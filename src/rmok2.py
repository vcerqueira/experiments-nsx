import torch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray import tune
from neuralforecast.auto import AutoRMoK
from neuralforecast.common._base_auto import BaseAuto, MockTrial
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import RMoK

from src.moe.gating import gate_supervision_loss

_GATE_LOSS_TYPE = "ib_softmax_mse_window"


class RMoK2(RMoK):
    """RMoK trained with the NSX objective.

    The experts and the gate are unchanged. Training minimizes the mean of the
    experts' own forecast losses, a detached per-series credit-assignment loss
    on the gate, and the mixture forecast loss. Validation still scores only
    the mixture.
    """

    def forward(self, windows_batch):
        forecast, _, _ = self._forecast_components(windows_batch)
        return forecast

    def _forecast_components(self, windows_batch):
        """Mixture forecast, gate softmax, and each expert forecast.

        Scores have shape ``[B * N, num_experts]``, one row per series, matching
        the ``transpose`` then ``reshape`` in RMoK. Expert forecasts are
        denormalized with the RevIN statistics of that same forward.
        """
        insample_y = windows_batch["insample_y"]
        batch_size, _, n_series = insample_y.shape
        normalized = self.rev(insample_y, "norm")
        flat = self.dropout(normalized).transpose(1, 2).reshape(batch_size * n_series, -1)

        scores = torch.softmax(self.gate(flat), dim=-1)
        raw = torch.stack(
            [expert(flat) for expert in self.experts],
            dim=-1,
        )
        mixed = torch.einsum("ble,be->bl", raw, scores)
        forecast = self._denormalize_forecast(mixed, batch_size, n_series)
        expert_forecasts = torch.stack(
            [
                self._denormalize_forecast(raw[:, :, index], batch_size, n_series)
                for index in range(self.num_experts)
            ],
            dim=0,
        )
        return forecast, scores, expert_forecasts

    def _denormalize_forecast(self, flat, batch_size, n_series):
        forecast = flat.reshape(
            batch_size,
            n_series,
            self.h * self.loss.outputsize_multiplier,
        ).permute(0, 2, 1)
        forecast = self.rev(forecast, "denorm")
        return forecast.reshape(batch_size, self.h, -1)

    def training_step(self, batch, batch_idx):
        if self.loss.is_distribution_output:
            raise ValueError("RMoK2 gate loss requires a point forecast loss")
        if self.RECURRENT:
            self.h = self.h_train

        y_idx = batch["y_idx"]
        (
            windows_temporal,
            static,
            static_cols,
            final_condition,
            sample_weight_windows,
            temporal_cols,
        ) = self._create_windows(batch, step="train")
        final_condition = self._shard_multivariate_windows(final_condition)
        n_windows = len(final_condition)
        if self.windows_batch_size is not None:
            if n_windows < self.windows_batch_size:
                window_idxs = torch.randint(
                    0,
                    n_windows,
                    size=(self.windows_batch_size,),
                    device=windows_temporal.device,
                )
            else:
                window_idxs = torch.randperm(n_windows, device=windows_temporal.device)[
                    : self.windows_batch_size
                ]
        else:
            window_idxs = torch.arange(n_windows, device=windows_temporal.device)

        windows = self._sample_windows(
            windows_temporal=windows_temporal,
            static=static,
            static_cols=static_cols,
            temporal_cols=temporal_cols,
            w_idxs=window_idxs,
            final_condition=final_condition,
            sample_weight=sample_weight_windows,
        )
        windows = self._normalization(windows=windows, y_idx=y_idx)
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        sample_weight = windows.get("sample_weight", None)
        if sample_weight is not None:
            outsample_mask = outsample_mask * sample_weight

        windows_batch = dict(
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )
        output, scores, expert_forecasts = self._forecast_components(windows_batch)
        output = self.loss.domain_map(output)

        def point_loss(forecast):
            return self.loss(
                y=outsample_y,
                y_hat=forecast,
                y_insample=insample_y,
                mask=outsample_mask,
            )

        combined_loss = point_loss(output)
        expert_losses = []
        per_series_losses = []
        for index in range(self.num_experts):
            expert_forecast = self.loss.domain_map(expert_forecasts[index])
            expert_losses.append(point_loss(expert_forecast))
            per_series_losses.append(
                self._per_series_point_loss(outsample_y, expert_forecast, outsample_mask)
            )

        expert_loss = torch.stack(expert_losses).mean()
        gate_loss = gate_supervision_loss(
            _GATE_LOSS_TYPE,
            scores,
            torch.stack(expert_losses),
            per_series_losses,
            output,
            outsample_y,
            expert_forecasts,
        )
        total_loss = expert_loss + gate_loss + combined_loss

        if torch.isnan(total_loss):
            raise Exception("Loss is NaN, training stopped.")

        batch_size = outsample_y.size(0)
        self.log("combined_loss", combined_loss.detach(), batch_size=batch_size, on_epoch=True)
        self.log("expert_loss", expert_loss.detach(), batch_size=batch_size, on_epoch=True)
        self.log("gate_loss", gate_loss.detach(), batch_size=batch_size, on_epoch=True)
        train_loss_log = total_loss.detach().item()
        self.log(
            "train_loss",
            train_loss_log,
            batch_size=batch_size,
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append((self.global_step, train_loss_log))
        self.h = self.horizon_backup
        return total_loss

    @staticmethod
    def _per_series_point_loss(y, y_hat, mask):
        """Horizon MAE for each series, flattened to the gate's row order."""
        error = (y_hat - y).abs()
        if mask is None:
            per_series = error.mean(dim=1)
        else:
            weights = mask.to(error.dtype)
            if weights.ndim == 2:
                weights = weights.unsqueeze(-1)
            weights = weights.expand_as(error)
            per_series = (error * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return per_series.reshape(-1)


class AutoRMoK2(AutoRMoK):
    """Automatic hyperparameter optimization for ``RMoK2``.

    The search space matches ``AutoRMoK``. Trials train ``RMoK2``.
    """

    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_series": None,
        "taylor_order": tune.choice([3, 4, 5]),
        "jacobi_degree": tune.choice([4, 5, 6]),
        "wavelet_function": tune.choice(
            ["mexican_hat", "morlet", "dog", "meyer", "shannon"]
        ),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard", "identity"]),
        "max_steps": tune.choice([1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(
        self,
        h,
        n_series,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        time_budget=None,
        refit_with_val=False,
        cpus=None,
        gpus=None,
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
        ray_options=None,
        optuna_options=None,
    ):
        if config is None:
            config = self.get_default_config(h=h, backend=backend, n_series=n_series)

        if backend == "ray":
            config["n_series"] = n_series
        elif backend == "optuna":
            mock_trial = MockTrial()
            if (
                "n_series" in config(mock_trial)
                and config(mock_trial)["n_series"] != n_series
            ) or ("n_series" not in config(mock_trial)):
                raise Exception(f"config needs 'n_series': {n_series}")

        super(AutoRMoK, self).__init__(
            cls_model=RMoK2,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            time_budget=time_budget,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
            ray_options=ray_options,
            optuna_options=optuna_options,
        )