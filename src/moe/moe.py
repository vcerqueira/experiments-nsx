import torch
from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_model import BaseModel

from src.moe.experts import build_experts
from src.moe.gating import (
    GATE_NAMES,
    build_gate,
    gate_supervision_loss,
    validate_gate_loss_type,
)
from src.moe.online_eg import ExponentiatedGradient
from src.moe.pooling import POOLING_NAMES, build_pooling
from src.moe.series_state import (
    accumulate_losses,
    arm_predict,
    arm_validation,
    consume_predict,
    consume_validation,
    hedge_weights,
    prepare_fit,
    window_series_index,
    wrap_predict_dataset,
)


class NSX(BaseModel):
    """Neural-based time-series mixture of experts.

    The forecast is the pooled mixture of NeuralForecast experts. Gate losses
    supervise the dense softmax over every expert, which is what still has a
    gradient for experts left out by sparse pooling.

    Args:
        h (int): Forecast horizon.
        input_size (int): Autoregressive lag window.
        loss (callable, optional): Training loss. Default is MAE().
        experts (list, optional): Expert modules. Built from ``expert_arch`` when omitted.
        num_experts (int, optional): Experts to build. Default 6.
        expert_arch (str, optional): ``mlp``, ``kan``, or ``nbeats``. Default ``mlp``.
        expert_kwargs (dict, optional): Extra expert constructor arguments.
        pooling (str, optional): ``dense``, ``sparse``, or ``straight_through``.
            Default ``dense``.
        k (int, optional): Top-k experts for sparse pooling. Default 3.
        gate (str, optional): ``linear``, ``linear_bias``, or ``mlp``. Default ``linear``.
        gate_loss_type (str, optional): One of ``ib_softmax_mse``, ``softmax_mse``,
            ``ib_softmax_mse_grad``, ``ib_softmax_mse_window``,
            ``ib_softmax_mse_grad_window``. Default ``ib_softmax_mse``.
        online_eg (bool, optional): One shared exponentiated-gradient mixture.
            Requires MAE and dense pooling. Default False.
        specialize (bool, optional): Train each expert in proportion to the played
            mixture, and balance the dense gate. Default False.
        load_balance_coef (float, optional): Weight of that dense-gate penalty.
            Default 0.01.
        series_state (bool, optional): Per-series Hedge table. Default False.
        series_state_eta (float, optional): Hedge step size. Must be positive.
            Default 1.0.
        **trainer_kwargs: Remaining arguments are NeuralForecast ``BaseModel``
            training arguments (learning rate, batch size, scaler, and so on).
    """

    SAMPLING_TYPE = 'univariate'
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    EXOGENOUS_CAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(self,
                 h,
                 input_size,
                 stat_exog_list=None,
                 futr_exog_list=None,
                 hist_exog_list=None,
                 loss=MAE(),
                 valid_loss=None,
                 max_steps: int = 4000,
                 learning_rate: float = 1e-3,
                 num_lr_decays: int = -1,
                 early_stop_patience_steps: int = -1,
                 val_check_steps: int = 100,
                 batch_size: int = 32,
                 valid_batch_size: int = 32,
                 windows_batch_size: int = 32,
                 inference_windows_batch_size: int = 32,
                 start_padding_enabled: bool = False,
                 step_size: int = 1,
                 scaler_type: str = 'identity',
                 random_seed: int = 1,
                 drop_last_loader: bool = False,
                 optimizer=None,
                 optimizer_kwargs=None,
                 lr_scheduler=None,
                 lr_scheduler_kwargs=None,
                 dataloader_kwargs=None,
                 experts=None,
                 num_experts: int = 6,
                 expert_arch: str = 'mlp',
                 expert_kwargs=None,
                 pooling: str = 'dense',
                 k: int = 3,
                 gate: str = 'linear',
                 gate_loss_type: str = 'ib_softmax_mse',
                 online_eg: bool = False,
                 specialize: bool = False,
                 load_balance_coef: float = 0.01,
                 series_state: bool = False,
                 series_state_eta: float = 1.0,
                 **trainer_kwargs):
        self._validate_modes(
            k=k,
            pooling=pooling,
            gate=gate,
            gate_loss_type=gate_loss_type,
            online_eg=online_eg,
            specialize=specialize,
            load_balance_coef=load_balance_coef,
            series_state=series_state,
            series_state_eta=series_state_eta,
            loss=loss,
        )

        super(NSX, self).__init__(h=h,
                                  input_size=input_size,
                                  stat_exog_list=stat_exog_list,
                                  futr_exog_list=futr_exog_list,
                                  hist_exog_list=hist_exog_list,
                                  loss=loss,
                                  valid_loss=valid_loss,
                                  max_steps=max_steps,
                                  learning_rate=learning_rate,
                                  num_lr_decays=num_lr_decays,
                                  early_stop_patience_steps=early_stop_patience_steps,
                                  val_check_steps=val_check_steps,
                                  batch_size=batch_size,
                                  valid_batch_size=valid_batch_size,
                                  windows_batch_size=windows_batch_size,
                                  inference_windows_batch_size=inference_windows_batch_size,
                                  start_padding_enabled=start_padding_enabled,
                                  step_size=step_size,
                                  scaler_type=scaler_type,
                                  random_seed=random_seed,
                                  drop_last_loader=drop_last_loader,
                                  optimizer=optimizer,
                                  optimizer_kwargs=optimizer_kwargs,
                                  lr_scheduler=lr_scheduler,
                                  lr_scheduler_kwargs=lr_scheduler_kwargs,
                                  dataloader_kwargs=dataloader_kwargs,
                                  **trainer_kwargs)

        self.input_size = input_size
        self.h = h

        if experts is not None:
            self.experts = experts if isinstance(experts, torch.nn.ModuleList) else torch.nn.ModuleList(experts)
        else:
            self.experts = build_experts(
                h=self.h,
                input_size=self.input_size,
                num_experts=num_experts,
                expert_arch=expert_arch,
                expert_kwargs=expert_kwargs,
                random_seed=random_seed,
            )

        self.num_experts = len(self.experts)
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        self.k = min(k, self.num_experts)

        self.gate = build_gate(gate, self.input_size, self.num_experts)
        self.pooling = build_pooling(pooling, self.k)

        self.gate_loss_type = gate_loss_type
        self.online_eg = bool(online_eg)
        self.specialize = bool(specialize)
        self.load_balance_coef = float(load_balance_coef)
        self.series_state = bool(series_state)
        self.series_state_eta = float(series_state_eta)
        self._val_hedge_weights = None
        self._val_hedge_cursor = 0
        self._predict_hedge_weights = None
        self._predict_hedge_cursor = 0

        if self.online_eg or self.series_state:
            for parameter in self.gate.parameters():
                parameter.requires_grad_(False)
        if self.online_eg:
            self.eg = ExponentiatedGradient(self.num_experts)

    @staticmethod
    def _validate_modes(
        k,
        pooling,
        gate,
        gate_loss_type,
        online_eg,
        specialize,
        load_balance_coef,
        series_state,
        series_state_eta,
        loss,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if pooling not in POOLING_NAMES:
            expected = ", ".join(repr(name) for name in POOLING_NAMES)
            raise ValueError(f"Unknown pooling={pooling!r}; expected {expected}")
        if gate not in GATE_NAMES:
            expected = ", ".join(repr(name) for name in GATE_NAMES)
            raise ValueError(f"Unknown gate={gate!r}; expected {expected}")
        validate_gate_loss_type(gate_loss_type)
        if online_eg and specialize:
            raise ValueError(
                "online_eg and specialize are separate modes; "
                "online_eg learns one mixture, specialize learns an input-dependent gate"
            )
        if series_state and (online_eg or specialize):
            raise ValueError(
                "series_state plays the per-series Hedge table and cannot be combined with online_eg or specialize"
            )
        if series_state and series_state_eta <= 0:
            raise ValueError(
                f"series_state_eta must be > 0, got {series_state_eta}"
            )
        if specialize and load_balance_coef < 0:
            raise ValueError(
                f"load_balance_coef must be >= 0, got {load_balance_coef}"
            )
        if online_eg and pooling != 'dense':
            raise ValueError(
                "online_eg learns a dense simplex and cannot be combined with "
                f"pooling={pooling!r}"
            )
        if online_eg and not isinstance(loss, MAE):
            raise ValueError(
                "online_eg requires MAE so the gate update is the absolute-loss subgradient"
            )

    def forward(self, windows_batch: dict, return_components: bool = False):
        """
        Args:
            windows_batch (dict): Input batch
            return_components (bool): If True, returns individual expert outputs and gate weights
        Returns:
            torch.Tensor if return_components=False: Final weighted predictions
            Tuple if return_components=True:
                (combined_output, expert_outputs, full_gate_weights, mixture_weights)
                mixture_weights are the weights that form the forecast.
                full_gate_weights are the dense softmax the gate loss supervises.
        """
        insample_y = windows_batch['insample_y']  # [B, L] or [B, L, 1]
        hedge_weights = windows_batch.get("hedge_weights")
        if hedge_weights is None and self.series_state and self._is_validating():
            # Validation does not pass hedge weights. Consume the table armed
            # for this batch. Predict passes weights explicitly so extra
            # explain forwards cannot walk the same cursor.
            hedge_weights = consume_validation(self, insample_y.shape[0])
        if hedge_weights is not None:
            gate_weights = hedge_weights
            full_gate_weights = gate_weights
        elif self.online_eg:
            # Training plays the EG state from previous steps. Prediction plays
            # the average of those played weights. Dense simplex weights only.
            gate_weights = self.eg.played_weights(insample_y.shape[0])
            full_gate_weights = gate_weights
        else:
            gate_logits = self.gate(insample_y.squeeze(-1))
            full_gate_weights = torch.softmax(gate_logits, dim=1)  # [B, num_experts]
            # The forecast uses the pooled mixture. Expert losses and gate targets
            # still see every expert through full_gate_weights.
            gate_weights = self.pooling(gate_logits)

        expert_outputs = []
        for expert_module in self.experts:
            expert_output = expert_module(windows_batch)
            if expert_output.ndim == 3:
                expert_output = expert_output.squeeze(-1)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, h]

        combined_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
        combined_output = combined_output.unsqueeze(-1)  # [B, h, 1] for NeuralForecast

        if return_components:
            return combined_output, expert_outputs, full_gate_weights, gate_weights
        return combined_output

    def _is_validating(self):
        trainer = getattr(self, "_trainer", None)
        return trainer is not None and bool(getattr(trainer, "validating", False))

    def eg_regret(self):
        """Regret of the played gate on the training-step sequence."""
        eg = getattr(self, "eg", None)
        if eg is None:
            raise RuntimeError("online_eg has not recorded any training steps")
        return eg.regret()

    def on_fit_start(self):
        super().on_fit_start()
        if not self.series_state:
            return
        prepare_fit(self)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if not self.series_state or getattr(self, "val_size", 0) == 0:
            return
        arm_validation(self, batch)

    def predict(self, dataset, **kwargs):
        if self.series_state:
            dataset = wrap_predict_dataset(dataset)
        return super().predict(dataset, **kwargs)

    def _predict_step_direct(self, batch, batch_idx, recursive=False):
        if self.series_state and not recursive:
            arm_predict(self, batch)
        return super()._predict_step_direct(batch, batch_idx, recursive=recursive)

    def _predict_step_direct_batch(
        self, insample_y, insample_mask, hist_exog, futr_exog, stat_exog, y_idx
    ):
        # Kept so predict can pass hedge weights into forward. The parent
        # method builds the batch itself, and explain calls the model again
        # without those weights.
        windows_batch = dict(
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )
        if self.series_state:
            windows_batch["hedge_weights"] = consume_predict(self, insample_y.shape[0])
        output_batch = self(windows_batch)
        output_batch = self.loss.domain_map(output_batch)
        if self.loss.is_distribution_output:
            y_loc, y_scale = self._get_loc_scale(y_idx)
            distr_args = self.loss.scale_decouple(
                output=output_batch, loc=y_loc, scale=y_scale
            )
            _, sample_mean, quants = self.loss.sample(distr_args=distr_args)
            y_hat = torch.concat((sample_mean, quants), axis=-1)
            if self.loss.return_params:
                distr_args = torch.stack(distr_args, dim=-1)
                if distr_args.ndim > 4:
                    distr_args = distr_args.flatten(-2, -1)
                y_hat = torch.concat((y_hat, distr_args), axis=-1)
            return y_hat
        return self._inv_normalization(y_hat=output_batch, y_idx=y_idx)

    def _responsibility_expert_loss(self, per_window_losses, played_weights):
        """Train expert i on window b in proportion to w_{b,i}.

        w is detached, so this term updates experts and leaves the gate to the
        mixture loss.
        """
        per_window = torch.stack(per_window_losses, dim=1)
        weights = played_weights.detach()
        if per_window.shape != weights.shape:
            raise ValueError(
                f"responsibility weights {tuple(weights.shape)} do not match "
                f"per-window losses {tuple(per_window.shape)}"
            )
        return (weights * per_window).sum() / weights.sum().clamp_min(1e-8)

    def _load_balance_penalty(self, gate_weights):
        """Small push toward using every expert.

        E * sum_i usage_i^2 is 1 when the gate is uniform and grows to E if one
        expert takes the batch. load_balance_coef keeps that far below a full
        extra copy of every expert's MAE.

        ``gate_weights`` is the dense softmax. Under sparse pooling that term
        is what still sends a gradient to experts outside the top-k.
        """
        usage = gate_weights.mean(dim=0)
        imbalance = self.num_experts * usage.square().sum()
        return self.load_balance_coef * imbalance

    def training_step(self, batch, batch_idx):
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
                w_idxs = torch.randint(
                    0,
                    n_windows,
                    size=(self.windows_batch_size,),
                    device=windows_temporal.device,
                )
            else:
                w_idxs = torch.randperm(n_windows, device=windows_temporal.device)[
                    : self.windows_batch_size
                ]
        else:
            w_idxs = torch.arange(n_windows, device=windows_temporal.device)

        windows = self._sample_windows(
            windows_temporal=windows_temporal,
            static=static,
            static_cols=static_cols,
            temporal_cols=temporal_cols,
            w_idxs=w_idxs,
            final_condition=final_condition,
            sample_weight=sample_weight_windows,
        )
        original_outsample_y = torch.clone(windows["temporal"][:, self.input_size:, y_idx])
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows
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
        if self.series_state:
            series_ids = window_series_index(self, batch, final_condition, w_idxs)
            windows_batch["hedge_weights"] = hedge_weights(self, series_ids)

        # Get predictions and component analysis
        output, expert_outputs, gate_weights, mixture_weights = self(
            windows_batch, return_components=True
        )
        output = self.loss.domain_map(output)

        expert_losses = []
        per_window_losses = []
        mapped_experts = []
        for i in range(expert_outputs.size(1)):
            expert_output = self.loss.domain_map(expert_outputs[:, i].unsqueeze(-1))
            if self.online_eg:
                if expert_output.ndim == 3 and expert_output.size(-1) == 1:
                    mapped_experts.append(expert_output.squeeze(-1))
                else:
                    mapped_experts.append(expert_output)
            if self.loss.is_distribution_output:
                y_loc, y_scale = self._get_loc_scale(y_idx)
                distr_args = self.loss.scale_decouple(output=expert_output, loc=y_loc, scale=y_scale)
                expert_loss = self.loss(y=original_outsample_y, distr_args=distr_args, mask=outsample_mask)
            else:
                expert_loss = self._point_loss(
                    y=outsample_y,
                    y_hat=expert_output,
                    mask=outsample_mask,
                    insample_y=insample_y,
                )
                per_window_losses.append(
                    self._per_window_point_loss(outsample_y, expert_output, outsample_mask)
                )
            expert_losses.append(expert_loss)

        expert_losses_tensor = torch.stack(expert_losses)
        expert_loss = expert_losses_tensor.mean()

        if self.series_state:
            if not per_window_losses:
                raise ValueError("series_state requires a point forecast loss")
            accumulate_losses(self, batch, final_condition, w_idxs, per_window_losses)
        elif self.online_eg:
            eta, eg_grad, played_weights = self.eg.update(
                y_hat=output,
                y=outsample_y,
                expert_forecasts=torch.stack(mapped_experts, dim=1),
                mask=outsample_mask,
                loss=self.loss,
            )
        elif self.specialize:
            if not per_window_losses:
                raise ValueError("specialize requires a point forecast loss")
            # Experts follow the played mixture. The gate does not: its only
            # routing signal is the mixture loss below, plus a light balance term.
            expert_loss = self._responsibility_expert_loss(per_window_losses, mixture_weights)
            load_balance = self._load_balance_penalty(gate_weights)
        else:
            # gate_weights is the dense softmax. The forecast used mixture_weights.
            gate_loss = gate_supervision_loss(
                self.gate_loss_type,
                gate_weights,
                expert_losses_tensor,
                per_window_losses,
                output,
                outsample_y,
                expert_outputs,
            )

        # Compute combined loss
        if self.loss.is_distribution_output:
            y_loc, y_scale = self._get_loc_scale(y_idx)
            outsample_y = original_outsample_y
            distr_args = self.loss.scale_decouple(output=output, loc=y_loc, scale=y_scale)
            combined_loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            combined_loss = self._point_loss(
                y=outsample_y,
                y_hat=output,
                mask=outsample_mask,
                insample_y=insample_y,
            )

        if self.online_eg or self.series_state:
            total_loss = expert_loss + combined_loss
        elif self.specialize:
            total_loss = expert_loss + combined_loss + load_balance
        else:
            total_loss = expert_loss + gate_loss + combined_loss

        # Log all components
        self.log("combined_loss", combined_loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)
        self.log("expert_loss", expert_loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)
        if self.online_eg:
            self.eg.note_step(
                step=int(self.global_step),
                played_loss=combined_loss.detach().item(),
                expert_losses=[loss.detach().item() for loss in expert_losses],
                eta=eta.item(),
                grad=eg_grad.tolist(),
                played=played_weights.tolist(),
            )
            self.log("eg_eta", eta, batch_size=outsample_y.size(0), on_epoch=True)
            for i, grad_value in enumerate(eg_grad):
                self.log(
                    f"eg_g_{i}",
                    grad_value,
                    batch_size=outsample_y.size(0),
                    on_epoch=True,
                )
        elif self.specialize:
            self.log("load_balance", load_balance.detach(), batch_size=outsample_y.size(0), on_epoch=True)
        elif not self.series_state:
            self.log("gate_loss", gate_loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)

        # Log individual expert losses and the mixture that formed the forecast.
        for i, loss in enumerate(expert_losses):
            self.log(f"expert_{i}_loss", loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)
            self.log(
                f"expert_{i}_usage",
                mixture_weights[:, i].mean(),
                batch_size=outsample_y.size(0),
                on_epoch=True,
            )

        if torch.isnan(total_loss):
            print("Model Parameters", self.hparams)
            print("insample_y", torch.isnan(insample_y).sum())
            print("outsample_y", torch.isnan(outsample_y).sum())
            raise Exception("Loss is NaN, training stopped.")

        train_loss_log = total_loss.detach().item()
        self.log(
            "train_loss",
            train_loss_log,
            batch_size=outsample_y.size(0),
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append((self.global_step, train_loss_log))

        self.h = self.horizon_backup
        return total_loss

    def _per_window_point_loss(self, y, y_hat, mask):
        y = y.squeeze(-1)
        y_hat = y_hat.squeeze(-1)
        err = (y_hat - y).abs()
        if mask is None:
            return err.mean(dim=-1)
        mask = mask.squeeze(-1).to(err.dtype)
        return (err * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    def _point_loss(self, y, y_hat, mask, insample_y):
        kwargs = dict(y=y, y_hat=y_hat, mask=mask, y_insample=insample_y)
        try:
            return self.loss(**kwargs)
        except TypeError:
            return self.loss(y=y, y_hat=y_hat, mask=mask)
