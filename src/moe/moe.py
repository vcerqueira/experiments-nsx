import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_model import BaseModel
from neuralforecast.common._modules import MLP as MLPLayer
from neuralforecast.models import MLP, KAN, NBEATS

from src.moe.pooling import DensePooling, SparsePooling

EXPERT_REGISTRY = {
    "mlp": MLP,
    "kan": KAN,
    "nbeats": NBEATS,
}


def _expert_init_kwargs(expert_arch, h, expert_kwargs):
    kwargs = dict(expert_kwargs or {})
    if expert_arch == "nbeats":
        kwargs.setdefault("stack_types", ["identity"])
        kwargs.setdefault("n_blocks", [1])
        kwargs.setdefault("mlp_units", [[128, 128]])
    elif expert_arch == "kan":
        kwargs.setdefault("hidden_size", 256)
    elif expert_arch == "mlp":
        kwargs.setdefault("hidden_size", 128)

    return kwargs


class _SeriesIndexedDataset(Dataset):
    """Dataset wrapper that keeps NeuralForecast's integer series index."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        item["series_idx"] = int(idx)
        return item

    def __getattr__(self, name):
        return getattr(self.base, name)


def _install_series_collate():
    """Keep series_idx in batches. The stock collate drops every key it does not know."""
    from neuralforecast.tsdataset import TimeSeriesLoader

    if getattr(TimeSeriesLoader, "_nsx_series_collate", False):
        return
    original = TimeSeriesLoader._collate_fn

    def _collate_fn(self, batch):
        collated = original(self, batch)
        if batch and isinstance(batch[0], dict) and "series_idx" in batch[0]:
            collated["series_idx"] = torch.tensor(
                [int(item["series_idx"]) for item in batch], dtype=torch.long
            )
        return collated

    TimeSeriesLoader._collate_fn = _collate_fn
    TimeSeriesLoader._nsx_series_collate = True


def _build_experts(h, input_size, num_experts, expert_arch, expert_kwargs):
    if expert_arch not in EXPERT_REGISTRY:
        raise ValueError(
            f"Unknown expert_arch={expert_arch!r}; expected one of {sorted(EXPERT_REGISTRY)}"
        )
    if num_experts < 1:
        raise ValueError(f"num_experts must be >= 1, got {num_experts}")
    cls = EXPERT_REGISTRY[expert_arch]
    kwargs = _expert_init_kwargs(expert_arch, h, expert_kwargs)
    return nn.ModuleList([
        cls(h=h, input_size=input_size, random_seed=random.randint(1, 1000), **kwargs)
        for _ in range(num_experts)
    ])


class NSX(BaseModel):
    """
    Neural-based time-Series mixture of eXperts

    Attributes:
        SAMPLING_TYPE (str): Type of sampling used, default is 'univariate'.
        EXOGENOUS_FUTR (bool): Indicates if future exogenous variables are used, default is False.
        EXOGENOUS_HIST (bool): Indicates if historical exogenous variables are used, default is False.
        EXOGENOUS_STAT (bool): Indicates if static exogenous variables are used, default is False.
    Args:
        h (int): Forecast horizon.
        input_size (int): Size of the input features.
        dropout (float, optional): Dropout rate, default is 0.1.
        futr_exog_list (list, optional): List of future exogenous variables, default is None.
        hist_exog_list (list, optional): List of historical exogenous variables, default is None.
        stat_exog_list (list, optional): List of static exogenous variables, default is None.
        loss (callable, optional): Loss function, default is SMAPE().
        valid_loss (callable, optional): Validation loss function, default is None.
        max_steps (int, optional): Maximum number of training steps, default is 1000.
        learning_rate (float, optional): Learning rate, default is 1e-3.
        num_lr_decays (int, optional): Number of learning rate decays, default is -1.
        early_stop_patience_steps (int, optional): Early stopping patience steps, default is -1.
        val_check_steps (int, optional): Validation check steps, default is 100.
        batch_size (int, optional): Batch size for training, default is 32.
        valid_batch_size (int, optional): Batch size for validation, default is 32.
        windows_batch_size (int, optional): Batch size for windows, default is 32.
        inference_windows_batch_size (int, optional): Batch size for inference windows, default is 32.
        start_padding_enabled (bool, optional): If start padding is enabled, default is False.
        step_size (int, optional): Step size, default is 1.
        scaler_type (str, optional): Type of scaler, default is 'identity'.
        random_seed (int, optional): Random seed, default is 1.
        drop_last_loader (bool, optional): If the last loader should be dropped, default is False.
        optimizer (callable, optional): Optimizer, default is None.
        optimizer_kwargs (dict, optional): Optimizer keyword arguments, default is None.
        lr_scheduler (callable, optional): Learning rate scheduler, default is None.
        lr_scheduler_kwargs (dict, optional): Learning rate scheduler keyword arguments, default is None.
        dataloader_kwargs (dict, optional): Data loader keyword arguments, default is None.
        **trainer_kwargs: Additional trainer keyword arguments.
    Methods:
        forward(windows_batch):
            Forward pass of the model.
            Args:
                windows_batch (dict): Batch of windowed time series data.
            Returns:
                torch.Tensor: Weighted sum of the experts' outputs.
    """
    # Class attributes
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
                 expert_arch: str = 'mlp',  # ['mlp','kan','nbeats']
                 expert_kwargs=None,
                 pooling: str = 'dense',  # ['dense','sparse','soft']
                 k: int = 3,  # Number of top experts for sparse pooling
                 gate: str = 'linear',  # ['linear','linear_bias','mlp']
                 gate_loss_type: str = 'ib_softmax_mse',  # ['ib_softmax_mse','softmax_mse','ib_softmax_mse_grad','kl','ib_softmax_mse_window','ib_softmax_mse_grad_window']
                 online_eg: bool = False,
                 specialize: bool = False,
                 load_balance_coef: float = 0.01,
                 series_state: bool = False,
                 series_state_eta: float = 1.0,
                 **trainer_kwargs):

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
            self.experts = experts if isinstance(experts, nn.ModuleList) else nn.ModuleList(experts)
        else:
            self.experts = _build_experts(
                h=self.h,
                input_size=self.input_size,
                num_experts=num_experts,
                expert_arch=expert_arch,
                expert_kwargs=expert_kwargs,
            )

        self.num_experts = len(self.experts)
        if gate == 'linear':
            self.gate = nn.Linear(self.input_size, self.num_experts, bias=False)
        elif gate == 'linear_bias':
            self.gate = nn.Linear(self.input_size, self.num_experts, bias=True)
        elif gate == 'mlp':
            self.gate = MLPLayer(
                self.input_size,
                self.num_experts,
                activation='ReLU',
                hidden_size=32,
                num_layers=1,
                dropout=0.1,
            )
        else:
            raise ValueError(f"Unknown gate={gate!r}; expected 'linear', 'linear_bias', or 'mlp'")

        self.softmax = nn.Softmax(dim=1)
        k = min(k, self.num_experts)
        self.k = k
        if pooling == 'dense':
            self.pooling = DensePooling()
        elif pooling == 'sparse':
            self.pooling = SparsePooling(k=k)
        else:
            raise ValueError(f"Unknown pooling={pooling!r}; expected 'dense', 'sparse'")

        self.gate_loss_type = gate_loss_type
        self.online_eg = bool(online_eg)
        self.specialize = bool(specialize)
        self.load_balance_coef = float(load_balance_coef)
        self.series_state = bool(series_state)
        self.series_state_eta = float(series_state_eta)
        self.series_cumloss = None
        self.series_visits = None
        if self.online_eg and self.specialize:
            raise ValueError(
                "online_eg and specialize are separate modes; "
                "online_eg learns one mixture, specialize learns an input-dependent gate"
            )
        if self.series_state and (self.online_eg or self.specialize):
            raise ValueError(
                "series_state plays the per-series Hedge table and cannot be combined with online_eg or specialize"
            )
        if self.series_state and self.series_state_eta <= 0:
            raise ValueError(
                f"series_state_eta must be > 0, got {self.series_state_eta}"
            )
        if self.specialize and self.load_balance_coef < 0:
            raise ValueError(
                f"load_balance_coef must be >= 0, got {self.load_balance_coef}"
            )
        if self.online_eg:
            if not isinstance(self.loss, MAE):
                raise ValueError(
                    "online_eg requires MAE so the gate update is the absolute-loss subgradient"
                )
            for parameter in self.gate.parameters():
                parameter.requires_grad_(False)
            self.register_buffer("eg_logits", torch.zeros(self.num_experts))
            self.register_buffer("eg_weight_sum", torch.zeros(self.num_experts))
            self.register_buffer("eg_steps", torch.zeros((), dtype=torch.long))
            self.register_buffer("eg_grad_bound", torch.zeros(()))
            self.eg_history = []
        if self.series_state:
            for parameter in self.gate.parameters():
                parameter.requires_grad_(False)

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
        """
        insample_y = windows_batch['insample_y']  # [B, L] or [B, L, 1]
        hedge_weights = windows_batch.get("hedge_weights")
        if hedge_weights is not None:
            gate_weights = hedge_weights
            full_gate_weights = gate_weights
        elif self.online_eg:
            # Training plays the EG state from previous steps. Prediction plays
            # the average of those played weights. Dense simplex weights only.
            gate_weights = self._eg_played_weights(insample_y.shape[0])
            full_gate_weights = gate_weights
        else:
            gate_logits = self.gate(insample_y.squeeze(-1))
            full_gate_weights = self.softmax(gate_logits)  # [B, num_experts]
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

    def _eg_weight_vector(self):
        if self.training or int(self.eg_steps) == 0:
            return torch.softmax(self.eg_logits, dim=0)
        averaged = self.eg_weight_sum / self.eg_steps.to(self.eg_weight_sum.dtype)
        return averaged / averaged.sum().clamp_min(1e-12)

    def _eg_played_weights(self, batch_size):
        weights = self._eg_weight_vector()
        if self.training:
            self._eg_round_weights = weights.detach()
        return weights.detach().unsqueeze(0).expand(batch_size, -1)

    def _mae_weight_grad(self, y_hat, y, expert_forecasts, mask):
        """Subgradient of weighted MAE with respect to the mixture weights."""
        y_hat = y_hat.detach()
        expert_forecasts = expert_forecasts.detach()
        residual_sign = torch.sign(y_hat - y)
        sample_weights = self.loss._compute_weights(y=y, mask=mask).to(residual_sign.dtype)
        if residual_sign.ndim == 3 and residual_sign.size(-1) == 1:
            residual_sign = residual_sign.squeeze(-1)
            sample_weights = sample_weights.squeeze(-1)
        numer = torch.einsum("bh,beh->e", sample_weights * residual_sign, expert_forecasts)
        denom = sample_weights.sum().clamp_min(1e-8)
        return numer / denom

    def _eg_update(self, y_hat, y, expert_forecasts, mask):
        """One exponentiated-gradient step. The played weights stay those of this round."""
        grad = self._mae_weight_grad(y_hat, y, expert_forecasts, mask)
        grad_inf = grad.abs().max()
        self.eg_grad_bound.copy_(torch.maximum(self.eg_grad_bound, grad_inf))
        step = self.eg_steps.to(device=grad.device, dtype=grad.dtype) + 1
        log_experts = torch.log(grad.new_tensor(float(self.num_experts)))
        grad_bound = self.eg_grad_bound.clamp_min(1e-8)
        eta = torch.sqrt(log_experts / (step * grad_bound.square()))
        played = self._eg_round_weights.detach()
        self.eg_logits.sub_(eta * grad)
        self.eg_logits.sub_(self.eg_logits.mean())
        self.eg_weight_sum.add_(played)
        self.eg_steps.add_(1)
        return eta.detach(), grad.detach(), played.detach()

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
        """
        usage = gate_weights.mean(dim=0)
        imbalance = self.num_experts * usage.square().sum()
        return self.load_balance_coef * imbalance

    def eg_regret(self):
        """Regret of the played gate on the training-step sequence."""
        if not getattr(self, "eg_history", None):
            raise RuntimeError("online_eg has not recorded any training steps")
        played_loss = 0.0
        expert_totals = None
        linearized = 0.0
        linearized_expert = None
        for row in self.eg_history:
            played_loss += row["played_loss"]
            if expert_totals is None:
                expert_totals = [0.0] * len(row["expert_losses"])
                linearized_expert = [0.0] * len(row["g"])
            for index, loss_value in enumerate(row["expert_losses"]):
                expert_totals[index] += loss_value
            linearized += sum(
                weight * grad_value for weight, grad_value in zip(row["w"], row["g"])
            )
            for index, grad_value in enumerate(row["g"]):
                linearized_expert[index] += grad_value
        return {
            "mae_regret_vs_best_expert": played_loss - min(expert_totals),
            "linearized_regret": linearized - min(linearized_expert),
            "steps": len(self.eg_history),
        }

    def on_fit_start(self):
        super().on_fit_start()
        if not self.series_state:
            return
        _install_series_collate()
        dataset = self.trainer.datamodule.dataset
        if not isinstance(dataset, _SeriesIndexedDataset):
            dataset = _SeriesIndexedDataset(dataset)
            self.trainer.datamodule.dataset = dataset
        n_series = len(dataset)
        self.series_cumloss = torch.zeros(
            n_series, self.num_experts, device=self.device
        )
        self.series_visits = torch.zeros(n_series, dtype=torch.long, device=self.device)

    def _train_windows_per_serie(self, temporal):
        window_size = self.input_size + self.h
        if self.val_size + self.test_size > 0:
            cutoff = -self.val_size - self.test_size
            temporal = temporal[:, :, :cutoff]
        temporal = self.padder_train(temporal)
        length = temporal.shape[-1]
        return (length - window_size) // self.step_size + 1

    def _window_series_index(self, batch, final_condition, w_idxs):
        if "series_idx" not in batch:
            raise RuntimeError(
                "series_state requires series_idx on the batch"
            )
        flat = final_condition[w_idxs]
        windows_per_serie = self._train_windows_per_serie(batch["temporal"])
        local = torch.div(flat, windows_per_serie, rounding_mode="floor")
        series_idx = batch["series_idx"].to(device=local.device)
        return series_idx[local]

    def _hedge_weights(self, series_index):
        if self.series_cumloss is None:
            raise RuntimeError("series_state weights require the cumulative table from fit")
        losses = self.series_cumloss.to(device=series_index.device, dtype=torch.float32)
        n_series = losses.shape[0]
        valid = (series_index >= 0) & (series_index < n_series)
        safe_index = series_index.clamp(0, max(n_series - 1, 0))
        gathered = losses[safe_index]
        weights = torch.softmax(-self.series_state_eta * gathered, dim=-1)
        if not torch.all(valid):
            uniform = torch.full_like(weights, 1.0 / self.num_experts)
            weights = torch.where(valid.unsqueeze(-1), weights, uniform)
        return weights.detach()

    def predict(self, dataset, **kwargs):
        if self.series_state:
            _install_series_collate()
            if not isinstance(dataset, _SeriesIndexedDataset):
                dataset = _SeriesIndexedDataset(dataset)
        return super().predict(dataset, **kwargs)

    def _predict_windows_per_serie(self, batch):
        temporal = batch["temporal"]
        window_size = self.input_size + self.h
        initial_input = temporal.shape[-1] - self.test_size
        if initial_input <= self.input_size:
            temporal = F.pad(
                temporal,
                pad=(self.input_size - initial_input, 0),
                mode="constant",
                value=0.0,
            )
        cutoff = -self.input_size - self.test_size
        temporal = temporal[:, :, cutoff:]
        if self.test_size == 0 and len(self.futr_exog_list) == 0:
            temporal = F.pad(temporal, pad=(0, self.h), mode="constant", value=0.0)
        length = temporal.shape[-1]
        return (length - window_size) // self.predict_step_size + 1

    def _arm_predict_hedge(self, batch):
        if "series_idx" not in batch:
            raise RuntimeError("series_state predict requires series_idx on the batch")
        windows_per_serie = self._predict_windows_per_serie(batch)
        n_series = batch["series_idx"].shape[0]
        flat = torch.arange(
            n_series * windows_per_serie, device=batch["series_idx"].device
        )
        local = torch.div(flat, windows_per_serie, rounding_mode="floor")
        series = batch["series_idx"].to(device=flat.device)[local]
        self._predict_hedge_weights = self._hedge_weights(series)
        self._predict_hedge_cursor = 0

    def _predict_step_direct(self, batch, batch_idx, recursive=False):
        if self.series_state and not recursive:
            self._arm_predict_hedge(batch)
        return super()._predict_step_direct(batch, batch_idx, recursive=recursive)

    def _predict_step_direct_batch(
        self, insample_y, insample_mask, hist_exog, futr_exog, stat_exog, y_idx
    ):
        windows_batch = dict(
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )
        if self.series_state:
            n_windows = insample_y.shape[0]
            start = self._predict_hedge_cursor
            weights = self._predict_hedge_weights[start : start + n_windows]
            if weights.shape[0] != n_windows:
                raise RuntimeError(
                    "series_state predict window count does not match the fitted table"
                )
            self._predict_hedge_cursor = start + n_windows
            windows_batch["hedge_weights"] = weights
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

    def _accumulate_series_state(self, batch, final_condition, w_idxs, per_window_losses):
        global_idx = self._window_series_index(batch, final_condition, w_idxs)
        losses = torch.stack(per_window_losses, dim=1).detach()
        self.series_cumloss = self.series_cumloss.to(device=losses.device)
        self.series_visits = self.series_visits.to(device=global_idx.device)
        self.series_cumloss.index_add_(0, global_idx, losses)
        self.series_visits.index_add_(
            0,
            global_idx,
            torch.ones(global_idx.shape[0], dtype=torch.long, device=global_idx.device),
        )

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
            series_ids = self._window_series_index(batch, final_condition, w_idxs)
            windows_batch["hedge_weights"] = self._hedge_weights(series_ids)

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
            self._accumulate_series_state(
                batch, final_condition, w_idxs, per_window_losses
            )

        if self.series_state:
            pass
        elif self.online_eg:
            eta, eg_grad, played_weights = self._eg_update(
                y_hat=output,
                y=outsample_y,
                expert_forecasts=torch.stack(mapped_experts, dim=1),
                mask=outsample_mask,
            )
        elif self.specialize:
            if not per_window_losses:
                raise ValueError("specialize requires a point forecast loss")
            # Experts follow the played mixture. The gate does not: its only
            # routing signal is the mixture loss below, plus a light balance term.
            expert_loss = self._responsibility_expert_loss(per_window_losses, mixture_weights)
            load_balance = self._load_balance_penalty(gate_weights)
        elif self.gate_loss_type == 'ib_softmax_mse':
            # Softmax-based weighting
            expert_scores = -expert_losses_tensor  # Convert losses to scores
            target_weights = self._gate_target(F.softmax(expert_scores, dim=0))

            target_weights_expanded = target_weights.unsqueeze(0).expand(gate_weights.size(0), -1)

            # Compute per-instance gate loss
            gate_loss = F.mse_loss(gate_weights, target_weights_expanded)
        elif self.gate_loss_type == 'softmax_mse':
            expert_scores = -expert_losses_tensor  # Convert losses to scores
            target_weights = self._gate_target(F.softmax(expert_scores, dim=0))

            gate_loss = F.mse_loss(gate_weights.mean(0), target_weights)
        elif self.gate_loss_type == 'ib_softmax_mse_grad':
            expert_contributions = (
                -torch.sign(output.squeeze(-1) - outsample_y.squeeze(-1)).unsqueeze(1)
                * expert_outputs
            )

            expert_scores = expert_contributions.mean(dim=[0, -1])  # Average over batch and horizon
            target_weights = self._gate_target(F.softmax(expert_scores, dim=0))

            # Expand to match batch size
            target_weights_expanded = target_weights.unsqueeze(0).expand(gate_weights.size(0), -1)

            # Compute gate loss
            gate_loss = F.mse_loss(gate_weights, target_weights_expanded)
        elif self.gate_loss_type == 'ib_softmax_mse_window':
            per_window = torch.stack(per_window_losses, dim=1)
            target_weights = self._gate_target(F.softmax(-per_window, dim=1))
            gate_loss = F.mse_loss(gate_weights, target_weights)
        elif self.gate_loss_type == 'ib_softmax_mse_grad_window':
            expert_contributions = (
                -torch.sign(output.squeeze(-1) - outsample_y.squeeze(-1)).unsqueeze(1)
                * expert_outputs
            )
            expert_scores = expert_contributions.mean(dim=-1)
            target_weights = self._gate_target(F.softmax(expert_scores, dim=1))
            gate_loss = F.mse_loss(gate_weights, target_weights)
        else:
            # "kl" and any other name: rank experts by softmax(-loss).
            raise ValueError(f"Unknown gate_loss_type={self.gate_loss_type!r}; expected 'ib_softmax_mse', 'softmax_mse', 'ib_softmax_mse_grad', 'ib_softmax_mse_window', or 'ib_softmax_mse_grad_window'")

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
            self.eg_history.append({
                "step": int(self.global_step),
                "played_loss": combined_loss.detach().item(),
                "expert_losses": [loss.detach().item() for loss in expert_losses],
                "eta": eta.item(),
                "g": eg_grad.tolist(),
                "w": played_weights.tolist(),
            })
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

        # Log individual expert losses and usage
        for i, loss in enumerate(expert_losses):
            self.log(f"expert_{i}_loss", loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)
            self.log(f"expert_{i}_usage", gate_weights[:, i].mean(), batch_size=outsample_y.size(0), on_epoch=True)

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

    def _gate_target(self, target_weights):
        return target_weights.detach()

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
