from contextlib import nullcontext

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
from src.moe.pooling import POOLING_NAMES, build_pooling


class _NSXMixin:
    """Shared mixture-of-experts forecast and training step.

    ``NSX`` and ``NSXFrozen`` both subclass ``BaseModel``. This mixin is not a
    NeuralForecast model.
    """

    def _check_passed_experts(self):
        model_name = type(self).__name__
        for index, expert in enumerate(self.experts):
            if expert.h != self.h or expert.input_size != self.input_size:
                raise ValueError(
                    f"expert {index} was built with h={expert.h}, input_size={expert.input_size}; "
                    f"{model_name} has h={self.h}, input_size={self.input_size}"
                )

    def _use_experts(self, experts):
        if isinstance(experts, torch.nn.ModuleList):
            self.experts = experts
        else:
            self.experts = torch.nn.ModuleList(list(experts))
        self._check_passed_experts()

    def _install_experts(self, experts, num_experts, expert_arch, expert_kwargs, random_seed):
        if experts is not None:
            self._use_experts(experts)
            return
        self.experts = build_experts(
            h=self.h,
            input_size=self.input_size,
            num_experts=num_experts,
            expert_arch=expert_arch,
            expert_kwargs=expert_kwargs,
            random_seed=random_seed,
        )

    def _install_gate(self, k, pooling, gate, gate_loss_type):
        self.num_experts = len(self.experts)
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        self.k = min(k, self.num_experts)
        self.gate = build_gate(gate, self.input_size, self.num_experts)
        self.pooling = build_pooling(pooling, self.k)
        self.gate_loss_type = gate_loss_type

    @staticmethod
    def _validate_modes(
        k,
        pooling,
        gate,
        gate_loss_type,
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
        gate_logits = self.gate(insample_y.squeeze(-1))
        full_gate_weights = torch.softmax(gate_logits, dim=1)  # [B, num_experts]
        # The forecast uses the pooled mixture. Expert losses and gate targets
        # still see every expert through full_gate_weights.
        gate_weights = self.pooling(gate_logits)

        with self._expert_grad_context():
            expert_outputs = self._run_experts(windows_batch)

        combined_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
        combined_output = combined_output.unsqueeze(-1)  # [B, h, 1] for NeuralForecast

        if return_components:
            return combined_output, expert_outputs, full_gate_weights, gate_weights
        return combined_output

    def _run_experts(self, windows_batch):
        expert_outputs = []
        for expert_module in self.experts:
            expert_output = expert_module(windows_batch)
            if expert_output.ndim == 3:
                expert_output = expert_output.squeeze(-1)
            expert_outputs.append(expert_output)
        return torch.stack(expert_outputs, dim=1)  # [B, E, h]

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

        # Get predictions and component analysis
        output, expert_outputs, gate_weights, mixture_weights = self(
            windows_batch, return_components=True
        )
        output = self.loss.domain_map(output)

        expert_losses = []
        per_window_losses = []
        for i in range(expert_outputs.size(1)):
            expert_output = self.loss.domain_map(expert_outputs[:, i].unsqueeze(-1))
            if self.loss.is_distribution_output:
                y_loc, y_scale = self._get_loc_scale(y_idx)
                distr_args = self.loss.scale_decouple(output=expert_output, loc=y_loc, scale=y_scale)
                expert_loss = self.loss(y=original_outsample_y, distr_args=distr_args, mask=outsample_mask)
            else:
                expert_loss = self.loss(
                    y=outsample_y,
                    y_hat=expert_output,
                    mask=outsample_mask,
                )
                per_window_losses.append(
                    self._per_window_point_loss(outsample_y, expert_output, outsample_mask)
                )
            expert_losses.append(expert_loss)

        expert_losses_tensor = torch.stack(expert_losses)
        expert_loss = expert_losses_tensor.mean()

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
            combined_loss = self.loss(
                y=outsample_y,
                y_hat=output,
                mask=outsample_mask,
            )

        total_loss = self._total_loss(expert_loss, gate_loss, combined_loss)

        # Log all components
        self.log("combined_loss", combined_loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)
        self.log("expert_loss", expert_loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)
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


class NSX(_NSXMixin, BaseModel):
    """Neural-based time-series mixture of experts.

    The forecast is the pooled mixture of NeuralForecast experts. Experts are
    trained together with the gate. Gate losses supervise the dense softmax
    over every expert, which is what still has a gradient for experts left
    out by sparse pooling.

    Args:
        h (int): Forecast horizon.
        input_size (int): Autoregressive lag window.
        loss (callable, optional): Training loss. Default is MAE().
        experts (list, optional): Expert modules. Built from ``expert_arch`` when omitted.
        num_experts (int, optional): Experts to build. Default 6.
        expert_arch (str, optional): ``mlp``, ``kan``, or ``nbeats``. Default ``mlp``.
        expert_kwargs (dict, optional): Extra expert constructor arguments.
        pooling (str, optional): ``dense`` or ``sparse``. Default ``dense``.
        k (int, optional): Top-k experts for sparse pooling. Default 3.
        gate (str, optional): ``linear``, ``linear_bias``, or ``mlp``. Default ``linear``.
        gate_loss_type (str, optional): One of ``ib_softmax_mse``, ``softmax_mse``,
            ``ib_softmax_mse_grad``, ``ib_softmax_mse_window``,
            ``ib_softmax_mse_grad_window``. Default ``ib_softmax_mse``.
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
                 **trainer_kwargs):
        self._validate_modes(
            k=k,
            pooling=pooling,
            gate=gate,
            gate_loss_type=gate_loss_type,
        )

        super().__init__(h=h,
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
        self._install_experts(experts, num_experts, expert_arch, expert_kwargs, random_seed)
        self._install_gate(k, pooling, gate, gate_loss_type)

    def _expert_grad_context(self):
        return nullcontext()

    def _total_loss(self, expert_loss, gate_loss, combined_loss):
        return expert_loss + gate_loss + combined_loss


class NSXFrozen(_NSXMixin, BaseModel):
    """Mixture of experts with fixed pretrained experts.

    Only the gate is trained. Passed experts stay in eval mode and do not
    receive gradients. The forecast column stays ``NSX`` so result files match
    the trainable model.

    Args:
        h (int): Forecast horizon.
        input_size (int): Autoregressive lag window.
        experts (list): Trained NeuralForecast modules used as experts.
        loss (callable, optional): Training loss. Default is MAE().
        alias (str, optional): Forecast column name. Default ``NSX``.
        pooling (str, optional): ``dense`` or ``sparse``. Default ``dense``.
        k (int, optional): Top-k experts for sparse pooling. Default 3.
        gate (str, optional): ``linear``, ``linear_bias``, or ``mlp``. Default ``linear``.
        gate_loss_type (str, optional): One of ``ib_softmax_mse``, ``softmax_mse``,
            ``ib_softmax_mse_grad``, ``ib_softmax_mse_window``,
            ``ib_softmax_mse_grad_window``. Default ``ib_softmax_mse``.
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
                 experts,
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
                 alias: str = 'NSX',
                 pooling: str = 'dense',
                 k: int = 3,
                 gate: str = 'linear',
                 gate_loss_type: str = 'ib_softmax_mse',
                 **trainer_kwargs):
        if not experts:
            raise ValueError('NSXFrozen requires trained modules in experts')
        self._validate_modes(
            k=k,
            pooling=pooling,
            gate=gate,
            gate_loss_type=gate_loss_type,
        )

        super().__init__(h=h,
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
                         alias=alias,
                         **trainer_kwargs)

        self.input_size = input_size
        self.h = h
        self._use_experts(experts)
        for parameter in self.experts.parameters():
            parameter.requires_grad_(False)
        self.experts.eval()
        self._install_gate(k, pooling, gate, gate_loss_type)

    def train(self, mode=True):
        super().train(mode)
        self.experts.eval()
        return self

    def configure_optimizers(self):
        parameters = self.parameters
        self.parameters = lambda recurse=True: (
            parameter for parameter in parameters(recurse) if parameter.requires_grad
        )
        try:
            return super().configure_optimizers()
        finally:
            self.parameters = parameters

    def _expert_grad_context(self):
        return torch.no_grad()

    def _total_loss(self, expert_loss, gate_loss, combined_loss):
        # Expert weights are fixed, so expert_loss cannot train them. It
        # stays in the logs because the gate target is built from it.
        return gate_loss + combined_loss
