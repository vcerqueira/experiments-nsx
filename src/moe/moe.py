import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_model import BaseModel
from neuralforecast.models import MLP, KAN, NBEATS

from src.moe.pooling import DensePooling, SparsePooling, SoftPooling

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
        kwargs.setdefault("hidden_size", 128)
    elif expert_arch == "mlp":
        kwargs.setdefault("hidden_size", 128)

    return kwargs


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
                 gate_loss_type: str = 'ib_softmax_mse',  # ['ib_softmax_mse','softmax_mse','ib_softmax_mse_grad','kl','ib_softmax_mse_window','ib_softmax_mse_grad_window']
                 pooled_combined_loss: bool = False,
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
        self.gate = nn.Linear(self.input_size, self.num_experts, bias=False)

        self.softmax = nn.Softmax(dim=1)
        k = min(k, self.num_experts)
        self.k = k
        if pooling == 'dense':
            self.pooling = DensePooling()
        elif pooling == 'sparse':
            self.pooling = SparsePooling(k=k)
        elif pooling == 'soft':
            self.pooling = SoftPooling()
        else:
            raise ValueError(f"Unknown pooling={pooling!r}; expected 'dense', 'sparse', or 'soft'")

        self.gate_loss_type = gate_loss_type
        self.pooled_combined_loss = pooled_combined_loss

    def forward(self, windows_batch: dict, return_components: bool = False):
        """
        Args:
            windows_batch (dict): Input batch
            return_components (bool): If True, returns individual expert outputs and gate weights
        Returns:
            torch.Tensor if return_components=False: Final weighted predictions
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] if return_components=True:
                (combined_output, expert_outputs, full_gate_weights)
        """
        insample_y = windows_batch['insample_y']  # [B, L] or [B, L, 1]
        gate_logits = self.gate(insample_y.squeeze(-1))
        full_gate_weights = self.softmax(gate_logits)  # [B, num_experts]
        pooled_gate_weights = self.pooling(gate_logits)
        # Expert losses and gate targets always see every expert. The forecast
        # mixture is pooled in training only when pooled_combined_loss is set.
        if return_components and not self.pooled_combined_loss:
            gate_weights = full_gate_weights
        else:
            gate_weights = pooled_gate_weights

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
            return combined_output, expert_outputs, full_gate_weights
        return combined_output

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
        output, expert_outputs, gate_weights = self(windows_batch, return_components=True)
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

        if self.gate_loss_type == 'ib_softmax_mse':
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
            expert_scores = -expert_losses_tensor
            target_weights = self._gate_target(F.softmax(expert_scores, dim=0))
            gate_loss = F.mse_loss(gate_weights.mean(0), target_weights)

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

        total_loss = expert_loss + gate_loss + combined_loss

        # Log all components
        self.log("combined_loss", combined_loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)
        self.log("expert_loss", expert_loss.detach(), batch_size=outsample_y.size(0), on_epoch=True)
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
