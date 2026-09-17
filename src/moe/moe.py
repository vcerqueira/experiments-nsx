from typing import Tuple, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_model import BaseModel
from neuralforecast.common._modules import MLP as MLPLayer, AttentionLayer
from neuralforecast.models import MLP, KAN, NBEATS
from neuralforecast.losses.pytorch import MAE, _weighted_mean, BasePointLoss

from src.moe.gates import RNNGate, AttentionGate
from src.moe.losses import MAEGrad
from src.moe.pooling import DensePooling, SparsePooling, SoftPooling, StraightThroughPooling

EXPERT_REGISTRY = {
    "mlp": MLP,
    "kan": KAN,
    "nbeats": NBEATS,
}


def _expert_init_kwargs(expert_arch, h, expert_kwargs):
    kwargs = dict(expert_kwargs or {})
    if expert_arch == "nbeats" and h == 1:
        kwargs.setdefault("stack_types", ["identity"])
        kwargs.setdefault("n_blocks", [1])
        kwargs.setdefault("mlp_units", [[512, 512]])
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

    def __init__(self,
                 h,
                 input_size,
                 dropout: float = 0.1,
                 futr_exog_list=None,
                 hist_exog_list=None,
                 stat_exog_list=None,
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
                 gate='mlp',  # ['mlp','attention','linear','rnn']
                 pooling: str = 'dense',  # ['dense','sparse','soft','ste']
                 k: int = 3,  # Number of top experts for sparse pooling
                 temperature: float = 1.0,  # Temperature for soft pooling
                 gate_loss_type: str = 'ib_softmax_mse',  # ['ib_softmax_mse','softmax_mse','kl']
                 add_specialization_loss: bool = False,
                 specialization_factor: float = 0.2,
                 add_ncl_loss: bool = False,
                 ncl_factor: float = 0.1,
                 add_balance_loss: bool = False,
                 balance_factor: float = 0.01,
                 include_combined_loss: bool = False,
                 anneal_temperature: bool = False,
                 total_loss_type: str = 'annealing',  # ['random', 'annealing','sumsqr']
                 annealing_temperature: int = 1000,
                 **trainer_kwargs):

        super(NSX, self).__init__(h=h,
                                  input_size=input_size,
                                  stat_exog_list=None,
                                  futr_exog_list=None,
                                  hist_exog_list=None,
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
                                  # lr_scheduler=lr_scheduler,
                                  # lr_scheduler_kwargs=lr_scheduler_kwargs,
                                  # dataloader_kwargs=dataloader_kwargs,
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
        self.current_step = 0

        if gate == 'mlp':
            self.gate = MLPLayer(
                self.input_size,
                self.num_experts,
                activation="ReLU",
                hidden_size=32,
                num_layers=1,
                dropout=0.1)
        elif gate == 'attention':
            self.gate = AttentionGate(self.input_size, self.num_experts)
        elif gate == 'linear':
            self.gate = nn.Linear(self.input_size, self.num_experts, bias=False)
        else:
            self.gate = RNNGate(input_size=self.input_size, num_experts=self.num_experts, )

        self.softmax = nn.Softmax(dim=1)
        k = min(k, self.num_experts)
        self.k = k
        self.temperature = temperature
        if pooling == 'dense':
            self.pooling = DensePooling()
        elif pooling == 'sparse':
            self.pooling = SparsePooling(k=k)
        elif pooling == 'soft':
            self.pooling = SoftPooling(temperature=temperature)
        elif pooling == 'ste':
            self.pooling = StraightThroughPooling()
        else:
            raise ValueError(f"Unknown pooling={pooling!r}; expected 'dense', 'sparse', 'soft', or 'ste'")
        self.add_specialization_loss = add_specialization_loss
        self.specialization_factor = specialization_factor
        self.add_ncl_loss = add_ncl_loss
        self.ncl_factor = ncl_factor
        self.add_balance_loss = add_balance_loss
        self.balance_factor = balance_factor
        self.include_combined_loss = include_combined_loss
        self.anneal_temperature = anneal_temperature

        self.gate_loss_type = gate_loss_type
        self.total_loss_type = total_loss_type
        self.annealing_temperature = annealing_temperature

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
        # Training (return_components) always mixes densely so every expert gets a
        # gradient. Sparse / STE / temperature pooling is inference-only.
        gate_weights = full_gate_weights if return_components else self.pooling(gate_logits)

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
        y_idx = batch["y_idx"]
        temporal_cols = batch["temporal_cols"]
        windows_temporal, static, static_cols = self._create_windows(batch, step="train")
        windows = self._sample_windows(
            windows_temporal, static, static_cols, temporal_cols, step="train"
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

        windows_batch = dict(
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )

        # Get predictions and component analysis
        output, expert_outputs, gate_weights = self(windows_batch, return_components=True)

        # Compute individual expert losses
        expert_losses = []
        for i in range(expert_outputs.size(1)):  # Loop through each expert
            expert_output = expert_outputs[:, i].unsqueeze(-1)
            if self.loss.is_distribution_output:
                y_loc, y_scale = self._get_loc_scale(y_idx)
                distr_args = self.loss.scale_decouple(output=expert_output, loc=y_loc, scale=y_scale)
                expert_loss = self.loss(y=original_outsample_y, distr_args=distr_args, mask=outsample_mask)
            else:
                try:
                    expert_loss = self.loss(y=outsample_y, y_hat=expert_output, y_hat_c=output, mask=outsample_mask)
                except TypeError:
                    expert_loss = self.loss(y=outsample_y, y_hat=expert_output, mask=outsample_mask)

            expert_losses.append(expert_loss)

        expert_losses_tensor = torch.stack(expert_losses)  # [num_experts]
        expert_loss = expert_losses_tensor.mean()

        temperature = self.get_temperature() if self.anneal_temperature else 1

        if self.gate_loss_type == 'ib_softmax_mse':
            # Softmax-based weighting
            expert_scores = -expert_losses_tensor  # Convert losses to scores
            target_weights = F.softmax(expert_scores / temperature, dim=0)

            target_weights_expanded = target_weights.unsqueeze(0).expand(gate_weights.size(0), -1)

            # Compute per-instance gate loss
            gate_loss = F.mse_loss(gate_weights, target_weights_expanded)
        elif self.gate_loss_type == 'softmax_mse':
            expert_scores = -expert_losses_tensor  # Convert losses to scores
            target_weights = F.softmax(expert_scores / temperature, dim=0)

            gate_loss = F.mse_loss(gate_weights.mean(0), target_weights)
        elif self.gate_loss_type == 'ib_softmax_mse_grad':
            #  gradient trick for gate weights computation
            # Compute sign of the overall error

            # error_sign = torch.sign(output - outsample_y)  # [batch, horizon]
            #
            # # Compute mean error direction per expert across batch
            # expert_directions = []
            # for i in range(expert_outputs.size(1)):
            #     expert_output = expert_outputs[:, i]  # [batch, horizon]
            #     direction = (error_sign * expert_output).mean()  # Scalar
            #     expert_directions.append(direction)
            #
            # expert_scores = -torch.tensor(expert_directions, device=output.device)  # [num_experts]
            # target_weights = F.softmax(expert_scores / temperature, dim=0)  # [num_experts]

            # More explicit about what we want
            # expert_contributions = -(output - outsample_y).unsqueeze(1) * expert_outputs  # Higher is better
            # WITH gradient trick
            expert_contributions = (
                -torch.sign(output.squeeze(-1) - outsample_y.squeeze(-1)).unsqueeze(1)
                * expert_outputs
            )

            expert_scores = expert_contributions.mean(dim=[0, -1])  # Average over batch and horizon
            target_weights = F.softmax(expert_scores / temperature, dim=0)

            # Expand to match batch size
            target_weights_expanded = target_weights.unsqueeze(0).expand(gate_weights.size(0), -1)

            # Compute gate loss
            gate_loss = F.mse_loss(gate_weights, target_weights_expanded)
        else:
            expert_scores = -expert_losses_tensor  # Convert losses to scores
            target_weights = expert_scores / expert_scores.sum()
            # target_weights = expert_scores / expert_scores.sum(dim=0)
            # target_weights_expanded = target_weights.unsqueeze(0).expand(gate_weights.size(0), -1)

            gate_loss = F.mse_loss(gate_weights.mean(0), target_weights)
            # gate_loss = F.mse_loss(gate_weights, target_weights_expanded)

        # Compute combined loss
        if self.loss.is_distribution_output:
            y_loc, y_scale = self._get_loc_scale(y_idx)
            distr_args = self.loss.scale_decouple(output=output, loc=y_loc, scale=y_scale)
            combined_loss = self.loss(y=original_outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            combined_loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

        if self.total_loss_type == 'random':
            p = np.random.random()
            total_loss = p * expert_loss + (1 - p) * gate_loss
        elif self.total_loss_type == 'annealing':
            expert_weight = min(1.0,
                                self.current_step / self.annealing_temperature)  # Gradually increase expert importance
            total_loss = expert_weight * expert_loss + (1 - expert_weight) * gate_loss
        elif self.total_loss_type == 'sum':
            # sum
            total_loss = (expert_loss + gate_loss)
        else:
            print(self.total_loss_type)
            raise ValueError("self.total_loss_type")

        if self.add_specialization_loss:
            spec_loss = self.specialization_loss(expert_outputs, gate_weights)
            total_loss += self.specialization_factor * spec_loss  # Small weight for specialization

        if self.add_ncl_loss:
            ncl = self.ncl_loss(expert_outputs)
            total_loss += self.ncl_factor * ncl
            self.log("ncl_loss", ncl.detach(), batch_size=outsample_y.size(0), on_epoch=True)

        if self.add_balance_loss:
            bal = self.balance_loss(gate_weights)
            total_loss += self.balance_factor * bal
            self.log("balance_loss", bal.detach(), batch_size=outsample_y.size(0), on_epoch=True)

        if self.include_combined_loss:
            total_loss = total_loss + combined_loss

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
            print("output", torch.isnan(output).sum())
            raise Exception("Loss is NaN, training stopped.")

        self.log(
            "train_loss",
            total_loss.detach().item(),
            batch_size=outsample_y.size(0),
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append((self.global_step, total_loss.detach().item()))

        self.current_step += 1
        return total_loss

    def get_temperature(self):
        init_temp = 2.0
        final_temp = 0.5
        progress = min(1.0, self.current_step / self.annealing_temperature)
        return max(final_temp, init_temp - progress * (init_temp - final_temp))

    def ncl_loss(self, expert_outputs):
        """Negative correlation learning: penalize experts that co-vary with the ensemble."""
        num_experts = expert_outputs.size(1)
        ensemble_mean = expert_outputs.mean(dim=1, keepdim=True)

        ncl_terms = []
        for i in range(num_experts):
            expert_pred = expert_outputs[:, i:i + 1]
            others = torch.cat([expert_outputs[:, :i], expert_outputs[:, i + 1:]], dim=1)
            others_mean = others.mean(dim=1, keepdim=True)
            correlation = ((expert_pred - ensemble_mean) * (others_mean - ensemble_mean)).mean()
            ncl_terms.append(correlation)

        return torch.stack(ncl_terms).mean()

    def balance_loss(self, gate_weights):
        """KL of mean gate usage toward uniform, minus entropy so the gate does not collapse."""
        expert_usage = gate_weights.mean(0)
        target_usage = torch.ones_like(expert_usage) / gate_weights.size(1)
        kl = F.kl_div(expert_usage.log(), target_usage, reduction='batchmean')
        entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(1).mean()
        return kl - 0.1 * entropy

    def specialization_loss(self, expert_outputs, gate_weights):
        """
        Encourage each expert to specialize by penalizing when multiple experts
        make similar predictions on samples where they have high gate weights
        """
        # Get pairwise differences between expert predictions
        # [batch, num_experts, num_experts, horizon]
        pairwise_diffs = (expert_outputs.unsqueeze(2) - expert_outputs.unsqueeze(1)).abs()

        # Weight the differences by gate weights
        # Only care when both experts have high weights
        gate_products = gate_weights.unsqueeze(2) * gate_weights.unsqueeze(1)

        # Compute loss - we want large differences when gate weights are high
        spec_loss = -(pairwise_diffs.mean(-1) * gate_products).mean()

        return spec_loss
