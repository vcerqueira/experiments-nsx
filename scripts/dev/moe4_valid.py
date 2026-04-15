# backup...this one works
from typing import Tuple, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.common._modules import MLP as MLPLayer, AttentionLayer
from neuralforecast.models import MLP as MLP

from src.pooling import SparsePooling

from neuralforecast.models.nbeats import NBEATS


class EnhancedAttentionGate(nn.Module):
    def __init__(self, input_size, num_experts, n_heads=4, d_model=64, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(1, d_model)  # Project scalar values to d_model

        # Multi-head attention
        self.attention = FullAttention(
            mask_flag=True,  # Use causal masking
            attention_dropout=dropout,
            output_attention=True
        )

        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_experts)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_length]
        B, L = x.shape
        H = self.n_heads

        # Project input
        x = x.unsqueeze(-1)  # [B, L, 1]
        x = self.input_proj(x)  # [B, L, d_model]

        # Prepare Q, K, V
        q = self.q_proj(x).view(B, L, H, -1)  # [B, L, H, d_model/H]
        k = self.k_proj(x).view(B, L, H, -1)  # [B, L, H, d_model/H]
        v = self.v_proj(x).view(B, L, H, -1)  # [B, L, H, d_model/H]

        # Apply attention
        out, attn = self.attention(q, k, v, attn_mask=None)  # [B, L, H, d_model/H]

        # Reshape and project to num_experts
        out = out.contiguous().view(B, L, self.d_model)  # [B, L, d_model]
        out = out[:, -1]  # Take last timestep: [B, d_model]

        return self.out_proj(out)  # [B, num_experts]




class RNNGate(nn.Module):
    def __init__(self, input_size, num_experts, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,  # Process one timestep at a time
            hidden_size=hidden_size,
            batch_first=True
        )
        self.proj = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # x shape: [batch_size, seq_length]
        x = x.unsqueeze(-1)  # Add feature dim: [batch, seq, 1]
        _, h_n = self.gru(x)  # h_n shape: [1, batch, hidden]
        return self.proj(h_n.squeeze(0))  # [batch, num_experts]
class AttentionGate(nn.Module):
    def __init__(self, input_size, num_experts, hidden_size=32):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_proj = nn.Linear(1, hidden_size)
        self.value_proj = nn.Linear(1, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # x shape: [batch_size, seq_length]
        x = x.unsqueeze(-1)  # [batch, seq, 1]
        keys = self.key_proj(x)  # [batch, seq, hidden]
        values = self.value_proj(x)  # [batch, seq, hidden]

        # Compute attention scores
        scores = torch.matmul(keys, self.query) / math.sqrt(keys.size(-1))
        attn_weights = F.softmax(scores, dim=1)  # [batch, seq]

        # Weighted sum of values
        context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)
        return self.out_proj(context)


class SimpleMoe(BaseWindows):
    """
    Simple Mixture of Experts (MoE) model for time series forecasting.
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
                 gate=None,
                 pooling=None,
                 **trainer_kwargs):

        super(SimpleMoe, self).__init__(h=h,
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
                                        lr_scheduler=lr_scheduler,
                                        lr_scheduler_kwargs=lr_scheduler_kwargs,
                                        dataloader_kwargs=dataloader_kwargs,
                                        **trainer_kwargs)

        self.input_size = input_size
        self.h = h

        if experts is not None:
            self.experts = experts
        else:
            self.experts = nn.ModuleList([
                MLP(h=self.h, input_size=self.input_size, random_seed=random.randint(1, 1000)),
                MLP(h=self.h, input_size=self.input_size, random_seed=random.randint(1, 1000)),
                MLP(h=self.h, input_size=self.input_size, random_seed=random.randint(1, 1000)),
                MLP(h=self.h, input_size=self.input_size, random_seed=random.randint(1, 1000)),
                MLP(h=self.h, input_size=self.input_size, random_seed=random.randint(1, 1000)),
                MLP(h=self.h, input_size=self.input_size, random_seed=random.randint(1, 1000)),
            ])

        self.num_experts = len(self.experts)
        self.current_step = 0

        if gate is not None:
            self.gate = gate
        else:
            # self.gate = MLPLayer(
            #     self.input_size,
            #     self.num_experts,
            #     activation="Sigmoid",
            #     hidden_size=32,
            #     num_layers=1,
            #     dropout=0.1)

            # self.gate = AttentionGate(self.input_size, self.num_experts)

            # self.gate = EnhancedAttentionGate(
            #     input_size=self.input_size,
            #     num_experts=self.num_experts,
            #     n_heads=4,
            #     d_model=64,
            #     dropout=0.1
            # )

            # self.gate=RNNGate(input_size=self.input_size,
            #     num_experts=self.num_experts,)

            self.gate = nn.Linear(self.input_size, self.num_experts, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.k = 3

        # if pooling is not None:
        #     self.pooling = pooling
        # else:
        #     self.pooling = SparsePooling(self.experts, self.gate, self.h, k=2)

    def get_temperature(self):
        init_temp = 2.0
        final_temp = 0.5
        progress = min(1.0, self.current_step / 1000)  # Anneal over 1000 steps
        return max(final_temp, init_temp - progress * (init_temp - final_temp))

    def forward2(self, windows_batch: dict, return_components: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            windows_batch (dict): Input batch
            return_components (bool): If True, returns individual expert outputs and gate weights
        Returns:
            torch.Tensor if return_components=False: Final weighted predictions
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] if return_components=True:
                (combined_output, expert_outputs, gate_weights)
        """
        insample_y = windows_batch['insample_y']
        batch_size = insample_y.size(0)

        # Compute gate logits and probabilities
        gate_logits = self.gate(insample_y)
        gate_weights = self.softmax(gate_logits)  # [batch_size, num_experts]

        if return_components:
            # Initialize tensor to store all expert outputs
            expert_outputs = torch.zeros(batch_size, len(self.experts), self.h, device=insample_y.device)

            # Compute all expert outputs for analysis
            for expert_idx, expert in enumerate(self.experts):
                expert_output = expert(windows_batch)
                expert_outputs[:, expert_idx] = expert_output

            combined_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
            return combined_output, expert_outputs, gate_weights

        else:
            # Efficient implementation for inference
            weighted_sum = torch.zeros(batch_size, self.h, device=insample_y.device)

            for expert_idx, expert in enumerate(self.experts):
                expert_output = expert(windows_batch)
                weighted_sum += expert_output * gate_weights[:, expert_idx].unsqueeze(1)

            return weighted_sum

    def forward(self, windows_batch: dict, return_components: bool = False):
        insample_y = windows_batch['insample_y']
        batch_size = insample_y.size(0)

        # Compute gate logits and full probabilities for training
        gate_logits = self.gate(insample_y)
        full_gate_weights = self.softmax(gate_logits)  # [batch_size, num_experts]

        # Get top-k for sparse routing
        topk_values, topk_indices = torch.topk(gate_logits, k=3, dim=1)
        sparse_gate_weights = torch.zeros_like(full_gate_weights)

        sparse_gate_weights.scatter_(1, topk_indices, self.softmax(topk_values))

        # temp
        # temperature1 = self.get_temperature()
        # sparse_gate_weights.scatter_(1, topk_indices,
        #                              self.softmax(topk_values / temperature1))

        if return_components:
            expert_outputs = torch.zeros(batch_size, len(self.experts), self.h, device=insample_y.device)

            # Use enumerate to get both index and expert module
            for expert_idx, expert_module in enumerate(self.experts):
                # Only compute for experts used in sparse routing
                if (topk_indices == expert_idx).any():
                    expert_output = expert_module(windows_batch)
                    expert_outputs[:, expert_idx] = expert_output

            combined_output = (expert_outputs * sparse_gate_weights.unsqueeze(-1)).sum(dim=1)
            return combined_output, expert_outputs, full_gate_weights

        else:
            weighted_sum = torch.zeros(batch_size, self.h, device=insample_y.device)

            # Use enumerate to get both index and expert module
            for expert_idx, expert_module in enumerate(self.experts):
                if (topk_indices == expert_idx).any():
                    expert_output = expert_module(windows_batch)
                    weighted_sum += expert_output * sparse_gate_weights[:, expert_idx].unsqueeze(1)

            return weighted_sum
    def get_lambda_div(self):
        """Dynamic diversity weight based on training progress"""
        # Start small and increase
        # return min(0.5, self.current_step / 2000)

        # OR start large and decrease
        return max(0.1, 1.0 - self.current_step / 2000)

    def ncl_loss(self, expert_outputs, target, lambda_div=0.1):
        """
        Implement Negative Correlation Learning
        expert_outputs: [batch_size, num_experts, horizon]
        target: [batch_size, horizon]
        """
        batch_size, num_experts, horizon = expert_outputs.size()

        # Get ensemble mean prediction
        ensemble_mean = expert_outputs.mean(dim=1, keepdim=True)  # [batch_size, 1, horizon]

        # NCL penalty term for each expert
        ncl_terms = []
        for i in range(num_experts):
            expert_pred = expert_outputs[:, i:i + 1]  # [batch_size, 1, horizon]

            # Correlation term: (f_i - f_bar)(f_j - f_bar) for j≠i
            others = torch.cat([expert_outputs[:, :i], expert_outputs[:, i + 1:]], dim=1)
            others_mean = others.mean(dim=1, keepdim=True)  # [batch_size, 1, horizon]

            correlation = (expert_pred - ensemble_mean) * (others_mean - ensemble_mean)
            correlation = correlation.mean()

            ncl_terms.append(correlation)

        ncl_penalty = torch.stack(ncl_terms).mean()
        return ncl_penalty

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

    def training_step(self, batch, batch_idx):
        # Create and normalize windows [Ws, L+H, C]
        windows = self._create_windows(batch, step="train")
        y_idx = batch["y_idx"]
        original_outsample_y = torch.clone(windows["temporal"][:, -self.h:, y_idx])
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
            expert_output = expert_outputs[:, i]
            if self.loss.is_distribution_output:
                _, y_loc, y_scale = self._inv_normalization(
                    y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
                )
                distr_args = self.loss.scale_decouple(output=expert_output, loc=y_loc, scale=y_scale)
                expert_loss = self.loss(y=original_outsample_y, distr_args=distr_args, mask=outsample_mask)
            else:
                expert_loss = self.loss(y=outsample_y, y_hat=expert_output, mask=outsample_mask)
            expert_losses.append(expert_loss)

        expert_losses_tensor = torch.stack(expert_losses)  # [num_experts]
        expert_loss = expert_losses_tensor.mean()

        # Softmax-based weighting
        temperature = 1
        expert_scores = -expert_losses_tensor  # Convert losses to scores
        target_weights = F.softmax(expert_scores / temperature, dim=0)

        target_weights_expanded = target_weights.unsqueeze(0).expand(gate_weights.size(0), -1)

        # Compute per-instance gate loss
        gate_loss = F.mse_loss(gate_weights, target_weights_expanded)
        # gate_loss = F.mse_loss(gate_weights.mean(0),target_weights)

        # print("gate_loss")
        # print(gate_loss)
        # Compute combined loss
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
            )
            distr_args = self.loss.scale_decouple(output=output, loc=y_loc, scale=y_scale)
            combined_loss = self.loss(y=original_outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            combined_loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

        # p = np.random.random()
        # total_loss = p * expert_loss + (1-p) * gate_loss

        # Option 1: Use fixed weights but alternate focus
        # if self.current_step % 2 == 0:
        #     total_loss = expert_loss
        # else:
        #     total_loss = gate_loss
        #     total_loss = gate_loss * expert_loss

        # print("total_loss")
        # print(total_loss)

        # # Option 2: Use annealing schedule
        expert_weight = min(1.0, self.current_step / 1000)  # Gradually increase expert importance
        total_loss = expert_weight * expert_loss + (1 - expert_weight) * gate_loss

        # Compute specialization loss
        # expert_weight = min(1.0, self.current_step / 1000)  # Gradually increase expert importance
        # spec_loss = self.specialization_loss(expert_outputs, gate_weights)
        # total_loss = (expert_weight * expert_loss +
        #               (1 - expert_weight) * gate_loss +
        #               0.1 * spec_loss)  # Small weight for specialization

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
