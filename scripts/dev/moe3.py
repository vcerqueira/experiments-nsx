from typing import Tuple, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.common._modules import MLP as MLPLayer
from neuralforecast.models import MLP as MLP

from src.pooling import SparsePooling

from neuralforecast.models.nbeats import NBEATS


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
            self.gate = MLPLayer(
                self.input_size,
                self.num_experts,
                activation="Sigmoid",
                hidden_size=32,
                num_layers=1,
                dropout=0.1)

            # self.gate = nn.Linear(self.input_size, self.num_experts, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.k = 3

        # if pooling is not None:
        #     self.pooling = pooling
        # else:
        #     self.pooling = SparsePooling(self.experts, self.gate, self.h, k=2)

    def forward(self, windows_batch: dict, return_components: bool = False) -> Union[
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

    @staticmethod
    def negative_minmax_norm(x: torch.Tensor) -> torch.Tensor:
        """
        Performs min-max normalization on the negative of a vector.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Normalized tensor in [0,1] range
        """
        neg_x = -x  # Take negative of input
        min_val = neg_x.min()
        max_val = neg_x.max()

        # Min-max normalization: (x - min)/(max - min)
        normalized = (neg_x - min_val) / (max_val - min_val)
        return normalized

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

        # target_weights = 1.0 / (expert_losses_tensor + 1e-10)  # Add epsilon for numerical stability
        # target_weights = target_weights / target_weights.sum()

        # Softmax-based weighting
        temperature=1
        expert_scores = -expert_losses_tensor  # Convert losses to scores
        target_weights = F.softmax(expert_scores / temperature, dim=0)
        # target_weights = self.negative_minmax_norm(expert_losses_tensor)

        # gate_loss = F.kl_div(
        #     gate_weights.mean(0).log(),  # Average predicted weights across batch
        #     target_weights,
        #     reduction='batchmean'
        # )
        gate_loss = F.mse_loss(gate_weights.mean(0),target_weights)

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

        # Gate balance loss - encourage uniform expert utilization
        # gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(1).mean()
        # target_usage = torch.ones_like(gate_weights) / gate_weights.size(1)
        # balance_loss = F.kl_div(
        #     gate_weights.log(),
        #     target_usage,
        #     reduction='batchmean'
        # )

        # Combine losses with weights
        # expert_loss = torch.stack(expert_losses).mean()
        # gate_loss = balance_loss - 0.1 * gate_entropy  # Encourage diversity with negative entropy
        # print("expert_loss")
        # print(expert_loss)
        # print("gate_loss")
        # print(gate_loss)

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

        self.current_step+=1
        return total_loss
