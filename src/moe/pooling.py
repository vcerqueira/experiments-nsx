from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Hidden dimension D as shown in the paper
        """
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)  # W ∈ R^{D×1}
        self.V = nn.Linear(input_dim, hidden_dim)  # V ∈ R^{D×1}

    def swish(self, x: torch.Tensor) -> torch.Tensor:
        """Swish activation function: x * sigmoid(x)"""
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements h_t = SwiGLU(x_t) = Swish(Wx_t) ⊗ (Vx_t)
        where ⊗ represents element-wise multiplication
        """
        return self.swish(self.W(x)) * self.V(x)


class BasePooling(nn.Module):
    """
    Base class for pooling strategies in a Mixture-of-Experts setting.
    It holds a list of expert modules and a gating network.
    """

    def __init__(
            self,
            experts: List[nn.Module],
            gate: nn.Module,
            out_features: int,
            device: Optional[torch.device] = None
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features (each expert’s output size).
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
        """
        super(BasePooling, self).__init__()

        self.experts: nn.ModuleList = nn.ModuleList(experts)
        self.gate: nn.Module = gate
        self.out_features: int = out_features
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.softmax: nn.Softmax = nn.Softmax(dim=1)


    def forward(self, insample_y: torch.Tensor) -> torch.Tensor:
        """
        Forward method to be implemented by subclasses.

        Args:
            insample_y (torch.Tensor): Input tensor with shape (batch_size, input_size).

        Returns:
            torch.Tensor: The weighted sum of experts' outputs.
        """
        raise NotImplementedError("Subclasses should implement this method.")


###############################################################################
# Dense Pooling
###############################################################################
class DensePooling(BasePooling):
    """
    Dense pooling uses all experts. The gate is computed and softmax-normalized
    (similar to your provided snippet), then each expert’s output is weighted
    accordingly and summed.
    """

    def forward(self, windows_batch: dict) -> torch.Tensor:
        insample_y = windows_batch['insample_y']
        # Compute the gate and normalize it.
        gate_logits: torch.Tensor = self.gate(insample_y)
        gate_probs: torch.Tensor = self.softmax(gate_logits)

        # Initialize the weighted sum.
        weighted_sum: torch.Tensor = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )
        # Sum over all experts.
        for i, expert in enumerate(self.experts):
            weighted_sum += gate_probs[:, i].unsqueeze(1) * expert(windows_batch)
        return weighted_sum


###############################################################################
# Sparse Pooling
###############################################################################
class SparsePooling(BasePooling):
    """
    Sparse pooling uses only the top-k experts (as determined by the gating network).
    """

    def __init__(
            self,
            experts: List[nn.Module],
            gate: nn.Module,
            out_features: int,
            k: int = 3,
            device: Optional[torch.device] = None
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features.
            k (int, optional): The number of top experts to select. Defaults to 1.
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
        """
        super(SparsePooling, self).__init__(experts, gate, out_features, device)
        self.k: int = k
        # Add SwiGLU embedding layer
        self.embedding = SwiGLU(
            input_dim=self.experts[0].input_size,  # Assuming all experts have same input size
            hidden_dim=18
        )



    def forward2(self, windows_batch: dict) -> torch.Tensor:


        insample_y = windows_batch['insample_y']

        # embedded_input = self.embedding(insample_y)  # Shape: [batch, embedding_dim]
        # windows_batch = {**windows_batch, 'insample_y': embedded_input}

        # Compute the gate logits. Shape: [batch, num_experts]
        gate_logits: torch.Tensor = self.gate(insample_y)

        # Select the top-k experts for each sample.
        # topk_values & topk_indices have shape: [batch, k]
        topk_values, topk_indices = torch.topk(gate_logits, self.k, dim=1)

        # Compute probabilities for the top-k experts using softmax.
        gate_probs: torch.Tensor = self.softmax(topk_values)

        # Initialize the weighted sum output.
        weighted_sum = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )

        num_experts = len(self.experts)
        # Group contributions by expert.
        for expert_idx in range(num_experts):

            # Create a mask of shape [batch, k] indicating where expert_idx was selected.
            expert_mask = (topk_indices == expert_idx)
            if expert_mask.sum() == 0:
                continue  # This expert was not selected in any top-k.

            # Sum the corresponding probabilities for each sample.
            # This gives a weight for each sample for expert_idx.
            expert_weight = (gate_probs * expert_mask.float()).sum(dim=1)  # Shape: [batch]
            # print('expert_weight')
            # print(expert_weight)

            # Compute expert output for the entire batch.
            # print("self.experts[expert_idx]")
            # print(self.experts[expert_idx])
            # print(windows_batch.keys())
            expert_output = self.experts[expert_idx](windows_batch)  # Shape: [batch, out_features]
            # print("expert_output")
            # print(expert_output)

            # Add the weighted expert output.
            weighted_sum += expert_output * expert_weight.unsqueeze(1)

        return weighted_sum

    def forward3(self, windows_batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple containing:
            - combined_output: Final weighted sum [batch, out_features]
            - expert_outputs: Individual expert predictions [batch, num_experts, out_features]
            - gate_weights: Gate weights for each expert [batch, num_experts]
        """
        insample_y = windows_batch['insample_y']
        batch_size = insample_y.size(0)

        # Compute gate logits and top-k selection
        gate_logits: torch.Tensor = self.gate(insample_y)
        topk_values, topk_indices = torch.topk(gate_logits, self.k, dim=1)
        topk_probs: torch.Tensor = self.softmax(topk_values)

        # Initialize full gate weights tensor (for all experts)
        gate_weights = torch.zeros(
            batch_size, len(self.experts),
            device=insample_y.device
        )



        # Initialize tensor to store all expert outputs
        expert_outputs = torch.zeros(
            batch_size, len(self.experts), self.out_features,
            device=insample_y.device
        )

        # Compute outputs for all experts
        for expert_idx, expert in enumerate(self.experts):
            # Get expert predictions
            expert_output = expert(windows_batch)
            expert_outputs[:, expert_idx] = expert_output

            # Create mask for this expert's selection
            expert_mask = (topk_indices == expert_idx)
            if expert_mask.sum() == 0:
                continue

            # Convert top-k probabilities to full expert weights
            expert_weight = (topk_probs * expert_mask.float()).sum(dim=1)
            gate_weights[:, expert_idx] = expert_weight

        # Compute final weighted sum
        combined_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)

        return combined_output, expert_outputs, gate_weights

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
            expert_outputs = torch.zeros(
                batch_size, len(self.experts), self.out_features,
                device=insample_y.device
            )

            # Compute all expert outputs for analysis
            for expert_idx, expert in enumerate(self.experts):
                expert_output = expert(windows_batch)
                expert_outputs[:, expert_idx] = expert_output

            combined_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
            return combined_output, expert_outputs, gate_weights

        else:
            # Original efficient implementation for inference
            weighted_sum = torch.zeros(batch_size, self.out_features, device=insample_y.device)

            for expert_idx, expert in enumerate(self.experts):
                expert_output = expert(windows_batch)
                weighted_sum += expert_output * gate_weights[:, expert_idx].unsqueeze(1)

            return weighted_sum


###############################################################################
# Soft Pooling
###############################################################################
#### TODO: DO THIS ONE
class SoftPooling(BasePooling):
    """
    Soft pooling applies a temperature-scaled softmax to the gate before weighting
    and summing the experts' outputs.
    """

    def __init__(
            self,
            experts: List[nn.Module],
            gate: nn.Module,
            out_features: int,
            temperature: float = 1.0,
            device: Optional[torch.device] = None
    ) -> None:
        """
        Args:
            experts (List[nn.Module]): List of expert models.
            gate (nn.Module): Gating network that computes the weights.
            out_features (int): The number of output features.
            temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.
            device (Optional[torch.device], optional): Device to run on. Defaults to CPU.
        """
        super(SoftPooling, self).__init__(experts, gate, out_features, device)
        self.temperature: float = temperature

    def forward(self, windows_batch: dict) -> torch.Tensor:
        insample_y = windows_batch['insample_y']

        # Compute the gate logits and apply temperature-scaled softmax.
        gate_logits: torch.Tensor = self.gate(insample_y)
        soft_gate: torch.Tensor = F.softmax(gate_logits / self.temperature, dim=1)

        weighted_sum: torch.Tensor = torch.zeros(
            insample_y.size(0), self.out_features, device=insample_y.device
        )
        # Sum over all experts.
        for i, expert in enumerate(self.experts):
            weighted_sum += soft_gate[:, i].unsqueeze(1) * expert(windows_batch)
        return weighted_sum
