import torch
import torch.nn as nn
import torch.nn.functional as F


class DensePooling(nn.Module):
    """Softmax over all experts."""

    def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(gate_logits, dim=1)


class SparsePooling(nn.Module):
    """Top-k experts; softmax only over those k, zeros elsewhere."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

    def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
        topk_values, topk_indices = torch.topk(gate_logits, k=self.k, dim=1)
        gate_weights = torch.zeros_like(gate_logits)
        gate_weights.scatter_(1, topk_indices, F.softmax(topk_values, dim=1))
        return gate_weights


class SoftPooling(nn.Module):
    """Temperature-scaled softmax over all experts."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(gate_logits / self.temperature, dim=1)

#
# class StraightThroughPooling(nn.Module):
#     """Hard one-hot in the forward pass, softmax gradients in the backward pass."""
#
#     def __init__(self, temperature: float = 1.0, hard: bool = True):
#         super().__init__()
#         self.temperature = temperature
#         self.hard = hard
#
#     def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
#         gates_soft = F.softmax(gate_logits / self.temperature, dim=-1)
#         if not self.hard:
#             return gates_soft
#
#         gates_hard = F.one_hot(
#             gates_soft.argmax(dim=-1),
#             num_classes=gate_logits.size(-1),
#         ).float()
#         return (gates_hard - gates_soft).detach() + gates_soft
