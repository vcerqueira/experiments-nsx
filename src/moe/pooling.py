import torch
import torch.nn as nn
import torch.nn.functional as F

POOLING_NAMES = ("dense", "sparse")


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


def build_pooling(name: str, k: int) -> nn.Module:
    if name == "dense":
        return DensePooling()
    if name == "sparse":
        return SparsePooling(k=k)

    expected = ", ".join(repr(item) for item in POOLING_NAMES)
    raise ValueError(f"Unknown pooling={name!r}; expected {expected}")
