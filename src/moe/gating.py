import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralforecast.common._modules import MLP as MLPLayer

GATE_NAMES = ("linear", "linear_bias", "mlp")

GATE_LOSS_TYPES = (
    "ib_softmax_mse",
    "softmax_mse",
    "ib_softmax_mse_grad",
    "ib_softmax_mse_window",
    "ib_softmax_mse_grad_window",
)

_WINDOW_GATE_LOSSES = (
    "ib_softmax_mse_window",
    "ib_softmax_mse_grad_window",
)


def validate_gate_loss_type(gate_loss_type):
    if gate_loss_type not in GATE_LOSS_TYPES:
        expected = ", ".join(repr(name) for name in GATE_LOSS_TYPES)
        raise ValueError(
            f"Unknown gate_loss_type={gate_loss_type!r}; expected {expected}"
        )


def build_gate(kind, input_size, num_experts):
    if kind == "linear":
        return nn.Linear(input_size, num_experts, bias=False)
    if kind == "linear_bias":
        return nn.Linear(input_size, num_experts, bias=True)
    if kind == "mlp":
        return MLPLayer(
            input_size,
            num_experts,
            activation="ReLU",
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
        )
    expected = ", ".join(repr(name) for name in GATE_NAMES)
    raise ValueError(f"Unknown gate={kind!r}; expected {expected}")


def _detached_softmax(scores, dim):
    return F.softmax(scores, dim=dim).detach()


def _mse_broadcast(gate_weights, target):
    expanded = target.unsqueeze(0).expand(gate_weights.size(0), -1)
    return F.mse_loss(gate_weights, expanded)


def _signed_contributions(output, outsample_y, expert_outputs):
    return (
        -torch.sign(output.squeeze(-1) - outsample_y.squeeze(-1)).unsqueeze(1)
        * expert_outputs
    )


def gate_supervision_loss(
    gate_loss_type,
    gate_weights,
    expert_losses,
    per_window_losses,
    output,
    outsample_y,
    expert_outputs,
):
    """MSE between the dense gate softmax and a detached target.

    The forecast uses the pooled mixture. This loss supervises the dense
    softmax, so every expert stays in the gradient. ``gate_weights`` must be
    that dense softmax, not the pooled mixture.
    """
    validate_gate_loss_type(gate_loss_type)
    if gate_loss_type in _WINDOW_GATE_LOSSES and not per_window_losses:
        raise ValueError(f"{gate_loss_type} requires a point forecast loss")

    if gate_loss_type == "ib_softmax_mse":
        target = _detached_softmax(-expert_losses, dim=0)
        return _mse_broadcast(gate_weights, target)
    if gate_loss_type == "softmax_mse":
        target = _detached_softmax(-expert_losses, dim=0)
        return F.mse_loss(gate_weights.mean(0), target)
    if gate_loss_type == "ib_softmax_mse_grad":
        scores = _signed_contributions(output, outsample_y, expert_outputs).mean(dim=[0, -1])
        target = _detached_softmax(scores, dim=0)
        return _mse_broadcast(gate_weights, target)
    if gate_loss_type == "ib_softmax_mse_window":
        per_window = torch.stack(per_window_losses, dim=1)
        target = _detached_softmax(-per_window, dim=1)
        return F.mse_loss(gate_weights, target)
    if gate_loss_type == "ib_softmax_mse_grad_window":
        scores = _signed_contributions(output, outsample_y, expert_outputs).mean(dim=-1)
        target = _detached_softmax(scores, dim=1)
        return F.mse_loss(gate_weights, target)
    raise ValueError(f"Unknown gate_loss_type={gate_loss_type!r}")
