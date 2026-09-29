import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralforecast.common._modules import MLP as MLPLayer

GATE_NAMES = ("linear", "linear_bias", "mlp")

# Grad targets are -residual * feature.
# residual: sign(mixture - y), or the residual itself.
# feature: expert forecast, expert - y, or expert - mixture.
_GRAD_GATE_SPECS = {
    "ib_softmax_mse_grad": ("sign", "level", False),
    "ib_softmax_mse_grad_window": ("sign", "level", True),
    "ib_softmax_mse_grad_error": ("sign", "error", False),
    "ib_softmax_mse_grad_error_window": ("sign", "error", True),
    "ib_softmax_mse_grad_advantage": ("sign", "advantage", False),
    "ib_softmax_mse_grad_advantage_window": ("sign", "advantage", True),
    "ib_softmax_mse_grad_resid": ("value", "level", False),
    "ib_softmax_mse_grad_resid_window": ("value", "level", True),
    "ib_softmax_mse_grad_resid_error": ("value", "error", False),
    "ib_softmax_mse_grad_resid_error_window": ("value", "error", True),
    "ib_softmax_mse_grad_resid_advantage": ("value", "advantage", False),
    "ib_softmax_mse_grad_resid_advantage_window": ("value", "advantage", True),
}

GATE_LOSS_TYPES = (
    "ib_softmax_mse",
    "softmax_mse",
    "ib_softmax_mse_window",
    *_GRAD_GATE_SPECS,
)

_WINDOW_GATE_LOSSES = (
    "ib_softmax_mse_window",
    *(name for name, (_, _, window) in _GRAD_GATE_SPECS.items() if window),
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


def _grad_scores(output, outsample_y, expert_outputs, residual, feature):
    """Negative mixture gradient of a point loss with respect to expert weights."""
    mixture = output.squeeze(-1)
    target = outsample_y.squeeze(-1)
    resid = mixture - target
    if residual == "sign":
        resid = torch.sign(resid)
    elif residual != "value":
        raise ValueError(f"Unknown grad residual={residual!r}")
    if feature == "level":
        values = expert_outputs
    elif feature == "error":
        values = expert_outputs - target.unsqueeze(1)
    elif feature == "advantage":
        values = expert_outputs - mixture.unsqueeze(1)
    else:
        raise ValueError(f"Unknown grad feature={feature!r}")
    return -resid.unsqueeze(1) * values


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
    if gate_loss_type == "ib_softmax_mse_window":
        per_window = torch.stack(per_window_losses, dim=1)
        target = _detached_softmax(-per_window, dim=1)
        return F.mse_loss(gate_weights, target)
    if gate_loss_type in _GRAD_GATE_SPECS:
        residual, feature, per_window = _GRAD_GATE_SPECS[gate_loss_type]
        scores = _grad_scores(output, outsample_y, expert_outputs, residual, feature)
        if per_window:
            target = _detached_softmax(scores.mean(dim=-1), dim=1)
            return F.mse_loss(gate_weights, target)
        target = _detached_softmax(scores.mean(dim=[0, -1]), dim=0)
        return _mse_broadcast(gate_weights, target)
    raise ValueError(f"Unknown gate_loss_type={gate_loss_type!r}")
