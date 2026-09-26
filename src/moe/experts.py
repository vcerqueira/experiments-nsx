import torch.nn as nn
from neuralforecast.models import MLP, KAN, NBEATS

EXPERT_REGISTRY = {
    "mlp": MLP,
    "kan": KAN,
    "nbeats": NBEATS,
}


def expert_init_kwargs(expert_arch, expert_kwargs):
    kwargs = dict(expert_kwargs or {})
    if expert_arch == "nbeats":
        kwargs.setdefault("stack_types", ["identity"])
        kwargs.setdefault("n_blocks", [1])
        kwargs.setdefault("mlp_units", [[128, 128]])
    elif expert_arch == "kan":
        kwargs.setdefault("hidden_size", 256)
    elif expert_arch == "mlp":
        kwargs.setdefault("hidden_size", 128)
    return kwargs


def build_experts(h, input_size, num_experts, expert_arch, expert_kwargs, random_seed):
    if expert_arch not in EXPERT_REGISTRY:
        raise ValueError(
            f"Unknown expert_arch={expert_arch!r}; expected one of {sorted(EXPERT_REGISTRY)}"
        )
    if num_experts < 1:
        raise ValueError(f"num_experts must be >= 1, got {num_experts}")
    cls = EXPERT_REGISTRY[expert_arch]
    kwargs = expert_init_kwargs(expert_arch, expert_kwargs)
    # Distinct seeds. A shared draw from 1..1000 can initialize two experts identically.
    return nn.ModuleList([
        cls(
            h=h,
            input_size=input_size,
            random_seed=random_seed + index + 1,
            **kwargs,
        )
        for index in range(num_experts)
    ])
