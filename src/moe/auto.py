from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from neuralforecast.common._base_auto import BaseAuto
from neuralforecast.losses.pytorch import MAE

from src.moe.experts import EXPERT_REGISTRY
from src.moe.gating import GATE_LOSS_TYPES, GATE_NAMES
from src.moe.moe import NSX, NSXFrozen
from src.moe.pooling import POOLING_NAMES


# Training ranges follow the commented search in config_pool. Architecture
# choices are the values each model actually accepts.
_TRAINING_CONFIG = {
    "h": None,
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "scaler_type": tune.choice(["revin", "robust", "standard"]),
    "max_steps": tune.choice([1500, 2000, 2500, 3000]),
    "batch_size": tune.choice([64, 128, 256]),
    "windows_batch_size": tune.choice([128, 256, 512]),
    "loss": None,
    "random_seed": tune.randint(1, 20),
}

_GATE_CONFIG = {
    "pooling": tune.choice(list(POOLING_NAMES)),
    "k": tune.randint(2, 6),
    "gate": tune.choice(list(GATE_NAMES)),
    "gate_loss_type": tune.choice(list(GATE_LOSS_TYPES)),
}


def _with_horizon_sizes(config, h):
    """Turn horizon multipliers into ``input_size`` choices, as Auto models do."""
    config = config.copy()
    translations = {
        "input_size_multiplier": "input_size",
        "inference_input_size_multiplier": "inference_input_size",
    }
    for multiplier_key, size_key in translations.items():
        if multiplier_key not in config:
            continue
        multipliers = config.pop(multiplier_key)
        config[size_key] = tune.choice([h * multiplier for multiplier in multipliers])
    return config


def _maybe_optuna(config, backend):
    if backend == "optuna":
        return BaseAuto._ray_config_to_optuna(config)
    return config


class AutoNSX(BaseAuto):
    """Automatic hyperparameter optimization for ``NSX``.

    ``config=None`` uses ``default_config``. ``h``, ``loss``, and ``valid_loss``
    are taken from the constructor. ``input_size_multiplier`` is a list of
    horizon multiples and is translated into ``input_size``.
    """

    default_config = {
        "input_size_multiplier": [1, 2, 3],
        **_TRAINING_CONFIG,
        "num_experts": tune.choice([10, 15, 20]),
        "expert_arch": tune.choice(list(EXPERT_REGISTRY)),
        **_GATE_CONFIG,
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        time_budget=None,
        refit_with_val=False,
        cpus=None,
        gpus=None,
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
        ray_options=None,
        optuna_options=None,
    ):
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoNSX, self).__init__(
            cls_model=NSX,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            time_budget=time_budget,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
            ray_options=ray_options,
            optuna_options=optuna_options,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = _with_horizon_sizes(cls.default_config, h)
        return _maybe_optuna(config, backend)


def _store_experts(experts):
    """Park pretrained modules in the Ray object store.

    Tune pickles ``AutoNSXFrozen`` into every trial actor, because
    ``BaseAuto._train_tune`` is a bound method. Fitted experts are larger than
    Ray's function-size limit; the actor has to capture an ``ObjectRef``.
    """
    import ray
    from ray import ObjectRef

    if isinstance(experts, ObjectRef):
        return experts
    return ray.put(experts)


def _load_experts(experts):
    import ray
    from ray import ObjectRef

    if isinstance(experts, ObjectRef):
        return ray.get(experts)
    return experts


class AutoNSXFrozen(BaseAuto):
    """Automatic hyperparameter optimization for ``NSXFrozen``.

    ``experts`` are fixed pretrained modules. Every trial uses their ``h`` and
    ``input_size``, so the search covers the gate and the training knobs only.
    With the Ray backend those modules live in the object store, not on this
    instance, so trial actors stay under Ray's function-size limit.
    """

    default_config = {
        **_TRAINING_CONFIG,
        **_GATE_CONFIG,
    }

    def __init__(
        self,
        h,
        experts,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        time_budget=None,
        refit_with_val=False,
        cpus=None,
        gpus=None,
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
        ray_options=None,
        optuna_options=None,
    ):
        input_size = _frozen_input_size(experts, h)
        if config is None:
            config = self.get_default_config(h=h, backend=backend)
        if backend == "ray":
            # Lightning deepcopies every constructor local into ``_hparams_initial``.
            # The name has to already be an ObjectRef before ``super().__init__``.
            experts = _store_experts(experts)
        config = _fix_frozen_config(config, experts, input_size)

        super(AutoNSXFrozen, self).__init__(
            cls_model=NSXFrozen,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            time_budget=time_budget,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
            ray_options=ray_options,
            optuna_options=optuna_options,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        return _maybe_optuna(cls.default_config.copy(), backend)

    def _fit_model(
        self, cls_model, config, dataset, val_size, test_size, distributed_config=None
    ):
        if isinstance(config, dict) and "experts" in config:
            config = dict(config)
            config["experts"] = _load_experts(config["experts"])
        return super()._fit_model(
            cls_model=cls_model,
            config=config,
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
            distributed_config=distributed_config,
        )


def _frozen_input_size(experts, h):
    if not experts:
        raise ValueError("AutoNSXFrozen requires trained modules in experts")
    input_sizes = {expert.input_size for expert in experts}
    horizons = {expert.h for expert in experts}
    if len(input_sizes) != 1 or len(horizons) != 1:
        raise ValueError("experts must share a single h and input_size")
    horizon = horizons.pop()
    if horizon != h:
        raise ValueError(
            f"experts were built with h={horizon}; AutoNSXFrozen has h={h}"
        )
    return input_sizes.pop()


def _fix_frozen_config(config, experts, input_size):
    """Pin the checkpoints onto a Ray dict or an Optuna config function."""
    if callable(config):
        def config_fn(trial, config=config):
            sampled = dict(config(trial))
            sampled["experts"] = experts
            sampled["input_size"] = input_size
            return sampled

        return config_fn

    config = dict(config)
    config["experts"] = experts
    config["input_size"] = input_size
    return config
