from typing import Optional
from copy import deepcopy

from ray.tune.schedulers import ASHAScheduler
from neuralforecast.common._base_auto import RayOptions
from neuralforecast.losses.pytorch import MAE
from neuralforecast.auto import (AutoNBEATS,
                                 AutoTiDE,
                                 AutoNLinear,
                                 AutoKAN,
                                 AutoMLP,
                                 AutoDLinear,
                                 AutoNHITS,
                                 AutoPatchTST,
                                 AutoTFT,
                                 AutoDeepNPTS)


class ModelsConfig:
    AUTO_MODEL_CLASSES = {
        'AutoTFT': AutoTFT,
        'AutoNBEATS': AutoNBEATS,
        'AutoTiDE': AutoTiDE,
        'AutoNLinear': AutoNLinear,
        'AutoKAN': AutoKAN,
        'AutoMLP': AutoMLP,
        'AutoDLinear': AutoDLinear,
        'AutoNHITS': AutoNHITS,
        'AutoDeepNPTS': AutoDeepNPTS,
        'AutoPatchTST': AutoPatchTST,
    }

    @classmethod
    def get_auto_nf_models(cls,
                           horizon: int,
                           n_samples: int,
                           engine: str = 'cpu',
                           limit_epochs: bool = False,
                           limit_val_batches: Optional[int] = None):

        models = []
        for mod_name, mod in cls.AUTO_MODEL_CLASSES.items():
            config = deepcopy(mod.default_config)
            config['accelerator'] = engine

            if limit_epochs:
                config['max_steps'] = 2

            if limit_val_batches is not None:
                config['limit_val_batches'] = limit_val_batches

            model_instance = mod(
                h=horizon,
                config=config,
                num_samples=n_samples,
                alias=mod_name,
                valid_loss=MAE(),
                refit_with_val=True,
                backend="ray",
                ray_options=RayOptions(
                    scheduler=ASHAScheduler(
                        max_t=30,
                        grace_period=1,
                        reduction_factor=4,
                        brackets=1,
                    )
                ),
            )

            models.append(model_instance)

        return models
