import random
import hashlib
import copy
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import SEED, N_SAMPLES
from src.moe.moe import NSX


class ConfigSampler:
    BAD_CONFIGS = []

    @classmethod
    def generate_samples(cls,
                         config_pool: Dict,
                         num_samples: int = N_SAMPLES,
                         random_state: int = SEED,
                         remove_bad_configs: bool = True,
                         return_df: bool = False):

        """
        Uninformed Random Sampling
        """

        cls.set_seeds(random_state)

        sample_list = []
        for i in range(num_samples):
            sample = {
                k: (v.sample() if hasattr(v, 'sample') else v)
                for k, v in config_pool.items()
            }

            sample['config_id'] = cls.get_config_id(sample)

            # if sample['batch_size'] > 32:
            #     continue

            sample_list.append(sample)

        if remove_bad_configs:
            sample_list = [sample for sample in sample_list if sample['config_id'] not in cls.BAD_CONFIGS]

        if return_df:
            df = pd.DataFrame(sample_list).set_index('config_id')
            return df

        return sample_list

    @staticmethod
    def set_seeds(seed: int = SEED):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def get_config_id(config):
        hash_len = 20

        config_str = json.dumps(config, sort_keys=True)
        config_id = hashlib.md5(config_str.encode()).hexdigest()[:hash_len]

        return config_id

    @staticmethod
    def create_model_instance(model_config: Dict,
                              horizon: int,
                              input_size: int,
                              engine: str,
                              limit_epochs: bool = False,
                              limit_val_batches: Optional[int] = None,
                              experts=None):

        model_config = copy.deepcopy(model_config)

        if 'config_id' in model_config:
            config_id = model_config.pop('config_id')

        input_multiplier = model_config.pop('input_size_multiplier')

        base_config = {'accelerator': engine,
                       'h': horizon,
                       'input_size': input_size * input_multiplier, }

        if 'inference_input_size_multiplier' in model_config:
            inference_input_size_multiplier = model_config.pop('inference_input_size_multiplier')
            base_config['inference_input_size'] = input_size * inference_input_size_multiplier

        config = {**model_config, **base_config}

        if limit_epochs:
            config['max_steps'] = 2

        if limit_val_batches is not None:
            config['limit_val_batches'] = limit_val_batches

        if experts is not None:
            config['experts'] = experts

        model_instance = NSX(**config)

        return model_instance
