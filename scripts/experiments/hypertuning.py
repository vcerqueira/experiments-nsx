from pprint import pprint
import warnings
from functools import partial
from pathlib import Path

import pandas as pd

from neuralforecast import NeuralForecast

from src.loaders import ChronosDataset, LongHorizonDatasetR
from metaforecast.evaluation import ModelRadar
from utilsforecast.losses import mase

from src.moe.config_pool import CONFIG_POOL
from src.hypertuning import ConfigSampler
from src.config import SEED, N_SAMPLES, MAX_SAMPLES, ENGINE, LIMIT_EPOCHS, DATASETS, LH_DATASETS

warnings.filterwarnings('ignore')

RESULTS_PATH = Path('../../assets/results')

if __name__ == '__main__':
    print(RESULTS_PATH.absolute())

    config_pool = CONFIG_POOL['NSX']

    for target in DATASETS:
        if target in LH_DATASETS:
            _, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to='D')
            df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(
                target,
                min_n_instances=2 * (n_lags + horizon),
                resample_to='D',
            )
        else:
            _, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
            df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(
                target,
                min_n_instances=2 * (n_lags + horizon),
            ) 

        cv_setup = {'val_size': horizon, 'test_size': horizon, 'step_size': 1, 'n_windows': None}
        train, _ = ChronosDataset.time_wise_split(df, horizon)

        config_list = ConfigSampler.generate_samples(config_pool=config_pool,
                                                     num_samples=N_SAMPLES,
                                                     random_state=SEED)

        for config_sample in config_list:
            pprint(config_sample)

            cfg_id = config_sample.pop('config_id')

            # Count every finished config for this target, not this config id alone.
            config_pattern = f"NSX,{cfg_id},{target}"
            config_files = list(RESULTS_PATH.glob(f"NSX,*,{target}.csv"))
            n_configs = len(config_files)
            if n_configs >= MAX_SAMPLES:
                print(f"No of configs reached MAX_SAMPLES for {target}")
                break

            fp = RESULTS_PATH / f"{config_pattern}.csv"

            if fp.exists():
                print(f"Skipping {target},{cfg_id},outer.csv -- Already exists")
                continue

            print(f"Running config {n_configs} / {MAX_SAMPLES}")
            try:
                model = ConfigSampler.create_model_instance(model_config=config_sample,
                                                        horizon=horizon,
                                                        input_size=n_lags,
                                                        engine=ENGINE,
                                                        limit_epochs=LIMIT_EPOCHS)
            except ValueError as e:
                continue

            try:
                nf = NeuralForecast(models=[model], freq=freq)
                cv = nf.cross_validation(df=df, **cv_setup)

                radar_outer = ModelRadar(
                    cv_df=cv,
                    metrics=[partial(mase, seasonality=seas_len)],
                    train_df=train,
                )

                err_outer = radar_outer.evaluate()
            except Exception as e:
                if "Loss is NaN, training stopped." not in str(e):
                    raise
                print(f"Loss is NaN on {target},{cfg_id}")
                err_outer = pd.Series([float("nan")], name="Overall")


            print(err_outer)

            err_outer.to_csv(fp, index=False)
