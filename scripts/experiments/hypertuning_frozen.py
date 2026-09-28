from pprint import pprint
import warnings
import copy
from functools import partial
from pathlib import Path

import pandas as pd

from neuralforecast import NeuralForecast

from src.loaders import ChronosDataset, LongHorizonDatasetR
from metaforecast.evaluation import ModelRadar
from utilsforecast.losses import mase

from build_experts import load_experts
from build_sota import load_sota
from src.moe.config_pool import CONFIG_POOL
from src.neuralnets import BaseModelsConfig
from src.hypertuning import ConfigSampler
from src.config import SEED, N_SAMPLES, MAX_SAMPLES, ENGINE, LIMIT_EPOCHS, DATASETS, LH_DATASETS

warnings.filterwarnings('ignore')

RESULTS_PATH = Path('../../assets/results')


def error_frame(scored, train, seas_len):
    radar = ModelRadar(
        cv_df=scored,
        metrics=[partial(mase, seasonality=seas_len)],
        train_df=train,
    )
    return pd.DataFrame(radar.evaluate()).T


if __name__ == '__main__':
    print(RESULTS_PATH.absolute())

    config_pool = CONFIG_POOL['NSX-frozen']

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

        train, test = ChronosDataset.time_wise_split(df, horizon)

        models = BaseModelsConfig.get_nf_models(
            horizon=horizon,
            input_size=n_lags * 2,
            engine=ENGINE,
            limit_epochs=LIMIT_EPOCHS,
        )
        nfe = NeuralForecast(models=models, freq=freq)
        nfe.fit(df=train, val_size=horizon)
        sota_fcst = nfe.predict()

        experts = copy.deepcopy(nfe.models)

        config_list = ConfigSampler.generate_samples(config_pool=config_pool,
                                                     num_samples=N_SAMPLES,
                                                     random_state=SEED)

        for config_sample in config_list:
            pprint(config_sample)

            cfg_id = config_sample.pop('config_id')

            config_pattern = f"NSX-frozen,{cfg_id},{target}"
            config_files = list(RESULTS_PATH.glob(f"NSX-frozen,*,{target}.csv"))
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
                experts_ = copy.deepcopy(experts)
                model = ConfigSampler.create_model_instance(model_config=config_sample,
                                                            horizon=horizon,
                                                            input_size=n_lags,
                                                            engine=ENGINE,
                                                            limit_epochs=LIMIT_EPOCHS,
                                                            experts=experts_)
            except ValueError as e:
                print(f"Skipping invalid config {cfg_id}: {e}")
                continue

            nf = NeuralForecast(models=[model], freq=freq)
            nf.fit(df=train)
            fcst = nf.predict().merge(sota_fcst, on=['unique_id', 'ds'])
            scored = test.merge(fcst, on=['unique_id', 'ds'], how='left')
            err_outer = error_frame(scored, train, seas_len)

            print(err_outer)

            err_outer.to_csv(fp, index=False)
