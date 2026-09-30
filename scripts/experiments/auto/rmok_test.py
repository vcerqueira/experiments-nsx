import warnings
from functools import partial
from pathlib import Path

import pandas as pd

from neuralforecast import NeuralForecast

from src.loaders import ChronosDataset, LongHorizonDatasetR
from metaforecast.evaluation import ModelRadar
from utilsforecast.losses import mase

from src.neuralnets import ModelsConfig
from src.config import ENGINE, LIMIT_EPOCHS, DATASETS, LH_DATASETS, N_SAMPLES

warnings.filterwarnings('ignore')

RESULTS_PATH = Path(__file__).resolve().parents[3] / 'assets' / 'results'

if __name__ == '__main__':
    print(RESULTS_PATH.absolute())
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    for target in DATASETS:
        # target = 'monash_m1_monthly'
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

        models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                                 n_samples=N_SAMPLES,
                                                 engine=ENGINE,
                                                 limit_epochs=LIMIT_EPOCHS,
                                                 skip_nsx=True,
                                                 input_size=n_lags * 2,
                                                 n_series=df['unique_id'].nunique())

        nf = NeuralForecast(models=models, freq=freq)
        nf.fit(df=train)
        fcst = nf.predict()

        cv = test.merge(fcst, on=['unique_id', 'ds'], how='left')
        print(cv)

        radar_outer = ModelRadar(
            cv_df=cv,
            metrics=[partial(mase, seasonality=seas_len)],
            train_df=train,
        )

        err_outer = radar_outer.evaluate()
        err_outer = pd.DataFrame(err_outer).T

        fp = RESULTS_PATH / f"RMOK_TEST,{target}.csv"

        err_outer.to_csv(fp, index=False)
