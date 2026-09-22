import warnings
from functools import partial
from pathlib import Path

from neuralforecast import NeuralForecast

from src.loaders import ChronosDataset, LongHorizonDatasetR

from src.moe.config_pool import CONFIG_POOL
from src.hypertuning import ConfigSampler
from src.config import SEED, N_SAMPLES, MAX_SAMPLES, ENGINE, LIMIT_EPOCHS

from metaforecast.evaluation import ModelRadar
from utilsforecast.losses import mase

warnings.filterwarnings('ignore')

N_SAMPLES = 10

# ---- data loading and partitioning
target = 'monash_m1_monthly'
# _, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to='D')
_, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target, min_n_instances=2 * (n_lags + horizon))
# df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target,
#                                                                           min_n_instances=2 * (n_lags + horizon),
#                                                                           resample_to='D')
train, _ = ChronosDataset.time_wise_split(df, horizon)

CV_SETUP = {'val_size': horizon, 'test_size': horizon, 'step_size': 1, 'n_windows': None, }

RESULTS_PATH = Path('../assets/results')

config_pool = CONFIG_POOL['NSX']
config_list = ConfigSampler.generate_samples(config_pool=config_pool,
                                             num_samples=N_SAMPLES,
                                             random_state=SEED)

config_sample = config_list[0]

cfg_id = config_sample.pop('config_id')


model = ConfigSampler.create_model_instance(model_config=config_sample,
                                            horizon=horizon,
                                            input_size=n_lags,
                                            engine=ENGINE,
                                            limit_epochs=LIMIT_EPOCHS)

nf = NeuralForecast(models=[model], freq=freq)
cv = nf.cross_validation(df=df, **CV_SETUP)

radar_outer = ModelRadar(
    cv_df=cv,
    metrics=[partial(mase, seasonality=seas_len)],
    train_df=train,
)

err_outer = radar_outer.evaluate()
print(err_outer)

err_outer.to_csv('adads.csv', index=False)
