import os
import warnings
from functools import partial
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.numpy import mase
from neuralforecast.models import MLP, NHITS
from utilsforecast.evaluation import evaluate
from metaforecast.evaluation import ModelRadar

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.moe.moe import NSX

from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler

warnings.filterwarnings('ignore')

# ---- data loading and partitioning
target = 'monash_m1_monthly'
# _, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to='D')
_, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target, min_n_instances=2 * (n_lags + horizon))
# df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target,
#                                                                           min_n_instances=2 * (n_lags + horizon),
#                                                                           resample_to='D')

CV_SETUP = {'val_size': horizon, 'test_size': horizon, 'step_size': 1, 'n_windows': None, }

RESULTS_PATH = '../assets/results{}'
# results_dir = Path('../assets/results')

if __name__ == '__main__':
    print(results_dir.absolute())

    config_pool = NEURAL_CONFIG_POOL[model_nm]
    config_list = ConfigSampler.generate_samples(config_pool=config_pool, num_samples=N_SAMPLES, random_state=SEED)

    for config_sample in config_list:
        print(config_sample)

        # check if no of configs reaches MAX_SAMPLES
        config_pattern = f"{model_nm},{target}"
        config_files = list(results_dir.glob(f"{config_pattern},*outer.csv"))
        n_configs = len(config_files)
        if n_configs >= MAX_SAMPLES:
            print(f"No of configs reached MAX_SAMPLES for {model_nm},{target}")
            break

        cfg_id = config_sample.pop('config_id')

        fp = results_dir / f'{target},{cfg_id},outer.csv'

        if fp.exists():
            print(f"Skipping {target},{cfg_id},outer.csv -- Already exists")
            continue

        print(f"Running config {n_configs} / {MAX_SAMPLES}")

        model = ModelsConfig.create_model_instance(model_class=model_nm,
                                                   model_config=config_sample,
                                                   horizon=horizon,
                                                   input_size=n_lags,
                                                   try_mps=TRY_MPS)

        try:
            nf_outer = NeuralForecast(models=[model], freq=freq)
            cv_outer = nf_outer.cross_validation(df=df, **CV_SETUP)

            cv_outer.to_csv(fp, index=False)
        except (NotImplementedError, IndexError) as e:
            print(f"Error on {target},{cfg_id}")
            continue



models = [
    NSX(h=horizon,
        input_size=input_size,
        accelerator='cpu',
        max_steps=20,
        scaler_type='standard',
        batch_size=32,
        gate='linear',
        total_loss_type='annealing',
        gate_loss_type='ib_softmax_mse_grad',
        pooling='dense',
        add_specialization_loss=False,
        specialization_factor=.1,
        annealing_temperature=1000),
    MLP(h=horizon,
        input_size=input_size,
        accelerator='cpu',
        max_steps=20,
        scaler_type='standard'),
]

nf = NeuralForecast(models=models, freq=freq)
nf.fit(df=train, val_size=horizon)
fcst = nf.predict()
print(fcst)

cv = test.merge(fcst, on=['unique_id', 'ds'], how="left")

# fazer com o metaforecast...
radar_outer = ModelRadar(
    cv_df=cv,
    metrics=[partial(mase, seasonality=seas_len)],
    model_names=['NSX','MLP'],
    train_df=train,
    hardness_reference='MLP',
    ratios_reference='MLP',
)

err_outer = radar_outer.evaluate(keep_uids=False)
print(err_outer)
