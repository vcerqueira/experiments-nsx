import os
import warnings
from functools import partial
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import MLP, NHITS
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mase
from metaforecast.evaluation import ModelRadar

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.moe.moe import NSX

warnings.filterwarnings('ignore')

# ---- data loading and partitioning
target = 'monash_m1_monthly'
df, horizon, input_size, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, _, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')


RESULTS_PATH = '../assets/results{}'
# results_dir = Path('../assets/results')

train, test = ChronosDataset.time_wise_split(df, horizon)

models = [
    NSX(h=horizon,
        input_size=input_size,
        accelerator='cpu',
        max_steps=10,
        scaler_type='standard',
        batch_size=32,
        gate='linear',
        total_loss_type='annealing',
        gate_loss_type='ib_softmax_mse_grad',
        pooling='dense',
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

radar_outer = ModelRadar(
    cv_df=cv,
    metrics=[partial(mase, seasonality=seas_len)],
    train_df=train,
    hardness_reference='MLP',
    ratios_reference='MLP',
)

err_outer = radar_outer.evaluate()
print(err_outer)
