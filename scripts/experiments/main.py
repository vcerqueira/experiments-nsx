import os
import warnings
from functools import partial
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.numpy import mase, smape
from neuralforecast.models import MLP, NHITS
from utilsforecast.evaluation import evaluate
from modelradar.evaluate.radar import ModelRadar

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.moe.moe import TiMEx, MAEGrad

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
target = 'monash_m1_monthly'
df, horizon, input_size, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, _, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')

# df['unique_id'].value_counts().value_counts().sort_index()
# from pprint import pprint
# dt = ChronosDataset.get_chronos_datasets_names()
# pprint(dt)

RESULTS_PATH = '../assets/results{}'
# results_dir = Path('../assets/results')

train, test = ChronosDataset.time_wise_split(df, horizon)

models = [
    TiMEx(h=horizon,
          input_size=input_size,
          accelerator='cpu',
          loss=MAEGrad(),
          max_steps=2000,
          scaler_type='standard',
          batch_size=32,
          gate='linear',
          total_loss_type='annealing',
          gate_loss_type='ib_softmax_mse_grad',
          sparse_gate=False,
          add_specialization_loss=False,
          specialization_factor=.1,
          annealing_temperature=1000),
    TiMEx(h=horizon,
          input_size=input_size,
          accelerator='cpu',
          loss=MAE(),
          gate_loss_type='ib_softmax_mse_grad',
          max_steps=2000,
          scaler_type='standard',
          batch_size=32,
          gate='linear',
          total_loss_type='annealing',
          sparse_gate=False,
          add_specialization_loss=False,
          specialization_factor=.1,
          annealing_temperature=1000),
    MLP(h=horizon,
        input_size=input_size,
        accelerator='cpu',
        max_steps=2000,
        scaler_type='standard'),
    # NHITS(h=horizon, input_size=n_lags, accelerator='mps')
]
# SimpleMoe     0.934051

# gate='mlp',  # ['mlp','attention','linear','rnn']
#                  sparse_gate: bool = False,  # [True,False]
#                  gate_loss_type: str = 'ib_softmax_mse',  # ['ib_softmax_mse','softmax_mse','kl']
#                  add_specialization_loss: bool = False,
#                  total_loss_type: str = 'annealing',# ['random', 'annealing']
#

nf = NeuralForecast(models=models, freq=freq)
nf.fit(df=train, val_size=horizon)
fcst = nf.predict()
print(fcst)

test_eval = test.merge(fcst, on=['unique_id', 'ds'], how="left")

print(smape(test_eval['y'], test_eval['TiMEx']))
print(smape(test_eval['y'], test_eval['TiMEx1']))
print(smape(test_eval['y'], test_eval['MLP']))

evaluation_df = evaluate(df=test_eval,
                         metrics=[partial(mase, seasonality=seas_len)],
                         models=['TiMEx', 'TiMEx1', 'MLP'],
                         train_df=train)
# evaluation_df = evaluate(test_eval, [smape], train_df=train)


radar_outer = ModelRadar(
    cv_df=test_eval,
    #metrics=[partial(mase, seasonality=seas_len)],
    metrics=[smape],
    model_names=['TiMEx', 'TiMEx1', 'MLP'],
    #train_df=train,
    hardness_reference='MLP',
    ratios_reference='MLP',
)

err_outer = radar_outer.evaluate(keep_uids=False)
print(err_outer)
