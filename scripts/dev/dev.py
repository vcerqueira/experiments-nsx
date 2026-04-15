import os
from functools import partial
from typing import Union

import torch
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NHITS
from neuralforecast.losses.pytorch import MAE, _weighted_mean, BasePointLoss

from utilsforecast.losses import mase, smape

from utilsforecast.evaluation import evaluate
from metaforecast.utils.data import DataUtils

from utils.load_data.config import DATASETS

from src.moe import SimpleMoe

data_name = 'M4'
# data_name = 'Gluonts'
group = 'Monthly'
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)
df = data_loader.prune_df_by_size(df, min_n_instances=500)

print(df['unique_id'].value_counts())
print(df.shape)
horizon = 3

# SPLITS AND MODELS
train, test = DataUtils.train_test_split(df, horizon)




models = [
    SimpleMoe(h=horizon, input_size=n_lags, accelerator='mps',
              # loss=MAEGrad(),
              loss=MAE(),
              max_steps=2000, scaler_type='standard', batch_size=32,
              gate='linear', total_loss_type='annealing',
              gate_loss_type='ib_softmax_mse_grad',
              sparse_gate=False,
              add_specialization_loss=False, specialization_factor=.1,
              annealing_temperature=1000),
    SimpleMoe(h=horizon, input_size=n_lags, accelerator='mps',
              # loss=MAEGrad(),
              loss=MAE(),
              gate_loss_type='ib_softmax_mse',
              max_steps=2000, scaler_type='standard', batch_size=32,
              gate='linear', total_loss_type='annealing', sparse_gate=False,
              add_specialization_loss=False, specialization_factor=.1,
              annealing_temperature=1000),
    # MLP(h=horizon, input_size=n_lags, accelerator='mps', max_steps=1000, scaler_type='standard'),
    # NHITS(h=horizon, input_size=n_lags, accelerator='mps')
]
# SimpleMoe     0.934051

# gate='mlp',  # ['mlp','attention','linear','rnn']
#                  sparse_gate: bool = False,  # [True,False]
#                  gate_loss_type: str = 'ib_softmax_mse',  # ['ib_softmax_mse','softmax_mse','kl']
#                  add_specialization_loss: bool = False,
#                  total_loss_type: str = 'annealing',# ['random', 'annealing']
#

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train, val_size=horizon)
fcst = nf.predict()
print(fcst)

test_eval = test.merge(fcst, on=['unique_id', 'ds'], how="left")
evaluation_df = evaluate(test_eval, [partial(mase, seasonality=freq_int)], train_df=train)
# evaluation_df = evaluate(test_eval, [smape], train_df=train)

print(evaluation_df.mean(numeric_only=True))

# a=evaluation_df#.mean(numeric_only=True)
# b=a['SimpleMoe']-a['SimpleMoe1']
# b.describe()

# import torch
# a=torch.tensor([0.7701, 0.7661, 0.7785, 0.7845, 0.7890, 0.7772])
# -a
