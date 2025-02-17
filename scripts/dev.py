import os
from functools import partial

import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NHITS

from utilsforecast.losses import mase, smape

from utilsforecast.evaluation import evaluate
from metaforecast.utils.data import DataUtils

from utils.load_data.config import DATASETS

from src.moe import SimpleMoe

data_name = 'Tourism'
# data_name = 'Gluonts'
group = 'Monthly'
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)
df = data_loader.prune_df_by_size(df,min_n_instances=200)

print(df['unique_id'].value_counts())
print(df.shape)
horizon=3

# SPLITS AND MODELS
train, test = DataUtils.train_test_split(df, horizon)

models = [
    SimpleMoe(h=horizon, input_size=n_lags, accelerator='mps', max_steps=2500, scaler_type='standard', batch_size=32),
    # MLP(h=horizon, input_size=n_lags, accelerator='mps', max_steps=1000, scaler_type='standard'),
    # NHITS(h=horizon, input_size=n_lags, accelerator='mps')
]
# SimpleMoe    0.953951
# SimpleMoe    0.936725
# SimpleMoe    0.935066 attention
# SimpleMoe    0.938058 rnn
# SimpleMoe    0.934051 linear
# MLP          0.983470

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train, val_size=horizon)
fcst = nf.predict()
print(fcst)

test_eval = test.merge(fcst, on=['unique_id', 'ds'], how="left")
evaluation_df = evaluate(test_eval, [partial(mase, seasonality=freq_int)], train_df=train)
# evaluation_df = evaluate(test_eval, [smape], train_df=train)

print(evaluation_df.mean(numeric_only=True))

# import torch
# a=torch.tensor([0.7701, 0.7661, 0.7785, 0.7845, 0.7890, 0.7772])
# -a
