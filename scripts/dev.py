import os
from functools import partial

import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP

from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from metaforecast.utils.data import DataUtils

from utils.load_data.config import DATASETS

from src.moe import SimpleMoe

data_name = 'M3'
group = 'Monthly'
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)

print(df['unique_id'].value_counts())
print(df.shape)

# SPLITS AND MODELS
train, test = DataUtils.train_test_split(df, horizon)

models = [
    SimpleMoe(h=horizon, input_size=n_lags, accelerator='mps', max_steps=2500, scaler_type='standard'),
    # MLP(h=horizon, input_size=n_lags, accelerator='mps', max_steps=1000, scaler_type='standard')
]

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train, val_size=horizon)
fcst = nf.predict()

test_eval = test.merge(fcst, on=['unique_id', 'ds'], how="left")
evaluation_df = evaluate(test_eval, [partial(mase, seasonality=freq_int), smape], train_df=train)

print(evaluation_df.mean(numeric_only=True))
