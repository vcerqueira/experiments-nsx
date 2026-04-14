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
df = data_loader.prune_df_by_size(df, min_n_instances=200)

print(df['unique_id'].value_counts())
print(df.shape)
horizon = 3

# SPLITS AND MODELS
train, test = DataUtils.train_test_split(df, horizon)

models = []
# for gate_ in ['mlp', 'attention', 'linear', 'rnn']:
for gate_ in ['attention']:

    # for sparse_gate_ in [True, False]:
    for sparse_gate_ in [True]:

        # for gate_loss_type_ in ['ib_softmax_mse', 'softmax_mse', 'minmax_mse']:
        for gate_loss_type_ in ['ib_softmax_mse']:

            for add_spec in [True, False]:
            # for add_spec in [False]:

                # for total_loss_type_ in ['random', 'annealing']:
                for total_loss_type_ in ['annealing']:
                    base = dict(h=horizon, input_size=n_lags,
                                accelerator='mps', max_steps=2000,
                                scaler_type='standard',
                                batch_size=32)

                    pars = {'gate': gate_,
                            'sparse_gate': sparse_gate_,
                            'gate_loss_type': gate_loss_type_,
                            'add_specialization_loss': add_spec,
                            'total_loss_type': total_loss_type_}

                    mod = SimpleMoe(**base, **pars)

                    models.append(mod)

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
