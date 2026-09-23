from pathlib import Path

import pandas as pd

from src.config import N_SAMPLES, SEED
from src.hypertuning import ConfigSampler
from src.moe.config_pool import CONFIG_POOL

# RESULTS_PATH = Path(__file__).resolve().parents[2] / 'assets' / 'results'
RESULTS_PATH = Path('./assets/results')

configs = ConfigSampler.generate_samples(
    config_pool=CONFIG_POOL['NSX'],
    num_samples=N_SAMPLES,
    random_state=SEED,
    return_df=True,
)

rows = []
for fp in sorted(RESULTS_PATH.glob('NSX,*,*.csv')):
    _, config_id, _ = fp.stem.split(',', 2)
    score = pd.read_csv(fp)
    # NaN-loss runs are written as Overall,"" instead of a float.
    score = score.apply(pd.to_numeric, errors='coerce')
    score.insert(0, 'config_id', config_id)
    rows.append(score)

scores = pd.concat(rows, ignore_index=True)
results = scores.merge(configs.reset_index(), on='config_id', how='left')

results.set_index('config_id', inplace=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


results['Overall'].describe()
results['Overall'].isna().mean()


results.groupby('loss').median(numeric_only=True)
results.groupby('scaler_type').median(numeric_only=True)
results.groupby('max_steps').median(numeric_only=True)
results.groupby('start_padding_enabled').median(numeric_only=True)
results.groupby('num_experts').median(numeric_only=True)
print(results.groupby('gate').median(numeric_only=True)['Overall'])
print(results.groupby('gate').mean(numeric_only=True)['Overall'])
print(results.groupby('pooling').mean(numeric_only=True)['Overall'])
results.groupby('pooling').median(numeric_only=True)
results.groupby('gate_loss_type').median(numeric_only=True)
results.groupby('detach_gate_targets').mean(numeric_only=True)
results.groupby('scale_maeg_by_gate').mean(numeric_only=True)
results.groupby('total_loss_type').median(numeric_only=True)
results.groupby('include_combined_loss').median(numeric_only=True)
results.groupby('anneal_temperature').median(numeric_only=True)
results.groupby('add_specialization_loss').median(numeric_only=True)
results.groupby('add_ncl_loss').median(numeric_only=True)
results.groupby('add_balance_loss').median(numeric_only=True)


results_na = results.loc[results['Overall'].isna(),:]


results_na.select_dtypes(include=['object']).value_counts()

results.iloc[0]

base_rate = results['Overall'].isna().mean()

nan_rates = []
for column in results.columns.drop('Overall'):
    if results[column].nunique(dropna=False) > 25:
        continue
    grouped = results.groupby(column, dropna=False)['Overall']
    summary = grouped.agg(n='size', n_nan=lambda s: int(s.isna().sum()))
    summary['rate'] = summary['n_nan'] / summary['n']
    summary['lift'] = summary['rate'] / base_rate
    summary.insert(0, 'parameter', column)
    nan_rates.append(summary.reset_index(names='value'))

nan_rates = pd.concat(nan_rates, ignore_index=True).sort_values('lift', ascending=False)

print(results)
