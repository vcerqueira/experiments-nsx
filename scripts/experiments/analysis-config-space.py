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


print(results['Overall'].isna().mean())

results.sort_values('Overall')

results.groupby('scaler_type').median(numeric_only=True)
results.groupby('max_steps').median(numeric_only=True)
results.groupby('num_experts').median(numeric_only=True)
print(results.groupby('pooling').mean(numeric_only=True)['Overall'])
print(results.groupby('pooling').mean(numeric_only=True)['Overall'])
results.groupby('pooling').median(numeric_only=True)
results.groupby('gate_loss_type').median(numeric_only=True)


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


def _levels(series, max_levels=25):
    if series.nunique(dropna=False) <= max_levels:
        return series
    return pd.qcut(series, q=4, duplicates='drop')


effects = []
global_median = results['Overall'].median()
for column in results.columns.drop('Overall'):
    level = _levels(results[column])
    for value, score in results.groupby(level, dropna=False, observed=False)['Overall']:
        finite = score.dropna()
        effects.append({
            'parameter': column,
            'value': value,
            'n': int(score.size),
            'median': finite.median(),
            'vs_global': finite.median() - global_median,
        })

effects = pd.DataFrame(effects)
gap = effects.groupby('parameter')['median'].agg(lambda s: s.max() - s.min())
effects = effects.join(gap.rename('gap'), on='parameter')
effects = effects.sort_values(['gap', 'parameter', 'median'], ascending=[False, True, True])

print("\n\n\n")
print(effects.drop(columns='gap').to_string(index=False))

