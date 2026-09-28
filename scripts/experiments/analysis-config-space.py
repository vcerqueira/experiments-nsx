from pathlib import Path

import pandas as pd

from src.config import N_SAMPLES, SEED
from src.hypertuning import ConfigSampler
from src.moe.config_pool import CONFIG_POOL

# RESULTS_PATH = Path(__file__).resolve().parents[2] / 'assets' / 'results'
RESULTS_PATH = Path('./assets/results')
# Set True to analyse the pretrained-expert runs in NSX-frozen result files.
FROZEN = True
# Set True when each result file is one row of model scores, with NSX as the
# configured model and the remaining columns as fixed baselines.
IS_VECTOR = True

model_name = 'NSX-frozen' if FROZEN else 'NSX'

configs = ConfigSampler.generate_samples(
    config_pool=CONFIG_POOL[model_name],
    num_samples=N_SAMPLES,
    random_state=SEED,
    return_df=True,
)
configs = configs[~configs.index.duplicated(keep='first')]

rows = []
baseline_columns = []
for fp in sorted(RESULTS_PATH.glob(f'{model_name},*,*.csv')):
    _, config_id, target = fp.stem.split(',', 2)
    score = pd.read_csv(fp)
    # NaN-loss runs are written as empty cells instead of a float.
    score = score.apply(pd.to_numeric, errors='coerce')
    if IS_VECTOR:
        if not baseline_columns:
            baseline_columns = [column for column in score.columns if column != 'NSX']
        score = score.rename(columns={'NSX': 'Overall'})
    score.insert(0, 'config_id', config_id)
    score.insert(1, 'target', target)
    rows.append(score)

scores = pd.concat(rows, ignore_index=True)
results = scores.merge(configs.reset_index(), on='config_id', how='left')

results.set_index('config_id', inplace=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


print(results['Overall'].isna().mean())



results.groupby('scaler_type').median(numeric_only=True)
results.groupby('max_steps').median(numeric_only=True)
if 'num_experts' in results.columns:
    results.groupby('num_experts').median(numeric_only=True)
print(results.groupby('pooling').mean(numeric_only=True)['Overall'])
print(results.groupby('pooling').mean(numeric_only=True)['Overall'])
results.groupby('pooling').median(numeric_only=True)
results.groupby('gate_loss_type').median(numeric_only=True)


results_na = results.loc[results['Overall'].isna(),:]


results_na.select_dtypes(include=['object']).value_counts()

results.iloc[0]

base_rate = results['Overall'].isna().mean()

score_skip = ['Overall', *baseline_columns]
nan_rates = []
for column in results.columns.drop(score_skip):
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


effects = []
global_median = results['Overall'].median()
# Range index: config_id repeats once per dataset, and a Series grouper
# aligned on that index does not count result files.
work = results.reset_index(drop=True)
for column in work.columns.drop(score_skip):
    series = work[column]
    if series.nunique(dropna=False) > 25:
        groups = pd.qcut(series, q=4, duplicates='drop')
        grouped = work.groupby(groups, observed=True)['Overall']
    else:
        grouped = work.groupby(column, dropna=False, observed=True)['Overall']
    for value, score in grouped:
        finite = score.dropna()
        effects.append({
            'parameter': column,
            'value': value,
            'n': int(len(score)),
            'median': finite.median(),
            'vs_global': finite.median() - global_median,
        })

effects = pd.DataFrame(effects)
gap = effects.groupby('parameter')['median'].agg(lambda s: s.max() - s.min())
effects = effects.join(gap.rename('gap'), on='parameter')
effects = effects.sort_values(['gap', 'parameter', 'median'], ascending=[False, True, True])
print("\n\n\n\n\n\n\n\n\n")
print(results.sort_values('Overall').iloc[:,-4:])
print("\n\n\n")
print(effects.drop(columns='gap').to_string(index=False))

