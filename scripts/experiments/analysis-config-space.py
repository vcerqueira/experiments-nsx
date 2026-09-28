from pathlib import Path

import pandas as pd

from src.config import DATASETS, N_SAMPLES, SEED
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


def _dataset_order(targets):
    order = {name: index for index, name in enumerate(DATASETS)}
    return targets.map(lambda target: order.get(target, len(order)))


def _pick_configuration(group, how):
    """One result row. ``how`` is ``best`` (lowest score) or ``median``."""
    finite = group.dropna(subset=['Overall'])
    if finite.empty:
        return group.iloc[0]
    if how == 'best':
        return finite.sort_values(['Overall', 'config_id']).iloc[0]
    median_value = finite['Overall'].median()
    distance = (finite['Overall'] - median_value).abs()
    # Equidistant rows take the worse score, so an even count does not
    # report a configuration better than the median.
    ranked = finite.assign(_distance=distance).sort_values(
        ['_distance', 'Overall', 'config_id'],
        ascending=[True, False, True],
    )
    return ranked.iloc[0]


def _configuration_table(frame, how, baseline_columns):
    chosen = [
        _pick_configuration(group, how)
        for _, group in frame.groupby('target', sort=False)
    ]
    table = pd.DataFrame(chosen).reset_index(drop=True)
    counts = frame.groupby('target')['Overall'].apply(lambda scores: int(scores.notna().sum()))
    table.insert(1, 'n', table['target'].map(counts).astype('Int64'))
    others = table[baseline_columns]
    has_other = others.notna().any(axis=1)
    table['best_other'] = others.min(axis=1, skipna=True).where(has_other)
    table['best_other_model'] = pd.Series(pd.NA, index=table.index, dtype='object')
    if has_other.any():
        table.loc[has_other, 'best_other_model'] = others.loc[has_other].idxmin(axis=1)
    table['gap'] = table['Overall'] - table['best_other']
    table['_order'] = _dataset_order(table['target'])
    return table.sort_values(['_order', 'target']).drop(columns='_order').reset_index(drop=True)


def _print_baseline_comparison(table, how, baseline_columns):
    label = 'Best' if how == 'best' else 'Median'
    score_name = f'NSX {how}'
    shown = table.rename(columns={'Overall': score_name})
    compact = [
        'target', 'n', 'config_id', score_name,
        'best_other', 'best_other_model', 'gap',
    ]
    print(f'\n{label} NSX configuration against the best other model in the same file')
    print('Other models are the columns of that file. A dataset can have different baseline scores across files.')
    print('gap is NSX minus the best of those columns. A negative gap means NSX is more accurate.')
    print(shown[compact].to_string(index=False))

    wide_columns = ['target', 'config_id', score_name, *baseline_columns]
    print(f'\n{label} NSX configuration and the other models in the same file')
    print(shown[wide_columns].to_string(index=False))

    shared = shown.dropna(subset=[score_name])
    shared = shared.loc[shared[baseline_columns].notna().any(axis=1)]
    if shared.empty:
        return
    score_columns = [score_name, *baseline_columns]
    print(f'\nMean score on {len(shared)} datasets with a finite {label.lower()} NSX and another model')
    print(shared[score_columns].mean().sort_values().to_string())
    print('\nAverage rank on those datasets')
    ranks = shared[score_columns].rank(axis=1, method='min', ascending=True, na_option='keep')
    print(ranks.mean().sort_values().to_string())
    wins = int((shared['gap'] < 0).sum())
    print(f'\n{label} NSX beats the best other model on {wins} of {len(shared)} datasets')


if IS_VECTOR and baseline_columns:
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda value: f'{value:.3f}')
    vector_results = results.reset_index()
    for selection in ('best', 'median'):
        comparison = _configuration_table(vector_results, selection, baseline_columns)
        _print_baseline_comparison(comparison, selection, baseline_columns)

