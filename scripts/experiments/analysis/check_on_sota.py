from pathlib import Path

import pandas as pd

from src.config import DATASETS

ROOT = Path(__file__).resolve().parents[2]
NSX_PATH = ROOT / 'assets' / 'results'
SOTA_PATH = ROOT / 'assets' / 'results_pretrained'


def load_nsx(path: Path) -> pd.DataFrame:
    rows = []
    for fp in sorted(path.glob('NSX*.csv')):
        model, config_id, target = fp.stem.split(',', 2)
        score = pd.read_csv(fp).apply(pd.to_numeric, errors='coerce')
        rows.append({
            'model': model,
            'config_id': config_id,
            'target': target,
            'Overall': score['Overall'].iloc[0],
        })
    if not rows:
        raise FileNotFoundError(f'No NSX result files in {path}')
    return pd.DataFrame(rows)


def load_sota(path: Path) -> pd.DataFrame:
    frames = []
    for fp in sorted(path.glob('SOTA,*.csv')):
        _, target = fp.stem.split(',', 1)
        score = pd.read_csv(fp).apply(pd.to_numeric, errors='coerce')
        score.insert(0, 'target', target)
        frames.append(score)
    if not frames:
        raise FileNotFoundError(f'No SOTA result files in {path}')
    return pd.concat(frames, ignore_index=True)


def _best_model(row: pd.Series, columns: list) -> object:
    values = row[columns]
    if not values.notna().any():
        return pd.NA
    return values.idxmin()


def _dataset_order(targets: pd.Series) -> pd.Series:
    order = {name: i for i, name in enumerate(DATASETS)}
    return targets.map(lambda target: order.get(target, len(order)))


def compare(nsx: pd.DataFrame, sota: pd.DataFrame, model: str) -> pd.DataFrame:
    scores = nsx.loc[nsx['model'] == model]
    grouped = scores.groupby('target')['Overall'].agg(
        n='size',
        best='min',
        median='median',
    ).reset_index()

    sota_models = [column for column in sota.columns if column != 'target']
    reference = sota[['target', *sota_models]].copy()
    reference['sota_best'] = reference[sota_models].min(axis=1, skipna=True)
    reference['sota_best_model'] = reference.apply(
        lambda row: _best_model(row, sota_models),
        axis=1,
    )

    table = grouped.merge(reference, on='target', how='outer')
    beats_best = []
    for _, row in table.iterrows():
        group = scores.loc[scores['target'] == row['target'], 'Overall'].dropna()
        if pd.isna(row['sota_best']) or group.empty:
            beats_best.append(pd.NA)
        else:
            beats_best.append(int((group < row['sota_best']).sum()))
    table['n'] = table['n'].astype('Int64')
    table['n_better_than_sota'] = pd.Series(beats_best, dtype='Int64')
    table['gap'] = table['best'] - table['sota_best']
    table['_order'] = _dataset_order(table['target'])
    return table.sort_values(['_order', 'target']).drop(columns='_order').reset_index(drop=True)


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda value: f'{value:.3f}')

    nsx = load_nsx(NSX_PATH)
    sota = load_sota(SOTA_PATH)
    sota_models = [column for column in sota.columns if column != 'target']

    print('NSX scores by dataset')
    summary = nsx.groupby(['model', 'target'])['Overall'].agg(
        n='size',
        best='min',
        median='median',
        mean='mean',
    )
    summary = summary.reset_index()
    summary['_order'] = _dataset_order(summary['target'])
    summary = summary.sort_values(['model', '_order', 'target']).drop(columns='_order')
    print(summary.to_string(index=False))

    for model in summary['model'].unique():
        table = compare(nsx, sota, model)
        best_col = f'{model} best'
        median_col = f'{model} median'

        print(f'\n{model} against the best SOTA model')
        print('gap is NSX best minus the best SOTA score. A negative gap means NSX is more accurate.')
        against_best = table[[
            'target', 'n', 'best', 'median', 'sota_best', 'sota_best_model',
            'gap', 'n_better_than_sota',
        ]].rename(columns={'best': best_col, 'median': median_col})
        print(against_best.to_string(index=False))

        wide = table[['target', 'best', 'median', *sota_models]].rename(
            columns={'best': best_col, 'median': median_col},
        )
        print(f'\n{model} and SOTA MASE by dataset')
        print(wide.to_string(index=False))

        shared = wide.dropna(subset=[best_col])
        shared = shared.loc[shared[sota_models].notna().any(axis=1)]
        if shared.empty:
            continue
        score_cols = [column for column in wide.columns if column != 'target']
        print(f'\nMean MASE on {len(shared)} datasets with both {model} and SOTA scores')
        print(shared[score_cols].mean().sort_values().to_string())
        print('\nAverage rank on those datasets')
        print(shared[score_cols].rank(axis=1, method='min', ascending=True).mean().sort_values().to_string())


if __name__ == '__main__':
    main()
