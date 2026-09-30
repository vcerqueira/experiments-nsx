from pathlib import Path

import pandas as pd

from src.config import DATASETS

RESULTS_PATH = Path('assets/').resolve() / 'results_pretrained'
FOCUS = 'AutoNSXF'


def load_results(path):
    rows = []
    for fp in sorted(path.glob('SOTA,*.csv')):
        target = fp.stem.split(',', 1)[1]
        score = pd.read_csv(fp).apply(pd.to_numeric, errors='coerce')
        score.insert(0, 'target', target)
        rows.append(score)
    if not rows:
        raise FileNotFoundError(f'No SOTA result files in {path}')

    table = pd.concat(rows, ignore_index=True)
    order = {name: index for index, name in enumerate(DATASETS)}
    table['_order'] = table['target'].map(lambda name: order.get(name, len(order)))
    table = table.sort_values(['_order', 'target']).drop(columns='_order')
    return table.reset_index(drop=True)


table = load_results(RESULTS_PATH)
models = [column for column in table.columns if column != 'target']
others = [column for column in models if column != FOCUS]
scores = table[models]
has_score = scores.notna().any(axis=1)
table['best'] = pd.Series(pd.NA, index=table.index, dtype='object')
table.loc[has_score, 'best'] = scores.loc[has_score].idxmin(axis=1)

df = table.drop(10)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 240)
pd.set_option('display.float_format', lambda value: f'{value:.3f}')

df.set_index('target', inplace=True)
df.drop(columns='best', inplace=True)


df.rank(axis=1).mean().sort_values()
df.mean().sort_values()