import os
import warnings
from pathlib import Path

from src.loaders import ChronosDataset, LongHorizonDatasetR

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
target = 'monash_m3_monthly'
df, horizon, input_size, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, _, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')

# df['unique_id'].value_counts().value_counts().sort_index()
# from pprint import pprint
# dt = ChronosDataset.get_chronos_datasets_names()
# pprint(dt)

RESULTS_PATH = '../../assets/results{}'
# results_dir = Path('../assets/results')

train, test = ChronosDataset.time_wise_split(df, horizon)
