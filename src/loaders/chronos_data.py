from typing import Optional

import datasets
import numpy as np
import pandas as pd
from huggingface_hub import dataset_info

from src.loaders.base import DatasetLoader


class ChronosDataset(DatasetLoader):
    # https://github.com/autogluon/fev/blob/main/benchmarks/chronos_zeroshot/results/auto_arima.csv
    # https://github.com/SalesforceAIResearch/gift-eval/blob/main/results/naive/all_results.csv

    DATASET_NAME = 'CHRONOS'
    REPO_ID = 'autogluon/chronos_datasets'

    M4_HORIZON_MAP = {
        "Y": 6,
        "Q": 4,
        "M": 12,
        "MS": 12,
        "ME": 12,
        "W": 13,
        "D": 14,
        "H": 48,
    }

    HORIZON_MAP = {
        "Y": 3,
        "Q": 4,
        "M": 12,
        "MS": 12,
        "ME": 12,
        "W": 8,
        "D": 30,
        "H": 48,
        "T": 48,
        "S": 60,
    }

    SPECIAL_HORIZON_MAP = {
        'monash_m1_monthly': 6,  # time series are too short for 12 or 18
        'monash_m1_quarterly': 2,  # time series are too short for 4
        'monash_m1_yearly': 2,  # time series are too short for 2
        'monash_tourism_yearly': 2,  # time series are too short for 2
        'm4_yearly': 3,  # time series are too short for 6
    }

    FREQUENCY_MAP = {
        "Y": 1,
        "Q": 4,
        "M": 12,
        "MS": 12,
        "ME": 12,
        "W": 52,  # ?
        "D": 365,  # 7?
        "H": 24,
        "T": 1,  # ?
        "S": 1,  # ?
    }

    # LAGS_BY_FREQUENCY = {k: int(v * 1.25) for k, v in HORIZON_MAP.items()}

    FREQUENCY_MAP_DATASETS = {
        'monash_m1_monthly': 'M',
        'monash_m1_quarterly': 'Q',
        'monash_m1_yearly': 'Y',
        'monash_m3_monthly': 'M',
        'monash_m3_quarterly': 'Q',
        'monash_m3_yearly': 'Y',
        'monash_tourism_monthly': 'M',
        'monash_tourism_quarterly': 'Q',
        'monash_tourism_yearly': 'Y',
        'm4_hourly': 'H',
        'm4_monthly': 'M',
        'm4_quarterly': 'Q',
        'm4_weekly': 'W',
        'm4_daily': 'D',
        'm4_yearly': 'Y',
        'monash_hospital': 'MS',
        'monash_car_parts': 'MS',
    }

    @classmethod
    def load_data(cls,
                  group: str,
                  split: str = 'train',
                  min_n_instances: Optional[int] = None,
                  id_col: str = 'unique_id',
                  time_col: str = 'ds',
                  target_col: str = 'y'):

        assert group in [*cls.FREQUENCY_MAP_DATASETS], 'Unknown dataset'

        ds = datasets.load_dataset(path=cls.REPO_ID, name=group, split=split)
        ds.set_format("numpy")

        df = cls.to_pandas(ds).reset_index(drop=True)
        df = df.rename(columns={'id': id_col,
                                'timestamp': time_col,
                                'target': target_col})

        if 'category' in df.columns:
            df = df.drop(columns=['category'])

        if min_n_instances is not None:
            df = cls.prune_uids_by_size(df, min_n_instances)

        # todo this was done for yearly time series (m4_yearly) ... format may not be appropriate for others
        if df[time_col].dtype == 'O':
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except pd.errors.OutOfBoundsDatetime:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce', format="%Y-%m-%dT%H:%M:%S.%f")
                if df[time_col].isna().any():
                    df[time_col] = (
                        df[time_col]
                        .astype(str)
                        .str.slice(0, 10)
                    )
                    df['ds'] = pd.to_datetime(df[time_col], errors='coerce', format="%Y-%m-%d")

        return df

    @classmethod
    def load_everything(cls,
                        group: str,
                        split: str = 'train',
                        min_n_instances: Optional[int] = None,
                        sample_n_uid: Optional[int] = None):

        # if split == 'both':
        #     df_train = cls.load_data(group=group, split='train', min_n_instances=min_n_instances)
        #     df_test = cls.load_data(group=group, split='test', min_n_instances=min_n_instances)
        #     df = pd.concat([df_train, df_test], axis=0).sort_values([id_col, time_col]).reset_index(drop=True)

        df = cls.load_data(group=group, split=split, min_n_instances=min_n_instances)

        freq = cls.FREQUENCY_MAP_DATASETS.get(group)

        if group in [*cls.SPECIAL_HORIZON_MAP]:
            horizon = cls.SPECIAL_HORIZON_MAP[group]
        else:
            if group.startswith('m4'):
                horizon = cls.M4_HORIZON_MAP.get(freq)
            else:
                horizon = cls.HORIZON_MAP.get(freq)

        n_lags = int(horizon * 1.25)

        seas_len = cls.FREQUENCY_MAP.get(freq)

        if sample_n_uid is not None:
            assert isinstance(df, pd.DataFrame)
            df = cls.sample_first_uids(df, sample_n_uid)

        df = df.reset_index(drop=True)

        return df, horizon, n_lags, freq, seas_len

    @staticmethod
    def get_chronos_datasets_names(repo_id='autogluon/chronos_datasets'):
        ds_info = dataset_info(repo_id)

        dataset_names = np.unique([s.rfilename.split('/')[0] for s in ds_info.siblings])
        dataset_names = [x for x in dataset_names if x not in ['.gitattributes', 'README.md']]

        return dataset_names

    @staticmethod
    def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
        """Convert dataset to long data frame format."""
        sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
        return ds.to_pandas().explode(sequence_columns).infer_objects()
