import os
from typing import Optional
from pathlib import Path

import pandas as pd
from datasetsforecast.long_horizon import LongHorizon
from dotenv import load_dotenv

from src.loaders.base import DatasetLoader


# load_dotenv()
# DATASET_PATH = Path(os.environ["DATA_DIR"])
# ds, *_ = LongHorizon.load(directory=DATASET_PATH, group='Weather')
# ds['unique_id'].value_counts()
# print(ds)

# ds, *_ = LongHorizonDatasetR.load_everything(group='ETTm1', resample_to='H')
# ds['unique_id'].value_counts()


class LongHorizonDataset(DatasetLoader):
    load_dotenv()
    DATASET_PATH = Path(os.environ["DATA_DIR"])

    DATASET_NAME = 'LONGHORIZON'

    HORIZON_MAP = {
        "ETTm1": 96,
        "ETTm2": 96,
        "ECL": 96,
        "Exchange": 14,
        "TrafficL": 96,
        "Weather": 144,
    }

    FREQUENCY_MAP = {
        "ETTm1": 96,
        "ETTm2": 96,
        "ECL": 24,
        "Exchange": 365,
        "TrafficL": 24,
        "Weather": 240,
    }

    FREQUENCY_MAP_DATASETS = {
        "ETTm1": '15T',
        "ETTm2": '15T',
        "ECL": 'H',
        "Exchange": 'D',
        "TrafficL": 'H',
        "Weather": '10T',
    }

    @classmethod
    def load_data(cls, group: str, min_n_instances: Optional[int] = None, **kwargs):

        assert group in [*cls.FREQUENCY_MAP_DATASETS], 'Unknown dataset'

        df, *_ = LongHorizon.load(directory=cls.DATASET_PATH, group=group)
        df['ds'] = pd.to_datetime(df['ds'])

        if min_n_instances is not None:
            df = cls.prune_uids_by_size(df, min_n_instances)

        return df

    @classmethod
    def load_everything(cls,
                        group: str,
                        min_n_instances: Optional[int] = None,
                        sample_n_uid: Optional[int] = None,
                        **kwargs):

        df = cls.load_data(group=group, min_n_instances=min_n_instances)

        freq = cls.FREQUENCY_MAP_DATASETS.get(group)
        horizon = cls.HORIZON_MAP.get(group)
        seas_len = cls.FREQUENCY_MAP.get(group)

        n_lags = int(horizon * 1.25)

        if sample_n_uid is not None:
            assert isinstance(df, pd.DataFrame)
            df = cls.sample_first_uids(df, sample_n_uid)

        df = df.reset_index(drop=True)

        return df, horizon, n_lags, freq, seas_len


class LongHorizonDatasetR(LongHorizonDataset):

    @classmethod
    def load_everything(cls,
                        group: str,
                        resample_to: str = 'D',
                        min_n_instances: Optional[int] = None,
                        sample_n_uid: Optional[int] = None,
                        **kwargs):
        df, horizon, n_lags, freq, seas_len = (
            super().load_everything(group=group,
                                    min_n_instances=min_n_instances,
                                    sample_n_uid=sample_n_uid))

        if resample_to == 'D':
            if group == 'Exchange':
                return df, horizon, n_lags, freq, seas_len

            d_horizon = 14
            d_n_lags = int(d_horizon * 1.25)
            freq = 'D'
            seas_len = 365

            daily_df = (
                df.groupby('unique_id')
                .resample(freq, on='ds')
                .mean(numeric_only=True)
                .reset_index()
            )

            return daily_df, d_horizon, d_n_lags, freq, seas_len
        elif resample_to == 'H':
            if group in ['Exchange', 'TrafficL', 'ECL']:
                return df, horizon, n_lags, freq, seas_len

            d_horizon = 48
            d_n_lags = int(d_horizon * 1.25)
            freq = 'H'
            seas_len = 24

            hourly_df = (
                df.groupby('unique_id')
                .resample(freq, on='ds')
                .mean(numeric_only=True)
                .reset_index()
            )

            return hourly_df, d_horizon, d_n_lags, freq, seas_len
        else:
            raise NotImplementedError

