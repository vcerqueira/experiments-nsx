from typing import Optional

import pandas as pd


class DatasetLoader:
    DATASET_NAME = ''

    HORIZON_MAP = {}

    SPECIAL_HORIZON_MAP = {}

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

    FREQUENCY_MAP_DATASETS = {
        'monash_m1_monthly': 'M',
    }

    @classmethod
    def load_data(cls,
                  group: str,
                  split: str = 'train',
                  min_n_instances: Optional[int] = None,
                  id_col: str = 'unique_id',
                  time_col: str = 'ds',
                  target_col: str = 'y'):

        pass

    @classmethod
    def load_everything(cls,
                        group: str,
                        split: str = 'train',
                        min_n_instances: Optional[int] = None,
                        sample_n_uid: Optional[int] = None,
                        **kwargs):

        pass

    @staticmethod
    def prune_uids_by_size(df: pd.DataFrame,
                           min_n_instances: int,
                           id_col: str = 'unique_id'):
        large_ts = df[id_col].value_counts() >= min_n_instances
        large_ts_uid = large_ts[large_ts].index.tolist()

        df = df.query(f'{id_col}== @large_ts_uid').reset_index(drop=True)

        return df

    @staticmethod
    def sample_first_uids(df: pd.DataFrame, n_uid: int, id_col: str = 'unique_id'):
        uid_sample = df[id_col].unique()[:n_uid].tolist()
        df = df.query(f'{id_col}==@uid_sample').reset_index(drop=True)

        return df

    @staticmethod
    def sample_uids(df: pd.DataFrame, frac_uid: float, id_col: str = 'unique_id'):

        unique_ids = pd.Series(df[id_col].unique())
        sampled_ids = unique_ids.sample(frac=frac_uid, random_state=123)

        sampled_df = df[df['unique_id'].isin(sampled_ids)].reset_index(drop=True)

        return sampled_df

    @staticmethod
    def dummify_series(df, id_col: str = 'unique_id', target_col: str = 'y'):
        df_uid = df.copy().groupby(id_col)

        dummied_l = []
        for g, uid_df in df_uid:
            uid_df[target_col] = range(uid_df.shape[0])

            dummied_l.append(uid_df)

        dummy_df = pd.concat(dummied_l, axis=0).reset_index(drop=True)

        return dummy_df

    @staticmethod
    def get_uid_tails(df, tail_size: int, id_col: str = 'unique_id'):
        df_list = []
        for g, df_ in df.groupby(id_col):
            df_list.append(df_.tail(tail_size))

        tail_df = pd.concat(df_list, axis=0).reset_index(drop=True)

        return tail_df

    @staticmethod
    def time_wise_split(df: pd.DataFrame,
                        horizon: int,
                        id_col: str = 'unique_id',
                        time_col: str = 'ds'):
        df_by_unq = df.groupby(id_col)

        train_l, test_l = [], []
        for g, df_ in df_by_unq:
            df_ = df_.sort_values(time_col)

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df

    @staticmethod
    def difference_series(df):
        df_uid = df.copy().groupby('unique_id')

        diff_l = []
        for g, uid_df in df_uid:
            uid_df['y'] = uid_df['y'].diff()

            diff_l.append(uid_df.tail(-1))

        diff_df = pd.concat(diff_l, axis=0).reset_index(drop=True)

        return diff_df

    @staticmethod
    def concat_time_wise_tr_ts(tr: pd.DataFrame, ts: pd.DataFrame):
        return pd.concat([tr, ts], axis=0).sort_values(['unique_id', 'ds']).reset_index(drop=True)
