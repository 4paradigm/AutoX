import pandas as pd
from tqdm import tqdm
from ..CONST import FEATURE_TYPE
from datetime import timedelta

def roll_mean_features(df, windows, val, keys, op):
    names = []
    for window in windows:
        name = f"{'__'.join(keys)}__{val}_roll_{op}_" + str(window)
        names.append(name)
        if op == 'mean':
            df[name] = df.groupby(keys)[val].transform(
                lambda x: x.rolling(window=window, min_periods=3, win_type="triang").mean())
        if op == 'std':
            df[name] = df.groupby(keys)[val].transform(
                lambda x: x.rolling(window=window, min_periods=3).std())
        if op == 'median':
            df[name] = df.groupby(keys)[val].transform(
                lambda x: x.rolling(window=window, min_periods=3).median())
        if op == 'max':
            df[name] = df.groupby(keys)[val].transform(
                lambda x: x.rolling(window=window, min_periods=3).max())
        if op == 'min':
            df[name] = df.groupby(keys)[val].transform(
                lambda x: x.rolling(window=window, min_periods=3).min())
    return df[names]

class FeatureRollingStatTS:
    def __init__(self):
        self.id_ = None
        self.target = None
        self.df_feature_type = None
        self.time_col = None
        self.ts_unit = None
        self.silence_cols = None
        self.ops = []

    def fit(self, df, id_, target, df_feature_type, time_col, ts_unit, silence_cols=[]):
        self.id_ = id_
        self.target = target
        self.df_feature_type = df_feature_type
        self.time_col = time_col
        self.ts_unit = ts_unit
        self.silence_cols = silence_cols

        for col in self.df_feature_type.keys():
            if col in self.silence_cols + id_ + [time_col]:
                continue
            if df.loc[df[self.target].isnull(), col].nunique() == df.loc[df[self.target].isnull(), col].shape[0]:
                continue
            if self.df_feature_type[col] == FEATURE_TYPE['num']:
                self.ops.append(col)

        if self.ts_unit == 'D':
            one_unit = timedelta(days=1)
            intervals = int((pd.to_datetime(df.loc[df[self.target].isnull(), self.time_col].max()) - pd.to_datetime(
            df.loc[df[self.target].isnull(), self.time_col].min())) / one_unit + 1)
            self.windows = [intervals+7, intervals+7*2, intervals*2]
            self.windows = list(dict.fromkeys(self.windows))
        if self.ts_unit == 'W':
            one_unit = timedelta(days=7)
            intervals = int((pd.to_datetime(df.loc[df[self.target].isnull(), self.time_col].max()) - pd.to_datetime(
            df.loc[df[self.target].isnull(), self.time_col].min())) / one_unit + 1)
            self.windows = [intervals+3, intervals+4, intervals+5]
            self.windows = list(dict.fromkeys(self.windows))

    def get_ops(self):
        return self.ops

    def set_ops(self, ops):
        self.ops = ops

    def get_windows(self):
        return self.windows

    def set_windows(self, windows):
        self.windows = windows

    def transform(self, df):
        df_copy = df.copy()
        df_copy.sort_values(by=self.time_col, axis=0, inplace=True)
        flag = True
        for col in tqdm(self.ops):
            for op in ['mean', 'std', 'median', 'max', 'min']:
                df_temp = roll_mean_features(df_copy, self.windows, col, self.id_, op)
                df_temp = df_temp.loc[df.index]
                if flag:
                    result = df_temp
                    flag = False
                else:
                    result = pd.concat([result, df_temp], axis=1)
        return result

    def fit_transform(self, df, id_, target, df_feature_type, time_col, ts_unit, silence_cols=[]):
        self.fit(df, id_, target, df_feature_type, time_col, ts_unit, silence_cols=silence_cols)
        return self.transform(df)


