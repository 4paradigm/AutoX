import pandas as pd
from tqdm import tqdm
from ..CONST import FEATURE_TYPE
from datetime import timedelta

def lag_features(df, lags, val, keys):
    df_temp = df[keys + [val]]
    names = []
    for lag in lags:
        name = f"{'__'.join(keys)}__{val}__lag_" + str(lag)
        names.append(name)
        df_temp[name] = df_temp.groupby(keys)[val].transform(lambda x: x.shift(lag))
    return df_temp[names]

class FeatureShiftTS:
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
            self.lags = [intervals, intervals + 1, intervals + 2, intervals + 3, intervals + 7,
                             intervals + 7 * 2, intervals + 7 * 3, intervals + 30, intervals * 2, intervals * 3]

        if self.ts_unit == 'W':
            one_unit = timedelta(days=7)
            intervals = int((pd.to_datetime(df.loc[df[self.target].isnull(), self.time_col].max()) - pd.to_datetime(
            df.loc[df[self.target].isnull(), self.time_col].min())) / one_unit + 1)
            self.lags = [intervals, intervals + 1, intervals + 2, intervals + 3]


    def get_ops(self):
        return self.ops

    def set_ops(self, ops):
        self.ops = ops

    def get_lags(self):
        return self.lags

    def set_lags(self, lags):
        self.lags = lags

    def transform(self, df):
        df_copy = df.copy()
        df_copy.sort_values(by=self.time_col, axis=0, inplace=True)

        for i, col in tqdm(enumerate(self.ops)):
            df_temp = lag_features(df_copy, self.lags, col, self.id_)
            df_temp = df_temp.loc[df.index]
            if i == 0:
                result = df_temp
            else:
                result = pd.concat([result, df_temp], axis=1)
        return result

    def fit_transform(self, df, id_, target, df_feature_type, time_col, ts_unit, silence_cols=[]):
        self.fit(df, id_, target, df_feature_type, time_col, ts_unit, silence_cols=silence_cols)
        return self.transform(df)


