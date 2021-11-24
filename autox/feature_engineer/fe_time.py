from autox.process_data import Feature_type_recognition
from ..CONST import FEATURE_TYPE
import pandas as pd
from tqdm import tqdm

class FeatureTime:
    def __init__(self):
        self.df_feature_type = None
        self.silence_cols = []
        self.select_all = None
        self.max_num = None
        self.ops = []

    def fit(self, df, df_feature_type = None, silence_cols = []):

        self.df_feature_type = df_feature_type
        self.silence_cols = silence_cols

        if self.df_feature_type is None:
            feature_type_recognition = Feature_type_recognition()
            feature_type = feature_type_recognition.fit(df)
            self.df_feature_type = feature_type

        for feature in self.df_feature_type.keys():
            if self.df_feature_type[feature] == FEATURE_TYPE['datetime'] and feature not in self.silence_cols:
                self.ops.append(feature)

    def get_ops(self):
        return self.ops

    def set_keys(self, ops):
        self.ops = ops

    def transform(self, df):

        df_copy = df[self.ops].copy()
        for col in tqdm(self.ops):

            prefix = col + "_"
            df_copy[col] = pd.to_datetime(df_copy[col])
            df_copy[prefix + 'year'] = df_copy[col].dt.year
            df_copy[prefix + 'month'] = df_copy[col].dt.month
            df_copy[prefix + 'day'] = df_copy[col].dt.day
            df_copy[prefix + 'hour'] = df_copy[col].dt.hour
            df_copy[prefix + 'weekofyear'] = df_copy[col].dt.weekofyear
            df_copy[prefix + 'dayofweek'] = df_copy[col].dt.dayofweek
            df_copy[prefix + 'is_wknd'] = df_copy[col].dt.dayofweek // 5
            df_copy[prefix + 'quarter'] = df_copy[col].dt.quarter
            df_copy[prefix + 'is_month_start'] = df_copy[col].dt.is_month_start.astype(int)
            df_copy[prefix + 'is_month_end'] = df_copy[col].dt.is_month_end.astype(int)

        df_copy.drop(self.ops, axis=1, inplace=True)
        return df_copy

    def fit_transform(self, df, df_feature_type = None, silence_cols = []):
        self.fit(df, df_feature_type=df_feature_type, silence_cols=silence_cols)
        return self.transform(df)