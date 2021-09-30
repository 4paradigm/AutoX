import pandas as pd
from ..CONST import FEATURE_TYPE
from autox.process_data import Feature_type_recognition
from tqdm import tqdm

class FeatureShift:
    def __init__(self):
        self.target = None
        self.df_feature_type = None
        self.silence_group_cols = []
        self.silence_agg_cols = []
        self.select_all = None
        self.max_num = None
        self.ops = {}

    def fit(self, df, target=None, df_feature_type=None, silence_group_cols=[], silence_agg_cols=[],
            select_all=True, max_num=None):
        self.target = target
        self.df_feature_type = df_feature_type
        self.silence_group_cols = silence_group_cols
        self.silence_agg_cols = silence_agg_cols
        self.select_all = select_all
        self.max_num = max_num

        if self.df_feature_type is None:
            feature_type_recognition = Feature_type_recognition()
            feature_type = feature_type_recognition.fit(df)
            self.df_feature_type = feature_type

        for group_col in self.df_feature_type.keys():
            if self.df_feature_type[group_col] == FEATURE_TYPE['cat'] and group_col not in self.silence_group_cols:
                if df[group_col].nunique() == df.shape[0]:
                    continue
                self.ops[(group_col)] = []
                for agg_col in self.df_feature_type.keys():
                    if group_col == agg_col:
                        continue
                    if agg_col not in self.silence_agg_cols:
                        if self.df_feature_type[agg_col] == FEATURE_TYPE['num']:
                            self.ops[(group_col)].append(agg_col)

        if not self.select_all:
            if self.target is not None:
                # 训练模型，对group_col进行筛选
                pass
            else:
                # 通过统计信息进行筛选
                del_group_cols = []
                for group_col in self.ops.keys():
                    if df[group_col].nunique() > df.shape[0] * 0.2 or df[group_col].nunique() < 5:
                        del_group_cols.append(group_col)
                for group_col in del_group_cols:
                    del self.ops[group_col]

    def get_ops(self):
        return self.ops

    def set_ops(self, ops):
        self.ops = ops

    def transform(self, df):
        result = pd.DataFrame()
        for group_col in tqdm(self.ops.keys()):
            agg_cols = self.ops[group_col]
            for agg_col in agg_cols:
                for i in [-30, -24, -7, -3, -2, -1, 1, 2, 3, 7, 24, 30]:
                    shift_value = df.groupby(group_col)[agg_col].shift(i).values
                    if type(group_col) == tuple:
                        name = f'{"__".join(group_col)}__{agg_col}__shift__{i}'
                    else:
                        name = f'{group_col}__{agg_col}__shift__{i}'
                    result[name] = shift_value
        return result

    def fit_transform(self, df, target=None, df_feature_type=None, silence_group_cols=[], silence_agg_cols=None,
                      select_all=True, max_num=None):
        self.fit(df, target=target, df_feature_type=df_feature_type, silence_group_cols=silence_group_cols,
                 silence_agg_cols=silence_agg_cols, select_all=select_all, max_num=max_num)
        return self.transform(df)


