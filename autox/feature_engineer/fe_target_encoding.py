import pandas as pd
from ..CONST import FEATURE_TYPE
from autox.process_data import Feature_type_recognition
import numpy as np

def FE_target_encoding(train, test, keys, col_label, k = 5):
    oof_train, oof_test = np.zeros(train.shape[0]), np.zeros(test.shape[0]) 
    from sklearn.model_selection import KFold
    skf = KFold(n_splits = k).split(train)
    for i, (train_idx, valid_idx) in enumerate(skf):
        df_train = train[keys + [col_label]].loc[train_idx]
        df_valid = train[keys].loc[valid_idx]
        df_map = df_train.groupby(keys)[[col_label]].agg('mean')
        oof_train[valid_idx] = df_valid.merge(df_map, on = keys, how = 'left')[col_label].fillna(-1).values
        oof_test += test[keys].merge(df_map, on = keys, how = 'left')[col_label].fillna(-1).values / k
    return oof_train, oof_test

class FeatureTargetEncoding:
    def __init__(self):
        self.target = None
        self.df_feature_type = None
        self.silence_cols = []
        self.select_all = None
        self.max_num = None
        self.ops = []

    def fit(self, df, target, df_feature_type = None, silence_cols = [], select_all = True,
            max_num = None):

        self.target = target
        self.df_feature_type = df_feature_type
        self.silence_cols = silence_cols
        self.select_all = select_all
        self.max_num = max_num

        if self.df_feature_type is None:
            feature_type_recognition = Feature_type_recognition()
            feature_type = feature_type_recognition.fit(df)
            self.df_feature_type = feature_type

        for feature in self.df_feature_type.keys():
            if self.df_feature_type[feature] == FEATURE_TYPE['cat']:
                if feature != target:
                    self.ops.append([feature])

        if not self.select_all:
            # 通过统计信息进行筛选
            del_targetencoding_cols = []

            # 用cat列在train和test上的分布来筛选, test做了target encoding之后，有值的部分要大于90%
            # todo:其他特征中也可以采用这种方式
            train = df[~df[target].isnull()]
            test = df[df[target].isnull()]
            for targetencoding_col in self.ops:
                if df.drop_duplicates(targetencoding_col).shape[0] > df.shape[0] * 0.001:
                    if targetencoding_col not in del_targetencoding_cols:
                        del_targetencoding_cols.append(targetencoding_col)
                if test.loc[test[targetencoding_col[0]].isin(train[targetencoding_col[0]].unique())].shape[0] / test.shape[0] < 0.999:
                    if targetencoding_col not in del_targetencoding_cols:
                        del_targetencoding_cols.append(targetencoding_col)

            for targetencoding_col in del_targetencoding_cols:
                self.ops.remove(targetencoding_col)

    def get_ops(self):
        return self.ops

    def set_keys(self, ops):
        self.ops = ops

    def transform(self, df):
        col_target = self.target
        result = pd.DataFrame()
        
        for keys in self.ops:
            name = f'TARGET_ENCODING_{"__".join(keys)}'
            
            train = df[~df[col_target].isnull()]
            test = df[df[col_target].isnull()]
            oof_train, oof_test = FE_target_encoding(train, test, keys, col_target, k = 5)
            train[name] = oof_train
            test[name] = oof_test
            result[name] = pd.concat([train[name], test[name]], axis = 0).loc[df.index]
        return result

    def fit_transform(self, df, target, df_feature_type = None, silence_cols = None, select_all = True,
            max_num = None):
        self.fit(df, target=target, df_feature_type=df_feature_type, silence_cols=silence_cols,
                        select_all=select_all, max_num=max_num)
        return self.transform(df)