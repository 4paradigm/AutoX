from tqdm import tqdm
from autox.process_data import Feature_type_recognition
from ..CONST import FEATURE_TYPE

class FeatureCount:
    def __init__(self):
        self.target = None
        self.df_feature_type = None
        self.silence_cols = []
        self.select_all = None
        self.max_num = None
        self.ops = []

    def fit(self, df, degree=1, target = None, df_feature_type = None, silence_cols = [], select_all = True,
            max_num = None):

        assert(degree == 1 or degree == 2)

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
            if self.df_feature_type[feature] == FEATURE_TYPE['cat'] and feature not in self.silence_cols:
                self.ops.append([feature])

        if not self.select_all:
            if self.target is not None:
                # 训练模型，对group_col进行筛选
                pass
            else:
                # 通过统计信息进行筛选
                del_count_cols = []
                for count_col in self.ops:
                    if df.drop_duplicates(count_col).shape[0] > df.shape[0] * 0.2:
                        del_count_cols.append(count_col)
                for count_col in del_count_cols:
                    self.ops.remove(count_col)

        if degree == 2:
            ops_degree_1 = self.ops
            ops = []
            for col_1 in ops_degree_1:
                for col_2 in ops_degree_1:
                    if col_1 == col_2:
                        continue
                    else:
                        ops.append(col_1 + col_2)
            self.ops = ops + ops_degree_1

    def get_ops(self):
        return self.ops

    def set_keys(self, ops):
        self.ops = ops

    def transform(self, df):
        name_list = []
        for op in tqdm(self.ops):
            if len(op) == 1:
                name = f'COUNT_{"__".join(op)}'
                name_list.append(name)
                df[name] = df.groupby(op)[op].transform('count')
            else:
                col_1, col_2 = op
                name = f'COUNT_{col_1}__{col_2}'
                name_list.append(name)
                df_map = df.groupby([col_1, col_2]).size().to_frame()
                df_map.columns = [name]
                df = df.merge(df_map, on=[col_1, col_2], how='left')
        result = df[name_list]
        df.drop(name_list, axis=1, inplace=True)
        return result

    def fit_transform(self, df, target = None, df_feature_type = None, silence_cols = [], select_all = True,
            max_num = None):
        self.fit(df, target=target, df_feature_type=df_feature_type, silence_cols=silence_cols,
                        select_all=select_all, max_num=max_num)
        return self.transform(df)