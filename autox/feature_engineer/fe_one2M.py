import pandas as pd
from ..CONST import FEATURE_TYPE

class FeatureOne2M:
    def __init__(self):
        self.relations = None
        self.relations_one2M = None

    def fit(self, relations, train_name, feature_types):
        self.relations = relations
        self.ops = {}

        relations_one2M = [x for x in relations if
                           x['type'] == '1-M' and x['left_entity'] == train_name]
        self.relations_one2M = relations_one2M

        for idx, cur_relation in enumerate(relations_one2M):
            cur_right_entity = cur_relation['right_entity']
            cur_feature_type = feature_types[cur_right_entity]
            agg_dict = {}
            for col in cur_feature_type:
                if col in cur_relation['right_on']:
                    continue
                else:
                    if cur_feature_type[col] == FEATURE_TYPE['num']:
                        agg_dict[col] = ['max', 'min', 'median', 'mean', 'std']
                    elif cur_feature_type[col] == FEATURE_TYPE['cat']:
                        agg_dict[col] = ['nunique']
            self.ops[cur_right_entity] = agg_dict

    def get_ops(self):
        return self.ops

    def set_ops(self, ops):
        self.ops = ops

    def transform(self, df, dfs):
        res_df = pd.DataFrame()
        for idx, cur_relation in enumerate(self.relations_one2M):
            if idx == 0:
                res_df = df[cur_relation['left_on']]
            cur_right_entity = cur_relation['right_entity']
            cur_df = dfs[cur_right_entity]

            temp_df = cur_df.groupby(cur_relation['right_on']).agg(self.ops[cur_right_entity])

            # rename temp_df
            feas_name = []
            for i in range(len(temp_df.columns)):
                cur_name = cur_right_entity + '__' + '__'.join(temp_df.columns[i])
                feas_name.append(cur_name)
            temp_df.columns = feas_name

            res_df = pd.merge(res_df, temp_df, how='left', left_on=cur_relation['left_on'], right_index=True)

        res_df.drop(cur_relation['left_on'], axis=1, inplace=True)
        return res_df

    def fit_transform(self, df, dfs, relations, train_name, feature_types):
        self.fit(relations, train_name, feature_types)
        return self.transform(df, dfs)