import itertools
import warnings
import lightgbm as lgb
import pandas as pd
warnings.filterwarnings('ignore')


class FeatureCross:
    """**synthetic feature formed by multiplying (crossing) two features.**
        """
    def __init__(self):
        pass

    def fit(self, X, y, objective, category_cols, top_k=10, used_cols=[]):
        '''
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features). Training vector, where n_samples is the number of samples and n_features is the number of features.
        :param y: array-like of shape (n_samples,). Target vector relative to X.
        :param objective: str, objective equal to 'binary' or 'regression'.
        :param category_cols: list, column names of categorical features.
        :param top_k: int, keep the top_k importance cross features, default top_k = 10.
        :param used_cols: list, columns will be used for training model, default top_k = 10.
        '''

        self.category_cols = category_cols
        if len(used_cols) > 0:
            self.used_cols = used_cols
        else:
            self.used_cols = list(X.columns)
        self.used_cols = [x for x in list(X.describe().columns) if x in self.used_cols]
        self.top_k = top_k

        assert (objective in ['binary', 'regression'])

        params = {'objective': objective,
                  'boosting': 'gbdt',
                  'learning_rate': 0.01,
                  'num_leaves': 2 ** 3,
                  'bagging_fraction': 0.95,
                  'bagging_freq': 1,
                  'bagging_seed': 66,
                  'feature_fraction': 0.7,
                  'feature_fraction_seed': 66,
                  'max_depth': -1
                  }
        N_round = 100
        trn_data = lgb.Dataset(X[self.used_cols], label=y, categorical_feature=category_cols)
        self.clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data], verbose_eval=False)

        self.feature_importances = pd.DataFrame()

        self.feature_importances['feature'] = X[self.used_cols].columns
        self.feature_importances['imp'] = self.clf.feature_importance()
        self.feature_importances = self.feature_importances.sort_values(by="imp", ascending=False)
        self.feature_importances.index = range(len(self.feature_importances))

        self.top_k_features = [x for x in self.feature_importances['feature'] if x in category_cols][:top_k]
        self.cross_features = []
        for item in list(itertools.product(self.top_k_features, self.top_k_features)):
            f1 = item[0]
            f2 = item[1]
            if f1 == f2:
                pass
            else:
                if f1 in category_cols and f2 in category_cols:
                    self.cross_features.append([f1, f2])

    def transform(self, X):
        '''
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features). Training vector, where n_samples is the number of samples and n_features is the number of features.
        :return: dataframe, cross features.
        '''
        result = pd.DataFrame()
        for [f1, f2] in self.cross_features:
            result[f'{f1}_cross_{f2}'] = X[f1].astype(str) + '__' + X[f2].astype(str)

        return result

    def fit_transform(self, X, y, objective, category_cols, top_k=10, used_cols=[]):
        self.fit(X, y, objective, category_cols=category_cols, used_cols=used_cols, top_k=top_k)
        return self.transform(X)