import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureOne2Many:
    """**Create features from table with one to many relationship.**
    1. Pass the label of the primary table to the secondary table;
    2. Train model (5-fold cross validation) and predict in the secondary table;
    3. Aggregate the prediction results of the secondary table to obtain statistical information.

    Example::
        `feature_one2many_autox <https://www.kaggle.com/code/poteman/feature-one2many-autox>`_

    """
    def __init__(self):
        pass

    def fit(self, t1, t2, on, target):
        """
        :param t1: dataframe, primary table
        :param t2: dataframe, secondary table
        :param on: list, column names to join on
        :param target: str, target column name
        """
        self.on = on
        self.target = target

        t2 = t2.merge(t1[on + [target]], on=on, how='left')
        used = [x for x in t2.columns if x not in on + [target]]

        cat_cols = []
        for f in t2.columns:
            if 'O' == t2[f].dtype and f not in on + [target]:
                lbl = LabelEncoder()
                t2[f] = lbl.fit_transform(list(t2[f].astype(str)))
                cat_cols.append(f)

        lr = 0.1
        Early_Stopping_Rounds = 150
        N_round = 200
        Verbose = False
        params = {'num_leaves': 41,
                  'min_child_weight': 0.03454472573214212,
                  'feature_fraction': 0.3797454081646243,
                  'bagging_fraction': 0.4181193142567742,
                  'min_data_in_leaf': 96,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': lr,
                  "boosting_type": "gbdt",
                  "bagging_seed": 11,
                  "metric": 'auc',
                  "verbosity": -1,
                  'reg_alpha': 0.3899927210061127,
                  'reg_lambda': 0.6485237330340494,
                  'random_state': 47,
                  'num_threads': 16
                  }

        t2['one2many_predict'] = -1
        groups = np.array(t2[on])
        gss = GroupShuffleSplit(n_splits=5, random_state=42)
        for train_idx, test_idx in tqdm(gss.split(t2[used], t2[target], groups), total=5):
            train_idx = list(t2.loc[train_idx].loc[t2.loc[train_idx][target].notnull()].index)
            trn_data = lgb.Dataset(t2.loc[train_idx][used], label=t2.loc[train_idx][target],
                                   categorical_feature=cat_cols)
            clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data], verbose_eval=Verbose,
                            early_stopping_rounds=Early_Stopping_Rounds)

            t2.loc[test_idx, 'one2many_predict'] = clf.predict(t2.loc[test_idx][used])

        result = t2.groupby(on).agg(
            {'one2many_predict': ['max', 'min', 'median', 'mean', 'std', 'count']}).reset_index()
        result.columns = ['_'.join(x) if x[1] != '' else ''.join(x) for x in list(result.columns)]

        self.result = result

    def transform(self, df):
        """
        :param df: dataframe, primary table that requires feature construction
        :return: dataframe, created features
        """
        return df[self.on].merge(self.result, on=self.on, how='left')