import datetime
from time import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from ..util import log

class CrossLgbRegression(object):
    def __init__(self, params=None, n_fold=5):
        self.models = []
        self.feature_importances_ = pd.DataFrame()
        self.n_fold = n_fold
        self.params_ = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 2 ** 5,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 66,
            'feature_fraction': 0.7,
            'feature_fraction_seed': 66,
            'max_bin': 100,
            'max_depth': 5,
            'verbose': -1
        }
        if params is not None:
            self.params_ = params
        self.Early_Stopping_Rounds = 30
        self.N_round = 600
        self.Verbose = 10

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def fit(self, X, y, Early_Stopping_Rounds=None, N_round=None, Verbose=None):
        log(X.shape)
        if Early_Stopping_Rounds is not None:
            self.Early_Stopping_Rounds = Early_Stopping_Rounds
        if N_round is not None:
            self.N_round = N_round
        if Verbose is not None:
            self.Verbose = Verbose

        folds = KFold(n_splits=self.n_fold, shuffle=True, random_state=889)
        MSEs = []
        self.feature_importances_['feature'] = X.columns

        for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

            start_time = time()
            print('Training on fold {}'.format(fold_n + 1))

            trn_data = lgb.Dataset(X.iloc[train_index],
                                   label=y.iloc[train_index], categorical_feature="")
            val_data = lgb.Dataset(X.iloc[valid_index],
                                   label=y.iloc[valid_index], categorical_feature="")
            clf = lgb.train(self.params_, trn_data, num_boost_round=self.N_round, valid_sets=[trn_data, val_data],
                            verbose_eval=self.Verbose,
                            early_stopping_rounds=self.Early_Stopping_Rounds)
            self.models.append(clf)
            self.feature_importances_['fold_{}'.format(fold_n + 1)] = clf.feature_importance()
            val = clf.predict(X.iloc[valid_index])
            mse_ = mean_squared_error(y.iloc[valid_index], val)
            print('MSE: {}'.format(mse_))
            MSEs.append(mse_)
            print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(
                                                                 seconds=time() - start_time))))
        self.feature_importances_['average'] = self.feature_importances_[
            [x for x in self.feature_importances_.columns if x != "feature"]].mean(axis=1)
        self.feature_importances_ = self.feature_importances_.sort_values(by="average", ascending=False)
        self.feature_importances_.index = range(len(self.feature_importances_))

    def predict(self, test):
        for idx, clf in enumerate(self.models):
            if idx == 0:
                result = clf.predict(test) / self.n_fold
            else:
                result += clf.predict(test) / self.n_fold
        return result