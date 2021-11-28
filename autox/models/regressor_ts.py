import datetime
from time import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from ..util import log, weighted_mae_lgb, weighted_mae_xgb
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from datetime import timedelta
import gc

class XgbRegressionTs(object):
    def __init__(self, params=None):
        self.models = []
        self.feature_importances_ = pd.DataFrame()
        self.params_ = {
            'eta': 0.01,
            'max_depth': 11,
            'subsample': 0.6,
            'n_estimators': 3000,
            'reg_alpha': 40,
            'reg_lambda': 18,
            'min_child_weight': 16,
            'metric': 'rmse',
            # 'tree_method': 'gpu_hist',
            'verbose_eval': 100
        }
        if params is not None:
            self.params_ = params
        self.log1p = None

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def fit(self, train, test, used_features, target, time_col, ts_unit, log1p=True, custom_metric=None,
            weight_for_mae=10):
        log(train[used_features].shape)
        if train[target].min() < 0:
            log1p = False
        self.log1p = log1p
        self.feature_importances_['feature'] = train[used_features].columns
        if log1p:
            log("log1p!")
            train[target] = np.log1p(train[target])

        if ts_unit == 'D':
            one_unit = timedelta(days=1)
        elif ts_unit == 'W':
            one_unit = timedelta(days=7)
        intervals = int((pd.to_datetime(test[time_col].max()) - pd.to_datetime(test[time_col].min())) / one_unit + 1)
        split_time = pd.to_datetime(train[time_col].max()) - intervals * one_unit
        train_idx = train.loc[~(train[time_col] > str(split_time))].index
        valid_idx = train.loc[train[time_col] > str(split_time)].index

        MSEs = []
        start_time = time()
        print('Training with validation')
        X_train, y_train = train.iloc[train_idx][used_features], train.iloc[train_idx][target]
        X_valid, y_valid = train.iloc[valid_idx][used_features], train.iloc[valid_idx][target]
        model = xgb.XGBRegressor(**self.params_)
        if custom_metric == 'weighted_mae':
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)], verbose=100, early_stopping_rounds=100,
                      eval_metric=weighted_mae_xgb(weight=weight_for_mae))
        else:
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)], verbose=100, early_stopping_rounds=100)

        val = model.predict(train.iloc[valid_idx][used_features])
        if log1p:
            mse_ = mean_squared_error(np.expm1(train.iloc[valid_idx][target]), np.expm1(val), squared=True)
        else:
            mse_ = mean_squared_error(train.iloc[valid_idx][target], val, squared=True)
        print('MSE: {}'.format(mse_))
        MSEs.append(mse_)
        print('Finished in {}'.format(str(datetime.timedelta(seconds=time() - start_time))))

        start_time = time()
        ## 用数据集重新训练
        print("ReTraining on all data")
        gc.enable()
        del X_valid, y_valid
        gc.collect()
        X_train, y_train = train[used_features], train[target]
        X_valid, y_valid = train[used_features], train[target]
        self.params_['n_estimators'] = int(model.get_booster().best_iteration * 1.15)
        model = xgb.XGBRegressor(**self.params_)
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)], verbose=100)
        self.models.append(model)
        self.feature_importances_['feature_importance'] = model.feature_importances_
        if log1p:
            train[target] = np.expm1(train[target])
        print('Finished in {}'.format(str(datetime.timedelta(seconds=time() - start_time))))


    def predict(self, test, used_features):
        for idx, model in enumerate(self.models):
            if idx == 0:
                result = model.predict(test[used_features])
            else:
                result += model.predict(test[used_features])
        result /= len(self.models)
        if self.log1p:
            result = np.expm1(result)
        return result

class LgbRegressionTs(object):
    def __init__(self, params=None):
        self.models = []
        self.feature_importances_ = pd.DataFrame()
        self.params_ = {
            'objective': 'regression',
            'boosting': 'gbdt',
            'learning_rate': 0.01,
            "metric": 'rmse',
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
        self.Early_Stopping_Rounds = 150
        self.N_round = 8000
        self.Verbose = 100
        self.log1p = None

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def fit(self, train, test, used_features, target, time_col, ts_unit, Early_Stopping_Rounds=None, N_round=None,
            Verbose=None, log1p=True, custom_metric=None, weight_for_mae=10):
        log(train[used_features].shape)
        if train[target].min() < 0:
            log1p = False
        self.log1p = log1p
        if log1p:
            log("log1p!")
            train[target] = np.log1p(train[target])

        if ts_unit == 'D':
            one_unit = timedelta(days=1)
        elif ts_unit == 'W':
            one_unit = timedelta(days=7)
        intervals = int((pd.to_datetime(test[time_col].max()) - pd.to_datetime(test[time_col].min())) / one_unit + 1)
        split_time = pd.to_datetime(train[time_col].max()) - intervals * one_unit
        train_idx = train.loc[~(train[time_col] > str(split_time))].index
        valid_idx = train.loc[train[time_col] > str(split_time)].index

        if Early_Stopping_Rounds is not None:
            self.Early_Stopping_Rounds = Early_Stopping_Rounds
        if N_round is not None:
            self.N_round = N_round
        if Verbose is not None:
            self.Verbose = Verbose

        MSEs = []
        self.feature_importances_['feature'] = train[used_features].columns

        start_time = time()
        print('Training with validation')
        trn_data = lgb.Dataset(train.iloc[train_idx][used_features], label=train.iloc[train_idx][target],
                               categorical_feature='')
        val_data = lgb.Dataset(train.iloc[valid_idx][used_features], label=train.iloc[valid_idx][target],
                               categorical_feature='')

        if custom_metric=='weighted_mae':
            model = lgb.train(self.params_, trn_data, num_boost_round=self.N_round, valid_sets=[trn_data, val_data],
                              verbose_eval=self.Verbose,
                              early_stopping_rounds=self.Early_Stopping_Rounds,
                              feval=weighted_mae_lgb(weight=weight_for_mae))
        else:
            model = lgb.train(self.params_, trn_data, num_boost_round=self.N_round, valid_sets=[trn_data, val_data],
                            verbose_eval=self.Verbose,
                            early_stopping_rounds=self.Early_Stopping_Rounds)
        val = model.predict(train.iloc[valid_idx][used_features])
        if log1p:
            mse_ = mean_squared_error(np.expm1(train.iloc[valid_idx][target]), np.expm1(val))
        else:
            mse_ = mean_squared_error(train.iloc[valid_idx][target], val)
        print('MSE: {}'.format(mse_))
        MSEs.append(mse_)
        print('Finished in {}'.format(str(datetime.timedelta(seconds=time() - start_time))))

        start_time = time()
        ## 用数据集重新训练
        print("ReTraining on all data")
        gc.enable()
        del trn_data, val_data
        gc.collect()
        trn_data = lgb.Dataset(train[used_features], label=train[target], categorical_feature='')
        model = lgb.train(self.params_, trn_data, num_boost_round=int(model.best_iteration * 1.15),
                        valid_sets=[trn_data], verbose_eval=self.Verbose)  # , feval=evalerror

        self.models.append(model)
        self.feature_importances_['feature_importance'] = model.feature_importance()
        if log1p:
            train[target] = np.expm1(train[target])
        print('Finished in {}'.format(str(datetime.timedelta(seconds=time() - start_time))))

    def predict(self, test, used_features):
        for idx, model in enumerate(self.models):
            if idx == 0:
                result = model.predict(test[used_features]) / len(self.models)
            else:
                result += model.predict(test[used_features]) / len(self.models)
        if self.log1p:
            result = np.expm1(result)
        return result