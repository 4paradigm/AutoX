import datetime
from time import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from ..util import log
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import torch.optim as optim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CrossTabnetRegression(object):
    def __init__(self, params=None, n_fold=10):
        self.models = []
        self.scaler = None
        self.features_median = None
        self.feature_importances_ = pd.DataFrame()
        self.n_fold = n_fold
        self.params_ = {
            'n_steps': 11,
            'gamma': 0.6,
            'device_name': DEVICE,
            'optimizer_fn': optim.Adam,
            'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
            'mask_type': "entmax",
            'max_epochs': 200,
            'patience': 15
        }
        if params is not None:
            self.params_ = params

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def optuna_tuning(self, X, y, Debug=False):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)

        def objective(trial):
            param_grid = {
                'n_steps': trial.suggest_int('n_steps', 3, 10),
                'gamma': trial.suggest_discrete_uniform('gamma', 1.0, 2.0, 0.1),
            }
            reg = TabNetRegressor(**param_grid)
            reg.fit(X_train, y_train.values.reshape(-1,1),
                    eval_set=[(X_valid, y_valid.values.reshape(-1,1))])
            return mean_squared_error(y_valid, reg.predict(X_valid), squared=True)

        train_time = 1 * 10 * 60  # h * m * s
        if Debug:
            train_time = 1 * 1 * 60  # h * m * s
        study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='TabNetRegressor')
        study.optimize(objective, timeout=train_time)

        log(f'Number of finished trials: {len(study.trials)}')
        log('Best trial:')
        trial = study.best_trial

        log(f'\tValue: {trial.value}')
        log('\tParams: ')
        for key, value in trial.params.items():
            log('\t\t{}: {}'.format(key, value))

        self.params_ = trial.params

    def fit(self, X, y, tuning=True, Debug=False):
        log(X.shape)
        self.feature_importances_['feature'] = X.columns

        # fillna
        self.features_median = X.median()
        X = X.fillna(self.features_median)

        # scaler
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        if tuning:
            log("[+]tuning params")
            self.optuna_tuning(X, y, Debug=Debug)

        folds = KFold(n_splits=self.n_fold, shuffle=True)
        RMSEs = []

        for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
            start_time = time()
            print('Training on fold {}'.format(fold_n + 1))
            X_train, y_train = X[train_index,:], y.values[train_index]
            X_valid, y_valid = X[valid_index,:], y.values[valid_index]
            model = TabNetRegressor(**self.params_)
            model.fit(X_train, y_train.reshape(-1,1), eval_set=[(X_valid, y_valid.reshape(-1,1))])

            self.models.append(model)
            self.feature_importances_['fold_{}'.format(fold_n + 1)] = model.feature_importances_
            val = model.predict(X[valid_index])
            mse_ = mean_squared_error(y.iloc[valid_index], val, squared=True)
            print('MSE: {}'.format(mse_))
            RMSEs.append(mse_)
            print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(
                seconds=time() - start_time))))
        log(f'Average KFold RMSE: {np.mean(RMSEs)}')
        self.feature_importances_['average'] = self.feature_importances_[
            [x for x in self.feature_importances_.columns if x != "feature"]].mean(axis=1)
        self.feature_importances_ = self.feature_importances_.sort_values(by="average", ascending=False)
        self.feature_importances_.index = range(len(self.feature_importances_))

    def predict(self, test):
        test = test.fillna(self.features_median)
        test = self.scaler.transform(test)
        for idx, model in enumerate(self.models):
            if idx == 0:
                result = model.predict(test)
            else:
                result += model.predict(test)
        result /= self.n_fold
        return result

class CrossXgbRegression(object):
    def __init__(self, metric, params=None, n_fold=10):
        self.models = []
        self.scaler = None
        self.feature_importances_ = pd.DataFrame()
        self.n_fold = n_fold
        self.metric = metric
        self.params_ = {
            'eta': 0.01,
            'max_depth': 11,
            'subsample': 0.6,
            'n_estimators': 3000,
            'reg_alpha': 40,
            'reg_lambda': 18,
            'min_child_weight': 16,
            # 'tree_method': 'gpu_hist',
            'verbose_eval': 100
        }
        if params is not None:
            self.params_ = params
        self.params_['eval_metric'] = self.metric
        self.log1p = None

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def optuna_tuning(self, X, y, metric, Debug=False, log1p=True):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)
        def objective(trial):
            param_grid = {
                'max_depth': trial.suggest_int('max_depth', 6, 15),
                'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 500, 3500, 100),
                'eta': 0.01,
                'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
                'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            }
            reg = xgb.XGBRegressor(**param_grid) # tree_method='gpu_hist',
            reg.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)], eval_metric=metric,
                    verbose=False)
            if log1p:
                mse_ = mean_squared_error(np.expm1(y_valid), np.expm1(reg.predict(X_valid)), squared=True)
            else:
                mse_ = mean_squared_error(y_valid, reg.predict(X_valid), squared=True)
            return mse_

        train_time = 1 * 10 * 60  # h * m * s
        if Debug:
            train_time = 1 * 1 * 60  # h * m * s
        study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='XgbRegressor')
        study.optimize(objective, timeout=train_time)

        log(f'Number of finished trials: {len(study.trials)}')
        log('Best trial:')
        trial = study.best_trial

        log(f'\tValue: {trial.value}')
        log('\tParams: ')
        for key, value in trial.params.items():
            log('\t\t{}: {}'.format(key, value))

        self.params_ = trial.params
        self.params_['eta'] = 0.01
        # self.params_['tree_method'] = 'gpu_hist'

    def fit(self, X, y, tuning=True, Debug=False, log1p=True):
        log(X.shape)
        if y.min() < 0:
            log1p = False
        self.log1p = log1p
        self.feature_importances_['feature'] = X.columns
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        if log1p:
            y = np.log1p(y)

        if tuning:
            log("[+]tuning params")
            self.optuna_tuning(X, y, metric=self.metric, Debug=Debug, log1p=log1p)

        folds = KFold(n_splits=self.n_fold, shuffle=True)
        RMSEs = []

        for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

            start_time = time()
            print('Training on fold {}'.format(fold_n + 1))
            X_train, y_train = X[train_index], y.iloc[train_index]
            X_valid, y_valid = X[valid_index], y.iloc[valid_index]
            model = xgb.XGBRegressor(**self.params_)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)], verbose=100, early_stopping_rounds=100)

            self.models.append(model)
            self.feature_importances_['fold_{}'.format(fold_n + 1)] = model.feature_importances_
            val = model.predict(X[valid_index])
            if log1p:
                mse_ = mean_squared_error(np.expm1(y.iloc[valid_index]), np.expm1(val), squared=True)
            else:
                mse_ = mean_squared_error(y.iloc[valid_index], val, squared=True)
            print('MSE: {}'.format(mse_))
            RMSEs.append(mse_)
            print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(
                                                                 seconds=time() - start_time))))
        log(f'Average KFold RMSE: {np.mean(RMSEs)}')
        self.feature_importances_['average'] = self.feature_importances_[
            [x for x in self.feature_importances_.columns if x != "feature"]].mean(axis=1)
        self.feature_importances_ = self.feature_importances_.sort_values(by="average", ascending=False)
        self.feature_importances_.index = range(len(self.feature_importances_))

    def predict(self, test):
        test = self.scaler.transform(test)
        for idx, model in enumerate(self.models):
            if idx == 0:
                result = model.predict(test)
            else:
                result += model.predict(test)
        result /= self.n_fold
        if self.log1p:
            result = np.expm1(result)
        return result

class CrossLgbRegression(object):
    def __init__(self, metric, params=None, n_fold=5):
        self.models = []
        self.feature_importances_ = pd.DataFrame()
        self.n_fold = n_fold
        self.metric = metric
        self.params_ = {
            'objective': 'regression',
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
        self.params_['metric'] = self.metric
        self.Early_Stopping_Rounds = 150
        self.N_round = 8000
        self.Verbose = 100
        self.log1p = None

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def optuna_tuning(self, X, y, metric, Debug=False, log1p=True):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            param_grid = {
                'num_leaves': trial.suggest_int('num_leaves', 2 ** 3, 2 ** 9),
                'num_boost_round': trial.suggest_int('num_boost_round', 100, 8000),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'objective': 'regression',
                'boosting': 'gbdt',
                'learning_rate': 0.01,
                'bagging_fraction': 0.95,
                'bagging_freq': 1,
                'bagging_seed': 66,
                'feature_fraction': 0.7,
                'feature_fraction_seed': 66,
                'max_bin': 100,
                'verbose': -1
            }
            param_grid['metric'] = metric
            trn_data = lgb.Dataset(X_train, label=y_train, categorical_feature="")
            val_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature="")
            clf = lgb.train(param_grid, trn_data, valid_sets=[trn_data, val_data], verbose_eval=False,
                            early_stopping_rounds=self.Early_Stopping_Rounds)
            pred_val = clf.predict(X_valid)
            if log1p:
                mse_ = mean_squared_error(np.expm1(y_valid), np.expm1(pred_val))
            else:
                mse_ = mean_squared_error(y_valid, pred_val)

            return mse_

        train_time = 1 * 10 * 60  # h * m * s
        if Debug:
            train_time = 1 * 1 * 60  # h * m * s
        study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='LgbRegressor')
        study.optimize(objective, timeout=train_time)

        log(f'Number of finished trials: {len(study.trials)}')
        log('Best trial:')
        trial = study.best_trial

        log(f'\tValue: {trial.value}')
        log('\tParams: ')
        for key, value in trial.params.items():
            log('\t\t{}: {}'.format(key, value))

        self.params_['num_leaves'] = trial.params['num_leaves']
        self.params_['max_depth'] = trial.params['max_depth']
        self.N_round = trial.params['num_boost_round']

    def fit(self, X, y, Early_Stopping_Rounds=None, N_round=None, Verbose=None, tuning=True, Debug=False, log1p=True):
        log(X.shape)
        if y.min() < 0:
            log1p = False
        self.log1p = log1p
        if log1p:
            y = np.log1p(y)

        if tuning:
            log("[+]tuning params")
            self.optuna_tuning(X, y, metric=self.metric, Debug=Debug, log1p=log1p)

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
            model = lgb.train(self.params_, trn_data, num_boost_round=self.N_round, valid_sets=[trn_data, val_data],
                            verbose_eval=self.Verbose,
                            early_stopping_rounds=self.Early_Stopping_Rounds)
            self.models.append(model)
            self.feature_importances_['fold_{}'.format(fold_n + 1)] = model.feature_importance()
            val = model.predict(X.iloc[valid_index])
            if log1p:
                mse_ = mean_squared_error(np.expm1(y.iloc[valid_index]), np.expm1(val))
            else:
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
        for idx, model in enumerate(self.models):
            if idx == 0:
                result = model.predict(test) / self.n_fold
            else:
                result += model.predict(test) / self.n_fold
        if self.log1p:
            result = np.expm1(result)
        return result