import datetime
from time import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from ..util import log
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

class CrossTabnetBiClassifier(object):
    pass
    # https://pypi.org/project/pytorch-tabnet/



class CrossXgbBiClassifier(object):
    def __init__(self, params=None, n_fold=10):
        self.models = []
        self.scaler = None
        self.feature_importances_ = pd.DataFrame()
        self.n_fold = n_fold
        self.params_ = {
            'eta': 0.01,
            'max_depth': 5,
            'subsample': 0.6,
            'n_estimators': 2500,
            'reg_alpha': 40,
            'reg_lambda': 18,
            'min_child_weight': 16,
            # 'tree_method': 'gpu_hist'
        }
        if params is not None:
            self.params_ = params

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def optuna_tuning(self, X, y, Debug=False):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.4)
        def objective(trial):
            param_grid = {
                'max_depth': trial.suggest_int('max_depth', 4, 15),
                'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000, 100),
                'eta': 0.01,
                'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
                'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            }
            reg = xgb.XGBClassifier(**param_grid) # tree_method='gpu_hist',
            reg.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)], eval_metric='auc',
                    verbose=False)
            return roc_auc_score(y_valid, reg.predict_proba(X_valid)[:,1])

        train_time = 1 * 10 * 60  # h * m * s
        if Debug:
            train_time = 1 * 1 * 60  # h * m * s
        study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name='XGBClassifier')
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

    def fit(self, X, y, tuning=True, Debug=False):
        log(X.shape)
        self.feature_importances_['feature'] = X.columns
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        if tuning:
            log("[+]tuning params")
            self.optuna_tuning(X, y, Debug=Debug)

        folds = KFold(n_splits=self.n_fold, shuffle=True)
        AUCs = []

        for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

            start_time = time()
            print('Training on fold {}'.format(fold_n + 1))
            X_train, y_train = X[train_index], y.iloc[train_index]
            X_valid, y_valid = X[valid_index], y.iloc[valid_index]
            model = xgb.XGBClassifier(**self.params_)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      eval_metric='auc', verbose=False)

            self.models.append(model)
            self.feature_importances_['fold_{}'.format(fold_n + 1)] = model.feature_importances_
            val = model.predict_proba(X[valid_index])[:,1]
            auc_ = roc_auc_score(y.iloc[valid_index], val)
            print('AUC: {}'.format(auc_))
            AUCs.append(auc_)
            print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(
                                                                 seconds=time() - start_time))))
        log(f'Average KFold AUC: {np.mean(AUCs)}')
        self.feature_importances_['average'] = self.feature_importances_[
            [x for x in self.feature_importances_.columns if x != "feature"]].mean(axis=1)
        self.feature_importances_ = self.feature_importances_.sort_values(by="average", ascending=False)
        self.feature_importances_.index = range(len(self.feature_importances_))

    def predict(self, test):
        test = self.scaler.transform(test)
        for idx, model in enumerate(self.models):
            if idx == 0:
                result = model.predict_proba(test)[:,1]
            else:
                result += model.predict_proba(test)[:,1]
        result /= self.n_fold
        return result

class CrossLgbBiClassifier(object):
    def __init__(self, params=None, n_fold=5):
        self.models = []
        self.feature_importances_ = pd.DataFrame()
        self.n_fold = n_fold
        self.params_ = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
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
        self.Early_Stopping_Rounds = 150
        self.N_round = 8000
        self.Verbose = 100

    def get_params(self):
        return self.params_

    def set_params(self, params):
        self.params_ = params

    def optuna_tuning(self, X, y, Debug=False):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        def objective(trial):
            param_grid = {
                'num_leaves': trial.suggest_int('num_leaves', 2 ** 3, 2 ** 9),
                'num_boost_round': trial.suggest_int('num_boost_round', 100, 8000),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'bagging_fraction': 0.95,
                'bagging_freq': 1,
                'bagging_seed': 66,
                'feature_fraction': 0.7,
                'feature_fraction_seed': 66,
                'max_bin': 100,
                'verbose': -1
            }
            trn_data = lgb.Dataset(X_train, label=y_train, categorical_feature="")
            val_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature="")
            clf = lgb.train(param_grid, trn_data, valid_sets=[trn_data, val_data], verbose_eval=False,
                            early_stopping_rounds=self.Early_Stopping_Rounds)
            pred_val = clf.predict(X_valid)
            auc_ = roc_auc_score(y_valid, pred_val)

            return auc_

        train_time = 1 * 10 * 60  # h * m * s
        if Debug:
            train_time = 1 * 1 * 60  # h * m * s
        study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name='LgbClassifier')
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

    def fit(self, X, y, Early_Stopping_Rounds=None, N_round=None, Verbose=None, tuning=True, Debug=False):
        log(X.shape)

        if tuning:
            log("[+]tuning params")
            self.optuna_tuning(X, y, Debug=Debug)

        if Early_Stopping_Rounds is not None:
            self.Early_Stopping_Rounds = Early_Stopping_Rounds
        if N_round is not None:
            self.N_round = N_round
        if Verbose is not None:
            self.Verbose = Verbose

        folds = KFold(n_splits=self.n_fold, shuffle=True, random_state=889)
        AUCs = []
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
            auc_ = roc_auc_score(y.iloc[valid_index], val)
            print('AUC: {}'.format(auc_))
            AUCs.append(auc_)
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
        return result