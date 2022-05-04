import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, StratifiedKFold

import pandas as pd
import numpy as np


class BaggingRegressor():
    def __init__(self, regressors, seeds = [2022], n_fold=5):
        self.regressors = regressors
        self.n_regressors = 1 if type(self.regressors) != list else len(self.regressors)
        self.fitted_regressors = []
        self.seeds = seeds
        self.n_seeds = len(self.seeds)
        self.n_fold = n_fold
        self.folds = None

    def fit(self, X, y):
        for idx, cur_regressor in enumerate(self.regressors):
            cur_fitted_regressors = []
            for seed in self.seeds:
                self.folds = KFold(n_splits=self.n_fold, shuffle=True, random_state=seed)
                for fold_n, (train_index, valid_index) in enumerate(self.folds.split(X, y)):
                    clf = cur_regressor.fit(X.loc[train_index], y.loc[train_index],
                                            eval_set=[(X.loc[valid_index], y.loc[valid_index])], 
                                            verbose = 50)
                    cur_fitted_regressors.append(clf)    
            self.fitted_regressors.append(cur_fitted_regressors)
        print('Training Done')

    def predict(self, X, regressor_weights = []):
        predict_test = pd.DataFrame()
        if np.sum(regressor_weights) != 1:
            regressor_weights = np.ones(self.n_regressors) / self.n_regressors

        for idx, cur_fitted_regressors in enumerate(self.fitted_regressors):
            for i, cur_fitted_regressor in enumerate(cur_fitted_regressors):
                if i == 0:
                    pred = cur_fitted_regressor.predict(X) / float(self.n_fold) / float(self.n_seeds)
                else:
                    pred += cur_fitted_regressor.predict(X) / float(self.n_fold) / float(self.n_seeds)
            predict_test['model_%d_predict' % (idx)] = pred * regressor_weights[idx]
        self.result = predict_test.sum(axis = 1)
        print('Prediction Done')
        return self.result

class BaggingClassifier():
    def __init__(self, classifiers, seeds = [2022], n_fold=5):
        self.classifiers = classifiers
        self.n_classifiers = 1 if type(self.classifiers) != list else len(self.classifiers)
        self.fitted_classifiers = []
        self.seeds = seeds
        self.n_seeds = len(self.seeds)
        self.n_fold = n_fold
        self.folds = None

    def fit(self, X, y, custom_metric_list = []):
        for idx, cur_classifier in enumerate(self.classifiers):
            cur_fitted_classifiers = []
            if idx < len(custom_metric_list):
                cur_metric = custom_metric_list[idx]
            else:
                cur_metric = 'auc'
            for seed in self.seeds:
                self.folds = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=seed)
                for fold_n, (train_index, valid_index) in enumerate(self.folds.split(X, y)):
                    clf = cur_classifier.fit(X.loc[train_index], y.loc[train_index],
                                            eval_set=[(X.loc[valid_index], y.loc[valid_index])], 
                                            eval_metric = cur_metric,
                                            verbose = 50)
                    cur_fitted_classifiers.append(clf)    
            self.fitted_classifiers.append(cur_fitted_classifiers)
        print('Training Done')

    def predict_proba(self, X, classifier_weights = []):
        predict_proba_test = pd.DataFrame()
        if np.sum(classifier_weights) != 1:
            classifier_weights = np.ones(len(self.classifiers)) / len(self.classifiers)

        for idx, cur_fitted_classifiers in enumerate(self.fitted_classifiers):
            for i, cur_fitted_classifier in enumerate(cur_fitted_classifiers):
                if i == 0:
                    pred = cur_fitted_classifier.predict_proba(X)[:, 1] / float(self.n_fold) / float(self.n_seeds)
                else:
                    pred += cur_fitted_classifier.predict_proba(X)[:, 1] / float(self.n_fold) / float(self.n_seeds)
            predict_proba_test['model_%d_predict_proba' % (idx)] = pred * classifier_weights[idx]
        self.result = predict_proba_test.sum(axis = 1)
        print('Prediction Done')
        return self.result

    def predict(self, X, classifier_weights = []):
        predict_proba_test = pd.DataFrame()
        if np.sum(classifier_weights) != 1:
            classifier_weights = np.ones(len(self.classifiers)) / len(self.classifiers)

        for idx, cur_fitted_classifiers in enumerate(self.fitted_classifiers):
            for i, cur_fitted_classifier in enumerate(cur_fitted_classifiers):
                if i == 0:
                    pred = cur_fitted_classifier.predict_proba(X)[:, 1] / float(self.n_fold) / float(self.n_seeds)
                else:
                    pred += cur_fitted_classifier.predict_proba(X)[:, 1] / float(self.n_fold) / float(self.n_seeds)
            predict_proba_test['model_%d_predict_proba' % (idx)] = pred * classifier_weights[idx]
        self.result = predict_proba_test.sum(axis = 1)
        print('Prediction Done')
        return self.result.argmax(axis = 1)
    
 