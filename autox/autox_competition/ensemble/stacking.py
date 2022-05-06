import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, StratifiedKFold

import pandas as pd
import numpy as np


class StackingRegressor():
    def __init__(self, regressors, meta_regressor, n_fold=5):
        self.regressors = regressors
        self.fitted_regressors = []
        self.meta_regressor = meta_regressor
        self.train_meta = pd.DataFrame()
        self.test_meta = pd.DataFrame()
        for i in range(len(regressors)):
            self.train_meta[f"meta_feature_{i}"] = 0
            self.test_meta[f"meta_feature_{i}"] = 0
        self.n_fold = n_fold
        folds = KFold(n_splits=self.n_fold, shuffle=True, random_state=889)
        self.folds = folds

    def fit(self, X, y):
        self.train_meta = pd.DataFrame(np.zeros([X.shape[0], len(self.regressors)]))
        self.train_meta.columns = [f"meta_feature_{i}" for i in range(1, len(self.regressors)+1)]
        for idx, cur_regressor in enumerate(self.regressors):
            cur_fitted_regressors = []
            for fold_n, (train_index, valid_index) in enumerate(self.folds.split(X)):
                print('Training on fold {}'.format(fold_n + 1))
                clf = cur_regressor.fit(X, y)
                cur_fitted_regressors.append(clf)
                val = clf.predict(X.iloc[valid_index])
                self.train_meta.loc[valid_index, f"meta_feature_{idx+1}"] = val
            self.fitted_regressors.append(cur_fitted_regressors)
        self.meta_regressor.fit(self.train_meta, y)

    def predict(self, X):
        self.test_meta = pd.DataFrame(np.zeros([X.shape[0], len(self.regressors)]))
        self.test_meta.columns = [f"meta_feature_{i}" for i in range(1, len(self.regressors) + 1)]
        for idx, cur_fitted_regressors in enumerate(self.fitted_regressors):
            for i, cur_fitted_regressor in enumerate(cur_fitted_regressors):
                if i == 0:
                    pred = cur_fitted_regressor.predict(X) / float(self.n_fold)
                else:
                    pred += cur_fitted_regressor.predict(X) / float(self.n_fold)
            self.test_meta[f'meta_feature_{idx+1}'] = pred
        self.result = self.meta_regressor.predict(self.test_meta)

class StackingClassifier():
    def __init__(self, classifiers, meta_classifier, n_fold=5):
        self.classifiers = classifiers
        self.fitted_classifiers = []
        self.meta_classifier = meta_classifier
        self.train_meta = pd.DataFrame()
        self.test_meta = pd.DataFrame()
        for i in range(len(classifiers)):
            self.train_meta[f"meta_feature_{i}"] = 0
            self.test_meta[f"meta_feature_{i}"] = 0
        self.n_fold = n_fold
        folds = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=2022)
        self.folds = folds

    def fit(self, X, y, custom_metric_list = []):
        self.train_meta = pd.DataFrame(np.zeros([X.shape[0], len(self.classifiers)]))
        self.train_meta.columns = [f"meta_feature_{i}" for i in range(1, len(self.classifiers)+1)]
        for idx, cur_classifier in enumerate(self.classifiers):
            cur_fitted_classifiers = []
            if idx < len(custom_metric_list):
                cur_metric = custom_metric_list[idx]
            else:
                cur_metric = 'auc'
            for fold_n, (train_index, valid_index) in enumerate(self.folds.split(X, y)):
                print('Training on fold {}'.format(fold_n + 1))
                clf = cur_classifier.fit(X.loc[train_index], y.loc[train_index],
                                        eval_set=[(X.loc[valid_index], y.loc[valid_index])], 
                                        eval_metric = cur_metric,
                                        verbose = 50)
                cur_fitted_classifiers.append(clf)
                val = clf.predict_proba(X.iloc[valid_index])[:,1]
                self.train_meta.loc[valid_index, f"meta_feature_{idx+1}"] = val
            self.fitted_classifiers.append(cur_fitted_classifiers)
        self.meta_classifier.fit(self.train_meta, y)
        print('Training Done')

    def predict_proba(self, X):
        self.test_meta = pd.DataFrame(np.zeros([X.shape[0], len(self.classifiers)]))
        self.test_meta.columns = [f"meta_feature_{i}" for i in range(1, len(self.classifiers) + 1)]
        for idx, cur_fitted_classifiers in enumerate(self.fitted_classifiers):
            for i, cur_fitted_classifier in enumerate(cur_fitted_classifiers):
                if i == 0:
                    pred = cur_fitted_classifier.predict_proba(X)[:,1] / float(self.n_fold)
                else:
                    pred += cur_fitted_classifier.predict_proba(X)[:,1] / float(self.n_fold)
            self.test_meta[f'meta_feature_{idx+1}'] = pred
        self.result = self.meta_classifier.predict_proba(self.test_meta)
        print('Prediction Done')
        return self.result

    def predict(self, X):
        self.test_meta = pd.DataFrame(np.zeros([X.shape[0], len(self.classifiers)]))
        self.test_meta.columns = [f"meta_feature_{i}" for i in range(1, len(self.classifiers) + 1)]
        for idx, cur_fitted_classifiers in enumerate(self.fitted_classifiers):
            for i, cur_fitted_classifier in enumerate(cur_fitted_classifiers):
                if i == 0:
                    pred = cur_fitted_classifier.predict_proba(X)[:,1] / float(self.n_fold)
                else:
                    pred += cur_fitted_classifier.predict_proba(X)[:,1] / float(self.n_fold)
            self.test_meta[f'meta_feature_{idx+1}'] = pred
        self.result = self.meta_classifier.predict_proba(self.test_meta)
        print('Prediction Done')
        return self.result.argmax(axis = 1) 
    
 