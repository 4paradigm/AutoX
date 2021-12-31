import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
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


