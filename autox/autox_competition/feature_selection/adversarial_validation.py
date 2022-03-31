import datetime
from time import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from autox.autox_competition.util import log
import warnings
warnings.filterwarnings("ignore")

class AdversarialValidation:
    """**Remove features with inconsistent distribution between train and test.**

    Example::
        `elo_AdversarialValidation_AutoX <https://www.kaggle.com/code/poteman/elo-adversarialvalidation-autox>`_

    """
    def __init__(self):
        self.removed_features = []

    def fit(self, train, test, id_, target, categorical_features=[], p=0.6):
        """
        :param train: dataframe, the training input samples.
        :param test: dataframe, the testing input samples.
        :param id_: list, columns as id.
        :param target: str, target column.
        :param categorical_features: list, columns with categorical type.
        :param p: float, threshold. If the auc is greater than this threshold, the algorithm will continuously remove the most important feature.
        """
        assert (p > 0.5)
        self.categorical_features = categorical_features

        train[target] = 0
        test[target] = 1
        train_test = pd.concat([train, test], axis=0)

        n_fold = 5
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=889)

        lr = 0.1
        Early_Stopping_Rounds = 20
        N_round = 50
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

        not_used = id_ + [target]
        used_features = [x for x in test.columns if x not in not_used]

        while True:
            log(f'used_features: {used_features}')
            log(f'categorical_features: {categorical_features}')
            AUCs = []
            feature_importances = pd.DataFrame()
            feature_importances['feature'] = train_test[used_features].columns

            for fold_n, (train_index, valid_index) in enumerate(folds.split(train_test[used_features])):
                start_time = time()
                log('Training on fold {}'.format(fold_n + 1))

                trn_data = lgb.Dataset(train_test[used_features].iloc[train_index],
                                       label=train_test[target].iloc[train_index],
                                       categorical_feature=categorical_features)
                val_data = lgb.Dataset(train_test[used_features].iloc[valid_index],
                                       label=train_test[target].iloc[valid_index],
                                       categorical_feature=categorical_features)
                clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data],
                                verbose_eval=Verbose,
                                early_stopping_rounds=Early_Stopping_Rounds)

                feature_importances['fold_{}'.format(fold_n + 1)] = clf.feature_importance()

                val = clf.predict(train_test[used_features].iloc[valid_index])

                auc_score = roc_auc_score(train_test[target].iloc[valid_index], val)
                log('AUC: {}'.format(auc_score))
                AUCs.append(auc_score)
                log('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(seconds=time() - start_time))))

            log(AUCs)
            mean_auc = np.mean(AUCs)
            log(f'Mean AUC: {mean_auc}')
            log('#' * 50)

            feature_importances['average'] = feature_importances[
                [x for x in feature_importances.columns if x != "feature"]].mean(axis=1)
            feature_importances = feature_importances.sort_values(by="average", ascending=False)
            feature_importances.index = range(len(feature_importances))
            feature_importances.head()

            if mean_auc > p:
                cur_removed_feature = feature_importances.loc[0, 'feature']
                log(f"remove feature {cur_removed_feature}")
                self.removed_features.append(cur_removed_feature)
                not_used = not_used + [cur_removed_feature]
                used_features = [x for x in test.columns if x not in not_used]
                categorical_features = [col for col in self.categorical_features if col in used_features]
            else:
                log(f"removed_features: {self.removed_features}")
                break

    def transform(self, df):
        """
        :param df: dataframe, dataframe needs to be transformed.
        :return: dataframe, transformed dataframe.
        """
        used = [x for x in df.columns if x not in self.removed_features]
        return df[used]
