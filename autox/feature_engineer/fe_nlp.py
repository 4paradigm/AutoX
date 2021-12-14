from tqdm import tqdm
from autox.process_data import Feature_type_recognition
from ..CONST import FEATURE_TYPE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import lightgbm as lgb
from time import time
import datetime
import pandas as pd
import numpy as np

class FeatureNlp:
    def __init__(self):
        self.target = None
        self.df_feature_type = None
        self.silence_cols = []
        self.select_all = None
        self.max_num = None
        self.ops = []

    def fit(self, df, target, df_feature_type = None, silence_cols = [], select_all = True,
            max_num = None):

        self.target = target
        self.df_feature_type = df_feature_type
        self.silence_cols = silence_cols
        self.select_all = select_all
        self.max_num = max_num

        if self.df_feature_type is None:
            feature_type_recognition = Feature_type_recognition()
            feature_type = feature_type_recognition.fit(df)
            self.df_feature_type = feature_type

        for feature in self.df_feature_type.keys():
            if self.df_feature_type[feature] == FEATURE_TYPE['txt'] and feature not in self.silence_cols:
                self.ops.append([feature])

    def get_ops(self):
        return self.ops

    def set_keys(self, ops):
        self.ops = ops

    def transform(self, df):
        result = pd.DataFrame()

        for op in tqdm(self.ops):
            col = op[0]
            tfidfVectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                    analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), use_idf=1,
                                    smooth_idf=1,
                                    sublinear_tf=1)
            X_tfidf = tfidfVectorizer.fit_transform(df[col].astype(str))
            shape_of_train = df[df[self.target].notnull()].shape[0]
            shape_of_test = df[df[self.target].isnull()].shape[0]
            train = X_tfidf[:shape_of_train]
            test = X_tfidf[shape_of_train:]
            y = df.loc[df[self.target].notnull(), [self.target]]

            # model
            print(train.shape, test.shape)
            n_fold = 5
            folds = KFold(n_splits=n_fold, shuffle=True, random_state=889)

            lr = 0.1
            Early_Stopping_Rounds = 150
            N_round = 500
            Verbose = 20

            params = {'num_leaves': 41,  # 当前base 61
                      'min_child_weight': 0.03454472573214212,
                      'feature_fraction': 0.3797454081646243,
                      'bagging_fraction': 0.4181193142567742,
                      'min_data_in_leaf': 96,  # 当前base 106
                      'objective': 'binary',
                      'max_depth': -1,
                      'learning_rate': lr,  # 快速验证
                      "boosting_type": "gbdt",
                      "bagging_seed": 11,
                      "metric": 'auc',
                      "verbosity": -1,
                      'reg_alpha': 0.3899927210061127,
                      'reg_lambda': 0.6485237330340494,
                      'random_state': 47,
                      'num_threads': 16
                      }

            # save meta feature
            meta_train = pd.DataFrame(np.zeros([shape_of_train, 1]))
            meta_test = pd.DataFrame(np.zeros([shape_of_test, 1]))

            meta_train.columns = ["meta_feature"]
            meta_test.columns = ["meta_feature"]

            N_MODEL = 1.0
            for model_i in tqdm(range(int(N_MODEL))):

                if N_MODEL != 1.0:
                    params['seed'] = model_i + 1123

                for fold_n, (train_index, valid_index) in enumerate(folds.split(train)):

                    start_time = time()
                    print('Training on model {} - fold {}'.format(model_i + 1, fold_n + 1))

                    trn_data = lgb.Dataset(train[train_index], label=y.iloc[train_index], categorical_feature="")
                    val_data = lgb.Dataset(train[valid_index], label=y.iloc[valid_index], categorical_feature="")
                    clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data],
                                    verbose_eval=Verbose,
                                    early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror

                    val = clf.predict(train[valid_index])
                    pred = clf.predict(test)

                    # meta feature
                    meta_train.loc[valid_index, "meta_feature"] = val
                    meta_test["meta_feature"] += pred / float(n_fold)

                    print('Model {} - Fold {} finished in {}'.format(model_i + 1, fold_n + 1,
                                                                     str(datetime.timedelta(
                                                                         seconds=time() - start_time))))
            print("done!")
            meta_txt = meta_train
            meta_txt = meta_txt.append(meta_test)
            meta_txt.index = range(len(meta_txt))
            meta_txt.columns = [col + "_nlp"]
            result[col + "_nlp"] = meta_txt[col + "_nlp"]

        return result

    def fit_transform(self, df, target, df_feature_type = None, silence_cols = [], select_all = True,
            max_num = None):
        self.fit(df, target=target, df_feature_type=df_feature_type, silence_cols=silence_cols,
                        select_all=select_all, max_num=max_num)
        return self.transform(df)