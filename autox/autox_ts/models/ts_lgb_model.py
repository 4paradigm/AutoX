import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
from tqdm import tqdm
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import gc
from autox.autox_competition.util import log
import warnings
warnings.filterwarnings('ignore')
from autox.autox_ts.metrics import _get_score_metric

def ts_lgb_model(train, test, id_col, time_col, target_col, used_features, category_cols,
                 time_interval_num, time_interval_unit, forecast_period,
                 label_log, metric):
    # 切线下验证集: 最后一部分的数据作为线下验证
    if time_interval_unit == 'minute':
        delta = timedelta(minutes=time_interval_num * (forecast_period - 1))
        valid_time_split = train[time_col].max() - delta

    lgb_sub = test[[id_col, time_col]].copy()
    lgb_sub[target_col] = 0

    # target_col取log
    label_log = label_log
    if label_log:
        train[target_col] = train[target_col].apply(lambda x: np.log1p(x))

    valid_idx = train.loc[train[time_col] > valid_time_split].index
    train_idx = train.loc[train[time_col] <= valid_time_split].index

    print(train.iloc[train_idx][time_col].min(), train.iloc[train_idx][time_col].max())
    print(train.iloc[valid_idx][time_col].min(), train.iloc[valid_idx][time_col].max())

    print(train.shape, test.shape)

    print(f'used_features: {used_features}')
    print(len(used_features))

    quick = True
    if quick:
        lr = 0.1
        Early_Stopping_Rounds = 150
    else:
        lr = 0.006883242363721497
        Early_Stopping_Rounds = 300

    N_round = 5000
    Verbose_eval = 100

    params = {'num_leaves': 61,
              'min_child_weight': 0.03454472573214212,
              'feature_fraction': 0.3797454081646243,
              'bagging_fraction': 0.4181193142567742,
              'min_data_in_leaf': 96,
              'objective': 'regression',
              "metric": metric,
              'max_depth': -1,
              'learning_rate': lr,
              "boosting_type": "gbdt",
              "bagging_seed": 11,
              "verbosity": -1,
              'reg_alpha': 0.3899927210061127,
              'reg_lambda': 0.6485237330340494,
              'random_state': 47,
              'num_threads': 16,
              'lambda_l1': 1,
              'lambda_l2': 1
              }

    category = category_cols

    folds_metrics = []
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = train[used_features].columns

    N_MODEL = 1.0
    for model_i in tqdm(range(int(N_MODEL))):

        if N_MODEL != 1.0:
            params['seed'] = model_i + 1123

        start_time = time()
        print('Training on model {}'.format(model_i + 1))

        ## All data
        trn_data = lgb.Dataset(train.iloc[train_idx][used_features], label=train.iloc[train_idx][target_col],
                               categorical_feature=category)
        val_data = lgb.Dataset(train.iloc[valid_idx][used_features], label=train.iloc[valid_idx][target_col],
                               categorical_feature=category)

        clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data],
                        verbose_eval=Verbose_eval, early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror

        val = clf.predict(train.iloc[valid_idx][used_features])

        cur_metric = _get_score_metric(train.iloc[valid_idx][target_col], val, metric)
        print(f'{metric}: {cur_metric}')

        folds_metrics.append(cur_metric)
        feature_importances['model_{}'.format(model_i + 1)] = clf.feature_importance()

        ## 用数据集重新训练
        print("ReTraining on all data")

        gc.enable()
        del trn_data, val_data
        gc.collect()

        trn_data = lgb.Dataset(train[used_features], label=train[target_col], categorical_feature=category)
        clf = lgb.train(params, trn_data, num_boost_round=int(clf.best_iteration * 1.15),
                        valid_sets=[trn_data], verbose_eval=Verbose_eval)  # , feval=evalerror

        pred = clf.predict(test[used_features])
        lgb_sub[target_col] = lgb_sub[target_col] + pred / N_MODEL

        print('Model {} finished in {}'.format(model_i + 1, str(timedelta(seconds=time() - start_time))))

    lgb_sub_mean = lgb_sub.groupby([id_col, time_col]).agg({target_col: ['mean']}).reset_index()
    lgb_sub_mean.columns = ['_'.join(x) if x[1] != '' else x[0] for x in list(lgb_sub_mean.columns)]

    feature_importances['average'] = feature_importances[
        [x for x in feature_importances.columns if x != "feature"]].mean(axis=1)
    feature_importances = feature_importances.sort_values(by="average", ascending=False)
    feature_importances.index = range(len(feature_importances))

    return lgb_sub_mean, feature_importances
