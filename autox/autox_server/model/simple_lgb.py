import warnings
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log

from autox.autox_server.model import model_util
SAMPLE_LIMIT = model_util.SAMPLE_LIMIT

def simple_lgb(G_df_dict, G_data_info, G_hist, is_train, remain_time, loop_num = None):
    """
    """

    log('[+] simple lightgbm')

    time_budget = G_data_info['time_budget']
    Id = G_data_info['target_id']
    target = G_data_info['target_label']
    main_table_name = G_data_info['target_entity']

    if is_train:
        start = time.time()

        G_hist['simple_lgb'] = {}

        # 获得used_size, 训练数据不断增加
        data_size = G_df_dict['BIG'].shape[0]
        half_size = data_size // 2

        # todo: 优化训练数据集大小
        used_size = [2 ** i * 1000 for i in range(12)]
        used_size = [x for x in used_size if x < half_size]
        used_size.extend([half_size, data_size])
        if loop_num in range(0, 3):
            try:
                used_size = [used_size[loop_num]]
            except:
                used_size = [half_size]
        elif loop_num == 3:
            used_size = [half_size]

        end = time.time()
        remain_time -= (end - start)
        log("remain_time: {} s".format(remain_time))

        G_hist['simple_lgb']['simple_lgb_models'] = []
        G_hist['simple_lgb']['AUCs'] = []
        G_hist['simple_lgb']['used_features'] = []
        G_hist['simple_lgb']['feature_importances'] = []

        for rum_num in range(len(used_size)):

            start = time.time()

            train = G_df_dict['BIG'].sample(used_size[rum_num])

            # 如果数据集超过一定的数量，采用采样的方式
            if train.shape[0] >= SAMPLE_LIMIT:
                train = train.sample(SAMPLE_LIMIT)

            log("used size: {}".format(train.shape[0]))

            not_used = Id + [target, 'istrain']
            used_features = [x for x in list(train.describe().columns) if x not in not_used]

            G_hist['simple_lgb']['used_features'].append(used_features)

            # 保存feature imp
            feature_importances = pd.DataFrame()
            feature_importances['feature'] = train[used_features].columns

            log("feature size: {}".format(train[used_features].shape[1]))

            n_fold = 5
            folds = KFold(n_splits=n_fold, shuffle=True, random_state=889)

            quick = False
            if quick:
                lr = 0.1
                Early_Stopping_Rounds = 150
                N_round = 500
                Verbose = 20
            else:
                lr = 0.006883242363721497
                Early_Stopping_Rounds = 300
                N_round = 2000
                Verbose = 50

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
                      #           'is_unbalance':True
                      }

            for fold_n, (train_index, valid_index) in enumerate(folds.split(train[used_features])):

                if fold_n != 0:
                    break

                #                 log('Training on fold {}'.format(fold_n + 1))

                trn_data = lgb.Dataset(train[used_features].iloc[train_index], label=train[target].iloc[train_index],
                                       categorical_feature="")
                val_data = lgb.Dataset(train[used_features].iloc[valid_index], label=train[target].iloc[valid_index],
                                       categorical_feature="")
                clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data],
                                verbose_eval=Verbose,
                                early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror

                val = clf.predict(train[used_features].iloc[valid_index])

                # 对于不平衡数据，可能存在label全为一类的情况
                if train[target].iloc[valid_index].nunique() == 1:
                    log("only one calss of label in valid data. set auc = 0")
                    auc_score = 0.0
                else:
                    auc_score = roc_auc_score(train[target].iloc[valid_index], val)
                log('AUC: {}'.format(auc_score))
                G_hist['simple_lgb']['AUCs'].append(auc_score)

            G_hist['simple_lgb']['simple_lgb_models'].append(clf)

            end = time.time()
            remain_time -= (end - start)
            log("time consumption: {}".format(str(end - start)))
            log("remain_time: {} s".format(remain_time))
            log("#" * 50)

            feature_importances['average'] = clf.feature_importance()
            feature_importances = feature_importances.sort_values(by="average", ascending=False)
            G_hist['simple_lgb']['feature_importances'].append(feature_importances)

    else:
        start = time.time()

        Id = G_data_info['target_id']
        target = G_data_info['target_label']

        test = G_df_dict['BIG']
        test = test.loc[test['istrain'] == False]

        sub = test[Id]
        used_features = G_hist['simple_lgb']['used_features'][-1]
        used_model = G_hist['simple_lgb']['simple_lgb_models'][-1]
        sub[target] = used_model.predict(test[used_features])
        G_hist['predict']['simple_lgb'] = sub

        end = time.time()
        remain_time -= (end - start)
        log("remain_time: {} s".format(remain_time))

    return remain_time

