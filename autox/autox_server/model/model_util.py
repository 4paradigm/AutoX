import warnings
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
import time
import gc
from autox.autox_server.util import log

# KEEP_TIME = 100 # 上线用
KEEP_TIME = -100000 # 测试效果用

SAMPLE_LIMIT = 2e7

def identify_zero_importance_features(df, G_data_info, G_hist, is_train, remain_time, exp_name, params, lgb_para_dict, data_name):
    category = ""
    Verbose = lgb_para_dict['Verbose']

    log('[+] exp_name: {}'.format(exp_name))

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    G_hist[exp_name] = {}
    G_hist[exp_name]['success'] = True

    train = df
    # 如果数据集超过一定的数量，采用采样的方式
    if train.shape[0] >= SAMPLE_LIMIT:
        train = train.sample(int(SAMPLE_LIMIT))

    log('{}| train.shape: {}'.format(exp_name, train.shape))

    not_used = Id + [target, 'istrain']
    used_features = [x for x in list(train.describe().columns) if x not in not_used]
    used_features = sorted(used_features)

    if len(used_features) < 100:
        log("{}| used_features: {}".format(exp_name, used_features))
    G_hist[exp_name]['used_features'] = used_features

    # 保存feature imp
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = train[used_features].columns

    log('{}| train[used_features].shape: {}'.format(exp_name, train[used_features].shape))

    # 切分数据集
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=889)

    # 时序数据,切线下验证集: 最后一部分的数据作为线下验证
    if G_data_info['time_series_data'] == 'true':
        log("{}| split data with time".format(exp_name))
        train = train.sort_values(by=G_data_info['target_time'])
        train_index_time = train.loc[:int(len(train) * 0.8)].index
        valid_index_time = train.loc[int(len(train) * 0.8) + 1:].index

    for fold_n, (train_index, valid_index) in enumerate(folds.split(train[used_features])):

        if fold_n != 0:
            break

        if G_data_info['target_time'] != '':
            log("{}| use time series index".format(exp_name))
            train_index = train_index_time
            valid_index = valid_index_time

        # 重排序
        train = train.sort_index()
        train_index = sorted(list(train_index))
        valid_index = sorted(list(valid_index))

        log("{}| {} for train, {} for valid".format(exp_name, len(train_index), len(valid_index)))

        ## 用数据集重新训练
        log("{}| Training on all data".format(exp_name))
        log("{}| train[used_features]:{}".format(exp_name, train[used_features].shape))

        trn_data = lgb.Dataset(train[used_features], label=train[target], categorical_feature=category)
        clf = lgb.train(params, trn_data, num_boost_round = 200,
                        valid_sets=[trn_data], verbose_eval=Verbose)  # , feval=evalerror

    G_hist[exp_name]['model'] = clf

    feature_importances['average'] = clf.feature_importance()
    feature_importances = feature_importances.sort_values(by="average", ascending=False)
    G_hist[exp_name]['feature_importances'] = feature_importances
    log("{}| feature_importances:".format(exp_name))
    log(feature_importances.set_index(["feature"])['average'].to_dict())

    zero_importance_features = list(feature_importances.loc[feature_importances['average'] == 0, 'feature'])
    return zero_importance_features


def lgb_model(df, G_data_info, G_hist, is_train, remain_time, exp_name, params, lgb_para_dict, data_name):
    category = ""
    Early_Stopping_Rounds = lgb_para_dict['Early_Stopping_Rounds']
    N_round = lgb_para_dict['N_round']
    Verbose = lgb_para_dict['Verbose']

    log('[+] exp_name: {}'.format(exp_name))

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:

        start = time.time()

        G_hist[exp_name] = {}
        G_hist[exp_name]['success'] = True

        train = df

        # 如果数据集超过一定的数量，采用采样的方式
        if train.shape[0] >= SAMPLE_LIMIT:
            train = train.sample(int(SAMPLE_LIMIT))

        log('{}| train.shape: {}'.format(exp_name, train.shape))

        not_used = Id + [target, 'istrain']
        used_features = [x for x in list(train.describe().columns) if x not in not_used]
        used_features = sorted(used_features)

        if len(used_features) < 100:
            log("{}| used_features: {}".format(exp_name, used_features))
        G_hist[exp_name]['used_features'] = used_features

        # # develop_tune直接return
        # end = time.time()
        # remain_time -= (end - start)
        # log("time consumption: {}".format(str(end - start)))
        # log("remain_time: {} s".format(remain_time))
        # return remain_time


        # 保存feature imp
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = train[used_features].columns

        log('{}| train[used_features].shape: {}'.format(exp_name, train[used_features].shape))

        # 切分数据集
        n_fold = 5
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=889)

        # 时序数据,切线下验证集: 最后一部分的数据作为线下验证
        if G_data_info['time_series_data'] == 'true':
            log("{}| split data with time".format(exp_name))
            train = train.sort_values(by=G_data_info['target_time'])
            train_index_time = train.loc[:int(len(train) * 0.8)].index
            valid_index_time = train.loc[int(len(train) * 0.8) + 1:].index

        for fold_n, (train_index, valid_index) in enumerate(folds.split(train[used_features])):

            if fold_n != 0:
                break

            if G_data_info['target_time'] != '':
                log("{}| use time series index".format(exp_name))
                train_index = train_index_time
                valid_index = valid_index_time

            # 重排序
            train = train.sort_index()
            train_index = sorted(list(train_index))
            valid_index = sorted(list(valid_index))

            log("{}| {} for train, {} for valid".format(exp_name, len(train_index), len(valid_index)))

            trn_data = lgb.Dataset(train[used_features].iloc[train_index], label=train[target].iloc[train_index],
                                   categorical_feature=category, free_raw_data=False)
            val_data = lgb.Dataset(train[used_features].iloc[valid_index], label=train[target].iloc[valid_index],
                                   categorical_feature=category, free_raw_data=False)

            log("{}| [+] Pre training for estimate time".format(exp_name))
            # todo:第一轮的耗时比较大,2-10轮耗时相对较少,考虑用内置评分函数返回时间
            start_time_small_lgb = time.time()
            clf = lgb.train(params, trn_data, num_boost_round=10, valid_sets=[trn_data, val_data], verbose_eval=Verbose,
                            early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror
            lgb_10_round = time.time() - start_time_small_lgb
            estimated_time = lgb_10_round * (N_round / 10)
            log("{}| estimate time: {}".format(exp_name, estimated_time))

            if estimated_time + KEEP_TIME > remain_time:
                G_hist[exp_name]['success'] = False
                end = time.time()
                remain_time -= (end - start)
                log("{}| estimate time exceed the remain time.".format(exp_name))
                return remain_time

            log("[+] Formal training")
            clf = lgb.train(params, trn_data, num_boost_round=N_round, valid_sets=[trn_data, val_data],
                            verbose_eval=Verbose,
                            early_stopping_rounds=Early_Stopping_Rounds)  # , feval=evalerror

            val = clf.predict(train[used_features].iloc[valid_index])
            auc_score = roc_auc_score(train[target].iloc[valid_index], val)
            log('{}| AUC: {}'.format(exp_name, auc_score))
            G_hist['val_auc'][exp_name] = auc_score

            ## 用数据集重新训练
            log("{}| ReTraining on all data".format(exp_name))
            log("{}| train[used_features]:{}".format(exp_name, train[used_features].shape))


            # #######################   debug   ################
            # if train[used_features].shape[0] < 100 * 10000:
            #     import os
            #     path_output = './temp/'
            #     os.makedirs(path_output, exist_ok=True)
            #     log("{}| save train[used_features] in {}".format(exp_name, path_output))
            #     train[used_features].to_hdf("./temp/online_train_all_{}.hdf".format(data_name), 'w', complib='blosc', complevel=5)
            # #######################   debug   ################



            gc.enable()
            del trn_data, val_data
            gc.collect()

            # todo: 不同的迭代倍数尝试
            if G_data_info['time_series_data'] == 'true':
                amplification_factor = 1.3
            else:
                amplification_factor = 1.15

            trn_data = lgb.Dataset(train[used_features], label=train[target], categorical_feature=category)
            clf = lgb.train(params, trn_data, num_boost_round = max(int(clf.best_iteration * amplification_factor), 140),
                            valid_sets=[trn_data], verbose_eval=Verbose)  # , feval=evalerror

        G_hist[exp_name]['model'] = clf

        end = time.time()
        remain_time -= (end - start)
        log("time consumption: {}".format(str(end - start)))
        log("remain_time: {} s".format(remain_time))
        log("#" * 50)

        feature_importances['average'] = clf.feature_importance()
        feature_importances = feature_importances.sort_values(by="average", ascending=False)
        G_hist[exp_name]['feature_importances'] = feature_importances
        log("{}| feature_importances:".format(exp_name))
        log(feature_importances.set_index(["feature"])['average'].to_dict())
    else:
        start = time.time()

        if G_hist[exp_name]['success']:
            Id = G_data_info['target_id']
            target = G_data_info['target_label']

            test = df
            test = test.loc[test['istrain'] == False]

            sub = test[Id]
            used_features = G_hist[exp_name]['used_features']
            used_model = G_hist[exp_name]['model']
            sub[target] = used_model.predict(test[used_features])
            G_hist['predict'][exp_name] = sub

        else:
            log("without {}".format(exp_name))

        end = time.time()
        remain_time -= (end - start)
        log("time consumption: {}".format(str(end - start)))
        log("remain_time: {} s".format(remain_time))

    return remain_time


###################### 第1组参数 #################
lgb_para_dict_1 = {
    'Early_Stopping_Rounds': 100,
    'N_round': 3500,
    'Verbose': 20
}
params_1 = {
    "objective": "binary", "metric": "auc", 'verbosity': -1, "seed": 0, 'two_round': False,
    'num_leaves': 20, 'learning_rate': 0.05, 'bagging_fraction': 0.9, 'bagging_freq': 3,
    'feature_fraction': 0.9, 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5,
    'lambda_l2': 0.5, 'min_data_in_leaf': 50, 'num_threads': 16
}

###################### 第2组参数 #################
lgb_para_dict_2 = {
    'Early_Stopping_Rounds': 200,
    'N_round': 3500,
    'Verbose': 20
}
params_2 = {'num_leaves': 61,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 50,  # 当前base 106
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
          'num_threads': 16
         }








lr_params = {
        'C': 0.09536298444122952,
        'class_weight': 'balanced',
        'max_iter': 1000,
        'solver': 'lbfgs',
        'n_jobs': -1
}
from sklearn.linear_model import LogisticRegression

def lr_model(df, G_data_info, G_hist, is_train, remain_time, name, lr_params):

    log('[+] {} lr model'.format(name))

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:

        start = time.time()

        G_hist[name] = {}
        G_hist[name]['success'] = True

        train = df

        # 如果数据集超过一定的数量，采用采样的方式
        if train.shape[0] >= SAMPLE_LIMIT:
            train = train.sample(SAMPLE_LIMIT)

        log('train.shape: {}'.format(train.shape))

        not_used = Id + [target, 'istrain']
        used_features = [x for x in list(train.describe().columns) if x not in not_used]
        used_features = sorted(used_features)

        if len(used_features) < 100:
            log("used_features: {}".format(used_features))
        G_hist[name]['used_features'] = used_features

        # # 保存feature imp
        # feature_importances = pd.DataFrame()
        # feature_importances['feature'] = train[used_features].columns

        log('train[used_features].shape: {}'.format(train[used_features].shape))
        # 切分数据集
        n_fold = 5
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=889)

        # 时序数据,切线下验证集: 最后一部分的数据作为线下验证
        if G_data_info['target_time'] != '':
            log("split data with time")
            train = train.sort_values(by=G_data_info['target_time'])
            train_index_time = train.loc[:int(len(train) * 0.8)].index
            valid_index_time = train.loc[int(len(train) * 0.8) + 1:].index

        for fold_n, (train_index, valid_index) in enumerate(folds.split(train[used_features])):

            if fold_n != 0:
                break

            if G_data_info['target_time'] != '':
                log("use time series index")
                train_index = train_index_time
                valid_index = valid_index_time

            # 重排序
            train = train.sort_index()
            train_index = sorted(list(train_index))
            valid_index = sorted(list(valid_index))

            log("{} for train, {} for valid".format(len(train_index), len(valid_index)))


            clf = LogisticRegression(C=lr_params['C'], class_weight = lr_params['class_weight'],
                                     max_iter=lr_params['max_iter'],
                                     solver=lr_params['solver'], n_jobs=lr_params['n_jobs'])
            clf.fit(train[used_features].iloc[train_index], train[target].iloc[train_index])

            val = clf.predict(train[used_features].iloc[valid_index])
            auc_score = roc_auc_score(train[target].iloc[valid_index], val)
            log('AUC: {}'.format(auc_score))
            G_hist['val_auc'][name] = auc_score

            ## 用数据集重新训练
            log("ReTraining on all data")
            log("train[used_features]:{}".format(train[used_features].shape))

            #######################   debug   ################
            import os
            path_output = './temp/'
            os.makedirs(path_output, exist_ok=True)
            log("save train[used_features] in {}".format(path_output))
            train[used_features].to_hdf("./temp/online_train_all.hdf", 'w', complib='blosc', complevel=5)
            #######################   debug   ################

            # todo: 不同的迭代倍数尝试
            clf = LogisticRegression(C=lr_params['C'], class_weight=lr_params['class_weight'],
                                     max_iter=lr_params['max_iter'] * 1.15,
                                     solver=lr_params['solver'], n_jobs=lr_params['n_jobs'])
            clf.fit(train[used_features], train[target])

        G_hist[name]['model'] = clf

        end = time.time()
        remain_time -= (end - start)
        log("remain_time: {} s".format(remain_time))
        log("#" * 50)

    else:
        start = time.time()

        if G_hist[name]['success']:
            Id = G_data_info['target_id']
            target = G_data_info['target_label']

            test = df
            test = test.loc[test['istrain'] == False]

            sub = test[Id]
            used_features = G_hist[name]['used_features']
            used_model = G_hist[name]['model']
            sub[target] = used_model.predict(test[used_features])
            G_hist['predict'][name] = sub

        else:
            log("without {}".format(name))

        end = time.time()
        remain_time -= (end - start)
        log("remain_time: {} s".format(remain_time))

    return remain_time