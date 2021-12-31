import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from autox.autox_server.model import model_util
from autox.autox_server.util import log

def lgb_para_opt(G_df_dict, G_data_info, G_hist, is_train, remain_time, params, lgb_para_dict):
    start = time.time()

    # 训练过程进行调参,预测过程使用最优参数进行预测
    if is_train:

        lgb_params_opt = []
        cnt = 0
        for N_leaves in [201]:
            params['num_leaves'] = N_leaves

            remain_time = model_util.lgb_model(G_df_dict['BIG_FE'], G_data_info, G_hist, is_train, remain_time, 'fe_lgb_opt',
                                    params.copy(), lgb_para_dict)

            if G_hist['fe_lgb_opt']['success']:
                used_features = G_hist['fe_lgb_opt']['used_features']
                auc = G_hist['val_auc']['fe_lgb_opt']
                model = G_hist['fe_lgb_opt']['model']
                feature_imp = G_hist['fe_lgb_opt']['feature_importances']
                lgb_params_opt.append([auc, params.copy(), model, used_features, feature_imp])

            cnt += 1

        lgb_params_opt = pd.DataFrame(lgb_params_opt)
        if lgb_params_opt.shape[0] > 0:
            lgb_params_opt.columns = ['auc', 'params', 'model', 'used_features', 'feature_imp']
            G_hist['lgb_params_opt'] = lgb_params_opt

        end = time.time()
        remain_time -= (end - start)

    else:
        if G_hist['fe_lgb_opt']['success']:

            G_hist['lgb_params_opt'] = G_hist['lgb_params_opt'].sort_values(by='auc')

            Id = G_data_info['target_id']
            target = G_data_info['target_label']

            test = G_df_dict['BIG_FE']
            test = test.loc[test['istrain'] == False]

            sub = test[Id]

            used_features = G_hist['lgb_params_opt'].loc[len(G_hist['lgb_params_opt']) - 1, 'used_features']
            used_model = G_hist['lgb_params_opt'].loc[len(G_hist['lgb_params_opt']) - 1, 'model']
            best_val_auc = G_hist['lgb_params_opt'].loc[len(G_hist['lgb_params_opt']) - 1, 'auc']

            sub[target] = used_model.predict(test[used_features])
            G_hist['predict']['fe_lgb_opt'] = sub
            G_hist['val_auc']['fe_lgb_opt'] = best_val_auc

        else:
            log("without fe_lgb_opt")

        end = time.time()
        remain_time -= (end - start)

    log("remain_time: {} s".format(remain_time))

    return remain_time
