import time
from autox.autox_server.util import log
from autox.autox_server.model import model_util
import warnings

warnings.filterwarnings('ignore')

def lgb_for_feature_selection(G_df_dict, G_data_info, G_hist, is_train, remain_time, params, lgb_para_dict, data_name, exp_name):

    start = time.time()
    log('[+] feature selection')

    if is_train:
        zero_importance_features = model_util.identify_zero_importance_features(G_df_dict['BIG_FE'], G_data_info,
                                                                                G_hist, is_train, remain_time, exp_name,
                                                                                params,
                                                                                lgb_para_dict, data_name)
        G_hist['zero_importance_features'] = zero_importance_features
        log("zero_importance_features: {}".format(zero_importance_features))

    used_features = [x for x in G_df_dict['BIG_FE'].columns if x not in G_hist['zero_importance_features']]
    G_df_dict['BIG_FE'] = G_df_dict['BIG_FE'][used_features]

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time