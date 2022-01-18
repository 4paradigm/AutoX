import warnings
warnings.filterwarnings('ignore')
from autox.autox_server.model import model_util

def base_lgb(G_df_dict, G_data_info, G_hist, is_train, remain_time, params, lgb_para_dict, exp_name):
    remain_time = model_util.lgb_model(G_df_dict['BIG'], G_data_info, G_hist, is_train, remain_time, 'base_lgb', params,
                            lgb_para_dict, exp_name)

    return remain_time