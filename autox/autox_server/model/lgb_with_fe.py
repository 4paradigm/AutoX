import warnings
warnings.filterwarnings('ignore')
from autox.autox_server.model import model_util

def lgb_with_fe(G_df_dict, G_data_info, G_hist, is_train, remain_time, params, lgb_para_dict, data_name, exp_name):
    remain_time = model_util.lgb_model(G_df_dict['BIG_FE'], G_data_info, G_hist, is_train, remain_time, exp_name, params,
                            lgb_para_dict, data_name)

    return remain_time