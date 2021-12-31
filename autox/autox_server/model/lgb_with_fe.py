import warnings
warnings.filterwarnings('ignore')
from autox.autox_server.model import model_util

def lgb_with_fe(G_df_dict, G_data_info, G_hist, is_train, remain_time, params, lgb_para_dict):
    remain_time = model_util.lgb_model(G_df_dict['BIG_FE'], G_data_info, G_hist, is_train, remain_time, 'fe_lgb', params,
                            lgb_para_dict)

    return remain_time