import warnings
warnings.filterwarnings('ignore')
from autox.autox_server.feature_engineer.fe_count_ratio import fe_count_ratio
from autox.autox_server.feature_engineer.fe_count_map import fe_count_map

def fe_count(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):

    remain_time = fe_count_map(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE)

    # if G_data_info['time_series_data'] == 'true':
    #     # 对于时序数据的count特征，test部分通过train保留的字典进行映射
    #     remain_time = fe_count_map(G_df_dict, G_data_info, G_hist, is_train, remain_time)
    # else:
    #     # 对于非时序数据的count特征，计算count数和总体数据量的比例，test部分计算时拼上train再操作
    #     remain_time = fe_count_ratio(G_df_dict, G_data_info, G_hist, is_train, remain_time)

    return remain_time