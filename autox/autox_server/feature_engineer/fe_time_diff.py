import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
import numpy as np
from autox.autox_server.util import log
from tqdm import tqdm

def fe_time_diff(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, time_diff')

    time_col = G_data_info['target_time']

    G_df_dict['FE_cat_time_diff'] = pd.DataFrame()

    window_size = 3
    if G_data_info['time_series_data'] == 'true':
        window = range(1, window_size + 1)
    else:
        window = range(-window_size, window_size + 1)
    window = [x for x in window if x != 0]

    log("big_cols_Unix_timestamp: {}".format(G_hist['big_cols_Unix_timestamp']))
    log("big_cols_datetime: {}".format(G_hist['big_cols_datetime']))

    for cat_col in tqdm(G_hist['big_cols_cat'] + G_hist['concat_features']):
        for i in window:
            for time_col_unix in G_hist['big_cols_Unix_timestamp']:
                if G_data_info['time_series_data'] == 'true':
                    if time_col != time_col_unix:
                        temp_df = G_df_dict['BIG'][[cat_col, time_col_unix, time_col]].sort_values([cat_col, time_col])
                    else:
                        temp_df = G_df_dict['BIG'][[cat_col, time_col]].sort_values([cat_col, time_col])
                else:
                    temp_df = G_df_dict['BIG'][[cat_col, time_col_unix]].sort_values([cat_col, time_col_unix])

                cat_diff  = temp_df[cat_col].astype('category').cat.codes.diff(i)
                time_diff = temp_df[time_col_unix].diff(i)
                time_diff[cat_diff != 0] = np.nan

                G_df_dict['FE_cat_time_diff'][f'{cat_col}_{time_col_unix}_diff_{i}'] = time_diff.reindex(G_df_dict['BIG'].index)

            for time_col_data_time in G_hist['big_cols_datetime']:
                if G_data_info['time_series_data'] == 'true':
                    if time_col != time_col_data_time:
                        temp_df = G_df_dict['BIG'][[cat_col, time_col_data_time, time_col]].sort_values([cat_col, time_col])
                    else:
                        temp_df = G_df_dict['BIG'][[cat_col, time_col]].sort_values([cat_col, time_col])
                else:
                    temp_df = G_df_dict['BIG'][[cat_col, time_col_data_time]].sort_values([cat_col, time_col_data_time])

                cat_diff = temp_df[cat_col].astype('category').cat.codes.diff(i)
                time_diff = temp_df[time_col_data_time].diff(i).dt.seconds
                time_diff[cat_diff != 0] = np.nan

                G_df_dict['FE_cat_time_diff'][f'{cat_col}_{time_col_data_time}_diff_{i}'] = time_diff.reindex(G_df_dict['BIG'].index)

    end = time.time()
    remain_time -= (end-start)
    log("remain_time: {} s".format(remain_time))
    return remain_time
