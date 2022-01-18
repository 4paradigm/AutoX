import warnings
import pandas as pd
import time
from autox.autox_server.util import log
from tqdm import tqdm
warnings.filterwarnings('ignore')


def fe_accumulate(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, accumulate')
    time_col = G_data_info['target_time']

    if is_train:
        G_hist['FE_Accumulate'] = {}
        G_hist['FE_Accumulate']['normal'] = []
        G_hist['FE_Accumulate']['time'] = []

        for col in tqdm(G_hist['big_cols_cat']):
            G_hist['FE_Accumulate']['normal'].append(col)
        log("accumulate normal features: {}".format(G_hist['FE_Accumulate']['normal']))

        if G_data_info['time_series_data'] == 'true':
            if G_hist['big_data_type'][time_col] == 'Unix_timestamp':
                G_df_dict['BIG'] = G_df_dict['BIG'].sort_values(by=time_col)

                for col in tqdm(G_hist['big_cols_cat']):
                    G_hist['FE_Accumulate']['time'].append(col)
                log("window features: {}".format(G_hist['FE_Accumulate']['time']))

    G_df_dict['FE_Accumulate'] = pd.DataFrame()
    for col in tqdm(G_hist['FE_Accumulate']['normal']):
        G_df_dict['FE_Accumulate'][f'{col}_acc_cnt'] = G_df_dict['BIG'].groupby(col).cumcount()

    for col in tqdm(G_hist['FE_Accumulate']['time']):
        G_df_dict['FE_Accumulate'][f'{col}_min_{time_col}'] = G_df_dict['BIG'].groupby(col)[time_col].transform('min')
        G_df_dict['FE_Accumulate'][f'{col}_acc_cnt_div_delta_time'] = G_df_dict['FE_Accumulate'][f'{col}_acc_cnt'] / \
                (G_df_dict['BIG'][time_col] - G_df_dict['FE_Accumulate'][f'{col}_min_{time_col}'] + 1)

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time