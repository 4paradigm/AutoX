import warnings
import pandas as pd
import time
from autox.autox_server.util import log
from tqdm import tqdm
warnings.filterwarnings('ignore')

def fe_time_count(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征
    start = time.time()
    log('[+] feature engineer, time count')
    time_col = G_data_info['target_time']

    if is_train:
        G_hist['FE_time_count'] = []
        size_of_big = G_df_dict['BIG'].shape[0]
        if G_data_info['time_series_data'] == 'true':
            G_df_dict['BIG'] = G_df_dict['BIG'].sort_values(by=time_col)
            for col in G_hist['big_cols_cat']:
                if G_df_dict['BIG'][col].nunique() < size_of_big * 0.8:
                    G_hist['FE_time_count'].append(col)

        if G_hist['big_data']:
            G_hist['FE_time_count'] = []

        log("time count features: {}".format(G_hist['FE_time_count']))

    if not AMPERE:
        G_df_dict['FE_time_count'] = pd.DataFrame()
        for col in tqdm(G_hist['FE_time_count']):
            G_df_dict['FE_time_count'][f'{col}__time_count'] = G_df_dict['BIG'].groupby([col, time_col])[col].transform('count')

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time