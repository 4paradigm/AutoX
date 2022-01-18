import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log
from tqdm import tqdm
from itertools import combinations

def _groupby_agg__window_count__equal_to_cur(df, keys, col, op, k, col_time):
    if type(k) == int:
        s = df.groupby(keys)[[col]]
        df_gp = s.rolling(window = k, min_periods=1).agg(op)[col]
        temp = pd.DataFrame(list(df_gp), index = list(df_gp.index.codes[2])).reindex(df.index)
    else:
        closed = 'right'
        s = df.groupby(keys)[[col_time, col]]
        df_gp = s.rolling(window = k, on = col_time, closed = closed).agg(op)[col]
        temp = pd.DataFrame(list(df_gp), index = list(df_gp.index.codes[2])).reindex(df.index)
    return temp[0]

def fe_window_count(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征

    # 聚合uid, 求窗口内iid等于当前iid的count数

    start = time.time()
    log('[+] feature engineer, window_count')

    big_size = G_df_dict['BIG'].shape[0]
    time_col = G_data_info['target_time']

    if is_train:
        G_hist['FE_window_count'] = {}
        G_hist['FE_window_count']['key_features'] = []
        G_hist['FE_window_count']['col_features'] = []

        if G_data_info['time_series_data'] == 'true':
            G_df_dict['BIG'] = G_df_dict['BIG'].sort_values(by=time_col)

            key_features = []
            for col in G_hist['big_cols_cat']:
                if big_size * 0.15 < G_df_dict['BIG'][col].nunique() < big_size * 0.2:
                    key_features.append(col)

            if G_hist['big_data']:
                key_features += [x for x in G_hist['big_cols_cat'] if G_df_dict['BIG'][x].nunique() == 2]

            G_hist['FE_window_count']['key_features'] = key_features
            log("FE_window_count key_features: {}".format(key_features))

            col_features = []
            for col in G_hist['big_cols_cat']:
                if big_size * 0.0005 < G_df_dict['BIG'][col].nunique() < big_size * 0.001:
                    col_features.append(col)

            if G_hist['big_data']:
                col_features += [x for x in G_hist['big_cols_cat'] if 30 < G_df_dict['BIG'][x].nunique() < 50]

            G_hist['FE_window_count']['col_features'] = col_features
            log("FE_window_count col_features: {}".format(col_features))

    if not AMPERE:
        G_df_dict['FE_window_count'] = pd.DataFrame()
        for w in tqdm([10]):
            for key in G_hist['FE_window_count']['key_features']:
                for col in G_hist['FE_window_count']['col_features']:
                    if key == col:
                        continue
                    ans = _groupby_agg__window_count__equal_to_cur(G_df_dict['BIG'], [key] + [col], col, 'count', w, time_col)
                    G_df_dict['FE_window_count'][f'{key}__with__{col}__window_count__{w}'] = ans

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time