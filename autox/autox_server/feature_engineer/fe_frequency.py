import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log

def _groupby_agg_rolling(df, keys, col, op, k, col_time):
    name = 'WIN_{}_{}_({})_({})'.format(k, op.upper(), '_'.join(keys), col)
    if type(k) == int:
        s = df.groupby(keys)[[col]]
        df_gp = s.rolling(window = k).agg(op) # rolling by number
    else:
        closed = 'left' # [left, right)
        # closed = 'both' # [left, right]
        s = df.groupby(keys)[[col_time, col]]
        df_gp = s.rolling(window = k, on = col_time, closed = closed).agg(op).iloc[:, -1:] # rolling by time
    df_gp.columns = [name]
    df_gp = df_gp.sort_index(level = 1).reset_index(drop = True)
    return df_gp

def fe_frequency(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, frequency')

    big_size = G_df_dict['BIG'].shape[0]
    time_col = G_data_info['target_time']

    if is_train:
        G_hist['FE_frequency'] = {}
        G_hist['FE_frequency']['keys'] = []
        G_hist['FE_frequency']['cols'] = []

        if G_data_info['time_series_data'] == 'true':
            # !先对df排序
            G_df_dict['BIG'] = G_df_dict['BIG'].sort_values(by=time_col)

            keys_features = []
            for col in G_hist['big_cols_cat']:
                if big_size * 0.005 < G_df_dict['BIG'][col].nunique() < big_size * 0.01:
                    keys_features.append(col)
            G_hist['FE_frequency']['keys'] = keys_features
            log("FE_frequency keys: {}".format(keys_features))

            cols_features = []
            for col in G_hist['big_cols_cat']:
                if big_size * 0.6 < G_df_dict['BIG'][col].nunique() < big_size * 0.8:
                    cols_features.append(col)
            G_hist['FE_frequency']['cols'] = cols_features
            log("FE_frequency cols: {}".format(cols_features))

    if not AMPERE:
        G_df_dict['FE_frequency'] = pd.DataFrame()
        for col in G_hist['FE_frequency']['cols']:
            for key_ in G_hist['FE_frequency']['keys']:
                df = G_df_dict['BIG'][[key_, col]].copy()
                keys = [key_]
                df['x'] = df.groupby(keys + [col])[col].transform('count') / df.groupby(keys)[col].transform('count')
                df['y'] = df.groupby(keys)['x'].transform('max')
                G_df_dict['FE_frequency'][f'{key_}__with__{col}__frequency'] = df['y']

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time