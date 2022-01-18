import warnings
import pandas as pd
import time
from autox.autox_server.util import log
from tqdm import tqdm
warnings.filterwarnings('ignore')

def f_rolling_count(x):
    x.iloc[:] = range(1, len(x) + 1)
    return x

# 同一时间内，cat类型的值出现的次数
def fe_time_rolling_count(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征
    start = time.time()
    log('[+] feature engineer, time rolling count')
    time_col = G_data_info['target_time']
    
    if is_train:
        G_hist['FE_time_rolling_count'] = []

        if G_hist['big_data']:
            G_hist['time_col_is_unix'] = 'false'
            pass
        else:
            if G_data_info['time_series_data'] == 'true':
                G_hist['time_col_is_unix'] = 'false'
                if G_hist['big_data_type'][time_col] == 'Unix_timestamp':
                    G_hist['time_col_is_unix'] = 'true'
                G_df_dict['BIG'] = G_df_dict['BIG'].sort_values(by=time_col)

                for col in tqdm(G_hist['big_cols_cat']):
                    if G_df_dict['BIG'][col].nunique() / G_df_dict['BIG'].shape[0] < 0.9:
                        G_hist['FE_time_rolling_count'].append(col)
                log("time rolling count features: {}".format(G_hist['FE_time_rolling_count']))

    if not AMPERE:
        G_df_dict['FE_time_rolling_count'] = pd.DataFrame()
        if G_data_info['time_series_data'] == 'true':
            time_col_is_unix = G_hist['time_col_is_unix']
        for col in tqdm(G_hist['FE_time_rolling_count']):
            G_df_dict['FE_time_rolling_count'][f'{col}_time_rolling_count_unixTime_{time_col_is_unix}'] = G_df_dict['BIG'].groupby([col, time_col])[col].apply(f_rolling_count)

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time