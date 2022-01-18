import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log
from tqdm import tqdm


def fe_window(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, window')

    big_size = G_df_dict['BIG'].shape[0]
    time_col = G_data_info['target_time']

    if is_train:
        G_hist['FE_window'] = []

        if G_data_info['time_series_data'] == 'true':
            if G_hist['big_data_type'][time_col] == 'Unix_timestamp':
                G_df_dict['BIG'] = G_df_dict['BIG'].sort_values(by=time_col)

                window_features = []
                for col in G_hist['big_cols_cat']:
                    if big_size * 0.01 < G_df_dict['BIG'][col].nunique() < big_size * 0.3:
                        window_features.append(col)
                G_hist['FE_window'] = window_features
                log("window features: {}".format(window_features))

    G_df_dict['FE_window'] = pd.DataFrame()
    for col in tqdm(G_hist['FE_window']):
        for w in [3]:
            f_std  = lambda x: x.rolling(window=w, min_periods=1).std()
            # f_mean = lambda x: x.rolling(window=w, min_periods=1).mean()
            G_df_dict['FE_window'][f'{col}_{time_col}_rolling_std_{w}'] = G_df_dict['BIG'].groupby([col])[time_col].apply(f_std)
            # G_df_dict['FE_window'][f'{col}_{time_col}_rolling_mean_{w}'] = G_df_dict['BIG'].groupby([col])[time_col].apply(f_mean)

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time