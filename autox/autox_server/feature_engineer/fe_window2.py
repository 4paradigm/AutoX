import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log
from tqdm import tqdm

def fe_window2(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, window2')

    big_size = G_df_dict['BIG'].shape[0]
    time_col = G_data_info['target_time']

    if is_train:
        G_hist['FE_window2'] = {}
        G_hist['FE_window2']['num_agg_features'] = []
        G_hist['FE_window2']['cat_agg_features'] = []
        G_hist['FE_window2']['window_cat_features'] = []

        if G_hist['big_data']:
            pass
        else:
            if G_data_info['time_series_data'] == 'true':
                G_df_dict['BIG'] = G_df_dict['BIG'].sort_values(by=time_col)

                window_cat_features = []
                for col in G_hist['big_cols_cat']:
                    if big_size * 0.02 < G_df_dict['BIG'][col].nunique() < big_size * 0.1:
                        window_cat_features.append(col)

                # 增加前缀一致的特征组
                if len(G_hist['same_prefix_cols']) > 0:
                    window_cat_features.extend(G_hist['same_prefix_cols'])

                G_hist['FE_window2']['window_cat_features'] = window_cat_features
                log("window2 cat features: {}".format(window_cat_features))

                num_agg_features = []
                for col in G_hist['big_cols_num']:
                    num_agg_features.append(col)
                G_hist['FE_window2']['num_agg_features'] = num_agg_features
                log("FE_window2 num_agg_features: {}".format(num_agg_features))

                cat_agg_features = []
                for col in G_hist['big_cols_cat']:
                    if big_size * 0.6 < G_df_dict['BIG'][col].nunique() < big_size * 0.8:
                        cat_agg_features.append(col)
                G_hist['FE_window2']['cat_agg_features'] = cat_agg_features
                log("FE_window2 cat_agg_features: {}".format(cat_agg_features))

    if not AMPERE:
        G_df_dict['FE_window2'] = pd.DataFrame()
        for w in tqdm([10, '432001s']):
            for cat_col in G_hist['FE_window2']['window_cat_features']:
                for num_col in G_hist['FE_window2']['num_agg_features']:
                    if type(w) == int:
                        # f_std = lambda x: x.rolling(window=w, min_periods=1).std()
                        f_mean = lambda x: x.rolling(window=w, min_periods=1).mean()
                        f_sum = lambda x: x.rolling(window=w, min_periods=1).sum()
                        # G_df_dict['FE_window2'][f'{cat_col}_{num_col}_rolling_std_{w}'] = G_df_dict['BIG'].groupby([cat_col])[num_col].apply(f_std)
                        G_df_dict['FE_window2'][f'{cat_col}__with__{num_col}_rolling_mean_{w}'] = G_df_dict['BIG'].groupby([cat_col])[num_col].apply(f_mean)
                        G_df_dict['FE_window2'][f'{cat_col}__with__{num_col}_rolling_sum_{w}'] = G_df_dict['BIG'].groupby([cat_col])[num_col].apply(f_sum)
                    else:
                        s = G_df_dict['BIG'].groupby(cat_col)[[time_col, num_col]]
                        df_gp_mean = s.rolling(window=w, on=time_col, closed='right').agg('mean').iloc[:, -1:]
                        df_gp_sum  = s.rolling(window=w, on=time_col, closed='right').agg('sum').iloc[:, -1:]

                        df_gp_mean = df_gp_mean.reset_index()
                        df_gp_mean.drop([cat_col], axis=1, inplace=True)
                        df_gp_mean = df_gp_mean.set_index('level_1')
                        df_gp_mean = df_gp_mean.reindex(G_df_dict['BIG'].index)

                        df_gp_sum = df_gp_sum.reset_index()
                        df_gp_sum.drop([cat_col], axis=1, inplace=True)
                        df_gp_sum = df_gp_sum.set_index('level_1')
                        df_gp_sum = df_gp_sum.reindex(G_df_dict['BIG'].index)

                        G_df_dict['FE_window2'][f'{cat_col}__with__{num_col}_rolling_mean_{w}'] = df_gp_mean.iloc[:, 0]
                        G_df_dict['FE_window2'][f'{cat_col}__with__{num_col}_rolling_sum_{w}']  = df_gp_sum.iloc[:, 0]

                for cat_agg_col in G_hist['FE_window2']['cat_agg_features']:
                    if type(w) == int:
                        s = G_df_dict['BIG'].groupby(cat_col)[[cat_agg_col]]
                        df_gp_unique = s.rolling(window=w, min_periods=1).agg(pd.Series.nunique).iloc[:, -1:]
                        df_gp_unique = df_gp_unique.reset_index()
                        df_gp_unique.drop([cat_col], axis=1, inplace=True)
                        df_gp_unique = df_gp_unique.set_index('level_1')
                        df_gp_unique = df_gp_unique.reindex(G_df_dict['BIG'].index)

                        G_df_dict['FE_window2'][f'{cat_col}__with__{cat_agg_col}_rolling_unique_{w}'] = df_gp_unique.iloc[:, 0]
                    else:
                        s = G_df_dict['BIG'].groupby(cat_col)[[time_col, cat_agg_col]]
                        df_gp_unique = s.rolling(window=w, on=time_col, closed='right').agg(pd.Series.nunique).iloc[:, -1:]
                        df_gp_unique = df_gp_unique.reset_index()
                        df_gp_unique.drop([cat_col], axis=1, inplace=True)
                        df_gp_unique = df_gp_unique.set_index('level_1')
                        df_gp_unique = df_gp_unique.reindex(G_df_dict['BIG'].index)
                        G_df_dict['FE_window2'][f'{cat_col}__with__{cat_agg_col}_rolling_unique_{w}'] = df_gp_unique.iloc[:, 0]

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time