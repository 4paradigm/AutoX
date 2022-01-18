import warnings
import pandas as pd
import time
from autox.autox_server.util import log
from tqdm import tqdm
warnings.filterwarnings('ignore')

def fe_count_ratio(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征
    start = time.time()
    log('[+] feature engineer, count ratio')

    if is_train:
        G_hist['FE_count_ratio'] = {}
        size_of_big = G_df_dict['BIG'].shape[0]

        cnt_ratio_features = []
        for col in G_hist['big_cols_cat'] + G_hist['big_cols_num']:
            if G_df_dict['BIG'][col].nunique() < size_of_big * 0.8:
                cnt_ratio_features.append(col)
        G_hist['FE_count_ratio'] = cnt_ratio_features
        log("count ratio features: {}".format(cnt_ratio_features))

    G_df_dict['FE_count_ratio'] = pd.DataFrame()
    for col in tqdm(G_hist['FE_count_ratio']):
        G_df_dict['FE_count_ratio'][col + "_cnt_ratio"] = G_df_dict['BIG'].groupby(col)[col].transform('count') / \
                                            G_df_dict['BIG'].shape[0]

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time
