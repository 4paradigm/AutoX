import warnings
import pandas as pd
import numpy as np
import time
from autox.autox_server.util import log
from tqdm import tqdm
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from pypinyin import pinyin, lazy_pinyin, Style

def str2map(s):
    if str(s) == 'None':
        return {}
    return {si.split(':')[0]: si.split(':')[1] for si in s.split(',')}

def get_keys(kv):
    return list(kv.keys())

def fe_kv(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征
    start = time.time()
    log('[+] feature engineer, kv')

    if is_train:
        G_hist['FE_kv'] = {}
        G_hist['FE_kv']['cols'] = []
        G_hist['FE_kv']['col_top_keys'] = {}

        cols_kv = [x for x in G_hist['big_cols_kv'] if x in G_df_dict['BIG'].columns]
        G_hist['FE_kv']['cols'] = cols_kv
        log("kv features: {}".format(G_hist['FE_kv']['cols']))

        for col in cols_kv:
            temp = G_df_dict['BIG'][[col]].copy()
            temp[col] = temp[col].apply(lambda x: str2map(x))
            temp[col + '_keys'] = temp[col].apply(lambda x: get_keys(x))

            vectorizer = CountVectorizer(max_features=100)
            vectorizer.fit_transform(temp[col + '_keys'].astype(str))
            G_hist['FE_kv']['col_top_keys'][col] = vectorizer.get_feature_names()

    if not AMPERE:
        G_df_dict['FE_kv'] = pd.DataFrame()
        for col in tqdm(G_hist['FE_kv']['cols']):
            for key_ in G_hist['FE_kv']['col_top_keys'][col]:
                temp = G_df_dict['BIG'][[col]].copy()
                temp[col] = temp[col].apply(lambda x: str2map(x))
                try:
                    G_df_dict['FE_kv'][f"{col}__{key_}__kv"] = temp[col].apply(lambda x: float(x.get(key_, np.nan)))
                except:
                    pass

        G_hist['FE_kv']['rename'] = {}
        cols_name = []
        for i, col in enumerate(G_df_dict['FE_kv'].columns):
            col_rename = ''.join(lazy_pinyin(col)) + f'__idx{i}'
            cols_name.append(col_rename)
            G_hist['FE_kv']['rename'][col_rename] = col
        G_df_dict['FE_kv'].columns = cols_name


    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time
