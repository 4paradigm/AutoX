import time
import warnings

import pandas as pd
from tqdm import tqdm
from autox.autox_server.util import log
warnings.filterwarnings('ignore')

def fe_hash_discrete(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征
    start = time.time()
    log('[+] feature engineer, hash_discrete')

    if is_train:
        G_hist['FE_hash_discrete'] = []
        col_hash_discrete = []
        if G_hist['super_big_data']:
            for col in G_hist['big_cols_cat']:
                # nunique大于10000的特征，截断保留4位
                if G_df_dict['BIG'][col].nunique() > 10000:
                    col_hash_discrete.append(col)

        G_hist['FE_hash_discrete'] = col_hash_discrete
        log("hash_discrete features: {}".format(G_hist['FE_hash_discrete']))

    if not AMPERE:
        G_df_dict['FE_hash_discrete'] = pd.DataFrame()
        for col in tqdm(G_hist['FE_c']):
            G_df_dict['FE_kv'][f"{col}_hash_discrete"] = G_df_dict['BIG'][col].apply(lambda x: str(x)[-4:])

        # todo: 对应feql直接discrete签名

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time
