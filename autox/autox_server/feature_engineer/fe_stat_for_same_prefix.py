import warnings
import pandas as pd
import time
from autox.autox_server.util import log
from tqdm import tqdm
warnings.filterwarnings('ignore')
import re

def fe_stat_for_same_prefix(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征
    start = time.time()
    log('[+] feature engineer, stat_for_same_prefix')

    if is_train:
        G_hist['FE_stat_for_same_prefix'] = []

        cols_agg_list = []
        cols = G_df_dict['BIG'].columns
        c_1_list = [col for col in cols if bool(re.search(r'_1$', str(col)))]
        for c_1 in c_1_list:
            c_list = [c_1]
            for i in range(2, 20):
                c_i = c_1.replace('_1', '_{}'.format(i))
                if c_i in cols:
                    c_list.append(c_i)
            num_flag = True
            for item in c_list:
                if str(G_df_dict['BIG'][item].dtype) == 'object':
                    num_flag = False
            if num_flag and 3 <= len(c_list) <= 3:
                cols_agg_list.append(c_list)
        G_hist['FE_stat_for_same_prefix'] = cols_agg_list
        log("stat_for_same_prefix features: {}".format(G_hist['FE_stat_for_same_prefix']))

    if not AMPERE:
        G_df_dict['FE_stat_for_same_prefix'] = pd.DataFrame()
        for cols_agg in tqdm(G_hist['FE_stat_for_same_prefix']):
            G_df_dict['FE_stat_for_same_prefix']['{}__stat_for_same_prefix__mean'.format('__col__'.join(cols_agg))] = G_df_dict['BIG'][cols_agg].mean(axis = 1)
            # G_df_dict['FE_stat_for_same_prefix']['{}__stat_for_same_prefix__median'.format('__col__'.join(cols_agg))] = G_df_dict['BIG'][cols_agg].median(axis = 1)
            G_df_dict['FE_stat_for_same_prefix']['{}__stat_for_same_prefix__min'.format('__col__'.join(cols_agg))] = G_df_dict['BIG'][cols_agg].min(axis = 1)
            G_df_dict['FE_stat_for_same_prefix']['{}__stat_for_same_prefix__max'.format('__col__'.join(cols_agg))] = G_df_dict['BIG'][cols_agg].max(axis = 1)
            # G_df_dict['FE_stat_for_same_prefix']['{}__stat_for_same_prefix__std'.format('__col__'.join(cols_agg))] = G_df_dict['BIG'][cols_agg].std(axis = 1)

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time
