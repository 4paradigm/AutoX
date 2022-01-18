from itertools import combinations
from autox.autox_server.util import cols_concat
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log

def fe_concat_count(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, concat count')

    if is_train:
        size_of_big = G_df_dict['BIG'].shape[0]
        G_hist['FE_concat_count'] = {}
        G_hist['FE_concat_count']['feature_map'] = {}

        top_k = 6
        used_concat_cols = []
        cnt = 0
        for i in range(len(G_hist['base_lgb']['feature_importances'])):
            if cnt == top_k:
                break
            cur_feature = G_hist['base_lgb']['feature_importances'].loc[i, 'feature']
            if cur_feature in G_hist['big_cols_cat']:
                used_concat_cols.append(cur_feature)
                cnt += 1

        G_hist['FE_concat_count']['used_concat_cols'] = used_concat_cols
        log("concat count features: {}".format(used_concat_cols))

        concat_count_df = G_df_dict['BIG'][used_concat_cols + ['istrain']]
        concat_features = []
        for col_1, col_2 in combinations(used_concat_cols, 2):
            if col_1 == 'istrain' or col_2 == 'istrain':
                continue
            cur_col = col_1 + "__" + col_2
            log(cur_col)
            concat_features.append(cur_col)
            concat_count_df = cols_concat(concat_count_df, [col_1, col_2])

        G_hist['concat_features'] = concat_features
        for f in concat_features:
            if concat_count_df[f].nunique() > size_of_big * 0.75:
                continue
            temp = pd.DataFrame(concat_count_df.loc[concat_count_df['istrain'] == True][f])
            temp[f + '_cnt'] = temp.groupby([f])[f].transform('count')
            temp.index = temp[f]
            temp = temp.drop(f, axis=1)
            faeture_map = temp.to_dict()[f + '_cnt']
            G_hist['FE_concat_count']['feature_map'][f] = faeture_map

    if is_train:
        G_df_dict['FE_concat_count'] = concat_count_df
        G_df_dict['FE_concat_count'].drop(used_concat_cols + ['istrain'], axis=1, inplace=True)
    else:
        G_df_dict['FE_concat_count'] = G_df_dict['BIG'][G_hist['FE_concat_count']['used_concat_cols']]
        for col_1, col_2 in combinations(G_hist['FE_concat_count']['used_concat_cols'], 2):
            G_df_dict['FE_concat_count'] = cols_concat(G_df_dict['FE_concat_count'], [col_1, col_2])

    for f in G_hist['concat_features']:
        G_df_dict['FE_concat_count'][f + "_cnt"] = G_df_dict['FE_concat_count'][f].map(G_hist['FE_concat_count']['feature_map'][f])
        G_df_dict['FE_concat_count'].drop(f, axis=1, inplace=True)

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time


