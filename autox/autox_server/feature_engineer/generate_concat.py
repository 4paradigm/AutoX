import warnings
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log, cols_concat
from itertools import combinations

def generate_concat(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展
    start = time.time()
    log('[+] feature engineer, generate_concat')

    if is_train:

        # 各种数据类型
        data_type_ = G_data_info['entities'][G_data_info['target_entity']]['columns']
        big_data_type = {}
        for item in data_type_:
            key_ = list(item.keys())[0]
            value_ = list(item.values())[0]
            big_data_type[key_] = value_
        G_hist['big_data_type'] = big_data_type

        big_cols_datetime = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'DateTime']
        big_cols_datetime = [x for x in big_cols_datetime if x in G_df_dict['BIG'].columns]
        G_hist['big_cols_datetime'] = big_cols_datetime

        big_cols_Unix_timestamp = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Unix_timestamp']
        big_cols_Unix_timestamp = [x for x in big_cols_Unix_timestamp if x in G_df_dict['BIG'].columns]
        G_hist['big_cols_Unix_timestamp'] = big_cols_Unix_timestamp

        big_cols_cat = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Str']
        big_cols_cat = [x for x in big_cols_cat if x in G_df_dict['BIG'].columns]
        G_hist['big_cols_cat'] = big_cols_cat

        big_cols_num = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Num']
        big_cols_num = [x for x in big_cols_num if x in G_df_dict['BIG'].columns]
        G_hist['big_cols_num'] = big_cols_num

        big_cols_multi_value = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Multi_value']
        big_cols_multi_value = [x for x in big_cols_multi_value if x in G_df_dict['BIG'].columns]
        G_hist['big_cols_multi_value'] = big_cols_multi_value

        top_k = 5
        used_concat_cols = []
        cnt = 0
        for i in range(len(G_hist['base_lgb']['feature_importances'])):
            if cnt >= top_k:
                break
            cur_feature = G_hist['base_lgb']['feature_importances'].loc[i, 'feature']
            if cur_feature in [x + '_label_encoder' for x in G_hist['big_cols_cat']]:
                used_concat_cols.append(cur_feature[:-14])
                cnt += 1

        # 增加所有的multi_valve特征
        for cur_feature in G_hist['big_cols_multi_value']:
            used_concat_cols.append(cur_feature)

        G_hist['used_concat_cols'] = used_concat_cols
        log("used_concat_cols: {}".format(used_concat_cols))

        concat_features = []
        for col_1, col_2 in combinations(used_concat_cols, 2):
            cur_col = col_1 + "__" + col_2
            concat_features.append(cur_col)

        G_hist['concat_features'] = concat_features
        log("concat_features: {}".format(concat_features))

    for item in G_hist['concat_features']:
        col_1 = item.split('__')[0]
        col_2 = item.split('__')[1]
        G_df_dict['BIG'] = cols_concat(G_df_dict['BIG'], [col_1, col_2])

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time