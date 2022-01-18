import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')
from autox.autox_server.util import del_invalid_features, rename_columns
from autox.autox_server.util import log
from tqdm import tqdm
import re

def is_in(x):
    sub = str(x[0])
    total = str(x[1])
    if ',' in total:
        total = total.split(',')
    if ':' in total:
        total = total.split(':')
    if type(total) == list and sub in total:
        return 1
    return 0

def label_encoder(df_table, G_data_info, G_hist, is_train):
    # 类别型特征label_encoder

    Id = G_data_info['target_id']

    if is_train:
        G_hist['preprocess_3']['label_encoder'] = {}
        G_hist['label_encoder_nunique'] = {}

        for col in df_table.columns:
            if col in Id:
                continue
            if 'O' == df_table[col].dtype:
                lbl = LabelEncoder()
                temp = pd.DataFrame(df_table[col].astype(str))
                col_nunique = temp[col].nunique()
                log('col: {}, nunique: {}'.format(col, col_nunique))
                G_hist['label_encoder_nunique'][col + '_label_encoder'] = col_nunique
                lbl.fit(temp[col])
                G_hist['preprocess_3']['label_encoder'][col] = lbl

    for col in G_hist['preprocess_3']['label_encoder']:
        lbl = G_hist['preprocess_3']['label_encoder'][col]
        kv = dict(zip(lbl.classes_, lbl.transform(lbl.classes_)))
        df_table[col + '_label_encoder'] = df_table[col].astype(str).map(kv).fillna(-1).astype(int)

    return df_table


def process_multi_value(df, G_data_info, G_hist, is_train):

    if is_train:
        G_hist['preprocess_3']['multi_value_cols'] = {}
        G_hist['preprocess_3']['is_in'] = []

        for col in tqdm(G_hist['big_cols_multi_value']):

            # 默认sep为':'
            # G_hist['preprocess_3']['multi_value_cols'][col] = ':'

            # 获得sep
            for sep in [':', ',']:
                temp = df[col].str.split(sep, expand=True)
                if temp.shape[1] != 1:
                    G_hist['preprocess_3']['multi_value_cols'][col] = sep


        for col_in in tqdm(G_hist['big_cols_cat']):  # + G_hist['big_cols_num']
            for col_out in G_hist['big_cols_multi_value']:
                df['{}_in_{}'.format(col_in, col_out)] = df[[col_in, col_out]].apply(lambda x: is_in(x), axis=1)
                if df['{}_in_{}'.format(col_in, col_out)].nunique() == 1:
                    df.drop('{}_in_{}'.format(col_in, col_out), axis=1, inplace=True)
                else:
                    log("{} in {}".format(col_in, col_out))
                    G_hist['preprocess_3']['is_in'].append('{}_in_{}'.format(col_in, col_out))
    else:
        ## is_in特征
        for col in G_hist['preprocess_3']['is_in']:
            col_in = col.split("_in_")[0]
            col_out = col.split("_in_")[1]
            df['{}_in_{}'.format(col_in, col_out)] = df[[col_in, col_out]].apply(lambda x: is_in(x), axis=1)


    ## 将multi_value分解获得特征
    for col in tqdm(G_hist['big_cols_multi_value']):
        sep = G_hist['preprocess_3']['multi_value_cols'][col]
        temp = df[col].str.split(sep, expand=True)
        temp.columns = ['multi_split_' + col + '_' + str(x) for x in temp.columns]
        used_cols = list(temp.columns)[:3]
        temp = temp[used_cols]
        df = pd.concat([df, temp], axis=1)

    return df

def get_same_prefix_cols(df):
    cols = list(df.columns)
    cols_agg_list = []
    c_1_list = [col for col in cols if bool(re.search(r'1$', str(col)))]
    for c_1 in c_1_list:
        c_list = [c_1]
        for i in range(2, 20):
            c_i = c_1.replace('1', '{}'.format(i))
            if c_i in cols:
                c_list.append(c_i)
        num_flag = True
        for item in c_list:
            if str(df[item].dtype) == 'object':
                num_flag = False
        if num_flag and 6 <= len(c_list) <= 6:
            cols_agg_list.append(c_list)
    return cols_agg_list

def get_info_from_big(df, G_data_info, G_hist, is_train):
    if is_train:
        # 各种数据类型
        target = G_data_info['target_label']
        # 只包含主表的数据列
        data_type_ = G_data_info['entities'][G_data_info['target_entity']]['columns']
        G_hist['data_type_'] = data_type_

        big_data_type = {}
        for item in data_type_:
            key_ = list(item.keys())[0]
            value_ = list(item.values())[0]
            big_data_type[key_] = value_
        G_hist['big_data_type'] = big_data_type
        log(f"[+]get_info_from_big: big_data_type: {G_hist['big_data_type']}")

        multi_value_cols = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Multi_value']
        multi_value_cols = [x for x in multi_value_cols if x in df.columns]
        G_hist['big_cols_multi_value'] = multi_value_cols
        log(f"[+]get_info_from_big: big_cols_multi_value: {G_hist['big_cols_multi_value']}")

        txt_cols = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Txt']
        txt_cols = [x for x in txt_cols if x in df.columns]
        G_hist['big_cols_txt'] = txt_cols
        log(f"[+]get_info_from_big: big_cols_txt: {G_hist['big_cols_txt']}")

        big_cols_datetime = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'DateTime']
        big_cols_datetime = [x for x in big_cols_datetime if x in df.columns]
        G_hist['big_cols_datetime'] = big_cols_datetime
        log(f"[+]get_info_from_big: big_cols_datetime: {G_hist['big_cols_datetime']}")

        big_cols_Unix_timestamp = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Unix_timestamp']
        big_cols_Unix_timestamp = [x for x in big_cols_Unix_timestamp if x in df.columns]
        G_hist['big_cols_Unix_timestamp'] = big_cols_Unix_timestamp
        log(f"[+]get_info_from_big: big_cols_Unix_timestamp: {G_hist['big_cols_Unix_timestamp']}")

        big_cols_cat = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Str']
        big_cols_cat = [x for x in big_cols_cat if x in df.columns]
        big_cols_cat = [x for x in big_cols_cat if x != target]
        G_hist['big_cols_cat'] = big_cols_cat
        log(f"[+]get_info_from_big: big_cols_cat: {G_hist['big_cols_cat']}")

        big_cols_num = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Num']
        big_cols_num = [x for x in big_cols_num if x in df.columns]
        big_cols_num = [x for x in big_cols_num if x != target]
        G_hist['big_cols_num'] = big_cols_num
        log(f"[+]get_info_from_big: big_cols_num: {G_hist['big_cols_num']}")

        big_cols_kv = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'KVString(,)[:]']
        big_cols_kv = [x for x in big_cols_kv if x in df.columns]
        big_cols_kv = [x for x in big_cols_kv if x != target]
        G_hist['big_cols_kv'] = big_cols_kv
        log(f"[+]get_info_from_big: big_cols_kv: {G_hist['big_cols_kv']}")

        G_hist['same_prefix_cols'] = []
        same_prefix_cols = get_same_prefix_cols(df)
        G_hist['same_prefix_cols'] = same_prefix_cols
        log(f"[+]get_info_from_big: same_prefix_cols: {G_hist['same_prefix_cols']}")


def preprocess_3(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 删除无效(nunique为1的)特征
    log('[+] preprocess_3')
    start = time.time()

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:
        G_hist['preprocess_3'] = {}

    log('[+] preprocess_3: rename columns')
    G_df_dict['BIG'] = rename_columns(G_df_dict['BIG'], G_data_info, G_hist, is_train)

    log('[+] preprocess_3: del invalid features')
    G_df_dict['BIG'] = del_invalid_features(G_df_dict['BIG'], G_data_info, G_hist, is_train, 'preprocess_3')

    # 定义数据列的类型
    log('[+] preprocess_3: get information of big')
    get_info_from_big(G_df_dict['BIG'], G_data_info, G_hist, is_train)

    log('[+] preprocess_3: process multi_value features')
    G_df_dict['BIG'] = process_multi_value(G_df_dict['BIG'], G_data_info, G_hist, is_train)

    log('[+] preprocess_3: label encoding')
    G_df_dict['BIG'] = label_encoder(G_df_dict['BIG'], G_data_info, G_hist, is_train)

    ## debug for "[LightGBM] [Fatal] Do not support special JSON characters in feature name."
    G_df_dict['BIG'] = G_df_dict['BIG'].rename(columns=lambda x: re.sub('[^A-Za-z0-9_-]+', '', x))

    # 是否为不平衡数据
    G_hist['unbalanced'] = False
    label_counts = list(G_df_dict['BIG'][G_data_info['target_label']].value_counts())
    label_counts.sort()
    if label_counts[0] / label_counts[1] < 0.005:
        G_hist['unbalanced'] = True


    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time