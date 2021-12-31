import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')
from autox.autox_server.util import del_invalid_features, rename_columns, log
from tqdm import tqdm


def label_encoder(df_table, G_data_info, G_hist, is_train):
    # 类别型特征label_encoder

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:
        G_hist['preprocess_3']['label_encoder'] = {}

        gen_kv_table = df_table.loc[df_table['istrain'] == True]

        for col in gen_kv_table.columns:
            if col in Id:
                continue
            if 'O' == gen_kv_table[col].dtype:
                lbl = LabelEncoder()
                temp = pd.DataFrame(gen_kv_table[col].astype(str))
                log('col: {}, nunique: {}'.format(col, temp[col].nunique()))
                lbl.fit(temp[col])
                G_hist['preprocess_3']['label_encoder'][col] = lbl

    for col in G_hist['preprocess_3']['label_encoder']:
        lbl = G_hist['preprocess_3']['label_encoder'][col]
        kv = dict(zip(lbl.classes_, lbl.transform(lbl.classes_)))
        df_table[col + "_label_encoder"] = df_table[col].astype(str).map(kv).fillna(-1).astype(int)

    return df_table


def process_multi_value(df, G_data_info, G_hist, is_train):
    data_type_ = G_data_info['entities'][G_data_info['target_entity']]['columns']
    multi_value_cols = [list(x.keys())[0] for x in data_type_ if list(x.values())[0] == 'Multi_value']
    multi_value_cols = [x for x in multi_value_cols if x in df.columns]
    G_hist['big_cols_multi_value'] = multi_value_cols

    if is_train:
        G_hist['preprocess_3']['multi_value_cols'] = {}

        for col in tqdm(multi_value_cols):
            # 默认sep为':'
            G_hist['preprocess_3']['multi_value_cols'][col] = ':'

            # 获得sep
            # for sep in [':']:
            #     temp = df[col].str.split(sep, expand=True)
            #     if temp.shape[1] != 1:
            #         G_hist['preprocess_3']['multi_value_cols'][col] = sep

    # for col in tqdm(multi_value_cols):
    #     sep = G_hist['preprocess_3']['multi_value_cols'][col]
    #     temp = df[col].str.split(sep, expand=True)
    #     temp.columns = ['multi_' + col + '_' + str(x) for x in temp.columns]
    #     used_cols = list(temp.columns)[:3]
    #     temp = temp[used_cols]
    #     df = pd.concat([df, temp], axis=1)

    return df

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

    log('[+] preprocess_3: process multi_value features')
    G_df_dict['BIG'] = process_multi_value(G_df_dict['BIG'], G_data_info, G_hist, is_train)

    log('[+] preprocess_3: label encoding')
    G_df_dict['BIG'] = label_encoder(G_df_dict['BIG'], G_data_info, G_hist, is_train)

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time