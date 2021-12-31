import time
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# log
import logging
LOGGER = logging.getLogger('run-time-adaptive_automl')
LOG_LEVEL = 'INFO'
LOGGER.setLevel(getattr(logging, LOG_LEVEL))
simple_formatter = logging.Formatter('%(levelname)7s -> %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(simple_formatter)
LOGGER.addHandler(console_handler)
LOGGER.propagate = False
nesting_level = 0

def log(entry, level='info'):
    if level not in ['debug', 'info', 'warning', 'error']:
        LOGGER.error('Wrong level input')

    global nesting_level
    space = '-' * (4 * nesting_level)

    getattr(LOGGER, level)(f"{space} {entry}")

def cols_concat(df, con_list):
    name = "__".join(con_list)
    df[name] = df[con_list[0]].astype(str)
    for item in con_list[1:]:
        df[name] = df[name] + '__' + df[item].astype(str)
    return df

# 获得unique为1的特征
def get_invalid_features(df):
    del_cols = []
    for col in df.columns:
        if df[col].nunique() in [0, 1]:
            del_cols.append(col)

    return del_cols

def del_invalid_features(df_table, G_data_info, G_hist, is_train, process_name):
    # 删除无效(nunique为1的)特征

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:
        G_hist['delete_column'][process_name] = {}
        del_cols = get_invalid_features(df_table)
        del_cols = [x for x in del_cols if x not in Id + [target]]
        del_cols = [x for x in del_cols if x != 'istrain']
        if len(del_cols) != 0:
            G_hist['delete_column'][process_name] = del_cols
        log('delete column: {}'.format(G_hist['delete_column'][process_name]))

    del_cols = G_hist['delete_column'][process_name]
    df_table.drop(del_cols, axis=1, inplace=True)

    return df_table


def rename_columns(df_table, G_data_info, G_hist, is_train):
    features_name = []
    for col_name in df_table.columns:
        if type(col_name) == tuple:
            col_name = '__'.join(list(col_name))
        features_name.append(col_name)

    df_table.columns = features_name
    return df_table


def merge_table(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 删除无效(nunique为1的)特征
    log('[+] merge_table')
    start = time.time()

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:
        G_hist['merge_table'] = {}

    df = G_df_dict['BIG']
    log("shape of BIG: {}".format(df.shape))
    for table_name in G_df_dict.keys():
        if table_name.startswith("FE_"):

            # del invalid features
            cur_table = del_invalid_features(G_df_dict[table_name], G_data_info, G_hist, is_train, table_name)

            # 删除fe表中的Id列
            for item in Id:
                if item in cur_table.columns:
                    cur_table.drop(item, axis=1, inplace=True)

            log("shape of {}: {}".format(table_name, cur_table.shape))
            df = pd.concat([df, cur_table], axis=1)
    log("shape after fe: {}".format(df.shape))
    G_df_dict['BIG_FE'] = df

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time