import warnings
import time
import pandas as pd
warnings.filterwarnings('ignore')
from autox.autox_server.util import log


def get_time_feature(df, col, keep=False):
    """
    为df增加时间特征列,包括:年,月,日,小时,dayofweek,weekofyear
    :param df:
    :param col: 时间列的列名
    :param keep: 是否保留原始时间列
    :return:
    """
    df_copy = df.copy()
    prefix = col + "_"

    df_copy[col] = pd.to_datetime(df_copy[col])
    df_copy[prefix + 'year'] = df_copy[col].dt.year
    df_copy[prefix + 'month'] = df_copy[col].dt.month
    df_copy[prefix + 'day'] = df_copy[col].dt.day
    df_copy[prefix + 'hour'] = df_copy[col].dt.hour
    df_copy[prefix + 'weekofyear'] = df_copy[col].dt.weekofyear
    df_copy[prefix + 'dayofweek'] = df_copy[col].dt.dayofweek
    # df_copy[prefix + 'holiday'] = df_copy[[prefix + 'year', prefix + 'month', prefix + 'day']] \
    #                                                 .apply(lambda x: _getholiday(x), axis=1)
    if keep:
        return df_copy
    else:
        return df_copy.drop([col], axis=1)


def drop_nan_target(df, target):
    df.index = range(len(df))
    drop_rows_idx = df.loc[df[target].isnull()].index
    df = df.loc[~df.index.isin(drop_rows_idx)]
    df.index = range(len(df))
    return df, drop_rows_idx


def delete_row(G_df_dict, G_data_info, G_hist, is_train, target, main_table_name):
    # 删除主表Label异常的数据: 只对train的主表进行操作
    if is_train:
        drop_rows_idx = []
        if target in G_df_dict[main_table_name].columns:
            G_df_dict[main_table_name], drop_rows_idx = drop_nan_target(G_df_dict[main_table_name], target)
        log("drop_rows_idx:{}".format(list(drop_rows_idx)))
        G_hist['drop_rows_idx'] = drop_rows_idx
    else:
        if len(list(G_hist['drop_rows_idx'])) != 0:
            G_df_dict[main_table_name] = G_df_dict[main_table_name].loc[G_df_dict[main_table_name].index.isin(G_hist['drop_rows_idx'])]
            G_df_dict[main_table_name].index = range(len(G_df_dict[main_table_name]))

def parsing_time(G_df_dict, G_data_info, G_hist, is_train):
    # 获取G_hist
    if is_train:
        G_hist['preprocess']['parsing_time'] = {}
        for table in G_data_info['entities'].keys():
            time_cols = [list(x.keys())[0] for x in G_data_info['entities'][table]['columns'] if
                         "DateTime" == list(x.values())[0]]
            if len(time_cols) != 0:
                G_hist['preprocess']['parsing_time'][table] = time_cols

        log(G_hist['preprocess']['parsing_time'])

    # 根据G_hist进行操作
    log("time columns: {}".format(G_hist['preprocess']['parsing_time']))
    for table in G_hist['preprocess']['parsing_time'].keys():
        log("table: {}".format(table))
        for col in G_hist['preprocess']['parsing_time'][table]:
            log("col: {}".format(col))
            G_df_dict[table] = get_time_feature(G_df_dict[table], col, keep=True)


def preprocess(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    """数据预处理

    1.去除异常行: 只对train的主表进行操作
    2.分解时间: 对所有表进行操作
    3.若数据较大，对主表进行采样
    """
    start = time.time()

    if is_train:
        G_hist['preprocess'] = {}

    time_budget = G_data_info['time_budget']
    Id = G_data_info['target_id']
    target = G_data_info['target_label']
    main_table_name = G_data_info['target_entity']

    if is_train:
        G_hist['big_data'] = False
        G_hist['super_big_data'] = False

        super_big_limit = 1.5e8
        if G_df_dict[main_table_name].shape[0] > super_big_limit:
            G_hist['super_big_data'] = True

        # 行数较多的数据集采样
        SAMPLE_LIMIT_FOR_BIG = 1e7
        SAMPLE_SIZE = 1e6
        if G_df_dict[main_table_name].shape[0] > SAMPLE_LIMIT_FOR_BIG:
            log('[+] sample for big data')
            G_hist['big_data'] = True
            G_df_dict[main_table_name] = G_df_dict[main_table_name].sample(int(SAMPLE_SIZE))
            if 'action' in G_df_dict:
                relation = [x for x in G_data_info['relations'] if x['right_entity'] == 'action'][0]
                main_table_id = G_df_dict[main_table_name][relation['left_on'][0]]
                G_df_dict['action'] = G_df_dict['action'].loc[
                    G_df_dict['action'][relation['right_on'][0]].isin(main_table_id)]

        # 列数较多的数据集采样
        SAMPLE_LIMIT_FOR_BIG_COL = 4e6
        COL_LIMIT = 600
        SAMPLE_SIZE_FOR_BIG_COL = 1e6
        if G_df_dict[main_table_name].shape[0] > SAMPLE_LIMIT_FOR_BIG_COL and G_df_dict[main_table_name].shape[1] > COL_LIMIT:
            log('[+] sample for big data')
            G_hist['big_data'] = True
            G_df_dict[main_table_name] = G_df_dict[main_table_name].sample(int(SAMPLE_SIZE_FOR_BIG_COL))
            if 'action' in G_df_dict:
                relation = [x for x in G_data_info['relations'] if x['right_entity'] == 'action'][0]
                main_table_id = G_df_dict[main_table_name][relation['left_on'][0]]
                G_df_dict['action'] = G_df_dict['action'].loc[
                    G_df_dict['action'][relation['right_on'][0]].isin(main_table_id)]

    log('[+] preprocess, delete_row')
    # 只对主表进行操作
    delete_row(G_df_dict, G_data_info, G_hist, is_train, target, main_table_name)

    log('[+] preprocess, parsing_time')
    parsing_time(G_df_dict, G_data_info, G_hist, is_train)


    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time

