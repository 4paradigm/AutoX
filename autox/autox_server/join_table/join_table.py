import warnings
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import rename_columns, log

def join_simple_tables(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    """
    获得G_df_dict['BIG']
    """
    start = time.time()

    if is_train:
        if 'relations' in G_data_info:
            G_hist['join_simple_tables'] = [x for x in G_data_info['relations'] if
                                            x['type'] == '1-1' and x['related_to_main_table'] == 'true']
        else:
            G_hist['join_simple_tables'] = []

    time_budget = G_data_info['time_budget']
    Id = G_data_info['target_id']
    target = G_data_info['target_label']
    main_table_name = G_data_info['target_entity']

    log('[+] join simple tables')
    G_df_dict['BIG'] = G_df_dict[main_table_name]

    # 如果为时序数据，对BIG表排序
    if G_data_info['target_time'] != '':
        G_df_dict['BIG'].sort_values(by=G_data_info['target_time'])

    for relation in G_hist['join_simple_tables']:
        left_table_name = relation['left_entity']
        right_table_name = relation['right_entity']
        left_on = relation['left_on']
        right_on = relation['right_on']
        if main_table_name == left_table_name:
            merge_table_name = right_table_name
            skip_name = right_on
        else:
            merge_table_name = left_table_name
            left_on, right_on = right_on, left_on
            skip_name = left_on
        log(merge_table_name)
        merge_table = G_df_dict[merge_table_name].copy()
        merge_table.columns = [x if x in skip_name else merge_table_name + "_" + x for x in merge_table.columns]

        G_df_dict['BIG'] = G_df_dict['BIG'].merge(merge_table, left_on=left_on, right_on=right_on, how='left')
        log(f"G_df_dict['BIG'].shape: {G_df_dict['BIG'].shape}")
    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time


def join_1_to_M_tables(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    log('[+] join 1_to_M tables')
    start = time.time()

    if is_train:
        if 'relations' in G_data_info:
            G_hist['join_1_to_M_tables'] = [x for x in G_data_info['relations'] if
                                            x['type'] == '1-M' and x['related_to_main_table'] == 'true']
        else:
            G_hist['join_1_to_M_tables'] = []

    time_budget = G_data_info['time_budget']
    Id = G_data_info['target_id']
    target = G_data_info['target_label']
    main_table_name = G_data_info['target_entity']

    for cur_relation in G_hist['join_1_to_M_tables']:

        merge_table_name = cur_relation['right_entity']
        log("process table : {}".format(merge_table_name))

        merge_table_key = cur_relation['right_on']
        merge_table_info = G_data_info['entities'][merge_table_name]
        merge_table = G_df_dict[merge_table_name]

        merge_table.columns = [x if x in merge_table_key else merge_table_name + "_" + x for x in merge_table.columns]

        # 对于非静态表,(且左右表均有时间列的情况下),将主表的时间列拼到副表,只保留副表中时间列小于主表时间列的行
        if merge_table_info['is_static'] == 'false':
            if 'left_time_col' in cur_relation and 'right_time_col' in cur_relation:
                temp_used_cols = cur_relation['left_on'].copy()
                temp_used_cols.append(cur_relation['left_time_col'])

                merge_table = merge_table.merge(G_df_dict[cur_relation['left_entity']][temp_used_cols],
                                                left_on=cur_relation['left_on'],
                                                right_on=cur_relation['right_on'], how='left')

                left_time_col = cur_relation['left_time_col']
                if merge_table_key == cur_relation['right_time_col']:
                    right_time_col = cur_relation['right_time_col']
                else:
                    right_time_col = merge_table_name + "_" + cur_relation['right_time_col']
                merge_table = merge_table.loc[merge_table[right_time_col] <= merge_table[left_time_col]]
                merge_table.drop(left_time_col, axis=1, inplace=True)

        cat_cols = [x for x in merge_table_info['columns'] if list(x.values())[0] == 'Str']
        num_cols = [x for x in merge_table_info['columns'] if list(x.values())[0] == 'Num']

        # 将C表(A->B-C)聚合的特征也考虑进来
        entities_cols = [list(x.keys())[0] for x in merge_table_info['columns']]
        entities_cols = [merge_table_name + "_" + x for x in entities_cols]
        for col in [x for x in merge_table.columns if x not in entities_cols]:
            num_cols.append({col: 'Num'})

        temp = merge_table.drop_duplicates(merge_table_key)
        temp = temp[merge_table_key]
        temp.index = range(len(temp))

        # 对于副表为非静态表的情况,计算如下特征:the stat of time between a id‘s behavior and that id‘s previous behavior.
        if merge_table_info['is_static'] == 'false':
            merge_table = merge_table.sort_values(by=[merge_table_name + "_" + x for x in merge_table_info['time_col']])
            merge_table[merge_table_name + '_delta_time'] = merge_table[[merge_table_name + "_" + x for x in
                                                                         merge_table_info['time_col']]] - \
                                                            merge_table.groupby(merge_table_key)[
                                                                [merge_table_name + "_" + x for x in
                                                                 merge_table_info['time_col']]].shift(1)
            cur_temp = merge_table.groupby(merge_table_key).agg(
                {merge_table_name + "_delta_time": ['max', 'min', 'median', 'mean', 'std']})
            temp = temp.merge(cur_temp, on=merge_table_key, how='left')

        for cur_num in num_cols:
            cur_num_col = list(cur_num.keys())[0]
            if cur_num_col in merge_table_key:
                continue
            if merge_table_name + "_" + cur_num_col in entities_cols:
                cur_temp = merge_table.groupby(merge_table_key).agg(
                    {merge_table_name + "_" + cur_num_col: ['max', 'min', 'median', 'mean', 'std']})
            else:
                cur_temp = merge_table.groupby(merge_table_key).agg(
                    {cur_num_col: ['max', 'min', 'median', 'mean', 'std']})
            temp = temp.merge(cur_temp, on=merge_table_key, how='left')

        for cur_cat in cat_cols:
            cur_cat_col = list(cur_cat.keys())[0]
            if cur_cat_col in merge_table_key:
                continue
            cur_temp = merge_table.groupby(merge_table_key).agg({merge_table_name + "_" + cur_cat_col: ['nunique']})
            temp = temp.merge(cur_temp, on=merge_table_key, how='left')

        G_df_dict['BIG'] = G_df_dict['BIG'].merge(temp, on=merge_table_key, how='left')

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))

    return remain_time


def join_indirect_1_to_M_tables(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    log('[+] join indirect 1_to_M tables')
    start = time.time()

    if is_train:
        if 'relations' in G_data_info:
            G_hist['join_indirect_1_to_M_tables'] = [x for x in G_data_info['relations'] if
                                                     x['type'] == '1-M' and x['related_to_main_table'] == 'false']
        else:
            G_hist['join_indirect_1_to_M_tables'] = []

    time_budget = G_data_info['time_budget']
    Id = G_data_info['target_id']
    target = G_data_info['target_label']
    main_table_name = G_data_info['target_entity']

    for cur_relation in G_hist['join_indirect_1_to_M_tables']:

        left_table_name = cur_relation['left_entity']
        merge_table_name = cur_relation['right_entity']
        log("process table : {}".format(merge_table_name))

        left_on = cur_relation['left_on']
        merge_table_key = cur_relation['right_on']
        merge_table_info = G_data_info['entities'][merge_table_name]
        merge_table = G_df_dict[merge_table_name]

        merge_table.columns = [x if x in merge_table_key else merge_table_name + "_" + x for x in merge_table.columns]

        cat_cols = [x for x in merge_table_info['columns'] if list(x.values())[0] == 'Str']
        num_cols = [x for x in merge_table_info['columns'] if list(x.values())[0] == 'Num']

        temp = merge_table.drop_duplicates(merge_table_key)
        temp = temp[merge_table_key]
        temp.index = range(len(temp))

        # 对于副表为非静态表的情况,计算如下特征:the stat of time between a id‘s behavior and that id‘s previous behavior.
        if merge_table_info['is_static'] == 'false':
            merge_table = merge_table.sort_values(by=[merge_table_name + "_" + x for x in merge_table_info['time_col']])
            merge_table[merge_table_name + '_delta_time'] = merge_table[[merge_table_name + "_" + x for x in
                                                                         merge_table_info['time_col']]] - \
                                                            merge_table.groupby(merge_table_key)[
                                                                [merge_table_name + "_" + x for x in
                                                                 merge_table_info['time_col']]].shift(1)
            cur_temp = merge_table.groupby(merge_table_key).agg(
                {merge_table_name + "_delta_time": ['max', 'min', 'median', 'mean', 'std']})
            temp = temp.merge(cur_temp, on=merge_table_key, how='left')

        for cur_num in num_cols:
            cur_num_col = list(cur_num.keys())[0]
            if cur_num_col in merge_table_key:
                continue

            cur_temp = merge_table.groupby(merge_table_key).agg(
                {merge_table_name + "_" + cur_num_col: ['max', 'min', 'median', 'mean', 'std']})
            temp = temp.merge(cur_temp, on=merge_table_key, how='left')

        for cur_cat in cat_cols:
            cur_cat_col = list(cur_cat.keys())[0]
            if cur_cat_col in merge_table_key:
                continue
            cur_temp = merge_table.groupby(merge_table_key).agg({merge_table_name + "_" + cur_cat_col: ['nunique']})
            temp = temp.merge(cur_temp, on=merge_table_key, how='left')

        G_df_dict[left_table_name] = G_df_dict[left_table_name].merge(temp, left_on=left_on, right_on=merge_table_key,
                                                                      how='left')

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))

    return remain_time


def preprocess_after_join_indirect_tables(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    log('[+] preprocess_after_join_indirect_tables')
    start = time.time()

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:
        G_hist['preprocess_after_join_indirect_tables'] = {}

    log('[+] preprocess_after_join_indirect_tables: rename columns')
    for table_name in G_df_dict.keys():
        G_df_dict[table_name] = rename_columns(G_df_dict[table_name], G_data_info, G_hist, is_train)

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))

    return remain_time