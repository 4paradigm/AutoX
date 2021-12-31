import json
import time
import warnings
import pandas as pd
from autox.autox_server.util import log
warnings.filterwarnings('ignore')

def load_json(path):
    with open(path) as f:
        config = json.load(f)
    return config

# test阶段拼上train的数据
# 预测的时候需要传入remain_time
def read_data(train_set_path=None, test_set_path=None, data_info_path=None, data_info=None, df_dict=None, is_train=True, debug=True, remain_time=None):
    start = time.time()
    if is_train:
        assert(train_set_path is not None)
        assert(data_info_path is not None)
        assert(test_set_path is None)
        assert(data_info is None)
        assert(df_dict is None)
    else:
        assert (train_set_path is None)
        assert (data_info_path is None)
        assert (test_set_path is not None)
        assert (data_info is not None)
        assert (df_dict is not None)

    path_input = train_set_path if (test_set_path is None) else test_set_path

    if is_train:
        log('read data_info.json')
        data_info = load_json(data_info_path)
        log("data_info: {}".format(data_info))


    if is_train:
        remain_time = data_info['time_budget']

    if debug:
        nrows = 1000
    else:
        nrows = None
    main_table_name = data_info['target_entity']

    if is_train:
        df_dict = {}

    for table in data_info['entities'].keys():
        if is_train:
            f_name = data_info['entities'][table]['file_name'].replace(".csv", "_train.csv")
            log('[+] read {}'.format(f_name))
            df = pd.read_csv(path_input + '/' + f_name, nrows=nrows)

            if table == main_table_name:
                df['istrain'] = True

        if not is_train:
            f_name = data_info['entities'][table]['file_name'].replace(".csv", "_test.csv")
            df_test = pd.read_csv(path_input + '/' + f_name, nrows=nrows)

            if table == main_table_name:
                df_test['istrain'] = False

            df = df_dict[table]
            df = df.append(df_test)
            df.index = range(len(df))

        log('key = {}, shape = {}'.format(table, df.shape))
        df_dict[table] = df

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))

    return df_dict, data_info, remain_time