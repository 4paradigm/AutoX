import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
import datetime
import json
import pickle


# log
import logging
LOGGER = logging.getLogger('run-time-adaptive_automl')
# LOG_LEVEL = 'INFO'
LOG_LEVEL = 'DEBUG'
LOGGER.setLevel(getattr(logging, LOG_LEVEL))
simple_formatter = logging.Formatter('%(levelname)7s -> %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(simple_formatter)
LOGGER.addHandler(console_handler)
LOGGER.propagate = False
nesting_level = 0
ensemble_top_k = 1


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def unique_cols(items):
    res = []
    unique_keys = []
    for item in items:
        key_ = list(item.keys())[0]
        if key_ not in unique_keys:
            unique_keys.append(key_)
            res.append(item)
    return res

def clean_meta(meta):
    for tabel in meta['data_info']['entities'].keys():
        table_cols = meta['data_info']['entities'][tabel]['columns']
        table_cols = unique_cols(table_cols)
        meta['data_info']['entities'][tabel]['columns'] = table_cols
    return meta

def load_json(path):
    with open(path) as f:
        config = json.load(f)
    return config

def get_data_info(conf_file_path):

    meta_config = load_json(conf_file_path)
    data_contents = {}
    for item in meta_config['data']:
        data_contents[item['name']] = item['uri']
    meta_config['data'] = data_contents

    # 转化config格式: from fz to taac, 并保留原始数据表的schema
    config_dict = meta_config['config_dict']
    hc = config_dict
    fe_config = {}
    fe_config['tableInfo'] = {}
    for table_ in hc['entity_detail'].keys():
        cur_res = []
        for item in hc['entity_detail'][table_]['features']:
            name_ = item["id"].split(".")[-1]
            type_ = item["feature_type"]
            cur_item = {"name": name_, "type": type_}
            cur_res.append(cur_item)
        fe_config['tableInfo'][table_] = cur_res

    dtype_map = {
        "SingleString": "Str",
        "ContinueNum": "Num",
        "DiscreteNum": "Num",
        "ArrayString(,)": "Multi_value",
        "Timestamp": "DateTime",
        "DiscreteLabel": "Num",
        "ContinueLabel": "Num",
        "SplitID": "SplitID",
        "KVString(,)[:]": "KVString(,)[:]"
    }
    # hc中时间不区分Unix_timestamp和DateTime

    data_info = {}
    #     data_info['dataset_id'] = data_name
    data_info['recom_metrics'] = ['auc']
    data_info['target_entity'] = hc['target_entity']
    data_info['target_id'] = [hc['target_entity_index']]
    data_info['target_label'] = hc['target_label'].split(".")[0] + "_" + hc['target_label'].split(".")[1]
    data_info['target_time'] = hc['target_pivot_timestamp']
    data_info['time_budget'] = 18000
    data_info['time_series_data'] = "true" if len(hc['target_pivot_timestamp']) > 0 else "false"
    data_info['entities'] = {}

    for idx, key_ in enumerate(hc['entity_detail']):
        #     print(key_)
        #     print(hc['entity_detail'][key_])

        data_info['entities'][key_] = {}
        data_info['entities'][key_]['file_name'] = key_
        data_info['entities'][key_]['format'] = 'parquet'
        data_info['entities'][key_]['header'] = 'true'

        data_info['entities'][key_]['is_static'] = \
            "false" if (len(hc['target_pivot_timestamp']) > 0 and key_ == hc['target_entity']) else "true"
        data_info['entities'][key_]['time_col'] = \
            [] if data_info['entities'][key_]['is_static'] == "true" else [hc['target_pivot_timestamp']]

        data_info['entities'][key_]['columns'] = []
        data_info['entities'][key_]['skip_columns'] = []
        for item in hc['entity_detail'][key_]['features']:
            col_ = item['id'].split(".")[-1]
            type_ = dtype_map[item['data_type']]
            if type_ != "SplitID":
                cur = {col_: type_}
                data_info['entities'][key_]['columns'].append(cur)
            if item['skip']:
                data_info['entities'][key_]['skip_columns'].append(col_)

    data_info['relations'] = []
    for item in hc['relations']:
        cur_relation = {}
        cur_relation['related_to_main_table'] = "true"
        if item['from_entity'] != hc['target_entity'] and item['to_entity'] != hc['target_entity']:
            cur_relation['related_to_main_table'] = "false"
        cur_relation['left_entity'] = item['from_entity']
        cur_relation['left_on'] = item['from_entity_keys']
        cur_relation['right_entity'] = item['to_entity']
        cur_relation['right_on'] = item['to_entity_keys']
        cur_relation['type'] = item['type']

        if cur_relation['type'] == "SLICE":
            cur_relation['type'] = "1-M"

        cur_relation['time_windows'] = item['time_windows']
        cur_relation['window_delay'] = item['window_delay']

        if item['from_entity_time_col'] != '':
            cur_relation['left_time_col'] = item['from_entity_time_col']
            cur_relation['right_time_col'] = item['to_entity_time_col']

        data_info['relations'].append(cur_relation)

    meta_config['config_dict'] = {}
    meta_config['data_info'] = data_info
    meta_config['fe_config'] = fe_config

    return meta_config

pico_fe_sel_dict = {'app_name': 'gbm',
 'env': {'resources': {'learner': {'cpu': 1, 'mem': 100000, 'num': 4},
   'pserver': {'cpu': 1, 'mem': 100000, 'num': 1}}},
 'framework': {'accumulator': {'max_num_of_snapshots_saved': 30,
   'report_accumulator_json_path': 'hdfs://m7-model-hdp01/user/caihengxing/zhouhao/memory/used/running/applications/output_reg_v1/case__name/feature_selection/accumulator.json',
   'report_accumulator_pretty': True,
   'report_interval_in_sec': 30},
  'channel': {'bounded_capacity': 1024},
  'communication': {'heartbeat_timeout': 120,
   'io_thread_num': 4,
   'max_socket_num': 65535,
   'request_rcvhwm': 4,
   'resend_message_queue_size_limit': 2048,
   'resend_timeout': 32},
  'dense_table': '',
  'execution_info': {'finished_state': 'finished',
   'max_execution_to_show': 20,
   'max_plan_to_show': 20,
   'running_state': 'running'},
  'global_concurrency': 10,
  'hadoop_bin': 'hdfs dfs',
  'lemon': {'dump_file_number': 4,
   'local_shard_num': 4,
   'message_compress_algorithm': '',
   'server_c2s_thread_num': 4,
   'server_load_block_size': 1000},
  'performance': {'is_evaluate_performance': False},
  'process': {'block_size': 10, 'cpu_concurrency': 40, 'io_concurrency': 4},
  'sparse_ps': '',
  'sparse_table': ''},
 'gbm': {'boost_regex': '500dt',
  'cache_shard_num': 1,
  'cache_uri': 'file://.?compress=lz4&format=archive',
  'input_path': ['hdfs://m7-model-hdp01/user/caihengxing/zhouhao/memory/used/running/applications/output_reg_v1/case__name/fe_sel_data?format=gc&block_size=4000'],
  'input_validation_path': ['hdfs://m7-model-hdp01/user/caihengxing/zhouhao/memory/used/running/applications/output_reg_v1/case__name/fe_sel_data?format=gc&block_size=4000'],
  'link_function_for_label': 'identity',
  'loss_type': 'logloss',
  'mini_batch_size': 100,
  'model_output_path': 'hdfs://m7-model-hdp01/user/caihengxing/zhouhao/memory/used/running/applications/output_reg_v1/case__name/feature_selection',
  'param': [{'col_sample_ratio': 0.8662117781339908,
    'l0_coef': 0.9013993833581042,
    'l2_coef': 0.5445281769282307,
    'label_cnt': 2,
    'learning_rate': 0.00876650871797269,
    'max_depth': 6,
    'max_leaf_n': 100000,
    'min_child_n': 0,
    'min_child_ratio': 0,
    'min_child_weight': 1.1591388511561993,
    'min_split_gain': 0.42573399521268684,
    'name': 'dt',
    'propose': {'discrete_max_n': 100,
     'eps': 1e-06,
     'max_n': 100,
     'type': 'quantile',
     'use_local_propose': False},
    'row_sample_ratio': 0.8174205227596796,
    'type': 'dtree'}],
  'random_seed': 0,
  'sink_model_interval': 0,
  'stop_criteria_max_selected_featue_num': 50,
  'validate_model_interval': 1}}

# Unix_timestamp列识别
def get_Unix_timestamp_cols(df):
    Unix_timestamp_cols = []
    for col in df.columns:
        if str(df[col].dtype) in ['int64', 'float64', 'object']:
                try:
                    ts_min = int(float(df.loc[~(df[col] == '') & (df[col].notnull()), col].min()))
                    ts_max = int(float(df.loc[~(df[col] == '') & (df[col].notnull()), col].max()))
                    datetime_min = datetime.datetime.utcfromtimestamp(ts_min).strftime('%Y-%m-%d %H:%M:%S')
                    datetime_max = datetime.datetime.utcfromtimestamp(ts_max).strftime('%Y-%m-%d %H:%M:%S')
                    if datetime_min > '2000-00-00 00:00:01' and datetime_max < '2025-00-00 00:00:01' and datetime_max > datetime_min:
                        log("Unix_timestamp_col: {}, datetime_min: {}, datetime_max: {}".format(col, datetime_min, datetime_max))
                        Unix_timestamp_cols.append(col)
                except:
                    pass
    return Unix_timestamp_cols

# 修改G_data_info中的Unix_timestamp列
def modify_Unix_timestamp_col_in_data_info(G_data_info, table, Unix_timestamp_cols):
    for item in G_data_info['entities'][table]['columns']:
        keys = list(item.keys())[0]
        values = list(item.values())[0]
        if keys in Unix_timestamp_cols:
            item[keys] = 'Unix_timestamp'
            log("col: {}, set from {} to Unix_timestamp".format(keys, values))
    return G_data_info

def remove_space_from_df(df):
    cols_name = []
    for col in df.columns:
        if ' ' in col:
            cols_name.append(col.replace(' ', ''))
        else:
            cols_name.append(col)
    df.columns = cols_name
    return df

def log(entry, level='info'):
    if level not in ['debug', 'info', 'warning', 'error']:
        LOGGER.error('Wrong level input')

    global nesting_level
    space = '-' * (4 * nesting_level)

    getattr(LOGGER, level)(f"{space} {entry}")

import os
def execute_shell(command_line):
    log(command_line)
    os.system(command_line)

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


def taac2ampere(data_path, config):
    new_config = {}
    unit_map = {
        "Day": "d",
        "Year" :"y",
        "Mounth" : "m",
        "Hour" : "h",
        "Second" : "s"
    }
    dtype_map = {
        "Cat" : "Integer",
        "Str" : "Category",
        "Num" : "Numeric",
        "Timestamp": "Datetime",
        "DateTime" : "Datetime",
        "Unix_timestamp" : "Timestamp",
        'KVString(,)[:]': "FKV",
        'Multi_value': "MultiValue",
    }

    new_config["fact_entity"] = config["target_entity"]
    new_config["fact_label"] = config["target_label"]
    new_config["fact_pivot_timestamp"]=config.get("target_time_col", None) or config.get("target_time", None)
    new_config["entity"] = {}
    new_config["relation"] = []
    unit = unit_map.get(config.get("unit", None), None)

    def build_relation(relations):
        if len(relations) == 7:
            return {
                "from_entity": relations[0],
                "from_entity_keys" : relations[1],
                "from_time_col": relations[2],
                "to_entity": relations[3],
                "to_entity_keys": relations[4],
                "to_time_col": relations[5],
                "type": relations[6]
            }
        else:
            return  {
                "from_entity": relations[0],
                "from_entity_keys" : relations[1],
                "to_entity": relations[2],
                "to_entity_keys": relations[3],
                "type": relations[4]
            }

    def parse_schema(columns, unit=None):
        schema = []
        for item in columns:
            for feature_name in item:
                feature_type = dtype_map.get(item[feature_name], "Category")
                if item[feature_name] == "Cat":
                    new_feature = {
                        "name":feature_name,
                        "type":feature_type,
                        "tags": {"category":True}
                    }
                elif feature_type == "FKV":
                    new_feature = {
                    "name": feature_name,
                    "type": feature_type,
                    "sub_type": "Numeric",
                    "sep" : ",",
                    "kvsep": ":"
                    }
                elif feature_type == "MultiValue":
                    new_feature = {
                    "name": feature_name,
                    "type": feature_type,
                    "sep" : ",",
                    }
                else:
                    new_feature = {
                        "name":feature_name,
                        "type":feature_type,
                    }
                if feature_name in config["target_id"]:

                    if "tags" in new_feature:
                        new_feature["tags"]["id"] = True
                    else:
                        new_feature["tags"] = {"id" :True}
                    if item[feature_name] == "Num":
                        new_feature["type"] = "Integer"
                        new_feature["tags"]["category"] = True
                schema.append(new_feature)
        return schema
    for key in config["entities"]:
        entity = config["entities"][key]
        new_entity = {}
        new_entity["data_header"] = 0 if entity['header'] == "true" else None
        new_entity["data_format"] =  entity['format']
        new_entity["data_path"] = data_path[entity["file_name"]]
        new_entity["is_static"] = False if entity["is_static"] == 'false' else True
        if "origin_columns" in entity.keys():
          new_entity["schema"]= parse_schema(entity["origin_columns"], unit)
        else:
          new_entity["schema"]= parse_schema(entity["columns"], unit)
        if key == new_config["fact_entity"] and new_config["fact_pivot_timestamp"] :
            new_entity["pivot_timestamp"] = new_config["fact_pivot_timestamp"]
        new_config["entity"][key] = (new_entity)

    if "relations" in config:
        for relation in config["relations"]:
            if relation["type"] == "1-1" or relation["type"] == "1-M" or relation["type"] == "M-M":
                if relation.get("left_time_col", None) and relation.get("left_time_col", None) != "" and relation.get("right_time_col", None) and relation.get("right_time_col", None) != "":
                    new_config["relation"].append(build_relation([relation["left_entity"], relation["left_on"], relation["left_time_col"], relation["right_entity"], relation["right_on"], relation["right_time_col"],relation["type"]]))
                else:
                    new_config["relation"].append(build_relation([relation["left_entity"], relation["left_on"], relation["right_entity"], relation["right_on"],relation["type"]]))

    if new_config['fact_pivot_timestamp'] == '':
        del new_config['fact_pivot_timestamp']

    return new_config


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
    log("columns of BIG: {}".format(list(df.columns)), level='debug')
    for table_name in G_df_dict.keys():
        if table_name.startswith("FE_"):

            # del invalid features
            cur_table = del_invalid_features(G_df_dict[table_name], G_data_info, G_hist, is_train, table_name)

            # 删除fe表中的Id列
            for item in Id:
                if item in cur_table.columns:
                    cur_table.drop(item, axis=1, inplace=True)

            log("shape of {}: {}".format(table_name, cur_table.shape))
            log("columns of {}: {}".format(table_name, list(cur_table.columns)), level='debug')
            if cur_table.shape[1] != 0:
                df = pd.concat([df, cur_table], axis=1)
    log("shape after fe: {}".format(df.shape))

    ## debug for "[LightGBM] [Fatal] Do not support special JSON characters in feature name."
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_-]+', '', x))
    assert(len(list(df.columns)) == len(set(list(df.columns))))

    G_df_dict['BIG_FE'] = df

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time

buddha_bless =\
'''
                   _ooOoo_
                  o8888888o
                  88" . "88
                  (| -_- |)
                  O\  =  /O
               ____/`---'\____
             .'  \\\\|     |//  `.
            /  \\\\|||  :  |||//  \\
           /  _||||| -:- |||||-  \\
           |   | \\\\\\  -  /// |   |
           | \_|  ''\---/''  |   |
           \  .-\__  `-`  ___/-. /
         ___`. .'  /--.--\  `. . __
      ."" '<  `.___\_<|>_/___.'  >'"".
     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
     \  \ `-.   \_ __\ /__ _/   .-` /  /
======`-.____`-.___\_____/___.-`____.-'======
                   `=---='
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     佛祖保佑       永不宕机     永无BUG 
'''
