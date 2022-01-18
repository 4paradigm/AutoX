import warnings
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import del_invalid_features
from autox.autox_server.util import log

def preprocess_2(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']进行操作
    # 删除无效(nunique为1的)特征
    # 类别型特征label_encoder

    log('[+] preprocess_2')
    start = time.time()
    if is_train:
        G_hist['preprocess_2'] = {}

    log('[+] preprocess_2: del invalid features')
    G_df_dict['BIG'] = del_invalid_features(G_df_dict['BIG'], G_data_info, G_hist, is_train, 'preprocess_2')

    #     log('[+] preprocess_2: label encoding')
    #     G_df_dict['BIG'] = label_encoder(G_df_dict['BIG'], G_data_info, G_hist, is_train)

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time