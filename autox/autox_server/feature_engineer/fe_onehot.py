import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
from autox.autox_server.util import log

onehot_N = 64
def fe_onehot(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, onehot')

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:
        G_hist['FE_onehot'] = {}

        size_of_big = G_df_dict['BIG'].shape[0]

        onehot_features = []
        for col in G_hist['big_cols_cat']:
            if col in [target] + Id:
                continue
            if 'int' in str(G_df_dict['BIG'][col].dtype):
                if G_df_dict['BIG'][col].nunique() < size_of_big * 0.005 and G_df_dict['BIG'][col].nunique() <= onehot_N:
                    onehot_features.append(col)
        G_hist['FE_onehot'] = onehot_features

    log("onehot features: {}".format(G_hist['FE_onehot']))

    G_df_dict['FE_onehot'] = G_df_dict['BIG'][Id]

    for f in G_hist['FE_onehot']:
        df_temp = pd.get_dummies(G_df_dict['BIG'][f], prefix=f)
        G_df_dict['FE_onehot'] = pd.concat([G_df_dict['FE_onehot'], df_temp], axis=1)
        G_df_dict['FE_onehot'].columns = ["onehot_" + str(x) for x in list(G_df_dict['FE_onehot'].columns)]

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time