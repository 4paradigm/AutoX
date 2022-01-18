import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
from autox.autox_server.util import log
from sklearn.feature_extraction.text import CountVectorizer


def fe_txt(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, txt')

    if is_train:
        G_hist['FE_txt'] = {}
        max_features = 500
        for col in G_hist['big_cols_txt']:
            log('processing: {}'.format(col))
            vec = CountVectorizer(stop_words=None,
                                  max_features=max_features,
                                  binary=True,
                                  ngram_range=(1, 2))
            vec_res = vec.fit(G_df_dict['BIG'][col].astype(str))
            G_hist['FE_txt'][col] = vec_res

    log("txt features: {}".format(list(G_hist['FE_txt'].keys())))

    G_df_dict['FE_txt'] = pd.DataFrame()
    for col in G_hist['FE_txt'].keys():
        temp = pd.DataFrame(G_hist['FE_txt'][col].transform(G_df_dict['BIG'][col].astype(str)).todense())
        temp.columns = [col + '_txt_' + str(i) for i in range(temp.shape[1])]
        G_df_dict['FE_txt'] = pd.concat([G_df_dict['FE_txt'], temp], axis=1)

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time