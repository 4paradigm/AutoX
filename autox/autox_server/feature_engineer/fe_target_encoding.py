import warnings
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log

def fe_target_encoding(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, target encoding')

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    if is_train:
        G_hist['FE_target_encoding'] = {}
        G_hist['FE_target_encoding']['mean'] = {}
        G_hist['FE_target_encoding']['var'] = {}

        size_of_big = G_df_dict['BIG'].shape[0]

        target_encoding_features = []
        for col in G_df_dict['BIG'].columns:
            if col in [target] + Id:
                continue
            if '_label_encoder' in col:
                continue
            if 'int' in str(G_df_dict['BIG'][col].dtype):
                if G_df_dict['BIG'][col].nunique() < size_of_big * 0.05:
                    target_encoding_features.append(col)

        for f in target_encoding_features:
            temp = G_df_dict['BIG'].groupby([f])[target].mean().reset_index()
            temp.index = temp[f]
            temp = temp.drop(f, axis=1)
            faeture_map = temp.to_dict()[target]
            G_hist['FE_target_encoding']['mean'][f] = faeture_map

            temp = G_df_dict['BIG'].groupby([f])[target].var().reset_index()
            temp.index = temp[f]
            temp = temp.drop(f, axis=1)
            faeture_map = temp.to_dict()[target]
            G_hist['FE_target_encoding']['var'][f] = faeture_map

    log("target encoding feature: {}".format(list(G_hist['FE_target_encoding']['mean'].keys())))

    G_df_dict['FE_target_encoding'] = G_df_dict['BIG'][Id]
    for f in G_hist['FE_target_encoding']['mean'].keys():
        G_df_dict['FE_target_encoding'][f + "_target_encoding"] = G_df_dict['BIG'][f].map(
            G_hist['FE_target_encoding']['mean'][f])
        # G_df_dict['FE_target_encoding'][f + "_target_encoding_var"] = G_df_dict['BIG'][f].map(
        #     G_hist['FE_target_encoding']['var'][f])

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time