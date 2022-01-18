import time
import warnings
from autox.autox_server.model import simple_lgb
from autox.autox_server.util import log
warnings.filterwarnings('ignore')

def ensemble(G_df_dict, G_data_info, G_hist, is_train, remain_time, top_k=3):
    start = time.time()

    # 训练过程确定要使用的结果,预测过程将选中的结果进行ensemble
    if is_train:

        used_model_names = sorted(G_hist['val_auc'], key=G_hist['val_auc'].get, reverse=True)[:top_k]
        G_hist['used_model_names'] = used_model_names
        log('used_model_names: {}'.format(used_model_names))
        end = time.time()
        remain_time -= (end - start)

    else:

        if len(G_hist['used_model_names']) != 0:
            target = G_data_info['target_label']

            ens_pred = G_hist['predict'][G_hist['used_model_names'][0]]
            for cur_name in G_hist['used_model_names'][1:]:
                cur_pred = G_hist['predict'][cur_name]
                ens_pred[target] += cur_pred[target]

            ens_pred[target] /= len(G_hist['used_model_names'])

            G_hist['predict']['ensemble'] = ens_pred

        else:
            log("[+] use simple lgb in ensemble")
            remain_time = simple_lgb.simple_lgb(G_df_dict, G_data_info, G_hist, is_train, remain_time)
            G_hist['predict']['ensemble'] = G_hist['predict']['simple_lgb']

        end = time.time()
        remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))

    return remain_time