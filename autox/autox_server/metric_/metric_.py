import pandas as pd
from sklearn import metrics
from autox.autox_server.util import log

def lb(G_data_info, G_hist, path_input):

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    solution = pd.read_csv(path_input + 'train_test/solution.csv')
    y = solution[target]

    pred = G_hist['predict']['ensemble'][target]
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc_ = metrics.auc(fpr, tpr)

    log("auc: {}".format(auc_))
    return auc_