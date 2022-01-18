import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import time
import pandas as pd
import numpy as np
import os
from autox.autox_server.util import log



top_k = 5
used_concat_cols = []
cnt = 0
for i in range(len(G_hist['base_lgb']['feature_importances'])):
    if cnt == top_k:
        break
    cur_feature = G_hist['base_lgb']['feature_importances'].loc[i, 'feature']
    if cur_feature in G_hist['big_cols_cat']:
        used_concat_cols.append(cur_feature)
        cnt += 1


def cols_concat(df, con_list):
    name = "__".join(con_list)
    df[name] = df[con_list[0]].astype(str)
    for item in con_list[1:]:
        df[name] = df[name] + '__' + df[item].astype(str)
    return df
concat_features = []

for col_1, col_2 in combinations(used_concat_cols, 2):
    cur_col = col_1 + "__" + col_2
    print(cur_col)
    concat_features.append(cur_col)
    G_df_dict['FE_concat'] = cols_concat(G_df_dict['FE_concat'], [col_1, col_1])