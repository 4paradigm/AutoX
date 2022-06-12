import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
from .type_recognition import Feature_type_recognition

def user_feature_engineer(samples, data, uid, iid, time_col):
    df = data[data[uid].isin(samples[uid].unique())]

    group = df.groupby(uid)

    tmp = group.agg({iid: ['count', 'nunique']}).reset_index()
    tmp.columns = [uid, 'n_purchase', 'n_purchase_nunique']
    samples = samples.merge(tmp, on=uid, how='left')

    feature_type_recognition = Feature_type_recognition()
    used = [x for x in data.columns if x not in [uid, iid, time_col]]

    if len(used) > 0:
        feature_type = feature_type_recognition.fit(data[used])

        num_cols = [x for x in feature_type if feature_type[x] == 'num']
        cat_cols = [x for x in feature_type if feature_type[x] == 'cat']

        print(f'num_cols: {num_cols}')
        print(f'cat_cols: {cat_cols}')
        for cur_num_col in num_cols:
            tmp = group.agg({cur_num_col: ['min', 'max', 'mean', 'std', 'median', 'sum',
                                           lambda x: sum(np.array(x) > x.mean())]}).reset_index()
            tmp.columns = [uid, f'{cur_num_col}_min', f'{cur_num_col}_max', f'{cur_num_col}_mean',
                           f'{cur_num_col}_std', f'{cur_num_col}_median', f'{cur_num_col}_sum',
                           f'{cur_num_col}_n_transactions_bigger_mean']
            samples = samples.merge(tmp, on=uid, how='left')

        for cur_num_col in cat_cols:
            tmp = group.agg({cur_num_col: ['nunique']}).reset_index()
            tmp.columns = [uid, f'{cur_num_col}_nunique']
            samples = samples.merge(tmp, on=uid, how='left')

        del tmp
        gc.collect()

    return samples