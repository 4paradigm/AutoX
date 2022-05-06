import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc

def user_feature_engineer(samples, data, uid, iid, time_col):

    df = data[data[uid].isin(samples[uid].unique())]

    group = df.groupby(uid)

    tmp = group.agg({
        iid: 'count',
        'price': lambda x: sum(np.array(x) > x.mean()),
        'sales_channel_id': lambda x: sum(x == 1),
    }).rename(columns={
        iid: 'n_purchase',
        'price': 'n_transactions_bigger_mean',
        'sales_channel_id': 'n_online_articles'
    }).reset_index()
    samples = samples.merge(tmp, on=uid, how='left')

    agg_cols = ['min', 'max', 'mean', 'std', 'median', 'sum']
    tmp = group['price'].agg(agg_cols).reset_index()
    tmp.columns = [uid] + ['customer_price_{}'.format(col) for col in agg_cols]
    tmp['customer_price_max_minus_min'] = tmp['customer_price_max'] - tmp['customer_price_min']
    samples = samples.merge(tmp, on=uid, how='left')

    tmp = group.agg({
        iid: 'nunique',
        'sales_channel_id': lambda x: sum(x == 1),
    }).rename(columns={
        iid: 'n_purchase_nuniq',
        'sales_channel_id': 'n_store_articles'
    }).reset_index()
    samples = samples.merge(tmp, on=uid, how='left')

    del tmp
    gc.collect()

    return samples