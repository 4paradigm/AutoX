import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
import gc

def history_recall(uids, data, date, uid, iid, time_col, last_days=7, recall_num=100, dtype='train'):
    assert dtype in ['train', 'test']

    if dtype == 'train':
        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date)

        data_hist = data[data[time_col] <= begin_date]

        target = data[(data[time_col] <= date) & (data[time_col] > begin_date)]
        print(target[time_col].min(), target[time_col].max())

        target = target.groupby(uid)[iid].agg(list).reset_index()
        target.columns = [uid, 'label']

        purchase_df = data_hist[data_hist[uid].isin(target[uid].unique())].groupby(uid).tail(recall_num).reset_index(
            drop=True)
        purchase_df = purchase_df[[uid, iid]]
        purchase_df = purchase_df.groupby(uid)[iid].agg(list).reset_index()
        purchase_df = purchase_df.merge(target, on=uid, how='left')

        samples = []
        for cur_uid, cur_iids, label in tqdm(purchase_df.values, total=len(purchase_df)):
            for cur_iid in cur_iids:
                if cur_iid in label:
                    samples.append([cur_uid, cur_iid, 1])
                else:
                    samples.append([cur_uid, cur_iid, 0])

        samples = pd.DataFrame(samples, columns=[uid, iid, 'label'])

        return samples

    elif dtype == 'test':

        purchase_df = data.loc[data[uid].isin(uids)].groupby(uid).tail(recall_num).reset_index(drop=True)
        samples = purchase_df[[uid, iid]]

        return samples
