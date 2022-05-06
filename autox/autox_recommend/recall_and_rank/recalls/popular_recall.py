import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
import gc


def popular_recall(uids, data, date, uid, iid, time_col, last_days=7, recall_num=100, dtype='train'):
    assert dtype in ['train', 'test']

    if dtype == 'train':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date)

        pop_begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(
            days=last_days)
        pop_begin_date = str(pop_begin_date)

        target = data[(data[time_col] <= date) & (data[time_col] > begin_date)]
        print(target[time_col].min(), target[time_col].max())
        target = target.groupby(uid)[iid].agg(list).reset_index()
        target.columns = [uid, 'label']

        data_lw = data[(data[time_col] >= pop_begin_date) & (data[time_col] <= begin_date)]
        popular_item = list(data_lw[iid].value_counts().index[:recall_num])

        samples = []
        hit = 0
        for cur_uid, labels in tqdm(target.values):
            h = 0
            for cur_iid in popular_item:
                if cur_iid in labels:
                    sample = [cur_uid, cur_iid, 1]
                    h += 1
                else:
                    sample = [cur_uid, cur_iid, 0]
                samples.append(sample)
            hit += h / len(labels)
        print('HIT: ', hit / len(target))
        samples = pd.DataFrame(samples, columns=[uid, iid, 'label'])

        return samples

    elif dtype == 'test':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date)

        data_lw = data[(data[time_col] >= begin_date) & (data[time_col] <= date)]
        print(data_lw[time_col].min(), data_lw[time_col].max())
        popular_item = list(data_lw[iid].value_counts().index[:recall_num])

        samples = []
        for cur_uid in tqdm(uids):
            for cur_iid in popular_item:
                samples.append([cur_uid, cur_iid])
        samples = pd.DataFrame(samples, columns=[uid, iid])

        return samples
