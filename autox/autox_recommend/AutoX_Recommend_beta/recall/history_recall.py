import datetime

import pandas as pd
from tqdm import tqdm

from utils.logger import logger


class HistoryRecall():
    def __init__(self, uid, iid, time_col, last_days=7, recall_num=100, required_attrs=None, **kwargs):
        self.uid = uid
        self.iid = iid
        self.time_col = time_col
        self.last_days = last_days
        self.recall_num = recall_num

    def fit(self, interactions, mode='train'):
        logger.info("History Recall fitting...")
        temp_date = datetime.datetime.strptime(str(interactions[self.time_col].max()), '%Y-%m-%d %H:%M:%S') + \
                    datetime.timedelta(days=1)
        valid_date = str(datetime.datetime(temp_date.year, temp_date.month, temp_date.day))
        self.valid_date = valid_date

        train_date = datetime.datetime.strptime(valid_date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=7)
        train_date = str(train_date)

        date = train_date
        if mode == 'valid':
            date = valid_date


        begin_date = datetime.datetime.strptime(
            date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=self.last_days)
        begin_date = str(begin_date)

        data_hist = interactions[interactions[self.time_col] <= begin_date]

        target = interactions[(interactions[self.time_col] <= date) & (
            interactions[self.time_col] > begin_date)]
        print(target[self.time_col].min(), target[self.time_col].max())

        target = target.groupby(self.uid)[self.iid].agg(list).reset_index()
        target.columns = [self.uid, 'label']

        purchase_df = data_hist[data_hist[self.uid].isin(target[self.uid].unique())].groupby(
            self.uid).tail(self.recall_num).reset_index(drop=True)
        purchase_df = purchase_df[[self.uid, self.iid]]
        purchase_df = purchase_df.groupby(
            self.uid)[self.iid].agg(list).reset_index()
        purchase_df = purchase_df.merge(target, on=self.uid, how='left')

        samples = []
        for cur_uid, cur_iids, label in tqdm(
                purchase_df.values, total=len(purchase_df)):
            for cur_iid in cur_iids:
                if cur_iid in label:
                    samples.append([cur_uid, cur_iid, 1])
                else:
                    samples.append([cur_uid, cur_iid, 0])

        samples = pd.DataFrame(samples, columns=[self.uid, self.iid, 'label'])

        return samples

    def predict(self, interactions, target_uids):
        purchase_df = interactions.loc[interactions[self.uid].isin(target_uids)].groupby(
            self.uid).tail(self.recall_num).reset_index(drop=True)
        samples = purchase_df[[self.uid, self.iid]]

        return samples
