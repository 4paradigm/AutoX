import datetime
import pandas as pd
from tqdm import tqdm
from utils.logger import logger


class PopularRecall():
    def __init__(self, uid, iid, time_col, last_days=7, recall_num=100, required_attrs=None, **kwargs):
        self.uid = uid
        self.iid = iid
        self.time_col = time_col
        self.last_days = last_days
        self.recall_num = recall_num

        self.valid_date = None

    def fit(self, interactions, mode='train'):
        """
        PopularRecall 没有训练的过程，但考虑形式的一致性，并且调用fit时是整个模型的训练阶段，
        因此此处仍命名为fit和predict

        :param interactions: interaction data
        :return:
        """
        logger.info("Populary Recall fitting...")

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

        pop_begin_date = datetime.datetime.strptime(
            begin_date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=self.last_days)
        pop_begin_date = str(pop_begin_date)

        target = interactions[(interactions[self.time_col] <= date) & (
            interactions[self.time_col] > begin_date)]
        print(target[self.time_col].min(), target[self.time_col].max())
        target = target.groupby(self.uid)[self.iid].agg(list).reset_index()
        target.columns = [self.uid, 'label']

        data_lw = interactions[(interactions[self.time_col] >= pop_begin_date) & (
            interactions[self.time_col] <= begin_date)]
        popular_item = list(
            data_lw[self.iid].value_counts().index[:self.recall_num])

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
        logger.info('HIT: %.4f' % (hit / len(target)))
        samples = pd.DataFrame(samples, columns=[self.uid, self.iid, 'label'])

        return samples

    def predict(self, interaction, target_uids):
        date = self.valid_date
        begin_date = datetime.datetime.strptime(
            date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=self.last_days)
        begin_date = str(begin_date)

        data_lw = interaction[(interaction[self.time_col] >= begin_date) & (
            interaction[self.time_col] <= date)]
        print(data_lw[self.time_col].min(), data_lw[self.time_col].max())
        popular_item = list(
            data_lw[self.iid].value_counts().index[:self.recall_num])

        samples = []
        for cur_uid in tqdm(target_uids):
            for cur_iid in popular_item:
                samples.append([cur_uid, cur_iid])
        samples = pd.DataFrame(samples, columns=[self.uid, self.iid])

        return samples
