import gc
import math
import warnings
import pandas as pd
from tqdm import tqdm
import datetime

from utils.logger import logger

warnings.filterwarnings('ignore')


# TODO
# 1. 效果输出
class BinaryNetRecall(object):
    def __init__(
            self,
            uid,
            iid,
            time_col,
            last_days=7,
            recall_num=100,
            topk=1000,
            required_attrs=None,
            **kwargs):
        self.uid = uid
        self.iid = iid
        self.time_col = time_col
        self.last_days = last_days
        self.recall_num = recall_num
        self.topk = topk

    def BinaryNet_Recommend(
            self,
            sim_item,
            user_item_dict,
            user_time_dict,
            user_id,
            rt_dict=False):
        rank = {}
        interacted_items = user_item_dict[user_id]
        interacted_times = user_time_dict[user_id]
        for loc, i in enumerate(interacted_items):
            time = interacted_times[loc]
            items = sorted(sim_item[i].items(), reverse=True)[0:self.top_k]
            for j, wij in items:
                rank.setdefault(j, 0)
                rank[j] += wij * 0.8 ** time

        if rt_dict:
            return rank

        return sorted(
            rank.items(),
            key=lambda d: d[1],
            reverse=True)[
            :self.recall_num]

    def get_sim_item_binary(self, df, time_max):
        user_item_ = df.groupby(self.uid)[self.iid].agg(list).reset_index()
        user_item_dict = dict(zip(user_item_[self.uid], user_item_[self.iid]))

        item_user_ = df.groupby(self.iid)[self.uid].agg(list).reset_index()
        item_user_dict = dict(zip(item_user_[self.iid], item_user_[self.uid]))

        df['date'] = (time_max - df[self.time_col]).dt.days
        user_time_ = df.groupby(self.uid)['date'].agg(
            list).reset_index()  # 引入时间因素
        user_time_dict = dict(zip(user_time_[self.uid], user_time_['date']))

        del df['date']
        gc.collect()

        sim_item = {}
        for item, users in tqdm(item_user_dict.items()):
            sim_item.setdefault(item, {})
            for u in users:
                tmp_len = len(user_item_dict[u])
                for relate_item in user_item_dict[u]:
                    sim_item[item].setdefault(relate_item, 0)
                    sim_item[item][relate_item] += 1 / \
                        (math.log(len(users) + 1) * math.log(tmp_len + 1))

        return sim_item, user_item_dict, user_time_dict

    def get_binaryNet_recall(self, custs, target_df, df, time_max):
        time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d %H:%M:%S')
        sim_item, user_item_dict, user_time_dict, = self.get_sim_item_binary(
            df, time_max)

        samples = []
        target_df = target_df[target_df[self.uid].isin(custs)]
        for cust in tqdm(custs):
            if cust not in user_item_dict:
                continue
            rec = self.BinaryNet_Recommend(
                sim_item, user_item_dict, user_time_dict, cust)
            for k, v in rec:
                samples.append([cust, k, v])
        samples = pd.DataFrame(
            samples,
            columns=[
                self.uid,
                self.iid,
                'binary_score'])
        print(samples.shape)
        target_df['label'] = 1
        samples = samples.merge(target_df[[self.uid, self.iid, 'label']], on=[
                                self.uid, self.iid], how='left')
        samples['label'] = samples['label'].fillna(0)
        print('BinaryNet recall: ', samples.shape)
        print(samples.label.mean())

        return samples

    def fit(self, interactions, mode='train'):
        logger.info("Binary Net Recall fitting...")

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

        target_df = interactions[(interactions[self.time_col] <= date) &
                                 (interactions[self.time_col] > begin_date)]
        target = target_df.groupby(self.uid)[self.iid].agg(list).reset_index()
        target.columns = [self.uid, 'label']
        data_hist = interactions[interactions[self.time_col] <= begin_date]

        # BinaryNet进行召回
        binary_samples = self.get_binaryNet_recall(
            target[self.uid].unique(), target_df, data_hist, begin_date)

        return binary_samples

    def predict(self, data, target_uids):
        date = self.valid_date
        time_max = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

        sim_item, user_item_dict, user_time_dict = self.get_sim_item_binary(
            data, time_max)

        samples = []
        for cust in tqdm(target_uids):
            if cust not in user_item_dict:
                continue

            rec = self.BinaryNet_Recommend(
                sim_item, user_item_dict, user_time_dict, cust)
            for k, v in rec:
                samples.append([cust, k, v])

        samples = pd.DataFrame(
            samples,
            columns=[
                self.uid,
                self.iid,
                'binary_score'])
        return samples
