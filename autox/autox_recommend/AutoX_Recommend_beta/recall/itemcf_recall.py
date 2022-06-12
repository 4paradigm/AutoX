import gc
import math
import warnings
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import datetime
from utils.logger import logger

warnings.filterwarnings('ignore')


class ItemCFRecall(object):
    def __init__(
            self,
            uid,
            iid,
            time_col,
            last_days=7,
            recall_num=100,
            topk=1000,
            sim_last_days=14,
            time_decay=0.8, required_attrs=None,
            **kwargs):
        self.uid = uid
        self.iid = iid
        self.time_col = time_col
        self.last_days = last_days
        self.recall_num = recall_num
        self.topk = topk
        self.sim_last_days = sim_last_days
        self.time_decay = time_decay

    def ItemCF_Recommend(
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
            if i in sim_item:
                time = interacted_times[loc]
                items = sorted(sim_item[i].items(), reverse=True)[0:self.topk]
                for j, wij in items:
                    rank.setdefault(j, 0)
                    rank[j] += wij * self.time_decay ** time

        if rt_dict:
            return rank
        return sorted(
            rank.items(),
            key=lambda d: d[1],
            reverse=True)[
            :self.recall_num]

    def get_sim_item(self, df, use_iif=False, time_max=None):
        user_item_ = df.groupby(self.uid)[self.iid].agg(list).reset_index()
        user_item_dict = dict(zip(user_item_[self.uid], user_item_[self.iid]))

        df['date'] = (time_max - df[self.time_col]).dt.days
        user_time_ = df.groupby(self.uid)['date'].agg(
            list).reset_index()  # 引入时间因素
        user_time_dict = dict(zip(user_time_[self.uid], user_time_['date']))

        del df['date']
        gc.collect()

        sim_item = {}
        item_cnt = defaultdict(int)  # 商品被点击次数

        for user, items in tqdm(user_item_dict.items()):
            for loc1, item in enumerate(items):
                item_cnt[item] += 1
                sim_item.setdefault(item, {})
                for loc2, relate_item in enumerate(items):
                    sim_item[item].setdefault(relate_item, 0)
                    if not use_iif:
                        t1 = user_time_dict[user][loc1]
                        t2 = user_time_dict[user][loc2]
                        if loc1 - loc2 > 0:
                            sim_item[item][relate_item] += 0.7 * \
                                (self.time_decay ** (t1 - t2)) / math.log(1 + len(items))
                        else:
                            sim_item[item][relate_item] += 1.0 * \
                                (self.time_decay ** (t2 - t1)) / math.log(1 + len(items))
                    else:
                        sim_item[item][relate_item] += 1 / \
                            math.log(1 + len(items))

        sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
        return sim_item_corr, user_item_dict, user_time_dict

    def get_itemcf_recall(self, data, target_df, df, time_max, use_iif=False):
        time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d %H:%M:%S')
        logger.info('calculate similarity')
        sim_item_corr, user_item_dict, user_time_dict = self.get_sim_item(
            df, use_iif=use_iif, time_max=time_max)

        samples = []
        target_df = target_df[target_df[self.uid].isin(
            data[self.uid].unique())]
        logger.info('ItemCF recommend')
        # todo: 并行优化
        for cust, hist_arts, dates in tqdm(
                data[[self.uid, self.iid, self.time_col]].values):
            rec = self.ItemCF_Recommend(
                sim_item_corr,
                user_item_dict,
                user_time_dict,
                cust,
                rt_dict=False)
            for k, v in rec:
                samples.append([cust, k, v])
        samples = pd.DataFrame(
            samples,
            columns=[
                self.uid,
                self.iid,
                'itemcf_score'])

        # print(samples.shape)
        target_df['label'] = 1
        samples = samples.merge(target_df[[self.uid, self.iid, 'label']], on=[
                                self.uid, self.iid], how='left')
        samples['label'] = samples['label'].fillna(0)
        logger.info('ItemCF recall: {}'.format(samples.shape))
        logger.info('mean: {}'.format(samples.label.mean()))
        logger.info('sum: {}'.format(samples.label.sum()))
        return samples

    def fit(self, interactions, mode='train', use_iif=False):
        logger.info("ItemCF Recall fitting...")

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

        target_df = interactions[(interactions[self.time_col] <= date) & (
            interactions[self.time_col] > begin_date)]
        # print(target_df[self.time_col].min(), target_df[self.time_col].max())

        target = target_df.groupby(self.uid)[self.iid].agg(list).reset_index()
        target.columns = [self.uid, 'label']

        data_hist = interactions[interactions[self.time_col] <= begin_date]

        # ItemCF进行召回
        data_hist_ = data_hist[data_hist[self.uid].isin(
            target[self.uid].unique())]
        df_hist = data_hist_.groupby(
            self.uid)[self.iid].agg(list).reset_index()
        tmp = data_hist_.groupby(
            self.uid)[self.time_col].agg(list).reset_index()
        df_hist = df_hist.merge(tmp, on=self.uid, how='left')

        samples = self.get_itemcf_recall(
            df_hist,
            target_df,
            data_hist,
            time_max=begin_date,
            use_iif=use_iif)

        return samples

    def predict(self, interactions, target_uids, use_iif=False):
        date = self.valid_date
        time_max = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        sim_item_corr, user_item_dict, user_time_dict = self.get_sim_item(
            interactions, use_iif=use_iif, time_max=time_max)

        data_ = interactions[interactions[self.uid].isin(target_uids)]

        df_hist = data_.groupby(self.uid)[self.iid].agg(list).reset_index()
        tmp = data_.groupby(self.uid)[self.time_col].agg(list).reset_index()
        df_hist = df_hist.merge(tmp, on=self.uid, how='left')

        samples = []
        for cust, hist_arts, dates in tqdm(
                df_hist[[self.uid, self.iid, self.time_col]].values):

            if cust not in user_item_dict:
                continue

            rec = self.ItemCF_Recommend(
                sim_item_corr, user_item_dict, user_time_dict, cust, False)
            for k, v in rec:
                samples.append([cust, k, v])

        samples = pd.DataFrame(
            samples,
            columns=[
                self.uid,
                self.iid,
                'itemcf_score'])

        return samples
