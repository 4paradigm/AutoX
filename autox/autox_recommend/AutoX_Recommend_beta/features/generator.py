import datetime

from features.item_gen import *
from features.user_gen import user_feature_engineer
from features.interaction_gen import interact_feature_engineer
from utils.logger import logger


class FeatureGenerator(object):
    def __init__(self, uid, iid, required_attrs=None):
        self.uid = uid
        self.iid = iid
        self.required_attrs = required_attrs

    # TODO
    # 目前特征工程方法没有修改，仍然依赖时间，与HM数据集强相关
    # 根据以后更多的特征工程方法，进行修改

    def time_based_gen(self, samples, interactions_data, users_data, items_data, time_col, last_days=7, mode='train'):
        assert mode in ['train', 'test', 'valid']
        temp_date = datetime.datetime.strptime(str(interactions_data[time_col].max()), '%Y-%m-%d %H:%M:%S') + \
                    datetime.timedelta(days=1)
        valid_date = str(datetime.datetime(temp_date.year, temp_date.month, temp_date.day))
        date = valid_date

        if mode == 'train':
            begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
            date = str(begin_date)
            data_hist = interactions_data[interactions_data[time_col] <= date]
            interactions_data = data_hist

        logger.info('User Feature engineer')
        samples = user_feature_engineer(samples, interactions_data, self.uid, self.iid, time_col)

        if users_data is not None:
            samples = samples.merge(users_data, on=self.uid, how='left')
        if items_data is not None:
            samples = samples.merge(items_data, on=self.iid, how='left')

        print('Interaction Feature Engineer')
        samples = interact_feature_engineer(samples, interactions_data, self.uid, self.iid, time_col)

        iid2idx = {}
        idx2iid = {}
        for idx, cur_iid in enumerate(samples[self.iid].unique()):
            iid2idx[cur_iid] = idx
            idx2iid[idx] = cur_iid
        samples[self.iid + '_idx'] = samples[self.iid].map(iid2idx)

        return samples

    def generate(self, samples, interactions_data, users_data, items_data, mode='train'):
        # TODO
        # auto select methods by columns
        pass
