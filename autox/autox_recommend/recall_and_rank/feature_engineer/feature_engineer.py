import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
from .interact_feature_engineer import interact_feature_engineer
from .user_feature_engineer import user_feature_engineer


def feature_engineer(samples, data, date,
                     user_df, item_df,
                     uid, iid, time_col, last_days=7, dtype='train'):
    assert dtype in ['train', 'test']

    if dtype == 'train':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date)
        data_hist = data[data[time_col] <= begin_date]

        print('customer feature engineer')
        samples = user_feature_engineer(samples, data_hist, uid, iid, time_col)

        #         print('article feature engineer')
        #         samples = article_feature_engineer(samples, data_hist, uid, iid, time_col)

        if user_df is not None:
            samples = samples.merge(user_df, on=uid, how='left')
        if item_df is not None:
            samples = samples.merge(item_df, on=iid, how='left')

        #         print(samples.head())
        #         print(data_hist.head())
        print('interact feature engineer')
        samples = interact_feature_engineer(samples, data_hist, uid, iid, time_col)

    elif dtype == 'test':

        print('customer feature engineer')
        samples = user_feature_engineer(samples, data, uid, iid, time_col)

        #         print('article feature engineer')
        #         samples = article_feature_engineer(samples, data, uid, iid, time_col)

        if user_df is not None:
            samples = samples.merge(user_df, on=uid, how='left')
        if item_df is not None:
            samples = samples.merge(item_df, on=iid, how='left')

        print('interact feature engineer')
        samples = interact_feature_engineer(samples, data, uid, iid, time_col)

    return samples