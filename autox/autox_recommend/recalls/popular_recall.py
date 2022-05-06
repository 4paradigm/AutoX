import datetime
import pandas as pd
from tqdm import tqdm

class PopularRecall():
    def __init__(self):
        pass

    def fit(self, inter_df, user_df, item_df,
                  uid, iid, time_col,
                  recall_num):
        self.uid = uid
        self.iid = iid
        self.time_col = time_col

        date = str(inter_df[time_col].max())
        last_days = 7

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date)

        data_lw = inter_df[(inter_df[time_col] >= begin_date) & (inter_df[time_col] <= date)]
        self.popular_item = list(data_lw[iid].value_counts().index[:recall_num])

    def transform(self, uids):

        samples = []
        for cur_uid in tqdm(uids):
            for cur_iid in self.popular_item:
                samples.append([cur_uid, cur_iid])
        samples = pd.DataFrame(samples, columns=[self.uid, self.iid])

        samples = samples.groupby(self.uid)[self.iid].agg(list).reset_index()
        samples.columns = [self.uid, 'recommend']
        return samples
