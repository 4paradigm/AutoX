import datetime
import pandas as pd
from tqdm import tqdm
from autox.autox_server.util import save_obj, load_obj

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
        samples.columns = [self.uid, 'prediction']
        return samples

    def save(self, path):
        save_obj(self.uid, f'{path}/uid.pkl')
        save_obj(self.iid, f'{path}/iid.pkl')
        save_obj(self.time_col, f'{path}/time_col.pkl')
        save_obj(self.popular_item, f'{path}/popular_item.pkl')

    def load(self, path):
        self.uid = load_obj(f'{path}/uid.pkl')
        self.iid = load_obj(f'{path}/iid.pkl')
        self.time_col = load_obj(f'{path}/time_col.pkl')
        self.popular_item = load_obj(f'{path}/popular_item.pkl')

