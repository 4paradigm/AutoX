import gc
import math
import warnings
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
import datetime

def ItemCF_Recommend(sim_item, user_item_dict, user_time_dict, user_price_dict, user_id, top_k, item_num, time_max,
                     rt_dict=False):
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_times = user_time_dict[user_id]
    interacted_prices = user_price_dict[user_id]
    for loc, i in enumerate(interacted_items):
        if i in sim_item:
            time = interacted_times[loc]  # datetime.datetime.strptime(interacted_times[loc], '%Y-%m-%d')
            price = interacted_prices[loc]
            items = sorted(sim_item[i].items(), reverse=True)[0:top_k]
            for j, wij in items:
                rank.setdefault(j, 0)
                rank[j] += wij * 0.8 ** time * price

    if rt_dict:
        return rank
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]


def get_sim_item(df,
                 user_col, item_col, time_col,
                 use_iif=False, time_max=None):
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    df['date'] = (time_max - df[time_col]).dt.days
    user_time_ = df.groupby(user_col)['date'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['date']))

    del df['date']
    gc.collect()

    user_price = df.groupby(user_col)['price'].agg(list).reset_index()
    user_price_dict = dict(zip(user_price[user_col], user_price['price']))

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
                        sim_item[item][relate_item] += 0.7 * (0.8 ** (t1 - t2)) / math.log(1 + len(items))
                    else:
                        sim_item[item][relate_item] += 1.0 * (0.8 ** (t2 - t1)) / math.log(1 + len(items))
                else:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    return sim_item_corr, user_item_dict, user_time_dict, user_price_dict


def get_itemcf_recall(data, target_df, df,
                      uid, iid, time_col,
                      time_max, topk=200, rec_num=100, use_iif=False):
    time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d %H:%M:%S')
    sim_item_corr, user_item_dict, user_time_dict, user_price_dict = get_sim_item(df,
                                                                                  uid, iid, time_col,
                                                                                  use_iif=use_iif,
                                                                                  time_max=time_max)

    samples = []
    target_df = target_df[target_df[uid].isin(data[uid].unique())]

    for cust, hist_arts, dates in tqdm(data[[uid, iid, time_col]].values):
        rec = ItemCF_Recommend(sim_item_corr, user_item_dict, user_time_dict, user_price_dict, cust, topk, rec_num,
                               time_max, )
        for k, v in rec:
            samples.append([cust, k, v])
    samples = pd.DataFrame(samples, columns=[uid, iid, 'itemcf_score'])

    print(samples.shape)
    target_df['label'] = 1
    samples = samples.merge(target_df[[uid, iid, 'label']], on=[uid, iid], how='left')
    samples['label'] = samples['label'].fillna(0)
    print('ItemCF recall: ', samples.shape)
    print(samples.label.mean())
    return samples


def itemcf_recall(uids, data, date,
                  uid, iid, time_col,
                  last_days=7, recall_num=100, dtype='train',
                  topk=1000, use_iif=False, sim_last_days=14):
    assert dtype in ['train', 'test']

    if dtype == 'train':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date)

        target_df = data[(data[time_col] <= date) & (data[time_col] > begin_date)]
        print(target_df[time_col].min(), target_df[time_col].max())

        target = target_df.groupby(uid)[iid].agg(list).reset_index()
        target.columns = [uid, 'label']

        data_hist = data[data[time_col] <= begin_date]

        # ItemCF进行召回
        data_hist_ = data_hist[data_hist[uid].isin(target[uid].unique())]
        df_hist = data_hist_.groupby(uid)[iid].agg(list).reset_index()
        tmp = data_hist_.groupby(uid)[time_col].agg(list).reset_index()
        df_hist = df_hist.merge(tmp, on=uid, how='left')

        samples = get_itemcf_recall(df_hist, target_df, data_hist,
                                    uid, iid, time_col,
                                    begin_date, topk=topk,
                                    rec_num=recall_num, use_iif=use_iif
                                    )

        return samples

    elif dtype == 'test':

        time_max = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        sim_item_corr, user_item_dict, user_time_dict, user_price_dict = get_sim_item(data,
                                                                                      uid, iid, time_col,
                                                                                      use_iif=use_iif,
                                                                                      time_max=time_max)

        data_ = data[data[uid].isin(uids)]

        df_hist = data_.groupby(uid)[iid].agg(list).reset_index()
        tmp = data_.groupby(uid)[time_col].agg(list).reset_index()
        df_hist = df_hist.merge(tmp, on=uid, how='left')

        samples = []
        for cust, hist_arts, dates in tqdm(df_hist[[uid, iid, time_col]].values):

            if cust not in user_item_dict:
                continue

            rec = ItemCF_Recommend(sim_item_corr, user_item_dict, user_time_dict, user_price_dict, cust, topk,
                                   recall_num, time_max)
            for k, v in rec:
                samples.append([cust, k, v])

        samples = pd.DataFrame(samples, columns=[uid, iid, 'itemcf_score'])

        return samples
