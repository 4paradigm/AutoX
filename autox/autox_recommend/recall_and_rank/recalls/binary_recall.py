import gc
import math
import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
import datetime


def BinaryNet_Recommend(sim_item, user_item_dict, user_time_dict, user_id, top_k, item_num, time_max,
                        rt_dict=False):
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_times = user_time_dict[user_id]
    for loc, i in enumerate(interacted_items):
        time = interacted_times[loc]
        items = sorted(sim_item[i].items(), reverse=True)[0:top_k]
        for j, wij in items:
            rank.setdefault(j, 0)
            rank[j] += wij * 0.8 ** time

    if rt_dict:
        return rank

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]


def get_sim_item_binary(df,
                        user_col, item_col, time_col,
                        time_max):
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    item_user_ = df.groupby(item_col)[user_col].agg(list).reset_index()
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

    df['date'] = (time_max - df[time_col]).dt.days
    user_time_ = df.groupby(user_col)['date'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['date']))

    del df['date']
    gc.collect()

    sim_item = {}
    for item, users in tqdm(item_user_dict.items()):
        sim_item.setdefault(item, {})
        for u in users:
            tmp_len = len(user_item_dict[u])
            for relate_item in user_item_dict[u]:
                sim_item[item].setdefault(relate_item, 0)
                sim_item[item][relate_item] += 1 / (math.log(len(users) + 1) * math.log(tmp_len + 1))

    return sim_item, user_item_dict, user_time_dict


def get_binaryNet_recall(custs, target_df, df,
                         uid, iid, time_col,
                         time_max, topk=200, rec_num=100):
    time_max = datetime.datetime.strptime(time_max, '%Y-%m-%d %H:%M:%S')
    sim_item, user_item_dict, user_time_dict, = get_sim_item_binary(df,
                                                                    uid, iid, time_col,
                                                                    time_max)

    samples = []
    target_df = target_df[target_df[uid].isin(custs)]
    for cust in tqdm(custs):
        if cust not in user_item_dict:
            continue
        rec = BinaryNet_Recommend(sim_item, user_item_dict, user_time_dict, cust, topk, rec_num,
                                  time_max)
        for k, v in rec:
            samples.append([cust, k, v])
    samples = pd.DataFrame(samples, columns=[uid, iid, 'binary_score'])
    print(samples.shape)
    target_df['label'] = 1
    samples = samples.merge(target_df[[uid, iid, 'label']], on=[uid, iid], how='left')
    samples['label'] = samples['label'].fillna(0)
    print('BinaryNet recall: ', samples.shape)
    print(samples.label.mean())

    return samples


def binary_recall(uids, data, date,
                  uid, iid, time_col,
                  last_days=7, recall_num=100, dtype='train', topk=1000):
    assert dtype in ['train', 'test']

    if dtype == 'train':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date)

        target_df = data[(data[time_col] <= date) & (data[time_col] > begin_date)]
        target = target_df.groupby(uid)[iid].agg(list).reset_index()
        target.columns = [uid, 'label']
        data_hist = data[data[time_col] <= begin_date]

        # BinaryNet进行召回
        binary_samples = get_binaryNet_recall(target[uid].unique(), target_df, data_hist,
                                              uid, iid, time_col,
                                              begin_date, topk=topk,
                                              rec_num=recall_num)

        return binary_samples

    elif dtype == 'test':

        time_max = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

        sim_item, user_item_dict, user_time_dict = get_sim_item_binary(
            data,
            uid, iid, time_col,
            time_max)

        samples = []
        for cust in tqdm(uids):
            if cust not in user_item_dict:
                continue

            rec = BinaryNet_Recommend(sim_item, user_item_dict, user_time_dict, cust, topk,
                                      recall_num,
                                      time_max)
            for k, v in rec:
                samples.append([cust, k, v])

        samples = pd.DataFrame(samples, columns=[uid, iid, 'binary_score'])
        return samples
