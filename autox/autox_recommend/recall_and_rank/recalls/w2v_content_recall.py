import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
import datetime
from time import time

def gen_detail_content_recall(sim_dict, target_df, data_hist,
                              uid, iid, time_col,
                              topn=20, topk=100, prefix='detail'):
    def REC(sim_dict, hists, topn=20, topk=100):
        rank = {}
        for art in hists:
            if art not in sim_dict:
                continue
            cnt = 0
            for sart, v in sim_dict[art].items():
                rank[sart] = max(rank.get(sart, 0), v)
                cnt += 1
                if cnt > topn:
                    break
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:topk]

    df = target_df.copy()
    tmp = data_hist.groupby(uid)[iid].agg(list).reset_index()
    df = df.merge(tmp, on=uid, how='left')

    samples = []
    for cur_uid, label, hists in tqdm(df.values):
        if hists is np.nan:
            continue
        rec = REC(sim_dict, hists, topn, topk)
        for k, v in rec:
            if k in label:
                samples.append([cur_uid, k, v, 1])
            else:
                samples.append([cur_uid, k, v, 0])
    samples = pd.DataFrame(samples, columns=[uid, iid, '{}_content_sim_score'.format(prefix), 'label'])
    print('{} content recall: '.format(prefix), samples.shape, samples.label.mean())

    return samples

def gen_detail_content_recall_test(sim_dict, data_hist,
                                   uid, iid, time_col,
                                   topn=20, topk=100, prefix='detail'):
    def REC(sim_dict, hists, topn=20, topk=100):
        rank = {}
        for art in hists:
            if art not in sim_dict:
                continue
            cnt = 0
            for sart, v in sim_dict[art].items():
                rank[sart] = max(rank.get(sart, 0), v)
                cnt += 1
                if cnt > topn:
                    break
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:topk]

    df = data_hist.groupby(uid)[iid].agg(list).reset_index()

    samples = []
    for cur_uid, hists in tqdm(df.values):
        if hists is np.nan:
            continue
        rec = REC(sim_dict, hists, topn, topk)
        for k, v in rec:
            samples.append([cur_uid, k, v])

    samples = pd.DataFrame(samples, columns=[uid, iid, '{}_content_sim_score'.format(prefix)])
    print('{} content recall: '.format(prefix), samples.shape)

    return samples

def get_art_sim_dict(df, art_map_dic,
                     uid, iid, time_col,
                     topn=100):
    feats = [c for c in df.columns if c not in [iid]]
    split_size = 2000
    split_num = int(len(df) / split_size)
    if len(df) % split_size != 0:
        split_num += 1

    w2v_vec = df[feats].values

    l2norm = np.linalg.norm(w2v_vec, axis=1, keepdims=True)
    w2v_vec = w2v_vec / (l2norm + 1e-9)

    w2v_vec_T = w2v_vec.T

    art_sim_dict = {}
    cnt = 0
    for i in tqdm(range(split_num)):

        vec = w2v_vec[i * split_size:(i + 1) * split_size]
        sim = vec.dot(w2v_vec_T)

        idx = (-sim).argsort(axis=1)
        sim = (-sim)
        sim.sort(axis=1)

        idx = idx[:, :topn]
        score = sim[:, :topn]
        score = -score

        for idx_, score_ in zip(idx, score):
            idx_ = [art_map_dic[j] for j in idx_]
            art_sim_dict[art_map_dic[cnt]] = dict(zip(idx_, score_))
            cnt += 1

    return art_sim_dict

def train_model(data, size=10, save_path='w2v_model/', iter=5, window=20):
    """训练模型"""
    print('Begin training w2v model')
    begin_time = time()
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = Word2Vec(data, vector_size=size, window=window, min_count=0, workers=20,
                     seed=1997, epochs=iter, sg=1, hs=1, compute_loss=True)
    print(model.get_latest_training_loss())

    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', round(run_time, 2))
    return model

def get_w2v_model(df_, date,
                  uid, iid, time_col,
                  last_days=30, size=10, iter=5, save_path='w2v_model/', window=20):
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
    begin_date = str(begin_date)

    df = df_[(df_[time_col] <= date) & (df_[time_col] >= begin_date)]

    user_item = df.groupby(uid)[iid].agg(list).reset_index()
    model = train_model(user_item[iid].values, size=size, iter=iter, save_path=save_path, window=window)

    return model

def generate_w2v_sim(transactions_train, date,
                     uid, iid, time_col,
                     last_days = 180, size=5):

    w2v_model = get_w2v_model(transactions_train, date,
                              uid, iid, time_col,
                              size=size, last_days=last_days)
    w2v_df = pd.DataFrame()
    w2v_df[iid] = w2v_model.wv.index_to_key
    w2v_vectors = pd.DataFrame(w2v_model.wv.vectors,
                               columns=[f'{iid}_w2v_dim{i}' for i in range(w2v_model.wv.vector_size)])
    w2v_df = pd.concat([w2v_df, w2v_vectors], axis=1)

    pop_num = 6000
    begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=7)
    begin_date = str(begin_date)

    data_lw = transactions_train[
        (transactions_train[time_col] >= begin_date) & (transactions_train[time_col] <= date)]
    dummy_dict = data_lw[iid].value_counts()
    recent_active_items = list(dummy_dict.index[:pop_num])

    df = w2v_df[w2v_df[iid].isin(recent_active_items)]
    art_map_dic = dict(zip(range(len(df)), df[iid].values.tolist()))

    art_sim_dict = get_art_sim_dict(df, art_map_dic,
                                    uid, iid, time_col,
                                    topn=200)
    return art_sim_dict

def w2v_concent_recall(uids, data, date,
                  uid, iid, time_col,
                  last_days=7, dtype='train',
                  topn=20, topk=100, prefix='w2v'):
    assert dtype in ['train', 'test']

    if dtype == 'train':

        begin_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=last_days)
        begin_date = str(begin_date)

        target_df = data[(data[time_col] <= date) & (data[time_col] > begin_date)]
        print(target_df[time_col].min(), target_df[time_col].max())

        target = target_df.groupby(uid)[iid].agg(list).reset_index()
        target.columns = [uid, 'label']

        data_hist = data[data[time_col] <= begin_date]
        data_hist_ = data_hist[data_hist[uid].isin(target[uid].unique())]

        last_days, size = 180, 32
        sim_dict = generate_w2v_sim(data, date,
                                    uid, iid, time_col,
                                    last_days=last_days, size=size)

        samples = gen_detail_content_recall(sim_dict, target, data_hist_,
                                            uid, iid, time_col,
                                            topn=topn, topk=topk, prefix=prefix)

        return samples

    elif dtype == 'test':

        last_days, size = 180, 32
        sim_dict = generate_w2v_sim(data, date,
                                    uid, iid, time_col,
                                    last_days=last_days, size=size)

        data_ = data[data[uid].isin(uids)]
        samples = gen_detail_content_recall_test(sim_dict, data_,
                                                 uid, iid, time_col,
                                                 topn=topn, topk=topk, prefix=prefix)

        return samples
