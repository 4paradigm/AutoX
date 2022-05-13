import datetime
import pandas as pd
from .feature_engineer import feature_engineer
from .ranker import ranker, ranker_test, inference
from .recalls import binary_recall
from .recalls import history_recall
from .recalls import itemcf_recall
from .recalls import popular_recall
from ..metrics import mapk

class RecallAndRank():
    def __init__(self):
        pass

    def fit(self, inter_df, user_df, item_df,
                  uid, iid, time_col,
                  recall_num, debug=False):

        self.inter_df = inter_df
        self.user_df = user_df
        self.item_df = item_df

        self.uid = uid
        self.iid = iid
        self.time_col = time_col
        self.recall_num = recall_num

        if debug:
            import os
            path_output = './temp'
            os.makedirs(path_output, exist_ok=True)

        temp_date = datetime.datetime.strptime(str(inter_df[time_col].max()), '%Y-%m-%d %H:%M:%S') + \
                    datetime.timedelta(days=1)
        valid_date = str(datetime.datetime(temp_date.year, temp_date.month, temp_date.day))
        self.valid_date = valid_date

        train_date = datetime.datetime.strptime(valid_date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=7)
        train_date = str(train_date)


        print('\npopular_recall')
        print('train')
        popular_recall_train = popular_recall(None, inter_df, date=train_date,
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=recall_num, dtype='train')
        print('valid')
        popular_recall_valid = popular_recall(None, inter_df, date=valid_date,
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=recall_num, dtype='train')

        print('\nhistory_recall')
        print('train')
        history_recall_train = history_recall(None, inter_df, date=train_date,
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=recall_num, dtype='train')
        print('valid')
        history_recall_valid = history_recall(None, inter_df, date=valid_date,
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=recall_num, dtype='train')

        print('\nitemcf_recall')
        print('train')
        itemcf_recall_train = itemcf_recall(None, inter_df, date=train_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=recall_num, dtype='train',
                                            topk=1000, use_iif=False, sim_last_days=14)
        print('valid')
        itemcf_recall_valid = itemcf_recall(None, inter_df, date=valid_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=recall_num, dtype='train',
                                            topk=1000, use_iif=False, sim_last_days=14)
        if debug:
            itemcf_recall_train.to_hdf(f'{path_output}/itemcf_recall_train.hdf', 'w', complib='blosc', complevel=5)
            itemcf_recall_valid.to_hdf(f'{path_output}/itemcf_recall_valid.hdf', 'w', complib='blosc', complevel=5)

        print('\nbinary_recall')
        print('train')
        binary_recall_train = binary_recall(None, inter_df, date=train_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=recall_num, dtype='train', topk=1000)

        print('valid')
        binary_recall_valid = binary_recall(None, inter_df, date=valid_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=recall_num, dtype='train', topk=1000)
        if debug:
            binary_recall_train.to_hdf(f'{path_output}/binary_recall_train.hdf', 'w', complib='blosc', complevel=5)
            binary_recall_valid.to_hdf(f'{path_output}/binary_recall_valid.hdf', 'w', complib='blosc', complevel=5)


        # 合并召回数据
        print('\nmerge recalls')
        print('train')
        history_recall_train.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        itemcf_recall_train.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        binary_recall_train.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        train = popular_recall_train.append(history_recall_train)
        train.drop_duplicates(subset=[uid, iid], keep='first', inplace=True)
        train = train.merge(itemcf_recall_train, on=[uid, iid, 'label'], how='outer')
        train = train.merge(binary_recall_train, on=[uid, iid, 'label'], how='outer')

        print('valid')
        history_recall_valid.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        itemcf_recall_valid.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        binary_recall_valid.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        valid = popular_recall_valid.append(history_recall_valid)
        valid.drop_duplicates(subset=[uid, iid], keep='first', inplace=True)
        valid = valid.merge(itemcf_recall_valid, on=[uid, iid, 'label'], how='outer')
        valid = valid.merge(binary_recall_valid, on=[uid, iid, 'label'], how='outer')

        # 特征工程
        print('\nfeature engineer')
        print('train')
        train_fe = feature_engineer(train, inter_df,
                                    date=train_date,
                                    user_df=user_df, item_df=item_df,
                                    uid=uid, iid=iid, time_col=time_col,
                                    last_days=7, dtype='train')
        print('valid')
        valid_fe = feature_engineer(valid, inter_df,
                                    date=valid_date,
                                    user_df=user_df, item_df=item_df,
                                    uid=uid, iid=iid, time_col=time_col,
                                    last_days=7, dtype='train')

        if debug:
            train_fe.to_hdf(f'{path_output}/train_fe.hdf', 'w', complib='blosc', complevel=5)
            valid_fe.to_hdf(f'{path_output}/valid_fe.hdf', 'w', complib='blosc', complevel=5)


        iid2idx = {}
        idx2iid = {}
        for idx, cur_iid in enumerate(train_fe[iid].unique()):
            iid2idx[cur_iid] = idx
            idx2iid[idx] = cur_iid
        self.iid2idx = iid2idx
        train_fe[iid + '_idx'] = train_fe[iid].map(iid2idx)
        valid_fe[iid + '_idx'] = valid_fe[iid].map(iid2idx)

        print(f"train_fe shape: {train_fe.shape}")
        print(f"valid_fe shape: {valid_fe.shape}")

        print('\nranker')
        lgb_ranker, valid_pred = ranker(train_fe, valid_fe,
                                        uid=uid, iid=iid, time_col=time_col)
        
        print('\nlocal result calculation')
        # 离线结果打印
        valid_pred = valid_pred.sort_values('prob', ascending=False)
        valid_pred = valid_pred.groupby(uid).head(12).groupby(uid)[iid].agg(list).reset_index()

        begin_date = datetime.datetime.strptime(valid_date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=7)
        begin_date = str(begin_date)

        valid_true = inter_df.loc[inter_df[uid].isin(valid_pred[uid])]
        valid_true = valid_true[(valid_true[time_col] <= valid_date) & (valid_true[time_col] > begin_date)]

        print(valid_true[time_col].min(), valid_true[time_col].max())
        valid_true = valid_true.groupby(uid)[iid].agg(list).reset_index()

        print("mAP Score on Validation set:", mapk(valid_true[iid], valid_pred[iid]))

        self.best_iteration_ = lgb_ranker.best_iteration_


        print("#" * 30)
        print('retrain')
        # 重新训练
        train_date = valid_date
        # train_date = '2022-04-07 00:00:00'

        print('\npopular_recall')
        popular_recall_train = popular_recall(None, inter_df, date=train_date,
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=recall_num, dtype='train')
        print('\nhistory_recall')
        history_recall_train = history_recall(None, inter_df, date=train_date,
                                              uid=uid, iid=iid, time_col=time_col,
                                              last_days=7, recall_num=recall_num, dtype='train')
        print('\nitemcf_recall')
        itemcf_recall_train = itemcf_recall(None, inter_df, date=train_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=recall_num, dtype='train',
                                            topk=1000, use_iif=False, sim_last_days=14)

        if debug:
            itemcf_recall_train.to_hdf(f'{path_output}/itemcf_recall_train_all.hdf', 'w', complib='blosc', complevel=5)

        print('\nbinary_recall')
        binary_recall_train = binary_recall(None, inter_df, date=train_date,
                                            uid=uid, iid=iid, time_col=time_col,
                                            last_days=7, recall_num=recall_num, dtype='train', topk=1000)
        if debug:
            binary_recall_train.to_hdf(f'{path_output}/binary_recall_train_all.hdf', 'w', complib='blosc', complevel=5)


        # 合并召回数据
        print('\nmerge recalls')
        history_recall_train.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        itemcf_recall_train.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        binary_recall_train.drop_duplicates(subset=[uid, iid, 'label'], keep='first', inplace=True)
        train = popular_recall_train.append(history_recall_train)
        train.drop_duplicates(subset=[uid, iid], keep='first', inplace=True)
        train = train.merge(itemcf_recall_train, on=[uid, iid, 'label'], how='outer')
        train = train.merge(binary_recall_train, on=[uid, iid, 'label'], how='outer')

        # 特征工程
        print('\nfeature engineer')
        train_fe = feature_engineer(train, inter_df,
                                    date=train_date,
                                    user_df=user_df, item_df=item_df,
                                    uid=uid, iid=iid, time_col=time_col,
                                    last_days=7, dtype='train')
        if debug:
            train_fe.to_hdf(f'{path_output}/train_fe_all.hdf', 'w', complib='blosc', complevel=5)


        train_fe[iid + '_idx'] = train_fe[iid].map(iid2idx)
        print(f"train_fe shape: {train_fe.shape}")

        print('\nranker')
        self.model, self.feats = ranker_test(train_fe, self.best_iteration_, 
                                   uid=uid, iid=iid, time_col=time_col)
        

    def transform(self, uids):

        test_date = self.valid_date
        # test_date = '2022-04-07 00:00:00'

        print('\npopular recall, test')
        popular_recall_test = popular_recall(uids, self.inter_df, date=test_date,
                                             uid=self.uid, iid=self.iid, time_col=self.time_col,
                                             last_days=7, recall_num=self.recall_num, dtype='test')
        print('\nhistory recall, test')
        history_recall_test = history_recall(uids, self.inter_df, date=test_date,
                                             uid=self.uid, iid=self.iid, time_col=self.time_col,
                                             last_days=7, recall_num=self.recall_num, dtype='test')
        print('\nitemcf recall, test')
        itemcf_recall_test = itemcf_recall(uids, self.inter_df, date=test_date,
                                           uid=self.uid, iid=self.iid, time_col=self.time_col,
                                           last_days=7, recall_num=self.recall_num, dtype='test',
                                           topk=1000, use_iif=False, sim_last_days=14)
        print('\nbinary recall, test')
        binary_recall_test = binary_recall(uids, self.inter_df, date=test_date,
                                           uid=self.uid, iid=self.iid, time_col=self.time_col,
                                           last_days=7, recall_num=self.recall_num, dtype='test', topk=1000)

        print('\nmerge recalls')
        history_recall_test.drop_duplicates(subset=[self.uid, self.iid], keep='first', inplace=True)
        itemcf_recall_test.drop_duplicates(subset=[self.uid, self.iid], keep='first', inplace=True)
        binary_recall_test.drop_duplicates(subset=[self.uid, self.iid], keep='first', inplace=True)
        test = popular_recall_test.append(history_recall_test)
        test.drop_duplicates(subset=[self.uid, self.iid], keep='first', inplace=True)
        test = test.merge(itemcf_recall_test, on=[self.uid, self.iid], how='outer')
        test = test.merge(binary_recall_test, on=[self.uid, self.iid], how='outer')

        print('\nfeature engineer')
        test_fe = feature_engineer(test, self.inter_df,
                                   date=test_date,
                                   user_df=self.user_df, item_df=self.item_df,
                                   uid=self.uid, iid=self.iid, time_col=self.time_col,
                                   last_days=7, dtype='test')
        test_fe[self.iid + '_idx'] = test_fe[self.iid].map(self.iid2idx)
        print(f"test_fe shape: {test_fe.shape}")

        print('\ninference')
        bs = 60000
        recs = inference(self.model, self.feats, test_fe, uids,
                         uid=self.uid, iid=self.iid, time_col=self.time_col,
                         batch_size=bs)

        return recs
