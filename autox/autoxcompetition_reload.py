import pandas as pd
from autox.autox_competition.feature_engineer import fe_ima2vec
from autox.autox_competition.process_data import auto_encoder
from autox.autox_competition.process_data import feature_combination, train_test_divide, clip_label
from autox.autox_competition.util import log


class AutoXReload():
    """AutoX主函数描述"""

    def __init__(self, autox, test_path, test=None):
        self.info_ = autox.info_
        self.dfs_ = autox.dfs_
        self.featureOne2M = autox.featureOne2M
        self.featureTime = autox.featureTime
        self.featureCumsum = autox.featureCumsum
        self.featureShift = autox.featureShift
        self.featureDiff = autox.featureDiff
        self.featureStat = autox.featureStat
        self.featureNlp = autox.featureNlp
        self.featureCount = autox.featureCount
        self.featureRank = autox.featureRank
        self.used_features = autox.used_features

        self.model_lgb = autox.model_lgb
        self.model_xgb = autox.model_xgb

        if test_path is None:
            autox.dfs_[self.info_['test_name']] = test
        else:
            autox.dfs_[self.info_['test_name']] = pd.read_csv(test_path)

        self.concat_train_test()
        self.dfs_['FE_all'] = None
        self.sub = None

    def concat_train_test(self):
        self.info_['shape_of_test'] = len(self.dfs_[self.info_['test_name']])
        self.dfs_['train_test'] = self.dfs_[self.info_['train_name']].append(self.dfs_[self.info_['test_name']])
        self.dfs_['train_test'].index = range(len(self.dfs_['train_test']))

    def split_train_test(self):
        self.dfs_['FE_train'] = self.dfs_['FE_all'][:self.info_['shape_of_train']]
        self.dfs_['FE_test'] = self.dfs_['FE_all'][self.info_['shape_of_train']:]

    def get_submit(self):
        self.topk_feas = self.get_top_features(return_df = False)

        # 模型训练
        log("start training xgboost model")
        # pass

        # 模型预测
        predict_lgb = self.model_lgb.predict(self.test[self.used_features])
        predict_xgb = self.model_xgb.predict(self.test[self.used_features].astype('float64'))
        # predict_tabnet = model_tabnet.predict(test[used_features])
        predict = (predict_xgb + predict_lgb) / 2

        # 预测结果后处理
        min_ = self.info_['min_target']
        max_ = self.info_['max_target']
        predict = clip_label(predict, min_, max_)

        # 获得结果
        sub = self.test[self.info_['id']]
        sub[self.info_['target']] = predict
        sub.index = range(len(sub))

        return sub

    def get_top_features(self, topk = 50, return_df = True):

        id_ = self.info_['id']
        target = self.info_['target']

        # 特征工程
        log("start feature engineer")
        df = self.dfs_['train_test']
        feature_type = self.info_['feature_type']['train_test']

        # 1-M拼表特征
        # one2M拼表特征
        log("feature engineer: one2M")
        featureOne2M = self.featureOne2M
        log(f"featureOne2M ops: {featureOne2M.get_ops()}")
        if len(featureOne2M.get_ops()) != 0:
            self.dfs_['FE_One2M'] = featureOne2M.transform(df, self.dfs_)
        else:
            self.dfs_['FE_One2M'] = None
            log("ignore featureOne2M")

        # 时间特征
        log("feature engineer: time")
        featureTime = self.featureTime
        log(f"featureTime ops: {featureTime.get_ops()}")
        self.dfs_['FE_time'] = featureTime.transform(df)


        # cumsum特征
        log("feature engineer: Cumsum")
        featureCumsum = self.featureCumsum

        fe_cumsum_cnt = 0
        for key_ in featureCumsum.get_ops().keys():
            fe_cumsum_cnt += len(featureCumsum.get_ops()[key_])
        if fe_cumsum_cnt < 30:
            self.dfs_['FE_cumsum'] = featureCumsum.transform(df)
            log(f"featureCumsum ops: {featureCumsum.get_ops()}")
        else:
            self.dfs_['FE_cumsum'] = None
            log("ignore featureCumsum")


        # shift特征
        log("feature engineer: Shift")
        featureShift = self.featureShift
        fe_shift_cnt = 0
        for key_ in featureShift.get_ops().keys():
            fe_shift_cnt += len(featureShift.get_ops()[key_])
        if fe_shift_cnt < 30:
            self.dfs_['FE_shift'] = featureShift.transform(df)
            log(f"featureShift ops: {featureShift.get_ops()}")
        else:
            self.dfs_['FE_shift'] = None
            log("ignore featureShift")


        # diff特征
        log("feature engineer: Diff")
        featureDiff = self.featureDiff
        fe_diff_cnt = 0
        for key_ in featureDiff.get_ops().keys():
            fe_diff_cnt += len(featureDiff.get_ops()[key_])
        if fe_diff_cnt < 30:
            self.dfs_['FE_diff'] = featureDiff.transform(df)
            log(f"featureDiff ops: {featureDiff.get_ops()}")
        else:
            self.dfs_['FE_diff'] = None
            log("ignore featureDiff")


        # 统计特征
        log("feature engineer: Stat")
        featureStat = self.featureStat
        fe_stat_cnt = 0
        for key_ in featureStat.get_ops().keys():
            aggs = featureStat.get_ops()[key_]
            for cur_agg in aggs:
                fe_stat_cnt += len(featureStat.get_ops()[key_][cur_agg])
        if fe_stat_cnt < 1500:
            self.dfs_['FE_stat'] = featureStat.transform(df)
            log(f"featureStat ops: {featureStat.get_ops()}")
        else:
            self.dfs_['FE_stat'] = None
            log("ignore featureStat")


        # nlp特征
        log("feature engineer: NLP")
        featureNlp = self.featureNlp
        self.dfs_['FE_nlp'] = featureNlp.transform(df)
        log(f"featureNlp ops: {featureNlp.get_ops()}")


        # count特征
        log("feature engineer: Count")
        # degree自动调整
        featureCount = self.featureCount
        log(f"featureCount ops: {featureCount.get_ops()}")
        self.dfs_['FE_count'] = featureCount.transform(df)


        # rank特征
        log("feature engineer: Rank")
        featureRank = self.featureRank
        fe_rank_cnt = 0
        for key_ in featureRank.get_ops().keys():
            fe_rank_cnt += len(featureRank.get_ops()[key_])
        if fe_rank_cnt < 500:
            self.dfs_['FE_rank'] = featureRank.transform(df)
            log(f"featureRank ops: {featureRank.get_ops()}")
        else:
            self.dfs_['FE_rank'] = None
            log("ignore featureRank")


        # image特征
        log("feature engineer: Image")
        if self.info_['image_info']:
            self.dfs_['FE_image'] = fe_ima2vec(df, self.info_['image_info']['image_path'],
                                               self.info_['image_info']['image_col'],
                                               self.info_['image_info']['filename_extension'])
        else:
            self.dfs_['FE_image'] = None
            log("ignore image feature")

        # auto_encoder
        df = auto_encoder(df, feature_type, id_)

        # 特征合并
        log("feature combination")
        df_list = [df, self.dfs_['FE_nlp'], self.dfs_['FE_count'], self.dfs_['FE_stat'], self.dfs_['FE_rank'], 
                   self.dfs_['FE_shift'], self.dfs_['FE_diff'], self.dfs_['FE_cumsum'], self.dfs_['FE_One2M'], 
                   self.dfs_['FE_image']]
        self.dfs_['FE_all'] = feature_combination(df_list)

        # # 内存优化
        # self.dfs_['FE_all'] = reduce_mem_usage(self.dfs_['FE_all'])

        # train和test数据切分
        train_length = self.info_['shape_of_train']
        self.train, self.test = train_test_divide(self.dfs_['FE_all'], train_length)
        log(f"shape of FE_all: {self.dfs_['FE_all'].shape}, shape of train: {self.train.shape}, shape of test: {self.test.shape}")

        # 特征过滤
        log("feature filter")
        # pass
        log(f"used_features: {self.used_features}")

        # 模型训练
        log("start training lightgbm model")
        # pass

        # 特征重要性
        fimp = self.model_lgb.feature_importances_
        log("feature importance")
        log(fimp)

        topk_feas = [x for x in list(fimp['feature']) if x not in df.columns][:topk]
        if return_df:
            return topk_feas, self.train[id_ + topk_feas], self.test[id_ + topk_feas]
        else:
            return topk_feas
