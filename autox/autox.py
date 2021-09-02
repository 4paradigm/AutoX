from .feature_engineer.fe_count import FeatureCount
from .feature_engineer.fe_stat import FeatureStat
from .file_io.read_data import read_data_from_path
from .models.regressor import CrossLgbRegression, CrossXgbRegression, CrossTabnetRegression
from .models.classifier import CrossLgbBiClassifier, CrossXgbBiClassifier, CrossTabnetBiClassifier
from .process_data import feature_combination, train_test_divide, clip_label
from .process_data import feature_filter, auto_label_encoder
from .process_data.feature_type_recognition import Feature_type_recognition
from .util import log, reduce_mem_usage

class AutoX():
    def __init__(self, target, train_name, test_name, path, feature_type = {}, relations = [], id = [], data_type = 'regression', Debug = False):
        self.Debug = Debug
        self.data_type = data_type
        self.info_ = {}
        self.info_['id'] = id
        self.info_['target'] = target
        self.info_['feature_type'] = feature_type
        self.info_['relations'] = relations
        self.info_['train_name'] = train_name
        self.info_['test_name'] = test_name
        self.dfs_ = read_data_from_path(path)
        if Debug:
            log("Debug mode, sample data")
            self.dfs_[train_name] = self.dfs_[train_name].sample(5000)
        self.info_['max_target'] = self.dfs_[train_name][target].max()
        self.info_['min_target'] = self.dfs_[train_name][target].min()
        if feature_type == {}:
            for table_name in self.dfs_.keys():
                df = self.dfs_[table_name]
                feature_type_recognition = Feature_type_recognition()
                feature_type = feature_type_recognition.fit(df)
                self.info_['feature_type'][table_name] = feature_type
        self.join_simple_tables()
        self.concat_train_test()

        self.dfs_['FE_all'] = None
        self.sub = None

        # 识别任务类型
        if self.dfs_[self.info_['train_name']][self.info_['target']].nunique() == 2:
            self.info_['task_type'] = 'binary'
        else:
            self.info_['task_type'] = 'regression'

    def join_simple_tables(self):
        simple_relations = [x for x in self.info_['relations'] if x['type'] == '1-1' and x['related_to_main_table'] == 'true']
        for relation in simple_relations:
            left_table_name = relation['left_entity']
            right_table_name = relation['right_entity']
            left_on = relation['left_on']
            right_on = relation['right_on']
            if right_table_name in [self.info_['train_name'], self.info_['test_name']]:
                left_table_name, right_table_name = right_table_name, left_table_name
                left_on, right_on = right_on, left_on

            skip_name = right_on
            merge_table_name = right_table_name
            merge_table = self.dfs_[merge_table_name].copy()

            self.dfs_[left_table_name] = self.dfs_[left_table_name].merge(merge_table, left_on=left_on,
                                                                            right_on=right_on, how='left')
            del merge_table
            for key_ in self.info_['feature_type'][merge_table_name]:
                if key_ not in self.info_['feature_type'][left_table_name] and key_ not in skip_name:
                    self.info_['feature_type'][left_table_name][key_] = self.info_['feature_type'][merge_table_name][key_]

    def concat_train_test(self):
        self.info_['shape_of_train'] = len(self.dfs_[self.info_['train_name']])
        self.info_['shape_of_test'] = len(self.dfs_[self.info_['test_name']])
        self.dfs_['train_test'] = self.dfs_[self.info_['train_name']].append(self.dfs_[self.info_['test_name']])
        self.dfs_['train_test'].index = range(len(self.dfs_['train_test']))

        feature_type_train_test = {}
        for col in self.dfs_['train_test'].columns:
            if col in self.info_['feature_type'][self.info_['train_name']]:
                feature_type_train_test[col] = self.info_['feature_type'][self.info_['train_name']][col]
            else:
                feature_type_train_test[col] = self.info_['feature_type'][self.info_['test_name']][col]
        self.info_['feature_type']['train_test'] = feature_type_train_test

    def split_train_test(self):
        self.dfs_['FE_train'] = self.dfs_['FE_all'][:self.info_['shape_of_train']]
        self.dfs_['FE_test'] = self.dfs_['FE_all'][self.info_['shape_of_train']:]

    def get_submit(self):

        id_ = self.info_['id']
        target = self.info_['target']

        # 特征工程
        log("start feature engineer")
        df = self.dfs_['train_test']
        feature_type = self.info_['feature_type']['train_test']

        # 统计特征
        log("feature engineer: Stat")
        featureStat = FeatureStat()
        featureStat.fit(df, df_feature_type=feature_type, silence_group_cols= id_ + [target],
                        silence_agg_cols= id_ + [target], select_all=False)
        self.dfs_['FE_stat'] = featureStat.transform(df)
        log(f"featureStat ops: {featureStat.get_ops()}")

        # count特征
        log("feature engineer: Count")
        featureCount = FeatureCount()
        featureCount.fit(df, degree=2, df_feature_type=feature_type, silence_cols= id_ + [target], select_all=False)
        self.dfs_['FE_count'] = featureCount.transform(df)
        log(f"featureCount ops: {featureCount.get_ops()}")

        # label_encoder
        df = auto_label_encoder(df, feature_type, silence_cols = id_ + [target])

        # 特征合并
        log("feature combination")
        df_list = [df, self.dfs_['FE_count'], self.dfs_['FE_stat']]
        self.dfs_['FE_all'] = feature_combination(df_list)

        # # 内存优化
        # self.dfs_['FE_all'] = reduce_mem_usage(self.dfs_['FE_all'])

        # train和test数据切分
        train_length = self.info_['shape_of_train']
        train, test = train_test_divide(self.dfs_['FE_all'], train_length)
        log(f"shape of FE_all: {self.dfs_['FE_all'].shape}, shape of train: {train.shape}, shape of test: {test.shape}")

        # 特征过滤
        log("feature filter")
        used_features = feature_filter(train, test, id_, target)
        log(f"used_features: {used_features}")

        # 模型训练
        log("start training model")
        if self.data_type == 'regression':
            model_lgb = CrossLgbRegression()
            model_lgb.fit(train[used_features], train[target], tuning=True, Debug=self.Debug)

            model_xgb = CrossXgbRegression()
            model_xgb.fit(train[used_features], train[target], tuning=True, Debug=self.Debug)

            # model_tabnet = CrossTabnetRegression()
            # model_tabnet.fit(train[used_features], train[target], tuning=True, Debug=self.Debug)

        elif self.data_type == 'binary':
            model_lgb = CrossLgbBiClassifier()
            model_lgb.fit(train[used_features], train[target], tuning=True, Debug=self.Debug)

            model_xgb = CrossXgbBiClassifier()
            model_xgb.fit(train[used_features], train[target], tuning=True, Debug=self.Debug)

            # model_tabnet = CrossTabnetBiClassifier()
            # model_tabnet.fit(train[used_features], train[target], tuning=True, Debug=self.Debug)

        # 特征重要性
        fimp = model_lgb.feature_importances_
        log("feature importance")
        log(fimp)

        # 模型预测
        predict_lgb = model_lgb.predict(test[used_features])
        predict_xgb = model_xgb.predict(test[used_features])
        # predict_tabnet = model_tabnet.predict(test[used_features])
        predict = (predict_xgb + predict_lgb) / 2

        # 预测结果后处理
        min_ = self.info_['min_target']
        max_ = self.info_['max_target']
        predict = clip_label(predict, min_, max_)

        # 获得结果
        sub = test[id_]
        sub[target] = predict
        sub.index = range(len(sub))

        return sub
