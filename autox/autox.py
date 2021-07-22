from .feature_engineer.fe_count import FeatureCount
from .feature_engineer.fe_stat import FeatureStat
from .file_io.read_data import read_data_from_path
from .models.regressor import CrossLgbRegression
from .process_data import feature_combination, train_test_divide, clip_label
from .process_data import feature_filter
from .process_data.feature_type_recognition import Feature_type_recognition
from .util import log

class AutoX():
    def __init__(self, target, train_name, test_name, path, feature_type = {}, id = []):
        self.info_ = {}
        self.info_['id'] = id
        self.info_['target'] = target
        self.info_['feature_type'] = feature_type
        self.info_['train_name'] = train_name
        self.info_['test_name'] = test_name
        self.dfs_ = read_data_from_path(path)
        self.info_['max_target'] = self.dfs_[train_name][target].max()
        self.info_['min_target'] = self.dfs_[train_name][target].min()
        self.concat_train_test()
        if feature_type == {}:
            for table_name in self.dfs_.keys():
                df = self.dfs_[table_name]
                feature_type_recognition = Feature_type_recognition()
                feature_type = feature_type_recognition.fit(df)
                self.info_['feature_type'][table_name] = feature_type

        self.dfs_['FE_all'] = None
        self.sub = None

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

        # 特征合并
        log("feature combination")
        df_list = [df, self.dfs_['FE_count'], self.dfs_['FE_stat']]
        self.dfs_['FE_all'] = feature_combination(df_list)

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
        crossLgbRegression = CrossLgbRegression()
        crossLgbRegression.fit(train[used_features], train[target], Early_Stopping_Rounds=100, N_round=4000, Verbose=50)

        # 特征重要性
        fimp = crossLgbRegression.feature_importances_
        log("feature importance")
        log(fimp)

        # 模型预测
        predict = crossLgbRegression.predict(test[used_features])

        # 预测结果后处理
        min_ = self.info_['min_target']
        max_ = self.info_['max_target']
        predict = clip_label(predict, min_, max_)

        # 获得结果
        sub = test[id_]
        sub[target] = predict
        sub.index = range(len(sub))

        return sub
