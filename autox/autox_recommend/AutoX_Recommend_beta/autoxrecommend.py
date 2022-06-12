import datetime

from metrics.mapk import mapk
from recall.recall_model import RecallModel
from rank.rank_model import RankModel
from features.generator import FeatureGenerator
from utils.load_config import get_config
from utils.logger import logger


class AutoXRecommend():
    def __init__(self, config='config.json'):
        logger.info("############  AutoX Recommend   ##########")
        logger.info("Loading Config File {}".format(config))
        config = get_config(config)

        # TODO
        # 1. 为 User、Item、Interaction分别构建类，便于存储和传递属性
        # 2. config中支持的属性分为两类，（1）特定属性，可直接明确输入字段，如time
        #                            （2） 通用属性，没有特定操作，可进一步区分为离散型和连续型

        try:
            self.user_id = config['USERS']['userId']
            self.item_id = config['ITEMS']['itemId']
        except KeyError:
            logger.error('Config Error: Required Key Not Found')

        assert config['INTERACTIONS']['userId'] == self.user_id \
               and config['INTERACTIONS']['itemId'] == self.item_id, "Interaction Info Error"

        self.interaction_time = config['INTERACTIONS'].get('timestamp', None)
        self.user_features = config['USERS'].get('attrs', None)
        self.item_features = config['ITEMS'].get('attrs', None)
        self.interaction_features = config['INTERACTIONS'].get('attrs', None)

        self.recall_num = config['METHODS'].get('recallNum', 100)
        self.recall_model = RecallModel(config['METHODS']['RECALL'], self.user_id, self.item_id, self.recall_num)
        self.rank_model = RankModel(config['METHODS']['RANK'], self.user_id, self.item_id)

        self.feature_generator = FeatureGenerator(self.user_id, self.item_id, )


    def fit(self, inter_df, user_df, item_df):
        self.interactions_data = inter_df
        self.users_data = user_df
        self.items_data = item_df

        # Recall
        logger.info("############  Recall   ##########")
        train = self.recall_model.fit(inter_df, user_df, item_df, mode='train')
        valid = self.recall_model.fit(inter_df, user_df, item_df, mode='valid')


        # Feature Engineering
        logger.info("############  Feature Engineering   ##########")
        # train_fe = self.feature_generator.generate(train, inter_df, user_df, item_df, mode='train')
        # TODO
        # 完善feature engineer后需要修改
        train_fe = self.feature_generator.time_based_gen(
            train, inter_df, user_df, item_df, self.interaction_time, mode='train')
        valid_fe = self.feature_generator.time_based_gen(
            valid, inter_df, user_df, item_df, self.interaction_time, mode='valid')

        # Rank
        logger.info("############  Rank   ##########")

        # TODO
        # 目前方法和time_col强耦合，recall、rank和效果计算每个部分都受影响
        # 后续需要改变数据划分策略，划分数据后，再调用每个方法
        # 目前为了保持fit()流程的简洁，recall阶段将划分的部分嵌入到了各个方法中
        # 但rank阶段，目前方法本身不需要time_col数据，所以仍然在方法外计算了date，后续需要修改
        temp_date = datetime.datetime.strptime(str(inter_df[self.interaction_time].max()), '%Y-%m-%d %H:%M:%S') + \
                    datetime.timedelta(days=1)
        valid_date = str(datetime.datetime(temp_date.year, temp_date.month, temp_date.day))

        valid_pred = self.rank_model.fit(train_fe, valid_fe)


        ###############################################################################
        logger.info('Local result calculation')
        # 离线结果打印
        valid_pred = valid_pred.sort_values('prob', ascending=False)
        valid_pred = valid_pred.groupby(self.user_id).head(12).groupby(self.user_id)[self.item_id].agg(list).reset_index()

        begin_date = datetime.datetime.strptime(valid_date, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=7)
        begin_date = str(begin_date)

        valid_true = inter_df.loc[inter_df[self.user_id].isin(valid_pred[self.user_id])]
        valid_true = valid_true[(valid_true[self.interaction_time] <= valid_date) & (valid_true[self.interaction_time] > begin_date)]

        valid_true = valid_true.groupby(self.item_id)[self.item_id].agg(list).reset_index()

        logger.info("mAP Score on Validation set:", mapk(valid_true[self.item_id], valid_pred[self.item_id]))
        ###############################################################################

        # Retrain
        logger.info("############  Retrain   ##########")
        retrain_recall = self.recall_model.fit(inter_df, user_df, item_df, mode='valid')
        retrain_fe = self.feature_generator.time_based_gen(
            retrain_recall, inter_df, user_df, item_df, self.interaction_time, mode='valid')
        self.features = self.rank_model.fit(retrain_fe, None)

        logger.info('Fit Complete')



    def transform(self, target_users):
        recall_output = self.recall_model.predict(self.interactions_data, target_users)
        fe_output = self.feature_generator.time_based_gen(
            recall_output, self.interactions_data, self.users_data, self.items_data, self.interaction_time, mode='valid')
        rank_output = self.rank_model.predict(self.features, fe_output, target_users)
        return rank_output


if __name__ == '__main__':
    model = AutoXRecommend()
