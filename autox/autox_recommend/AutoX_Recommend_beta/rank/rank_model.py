from datetime import datetime

from rank.lightgbm_rank import LightGBMRanker
from utils.logger import logger

rank_models = {
    'LightGBM': LightGBMRanker,
}


class RankModel():
    def __init__(self, models, uid, iid):
        self.uid = uid
        self.iid = iid
        self.models = []
        logger.info("Start Initializing Rank Models.")
        for model in models:
            model_name = model['name']
            parameters = model.get('parameters', {})
            self.models.append(rank_models[model_name](uid, iid, **parameters))
            logger.info(" * Initial Rank Model: {}".format(model_name))

    def fit(self, train_data, valid_date):
        # TODO
        # 考虑多个 Rank model，如何拼接
        # 目前只有一个model
        res = self.models[0].fit(train_data, valid_date)

        return res

    def predict(self, features, data, target_users):
        # TODO
        # 考虑多个 Rank model，如何拼接
        # 目前只有一个model
        res = self.models[0].predict(features, data, target_users)
        return res
