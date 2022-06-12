import pandas as pd

from recall.binary_recall import BinaryNetRecall
from recall.history_recall import HistoryRecall
from recall.itemcf_recall import ItemCFRecall
from recall.popular_recall import PopularRecall
from recall.w2v_content_recall import W2VContentRecall
from utils.logger import logger

recall_models = {
    'Popular': PopularRecall,
    'History': HistoryRecall,
    'ItemCF': ItemCFRecall,
    'Binary': BinaryNetRecall,
    'Word2Vec': W2VContentRecall,
}

class RecallModel(object):
    def __init__(self, models, uid, iid, recall_num):
        self.uid = uid
        self.iid = iid
        self.recall_num = recall_num

        self.models = []
        logger.info("Start Initializing Recall Models.")
        for model in models:
            model_name = model['name']
            parameters = model.get('parameters', {})
            # TODO
            # 每个模型在fit阶段，都可以按照required_attrs进行判断，是否可以应用当前方法
            # 目前未加入
            parameters['required_attrs'] = model.get('required_attrs', [])
            self.models.append(recall_models[model_name](uid, iid, **parameters))
            logger.info(" * Initial Recall Model: {}".format(model_name))

        # TODO
        # 需要记录模型效果

    def fit(self, interactions, users, items, mode='train'):

        # TODO:
        # model 所需数据可能不止interactions
        # 需要每个model 设置属性，按属性输入数据
        # 目前model都只是用interactions，因此直接input interactions

        output_df = pd.DataFrame(data=None, columns=[self.uid, self.iid, 'label'])

        for model in self.models:
            tmp_output = model.fit(interactions, mode=mode)
            tmp_output.drop_duplicates(subset=[self.uid, self.iid, 'label'], keep='first', inplace=True)
            output_df.merge(tmp_output, on=[self.uid, self.iid, 'label'], how='outer')

        return output_df


    def predict(self, interactions, target_users):
        output_df = pd.DataFrame(data=None, columns=[self.uid, self.iid, 'label'])

        for model in self.models:
            tmp_output = model.predict(interactions, target_users)
            tmp_output.drop_duplicates(subset=[self.uid, self.iid, 'label'], keep='first', inplace=True)
            output_df.merge(tmp_output, on=[self.uid, self.iid, 'label'], how='outer')

        return output_df
