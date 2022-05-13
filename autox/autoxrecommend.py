import datetime
import pandas as pd
from tqdm import tqdm

from autox.autox_recommend.recalls import PopularRecall
from autox.autox_recommend.recall_and_rank import RecallAndRank

class AutoXRecommend():
    def __init__(self):
        pass

    def fit(self, inter_df, user_df, item_df,
                  uid, iid, time_col,
                  recall_num, mode='recall_and_rank', recall_method=None,
                  debug=False, debug_save_path=None):

        assert mode in ['recalls', 'recall_and_rank']

        if mode == 'recalls':
            assert recall_method in ['popular', 'history', 'itemcf', 'binary']

            if recall_method == 'popular':
                self.recommend = PopularRecall()
                self.recommend.fit(inter_df=inter_df, user_df=user_df, item_df=item_df,
                                   uid=uid, iid=iid, time_col=time_col,
                                   recall_num=recall_num)

        elif mode == 'recall_and_rank':
            self.recommend = RecallAndRank()
            self.recommend.fit(inter_df=inter_df, user_df=user_df, item_df=item_df,
                               uid=uid, iid=iid, time_col=time_col,
                               recall_num=recall_num,
                               debug=debug, debug_save_path=debug_save_path)


    def transform(self, uids):

        result = self.recommend.transform(uids)

        return result
