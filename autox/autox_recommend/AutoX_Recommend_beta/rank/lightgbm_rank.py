import lightgbm as lgb
import pandas as pd
import gc
from utils.logger import logger


class LightGBMRanker(object):
    def __init__(self, uid, iid, **kwargs):
        self.uid = uid
        self.iid = iid
        self.boosting_type = kwargs.get('boosting_type', 'gbdt')
        self.num_leaves = kwargs.get('num_leaves', 31)
        self.reg_alpha = kwargs.get('reg_alpha', 0.0)
        self.reg_lambda = kwargs.get('reg_lambda', 1)
        self.max_depth = kwargs.get('max_depth', -1)
        self.n_estimators = kwargs.get('n_estimators', 2000)
        self.subsample = kwargs.get('subsample', 0.7)
        self.colsample_bytree = kwargs.get('colsample_bytree', 0.7)
        self.subsample_freq = kwargs.get('subsample_freq', 1)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.min_child_weight = kwargs.get('min_child_weight', 50)
        self.random_state = kwargs.get('random_state', 2018)
        self.n_jobs = kwargs.get('n_jobs', -1)
        self.verbose = kwargs.get('verbose', 100)
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', 100)
        self.eval_at = kwargs.get('eval_at', [12])
        self.eval_metric = kwargs.get('eval_metric', ['map'])

        self.model = lgb.LGBMRanker(
            boosting_type=self.boosting_type, num_leaves=self.num_leaves,
            reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
            max_depth=self.max_depth, n_estimators=self.n_estimators,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            subsample_freq=self.subsample_freq, learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight, random_state=self.random_state,
            n_jobs=self.n_jobs
        )

    def fit(self, train_data, valid_data):
        train_data.sort_values(by=[self.uid], inplace=True)
        g_train = train_data.groupby([self.uid], as_index=False).count()["label"].values

        del_cols = []
        feats = [f for f in train_data.describe().columns if f not in [self.uid, self.iid, 'label'] + del_cols]

        if valid_data is not None:
            valid_data.sort_values(by=[self.uid], inplace=True)
            g_valid = valid_data.groupby([self.uid], as_index=False).count()["label"].values
            eval_set = [(valid_data[feats], g_valid['label'])]
        else:
            eval_set = [(train_data[feats], train_data['label'])]
            self.early_stopping_rounds = None

        self.model.fit(train_data[feats], group=g_train, eval_set=eval_set,
                       eval_metric=['map'],
                       early_stopping_rounds=self.early_stopping_rounds,
                       eval_at=self.eval_at,
                       eval_names=self.eval_metric,
                       verbose=self.verbose)

        self.best_iteration_ = self.model.best_iteration_

        logger.info('LightGBM ranker training done.')
        logger.info(self.model.best_score_)

        importance_df = pd.DataFrame()
        importance_df["feature"] = feats
        importance_df["importance"] = self.model.feature_importances_

        logger.info(importance_df.sort_values('Importance', ascending=False).head(20))

        if valid_data is not None:
            valid_data['prob'] = self.model.predict(valid_data[feats], num_iteration=self.model.best_iteration_)
            return valid_data[[self.uid, self.iid, 'prob']]
        else:
            return feats

    def predict(self, feats, data, all_user, batch_size=30000):
        logger.info('LightGBM ranker predicting...')

        batch_num = len(all_user) // batch_size + 1
        recs = []
        for i in range(batch_num):
            logger.info('[{}/{}]'.format(i + 1, batch_num))
            custs = all_user[i * batch_size: (i + 1) * batch_size]
            cur_test = data.loc[data[self.uid].isin(custs)]
            cur_test['prob'] = self.model.predict(cur_test[feats])
            rec = cur_test.sort_values('prob', ascending=False).groupby(self.uid)[self.iid].agg(
                list).reset_index()
            rec.columns = [self.uid, 'prediction']
            recs.append(rec)

            del rec
            gc.collect()

        recs = pd.concat(recs)

        return recs
