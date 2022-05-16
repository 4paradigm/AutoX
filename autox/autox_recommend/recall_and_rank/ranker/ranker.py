import lightgbm as lgb
import pandas as pd
import gc

def ranker(train, valid, uid, iid, time_col):
    train.sort_values(by=[uid], inplace=True)
    g_train = train.groupby([uid], as_index=False).count()["label"].values

    valid.sort_values(by=[uid], inplace=True)
    g_val = valid.groupby([uid], as_index=False).count()["label"].values

    del_cols = []
    feats = [f for f in train.describe().columns if f not in [uid, iid, 'label'] + del_cols]

    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=2000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=-1)

    lgb_ranker.fit(train[feats], train['label'], group=g_train,
                   eval_set=[(valid[feats], valid['label'])],
                   eval_group=[g_val], eval_at=[12], eval_metric=['map'],
                   early_stopping_rounds=100,
                   verbose=100,
                   )

    print(lgb_ranker.best_score_)

    importance_df = pd.DataFrame()
    importance_df["feature"] = feats
    importance_df["importance"] = lgb_ranker.feature_importances_

    print(importance_df.sort_values('importance', ascending=False).head(20))

    valid['prob'] = lgb_ranker.predict(valid[feats], num_iteration=lgb_ranker.best_iteration_)

    return lgb_ranker, valid[[uid, iid, 'prob']]


def ranker_test(train, epoch, uid, iid, time_col):
    train.sort_values(by=[uid], inplace=True)

    g_train = train.groupby([uid], as_index=False).count()["label"].values

    del_cols = []
    feats = [f for f in train.describe().columns if f not in [uid, iid, 'label'] + del_cols]

    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=epoch, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=-1)

    lgb_ranker.fit(train[feats], train['label'], group=g_train,
                   eval_set=[(train[feats], train['label'])],
                   eval_group=[g_train], eval_at=[12], eval_metric=['map', ],
                   verbose=100
                   )

    print(lgb_ranker.best_score_)

    importance_df = pd.DataFrame()
    importance_df["feature"] = feats
    importance_df["importance"] = lgb_ranker.feature_importances_

    del train
    gc.collect()

    print(importance_df.sort_values('importance', ascending=False).head(20))

    return lgb_ranker, feats


def inference(model, feats, test, all_user,
              uid, iid, time_col,
              batch_size=30000):
    batch_num = len(all_user) // batch_size + 1

    recs = []

    for i in range(batch_num):
        print('[{}/{}]'.format(i + 1, batch_num))

        custs = all_user[i * batch_size: (i + 1) * batch_size]

        cur_test = test.loc[test[uid].isin(custs)]

        print(cur_test.shape)

        cur_test['prob'] = model.predict(cur_test[feats])

        rec = cur_test.sort_values('prob', ascending=False).groupby(uid)[iid].agg(
            list).reset_index()

        rec.columns = [uid, 'prediction']

        recs.append(rec)

        del rec

        gc.collect()

    recs = pd.concat(recs)

    return recs