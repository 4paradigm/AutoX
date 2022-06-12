from autoxrecommend import AutoXRecommend
import pandas as pd


path = '../data/datasets/HM/'
print('Loading data...')

inter_df = pd.read_csv(path + 'transactions_train.csv', dtype={'article_id': str})
user_df = pd.read_csv(path + 'customers.csv')
item_df = pd.read_csv(path + 'articles.csv', dtype={'article_id': str})
sample_submission = pd.read_csv(path + 'sample_submission.csv')


uid = 'customer_id'
iid = 'article_id'
time_col = 't_dat'
recall_num = 50

inter_df[time_col] = pd.to_datetime(inter_df[time_col])

autoXRecommend = AutoXRecommend(config='../config.json')

autoXRecommend.fit(inter_df = inter_df, user_df = user_df, item_df = item_df)

test_users = list(sample_submission[uid])
res = autoXRecommend.transform(test_users)

res['prediction'] = res['prediction'].apply(lambda x: ' '.join(x))

res.to_csv('./AutoX_hm.csv', index = False)