from autox import AutoX

data_name = 'zhidemai'
path = f'./data/{data_name}'

feature_type = {
    'train.csv': {
        'article_id': 'cat',
         'date': 'num',
         'baike_id_1h': 'cat',
         'price': 'num',
         'price_diff': 'num',
         'author': 'cat',
         'level1': 'cat',
         'level2': 'cat',
         'level3': 'cat',
         'level4': 'cat',
         'brand': 'cat',
         'mall': 'cat',
         'url': 'cat',
         'comments_1h': 'num',
         'zhi_1h': 'num',
         'buzhi_1h': 'num',
         'favorite_1h': 'num',
         'orders_1h': 'num',
         'baike_id_2h': 'cat',
         'comments_2h': 'num',
         'zhi_2h': 'num',
         'buzhi_2h': 'num',
         'favorite_2h': 'num',
         'orders_2h': 'num',
         'orders_3h_15h': 'num'
    },
    'test.csv': {
        'article_id': 'cat',
         'date': 'num',
         'baike_id_1h': 'cat',
         'price': 'num',
         'price_diff': 'num',
         'author': 'cat',
         'level1': 'cat',
         'level2': 'cat',
         'level3': 'cat',
         'level4': 'cat',
         'brand': 'cat',
         'mall': 'cat',
         'url': 'cat',
         'comments_1h': 'num',
         'zhi_1h': 'num',
         'buzhi_1h': 'num',
         'favorite_1h': 'num',
         'orders_1h': 'num',
         'baike_id_2h': 'cat',
         'comments_2h': 'num',
         'zhi_2h': 'num',
         'buzhi_2h': 'num',
         'favorite_2h': 'num',
         'orders_2h': 'num'
    }
}

autox = AutoX(target ='orders_3h_15h', train_name ='train.csv', test_name ='test.csv',
               id = ['article_id'], path = path, feature_type = feature_type)
submit = autox.get_submit()
submit.to_csv("./submit.csv", index=False)