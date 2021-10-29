from autox import AutoX
import argparse
from time import time
import datetime

start_time = time()
ap = argparse.ArgumentParser(description='run_autox.py')
ap.add_argument('path_input', nargs=1, action="store", type=str)
ap.add_argument('path_output', nargs=1, action="store", type=str)
pa = ap.parse_args()
path_input = pa.path_input[0]
path_output = pa.path_output[0]
print("path_input: ", path_input)
print("path_output: ", path_output)

# 配置数据信息, 选择数据集
data_name = path_input.split('/')[-1]

if data_name == 'kaggle_tabular_aug_2021':
    autox = AutoX(target='loss', train_name='train.csv', test_name='test.csv',
                    id=['id'], path = path_input)
elif data_name == 'kaggle_house_price':
    autox = AutoX(target='SalePrice', train_name='train.csv', test_name='test.csv',
                    id=['Id'], path = path_input)
elif data_name == 'titanic':
    autox = AutoX(target='Survived', train_name='train.csv', test_name='test.csv',
                  id=['PassengerId'], path = path_input)
elif data_name == 'kaggle_ieee':
    relations = [
        {
            "related_to_main_table": "true",  # 是否为和主表的关系
            "left_entity": "train_transaction.csv",  # 左表名字
            "left_on": ["TransactionID"],  # 左表拼表键
            "right_entity": "train_identity.csv",  # 右表名字
            "right_on": ["TransactionID"],  # 右表拼表键
            "type": "1-1"  # 左表与右表的连接关系
        },
        {
            "related_to_main_table": "true",  # 是否为和主表的关系
            "left_entity": "test_transaction.csv",  # 左表名字
            "left_on": ["TransactionID"],  # 左表拼表键
            "right_entity": "test_identity.csv",  # 右表名字
            "right_on": ["TransactionID"],  # 右表拼表键
            "type": "1-1"  # 左表与右表的连接关系
        }
    ]
    autox = AutoX(target='isFraud', train_name='train_transaction.csv', test_name='test_transaction.csv',
                  id=['TransactionID'], path = path_input, relations=relations)
elif data_name == 'kaggle_springleaf':
    autox = AutoX(target='target', train_name='train.csv', test_name='test.csv',
                  id=['ID'], path = path_input)
elif data_name == 'stumbleupon':
    autox = AutoX(target = 'label', train_name = 'train.csv', test_name = 'test.csv',
                    id = ['urlid'], path = path_input)
elif data_name == 'santander':
    autox = AutoX(target='target', train_name='train.csv', test_name='test.csv',
                  id=['ID_code'], path = path_input)
elif data_name == 'ventilator':
    feature_type = {
        'train.csv': {
            'id': 'cat',
            'breath_id': 'cat',
            'R': 'num',
            'C': 'num',
            'time_step': 'num',
            'u_in': 'num',
            'u_out': 'num',
            'pressure': 'num'
        },
        'test.csv': {
            'id': 'cat',
            'breath_id': 'cat',
            'R': 'num',
            'C': 'num',
            'time_step': 'num',
            'u_in': 'num',
            'u_out': 'num'
        }
    }
    autox = AutoX(target='pressure', train_name='train.csv', test_name='test.csv',
                  id=['id'], path = path_input, feature_type=feature_type, metric='mae')
elif data_name == 'allstate_claims':
    autox = AutoX(target = 'loss', train_name = 'train.csv', test_name = 'test.csv',
                    id = ['id'], path = path_input, metric = 'mae')
elif data_name == 'RestaurantRevenue':
    autox = AutoX(target='revenue', train_name='train.csv', test_name='test.csv',
                  id=['Id'], path = path_input)

sub = autox.get_submit()

sub.to_csv(f"{path_output}/autox_{data_name}_oneclick.csv", index = False)

total_time = str(datetime.timedelta(seconds=time() - start_time))
with open(f"{path_output}/{data_name}_time.txt", "w") as text_file:
    text_file.write(total_time)