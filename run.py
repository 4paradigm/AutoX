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
data_name = 'allstate_claims'
path = f'{path_input}/{data_name}'

autox = AutoX(target = 'loss', train_name = 'train.csv', test_name = 'test.csv',
               id = ['id'], path = path, metric = 'mae')

sub = autox.get_submit()

sub.to_csv(f"{path_output}/autox_{data_name}_oneclick.csv", index = False)

total_time = str(datetime.timedelta(seconds=time() - start_time))
with open(f"{path_output}/{data_name}_time.txt", "w") as text_file:
    text_file.write(total_time)