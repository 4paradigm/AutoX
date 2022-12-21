import argparse
import pandas as pd
from autox import AutoX
import os

ap = argparse.ArgumentParser(description='run_autox.py')
ap.add_argument('--train_path', nargs=1, action="store", type=str)
ap.add_argument('--test_path', nargs=1, action="store", type=str)
ap.add_argument('--output_path', nargs=1, action="store", type=str)
ap.add_argument('--exp', nargs=1, action="store", type=str)
ap.add_argument('--id', nargs=1, action="store", type=str)
ap.add_argument('--target', nargs=1, action="store", type=str)

pa = ap.parse_args()
train_path = pa.train_path[0]
test_path = pa.test_path[0]
output_path = pa.output_path[0]
exp = pa.exp[0]
id = pa.id[0]
target = pa.target[0]

train_name = train_path.split('/')[-1]
test_name = test_path.split('/')[-1]

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

autox = AutoX(id = [id],
              target = target,
              train_name = train_name,
              test_name = test_name,
              path = None,
              dfs = {train_name: train, test_name: test})

sub = autox.get_submit()
output_file = os.path.join(output_path, f'{exp}.csv')
sub.to_csv(output_file, index=False)

# conda run -n myenv python script.py
# python run_autox.py \
#        --train_path /home/caihengxing/data/santander/train.csv \
#        --test_path /home/caihengxing/data/santander/test.csv \
#        --output_path /home/caihengxing/temp \
#        --exp santander_exp2 \
#        --id ID_code \
#        --target target