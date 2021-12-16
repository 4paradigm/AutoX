import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from ..util import log, reduce_mem_usage

def read_data_from_path(path, train_name, target, target_map, file_type='csv'):
    G_df_dict = {}
    files = os.listdir(path)
    files = [x for x in files if x.endswith('.' + file_type)]
    for item in files:
        log('[+] read {}'.format(item))
        df = pd.read_csv(os.path.join(path, item))

        if item == train_name and len(target_map) > 0:
            df[target] = df[target].map(target_map)

        # 内存优化
        df = reduce_mem_usage(df)
        log('table = {}, shape = {}'.format(item, df.shape))
        name = item
        G_df_dict[name] = df
    return G_df_dict