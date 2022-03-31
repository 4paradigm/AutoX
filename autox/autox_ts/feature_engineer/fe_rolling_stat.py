from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from autox.autox_competition.util import log

def fe_rolling_stat(df, id_col, time_col, time_varying_cols, window_size):
    log('[+] fe_rolling_stat')
    result = df[[id_col, time_col]]
    df = df.sort_values(by = time_col)
    add_feas = []
    key = id_col
    for cur_ws in tqdm(window_size):
        for val in time_varying_cols:
            for op in ['mean', 'std', 'median', 'max', 'min', 'kurt', 'skew']:
                name = f'{key}__{val}__{cur_ws}__{op}'
                add_feas.append(name)
                if op == 'mean':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).mean())
                if op == 'std':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).std())
                if op == 'median':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).median())
                if op == 'max':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).max())
                if op == 'min':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).min())
                if op == 'kurt':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).kurt())
                if op == 'skew':
                    df[name] = df.groupby(key)[val].transform(
                        lambda x: x.rolling(window=cur_ws).skew())
    return result.merge(df[[id_col, time_col] + add_feas], on = [id_col, time_col], how = 'left')[add_feas]