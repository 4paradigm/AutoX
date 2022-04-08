import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from autox.autox_competition.util import log

from sklearn.preprocessing import MinMaxScaler

def normalization(df, cols):
    for col in cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[col] = scaler.fit_transform(df[col].values.reshape((-1, 1)))
    return df

def feature_combination(df_list):
    log('[+] feature combination')
    result = df_list[0]
    for df in tqdm(df_list[1:], total=len(df_list[1:])):
        if df is None or df.shape[0] == 0:
            continue
        assert(result.shape[0] == df.shape[0])
        result = pd.concat([result, df], axis=1)
    return result


def construct_data(df, id_col, time_col, target_col,
                   time_interval_num, time_interval_unit, forecast_period,
                   new_target_col, add_time_col):

    log('[+] construct data')

    def get_t2(x):
        t1, k = x[0], x[1]
        if time_interval_unit == 'minute':
            delta = timedelta(minutes=k * time_interval_num)
        return t1 + delta

    SAMPLE_DATA = False
    SAMPLE_DATA_threshold = 500 * 10000
    if (df.shape[0] * forecast_period) > SAMPLE_DATA_threshold:
        SAMPLE_DATA = True
        Frac = SAMPLE_DATA_threshold / (df.shape[0] * forecast_period)

    train_test = pd.DataFrame()
    for k_step in tqdm(range(1, forecast_period + 1)):
        temp = df.copy()
        temp['k_step'] = k_step
        temp[add_time_col] = temp[[time_col, 'k_step']].apply(lambda x: get_t2(x), axis=1)
        temp[new_target_col] = temp.groupby(id_col)[target_col].shift(-k_step)
        if SAMPLE_DATA:
            log(f'[+] sample data, frac={Frac}')
            train_test = train_test.append(temp.sample(frac=Frac, random_state=1).copy())
        else:
            train_test = train_test.append(temp.copy())

    train = train_test.loc[train_test[new_target_col].notnull()]
    test = train_test.loc[train_test[new_target_col].isnull()]
    train.index = range(len(train))
    test.index = range(len(test))
    del train_test

    return train, test