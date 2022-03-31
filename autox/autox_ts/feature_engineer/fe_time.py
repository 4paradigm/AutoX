import pandas as pd
from autox.autox_competition.util import log

def fe_time(df, time_col):
    log('[+] fe_time')
    result = pd.DataFrame()
    prefix = time_col + "_"

    df[time_col] = pd.to_datetime(df[time_col])

    result[prefix + 'year'] = df[time_col].dt.year
    result[prefix + 'month'] = df[time_col].dt.month
    result[prefix + 'day'] = df[time_col].dt.day
    result[prefix + 'hour'] = df[time_col].dt.hour
    result[prefix + 'weekofyear'] = df[time_col].dt.weekofyear
    result[prefix + 'dayofweek'] = df[time_col].dt.dayofweek
    result[prefix + 'is_wknd'] = df[time_col].dt.dayofweek // 5
    result[prefix + 'quarter'] = df[time_col].dt.quarter
    result[prefix + 'is_month_start'] = df[time_col].dt.is_month_start.astype(int)
    result[prefix + 'is_month_end'] = df[time_col].dt.is_month_end.astype(int)

    return result