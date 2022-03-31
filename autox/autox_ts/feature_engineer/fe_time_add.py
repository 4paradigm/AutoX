import pandas as pd
from autox.autox_competition.util import log

def fe_time_add(df, time_col):
    log('[+] fe_time_add')
    prefix = time_col + "_"

    df[time_col] = pd.to_datetime(df[time_col])

    df[prefix + 'year'] = df[time_col].dt.year
    df[prefix + 'month'] = df[time_col].dt.month
    df[prefix + 'day'] = df[time_col].dt.day
    df[prefix + 'hour'] = df[time_col].dt.hour
    df[prefix + 'weekofyear'] = df[time_col].dt.weekofyear
    df[prefix + 'dayofweek'] = df[time_col].dt.dayofweek
    df[prefix + 'is_wknd'] = df[time_col].dt.dayofweek // 5
    df[prefix + 'quarter'] = df[time_col].dt.quarter
    df[prefix + 'is_month_start'] = df[time_col].dt.is_month_start.astype(int)
    df[prefix + 'is_month_end'] = df[time_col].dt.is_month_end.astype(int)