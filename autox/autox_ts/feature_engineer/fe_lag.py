from autox.autox_competition.util import log

def fe_lag(df, id_col, time_col, time_varying_cols, lag):
    log('[+] fe_lag')
    result = df[[id_col, time_col]].copy()
    df = df.sort_values(by = time_col)
    add_feas = []
    key = id_col
    for value in time_varying_cols:
        for cur_lag in lag:
            name = f'{key}__{value}__lag__{cur_lag}'
            add_feas.append(name)
            df[name] = df.groupby(key)[value].shift(cur_lag)
    return result.merge(df[[id_col, time_col] + add_feas], on = [id_col, time_col], how = 'left')[add_feas]