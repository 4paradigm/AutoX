import pandas as pd
from autox.autox_competition.util import log

def fe_onehot(df, cols):
    log('[+] fe_onehot')
    result = pd.DataFrame()
    for cur_col in cols:
        df_temp = pd.get_dummies(df[cur_col], prefix = cur_col)
        result = pd.concat([result, df_temp], axis = 1)
    return result