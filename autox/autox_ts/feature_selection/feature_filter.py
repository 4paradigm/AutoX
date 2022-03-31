from tqdm import tqdm
from autox.autox_competition.util import log

def feature_filter(train, test, time_col, target_col):
    log('[+] feature_filter')
    not_used = []

    # nunique为1
    # train的最小值比test的最大值大 or train的最大值比test的最小值小
    for col in tqdm(test.columns):

        col_dtype = str(test[col].dtype)
        if not col_dtype.startswith('int') and not col_dtype.startswith('float'):
            not_used.append(col)
        elif train[col].nunique() <= 1:
            not_used.append(col)
        elif train[col].min() > test[col].max() or train[col].max() < test[col].min():
            not_used.append(col)

    not_used = list(set(not_used + [time_col, target_col]))
    print(f'not_used: {not_used}')

    used_features = [x for x in test.columns if x not in not_used]
    return used_features