from tqdm import tqdm
from ..util import log

def feature_filter(train, test, id_, target):
    not_used = id_ + [target]

    # 过滤掉test中全为nan的特征
    for col in tqdm(test.columns):
        # test中全为Nan的特征
        if test.loc[test[col].isnull()].shape[0] == test.shape[0]:
            if col not in not_used:
                not_used += [col]

        # test中的值都比train中的值要大(或小)的特征
        if test[col].min() > train[col].max() or test[col].max() < train[col].min():
            if col not in not_used:
                not_used += [col]
    log(f"filtered features: {not_used}")
    used_features = [x for x in list(train.describe().columns) if x not in not_used]
    return used_features