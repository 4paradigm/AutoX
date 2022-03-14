from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

def get_pseudo_label(train, test, id_, target, used_cols, p = 0.99):
    assert 0.5 < p < 1
    sub = test[[id_]]
    sub[target] = 0

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in tqdm(skf.split(train[used_cols], train[target]), total=skf.n_splits):
        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
        clf.fit(train.loc[train_index][used_cols], train.loc[train_index][target])

        pred = clf.predict_proba(test[used_cols])[:,1]
        sub[target] = sub[target] + pred / skf.n_splits

    pseudo_test = sub[(sub[target] <= (1-p)) | (sub[target] >= p)].copy()
    pseudo_test.loc[pseudo_test[target] >= 0.5, target] = 1
    pseudo_test.loc[pseudo_test[target] < 0.5, target] = 0
    pseudo_test.index = range(len(pseudo_test))
    pseudo_test = pseudo_test.merge(test, on=id_, how='left')
    pseudo_test = pseudo_test[train.columns]
    return  pseudo_test