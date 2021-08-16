import pandas as pd
from tqdm import tqdm
from ..util import log

def auto_label_encoder(df, df_feature_type=None,silence_cols=[]):
    df_copy = df.copy()
    label_encoder_list = []
    if df_feature_type is not None:
        for f in tqdm(df_feature_type.keys()):
            if f in silence_cols:
                continue
            if df_feature_type[f] == 'cat':
                label_encoder_list.append(f)
                temp = pd.DataFrame(df_copy[f].astype(str))
                temp.index = range(len(temp))
                temp[f] = temp[[f]].apply(lambda x: x.astype('category').cat.codes)
                df_copy[f] = temp[f].values
    else:
        for f in tqdm(df_copy.columns):
            if silence_cols is not None and f in silence_cols:
                continue
            if 'O' == df[f].dtype:
                label_encoder_list.append(f)
                temp = pd.DataFrame(df_copy[f].astype(str))
                temp.index = range(len(temp))
                temp[f] = temp[[f]].apply(lambda x: x.astype('category').cat.codes)
                df_copy[f] = temp[f].values
    log(f"label_encoder_list: {label_encoder_list}")
    return df_copy