import pandas as pd
from tqdm import tqdm
from ..util import log
from sklearn.preprocessing import OrdinalEncoder

def auto_encoder(df, df_feature_type, id):
    df_copy = df.copy()
    label_encoder_list = []
    ordinal_encoder_list = []
    for f in tqdm(df_feature_type.keys()):
        if df_feature_type[f] == 'cat':
            label_encoder_list.append(f)
            temp = pd.DataFrame(df_copy[f].astype(str))
            temp.index = range(len(temp))
            temp[f] = temp[[f]].apply(lambda x: x.astype('category').cat.codes)
            if id is not None:
                if f in id:
                    df_copy[f + '_encoder'] = temp[f].values
            else:
                df_copy[f] = temp[f].values
        if df_feature_type[f] == 'ord':
            ordinal_encoder_list.append(f)
            ord_encoder = OrdinalEncoder()
            df_copy[f] = ord_encoder.fit_transform(pd.DataFrame(df_copy[f]))

    log(f"label_encoder_list: {label_encoder_list}")
    log(f"ordinal_encoder_list: {ordinal_encoder_list}")
    return df_copy