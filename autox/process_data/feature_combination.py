import pandas as pd
from tqdm import tqdm

def feature_combination(df_list):
    result = df_list[0]
    for df in tqdm(df_list[1:], total=len(df_list[1:])):
        if df is None or df.shape[0] == 0:
            continue
        assert(result.shape[0] == df.shape[0])
        result = pd.concat([result, df], axis=1)
    return result