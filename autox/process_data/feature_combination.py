import pandas as pd

def feature_combination(df_list):
    result = df_list[0]
    for df in df_list[1:]:
        if df.shape[0] == 0:
            continue
        assert(result.shape[0] == df.shape[0])
        result = pd.concat([result, df], axis=1)
    return result