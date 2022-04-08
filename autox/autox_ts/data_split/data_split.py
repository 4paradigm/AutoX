import numpy as np
from tqdm import tqdm

def get_train_valid(df_all, id_col, time_varying_cols, target_col, used_cols, forecast_period):
    n_steps_in, n_steps_out = forecast_period, forecast_period

    X_train, X_train_other, y_train = list(), list(), list()
    X_valid, X_valid_other, y_valid = list(), list(), list()
    for cur_id in tqdm(df_all[id_col].unique()):
        cur_df = df_all.loc[df_all[id_col] == cur_id].copy()
        cur_df.index = range(len(cur_df))
        cur_X, cur_X_other, cur_y = split_sequences(cur_df, n_steps_in, n_steps_out, time_varying_cols,
                                                    target_col, used_cols)
        X_train.extend(cur_X[:-n_steps_out])
        X_train_other.extend(cur_X_other[:-n_steps_out])
        y_train.extend(cur_y[:-n_steps_out])

        X_valid.extend(cur_X[-n_steps_out:])
        X_valid_other.extend(cur_X_other[-n_steps_out:])
        y_valid.extend(cur_y[-n_steps_out:])

    X_train = np.array(X_train).astype('float32')
    X_train_other = np.array(X_train_other).astype('float32')
    y_train = np.array(y_train).astype('float32')

    X_valid = np.array(X_valid).astype('float32')
    X_valid_other = np.array(X_valid_other).astype('float32')
    y_valid = np.array(y_valid).astype('float32')

    return [X_train, X_train_other, y_train], [X_valid, X_valid_other, y_valid]

def split_sequences(df, n_steps_in, n_steps_out, time_varying_cols, target_col, used_cols):
    sequences = df[[x for x in time_varying_cols if x != target_col] + [target_col]].values

    X, X_other, y = list(), list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, -1]
        X.append(seq_x)

        # X_other的信息对应特征的最后一个时刻
        X_other.append(df.loc[end_ix - 1, used_cols].values)

        y.append(seq_y)

    return X, X_other, y


# split a multivariate sequence into samples
def split_sequences_test(df, n_steps_in, n_steps_out, time_varying_cols, target_col, used_cols):
    sequences = df[[x for x in time_varying_cols if x != target_col] + [target_col]].values

    X, X_other = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        if out_end_ix <= len(sequences):
            continue
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :]
        X.append(seq_x)

        # X_other的信息对应特征的最后一个时刻
        X_other.append(df.loc[end_ix - 1, used_cols].values)

    return X, X_other