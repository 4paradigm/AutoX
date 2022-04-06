import numpy as np

def SMAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    score = np.where(np.isnan(score), 0, score)
    return score

def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    idx = y_true > 0
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    score = np.abs(y_pred - y_true) / np.abs(y_true)
    return np.mean(score)

def _get_score_metric(y_true, y_pred, metric='mape'):
    '''
    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs). Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs). Estimated target values.
    :param metric: str, one of ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle', 'smape'], default = 'mape'.
    :return: metric.
    '''
    y_true = np.array(y_true) if type(y_true) == list else y_true
    y_pred = np.array(y_pred) if type(y_true) == list else y_pred

    if metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    if metric == 'mape':
        return MAPE(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'rmse':
        return np.mean((y_true - y_pred) ** 2) ** 0.5
    elif metric == 'msle':
        return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
    elif metric == 'rmsle':
        return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2) ** 0.5
    elif metric == 'smape':
        return np.mean(SMAPE(y_true, y_pred))
    return (y_true == y_pred).sum()