import warnings
warnings.filterwarnings('ignore')
import numpy as np

# log
import logging
LOGGER = logging.getLogger('run-time-adaptive_automl')
LOG_LEVEL = 'INFO'
# LOG_LEVEL = 'DEBUG'
LOGGER.setLevel(getattr(logging, LOG_LEVEL))
simple_formatter = logging.Formatter('%(levelname)7s -> %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(simple_formatter)
LOGGER.addHandler(console_handler)
LOGGER.propagate = False
nesting_level = 0

def log(entry, level='info'):
    if level not in ['debug', 'info', 'warning', 'error']:
        LOGGER.error('Wrong level input')

    global nesting_level
    space = '-' * (4 * nesting_level)

    getattr(LOGGER, level)(f"{space} {entry}")


def weighted_mae_lgb(preds, train_data, weight=10.0):
    labels = train_data.get_label()

    masks_small = (preds - labels) < 0
    masks_big = (preds - labels) >= 0

    loss = np.mean(abs(preds - labels) * masks_small * weight + abs(preds - labels) * masks_big)
    return 'weighted_mae', loss, False

def weighted_mae_xgb(preds, train_data, weight=10.0):
    labels = train_data.get_label()

    masks_small = (preds - labels) < 0
    masks_big = (preds - labels) >= 0

    loss = np.mean(abs(preds - labels) * masks_small * weight + abs(preds - labels) * masks_big)
    return 'weighted_mae', loss

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    log('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    log('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    log('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df