import numpy as np

def clip_label(pred, min_, max_):
    pred = np.clip(pred, min_, max_)
    return pred