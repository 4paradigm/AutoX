import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log


def fe_shift(G_df_dict, G_data_info, G_hist, is_train, remain_time, AMPERE):
    # 对G_df_dict['BIG']表做扩展特征

    start = time.time()
    log('[+] feature engineer, shift')

    Id = G_data_info['target_id']
    target = G_data_info['target_label']
    big_size = G_df_dict['BIG'].shape[0]

    time_col = None
    if 'target_time' in G_data_info.keys() and G_data_info['target_time'] != '':
        time_col = G_data_info['target_time']
        G_df_dict['BIG'] = G_df_dict['BIG'].sort_values(by=time_col)

    if is_train:
        G_hist['FE_shift'] = {}
        G_hist['FE_shift_window'] = []

        if time_col:
            shift_id_potential = []
            for key_ in G_hist['big_cols_cat']:
                if 100 < G_df_dict['BIG'][key_].nunique() < big_size * 0.8:
                    shift_id_potential.append(key_)
            G_hist['FE_shift'] = shift_id_potential

            log("shift features: {}".format(shift_id_potential))

        shift_window = 3
        if G_data_info['time_series_data'] == 'true':
            window = range(1, shift_window + 1)
        else:
            window = [x for x in range(-shift_window, shift_window + 1) if x != 0]
        G_hist['FE_shift_window'] = window
        log("shift window: {}".format(window))

    if not AMPERE:

        G_df_dict['FE_shift'] = pd.DataFrame()
        for shift_id in G_hist['FE_shift']:
            data_shift = G_df_dict['BIG'][[shift_id, time_col, target]].copy()

            for i in G_hist['FE_shift_window']:
                data_shift[target + "(t-{})".format(i)] = data_shift.groupby(shift_id)[target].shift(i)
            data_shift.drop(target, axis=1, inplace=True)

            for i in G_hist['FE_shift_window']:
                G_df_dict['FE_shift'][target + "_shift_t-" + str(i) + "_with_" + shift_id] = data_shift[
                    target + '(t-' + str(i) + ')']

    end = time.time()
    remain_time -= (end - start)
    log("time consumption: {}".format(str(end - start)))
    log("remain_time: {} s".format(remain_time))
    return remain_time