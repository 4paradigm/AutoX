from autox.autox_server.ensemble import ensemble
from autox.autox_server.feature_engineer import fe_count, fe_onehot, fe_shift, fe_time_diff
from autox.autox_server.feature_engineer import fe_kv, fe_stat_for_same_prefix, fe_frequency
from autox.autox_server.feature_engineer import fe_time_count, fe_window_count, fe_time_rolling_count
from autox.autox_server.feature_engineer import fe_window2, fe_txt
from autox.autox_server.join_table import join_table
from autox.autox_server.model import lgb_with_fe, lgb_for_feature_selection
from autox.autox_server.model import model_util
from autox.autox_server.pre_process import process_1, process_2, process_3
from autox.autox_server.read_data import read_data
from autox.autox_server.util import log, load_obj
from autox.autox_server.util import merge_table, save_obj

class AutoXServer():
    def __init__(self, is_train, server_name, data_info_path=None, train_set_path=None):
        if is_train:
            assert(data_info_path is not None and train_set_path is not None)
        else:
            assert (data_info_path is None and train_set_path is None)

        self.is_train = is_train
        self.data_info_path = data_info_path
        self.train_set_path = train_set_path
        self.server_name = server_name

    def fit(self):

        data_name = self.server_name
        log("data name: {}".format(data_name))
        lgb_para_dict_1 = model_util.lgb_para_dict_1
        lgb_para_dict_2 = model_util.lgb_para_dict_2
        params_1 = model_util.params_1
        params_2 = model_util.params_2

        self.G_hist = {}
        self.G_hist['val_auc'] = {}
        self.G_hist['predict'] = {}
        self.G_hist['delete_column'] = {}

        phase = 'train'
        log("*** phase: {}".format(phase))
        is_train = True if phase == 'train' else False
        self.G_df_dict, self.G_data_info, remain_time = read_data.read_data(data_info_path=self.data_info_path,
                                                                            train_set_path=self.train_set_path, is_train=is_train, debug=False)


        remain_time = process_1.preprocess(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)
        remain_time = join_table.join_simple_tables(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)
        remain_time = process_2.preprocess_2(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)


        remain_time = join_table.join_indirect_1_to_M_tables(self.G_df_dict, self.G_data_info, self.G_hist, is_train=is_train, remain_time=remain_time)
        remain_time = join_table.preprocess_after_join_indirect_tables(self.G_df_dict, self.G_data_info, self.G_hist, is_train=is_train, remain_time=remain_time)
        remain_time = join_table.join_1_to_M_tables(self.G_df_dict, self.G_data_info, self.G_hist, is_train=is_train, remain_time=remain_time)
        remain_time = process_3.preprocess_3(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)


        remain_time = fe_kv.fe_kv(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_stat_for_same_prefix.fe_stat_for_same_prefix(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_frequency.fe_frequency(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_count.fe_count(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_shift.fe_shift(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_time_diff.fe_time_diff(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_time_count.fe_time_count(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_window_count.fe_window_count(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_time_rolling_count.fe_time_rolling_count(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_window2.fe_window2(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_onehot.fe_onehot(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)
        remain_time = fe_txt.fe_txt(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)


        remain_time = merge_table(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)

        exp_name = 'feature_selection'
        remain_time = lgb_for_feature_selection.lgb_for_feature_selection(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, params_1, lgb_para_dict_1, data_name, exp_name)
        exp_name_1 = 'fe_lgb'
        remain_time = lgb_with_fe.lgb_with_fe(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, params_1, lgb_para_dict_1, data_name, exp_name_1)
        exp_name_2 = 'fe_lgb_2'
        remain_time = lgb_with_fe.lgb_with_fe(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, params_2, lgb_para_dict_2, data_name, exp_name_2)

        _ = ensemble.ensemble(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, top_k=2)

    def predict(self, df=None, test_set_path=None):
        assert ((df is None and test_set_path is not None) or (df is not None and test_set_path is None))

        data_name = self.server_name
        lgb_para_dict_1 = model_util.lgb_para_dict_1
        lgb_para_dict_2 = model_util.lgb_para_dict_2
        params_1 = model_util.params_1
        params_2 = model_util.params_2

        phase = 'test'
        log("*** phase: {}".format(phase))
        remain_time = 1e10
        is_train = True if phase == 'train' else False
        self.G_df_dict, self.G_data_info, remain_time = read_data.read_data(data_info=self.G_data_info, test_set_path=test_set_path, df_dict=self.G_df_dict,
                                                                            is_train=is_train, debug=False, remain_time=remain_time)


        remain_time = process_1.preprocess(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)
        remain_time = join_table.join_simple_tables(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)
        remain_time = process_2.preprocess_2(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)


        remain_time = join_table.join_indirect_1_to_M_tables(self.G_df_dict, self.G_data_info, self.G_hist, is_train=is_train, remain_time=remain_time)
        remain_time = join_table.preprocess_after_join_indirect_tables(self.G_df_dict, self.G_data_info, self.G_hist, is_train=is_train, remain_time=remain_time)
        remain_time = join_table.join_1_to_M_tables(self.G_df_dict, self.G_data_info, self.G_hist, is_train=is_train, remain_time=remain_time)
        remain_time = process_3.preprocess_3(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)


        remain_time = fe_kv.fe_kv(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_stat_for_same_prefix.fe_stat_for_same_prefix(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_frequency.fe_frequency(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_count.fe_count(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_shift.fe_shift(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_time_diff.fe_time_diff(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_time_count.fe_time_count(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_window_count.fe_window_count(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_time_rolling_count.fe_time_rolling_count(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_window2.fe_window2(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, False)
        remain_time = fe_onehot.fe_onehot(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)
        remain_time = fe_txt.fe_txt(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)


        remain_time = merge_table(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time)

        exp_name = 'feature_selection'
        remain_time = lgb_for_feature_selection.lgb_for_feature_selection(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, params_1, lgb_para_dict_1, data_name, exp_name)
        exp_name_1 = 'fe_lgb'
        remain_time = lgb_with_fe.lgb_with_fe(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, params_1, lgb_para_dict_1, data_name, exp_name_1)
        exp_name_2 = 'fe_lgb_2'
        remain_time = lgb_with_fe.lgb_with_fe(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, params_2, lgb_para_dict_2, data_name, exp_name_2)

        _ = ensemble.ensemble(self.G_df_dict, self.G_data_info, self.G_hist, is_train, remain_time, top_k=2)

        sub = self.G_hist['predict']['ensemble']
        sub.index = range(len(sub))
        return sub

    def save_server(self, path):
        data_name = self.server_name
        save_obj(self.G_df_dict, path + f'/{data_name}_G_df_dict.pkl')
        save_obj(self.G_data_info, path + f'/{data_name}_G_data_info.pkl')
        save_obj(self.G_hist, path + f'/{data_name}_G_hist.pkl')

    def load_server(self, path):

        data_name = self.server_name
        self.G_df_dict = load_obj(path + f'/{data_name}_G_df_dict.pkl')
        self.G_data_info = load_obj(path + f'/{data_name}_G_data_info.pkl')
        self.G_hist = load_obj(path + f'/{data_name}_G_hist.pkl')

