from autox.autox_ts.feature_engineer import fe_rolling_stat
from autox.autox_ts.feature_engineer import fe_lag
from autox.autox_ts.feature_engineer import fe_diff
from autox.autox_ts.feature_engineer import fe_time
from autox.autox_ts.feature_engineer import fe_time_add
from autox.autox_ts.feature_selection import feature_filter
from autox.autox_ts.models import ts_lgb_model
from autox.autox_ts.util import feature_combination
from autox.autox_ts.util import construct_data
# from autox.autox_ts.baseline import prophet_predict
from autox.autox_competition.util import log
from autox.autox_competition.process_data import clip_label

class AutoTS():
    def __init__(self,
                df,
                id_col,
                time_col,
                target_col,
                time_varying_cols,
                time_interval_num,
                time_interval_unit,
                forecast_period,
                mode='auto',
                metric='rmse'):

        assert (mode in ['auto', 'prophet'])

        self.df = df
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.time_varying_cols = time_varying_cols
        self.time_interval_num = time_interval_num
        self.time_interval_unit = time_interval_unit
        self.forecast_period = forecast_period
        self.mode = mode
        self.metric = metric

    def get_result(self):
        if self.mode == 'auto':
            sub = self.kdata_lgb()
        # elif self.mode == 'prophet':
        #     sub = self.baseline_prophet()

        log('[+] post process')
        sub[self.target_col] = clip_label(sub[self.target_col], self.df[self.target_col].min(), self.df[self.target_col].max())

        return sub

    # def baseline_prophet(self):
    #     sub = prophet_predict(df=self.df,
    #                           id_col=self.id_col,
    #                           time_col=self.time_col,
    #                           target_col=self.target_col,
    #                           time_interval_num=self.time_interval_num,
    #                           time_interval_unit=self.time_interval_unit,
    #                           forecast_period=self.forecast_period)
    #     return sub

    def kdata_lgb(self):

        log('[+] feature engineer')

        # rolling 窗口特征
        df_rolling_stat = fe_rolling_stat(self.df,
                                          id_col=self.id_col,
                                          time_col=self.time_col,
                                          time_varying_cols=self.time_varying_cols,
                                          window_size=[4, 16, 64, 256])
        # lag 特征
        df_lag = fe_lag(self.df,
                        id_col=self.id_col,
                        time_col=self.time_col,
                        time_varying_cols=self.time_varying_cols,
                        lag=[1, 2, 3])

        # diff特征
        df_diff = fe_diff(self.df,
                          id_col=self.id_col,
                          time_col=self.time_col,
                          time_varying_cols=self.time_varying_cols,
                          lag=[1, 2, 3])

        # 时间特征
        df_time = fe_time(self.df, time_col=self.time_col)

        # 合并所有特征
        df_all = feature_combination([self.df, df_rolling_stat, df_lag, df_diff, df_time])

        # 构造数据
        new_target_col = 'y'
        add_time_col = 't2'
        train, test = construct_data(df_all,
                                     id_col=self.id_col,
                                     time_col=self.time_col,
                                     target_col=self.target_col,
                                     time_interval_num=self.time_interval_num,
                                     time_interval_unit=self.time_interval_unit,
                                     forecast_period=self.forecast_period,
                                     new_target_col=new_target_col,
                                     add_time_col=add_time_col)

        # 补充特征
        fe_time_add(train, add_time_col)
        fe_time_add(test, add_time_col)

        # 特征选择
        used_features = feature_filter(train, test, self.time_col, target_col=new_target_col)
        category_cols = [self.id_col, 'k_step']

        log('[+] train model')
        # 模型
        sub, feature_importances = \
            ts_lgb_model(train, test,
                         id_col=self.id_col,
                         time_col=add_time_col,
                         target_col=new_target_col,
                         used_features=used_features,
                         category_cols=category_cols,
                         time_interval_num=self.time_interval_num,
                         time_interval_unit=self.time_interval_unit,
                         forecast_period=self.forecast_period,
                         label_log=False,
                         metric=self.metric)
        self.feature_importances = feature_importances

        return sub
