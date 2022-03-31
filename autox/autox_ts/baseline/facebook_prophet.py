# import pandas as pd
# from fbprophet import Prophet
# from tqdm import tqdm
#
# def prophet_predict(df, id_col, time_col, target_col, time_interval_num, time_interval_unit, forecast_period):
#     result = pd.DataFrame()
#     for cur_TurbID in tqdm(df[id_col].unique(), total=df[id_col].nunique()):
#         Prophet_df = df.loc[df[id_col] == cur_TurbID, [time_col, target_col]]
#         Prophet_df.columns = ['ds', 'y']
#         Prophet_df.index = range(len(Prophet_df))
#         Prophet_df['ds'] = pd.to_datetime(Prophet_df['ds'])
#
#         m = Prophet()
#         m.fit(Prophet_df)
#
#         freq = str(time_interval_num) + time_interval_unit
#         future = m.make_future_dataframe(periods=forecast_period, freq=freq)
#         forecast = m.predict(future)
#
#         cur_forecast = forecast.loc[forecast['ds'] > df[time_col].max(), ['ds', 'yhat']]
#         cur_forecast[id_col] = cur_TurbID
#         result = result.append(cur_forecast)
#         result.index = range(len(result))
#
#     result.rename({'ds': time_col, 'yhat': target_col}, axis=1, inplace=True)
#     return result[[id_col, time_col, target_col]]