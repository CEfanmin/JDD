import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# predict = pd.read_csv('./v10-prediction_XGB.csv')
# predict[predict<1.5] = 0
# predict.to_csv('./v10-prediction_XGB.csv')

# predict = pd.read_csv('./predictionScore/v7-December_Loan_Forecasting.csv')
# predict[predict < 1] = 0
# predict.to_csv('./predictionScore/v7-December_Loan_Forecasting.csv')

# drop out t_user active_time col
# user_data = pd.read_csv('./rawData/t_user.csv')
# new_user_data = user_data.drop('active_date', axis=1)
# new_user_data.to_csv('./dropData/new_user_data.csv')

# drop out t_loan_sum month col
# loan_sum_data = pd.read_csv('./t_loan_sum.csv')
# new_loan_sum_data = loan_sum_data.drop('month', axis=1)
# new_loan_sum_data.to_csv('./new_loan_sum_data.csv')

# statistics month click count
# user_id = pd.read_csv('./monthData/user_id.csv')
# click_data = pd.read_csv('./rawData/t_click.csv')
# merge_data = pd.merge(user_id, click_data, how='left', on='uid')
# merge_data['click_time'] = pd.to_datetime(merge_data['click_time'])
# merge_data = merge_data.set_index('click_time')
#
# i = 8
# for month_data in [merge_data['2016-08'], merge_data['2016-09'], merge_data['2016-10'], merge_data['2016-11']]:
# 	month_data_change = month_data.groupby('uid', sort=False)\
# 		.agg({'param': len})\
# 		.rename(columns={'param': 'click_time_sum'+str(i)})
# 	print('click_time' + str(i)+'.csv')
# 	month_data_change.to_csv('./click_time' + str(i)+'.csv')
# 	i = i + 1
#
# click_time08 = pd.read_csv('./click_time8.csv')
# pd.merge(user_id, click_time08, on='uid', how='left').fillna(0).to_csv('./monthData/click_time08.csv')
#
# click_time09 = pd.read_csv('./click_time9.csv')
# pd.merge(user_id, click_time09, on='uid', how='left').fillna(0).to_csv('./monthData/t_click/click_time09.csv')
#
# click_time10 = pd.read_csv('./click_time10.csv')
# pd.merge(user_id, click_time10, on='uid', how='left').fillna(0).to_csv('./monthData/t_click/click_time10.csv')
#
# click_time11 = pd.read_csv('./click_time11.csv')
# pd.merge(user_id, click_time11, on='uid', how='left').fillna(0).to_csv('./monthData/t_click/click_time11.csv')

# # statistics month order count
# order_data = pd.read_csv('./rawData/t_order.csv')
# merge_data = pd.merge(user_id, order_data, how='left', on='uid')
# merge_data['buy_time'] = pd.to_datetime(merge_data['buy_time'])
# merge_data = merge_data.set_index('buy_time')

# i = 8
# for month_data in [merge_data['2016-08'], merge_data['2016-09'], merge_data['2016-10'], merge_data['2016-11']]:
# 	month_data_change = month_data.groupby('uid', sort=False)\
# 		.agg({'discount': len, 'price': lambda x: sum(x)})\
# 		.rename(columns={'discount': 'order_count_sum'+str(i), 'price': 'order_price_sum'+str(i)})
# 	print('order_time' + str(i)+'.csv')
# 	month_data_change.to_csv('./order_time' + str(i)+'.csv')
# 	i = i + 1

# order_time08 = pd.read_csv('./order_time8.csv')
# pd.merge(user_id, order_time08, on='uid', how='left').fillna(0).to_csv('./monthData/t_order/order_time08.csv')

# order_time09 = pd.read_csv('./order_time9.csv')
# pd.merge(user_id, order_time09, on='uid', how='left').fillna(0).to_csv('./monthData/t_order/order_time09.csv')
#
# order_time10 = pd.read_csv('./order_time10.csv')
# pd.merge(user_id, order_time10, on='uid', how='left').fillna(0).to_csv('./monthData/t_order/order_time10.csv')
#
# order_time11 = pd.read_csv('./order_time11.csv')
# pd.merge(user_id, order_time11, on='uid', how='left').fillna(0).to_csv('./monthData/t_order/order_time11.csv')
#
# # statistics month loan count
# loan_data = pd.read_csv('./rawData/t_loan.csv')
# merge_data = pd.merge(user_id, loan_data, how='left', on='uid')
# merge_data['loan_time'] = pd.to_datetime(merge_data['loan_time'])
# merge_data = merge_data.set_index('loan_time')
# i = 8
# for month_data in [merge_data['2016-08'], merge_data['2016-09'], merge_data['2016-10'], merge_data['2016-11']]:
# 	month_data_change = month_data.groupby('uid', sort=False)\
# 		.agg({'plannum': len})\
# 		.rename(columns={'plannum': 'loan_count_sum'+str(i)})
# 	print ('loan_time' + str(i)+'.csv')
# 	month_data_change.to_csv('./loan_time' + str(i)+'.csv')
# 	i = i + 1
#
# loan_time08 = pd.read_csv('./loan_time8.csv')
# pd.merge(user_id, loan_time08, on='uid', how='left').fillna(0).to_csv('./monthData/t_loan/loan_time08.csv')
#
# loan_time09 = pd.read_csv('./loan_time9.csv')
# pd.merge(user_id, loan_time09, on='uid', how='left').fillna(0).to_csv('./monthData/t_loan/loan_time09.csv')
#
# loan_time10 = pd.read_csv('./loan_time10.csv')
# pd.merge(user_id, loan_time10, on='uid', how='left').fillna(0).to_csv('./monthData/t_loan/loan_time10.csv')
#
# loan_time11 = pd.read_csv('./loan_time11.csv')
# pd.merge(user_id, loan_time11, on='uid', how='left').fillna(0).to_csv('./monthData/t_loan/loan_time11.csv')

# statistics consumer model
# raw_data = pd.read_csv('./rawSumFeatureData.csv')
# loan_price_sum = raw_data['loan_price_sum']
# order_price_sum = raw_data['order_price_sum']
# consumer_model = loan_price_sum/order_price_sum
# consumer_model.to_csv('./consumer_model.csv')
# print(consumer_model)
# loan_count_sum = raw_data['loan_count_sum']
# order_count_sum = raw_data['order_count_sum']
# consumer_model = loan_count_sum/order_count_sum
# consumer_model.to_csv('./count_model.csv')
# print(consumer_model)


# statistics 8-10 month data
# raw_data = pd.read_csv('../data/rawBigTable_fixed.csv')
# for title in ['loan_price_sum', 'loan_count_sum', 'plannum', 'click_count_sum', 'order_price_sum', 'order_count_sum']:
# 	raw_data[title] = raw_data[title+'8'] + raw_data[title +'9'] + raw_data[title +'10']
# 	# print(raw_data[title])
# 	raw_data[title].to_csv('../data/'+title+'8-10.csv')
#
# for title in ['loan_price_sum', 'loan_count_sum', 'plannum', 'click_count_sum', 'order_price_sum', 'order_count_sum']:
# 	raw_data[title] = raw_data[title+'9'] + raw_data[title +'10'] + raw_data[title +'11']
# 	# print(raw_data[title])
# 	raw_data[title].to_csv('../data/'+title+'9-11.csv')


merge_data1 = pd.read_csv('../result/v10-prediction_XGB.csv')
loan_sum1 = merge_data1['loan_sum']
merge_data2 = pd.read_csv('../result/v3-prediction_ET.csv')
loan_sum2 = merge_data2['loan_sum']

loan_sum = pd.DataFrame(0.5*loan_sum2 + 0.5*loan_sum1)
loan_sum[loan_sum <1] = 0
loan_sum.to_csv('../result/addModel.csv')
print(loan_sum)

