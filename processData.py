import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# drop out t_user active_time col
# user_data = pd.read_csv('./rawData/t_user.csv')
# new_user_data = user_data.drop('active_date', axis=1)
# new_user_data.to_csv('./dropData/new_user_data.csv')

# drop out t_loan_sum month col
# loan_sum_data = pd.read_csv('./t_loan_sum.csv')
# new_loan_sum_data = loan_sum_data.drop('month', axis=1)
# new_loan_sum_data.to_csv('./new_loan_sum_data.csv')

# statistics month click count
user_id = pd.read_csv('./monthData/user_id.csv')
click_data = pd.read_csv('./rawData/t_click.csv')
merge_data = pd.merge(user_id, click_data, how='left', on='uid')
merge_data['click_time'] = pd.to_datetime(merge_data['click_time'])
merge_data = merge_data.set_index('click_time')

i = 8
for month_data in [merge_data['2016-08'],merge_data['2016-09'],merge_data['2016-10'], merge_data['2016-11']]:
	month_data_change = month_data.groupby('uid', sort=False)\
		.agg({'pid': len})\
		.rename(columns={'pid': 'click_time_sum'+str(i)})
	print ('click_time' + str(i)+'.csv')
	month_data_change.to_csv('./click_time' + str(i)+'.csv')
	i = i + 1

click_time08 = pd.read_csv('./click_time8.csv')
pd.merge(user_id, click_time08, on='uid', how='left').fillna(0).to_csv('./monthData/t_click/click_time08.csv')

click_time09 = pd.read_csv('./click_time9.csv')
pd.merge(user_id, click_time09, on='uid', how='left').fillna(0).to_csv('./monthData/t_click/click_time09.csv')

click_time10 = pd.read_csv('./click_time10.csv')
pd.merge(user_id, click_time10, on='uid', how='left').fillna(0).to_csv('./monthData/t_click/click_time10.csv')

click_time11 = pd.read_csv('./click_time11.csv')
pd.merge(user_id, click_time11, on='uid', how='left').fillna(0).to_csv('./monthData/t_click/click_time11.csv')

# statistics month order count
order_data = pd.read_csv('./rawData/t_order.csv')
merge_data = pd.merge(user_id, order_data, how='left', on='uid')
merge_data['buy_time'] = pd.to_datetime(merge_data['buy_time'])
merge_data = merge_data.set_index('buy_time')

new_order_data = order_data.groupby('uid',sort=False)\
	.agg({'buy_time':len, 'price': lambda x:sum(x)})\
	.rename(columns={'buy_time':'order_count', 'price':'order_price_sum'})
new_order_data.to_csv('./new_order_data.csv')

i = 8
for month_data in [merge_data['2016-08'], merge_data['2016-09'], merge_data['2016-10'], merge_data['2016-11']]:
	month_data_change = month_data.groupby('uid', sort=False)\
		.agg({'qty': len, 'price': lambda x: sum(x)})\
		.rename(columns={'qty': 'order_count_sum'+str(i), 'price':'order_price_sum'+str(i)})
	print ('order_time' + str(i)+'.csv')
	month_data_change.to_csv('./order_time' + str(i)+'.csv')
	i = i + 1

order_time08 = pd.read_csv('./order_time8.csv')
pd.merge(user_id, order_time08, on='uid', how='left').fillna(0).to_csv('./monthData/t_order/order_time08.csv')

order_time09 = pd.read_csv('./order_time9.csv')
pd.merge(user_id, order_time09, on='uid', how='left').fillna(0).to_csv('./monthData/t_order/order_time09.csv')

order_time10 = pd.read_csv('./order_time10.csv')
pd.merge(user_id, order_time10, on='uid', how='left').fillna(0).to_csv('./monthData/t_order/order_time10.csv')

order_time11 = pd.read_csv('./order_time11.csv')
pd.merge(user_id, order_time11, on='uid', how='left').fillna(0).to_csv('./monthData/t_order/order_time11.csv')

# statistics month loan count
loan_data = pd.read_csv('./rawData/t_loan.csv')
merge_data = pd.merge(user_id, loan_data, how='left', on='uid')
merge_data['loan_time'] = pd.to_datetime(merge_data['loan_time'])
merge_data = merge_data.set_index('loan_time')
i = 8
for month_data in [merge_data['2016-08'], merge_data['2016-09'], merge_data['2016-10'], merge_data['2016-11']]:
	month_data_change = month_data.groupby('uid', sort=False)\
		.agg({'plannum': len})\
		.rename(columns={'plannum': 'loan_count_sum'+str(i)})
	print ('loan_time' + str(i)+'.csv')
	month_data_change.to_csv('./loan_time' + str(i)+'.csv')
	i = i + 1

loan_time08 = pd.read_csv('./loan_time8.csv')
pd.merge(user_id, loan_time08, on='uid', how='left').fillna(0).to_csv('./monthData/t_loan/loan_time08.csv')

loan_time09 = pd.read_csv('./loan_time9.csv')
pd.merge(user_id, loan_time09, on='uid', how='left').fillna(0).to_csv('./monthData/t_loan/loan_time09.csv')

loan_time10 = pd.read_csv('./loan_time10.csv')
pd.merge(user_id, loan_time10, on='uid', how='left').fillna(0).to_csv('./monthData/t_loan/loan_time10.csv')

loan_time11 = pd.read_csv('./loan_time11.csv')
pd.merge(user_id, loan_time11, on='uid', how='left').fillna(0).to_csv('./monthData/t_loan/loan_time11.csv')
