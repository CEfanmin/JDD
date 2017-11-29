import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# feature1: sum
'''t_loan'''
# raw_loan_data = pd.read_csv('./rawData/t_loan.csv')
# new_loan_data = raw_loan_data.groupby('uid', sort=False)\
# 	.agg({'loan_time': len, 'loan_amount':lambda x: sum(x), 'plannum': lambda y: sum(y)})\
# 	.rename(columns={'loan_time': 'loan_count_sum', 'loan_amount': 'loan_price_sum', 'plannum': 'plannum_sum'})
# new_loan_data.to_csv('./featureData/new_loan_data.csv')
#
# user_data = pd.read_csv('./user_id.csv')
# loan_data = pd.read_csv('./featureData/new_loan_data.csv')
# merge_loan_data = pd.merge(user_data, loan_data, how='left', on='uid')
# merge_loan_data.to_csv('./featureData/merge_loan_data.csv')
'''t_click'''
# raw_loan_data = pd.read_csv('./rawData/t_click.csv')
# new_loan_data = raw_loan_data.groupby('uid', sort=False)\
# 	.agg({'click_time': len,})\
# 	.rename(columns={'click_time': 'click_count_sum'})
# new_loan_data.to_csv('./featureData/new_click_data.csv')
#
# user_data = pd.read_csv('./user_id.csv')
# loan_data = pd.read_csv('./featureData/new_click_data.csv')
# merge_loan_data = pd.merge(user_data, loan_data, how='left', on='uid')
# merge_loan_data.to_csv('./featureData/merge_click_data.csv')
'''t_order'''
# raw_loan_data = pd.read_csv('./rawData/t_order.csv')
# new_loan_data = raw_loan_data.groupby('uid', sort=False)\
# 	.agg({'buy_time': len, 'price':lambda x: sum(x), })\
# 	.rename(columns={'buy_time': 'order_count_sum', 'price': 'order_price_sum'})
# new_loan_data.to_csv('./featureData/new_order_data.csv')
#
# user_data = pd.read_csv('./user_id.csv')
# loan_data = pd.read_csv('./featureData/new_order_data.csv')
# merge_loan_data = pd.merge(user_data, loan_data, how='left', on='uid')
# merge_loan_data.to_csv('./featureData/merge_order_data.csv')

'''user ability'''
raw_data = pd.read_csv('../data/userInfoSum_9-11.csv')
order_ability = pd.Series(raw_data['order_price_sum'])/pd.Series(raw_data['order_count_sum'])
loan_ability = pd.Series(raw_data['loan_price_sum'])/pd.Series(raw_data['loan_count_sum'])
repayment_ability = pd.Series(raw_data['loan_price_sum'])/pd.Series(raw_data['plannum_sum'])
active_ability = pd.Series(raw_data['click_count_sum'])/3.0
consumer_model = pd.Series(raw_data['loan_price_sum'])/pd.Series(raw_data['order_price_sum'])
click_model = pd.Series(raw_data['loan_count_sum'])/pd.Series(raw_data['order_count_sum'])

order_ability.to_csv('../data/ability/order_ability.csv')
loan_ability.to_csv('../data/ability/loan_ability.csv')
repayment_ability.to_csv('../data/ability/repayment_ability.csv')
active_ability.to_csv('../data/ability/active_ability.csv')
consumer_model.to_csv('../data/ability/consumer_model.csv')
click_model.to_csv('../data/ability/click_model.csv')


# process Nan value with mean
# merge_data = pd.read_csv('./v8-prediction.csv')
# merge_data = pd.DataFrame(merge_data).clip(0,)
# print (merge_data.head())

# merge_data = merge_data.iloc[:, 1:10].fillna(merge_data.mode().iloc[0])
# print (merge_data.head(100))
# merge_data.to_csv('./merge_data_mode.csv')

# analysis click_count
# fig, axes = plt.subplots(nrows=2, ncols=1)
# merge_data['score'].hist(bins=100, ax=axes[0])
# merge_data['score'] = np.log1p(merge_data['score'])
# # merge_data['click_count'].to_csv('./logData/log_click_count.csv')
# merge_data['score'].hist(bins=100, ax=axes[1])
# plt.title('score')
# plt.show()
# merge_data = pd.read_csv('./rawSumFeatureData.csv')
# fig, axes = plt.subplots(nrows=2, ncols=1)
# merge_data['loan_count_sum'].hist(bins=100, ax=axes[0])
# merge_data['loan_count_sum'] = np.log1p(merge_data['loan_count_sum'])
# merge_data['loan_count_sum'].to_csv('./logData/log_loan_count_sum.csv')
# merge_data['loan_count_sum'].hist(bins=100, ax=axes[1])
# plt.title('loan_count_sum')
# plt.show()

# analysis order_price_sum
# fig, axes = plt.subplots(nrows=2, ncols=1)
# merge_data['order_price_sum'].hist(bins=100, ax=axes[0])
# merge_data['order_price_sum'] = np.log1p(merge_data['order_price_sum'])
# # merge_data['order_price_sum'].to_csv('./logData/log_order_price_sum.csv')
# merge_data['order_price_sum'].hist(bins=100, ax=axes[1])
# plt.title('order_price_sum')
# plt.show()

# analysis order_count
# fig, axes = plt.subplots(nrows=2, ncols=1)
# merge_data['order_count'].hist(bins=100, ax=axes[0])
# merge_data['order_count'] = np.log1p(merge_data['order_count'])
# # merge_data['loan_price_sum'].to_csv('./logData/log_loan_price_sum.csv')
# merge_data['order_count'].hist(bins=100, ax=axes[1])
# plt.title('order_count')
# plt.show()

# analysis loan_price_sum
# fig, axes = plt.subplots(nrows=2, ncols=1)
# merge_data['loan_price_sum11'].hist(bins=100, ax=axes[0])
# merge_data['loan_price_sum11'] = np.log1p(merge_data['loan_price_sum11'])
# merge_data['loan_price_sum11'].to_csv('./logData/log_loan_price_sum11.csv')
# merge_data['loan_price_sum11'].hist(bins=100, ax=axes[1])
# plt.title('loan_price_sum11')
# plt.show()

# analysis plannum
# fig, axes = plt.subplots(nrows=2, ncols=1)
# merge_data['plannum11'].hist(bins=100, ax=axes[0])
# merge_data['plannum11'] = np.log1p(merge_data['plannum11'])
# merge_data['plannum11'].to_csv('./logData/plannum11.csv')
# merge_data['plannum11'].hist(bins=100, ax=axes[1])
# plt.title('plannum11')
# plt.show()

# analysis loan_sum
# fig, axes = plt.subplots(nrows=2, ncols=1)
# merge_data['loan_sum'].hist(bins=100, ax=axes[0])
# merge_data['loan_sum'] = np.log1p(merge_data['loan_sum'])
# # merge_data['loan_price_sum'].to_csv('./logData/log_loan_price_sum.csv')
# merge_data['loan_sum'].hist(bins=100, ax=axes[1])
# plt.title('loan_sum')
# plt.show()


# fill loan_sum nan in merge_data_mode with RF
# merge_data = pd.read_csv('./merge_data_mode.csv')
# loan_sum_no_nan = merge_data.dropna(axis=0, how='any').to_csv('./loan_sum_no_nan.csv')
# print(loan_sum_no_nan.shape)
# loan_sum_with_nan = merge_data[merge_data.isnull().any(axis=1)].to_csv('./loan_sum_with_nan.csv')

# merge loan_sum_no_nan and loan_sum_with_loan
# loan_sum_no_nan = pd.read_csv('./loan_sum_no_nan.csv')
# loan_sum_with_nan = pd.read_csv('./loan_sum_with_nan.csv')
# user_id = pd.read_csv('./user_id.csv')
# merge_data_mode_pre = loan_sum_with_nan.append(loan_sum_no_nan)
# merge_data_mode_pre.to_csv('./merge_data_mode_pre.csv')
# print (merge_data_mode_pre)

# merge_data_mode_pre = pd.merge(user_id, merge_data_mode_pre, how='left', on='uid')
# merge_data_mode_pre.to_csv('./merge_data_mode_pre1.csv')


