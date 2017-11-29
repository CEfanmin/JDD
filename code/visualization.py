import pandas as pd
from pandas import DataFrame
from math import exp
import matplotlib.pyplot as plot
import scipy.stats as stats


merge_data = pd.read_csv('../result/t_loan_sum.csv')
loan_sum = merge_data['loan_sum'].tolist()
stats.probplot(loan_sum, dist="norm", plot=plot.subplot(221))
plot.title('t_loan_sum')

merge_data = pd.read_csv('../result/v10-prediction_XGB.csv')
loan_sum = merge_data['loan_sum'].tolist()
stats.probplot(loan_sum, dist="norm", plot=plot.subplot(222))
plot.title('v10-prediction_XGB')

merge_data = pd.read_csv('../result/addModel.csv')
loan_sum = merge_data['loan_sum'].tolist()
stats.probplot(loan_sum, dist="norm", plot=plot.subplot(223))
plot.title('addModel')

merge_data = pd.read_csv('../result/v3-prediction_ET.csv')
loan_sum = merge_data['loan_sum'].tolist()
stats.probplot(loan_sum, dist="norm", plot=plot.subplot(224))
plot.title('v3-prediction_ET')

# merge_data = pd.read_csv('../result/v2-prediction_StackingModel.csv')
# loan_sum = merge_data['loan_sum'].tolist()
# stats.probplot(loan_sum, dist="norm", plot=plot.subplot(325))
# plot.title('v2-prediction_StackingModel')
plot.show()


# heat map
# merge_data = pd.read_csv('../data/userInfo_8-10.csv')
# print(merge_data.shape)
# corMat = DataFrame(merge_data.iloc[:, 1:22].corr())
# plot.pcolor(corMat)
plot.show()

# parallel plot
# summary = merge_data.describe()
# print (summary)
# mean_loan_sum = summary.iloc[1, 7]
# std_loan_sum = summary.iloc[2, 7]
# nrows = len(merge_data.index)
# for i in range(1000):
# 	dataRow = merge_data.iloc[i, 1:10]
# 	normTarget = (merge_data.iloc[i, 10]-mean_loan_sum)/std_loan_sum
# 	labelColor = 1.0/(1.0+exp(-normTarget))
# 	dataRow.plot(color=plot.cm.RdYlBu(labelColor),alpha=0.5)
#
# plot.xlabel('Attribute Index')
# plot.ylabel('Attribute Value')
# plot.show()

# box plot
# merge_dataNormalized = merge_data.iloc[:, 1:11]
# for i in range(10):
#     mean = summary.iloc[1, i]
#     sd = summary.iloc[2, i]
#     merge_dataNormalized.iloc[:, i:(i + 1)] = (merge_dataNormalized.iloc[:, i:(i + 1)] - mean) / sd
#
# print(merge_dataNormalized)
# array3 = merge_dataNormalized.values
# boxplot(array3)
# plot.xlabel("Attribute Index")
# plot.ylabel("Quartile Ranges - Normalized ")
# plot.show()

