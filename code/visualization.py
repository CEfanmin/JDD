import pandas as pd
from pandas import DataFrame
from math import exp
from pylab import *
import matplotlib.pyplot as plot

# heat map
merge_data = pd.read_csv('../data/userInfo_8-10.csv')
print(merge_data.shape)
corMat = DataFrame(merge_data.iloc[:, 1:22].corr())
plot.pcolor(corMat)
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

