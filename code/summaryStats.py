import numpy as np
import sys,pylab
import pandas as pd
import scipy.stats as stats

# generate summary statistics for some column
merge_data = pd.read_csv('./merge_data_mean.csv')
colData = merge_data['loan_sum']
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +"Standard Deviation = " + '\t ' + str(colsd) + "\n")


# calculate quantile boundaries
ntile = 4
percentBdry = []
for i in range(ntile+1):
	percentBdry.append(np.percentile(colArray, i*(100)/ ntile))
print('Boundaries for 4 Equal Percentiles', percentBdry)

# run again with 10 equal intervals
ntile = 10
percentBdry = []
for i in range(ntile+1):
	percentBdry.append(np.percentile(colArray, i*(100)/ ntile))
print('Boundaries for 10 Equal Percentiles', percentBdry)

# plot quantile-quantile
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()

