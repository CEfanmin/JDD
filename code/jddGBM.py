import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot
from sklearn.externals import joblib
import pandas as pd


raw_data = pd.read_csv('./rawSumFeatureData.csv')
all_data = raw_data.fillna(0)
labels = all_data['loan_sum']
xList = all_data.iloc[:, 1:14]
X = np.array(xList)
y = np.array(labels)

featureNames = np.array(['age', 'sex', 'limit', 'order_price_sum', 'order_count_sum',
                         'loan_price_sum', 'loan_count_sum', 'plannum_sum', 'click_count_sum',
                         'repayment_ability', 'active_ability', 'order_ability', 'loan_ability'])

# take fixed holdout set 10% of data rows
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.10, random_state=531)
nEst = 500
depth = None
learnRate = 0.01
subSamp = 0.5

loanGBMModel = ensemble.GradientBoostingRegressor(n_estimators=nEst, max_depth=depth, learning_rate=learnRate,
                                                  subsample=subSamp, loss='ls')

loanGBMModel.fit(xTrain, yTrain)
# compute mse on test set
msError = []
predictions = loanGBMModel.staged_predict(xTest)

# save model and load model
# joblib.dump(loanRFModel, 'D:\JDDModels\RFwithMode\clf.pkl')
# finalClasss = joblib.load('D:\JDDModels\RFwithModeMean\clf.pkl')
finalPrediction = loanGBMModel.predict(X)
data = pd.DataFrame(finalPrediction)
data.to_csv('./v7-prediction.csv')

for p in predictions:
    msError.append(mean_squared_error(yTest, p))

print("MSE")
print(min(msError))
print(msError.index(min(msError)))

# plot training and test errors vs number of trees in ensemble
plot.figure()
plot.plot(range(1, nEst+1), loanGBMModel.train_score_, label='Training Set MSE')
plot.plot(range(1, nEst + 1), msError, label='Test Set MSE')
plot.legend(loc='upper right')
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
plot.show()

# Plot feature importance
featureImportance = loanGBMModel.feature_importances_
# normalize by max importance
featureImportance = featureImportance / featureImportance.max()
idxSorted = np.argsort(featureImportance)
barPos = np.arange(idxSorted.shape[0]) + .5
plot.barh(barPos, featureImportance[idxSorted], align='center')
plot.yticks(barPos, featureNames[idxSorted])
plot.xlabel('Variable Importance')
plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plot.show()

