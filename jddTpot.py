import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from tpot.builtins import StackingEstimator
import pylab as plot
from sklearn.metrics import mean_squared_error


# load data
all_data = pd.read_csv('./merge_data_log_all.csv')
labels = all_data['loan_sum']
xList = all_data.iloc[:, 1:18]
X = np.array(xList)
y = np.array(labels)

featureNames = np.array(['age', 'sex', 'limit', 'order_count', 'order_price_sum',
                         'click_count', 'plannum', 'loan_price_sum', 'loan_count',
                         'loan_price_sum8','plannum8','loan_price_sum9','plannum9',
                         'loan_price_sum10','plannum10','loan_price_sum11','plannum11'])

xTrain, xTest, yTrain, yTest= train_test_split(X, y, random_state=42)

# Score on the training set was: 0.106589360722
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    MaxAbsScaler(),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="huber", max_depth=6, max_features=1.0, min_samples_leaf=20, min_samples_split=14, n_estimators=100, subsample=0.75)
)

exported_pipeline.fit(xTrain, yTrain)


# compute mse on test set
msError = []
predictions = exported_pipeline.predict(xTest)
for p in predictions:
	msError.append(mean_squared_error(yTest, p))

print("MSE")
print(min(msError))
print(msError.index(min(msError)))

# # Plot feature importance
# featureImportance = GradientBoostingRegressor.feature_importances_
# # normalize by max importance
# featureImportance = featureImportance / featureImportance.max()
# idxSorted = np.argsort(featureImportance)
# barPos = np.arange(idxSorted.shape[0]) + .5
# plot.barh(barPos, featureImportance[idxSorted], align='center')
# plot.yticks(barPos, featureNames[idxSorted])
# plot.xlabel('Variable Importance')
# plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
# plot.show()
