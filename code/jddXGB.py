import numpy as np
import pandas as pd
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor

raw_data = pd.read_csv('./userInfoTime_8-10.csv')
all_data = raw_data.fillna(0)
labels = all_data['loan_sum']
xList = all_data.iloc[:, 1:24]
X = np.array(xList)
y = np.array(labels)

training_features, testing_features, training_target, testing_target = train_test_split(X, y,test_size=0.1, random_state=42)
print("training_features len is: ", len(training_features))
print("testing_target len is: ", len(testing_target))
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=1, min_child_weight=13, n_estimators=100, nthread=1, subsample=0.15)),
    ZeroCount(),
    ExtraTreesRegressor(bootstrap=True, max_features=0.45, min_samples_leaf=15, min_samples_split=18, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)

# validation rmse
results = exported_pipeline.predict(testing_features)
test_mse = mean_squared_error(results, testing_target)
test_rmse = math.sqrt(test_mse)
print("validation rmse score is: ", test_rmse)

# prediction rmse
finalPrediction = exported_pipeline.predict(X)
total_mse = mean_squared_error(finalPrediction,y)
total_rmse = math.sqrt(total_mse)
print("total rmse score is: ", total_rmse)

# prediction
predData = pd.read_csv('./userInfoTime_9-11.csv')
xList = predData.iloc[:, 1:24]
X = np.array(xList)
print(len(X))
finalPrediction = exported_pipeline.predict(X)
data = pd.DataFrame(finalPrediction, index=np.arange(1, len(X) + 1))
data[data < 1] = 0
data.to_csv('./prediction_xgboost.csv')


