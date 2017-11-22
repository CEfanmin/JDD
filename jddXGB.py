import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor

raw_data = pd.read_csv('./rawSumFeatureData.csv')
all_data = raw_data.fillna(0)
labels = all_data['loan_sum']
xList = all_data.iloc[:, 1:14]
X = np.array(xList)
y = np.array(labels)

training_features, testing_features, training_target, testing_target = \
            train_test_split(X, y, random_state=42)

# Score on the training set was:-2.25928065111
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=1, min_child_weight=13, n_estimators=100, nthread=1, subsample=0.15)),
    ZeroCount(),
    ExtraTreesRegressor(bootstrap=True, max_features=0.45, min_samples_leaf=15, min_samples_split=18, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
finalPrediction = exported_pipeline.predict(X)
data = pd.DataFrame(finalPrediction)
data[data < 1] = 0
data.to_csv('./prediction.csv')


