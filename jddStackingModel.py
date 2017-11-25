import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor


# stacking model class
class Ensemble(object):
	def __init__(self, n_folds, stacker, base_models):
		self.n_folds = n_folds
		self.stacker = stacker
		self.base_models = base_models

	def fit_predict(self, X, y, T):  # TrainData:X  TrainData:y  TestData: T
		X = np.array(X)
		y = np.array(y)
		T = np.array(T)
		folds = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=2017).split(X))
		S_train = np.zeros((X.shape[0], len(self.base_models)))
		S_test = np.zeros((T.shape[0], len(self.base_models)))
		for i, clf in enumerate(self.base_models):
			S_test_i = np.zeros((T.shape[0], len(folds)))
			for j, (train_idx, test_idx) in enumerate(folds):
				X_train = X[train_idx]
				y_train = y[train_idx]
				X_holdout = X[test_idx]
				# y_holdout = y[test_idx]
				clf.fit(X_train, y_train)
				y_pred = clf.predict(X_holdout)[:]
				S_train[test_idx, i] = y_pred
				S_test_i[:, j] = clf.predict(T)[:]
			S_test[:, i] = S_test_i.mean(1)

		self.stacker.fit(S_train, y)
		y_pred = self.stacker.predict(S_test)[:]
		return y_pred


# class to extend sklearn classifier
class SklearnHelper(object):
	def __init__(self, clf, seed=0, params=None):
		self.clf = clf(**params)

	def train(self, x_train, y_train):
		self.clf.fit(x_train, y_train)

	def predict(self, x_test):
		return self.clf.predict(x_test)

	def fit(self, x, y):
		return self.clf.fit(x, y)

	def feature_importances(self, x, y):
		print(self.clf.fit(x, y).feature_importances_)

# load data and split date to test data and training data
raw_data = pd.read_csv('./rawSumFeatureData.csv')
all_data = raw_data.fillna(0)
labels = all_data['loan_sum']
xList = all_data.iloc[:, 1:14]
X = np.array(xList)
y = np.array(labels)
mseOos=[]
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.10, random_state=42)

# load base model
# Random Forest parameters
rf_params = {
	'n_jobs': 3,
	'oob_score': True,
	'max_depth': None,
	'max_features': 4,
	'n_estimators': 2000,
	'random_state': 531
}
# Gradient Boosting parameters
gb_params = {
	'n_estimators': 500,
	'subsample': 0.5,
	'learning_rate': 0.01,
	'loss': 'ls'
}
# XGB parameters
XGB_params = {
	'learning_rate': 0.1,
	'max_depth': 1,
	'min_child_weight': 13,
	'n_estimators': 100,
	'nthread': 1,
	'subsample':0.15
}
# Extra Trees parameters
et_params = {
	'bootstrap': True,
	'max_features': 0.45,
	'min_samples_leaf': 15,
	'min_samples_split': 18,
	'n_estimators': 100
}

rf = SklearnHelper(clf=ensemble.RandomForestRegressor, params=rf_params)
gb = SklearnHelper(clf=ensemble.GradientBoostingRegressor, params=gb_params)
XGB = SklearnHelper(clf=XGBRegressor, params=XGB_params)
et = SklearnHelper(clf=ensemble.ExtraTreesRegressor, params=et_params)

base_models = [rf, gb, XGB, et]
stackerModel = Ensemble(n_folds=4, stacker=XGB, base_models=base_models)
prediction = stackerModel.fit_predict(xTrain, yTrain, xTest)
mseOos.append(mean_squared_error(yTest, prediction))
print("MSE")
print(min(mseOos))
print(mseOos.index(min(mseOos)))

data = pd.DataFrame(prediction)
data[data < 1] = 0
data.to_csv('./stackingModelPrediction.csv')
