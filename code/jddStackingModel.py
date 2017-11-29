import pandas as pd
import numpy as np
import math
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor


start_time = time.time()
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
		print("T length is: ", T.shape[0])
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
	def __init__(self, clf, params=None):
		self.clf = clf(**params)

	def train(self, x_train, y_train):
		self.clf.fit(x_train, y_train)

	def predict(self, x_test):
		return self.clf.predict(x_test)

	def fit(self, x, y):
		return self.clf.fit(x, y)

	def feature_importances(self, x, y):
		print(self.clf.fit(x, y).feature_importances_)



# load data and split data to test data and training data
def main():
	raw_data = pd.read_csv('../data/userInfoTime_8-10.csv')
	all_data = raw_data.fillna(0)
	labels = all_data['loan_sum']
	xList = all_data.iloc[:, 1:24]
	X = np.array(xList)
	y = np.array(labels)
	xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
	pTest = np.array(pd.read_csv('../data/userInfoTime_9-11.csv').iloc[:, 1:24])
	# Random Forest parameters
	rf_params = {
		'n_jobs': -1,
		'oob_score': True,
		'max_depth': None,
		'max_features': 5,
		'n_estimators': 800,
		'random_state': 2017
	}
	# Gradient Boosting parameters
	gb_params = {
		'n_estimators': 200,
		'subsample': 0.4,
		'learning_rate': 0.1,
		'loss': 'ls',
		'max_depth':2,
		'max_features':0.6,
		'min_samples_leaf':6,
		'min_samples_split':17
	}
	# XGB parameters
	XGB_params = {
		'learning_rate': 0.1,
		'max_depth': 1,
		'min_child_weight': 13,
		'n_estimators': 100,
		'n_jobs': -1,
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


	# level1 base model
	rf = SklearnHelper(clf=ensemble.RandomForestRegressor, params=rf_params)
	gb = SklearnHelper(clf=ensemble.GradientBoostingRegressor, params=gb_params)
	XGB = SklearnHelper(clf=XGBRegressor, params=XGB_params)
	et = SklearnHelper(clf=ensemble.ExtraTreesRegressor, params=et_params)
	base_models = [rf, gb, XGB, et]

	# level2 is XGB
	stackerModel = Ensemble(n_folds=4, stacker=XGB, base_models=base_models)


	# validation rmse
	prediction = stackerModel.fit_predict(xTrain, yTrain, xTest)
	test_mse = mean_squared_error(yTest, prediction)
	test_rmse = math.sqrt(test_mse)
	print("validation rmse score is: ", test_rmse)


	# prediction
	final_prediction = stackerModel.fit_predict(X, y, pTest)
	print("prediction len is: ", len(final_prediction))
	data = pd.DataFrame(final_prediction, index=np.arange(1, len(X) + 1))
	data[data < 1] = 0
	data.to_csv('../result/prediction_StackingModel.csv')
	print("submission time is: ", round(((time.time() - start_time) / 60), 2))


if __name__=="__main__":
	main()
