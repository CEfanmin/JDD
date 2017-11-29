import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pylab as plot
from sklearn.externals import joblib
import pandas as pd


if __name__ == '__main__':
    raw_data = pd.read_csv('../data/userInfoSum_8-10.csv')
    raw_data = raw_data.replace([np.inf, -np.inf], 99999)
    all_data = raw_data.fillna(0)
    labels = all_data['loan_sum']
    xList = all_data.iloc[:, 1:16]
    X = np.array(xList)
    y = np.array(labels)
    featureNames = np.array(['age', 'sex', 'limit',
                             'loan_count_sum', 'loan_price_sum','plannum_sum', 'click_count_sum', 'order_price_sum','order_count_sum',
                             'active_ability', 'click_model', 'consumer_model', 'loan_ability', 'order_ability', 'repayment_ability',
                             'loan_sum'])
    # take fixed holdout set 20% of data rows
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.20, random_state=531)

    # train random forest at a range of ensemble sizes in order to see how the mse changes
    iTrees = 750
    # nTreeList = range(100, 2000, 100)
    # for iTrees in nTreeList:
    depth = 10
    maxFeat = 4  # try tweaking
    parameter_space = {
        "n_estimators": [700, 750, 800],
        "min_samples_leaf": [2, 4, 6],
        'max_features': [4, 5],
        'max_depth': [7, 8, 9],
    }
    # loanRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
    #                                              oob_score=True, random_state=2017, min_samples_split=2, n_jobs=3)
    loanRFModel = ensemble.RandomForestRegressor(oob_score=True, random_state=2017, min_samples_split=2, n_jobs=3)
    print("Tuning hyper-parameters")
    grid = GridSearchCV(loanRFModel, parameter_space, cv=5)
    grid.fit(xTrain, yTrain)
    print("Best parameters set found on development set:")
    print(grid.best_params_)


    # Accumulate mse on test set
    prediction = grid.predict(xTest)
    rmse = math.sqrt(mean_squared_error(yTest, prediction))
    print("validation rmse is: ", rmse, ' iTrees is: ', iTrees)

    finalPrediction = grid.predict(X)
    total_mse = mean_squared_error(finalPrediction, y)
    total_rmse = math.sqrt(total_mse)
    print("total rmse score is: ", total_rmse)

    # prediction
    predData = pd.read_csv('../data/userInfoSum_9-11.csv')
    predData = predData.replace([np.inf, -np.inf], 99999)
    xList = predData.iloc[:, 1:16].fillna(0)
    X = np.array(xList)
    finalPrediction = grid.predict(X)
    data = pd.DataFrame(finalPrediction, index=np.arange(1, len(X) + 1))
    data[data < 1] = 0
    data.to_csv('../result/prediction_RF.csv')
    print('write to csv file')

    # plot training and test errors vs number of trees in ensemble
    # plot.figure()
    # plot.plot(nTreeList, mseOos)
    # plot.xlabel('Number of Trees in Ensemble')
    # plot.ylabel('Mean Squared Error')
    # plot.show()

    # save model and load model
    # joblib.dump(loanRFModel, 'D:\JDDModels\RFwithMode\clf.pkl')
    # finalClasss = joblib.load('D:\JDDModels\RFwithModeMean\clf.pkl')
    # finalPrediction = loanRFModel.predict(X)
    # data = pd.DataFrame(finalPrediction)
    # data.to_csv('./v6-prediction.csv')

    # Plot feature importance
    # featureImportance = loanRFModel.feature_importances_
    # featureImportance = featureImportance / featureImportance.max()  # scale by max importance
    # sorted_idx = np.argsort(featureImportance)
    # barPos = np.arange(sorted_idx.shape[0]) + .5
    # plot.barh(barPos, featureImportance[sorted_idx], align='center')
    # plot.yticks(barPos, featureNames[sorted_idx])
    # plot.xlabel('Variable Importance')
    # plot.show()



