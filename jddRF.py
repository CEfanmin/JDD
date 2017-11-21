import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pylab as plot
from sklearn.externals import joblib
import pandas as pd

if __name__ == '__main__':
    raw_data = pd.read_csv('./rawSumFeatureData.csv')
    all_data = raw_data.fillna(0)
    labels = all_data['loan_sum']
    xList = all_data.iloc[:, 1:14]
    X = np.array(xList)
    y = np.array(labels)
    featureNames = np.array(['age', 'sex', 'limit', 'order_price_sum', 'order_count_sum',
                             'loan_price_sum', 'loan_count_sum', 'plannum_sum', 'click_count_sum',
                             'repayment_ability', 'active_ability', 'order_ability', 'loan_ability'])

    # take fixed holdout set 30% of data rows
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.10, random_state=531)

    # train random forest at a range of ensemble sizes in order to see how the mse changes
    mseOos = []
    # iTrees = 2000
    nTreeList = range(100, 1000, 20)
    for iTrees in nTreeList:
        depth = None
        maxFeat = 4  # try tweaking
        loanRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
                                                     oob_score=True, random_state=531)

        loanRFModel.fit(xTrain, yTrain)
        # Accumulate mse on test set
        prediction = loanRFModel.predict(xTest)
        mseOos.append(mean_squared_error(yTest, prediction))

    print("MSE")
    print(min(mseOos))
    # plot training and test errors vs number of trees in ensemble
    plot.figure()
    plot.plot(nTreeList, mseOos)
    plot.xlabel('Number of Trees in Ensemble')
    plot.ylabel('Mean Squared Error')
    plot.show()

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



