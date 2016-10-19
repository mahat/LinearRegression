import statsmodels.api as sm
import numpy as np

def getRPredicted(Y,X):
    # in order to calculate RSquared Predicted,
    # systematically remove each observation from all observations
    # and calculate prediction power for removed observation
    # and return R Squared value
    realY = []
    predY = []
    for ind in X.index:
        tmpX = X[X.index != ind]
        tmpY = Y[Y.index != ind]
        model = sm.OLS(tmpY, tmpX)
        result = model.fit()
        realY.append(Y[Y.index == ind].values[0])
        predY.append(result.predict(X[X.index == ind])[0])

    # calculating R - Squared Predicted
    meanY = np.sum(realY) / len(realY)  # or sum(y)/len(y)
    return 1 - (np.sum([(p - r) ** 2 for p, r in zip(predY, realY)]) / np.sum((realY - meanY)**2))

def getStandartErrorEstimate(resid):
    return np.sqrt(np.sum(resid ** 2) / (len(resid) - 2))