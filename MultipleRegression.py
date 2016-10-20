import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from pandas import scatter_matrix
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from Utils import getRPredicted, getSampleStandardErrorEstimate

# read data
rawData = pd.read_csv('./datasets/regressionTestData.txt', delim_whitespace=True)
rawData.set_index('Index')
# describe of data
print rawData.describe()

# spliting data into two
# %80 for train %20 for test
msk = np.random.rand(len(rawData)) < 0.8
train = rawData[msk]
test = rawData[~msk]

# histogram of variables
train.hist()
plt.show()

# scatter plots of variables
scatter_matrix(train)
plt.show()

# check for linearity between IV vs DV
# significance
alpha = 0.01

print list(map(lambda x: pearsonr(train[x], train['B']), train.columns))
# finding correlated IV
corr_IV = list(filter((lambda x: abs(pearsonr(train[x], train['B'])[1]) <= alpha), train.columns))
# removing dependent value from list
corr_IV.remove('B')
# printing correlated independent variables
print 'List of linear correlated independent variables:'
print corr_IV

# checking multi - colinearity
# in linear regression all IV must be non-correlated each other
# |pearson| > 0.5 strong linear relationship
colinear = {}
colinearAlpha = 0.01
for var in corr_IV:
    colinear[var] = list(filter((lambda x: pearsonr(train[x], train[var])[1] <= colinearAlpha), corr_IV))

print 'Possible co linear IV:'
for k, v in colinear.iteritems():
    print '%s -> %s' % (k, ', '.join(v))

# usually colinear variables shouldn't be considered together,
# but for educational purposes it will continue as these variables are not collinear.

# creating the model
# because we have many variables, regression model should created incrementally.
# In other words, adding to many variables to model will result into overfitting.
# Therefore, picking right variables are very important.

# generating all possible models
models = []
results = []
resultDataFrame = pd.DataFrame(
    columns=('Index', 'Variables', 'Std Error', 'R-Squared ', 'R-Squared-Ajd', 'R-Squared-Pred'))
Y = train['B']
modelIndex = 0
for i in range(1, len(corr_IV) + 1):
    IVset = list(itertools.combinations(corr_IV, i))
    for varSet in IVset:
        X = train[list(varSet)]
        X = sm.add_constant(X)
        tmpModel = sm.OLS(Y, X)
        tmpResult = tmpModel.fit()

        models.append(tmpModel)
        results.append(tmpResult)
        R2Pred = getRPredicted(Y, X)
        # insert to result DF
        resultDataFrame.loc[modelIndex] = [modelIndex, ' '.join(list(varSet)),
                                           getSampleStandardErrorEstimate(tmpResult.resid.values), tmpResult.rsquared,
                                           tmpResult.rsquared_adj, R2Pred]
        modelIndex = modelIndex + 1



# print result dataframe
print 'Results for different models'
print resultDataFrame
# pick the model
# in order to pick the best model. there are three important steps
# 1st: Biggest Adjusted R square score
# 2nd: There should be too big difference between Adjusted R Square and Predicted R square, If there is a big difference that means model is overfitting
# 3rd: If we have different candidates which are good in first two steps, then it is usually good to pick simplest model. In other words, the model which requires least amount of independent variables.

#E.g.
#    Index Variables    Std Error  R-Squared   R-Squared-Ajd  R-Squared-Pred
# 0      0        A1  2314.063210    0.152336       0.140227        0.094816
# 1      1        A2  2230.725494    0.212292       0.201039        0.153679
# 2      2        A3  2032.909476    0.345802       0.336456        0.309627
# 3      3     A1 A2  2202.476706    0.232116       0.209858        0.148327
# 4      4     A1 A3  1958.865017    0.392590       0.374984        0.340680
# 5      5     A2 A3  1949.483008    0.398394       0.380956        0.346121
# 6      6  A1 A2 A3  1930.243557    0.410210       0.384190        0.344851

# Above there is a result table which is created by the code. Model #5 and #6 are very close candidates. They are good at Adj-R2 and Pred-R2, but model #5 is more suitable because it requires less number of variables.

# Last but not least: There is not the best method :) I generally look at these points. Maybe there are better ones. If you find one of these please use them.


# Important Step!
# plotting residuals to see they are normally distributed
fig = plt.figure()
results[5].resid.hist()
plt.title('Residual Histogram Plot')
plt.show()
