import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from pandas import scatter_matrix
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from Utils import getRPredicted, getStandartErrorEstimate

# read data
rawData = pd.read_csv('./datasets/Faculty.dat', delimiter=',')
# dropping Merit due to make problem simple.
rawData = rawData.drop(['Faculty', 'Merit'], 1)
# dependent variable is Salary

print rawData.describe()

# in this example Gender, Rank and Dept are categorical data.
# For categorical data, dummy variables are used
# In sort dummy variable is an encoding of categorical data by using number of categories minus 1 variables and each dummy variable can be 0 or 1
# in other words lets n is number of categorical data, n-1 dummy variable should introduced (for n > 1).
# if we have 2 category like Gender we can say 0 is one category 1 is different category.
# e.g.
# Dept have 3 category then we introduce 2 dummy variable, lets say dummy_1 and dummy_2Therefore
# Category 1 -> dummy_1 = 1, dummy_2 = 0
# Category 2 -> dummy_1 = 0, dummy_2 = 1
# Category 3 -> dummy_1 = 0, dummy_2 = 0

# I am not going into logic behind it but you can find why dummy variables are introduced in this way.



# Lets add dummy variable to dataframe

dept_dummies = pd.get_dummies(rawData['Dept']).rename(columns=lambda x: 'Dept_' + str(x))
rank_dummies = pd.get_dummies(rawData['Rank']).rename(columns=lambda x: 'Rank_' + str(x))

rawDataWithDummies = pd.concat([rawData, dept_dummies, rank_dummies], axis=1)

# droping extra dummy variables in order to satisfy encoding in previous example
rawDataWithDummies.drop(['Dept', 'Dept_3', 'Rank_3'], inplace=True, axis=1)
rawDataWithDummies = rawDataWithDummies.applymap(np.int)



# spliting data into two
# %80 for train %20 for test
msk = np.random.rand(len(rawData)) < 1
train = rawDataWithDummies[msk]
test = rawDataWithDummies[~msk]

# scatter plots of variables
scatter_matrix(train)
plt.show()

# Linearity check between ordinary variables not categorical ones
# signifance level
alpha = 0.01
print list(map(lambda x: pearsonr(train[x], train['Salary']), ['Years']))
# finding correlated IV
corr_IV = list(filter((lambda x: abs(pearsonr(train[x], train['Salary'])[1]) <= alpha), ['Years']))

# printing correlated independent variables
print 'List of linear correlated independent variables:'
print corr_IV

# in order to evaluate models dummy variables must think as a group
variableGroups = {
    'c1': ['Rank_1', 'Rank_2'],
    'c2': ['Dept_1', 'Dept_2'],
    'r1': ['Years']
}

# no need to co-linearity check because only one non categorical data
# But if we have more non-categorical data then we need to check co-linearity between them


# generating all possible models
models = []
resultDataFrame = pd.DataFrame(
    columns=('Index', 'Variables', 'Std Error', 'R-Squared ', 'R-Squared-Ajd', 'R-Squared-Pred'))
Y = train['Salary']
modelIndex = 0
for i in range(1, len(variableGroups.keys()) + 1):
    IVset = list(itertools.combinations(variableGroups.keys(), i))
    for keySet in IVset:
        # concat variable groups
        varSet = sum(map(lambda x : variableGroups[x],keySet), [])
        X = train[varSet]
        X = sm.add_constant(X)
        tmpModel = sm.OLS(Y, X)
        tmpResult = tmpModel.fit()
        # check it is adjusted if not re calculate
        # R2Adj = tmpResult.rsquared
        models.append(tmpModel)
        R2Pred = getRPredicted(Y, X)
        # insert to result DF
        resultDataFrame.loc[modelIndex] = [modelIndex, ' '.join(list(varSet)),
                                           getStandartErrorEstimate(tmpResult.resid.values), tmpResult.rsquared,
                                           tmpResult.rsquared_adj, R2Pred]
        modelIndex = modelIndex + 1


# print result dataframe
print 'Results for different models'
print resultDataFrame

