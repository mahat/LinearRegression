import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

# read data
rawData = pd.read_csv('./datasets/nuclear_time.dat', delim_whitespace=True, header=-1)
rawData.columns = ['Task', 'Nationality', 'Time_To_Complete', 'Complexity_Score']
rawData = rawData.drop(['Task', 'Nationality'], 1)
# rename columns for simplicity
rawData.rename(columns={'Complexity_Score': 'X', 'Time_To_Complete': 'Y'}, inplace=True)
# divide data into train and test
# %80 for train %20 for test
msk = np.random.rand(len(rawData)) < 0.8
train = rawData[msk]
test = rawData[~msk]

# plot data
plt.figure()
plt.scatter(train.X, train.Y)
plt.xlabel('Complexity Score', fontsize=20)
plt.ylabel('Time to Complete', fontsize=20)
plt.title('Scatter plot X vs. Y')
plt.grid(True)
plt.show()

# check correlation
# significance value
alpha = 0.001
corr = pearsonr(train.X, train.Y)
corr_Val = corr[0]
corr_pValue = corr[1]

print 'According to Pearson correlation results'
print 'Correlation coefficient = %f' % corr_Val
print '2 tailed p-value = %f' % corr_pValue

if corr_pValue < alpha:
    print 'For alpha = %.4f, it can be said that the is a linear correlation between X and Y.' % alpha
else:
    print 'For alpha = %.4f, it can not be said that the is a linear correlation between X and Y.' % alpha

# model
Y = train.Y
X = train.X
# adding constant for intercept value
X = sm.add_constant(X)

model = sm.OLS(Y, X)
result = model.fit()

# showing residuals
plt.plot(range(len(Y)), result.resid, '*')
plt.title('Residual Plot')
plt.show()

# finding predictions
trainPred = result.predict(X)
# plot predictions
plt.plot(range(len(Y)), Y, 'r*-', range(len(Y)), trainPred, 'bo-')
plt.title('Train dataset Real vs. Predicted Values')
plt.legend(['Real Values', 'Predicted Values'])
plt.show()

# plot residuals


# model stats
print result.summary()
print('Parameters: ', result.params)
print('R2: ', result.rsquared)
# in the summary we can see Adj. R-squared which is very important metric. It is between 0 - 1 and it should be high as much as possible.

# predict test data
testX = test.X
testX = sm.add_constant(testX)
testY = test.Y
testPred = result.predict(testX)

# plot test predictions
plt.plot(range(len(testY)), testY, 'r*-', range(len(testY)), testPred, 'bo-')
plt.title('Test Dataset Real vs. Predicted Values')
plt.legend(['Real Values', 'Predicted Values'])
plt.show()
