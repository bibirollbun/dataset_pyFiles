import numpy as np
import pandas as pd

import os

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pylab 
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train


train.isna().sum().sum()


test


test.isna().sum().sum()


submission


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

train.shape, test.shape


print(train.Sex.unique())


print(test.Sex.unique())


gender_dict = {'female':0, 'male':1}

train['Sex'] = train['Sex'].map(gender_dict)
test['Sex'] = test['Sex'].map(gender_dict)

print(train.Sex.value_counts())
print(test.Sex.value_counts())


# Plotting a basic histogram
plt.hist(train['Calories'], bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Calories Histogram')
 
# Display the plot
plt.show()


corr = train.corr()
sns.heatmap(corr)


y = train.pop('Calories')
X = train
X_test = test


X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.1, shuffle=True, random_state=42)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape


model = linear_model.GammaRegressor(max_iter=5000).fit(X_train,y_train)
model.score(X_train, y_train)


y_pred = model.predict(X_val)
y_pred


mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
rmse


def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE).
    
    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    
    Returns:
    float: RMSLE value.
    """
    # Ensure no negative values (logarithm is undefined for negatives)
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    
    # Compute the logarithmic differences
    log_true = np.log1p(y_true)  # log1p(x) = log(1 + x)
    log_pred = np.log1p(y_pred)
    
    # Calculate the mean squared logarithmic error
    msle = np.mean((log_true - log_pred) ** 2)
    
    # Return the square root of MSLE
    return np.sqrt(msle)

rmsl_error = rmsle(y_val, y_pred)
rmsl_error


df = pd.DataFrame({'y_val':y_val, 'y_pred':y_pred})
df


# Plotting a basic histogram
plt.hist(y_pred, bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('y_pred Histogram')
 
# Display the plot
plt.show()



stats.probplot(y_pred, dist="norm", plot=pylab)
pylab.show()


fig, ax = plt.subplots()
ax.scatter(y_val, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


pred = model.predict(X_test)
pred


submission['Calories'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

