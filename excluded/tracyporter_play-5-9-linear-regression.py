import pandas as pd
import numpy as np
import os

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.graphics.gofplots import qqplot

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


pd.set_option('display.max_columns', None)


train


train.info()


train.isna().sum().sum()


test


test.info()


test.isna().sum().sum()


submission


drop_col = []
alpha = 0.05

for col in test:
    pv, _ = stats.ks_2samp(train[col], test[col])
    if pv < alpha:
        drop_col.append(col)

print(drop_col)


train = train.drop('id', axis = 1)
test = test.drop('id', axis = 1)

train.shape, test.shape


target = train.pop("BeatsPerMinute")
target


# Create the histogram
plt.hist(target, bins=30, color='blue', edgecolor='black', alpha=0.7, label='target')
# Add a vertical line at x=0
plt.axvline(x=target.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
# Add labels and title
plt.title('Histogram of Target')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
# Show the plot
plt.show()


corr = train.corr()
train_heatmap = sns.heatmap(corr, cmap="viridis")


corr


y = target
X = train
X_test = test


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=None)
X_train.shape, X_val.shape,y_train.shape, y_val.shape, X_test.shape


model = LinearRegression().fit(X_train, y_train)
model.score(X_train, y_train)


y_pred = model.predict(X_val)
y_pred


# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_val, y_pred))  # Mean Squared Error
r2 = r2_score(y_val, y_pred)  # R-squared score


print("Model Coefficients (Slope):", model.coef_[0])
print("Model Intercept:", model.intercept_)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)


X_val.shape, y_val.shape


# Plot the results
fig, ax = plt.subplots()
ax.scatter(y_val, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


#qq plot
qqplot(y_pred,line='s')
plt.show()


pred = model.predict(X_test)
pred


# Create the histogram
plt.hist(pred, bins=30, color='blue', edgecolor='black', alpha=0.7, label='target')
# Add a vertical line at x=0
plt.axvline(x=target.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
# Add labels and title
plt.title('Histogram of Predictions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
# Show the plot
plt.show()


submission['BeatsPerMinute'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

