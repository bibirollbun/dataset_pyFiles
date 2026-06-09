# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')


train.shape


train.head()


train.isnull().sum()


test.isnull().sum()


train.info()


train['holiday'].value_counts()


cat_cols = train.select_dtypes(include='object').columns.to_list()
le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

train.head()
    


sns.histplot(x='accident_risk', bins=20, kde=True, data=train, fill=False)
plt.title('Histogram for distribution of target variables')
plt.xlabel('Accident risk')


corr_values = train.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_values, annot=True, fmt='.2f')


train.describe().T


train.shape


X = train.copy()
y = X.pop('accident_risk')


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=2)


lr = LinearRegression()
lr.fit(X_train, y_train)


y_pred = lr.predict(X_valid)


fig, ax = plt.subplots(figsize=(12, 6))
sns.regplot(
    x=y_valid, y=y_pred, color='#ff9f0ed2',
    scatter_kws={
        's': 70,           
        'alpha': 0.7,       
        'edgecolor': 'w',   
        'linewidth': 0.6
    },
    line_kws={'color': '#2c3e50', 'linewidth': 2},
    ci=95
)
ax.set_aspect('auto')
ax.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--')
ax.set_title("Actual vs Predicted (with perfect-prediction line)")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.legend()

plt.show()


def regression_results(y_true, y_pred, regr_type):

    # Regression metrics
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred) 
    mse = mean_squared_error(y_true, y_pred) 
    r2 = r2_score(y_true, y_pred)
    
    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('Explained_variance: ',  round(ev,4)) 
    print('R2: ', round(r2,4))
    print('MAE: ', round(mae,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print()

regression_results(y_valid, y_pred, 'LinearRegression')


residuals = np.array(y_valid) - np.array(y_pred)

mean_resid = residuals.mean()
rmse = mean_squared_error(y_valid, y_pred, squared=False)

print(f"Mean residual (bias) = {mean_resid:.4f}")
print(f"RMSE = {rmse:.4f}")

plt.figure(figsize=(10,5))
plt.scatter(y_pred, residuals, s=30, alpha=0.7, edgecolor='w')
plt.axhline(0, color='black', linestyle='--', linewidth=1) 
plt.xlabel('Predicted values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30, edgecolor='k', color='orange', alpha=0.7)
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Count')
plt.title('Residuals Distribution')
plt.tight_layout()
plt.show()



X_test = test.copy()
final_preds = lr.predict(X_test)


sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sub['accident_risk'] = final_preds
sub.to_csv('submission.csv', index=False)


sub.head()

