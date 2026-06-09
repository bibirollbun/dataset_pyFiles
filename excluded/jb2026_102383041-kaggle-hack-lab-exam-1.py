# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/sample_submission.csv')


print("Training Data shape: ",train_data.shape)
print("Training Data Info: ")
print(train_data.info())
print("Summary of Numerical Features: ")
print(train_data.describe())
print("Missing Values in Training Data: ")
print(train_data.isnull().sum())


import warnings

train_data.replace([np.inf, -np.inf], np.nan, inplace=True)

train_data.fillna(train_data.mean(), inplace=True)

warnings.filterwarnings("ignore", category=FutureWarning)

sns.pairplot(train_data,diag_kind='kde',corner=True)
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(train_data.drop(columns=['target']))
poly_feature_names = poly.get_feature_names_out(train_data.drop(columns=['target']).columns)


X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
X_poly_df['target'] = train_data['target']

print("Poly Feature Shape:", X_poly_df.shape)


X = X_poly_df.drop(columns=['target'])
y = X_poly_df['target']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)


y_valid_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_valid_pred)
print(f"Validation MAE: {mae}")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_maes = []

for train_idx, val_idx in kf.split(X_scaled):
    X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    y_fold_pred = model.predict(X_fold_val)
    fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
    fold_maes.append(fold_mae)

print(f"Mean MAE across folds: {np.mean(fold_maes)}")


train_features = train_data.drop(columns=['target'])
test_features = test_data[train_features.columns]

test_poly = poly.transform(test_features)

test_poly_df = pd.DataFrame(test_poly, columns=poly.get_feature_names_out(train_features.columns))

test_scaled = scaler.transform(test_poly_df)

test_predictions = model.predict(test_scaled)


submission = sample_submission.copy()
submission['target'] = test_predictions
submission.to_csv('My_submission.csv', index=False)
print("Submission file saved as My_submission.csv")

