# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')


#load dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.head()


test.head()


print("Train shape:", train.shape)


print("Test shape:", test.shape)


train.info()


test.info()


#check missing value
print(train.isnull().sum())


print(test.isnull().sum())


#Target distribution
plt.figure(figsize=(8,6))
sns.histplot(train['accident_risk'], kde=True, bins=30, color='royalblue')
plt.title('Distribution of Accident Risk')
plt.show()


#correlation only on numeric columns
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12,8))
sns.heatmap(train[numeric_cols].corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlations (Numeric Features Only)')
plt.show()


# Handle missing values (if any)
train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(test.median(numeric_only=True), inplace=True)

# Encoding categorical features
cat_cols = train.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


#create polynomial features
num_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
if 'accident_risk' in num_cols:
    num_cols.remove('accident_risk')
if 'id' in num_cols:
    num_cols.remove('id')

for col in num_cols:
    train[f'{col}_sq'] = train[col] ** 2
    test[f'{col}_sq'] = test[col] ** 2


X = train.drop(['accident_risk', 'id'], axis=1)
y = train['accident_risk']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
test_scaled = scaler.transform(test.drop('id', axis=1))


def evaluate_model(model):
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_valid_scaled)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse

models = {
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=300),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.05),
    "LightGBM": LGBMRegressor(random_state=42, n_estimators=300, learning_rate=0.05),
    "CatBoost": CatBoostRegressor(random_state=42, verbose=0, learning_rate=0.05)
}

results = {}
for name, model in models.items():
    rmse = evaluate_model(model)
    results[name] = rmse
    print(f"{name} RMSE: {rmse:.5f}")


result_df = pd.DataFrame(results.items(), columns=['Model', 'RMSE']).sort_values(by='RMSE')
sns.barplot(x='Model', y='RMSE', data=result_df, palette='coolwarm')
plt.title('Model Performance Comparison')
plt.show()

best_model_name = result_df.iloc[0]['Model']
print("Best Model:", best_model_name)


if best_model_name == "LightGBM":
    model = LGBMRegressor(random_state=42)
    param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1]
    }
elif best_model_name == "XGBoost":
    model = XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [6, 8],
        'learning_rate': [0.05, 0.1]
    }
else:
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [10, 15, 20]
    }

grid = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print("Best Params:", grid.best_params_)
print("Best RMSE:", -grid.best_score_)


final_model = grid.best_estimator_
final_model.fit(X, y)


preds = final_model.predict(test_scaled)


submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': np.clip(preds, 0, 1)  # ensure between 0 and 1
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")
submission.head()

