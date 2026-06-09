# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, rwarnings.filterwarnings("ignore")unning this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, warnings
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings("ignore")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor 
import lightgbm as lgb


train_data=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission=pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train_data.head()


train_data.info()


train_data.isnull().sum()


train_data.describe()


categorical_features=[feature for feature in train_data.columns if train_data[feature].dtype=='O']


rows = int(np.ceil(len(categorical_features) / 3))  # Ensure it aligns with categorical_features
col = 3

# Creating figure and setting size
fig, axes = plt.subplots(rows, col, figsize=(15, rows * 5))
axes = axes.flatten()

# Looping through features and creating subplots
for idx, feature in enumerate(categorical_features):
    data = train_data.copy()
    
    # Plot on current subplot
    sns.countplot(x=feature, data=data, ax=axes[idx])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel('')

# Hide any unused subplots
for idx in range(len(categorical_features), len(axes)):  # Aligning with categorical_features
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()



numerical_features=[feature for feature in train_data.columns if train_data[feature].dtype!='O' and train_data[feature].dtype!='bool']
numerical_features


rows = int(np.ceil(len(numerical_features) / 3))  # 3 columns layout
cols = 3

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
axes = axes.flatten()

for idx, feature in enumerate(numerical_features):
    data = train_data.copy()
    sns.histplot(data=data, x=feature, kde=True, bins=30, ax=axes[idx], color='teal')
    axes[idx].set_title(f"Distribution of {feature}", fontsize=12)
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Count')

# Hide unused axes
for idx in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()


#converting boolean features to int
train_data[train_data.select_dtypes('bool').columns] = train_data.select_dtypes('bool').astype(int)
test_data[test_data.select_dtypes('bool').columns] = test_data.select_dtypes('bool').astype(int)


# using one hot encoding 
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
train_data = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)
test_data = pd.get_dummies(test_data, columns=categorical_features, drop_first=True)


# Correlation heatmap
sns.heatmap(train_data.corr(), cmap='bwr')

# Or pairplots
#sns.pairplot(train_data, x_vars=['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents'], y_vars='accident_risk')


train_data.drop('id',axis=1,inplace=True)
test_data.drop('id',axis=1,inplace=True)


# Select features and target
X = train_data.drop('accident_risk', axis=1)
y = train_data['accident_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}

print("Model Comparison Results:")
print("=" * 50)

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse
    
    print(f"{name:<15} | RMSE: {rmse:.6f}")

# Find best model
best_model = min(results, key=results.get)
best_rmse = results[best_model]

print("=" * 50)
print(f"ðŸŽ¯ BEST MODEL: {best_model}")
print(f"ðŸ“Š BEST RMSE: {best_rmse:.6f}")


xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.2,
    random_state=42,
    verbosity=1,
    subsample=0.8
)


# Train model
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# âœ… Evaluate using RMSE
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb:.4f}")


test_pred = xgb_model.predict(test_data)


submission['accident_risk']=test_pred.round(2)


submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv file created with rounded predictions!")

