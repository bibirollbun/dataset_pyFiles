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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.head()


train.info()


train.describe()


print(train.isnull().sum())
print(test.isnull().sum())


#Distribution of Accident risk
plt.figure(figsize=(12,6))
sns.histplot(data=train, x="accident_risk")
plt.title('Distribution of Accident risk')
plt.show()


cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

plt.figure(figsize=(12, 8))
for i, col in enumerate(cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train[col], bins=20, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Categorical variables
categorical_cols = train.select_dtypes(include=['object', 'category']).columns

# Set figure size for multiple plots
plt.figure(figsize=(15, len(categorical_cols) * 4))

# Loop through categorical columns
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(len(categorical_cols), 1, i)
    sns.countplot(x=col, data=train, palette='pastel')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()

plt.show()


boolen_cols=['road_signs_present','public_road','holiday','school_season']
# Set figure size for multiple plots
plt.figure(figsize=(15, len(boolen_cols) * 4))

# Loop through categorical columns
for i, col in enumerate(boolen_cols, 1):
    plt.subplot(len(boolen_cols), 1, i)
    sns.countplot(x=col, data=train, palette='pastel')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()

plt.show()


#Boxplot to check for outliers
cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents','accident_risk']

n = len(cols)
rows = math.ceil(n / 2)  

plt.figure(figsize=(12, 4 * rows))

for i, col in enumerate(cols, 1):
    plt.subplot(rows, 2, i)
    sns.boxplot(x=train[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.xlabel('')
    plt.tight_layout()

plt.show()


#Correlation
cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents','accident_risk']

corr = train[cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


#transforming categorical variables
le = LabelEncoder()

categorical_cols = train.select_dtypes(include=['object', 'category']).columns


for col in categorical_cols:
    train[col] = le.fit_transform(train[col])

print(train[categorical_cols].head())


boolen_cols=['road_signs_present','public_road','holiday','school_season']
for col in boolen_cols:
    train[col] = le.fit_transform(train[col])

print(train[boolen_cols].head())


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance


X = train.drop(columns=['id', 'accident_risk'])
y = train['accident_risk']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


xgb_reg = XGBRegressor(
    n_estimators=300,        
    learning_rate=0.1,      
    max_depth=5,
    random_state=42,
    objective='reg:squarederror' 
)

xgb_reg.fit(X_train, y_train)


y_pred = xgb_reg.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


plt.figure(figsize=(8, 6))
plot_importance(xgb_reg, importance_type='gain')
plt.title('Feature Importance (XGBoost Regressor)')
plt.show()


from xgboost import cv, DMatrix

dtrain = DMatrix(X, label=y)
params = {
    'max_depth': 5,
    'eta': 0.05,
    'objective': 'reg:squarederror'
}

cv_results = cv(
    params,
    dtrain,
    num_boost_round=300,
    nfold=5,
    metrics='rmse',
    seed=42
)

print(cv_results.tail(1))


from sklearn.model_selection import cross_val_score, KFold

xgb_model = XGBRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    random_state=42
)

# Define 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model using negative RMSE
scores = cross_val_score(xgb_model, X, y, scoring='neg_root_mean_squared_error', cv=kfold)

print("Cross-validation RMSE scores:", -scores)
print("Average RMSE:", np.mean(-scores))


le = LabelEncoder()

categorical_cols = test.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    test[col] = le.fit_transform(test[col])

boolen_cols=['road_signs_present','public_road','holiday','school_season']
for col in boolen_cols:
    test[col] = le.fit_transform(test[col])


test_df=test.drop(columns=['id'])


prediction = xgb_reg.predict(test_df)
prediction


#submission
sub = pd.DataFrame({
    "id": test["id"],
    'accident_risk': prediction
})

# Save submission file
sub.to_csv("submission.csv", index=False)


sub.head()


X_new = train.drop(columns=['id', 'road_type', 'school_season', 'road_signs_present', 
                            'time_of_day', 'num_lanes', 'accident_risk'])
y_new = train['accident_risk']


X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)


xgb_reg_new = XGBRegressor(
    n_estimators=300,        
    learning_rate=0.1,      
    max_depth=5,
    random_state=42,
    objective='reg:squarederror' 
)

xgb_reg_new.fit(X_train_new, y_train_new)


y_pred_new = xgb_reg_new.predict(X_test_new)


mse = mean_squared_error(y_test_new, y_pred_new)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_new, y_pred_new)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


test_df_new=test.drop(columns=['id','road_type', 'school_season', 'road_signs_present', 
                            'time_of_day', 'num_lanes'])


prediction_new = xgb_reg_new.predict(test_df_new)
prediction_new


#submission
sub_new = pd.DataFrame({
    "id": test["id"],
    'accident_risk': prediction_new
})

# Save submission file
sub_new.to_csv("submission.csv", index=False)


sub_new.head()


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Base models
base_models = [
    ('xgb', XGBRegressor(random_state=42)),
    ('rf', RandomForestRegressor(random_state=42)),
    ('gbr', GradientBoostingRegressor(random_state=42)),
    ('lgbm', LGBMRegressor(random_state=42))
]

# Meta-model
meta_model = RidgeCV()

stacked = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stacked.fit(X_train_new, y_train_new)
y_pred_new_stacked = stacked.predict(X_test_new)

mse = mean_squared_error(y_test_new, y_pred_new_stacked)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_new, y_pred_new_stacked)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


prediction_stacked = stacked.predict(test_df_new)
prediction_stacked


#submission
sub_stacked = pd.DataFrame({
    "id": test["id"],
    'accident_risk': prediction_stacked
})

# Save submission file
sub_stacked.to_csv("submission.csv", index=False)


sub_stacked.head()

