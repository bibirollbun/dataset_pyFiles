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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

print("Train shape:", df.shape)
print("Test shape:", df_test.shape)


# Dataset Overview
display(Markdown("### Dataset Info"))
df.info()

display(Markdown("### First 5 Rows"))
df.head()


# Missing Values Analysis

missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]

if not missing.empty:
    plt.figure(figsize=(10,6))
    sns.barplot(x=missing.values, y=missing.index, palette="viridis")
    plt.title("Missing Values per Feature")
    plt.xlabel("Count of Missing Values")
    plt.ylabel("Feature")
    plt.show()

    sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu")
else:
    print("No missing values found in the dataset.")


numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()


# Check target variable distribution
sns.countplot(x='accident_risk', data=df, palette='viridis')
plt.title('Distribution of Accident Risk')
plt.show()


#Exploratory Data Analysis (EDA)

# Numerical Features
num_cols = df.select_dtypes(include=np.number).columns.drop('accident_risk', errors='ignore')

df[num_cols].describe().T

# Histograms
df[num_cols].hist(figsize=(14,8), bins=30, color="skyblue")
plt.suptitle("Numerical Features Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()


if 'latitude' in df.columns and 'longitude' in df.columns:
    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude",
                            color="accident_risk", size_max=10, zoom=5,
                            mapbox_style="carto-positron",
                            title="Accident Locations by Risk")
    fig.show()


#data processing
# Encode categorical variables
from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    df_test[col] = le.fit_transform(df_test[col])

# Features & target
X = df.drop('accident_risk', axis=1)
y = df['accident_risk']


# Train-test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Base model
lgb_model = lgb.LGBMRegressor(random_state=42)

# Parameter grid
param_grid = {
    'num_leaves': [31, 50, 100, 200],
    'max_depth': [-1, 10, 20, 30],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_samples': [20, 40, 60]
}

# Random search
search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    n_iter=20,  # increase for more thorough search
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best RMSE (CV):", -search.best_score_)

# Train final model with best params
best_lgb = search.best_estimator_
y_pred = best_lgb.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Final Test RMSE: {rmse:.4f}")
print(f"Final Test RÂ²: {r2:.4f}")



# Define the model
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define hyperparameter grid

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}

# Randomized Search for faster tuning
random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=param_grid,
    n_iter=30,              # number of random combinations
    scoring='neg_mean_squared_error',
    cv=2,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", random_search.best_params_)

# Train final model with best parameters
best_xgb = random_search.best_estimator_

# Predict
y_pred = best_xgb.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")


from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=10,
    loss_function='RMSE',
    random_seed=42,
    verbose=200
)

cat_model.fit(X_train, y_train)
y_pred = cat_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"CatBoost RMSE: {rmse:.4f}")
print(f"CatBoost RÂ²: {r2:.4f}")



from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

estimators = [
    ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)),
    ('lgbm', lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)),
    ('cat', CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, verbose=0, random_state=42))
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV()
)

stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Stacking RMSE: {rmse:.4f}")
print(f"Stacking RÂ²: {r2:.4f}")



y_df_pred=stack_model.predict(df_test)



submission1 = pd.DataFrame({
    'id': df_test['id'],
    'accident_risk': y_df_pred
})
submission1.to_csv('submission1.csv', index=False)
print("âœ… Submission saved!")
print(submission1.head())

