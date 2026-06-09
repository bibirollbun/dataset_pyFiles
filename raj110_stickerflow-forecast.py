print("JBB")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")



train.head()


test.head()


train.info()


train.describe()


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


# Extract time-based features
def add_time_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df


train = add_time_features(train)
test = add_time_features(test)


# More EDA - Visualizing sales trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=train, x='date', y='num_sold', hue='country')
plt.title("Sales Trends Over Time")
plt.show()


# Distribution of sales per store and item
plt.figure(figsize=(12, 6))
sns.boxplot(data=train, x='store', y='num_sold')
plt.title("Sales Distribution per Store")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train, x='product', y='num_sold')
plt.title("Sales Distribution per product")
plt.show()


# Encode categorical features
cat_features = ['country', 'store', 'product']
label_encoders = {}


for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le


# Fill missing values in num_sold with median
train['num_sold'].fillna(train['num_sold'].median(), inplace=True)


# Define features and target
features = ['country', 'store', 'product', 'year', 'month', 'day', 'day_of_week', 'is_weekend']
target = 'num_sold'


# Compute and plot correlation matrix
corr_matrix = train[features + [target]].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()


X_train, X_valid, y_train, y_valid = train_test_split(train[features], train[target], test_size=0.2, random_state=42)


# Train multiple models using pipelines
models = {
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("model", XGBRegressor(n_estimators=100, random_state=42))
    ]),
    "Ridge": Pipeline([
        ("model", Ridge(alpha=1.0))
    ])
}


for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mape = mean_absolute_percentage_error(y_valid, preds)
    print(f"{name} MAPE: {mape:.4f}")


# Select the best model (assuming lowest MAPE)
best_model = min(models.items(), key=lambda x: mean_absolute_percentage_error(y_valid, x[1].predict(X_valid)))[1]


# Predict on test set
test_preds = best_model.predict(test[features])


# Prepare submission
submission = sample_submission.copy()
submission['num_sold'] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")

