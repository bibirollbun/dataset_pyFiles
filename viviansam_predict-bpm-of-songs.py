import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# import library
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import optuna
import lightgbm as lgb
import xgboost as XGBRegressor
from sklearn.model_selection import KFold


df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
df.info()


df.head()


# Correlation analysis
correlation_matrix = df.corr(numeric_only=True)
bpm_corr = correlation_matrix['BeatsPerMinute'].sort_values(ascending=False)
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.show()


# Split data 
X = df.drop(columns=['id', 'BeatsPerMinute'])
y = df['BeatsPerMinute']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Modelling: LightGMB
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_eval])
gbm_y_pred = gbm.predict(X_test)

# RMSE
lgb_rmse = mean_squared_error(y_test, gbm_y_pred, squared=False)
print(f"RMSE: {lgb_rmse:.2f}")


df_test.info()


# Keep the 'ID' column separate
id_test = df_test['id']  

# Drop the 'ID' column from df_test
df_test = df_test.drop(columns=['id'])


# Predict
y_test = gbm.predict(df_test)


# Create a DataFrame with 'ID' and 'BeatsPerMinute' columns
output = pd.DataFrame({'id': id_test, 'BeatsPerMinute': y_test})
output.head()


output.to_csv('submission.csv', index=False)

