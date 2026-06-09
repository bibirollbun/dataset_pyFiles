import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb

import optuna
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


print(train.shape) 
print(test.shape)
print(sample_submission.shape)


train.head()


test.head()


sample_submission.head()


# Check for missing values
print(train.isnull().sum())


train.info()


# Feature summary
print(train.describe())


 # Visualize distributions
sns.histplot(train['Calories'], bins=50)


sns.countplot(x='Sex', data=train)
plt.title('Count of each Gender')
plt.show()


sns.distplot(train['Age'])


sns.distplot(train['Height'])


sns.distplot(train['Weight'])


sns.distplot(train['Duration'])


sns.distplot(train['Heart_Rate'])


sns.distplot(train['Body_Temp'])


# Converting text data to numerical values
train.replace({'Sex':{'male':0,'female':1}}, inplace=True)
test.replace({'Sex':{'male':0,'female':1}}, inplace=True)


train.head()


correlation=train.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='plasma')
plt.title('Feature Correlation Heatmap')
plt.show()


# Separate features and target
X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
X_test = test.drop(['id'], axis=1)


# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8)
xgb_model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_valid, y_valid)], verbose=False)


# Predict
xgb_preds = xgb_model.predict(X_valid)


# Evaluate using RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

print("XGBoost RMSLE:", rmsle(y_valid, xgb_preds))


from lightgbm import early_stopping

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8
)


lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[early_stopping(stopping_rounds=20)]
)


lgb_preds = lgb_model.predict(X_valid)
print("LightGBM RMSLE:", rmsle(y_valid, lgb_preds))


# Load the sample submission file
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

# Predict using your xgb_model
final_preds = xgb_model.predict(X_test)

# If needed, clip predictions to avoid negatives (RMSLE can't handle negative values)
final_preds = np.clip(final_preds, 0, None)

# Assign predictions to submission dataframe
submission['Calories'] = final_preds

# Save submission file
submission.to_csv("submission.csv", index=False)


print(submission.head())
print("Any NaNs in predictions?", submission['Calories'].isna().sum())
print("Any negative predictions?", (submission['Calories'] < 0).sum())


# Load the sample submission file
submission1 = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

# Predict using your xgb_model
final_preds = lgb_model.predict(X_test)

# If needed, clip predictions to avoid negatives (RMSLE can't handle negative values)
final_preds = np.clip(final_preds, 0, None)

# Assign predictions to submission dataframe
submission1['Calories'] = final_preds

# Save submission file
submission1.to_csv("submission1.csv", index=False)


print(submission1.head())
print("Any NaNs in predictions?", submission1['Calories'].isna().sum())
print("Any negative predictions?", (submission1['Calories'] < 0).sum())

