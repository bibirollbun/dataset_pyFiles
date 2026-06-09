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
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())
print("\nTest columns:", test.columns.tolist())

train.head()


print(train.isnull().sum())

print('\nDatatypes')
print(train.dtypes)


plt.figure(figsize=(8,5))
sns.histplot(train['BeatsPerMinute'], kde=True, bins=30)
plt.title("Distribution of BeatsPerMinute")
plt.xlabel("BeatsPerMinute")
plt.ylabel("Count")
plt.show()

print("Target Stats:")
print(train['BeatsPerMinute'].describe())


num_cols = train.select_dtypes(include=['int64','float64']).columns.drop('BeatsPerMinute')

train[num_cols].hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()


# Correlation heatmap
corr = train[num_cols.tolist() + ['BeatsPerMinute']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=14)
plt.show()

# Top correlated features with BeatsPerMinute
corr_target = corr['BeatsPerMinute'].drop('BeatsPerMinute').sort_values(ascending=False)
corr_target.head(10).plot(kind='barh', figsize=(6,4), color="green")
plt.title("Top Features Correlated with BeatsPerMinute")
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x=train['BeatsPerMinute'])
plt.title("Outliers in BeatsPerMinute")
plt.show()


plt.figure(figsize=(6,4))
sns.histplot(train['BeatsPerMinute'], color='blue', label='Train', kde=True)
plt.legend()
plt.title("Train BPM Distribution")
plt.show()

print("Test IDs range:", test['id'].min(), "-", test['id'].max())


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Split X and y
# X = train.drop(columns=["BeatsPerMinute"])
# y = train["BeatsPerMinute"]

# # Train Linear Regression
# lr = LinearRegression()
# lr.fit(X, y)

# # Predict
# train_preds = lr.predict(X)
# test_preds = lr.predict(test)

# # Evaluate
# rmse = np.sqrt(mean_squared_error(y, train_preds))
# # print(f"ðŸ“‰ Linear Regression RMSE (Train): {rmse:.4f}")

# # Submission
# submission = pd.DataFrame({
#     "id": test["id"],   
#     "BeatsPerMinute": test_preds
# })
# submission.to_csv("submission.csv", index=False)
# #print(" submission.csv saved")
# print(submission.head())


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# # 1. Load Data

# # Features and target
# X = train.drop(columns=["BeatsPerMinute", "id"])  # drop target + ID column
# y = train["BeatsPerMinute"]

# # Save test IDs for submission
# test_ids = test["id"]

# # 2. Train-Validation Split

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 3. Train Random Forest

# rf = RandomForestRegressor(
#     n_estimators=200,   # more trees for better performance
#     max_depth=10,       # limit depth to avoid overfitting
#     max_features="sqrt",
#     random_state=42,
#     n_jobs=-1
# )

# rf.fit(X_train, y_train)

# # 4. Evaluate with RMSE

# y_val_pred = rf.predict(X_val)
# rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
# # print(f" Validation RMSE: {rmse:.4f}")

# # 5. Predict on Test & Submission

# test_X = test.drop(columns=["id"])
# test_preds = rf.predict(test_X)

# submission = pd.DataFrame({
#     "id": test_ids,
#     "BeatsPerMinute": test_preds
# })

# submission.to_csv("submission.csv", index=False)
# print(" Submission file saved as submission.csv")
# print(submission.head())


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor


# # 1. Load Data

# # Features and target
# X = train.drop(columns=["BeatsPerMinute", "id"])  # drop target + ID
# y = train["BeatsPerMinute"]

# # Save test IDs
# test_ids = test["id"]

# # 2. Train-Validation Split

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 3. Train XGBoost Regressor

# xgb = XGBRegressor(
#     n_estimators=500,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )

# xgb.fit(X_train, y_train)

# # 4. Evaluate with RMSE

# y_val_pred = xgb.predict(X_val)
# rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
# # print(f"âœ… Validation RMSE (XGBoost): {rmse:.4f}")

# # 5. Predict on Test & Submission

# test_X = test.drop(columns=["id"])
# test_preds = xgb.predict(test_X)

# submission = pd.DataFrame({
#     "id": test_ids,
#     "BeatsPerMinute": test_preds
# })

# submission.to_csv("submission.csv", index=False)
# # print("âœ… Submission file saved as submission.csv")
# print(submission.head())


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Separate features and target
X = train.drop(columns=["BeatsPerMinute", "id"])  # drop target + ID
y = train["BeatsPerMinute"]

# Keep test IDs
test_ids = test["id"]   # replace "Id" with your test ID column name
X_test = test.drop(columns=["id"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=10000,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

y_pred_valid = lgb_model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
# print("Validation RMSE:", rmse)

y_pred_test = lgb_model.predict(X_test)

submission = pd.DataFrame({
    "id": test_ids,
    "target": y_pred_test
})
submission.to_csv("submission.csv", index=False)
# print("âœ… submission.csv file created!")
print(submission.head())

