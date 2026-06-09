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

train = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s4e12/sample_submission.csv")



train.head()
train.info()



train["Premium Amount"].describe()



import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(train["Premium Amount"], bins=50)
plt.show()



missing = train.isnull().sum().sort_values(ascending=False)
missing[missing > 0]



num_cols = train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = train.select_dtypes(include=["object"]).columns

num_cols, cat_cols



num_cols = num_cols.drop(['Premium Amount', 'id'])
num_cols



for col in num_cols:
    median_value = train[col].median()
    train[col].fillna(median_value, inplace=True)
    test[col].fillna(median_value, inplace=True)
train[num_cols].isnull().sum()



for col in cat_cols:
    train[col].fillna("Unknown", inplace=True)
    test[col].fillna("Unknown", inplace=True)
train[cat_cols].isnull().sum()



train["Policy Start Date"] = pd.to_datetime(train["Policy Start Date"])
test["Policy Start Date"] = pd.to_datetime(test["Policy Start Date"])

train["Policy Start Year"] = train["Policy Start Date"].dt.year
train["Policy Start Month"] = train["Policy Start Date"].dt.month

test["Policy Start Year"] = test["Policy Start Date"].dt.year
test["Policy Start Month"] = test["Policy Start Date"].dt.month

train.drop(columns=["Policy Start Date"], inplace=True)
test.drop(columns=["Policy Start Date"], inplace=True)
train[["Policy Start Year", "Policy Start Month"]].head()



cat_cols = train.select_dtypes(include=["object"]).columns
cat_cols



train_encoded = pd.get_dummies(train, columns=cat_cols, drop_first=True)
test_encoded  = pd.get_dummies(test,  columns=cat_cols, drop_first=True)
train_encoded.shape, test_encoded.shape




train_encoded, test_encoded = train_encoded.align(
    test_encoded,
    join='left',
    axis=1,
    fill_value=0
)

train_encoded.shape, test_encoded.shape



X = train_encoded.drop(columns=["Premium Amount"])
y = train_encoded["Premium Amount"]

X_test_final = test_encoded.copy()

X.shape, y.shape, X_test_final.shape



"Premium Amount" in test_encoded.columns


X_test_final = test_encoded.drop(columns=["Premium Amount"])
X_test_final.shape





X.shape, X_test_final.shape



set(X.columns) == set(X_test_final.columns)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.shape, X_val.shape, y_train.shape, y_val.shape



import numpy as np

def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)  # ØªÙ…Ù†Ø¹ Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ø³Ø§Ù„Ø¨Ø© (Ù…Ù‡Ù…)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))



from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
print("RMSLE:", rmsle(y_val, val_pred))



from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train)

val_pred_lgb = lgb_model.predict(X_val)

print("RMSLE (LightGBM):", rmsle(y_val, val_pred_lgb))



import numpy as np
from sklearn.model_selection import train_test_split

# 1) target log
y_log = np.log1p(y)

# 2) split Ø¹Ù„Ù‰ y_log
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

X_train.shape, X_val.shape, y_train_log.shape, y_val_log.shape



from lightgbm import LGBMRegressor
import numpy as np

# RMSLE Ø¹Ù„Ù‰ Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ø£ØµÙ„ÙŠØ©
def rmsle_original(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

lgb_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# ØªØ¯Ø±ÙŠØ¨ Ø¹Ù„Ù‰ Ø§Ù„Ù„ÙˆØº
lgb_model.fit(X_train, y_train_log)

# ØªÙ†Ø¨Ø¤ Ø¹Ù„Ù‰ Ø§Ù„Ù„ÙˆØº
val_pred_log = lgb_model.predict(X_val)

# Ø±Ø¬Ù‘Ø¹ Ù„Ù„Ø£ØµÙ„
val_pred = np.expm1(val_pred_log)
y_val_original = np.expm1(y_val_log)

print("RMSLE (original scale):", rmsle_original(y_val_original, val_pred))



import numpy as np
import pandas as pd

# 1) ØªÙˆÙ‚Ø¹ Ø¹Ù„Ù‰ test (log)
test_pred_log = lgb_model.predict(X_test_final)

# 2) Ø±Ø¬Ù‘Ø¹ Ù„Ù„Ø£ØµÙ„
test_pred = np.expm1(test_pred_log)

# (Ø§Ø®ØªÙŠØ§Ø±ÙŠ) ØªØ£ÙƒØ¯ Ù…Ø§ Ù�ÙŠ Ø³Ø§Ù„Ø¨
test_pred = np.maximum(test_pred, 0)

# 3) Ø§Ø¹Ù…Ù„ submission
submission = pd.DataFrame({
    "id": test["id"],
    "Premium Amount": test_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()



import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# --- X, y Ù„Ø§Ø²Ù… ÙŠÙƒÙˆÙ†ÙˆØ§ Ø¬Ø§Ù‡Ø²ÙŠÙ† Ù…Ù† Ù‚Ø¨Ù„ ---
# X = train_encoded.drop(columns=["Premium Amount"])
# y = train_encoded["Premium Amount"]

# Ø¬Ù‡Ù‘Ø² X_test_final
if "Premium Amount" in test_encoded.columns:
    X_test_final = test_encoded.drop(columns=["Premium Amount"])
else:
    X_test_final = test_encoded.copy()

# (Ù…Ù‡Ù…) Ø´ÙŠÙ„ id Ù…Ù† Ø§Ù„Ù…ÙŠØ²Ø§Øª Ø¥Ø°Ø§ Ù…ÙˆØ¬ÙˆØ¯
if "id" in X.columns:
    X = X.drop(columns=["id"])
if "id" in X_test_final.columns:
    X_test_final = X_test_final.drop(columns=["id"])

# (Ø§Ù„Ø£Ù‡Ù…) Ù†Ù�Ø³ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© + Ù†Ù�Ø³ Ø§Ù„ØªØ±ØªÙŠØ¨
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

# Ù†Ø¸Ù‘Ù� Ø£Ø³Ù…Ø§Ø¡ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ù…Ù† Ø§Ù„Ù…Ø³Ø§Ù�Ø§Øª (Ø§Ø®ØªÙŠØ§Ø±ÙŠ Ù„ÙƒÙ† Ù…Ù�ÙŠØ¯)
X.columns = X.columns.str.replace(" ", "_")
X_test_final.columns = X_test_final.columns.str.replace(" ", "_")

# log target
y_log = np.log1p(y)

# split Ø¹Ù„Ù‰ y_log
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

print("Shapes:", X_train.shape, X_val.shape, y_train_log.shape, y_val_log.shape)

lgb_base = LGBMRegressor(
    objective="regression",
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": [200, 300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [31, 50, 70, 100],
    "max_depth": [-1, 10, 20, 30],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_samples": [10, 20, 40, 60],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.0, 0.1, 0.5, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=lgb_base,
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_mean_squared_error",   # âœ… Ø¨Ø¯Ù„ neg_mean_squared_log_error
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train_log)

best_model = random_search.best_estimator_
print("\nBest Params:\n", random_search.best_params_)

# Validate: log -> expm1 -> RMSLE Ø§Ù„Ø­Ù‚ÙŠÙ‚ÙŠ
val_pred_log = best_model.predict(X_val)
val_pred = np.expm1(val_pred_log)
y_val_real = np.expm1(y_val_log)

print("\nRMSLE (Validation):", rmsle(y_val_real, val_pred))

# Train full
best_model.fit(X, y_log)

# Predict test
test_pred_log = best_model.predict(X_test_final)
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0)

submission = pd.DataFrame({
    "id": test["id"],            # Ù…Ù† dataframe Ø§Ù„Ø£ØµÙ„ÙŠ test
    "Premium Amount": test_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()


