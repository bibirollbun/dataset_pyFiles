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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings("ignore")


# Load the datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train = train_df.copy()
test = test_df.copy()


train.drop("id", axis=1, inplace=True)
train.head(10)


# Target encoding categorical features using KFold
def target_encoding(train, test, col, target, n_split=5):
    out_vals = pd.Series(index=train.index, dtype=float)
    test_encoded = pd.Series(index=test.index, dtype=float)
    kf = KFold(n_splits=n_split, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]

        means = X_train.groupby(col)[target].mean()
        out_vals.iloc[val_idx] = X_val[col].map(means)

    test_encoded = test[col].map(train.groupby(col)[target].mean())
    new_col = f"{col}_TE"
    train[new_col] = out_vals.fillna(train[target].mean())
    test[new_col] = test_encoded.fillna(train[target].mean())


# Filter numeric and categorical features
cat_cols = [col for col in train.columns if train[col].dtype in ["object", "category"]]
num_cols = [col for col in train.columns if train[col].dtype in ["float64"]]

target = "Listening_Time_minutes" # target feature

# Apply target encoding in categorical features
for col in cat_cols:
    target_encoding(train, test, col, target)


train.columns


# histogram with kde for target features
plt.figure(figsize=(12, 5))
sns.histplot(train["Listening_Time_minutes"], kde=True, bins=40)
plt.title("Distribution of Target Feature")
plt.xlabel("Listening Time (minutes)")
plt.ylabel("Frequency")
plt.show()


# Clip handle the outliers
upper_limit = train["Listening_Time_minutes"].quantile(0.995)
train["Listening_Time_minutes_Clipped"] = train[target].clip(upper=upper_limit)


# log transformation of traget feature
train["Listening_Time_minutes_Logged"] = np.log1p(train[target])


# Prepare for comparison
features = ["Podcast_Name_TE", "Episode_Title_TE", "Genre_TE", "Publication_Day_TE",
            "Publication_Time_TE", "Episode_Sentiment_TE", "Number_of_Ads", "Guest_Popularity_percentage",
            "Host_Popularity_percentage", "Episode_Length_minutes"
           ]
X = train[features]
y_clipped = train["Listening_Time_minutes_Clipped"]
y_logged = train["Listening_Time_minutes_Logged"]

# train test split
X_train, X_val, y_clipped_train, y_clipped_val, y_logged_train, y_logged_val = train_test_split(
    X, y_clipped, y_logged, test_size=0.2, random_state=42
)


# LightGBM
# Clipped
lgb_model_clip = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=1024,
    subsample=0.5,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1
)

# Logged
lgb_model_log = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=1024,
    subsample=0.5,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1
)


# Train on clipped
lgb_model_clip.fit(X_train, y_clipped_train)
y_pred_clipped = lgb_model_clip.predict(X_val)


# Train on logged
lgb_model_log.fit(X_train, y_logged_train)
y_pred_logged = lgb_model_log.predict(X_val)


# RMSE
rmse_clip= mean_squared_error(y_clipped_val, y_pred_clipped, squared=False)
rmse_log = mean_squared_error(np.expm1(y_logged_val), np.expm1(y_pred_logged), squared=False)

print(f"RMSE using CLIP: {rmse_clip}")
print(f"RMSE using LOG: {rmse_log}")


# Visualize
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_clipped_val, y=y_pred_clipped, alpha=0.5)
plt.plot([y_clipped_val.min(), y_clipped_val.max()], [y_clipped_val.min(), y_clipped_val.max()], 'r--')
plt.title("Predicted vs Actual [Target Clipped]")
plt.xlabel("Actual Listening Time")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_logged_val, y=y_pred_logged, alpha=0.5)
plt.plot([y_logged_val.min(), y_logged_val.max()], [y_logged_val.min(), y_logged_val.max()], 'r--')
plt.title("Predicted vs Actual [Target Logged]")
plt.xlabel("Actual Listening Time")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()


# submission
submission["Listening_Time_minutes"] = lgb_model_clip.predict(test[features])
submission.to_csv("submission.csv", index=False)
submission.head(10)




