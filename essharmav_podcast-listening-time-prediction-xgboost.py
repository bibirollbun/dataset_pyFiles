# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_ds = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_ds = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


print("Training Dataset Shape:",train_ds.shape)
print("Test Dataset Shape:",test_ds.shape)


train_ds.isna().sum()


train_ds['Episode_Length_minutes'] = train_ds['Episode_Length_minutes'].replace(np.nan, train_ds['Episode_Length_minutes'].mean())
train_ds['Guest_Popularity_percentage'] = train_ds['Guest_Popularity_percentage'].replace(np.nan, train_ds['Guest_Popularity_percentage'].mean())

test_ds['Episode_Length_minutes'] = test_ds['Episode_Length_minutes'].replace(np.nan, test_ds['Episode_Length_minutes'].mean())
test_ds['Guest_Popularity_percentage'] = test_ds['Guest_Popularity_percentage'].replace(np.nan, test_ds['Guest_Popularity_percentage'].mean())


train_ds = train_ds.dropna()


train_ds.head()


numerical_cols = train_ds.select_dtypes(exclude='object').columns
categorical_cols = train_ds.select_dtypes(include='object').columns


train_ds[categorical_cols].nunique()


correlation_matrix = train_ds[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap = 'crest', fmt='.2f', linewidths= 0.5)


plt.figure(figsize=(10, 6))
sns.regplot(
    x=train_ds['Episode_Length_minutes'], 
    y=train_ds['Listening_Time_minutes'], 
    scatter_kws={'alpha':0.5},  # adjust transparency of points
    line_kws={'color': 'red'}   # regression line color
)
plt.title("Episode Length vs Listening Time")
plt.xlabel("Episode Length (minutes)")
plt.ylabel("Listening Time (minutes)")
plt.grid(True)
plt.show()


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Selecting features and target
features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Number_of_Ads",
    "Genre",
    "Publication_Day",
    "Publication_Time",
    "Episode_Sentiment",
]
cleaned_train_ds = train_ds[features]
X_test = test_ds[features].copy()
Y = train_ds[["Listening_Time_minutes"]]

# Train/validation split
X_train, X_val, Y_train, Y_val = train_test_split(
    cleaned_train_ds, Y, test_size=0.2, random_state=42, stratify=Y
)

# Columns
standardise_cols = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Number_of_Ads",
]
encode_cols = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

# Standard Scaling
std_scaler = StandardScaler()
X_train[standardise_cols] = std_scaler.fit_transform(X_train[standardise_cols])
X_val[standardise_cols] = std_scaler.transform(X_val[standardise_cols])
X_test[standardise_cols] = std_scaler.transform(X_test[standardise_cols])

# Encoding
for col in encode_cols:
    unique_vals = X_train[col].nunique()

    if unique_vals > 4:
        # Label Encoding
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])


    else:
        # One-Hot Encoding
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(X_train[[col]])

        # Transform
        X_train_ohe = ohe.transform(X_train[[col]])
        X_val_ohe = ohe.transform(X_val[[col]])
        X_test_ohe = ohe.transform(X_test[[col]])

        # Convert to DataFrames
        columns = [f"{col}_{cat}" for cat in ohe.categories_[0]]
        X_train_df = pd.DataFrame(X_train_ohe, columns=columns, index=X_train.index)
        X_val_df = pd.DataFrame(X_val_ohe, columns=columns, index=X_val.index)
        X_test_df = pd.DataFrame(X_test_ohe, columns=columns, index=X_test.index)

        # Replace original column with encoded version
        X_train = pd.concat([X_train.drop(columns=[col]), X_train_df], axis=1)
        X_val = pd.concat([X_val.drop(columns=[col]), X_val_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=[col]), X_test_df], axis=1)



def xgb_regressor(X_train, Y_train, X_val, Y_val):
    n_estimators = [150, 200, 250]
    max_depth = [100, 150]
    learning_rate = [0.015, 0.02]
    errors = {}
    for n in n_estimators:
        for d in max_depth:
            for lr in learning_rate:

                xgr = XGBRegressor(
                    n_estimators=n,
                    max_depth=d,
                    learning_rate=lr,
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=20,
                )

                # xgr.fit(X_train, Y_train.squeeze())

                xgr.fit(
                    X_train, Y_train.squeeze(), eval_set=[(X_val, Y_val)], verbose=False
                )

                y_pred = xgr.predict(X_val)

                mse = mean_squared_error(Y_val.squeeze(), y_pred)
                mae = mean_absolute_error(Y_val.squeeze(), y_pred)

                errors[(n, d, lr)] = [mse, mae]

                print(
                    f"True MAE for {n} estimators and {d} depth and learning rate {lr}: {mae}"
                )
                print(
                    f"True MSE for {n} estimators and {d} depth and learning rate {lr}: {mse}"
                )

    mse_list = [j[0] for i, j in errors.items()]
    best_model_parameters = list(errors.keys())[np.argmin(mse_list)]

    return errors, best_model_parameters


errors, best_model_parameters = xgb_regressor(X_train, Y_train, X_val, Y_val)

n_estimators, max_depth, learning_rate = best_model_parameters



xgr = XGBRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=42,
    n_jobs=-1,
)

xgr.fit(X_train, Y_train.squeeze())

# Predict on test set
Y_pred_xgr = xgr.predict(X_test)

# Predict on validation set and calculate R²
Y_val_pred = xgr.predict(X_val)
r2_xgr = r2_score(Y_val, Y_val_pred)
print(f"R² Score: {r2_xgr:.4f}")

# Prepare the final submission DataFrame
Y_pred_df = pd.DataFrame(np.round(Y_pred_xgr, 3), columns=["Listening_Time_minutes"])
xgr_result = pd.concat([test_ds[["id"]].reset_index(drop=True), Y_pred_df], axis=1)



xgr_result.to_csv('submission.csv', index=False)




