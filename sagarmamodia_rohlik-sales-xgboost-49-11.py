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
import os
import sys
from tabulate import tabulate
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


INPUT_DIR = '/kaggle/input/rohlik-sales-forecasting-challenge-v2'
OUTPUT_DIR = '/kaggle/working'


df_sales = pd.read_csv(os.path.join(INPUT_DIR, "sales_train.csv"))
df_cal = pd.read_csv(os.path.join(INPUT_DIR, "calendar.csv"))
df_inv = pd.read_csv(os.path.join(INPUT_DIR, "inventory.csv"))


#converting datatype
df_sales["date"] = pd.to_datetime(df_sales["date"])
df_cal["date"] = pd.to_datetime(df_cal["date"])


# Handle Null Values
df_sales = df_sales.dropna()
df_cal["holiday_name"] = df_cal["holiday_name"].replace({np.NaN: "Unnamed"})


# Join df_sales, df_inv and df_cal
df_merged = (pd.merge(df_sales, df_inv, on="unique_id")
               .drop(columns=["warehouse_y"])
               .rename(columns={"warehouse_x": "warehouse"}))

df_merged = pd.merge(df_merged, df_cal, on=["date", "warehouse"])


df_merged["year"] = df_merged["date"].dt.year
df_merged["month"] = df_merged["date"].dt.month
df_merged["day"] = df_merged["date"].dt.day


categorical_cols = ["warehouse", "name", "L1_category_name_en", "L2_category_name_en",
                   "L3_category_name_en", "L4_category_name_en", "holiday_name"]


from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


df_merged[categorical_cols] = pd.DataFrame(ord_enc.fit_transform(df_merged[categorical_cols]),
                                          columns=categorical_cols)


df_merged = df_merged.drop(columns=["availability","date"])


# Train test split 
X_train = df_merged[df_merged["year"] < 2024].drop(columns= ["sales"])
y_train = df_merged[df_merged["year"] < 2024]["sales"]
X_val = df_merged[df_merged["year"]==2024].drop(columns=["sales"])
y_val = df_merged[df_merged["year"]==2024]["sales"]


xgb_reg = xgb.XGBRegressor(
    objective="reg:absoluteerror",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)


xgb_reg.fit(X_train, y_train)


# Evaluation
val_pred = xgb_reg.predict(X_val)
mae = mean_absolute_error(y_val, val_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"MAE as % of average target value: { 100*mae/np.mean(y_val)}")


df_test = pd.read_csv(os.path.join(INPUT_DIR, "sales_test.csv"))
df_test["date"] = pd.to_datetime(df_test["date"])
df_test["year"] = df_test["date"].dt.year
df_test["month"] = df_test["date"].dt.month
df_test["day"] = df_test["date"].dt.day


# Join df_test, df_inv and df_cal
df_merged_test = (pd.merge(df_test, df_inv, on="unique_id")
             .drop(columns=["warehouse_y"])
             .rename(columns={"warehouse_x": "warehouse"})
            )
df_merged_test = pd.merge(df_merged_test, df_cal, on=["date", "warehouse"])
df_merged_test = df_merged_test.drop(columns=["date"])


# categorical encoding
df_merged_test[categorical_cols] = ord_enc.transform(df_merged_test[categorical_cols])


df_merged_test = df_merged_test[X_train.columns]
test_pred = xgb_reg.predict(df_merged_test)


# prediction processing
unique_id = df_merged_test["unique_id"].astype('string')
year = df_merged_test["year"].astype('string')
month = df_merged_test["month"].apply(lambda x: str(x) if x>9 else f"0{x}")
day = df_merged_test["day"].apply(lambda x: str(x) if x>9 else f"0{x}")
date = year + "-" + month + "-" + day
test_id = unique_id + "_" + date

df_sub = pd.DataFrame({
    "id": test_id,
    "sales_hat": test_pred.astype("int64")
})


df_sub.to_csv("submission.csv", index=False)


df_sol = pd.read_csv(os.path.join(INPUT_DIR, "solution.csv"))


df_sol["id"].sort_values




