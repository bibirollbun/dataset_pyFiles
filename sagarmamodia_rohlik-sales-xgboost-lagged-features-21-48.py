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
df_test = pd.read_csv(os.path.join(INPUT_DIR, "sales_test.csv"))


df_cal_enriched = pd.read_csv('/kaggle/input/extended-calendar-dataset-for-rohlik-challenge/calendar_enriched.csv')
df_cal_enriched_2025 = pd.read_csv('/kaggle/input/extended-calendar-dataset-for-rohlik-challenge/calendar_enriched_2025-01-05.csv')


if "sales" in df_sales.columns:
    print("Not")


print(f"Train start date: {df_sales['date'].sort_values().min()}")
print(f"Train end date: {df_sales['date'].sort_values().max()}")
print(f"Test start date: {df_test['date'].sort_values().min()}")
print(f"Test end date: {df_test['date'].sort_values().max()}")


#converting datatype
df_sales["date"] = pd.to_datetime(df_sales["date"])
df_test["date"] = pd.to_datetime(df_test["date"])
df_cal_enriched_2025["date"] = pd.to_datetime(df_cal_enriched_2025["date"])


# Handle Null Values
df_sales = df_sales.dropna()
df_cal_enriched_2025["date_holiday_name"] = df_cal_enriched_2025["date_holiday_name"].replace({np.NaN: "Unnamed"})

# drop columns
df_sales = df_sales.drop(columns=["availability"])


# Feature generation functions
def add_date_features(df):
    df['date_year'] = df['date'].dt.year
    df['date_month'] = df['date'].dt.month
    df['date_day'] = df['date'].dt.day
    df['date_weekofyear'] = df['date'].dt.isocalendar().week
    df['date_weekday'] = df['date'].dt.weekday 
    df['date_dayofyear'] = df['date'].dt.dayofyear
    df['date_year_sin'] = np.sin((df['date_year'] - df['date_year'].min()) / (df['date_year'].max() - df['date_year'].min()) * 2 * np.pi)
    df['date_year_sin'] = np.sin(df['date_year'] / 1 * 2 * np.pi)
    df['date_month_sin'] = np.sin(df['date_month'] / 12 * 2 * np.pi)
    return df

periods = [14, 21, 28, 56, 112, 224, 365]
def add_lag_sales(df_sales):
    for period in periods:
        df_sales[f"sales_lag_{period}"] = df_sales.groupby("unique_id")["sales"].shift(period)
    
    return df_sales


df_test["sales"] = 0
df = pd.concat([df_sales, df_test], ignore_index=True).sort_values("date")
df = df.merge(df_cal_enriched_2025, on=['date', 'warehouse'], how='left')
df = df.merge(df_inv, on=['unique_id', 'warehouse'], how='left')


df = add_date_features(df)
df = add_lag_sales(df)


## Ensuring correct datatypes

for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype('category')


# Train Test split 
train_end_date = "2024-06-02"
test_start_date = "2024-06-03"
test_end_date = "2024-06-17"

X_train = df[df["date"] <= train_end_date].drop(columns= ["sales"])
y_train = df[df["date"] <= train_end_date]["sales"]
X_test = df[df["date"] >= test_start_date].drop(columns=["sales"])
y_test = df[df["date"] >= test_start_date]["sales"]


# Train-Test split visualized
df_plot = df.groupby("date", as_index=False).agg({"sales": "mean"})
plt.figure(figsize=(10, 5))
plt.plot(df_plot["date"], df_plot["sales"], label="Avg Sale")

# plt.axvline(x=pd.to_datetime(train_end_date), color="red", linestyle="--")
plt.axvline(x=pd.to_datetime(test_end_date), color="black", linestyle="--")

plt.grid(True)
plt.legend()


params = {
    "objective": "reg:absoluteerror",
    "eta": 0.021796506746095975,
    "max_depth": 20,
    "random_state": 42,
    "gamma": 0.1,
    "subsample": 0.6,
    "device": 'cuda',
}
num_boost_round = 400


dmat = xgb.DMatrix(X_train.drop(columns=["date"]), y_train, enable_categorical=True)
xgb_reg = xgb.train(params, dmat, num_boost_round=num_boost_round,  evals=[(dmat, 'train_set')])


xgb_reg.predict(dmat)


dmat = xgb.DMatrix(X_test.drop(columns=["date"]), enable_categorical=True)
test_pred = xgb_reg.predict(dmat)


# prediction processing
unique_id = X_test["unique_id"].astype('string')
date = X_test["date"].astype('string')
test_id = unique_id + "_" +  date

df_sub = pd.DataFrame({
    "id": test_id,
    "sales_hat": test_pred.astype("int64")
})


df_sub.to_csv("submission.csv", index=False)




