import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        paths.append(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install py7zr


import py7zr
for filenames in paths:
    with py7zr.SevenZipFile(filenames, mode='r') as z_ref:
        z_ref.extractall(path='/kaggle/working')


%%time
data = pd.read_csv("/kaggle/working/train.csv")
data["date"] =  pd.to_datetime(data["date"])


monthly_counts_data = data['date'].dt.to_period('M').value_counts().sort_index()

plt.figure(figsize=(12, 6))
monthly_counts_data.plot(kind='bar', color='skyblue')
plt.title('Count of Data for Each Month for All data', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


%%time
THRESHOLD_TRAIN_DATE = pd.to_datetime("2017-07-01")
THRESHOLD_TEST_DATE = pd.to_datetime("2017-08-01")

data[data["date"] < THRESHOLD_TRAIN_DATE].to_csv("/kaggle/working/train_data.csv")
data[(data["date"] >= THRESHOLD_TRAIN_DATE)&(data["date"] < THRESHOLD_TEST_DATE)].to_csv("/kaggle/working/test_data.csv")

%xdel data


%%time
df = pd.read_csv("/kaggle/working/train_data.csv")
test_df = pd.read_csv("/kaggle/working/test_data.csv")


df["date"] = pd.to_datetime(df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])


monthly_counts = df['date'].dt.to_period('M').value_counts().sort_index().reset_index()
monthly_counts_test = test_df['date'].dt.to_period('M').value_counts().sort_index().reset_index()


monthly_counts['type'] = 'Train'
monthly_counts_test['type'] = 'Test'
combined = pd.concat([monthly_counts, monthly_counts_test])

plt.figure(figsize=(12, 6))
sns.barplot(data=combined, x='date', y='count', hue='type', palette=['blue', 'red'])
plt.title('Count of Data for Each Month (Train vs Test)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


df.head()


holiday_events = pd.read_csv("/kaggle/working/holidays_events.csv")
print(holiday_events.shape)
holiday_events.head()


items = pd.read_csv("/kaggle/working/items.csv")
print(items.shape)
items.head()


oil = pd.read_csv("/kaggle/working/oil.csv")
print(oil.shape)
oil.head()


stores = pd.read_csv("/kaggle/working/stores.csv")
print(stores.shape)
stores.head()


from sklearn.dummy import DummyRegressor
dummy_model = DummyRegressor(strategy="mean")
X_train = df.drop(columns=["unit_sales", "date"]) 
y_train = df["unit_sales"]
dummy_model.fit(X_train, y_train)


test_df = test_df.merge(items[['item_nbr', 'perishable']], on='item_nbr', how='left')
test_df['weight'] = test_df['perishable'].apply(lambda x: 1.25 if x == 1 else 1.00)


X_test = test_df.drop(columns=["unit_sales", "date"]) 
y_test = test_df["unit_sales"]  

y_pred = dummy_model.predict(X_test)


y_test = np.maximum(y_test, 0)
valid_indices = y_test >= 0
y_test = y_test[valid_indices]
y_pred = y_pred[valid_indices]
test_df = test_df[valid_indices]

log_error = np.log(y_pred + 1) - np.log(y_test + 1)
weighted_squared_error = test_df['weight'] * log_error ** 2

# NWRMSLE calculation
nwrmsle = np.sqrt(weighted_squared_error.sum() / test_df['weight'].sum())
print(f"Baseline Model NWRMSLE: {nwrmsle}")


holidays = pd.read_csv("/kaggle/working/holidays_events.csv")
holidays['date'] = pd.to_datetime(holidays['date'])

df = df.merge(holidays[['date', 'type', 'locale']], on='date', how='left')
test_df = test_df.merge(holidays[['date', 'type', 'locale']], on='date', how='left')

df['holiday_flag'] = df['type'].notna().astype(int)
test_df['holiday_flag'] = test_df['type'].notna().astype(int)


items = pd.read_csv("/kaggle/working/items.csv")
df = df.merge(items[['item_nbr', 'family', 'class', 'perishable']], on='item_nbr', how='left')
test_df = test_df.merge(items[['item_nbr', 'family', 'class', 'perishable']], on='item_nbr', how='left')


oil = pd.read_csv("/kaggle/working/oil.csv")
oil['date'] = pd.to_datetime(oil['date'])

df = df.merge(oil[['date', 'dcoilwtico']], on='date', how='left')
test_df = test_df.merge(oil[['date', 'dcoilwtico']], on='date', how='left')
df['dcoilwtico'].fillna(method='ffill', inplace=True)
test_df['dcoilwtico'].fillna(method='ffill', inplace=True)


stores = pd.read_csv("/kaggle/working/stores.csv")

df = df.merge(stores, on='store_nbr', how='left')
test_df = test_df.merge(stores, on='store_nbr', how='left')


for dataset in [df, test_df]:
    dataset['day_of_week'] = dataset['date'].dt.dayofweek
    dataset['month'] = dataset['date'].dt.month
    dataset['weekend_flag'] = (dataset['day_of_week'] >= 5).astype(int)


df.to_csv("/kaggle/working/enhanced_train_data.csv", index=False)
test_df.to_csv("/kaggle/working/enhanced_test_data.csv", index=False)

print("Feature engineering complete! Datasets saved for training.")


from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/working/enhanced_train_data.csv")
test_df = pd.read_csv("/kaggle/working/enhanced_test_data.csv")

X = df.drop(columns=["unit_sales", "date"])
y = df["unit_sales"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


pip install lightgbm


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

# Define dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

lgb_model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
    early_stopping_rounds=50,
    verbose_eval=50,
)
y_pred_val = lgb_model.predict(X_val)



y_pred_val = np.maximum(y_pred_val, 0)

weights = test_df["weight"].iloc[:len(y_pred_val)] 
log_error = np.log1p(y_pred_val) - np.log1p(y_val)
weighted_squared_error = weights * log_error**2
nwrmsle = np.sqrt(weighted_squared_error.sum() / weights.sum())

print(f"LGBM Model NWRMSLE: {nwrmsle}")


pip install catboost


from catboost import CatBoostRegressor
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="RMSE",
    early_stopping_rounds=50,
    verbose=50,
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

y_pred_cat = cat_model.predict(X_val)


y_pred_cat = np.maximum(y_pred_cat, 0)
log_error = np.log1p(y_pred_cat) - np.log1p(y_val)
weighted_squared_error = weights * log_error**2
nwrmsle_cat = np.sqrt(weighted_squared_error.sum() / weights.sum())

print(f"CatBoost Model NWRMSLE: {nwrmsle_cat}")


import optuna

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
    }

    lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    y_pred = lgb_model.predict(X_val)
    y_pred = np.maximum(y_pred, 0)
    log_error = np.log1p(y_pred) - np.log1p(y_val)
    weighted_squared_error = weights * log_error**2
    nwrmsle = np.sqrt(weighted_squared_error.sum() / weights.sum())
    return nwrmsle

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
print(f"Best NWRMSLE: {study.best_value}")


X_test = test_df.drop(columns=["unit_sales", "date"])
y_test_pred = lgb_model.predict(X_test)
y_test_pred = np.maximum(y_test_pred, 0)

submission = pd.DataFrame({"id": test_df["id"], "unit_sales": y_test_pred})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved!")

