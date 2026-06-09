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


pd.set_option('display.max_columns',None)


sample_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet"
df_sample = pd.read_parquet(sample_path)

print("Shape:", df_sample.shape)
df_sample.head(2)


X = df_sample.filter(regex="^feature_")
y = df_sample["responder_6"]
w = df_sample["weight"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("weight shape:", w.shape)


# considering only only 200,000 rows as teh df_sample is very large (1.9 million rows)

df_200krows = df_sample.sample(n=200_000, random_state=42)

print("Original rows:", df_sample.shape[0])
print("Sampled rows:", df_200krows.shape[0])


# Train-test split (simple, random)

from sklearn.model_selection import train_test_split

X = df_200krows.filter(regex="^feature_")
y = df_200krows["responder_6"]
w = df_200krows["weight"]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Training using random forest model

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("First 10 predictions:")
print(y_pred[:10])

print("\nFirst 10 actual values:")
print(y_test.iloc[:10].values)


# Now I'll use a fast LightGBM model as it handles NaNs automatically and is much faster than RandomForest and also safe for real-time inference

import lightgbm as lgb

X = df_200krows.filter(regex="^feature_")
y = df_200krows["responder_6"]
w = df_200krows["weight"]

train_data = lgb.Dataset(X, label=y,weight=w)
params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 64, "verbosity": -1}

lgb_model = lgb.train(params, train_data, num_boost_round=100)


# predict function 

def predict(test, lags):
    X_test = test[[col for col in test.columns if col.startswith("feature_")]]
    preds = lgb_model.predict(X_test)

    return pd.DataFrame({
        "row_id": test["row_id"],
        "responder_6": preds
    })



import kaggle_evaluation.jane_street_inference_server as js

js.JSInferenceServer(predict).serve()

