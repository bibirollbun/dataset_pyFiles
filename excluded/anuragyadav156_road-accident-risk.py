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


traindf = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
testdf = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
traindf


print(traindf.shape, testdf.shape)


print(traindf.isnull().sum(), testdf.isnull().sum())


print(traindf.dtypes, testdf.dtypes)


from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()


# Select columns with object or bool types
cat_cols = traindf.select_dtypes(include=["object", "bool"])

# Apply LabelEncoder column-wise
for col in cat_cols:
    traindf[col] = Le.fit_transform(traindf[col])
    testdf[col] = Le.transform(testdf[col])  # use transform, NOT fit_transform on test data


traindf


testdf


from sklearn.model_selection import train_test_split
X = traindf.drop("accident_risk", axis=1)
y = traindf["accident_risk"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 42)



import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)

# Set parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt'
}

# Train model
model = lgb.train(params, train_data, num_boost_round=100)


import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

models = {
    "LightGBM": lgb.LGBMRegressor(verbose=0),
    "RandomForest": RandomForestRegressor(),
    "LinearRegression": LinearRegression()
}

results = {}

for name, model1 in models.items():
    model1.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_preds)
    r2 = r2_score(y_test, y_preds)
    
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")



# Predict probabilities for class 1 using CatBoost
best_model = models['LightGBM']
predictions = best_model.predict(X_test)


submission = pd.DataFrame({
    'id': X_test["id"],
    'y': predictions  
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("✅ Submission file 'submission.csv' created using LightGBM!")


submission.columns




