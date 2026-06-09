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


#Step 1:Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

#Step 2: Load Datasets
train_path = "/kaggle/input/playground-series-s5e1/train.csv"  
test_path = "/kaggle/input/playground-series-s5e1/test.csv"  
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

#Step 3: Preprocess the Data
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
train_df["num_sold"] = train_df.groupby(["country", "store", "product"])["num_sold"].transform(lambda x: x.fillna(x.median()))
train_df = train_df[train_df["num_sold"] > 0].copy()
train_df["num_sold"] = np.log1p(train_df["num_sold"])

#Step 4: Feature Engineering
for df in [train_df, test_df]:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

# Encode Categorical Features
label_encoders = {}
for col in ["country", "store", "product"]:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

#Step 5: Define Features and Target Variable
features = ["country", "store", "product", "year", "month", "day", "weekday", "weekofyear"]
X = train_df[features]
y = train_df["num_sold"]

#Step 6: Train-Validation Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 7: Train LightGBM Model with K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgbm_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "max_depth": 6,
    "num_leaves": 31,
    "random_state": 42
}
models = []
cv_scores = []
test_predictions = []

test_X = test_df[features]
for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgbm_params)
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]

    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks
    )

    models.append(model)
    val_pred = model.predict(X_val_fold)
    score = mean_absolute_percentage_error(np.expm1(y_val_fold), np.expm1(val_pred))
    cv_scores.append(score)
    print(f"Fold MAPE: {score:.4f}")
    test_predictions.append(np.expm1(model.predict(test_X)))

print(f"Mean MAPE: {np.mean(cv_scores):.4f}")

#Step 8: Generate Final Predictions
final_predictions = np.mean(test_predictions, axis=0)
final_predictions = np.clip(final_predictions, 0, None)  # Ensure no negative predictions
final_predictions = np.round(final_predictions).astype(int)  # Match submission format

#Step 9: Create Submission File
submission = pd.DataFrame({
    "id": test_df["id"],
    "num_sold": final_predictions
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file 'submission.csv' created successfully!")

