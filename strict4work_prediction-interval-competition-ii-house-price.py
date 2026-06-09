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


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")
sample = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")


train.shape, test.shape


train.info()


train.describe().transpose()


from sklearn.model_selection import train_test_split

X = train.drop(columns=["sale_price"], axis=1)
y = train[["sale_price"]]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_val = test.copy()

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_val.shape)

# Process sale_date
def process_date(df):
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_mth"] = df["sale_date"].dt.month
    df["sale_day"] = df["sale_date"].dt.day
    
    # Drop after used
    df.drop(columns=["sale_date", "id"], inplace=True)
    
    return df

X_train = process_date(X_train)
X_test = process_date(X_test)
X_val = process_date(X_val)


X_train.info()


# from sklearn.preprocessing import OrdinalEncoder
categorical = X_train.select_dtypes(exclude=["int32", "int64", "float64"]).columns.tolist()

X_train[categorical] = X_train[categorical].astype('category')
X_test[categorical] = X_test[categorical].astype('category')
X_val[categorical] = X_val[categorical].astype('category')


import lightgbm as lgb
from lightgbm import LGBMRegressor

# Regression Model (LightGBM)
hyper_params = { 
        "boosting_type" : "gbdt",
        "objective": "quantile",
        "learning_rate": 0.01,
        "max_depth": 4,
        "num_leaves": 4,
        "n_estimators": 100,
        "random_state": 42
}

# Initialize RandomForestClassifier
lgb_model_lower_quantile = LGBMRegressor(**hyper_params, alpha=0.05)
# Fit the classifier to the training data
lgb_model_lower_quantile.fit(X_train, np.ravel(y_train), eval_set=(X_test, np.ravel(y_test)))

# Initialize RandomForestClassifier
lgb_model_upper_quantile = LGBMRegressor(**hyper_params, alpha=0.95)
# Fit the classifier to the training data
lgb_model_upper_quantile.fit(X_train, np.ravel(y_train), eval_set=(X_test, np.ravel(y_test)))


pred_lower = lgb_model_lower_quantile.predict(X_test)
pred_upper = lgb_model_upper_quantile.predict(X_test)


alpha = 0.1
def winkler_score(ytest, lower, upper, alpha=0.1):
    ytest = np.asarray(ytest)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    
    score = np.mean(upper - lower)
    
    below = ytest < lower
    above = ytest > upper
    
    score += np.mean((2 / alpha) * (lower - ytest) * below)
    score += np.mean((2 / alpha) * (ytest - upper) * above)

    return score

score = winkler_score(y_test, pred_lower, pred_upper)
print(f"\nðŸ“Š OOF Winkler Score: {score:.2f}")


submission_pred_lower = lgb_model_lower_quantile.predict(X_val)
submission_pred_upper = lgb_model_upper_quantile.predict(X_val)


submission = pd.DataFrame({"id":test["id"], "pi_lower":submission_pred_lower, "pi_upper":submission_pred_upper})


submission[submission["pi_lower"] > submission["pi_upper"]]


submission.to_csv("submission.csv", index=False)


submission


# def winkler_score_batch(y_true, lower, upper, alpha=0.1, batch_size=10000):
#     y_true = np.asarray(y_true)
#     lower = np.asarray(lower)
#     upper = np.asarray(upper)

#     total_score = 0.0
#     counter = 0
#     n = len(y_true)

#     for i in range(0, n, batch_size):
#         if n - counter > 0:
#             print(f"start batch {i} to {i+batch_size}")
#             y = y_true[i:i+batch_size]
#             l = lower[i:i+batch_size]
#             u = upper[i:i+batch_size]

#             score = np.mean(upper - lower)
            
#             below = y < lower
#             above = y > upper
            
#             score += np.mean((2 / alpha) * (lower - y) * below)
#             score += np.mean((2 / alpha) * (y - upper) * above)
            
#             i += 1
#             counter -= batch_size
#             total_score += score
#         else:
#             print("finish batch")
        
#     return np.mean(total_score)

# scores = winkler_score_batch(y_test, pred_lower, pred_upper, alpha=0.1)
# scores

