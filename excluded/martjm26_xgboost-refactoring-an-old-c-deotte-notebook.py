# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

%load_ext cudf.pandas

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings

msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
train = train.drop(columns="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test=test.drop(columns="id")

train.drop_duplicates(inplace=True)

display(train.info(), test.info(), train.describe().T)


from sklearn.preprocessing import LabelEncoder

CATS = list(train.drop(columns=["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", 'Listening_Time_minutes']).columns)


'''for col in cats:
    train[col] = train[col].fillna(train[col].mode())
    test[col] = test[col].fillna(test[col].mode())
    
    le = LabelEncoder()
    le.fit(train[col])
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

for col in test.columns:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(test[col].mean())

display(train.info(), train.head())'''

CATS


COMBO = []
for i,c in enumerate(CATS):
    combine = pd.concat([train[c],test[c]],axis=0)
    combine,_ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]
    n = f"{c}_Episode_Length_minutes"
    train[n] = train[c]*100 + train["Episode_Length_minutes"]
    test[n] = test[c]*100 + test["Episode_Length_minutes"]
    COMBO.append(n)
for i,c in enumerate(CATS):
    combine = pd.concat([train[c],test[c]],axis=0)
    combine,_ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]
    n = f"{c}_Host_Popularity_percentage"
    train[n] = train[c]*100 + train["Host_Popularity_percentage"]
    test[n] = test[c]*100 + test["Host_Popularity_percentage"]
    COMBO.append(n)
for i,c in enumerate(CATS):
    combine = pd.concat([train[c],test[c]],axis=0)
    combine,_ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]
    n = f"{c}_Guest_Popularity_percentage"
    train[n] = train[c]*100 + train["Guest_Popularity_percentage"]
    test[n] = test[c]*100 + test["Guest_Popularity_percentage"]
    COMBO.append(n)
for i,c in enumerate(CATS):
    combine = pd.concat([train[c],test[c]],axis=0)
    combine,_ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]
    n = f"{c}_Number_of_Ads"
    train[n] = train[c]*100 + train["Number_of_Ads"]
    test[n] = test[c]*100 + test["Number_of_Ads"]
    COMBO.append(n)
print()
print(f"We engineer {len(COMBO)} new columns!")
print( COMBO )


FEATURES = CATS + ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"] + COMBO
print(f"We now have {len(FEATURES)} columns:")
print( FEATURES )
train.info()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
import cudf
import cupy as cp
print(f"XGBoost version",xgb.__version__)


# STATISTICS TO AGGEGATE FOR OUR FEATURE GROUPS
STATS = ["mean","std","count","nunique","median","min","max","skew"]
STATS2 = ["mean","std"]


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train))
pred = np.zeros(len(test))

# OUTER K-FOLD
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print(f"### OUTER Fold {i+1} ###")

    X_train = train.loc[train_index, FEATURES + ['Listening_Time_minutes']].reset_index(drop=True).copy()
    y_train = train.loc[train_index, 'Listening_Time_minutes']

    X_valid = train.loc[test_index, FEATURES].reset_index(drop=True).copy()
    y_valid = train.loc[test_index, 'Listening_Time_minutes']

    X_test = test[FEATURES].reset_index(drop=True).copy()

    # INNER K-FOLD (TO PREVENT LEAKAGE WHEN USING Listening_Time_minutes)
    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=42)   
    for j, (train_index2, test_index2) in enumerate(kf2.split(X_train)):
        print(f" ## INNER Fold {j+1} (Outer Fold {i+1}) ##")

        X_train2 = X_train.loc[train_index2, FEATURES + ['Listening_Time_minutes']].copy()
        X_valid2 = X_train.loc[test_index2, FEATURES].copy()

        ### FEATURE SET 1 (Uses Listening_Time_minutes) ###
        col = "Episode_Length_minutes"
        tmp = X_train2.groupby(col).Listening_Time_minutes.agg(STATS)
        tmp.columns = [f"TE1_Episode_Length_minutes_{s}" for s in STATS]
        X_valid2 = X_valid2.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_train.loc[test_index2, c] = X_valid2[c].values

        col = "Host_Popularity_percentage"
        tmp = X_train2.groupby(col).Listening_Time_minutes.agg(STATS)
        tmp.columns = [f"TE1_Host_Popularity_percentage_{s}" for s in STATS]
        X_valid2 = X_valid2.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_train.loc[test_index2, c] = X_valid2[c].values

        col = "Guest_Popularity_percentage"
        tmp = X_train2.groupby(col).Listening_Time_minutes.agg(STATS)
        tmp.columns = [f"TE1_Guest_Popularity_percentage_{s}" for s in STATS]
        X_valid2 = X_valid2.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_train.loc[test_index2, c] = X_valid2[c].values

        col = "Number_of_Ads"
        tmp = X_train2.groupby(col).Listening_Time_minutes.agg(STATS)
        tmp.columns = [f"TE1_Number_of_Ads_{s}" for s in STATS]
        X_valid2 = X_valid2.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_train.loc[test_index2, c] = X_valid2[c].values

        ### FEATURE SET 2 (Uses Listening_Time_minutes) ###
        for col in COMBO:
            tmp = X_train2.groupby(col).Listening_Time_minutes.agg(STATS2)
            tmp.columns = [f"TE2_{col}_{s}" for s in STATS2]
            X_valid2 = X_valid2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                X_train.loc[test_index2, c] = X_valid2[c].values

    ### FEATURE SET 1 (Uses Listening_Time_minutes) ###
    col = "Episode_Length_minutes"
    tmp = X_train.groupby(col).Listening_Time_minutes.agg(STATS)
    tmp.columns = [f"TE1_Episode_Length_minutes_{s}" for s in STATS]
    X_valid = X_valid.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")

    col = "Host_Popularity_percentage"
    tmp = X_train.groupby(col).Listening_Time_minutes.agg(STATS)
    tmp.columns = [f"TE1_Host_Popularity_percentage_{s}" for s in STATS]
    X_valid = X_valid.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")

    col = "Guest_Popularity_percentage"
    tmp = X_train.groupby(col).Listening_Time_minutes.agg(STATS)
    tmp.columns = [f"TE1_Guest_Popularity_percentage_{s}" for s in STATS]
    X_valid = X_valid.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")

    col = "Number_of_Ads"
    tmp = X_train.groupby(col).Listening_Time_minutes.agg(STATS)
    tmp.columns = [f"TE1_Number_of_Ads_{s}" for s in STATS]
    X_valid = X_valid.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")

    ### FEATURE SET 2 (Uses Listening_Time_minutes) ###
    for col in COMBO:
        tmp = X_train.groupby(col).Listening_Time_minutes.agg(STATS2)
        tmp.columns = [f"TE2_{col}_{s}" for s in STATS2]
        X_valid = X_valid.merge(tmp, on=col, how="left")
        X_test = X_test.merge(tmp, on=col, how="left")

    # CONVERT TO CATS SO XGBOOST RECOGNIZES THEM
    X_train[CATS] = X_train[CATS].astype("category")
    X_valid[CATS] = X_valid[CATS].astype("category")
    X_test[CATS] = X_test[CATS].astype("category")

    # DROP Listening_Time_minutes THAT WAS USED FOR TARGET ENCODING
    X_train = X_train.drop(['Listening_Time_minutes'], axis=1)

    # Convert to CuPy (for GPU acceleration)
    X_train = cp.asarray(X_train)
    X_valid = cp.asarray(X_valid)
    X_test = cp.asarray(X_test)

    y_train = cp.asarray(y_train)
    y_valid = cp.asarray(y_valid)

    # Convert to XGBoost DMatrix (GPU enabled)
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, nthread=-1)
    dtest = xgb.DMatrix(X_test, nthread=-1)

    # Set XGBoost parameters (GPU enabled)
    params = {
        "max_depth": 6,
        "colsample_bytree": 0.5,
        "subsample": 0.5,
        "learning_rate": 0.02,
        "min_child_weight": 10,
        "tree_method": "hist",
        "device":"cuda"
    }
    
    # Train the model
    evallist = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10_000,  # Instead of n_estimators
        evals=evallist,
        early_stopping_rounds=100,
        verbose_eval=300
    )


    # Predict OOF and Test using DMatrix
    oof[test_index] = model.predict(dvalid)
    pred += model.predict(dtest)

pred /= FOLDS


# COMPUTE OVERALL CV SCORE
true = train.Listening_Time_minutes.values
s = np.sqrt(np.mean( (oof-true)**2.0 ) )
print(f"=> Overall CV Score = {s}")


print(f"\nIn total, we used {dtrain.num_col()} features, Wow!\n")


import xgboost as xgb
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(model, importance_type='gain',ax=ax)
plt.title("Top Feature Importances (XGBoost)")
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub.Listening_Time_minutes = pred
sub.to_csv(f"submission_v1.csv",index=False)
sub.head()


plt.figure(figsize=(6,4))
plt.hist(sub.Listening_Time_minutes,bins=100)
plt.title("Test Predictions")
plt.show()




