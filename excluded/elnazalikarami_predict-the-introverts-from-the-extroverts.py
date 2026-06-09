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


### Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb


### Loading the data data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.info()


train.describe()


train["Personality"].value_counts(normalize=True)


##  Lable encoding the target
le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])



## Prepare features
X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])


### handling missing values

# Numeric imputation

numeric_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", 
                "Friends_circle_size", "Post_frequency"]
for col in numeric_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)


# Categorical imputation

cat_cols = X.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    X[col] = X[col].fillna("missing")
    X_test[col] = X_test[col].fillna("missing")



# Add interaction features to train and test
X["Alone_vs_Social"] = X["Time_spent_Alone"] / (X["Social_event_attendance"] + 1)
X["Friends_per_event"] = X["Friends_circle_size"] / (X["Social_event_attendance"] + 1)

X_test["Alone_vs_Social"] = X_test["Time_spent_Alone"] / (X_test["Social_event_attendance"] + 1)
X_test["Friends_per_event"] = X_test["Friends_circle_size"] / (X_test["Social_event_attendance"] + 1)



# Combine for consistent encoding
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)



### Encode Categorical Features
# Combine train and test features for consistent encoding
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)

# Encode categorical columns using OrdinalEncoder on combined data
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

# Split back into train and test
X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)

# Check that columns match
assert all(X.columns == X_test.columns), "Train and test columns mismatch!"






## Encoding categorical columns
#combined = pd.concat([X, X_test], axis=0)
#cat_cols = combined.select_dtypes(include="object").columns.tolist()
#encoder = OrdinalEncoder()
#combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

#X = combined.iloc[:len(X)].reset_index(drop=True)
#X_test = combined.iloc[len(X):].reset_index(drop=True)



##Optuna tuning
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42,
        "verbosity": 0
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        preds = model.predict(dval)
        oof_preds[val_idx] = preds

    oof_binary = (oof_preds > 0.5).astype(int)
    accuracy = accuracy_score(y, oof_binary)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # You can increase n_trials for better search

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")



best_params = trial.params

# Add parameters that are fixed and not in Optuna search space
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0
})

print("Best parameters for final training:")
print(best_params)



### Stratified K-Fold Cross-Validation
#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#oof_preds = np.zeros(len(X))
#test_preds = np.zeros(len(X_test))

#for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
   # X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    #y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    #dtrain = xgb.DMatrix(X_train, label=y_train)
    #dval = xgb.DMatrix(X_val, label=y_val)
    #dtest = xgb.DMatrix(X_test)

    #model = xgb.train(params, dtrain, num_boost_round=100,
                      #evals=[(dval, "valid")],
                      #early_stopping_rounds=10, verbose_eval=False)
    
    #oof_preds[val_idx] = model.predict(dval) > 0.5
    #test_preds += model.predict(dtest) / skf.n_splits



# Stratified K-Fold Cross Validation
params = best_params

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "validation")],
                      early_stopping_rounds=50,
                      verbose_eval=10)
    
    oof_preds[val_idx] = model.predict(dval)
    test_preds += model.predict(dtest) / skf.n_splits

oof_binary = (oof_preds > 0.5).astype(int)
cv_acc = accuracy_score(y, oof_binary)
print(f"Cross-Validation Accuracy: {cv_acc:.4f}")



### Create submission
final_preds = (test_preds > 0.5).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
submission.head()

