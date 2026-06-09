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

df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")

print(df.columns)


print(df.shape)


print(df.describe())


print(df.columns)


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')

X = df['num_reported_accidents']
y = df['accident_risk']

plt.scatter(X, y)
plt.xlabel('num_reported_accidents')
plt.ylabel('Accident Risk')
plt.show()  


common_accidents = df['num_reported_accidents'].value_counts()
common_risk = df['accident_risk'].value_counts()

print("Most common values in 'num_reported_accidents':")
print(common_accidents.head(10))
print("\nMost common values in 'accident_risk':")
print(common_risk.head(10))


print(df.head)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# 2. Separate features and target
X = train.drop(["accident_risk"], axis=1)
y = train["accident_risk"]

# 3. Encode categorical columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 4. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model (no early stopping, no eval_metric)
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# 6. Evaluate
y_pred_val = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse)

# 7. Predict on test set
test_preds = model.predict(test)

# 8. Save submission
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": test_preds
})
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv saved successfully!")



import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# --- Identify Columns ---
target_col = "accident_risk"
id_col = "id"

# --- Encode Categorical Columns ---
for col in train.columns:
    if train[col].dtype == 'object':
        lbl = LabelEncoder()
        full_data = pd.concat([train[col], test[col]], axis=0)
        lbl.fit(full_data.astype(str))
        train[col] = lbl.transform(train[col].astype(str))
        test[col] = lbl.transform(test[col].astype(str))

# --- Handle Missing Values ---
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

# --- Split Features and Target ---
X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# --- Train/Test Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Optimized XGBoost Regressor ---
xgb = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    reg_lambda=2,
    min_child_weight=1,
    random_state=42,
    tree_method="hist",
    objective="reg:squarederror",
    eval_metric="rmse"
)

# --- Train Model ---
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=200,
    early_stopping_rounds=100
)

# --- Validation Performance ---
y_val_pred = xgb.predict(X_val)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print("Validation RMSE:", round(rmse, 5))

# --- Predict on Test Data ---
test_preds = xgb.predict(X_test)

# --- Create Submission File ---
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: test_preds
})

submission.to_csv("submission_regression.csv", index=False)
print("âœ… submission_regression.csv created successfully!")



import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# --- Encode Categorical Columns ---
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

# --- Split Data ---
X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define Objective Function for Optuna ---
def objective(trial):
    params = {
        "n_estimators": 800,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "tree_method": "hist",
        "random_state": 42,
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

# --- Run Optimization ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)  # increase trials for better results

print("âœ… Best RMSE:", study.best_value)
print("ğŸ�† Best Params:", study.best_params)

# --- Train Final Model with Best Params ---
best_params = study.best_params
best_params["n_estimators"] = 1200
best_params["tree_method"] = "hist"
best_params["random_state"] = 42

model = XGBRegressor(**best_params)
model.fit(X, y, verbose=False)

# --- Predict and Save Submission ---
preds = model.predict(X_test)
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})
submission.to_csv("submission_optuna.csv", index=False)
print("âœ… submission_optuna.csv created successfully!")



import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# --- Encode categorical columns ---
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# --- Objective function for Optuna ---
def objective(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "tree_method": "hist",
        "random_state": 42,
    }

    # --- K-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# --- Run Optuna Search ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # can raise to 50â€“100 for deeper search

print("âœ… Best RMSE:", study.best_value)
print("ğŸ�† Best Params:", study.best_params)

# --- Train final model with best params ---
best_params = study.best_params
best_params["n_estimators"] = 1200
best_params["tree_method"] = "hist"
best_params["random_state"] = 42

final_model = XGBRegressor(**best_params)
final_model.fit(X, y, verbose=False)

# --- Predict on test set ---
preds = final_model.predict(X_test)

# --- Save submission ---
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})
submission.to_csv("submission_kfold_optuna.csv", index=False)
print("âœ… submission_kfold_optuna.csv created successfully!")



import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# --- Encode categorical columns ---
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# --- Objective function for Optuna ---
def objective(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.02),
        "max_depth": trial.suggest_int("max_depth", 7, 11),
        "subsample": trial.suggest_float("subsample", 0.8, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.9, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 9),
        "gamma": trial.suggest_float("gamma", 0.0, 0.02),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 0.8),
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # --- K-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# --- Run Optuna Search ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # can raise to 50â€“100 for deeper search

print("âœ… Best RMSE:", study.best_value)
print("ğŸ�† Best Params:", study.best_params)

# --- Train final model with best params ---
best_params = study.best_params
best_params["n_estimators"] = 1200
best_params["tree_method"] = "hist"
best_params["random_state"] = 42

final_model = XGBRegressor(**best_params)
final_model.fit(X, y, verbose=False)

# --- Predict on test set ---
preds = final_model.predict(X_test)

# --- Save submission ---
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})
submission.to_csv("submission_kfold_opt1.csv", index=False)
print("âœ… submission_kfold_opt1.csv created successfully!")


import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# --- Encode categorical columns ---
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# --- Objective function for Optuna ---
def objective(trial):
    params = {
        "n_estimators": 1000,
         "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.019),
        "max_depth": trial.suggest_int("max_depth", 9, 11),
        "subsample": trial.suggest_float("subsample", 0.90, 0.93),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.97, 0.99),
        "min_child_weight": trial.suggest_int("min_child_weight", 8, 18),
        "gamma": trial.suggest_float("gamma", 0.008, 0.011),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.50, 0.55),
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # --- K-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# --- Run Optuna Search ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # can raise to 50â€“100 for deeper search

print("âœ… Best RMSE:", study.best_value)
print("ğŸ�† Best Params:", study.best_params)

# --- Train final model with best params ---
best_params = study.best_params
best_params["n_estimators"] = 1200
best_params["tree_method"] = "hist"
best_params["device"] = "cuda"
best_params["random_state"] = 42

final_model = XGBRegressor(**best_params)
final_model.fit(X, y, verbose=False)

# --- Predict on test set ---
preds = final_model.predict(X_test)

# --- Save submission ---
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})
submission.to_csv("submission_kfold_opt2.csv", index=False)
print("âœ… submission_kfold_opt2.csv created successfully!")


import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# --- Encode categorical columns ---
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# --- Objective function for Optuna ---
def objective(trial):
    params = {
        "n_estimators": 1000,
         "learning_rate": trial.suggest_float("learning_rate", 0.014, 0.018),
        "max_depth": trial.suggest_int("max_depth", 9, 10),
        "subsample": trial.suggest_float("subsample", 0.86, 0.92),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.97, 0.99),
        "min_child_weight": trial.suggest_int("min_child_weight", 8, 18),
        "gamma": trial.suggest_float("gamma", 0.008, 0.011),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.50, 0.55),
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # --- K-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# --- Run Optuna Search ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # can raise to 50â€“100 for deeper search

print("âœ… Best RMSE:", study.best_value)
print("ğŸ�† Best Params:", study.best_params)

# --- Train final model with best params ---
best_params = study.best_params
best_params["n_estimators"] = 1200
best_params["tree_method"] = "hist"
best_params["device"] = "cuda"
best_params["random_state"] = 42

final_model = XGBRegressor(**best_params)
final_model.fit(X, y, verbose=False)

# --- Predict on test set ---
preds = final_model.predict(X_test)

# --- Save submission ---
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})
submission.to_csv("submission_kfold_opt3.csv", index=False)
print("âœ… submission_kfold_opt3.csv created successfully!")


import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# --- Encode categorical columns ---
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# --- Objective function for Optuna ---
def objective(trial):
    params = {
        "n_estimators": 1000,
         "learning_rate": trial.suggest_float("learning_rate", 0.014, 0.018),
        "max_depth": trial.suggest_int("max_depth", 9, 10),
        "subsample": trial.suggest_float("subsample", 0.86, 0.92),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.97, 0.99),
        "min_child_weight": trial.suggest_int("min_child_weight", 50, 100),
        "gamma": trial.suggest_float("gamma", 0.008, 0.011),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.50, 0.55),
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # --- K-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# --- Run Optuna Search ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # can raise to 50â€“100 for deeper search

print("âœ… Best RMSE:", study.best_value)
print("ğŸ�† Best Params:", study.best_params)

# --- Train final model with best params ---
best_params = study.best_params
best_params["n_estimators"] = 1200
best_params["tree_method"] = "hist"
best_params["device"] = "cuda"
best_params["random_state"] = 42

final_model = XGBRegressor(**best_params)
final_model.fit(X, y, verbose=False)

# --- Predict on test set ---
preds = final_model.predict(X_test)

# --- Save submission ---
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})
submission.to_csv("submission_kfold_opt4.csv", index=False)
print("âœ… submission_kfold_opt4.csv created successfully!")


import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# --- Encode categorical columns ---
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# --- Objective function for Optuna ---
def objective(trial):
    params = {
        "n_estimators": 1000,
         "learning_rate": trial.suggest_float("learning_rate", 0.014, 0.018),
        "max_depth": trial.suggest_int("max_depth", 9, 10),
        "subsample": trial.suggest_float("subsample", 0.86, 0.92),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.97, 0.99),
        "min_child_weight": trial.suggest_int("min_child_weight", 50, 100),
        "gamma": trial.suggest_float("gamma", 0.008, 0.011),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.50, 0.55),
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # --- K-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# --- Run Optuna Search ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # can raise to 50â€“100 for deeper search

print("âœ… Best RMSE:", study.best_value)
print("ğŸ�† Best Params:", study.best_params)

# --- Train final model with best params ---
best_params = study.best_params
best_params["n_estimators"] = 1200
best_params["tree_method"] = "hist"
best_params["device"] = "cuda"
best_params["random_state"] = 42

final_model = XGBRegressor(**best_params)
final_model.fit(X, y, verbose=False)

# --- Predict on test set ---
preds = final_model.predict(X_test)

# --- Save submission ---
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})
submission.to_csv("submission_kfold_opt4.csv", index=False)
print("âœ… submission_kfold_opt4.csv created successfully!")


# ==========================
# ğŸ§© Imports
# ==========================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# ==========================
# ğŸ“‚ Load Data
# ==========================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# ==========================
# ğŸ”  Encode Categorical Columns
# ==========================
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ==========================
# ğŸ§± Base Models
# ==========================
base_models = [
    ("xgb", XGBRegressor(
        n_estimators=700,
        learning_rate=0.015,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.98,
        min_child_weight=10,
        gamma=0.009,
        reg_lambda=0.53,
        tree_method="hist",
        device="cuda",
        random_state=42
    )),
    ("lgbm", LGBMRegressor(
        n_estimators=700,
        learning_rate=0.015,
        num_leaves=50,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        device="gpu"
    )),
    ("cat", CatBoostRegressor(
        iterations=700,
        learning_rate=0.015,
        depth=8,
        random_seed=42,
        verbose=False,
        task_type="GPU"
    ))
]

# ==========================
# ğŸ”� K-Fold Training for Stacking
# ==========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_features = np.zeros((X.shape[0], len(base_models)))
test_meta_features = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    print(f"ğŸ”¹ Training base model: {name}")
    test_fold_preds = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        meta_features[val_idx, i] = val_pred

        test_pred = model.predict(X_test)
        test_fold_preds.append(test_pred)

    test_meta_features[:, i] = np.mean(test_fold_preds, axis=0)

print("âœ… Base models training complete!")

# ==========================
# ğŸ§  Neural Network Meta Model
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaNN(nn.Module):
    def __init__(self, input_dim):
        super(MetaNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

X_meta = torch.tensor(meta_features, dtype=torch.float32).to(device)
y_meta = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
X_test_meta = torch.tensor(test_meta_features, dtype=torch.float32).to(device)

meta_model = MetaNN(input_dim=len(base_models)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

# ==========================
# ğŸ”� Train NN Meta Model
# ==========================
epochs = 200
for epoch in range(epochs):
    meta_model.train()
    optimizer.zero_grad()
    preds = meta_model(X_meta)
    loss = criterion(preds, y_meta)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        rmse = torch.sqrt(loss).item()
        print(f"Epoch [{epoch+1}/{epochs}] - RMSE: {rmse:.4f}")

# ==========================
# ğŸ”® Predictions
# ==========================
meta_model.eval()
with torch.no_grad():
    final_preds = meta_model(X_test_meta).cpu().numpy().flatten()

# ==========================
# ğŸ’¾ Save Submission
# ==========================
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_preds
})
submission.to_csv("submission_nn_stacking.csv", index=False)
print("ğŸ�� submission_nn_stacking.csv created successfully!")



# ======================================
# ğŸ§  Advanced NN Stacking (Improved RMSE)
# ======================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# ==========================
# ğŸ“‚ Load Data
# ==========================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target_col = "accident_risk"
id_col = "id"

# ==========================
# ğŸ”  Encode Categorical Columns
# ==========================
for col in train.columns:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

X = train.drop([target_col, id_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# ==========================
# ğŸ§± Base Models
# ==========================
base_models = [
    ("xgb", XGBRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.98,
        min_child_weight=10,
        gamma=0.009,
        reg_lambda=0.53,
        tree_method="hist",
        device="cuda",
        random_state=42
    )),
    ("lgbm", LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        num_leaves=70,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.4,
        random_state=42,
        device="gpu"
    )),
    ("cat", CatBoostRegressor(
        iterations=1200,
        learning_rate=0.015,
        depth=10,
        l2_leaf_reg=3.5,
        random_seed=42,
        verbose=False,
        task_type="GPU"
    ))
]

# ==========================
# ğŸ”� K-Fold for Meta Features
# ==========================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((X.shape[0], len(base_models)))
test_meta_features = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    print(f"ğŸ”¹ Training base model: {name}")
    test_fold_preds = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        meta_features[val_idx, i] = val_pred
        test_fold_preds.append(model.predict(X_test))

    test_meta_features[:, i] = np.mean(test_fold_preds, axis=0)

print("âœ… Base models training complete!")

# ==========================
# ğŸ“Š Scale Meta Features
# ==========================
scaler = StandardScaler()
X_meta = scaler.fit_transform(meta_features)
X_test_meta = scaler.transform(test_meta_features)

# ==========================
# ğŸ§  Neural Network Meta Model
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaNN(nn.Module):
    def __init__(self, input_dim):
        super(MetaNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# Data tensors
X_meta_tensor = torch.tensor(X_meta, dtype=torch.float32).to(device)
y_meta_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
X_test_meta_tensor = torch.tensor(X_test_meta, dtype=torch.float32).to(device)

meta_model = MetaNN(input_dim=len(base_models)).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(meta_model.parameters(), lr=0.0007, weight_decay=1e-4)

# ==========================
# ğŸ”� Train NN with Early Stopping
# ==========================
best_loss = float("inf")
patience = 30
wait = 0
epochs = 500

for epoch in range(epochs):
    meta_model.train()
    optimizer.zero_grad()
    preds = meta_model(X_meta_tensor)
    loss = criterion(preds, y_meta_tensor)
    loss.backward()
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_weights = meta_model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"â�¹ï¸� Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - RMSE: {torch.sqrt(loss).item():.5f}")

meta_model.load_state_dict(best_weights)

# ==========================
# ğŸ”® Predictions
# ==========================
meta_model.eval()
with torch.no_grad():
    final_preds = meta_model(X_test_meta_tensor).cpu().numpy().flatten()

# ==========================
# ğŸ’¾ Save Submission
# ==========================
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_preds
})
submission.to_csv("submission_nn_stacking_improved.csv", index=False)
print("ğŸ�� submission_nn_stacking_improved.csv created successfully!")



# baseline_lgbm.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ---------- params ----------
TRAIN_CSV = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_CSV  = "/kaggle/input/playground-series-s5e10/test.csv"
SUB_CSV   = "/kaggle/input/playground-series-s5e10/sample_submission.csv"
RANDOM_STATE = 42
NFOLDS = 5

# ---------- load ----------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
sub   = pd.read_csv(SUB_CSV)

# assume the target column is 'accident_risk' and id column is 'id'
TARGET = "accident_risk"
IDCOL = "id"

# ---------- quick preprocessing ----------
train_ids = train[IDCOL].copy()
test_ids  = test[IDCOL].copy()

y = train[TARGET].values
X = train.drop([IDCOL, TARGET], axis=1)
X_test = test.drop([IDCOL], axis=1)

# simple fill
X = X.fillna(-999)
X_test = X_test.fillna(-999)

# label-encode categorical columns
for c in X.columns:
    if X[c].dtype == "object":
        le = LabelEncoder()
        vals = pd.concat([X[c], X_test[c]], axis=0).astype(str)
        le.fit(vals)
        X[c] = le.transform(X[c].astype(str))
        X_test[c] = le.transform(X_test[c].astype(str))

# ---------- CV train ----------
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importance = pd.DataFrame()

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 20,
    "lambda_l2": 1.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": RANDOM_STATE
}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}/{NFOLDS}")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    # âœ… fixed indentation and API
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=100)
        ]
    )

    oof_preds[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / NFOLDS

    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importance(importance_type="gain"),
        "fold": fold + 1
    })
    feature_importance = pd.concat([feature_importance, fi], ignore_index=True)

# ---------- metrics & output ----------
rmse = np.sqrt(mean_squared_error(y, oof_preds))
print("OOF RMSE:", rmse)

# clip predictions to [0,1]
test_preds = np.clip(test_preds, 0, 1)

sub[TARGET] = test_preds
sub.to_csv("submission_baseline.csv", index=False)
print("âœ… Saved submission_baseline.csv")



!pip install autogluon --quiet



# autogluon_baseline.py
import pandas as pd
from autogluon.tabular import TabularPredictor

# ---------- paths ----------
TRAIN_CSV = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_CSV  = "/kaggle/input/playground-series-s5e10/test.csv"
SUB_CSV   = "/kaggle/input/playground-series-s5e10/sample_submission.csv"

# ---------- load ----------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
sub   = pd.read_csv(SUB_CSV)

TARGET = "accident_risk"
IDCOL = "id"

# ---------- train ----------
predictor = TabularPredictor(
    label=TARGET,
    eval_metric="root_mean_squared_error",
    problem_type="regression",
    path="AutogluonModels",
).fit(
    train_data=train.drop(columns=[IDCOL]),
    time_limit=3600,   # 1 hour max training (change if needed)
    presets="best_quality",  # high accuracy preset
    verbosity=2
)

# ---------- leaderboard (optional) ----------
leaderboard = predictor.leaderboard(silent=True)
print(leaderboard.head(10))

# ---------- predict ----------
preds = predictor.predict(test.drop(columns=[IDCOL]))

# ---------- clip predictions between 0 and 1 ----------
preds = preds.clip(0, 1)

# ---------- save submission ----------
sub[TARGET] = preds
sub.to_csv("submission_autogluon.csv", index=False)
print("âœ… Saved submission_autogluon.csv")





