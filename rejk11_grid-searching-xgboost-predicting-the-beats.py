import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib, os


OUTPUT_FOLDER="/kaggle/working"
IN_FOLDER="/kaggle/input/playground-series-s5e9"


traindf=pd.read_csv(f"{IN_FOLDER}/train.csv",index_col=['id'])
testdf=pd.read_csv(f"{IN_FOLDER}/test.csv",index_col=['id'])


len(traindf.columns)


len(testdf.columns)


traindf.head()


testdf.head()


assert "BeatsPerMinute" in traindf.columns, "Target column missing"
assert traindf.drop(columns=["BeatsPerMinute"]).shape[1] == testdf.shape[1], "Feature mismatch"

X = traindf.drop(columns=["BeatsPerMinute"])
y = traindf["BeatsPerMinute"].astype(float)

const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if const_cols:
    X = X.drop(columns=const_cols)
    testdf = testdf.drop(columns=const_cols)

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)


import torch 
use_cuda = torch.cuda.is_available()


from sklearn.model_selection import train_test_split, GridSearchCV
param_grid = {
    "n_estimators": [300, 600, 1000],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.03],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0.1],
    "reg_lambda": [1.0, 2.0],
    "gamma": [0.1],
}

model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device=("cuda" if use_cuda else "cpu"),
    early_stopping_rounds=100,
    max_delta_step=0,
)

gscv = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

gscv.fit(
    Xtr, ytr,
    eval_set=[(Xva, yva)],
    verbose=False
)

print("Best params:", gscv.best_params_)
print("Best CV score (neg RMSE):", gscv.best_score_)



Xfull_tr, Xfull_va, yfull_tr, yfull_va = train_test_split(X, y, test_size=0.1, random_state=123)
refit_model = XGBRegressor(
    **{**gscv.best_params_, 
       "objective": "reg:squarederror",
       "eval_metric": "rmse",
       "tree_method": "hist",
       "device": ("cuda" if use_cuda else "cpu"),
       "early_stopping_rounds": 100}
)
refit_model.fit(Xfull_tr, yfull_tr, eval_set=[(Xfull_va, yfull_va)], verbose=False)



os.makedirs(OUTPUT_FOLDER, exist_ok=True)
model_path = os.path.join(OUTPUT_FOLDER, "best_xgb_model.joblib")
joblib.dump(refit_model, model_path)
print(f"Saved: {model_path}")


test_pred = refit_model.predict(testdf)
submission = pd.DataFrame({"id": testdf.index, "BeatsPerMinute": test_pred}).reset_index(drop=True)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv; preview:", submission.head())

