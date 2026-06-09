import subprocess
import sys

def install_libraries():
    packages = [
        "autofeat",
        "catboost",
        "lightgbm",
        "scikit-learn>=1.0.0",
        "optuna",
        "numpy",
        "pandas",
        "xgboost"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-qq", package])

install_libraries()


import pandas as pd
import numpy as np
from autofeat import AutoFeatRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


ID_COL = "id"
TARGET = "Calories"
N_FOLDS = 5
SEED = 42


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

train["Sex"] = train["Sex"].map({"female": 0, "male": 1})
test["Sex"] = test["Sex"].map({"female": 0, "male": 1})

X = train.drop(columns=[ID_COL, TARGET])
y = np.log1p(train[TARGET])
X_test = test.drop(columns=[ID_COL])


oof_log = np.zeros(len(X))
test_log = np.zeros(len(X_test))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    af = AutoFeatRegressor(feateng_steps=1, verbose=0)
    X_train_af = af.fit_transform(X_train, y_train)
    X_val_af = af.transform(X_val)
    X_test_af = af.transform(X_test)

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        early_stopping_rounds=50,
        objective="reg:squarederror",
        random_state=SEED,
        verbosity=0
    )
    model.fit(X_train_af, y_train, eval_set=[(X_val_af, y_val)], verbose=False)

    oof_log[val_idx] = model.predict(X_val_af)
    test_log += model.predict(X_test_af) / N_FOLDS

    rmse = mean_squared_error(y_val, oof_log[val_idx], squared=False)
    print(f"Fold {fold} RMSE (log): {rmse:.4f}")


oof_pred = np.expm1(oof_log)
test_pred = np.expm1(test_log)

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: test_pred
})
submission.to_csv("submission.csv", index=False)
print("submission saved.")

