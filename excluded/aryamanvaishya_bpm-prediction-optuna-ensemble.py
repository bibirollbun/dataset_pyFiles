# ðŸŽµ September Playground 2025 â€” BPM Prediction | Optuna Ensemble ðŸŽ¶

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna

# ===============



# Load Data
# =======================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

X = train.drop(columns=["BeatsPerMinute", "id"])
y = train["BeatsPerMinute"]
X_test = test.drop(columns=["id"])


# =======================
# Define Models
# =======================
ridge = Ridge(alpha=10, random_state=42)
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
lgb = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, random_state=42, n_jobs=-1)

models = {"ridge": ridge, "rf": rf, "lgb": lgb}


# =======================
# Cross-validation RMSE
# =======================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error").mean())
    print(f"{name} CV RMSE: {rmse:.4f}")


# =======================
# Optuna for Ensemble Weights
# =======================
def objective(trial):
    w_ridge = 0.7937198069501389
    w_rf = 0.36748340851058126
    w_lgb = 0.9043618221269145

    # normalize weights
    total = w_ridge + w_rf + w_lgb + 1e-9
    w_ridge, w_rf, w_lgb = w_ridge/total, w_rf/total, w_lgb/total

    preds_oof = np.zeros(len(X))
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        ridge.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        lgb.fit(X_train, y_train)

        p_ridge = ridge.predict(X_val)
        p_rf = rf.predict(X_val)
        p_lgb = lgb.predict(X_val)

        preds_oof[val_idx] = w_ridge*p_ridge + w_rf*p_rf + w_lgb*p_lgb

    return mean_squared_error(y, preds_oof, squared=False)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1)
print("Best Weights:", study.best_params)
print("Best CV RMSE:", study.best_value)


# =======================
# Train on Full Data
# =======================
ridge.fit(X, y)
rf.fit(X, y)
lgb.fit(X, y)

p_ridge = ridge.predict(X_test)
p_rf = rf.predict(X_test)
p_lgb = lgb.predict(X_test)


w_ridge = study.best_params["w_ridge"]
w_rf = study.best_params["w_rf"]
w_lgb = study.best_params["w_lgb"]
total = w_ridge + w_rf + w_lgb + 1e-9
w_ridge, w_rf, w_lgb = w_ridge/total, w_rf/total, w_lgb/total

preds_final = w_ridge*p_ridge + w_rf*p_rf + w_lgb*p_lgb


# =======================
# Submission
# =======================
submission = pd.DataFrame({"id": test["id"], "BeatsPerMinute": preds_final})
submission.to_csv("submission.csv", index=False)
print("submission.csv created!")




