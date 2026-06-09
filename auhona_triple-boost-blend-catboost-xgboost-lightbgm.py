
"""
blended_boosting_pipeline.py
============================
Endâ€‘toâ€‘end pipeline for the Kaggle â€œCalorie Expenditureâ€� competition.

Improvements over the baseline:
â€¢Â Robust feature engineering (BMI, HR_frac, VO2, weather gaps, cyclic time features)
â€¢Â Groupâ€‘aware crossâ€‘validation                      (GroupKFold on Sex)
â€¢Â Deterministic, fullyâ€‘seeded training
â€¢Â GPU fallback for CatBoost
â€¢Â Ridge metaâ€‘learner to learn optimal blend weights
â€¢Â Strict reproducibility: saves all OOF preds & models
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
DATA_DIR = Path("/kaggle/input/calorie-expenditure")
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Data ---------------- #
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

# ---- Basic cleaning ---- #
train["Duration"] = train["Duration"].clip(lower=1)        # avoid div/0
test["Duration"]  = test["Duration"].clip(lower=1)

# ---------------- Feature engineering ---------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # BMI
    df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2

    # Heartâ€‘rate fraction of ageâ€‘predicted max
    df["HR_frac"]   = df["HeartRate"] / (220 - df["Age"])
    df["HR_frac_pct"] = df["HR_frac"] * 100                 # scale up

    # VO2 (simple linear surrogate)
    df["VO2_est"] = df["HR_frac"] * 3.5

    # Cardio load proxy
    df["CardioLoad"] = df["HeartRate"] * df["Duration"]

    # Ambient temperature gap from 22Â Â°C comfort
    df["TempGap"] = (df["Temperature"] - 22).clip(lower=-10, upper=17)

    # Cyclic time features (if timestamp present)
    if "Date" in df.columns:
        ts = pd.to_datetime(df["Date"])
        df["Month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
        df["Month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
        df["Dow_sin"]   = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["Dow_cos"]   = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    return df

train = add_features(train)
test  = add_features(test)

# Target: logâ€‘caloriesâ€‘perâ€‘min
y = np.log1p(train["Calories"] / train["Duration"])

features = [c for c in train.columns if c not in ["Calories", "Date", "Id"]]

X = train[features]
X_test = test[features]

# ---------------- Modeling ---------------- #
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

cat_params = dict(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=SEED,
    eval_metric="RMSE",
    verbose=False,
)
# GPU only if available
try:
    import torch, os
    if torch.cuda.is_available():
        cat_params["task_type"] = "GPU"
except Exception:
    pass

xgb_params = dict(
    n_estimators=2500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=SEED,
)

lgb_params = dict(
    n_estimators=3500,
    learning_rate=0.02,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="rmse",
    random_state=SEED,
)

oof_cat = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_lgb = np.zeros(len(train))

pred_cat = np.zeros(len(test))
pred_xgb = np.zeros(len(test))
pred_lgb = np.zeros(len(test))

# Use Sex as group to keep gender distribution stable
groups = train["Sex"]

kf = GroupKFold(n_splits=N_SPLITS)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # CatBoost
    cat = CatBoostRegressor(**cat_params)
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_idx] = cat.predict(X_val)
    pred_cat += cat.predict(X_test) / N_SPLITS
    joblib.dump(cat, MODEL_DIR / f"cat_fold{fold}.cbm")

    # XGBoost
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb.predict(X_val)
    pred_xgb += xgb.predict(X_test) / N_SPLITS
    joblib.dump(xgb, MODEL_DIR / f"xgb_fold{fold}.json")

    # LightGBM
    lgb = LGBMRegressor(**lgb_params)
    lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_lgb[val_idx] = lgb.predict(X_val)
    pred_lgb += lgb.predict(X_test) / N_SPLITS
    joblib.dump(lgb, MODEL_DIR / f"lgb_fold{fold}.txt")

# ---------------- Blending ---------------- #
meta = Ridge(alpha=1.0, fit_intercept=False, positive=True, random_state=SEED)
meta.fit(np.column_stack([oof_cat, oof_xgb, oof_lgb]), y)
w = meta.coef_ / meta.coef_.sum()

# Outâ€‘ofâ€‘fold score
oof_blend = (w[0]*oof_cat + w[1]*oof_xgb + w[2]*oof_lgb)
cv_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y * train["Duration"]),
                                          np.expm1(oof_blend * train["Duration"])))
print(f"CV RMSLE: {cv_rmsle:.6f}")
np.save(MODEL_DIR / "blend_weights.npy", w)

# Final prediction
blend_test = (w[0]*pred_cat + w[1]*pred_xgb + w[2]*pred_lgb)
final_test_pred = np.expm1(blend_test * test["Duration"])

# ---------------- Submission ---------------- #
sub = pd.DataFrame({"Id": test["Id"], "Calories": final_test_pred})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to /kaggle/working/submission.csv")

# ---------------- Save OOF for sanityâ€‘check ---------------- #
pd.DataFrame({
    "Id": train["Id"],
    "OOF_pred": np.expm1(oof_blend * train["Duration"]),
    "Calories": train["Calories"]
}).to_csv(MODEL_DIR / "oof_predictions.csv", index=False)

