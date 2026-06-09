import numpy as np
import pandas as pd


# --- Load data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

train.head()


test.head()


target = "accident_risk"
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
cat_cols = ["road_type", "lighting", "weather", "time_of_day"]


# Convert bool column to 0/1
for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)
train.head()


test.head()


from sklearn.preprocessing import LabelEncoder

# String category string only LabelEncoder
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(all_values)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

train.head()


test.head()


# --- Basic feature engineering ---
for df in [train, test]:
    df["speed_curvature"] = df["speed_limit"] * df["curvature"]
    df["lane_density"] = df["num_lanes"] / (df["speed_limit"] + 1)
    df["accident_per_lane"] = df["num_reported_accidents"] / (df["num_lanes"] + 1)


train.head()


test.head()


# --- Define base models ---
lgb_params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.02,
    num_leaves=150,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    random_state=42,
    n_estimators=3000
)
cb_params = dict(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=False
)
xgb_params = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=3000,
    random_state=42
)


from sklearn.model_selection import KFold

# --- 5-Fold Stacking ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

features = [col for col in train.columns if col not in ["id", target]]

oof_lgb, oof_cb, oof_xgb = np.zeros(len(train)), np.zeros(len(train)), np.zeros(len(train))
test_lgb, test_cb, test_xgb = np.zeros(len(test)), np.zeros(len(test)), np.zeros(len(test))


import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    print(f"===== Fold {fold+1} =====")
    X_train, X_valid = train.iloc[tr_idx][features], train.iloc[val_idx][features]
    y_train, y_valid = train.iloc[tr_idx][target], train.iloc[val_idx][target]

    # LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = model_lgb.predict(X_valid)
    test_lgb += model_lgb.predict(test[features]) / kf.n_splits

    # CatBoost
    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                 use_best_model=True, early_stopping_rounds=100)
    oof_cb[val_idx] = model_cb.predict(X_valid)
    test_cb += model_cb.predict(test[features]) / kf.n_splits

    # XGBoost
    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds=100, verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_valid)
    test_xgb += model_xgb.predict(test[features]) / kf.n_splits


# --- Create meta features ---
train_stack = pd.DataFrame({
    "lgb": oof_lgb,
    "cb": oof_cb,
    "xgb": oof_xgb
})
test_stack = pd.DataFrame({
    "lgb": test_lgb,
    "cb": test_cb,
    "xgb": test_xgb
})


import itertools

# Add interaction terms + mean/std among base models
def add_interactions(df):
    df_new = df.copy()
    for (c1, c2) in itertools.combinations(df.columns, 2):
        df_new[f"{c1}_x_{c2}"] = df[c1] * df[c2]
    df_new["mean"] = df.mean(axis=1)
    df_new["std"] = df.std(axis=1)
    return df_new

train_stack_ext = add_interactions(train_stack)
test_stack_ext = add_interactions(test_stack)


print(f"Base stacking features: {train_stack.shape[1]}")
print(f"Extended stacking features: {train_stack_ext.shape[1]}")


from sklearn.linear_model import RidgeCV

# --- RidgeCV as meta model ---
alphas = [1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
meta_model = RidgeCV(alphas=alphas, scoring="neg_root_mean_squared_error", cv=5)
meta_model.fit(train_stack_ext, train[target])


from sklearn.metrics import mean_squared_error

# Metamodel predictions
final_oof = meta_model.predict(train_stack_ext)
final_preds = meta_model.predict(test_stack_ext)

rmse = mean_squared_error(train[target], final_oof, squared=False)
print("===============================")
print(f"Final Stacking RMSE (RidgeCV + mean/std): {rmse:.5f}")
print(f"Best alpha: {meta_model.alpha_}")
print("===============================")

# --- Submission ---
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": final_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")

