import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation
import catboost
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings("ignore")


def safe_target_encode(train, test, col, target):
    """Leak-free target encoding using KFold."""
    train_new = train.copy()
    test_new = test.copy()
    global_mean = train[target].mean()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    encoded = pd.Series(index=train.index, dtype=float)

    for train_idx, val_idx in kf.split(train):
        means = train.iloc[train_idx].groupby(col)[target].mean()
        encoded.iloc[val_idx] = train.iloc[val_idx][col].map(means)
    train_new[col + "_mean_risk"] = encoded.fillna(global_mean)
    test_new[col + "_mean_risk"] = test[col].map(train.groupby(col)[target].mean()).fillna(global_mean)
    return train_new, test_new


def engineer_features(train, test):
    train = train.copy()
    test = test.copy()

    # --- Basic numeric interactions ---
    for df in [train, test]:
        df["speed_curvature_product"] = df["speed_limit"] * df["curvature"]
        df["speed_lanes_ratio"] = df["speed_limit"] / (df["num_lanes"] + 1)
        df["accidents_per_lane"] = df["num_reported_accidents"] / (df["num_lanes"] + 1)
        df["is_adverse_weather"] = df["weather"].isin(["Rainy", "Foggy"]).astype(int)
        df["is_low_visibility"] = df["lighting"].isin(["Dim", "Night"]).astype(int)

    # --- Fold-safe mean encodings ---
    for col in ["road_type", "weather", "lighting"]:
        train, test = safe_target_encode(train, test, col, target)

    # --- Categorical encoding ---
    cat_cols = ["road_type", "weather", "lighting", "time_of_day"]
    train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
    test = pd.get_dummies(test, columns=cat_cols, drop_first=True)

    # --- Align columns (avoid mismatch between train/test) ---
    train, test = train.align(test, join="left", axis=1, fill_value=0)

    return train, test


# Base path
base_path = "/kaggle/input/playground-series-s5e10"

train = pd.read_csv(base_path + "/train.csv")
test = pd.read_csv(base_path + "/test.csv")

target = "accident_risk"
id_col = "id"

print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())


print("\nMissing values:\n", train.isna().sum())

target = "accident_risk"
print("\nTarget summary:")
print(train[target].describe())


# Plot target distribution
sns.histplot(train[target], bins=30, kde=True)
plt.title("Distribution of Accident Risk (Continuous 0â€“1)")
plt.show()


cat_cols_train = train.select_dtypes(include=['object', 'category', 'bool']).columns
cat_cols_test = test.select_dtypes(include=['object', 'category', 'bool']).columns
# Compare sets
same_cats = set(cat_cols_train) == set(cat_cols_test)
print("Same categorical columns:", same_cats)

# If not same, show differences
if not same_cats:
    print("In train but not in test:", set(cat_cols_train) - set(cat_cols_test))
    print("In test but not in train:", set(cat_cols_test) - set(cat_cols_train))


# Find common categorical columns
common_cat_cols = set(cat_cols_train).intersection(set(cat_cols_test))

# Compare unique value counts
for col in common_cat_cols:
    train_unique = train[col].nunique(dropna=False)
    test_unique = test[col].nunique(dropna=False)
    print(f"{col}: Train = {train_unique}, Test = {test_unique}, Match = {train_unique == test_unique}")


# Apply feature engineering
train_fe, test_fe = engineer_features(train, test) 
train_fe.head()


train_fe.shape


corr = train_fe.corrwith(train_fe['accident_risk']).sort_values(ascending=False)
print(corr.head(20))
print()
print(corr.tail(20))


X = train_fe.drop(columns=[target, id_col])
y = train_fe[target]

# Drop columns with very low correlation (< 0.05 abs)
corr = train_fe.corrwith(train_fe[target]).abs().sort_values(ascending=False)
low_signal_cols = corr[corr < 0.02].index.tolist()
X = X.drop(columns=low_signal_cols, errors="ignore")
test_X = test_fe[X.columns]

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
bool_cols = X.select_dtypes(include=['bool']).columns

print(f"Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}, Boolean: {len(bool_cols)}")


print(f"âœ… Training with {X.shape[1]} features after filtering low-signal ones")


# final data after preprocessing
X


# final test data after preprocessing 
test_X


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train


# xgb = XGBRegressor(
#     n_estimators=800, learning_rate=0.05, max_depth=6,
#     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.2,
#     random_state=42, tree_method="hist"
# )

# lgb = LGBMRegressor(
#     n_estimators=800, learning_rate=0.05, max_depth=-1,
#     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
#     random_state=42
# )

# cat = CatBoostRegressor(
#     iterations=800, learning_rate=0.05, depth=6,
#     l2_leaf_reg=3, random_seed=42, verbose=False
# )


# --- Configuration ---
TARGET = "accident_risk"
ID_COL = "id"
RANDOM_STATE = 42

# Optuna control
MAX_TRIALS = 50              # 40â€“50 trials give strong results; 20 might underexplore
PATIENCE = 7                 # allow 7 trials without improvement
IMPROVEMENT_THRESHOLD = 2e-4 # more sensitive stopping (0.0005 RMSE diff)

def should_stop(study, patience=PATIENCE, threshold=IMPROVEMENT_THRESHOLD):
    vals = [t.value for t in study.trials if t.value is not None]
    if len(vals) < patience + 1:
        return False
    best_before = np.min(vals[:-patience])
    best_now = np.min(vals)
    return (best_before - best_now) < threshold

# Use 3 folds for tuning (faster), 5 folds for final model
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def objective_lgb(trial):
    params = {
        "n_estimators": 1500,  # reduced â€” early stopping will find optimal anyway
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 180),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.03),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.5),
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "device": "gpu",
    }

    rmses = []
    for tr_idx, val_idx in cv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(period=0)
            ],
        )
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

    return np.mean(rmses)


def objective_xgb(trial):
    params = {
        "n_estimators": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "gpu_hist" ,  # much faster than exact
        "predictor": "gpu_predictor"
    }

    rmses = []
    for tr_idx, val_idx in cv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

    return np.mean(rmses)


def objective_cat(trial):
    params = {
        "iterations": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth": trial.suggest_int("depth", 5, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 1.0),
        "random_state": RANDOM_STATE,
        "loss_function": "RMSE",
        "task_type": "GPU",
        "early_stopping_rounds": 50,
        "verbose": False,
    }

    rmses = []
    for tr_idx, val_idx in cv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

    return np.mean(rmses)



! pip install optuna


# ----------------------
# Run Optuna studies
# ----------------------
import optuna
print("===> Tuning LightGBM...")
study_lgb = optuna.create_study(direction="minimize")
for t in range(MAX_TRIALS):
    study_lgb.optimize(objective_lgb, n_trials=1)
    if should_stop(study_lgb):
        print(f"ðŸ›‘ Early stop triggered â€” no improvement in last {PATIENCE} trials.")
        break

print(f"\nâœ… Best RMSE: {study_lgb.best_value:.6f}")
print(f"Best params:\n{study_lgb.best_params}")

print("===> Tuning XGBoost...")
study_xgb = optuna.create_study(direction="minimize")
for t in range(MAX_TRIALS):
    study_xgb.optimize(objective_xgb, n_trials=1)
    if should_stop(study_xgb):
        print(f"ðŸ›‘ Early stop triggered â€” no improvement in last {PATIENCE} trials.")
        break
print(f"\nâœ… Best RMSE: {study_xgb.best_value:.6f}")
print(f"Best params:\n{study_xgb.best_params}")

print("===> Tuning CatBoost...")
study_cat = optuna.create_study(direction="minimize")
for t in range(MAX_TRIALS):
    study_cat.optimize(objective_cat, n_trials=1)
    if should_stop(study_cat):
        print(f"ðŸ›‘ Early stop triggered â€” no improvement in last {PATIENCE} trials.")
        break
print(f"\nâœ… Best RMSE: {study_cat.best_value:.6f}")
print(f"Best params:\n{study_cat.best_params}")


best_lgb = study_lgb.best_params.copy()
best_lgb.update({"n_estimators": 3000, "random_state": RANDOM_STATE})

best_xgb = study_xgb.best_params.copy()
best_xgb.update({"n_estimators": 3000, "random_state": RANDOM_STATE})

best_cat = study_cat.best_params.copy()
best_cat.update({"iterations": 3000, "random_state": RANDOM_STATE, "verbose": False})

model_lgb = LGBMRegressor(**best_lgb)
model_xgb = XGBRegressor(**best_xgb)
model_cat = CatBoostRegressor(**best_cat)


meta_model = LGBMRegressor(device="gpu", learning_rate=0.03, n_estimators=800, num_leaves=64, random_state=RANDOM_STATE)
stack = StackingRegressor(
estimators=[("xgb", model_xgb), ("lgb", model_lgb), ("cat", model_cat)],
final_estimator=meta_model,
passthrough=True,
n_jobs=-1
)


# print("===> CV evaluation of stacked model...")
# _rmse = np.sqrt(-cross_val_score(stack, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1))
# print(f"Stacking CV RMSE: {_rmse.mean():.5f} Â± {_rmse.std():.5f}")


# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# rmse_scores = np.sqrt(
#     -cross_val_score(stack_model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
# )
# print(f"StackingRegressor CV RMSE: {rmse_scores.mean():.5f} Â± {rmse_scores.std():.5f}")


print("===> Training final stack on full data...")
stack.fit(X, y)
pred_test = stack.predict(test_X)

submission = pd.DataFrame({"id": test[ID_COL], "accident_risk": np.clip(pred_test, 0, 1)})
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"âœ… Saved final submission to {output_path}")


# stack_model.fit(X, y)
# y_pred = stack_model.predict(X_val)

# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# mae = mean_absolute_error(y_val, y_pred)
# r2 = r2_score(y_val, y_pred)

# print(f"Validation RMSE: {rmse:.4f}")
# print(f"Validation MAE:  {mae:.4f}")
# print(f"Validation RÂ²:   {r2:.4f}")

# # Scatter plot of predictions
# plt.figure(figsize=(6,6))
# sns.scatterplot(x=y_val, y=y_pred, alpha=0.6)
# plt.xlabel("True Accident Risk")
# plt.ylabel("Predicted Accident Risk")
# plt.title("Predicted vs True Values")
# plt.plot([0,1],[0,1],'r--')
# plt.show()


# test_X


# test_pred = stack_model.predict(test_X)
# test_pred = np.clip(test_pred, 0, 1)


# submission = pd.DataFrame({
#     "id": test["id"],
#     "accident_risk": test_pred
# })

# submission.to_csv("submission.csv", index=False)


submission.head()




