# Ensemble of XGBoost and CatBoost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

train_df = pd.read_csv("/kaggle/input/carnival-risk-analytics-challenge/train.csv")
test_df = pd.read_csv("/kaggle/input/carnival-risk-analytics-challenge/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)



target_col = "Premium Amount"

# Drop rows with missing target
train_df = train_df.dropna(subset=[target_col])

# Combine train + test for consistent preprocessing
train_df["is_train"] = 1
test_df["is_train"] = 0
combined = pd.concat([train_df, test_df], ignore_index=True)

# Handle datetime features
if "Policy Start Date" in combined.columns:
    combined["Policy Start Date"] = pd.to_datetime(combined["Policy Start Date"], errors='coerce')
    combined["Policy_Year"] = combined["Policy Start Date"].dt.year
    combined["Policy_Month"] = combined["Policy Start Date"].dt.month
    combined.drop(columns=["Policy Start Date"], inplace=True)


# Encode categorical features
cat_cols = combined.select_dtypes(include=["object"]).columns
for col in cat_cols:
    combined[col] = combined[col].fillna("Unknown")
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Fill numeric NaNs
num_cols = combined.select_dtypes(include=[np.number]).columns
combined[num_cols] = combined[num_cols].fillna(combined[num_cols].median())

# Split back
train_df = combined[combined["is_train"] == 1].drop(columns=["is_train"])
test_df = combined[combined["is_train"] == 0].drop(columns=["is_train", target_col], errors="ignore")

# Features and target
X = train_df.drop(columns=[target_col])
y = train_df[target_col]



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)

cat_model = CatBoostRegressor(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=0
)


ensemble = VotingRegressor([
    ("xgb", xgb_model),
    ("cat", cat_model)
])

ensemble.fit(X_train, y_train)

preds_valid = ensemble.predict(X_valid)
mae = mean_absolute_error(y_valid, preds_valid)
print(f"Validation MAE: {mae:.4f}")

test_preds = ensemble.predict(test_df)

submission = pd.DataFrame({
    "id": test_df["id"],
    "Premium Amount": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv generated successfully!")


print(submission.head())


print(submission.shape)


 # Includes Feature Engineering + K-Fold + XGB, CatBoost, LightGBM + Ridge

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

#Preprocessing nd Feature Engineering

TARGET = "Premium Amount"
ID_COL = "id"

df_train = train_df.copy()
df_test = test_df.copy()

# Drop rows with missing target
df_train = df_train.dropna(subset=[TARGET])

# Parse dates
if "Policy Start Date" in df_train.columns:
    for df in [df_train, df_test]:
        df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors="coerce")
        df["policy_year"] = df["Policy Start Date"].dt.year
        df["policy_month"] = df["Policy Start Date"].dt.month
        df["policy_day"] = df["Policy Start Date"].dt.day



# Feature interactions
for df in [df_train, df_test]:
    df["Income_per_Dependent"] = df["Annual Income"] / (df["Number of Children"].fillna(0)+1)
    df["Health_per_Age"] = df["Health Score"] / (df["Age"]+1)
    df["Vehicle_per_Duration"] = df["Vehicle Age"] / (df["Insurance Duration"]+1)
    df["Income_per_VehicleAge"] = df["Annual Income"] / (df["Vehicle Age"] + 1)


# Binning skewed numeric features
for df in [df_train, df_test]:
    df["Income_bin"] = pd.qcut(df["Annual Income"].rank(method='first'), 10, labels=False)
    df["CreditScore_bin"] = pd.qcut(df["Credit Score"].rank(method='first'), 10, labels=False)


# Separate numeric & categorical
num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in [TARGET, ID_COL]]
cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()

# Fill missing
for c in num_cols:
    median_val = df_train[c].median()
    df_train[c] = df_train[c].fillna(median_val)
    df_test[c] = df_test[c].fillna(median_val)

for c in cat_cols:
    df_train[c] = df_train[c].fillna("Unknown").astype(str)
    df_test[c] = df_test[c].fillna("Unknown").astype(str)


# Encode categorical features for XGB/LightGBM
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([df_train[c], df_test[c]], axis=0))
    df_train[c] = le.transform(df_train[c])
    df_test[c] = le.transform(df_test[c])
    le_dict[c] = le


FEATURES = num_cols + cat_cols
X = df_train[FEATURES].copy()
y = np.log1p(df_train[TARGET].values)  #log transform target
X_test = df_test[FEATURES].copy()


N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

#Out of fold predictions
oof_preds = np.zeros(X.shape[0])
test_preds_xgb = np.zeros(X_test.shape[0])
test_preds_cb = np.zeros(X_test.shape[0])
test_preds_lgb = np.zeros(X_test.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n=== Fold {fold+1} ===")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        tree_method="hist"
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )
    val_pred_xgb = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test) / N_SPLITS
    
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=50
    )
    cat_model.fit(
        Pool(X_tr, y_tr),
        eval_set=Pool(X_val, y_val)
    )
    val_pred_cb = cat_model.predict(X_val)
    test_preds_cb += cat_model.predict(X_test) / N_SPLITS
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    val_pred_lgb = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test) / N_SPLITS
    
    val_pred_ensemble = (val_pred_xgb + val_pred_cb + val_pred_lgb)/3
    oof_preds[val_idx] = val_pred_ensemble
    
    fold_mae = mean_absolute_error(np.expm1(y_val), np.expm1(val_pred_ensemble))
    print(f"Fold {fold+1} MAE (original scale): {fold_mae:.4f}")


from sklearn.linear_model import Ridge

meta_train = np.vstack([oof_preds, oof_preds, oof_preds]).T  # simple stacking
meta_test = np.vstack([test_preds_xgb, test_preds_cb, test_preds_lgb]).T

ridge_meta = Ridge(alpha=1.0)
ridge_meta.fit(meta_train, np.log1p(df_train[TARGET]))
final_test_preds_log = ridge_meta.predict(meta_test)
final_test_preds = np.expm1(final_test_preds_log)

submission = pd.DataFrame({
    ID_COL: df_test[ID_COL].values,
    TARGET: final_test_preds
})
submission[TARGET] = submission[TARGET].clip(lower=0)
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved as submission.csv")
display(submission.head())


#Ensemble of XGBoost and CatBoost + Optuna Tuning for XGBoost


import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor


train_df = pd.read_csv("/kaggle/input/carnival-risk-analytics-challenge/train.csv")
test_df = pd.read_csv("/kaggle/input/carnival-risk-analytics-challenge/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


#Preprocess

target_col = "Premium Amount"
train_df = train_df.dropna(subset=[target_col])

train_df["is_train"] = 1
test_df["is_train"] = 0
combined = pd.concat([train_df, test_df], ignore_index=True)

if "Policy Start Date" in combined.columns:
    combined["Policy Start Date"] = pd.to_datetime(combined["Policy Start Date"], errors='coerce')
    combined["Policy_Year"] = combined["Policy Start Date"].dt.year
    combined["Policy_Month"] = combined["Policy Start Date"].dt.month
    combined.drop(columns=["Policy Start Date"], inplace=True)

cat_cols = combined.select_dtypes(include=["object"]).columns
for col in cat_cols:
    combined[col] = combined[col].fillna("Unknown")
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

num_cols = combined.select_dtypes(include=[np.number]).columns
combined[num_cols] = combined[num_cols].fillna(combined[num_cols].median())

train_df = combined[combined["is_train"] == 1].drop(columns=["is_train"])
test_df = combined[combined["is_train"] == 0].drop(columns=["is_train", target_col], errors="ignore")

X = train_df.drop(columns=[target_col])
y = train_df[target_col]


#Train / Validation Split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Optuna Hyperparameter Tuning (XGBoost)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": 42,
        "tree_method": "hist"
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    return mae


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25, show_progress_bar=True)
print("Best parameters:", study.best_params)


#Final Models (Using Best XGB Params)

xgb_best = XGBRegressor(**study.best_params)

cat_model = CatBoostRegressor(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=0
)

# Ensemble(Voting Regressor)
ensemble = VotingRegressor([
    ("xgb", xgb_best),
    ("cat", cat_model)
])


ensemble.fit(X_train, y_train)

preds_valid = ensemble.predict(X_valid)
mae = mean_absolute_error(y_valid, preds_valid)
print(f"Validation MAE after tuning: {mae:.4f}")


test_preds = ensemble.predict(test_df)

submission = pd.DataFrame({
    "id": test_df["id"],
    "Premium Amount": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv generated successfully!")


print(submission.head())

