import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# =====================
# Load Data
# =====================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


import numpy as np

def preprocess_and_engineer(df, ref_df=None):
    """Preprocess and engineer features for the dataset
    
    Args:
        df: DataFrame containing the original features
        ref_df: Reference DataFrame for outlier clipping (optional)
        
    Returns:
        DataFrame with preprocessed and additional engineered features
    """
    # Make a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # ---- Preprocessing Step ----
    outlier_cols = [
        "AudioLoudness", "VocalContent", "AcousticQuality", 
        "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore"
    ]
    
    # Use provided reference dataframe or the original one for outlier clipping
    if ref_df is None:
        ref_df = df_new
        
    for col in outlier_cols:
        low, high = ref_df[col].quantile([0.005, 0.995])
        df_new[col] = df_new[col].clip(lower=low, upper=high)
    
    # Apply log transformation to positively skewed columns
    pos_skew = ["VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood"]
    for col in pos_skew:
        df_new[col] = np.log1p(df_new[col])
    
    # Apply cubic root to AudioLoudness
    df_new["AudioLoudness"] = np.cbrt(df_new["AudioLoudness"])

    # ---- Feature Engineering Step ----
    # 1. Rhythm-based features
    df_new['Rhythm_Energy'] = df_new['RhythmScore'] * df_new['Energy']
    df_new['Rhythm_Loudness'] = df_new['RhythmScore'] * df_new['AudioLoudness']
    
    # 2. Duration-related features
    df_new['Duration_Minutes'] = df_new['TrackDurationMs'] / 60000  # Convert to minutes
    df_new['Duration_Energy_Ratio'] = df_new['TrackDurationMs'] / (df_new['Energy'] * 10000 + 1)  # Scaled for numerical stability
    
    # 3. Non-linear transformations
    df_new['RhythmScore_Squared'] = df_new['RhythmScore'] ** 2
    df_new['Energy_Squared'] = df_new['Energy'] ** 2
    df_new['Log_Duration'] = np.log1p(df_new['TrackDurationMs'])  # log(1+x) to handle zeros
    
    # 4. Musical character features
    df_new['Acoustic_Instrumental_Ratio'] = df_new['AcousticQuality'] / (df_new['InstrumentalScore'] + 0.01)  # Avoid division by zero
    df_new['Vocal_Energy'] = df_new['VocalContent'] * df_new['Energy']
    
    # 5. Performance and mood interactions
    df_new['Live_Energy'] = df_new['LivePerformanceLikelihood'] * df_new['Energy']
    df_new['Mood_Rhythm'] = df_new['MoodScore'] * df_new['RhythmScore']
    
    # 6. Composite metrics
    df_new['Audio_Intensity'] = (df_new['Energy'] * np.abs(df_new['AudioLoudness'])) / 10  # Scaled for better range
    df_new['Performance_Character'] = (df_new['LivePerformanceLikelihood'] + df_new['MoodScore']) / 2
    
    # 7. Ratios representing musical balance
    df_new['Energy_Loudness_Ratio'] = df_new['Energy'] / (np.abs(df_new['AudioLoudness']) + 0.01)
    df_new['Rhythm_Duration_Density'] = df_new['RhythmScore'] / df_new['Duration_Minutes']
    
    return df_new


# Example usage:
print("Creating and preprocessing features...")
train_fe = preprocess_and_engineer(train)
test_fe = preprocess_and_engineer(test, ref_df=train)


X = train_fe.drop(columns=["id", "BeatsPerMinute"])
y = train_fe["BeatsPerMinute"]
X_test = test_fe.drop(columns=["id"])

print("Data loaded")


# =====================
# LightGBM Optuna params (already tuned)
# =====================
best_lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.028551392479384357,
    "num_leaves": 42,
    "max_depth": 8,
    "feature_fraction": 0.9589137177373734,
    "bagging_fraction": 0.9509935933870622,
    "bagging_freq": 7,
    "min_data_in_leaf": 42,
    "lambda_l1": 0.007529602616697326,
    "lambda_l2": 4.811359028566392e-06,
    "seed": 42,
}


# =====================
# Optuna for XGBoost
# =====================
def objective_xgb(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "random_state": 42,
        "tree_method": "hist",
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=25, show_progress_bar=True)
best_xgb_params = study_xgb.best_trial.params
best_xgb_params["n_estimators"] = 5000
best_xgb_params["objective"] = "reg:squarederror"
best_xgb_params["eval_metric"] = "rmse"
best_xgb_params["random_state"] = 42
best_xgb_params["tree_method"] = "hist"


# =====================
# Optuna for CatBoost
# =====================
def objective_cat(trial):
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10.0, log=True),
        "random_seed": 42,
        "loss_function": "RMSE",
        "od_type": "Iter",
        "od_wait": 50,
        "verbose": False,
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(objective_cat, n_trials=25, show_progress_bar=True)
best_cat_params = study_cat.best_trial.params
best_cat_params["iterations"] = 5000
best_cat_params["random_seed"] = 42
best_cat_params["loss_function"] = "RMSE"
best_cat_params["od_type"] = "Iter"
best_cat_params["od_wait"] = 50
best_cat_params["verbose"] = False


# =====================
# OOF Stacking
# =====================
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_preds_lgb = np.zeros((NFOLDS, len(X_test)))
test_preds_xgb = np.zeros((NFOLDS, len(X_test)))
test_preds_cat = np.zeros((NFOLDS, len(X_test)))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"FOLD {fold+1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # LightGBM
    dtrain = lgb.Dataset(X_train_scaled, label=y_train)
    dval = lgb.Dataset(X_val_scaled, label=y_val)
    lgb_model = lgb.train(
        best_lgb_params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=10000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val_scaled)
    test_preds_lgb[fold] = lgb_model.predict(X_test_scaled)

    # XGBoost
    xgb_model = xgb.XGBRegressor(**best_xgb_params)
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    oof_xgb[val_idx] = xgb_model.predict(X_val_scaled)
    test_preds_xgb[fold] = xgb_model.predict(X_test_scaled)

    # CatBoost
    cat_model = CatBoostRegressor(**best_cat_params)
    cat_model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), verbose=False)
    oof_cat[val_idx] = cat_model.predict(X_val_scaled)
    test_preds_cat[fold] = cat_model.predict(X_test_scaled)


# =====================
# Meta Model (Ridge)
# =====================
stack_train = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
stack_test = np.vstack([
    test_preds_lgb.mean(axis=0),
    test_preds_xgb.mean(axis=0),
    test_preds_cat.mean(axis=0)
]).T

meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, y)
final_preds = meta_model.predict(stack_test)
print("done")


# =====================
# Evaluate
# =====================
rmse = np.sqrt(mean_squared_error(y, meta_model.predict(stack_train)))
print("OOF Stacking RMSE:", rmse)


# =====================
# Submission
# =====================
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": final_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")


submission

