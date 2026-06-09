import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from prettytable import PrettyTable


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Display dataset information
train.head(5)


train_copy=train.copy()
test_copy=test.copy()
target='Listening_Time_minutes'

table = PrettyTable()
table.field_names = ['Feature', 'Data Type', 'Train Missing %', 'Test Missing %' , "Discrete Ratio (Train)"]

for column in train_copy.columns:
    data_type = str(train_copy[column].dtype)
    
    # Calculate missing percentages
    non_null_count_train = np.round(100-train_copy[column].count()/train_copy.shape[0]*100, 1)
    
    if column != target:
        non_null_count_test = np.round(100-test_copy[column].count()/test_copy.shape[0]*100, 1)
    else:
        non_null_count_test = "NA"
        
    
    
    # Calculate discrete nature ratio (unique values / total values)
    discrete_ratio = np.round(train_copy[column].nunique() / train_copy.shape[0], 4)
    
    table.add_row([column, data_type, non_null_count_train, non_null_count_test, discrete_ratio])

print(table)


# Define features
RMV = ["id", "Listening_Time_minutes"]
FEATURES = [c for c in train.columns if c not in RMV]
CATS = [c for c in FEATURES if train[c].dtype == "object"]

print(f"Features: {len(FEATURES)} (Categorical: {len(CATS)})")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def optimize_xgboost(train, FEATURES, n_trials=20):
    
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            # Updated GPU parameters
            "tree_method": "hist",
            "device": "cuda",  # Changed from gpu_hist
            "sampling_method": "gradient_based",
        }

        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, valid_idx in kf.split(train):
            model = XGBRegressor(
                **params,
                n_estimators=1000,
                eval_metric="rmse",
                early_stopping_rounds=100,
                random_state=42,
                enable_categorical=True,
                verbosity=0
            )

            model.fit(
                train.iloc[train_idx][FEATURES], 
                train.iloc[train_idx]["Listening_Time_minutes"],
                eval_set=[(train.iloc[valid_idx][FEATURES], 
                         train.iloc[valid_idx]["Listening_Time_minutes"])],
                verbose=0
                
            )

            preds = model.predict(train.iloc[valid_idx][FEATURES])
            rmse = np.sqrt(mean_squared_error(train.iloc[valid_idx]["Listening_Time_minutes"], preds))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
#best_params = optimize_xgboost(train, FEATURES, n_trials=20)
#print("Best hyperparameters:", best_params)


# Best hyperparameters from Optuna ( Change every run)
best_params = {
    'max_depth': 10,
    'learning_rate': 0.03379119657569082,
    'colsample_bytree': 0.6588612968138808,
    'subsample': 0.8967584873358806,
    'min_child_weight': 4,
    'gamma': 1.6318053453600387,
    'reg_alpha':  5.521023013284561,
    'reg_lambda': 7.849683124657393
}


model = XGBRegressor(
    **best_params,
    n_estimators=5000,
    eval_metric="rmse",
    early_stopping_rounds=300,
    random_state=42,
    tree_method="hist",
    device="cuda",  # GPU acceleration
    enable_categorical=True,
    verbosity=0
)


%%time

from sklearn.metrics import mean_squared_error

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Changed to float arrays for regression values
oof_xgb = np.zeros(len(train), dtype=float)
pred_xgb = np.zeros(len(test), dtype=float)



for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1}")
    print("#"*25)

    X_train = train.iloc[train_idx][FEATURES]
    y_train = train.iloc[train_idx]["Listening_Time_minutes"]
    X_val = train.iloc[val_idx][FEATURES]
    y_val = train.iloc[val_idx]["Listening_Time_minutes"]
    
    # Train with categorical feature handling
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=300,  # Reduced verbosity for cleaner output
    )
    
    # Get predictions (regression uses predict() not predict_proba())
    oof_xgb[val_idx] = model.predict(X_val)
    fold_preds = model.predict(test[FEATURES])
    pred_xgb += fold_preds / FOLDS  # Average across folds
    




# Calculate overall OOF RMSE
final_rmse = np.sqrt(mean_squared_error(train["Listening_Time_minutes"], oof_xgb))
print(f"Overall OOF RMSE: {final_rmse:.4f}")


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
ensemble_preds = pred_xgb 
sub['Listening_Time_minutes'] = ensemble_preds
sub.to_csv("submission_xgb.csv", index=False)


# Get feature importances
feature_importance = model.feature_importances_


importance_df = pd.DataFrame({
    "Feature": FEATURES,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=True)  

# Plot
plt.figure(figsize=(10, 12))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("XGBoost Feature Importance", fontsize=14)
plt.grid(axis="x", alpha=0.3)
plt.show()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

def optimize_catboost(train, FEATURES, CATS, n_trials=15):
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
            'border_count': trial.suggest_int('border_count', 128, 255),  # Higher for GPU
            'random_strength': trial.suggest_float('random_strength', 1e-9, 15),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 8),  # For categoricals
            'bootstrap_type': 'Poisson'  # Required for GPU
        }

        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, valid_idx in kf.split(train):
            X_train = train.iloc[train_idx][FEATURES]
            y_train = train.iloc[train_idx]["Listening_Time_minutes"]
            X_valid = train.iloc[valid_idx][FEATURES]
            y_valid = train.iloc[valid_idx]["Listening_Time_minutes"]

            model = CatBoostRegressor(
                **params,
                eval_metric='RMSE',
                early_stopping_rounds=100,
                iterations=1000,
                cat_features=CATS,
                task_type='GPU',  # Enable GPU
                devices='0:0',    # Use first GPU
                thread_count=-1,
                verbose=0,
                allow_writing_files=False,
                random_seed=42
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                use_best_model=True
            )

            preds = model.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Usage example:
#best_params_cat = optimize_catboost(train, FEATURES, CATS, n_trials=15)
#print("Best CatBoost hyperparameters:", best_params_cat)


# Best hyperparameters from Optuna ( Change every run)
best_params_cat = {
    'depth': 10,
    'learning_rate': 0.0730463721579289,
    'l2_leaf_reg': 7.507677868474984,
    'border_count': 235,
    'random_strength': 14.78691512218887,
    'bagging_temperature': 0.38361736211619446,
    'min_data_in_leaf': 2,
    'grow_policy': 'Depthwise',
    'max_ctr_complexity': 1
    
}


model_cat = CatBoostRegressor(
    **best_params_cat,
    eval_metric="RMSE",
    allow_writing_files=False,
    early_stopping_rounds=300,
    iterations=5000,
    random_seed=42,
    verbose=100,
    thread_count=-1,
    cat_features=CATS,
    # GPU-specific parameters
    task_type='GPU',
    devices='0:0',
    bootstrap_type='Poisson',  # Required for GPU mode
    # Optional GPU optimizations
    used_ram_limit='10gb',  # Prevent memory overflows
)


%%time

from sklearn.metrics import mean_squared_error

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Initialize arrays for regression predictions
oof_cat = np.zeros(len(train), dtype=float)
pred_cat = np.zeros(len(test), dtype=float)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1}")
    print("#"*25)

    X_train = train.iloc[train_idx][FEATURES]
    y_train = train.iloc[train_idx]["Listening_Time_minutes"]
    X_val = train.iloc[val_idx][FEATURES]
    y_val = train.iloc[val_idx]["Listening_Time_minutes"]
    
    model_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=300,  # Reduced verbosity
        # Add categorical features if needed
        cat_features=CATS
    )
    
    # Get regression predictions
    oof_cat[val_idx] = model_cat.predict(X_val)
    fold_preds = model_cat.predict(test[FEATURES])
    pred_cat += fold_preds / FOLDS
    


# Calculate overall OOF RMSE
final_rmse = np.sqrt(mean_squared_error(train["Listening_Time_minutes"], oof_cat))
print(f"Overall OOF RMSE: {final_rmse:.4f}")


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
ensemble_preds = pred_cat 
sub['Listening_Time_minutes'] = ensemble_preds
sub.to_csv("submission_cat.csv", index=False)


feature_importance = model_cat.get_feature_importance()


importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=True)


plt.figure(figsize=(10, 12))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("CatBoost Feature Importance", fontsize=14)  
plt.grid(axis="x", alpha=0.3)
plt.show()


import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def optimize_lgbm(train, FEATURES, CATS, n_trials=12):
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 512),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'max_bin': trial.suggest_int('max_bin', 64, 255),  # Must be â‰¤ 255 for GPU
        }

        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, valid_idx in kf.split(train):
            X_train = train.iloc[train_idx][FEATURES]
            y_train = train.iloc[train_idx]["Listening_Time_minutes"]
            X_valid = train.iloc[valid_idx][FEATURES]
            y_valid = train.iloc[valid_idx]["Listening_Time_minutes"]

            model = LGBMRegressor(
                **params,
                objective='regression',
                metric='rmse',
                n_estimators=1000,
                early_stopping_round=100,  # Early stopping is enough
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                force_col_wise=True,  # Helps GPU stability
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                categorical_feature=CATS,  # Handle categorical features
            )

            preds = model.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)
        
    # optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Run optimization
#best_params_lgbm = optimize_lgbm(train, FEATURES, CATS, n_trials=12)
#print("Best LightGBM hyperparameters:", best_params_lgbm)


best_params_lgbm = {
    'num_leaves': 589,
    'max_depth': 12,
    'learning_rate': 0.023111458265010466,
    'min_child_samples': 10,
    'subsample': 0.8725177917814763,
    'colsample_bytree': 0.8584398483559579,
    'reg_alpha': 0.2571489639295177,
    'reg_lambda': 6.2143374851920505,
    'max_bin': 183,
}


model_lgbm = LGBMRegressor(
    **best_params_lgbm,
    objective='regression',
    metric='rmse',
    n_estimators=5000,
    early_stopping_round=250,
    random_state=42,
    device='gpu',
    n_jobs=-1,
    verbose=-1
)


%%time

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Initialize arrays for regression predictions
oof_lgbm = np.zeros(len(train), dtype=float)
pred_lgbm = np.zeros(len(test), dtype=float)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1}")
    print("#"*25)

    X_train = train.iloc[train_idx][FEATURES]
    y_train = train.iloc[train_idx]["Listening_Time_minutes"]
    X_val = train.iloc[val_idx][FEATURES]
    y_val = train.iloc[val_idx]["Listening_Time_minutes"]
    
    
    model_lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=CATS,  # Add if using categorical features
    )
    
    # Get regression predictions
    oof_lgbm[val_idx] = model_lgbm.predict(X_val)
    fold_preds = model_lgbm.predict(test[FEATURES])
    pred_lgbm += fold_preds / FOLDS
    


# Calculate overall OOF RMSE
final_rmse = np.sqrt(mean_squared_error(train["Listening_Time_minutes"], oof_lgbm))
print(f"\nOverall OOF RMSE: {final_rmse:.4f}")


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
ensemble_preds = pred_lgbm 
sub['Listening_Time_minutes'] = ensemble_preds
sub.to_csv("submission_lgbm.csv", index=False)


feature_importance = model_lgbm.booster_.feature_importance(importance_type='gain')

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=True)

# Plot
plt.figure(figsize=(10, 12))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("LightGBM Feature Importance", fontsize=14)
plt.grid(axis="x", alpha=0.3)
plt.show()


# Ensemble OOF predictions (simple average)
oof_ensemble = (oof_xgb + oof_cat + oof_lgbm ) / 3


# Calculate overall OOF RMSE
final_rmse = np.sqrt(mean_squared_error(train["Listening_Time_minutes"], oof_ensemble))
print(f"\nOverall OOF RMSE: {final_rmse:.4f}")


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
ensemble_preds = (pred_xgb + pred_cat + pred_lgbm ) / 3  # 50/50 blend
sub['Listening_Time_minutes'] = ensemble_preds
sub.to_csv("submission.csv", index=False)

