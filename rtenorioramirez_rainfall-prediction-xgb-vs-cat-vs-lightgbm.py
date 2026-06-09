import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Display dataset information
train.head(5)



# Define features
RMV = ["id", "rainfall"]
FEATURES = [c for c in train.columns if c not in RMV]
CATS = [c for c in FEATURES if train[c].dtype == "object"]

print(f"Features: {len(FEATURES)} (Categorical: {len(CATS)})")


for c in FEATURES:
    if train[c].dtype == "float64":
        train[c] = train[c].astype("float32")
        test[c] = test[c].astype("float32")
    elif train[c].dtype == "int64":
        train[c] = train[c].astype("int32")
        test[c] = test[c].astype("int32")


import optuna
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score



def optimize_xgboost(train, FEATURES, n_trials=30):
    def objective(trial):
        # Hyperparameter suggestions
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 5000),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }

        # 5-fold cross-validation
        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, valid_idx in kf.split(train):
            X_train = train.iloc[train_idx][FEATURES]
            y_train = train.iloc[train_idx]["rainfall"]
            X_valid = train.iloc[valid_idx][FEATURES]
            y_valid = train.iloc[valid_idx]["rainfall"]

            # Train XGBoost model
            model = XGBClassifier(
                **params,
                eval_metric="auc",
                early_stopping_rounds=650,
                random_state=42,
                tree_method="hist",
                enable_categorical=False,  # No categorical features in this dataset
                verbosity=0
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=0
            )

            # Evaluate on validation set
            preds = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, preds)
            auc_scores.append(auc)

        # Return mean AUC across folds
        return np.mean(auc_scores)

    # Create Optuna study
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="maximize")  # Maximize AUC
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Run optimization
best_params = optimize_xgboost(train, FEATURES, n_trials=30)
print("Best hyperparameters:", best_params)


# Best hyperparameters from Optuna ( Change every run)
best_params = {
    'max_depth': 3,
    'learning_rate': 0.06748736192996821,
    'colsample_bytree': 0.6025266237479818,
    'subsample': 0.7343826427838714,
    'min_child_weight': 6,
    'gamma': 0.9624240622807356,
    'scale_pos_weight': 1.4599566740647745,
    'n_estimators': 3832,
    'reg_alpha':  1.3841038719254324,
    'reg_lambda': 1.0889201239250892
}


model = XGBClassifier(
    **best_params,
    eval_metric="auc",
    early_stopping_rounds=300,
    random_state=42,
    tree_method="hist",
    enable_categorical=False,
)


%%time

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {fold+1}")
    print("#"*25)

    X_train = train.iloc[train_idx][FEATURES]
    y_train = train.iloc[train_idx]["rainfall"]
    X_val = train.iloc[val_idx][FEATURES]
    y_val =  train.iloc[val_idx]["rainfall"]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=200
    )
    
    oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
    pred_xgb += model.predict_proba(test[FEATURES])[:, 1] / FOLDS




print(f"OOF AUC: {roc_auc_score(train['rainfall'], oof_xgb):.4f}")


#sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
#ensemble_preds = pred_xgb 
#sub['rainfall'] = ensemble_preds
#sub.to_csv("submission_xgb.csv", index=False)


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


from catboost import CatBoostClassifier

def optimize_catboost(train, FEATURES, n_trials=30):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
        }

        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, valid_idx in kf.split(train):
            X_train = train.iloc[train_idx][FEATURES]
            y_train = train.iloc[train_idx]["rainfall"]
            X_valid = train.iloc[valid_idx][FEATURES]
            y_valid = train.iloc[valid_idx]["rainfall"]

            model = CatBoostClassifier(
                **params,
                eval_metric='AUC',
                early_stopping_rounds=100,
                cat_features=[],
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

            preds = model.predict_proba(X_valid)[:, 1]
            auc_scores.append(roc_auc_score(y_valid, preds))

        return np.mean(auc_scores)

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Run optimization
best_params_cat = optimize_catboost(train, FEATURES, n_trials=30)
print("Best CatBoost hyperparameters:", best_params_cat)


# Best hyperparameters from Optuna ( Change every run)
best_params_cat = {
    'iterations': 1192,
    'depth': 4,
    'learning_rate': 0.0670104241221738,
    'l2_leaf_reg': 4.817821099015548,
    'border_count': 139,
    'random_strength': 4.55425441455397,
    'bagging_temperature': 0.34244010223754345,
    'min_data_in_leaf': 10,
}



model_cat = CatBoostClassifier(
    **best_params_cat,
    eval_metric="AUC",
    allow_writing_files=False,
    early_stopping_rounds=300,
    random_seed=42,
    cat_features=[],  # Explicitly state no categorical features
    verbose=200,
    thread_count=-1  # Use all available cores
)


%%time

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    
    print("#"*25)
    print(f"### Fold {fold+1}")
    print("#"*25)

    X_train = train.iloc[train_idx][FEATURES]
    y_train = train.iloc[train_idx]["rainfall"]
    X_val = train.iloc[val_idx][FEATURES]
    y_val = train.iloc[val_idx]["rainfall"]
    
    model_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=200
    )
    
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    pred_cat += model_cat.predict_proba(test[FEATURES])[:, 1] / FOLDS


print(f"OOF AUC: {roc_auc_score(train['rainfall'], oof_cat):.4f}")


#sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
#semble_preds = pred_cat 
#sub['rainfall'] = ensemble_preds
#sub.to_csv("submission_cat.csv", index=False)



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


pip install -q optuna "optuna-integration[lightgbm]"


from lightgbm import LGBMClassifier
from optuna.integration import LightGBMPruningCallback

def optimize_lgbm(train, FEATURES, n_trials=30):
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        }

        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, valid_idx in kf.split(train):
            X_train = train.iloc[train_idx][FEATURES]
            y_train = train.iloc[train_idx]["rainfall"]
            X_valid = train.iloc[valid_idx][FEATURES]
            y_valid = train.iloc[valid_idx]["rainfall"]

            model = LGBMClassifier(
                **params,
                objective='binary',
                metric='auc',
                early_stopping_round=100,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[]
            )

            preds = model.predict_proba(X_valid)[:, 1]
            auc_scores.append(roc_auc_score(y_valid, preds))

        return np.mean(auc_scores)

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Run optimization
best_params_lgbm = optimize_lgbm(train, FEATURES, n_trials=30)
print("Best LightGBM hyperparameters:", best_params_lgbm)


best_params_LGBM = {
    'num_leaves': 139,
    'max_depth': 12,
    'learning_rate': 0.022138655089953965,
    'min_child_samples': 77,
    'subsample': 0.9286315128938976,
    'colsample_bytree': 0.7448197522526118,
    'reg_alpha': 1.9726169883011861,
    'reg_lambda': 6.733095030844662,
    'n_estimators': 950,
}


model_lgbm = LGBMClassifier(
    **best_params_LGBM,
    objective='binary',
    metric='auc',
    early_stopping_round=300,
    random_state=42,
    n_jobs=-1,  # Use all available cores
    verbose=-1  # -1 to suppress output, 0 for basic info
)



%%time

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_lgbm = np.zeros(len(train))
pred_lgbm = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    
    print("#"*25)
    print(f"### Fold {fold+1}")
    print("#"*25)

    X_train = train.iloc[train_idx][FEATURES]
    y_train = train.iloc[train_idx]["rainfall"]
    X_val = train.iloc[val_idx][FEATURES]
    y_val = train.iloc[val_idx]["rainfall"]
    
    model_lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    
    oof_lgbm[val_idx] = model_lgbm.predict_proba(X_val)[:, 1]
    pred_lgbm += model_lgbm.predict_proba(test[FEATURES])[:, 1] / FOLDS


print(f"LGBM OOF AUC: {roc_auc_score(train['rainfall'], oof_lgbm):.6f}")


#sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
#ensemble_preds = pred_lgbm
#sub['rainfall'] = ensemble_preds
#sub.to_csv("submission_lgbm.csv", index=False)


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


ensemble_auc = roc_auc_score(train['rainfall'], oof_ensemble)
print(f"Ensemble OOF AUC: {ensemble_auc:.6f}")


#sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
#ensemble_preds = (pred_xgb + pred_cat + pred_lgbm ) / 3  # 50/50 blend
#sub['rainfall'] = ensemble_preds
#b.to_csv("submission_UniformBleading.csv", index=False)


# Get individual model scores
xg_auc = roc_auc_score(train['rainfall'], oof_xgb)
cat_auc = roc_auc_score(train['rainfall'], oof_cat)
lgbm_auc = roc_auc_score(train['rainfall'], oof_lgbm)

# Calculate weights (softmax scaling)
total = np.exp(xg_auc) + np.exp(cat_auc) + np.exp(lgbm_auc)
weight_xg = np.exp(xg_auc) / total
weight_cat = np.exp(cat_auc) / total
weight_lgbm = np.exp(lgbm_auc) / total



print(f"- XGBoost weight: {weight_xg:.4f}")
print(f"- CatBoost weight: {weight_cat:.4f}")
print(f"- LightGBM weight: {weight_lgbm:.4f}")
print(f"Sum of weights: {weight_xg + weight_cat + weight_lgbm:.4f}")


oof_ensemble = (oof_xgb * weight_xg) + (oof_cat * weight_cat) + (oof_lgbm * weight_lgbm)
ensemble_auc = roc_auc_score(train['rainfall'], oof_ensemble)
print(f"Weighted Ensemble OOF AUC: {ensemble_auc:.6f}")



#ensemble_test_preds = (pred_xgb * weight_xg) + (pred_cat * weight_cat) + (pred_lgbm * weight_lgbm)

#sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
#sub['rainfall'] = ensemble_test_preds
#sub.to_csv("submission_OptimazedBleading.csv", index=False)




