# === Step 1: Install & Imports ===
# !pip install -q gplearn

import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# Square function
def _square(x):
    return np.power(x, 2)
square = make_function(function=_square, name='square', arity=1)

# Cube function
def _cube(x):
    return np.power(x, 3)
cube = make_function(function=_cube, name='cube', arity=1)

# Negation
def _neg(x):
    return -x
neg = make_function(function=_neg, name='neg', arity=1)

# === Step 2: Globals and Pipeline ===
le = LabelEncoder()
gp_transformer = None
scaler = None
SHAP_THRESHOLD = 0.001
filtered_programs = []

# === Step 3: Custom Symbolic Feature Filter ===
def transform_with_programs(X_raw, programs):
    features = []
    for prog in programs:
        features.append(prog.execute(X_raw))
    return np.column_stack(features) if features else np.empty((X_raw.shape[0], 0))

# === Step 4: Data Preparation ===
def prepare_data(df, fit=False):
    global gp_transformer, scaler, filtered_programs

    df = df.copy()
    df["Sex"] = le.fit_transform(df["Sex"]) if fit else le.transform(df["Sex"])
    X_raw = df.drop(columns=["Calories", "id"], errors="ignore").astype(float)

    if fit:
        X_sample = X_raw.sample(n=10000, random_state=42)
        y_sample = np.log1p(df.loc[X_sample.index, "Calories"])

        gp_transformer = SymbolicTransformer(
            generations=50,
            population_size=2000,
            hall_of_fame=200,
            n_components=50,
            function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'max', 'min', square, cube, neg),
            parsimony_coefficient=0.005,
            max_samples=0.3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        gp_transformer.fit(X_sample, y_sample)

        X_sym = gp_transformer.transform(X_raw)
        X_all = np.hstack([X_raw.values, X_sym])

        from shap import Explainer
        sample_idx = np.random.choice(len(X_raw), size=1000, replace=False)
        X_train_sample = X_all[sample_idx]
        y_train_sample = np.log1p(df.loc[sample_idx, "Calories"])
        model_sample = xgb.XGBRegressor(tree_method="hist", device="cuda").fit(X_train_sample, y_train_sample)
        shap_values = Explainer(model_sample, X_train_sample)(X_train_sample)

        raw_feature_len = X_raw.shape[1]
        shap_importance = np.abs(shap_values.values).mean(axis=0)[raw_feature_len:]
        all_programs = gp_transformer._best_programs
        filtered_programs = [p for p, s in zip(all_programs, shap_importance) if s > SHAP_THRESHOLD]

    X_sym_filtered = transform_with_programs(X_raw.values, filtered_programs)
    X_full = np.hstack([X_raw.values, X_sym_filtered])

    if fit:
        scaler = StandardScaler()
        return scaler.fit_transform(X_full)
    else:
        return scaler.transform(X_full)

# === Step 5: Load and Prepare Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
y = np.log1p(train["Calories"].clip(lower=0))

X = prepare_data(train, fit=True)
X_test = prepare_data(test, fit=False)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

N_FOLDS = 20
SEED = 42
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds_all = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nðŸ§ª Fold {fold + 1}/{N_FOLDS}")
    X_train_fold, y_train_fold = X[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X[val_idx], y[val_idx]
    
    model_fold = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        eval_metric='rmsle',
        early_stopping_rounds=20,
        random_state=SEED + fold,
        verbosity=0
    )

    model_fold.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )

    val_preds = model_fold.predict(X_val_fold)
    oof_preds[val_idx] = val_preds

    fold_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val_fold), np.expm1(val_preds)))
    fold_scores.append(fold_rmsle)
    print(f"ðŸ“‰ Fold {fold+1} RMSLE: {fold_rmsle:.5f}")

    test_preds_all.append(model_fold.predict(X_test))

# === Weighting based on inverse RMSLE
test_preds_all = np.array(test_preds_all)
fold_scores = np.array(fold_scores)
inv_rmsle = 1 / fold_scores
weights = inv_rmsle / inv_rmsle.sum()
print(f"\nðŸ“Š Fold Weights: {np.round(weights, 4)}")

# === Weighted test prediction
weighted_test_preds = np.average(test_preds_all, axis=0, weights=weights)

# === Final OOF Evaluation ===
rmsle_oof = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
print(f"\nâœ… Final OOF RMSLE (10-fold weighted): {rmsle_oof:.5f}")

# === Save Submission ===
final_preds = np.clip(np.expm1(weighted_test_preds), 0, None)
submission = pd.DataFrame({"id": test["id"], "Calories": final_preds})
submission.to_csv("submission.csv", index=False)
print("âœ… submission_xgb_10fold_weighted.csv saved.")

