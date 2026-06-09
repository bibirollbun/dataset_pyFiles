# Run this cell if required (Kaggle usually has these)
!pip install -q catboost lightgbm xgboost optuna


import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")

# Models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Kaggle dataset path (adjust if running locally)
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape :", test.shape)
train.head()



# Missing values
print("Missing in train:\n", train.isna().sum().sort_values(ascending=False).head(10))
print("\nTarget distribution:")
print(train['diagnosed_diabetes'].value_counts(normalize=True))

# Basic stats
train.describe().T.iloc[:10]


# Drop id, separate target
TARGET = "diagnosed_diabetes"
ID_COL = "id"

train_ids = train[ID_COL].copy()
test_ids  = test[ID_COL].copy()

y = train[TARGET].copy()
X = train.drop([ID_COL, TARGET], axis=1).reset_index(drop=True)
X_test = test.drop([ID_COL], axis=1).reset_index(drop=True)

print("Feature columns:", X.shape[1])

# Identify numeric columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print("Numerical columns:", len(num_cols))


cat_cols = X.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", cat_cols)



# Identify categorical columns
cat_cols = ['gender', 'ethnicity', 'education_level',
            'income_level', 'smoking_status', 'employment_status']

print("Categorical columns:", cat_cols)

# Convert categorical features to numeric codes
for c in cat_cols:
    X[c] = X[c].astype('category').cat.codes
    X_test[c] = X_test[c].astype('category').cat.codes



num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns for PCA:", num_cols)

pca_n_components = 8
pca = PCA(n_components=pca_n_components, random_state=SEED)
pca.fit(X[num_cols].fillna(0))


from sklearn.preprocessing import QuantileTransformer

def build_features(df, pca=None, pca_cols=None):
    df = df.copy()

    # Fill numeric NA with median
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())

    # Basic squared and log transforms
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[f'{c}_sq'] = df[c] ** 2
            df[f'{c}_log1p'] = np.log1p(np.clip(df[c], 0, None))

    # Simple interaction features
    numeric = [c for c in df.columns if df[c].dtype.kind in "biufc"]
    if len(numeric) >= 2:
        a, b = numeric[0], numeric[1]
        df[f'{a}_minus_{b}'] = (df[a] - df[b]).abs()
        df[f'{a}_div_{b}'] = df[a] / (df[b] + 1e-6)

    # Quantile transformation (stabilizes)
    sample_cols = numeric[:3]
    if len(sample_cols) > 0:
        qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
        df[sample_cols] = qt.fit_transform(df[sample_cols])

    # === PCA FIX (IMPORTANT) ===
    if pca is not None and pca_cols is not None:
        pcs = pca.transform(df[pca_cols].fillna(0))
        for i in range(pcs.shape[1]):
            df[f'pca_{i}'] = pcs[:, i]

    return df


X_fe = build_features(X, pca=pca, pca_cols=num_cols)
X_test_fe = build_features(X_test, pca=pca, pca_cols=num_cols)

print("Final train features:", X_fe.shape)
print("Final test features :", X_test_fe.shape)



from sklearn.model_selection import StratifiedKFold

N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X_fe, y))

print("Folds created:", len(folds))



import lightgbm as lgb  # ensure this is imported at top

def run_cv_model(model_name, model_init_fn, X, y, X_test, folds):
    """
    model_name: "catboost" | "lightgbm" | "xgb"
    model_init_fn: function that returns an untrained model
    """
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    
    for fold, (tr_idx, val_idx) in enumerate(folds):
        print(f"\n=== Fold {fold+1}/{len(folds)} ({model_name}) ===")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = model_init_fn()

        # --- CatBoost (use its native args) ---
        if model_name == "catboost":
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                verbose=200,
                use_best_model=True
            )

        # --- LightGBM (use LightGBM callbacks) ---
        elif model_name == "lightgbm":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200),
                    lgb.log_evaluation(period=0)
                ]
            )

        # --- XGBoost (use XGBoost early_stopping_rounds) ---
        elif model_name == "xgb":
            # XGBoost's sklearn wrapper accepts early_stopping_rounds and verbose
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=200,
                verbose=False
            )

        else:
            # fallback generic fit (shouldn't reach here)
            model.fit(X_tr, y_tr)

        # --- predictions ---
        # ensure predict_proba exists (all three do)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / len(folds)

        models.append(model)

        # report fold AUC using only fold val predictions
        fold_auc = roc_auc_score(y_val, oof[val_idx])
        print(f"Fold AUC: {fold_auc:.5f}")
    
    # overall CV AUC using full OOF
    cv_score = roc_auc_score(y, oof)
    print(f"\n=== Full {model_name} CV AUC: {cv_score:.5f} ===")
    
    return oof, test_preds, models, cv_score



from catboost import CatBoostClassifier

def catboost_init():
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=200
    )

print("Training CatBoost...")
cat_oof, cat_test, cat_models, cat_cv = run_cv_model(
    "catboost",
    catboost_init,
    X_fe, y,
    X_test_fe,
    folds
)

print("CatBoost CV AUC:", cat_cv)



import numpy as np

np.save("cat_oof.npy", cat_oof)
np.save("cat_test.npy", cat_test)


submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": cat_test  # use CatBoost test predictions
})



submission_path = "/kaggle/working/catboost_submission.csv"
submission.to_csv(submission_path, index=False)

submission.head(), submission_path



from lightgbm import LGBMClassifier


def lgb_init():
    return LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )


print("Training LightGBM...")

lgb_oof, lgb_test, lgb_models, lgb_cv = run_cv_model(
    "lightgbm",
    lgb_init,
    X_fe, y,
    X_test_fe,
    folds
)

print("LightGBM CV AUC:", lgb_cv)


import numpy as np

np.save("lgb_oof.npy", lgb_oof)
np.save("lgb_test.npy", lgb_test)


from xgboost import XGBClassifier


def xgb_init():
    return XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        random_state=SEED,
        tree_method="hist",   # fastest on CPU
        n_jobs=-1
    )


print("Training XGBoost...")

xgb_oof, xgb_test, xgb_models, xgb_cv = run_cv_model(
    "xgb",
    xgb_init,
    X_fe, y,
    X_test_fe,
    folds
)

print("XGBoost CV AUC:", xgb_cv)


import numpy as np

np.save("xgb_oof.npy", xgb_oof)
np.save("xgb_test.npy", xgb_test)


import numpy as np

cat_oof = np.load("/kaggle/working/cat_oof.npy")
lgb_oof = np.load("/kaggle/working/lgb_oof.npy")
xgb_oof = np.load("/kaggle/working/xgb_oof.npy")

cat_test = np.load("/kaggle/working/cat_test.npy")
lgb_test = np.load("/kaggle/working/lgb_test.npy")
xgb_test = np.load("/kaggle/working/xgb_test.npy")


import optuna
from sklearn.metrics import roc_auc_score

def blend_objective(trial):
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0)
    w3 = trial.suggest_float("w3", 0.0, 1.0)

    weights = np.array([w1, w2, w3])
    if weights.sum() == 0:
        weights = np.array([1.0, 0.0, 0.0])
    weights = weights / weights.sum()   # normalize

    blended_oof = (
        weights[0] * cat_oof +
        weights[1] * lgb_oof +
        weights[2] * xgb_oof
    )

    return roc_auc_score(y, blended_oof)

study = optuna.create_study(direction="maximize")
study.optimize(blend_objective, n_trials=200)


best = study.best_params
w = np.array([best["w1"], best["w2"], best["w3"]])
w = w / w.sum()

print("Best blend weights (Cat, LGB, XGB):", w)
print("Best OOF AUC:", study.best_value)


final_test_pred = (
    w[0] * cat_test +
    w[1] * lgb_test +
    w[2] * xgb_test
)


submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": final_test_pred
})

submission_path = "/kaggle/working/blended_submission.csv"
submission.to_csv(submission_path, index=False)

submission.head(), submission_path




