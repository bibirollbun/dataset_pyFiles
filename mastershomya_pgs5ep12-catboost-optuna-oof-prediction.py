import pandas as pd
import numpy as np
import warnings
import catboost as cb
from catboost import CatBoostClassifier, Pool
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_tr = df_train.copy()
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df_te = df_test.copy()


df_tr_ids = df_tr['id']
df_te_ids = df_te['id']
df_tr = df_tr.drop(columns=["id"])
df_te = df_te.drop(columns=["id"])


edu_map = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}
income_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
smoke_map = {'Never': 0, 'Former': 1, 'Current': 2}

for df in [df_tr, df_te]:
    df['education_level'] = df['education_level'].map(edu_map)
    df['income_level'] = df['income_level'].map(income_map)
    df['smoking_status'] = df['smoking_status'].map(smoke_map)


nominal_cols = ['gender', 'ethnicity', 'employment_status']
cat_features = nominal_cols


print(f"Categorical Features for CatBoost: {cat_features}")


df_tr['diagnosed_diabetes'] = df_tr['diagnosed_diabetes'].astype(int)
X = df_tr.drop(columns=["diagnosed_diabetes"])
y = df_tr["diagnosed_diabetes"]


SEED = 42
sampler = optuna.samplers.TPESampler(seed=SEED)

X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

def objective(trial):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'GPU',       # ENABLE GPU
        'devices': '0',
        'verbose': False,
        'random_seed': SEED,
        'iterations': 2500,
        
        # Hyperparameters
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'border_count': 64,
    }

    oof_preds = np.zeros(len(X_dev))
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    for train_idx, val_idx in kf.split(X_dev, y_dev):
        X_tr, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

        # CatBoost specific Pool creation
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=1000
        )
        
        # Predict on validation set
        oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]

    return roc_auc_score(y_dev, oof_preds)

print("Starting CatBoost GPU Tuning...")
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=30)

print("\nBest Params Found:")
print(study.best_params)

# --- 3. SANITY CHECK (HOLDOUT) ---
print("\nChecking on Holdout...")
final_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': 'GPU',
    'devices': '0',
    'verbose': False,
    'random_seed': SEED,
    'iterations': 2500,
    'border_count': 64,
    **study.best_params
}

# Pools for holdout
train_pool_dev = Pool(X_dev, y_dev, cat_features=cat_features)
holdout_pool = Pool(X_holdout, y_holdout, cat_features=cat_features)

model_check = CatBoostClassifier(**final_params)
model_check.fit(
    train_pool_dev,
    eval_set=holdout_pool,
    early_stopping_rounds=50,
    verbose=False
)
holdout_auc = roc_auc_score(y_holdout, model_check.predict_proba(holdout_pool)[:, 1])
print(f"Holdout AUC: {holdout_auc:.5f}")

# --- 4. FINAL OOF GENERATION ---
print("\nGenerating Final OOFs (for Stacking) and Test Predictions...")

oof_preds_full = np.zeros(len(X))
test_preds_full = np.zeros(len(df_te))

kf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Full Test Pool (Create once to save overhead)
test_pool_full = Pool(df_te, cat_features=cat_features)

for fold, (train_idx, val_idx) in enumerate(kf_full.split(X, y)):
    print(f"Fold {fold+1}...")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(**final_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        verbose=False # Set to False to keep output clean like LGBM
    )

    oof_preds_full[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds_full += model.predict_proba(test_pool_full)[:, 1] / 5

print(f"\nFinal Full OOF AUC: {roc_auc_score(y, oof_preds_full):.5f}")

# --- 5. SAVING FILES ---
df_oof_save = pd.DataFrame({
    'id': df_train['id'],
    'diagnosed_diabetes': y,
    'catboost_pred': oof_preds_full # Unique Name for Stacking
})
df_oof_save.to_csv('oof_catboost.csv', index=False)

df_test_save = pd.DataFrame({
    'id': df_test['id'],
    'catboost_pred': test_preds_full
})
df_test_save.to_csv('test_catboost.csv', index=False)

print("CatBoost Files saved!")


subm = pd.read_csv("/kaggle/working/test_catboost.csv")
subm.head()


subm = subm.rename(columns={'catboost_pred': 'diagnosed_diabetes'})
subm.to_csv('submission.csv', index=False)

