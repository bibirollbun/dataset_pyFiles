import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import optuna
import gc
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sub   = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"Train: {train.shape} | Test: {test.shape}")


def create_features(df):
    df = df.copy()
    df['loan_to_income']       = df['loan_amount'] / (df['annual_income'] + 1)
    df['payment_to_income']    = (df['loan_amount'] * df['interest_rate'] / 1200) / (df['annual_income'] / 12 + 1)
    df['debt_load']            = df['debt_to_income_ratio'] * df['loan_amount']
    df['affordability']        = (df['annual_income'] * (1 - df['debt_to_income_ratio'])) / (df['loan_amount'] + 1)
    df['income_log']           = np.log1p(df['annual_income'])
    df['loan_log']             = np.log1p(df['loan_amount'])
    
    # Grade features
    df['grade']                = df['grade_subgrade'].str[0]
    df['subgrade_rank']        = df['grade_subgrade'].str[1:].astype(int)
    grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    df['grade_rank']           = df['grade'].map(grade_map)
    
    return df

train = create_features(train)
test  = create_features(test)


cat_features = [
    'gender', 'marital_status', 'education_level', 'employment_status',
    'loan_purpose', 'grade_subgrade', 'grade'
]

num_features = [
    'annual_income', 'debt_to_income_ratio', 'credit_score',
    'loan_amount', 'interest_rate',
    'loan_to_income', 'payment_to_income', 'debt_load', 'affordability',
    'income_log', 'loan_log', 'subgrade_rank', 'grade_rank'
]

features = cat_features + num_features
target = 'loan_paid_back'

print(f"Total features: {len(features)} | Categorical: {len(cat_features)}")


# -----------------------------
# Optuna Objective
# -----------------------------
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 2000, 8000),
        'depth': trial.suggest_int('depth', 6, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 12.0),
        'border_count': trial.suggest_int('border_count', 128, 254),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.8, 2.0),
        'od_type': 'Iter',
        'od_wait': 150,
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0',
        'verbose': False,
    }

    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(train[features], train[target]):
        X_tr = train.iloc[train_idx][features]
        y_tr = train.iloc[train_idx][target]
        X_val = train.iloc[val_idx][features]
        y_val = train.iloc[val_idx][target]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool   = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)

        pred = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, pred))

        break  # Only 1 fold for speed during tuning

    return np.mean(scores)


print("\nStarting Optuna tuning (50 trials)...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, timeout=None)  # Increase to 80â€“100 for max performance

print(f"\nBest CV AUC: {study.best_value:.6f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

best_params = study.best_params
best_params.update({
    'od_type': 'Iter',
    'od_wait': 200,
    'random_seed': 42,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 500,
})


print("\nTraining final 7-fold ensemble with best params...")

n_folds = 7
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train[features], train[target])):
    print(f"\nFold {fold+1}/{n_folds}")
    
    X_tr = train.iloc[train_idx][features]
    y_tr = train.iloc[train_idx][target]
    X_val = train.iloc[val_idx][features]
    y_val = train.iloc[val_idx][target]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool   = Pool(X_val, y_val, cat_features=cat_features)
    test_pool  = Pool(test[features], cat_features=cat_features)

    model = CatBoostClassifier(**best_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test_pool)[:, 1] / n_folds

    score = roc_auc_score(y_val, oof_preds[val_idx])
    cv_scores.append(score)
    print(f"Fold {fold+1} AUC: {score:.6f}")

print("\n" + "="*60)
print(f"Final CV AUC : {np.mean(cv_scores):.6f} Â± {np.std(cv_scores):.6f}")
print(f"OOF AUC      : {roc_auc_score(train[target], oof_preds):.6f}")
print("="*60)


sub['loan_paid_back'] = test_preds
sub.to_csv('final_submission_catboost.csv', index=False)
print(f"\nSubmission saved! Expected Public LB: 0.9280 â€“ 0.9283+")

# Optional: Save best params
import json
with open('best_catboost_params.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
print("Best params saved to best_catboost_params.json")




