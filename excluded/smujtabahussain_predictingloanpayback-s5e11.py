import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"Train: {train.shape} | Test: {test.shape}")


print("\n" + "="*50)
print("EDA")
print("="*50)

print("\nTraining Data Info\n")
print(train.info())

print("\nMissing Values\n")
print(train.isnull().sum())

print("\nTarget Distribution\n")
print(train['loan_paid_back'].value_counts(normalize=True))

print("\nNumerical Features Statistics\n")
print(train.describe())



print("Creating ONLY essential features...")

def minimal_features(df):
    """Create only 5-6 most essential features"""
    df = df.copy()
    
    # Just the most basic ratios
    df['loan_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['debt_income_ratio_total'] = (df['annual_income'] * df['debt_to_income_ratio'] + df['loan_amount']) / (df['annual_income'] + 1)
    df['monthly_payment_est'] = df['loan_amount'] * df['interest_rate'] / 1200
    df['payment_income_ratio'] = df['monthly_payment_est'] / (df['annual_income'] / 12 + 1)
    #df['debt_burden'] = df['debt_amount'] / df['credit_score']
    
    # Grade numeric
    df['grade_letter'] = df['grade_subgrade'].str[0]
    df['grade_number'] = df['grade_subgrade'].str[1].astype(int)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_numeric'] = df['grade_letter'].map(grade_map) * 10 + df['grade_number']
    
    return df

train_fe = minimal_features(train)
test_fe = minimal_features(test)

print(f"Added features: {train_fe.shape[1] - train.shape[1]}")


print("\nPreprocessing...")

X = train_fe.drop(['id', 'loan_paid_back', 'grade_subgrade'], axis=1)
y = train_fe['loan_paid_back']
X_test = test_fe.drop(['id', 'grade_subgrade'], axis=1)

# Encode categoricals
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    if col in X_test.columns:
        X_test[col] = le.transform(X_test[col].astype(str))

print(f"Final features: {X.shape[1]}")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


print("\n" + "="*70)
print("TRAINING LIGHTGBM (Aggressive)")
print("="*70)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 84,  # Higher!
    'learning_rate': 0.008,  # Lower for more iterations
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'min_child_samples': 10,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'max_depth': -1,
    'verbose': -1,
    'random_state': RANDOM_STATE,
    'force_col_wise': True
}

lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params, 
        train_data, 
        num_boost_round=5000,  # Many iterations
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),  # Very patient
            lgb.log_evaluation(period=0)
        ]
    )
    
    lgb_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    lgb_test += model.predict(X_test, num_iteration=model.best_iteration) / cv.n_splits
    
    fold_score = roc_auc_score(y_val, lgb_oof[val_idx])
    print(f"Fold {fold+1:2d}: {fold_score:.5f}")

lgb_cv_score = roc_auc_score(y, lgb_oof)
print(f"\nLightGBM CV: {lgb_cv_score:.5f}")


print("\n" + "="*70)
print("TRAINING XGBOOST (Aggressive)")
print("="*70)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,  # Deeper
    'learning_rate': 0.01,  # Lower
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 1,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'gamma': 0,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist'
}

xgb_oof = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        train_data,
        num_boost_round=5000,
        evals=[(val_data, 'val')],
        early_stopping_rounds=200,
        verbose_eval=0
    )
    
    xgb_oof[val_idx] = model.predict(val_data, iteration_range=(0, model.best_iteration))
    xgb_test += model.predict(xgb.DMatrix(X_test), iteration_range=(0, model.best_iteration)) / cv.n_splits
    
    fold_score = roc_auc_score(y_val, xgb_oof[val_idx])
    print(f"Fold {fold+1:2d}: {fold_score:.5f}")

xgb_cv_score = roc_auc_score(y, xgb_oof)
print(f"\nXGBoost CV: {xgb_cv_score:.5f}")



print("\n" + "="*70)
print("TRAINING CATBOOST (Aggressive)")
print("="*70)

cat_params = {
    'iterations': 5000,
    'learning_rate': 0.01,
    'depth': 10,  # Deep!
    'l2_leaf_reg': 5,
    'eval_metric': 'AUC',
    'random_seed': RANDOM_STATE,
    'verbose': 0,
    'early_stopping_rounds': 200
}

cat_oof = np.zeros(len(X))
cat_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(**cat_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    cat_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    cat_test += model.predict_proba(X_test)[:, 1] / cv.n_splits
    
    fold_score = roc_auc_score(y_val, cat_oof[val_idx])
    print(f"Fold {fold+1:2d}: {fold_score:.5f}")

cat_cv_score = roc_auc_score(y, cat_oof)
print(f"\nCatBoost CV: {cat_cv_score:.5f}")


print("\n" + "="*70)
print("TESTING ENSEMBLE STRATEGIES")
print("="*70)

# Strategy 1: Equal weights
ensemble1 = (lgb_test + xgb_test + cat_test) / 3
ensemble1_oof = (lgb_oof + xgb_oof + cat_oof) / 3
print(f"Equal weights CV:     {roc_auc_score(y, ensemble1_oof):.5f}")

# Strategy 2: Best model heavy
if lgb_cv_score >= xgb_cv_score and lgb_cv_score >= cat_cv_score:
    ensemble2 = 0.5 * lgb_test + 0.25 * xgb_test + 0.25 * cat_test
    ensemble2_oof = 0.5 * lgb_oof + 0.25 * xgb_oof + 0.25 * cat_oof
elif xgb_cv_score >= lgb_cv_score and xgb_cv_score >= cat_cv_score:
    ensemble2 = 0.25 * lgb_test + 0.5 * xgb_test + 0.25 * cat_test
    ensemble2_oof = 0.25 * lgb_oof + 0.5 * xgb_oof + 0.25 * cat_oof
else:
    ensemble2 = 0.25 * lgb_test + 0.25 * xgb_test + 0.5 * cat_test
    ensemble2_oof = 0.25 * lgb_oof + 0.25 * xgb_oof + 0.5 * cat_oof
print(f"Best model heavy CV:  {roc_auc_score(y, ensemble2_oof):.5f}")

# Strategy 3: Weighted by performance
total = lgb_cv_score + xgb_cv_score + cat_cv_score
w_lgb = lgb_cv_score / total
w_xgb = xgb_cv_score / total
w_cat = cat_cv_score / total
ensemble3 = w_lgb * lgb_test + w_xgb * xgb_test + w_cat * cat_test
ensemble3_oof = w_lgb * lgb_oof + w_xgb * xgb_oof + w_cat * cat_oof
print(f"Performance weighted CV: {roc_auc_score(y, ensemble3_oof):.5f}")
print(f"  Weights: LGB={w_lgb:.3f}, XGB={w_xgb:.3f}, CAT={w_cat:.3f}")

# Choose best ensemble
ensembles = [
    (ensemble1, ensemble1_oof, "equal"),
    (ensemble2, ensemble2_oof, "best_heavy"),
    (ensemble3, ensemble3_oof, "weighted")
]
best_ensemble = max(ensembles, key=lambda x: roc_auc_score(y, x[1]))
final_predictions = best_ensemble[0]
print(f"\nUsing: {best_ensemble[2]} (CV: {roc_auc_score(y, best_ensemble[1]):.5f})")



print("\n" + "="*70)
print("CREATING SUBMISSION")
print("="*70)

submission['loan_paid_back'] = final_predictions
submission.to_csv('submission.csv', index=False)

print("Done!")
print(f"\nFINAL SCORES:")
print(f"   LightGBM:  {lgb_cv_score:.5f}")
print(f"   XGBoost:   {xgb_cv_score:.5f}")
print(f"   CatBoost:  {cat_cv_score:.5f}")
print(f"   Ensemble:  {roc_auc_score(y, best_ensemble[1]):.5f}")
print(f"\nStrategy: {best_ensemble[2]}")

