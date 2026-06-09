import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train.info()


train.describe()


print("Missing values in train:")
print(train.isnull().sum())
print("\n" + "="*80 + "\n")
print("Missing values in test:")
print(test.isnull().sum())


cat_cols = train.select_dtypes(include='object').columns.tolist()
print("\n" + "="*80 + "\n")
print("Categorical columns:", cat_cols)
for col in cat_cols:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts())


def create_features(df):
    """Create advanced features for the model"""
    df = df.copy()
    
    # Income-to-Loan ratio
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    
    # Loan-to-Income ratio
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    
    # Monthly payment estimate
    df['estimated_monthly_payment'] = (df['loan_amount'] * df['interest_rate'] / 100) / 12
    
    # Payment-to-Income ratio
    df['payment_to_income_ratio'] = (df['estimated_monthly_payment'] * 12) / (df['annual_income'] + 1)
    
    # Credit score bins
    df['credit_score_bin'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 800, 900], 
                                      labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    # Debt burden
    df['debt_burden'] = df['debt_to_income_ratio'] * df['annual_income']
    
    # Total financial burden
    df['total_burden'] = df['debt_burden'] + df['loan_amount']
    
    # Affordability score
    df['affordability_score'] = df['annual_income'] / (df['total_burden'] + 1)
    
    # Interest rate * loan amount (total interest cost indicator)
    df['interest_cost'] = df['interest_rate'] * df['loan_amount']
    
    # Risk score (lower credit score + higher debt ratio = higher risk)
    df['risk_score'] = (1000 - df['credit_score']) * df['debt_to_income_ratio']
    
    # Income per debt ratio unit
    df['income_per_debt'] = df['annual_income'] / (df['debt_to_income_ratio'] + 0.01)
    
    # Loan amount squared (non-linear relationship)
    df['loan_amount_squared'] = df['loan_amount'] ** 2
    
    # Credit score * income interaction
    df['credit_income_interaction'] = df['credit_score'] * df['annual_income'] / 100000
    
    return df

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

print("Feature engineering completed!")
print(f"New train shape: {train_fe.shape}")
print(f"New test shape: {test_fe.shape}")


def prepare_data(train_df, test_df):
    """Encode categorical variables and prepare feature matrices"""
    
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # Identify categorical columns (excluding target and id)
    cat_columns = ['gender', 'marital_status', 'education_level', 
                   'employment_status', 'loan_purpose', 'grade_subgrade', 'credit_score_bin']
    
    # Label encode categorical variables
    label_encoders = {}
    for col in cat_columns:
        if col in train_processed.columns:
            le = LabelEncoder()
            train_processed[col] = le.fit_transform(train_processed[col].astype(str))
            test_processed[col] = le.transform(test_processed[col].astype(str))
            label_encoders[col] = le
    
    # Separate features and target
    X = train_processed.drop(['id', 'loan_paid_back'], axis=1)
    y = train_processed['loan_paid_back']
    X_test = test_processed.drop(['id'], axis=1)
    
    return X, y, X_test, label_encoders

X, y, X_test, label_encoders = prepare_data(train_fe, test_fe)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")
print("\nFeatures used:")
print(X.columns.tolist())


# Setup cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store out-of-fold predictions
oof_predictions = {}
test_predictions = {}


# LightGBM with optimized parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Train LightGBM with cross-validation
oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

print("Training LightGBM model...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # Predictions
    oof_lgb[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_lgb += model.predict(X_test, num_iteration=model.best_iteration) / n_folds

lgb_score = roc_auc_score(y, oof_lgb)
print(f"\n{'='*80}")
print(f"LightGBM OOF ROC AUC Score: {lgb_score:.6f}")
print(f"{'='*80}")

oof_predictions['lgb'] = oof_lgb
test_predictions['lgb'] = test_lgb


# XGBoost with optimized parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

# Train XGBoost with cross-validation
oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

print("Training XGBoost model...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
    dtest = xgb.DMatrix(X_test)
    
    # Train model
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predictions
    oof_xgb[val_idx] = model.predict(dval)
    test_xgb += model.predict(dtest) / n_folds

xgb_score = roc_auc_score(y, oof_xgb)
print(f"\n{'='*80}")
print(f"XGBoost OOF ROC AUC Score: {xgb_score:.6f}")
print(f"{'='*80}")

oof_predictions['xgb'] = oof_xgb
test_predictions['xgb'] = test_xgb


# CatBoost with optimized parameters
cat_params = {
    'objective': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 50,
    'task_type': 'CPU'
}

# Train CatBoost with cross-validation
oof_cat = np.zeros(len(X))
test_cat = np.zeros(len(X_test))

print("Training CatBoost model...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = CatBoostClassifier(**cat_params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        use_best_model=True,
        verbose=100
    )
    
    # Predictions
    oof_cat[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    test_cat += model.predict_proba(X_test)[:, 1] / n_folds

cat_score = roc_auc_score(y, oof_cat)
print(f"\n{'='*80}")
print(f"CatBoost OOF ROC AUC Score: {cat_score:.6f}")
print(f"{'='*80}")

oof_predictions['cat'] = oof_cat
test_predictions['cat'] = test_cat


# Compare individual model scores
print("Individual Model Performance:")
print(f"LightGBM ROC AUC: {lgb_score:.6f}")
print(f"XGBoost ROC AUC:  {xgb_score:.6f}")
print(f"CatBoost ROC AUC: {cat_score:.6f}")

# Try different ensemble weights
best_score = 0
best_weights = None

print("\n" + "="*80)
print("Testing ensemble combinations...")
print("="*80)

# Test various weight combinations
for w1 in np.arange(0.2, 0.5, 0.05):
    for w2 in np.arange(0.2, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 >= 0.2 and w3 <= 0.5:
            ensemble_oof = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
            score = roc_auc_score(y, ensemble_oof)
            if score > best_score:
                best_score = score
                best_weights = (w1, w2, w3)
                print(f"New best! LGB:{w1:.2f} XGB:{w2:.2f} CAT:{w3:.2f} -> Score: {score:.6f}")

print("\n" + "="*80)
print(f"Best Ensemble Score: {best_score:.6f}")
print(f"Best Weights - LightGBM: {best_weights[0]:.3f}, XGBoost: {best_weights[1]:.3f}, CatBoost: {best_weights[2]:.3f}")
print("="*80)


# Create final ensemble predictions
final_test_predictions = (best_weights[0] * test_lgb + 
                          best_weights[1] * test_xgb + 
                          best_weights[2] * test_cat)

print(f"Final test predictions shape: {final_test_predictions.shape}")
print(f"Prediction range: [{final_test_predictions.min():.4f}, {final_test_predictions.max():.4f}]")
print(f"Mean prediction: {final_test_predictions.mean():.4f}")


# Create submission file
submission = pd.DataFrame({
    'id': sample_submission['id'],
    'loan_paid_back': final_test_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)

