import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Modeling
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Feature Engineering
from category_encoders import TargetEncoder

# Hyperparameter Tuning
import optuna
from optuna.samplers import TPESampler

print("Libraries imported successfully!")


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {train.columns.tolist()}")
print(f"\nTarget distribution:\n{train['loan_paid_back'].value_counts(normalize=True)}")

# Display first few rows
train.head()


# Check for missing values
print("Missing values in train:")
print(train.isnull().sum()[train.isnull().sum() > 0])
print("\nMissing values in test:")
print(test.isnull().sum()[test.isnull().sum() > 0])


def advanced_feature_engineering(df, is_train=True):
    """
    Apply advanced feature engineering techniques:
    - Mathematical ratios
    - Log transforms
    - Binning
    - Interaction features
    """
    df = df.copy()
    
    # === MATHEMATICAL RATIOS ===
    # Income to loan ratio - can the borrower afford this loan?
    if 'annual_income' in df.columns and 'loan_amount' in df.columns:
        df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    
    # Monthly debt ratio - monthly payment burden
    if 'installment' in df.columns and 'annual_income' in df.columns:
        df['monthly_debt_ratio'] = df['installment'] / ((df['annual_income'] / 12) + 1)
    
    # Loan to income ratio
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    
    # Interest rate to FICO ratio - risk indicator
    if 'interest_rate' in df.columns and 'fico_score' in df.columns:
        df['interest_to_fico_ratio'] = df['interest_rate'] / (df['fico_score'] + 1)
    
    # Total payment amount
    if 'installment' in df.columns and 'term' in df.columns:
        df['total_payment'] = df['installment'] * df['term']
        if 'loan_amount' in df.columns:
            df['total_interest_paid'] = df['total_payment'] - df['loan_amount']
    
    # === LOG TRANSFORMS (for skewed features) ===
    skewed_features = ['annual_income', 'loan_amount', 'installment']
    for feat in skewed_features:
        if feat in df.columns:
            df[f'{feat}_log'] = np.log1p(df[feat])
    
    # === BINNING ===
    # FICO score bands
    if 'fico_score' in df.columns:
        df['fico_band'] = pd.cut(df['fico_score'], 
                                  bins=[0, 580, 670, 740, 800, 850],
                                  labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        df['fico_band'] = df['fico_band'].astype(str)
    
    # DTI bands
    if 'dti' in df.columns:
        df['dti_band'] = pd.cut(df['dti'], 
                                bins=[0, 10, 20, 30, 40, 100],
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df['dti_band'] = df['dti_band'].astype(str)
    
    # Income bands
    if 'annual_income' in df.columns:
        df['income_band'] = pd.cut(df['annual_income'], 
                                    bins=[0, 30000, 50000, 75000, 100000, np.inf],
                                    labels=['Low', 'Medium', 'Upper Medium', 'High', 'Very High'])
        df['income_band'] = df['income_band'].astype(str)
    
    # === INTERACTION FEATURES ===
    # Interest rate * DTI - combined risk indicator
    if 'interest_rate' in df.columns and 'dti' in df.columns:
        df['interest_dti_interaction'] = df['interest_rate'] * df['dti']
    
    # FICO * Income - creditworthiness + capacity
    if 'fico_score' in df.columns and 'annual_income' in df.columns:
        df['fico_income_interaction'] = df['fico_score'] * np.log1p(df['annual_income'])
    
    # Loan amount * Interest rate - total cost indicator
    if 'loan_amount' in df.columns and 'interest_rate' in df.columns:
        df['loan_interest_interaction'] = df['loan_amount'] * df['interest_rate']
    
    # === ADDITIONAL FEATURES ===
    # Credit utilization (if revolving balance and credit line exist)
    if 'revol_bal' in df.columns and 'revol_util' in df.columns:
        df['credit_utilization'] = df['revol_util']
    
    # Employment length encoding
    if 'emp_length' in df.columns:
        emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, 
                   '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                   '8 years': 8, '9 years': 9, '10+ years': 10}
        df['emp_length_numeric'] = df['emp_length'].map(emp_map)
    
    print(f"Feature engineering complete. New shape: {df.shape}")
    return df

# Apply feature engineering
train_fe = advanced_feature_engineering(train, is_train=True)
test_fe = advanced_feature_engineering(test, is_train=False)

print(f"\nNew features created: {set(train_fe.columns) - set(train.columns)}")


def preprocess_data(train_df, test_df, target_col='loan_paid_back'):
    """
    Preprocess data:
    - Handle missing values
    - Target encoding for high-cardinality categoricals
    - Label encoding for low-cardinality categoricals
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Separate target and id
    y = train_df[target_col].values
    train_id = train_df['id'].values if 'id' in train_df.columns else None
    test_id = test_df['id'].values if 'id' in test_df.columns else None
    
    # Drop target and id from features
    X_train = train_df.drop([target_col, 'id'], axis=1, errors='ignore')
    X_test = test_df.drop(['id'], axis=1, errors='ignore')
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {len(numerical_cols)}")
    
    # === HANDLE MISSING VALUES ===
    # Numerical: fill with median
    for col in numerical_cols:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    
    # Categorical: fill with 'Missing'
    for col in categorical_cols:
        X_train[col].fillna('Missing', inplace=True)
        X_test[col].fillna('Missing', inplace=True)
    
    # === ENCODING ===
    # Identify high-cardinality categoricals (use Target Encoding)
    high_cardinality_cols = [col for col in categorical_cols 
                             if X_train[col].nunique() > 10]
    low_cardinality_cols = [col for col in categorical_cols 
                            if X_train[col].nunique() <= 10]
    
    print(f"\nHigh-cardinality columns (Target Encoding): {high_cardinality_cols}")
    print(f"Low-cardinality columns (Label Encoding): {low_cardinality_cols}")
    
    # Target Encoding for high-cardinality
    if high_cardinality_cols:
        te = TargetEncoder(cols=high_cardinality_cols, smoothing=1.0)
        X_train[high_cardinality_cols] = te.fit_transform(X_train[high_cardinality_cols], y)
        X_test[high_cardinality_cols] = te.transform(X_test[high_cardinality_cols])
    
    # Label Encoding for low-cardinality
    label_encoders = {}
    for col in low_cardinality_cols:
        le = LabelEncoder()
        # Combine train and test to ensure all categories are seen
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
    
    # Store categorical column indices for CatBoost
    cat_features = [i for i, col in enumerate(X_train.columns) if col in low_cardinality_cols]
    
    print(f"\nFinal feature shape: {X_train.shape}")
    print(f"CatBoost categorical features indices: {cat_features}")
    
    return X_train, X_test, y, test_id, cat_features

# Preprocess data
X_train, X_test, y, test_id, cat_features = preprocess_data(train_fe, test_fe)


def objective_xgb(trial, X, y, n_splits=3):
    """
    Optuna objective function for XGBoost hyperparameter tuning
    """
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cpu',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42
    }
    
    # Quick CV to evaluate parameters
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**param)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

# Run Optuna optimization
print("Starting Optuna hyperparameter tuning for XGBoost...")
print("This may take several minutes...\n")

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(lambda trial: objective_xgb(trial, X_train, y), n_trials=30, show_progress_bar=True)

print(f"\nBest ROC AUC: {study.best_value:.6f}")
print(f"Best parameters: {study.best_params}")

# Store best parameters
best_xgb_params = study.best_params
best_xgb_params['objective'] = 'binary:logistic'
best_xgb_params['eval_metric'] = 'auc'
best_xgb_params['tree_method'] = 'hist'
best_xgb_params['device'] = 'cpu'
best_xgb_params['random_state'] = 42


def train_with_kfold(X, y, X_test, n_splits=5):
    """
    Train XGBoost, LightGBM, and CatBoost with Stratified K-Fold CV
    Returns OOF predictions and test predictions for ensemble
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize OOF and test predictions
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))
    
    fold_scores = {'xgb': [], 'lgb': [], 'cat': [], 'ensemble': []}
    
    print("="*80)
    print(f"Starting {n_splits}-Fold Stratified Cross-Validation")
    print("="*80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*80}")
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        print(f"Train size: {len(X_tr)}, Validation size: {len(X_val)}")
        print(f"Target distribution - Train: {y_tr.mean():.4f}, Val: {y_val.mean():.4f}")
        
        # === XGBoost ===
        print("\n[1/3] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**best_xgb_params)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_splits
        
        xgb_score = roc_auc_score(y_val, oof_xgb[val_idx])
        fold_scores['xgb'].append(xgb_score)
        print(f"XGBoost ROC AUC: {xgb_score:.6f}")
        
        # === LightGBM ===
        print("\n[2/3] Training LightGBM...")
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
            'verbose': -1,
            'n_estimators': 500
        }
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits
        
        lgb_score = roc_auc_score(y_val, oof_lgb[val_idx])
        fold_scores['lgb'].append(lgb_score)
        print(f"LightGBM ROC AUC: {lgb_score:.6f}")
        
        # === CatBoost ===
        print("\n[3/3] Training CatBoost...")
        cat_params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50
        }
        
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            verbose=False
        )
        
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_preds_cat += cat_model.predict_proba(X_test)[:, 1] / n_splits
        
        cat_score = roc_auc_score(y_val, oof_cat[val_idx])
        fold_scores['cat'].append(cat_score)
        print(f"CatBoost ROC AUC: {cat_score:.6f}")
        
        # === Ensemble (Weighted Average) ===
        # Weight by individual model performance
        weights = np.array([xgb_score, lgb_score, cat_score])
        weights = weights / weights.sum()
        
        ensemble_oof = (oof_xgb[val_idx] * weights[0] + 
                        oof_lgb[val_idx] * weights[1] + 
                        oof_cat[val_idx] * weights[2])
        
        ensemble_score = roc_auc_score(y_val, ensemble_oof)
        fold_scores['ensemble'].append(ensemble_score)
        
        print(f"\n{'─'*80}")
        print(f"Fold {fold} Summary:")
        print(f"  XGBoost:  {xgb_score:.6f}")
        print(f"  LightGBM: {lgb_score:.6f}")
        print(f"  CatBoost: {cat_score:.6f}")
        print(f"  Ensemble: {ensemble_score:.6f} (weights: XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CAT={weights[2]:.3f})")
        print(f"{'─'*80}")
    
    # Final OOF scores
    print(f"\n{'='*80}")
    print("FINAL OUT-OF-FOLD SCORES")
    print(f"{'='*80}")
    
    final_xgb = roc_auc_score(y, oof_xgb)
    final_lgb = roc_auc_score(y, oof_lgb)
    final_cat = roc_auc_score(y, oof_cat)
    
    # Calculate final ensemble weights based on OOF performance
    final_weights = np.array([final_xgb, final_lgb, final_cat])
    final_weights = final_weights / final_weights.sum()
    
    oof_ensemble = (oof_xgb * final_weights[0] + 
                    oof_lgb * final_weights[1] + 
                    oof_cat * final_weights[2])
    final_ensemble = roc_auc_score(y, oof_ensemble)
    
    print(f"XGBoost:  {final_xgb:.6f} (±{np.std(fold_scores['xgb']):.6f})")
    print(f"LightGBM: {final_lgb:.6f} (±{np.std(fold_scores['lgb']):.6f})")
    print(f"CatBoost: {final_cat:.6f} (±{np.std(fold_scores['cat']):.6f})")
    print(f"\nEnsemble: {final_ensemble:.6f} (±{np.std(fold_scores['ensemble']):.6f})")
    print(f"Final weights: XGB={final_weights[0]:.3f}, LGB={final_weights[1]:.3f}, CAT={final_weights[2]:.3f}")
    print(f"{'='*80}")
    
    # Calculate final test predictions with optimal weights
    test_preds_ensemble = (test_preds_xgb * final_weights[0] + 
                           test_preds_lgb * final_weights[1] + 
                           test_preds_cat * final_weights[2])
    
    return test_preds_ensemble, fold_scores, final_weights

# Train models
test_predictions, fold_scores, final_weights = train_with_kfold(X_train, y, X_test, n_splits=5)


# Create submission dataframe
submission = pd.DataFrame({
    'id': test_id,
    'loan_paid_back': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(f"\nSubmission shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(submission['loan_paid_back'].describe())

