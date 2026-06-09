import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA
# ============================================
print("Loading data...")
# For Kaggle notebook, use the input path
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['diagnosed_diabetes'].value_counts(normalize=True)}")

# ============================================
# 2. ENCODE CATEGORICAL VARIABLES
# ============================================
target = 'diagnosed_diabetes'
id_col = 'id'

# Identify categorical columns
cat_columns = []
for col in train.columns:
    if train[col].dtype == 'object' and col not in [target, id_col]:
        cat_columns.append(col)

print(f"\nCategorical columns: {cat_columns}")

def encode_categoricals(train_df, test_df, cat_cols):
    """Label encode categorical columns"""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    encoders = {}
    for col in cat_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            # Fit on combined data to handle all categories
            combined = pd.concat([train_df[col].astype(str), 
                                 test_df[col].astype(str)])
            le.fit(combined)
            train_encoded[col] = le.transform(train_df[col].astype(str))
            test_encoded[col] = le.transform(test_df[col].astype(str))
            encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} categories")
    
    return train_encoded, test_encoded, encoders

train_encoded, test_encoded, encoders = encode_categoricals(train, test, cat_columns)

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
def create_features(df):
    """Create additional features"""
    df = df.copy()
    
    # BMI categories (medical standards)
    df['bmi_category'] = pd.cut(df['bmi'], 
                                 bins=[0, 18.5, 25, 30, 35, 100],
                                 labels=[0, 1, 2, 3, 4])
    df['bmi_category'] = df['bmi_category'].astype(int)
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 30, 40, 50, 60, 100],
                              labels=[0, 1, 2, 3, 4])
    df['age_group'] = df['age_group'].astype(int)
    
    # Health risk combinations
    df['bmi_age_interaction'] = df['bmi'] * df['age']
    df['bmi_systolic_interaction'] = df['bmi'] * df['systolic_bp']
    df['age_systolic_interaction'] = df['age'] * df['systolic_bp']
    
    # Lifestyle score (inverse of healthy behaviors)
    df['unhealthy_lifestyle'] = (
        (df['alcohol_consumption_per_week'] / 10) +
        (df['screen_time_hours_per_day'] / 5) -
        (df['physical_activity_minutes_per_week'] / 200) -
        (df['diet_score'] / 5) -
        (df['sleep_hours_per_day'] / 8)
    )
    
    # Physical activity categories
    df['activity_level'] = pd.cut(df['physical_activity_minutes_per_week'],
                                   bins=[0, 50, 150, 300, 1000],
                                   labels=[0, 1, 2, 3])
    df['activity_level'] = df['activity_level'].astype(int)
    
    # Sleep quality indicator
    df['sleep_quality'] = ((df['sleep_hours_per_day'] >= 7) & 
                           (df['sleep_hours_per_day'] <= 9)).astype(int)
    
    # Metabolic syndrome indicators
    df['metabolic_risk'] = (
        (df['bmi'] > 30).astype(int) +
        (df['waist_to_hip_ratio'] > 0.85).astype(int) +
        (df['systolic_bp'] > 130).astype(int)
    )
    
    # Polynomial features for key health indicators
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_squared'] = df['age'] ** 2
    df['systolic_bp_squared'] = df['systolic_bp'] ** 2
    
    # Ratios
    df['activity_to_screen_ratio'] = (df['physical_activity_minutes_per_week'] / 60) / (df['screen_time_hours_per_day'] + 1)
    df['diet_to_alcohol_ratio'] = df['diet_score'] / (df['alcohol_consumption_per_week'] + 1)
    
    # Blood pressure categories
    df['bp_category'] = pd.cut(df['systolic_bp'],
                                bins=[0, 120, 130, 140, 200],
                                labels=[0, 1, 2, 3])
    df['bp_category'] = df['bp_category'].astype(int)
    
    # Combined health score
    df['health_score'] = (
        df['diet_score'] * 0.3 +
        (df['physical_activity_minutes_per_week'] / 50) * 0.3 +
        df['sleep_hours_per_day'] * 0.2 -
        (df['bmi'] / 10) * 0.1 -
        (df['alcohol_consumption_per_week'] / 5) * 0.1
    )
    
    # Cholesterol features (if available)
    if 'hdl_cholesterol' in df.columns and 'ldl_cholesterol' in df.columns:
        df['cholesterol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
        df['total_to_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    
    # Blood sugar risk (if available)
    if 'blood_sugar' in df.columns:
        df['high_blood_sugar'] = (df['blood_sugar'] > 100).astype(int)
        df['blood_sugar_bmi'] = df['blood_sugar'] * df['bmi']
    
    if 'hba1c' in df.columns:
        df['high_hba1c'] = (df['hba1c'] > 5.7).astype(int)
    
    # Family + personal history combo (if available)
    if 'family_history_diabetes' in df.columns:
        df['total_risk_factors'] = (df['family_history_diabetes'] + 
                                     df['hypertension_history'] + 
                                     df['cardiovascular_history'])
        df['genetic_metabolic_risk'] = df['family_history_diabetes'] * df['metabolic_risk']
    
    return df

print("\nCreating features...")
train_fe = create_features(train_encoded)
test_fe = create_features(test_encoded)

# ============================================
# 4. PREPARE DATA FOR MODELING
# ============================================
features = [c for c in train_fe.columns if c not in [target, id_col]]

X = train_fe[features]
y = train_fe[target]
X_test = test_fe[features]

print(f"\nNumber of features: {len(features)}")

# ============================================
# 5. CROSS-VALIDATION SETUP
# ============================================
N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Store predictions
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

# ============================================
# 6. MODEL 1: LIGHTGBM
# ============================================
print("\n" + "="*50)
print("Training LightGBM...")
print("="*50)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=500)
        ]
    )
    
    oof_lgb[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds_lgb += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    
    fold_score = roc_auc_score(y_val, oof_lgb[val_idx])
    print(f"Fold {fold + 1} AUC: {fold_score:.6f}")

lgb_score = roc_auc_score(y, oof_lgb)
print(f"\nLightGBM OOF AUC: {lgb_score:.6f}")

# ============================================
# 7. MODEL 2: XGBOOST
# ============================================
print("\n" + "="*50)
print("Training XGBoost...")
print("="*50)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    oof_xgb[val_idx] = model.predict(dval)
    test_preds_xgb += model.predict(xgb.DMatrix(X_test)) / N_FOLDS
    
    fold_score = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"Fold {fold + 1} AUC: {fold_score:.6f}")

xgb_score = roc_auc_score(y, oof_xgb)
print(f"\nXGBoost OOF AUC: {xgb_score:.6f}")

# ============================================
# 8. MODEL 3: CATBOOST
# ============================================
print("\n" + "="*50)
print("Training CatBoost...")
print("="*50)

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.01,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 500,
    'task_type': 'CPU',
    'thread_count': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(**cat_params, iterations=5000, early_stopping_rounds=100)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_cat += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fold_score = roc_auc_score(y_val, oof_cat[val_idx])
    print(f"Fold {fold + 1} AUC: {fold_score:.6f}")

cat_score = roc_auc_score(y, oof_cat)
print(f"\nCatBoost OOF AUC: {cat_score:.6f}")

# ============================================
# 9. ENSEMBLE PREDICTIONS
# ============================================
print("\n" + "="*50)
print("Creating Ensemble...")
print("="*50)

# Weighted average ensemble (optimize weights based on CV scores)
weights = np.array([lgb_score, xgb_score, cat_score])
weights = weights / weights.sum()

print(f"\nEnsemble weights:")
print(f"LightGBM: {weights[0]:.4f}")
print(f"XGBoost:  {weights[1]:.4f}")
print(f"CatBoost: {weights[2]:.4f}")

oof_ensemble = (oof_lgb * weights[0] + 
                oof_xgb * weights[1] + 
                oof_cat * weights[2])

test_preds_ensemble = (test_preds_lgb * weights[0] + 
                       test_preds_xgb * weights[1] + 
                       test_preds_cat * weights[2])

ensemble_score = roc_auc_score(y, oof_ensemble)

print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"LightGBM OOF AUC: {lgb_score:.6f}")
print(f"XGBoost OOF AUC:  {xgb_score:.6f}")
print(f"CatBoost OOF AUC: {cat_score:.6f}")
print(f"Ensemble OOF AUC: {ensemble_score:.6f}")
print(f"{'='*50}")

# ============================================
# 10. CREATE SUBMISSION
# ============================================
submission = pd.DataFrame({
    'id': test_fe[id_col],
    'diagnosed_diabetes': test_preds_ensemble
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nPrediction statistics:")
print(submission['diagnosed_diabetes'].describe())




