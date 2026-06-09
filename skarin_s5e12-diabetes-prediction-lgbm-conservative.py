import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import warnings
import gc

# ------------------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
N_FOLDS = 5
TARGET = 'diagnosed_diabetes'

def load_data():
    print(">>> Loading Data...")
    train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
    
    # 1. Sanitize Column Names (CRITICAL FIX)
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    
    # 2. Verify Integrity
    if train.shape[0] < 600000:
        raise ValueError(f"FATAL: train.csv has {train.shape[0]} rows. Expected ~700,000. You are using the wrong file.")
        
    # 3. Store IDs and Drop
    test_ids = test['id']
    if 'id' in train.columns: train = train.drop(columns=['id'])
    if 'id' in test.columns: test = test.drop(columns=['id'])
    
    return train, test, test_ids

def feature_engineering(df):
    """
    Feature Engineering:
    Focus on Ratios and Log-Transforms to reduce variance.
    """
    # 1. Log Transform for Skewed Activity (Skew > 2.0)
    # This compresses outlier values like 158 mins/week into a cleaner range
    df['log_physical_activity'] = np.log1p(df['physical_activity_minutes_per_week'])
    
    # 2. Physiological Interactions (The Alpha)
    # Pulse Pressure: Proxy for arterial stiffness
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # MAP: Mean Arterial Pressure (Perfusion proxy)
    df['map'] = (df['systolic_bp'] + (2 * df['diastolic_bp'])) / 3
    
    # Metabolic Ratios (with epsilon safety)
    eps = 1e-6
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + eps)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + eps)
    
    # Interaction: BMI * Age (Classic risk multiplier)
    df['bmi_age'] = df['bmi'] * df['age']
    
    # Interaction: Waist * BMI (Central Obesity index)
    df['waist_bmi'] = df['waist_to_hip_ratio'] * df['bmi']
    
    return df

def run_pipeline():
    # Load & Sanitize
    train, test, test_ids = load_data()
    
    print(">>> Engineering Features...")
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    X_test = test.copy()
    
    # Categorical Encoding
    cat_cols = [col for col in X.columns if X[col].dtype == 'object']
    print(f"    Categoricals: {cat_cols}")
    
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[cat_cols] = enc.fit_transform(X[cat_cols])
    X_test[cat_cols] = enc.transform(X_test[cat_cols])
    
    # Convert to Category Type (Optimized for LightGBM)
    for col in cat_cols:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # Stratified CV
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    
    # --------------------------------------------------------------------------
    # THE ANVIL CONFIGURATION (Aggressive Regularization)
    # --------------------------------------------------------------------------
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.015,     
        'num_leaves': 16,           
        'max_depth': -1,
        'bagging_fraction': 0.6,    # Use only 60% of data per tree
        'bagging_freq': 5,
        'feature_fraction': 0.6,    # Use only 60% of features per tree
        'lambda_l1': 3.0,           
        'lambda_l2': 5.0,           
        'min_child_samples': 200,   
        'verbose': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    
    print(f"\n>>> Starting {N_FOLDS}-Fold De-Overfit CV...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Dataset
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
        lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols, reference=lgb_train)
        
        # Train
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=5000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=500)
            ]
        )
        
        # Predict
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)
        print(f"    Fold {fold+1} AUC: {score:.5f}")
        
        del X_train, X_val, y_train, y_val, lgb_train, lgb_val, model
        gc.collect()
        
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n>>> FINAL LOCAL CV AUC: {overall_auc:.5f}")
    print(f"    (Mean: {np.mean(scores):.5f} +/- {np.std(scores):.5f})")
    
    # Save
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': test_preds
    })
    submission.to_csv('submission.csv', index=False)
    print(">>> submission.csv saved successfully.")

if __name__ == "__main__":
    run_pipeline()

