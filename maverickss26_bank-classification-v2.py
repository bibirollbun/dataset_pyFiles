import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. ADVANCED FEATURE ENGINEERING
# =====================================================

def create_advanced_features(df):
    """Create powerful feature interactions and transformations"""
    df = df.copy()
    
    # 1. Interaction features between key predictors
    # Duration is the strongest predictor - create smart interactions
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_squared'] = df['duration'] ** 2
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_sqrt'] = np.sqrt(df['duration'])
    
    # 2. Age-based features
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                              labels=['young', 'early_adult', 'mid_adult', 'mature', 'senior', 'elderly'])
    df['age_squared'] = df['age'] ** 2
    
    # 3. Balance features (powerful predictor)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)
    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_bins'] = pd.qcut(df['balance'], q=10, labels=False, duplicates='drop')
    
    # 4. Previous contact features
    df['has_previous_contact'] = (df['previous'] > 0).astype(int)
    df['pdays_active'] = (df['pdays'] != -1).astype(int)
    df['pdays_bins'] = df['pdays'].replace(-1, 999)
    df['pdays_bins'] = pd.cut(df['pdays_bins'], bins=[-1, 0, 100, 200, 300, 400, 1000], labels=False)
    
    # 5. Campaign intensity
    df['campaign_intensity'] = df['campaign'] * df['duration']
    df['multiple_campaigns'] = (df['campaign'] > 1).astype(int)
    
    # 6. Contact pattern features
    df['contact_month_day_interaction'] = df['month'].astype(str) + '_' + df['day'].astype(str)
    
    # 7. Economic indicator combinations
    df['has_loan_and_housing'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)
    df['no_loan_no_housing'] = ((df['loan'] == 'no') & (df['housing'] == 'no')).astype(int)
    
    # 8. Job-Education interaction
    df['job_education'] = df['job'].astype(str) + '_' + df['education'].astype(str)
    
    # 9. Success indicator from previous campaign
    df['prev_success_indicator'] = ((df['poutcome'] == 'success') | 
                                     ((df['poutcome'] == 'other') & (df['previous'] > 2))).astype(int)
    
    # 10. Ratio features
    df['prev_to_campaign_ratio'] = df['previous'] / (df['campaign'] + 1)
    df['duration_to_age_ratio'] = df['duration'] / df['age']
    
    return df

def create_frequency_encoding(df, cols, train_data=None):
    """Frequency encoding for categorical variables"""
    df = df.copy()
    
    for col in cols:
        if train_data is not None:
            # Use training data frequencies for validation/test
            freq_map = train_data[col].value_counts(normalize=True).to_dict()
            df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
        else:
            # Calculate frequencies on current data (training)
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq)
    
    return df

def create_target_encoding(X, y, cols, n_splits=5):
    """Target encoding with cross-validation to prevent overfitting"""
    X = X.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for col in cols:
        X[f'{col}_target_enc'] = 0
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            # Calculate target mean for each category
            target_mean = y_train.groupby(X_train[col]).mean()
            
            # Map to validation fold
            X.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(target_mean).fillna(y_train.mean())
    
    return X

# =====================================================
# 2. ADVANCED ENSEMBLE WITH META-LEARNING
# =====================================================

def train_advanced_models(X_train, y_train, X_test, cat_features):
    """Train diverse models with different characteristics"""
    
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    
    FOLDS = 5
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    # Store predictions
    predictions = {}
    oof_predictions = {}
    
    # ========== Model 1: CatBoost with categorical features ==========
    print("Training CatBoost...")
    oof_cb = np.zeros(len(X_train))
    pred_cb = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostClassifier(
            iterations=3000,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            colsample_bylevel=0.8,
            random_seed=42 + fold,
            eval_metric='AUC',
            #task_type="GPU",
            early_stopping_rounds=100,
            verbose=0
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            cat_features=cat_features,
            verbose=False
        )
        
        oof_cb[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_cb += model.predict_proba(X_test)[:, 1] / FOLDS
    
    predictions['catboost'] = pred_cb
    oof_predictions['catboost'] = oof_cb
    print(f"CatBoost CV Score: {roc_auc_score(y_train, oof_cb):.5f}")
    
    # ========== Model 2: XGBoost with different parameters ==========
    print("Training XGBoost...")
    oof_xgb = np.zeros(len(X_train))
    pred_xgb = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = XGBClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=1,
            reg_alpha=4,
            reg_lambda=4,
            random_state=42 + fold,
            #device="cuda",
            eval_metric='auc',
            early_stopping_rounds=100
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_xgb += model.predict_proba(X_test)[:, 1] / FOLDS
    
    predictions['xgboost'] = pred_xgb
    oof_predictions['xgboost'] = oof_xgb
    print(f"XGBoost CV Score: {roc_auc_score(y_train, oof_xgb):.5f}")
    
    # ========== Model 3: LightGBM with dart mode ==========
    print("Training LightGBM (DART)...")
    oof_lgb = np.zeros(len(X_train))
    pred_lgb = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = LGBMClassifier(
            boosting_type='dart',
            n_estimators=1500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=4,
            reg_lambda=4,
            random_state=42 + fold,
            metric='auc',
            n_jobs=-1
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )
        
        oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_lgb += model.predict_proba(X_test)[:, 1] / FOLDS
    
    predictions['lightgbm_dart'] = pred_lgb
    oof_predictions['lightgbm_dart'] = oof_lgb
    print(f"LightGBM (DART) CV Score: {roc_auc_score(y_train, oof_lgb):.5f}")
    
    # ========== Model 4: Extra Trees for diversity ==========
    from sklearn.ensemble import ExtraTreesClassifier
    print("Training Extra Trees...")
    oof_et = np.zeros(len(X_train))
    pred_et = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42 + fold,
            n_jobs=-1
        )
        
        model.fit(X_tr, y_tr)
        
        oof_et[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_et += model.predict_proba(X_test)[:, 1] / FOLDS
    
    predictions['extra_trees'] = pred_et
    oof_predictions['extra_trees'] = oof_et
    print(f"Extra Trees CV Score: {roc_auc_score(y_train, oof_et):.5f}")
    
    return predictions, oof_predictions

# =====================================================
# 3. META-LEARNER STACKING
# =====================================================

def train_meta_learner(oof_predictions, y_train, test_predictions):
    """Train a meta-learner on OOF predictions"""
    from xgboost import XGBClassifier
    
    # Prepare meta features
    X_meta_train = pd.DataFrame(oof_predictions)
    X_meta_test = pd.DataFrame(test_predictions)
    
    # Add prediction statistics as features
    X_meta_train['mean_pred'] = X_meta_train.mean(axis=1)
    X_meta_train['std_pred'] = X_meta_train.std(axis=1)
    X_meta_train['max_pred'] = X_meta_train.max(axis=1)
    X_meta_train['min_pred'] = X_meta_train.min(axis=1)
    
    X_meta_test['mean_pred'] = X_meta_test.mean(axis=1)
    X_meta_test['std_pred'] = X_meta_test.std(axis=1)
    X_meta_test['max_pred'] = X_meta_test.max(axis=1)
    X_meta_test['min_pred'] = X_meta_test.min(axis=1)
    
    # Train meta-learner with CV
    FOLDS = 5
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    oof_meta = np.zeros(len(X_meta_train))
    pred_meta = np.zeros(len(X_meta_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_train, y_train)):
        X_tr, X_val = X_meta_train.iloc[train_idx], X_meta_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        meta_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42 + fold,
            eval_metric='auc'
        )
        
        meta_model.fit(X_tr, y_tr)
        
        oof_meta[val_idx] = meta_model.predict_proba(X_val)[:, 1]
        pred_meta += meta_model.predict_proba(X_meta_test)[:, 1] / FOLDS
    
    print(f"Meta-learner CV Score: {roc_auc_score(y_train, oof_meta):.5f}")
    
    return pred_meta, oof_meta

# =====================================================
# 4. MAIN PIPELINE
# =====================================================

def main_pipeline():
    """Main training pipeline"""
    
    # Load data
    print("Loading data...")
    train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    orig = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')
    
    # Process original data
    orig['y'] = orig['y'].map({'no': 0, 'yes': 1})
    
    # Store IDs and target
    train_ids = train['id']
    test_ids = test['id']
    y_train_synthetic = train['y']
    y_orig = orig['y']
    
    # Drop unnecessary columns
    train = train.drop(columns=['id', 'y'])
    test = test.drop(columns=['id'])
    orig = orig.drop(columns=['y'])
    
    # Combine synthetic and original data
    train_combined = pd.concat([train, orig], ignore_index=True)
    y_combined = pd.concat([y_train_synthetic, y_orig], ignore_index=True)
    
    print("Creating advanced features...")
    # Feature engineering
    train_combined = create_advanced_features(train_combined)
    test = create_advanced_features(test)
    
    # Identify categorical columns for encoding
    cat_cols_original = ['job', 'marital', 'education', 'default', 'housing', 
                         'loan', 'contact', 'month', 'poutcome']
    cat_cols_new = ['age_group', 'job_education', 'contact_month_day_interaction']
    all_cat_cols = cat_cols_original + cat_cols_new
    
    # Frequency encoding
    train_combined = create_frequency_encoding(train_combined, cat_cols_original)
    test = create_frequency_encoding(test, cat_cols_original, train_combined)
    
    # Target encoding for high-cardinality features
    high_card_cols = ['job_education', 'contact_month_day_interaction']
    train_combined = create_target_encoding(train_combined, y_combined, high_card_cols)
    
    # For test set, use mean target encoding from train
    for col in high_card_cols:
        target_means = y_combined.groupby(train_combined[col]).mean()
        test[f'{col}_target_enc'] = test[col].map(target_means).fillna(y_combined.mean())
    
    # Label encode categorical features
    print("Encoding categorical features...")
    le = LabelEncoder()
    for col in all_cat_cols:
        if col in train_combined.columns:
            train_combined[col] = train_combined[col].astype(str)
            test[col] = test[col].astype(str)
            
            # Fit on combined train data
            le.fit(list(train_combined[col].values) + list(test[col].values))
            train_combined[col] = le.transform(train_combined[col])
            test[col] = le.transform(test[col])
    
    # Scale numerical features
    num_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
                    'duration_per_campaign', 'duration_squared', 'duration_log', 'duration_sqrt',
                    'age_squared', 'balance_log', 'campaign_intensity', 'prev_to_campaign_ratio',
                    'duration_to_age_ratio']
    
    scaler = StandardScaler()
    train_combined[num_features] = scaler.fit_transform(train_combined[num_features].fillna(0))
    test[num_features] = scaler.transform(test[num_features].fillna(0))
    
    # Train models
    print("\nTraining diverse models...")
    predictions, oof_predictions = train_advanced_models(
        train_combined, y_combined, test, 
        cat_features=[col for col in all_cat_cols if col in train_combined.columns]
    )
    
    # Train meta-learner
    print("\nTraining meta-learner...")
    final_pred, final_oof = train_meta_learner(oof_predictions, y_combined, predictions)
    
    print(f"\nFinal CV Score: {roc_auc_score(y_combined, final_oof):.5f}")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        'y': final_pred
    })
    
    # Optional: Blend with best public submission if available
    try:
        best_public = pd.read_csv("/kaggle/input/ps-s5e8-binary-classification-hv-blend-bokeh/submission.csv")
        
        # Weighted average with emphasis on the best model output
        submission['y'] = 0.02 * submission['y'] + 0.98 * best_public['y']
        print("Blended with public best submission")
    except:
        print("Using standalone predictions")
    
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved! Shape: {submission.shape}")
    
    return submission

# Run the pipeline
if __name__ == "__main__":
    submission = main_pipeline()
    print("\nFirst 5 predictions:")
    print(submission.head())




