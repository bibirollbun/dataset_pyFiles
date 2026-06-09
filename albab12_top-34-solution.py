# Install required packages (run this in Kaggle notebook with internet enabled)
!pip install -q xgboost lightgbm catboost


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
# We will run the entire pipeline with these seeds and average the results
SEEDS = [42, 2024, 555] 
N_SPLITS = 5 # 5 folds x 3 seeds = 15 models total (Robust)
TARGET = 'diagnosed_diabetes'

# --- Load Data ---
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

train = train.drop(columns=['id'])
test_ids = test['id']
test = test.drop(columns=['id'])

# --- 1. Preprocessing & Mappings ---
def clean_data(df):
    df = df.copy()
    
    # Ordinal Mapping (Based on common sense hierarchy)
    # Mapping known values to risk levels helps trees split faster
    smoke_map = {'Never': 0, 'No': 0, 'Former': 1, 'Current': 2, 'Smoker': 2, 'Yes': 2}
    gender_map = {'Female': 0, 'Male': 1}
    
    if 'smoking_status' in df.columns:
        df['smoking_status'] = df['smoking_status'].map(smoke_map).fillna(0)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(gender_map).fillna(0)
        
    # Rounding float columns that look like integers with noise
    # Synthetic data often adds .0001 noise to integers like Age
    for col in ['age', 'sleep_hours_per_day', 'physical_activity_minutes_per_week']:
        if col in df.columns:
            df[col] = np.round(df[col], 1)
            
    return df

train = clean_data(train)
test = clean_data(test)

# --- 2. Feature Engineering: Interaction & Freq Encoding ---
def engineer_features(df, train_df=None):
    df = df.copy()
    
    # --- Frequency Encoding (Count Encoding) ---
    # Replaces 'Highschool' with the count of people who have 'Highschool'.
    # This works great for nominal cols like Ethnicity/Employment
    cat_cols_to_freq = ['ethnicity', 'education_level', 'income_level', 'employment_status']
    
    for col in cat_cols_to_freq:
        if col in df.columns:
            # Calculate counts on TRAIN to avoid leakage
            if train_df is not None:
                freq_map = train_df[col].value_counts(normalize=True).to_dict()
            else:
                # Fallback (only for the train set itself)
                freq_map = df[col].value_counts(normalize=True).to_dict()
                
            df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)

    # --- Medical Interactions ---
    df['Log_Trig'] = np.log1p(df['triglycerides'])
    df['Risk_Product'] = df['bmi'] * df['systolic_bp']
    df['Metabolic_Index'] = df['waist_to_hip_ratio'] * df['bmi']
    df['Cardio_Stress'] = df['heart_rate'] * (df['systolic_bp'] - df['diastolic_bp'])
    
    return df

# Apply FE
# Important: We pass 'train' as reference for frequency map to ensure test set gets mapped consistently
X = engineer_features(train, train_df=train)
X_test = engineer_features(test, train_df=train)

y = X[TARGET]
X = X.drop(columns=[TARGET])

# --- 3. Categorical Handling ---
# We keep the original strings for CatBoost, but encode for XGB
cat_cols = ['ethnicity', 'education_level', 'income_level', 'employment_status']

# Create encoded copies for XGB
X_enc = X.copy()
X_test_enc = X_test.copy()

lbl_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on both to catch all values
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_enc[col] = le.transform(X[col].astype(str))
    X_test_enc[col] = le.transform(X_test[col].astype(str))

# --- 4. Multi-Seed Training Loop ---
final_test_preds = np.zeros(len(X_test))
final_oof_preds = np.zeros(len(X))

print(f"Starting Multi-Seed Training ({len(SEEDS)} seeds)...")

for seed in SEEDS:
    print(f"\n--- Training Seed {seed} ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    
    seed_test_preds = np.zeros(len(X_test))
    seed_oof_preds = np.zeros(len(X))
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        # --- XGBoost (Uses Encoded Data) ---
        # Removed scale_pos_weight to focus on pure ranking (AUC)
        model_xgb = xgb.XGBClassifier(
            n_estimators=2500,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.5,
            reg_lambda=2.0,
            tree_method='hist',
            random_state=seed + fold,
            n_jobs=-1,
            early_stopping_rounds=100
        )
        model_xgb.fit(X_enc.iloc[trn_idx], y_trn, eval_set=[(X_enc.iloc[val_idx], y_val)], verbose=False)
        p_xgb = model_xgb.predict_proba(X_enc.iloc[val_idx])[:, 1]
        t_xgb = model_xgb.predict_proba(X_test_enc)[:, 1]

        # --- CatBoost (Uses Raw Categoricals) ---
        model_cb = cb.CatBoostClassifier(
            iterations=2500,
            learning_rate=0.015,
            depth=6,
            l2_leaf_reg=4,
            auto_class_weights='Balanced', # Catboost handles imbalance well
            random_seed=seed + fold,
            verbose=False,
            allow_writing_files=False
        )
        model_cb.fit(
            X.iloc[trn_idx], y_trn,
            cat_features=cat_cols,
            eval_set=(X.iloc[val_idx], y_val),
            early_stopping_rounds=100
        )
        p_cb = model_cb.predict_proba(X.iloc[val_idx])[:, 1]
        t_cb = model_cb.predict_proba(X_test)[:, 1]

        # --- Blend per Fold ---
        # 50/50 blend proved most stable in these scenarios
        seed_oof_preds[val_idx] = (0.5 * p_xgb) + (0.5 * p_cb)
        seed_test_preds += ((0.5 * t_xgb) + (0.5 * t_cb)) / N_SPLITS
        
    print(f"Seed {seed} OOF AUC: {roc_auc_score(y, seed_oof_preds):.5f}")
    
    # Add to final average
    final_test_preds += seed_test_preds / len(SEEDS)
    final_oof_preds += seed_oof_preds / len(SEEDS)

# --- 5. Final Evaluation ---
overall_auc = roc_auc_score(y, final_oof_preds)
print(f"\n==================================")
print(f"GRAND ENSEMBLE CV AUC: {overall_auc:.5f}")
print(f"==================================")

submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': final_test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission saved!")

