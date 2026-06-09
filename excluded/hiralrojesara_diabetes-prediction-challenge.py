# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import optuna
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

def engineer_features_advanced(df):
    df.columns = df.columns.str.lower()
    num_cols = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'age']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
            df[col] = df[col].replace(0, np.nan)
            df[col] = df.groupby('age')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    if 'age' in df.columns and 'glucose' in df.columns:
        df['age_glucose_ratio'] = df['age'] / (df['glucose'] + 1e-5)
        df['age_bmi_prod'] = df['age'] * df['bmi'] 
    if 'glucose' in df.columns and 'insulin' in df.columns:
        df['homa_ir'] = (df['glucose'] * df['insulin']) / 405
    
    return pd.get_dummies(df, drop_first=True)


print("ğŸš€ Loading Data...")
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
y = train["diagnosed_diabetes"]
X = engineer_features_advanced(train.drop(columns=["diagnosed_diabetes", "id"]))
X_test = engineer_features_advanced(test.drop(columns=["id"]))
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)


import pandas as pd
import numpy as np
import optuna
import gc
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. Load All Data
print("ğŸ“‚ Loading Data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Note: External data load (Optional usage based on your logic)
try:
    ext_data = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv')
except:
    print("âš ï¸� External data file not found, continuing with competition data only.")

# 2. Feature Alignment Function (Updated for consistency)
def align_features(df, source="comp"):
    new_df = pd.DataFrame()
    if source == "comp":
        new_df['bmi'] = df['bmi']
        new_df['age'] = df['age']
        if 'systolic_bp' in df.columns:
            new_df['high_bp'] = (df['systolic_bp'] > 130).astype(int)
        if 'smoking_status' in df.columns:
            new_df['smoker'] = df['smoking_status'].apply(lambda x: 1 if x in ['Current', 'Former'] else 0)
    else: # BRFSS
        new_df['bmi'] = df['BMI']
        new_df['age'] = df['Age']
        new_df['high_bp'] = df['HighBP']
        new_df['smoker'] = df['Smoker']
    return new_df

# 3. ğŸ”� Optuna Objective with Fix for XGBoost 2.0+
def objective(trial):
    # Model parameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2000, 5000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.6),
        'tree_method': 'hist', 
        'device': 'cuda', # Kaggle GPU environment ke liye
        'early_stopping_rounds': 100, # <--- Fix: Constructor mein moved
        'random_state': 42,
        'verbosity': 0
    }
    
    # 5-Fold CV tuning ke liye fast hota hai (500 trials ke liye)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    # Pre-processing targets and dropping non-feature columns
    X_full = train.drop(['id', 'diagnosed_diabetes'], axis=1, errors='ignore')
    y_full = train['diagnosed_diabetes']
    
    # Categorical Columns Encoding
    for col in X_full.select_dtypes('object').columns:
        X_full[col] = X_full[col].astype('category').cat.codes
    
    for t_idx, v_idx in skf.split(X_full, y_full):
        xt, xv = X_full.iloc[t_idx], X_full.iloc[v_idx]
        yt, yv = y_full.iloc[t_idx], y_full.iloc[v_idx]
        
        # Initialize Model
        model = XGBClassifier(**params)
        
        # Fit Model (Note: early_stopping_rounds here is removed)
        model.fit(
            xt, yt, 
            eval_set=[(xv, yv)],
            verbose=False
        )
        
        # Predict Probabilities
        preds = model.predict_proba(xv)[:, 1]
        scores.append(roc_auc_score(yv, preds))
        
        # Memory Cleanup
        del model; gc.collect()
        
    return np.mean(scores)

# --- Tuning Start ---
print("ğŸš€ Starting Optuna Optimization (Targeting Top Score)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # Change n_trials=500 for final run if time permits

print(f"ğŸ�† Best Trial Score: {study.best_value}")
print(f"ğŸ“‹ Best Params: {study.best_params}")


from sklearn.linear_model import Ridge

def run_top_rank_ensemble(train_X, train_y, test_X, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros((len(train_X), 3))
    preds = np.zeros((len(test_X), 3))
    
    # Best Params with Lower Learning Rate for more precision
    xgb_params = {**best_params, 'learning_rate': 0.008, 'n_estimators': 2500} 
    
    for f, (t, v) in enumerate(skf.split(train_X, train_y)):
        xt, xv, yt, yv = train_X.iloc[t], train_X.iloc[v], train_y.iloc[t], train_y.iloc[v]
        
        # 1. XGBoost (Slower learning)
        m1 = XGBClassifier(**xgb_params).fit(xt, yt)
        # 2. LightGBM (More estimators, slower learning)
        m2 = LGBMClassifier(n_estimators=3000, learning_rate=0.005, device='gpu', random_state=42).fit(xt, yt)
        # 3. CatBoost
        m3 = CatBoostClassifier(iterations=2000, learning_rate=0.01, task_type='GPU', verbose=0).fit(xt, yt)
        
        oof[v, 0], oof[v, 1], oof[v, 2] = m1.predict_proba(xv)[:, 1], m2.predict_proba(xv)[:, 1], m3.predict_proba(xv)[:, 1]
        preds[:, 0] += m1.predict_proba(test_X)[:, 1] / n_splits
        preds[:, 1] += m2.predict_proba(test_X)[:, 1] / n_splits
        preds[:, 2] += m3.predict_proba(test_X)[:, 1] / n_splits
        
        fold_auc = roc_auc_score(yv, (oof[v, 0] + oof[v, 1] + oof[v, 2]) / 3)
        print(f"âœ… Fold {f+1} Done | AUC: {fold_auc:.5f}")
        
    # Change 3: Meta-Learner as Ridge for better stability
    meta = Ridge(alpha=1.0).fit(oof, train_y)
    final_test_preds = meta.predict(preds)
    final_oof_preds = meta.predict(oof)
    
    return final_test_preds, final_oof_preds


def engineer_features_advanced(df):
    # Copy banana zaroori hai taaki original data kharab na ho
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    # Columns ke naam check karein (Safe handling)
    num_cols = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'age']
    
    # Missing value treatment & Outlier clipping
    for col in num_cols:
        if col in df.columns:
            # Outliers handle karein
            lower_limit = df[col].quantile(0.01)
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(lower_limit, upper_limit)
            
            # 0 ko NaN karein aur fill karein
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    # Health Indicators (Ab safety check ke saath)
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[1, 2, 3, 4]).astype(float)
    
    if 'glucose' in df.columns:
        df['glucose_risk'] = pd.cut(df['glucose'], bins=[0, 100, 140, 200, 1000], labels=[1, 2, 3, 4]).astype(float)
    
    # Interaction Features
    if 'age' in df.columns and 'glucose' in df.columns:
        df['age_glucose'] = df['age'] * df['glucose']
    
    if 'bmi' in df.columns and 'insulin' in df.columns:
        df['bmi_insulin'] = df['bmi'] * df['insulin']
        
    if 'glucose' in df.columns and 'insulin' in df.columns:
        df['homa_ir'] = (df['glucose'] * df['insulin']) / 405
        
    if 'glucose' in df.columns and 'bmi' in df.columns:
        df['metabolic_index'] = (df['glucose'] * df['bmi']) / 100
    
    if 'insulin' in df.columns:
        df['log_insulin'] = np.log1p(df['insulin'])
    
    # Sabse aakhir mein sirf wahi columns rakhein jo model ko chahiye
    return pd.get_dummies(df, drop_first=True)


def final_feature_factory(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # --- 1. BP & Heart (7 Features) ---
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-5)
    df['bp_crisis'] = ((df['systolic_bp'] > 180) | (df['diastolic_bp'] > 120)).astype(int)
    df['bp_heart_rate'] = df['systolic_bp'] * df['heart_rate']
    df['bp_sum'] = df['systolic_bp'] + df['diastolic_bp'] # New Safe
    df['heart_age_index'] = df['heart_rate'] * df['age'] # New Safe

    # --- 2. Lipids & Cholesterol (6 Features) ---
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
    df['total_lipids'] = df['cholesterol_total'] + df['triglycerides']
    df['lipid_index'] = (df['triglycerides'] * df['ldl_cholesterol']) / (df['hdl_cholesterol'] + 1e-5) # New Safe
    df['chol_age_ratio'] = df['cholesterol_total'] / (df['age'] + 1) # New Safe

    # --- 3. BMI & Physical (5 Features) ---
    df['bmi_age'] = df['bmi'] * df['age']
    df['waist_hip_bmi'] = df['waist_to_hip_ratio'] * df['bmi']
    df['is_obese_st2'] = (df['bmi'] >= 35).astype(int)
    df['bmi_per_age'] = df['bmi'] / (df['age'] + 1) # New Safe
    df['waist_age_ratio'] = df['waist_to_hip_ratio'] / (df['age'] + 1) # New Safe
    
    # --- 4. Lifestyle & Balance (5 Features) ---
    df['lifestyle_stress'] = df['screen_time_hours_per_day'] / (df['sleep_hours_per_day'] + 1e-5)
    df['activity_per_age'] = df['physical_activity_minutes_per_week'] / (df['age'] + 1)
    df['diet_sleep_balance'] = df['diet_score'] * df['sleep_hours_per_day']
    df['stress_to_activity'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week']/60 + 1) # New Safe
    df['sleep_per_age'] = df['sleep_hours_per_day'] / (df['age'] + 1) # New Safe
    
    # --- 5. Statistical Groupings (6 Features from Loop) ---
    for col in ['bmi', 'systolic_bp', 'cholesterol_total']:
        df[f'avg_{col}_age'] = df.groupby('age')[col].transform('mean')
        df[f'{col}_vs_age_avg'] = df[col] - df[f'avg_{col}_age']
    
    # --- 6. Original Features (Approx 16 Features) ---
    # (Age, Gender, Smoker, etc. included)

    # Auto-Handling Categoricals
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]
            
    return df

