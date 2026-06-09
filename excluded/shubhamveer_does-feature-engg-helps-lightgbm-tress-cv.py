from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train.drop('id', inplace=True, axis=1)
test.drop('id', inplace=True, axis=1)


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def add_features_numeric_only(df):
    df = df.copy()
    
    # Age features
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['is_young_adult'] = df['age'].between(18, 30).astype(int)
    df['age_decade'] = (df['age'] // 10) * 10
    df['age_zscore'] = (df['age'] - df['age'].mean()) / df['age'].std()
    df['age_bin'] = pd.cut(df['age'], bins=range(15, 100, 5), labels=False)
    
    # Label encode categorical columns safely
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    
    # Job features
    job_freq = df['job'].value_counts(normalize=True)
    df['job_freq'] = df['job'].map(job_freq).fillna(0)
    df['job_is_high_profile'] = df['job'].isin(['management', 'admin.', 'technician']).astype(int)
    df['is_self_employed'] = df['job'].isin(['self-employed', 'entrepreneur']).astype(int)
    
    # Marital features
    df['is_married'] = (df['marital'] == 'married').astype(int)
    df['is_single_or_divorced'] = df['marital'].isin(['single', 'divorced']).astype(int)
    
    # Education features
    df['is_educated'] = df['education'].isin(['tertiary', 'secondary']).astype(int)
    df['unknown_education'] = (df['education'] == 'unknown').astype(int)
    df['edu_job_match'] = ((df['education'] == 'tertiary') & df['job_is_high_profile'].astype(bool)).astype(int)
    
    # Financial features
    df['balance_log'] = df['balance'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    df['is_balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_zscore'] = (df['balance'] - df['balance'].mean()) / df['balance'].std()
    df['balance_bucket'] = pd.qcut(df['balance'], 5, labels=False, duplicates='drop')
    df['has_high_balance'] = (df['balance'] > df['balance'].quantile(0.75)).astype(int)
    
    # Loan & Housing features
    df['has_any_loan'] = ((df['loan'] == 'yes') | (df['housing'] == 'yes')).astype(int)
    df['has_both_loans'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)
    
    # Loan balance ratio: balance / loan indicator (replace 0 with NaN to avoid div by zero)
    loan_indicator = (df['loan'] == 'yes').astype(int).replace(0, np.nan)
    df['loan_balance_ratio'] = df['balance'] / loan_indicator
    
    # Contact features
    df['is_mobile_contact'] = (df['contact'] == 'cellular').astype(int)
    df['unknown_contact'] = (df['contact'] == 'unknown').astype(int)
    df['preferred_contact_score'] = df['contact'].map({'cellular': 2, 'telephone': 1, 'unknown': 0}).fillna(0)
    
    # Campaign & Previous Contact features
    df['calls_per_day'] = df['campaign'] / df['day'].replace(0, np.nan)
    df['multiple_contacts_flag'] = (df['campaign'] > 3).astype(int)
    df['pdays_flag'] = (df['pdays'] != -1).astype(int)
    df['days_since_last_contact_bucket'] = pd.cut(df['pdays'], bins=[-2,0,30,90,180,999], labels=False)
    df['previous_contact_ratio'] = df['previous'] / df['campaign'].replace(0, np.nan)
    
    # Temporal features
    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    df['month_enc'] = df['month'].map(month_map).fillna(0).astype(int)
    season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}  # winter=0,spring=1,summer=2,fall=3
    df['season'] = df['month_enc'].map(season_map).fillna(-1).astype(int)
    df['is_month_end'] = (df['day'] > 25).astype(int)
    df['day_of_week_estimate'] = df['day'] % 7
    df['is_q2'] = df['month'].isin(['apr','may','jun']).astype(int)
    
    # Interaction features (example 10)
    df['age_x_balance'] = df['age'] * df['balance']
    df['campaign_x_previous'] = df['campaign'] * df['previous']
    df['pdays_x_poutcome'] = df['pdays'] * df['poutcome_enc'].fillna(0)
    df['balance_per_campaign'] = df['balance'] / (df['campaign'].replace(0, np.nan))
    df['duration_per_campaign'] = df['duration'] / (df['campaign'].replace(0, np.nan))
    df['age_bin_x_is_married'] = df['age_bin'].fillna(-1).astype(int) * df['is_married']
    df['job_enc_x_loan'] = df['job_enc'] * (df['loan'] == 'yes').astype(int)
    df['balance_log_x_is_balance_pos'] = df['balance_log'] * df['is_balance_positive']
    df['calls_per_day_x_multiple_contacts'] = df['calls_per_day'] * df['multiple_contacts_flag']
    df['season_x_is_q2'] = df['season'] * df['is_q2']
    
    # Drop original categorical string columns to avoid dtype issues
    original_cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df = df.drop(columns=[c for c in original_cat_cols if c in df.columns])
    
    # Fill any remaining NaNs (for safety)
    df = df.fillna(0)
    
    # Ensure all columns are numeric type
    for c in df.columns:
        if df[c].dtype == 'bool':
            df[c] = df[c].astype(int)
        elif df[c].dtype.name == 'category':
            df[c] = df[c].astype(int)
    
    return df




# ====== Training example ======

def train_lightgbm_cv(train, test, target_col='y'):
    X = train.drop(columns=[target_col])
    y = train[target_col]
    
    X_test = test.copy()
    
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_probs = np.zeros(len(X_test))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nðŸ”¹ Training fold {fold + 1}/{n_splits} >>>")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        X_train_fe = add_features_numeric_only(X_train)
        X_val_fe = add_features_numeric_only(X_val)
        X_test_fe = add_features_numeric_only(X_test)
        
        model = lgb.LGBMClassifier(
            n_estimators=30000,
            class_weights='balanced',
            learning_rate=0.06,
            num_leaves=100,
            max_depth=10,
            min_child_samples=7,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_alpha=0.8,
            reg_lambda=0.3,
            max_bin=4859,
            random_state=2003,
            verbosity=-1,
            boosting_type='gbdt',
            eval_metric='auc',
            metric='auc'
        )
        
        model.fit(
            X_train_fe, y_train,
            eval_set=[(X_val_fe, y_val)],
            callbacks=[
                lgb.early_stopping(300),
                lgb.log_evaluation(500)
            ]
        )
        
        models.append(model)
        y_probs += model.predict_proba(X_test_fe)[:, 1] / n_splits
    
    print("\nâœ… Cross-validation training complete.")
    return y_probs, models

# ==== Example usage ====
y_probs, models = train_lightgbm_cv(train, test, target_col='y')




import pandas as pd

# Load the test file for 'id' column
testify = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# y_probs is already the averaged predictions from all folds
submission = pd.DataFrame({
    'id': testify['id'],
    'target': y_probs  # from CV loop
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file saved as 'submission.csv'")





