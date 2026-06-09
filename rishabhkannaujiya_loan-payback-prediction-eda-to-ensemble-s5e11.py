import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

!pip install -q category_encoders
import category_encoders as ce

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool



train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


FOLDS = 10
SEED = 42
TARGET = 'loan_paid_back'


# Drop ID columns
if 'id' in train_df.columns:
    train_df = train_df.drop(columns=['id'])
    test_df = test_df.drop(columns=['id'])


# Separate Target
y = train_df[TARGET]
X = train_df.drop(columns=[TARGET])
X_test = test_df.copy()

train_len = len(X)
full_df = pd.concat([X, X_test], axis=0).reset_index(drop=True)


num_cols = full_df.select_dtypes(exclude=['object', 'category']).columns.tolist()
cat_cols = full_df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in num_cols:
    full_df[col] = full_df[col].fillna(full_df[col].median())

for col in cat_cols:
    full_df[col] = full_df[col].fillna("Missing").astype(str)

print(f"Data Loaded. Train Shape: {X.shape}, Test Shape: {X_test.shape}")


print("Engineering Features (Binning & Aggregations)...")

def create_features(df):
    # Income buffering (+1) to avoid division by zero
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_burden'] = df['loan_amount'] / (df['annual_income'] / 12 + 1)
    df['interest_burden'] = df['loan_amount'] * (df['interest_rate'] / 100)
    
    # Log Transforms (Normalize skewed money data)
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_loan'] = np.log1p(df['loan_amount'])
    
    # Binning (Crucial for Tree Models)
    # 1. Credit Score Bins
    df['credit_score_bin'] = pd.cut(df['credit_score'], 
                                    bins=[0, 580, 670, 740, 800, 900], 
                                    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']).astype(str)
    
    # 2. Loan Amount Bins (Quantiles)
    df['loan_bin'] = pd.qcut(df['loan_amount'], q=10, labels=False).astype(str)

    # GroupBy Aggregations (Relative Performance)
    # Note: We use transform so the shape of the dataframe doesn't change
    
    # "Is this loan bigger than the average loan for this grade?"
    df['loan_vs_grade_mean'] = df['loan_amount'] / df.groupby('grade_subgrade')['loan_amount'].transform('mean')
    
    # "Is this person richer than the average person with this job?"
    df['income_vs_job_mean'] = df['annual_income'] / df.groupby('employment_status')['annual_income'].transform('mean')
    
    return df


full_df = create_features(full_df)

X = full_df.iloc[:train_len].copy()
X_test = full_df.iloc[train_len:].copy()

# Re-identify categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical Features: {len(cat_cols)} variables")


print("Running Adversarial Validation Check...")
# We train a quick model to see if it can distinguish Train from Test.
# If AUC is ~0.5, Train and Test are similar (Good). If > 0.70, we have drift.

adv_train = X.copy()
adv_test = X_test.copy()
adv_train['is_test'] = 0
adv_test['is_test'] = 1
adv_data = pd.concat([adv_train, adv_test], axis=0).reset_index(drop=True)
adv_y = adv_data['is_test']
adv_X = adv_data.drop(columns=['is_test'])

# Quick ordinal encoding for the check
adv_X_enc = ce.OrdinalEncoder(cols=cat_cols).fit_transform(adv_X)

adv_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, 
                          tree_method='hist', device='cuda', eval_metric='auc', random_state=SEED)
                          
adv_cv_score = cross_val_score(adv_model, adv_X_enc, adv_y, cv=3, scoring='roc_auc').mean()

print(f"Adversarial AUC: {adv_cv_score:.4f}")
if adv_cv_score > 0.70:
    print("âš ï¸� WARNING: Train and Test sets look different. Consider dropping drifting features.")
else:
    print("âœ… PASSED: Train and Test sets are similar. Proceeding...")

del adv_data, adv_X, adv_y, adv_X_enc, adv_train, adv_test
gc.collect()



print(f"Starting {FOLDS}-Fold Stacking Ensemble...")

# Initialize arrays for Out-of-Fold (OOF) preds and Test preds
xgb_oof = np.zeros(len(X))
lgbm_oof = np.zeros(len(X))
cat_oof = np.zeros(len(X))

xgb_test = np.zeros(len(X_test))
lgbm_test = np.zeros(len(X_test))
cat_test = np.zeros(len(X_test))



skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)


xgb_params = {
    'learning_rate': 0.06306402264031599,
    'max_depth': 6,
    'subsample': 0.9877243257173147,
    'colsample_bytree': 0.9424086534559548,
    'min_child_weight': 4,
    'reg_lambda': 0.008595178377302507,
    'reg_alpha': 0.05542369465609152,
    'gamma': 0.2691485091993745,
    
    # Fixed Parameters for GPU
    'n_estimators': 5000,
    'device': 'cuda',
    'tree_method': 'hist',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    'enable_categorical': False
}


lgbm_params = {
    'learning_rate': 0.11800628949217232,
    'num_leaves': 110,
    'max_depth': 5,
    'min_child_samples': 34,
    'subsample': 0.8554501523090093,
    'colsample_bytree': 0.507019653413801,
    'reg_alpha': 0.0922019636447731,
    'reg_lambda': 0.03375421066057399,
    
    # Fixed Params
    'n_estimators': 5000,
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'gpu_use_dp': False,
    'force_col_wise': True
}


cat_params = {
    'learning_rate': 0.13920677408646712,
    'depth': 5,
    'l2_leaf_reg': 0.5223157283384439,
    'random_strength': 3.690512444785991e-08,
    'border_count': 252,
    'subsample': 0.652240898735347,
    
    # Fixed GPU Params
    'iterations': 5000,
    'task_type': 'GPU',
    'devices': '0',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 100,
    'bootstrap_type': 'Bernoulli' 
}


for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y)), total=FOLDS):
    # 1. Split Data
    X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
    X_te = X_test.copy()
    
    # 2. Robust Target Encoding (Prevents Leakage)
    # Note: CatBoost handles categories natively, so we keep a separate copy for it
    te_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=20)
    
    # Fit on Train, Transform Val/Test
    X_tr_enc = te_encoder.fit_transform(X_tr, y_tr)
    X_val_enc = te_encoder.transform(X_val)
    X_te_enc = te_encoder.transform(X_te)
    
    # 3. Train XGBoost
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
    xgb_oof[val_idx] = model_xgb.predict_proba(X_val_enc)[:, 1]
    xgb_test += model_xgb.predict_proba(X_te_enc)[:, 1] / FOLDS
    
    # 4. Train LightGBM
    model_lgbm = LGBMClassifier(**lgbm_params)
    model_lgbm.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)])
    lgbm_oof[val_idx] = model_lgbm.predict_proba(X_val_enc)[:, 1]
    lgbm_test += model_lgbm.predict_proba(X_te_enc)[:, 1] / FOLDS
    
    # 5. Train CatBoost (Native Categories)
    # CatBoost works better with raw categories (strings)
    # Ensure numeric cols are filled and cats are strings
    X_tr_cat, X_val_cat, X_te_cat = X_tr.copy(), X_val.copy(), X_te.copy()
    
    model_cat = CatBoostClassifier(**cat_params, cat_features=cat_cols)
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val))
    
    cat_oof[val_idx] = model_cat.predict_proba(X_val_cat)[:, 1]
    cat_test += model_cat.predict_proba(X_te_cat)[:, 1] / FOLDS
    
    # Report Fold Score
    print(f"Fold {fold+1} AUC | XGB: {roc_auc_score(y_val, xgb_oof[val_idx]):.5f} | "
          f"LGBM: {roc_auc_score(y_val, lgbm_oof[val_idx]):.5f} | "
          f"CAT: {roc_auc_score(y_val, cat_oof[val_idx]):.5f}")


print("Training Meta-Model (Logistic Regression)...")

# Create Stacking Dataset
stack_train = pd.DataFrame({
    'xgb': xgb_oof,
    'lgbm': lgbm_oof,
    'cat': cat_oof
})
stack_test = pd.DataFrame({
    'xgb': xgb_test,
    'lgbm': lgbm_test,
    'cat': cat_test
})


meta_model = LogisticRegression()
meta_model.fit(stack_train, y)
stack_preds = meta_model.predict_proba(stack_test)[:, 1]
cv_score = roc_auc_score(y, meta_model.predict_proba(stack_train)[:, 1])

print(f"\n---> Final Stacked CV AUC: {cv_score:.5f}")
print(f"Meta-Model Coefficients: XGB:{meta_model.coef_[0][0]:.2f}, LGBM:{meta_model.coef_[0][1]:.2f}, CAT:{meta_model.coef_[0][2]:.2f}")



# A. Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(stack_train.corr(), annot=True, cmap='coolwarm', fmt=".3f")
plt.title("Correlation between Models")
plt.tight_layout()
plt.show()


# B. Feature Importance (LGBM)
import lightgbm as lgb
plt.figure(figsize=(10, 8))
lgb.plot_importance(model_lgbm, max_num_features=20, importance_type='gain', title='LGBM Feature Importance')
plt.tight_layout()
plt.show()




# C. Post-Processing (Calibration Clipping)
# Clips probability extremes to avoid confident errors
final_preds = np.clip(stack_preds, 0.001, 0.999)

# D. Save
submission[TARGET] = final_preds
submission.to_csv('submission.csv', index=False)

print("\nâœ… Done! File saved as 'submission.csv'")
print("Distribution of Predictions:")
print(submission[TARGET].describe())

plt.figure(figsize=(8, 4))
sns.histplot(final_preds, kde=True, color='green')
plt.title("Final Prediction Distribution")
plt.show()

