import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# ==============================================================================
# 2. Data Loading
# ==============================================================================
try:
    train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
    print(f"Data Loaded: Train shape {train.shape}, Test shape {test.shape}")
except FileNotFoundError:
    print("Error: CSV file not found, please check the path.")
    exit()

TARGET = 'diagnosed_diabetes'

train['is_train'] = 1
test['is_train'] = 0
test[TARGET] = np.nan 

df_all = pd.concat([train, test], axis=0, ignore_index=True)


# ==============================================================================
# 3. Feature Engineering
# ==============================================================================
print("Starting Robust Feature Engineering...")

# --- 3.1 Log Transform ---
skewed_cols = ['triglycerides', 'ldl_cholesterol', 'hdl_cholesterol', 'insulin', 'glucose']
for col in skewed_cols:
    if col in df_all.columns:
        # log1p = log(x + 1)
        df_all[f'Log_{col}'] = np.log1p(df_all[col])

# --- 3.2 Medical Ratios
# 1. Atherogenic Index of Plasma (AIP): log(Triglycerides / HDL)
if 'triglycerides' in df_all.columns and 'hdl_cholesterol' in df_all.columns:
    df_all['AIP_Index'] = np.log1p(df_all['triglycerides']) - np.log1p(df_all['hdl_cholesterol'])

# 2. Non-HDL Cholesterol: Total - HDL
if 'cholesterol_total' in df_all.columns and 'hdl_cholesterol' in df_all.columns:
    df_all['Non_HDL'] = df_all['cholesterol_total'] - df_all['hdl_cholesterol']
    df_all['Chol_HDL_Ratio'] = df_all['cholesterol_total'] / (df_all['hdl_cholesterol'] + 1e-5)

# 3. Glucose/Insulin
if 'glucose' in df_all.columns and 'insulin' in df_all.columns:
    df_all['HOMA_Proxy'] = df_all['glucose'] * df_all['insulin']

# 4. Pulse Pressure
if 'systolic_bp' in df_all.columns and 'diastolic_bp' in df_all.columns:
    df_all['Pulse_Pressure'] = df_all['systolic_bp'] - df_all['diastolic_bp']
    # MAP (Mean Arterial Pressure)
    df_all['MAP'] = df_all['diastolic_bp'] + (df_all['Pulse_Pressure'] / 3)

# --- 3.3 Interactions ---
# Obesity + Hypertension = High Risk
if 'bmi' in df_all.columns and 'systolic_bp' in df_all.columns:
    df_all['BMI_BP_Interact'] = df_all['bmi'] * df_all['systolic_bp']

# --- 3.4 Basic Encoding Processing ---
binary_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 'gender']
for col in binary_cols:
    if col in df_all.columns:
        df_all[col] = df_all[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}).fillna(0)

# --- 3.5 Categorical Feature Encoding ---
cat_cols = ['smoking_status', 'employment_status', 'education_level', 'income_level', 'ethnicity'] 
for col in cat_cols:
    if col in df_all.columns:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str))

# --- 3.6 Target Encoding ---
def add_kfold_target_encoding(df, cat_cols, target_col, n_folds=5):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_mask = df['is_train'] == 1
    test_mask = df['is_train'] == 0
    train_df = df[train_mask].copy()
    
    for col in cat_cols:
        if col not in df.columns: continue
        new_col_name = f'{col}_target_enc'
        df[new_col_name] = np.nan
        for tr_idx, val_idx in kf.split(train_df, train_df[target_col]):
            X_tr = train_df.iloc[tr_idx]
            means = X_tr.groupby(col)[target_col].mean()
            original_val_idx = train_df.iloc[val_idx].index
            df.loc[original_val_idx, new_col_name] = df.loc[original_val_idx, col].map(means)
        global_means = train_df.groupby(col)[target_col].mean()
        global_mean_val = train_df[target_col].mean()
        df.loc[test_mask, new_col_name] = df.loc[test_mask, col].map(global_means)
        df[new_col_name].fillna(global_mean_val, inplace=True)
    return df

target_enc_cols = ['ethnicity', 'education_level', 'employment_status'] 
df_all = add_kfold_target_encoding(df_all, target_enc_cols, TARGET)

# --- 3.7 Data Splitting ---
drop_cols = ['id', 'is_train', TARGET] 

features = [col for col in df_all.columns if col not in drop_cols]

train_processed = df_all[df_all['is_train'] == 1].reset_index(drop=True)
test_processed = df_all[df_all['is_train'] == 0].reset_index(drop=True)

X = train_processed[features]
y = train_processed[TARGET]
X_test = test_processed[features]

print(f"Feature Engineering Complete. Total Features: {len(features)}")


# ==============================================================================
# 4. Model Training
# ==============================================================================
neg_count, pos_count = np.bincount(y)
scale_weight = neg_count / pos_count
print(f"Class Imbalance Check -> Neg(0): {neg_count}, Pos(1): {pos_count}, Scale Weight: {scale_weight:.4f}")

xgb_params = {
    'n_estimators': 3000, 
    'learning_rate': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'n_jobs': -1,
    'random_state': 42,
    'early_stopping_rounds': 200,
    'reg_alpha': 10,
    'reg_lambda': 10,
    'scale_pos_weight': scale_weight,
    'tree_method': 'hist'
}

lgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'num_leaves': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'n_jobs': -1,
    'random_state': 42,
    'metric': 'auc',
    'verbosity': -1,
    'reg_alpha': 10,
    'reg_lambda': 10,
    'scale_pos_weight': scale_weight
}

cat_params = {
    'iterations': 3000,
    'learning_rate': 0.01,
    'depth': 8,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 200,
    'allow_writing_files': False,
    'l2_leaf_reg': 10,
    'auto_class_weights': 'Balanced'
}

def train_model(model_type, X, y, X_test, params, n_folds=5):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # Initialize feature importance storage
    importances = pd.DataFrame()
    importances['feature'] = X.columns
    importances['importance'] = 0
    
    print(f"--- Training {model_type.upper()} ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            importances['importance'] += model.feature_importances_ / n_folds
            
        elif model_type == 'lightgbm':
            callbacks = [lgb.early_stopping(200, verbose=False)]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=callbacks)
            importances['importance'] += model.feature_importances_ / n_folds
            
        elif model_type == 'catboost':
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            importances['importance'] += model.feature_importances_ / n_folds
            
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds
    
    auc = roc_auc_score(y, oof_preds)
    print(f"{model_type} OOF AUC: {auc:.5f}")
    return oof_preds, test_preds, importances

# Prepare LightGBM data (column name processing)
X_lgb = X.copy()
X_test_lgb = X_test.copy()
cols = [re.sub(r'[^\w]', '_', c) for c in X_lgb.columns]
X_lgb.columns = cols; X_test_lgb.columns = cols

# Train and get feature importance
oof_xgb, pred_xgb, imp_xgb = train_model('xgboost', X, y, X_test, xgb_params)
oof_lgb, pred_lgb, imp_lgb = train_model('lightgbm', X_lgb, y, X_test_lgb, lgb_params)
oof_cat, pred_cat, imp_cat = train_model('catboost', X, y, X_test, cat_params)


# ==============================================================================
# 5. Stacking Ensemble
# ==============================================================================
print("\n>>> Starting Stacking Optimization...")

# Construct the Meta-Learner training set (using OOF predictions)
train_stack = pd.DataFrame({
    'xgb': oof_xgb,
    'lgb': oof_lgb,
    'cat': oof_cat
})

# Construct the Meta-Learner test set
test_stack = pd.DataFrame({
    'xgb': pred_xgb,
    'lgb': pred_lgb,
    'cat': pred_cat
})

meta_model = LogisticRegression()
meta_model.fit(train_stack, y)

# Retrieve stacking weights/coefficients
weights = meta_model.coef_[0]
print(f"Stacking Weights -> XGB: {weights[0]:.3f}, LGB: {weights[1]:.3f}, CAT: {weights[2]:.3f}")

# Generate final predictions
stack_oof_pred = meta_model.predict_proba(train_stack)[:, 1]
stack_test_pred = meta_model.predict_proba(test_stack)[:, 1]

final_auc = roc_auc_score(y, stack_oof_pred)
print(f"\n>>> Final Stacking AUC: {final_auc:.5f}")


# ==============================================================================
# 6. Threshold Optimization and Submission
# ==============================================================================
best_thresh = 0.5
best_f1 = 0
thresholds = np.arange(0.3, 0.7, 0.005)

for thresh in thresholds:
    pred_binary = (stack_oof_pred > thresh).astype(int)
    score = f1_score(y, pred_binary)
    if score > best_f1:
        best_f1 = score
        best_thresh = thresh

print(f">>> Best F1 Threshold (For Reference Only): {best_thresh:.3f}")
print(f">>> Best F1 Score (Local CV): {best_f1:.5f}")

# Confusion Matrix Check (Local)
cm = confusion_matrix(y, (stack_oof_pred > best_thresh).astype(int))
print("\nConfusion Matrix (Best Threshold):")
print(cm)

if 'id' in test.columns:
    sub = pd.DataFrame({
        'id': test['id'], 
        TARGET: stack_test_pred
    })
    
    sub.to_csv('submission.csv', index=False)
    
    print("\n✅ Submission saved to 'submission.csv'")


# ==============================================================================
# 7. Visualization Analysis
# ==============================================================================
fig = plt.figure(figsize=(18, 12))

# 1. ROC Curve
ax1 = fig.add_subplot(2, 2, 1)
fpr, tpr, _ = roc_curve(y, stack_oof_pred) 
ax1.plot(fpr, tpr, label=f'Stacking AUC = {final_auc:.4f}', color='darkorange', linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Confusion Matrix
ax2 = fig.add_subplot(2, 2, 2)
cm = confusion_matrix(y, (stack_oof_pred > best_thresh).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_title(f'Confusion Matrix (Threshold={best_thresh:.3f})')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

# 3. Feature Importance
ax3 = fig.add_subplot(2, 2, 3)
top_features = imp_xgb.sort_values(by='importance', ascending=False).head(20)
sns.barplot(x='importance', y='feature', data=top_features, palette='viridis', ax=ax3)
ax3.set_title('Top 20 Features (XGBoost)')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# 4. Predicted Probability Distribution Check
ax4 = fig.add_subplot(2, 2, 4)
sns.kdeplot(stack_oof_pred[y==0], label='Healthy (0)', fill=True, color='blue', alpha=0.2, ax=ax4)
sns.kdeplot(stack_oof_pred[y==1], label='Diabetes (1)', fill=True, color='red', alpha=0.2, ax=ax4)
ax4.axvline(best_thresh, color='green', linestyle='--', label=f'Threshold {best_thresh:.2f}')
ax4.set_title('Predicted Probability Distribution')
ax4.set_xlabel('Probability')
ax4.legend()

plt.tight_layout()
plt.show()







