# --- 1. Libraries & Configuration ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb

# Configuration
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

print("Libraries Loaded Successfully.")


# --- 2. Load Data ---
# Please ensure the paths match your Kaggle directory
try:
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
    print(f"Train Shape: {df_train.shape}")
    print(f"Test Shape: {df_test.shape}")
except FileNotFoundError:
    print("âš ï¸� File not found. Please check your dataset path.")

# --- 3. Quick Inspection ---
print("\n--- Data Info ---")
df_train.info()

print("\n--- First 3 Rows ---")
display(df_train.head(3))


# --- 2.1 Target Distribution ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='loan_paid_back', data=df_train, palette='viridis')
plt.title('Target Distribution: Loan Paid Back vs Default', fontsize=12)
plt.xlabel('Loan Paid Back (0 = No, 1 = Yes)')
plt.ylabel('Count')

# Add percentages
total = len(df_train)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 5000
    ax.annotate(percentage, (x, y), ha='center')
plt.show()

# --- 2.2 The "Truth" Plot for Grade Subgrade ---
# Calculate repayment probability per grade
grade_trend = df_train.groupby('grade_subgrade')['loan_paid_back'].mean().reset_index()
grade_trend = grade_trend.sort_values('grade_subgrade') # Sort Alphabetically

plt.figure(figsize=(14, 6))
sns.barplot(x='grade_subgrade', y='loan_paid_back', data=grade_trend, color='skyblue', alpha=0.8)

# Add the Trend Line (The Red Line)
plt.plot(range(len(grade_trend)), grade_trend['loan_paid_back'], 
         color='red', marker='o', linewidth=2, label='Repayment Probability Trend')

plt.title('Hypothesis Test: Is Grade Ordinal? (Look at the Red Line)', fontsize=14)
plt.ylabel('Probability of Repayment (1.0 = Safe)', fontsize=12)
plt.xlabel('Grade Subgrade', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0.5, 1.0) # Zoom in to see the drop
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()


# --- 3.1 Preprocessing for Baseline ---
# Create a copy to avoid modifying original data
df_base = df_train.copy()
df_test_base = df_test.copy()

# Identify Categorical Columns
cat_cols = df_base.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical Columns to Encode: {cat_cols}")

# Simple Ordinal Encoding (Assigns 1, 2, 3 randomly based on string)
# We use this for the baseline to simulate "minimal effort"
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_base[cat_cols] = enc.fit_transform(df_base[cat_cols])
df_test_base[cat_cols] = enc.transform(df_test_base[cat_cols])

# --- 3.2 Stratified K-Fold Training Function ---
def train_model(df, features, model_name):
    X = df[features]
    y = df['loan_paid_back']
    
    # LightGBM Parameters (Optimized for Imbalance)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,            # Restrict depth to prevent overfitting
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    scores = []
    
    print(f"--- Training {model_name} ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        
        # Note: Early stopping is handled via callbacks in new sklearn API
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        score = roc_auc_score(y_val, val_preds)
        scores.append(score)
        print(f"Fold {fold+1} AUC: {score:.5f}")
        
    mean_score = np.mean(scores)
    print(f"\nâœ… {model_name} Average AUC: {mean_score:.5f}")
    return oof_preds, mean_score

# --- 3.3 Run Baseline ---
features_base = [c for c in df_base.columns if c not in ['id', 'loan_paid_back']]
oof_base, score_base = train_model(df_base, features_base, "Baseline Model")


# --- 4.1 Feature Engineering Function ---
def engineer_features(df_in):
    df = df_in.copy()
    
    # 1. The "Killer Feature": Custom Ordinal Grade Mapping
    # We dynamically sort them to ensure A1 < A2 < ... < G5
    # (Assumes the letters are statistically meaningful, which we proved)
    unique_grades = sorted(df_train['grade_subgrade'].unique())
    grade_map = {grade: i for i, grade in enumerate(unique_grades)}
    
    df['grade_risk_score'] = df['grade_subgrade'].map(grade_map)
    
    # Handle potential new grades in test set (fill with max risk)
    df['grade_risk_score'] = df['grade_risk_score'].fillna(max(grade_map.values()) + 1)
    
    # 2. Financial Ratios
    # Log transform income to reduce skewness impact
    df['log_annual_income'] = np.log1p(df['annual_income'])
    
    # Affordability Ratio
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    
    # Estimated Monthly Debt (Reverse engineering DTI)
    # DTI = (Monthly Debt / Monthly Income) * 100
    # So, Monthly Debt = (DTI / 100) * (Annual Income / 12)
    df['estimated_monthly_debt'] = (df['debt_to_income_ratio'] / 100) * (df['annual_income'] / 12)
    
    # Drop the original string column (it's now redundant)
    df = df.drop(columns=['grade_subgrade'])
    
    return df

# --- 4.2 Apply Engineering ---
print("Generating Features...")
df_fe = engineer_features(df_train)
df_test_fe = engineer_features(df_test)

# --- 4.3 Encoding Remaining Categoricals ---
# We encoded 'grade_subgrade' manually, but we still need to encode Gender, Purpose, etc.
# Use the same logic as baseline for the others
cat_cols_fe = df_fe.select_dtypes(include=['object']).columns.tolist()
print(f"Remaining Categoricals to Encode: {cat_cols_fe}")

enc_fe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_fe[cat_cols_fe] = enc_fe.fit_transform(df_fe[cat_cols_fe])
df_test_fe[cat_cols_fe] = enc_fe.transform(df_test_fe[cat_cols_fe])

# --- 4.4 Train Advanced Model ---
features_fe = [c for c in df_fe.columns if c not in ['id', 'loan_paid_back']]

# We use the EXACT SAME training function to ensure a fair comparison
oof_fe, score_fe = train_model(df_fe, features_fe, "Advanced Model (With FE)")


# --- 5.1 Plot ROC Curve Comparison ---
plt.figure(figsize=(10, 8))

fpr_base, tpr_base, _ = roc_curve(df_train['loan_paid_back'], oof_base)
fpr_fe, tpr_fe, _ = roc_curve(df_train['loan_paid_back'], oof_fe)

plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {score_base:.5f})', color='blue', linestyle='--')
plt.plot(fpr_fe, tpr_fe, label=f'Feature Eng (AUC = {score_fe:.5f})', color='red', alpha=0.7)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Baseline vs Advanced Model')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

# --- 5.2 Generate Submission File ---
# We need to make predictions on the TEST set
# Model 1 Predictions (Baseline)
model_base = lgb.LGBMClassifier(
    objective='binary', metric='auc', n_estimators=1000, learning_rate=0.05,
    max_depth=7, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model_base.fit(df_base.drop(columns=['id', 'loan_paid_back']), df_base['loan_paid_back'])
preds_base = model_base.predict_proba(df_test_base.drop(columns=['id']))[:, 1]

# Model 2 Predictions (Advanced)
model_fe = lgb.LGBMClassifier(
    objective='binary', metric='auc', n_estimators=1000, learning_rate=0.05,
    max_depth=7, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model_fe.fit(df_fe.drop(columns=['id', 'loan_paid_back']), df_fe['loan_paid_back'])
preds_fe = model_fe.predict_proba(df_test_fe.drop(columns=['id']))[:, 1]

# Ensemble (Average)
final_preds = (preds_base * 0.5) + (preds_fe * 0.5)



from xgboost import XGBClassifier

# --- 6.1 XGBoost Training Function ---
def train_xgb(df, features):
    X = df[features]
    y = df['loan_paid_back']
    
    # XGBoost Hyperparameters (Tuned for 80/20 Imbalance)
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist' # Faster training
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    scores = []
    
    print(f"--- Training XGBoost ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        score = roc_auc_score(y_val, val_preds)
        scores.append(score)
        print(f"Fold {fold+1} AUC: {score:.5f}")
        
    mean_score = np.mean(scores)
    print(f"\nâœ… XGBoost Average AUC: {mean_score:.5f}")
    return oof_preds, mean_score, model

# --- 6.2 Run XGBoost on our Best Data (Ordinal/Baseline) ---
# We use df_base because our FE didn't help much, and XGBoost likes raw ordinal data
oof_xgb, score_xgb, last_model_xgb = train_xgb(df_base, features_base)

# --- 6.3 Generate XGB Test Predictions ---
# Retrain on full data or use the last fold model (quick approximation for now)
# For best results, we usually average predictions from all 5 folds, 
# but for this step, let's just fit one model on full train to predict test.
full_xgb = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=6, 
    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
    eval_metric='auc', random_state=42, n_jobs=-1, tree_method='hist'
)
full_xgb.fit(df_base[features_base], df_base['loan_paid_back'])
preds_xgb = full_xgb.predict_proba(df_test_base[features_base])[:, 1]



# --- 7.1 Plotting the Battle of the Models ---
plt.figure(figsize=(12, 10))

# Calculate FPR/TPR for all models
fpr_base, tpr_base, _ = roc_curve(df_train['loan_paid_back'], oof_base)
fpr_fe, tpr_fe, _ = roc_curve(df_train['loan_paid_back'], oof_fe)
fpr_xgb, tpr_xgb, _ = roc_curve(df_train['loan_paid_back'], oof_xgb)

# Plot Curves
plt.plot(fpr_base, tpr_base, label=f'LGBM Baseline (AUC = {score_base:.5f}) ğŸ�†', color='blue', linewidth=2)
plt.plot(fpr_fe, tpr_fe, label=f'LGBM Feature Eng (AUC = {score_fe:.5f})', color='green', linestyle='--')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {score_xgb:.5f})', color='red', linestyle='-.')

# Plot Random Guess
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)

# Aesthetics
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final Model Comparison: Who is the Winner?', fontsize=15)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)

# Zoom in on the top left corner (Optional, to see the difference)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1.0)

plt.show()

# --- 7.2 Select the Winner & Submit ---
scores = {
    'LGBM_Baseline': score_base,
    'LGBM_FeatureEng': score_fe,
    'XGBoost': score_xgb
}

winner_name = max(scores, key=scores.get)
winner_score = scores[winner_name]

print(f"\nâ­�â­� AND THE WINNER IS: {winner_name} with AUC {winner_score:.5f} â­�â­�\n")

# --- 7.3 Generate Final Submission for the Winner ---
print("Generating submission file for the winner...")

# We need to ensure we have the predictions for the winner ready
# (Re-generating specifically for the best model to be 100% sure)

if winner_name == 'LGBM_Baseline':
    # We already trained this in Step 3, but let's be safe and use the preds from Step 5
    final_submission_preds = preds_base
    
elif winner_name == 'LGBM_FeatureEng':
    final_submission_preds = preds_fe
    
elif winner_name == 'XGBoost':
    final_submission_preds = preds_xgb

# Create CSV
submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': final_submission_preds
})

filename = f'submission_best_model_{winner_name}.csv'
submission.to_csv(filename, index=False)

print(f"âœ… Successfully saved: {filename}")
print(submission.head())

