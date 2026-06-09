import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


TARGET = 'diagnosed_diabetes'

CATS = ['gender', 'ethnicity', 'education_level', 'income_level',
        'smoking_status', 'employment_status']


BASE = [col for col in train.columns if col not in ['id', TARGET]]
BASE = [c for c in BASE if c in train.columns and c in orig.columns]
ORIG_FEATS = []


print("Starting Feature Engineering...")

for col in BASE:
    if col in orig.columns:
        # Mean Encoding
        mean_map = orig.groupby(col)[TARGET].mean().reset_index()
        new_mean_col_name = f"orig_mean_{col}"
        mean_map = mean_map.rename(columns={TARGET: new_mean_col_name})
        train = train.merge(mean_map, on=col, how='left')
        test = test.merge(mean_map, on=col, how='left')
        ORIG_FEATS.append(new_mean_col_name)

        # Count Encoding
        new_count_col_name = f"orig_count_{col}"
        count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
        train = train.merge(count_map, on=col, how='left')
        test = test.merge(count_map, on=col, how='left')
        ORIG_FEATS.append(new_count_col_name)


def engineer_features(df):
    df = df.copy()

    # Log Transforms (Handling skewed data)
    df['alcohol_log'] = np.log1p(df['alcohol_consumption_per_week'])
    df['screen_time_log'] = np.log1p(df['screen_time_hours_per_day'])
    df['triglycerides_log'] = np.log1p(df['triglycerides'])
    
    # --- NEW: Binning for better tree splits ---
    # BMI Categories: Underweight, Normal, Overweight, Obese
    df['bmi_cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3]).astype(int)
    # Age Groups (decades)
    df['age_bin'] = pd.cut(df['age'], bins=5, labels=False).astype(int)

    # Polynomials
    df['age_sq'] = df['age'] ** 2
    df['bmi_sq'] = df['bmi'] ** 2
    df['whr_sq'] = df['waist_to_hip_ratio'] ** 2
    
    # Clinical Interactions
    df['age_bmi'] = df['age'] * df['bmi']
    df['htn_sbp'] = df['hypertension_history'] * df['systolic_bp']
    
    # Blood Pressure metrics
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3

    # Cholesterol Ratios
    df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']

    # --- NEW: Comorbidity Score ---
    # Summing binary flags to create a risk score
    binary_cols = ['hypertension_history', 'family_history_diabetes', 'smoking_status'] 

    return df


train = engineer_features(train)
test = engineer_features(test)


FEATURES = BASE + ORIG_FEATS + ['alcohol_log', 'screen_time_log', 'triglycerides_log', 
                                'bmi_cat', 'age_bin', 'age_sq', 'bmi_sq', 'whr_sq', 
                                'age_bmi', 'htn_sbp', 'pulse_pressure', 'mean_arterial_pressure',
                                'chol_hdl_ratio', 'non_hdl_cholesterol']


FEATURES = [f for f in FEATURES if f in train.columns]

X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()


train.head()


for col in CATS:
    if col in X.columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')


cat_features_idx = [X.columns.get_loc(c) for c in CATS if c in X.columns]

print(f"Total Features: {len(FEATURES)}")


FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Storage for Raw Logits
oof_logits_xgb = np.zeros(len(X))
oof_logits_lgb = np.zeros(len(X))
oof_logits_cat = np.zeros(len(X))

test_logits_xgb = np.zeros(len(X_test))
test_logits_lgb = np.zeros(len(X_test))
test_logits_cat = np.zeros(len(X_test))

print("\n=== STAGE 1: Generating OOF Logits ===")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # --- A. CATBOOST (Slower LR, Deeper) ---
    cat_model = CatBoostClassifier(
        iterations=2000, 
        learning_rate=0.015, # Slower for better convergence
        depth=8,             # Deeper for complex interactions
        l2_leaf_reg=3,       # Regularization
        eval_metric='AUC', 
        verbose=0, 
        random_state=42,
        task_type='GPU', 
        devices='0', 
        allow_writing_files=False
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features_idx, early_stopping_rounds=100)
    oof_logits_cat[va_idx] = cat_model.predict(X_va, prediction_type='RawFormulaVal')
    test_logits_cat += cat_model.predict(X_test, prediction_type='RawFormulaVal') / FOLDS

    # --- B. LIGHTGBM (Extra Trees) ---
    lgb_model = LGBMClassifier(
        n_estimators=2000, 
        learning_rate=0.015,
        num_leaves=31,
        max_depth=8,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        extra_trees=True, # Adds randomness/diversity
        device='gpu', 
        random_state=42, 
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(100), log_evaluation(0)])
    oof_logits_lgb[va_idx] = lgb_model.predict(X_va, raw_score=True)
    test_logits_lgb += lgb_model.predict(X_test, raw_score=True) / FOLDS

    # --- C. XGBOOST (Regularized) ---
    xgb_model = XGBClassifier(
        n_estimators=2000, 
        learning_rate=0.015, 
        max_depth=6,
        subsample=0.8,         # Prevent overfitting
        colsample_bytree=0.8,  # Prevent overfitting
        reg_lambda=1.5,        # L2 Regularization
        use_label_encoder=False, 
        eval_metric='auc',
        enable_categorical=True, 
        tree_method='hist', 
        device='cuda',
        random_state=42
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=100)
    oof_logits_xgb[va_idx] = xgb_model.predict(X_va, output_margin=True)
    test_logits_xgb += xgb_model.predict(X_test, output_margin=True) / FOLDS

    print(f"Fold {fold} complete.")

# Create Base Margin (Average of Stage 1)
oof_base_margin = (oof_logits_cat + oof_logits_lgb + oof_logits_xgb) / 3
test_base_margin = (test_logits_cat + test_logits_lgb + test_logits_xgb) / 3


print("\n=== STAGE 2: Regularized Residual Boosting ===")

stage2_oof_preds = np.zeros(len(X))
stage2_test_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    margin_tr = oof_base_margin[tr_idx]
    margin_va = oof_base_margin[va_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, base_margin=margin_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, label=y_va, base_margin=margin_va, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, base_margin=test_base_margin, enable_categorical=True)

    # Stage 2 Params: High Regularization to prevent overfitting to residuals
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.01,   # Very slow learning
        'max_depth': 3,          # Shallow trees (fix bias, don't add variance)
        'reg_alpha': 2.0,        # L1 Reg
        'reg_lambda': 5.0,       # L2 Reg
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42 + fold # Seed variation
    }
    
    bst = xgb.train(
        params, 
        dtrain, 
        num_boost_round=2000, 
        evals=[(dvalid, "validation")], 
        early_stopping_rounds=100, 
        verbose_eval=False
    )

    stage2_oof_preds[va_idx] = bst.predict(dvalid)
    stage2_test_preds += bst.predict(dtest) / FOLDS
    
    print(f"Fold {fold} Stage 2 AUC: {roc_auc_score(y_va, stage2_oof_preds[va_idx]):.5f}")

final_auc = roc_auc_score(y, stage2_oof_preds)
print(f"\nFinal Ensemble AUC (Stage 2): {final_auc:.6f}")


# Create Submission
submission[TARGET] = stage2_test_preds
submission.to_csv('submission_improved_stacking.csv', index=False)
print("Submission saved successfully.")

# Plot ROC
fpr, tpr, _ = roc_curve(y, stage2_oof_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'Stage 2 Ensemble (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stage 2 Residual Boosting ROC')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()


# --- 1. SHAP Analysis (Using Stage 1 XGBoost for clarity) ---
print("Generating SHAP values (using last fold XGBoost model)...")

# Taking a sample for SHAP to save time
X_sample = X.sample(2000, random_state=42)

xgb_model.set_params(device='cpu') 
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 8))
plt.title("SHAP Feature Importance (Top 20 Features)")
shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
plt.tight_layout()
plt.show()

# --- 2. OOF vs Test Prediction Histogram ---
plt.figure(figsize=(10, 6))
sns.kdeplot(stage2_oof_preds, fill=True, label='OOF Predictions (Train)', color='blue', alpha=0.3)
sns.kdeplot(stage2_test_preds, fill=True, label='Test Predictions', color='orange', alpha=0.3)
plt.title("Distribution of Predictions: OOF (Train) vs Test")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


submission.head()

