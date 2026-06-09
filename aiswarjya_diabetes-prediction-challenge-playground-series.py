# Install required libraries if not present
!pip install category_encoders catboost shap -q



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
import warnings

# Configuration
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
SEED = 42

print("Setup Complete")



# Load competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')

# Load and integrate original dataset
try:
    original = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
    
    # Align columns
    original_aligned = original.reindex(columns=train.columns)
    
    # Combine datasets
    train_augmented = pd.concat([train, original_aligned], axis=0, ignore_index=True)
    
    # Remove rows with missing target in the augmented set
    train_augmented = train_augmented.dropna(subset=['diagnosed_diabetes'])
    
    print(f"Original Data Integrated. New Training Size: {train_augmented.shape}")
    train = train_augmented
    
except Exception as e:
    print(f"Note: Original dataset not found or could not be loaded. using competition data only.")
    print(f"Error: {e}")

print(f"Final Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")



# Target Balance
target_counts = train['diagnosed_diabetes'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(target_counts, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', 
        colors=['#2ecc71', '#e74c3c'], explode=(0, 0.1), shadow=True)
plt.title('Target Distribution : Diagnosed Diabetes', fontsize=14, fontweight='bold')
plt.show()



numerical_features = ['age', 'bmi', 'cholesterol_total', 'glucose', 'systolic_bp', 'diastolic_bp']
# Note: 'glucose' might not be in this specific dataset, limiting to available columns for safety
available_nums = [c for c in numerical_features if c in train.columns]

if available_nums:
    fig, axes = plt.subplots(len(available_nums)//3 + 1, 3, figsize=(18, 5*len(available_nums)//3))
    axes = axes.ravel()
    
    for idx, col in enumerate(available_nums):
        sns.histplot(data=train, x=col, hue='diagnosed_diabetes', bins=40, kde=True, 
                     palette=['#2ecc71', '#e74c3c'], ax=axes[idx], alpha=0.6)
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
    
    # Hide empty subplots
    for i in range(idx+1, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()



def engineer_features(df_in, is_train=True):
    df = df_in.copy()
    
    # --- Medical Ratios ---
    # Blood Pressure Indices
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    
    # Cholesterol Ratios
    if 'cholesterol_total' in df.columns and 'hdl_cholesterol' in df.columns:
        df['cholesterol_ratio'] = (df['cholesterol_total'] - df['hdl_cholesterol']) / (df['hdl_cholesterol'] + 1)
        df['non_hdl'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # --- Interaction Terms ---
    # Log transformations for skewed distributions
    for col in ['triglycerides', 'alcohol_consumption_per_week']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            
    # Metabolic & Risk Scores
    # Simple boolean sum of risk factors
    risk_factors = []
    if 'bmi' in df.columns: risk_factors.append((df['bmi'] > 30).astype(int))
    if 'systolic_bp' in df.columns: risk_factors.append((df['systolic_bp'] > 140).astype(int))
    if 'smoking_status' in df.columns: risk_factors.append((df['smoking_status'] == 'Current').astype(int))
    
    if risk_factors:
        df['risk_score_simple'] = sum(risk_factors)
        
    return df

# Apply engineering
print("Engineering features...")
# Concatenate for consistent processing
train['is_train'] = 1
test['is_train'] = 0
test['diagnosed_diabetes'] = -1 # Placeholder

combined = pd.concat([train, test], ignore_index=True)
combined = engineer_features(combined)

# Target Encoding for categorical variables
categorical_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    print(f"Target encoding: {categorical_cols}")
    # Split back for encoding to avoid leakage
    train_idx = combined['is_train'] == 1
    test_idx = combined['is_train'] == 0
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=10)
    
    # Split for encoding
    df_tr = combined[combined['is_train'] == 1].copy()
    df_te = combined[combined['is_train'] == 0].copy()
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=10)
    
    # Fit on TRAIN
    df_tr[categorical_cols] = encoder.fit_transform(df_tr[categorical_cols], df_tr['diagnosed_diabetes'])
    
    # Transform TEST
    df_te[categorical_cols] = encoder.transform(df_te[categorical_cols])
    
    # Recombine
    combined = pd.concat([df_tr, df_te], axis=0)

# Split back to train/test
df_train = combined[combined['is_train'] == 1].drop(['is_train', 'dataset_source'], axis=1, errors='ignore')
df_test = combined[combined['is_train'] == 0].drop(['is_train', 'diagnosed_diabetes', 'dataset_source'], axis=1, errors='ignore')

print(f"Processed Train Shape: {df_train.shape}")
print(f"Processed Test Shape: {df_test.shape}")



# Prepare Arrays
X = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = df_train['diagnosed_diabetes']
X_test = df_test.drop(['id'], axis=1)

# Define Models with Optimized Hyperparameters
models = {
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=SEED, n_jobs=-1
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.03,
        random_seed=SEED, verbose=0, allow_writing_files=False,
        eval_metric='AUC'
    )
}



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}
cv_scores = {name: [] for name in models}

print("Starting Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    for name, model in models.items():
        # Train
        model.fit(X_tr, y_tr)
        
        # Predict
        val_p = model.predict_proba(X_val)[:, 1]
        test_p = model.predict_proba(X_test)[:, 1]
        
        # Store
        oof_preds[name][val_idx] = val_p
        test_preds[name] += test_p / 5 # Average over folds
        
        score = roc_auc_score(y_val, val_p)
        cv_scores[name].append(score)
    
    print(f"Fold {fold} completed.")

# Results
print("\n--- CV ROC-AUC Scores ---")
for name, scores in cv_scores.items():
    print(f"{name}: {np.mean(scores):.5f} Â± {np.std(scores):.4f}")



# Simple Weighted Average based on inverse variable
# (Here we use simple equal weights or tuned manual weights for robustness)

# Example: High performing CatBoost gets slightly more weight
final_oof = 0.4 * oof_preds['CatBoost'] + 0.3 * oof_preds['XGBoost'] + 0.3 * oof_preds['LightGBM']
final_test_pred = 0.4 * test_preds['CatBoost'] + 0.3 * test_preds['XGBoost'] + 0.3 * test_preds['LightGBM']

final_score = roc_auc_score(y, final_oof)
print(f"ğŸ�† Ensemble OOF ROC-AUC: {final_score:.5f}")



# SHAP for XGBoost (fastest to compute for tree explainer)
explainer = shap.TreeExplainer(models['XGBoost'])
shap_values = explainer.shap_values(X.iloc[:1000]) # Sample for performance

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X.iloc[:1000], show=False)
plt.title('Feature Importance (SHAP) - XGBoost', fontsize=12)
plt.tight_layout()
plt.show()



submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': final_test_pred
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved successfully!")
print(submission.head())


