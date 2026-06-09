# Data manipulation and analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Models
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)

# Warnings
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("✓ ALL LIBRARIES IMPORTED SUCCESSFULLY")
print("="*80)


# assigning a variable to the path of the datasets
train_dir = "/kaggle/input/playground-series-s5e12/train.csv"
test_dir = "/kaggle/input/playground-series-s5e12/test.csv"
sample_submission = "/kaggle/input/playground-series-s5e12/sample_submission.csv"


# loading the datasets
train_df = pd.read_csv(train_dir)
test_df = pd.read_csv(test_dir)
submission_df = pd.read_csv(sample_submission)



print("\n" + "-"*80)
print("TRAIN DATA - FIRST 5 ROWS")
print("-"*80)
print(train_df.head())

print("\n" + "-"*80)
print("TEST DATA - FIRST 5 ROWS")
print("-"*80)
print(test_df.head())

print("\n" + "-"*80)
print("SUBMISSION FORMAT")
print("-"*80)
print(submission_df.head())


print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Define columns
id_col = 'id'
target_col = 'diagnosed_diabetes'

# Identify categorical and numerical columns
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                    'smoking_status', 'employment_status']
numerical_cols = ['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
                  'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
                  'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp',
                  'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
                  'triglycerides', 'family_history_diabetes', 'hypertension_history',
                  'cardiovascular_history']

print(f"\nID Column: {id_col}")
print(f"Target Column: {target_col}")
print(f"\nNumber of Features: {len(numerical_cols) + len(categorical_cols)}")
print(f"  - Numerical: {len(numerical_cols)}")
print(f"  - Categorical: {len(categorical_cols)}")



# Check missing values in train
print("\nMissing Values in Train:")
missing_train = train_df.isnull().sum()
if missing_train.sum() == 0:
    print("  No missing values!")
else:
    print(missing_train[missing_train > 0])


# Check missing values in test
print("\nMissing Values in Test:")
missing_test = test_df.isnull().sum()
if missing_test.sum() == 0:
    print("  No missing values!")
else:
    print(missing_test[missing_test > 0])


# Check target distribution
print(f"\nTarget Distribution ({target_col}):")
print(train_df[target_col].value_counts())
print("\nTarget Percentage:")
target_dist = train_df[target_col].value_counts(normalize=True) * 100
print(f"  No Diabetes (0): {target_dist[0.0]:.2f}%")
print(f"  Diabetes (1): {target_dist[1.0]:.2f}%")


# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_df[target_col].value_counts().plot(kind='bar', color=['skyblue', 'coral'], ax=axes[0])
axes[0].set_title('Target Distribution', fontsize=14)
axes[0].set_xlabel('Class (0: No Diabetes, 1: Diabetes)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

train_df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                          colors=['skyblue', 'coral'], ax=axes[1])
axes[1].set_title('Target Proportion', fontsize=14)
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Categorical features distribution
print("\nCategorical Features Distribution:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(train_df[col].value_counts())


# Visualize categorical features
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    train_df[col].value_counts().plot(kind='bar', ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'{col.replace("_", " ").title()}', fontsize=11)
    axes[idx].set_xlabel('')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# Correlation heatmap (numerical features only)
plt.figure(figsize=(16, 14))
correlation_matrix = train_df[numerical_cols + [target_col]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()


# Top correlations with target
target_corr = correlation_matrix[target_col].sort_values(ascending=False)
print("\nTop 10 Features Correlated with Target:")
print(target_corr[1:11])  # Exclude target itself



# Feature distributions
train_df[numerical_cols].hist(figsize=(18, 14), bins=30, edgecolor='black')
plt.suptitle('Numerical Feature Distributions', fontsize=16)
plt.tight_layout()
plt.show()


print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Prepare feature columns
feature_cols = numerical_cols + categorical_cols

# Separate features and target from training data
X_train_full = train_df[feature_cols].copy()
y_train_full = train_df[target_col].copy()

# Prepare test data (no target column)
X_test_kaggle = test_df[feature_cols].copy()
test_ids = test_df[id_col].copy()

print(f"\nFull Training Features Shape: {X_train_full.shape}")
print(f"Full Training Target Shape: {y_train_full.shape}")
print(f"Kaggle Test Features Shape: {X_test_kaggle.shape}")


# Encode categorical variables
print("\nEncoding categorical variables...")
from sklearn.preprocessing import LabelEncoder

# Create a copy for encoding
X_train_full_encoded = X_train_full.copy()
X_test_kaggle_encoded = X_test_kaggle.copy()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined data to ensure all categories are captured
    combined_data = pd.concat([X_train_full[col], X_test_kaggle[col]], axis=0)
    le.fit(combined_data)
    
    X_train_full_encoded[col] = le.transform(X_train_full[col])
    X_test_kaggle_encoded[col] = le.transform(X_test_kaggle[col])
    label_encoders[col] = le
    
    print(f"  {col}: {len(le.classes_)} unique values")

print("Categorical encoding completed.")


# Handle missing values (if any)
if X_train_full_encoded.isnull().sum().sum() > 0:
    print("\nImputing missing values in training data...")
    imputer = SimpleImputer(strategy='median')
    X_train_full_encoded = pd.DataFrame(
        imputer.fit_transform(X_train_full_encoded), 
        columns=X_train_full_encoded.columns
    )
else:
    imputer = None
    print("\nNo missing values in training data.")

if X_test_kaggle_encoded.isnull().sum().sum() > 0:
    print("Imputing missing values in test data...")
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train_full_encoded)
    X_test_kaggle_encoded = pd.DataFrame(
        imputer.transform(X_test_kaggle_encoded), 
        columns=X_test_kaggle_encoded.columns
    )
else:
    print("No missing values in test data.")



# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full_encoded, y_train_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train_full
)

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Kaggle Test set: {X_test_kaggle_encoded.shape}")



# Feature scaling for the models
print("\nScaling features for better performance of the model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_train_full_scaled = scaler.fit_transform(X_train_full_encoded)
X_test_kaggle_scaled = scaler.transform(X_test_kaggle_encoded)

print("Data preprocessing completed successfully!")



print("\n" + "="*80)
print("MODEL BUILDING & TRAINING")
print("="*80)

# Dictionary to store models and results
models = {}
results = {}

# XGBoost
print("\n[1/3] Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model
print("XGBoost training completed.")

# LightGBM 
print("\n[2/3] Training LightGBM...")
lgb_model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
models['LightGBM'] = lgb_model
print("LightGBM training completed.")

# CatBoost
print("\n[3/3] Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)
cat_model.fit(X_train, y_train)
models['CatBoost'] = cat_model
print("CatBoost training completed.")



print("\n" + "="*80)
print("MODEL EVALUATION ON VALIDATION SET")
print("="*80)

def evaluate_model(name, model, X_val_data, y_val):
    """Evaluate a single model on validation set"""
    if name == ' CatBoost':
        y_pred_proba = model.predict(X_val_data, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_val_data)
        y_pred_proba = model.predict_proba(X_val_data)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    print(f"\n{name} Validation Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return y_pred, y_pred_proba

# Evaluate each model
xgb_pred, xgb_proba = evaluate_model('XGBoost', models['XGBoost'], X_val, y_val)
lgb_pred, lgb_proba = evaluate_model('LightGBM', models['LightGBM'], X_val, y_val)
cat_pred, cat_proba = evaluate_model('CatBoost', models['CatBoost'], X_val, y_val)



print("\n" + "="*80)
print("ENSEMBLE MODEL (VOTING)")
print("="*80)

# Create ensemble using voting on validation set
ensemble_proba = (xgb_proba + lgb_proba + cat_proba ) / 3
ensemble_pred = (ensemble_proba > 0.5).astype(int)

# Evaluate ensemble
ensemble_acc = accuracy_score(y_val, ensemble_pred)
ensemble_prec = precision_score(y_val, ensemble_pred)
ensemble_rec = recall_score(y_val, ensemble_pred)
ensemble_f1 = f1_score(y_val, ensemble_pred)
ensemble_roc_auc = roc_auc_score(y_val, ensemble_proba)

results['Ensemble'] = {
    'Accuracy': ensemble_acc,
    'Precision': ensemble_prec,
    'Recall': ensemble_rec,
    'F1-Score': ensemble_f1,
    'ROC-AUC': ensemble_roc_auc
}

print(f"\nEnsemble Validation Results:")
print(f"  Accuracy:  {ensemble_acc:.4f}")
print(f"  Precision: {ensemble_prec:.4f}")
print(f"  Recall:    {ensemble_rec:.4f}")
print(f"  F1-Score:  {ensemble_f1:.4f}")
print(f"  ROC-AUC:   {ensemble_roc_auc:.4f}")



print("\n" + "="*80)
print("FINAL RESULTS COMPARISON")
print("="*80)

results_df = pd.DataFrame(results).T
print("\n", results_df)

# Plot comparison
results_df.plot(kind='bar', figsize=(14, 6))
plt.title('Model Performance Comparison', fontsize=16)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


best_model_name = results_df['ROC-AUC'].idxmax()
print(f"\n Best Model: {best_model_name} (ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f})")

if best_model_name == 'Ensemble':
    best_pred = ensemble_pred
elif best_model_name == 'XGBoost':
    best_pred = xgb_pred
elif best_model_name == 'LightGBM':
    best_pred = lgb_pred
elif best_model_name == 'CatBoost':
    best_pred = cat_pred
else:
    best_pred = ann_pred

cm = confusion_matrix(y_val, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title(f'Confusion Matrix - {best_model_name} (Validation Set)', fontsize=14)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.show()

print("\nClassification Report (Validation Set):")
print(classification_report(y_val, best_pred))


print("\n" + "="*80)
print("RETRAINING ON FULL DATA & GENERATING KAGGLE PREDICTIONS")
print("="*80)

# Retrain all models on full training data for better performance
print("\nRetraining models on full training data...")

# XGBoost
print("  [1/3] Retraining XGBoost...")
xgb_final = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_final.fit(X_train_full_encoded, y_train_full)

# LightGBM
print("  [2/3] Retraining LightGBM...")
lgb_final = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
lgb_final.fit(X_train_full_encoded, y_train_full)

# CatBoost
print("  [3/3] Retraining CatBoost...")
cat_final = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)
cat_final.fit(X_train_full_encoded, y_train_full)


print("\n All models retrained successfully!")



# Generate predictions on Kaggle test set
print("\nGenerating predictions on test set...")

# Get predictions from each model
xgb_test_pred = xgb_final.predict_proba(X_test_kaggle_encoded)[:, 1]
lgb_test_pred = lgb_final.predict_proba(X_test_kaggle_encoded)[:, 1]
cat_test_pred = cat_final.predict_proba(X_test_kaggle_encoded)[:, 1]
# Ensemble prediction (average of all models)
ensemble_test_pred = (xgb_test_pred + lgb_test_pred + cat_test_pred ) / 3

# Convert probabilities to class labels (0 or 1)
ensemble_test_labels = (ensemble_test_pred > 0.5).astype(int)

print(f"\n Predictions completed!")
print(f"\nPrediction distribution on test set:")
unique, counts = np.unique(ensemble_test_labels, return_counts=True)
for label, count in zip(unique, counts):
    percentage = (count / len(ensemble_test_labels)) * 100
    diabetes_status = "No Diabetes" if label == 0 else "Diabetes"
    print(f"  Class {label} ({diabetes_status}): {count:,} ({percentage:.2f}%)")



print("\n" + "="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

# Create submission dataframe
submission = submission_df.copy()
target_col_name = submission.columns[-1]  # Usually 'diagnosed_diabetes' or 'prediction'

# Update with predictions
submission[target_col_name] = ensemble_test_labels

print(f"\nSubmission file preview:")
print(submission.head(10))

# Save submission file
submission.to_csv('submission.csv', index=False)
print("\n Submission file saved as 'submission.csv'")

# Also save individual model predictions (optional, for ensembling later)
detailed_submission = pd.DataFrame({
    id_col: test_ids,
    'XGBoost_pred': (xgb_test_pred > 0.5).astype(int),
    'LightGBM_pred': (lgb_test_pred > 0.5).astype(int),
    'CatBoost_pred': (cat_test_pred > 0.5).astype(int),
    'Ensemble_pred': ensemble_test_labels,
    'Ensemble_proba': ensemble_test_pred
})

detailed_submission.to_csv('detailed_predictions.csv', index=False)
print("Detailed predictions saved as 'detailed_predictions.csv'")





