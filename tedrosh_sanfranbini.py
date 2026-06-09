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


# San Francisco Crime Classification - Comprehensive Analysis
# This script includes model training, evaluation, and extensive visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Evaluation metrics
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_squared_error,
                            confusion_matrix, ConfusionMatrixDisplay, 
                            classification_report, roc_curve, auc)

# Try to import XGBoost, install if missing
try:
    import xgboost as xgb
except ImportError:
    print("Installing XGBoost...")
    import subprocess
    subprocess.check_call(["pip", "install", "xgboost"])
    import xgboost as xgb


# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data():
    """Load San Francisco crime data with fallback options"""
    try:
        # Try Kaggle path first
        train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
        test = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')
        print("âœ“ Data loaded from Kaggle environment")
    except FileNotFoundError:
        try:
            # Try local path
            train = pd.read_csv('train.csv')
            test = pd.read_csv('test.csv')
            print("âœ“ Data loaded from local files")
        except FileNotFoundError:
            print("âš  Dataset not found. Creating sample data for demonstration...")
            # Create sample data for demonstration
            np.random.seed(42)
            n_samples = 10000
            
            categories = ['LARCENY/THEFT', 'OTHER OFFENSES', 'NON-CRIMINAL', 'ASSAULT', 
                         'VEHICLE THEFT', 'VANDALISM', 'WARRANTS', 'BURGLARY', 'ROBBERY']
            districts = ['CENTRAL', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 
                        'TENDERLOIN', 'MISSION', 'INGLESIDE', 'BAYVIEW']
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            train = pd.DataFrame({
                'Category': np.random.choice(categories, n_samples, p=[0.3, 0.15, 0.12, 0.1, 0.08, 0.08, 0.07, 0.05, 0.05]),
                'Dates': pd.date_range('2003-01-01', periods=n_samples, freq='H'),
                'DayOfWeek': np.random.choice(days, n_samples),
                'PdDistrict': np.random.choice(districts, n_samples),
                'X': np.random.normal(-122.4, 0.05, n_samples),
                'Y': np.random.normal(37.77, 0.05, n_samples)
            })
            
            test = pd.DataFrame({
                'Id': range(n_samples//4),
                'Dates': pd.date_range('2015-01-01', periods=n_samples//4, freq='H'),
                'DayOfWeek': np.random.choice(days, n_samples//4),
                'PdDistrict': np.random.choice(districts, n_samples//4),
                'X': np.random.normal(-122.4, 0.05, n_samples//4),
                'Y': np.random.normal(37.77, 0.05, n_samples//4)
            })
    
    return train, test

def preprocess(df):
    """
    Comprehensive preprocessing with NaN handling
    - Removes invalid coordinates (0,0) which are data errors
    - Extracts temporal features from datetime
    - Handles missing values with median imputation
    """
    print(f"Initial data shape: {df.shape}")
    
    # Remove invalid coordinates (common issue in SF crime data)
    df = df[(df['X'] != 0) & (df['Y'] != 0)]
    print(f"After removing invalid coordinates: {df.shape}")
    
    # Extract datetime features
    df['Dates'] = pd.to_datetime(df['Dates'])
    df['Year'] = df['Dates'].dt.year
    df['Month'] = df['Dates'].dt.month
    df['Day'] = df['Dates'].dt.day
    df['Hour'] = df['Dates'].dt.hour
    
    return df

def encode_features(df, numerical_cols):
    """
    One-hot encode categorical features and combine with numerical
    """
    day_dummies = pd.get_dummies(df['DayOfWeek'], prefix='Day')
    district_dummies = pd.get_dummies(df['PdDistrict'], prefix='District')
    numerical = df[numerical_cols]
    return pd.concat([day_dummies, district_dummies, numerical], axis=1)





# =============================================================================
# 2. LOAD AND PREPROCESS DATA
# =============================================================================

print("ğŸ”„ Loading and preprocessing data...")
train, test = load_data()
train = preprocess(train)

# Handle missing values in numerical columns
numerical_cols = ['X', 'Y', 'Year', 'Month', 'Day', 'Hour']
imputer = SimpleImputer(strategy='median')
train[numerical_cols] = imputer.fit_transform(train[numerical_cols])

# Encode target variable
le = LabelEncoder()
y_train = le.fit_transform(train['Category'])
print(f"Number of crime categories: {len(le.classes_)}")

# Feature engineering
X_train = encode_features(train, numerical_cols)
print(f"Feature matrix shape: {X_train.shape}")

# Train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)


# =============================================================================
# 3. MODEL TRAINING
# =============================================================================

print("\nğŸ¤– Training models...")

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train_split, y_train_split)

# XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_split, y_train_split)

# Calibrated SVM (with proper scaling)
print("Training Calibrated SVM...")
scaler = StandardScaler()
X_train_svm = X_train_split.copy()
X_train_svm[numerical_cols] = scaler.fit_transform(X_train_split[numerical_cols])

svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, 
                   tol=1e-3, random_state=42, n_jobs=-1)
calibrated_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=3)
calibrated_svm.fit(X_train_svm, y_train_split)

# Prepare validation data for SVM
X_val_svm = X_val_split.copy()
X_val_svm[numerical_cols] = scaler.transform(X_val_split[numerical_cols])

print("âœ“ All models trained successfully!")


# =============================================================================
# 4. PREDICTIONS AND METRICS CALCULATION
# =============================================================================

print("\nğŸ“Š Generating predictions and calculating metrics...")

# Generate predictions
rf_pred = rf.predict(X_val_split)
xgb_pred = xgb_model.predict(X_val_split)
svm_pred = calibrated_svm.predict(X_val_svm)

# Generate predicted probabilities
rf_proba = rf.predict_proba(X_val_split)
xgb_proba = xgb_model.predict_proba(X_val_split)
svm_proba = calibrated_svm.predict_proba(X_val_svm)

# Calculate all metrics
models = ['Random Forest', 'XGBoost', 'Calibrated SVM']
predictions = [rf_pred, xgb_pred, svm_pred]
probabilities = [rf_proba, xgb_proba, svm_proba]

metrics_results = {}
for model, pred, proba in zip(models, predictions, probabilities):
    metrics_results[model] = {
        'Accuracy': accuracy_score(y_val_split, pred),
        'F1-Score': f1_score(y_val_split, pred, average='weighted'),
        'Log-Loss': log_loss(y_val_split, proba),
        'RMSE': np.sqrt(mean_squared_error(y_val_split, pred))
    }


# =============================================================================
# 5. COMPREHENSIVE VISUALIZATIONS
# =============================================================================

print("\nğŸ“ˆ Creating visualizations...")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# 5.1 METRICS COMPARISON BAR PLOT
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'F1-Score', 'Log-Loss', 'RMSE']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    values = [metrics_results[model][metric] for model in models]
    
    bars = ax.bar(models, values, color=colors)
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylabel(metric)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # For log-loss and RMSE, lower is better - highlight the best
    if metric in ['Log-Loss', 'RMSE']:
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmax(values)
    bars[best_idx].set_color('#27ae60')

plt.tight_layout()
plt.show()


# 5.2 CONFUSION MATRICES COMPARISON
fig, axes = plt.subplots(1, 3, figsize=(60,18))
fig.suptitle('Confusion Matrix Comparison (Normalized)', fontsize=12, fontweight='bold')

for i, (model, pred) in enumerate(zip(models, predictions)):
    cm = confusion_matrix(y_val_split, pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=axes[i], xticks_rotation=90, colorbar=False, cmap='Blues')
    axes[i].set_title(f'{model}\nAccuracy: {metrics_results[model]["Accuracy"]:.3f}', 
                     fontweight='bold')

plt.tight_layout()
plt.show()


# 5.3 ROC CURVES FOR MULTICLASS CLASSIFICATION
print("Generating ROC curves...")

# Binarize labels for ROC calculation
label_binarizer = LabelBinarizer()
y_val_bin = label_binarizer.fit_transform(y_val_split)
n_classes = y_val_bin.shape[1]

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')

colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange'])

for idx, (model, proba) in enumerate(zip(models, probabilities)):
    ax = axes[idx]
    
    # Calculate ROC for each class
    for i, color in zip(range(min(n_classes, 8)), colors):  # Limit to 8 classes for visibility
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{le.classes_[i][:15]} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model}')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# 5.4 FEATURE IMPORTANCE (for tree-based models)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[0].barh(rf_importance['feature'], rf_importance['importance'])
axes[0].set_title('Random Forest - Top 15 Features')
axes[0].set_xlabel('Importance')

# XGBoost feature importance
xgb_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[1].barh(xgb_importance['feature'], xgb_importance['importance'])
axes[1].set_title('XGBoost - Top 15 Features')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()


# =============================================================================
# 6. DETAILED PERFORMANCE REPORT
# =============================================================================

print("\n" + "="*80)
print("ğŸ“‹ COMPREHENSIVE PERFORMANCE REPORT")
print("="*80)

# Print metrics table
print("\nğŸ�¯ EVALUATION METRICS SUMMARY:")
print("-" * 70)
print(f"{'Model':<15} {'Accuracy':<10} {'F1-Score':<10} {'Log-Loss':<10} {'RMSE':<10}")
print("-" * 70)

for model in models:
    metrics = metrics_results[model]
    print(f"{model:<15} {metrics['Accuracy']:<10.4f} {metrics['F1-Score']:<10.4f} "
          f"{metrics['Log-Loss']:<10.4f} {metrics['RMSE']:<10.4f}")

print("-" * 70)

# Best model identification
best_accuracy = max(models, key=lambda x: metrics_results[x]['Accuracy'])
best_f1 = max(models, key=lambda x: metrics_results[x]['F1-Score'])
best_logloss = min(models, key=lambda x: metrics_results[x]['Log-Loss'])
best_rmse = min(models, key=lambda x: metrics_results[x]['RMSE'])

print(f"\nğŸ�† BEST PERFORMERS:")
print(f"   Best Accuracy:  {best_accuracy} ({metrics_results[best_accuracy]['Accuracy']:.4f})")
print(f"   Best F1-Score:  {best_f1} ({metrics_results[best_f1]['F1-Score']:.4f})")
print(f"   Best Log-Loss:  {best_logloss} ({metrics_results[best_logloss]['Log-Loss']:.4f})")
print(f"   Best RMSE:      {best_rmse} ({metrics_results[best_rmse]['RMSE']:.4f})")

# Detailed classification reports
print(f"\nğŸ“Š DETAILED CLASSIFICATION REPORTS:")
print("-" * 80)

for model, pred in zip(models, predictions):
    print(f"\n{model.upper()} CLASSIFICATION REPORT:")
    print("-" * 50)
    report = classification_report(y_val_split, pred, target_names=le.classes_, 
                                 output_dict=False, zero_division=0)
    print(report)


# =============================================================================
# 7. RECOMMENDATIONS AND INSIGHTS
# =============================================================================

print("\n" + "="*80)
print("ğŸ’¡ INSIGHTS AND RECOMMENDATIONS")
print("="*80)

print("\nğŸ”� KEY FINDINGS:")
print(f"1. Overall Performance: The models achieve {metrics_results[best_accuracy]['Accuracy']:.1%} accuracy")
print(f"2. Log-loss scores range from {min(m['Log-Loss'] for m in metrics_results.values()):.3f} to {max(m['Log-Loss'] for m in metrics_results.values()):.3f}")
print(f"3. {best_logloss} performs best overall with the lowest log-loss")

print(f"\nğŸš€ IMPROVEMENT STRATEGIES:")
print("1. Feature Engineering: Add temporal patterns, weather data, events")
print("2. Ensemble Methods: Combine predictions from multiple models")
print("3. Hyperparameter Tuning: Use grid search or Bayesian optimization")
print("4. Class Balance: Address imbalanced classes with SMOTE or class weights")
print("5. External Data: Incorporate demographic, economic, or geographical data")

print(f"\nğŸ“ˆ COMPETITION CONTEXT:")
if 'Log-Loss' in metrics_results[best_logloss]:
    logloss_score = metrics_results[best_logloss]['Log-Loss']
    if logloss_score < 2.3:
        print(f"   Excellent! Score of {logloss_score:.3f} is competitive (top 10%)")
    elif logloss_score < 2.4:
        print(f"   Good! Score of {logloss_score:.3f} is above average")
    else:
        print(f"   Score of {logloss_score:.3f} needs improvement for competition")

print("\n" + "="*80)
print("âœ… ANALYSIS COMPLETE")
print("="*80)


# --- Prepare test data in the same way as training data ---
# Preprocess test set
test = preprocess(test)
test[numerical_cols] = imputer.transform(test[numerical_cols])
X_test = encode_features(test, numerical_cols)

# Ensure columns match between train and test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# For SVM, scale the numerical columns
X_test_svm = X_test.copy()
X_test_svm[numerical_cols] = scaler.transform(X_test_svm[numerical_cols])

# --- Choose your model here ---
# model = rf
# X_test_input = X_test

# model = xgb_model
# X_test_input = X_test

model = calibrated_svm
X_test_input = X_test_svm

# --- Generate predicted probabilities ---
probs = model.predict_proba(X_test_input)

# --- Create submission DataFrame ---
submission = pd.DataFrame(probs, columns=le.classes_)
submission.insert(0, 'Id', test['Id'])

# --- Save to CSV for Kaggle submission ---
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")





