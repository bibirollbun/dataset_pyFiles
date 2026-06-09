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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


print("DATA OVERVIEW")
print("="*60)
print(train.head())
print("\nData Info:")
print(train.info())
print("\nTarget Distribution:")
print(train['diagnosed_diabetes'].value_counts(normalize=True))

# Check for missing values
print("\nMissing Values:")
print(train.isnull().sum())



print("\n" + "="*60)
print("ENCODING CATEGORICAL VARIABLES")
print("="*60)

# Identify categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
if 'id' in categorical_cols:
    categorical_cols.remove('id')
if 'diagnosed_diabetes' in categorical_cols:
    categorical_cols.remove('diagnosed_diabetes')

print(f"Categorical columns: {categorical_cols}")

# Label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded {col}: {len(le.classes_)} unique values")


def create_features(df):
    """Create additional features"""
    df = df.copy()
    
    # BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], 
                                 bins=[0, 18.5, 25, 30, 100],
                                 labels=[0, 1, 2, 3])
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 30, 45, 60, 100],
                              labels=[0, 1, 2, 3])
    
    # Blood pressure categories
    df['bp_category'] = pd.cut(df['systolic_bp'],
                                bins=[0, 120, 130, 140, 200],
                                labels=[0, 1, 2, 3])
    
    # Health score (combination of positive factors)
    df['health_score'] = (
        df['physical_activity_minutes_per_week'] / 150 +  # WHO recommendation
        df['diet_score'] / 10 +
        df['sleep_hours_per_day'] / 8 -
        df['screen_time_hours_per_day'] / 8 -
        df['alcohol_consumption_per_week'] / 7
    )
    
    # Risk factors
    df['high_bmi'] = (df['bmi'] > 30).astype(int)
    df['high_bp'] = (df['systolic_bp'] > 130).astype(int)
    df['high_waist_hip'] = (df['waist_to_hip_ratio'] > 0.90).astype(int)
    df['low_activity'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
    df['poor_sleep'] = ((df['sleep_hours_per_day'] < 6) | (df['sleep_hours_per_day'] > 9)).astype(int)
    
    # Interaction features
    df['bmi_age_interaction'] = df['bmi'] * df['age']
    df['bp_age_interaction'] = df['systolic_bp'] * df['age']
    df['activity_age_ratio'] = df['physical_activity_minutes_per_week'] / (df['age'] + 1)
    
    # Convert categorical to numeric
    df['bmi_category'] = df['bmi_category'].astype(float)
    df['age_group'] = df['age_group'].astype(float)
    df['bp_category'] = df['bp_category'].astype(float)
    
    return df

print("\nCreating features...")
train = create_features(train)
test = create_features(test)



# Separate features and target
X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']
X_test = test.drop(['id'], axis=1)

print(f"\nFeatures shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")



print("TRAINING MODELS")
print("="*60)

# Define models
models = {
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss'
    ),
    'CatBoost': CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
}

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

predictions = {}
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    fold_predictions = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Validate
        val_pred = model.predict_proba(X_val)[:, 1]
        fold_score = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_score)
        
        print(f"  Fold {fold+1} ROC-AUC: {fold_score:.5f}")
    
    print(f"{name} Average ROC-AUC: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    
    # Train on full data
    model.fit(X, y)
    
    # Predictions
    predictions[name] = model.predict_proba(X_test)[:, 1]




print("CREATING ENSEMBLE")
print("="*60)

# Simple averaging ensemble
ensemble_pred = np.mean([predictions['LightGBM'], 
                         predictions['XGBoost'], 
                         predictions['CatBoost']], axis=0)

# Weighted ensemble (you can adjust weights based on CV scores)
weights = [0.35, 0.35, 0.30]  # LightGBM, XGBoost, CatBoost
weighted_ensemble = (weights[0] * predictions['LightGBM'] + 
                     weights[1] * predictions['XGBoost'] + 
                     weights[2] * predictions['CatBoost'])



print("\nTop 15 Important Features (LightGBM):")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': models['LightGBM'].feature_importances_
}).sort_values('importance', ascending=False).head(15)
print(feature_importance)




submission = sample_sub.copy()
submission['diagnosed_diabetes'] = weighted_ensemble

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n" + "="*60)
print("SUBMISSION SAVED!")
print("="*60)
print(submission.head(10))
print(f"\nPrediction statistics:")
print(submission['diagnosed_diabetes'].describe())




fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Feature Importance
feature_importance_plot = feature_importance.head(10).sort_values('importance')
axes[0, 0].barh(feature_importance_plot['feature'], feature_importance_plot['importance'])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Top 10 Feature Importance')

# Plot 2: Prediction Distribution
axes[0, 1].hist(submission['diagnosed_diabetes'], bins=50, edgecolor='black')
axes[0, 1].set_xlabel('Predicted Probability')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Predictions')

# Plot 3: Model Comparison
model_preds = pd.DataFrame(predictions)
axes[1, 0].boxplot([model_preds['LightGBM'], model_preds['XGBoost'], model_preds['CatBoost']])
axes[1, 0].set_xticklabels(['LightGBM', 'XGBoost', 'CatBoost'])
axes[1, 0].set_ylabel('Predicted Probability')
axes[1, 0].set_title('Model Predictions Comparison')

# Plot 4: Correlation of models
corr_matrix = model_preds.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Model Predictions Correlation')

plt.tight_layout()
plt.savefig('analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nNotebook execution complete!")
print("Analysis saved as 'analysis.png'")
print("Submission saved as 'submission.csv'")

