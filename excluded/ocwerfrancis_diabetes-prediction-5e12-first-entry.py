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


import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


train_df.head()


train_df.info()


print("\nMissing Values:")
print(train_df.isnull().sum())


train_df.dtypes


train_df.hist(bins=60,figsize=(30,20))


train_df['diagnosed_diabetes'].value_counts()



print("Target distribution:")
print(train_df['diagnosed_diabetes'].value_counts(normalize=True))

# Check categorical values
print("\nCategorical feature values:")
for col in [var for var in train_df.columns if train_df[var].dtype == "O"]:
    print(f"\n{col}:")
    print(train_df[col].value_counts())


categorical_cols = [col for col in train_df.columns if train_df[col].dtype == "O"]

for col in categorical_cols:
    plt.figure(figsize=(12, 6))

    # Correct color mapping
    palette = {0: '#51cf66', 1: '#ff6b6b'}  # Green = No diabetes, Red = Diabetes

    ax = sns.countplot(
        x=col,
        hue='diagnosed_diabetes',
        data=train_df,
        palette=palette,
        order=train_df[col].value_counts().index
    )

    plt.title(f'Diabetes Distribution by {col.replace("_", " ").title()}',
              fontsize=14, fontweight='bold')
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    total = len(train_df)

    # Add percentage labels
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + total * 0.005
        ax.annotate(percentage, (x, y), ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    plt.legend(title='Diabetes Diagnosis', labels=['No Diabetes', 'Diabetes'])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()



categorical_cols = [col for col in train_df.columns if train_df[col].dtype == 'O']
numerical_cols = [col for col in train_df.columns.drop('id') if train_df[col].dtype != 'O']
print("FEATURE TYPE SUMMARY")
print("="*80)
print(f"\n Numerical features ({len(numerical_cols)}):")
for i, col in enumerate(numerical_cols, 1):
    print(f"   {i}. {col}")

print(f"\n Categorical features ({len(categorical_cols)}):")
for i, col in enumerate(categorical_cols, 1):
    print(f"   {i}. {col}")

print(f"\n Total predictive features: {len(numerical_cols) + len(categorical_cols)}")


# Statistical summary of numerical features
print("NUMERICAL FEATURES - STATISTICAL SUMMARY")
print("="*80)

numerical_stats = train_df[numerical_cols].describe().T
numerical_stats['missing'] = train_df[numerical_cols].isnull().sum().values
numerical_stats['skewness'] = train_df[numerical_cols].skew().values
numerical_stats['kurtosis'] = train_df[numerical_cols].kurtosis().values

display(numerical_stats.style.background_gradient(cmap='coolwarm', subset=['mean', 'std', 'skewness', 'kurtosis']))


import math

num_plots = len(numerical_cols)
rows = math.ceil(num_plots / 2)

fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 4))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(train_df[col], bins=50, alpha=0.6, color='steelblue',
                   edgecolor='black', density=True, label='Histogram')
    
    train_df[col].plot(kind='kde', ax=axes[idx], color='red', linewidth=2, label='KDE')
    
    axes[idx].set_title(f'{col} Distribution', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].grid(alpha=0.3, linestyle='--')
    
    # Stats
    mean_val = train_df[col].mean()
    median_val = train_df[col].median()
    
    axes[idx].axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[idx].axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    axes[idx].legend(fontsize=9)

plt.tight_layout()
plt.savefig("Distribution.png", dpi=300, bbox_inches='tight')
plt.show()


# Categorical features summary
print("CATEGORICAL FEATURES - DETAILED ANALYSIS")
print("="*80)

for col in categorical_cols:
    print(f"Feature: {col.upper()}")
    print(f"{'='*80}")
    print(f"Unique values: {train_df[col].nunique()}")
    print(f"Most common: {train_df[col].mode()[0]}")
    print(f"\nValue Counts:")
    
    value_counts_df = pd.DataFrame({
        'Value': train_df[col].value_counts().index,
        'Count': train_df[col].value_counts().values,
        'Percentage': (train_df[col].value_counts(normalize=True) * 100).values
    })
    display(value_counts_df.head(10).style.background_gradient(cmap='Blues', subset=['Count', 'Percentage']))


train_df = train_df.copy()
test_df = test_df.copy()


# Clinical ratios and scores
def create_medical_features(df):
    
    # 1. Cardiovascular risk scores
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']  # Important predictor
    df['mean_arterial_pressure'] = (2 * df['diastolic_bp'] + df['systolic_bp']) / 3
    
    # 2. Cholesterol ratios (clinically significant)
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    df['triglyceride_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    
    # 3. Metabolic syndrome indicators
    df['metabolic_syndrome_score'] = (
        (df['bmi'] > 30).astype(int) +
        (df['waist_to_hip_ratio'] > 0.9).astype(int) +  # 0.9 for male, adjust if gender known
        (df['systolic_bp'] >= 130).astype(int) +
        (df['hdl_cholesterol'] < 40).astype(int) +  # For males
        (df['triglycerides'] >= 150).astype(int)
    )
    
    # 4. Blood pressure categories
    df['bp_category'] = np.select(
        [
            (df['systolic_bp'] < 120) & (df['diastolic_bp'] < 80),
            (df['systolic_bp'].between(120, 129)) & (df['diastolic_bp'] < 80),
            (df['systolic_bp'].between(130, 139)) | (df['diastolic_bp'].between(80, 89)),
            (df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)
        ],
        [0, 1, 2, 3],  # Normal, Elevated, Stage 1, Stage 2
        default=0
    )
    
    # 5. BMI categories (WHO standards)
    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 35, 40, 100],
        labels=[0, 1, 2, 3, 4, 5]  # Underweight, Normal, Overweight, Obese I, II, III
    ).astype('float')
    
    # 6. Heart rate variability (resting)
    df['heart_rate_zscore'] = (df['heart_rate'] - df['heart_rate'].mean()) / df['heart_rate'].std()
    
    # 7. Combined risk score
    df['diabetes_risk_score'] = (
        0.3 * (df['bmi'] / 30) +
        0.2 * (df['age'] / 60) +
        0.2 * (df['total_hdl_ratio'] / 5) +
        0.15 * (df['systolic_bp'] / 140) +
        0.15 * (df['waist_to_hip_ratio'] / 1.0)
    )

    # 1. Age-BMI interaction (risk increases with age and weight)
    df['age_bmi_interaction'] = df['age'] * df['bmi'] / 100
    
    # 2. Blood pressure and age
    df['age_systolic_interaction'] = df['age'] * df['systolic_bp'] / 100
    
    
    # 4. Metabolic age (biological vs chronological)
    df['metabolic_age'] = (
        df['bmi'] * 0.5 +
        df['waist_to_hip_ratio'] * 10 +
        df['total_hdl_ratio'] * 2 +
        (df['systolic_bp'] - 120) * 0.1
    )
    
    # 5. Hypertension risk combined with cholesterol
    df['cardiometabolic_risk'] = (
        (df['hypertension_history'] | (df['bp_category'] >= 2)).astype(int) *
        (df['total_hdl_ratio'] > 4).astype(int) *
        (df['bmi'] > 30).astype(int)
    )
    
    # 6. Physical activity compensation for poor diet
    df['activity_diet_balance'] = (
        df['physical_activity_minutes_per_week'] / 100 -
        (10 - df['diet_score']) / 2
    )
    
    # 7. Combined family history
    df['total_family_history'] = (
        df['family_history_diabetes'] +
        df['hypertension_history'] +
        df['cardiovascular_history']
    )
    
    # 8. Screen time and diet interaction
    df['screen_diet_interaction'] = (
        df['screen_time_hours_per_day'] * (10 - df['diet_score'])
    )
    
    # 9. Alcohol and triglycerides (important for metabolic health)
    df['alcohol_triglycerides'] = (
        df['alcohol_consumption_per_week'] * df['triglycerides'] / 100
    )
    
    # 10. Sleep and blood pressure
    df['sleep_bp_interaction'] = (
        abs(df['sleep_hours_per_day'] - 7) * df['systolic_bp'] / 100
    )
    
    
    return df

train_df = create_medical_features(train_df)
test_df = create_medical_features(test_df)


def create_features(df):
    df['cholesterol_ratio'] = df['cholesterol_total'] / df['hdl_cholesterol']
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['central_obesity_score'] = df['bmi'] * df['waist_to_hip_ratio']

    return df

train_df = create_features(train_df)
test_df = create_features(test_df)


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)


train_df.columns


input_cols = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'gender', 'ethnicity', 'education_level',
       'income_level', 'smoking_status', 'employment_status',
       'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'cholesterol_ratio',
       'pulse_pressure', 'central_obesity_score', 'mean_arterial_pressure',
       'total_hdl_ratio', 'ldl_hdl_ratio', 'triglyceride_hdl_ratio',
       'metabolic_syndrome_score', 'bp_category', 'bmi_category',
       'heart_rate_zscore', 'diabetes_risk_score', 'age_bmi_interaction',
       'age_systolic_interaction', 'metabolic_age', 'cardiometabolic_risk',
       'activity_diet_balance', 'total_family_history',
       'screen_diet_interaction', 'alcohol_triglycerides',
       'sleep_bp_interaction']

input_cols


target_col = 'diagnosed_diabetes'
target_col


# Training dataset inputs and target

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy() 

# Validation dataset inputs and target

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy() 

# Testing dataset inputs and target

test_inputs = test_df[input_cols].copy()



numerical_cols = [var for var in train_inputs.columns if train_inputs[var].dtype != 'O']

numerical_cols


categorical_cols = [var for var in train_inputs.columns if train_inputs[var].dtype == 'O']


categorical_cols


train_inputs[categorical_cols].isnull().sum()


train_inputs[numerical_cols].isnull().sum()


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs.loc[:, encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs.loc[:, encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs.loc[:, encoded_cols] = encoder.transform(test_inputs[categorical_cols])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_inputs[numerical_cols])

train_inputs[numerical_cols] = scaler.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = scaler.transform(val_inputs[numerical_cols])
test_inputs[numerical_cols] = scaler.transform(test_inputs[numerical_cols])


X_train = train_inputs[numerical_cols + encoded_cols]
X_val = val_inputs[numerical_cols + encoded_cols]
X_test = test_inputs[numerical_cols + encoded_cols]


# Check class distribution
print("\nðŸ“Š Class Distribution Analysis:")
print(f"Training set - Class 0: {np.sum(train_targets == 0):,} | Class 1: {np.sum(train_targets == 1):,}")
print(f"Validation set - Class 0: {np.sum(val_targets == 0):,} | Class 1: {np.sum(val_targets == 1):,}")

# Check if we have imbalance
class_ratio = np.sum(val_targets == 0) / np.sum(val_targets == 1)
print(f"Class ratio (0:1): {class_ratio:.2f}:1")


import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

xgb_params = {
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_estimators': 10000,
    'learning_rate': 0.05,
    'max_depth': 2,
    'max_leaves': 1000,
    'min_child_weight': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.9,
    'colsample_bynode': 0.9,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': 0.61,
}

print("Training XGBoost with K-Fold Cross Validation")
print("Parameters:")
for key, value in xgb_params.items():
    print(f"  {key}: {value}")

# Initialize K-Fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
models = []

# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, train_targets)):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*50}")
    
    # Split data for this fold
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = train_targets.iloc[train_idx], train_targets.iloc[val_idx]
    
    # Initialize and train model
    model = xgb.XGBClassifier(**xgb_params)
    
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=1000,
        early_stopping_rounds=200
    )
    
    # Predictions
    val_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_fold_val, val_pred_proba)
    auc_scores.append(auc)
    models.append(model)
    
    print(f"Fold {fold + 1} AUC: {auc:.4f}")

# Print overall results
print(f"\n{'='*50}")
print("K-Fold Cross Validation Results")
print(f"{'='*50}")
print(f"Mean AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
print(f"Individual Fold AUCs: {[f'{score:.4f}' for score in auc_scores]}")


test_preds = model.predict(X_test)


submission_df.head()


submission_df['diagnosed_diabetes'] = test_preds

# Verify the update
print("Updated submission preview:")
print(submission_df.head())
print(f"\nSubmission shape: {submission_df.shape}")

# Save the updated submission
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




