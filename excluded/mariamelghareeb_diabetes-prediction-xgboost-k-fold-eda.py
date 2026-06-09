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
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


print(f"Train dataset: {train_df.shape[0]:,} rows Ã— {train_df.shape[1]} columns")
print(f"Test dataset: {test_df.shape[0]:,} rows Ã— {test_df.shape[1]} columns")
print(f"Sample submission: {sample_submission.shape[0]:,} rows Ã— {sample_submission.shape[1]} columns")

print("First 5 rows of training data:")
print(train_df.head())

print("Dataset Information:")
print(train_df.info())


id_col = 'id'
target_col = 'diagnosed_diabetes'


# Numeric features
numeric_features = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides'
]

# Categorical features
categorical_features = [
    'gender', 'ethnicity', 'education_level', 'income_level',
    'smoking_status', 'employment_status'
]

# Binary features (can be treated as numeric)
binary_features = [
    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
]

print(f"\nğŸ“Š Feature Distribution:")
print(f"   â€¢ Numeric Features: {len(numeric_features)}")
print(f"   â€¢ Categorical Features: {len(categorical_features)}")
print(f"   â€¢ Binary Features: {len(binary_features)}")
print(f"   â€¢ Total Features: {len(numeric_features) + len(categorical_features) + len(binary_features)}")


print("\nTarget Distribution:")
print(train_df[target_col].value_counts())
print("\nTarget Proportion:")
target_prop = train_df[target_col].value_counts(normalize=True)
print(target_prop)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
target_counts = train_df[target_col].value_counts()
axes[0].bar(['No Diabetes (0)', 'Diabetes (1)'], target_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 1000, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(target_counts.values, labels=['No Diabetes (0)', 'Diabetes (1)'], 
           autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Target Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# Check for class imbalance
imbalance_ratio = target_counts.max() / target_counts.min()
print(f"\nâš ï¸� Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 1.5:
    print("   â†’ Moderate imbalance detected. Consider using stratified sampling.")
else:
    print("   â†’ Classes are relatively balanced.")



# 2.2 Missing Values Analysis
print("\n" + "-" * 100)
print("ğŸ“Š 2.2 MISSING VALUES ANALYSIS")
print("-" * 100)

missing_train = train_df.isnull().sum()
missing_train_pct = (missing_train / len(train_df)) * 100
missing_df = pd.DataFrame({
    'Feature': missing_train.index,
    'Missing_Count': missing_train.values,
    'Missing_Percentage': missing_train_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print("\nâš ï¸� Missing Values Found in Training Data:")
    print(missing_df.to_string(index=False))
else:
    print("\nâœ“ No missing values in training data!")

missing_test = test_df.isnull().sum()
missing_test_pct = (missing_test / len(test_df)) * 100
missing_test_df = pd.DataFrame({
    'Feature': missing_test.index,
    'Missing_Count': missing_test.values,
    'Missing_Percentage': missing_test_pct.values
})
missing_test_df = missing_test_df[missing_test_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_test_df) > 0:
    print("\nâš ï¸� Missing Values Found in Test Data:")
    print(missing_test_df.to_string(index=False))
else:
    print("\nâœ“ No missing values in test data!")

# 2.3 Numeric Features Analysis
print("\n" + "-" * 100)
print("ğŸ“Š 2.3 NUMERIC FEATURES ANALYSIS")
print("-" * 100)

print("\nStatistical Summary:")
print(train_df[numeric_features].describe())


# Distribution plots for numeric features
fig, axes = plt.subplots(5, 3, figsize=(18, 20))
axes = axes.flatten()

for idx, col in enumerate(numeric_features):
    axes[idx].hist(train_df[col], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[idx].set_title(f'{col}', fontweight='bold', fontsize=11)
    axes[idx].set_xlabel(col, fontsize=9)
    axes[idx].set_ylabel('Frequency', fontsize=9)
    axes[idx].axvline(train_df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[idx].axvline(train_df[col].median(), color='green', linestyle='--', linewidth=2, label='Median')
    axes[idx].legend(fontsize=8)

plt.suptitle('Distribution of Numeric Features', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()


# Categorical feature distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_features):
    value_counts = train_df[col].value_counts()
    axes[idx].bar(range(len(value_counts)), value_counts.values, color='coral', edgecolor='black')
    axes[idx].set_title(f'{col}', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Count', fontsize=10)
    axes[idx].set_xticks(range(len(value_counts)))
    axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)

plt.suptitle('Distribution of Categorical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Relationship between categorical features and target
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_features):
    ct = pd.crosstab(train_df[col], train_df[target_col], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[idx].set_title(f'{col} vs Diabetes', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Percentage (%)', fontsize=10)
    axes[idx].legend(['No Diabetes', 'Diabetes'], fontsize=9)
    axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle('Categorical Features vs Target Variable', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


print("\n" + "-" * 100)
print("ğŸ“Š 2.5 BINARY FEATURES ANALYSIS")
print("-" * 100)

print("\nBinary Feature Distributions:")
for col in binary_features:
    print(f"\n{col}:")
    print(train_df[col].value_counts())
    print(f"Proportion: {train_df[col].value_counts(normalize=True)}")

# Binary features visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, col in enumerate(binary_features):
    ct = pd.crosstab(train_df[col], train_df[target_col])
    ct.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[idx].set_title(f'{col} vs Diabetes', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Count', fontsize=10)
    axes[idx].legend(['No Diabetes', 'Diabetes'], fontsize=10)
    axes[idx].tick_params(axis='x', rotation=0)

plt.suptitle('Binary Features vs Target Variable', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# 2.6 Correlation Analysis
print("\n" + "-" * 100)
print("ğŸ“Š 2.6 CORRELATION ANALYSIS")
print("-" * 100)

# Correlation matrix for numeric features
all_numeric = numeric_features + binary_features + [target_col]
correlation_matrix = train_df[all_numeric].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix (Numeric + Binary Features)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Top correlations with target
target_corr = correlation_matrix[target_col].drop(target_col).sort_values(ascending=False)
print("\nğŸ�¯ Features Most Correlated with Diabetes:")
print(target_corr)

# Visualize top correlations
fig, ax = plt.subplots(figsize=(10, 8))
target_corr.plot(kind='barh', ax=ax, color=['green' if x > 0 else 'red' for x in target_corr.values])
ax.set_xlabel('Correlation with Diabetes', fontsize=12, fontweight='bold')
ax.set_title('Feature Correlation with Target', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()


print("\n" + "-" * 100)
print("ğŸ“Š 2.7 OUTLIER DETECTION")
print("-" * 100)

outlier_summary = []
for col in numeric_features:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)][col]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(train_df)) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Outlier_Count': outlier_count,
        'Outlier_Percentage': outlier_pct,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Count', ascending=False)
print("\nOutlier Summary (IQR Method):")
print(outlier_df.to_string(index=False))


print("\nğŸ�¯ Preprocessing Strategy:")
print("   1. Handle missing values (if any)")
print("   2. Encode categorical features")
print("   3. Keep binary features as numeric")
print("   4. Create feature copies for different scaling methods")
print("   5. Handle outliers (winsorization)")


train_processed = train_df.copy()
test_processed = test_df.copy()

# Save IDs
test_ids = test_processed[id_col]

# Separate features and target
X_train = train_processed.drop([id_col, target_col], axis=1)
y_train = train_processed[target_col]
X_test = test_processed.drop([id_col], axis=1)

print(f"\nâœ“ Initial shapes:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")


print("\n" + "-" * 100)
print("ğŸ”§ 3.2 ENCODING CATEGORICAL FEATURES")
print("-" * 100)

print(f"\nEncoding {len(categorical_features)} categorical features:")

# Label Encoding for categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le
    print(f"   âœ“ {col}: {len(le.classes_)} unique values encoded")


print("\n" + "-" * 100)
print("ğŸ”§ 3.3 HANDLING OUTLIERS")
print("-" * 100)

print("\nApplying Winsorization (clipping at 0.5th and 99.5th percentiles)...")
for col in numeric_features:
    lower = np.percentile(X_train[col], 0.5)
    upper = np.percentile(X_train[col], 99.5)
    
    X_train[col] = np.clip(X_train[col], lower, upper)
    X_test[col] = np.clip(X_test[col], lower, upper)

print("   âœ“ Outliers handled for all numeric features")

print(f"\nâœ“ Preprocessed shapes:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")


def create_advanced_features(df):
    """Create domain-specific features for diabetes prediction"""
    df = df.copy()
    
    print("\nğŸ”§ Creating advanced features:")
    feature_count = 0
    
    # 1. BMI Categories (WHO standards)
    if 'bmi' in df.columns:
        df['bmi_underweight'] = (df['bmi'] < 18.5).astype(int)
        df['bmi_normal'] = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).astype(int)
        df['bmi_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
        df['bmi_obese_1'] = ((df['bmi'] >= 30) & (df['bmi'] < 35)).astype(int)
        df['bmi_obese_2'] = (df['bmi'] >= 35).astype(int)
        df['bmi_squared'] = df['bmi'] ** 2
        df['bmi_log'] = np.log1p(df['bmi'])
        feature_count += 7
        print(f"   âœ“ BMI features: 7")
    
    # 2. Blood Pressure Features
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['bp_pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['bp_mean_arterial'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
        df['bp_hypertension_stage1'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
        df['bp_hypertension_stage2'] = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).astype(int)
        df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
        feature_count += 5
        print(f"   âœ“ Blood pressure features: 5")
    
    # 3. Cholesterol Ratios
    if all(col in df.columns for col in ['cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol']):
        df['chol_total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
        df['chol_ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
        df['chol_non_hdl'] = df['cholesterol_total'] - df['hdl_cholesterol']
        df['chol_risk_high'] = (df['cholesterol_total'] > 240).astype(int)
        df['hdl_low'] = (df['hdl_cholesterol'] < 40).astype(int)
        df['ldl_high'] = (df['ldl_cholesterol'] > 160).astype(int)
        feature_count += 6
        print(f"   âœ“ Cholesterol features: 6")
    
    # 4. Triglycerides
    if 'triglycerides' in df.columns:
        df['triglycerides_high'] = (df['triglycerides'] > 150).astype(int)
        df['triglycerides_very_high'] = (df['triglycerides'] > 200).astype(int)
        df['triglycerides_log'] = np.log1p(df['triglycerides'])
        feature_count += 3
        print(f"   âœ“ Triglycerides features: 3")
    
    # 5. Lifestyle Risk Score
    if all(col in df.columns for col in ['physical_activity_minutes_per_week', 'diet_score', 
                                          'sleep_hours_per_day', 'alcohol_consumption_per_week']):
        # Normalize and combine (lower is worse)
        df['lifestyle_risk'] = (
            (df['physical_activity_minutes_per_week'] < 150).astype(int) * 0.25 +
            (df['diet_score'] < 5).astype(int) * 0.25 +
            ((df['sleep_hours_per_day'] < 7) | (df['sleep_hours_per_day'] > 9)).astype(int) * 0.25 +
            (df['alcohol_consumption_per_week'] > 7).astype(int) * 0.25
        )
        
        df['sedentary'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
        df['poor_diet'] = (df['diet_score'] < 5).astype(int)
        df['poor_sleep'] = ((df['sleep_hours_per_day'] < 7) | (df['sleep_hours_per_day'] > 9)).astype(int)
        df['excessive_alcohol'] = (df['alcohol_consumption_per_week'] > 7).astype(int)
        feature_count += 5
        print(f"   âœ“ Lifestyle risk features: 5")
    
    # 6. Age-based features
    if 'age' in df.columns:
        df['age_squared'] = df['age'] ** 2
        df['age_high_risk'] = (df['age'] >= 45).astype(int)
        df['age_very_high_risk'] = (df['age'] >= 65).astype(int)
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
        feature_count += 4
        print(f"   âœ“ Age features: 4")
    
    # 7. Screen Time Risk
    if 'screen_time_hours_per_day' in df.columns:
        df['excessive_screen_time'] = (df['screen_time_hours_per_day'] > 4).astype(int)
        df['screen_time_squared'] = df['screen_time_hours_per_day'] ** 2
        feature_count += 2
        print(f"   âœ“ Screen time features: 2")
    
    # 8. Waist-to-Hip Ratio Risk
    if 'waist_to_hip_ratio' in df.columns:
        df['whr_high_risk_male'] = (df['waist_to_hip_ratio'] > 0.90).astype(int)
        df['whr_high_risk_female'] = (df['waist_to_hip_ratio'] > 0.85).astype(int)
        df['whr_squared'] = df['waist_to_hip_ratio'] ** 2
        feature_count += 3
        print(f"   âœ“ Waist-to-hip ratio features: 3")
    
    # 9. Heart Rate Features
    if 'heart_rate' in df.columns:
        df['heart_rate_high'] = (df['heart_rate'] > 100).astype(int)
        df['heart_rate_low'] = (df['heart_rate'] < 60).astype(int)
        feature_count += 2
        print(f"   âœ“ Heart rate features: 2")
    
    # 10. Combined Risk Factors
    if all(col in df.columns for col in ['family_history_diabetes', 'hypertension_history', 
                                          'cardiovascular_history']):
        df['total_medical_history'] = (df['family_history_diabetes'] + 
                                       df['hypertension_history'] + 
                                       df['cardiovascular_history'])
        df['multiple_risk_factors'] = (df['total_medical_history'] >= 2).astype(int)
        feature_count += 2
        print(f"   âœ“ Medical history features: 2")
    
    # 11. Interaction Features
    if 'bmi' in df.columns and 'age' in df.columns:
        df['bmi_age_interaction'] = df['bmi'] * df['age']
        feature_count += 1
        print(f"   âœ“ BMI-Age interaction: 1")
    
    if 'bmi' in df.columns and 'waist_to_hip_ratio' in df.columns:
        df['bmi_whr_interaction'] = df['bmi'] * df['waist_to_hip_ratio']
        feature_count += 1
        print(f"   âœ“ BMI-WHR interaction: 1")
    
    if 'physical_activity_minutes_per_week' in df.columns and 'bmi' in df.columns:
        df['activity_bmi_ratio'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
        feature_count += 1
        print(f"   âœ“ Activity-BMI ratio: 1")
    
    print(f"\n   ğŸ“Š Total new features created: {feature_count}")
    
    return df

# Apply feature engineering
print("\nğŸš€ Applying feature engineering...")
X_train_engineered = create_advanced_features(X_train)
X_test_engineered = create_advanced_features(X_test)

print(f"\nâœ“ Feature engineering complete!")
print(f"   Original features: {X_train.shape[1]}")
print(f"   Total features: {X_train_engineered.shape[1]}")
print(f"   New features added: {X_train_engineered.shape[1] - X_train.shape[1]}")



# StandardScaler
scaler_standard = StandardScaler()
X_train_standard = pd.DataFrame(
    scaler_standard.fit_transform(X_train_engineered),
    columns=X_train_engineered.columns,
    index=X_train_engineered.index
)
X_test_standard = pd.DataFrame(
    scaler_standard.transform(X_test_engineered),
    columns=X_test_engineered.columns,
    index=X_test_engineered.index
)
print("   âœ“ StandardScaler applied")


X_scaled = X_train_standard.values
X_test_scaled = X_test_standard.values
y = y_train





# Random Forest (CPU with all cores)
rf_params = {
    'n_estimators': 300,
    'max_depth': 15,
    'max_features': 'sqrt',
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'n_jobs': -1,  # Use all CPU cores
    'random_state': 42,
    'verbose': 0
}
print("âœ“ Random Forest configured (CPU multi-threaded)")

# LightGBM (GPU) - P100 Compatible
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 7,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1
}
print("âœ“ LightGBM configured with GPU support (P100 compatible)")

# XGBoost (GPU) - P100 Compatible
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'gpu_hist',  # GPU acceleration
    'predictor': 'gpu_predictor',
    'gpu_id': 0
}
print("âœ“ XGBoost configured with GPU support (P100 compatible)")



n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize arrays for meta-features
train_meta = np.zeros((len(X_scaled), 3))
test_meta = np.zeros((len(X_test_scaled), 3))

# Store OOF scores
oof_scores = {'rf': [], 'lgb': [], 'xgb': []}

print(f"\nTraining with {n_folds}-fold cross-validation...")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}/{n_folds}")
    print(f"{'='*60}")
    
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")


    print("\n[1/3] Training Random Forest (CPU)...")
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train_fold, y_train_fold)
    rf_pred_val = rf.predict_proba(X_val_fold)[:, 1]
    rf_pred_test = rf.predict_proba(X_test_scaled)[:, 1]
    train_meta[val_idx, 0] = rf_pred_val
    test_meta[:, 0] += rf_pred_test / n_folds
    rf_auc = roc_auc_score(y_val_fold, rf_pred_val)
    oof_scores['rf'].append(rf_auc)
    print(f"  âœ“ Random Forest AUC: {rf_auc:.4f}")


    print("\n[2/3] Training LightGBM (GPU)...")
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_pred_val = lgb_clf.predict_proba(X_val_fold)[:, 1]
    lgb_pred_test = lgb_clf.predict_proba(X_test_scaled)[:, 1]
    train_meta[val_idx, 1] = lgb_pred_val
    test_meta[:, 1] += lgb_pred_test / n_folds
    lgb_auc = roc_auc_score(y_val_fold, lgb_pred_val)
    oof_scores['lgb'].append(lgb_auc)
    print(f"  âœ“ LightGBM AUC: {lgb_auc:.4f}")


    print("\n[3/3] Training XGBoost (GPU)...")
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )
    xgb_pred_val = xgb_clf.predict_proba(X_val_fold)[:, 1]
    xgb_pred_test = xgb_clf.predict_proba(X_test_scaled)[:, 1]
    train_meta[val_idx, 2] = xgb_pred_val
    test_meta[:, 2] += xgb_pred_test / n_folds
    xgb_auc = roc_auc_score(y_val_fold, xgb_pred_val)
    oof_scores['xgb'].append(xgb_auc)
    print(f"  âœ“ XGBoost AUC: {xgb_auc:.4f}")
    
    print(f"\nFold {fold} Summary:")
    print(f"  RF: {rf_auc:.4f} | LightGBM: {lgb_auc:.4f} | XGB: {xgb_auc:.4f}")

print("\n" + "="*60)
print("BASE MODELS TRAINING COMPLETED!")
print("="*60)





print("MODEL EVALUATION & PERFORMANCE METRICS")
print("="*60)

# Calculate mean CV scores
print("\nCross-Validation Performance (Mean Â± Std):")
print(f"Random Forest: {np.mean(oof_scores['rf']):.4f} Â± {np.std(oof_scores['rf']):.4f}")
print(f"LightGBM:      {np.mean(oof_scores['lgb']):.4f} Â± {np.std(oof_scores['lgb']):.4f}")
print(f"XGBoost:       {np.mean(oof_scores['xgb']):.4f} Â± {np.std(oof_scores['xgb']):.4f}")

# Full OOF performance
print("\nOut-of-Fold (OOF) Performance:")
rf_oof_auc = roc_auc_score(y, train_meta[:, 0])
lgb_oof_auc = roc_auc_score(y, train_meta[:, 1])
xgb_oof_auc = roc_auc_score(y, train_meta[:, 2])
print(f"Random Forest: {rf_oof_auc:.4f}")
print(f"LightGBM:      {lgb_oof_auc:.4f}")
print(f"XGBoost:       {xgb_oof_auc:.4f}")


import pandas as pd
from sklearn.metrics import roc_auc_score

rf_oof_auc = roc_auc_score(y, train_meta[:, 0])
lgb_oof_auc = roc_auc_score(y, train_meta[:, 1])
xgb_oof_auc = roc_auc_score(y, train_meta[:, 2])

oof_aucs = {
    'Random Forest': rf_oof_auc,
    'LightGBM': lgb_oof_auc,
    'XGBoost': xgb_oof_auc
}

print("OOF AUC Scores:")
for model_name, auc in oof_aucs.items():
    print(f"{model_name}: {auc:.6f}")


best_model_name = max(oof_aucs, key=oof_aucs.get)
print(f"\nBest model based on OOF AUC: {best_model_name} ({oof_aucs[best_model_name]:.6f})")


model_idx = {'Random Forest': 0, 'LightGBM': 1, 'XGBoost': 2}[best_model_name]

submission = pd.DataFrame({
    'id': test_ids,
    'diabetes': test_meta[:, model_idx]  # Already averaged over folds
})

filename = f'submission_best_model_{best_model_name.lower().replace(" ", "_")}.csv'
submission.to_csv(filename, index=False)
print(f"Submission file created: {filename}")





