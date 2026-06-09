# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                              confusion_matrix, roc_curve)
import xgboost as xgb
import lightgbm as lgb

# Settings
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("All libraries imported successfully!")


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print("DATASET SHAPES:")
print(f"  Training set: {train_df.shape}")
print(f"  Test set: {test_df.shape}")
print(f"  Sample submission: {sample_submission.shape}")

print("\nFIRST 5 ROWS:")
display(train_df.head())


# Dataset information
print("DATASET INFO:")
print(train_df.info())

print("\nBASIC STATISTICS:")
display(train_df.describe())


# Check for missing values
print("MISSING VALUES CHECK:")
missing_values = train_df.isnull().sum()
if missing_values.sum() == 0:
    print("  No missing values found!")
else:
    print(missing_values[missing_values > 0])

# Target variable distribution
print("\nTARGET VARIABLE DISTRIBUTION:")
target_counts = train_df['diagnosed_diabetes'].value_counts()
target_percentage = train_df['diagnosed_diabetes'].value_counts(normalize=True) * 100
print(f"  Class 0 (No Diabetes): {target_counts[0]:,} ({target_percentage[0]:.2f}%)")
print(f"  Class 1 (Diabetes): {target_counts[1]:,} ({target_percentage[1]:.2f}%)")

# Visualize target distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='diagnosed_diabetes', palette='Set2')
plt.title('Target Variable Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosed Diabetes')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.show()


# Separate features by type
numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('id')
if 'diagnosed_diabetes' in numerical_features:
    numerical_features.remove('diagnosed_diabetes')

categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"FEATURE TYPES:")
print(f"\nNumerical Features ({len(numerical_features)}):")
for i, feat in enumerate(numerical_features, 1):
    print(f"  {i:2d}. {feat}")

print(f"\nCategorical Features ({len(categorical_features)}):")
for i, feat in enumerate(categorical_features, 1):
    print(f"  {i}. {feat}")


# Analyze categorical features
print("CATEGORICAL FEATURES ANALYSIS:\n")
for col in categorical_features:
   
    print(f"Feature: {col}")
 
    print(f"Unique values: {train_df[col].nunique()}")
    print(f"\nValue counts:")
    print(train_df[col].value_counts())
    
    # Visualize
    plt.figure(figsize=(10, 4))
    train_df[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'{col.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Correlation analysis with target
print("CORRELATION WITH TARGET VARIABLE:\n")
correlations = train_df[numerical_features + ['diagnosed_diabetes']].corr()['diagnosed_diabetes'].sort_values(ascending=False)
print(correlations)

# Visualize top correlations
top_features = correlations.drop('diagnosed_diabetes').abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
colors = ['green' if correlations[feat] > 0 else 'red' for feat in top_features.index]
plt.barh(range(len(top_features)), [correlations[feat] for feat in top_features.index], color=colors)
plt.yticks(range(len(top_features)), top_features.index)
plt.xlabel('Correlation with Diabetes')
plt.title('Top 10 Features Correlated with Diabetes', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nTOP 10 PREDICTIVE FEATURES:")
for i, (feature, corr_value) in enumerate(top_features.items(), 1):
    direction = "positive" if correlations[feature] > 0 else "negative"
    print(f"  {i:2d}. {feature:40s}: {correlations[feature]:7.4f} ({direction})")


# Outlier detection
print("OUTLIER DETECTION (IQR Method):\n")
outlier_summary = {}
for col in numerical_features:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]
    outlier_percentage = (len(outliers) / len(train_df)) * 100
    if outlier_percentage > 1:
        outlier_summary[col] = outlier_percentage
        print(f"   {col:40s}: {outlier_percentage:5.2f}% outliers")

if not outlier_summary:
    print(" No significant outliers detected (< 1% threshold)")


# Create copies for feature engineering
train_processed = train_df.copy()
test_processed = test_df.copy()

print("CREATING NEW FEATURES:\n")


# 1. BMI Categories (Clinical interpretation)
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

train_processed['bmi_category'] = train_processed['bmi'].apply(categorize_bmi)
test_processed['bmi_category'] = test_processed['bmi'].apply(categorize_bmi)

print("1. BMI Category - Clinical classification")
print(train_processed['bmi_category'].value_counts())
print()


# 2. Cholesterol Ratio (LDL/HDL - cardiovascular risk indicator)
train_processed['cholesterol_ratio'] = train_processed['ldl_cholesterol'] / (train_processed['hdl_cholesterol'] + 1e-5)
test_processed['cholesterol_ratio'] = test_processed['ldl_cholesterol'] / (test_processed['hdl_cholesterol'] + 1e-5)

print("2. Cholesterol Ratio (LDL/HDL)")
print(f"  Mean: {train_processed['cholesterol_ratio'].mean():.2f}")
print(f"  Range: [{train_processed['cholesterol_ratio'].min():.2f}, {train_processed['cholesterol_ratio'].max():.2f}]")
print()


# 3. Blood Pressure Categories
def categorize_bp(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return 'Normal'
    elif systolic < 130 and diastolic < 80:
        return 'Elevated'
    elif systolic < 140 or diastolic < 90:
        return 'Stage1_Hypertension'
    else:
        return 'Stage2_Hypertension'

train_processed['bp_category'] = train_processed.apply(
    lambda row: categorize_bp(row['systolic_bp'], row['diastolic_bp']), axis=1
)
test_processed['bp_category'] = test_processed.apply(
    lambda row: categorize_bp(row['systolic_bp'], row['diastolic_bp']), axis=1
)

print("3. Blood Pressure Categories")
print(train_processed['bp_category'].value_counts())
print()


# 4. Lifestyle Risk Score
train_processed['lifestyle_risk'] = (
    (train_processed['smoking_status'] == 'Current').astype(int) * 2 +
    (train_processed['alcohol_consumption_per_week'] > train_processed['alcohol_consumption_per_week'].median()).astype(int) +
    (train_processed['physical_activity_minutes_per_week'] < train_processed['physical_activity_minutes_per_week'].median()).astype(int) +
    (train_processed['screen_time_hours_per_day'] > train_processed['screen_time_hours_per_day'].median()).astype(int)
)
test_processed['lifestyle_risk'] = (
    (test_processed['smoking_status'] == 'Current').astype(int) * 2 +
    (test_processed['alcohol_consumption_per_week'] > train_processed['alcohol_consumption_per_week'].median()).astype(int) +
    (test_processed['physical_activity_minutes_per_week'] < train_processed['physical_activity_minutes_per_week'].median()).astype(int) +
    (test_processed['screen_time_hours_per_day'] > train_processed['screen_time_hours_per_day'].median()).astype(int)
)

print("4. Lifestyle Risk Score (0-5 scale)")
print(train_processed['lifestyle_risk'].value_counts().sort_index())
print()


# 5. Medical History Count
train_processed['medical_history_count'] = (
    train_processed['family_history_diabetes'] +
    train_processed['hypertension_history'] +
    train_processed['cardiovascular_history']
)
test_processed['medical_history_count'] = (
    test_processed['family_history_diabetes'] +
    test_processed['hypertension_history'] +
    test_processed['cardiovascular_history']
)

print("5. Medical History Count (0-3 scale)")
print(train_processed['medical_history_count'].value_counts().sort_index())
print()


# 6. Age Groups
def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle'
    elif age < 65:
        return 'Senior'
    else:
        return 'Elderly'

train_processed['age_group'] = train_processed['age'].apply(categorize_age)
test_processed['age_group'] = test_processed['age'].apply(categorize_age)

print("6. Age Groups")
print(train_processed['age_group'].value_counts())

print(f"\nNEW DATASET SHAPE: {train_processed.shape}")
print(f"  Added {train_processed.shape[1] - train_df.shape[1]} new features")


# Identify all categorical features (original + engineered)
all_categorical = categorical_features + ['bmi_category', 'bp_category', 'age_group']

print(f"CATEGORICAL FEATURES TO ENCODE ({len(all_categorical)}):")
for i, feat in enumerate(all_categorical, 1):
    print(f"  {i}. {feat}")

# One-hot encoding
train_encoded = pd.get_dummies(train_processed, columns=all_categorical, drop_first=True)
test_encoded = pd.get_dummies(test_processed, columns=all_categorical, drop_first=True)

# Align train and test columns
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

print(f"\nENCODING COMPLETE:")
print(f"  Training shape: {train_encoded.shape}")
print(f"  Test shape: {test_encoded.shape}")


# Separate features and target
X = train_encoded.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_encoded['diagnosed_diabetes']
X_test = test_encoded.drop(['id'], axis=1)

# Ensure X_test has same columns as X
missing_cols = set(X.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X.columns]

test_ids = test_encoded['id']

print(f"FEATURE MATRIX SHAPES:")
print(f"  X_train: {X.shape}")
print(f"  y_train: {y.shape}")
print(f"  X_test: {X_test.shape}")


# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("FEATURE SCALING APPLIED (StandardScaler)")
print("  Features scaled to mean=0, std=1")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTRAIN-VALIDATION SPLIT:")
print(f"  Training set: {X_train.shape[0]:,} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Validation set: {X_val.shape[0]:,} samples ({len(X_val)/len(X)*100:.0f}%)")


# Dictionary to store results
results = {}

print("TRAINING BASELINE MODELS:")



# 1. Logistic Regression
print("\n1. LOGISTIC REGRESSION")
print("-" * 70)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_val)
lr_pred_proba = lr_model.predict_proba(X_val)[:, 1]

lr_accuracy = accuracy_score(y_val, lr_pred)
lr_auc = roc_auc_score(y_val, lr_pred_proba)

results['Logistic Regression'] = {'accuracy': lr_accuracy, 'auc': lr_auc, 'model': lr_model}

print(f"Training complete")
print(f"  Accuracy: {lr_accuracy:.4f}")
print(f"  AUC-ROC:  {lr_auc:.4f}")


# 2. Random Forest
print("\n2. RANDOM FOREST")
print("-" * 70)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_pred_proba = rf_model.predict_proba(X_val)[:, 1]

rf_accuracy = accuracy_score(y_val, rf_pred)
rf_auc = roc_auc_score(y_val, rf_pred_proba)

results['Random Forest'] = {'accuracy': rf_accuracy, 'auc': rf_auc, 'model': rf_model}

print(f"Training complete")
print(f"  Accuracy: {rf_accuracy:.4f}")
print(f"  AUC-ROC:  {rf_auc:.4f}")


# 3. Gradient Boosting
print("\n3. GRADIENT BOOSTING")
print("-" * 70)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_val)
gb_pred_proba = gb_model.predict_proba(X_val)[:, 1]

gb_accuracy = accuracy_score(y_val, gb_pred)
gb_auc = roc_auc_score(y_val, gb_pred_proba)

results['Gradient Boosting'] = {'accuracy': gb_accuracy, 'auc': gb_auc, 'model': gb_model}

print(f"Training complete")
print(f"  Accuracy: {gb_accuracy:.4f}")
print(f"  AUC-ROC:  {gb_auc:.4f}")


# Model comparison
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'AUC-ROC': [results[m]['auc'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
print(comparison_df.to_string(index=False))

# Find best model
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
best_auc = results[best_model_name]['auc']

print(f"\nBEST MODEL: {best_model_name}")
print(f"  AUC-ROC: {best_auc:.4f}")


def engineer_features_v5(df):
    df = df.copy()
    
    # BMI Categories
    df['bmi_category'] = pd.cut(df['bmi'], 
                                 bins=[0, 18.5, 25, 30, 100],
                                 labels=[0, 1, 2, 3]).astype(int)
    
    # Cholesterol Ratios
    df['chol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    df['total_chol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    
    # Blood Pressure
    df['bp_category'] = 0
    df.loc[(df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80), 'bp_category'] = 1
    df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'bp_category'] = 2
    
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    df['hypertension'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
    
    # Age Categories
    df['age_category'] = pd.cut(df['age'], 
                                 bins=[0, 30, 45, 60, 100],
                                 labels=[0, 1, 2, 3]).astype(int)
    
    # Risk Scores
    df['medical_risk'] = (df['family_history_diabetes'] * 0.3 + 
                         df['hypertension_history'] * 0.3 + 
                         df['cardiovascular_history'] * 0.4)
    
    # Interaction Features
    df['age_bmi'] = df['age'] * df['bmi'] / 100
    df['age_chol'] = df['age'] * df['cholesterol_total'] / 100
    df['bmi_chol'] = df['bmi'] * df['cholesterol_total'] / 100
    df['family_age'] = df['family_history_diabetes'] * df['age'] / 10
    
    # Polynomial Features
    df['bmi_squared'] = df['bmi'] ** 2 / 100
    df['chol_squared'] = df['cholesterol_total'] ** 2 / 1000
    df['age_squared'] = df['age'] ** 2 / 1000
    
    # Additional Interactions
    df['chol_hdl'] = df['cholesterol_total'] * df['hdl_cholesterol'] / 100
    df['bp_bmi'] = df['systolic_bp'] * df['bmi'] / 100
    
    # Encode Categorical Variables
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0.5)
    
    if 'smoking_status' in df.columns:
        df['smoking_status'] = df['smoking_status'].map({
            'Never': 0, 'Former': 0.5, 'Current': 1
        }).fillna(0)
        df['lifestyle_risk'] = (df['smoking_status'] * 0.4 + 
                               (1 - df['physical_activity_minutes_per_week'] / 300) * 0.3 + 
                               (df['bmi'] > 30).astype(int) * 0.3)
    
    if 'employment_status' in df.columns:
        df['employment_status'] = df['employment_status'].map({
            'Unemployed': 0, 'Part-Time': 0.33, 'Self-Employed': 0.67, 'Full-Time': 1
        }).fillna(0.5)
    
    if 'education_level' in df.columns:
        df['education_level'] = df['education_level'].map({
            'No Formal Education': 0, 'Primary School': 1, 
            'High School': 2, 'Associate Degree': 3,
            "Bachelor's Degree": 4, "Master's Degree": 5, 'Doctorate': 6
        }).fillna(2)
    
    if 'income_level' in df.columns:
        df['income_level'] = df['income_level'].map({
            'Low': 0, 'Lower Middle': 1, 'Middle': 2, 
            'Upper Middle': 3, 'High': 4
        }).fillna(2)
    
    if 'ethnicity' in df.columns:
        ethnicity_dummies = pd.get_dummies(df['ethnicity'], prefix='ethnicity')
        df = pd.concat([df.drop('ethnicity', axis=1), ethnicity_dummies], axis=1)
    
    return df

print("Feature engineering function defined for V5")


# Apply V5 feature engineering
train_v5 = engineer_features_v5(train_df)
test_v5 = engineer_features_v5(test_df)

print(f"Features after V5 engineering: {train_v5.shape[1]}")


# Prepare V5 features
X_v5 = train_v5.drop(['id', 'diagnosed_diabetes'], axis=1)
y_v5 = train_v5['diagnosed_diabetes']
X_test_v5 = test_v5.drop(['id'], axis=1)

print(f"Final V5 feature set: {X_v5.shape[1]} features")

scaler_v5 = StandardScaler()
X_v5_scaled = scaler_v5.fit_transform(X_v5)
X_test_v5_scaled = scaler_v5.transform(X_test_v5)


# Cross-Validation Setup
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

train_meta = np.zeros((len(X_v5), 4))
test_meta = np.zeros((len(X_test_v5), 4))

print("5-Fold Cross-Validation initialized")


# Model 1: XGBoost
print("Training XGBoost with 5-Fold CV...")
xgb_params = {
    'n_estimators': 275,
    'max_depth': 5,
    'learning_rate': 0.045,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1.5,
    'scale_pos_weight': len(y_v5[y_v5==0]) / len(y_v5[y_v5==1]),
    'reg_alpha': 0.08,
    'reg_lambda': 0.8,
    'random_state': 42,
    'eval_metric': 'auc',
    'tree_method': 'hist'
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_v5_scaled, y_v5)):
    X_tr, X_val = X_v5_scaled[train_idx], X_v5_scaled[val_idx]
    y_tr, y_val = y_v5.iloc[train_idx], y_v5.iloc[val_idx]
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, verbose=False)
    
    train_meta[val_idx, 0] = model.predict_proba(X_val)[:, 1]
    test_meta[:, 0] += model.predict_proba(X_test_v5_scaled)[:, 1] / n_folds
    
    auc = roc_auc_score(y_val, train_meta[val_idx, 0])
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"XGBoost Overall CV AUC: {roc_auc_score(y_v5, train_meta[:, 0]):.4f}")


# Model 2: LightGBM
print("\nTraining LightGBM with 5-Fold CV...")
lgb_params = {
    'n_estimators': 275,
    'max_depth': 5,
    'learning_rate': 0.045,
    'num_leaves': 25,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 30,
    'reg_alpha': 0.08,
    'reg_lambda': 0.8,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_v5_scaled, y_v5)):
    X_tr, X_val = X_v5_scaled[train_idx], X_v5_scaled[val_idx]
    y_tr, y_val = y_v5.iloc[train_idx], y_v5.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)
    
    train_meta[val_idx, 1] = model.predict_proba(X_val)[:, 1]
    test_meta[:, 1] += model.predict_proba(X_test_v5_scaled)[:, 1] / n_folds
    
    auc = roc_auc_score(y_val, train_meta[val_idx, 1])
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"LightGBM Overall CV AUC: {roc_auc_score(y_v5, train_meta[:, 1]):.4f}")


# Model 3: Random Forest
print("\nTraining Random Forest with 5-Fold CV...")
rf_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 40,
    'min_samples_leaf': 20,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_v5_scaled, y_v5)):
    X_tr, X_val = X_v5_scaled[train_idx], X_v5_scaled[val_idx]
    y_tr, y_val = y_v5.iloc[train_idx], y_v5.iloc[val_idx]
    
    model = RandomForestClassifier(**rf_params)
    model.fit(X_tr, y_tr)
    
    train_meta[val_idx, 2] = model.predict_proba(X_val)[:, 1]
    test_meta[:, 2] += model.predict_proba(X_test_v5_scaled)[:, 1] / n_folds
    
    auc = roc_auc_score(y_val, train_meta[val_idx, 2])
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"Random Forest Overall CV AUC: {roc_auc_score(y_v5, train_meta[:, 2]):.4f}")


# # Model 4: Gradient Boosting
# print("\nTraining Gradient Boosting with 5-Fold CV...")
# gb_params = {
#     'n_estimators': 200,
#     'max_depth': 4,
#     'learning_rate': 0.05,
#     'subsample': 0.8,
#     'min_samples_split': 40,
#     'min_samples_leaf': 20,
#     'random_state': 42
# }

# for fold, (train_idx, val_idx) in enumerate(skf.split(X_v5_scaled, y_v5)):
#     X_tr, X_val = X_v5_scaled[train_idx], X_v5_scaled[val_idx]
#     y_tr, y_val = y_v5.iloc[train_idx], y_v5.iloc[val_idx]
    
#     model = GradientBoostingClassifier(**gb_params)
#     model.fit(X_tr, y_tr)
    
#     train_meta[val_idx, 3] = model.predict_proba(X_val)[:, 1]
#     test_meta[:, 3] += model.predict_proba(X_test_v5_scaled)[:, 1] / n_folds
    
#     auc = roc_auc_score(y_val, train_meta[val_idx, 3])
#     print(f"  Fold {fold+1}: AUC = {auc:.4f}")

# print(f"Gradient Boosting Overall CV AUC: {roc_auc_score(y_v5, train_meta[:, 3]):.4f}")


# Simple ensemble (equal weights)
test_pred_v5 = test_meta.mean(axis=1)
train_pred_v5 = train_meta.mean(axis=1)

print(f"Ensemble CV AUC: {roc_auc_score(y_v5, train_pred_v5):.4f}")
print(f"Predictions: {(test_pred_v5 > 0.5).sum() / len(test_pred_v5) * 100:.1f}% diabetes")


# Create V5 submission
submission = pd.DataFrame({
    'id': test_v5['id'],
    'diagnosed_diabetes': test_pred_v5
})

submission.to_csv('submission.csv', index=False)
print("Submission saved: submission.csv")

