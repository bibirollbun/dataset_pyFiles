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


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# Set some display options for better visualization
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("Libraries imported successfully!")



# Load the data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print(f"Training data shape: {df_train.shape}")
print(f"Test data shape: {df_test.shape}")
print(f"\nTarget distribution:")
print(df_train['diagnosed_diabetes'].value_counts(normalize=True))


print("Training Data Head:")
df_train.head()


print("\nTraining Data Info:")
df_train.info()


print("\nMissing Values in Train Data:")
print(df_train.isnull().sum())


print("\nMissing Values in Test Data:")
print(df_test.isnull().sum())


# Descriptive statistics for numerical columns
df_train.describe()


# Distribution of the target variable 'accident_risk'
plt.figure(figsize=(10, 6))
sns.countplot(x='diagnosed_diabetes', data=df_train, palette='pastel', edgecolor='black')
plt.title('Distribution of Diagnosed Diabetes')
plt.xlabel('Diagnosed Diabetes')
plt.ylabel('Count')
plt.show()


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# A more compact view of categorical features vs the target
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9, 0.66, 0.33])
target = 'diagnosed_diabetes'

for i, col in enumerate(categorical_features):
    grouped = df_train.groupby(col)[target].mean()
    axes[i].bar(grouped.index.astype(str), grouped.values, color=colors)
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()


numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'diagnosed_diabetes']]
print(numerical_features)


# Loop through all numerical features
for col in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # --- Left: Distribution (Histogram + KDE) ---
    sns.histplot(df_train[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f"Distribution of {col}", fontsize=12)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")
    
    # --- Right: Boxplot (Outliers) ---
    sns.boxplot(x=df_train[col], ax=axes[1], color='lightcoral')
    axes[1].set_title(f"Boxplot of {col}", fontsize=12)
    axes[1].set_xlabel(col)
    
    # Clean layout
    plt.tight_layout()
    plt.show()


# Save target and remove from training data
target = df_train['diagnosed_diabetes'].values
df_train = df_train.drop(['diagnosed_diabetes'], axis=1)

# Combine for feature engineering
df_combined = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
print(f"Combined shape: {df_combined.shape}")


# def create_features_safe(df):
#     """Create features with proper error handling"""
    
#     df_feat = df.copy()
    
#     print("Creating metabolic features...")
    
#     # --- BASIC HEALTH INDICATORS ---
    
#     # BMI Categories
#     df_feat['bmi_category'] = pd.cut(df['bmi'], 
#                                       bins=[0, 18.5, 25, 30, 35, 40, 100],
#                                       labels=[0, 1, 2, 3, 4, 5])  # Use numeric labels
#     df_feat['bmi_category'] = df_feat['bmi_category'].astype(float)
    
#     # Central Obesity Risk
#     df_feat['central_obesity_risk'] = 0
#     male_mask = df['gender'] == 'Male'
#     female_mask = df['gender'] == 'Female'
#     df_feat.loc[male_mask & (df['waist_to_hip_ratio'] > 0.90), 'central_obesity_risk'] = 1
#     df_feat.loc[female_mask & (df['waist_to_hip_ratio'] > 0.85), 'central_obesity_risk'] = 1
    
#     # Blood Pressure Category
#     df_feat['bp_stage'] = pd.cut(df['systolic_bp'], 
#                                   bins=[0, 120, 130, 140, 180, 300],
#                                   labels=[0, 1, 2, 3, 4]).astype(float)
    
#     # Metabolic Age
#     df_feat['metabolic_age'] = df['age'] * (df['bmi'] / 25) * df['waist_to_hip_ratio']
    
#     print("Creating lifestyle features...")
    
#     # --- LIFESTYLE FEATURES ---
    
#     # Physical Activity
#     df_feat['activity_adequate'] = (df['physical_activity_minutes_per_week'] >= 150).astype(int)
#     df_feat['activity_deficit'] = np.maximum(0, 150 - df['physical_activity_minutes_per_week'])
    
#     # Sleep Quality
#     df_feat['sleep_quality'] = 0
#     good_sleep = (df['sleep_hours_per_day'] >= 7) & (df['sleep_hours_per_day'] <= 9)
#     ok_sleep = (df['sleep_hours_per_day'] >= 6) & (df['sleep_hours_per_day'] <= 10)
#     df_feat.loc[good_sleep, 'sleep_quality'] = 1.0
#     df_feat.loc[ok_sleep & ~good_sleep, 'sleep_quality'] = 0.5
    
#     # Lifestyle Score
#     df_feat['lifestyle_score'] = (
#         df['diet_score'] / 10 + 
#         df_feat['sleep_quality'] + 
#         df_feat['activity_adequate'] - 
#         df['alcohol_consumption_per_week'] / 10 - 
#         df['screen_time_hours_per_day'] / 10
#     )
    
#     print("Creating cardiovascular risk features...")
    
#     # --- CARDIOVASCULAR FEATURES ---
    
#     # Lipid Ratios
#     df_feat['cholesterol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
#     df_feat['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
#     df_feat['triglyceride_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    
#     # Non-HDL Cholesterol
#     df_feat['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
#     # Atherogenic Index
#     df_feat['aip'] = np.log10((df['triglycerides'] + 1) / (df['hdl_cholesterol'] + 1))
    
#     # Blood Pressure Features
#     df_feat['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
#     df_feat['mean_arterial_pressure'] = df['diastolic_bp'] + (df_feat['pulse_pressure'] / 3)
    
#     print("Creating interaction features...")
    
#     # --- INTERACTIONS ---
    
#     # Age-BMI Interactions
#     df_feat['age_bmi_interaction'] = df['age'] * df['bmi']
#     df_feat['age_bmi_squared'] = df['age'] * (df['bmi'] ** 2)
    
#     # Family History Interactions
#     df_feat['family_risk_age'] = df['family_history_diabetes'] * df['age']
#     df_feat['family_risk_bmi'] = df['family_history_diabetes'] * df['bmi']
#     df_feat['family_risk_whr'] = df['family_history_diabetes'] * df['waist_to_hip_ratio']
    
#     # Comorbidity Count
#     df_feat['comorbidity_count'] = (
#         df['hypertension_history'] + 
#         df['cardiovascular_history'] + 
#         df['family_history_diabetes']
#     )
    
#     # Comprehensive Risk Score
#     df_feat['risk_score'] = (
#         (df['age'] / 100) * 2 +
#         (df['bmi'] / 50) * 1.5 +
#         df['waist_to_hip_ratio'] * 2 +
#         df['family_history_diabetes'] * 1.5 +
#         df['hypertension_history'] * 1.2 +
#         (df_feat['cholesterol_hdl_ratio'] / 10) +
#         (df['triglycerides'] / 200)
#     )
    
#     print("Creating polynomial features...")
    
#     # --- POLYNOMIALS ---
    
#     # BMI Polynomials
#     df_feat['bmi_squared'] = df['bmi'] ** 2
#     df_feat['bmi_log'] = np.log1p(df['bmi'])
    
#     # Age Polynomials
#     df_feat['age_squared'] = df['age'] ** 2
#     df_feat['age_log'] = np.log1p(df['age'])
    
#     # Waist-to-Hip Ratio Polynomials
#     df_feat['whr_squared'] = df['waist_to_hip_ratio'] ** 2
    
#     print("Creating binned features...")
    
#     # --- BINNING ---
    
#     # Age Groups
#     df_feat['age_group'] = pd.cut(df['age'], 
#                                   bins=[0, 40, 50, 60, 70, 100],
#                                   labels=[0, 1, 2, 3, 4]).astype(float)
    
#     # HDL Categories
#     df_feat['hdl_low'] = (df['hdl_cholesterol'] < 40).astype(int)
#     df_feat['hdl_optimal'] = (df['hdl_cholesterol'] >= 60).astype(int)
    
#     # Triglyceride Categories
#     df_feat['trig_high'] = (df['triglycerides'] >= 150).astype(int)
#     df_feat['trig_very_high'] = (df['triglycerides'] >= 200).astype(int)
    
#     print("Encoding categorical features...")
    
#     # --- ENCODE CATEGORICALS ---
    
#     # Simple label encoding for categorical features
#     categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
#                        'smoking_status', 'employment_status']
    
#     for col in categorical_cols:
#         if col in df_feat.columns:
#             le = LabelEncoder()
#             df_feat[f'{col}_encoded'] = le.fit_transform(df_feat[col].astype(str))
#             # Keep the encoded version, drop the original
#             df_feat = df_feat.drop(col, axis=1)
    
#     # Drop any remaining non-numeric columns
#     numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
#     df_feat = df_feat[numeric_cols]
    
#     print(f"Final feature count: {df_feat.shape[1]}")
    
#     return df_feat

# # Apply feature engineering
# print("Creating features...")
# df_features = create_features_safe(df_combined)


# from sklearn.preprocessing import LabelEncoder

# def create_features_optimized(df):
#     df_feat = df.copy()
    
#     # --- 1. MEDICAL DOMAIN FEATURES (Keep all these) ---
#     # BMI & Obesity
#     df_feat['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 40, 100], labels=False)
    
#     # Central Obesity (Gender specific)
#     df_feat['central_obesity_risk'] = 0
#     mask_m = (df['gender'] == 'Male') & (df['waist_to_hip_ratio'] > 0.90)
#     mask_f = (df['gender'] == 'Female') & (df['waist_to_hip_ratio'] > 0.85)
#     df_feat.loc[mask_m | mask_f, 'central_obesity_risk'] = 1

#     # Blood Pressure
#     df_feat['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
#     df_feat['map'] = df['diastolic_bp'] + (df_feat['pulse_pressure'] / 3)
    
#     # Metabolic & Lipids
#     # Log transform is good for skewed data like Triglycerides
#     df_feat['log_triglycerides'] = np.log1p(df['triglycerides']) 
#     df_feat['cholesterol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
#     df_feat['aip'] = np.log10((df['triglycerides'] + 1) / (df['hdl_cholesterol'] + 1))

#     # --- 2. COMORBIDITY (Strongest Features) ---
#     # Summing binary flags is very effective
#     df_feat['comorbidity_sum'] = (
#         df['hypertension_history'] + 
#         df['cardiovascular_history'] + 
#         df['family_history_diabetes']
#     )

#     # --- 3. CLEAN UP & ENCODING ---
    
#     # Drop polynomials (Squared, Log of age/BMI) - unnecessary for Trees
    
#     # Handle Categoricals (The XGBoost/LGBM Way)
#     categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
    
#     for col in categorical_cols:
#         if col in df_feat.columns:
#             # Convert to category type - highly efficient for LightGBM/XGBoost
#             df_feat[col] = df_feat[col].astype('category')
            
#             # If you strictly need numbers (e.g. for older XGBoost versions), 
#             # use factorize which is faster than LabelEncoder
#             # df_feat[col] = pd.factorize(df_feat[col])[0]

#     # Drop original ID or non-predictive columns if necessary
#     # (Assuming 'id' is handled outside)

#     print(f"Final feature count: {df_feat.shape[1]}")
    
#     return df_feat

# # Apply feature engineering
# print("Creating features...")
# df_features = create_features_optimized(df_combined)




def create_features(df):
    """Create sophisticated features based on medical domain knowledge"""
    
    df_feat = df.copy()
    
    # --- METABOLIC SYNDROME INDICATORS ---
    # These are key medical markers for diabetes risk
    
    # 1. BMI Categories (WHO classification)
    df_feat['bmi_category'] = pd.cut(df['bmi'], 
                                      bins=[0, 18.5, 25, 30, 35, 40, 100],
                                      labels=['underweight', 'normal', 'overweight', 
                                             'obese1', 'obese2', 'obese3'])
    
    # 2. Central Obesity Risk (waist-to-hip ratio thresholds)
    df_feat['central_obesity_risk'] = ((df['gender'] == 'Male') & (df['waist_to_hip_ratio'] > 0.90) |
                                       (df['gender'] == 'Female') & (df['waist_to_hip_ratio'] > 0.85)).astype(int)
    
    # 3. Hypertension stages
    df_feat['bp_category'] = pd.cut(df['systolic_bp'], 
                                    bins=[0, 120, 130, 140, 180, 300],
                                    labels=['normal', 'elevated', 'stage1', 'stage2', 'crisis'])
    
    # 4. Metabolic Age (biological vs chronological age indicator)
    df_feat['metabolic_age'] = df['age'] * (df['bmi'] / 25) * (df['waist_to_hip_ratio'])
    
    # --- LIFESTYLE RISK FACTORS ---
    
    # 5. Physical Activity Adequacy (WHO recommends 150 min/week)
    df_feat['activity_adequate'] = (df['physical_activity_minutes_per_week'] >= 150).astype(int)
    df_feat['activity_deficit'] = np.maximum(0, 150 - df['physical_activity_minutes_per_week'])
    
    # 6. Sleep Quality Score
    df_feat['sleep_quality'] = np.where(
        (df['sleep_hours_per_day'] >= 7) & (df['sleep_hours_per_day'] <= 9), 1,
        np.where((df['sleep_hours_per_day'] >= 6) & (df['sleep_hours_per_day'] <= 10), 0.5, 0)
    )
    
    # 7. Lifestyle Balance Score
    df_feat['lifestyle_score'] = (
        df['diet_score'] / 10 + 
        df_feat['sleep_quality'] + 
        df_feat['activity_adequate'] - 
        df['alcohol_consumption_per_week'] / 10 - 
        df['screen_time_hours_per_day'] / 10
    )
    
    # --- CARDIOVASCULAR RISK INDICATORS ---
    
    # 8. Lipid Ratios (important for cardiovascular disease)
    df_feat['cholesterol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df_feat['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df_feat['triglyceride_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
    
    # 9. Non-HDL Cholesterol (better predictor than LDL alone)
    df_feat['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # 10. Atherogenic Index of Plasma
    df_feat['aip'] = np.log10(df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5))
    
    # 11. Pulse Pressure (systolic - diastolic)
    df_feat['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # 12. Mean Arterial Pressure
    df_feat['mean_arterial_pressure'] = df['diastolic_bp'] + (df_feat['pulse_pressure'] / 3)
    
    # --- INTERACTION FEATURES ---
    
    # 13. Age-BMI Interaction (obesity impact increases with age)
    df_feat['age_bmi_interaction'] = df['age'] * df['bmi']
    df_feat['age_bmi_squared'] = df['age'] * (df['bmi'] ** 2)
    
    # 14. Family History Interactions
    df_feat['family_risk_age'] = df['family_history_diabetes'] * df['age']
    df_feat['family_risk_bmi'] = df['family_history_diabetes'] * df['bmi']
    
    # 15. Comorbidity Score
    df_feat['comorbidity_count'] = (df['hypertension_history'] + 
                                    df['cardiovascular_history'] + 
                                    df['family_history_diabetes'])
    
    # 16. Comprehensive Risk Score
    df_feat['risk_score'] = (
        (df['age'] / 100) * 2 +
        (df['bmi'] / 50) * 1.5 +
        df['waist_to_hip_ratio'] * 2 +
        df['family_history_diabetes'] * 1.5 +
        df['hypertension_history'] * 1.2 +
        (df_feat['cholesterol_hdl_ratio'] / 10) +
        (df['triglycerides'] / 200)
    )
    
    # --- POLYNOMIAL FEATURES FOR KEY INDICATORS ---
    
    # 17. BMI polynomials
    df_feat['bmi_squared'] = df['bmi'] ** 2
    df_feat['bmi_cubed'] = df['bmi'] ** 3
    df_feat['bmi_sqrt'] = np.sqrt(df['bmi'])
    
    # 18. Age polynomials  
    df_feat['age_squared'] = df['age'] ** 2
    df_feat['age_log'] = np.log1p(df['age'])
    
    # --- BINNING CONTINUOUS FEATURES ---
    
    # 19. Age groups (medical risk categories)
    df_feat['age_group'] = pd.cut(df['age'], 
                                  bins=[0, 40, 50, 60, 70, 100],
                                  labels=['<40', '40-50', '50-60', '60-70', '70+'])
    
    # 20. Triglyceride levels
    df_feat['triglyceride_level'] = pd.cut(df['triglycerides'],
                                           bins=[0, 150, 200, 500, 1000],
                                           labels=['normal', 'borderline', 'high', 'very_high'])
    
    # --- STATISTICAL AGGREGATIONS ---
    
    # 21. Normalized features within demographic groups
    for col in ['bmi', 'cholesterol_total', 'systolic_bp']:
        df_feat[f'{col}_gender_norm'] = df.groupby('gender')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-5)
        )
        df_feat[f'{col}_ethnicity_norm'] = df.groupby('ethnicity')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-5)
        )
    
    # --- ENCODE CATEGORICAL FEATURES ---
    
    categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                       'smoking_status', 'employment_status', 'bmi_category', 
                       'bp_category', 'age_group', 'triglyceride_level']
    
    for col in categorical_cols:
        if col in df_feat.columns:
            # Convert to category type - highly efficient for LightGBM/XGBoost
            df_feat[col] = df_feat[col].astype('category')
            
            # If you strictly need numbers (e.g. for older XGBoost versions), 
            # use factorize which is faster than LabelEncoder
            # df_feat[col] = pd.factorize(df_feat[col])[0]

    print(f"Final feature count: {df_feat.shape[1]}")
    
    return df_feat


# Apply feature engineering
print("Creating features...")
df_features = create_features(df_combined)


final_feature_list = [
    'age',
    'physical_activity_minutes_per_week',
    'diet_score',
    'waist_to_hip_ratio',
    'heart_rate',
    'ldl_cholesterol',
    'triglycerides',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
    'bmi_category',
    'metabolic_age',
    'activity_adequate',
    'activity_deficit',
    'ldl_hdl_ratio',
    'triglyceride_hdl_ratio',
    'aip',
    'age_bmi_interaction',
    'age_bmi_squared',
    'family_risk_age',
    'family_risk_bmi',
    'risk_score',
    'age_squared',
    'age_log',
    'bmi_ethnicity_norm',
    'cholesterol_total_gender_norm',
    'cholesterol_total_ethnicity_norm',
    'systolic_bp_ethnicity_norm',
]


# Split back to train/test
train_features = df_features.iloc[:len(df_train)]
test_features = df_features.iloc[len(df_train):]

# Remove id column if it exists
if 'id' in train_features.columns:
    train_features = train_features.drop(['id'], axis=1)
if 'id' in test_features.columns:
    test_ids = test_features['id'].values
    test_features = test_features.drop(['id'], axis=1)
else:
    test_ids = df_test['id'].values

print(f"Train shape: {train_features.shape}")
print(f"Test shape: {test_features.shape}")


train_features = train_features[final_feature_list]
test_features = test_features[final_feature_list]




# 1. Define XGBoost Parameters (I used optuna to hypertune them)
# xgb_params = {
#     'n_estimators': 20000,
#     'learning_rate': 0.02,
#     'max_depth': 6,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'random_state': 42,
#     'use_label_encoder': False,
#     'eval_metric': 'logloss',
#     'verbosity': 0,
#     'enable_categorical': True,  # Allows 'category' dtype
#     'tree_method': 'hist'
# }

xgb_params = {
    'n_estimators': 20000, # We control this via early stopping
    'random_state': 42,
    'enable_categorical': True,  # Allows 'category' dtype
    'booster': 'gbtree',
    #'device': 'cuda',
    'tree_method': 'hist',     # Fast training
    'enable_categorical': True,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'n_jobs': -1,
    'learning_rate': 0.009935633574318715, 
    'max_depth': 4, 
    'subsample': 0.7311056643140286, 
    'colsample_bytree': 0.6174097862624632, 
    'min_child_weight': 9, 
    'reg_alpha': 6.840824833732264, 
    'reg_lambda': 0.01304997677943209, 
    'gamma': 6.499378163345299e-06
}




# # 2. Prepare for Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store predictions
oof_preds = np.zeros(len(train_features))
final_test_predictions = np.zeros(len(test_features))

# DataFrame to store feature importance from every fold
feat_importance = pd.DataFrame(index=train_features.columns)
feat_importance['importance'] = 0

print(f"{'='*40}")
print("Starting XGBoost Cross-Validation")
print(f"{'='*40}")

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_features, target)):
    
    # Split data
    X_tr = train_features.iloc[train_idx]
    X_val = train_features.iloc[val_idx]
    y_tr = target[train_idx]
    y_val = target[val_idx]
    
    # Initialize fresh model
    model = xgb.XGBClassifier(**xgb_params)
    
    # Train
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=500 
    )
    
    # --- CAPTURE IMPORTANCE ---
    # We add the importance of this fold to the total
    feat_importance['importance'] += model.feature_importances_
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    final_test_predictions += model.predict_proba(test_features)[:, 1] / kfold.n_splits
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.6f}")

# --- PLOTTING CODE ---
# Average the importance over 5 folds
feat_importance['importance'] /= kfold.get_n_splits()
# Sort and take top 25 features
top_features = feat_importance.sort_values('importance', ascending=False).head(25)

plt.figure(figsize=(10, 12))
sns.barplot(x=top_features['importance'], y=top_features.index, palette='viridis')
plt.title('Top 25 Most Important Features (Averaged over 5 Folds)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

# Final CV Score
print(f"\nOVERALL CV AUC: {roc_auc_score(target, oof_preds):.6f}")




feat_importance.sort_values('importance', ascending=False)


# 5. Post-Processing & Submission
# Clip predictions to avoid extremes (optional but recommended based on your previous code)
final_test_predictions = np.clip(final_test_predictions, 0.001, 0.999)

submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': final_test_predictions
})


# Save submission
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission saved!")
print(f"Shape: {submission.shape}")

