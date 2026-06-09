import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')  # Assuming the competition name is s4e2 or similar; adjust if needed
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.head()


# Display basic info
print("Train shape:", train.shape)
print("\nTrain info:")
train.info()


print("Test shape:", test.shape)
print("\nTest info:")
test.info()


# Exploratory Data Analysis (EDA)
# Check for missing values
print("Missing values in train:")
print(train.isnull().sum())


print("\nMissing values in test:")
print(test.isnull().sum())


# 1. Check unique values in target to confirm it's binary
print("Unique values in diagnosed_diabetes:")
print(train['diagnosed_diabetes'].value_counts())
print(f"\nTarget imbalance ratio: {train['diagnosed_diabetes'].value_counts()[1] / len(train):.4f} (positive class)")


# 3. Target distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='diagnosed_diabetes')
plt.title('Distribution of Diagnosed Diabetes')
plt.show()



# 4. Distributions of key numerical features (sample for efficiency)
sample_train = train.sample(10000, random_state=42)  # Sample for faster plotting

numerical_features = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'diet_score', 'sleep_hours_per_day', 
                      'physical_activity_minutes_per_week', 'alcohol_consumption_per_week']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    sns.histplot(data=sample_train, x=col, kde=True, ax=axes[idx])
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# 5. Boxplots of numerical features by target (sample)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    sns.boxplot(data=sample_train, x='diagnosed_diabetes', y=col, ax=axes[idx])
    axes[idx].set_title(f'{col} by Diabetes Diagnosis')

plt.tight_layout()
plt.show()


# 6. Categorical features distributions
categorical_features = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    sns.countplot(data=train, x=col, ax=axes[idx], order=train[col].value_counts().index)
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# 7. Categorical features vs target (countplots)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    sns.countplot(data=train, x=col, hue='diagnosed_diabetes', ax=axes[idx], 
                  order=train[col].value_counts().index)
    axes[idx].set_title(f'{col} by Diabetes Diagnosis')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# 8. Correlation heatmap for numerical features
plt.figure(figsize=(12, 10))
corr_matrix = sample_train[numerical_features + ['diagnosed_diabetes']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()



# 9. Top correlations with target
corr_with_target = corr_matrix['diagnosed_diabetes'].drop('diagnosed_diabetes').sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
plt.title('Correlation of Numerical Features with Diagnosed Diabetes')
plt.xlabel('Correlation Coefficient')
plt.show()


# 10. Check for outliers in key numerical features (using IQR method)
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return len(outliers)

outlier_summary = {}
for col in numerical_features:
    outlier_summary[col] = detect_outliers(train, col)

print("Outlier counts per feature:")
for col, count in outlier_summary.items():
    print(f"{col}: {count} ({count/len(train)*100:.2f}%)")


# 11. Binary features vs target
binary_features = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, col in enumerate(binary_features):
    sns.countplot(data=train, x=col, hue='diagnosed_diabetes', ax=axes[idx])
    axes[idx].set_title(f'{col} by Diabetes Diagnosis')

plt.tight_layout()
plt.show()


# 1. Feature Engineering

# Create BMI categories (Underweight, Normal, Overweight, Obese)
def create_bmi_categories(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif bmi < 25:
        return 1  # Normal
    elif bmi < 30:
        return 2  # Overweight
    else:
        return 3  # Obese

train['bmi_category'] = train['bmi'].apply(create_bmi_categories)
test['bmi_category'] = test['bmi'].apply(create_bmi_categories)

# Blood Pressure categories (Normal, Elevated, Hypertension Stage 1, Stage 2)
def create_bp_category(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return 0  # Normal
    elif systolic < 130 and diastolic < 80:
        return 1  # Elevated
    elif systolic < 140 or diastolic < 90:
        return 2  # Stage 1
    else:
        return 3  # Stage 2

train['bp_category'] = train.apply(lambda row: create_bp_category(row['systolic_bp'], row['diastolic_bp']), axis=1)
test['bp_category'] = test.apply(lambda row: create_bp_category(row['systolic_bp'], row['diastolic_bp']), axis=1)

# Interaction feature: age * bmi (risk increases with age and obesity)
train['age_bmi_interaction'] = train['age'] * train['bmi']
test['age_bmi_interaction'] = test['age'] * test['bmi']

# Risk score: simple sum of binary risks
risk_features = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 'bmi_category']
train['risk_score'] = train[risk_features].sum(axis=1)
test['risk_score'] = test[risk_features].sum(axis=1)

print("New features created. Updated train shape:", train.shape)


train.head()


# 2. Handle Class Imbalance
imbalance_ratio = train['diagnosed_diabetes'].value_counts()[1] / len(train)
print(f"Positive class ratio: {imbalance_ratio:.4f}")

if imbalance_ratio < 0.3:  # Threshold for imbalance
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    # Will use in modeling below
    print("Imbalance detected. Will use SMOTE in CV.")
else:
    print("No significant imbalance.")


# 3. Preprocessing 

updated_categorical = categorical_features + ['bmi_category', 'bp_category']  # bmi_category is int, but encode if needed

label_encoders = {}
for col in updated_categorical:
    if col in train.columns and train[col].dtype == 'object':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        label_encoders[col] = le

# Prepare features (include new ones)
feature_columns = [col for col in train.columns if col not in ['id', 'diagnosed_diabetes']]
X = train[feature_columns]
y = train['diagnosed_diabetes']
X_test = test[feature_columns]

# Scale (include new numerical)
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("Features prepared. X shape:", X.shape)


# 4. Cross-Validation Setup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Define CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline CV for XGBoost
xgb_scores = []
for train_idx, val_idx in cv.split(X, y):
    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        scale_pos_weight=1 / imbalance_ratio if imbalance_ratio < 0.3 else 1  # Handle imbalance
    )
    model.fit(X_tr, y_tr)
    val_probs = model.predict_proba(X_vl)[:, 1]
    score = roc_auc_score(y_vl, val_probs)
    xgb_scores.append(score)

print(f"XGBoost CV AUC: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores)*2:.4f})")


# Train final XGBoost model on full data using the same parameters
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=1 / imbalance_ratio if imbalance_ratio < 0.3 else 1  # Handle imbalance
)

final_model.fit(X, y)

# Predict probabilities on test set
test_probs = final_model.predict_proba(X_test)[:, 1]

# Prepare submission
submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_probs
})

# Display first few rows
print(submission.head())

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv' using XGBoost baseline model!")




