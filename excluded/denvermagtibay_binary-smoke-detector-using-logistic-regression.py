import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Load datasets
train = pd.read_csv("/kaggle/input/binary-smoke-detector/train.csv", sep=",")
test = pd.read_csv("/kaggle/input/binary-smoke-detector/test.csv", sep=",")

# Quick overview
print(train.head())
print(test.head())


# Shape of datasets
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Data types and missing values
print("\nTrain Info:")
print(train.info())

print("\nTest Info:")
print(test.info())


# Missing values count and percentage
missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print("\nMissing values in Train set:\n", missing_train)

missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
print("\nMissing values in Test set:\n", missing_test)


# Describe numerical features
print("\nTrain Describe:")
print(train.describe())

print("\nTest Describe:")
print(test.describe())


# Identify numerical and categorical columns
numerical_features = train.drop(columns=['id', 'smoking']).select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train.drop(columns=['id', 'smoking']).select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical Features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")


import matplotlib.pyplot as plt

# Smoking target distribution
plt.figure(figsize=(5,3))
train['smoking'].value_counts(normalize=True).plot(kind='bar', title='Target distribution (smoking)')
plt.xlabel('Smoking (0=No, 1=Yes)')
plt.ylabel('Proportion')
plt.show()


# Correlation with the target
correlations = train[numerical_features + ['smoking']].corr()['smoking'].sort_values(ascending=False)
print("\nCorrelation with Target:\n", correlations)

# Visualize the correlation matrix
import seaborn as sns

plt.figure(figsize=(12,10))
corr_matrix = train[numerical_features].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()


# ----------------------------------------
# Feature Engineering - Step-by-Step
# ----------------------------------------

# 1. Create Ratios and Asymmetry Features

# BMI: weight (kg) / (height (m))Â²
train['BMI'] = train['weight(kg)'] / (train['height(cm)'] / 100) ** 2
test['BMI'] = test['weight(kg)'] / (test['height(cm)'] / 100) ** 2

# Waist-to-height ratio
train['waist_height_ratio'] = train['waist(cm)'] / train['height(cm)']
test['waist_height_ratio'] = test['waist(cm)'] / test['height(cm)']

# Eyesight asymmetry (left - right)
train['eyesight_diff'] = train['eyesight(left)'] - train['eyesight(right)']
test['eyesight_diff'] = test['eyesight(left)'] - test['eyesight(right)']

# Hearing asymmetry (left - right)
train['hearing_diff'] = train['hearing(left)'] - train['hearing(right)']
test['hearing_diff'] = test['hearing(left)'] - test['hearing(right)']

# ----------------------------------------

# 2. Bin / Group Continuous Variables

# 2.1 Binning Age into Age Groups
train['age_group'] = pd.cut(train['age'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4])
test['age_group'] = pd.cut(test['age'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4])

# 2.2 Binning Systolic Blood Pressure
train['bp_category'] = pd.cut(train['systolic'], bins=[0, 120, 130, 140, 180, 300], labels=[0, 1, 2, 3, 4])
test['bp_category'] = pd.cut(test['systolic'], bins=[0, 120, 130, 140, 180, 300], labels=[0, 1, 2, 3, 4])

# 2.3 Handle missing bins (add category -1 and fill)
train['age_group'] = train['age_group'].cat.add_categories(-1).fillna(-1).astype(int)
test['age_group'] = test['age_group'].cat.add_categories(-1).fillna(-1).astype(int)

train['bp_category'] = train['bp_category'].cat.add_categories(-1).fillna(-1).astype(int)
test['bp_category'] = test['bp_category'].cat.add_categories(-1).fillna(-1).astype(int)


# ----------------------------------------

# 3. Interaction Features (Multiplying Features)

# Age Ã— Gtp
train['age_Gtp'] = train['age'] * train['Gtp']
test['age_Gtp'] = test['age'] * test['Gtp']

# Age Ã— triglyceride
train['age_triglyceride'] = train['age'] * train['triglyceride']
test['age_triglyceride'] = test['age'] * test['triglyceride']

# waist(cm) Ã— fasting blood sugar
train['waist_fbs'] = train['waist(cm)'] * train['fasting blood sugar']
test['waist_fbs'] = test['waist(cm)'] * test['fasting blood sugar']

# ----------------------------------------

# 4. Polynomial Features

# (waist(cm))Â²
train['waist_squared'] = train['waist(cm)'] ** 2
test['waist_squared'] = test['waist(cm)'] ** 2

# (age Ã— LDL)
train['age_LDL'] = train['age'] * train['LDL']
test['age_LDL'] = test['age'] * test['LDL']



# Check new shape
print(f"New train shape: {train.shape}")
print(f"New test shape: {test.shape}")

# Check first few rows
print("\nTrain head:")
print(train.head())

print("\nTest head:")
print(test.head())


# Separate features and target
X = train.drop(columns=['id', 'smoking'])  # all features except id and target
y = train['smoking']

X_test = test.drop(columns=['id'])  # test set features


# Initialize scaler
scaler = StandardScaler()

# Fit only on training data
X_scaled = scaler.fit_transform(X)

# Transform test data too
X_test_scaled = scaler.transform(X_test)


# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create arrays to hold out-of-fold predictions and test predictions
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])

# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(cv.split(X_scaled, y)):
    X_train_fold, y_train_fold = X_scaled[train_idx], y.iloc[train_idx]
    X_valid_fold, y_valid_fold = X_scaled[valid_idx], y.iloc[valid_idx]
    
    # Model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_fold, y_train_fold)
    
    # Predict validation and test
    oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
    test_preds += model.predict_proba(X_test_scaled)[:, 1] / cv.n_splits
    
    # Fold AUC
    fold_auc = roc_auc_score(y_valid_fold, oof_preds[valid_idx])
    print(f'Fold {fold+1} AUC: {fold_auc:.5f}')

# Overall CV AUC
cv_score = roc_auc_score(y, oof_preds)
print(f'\nOverall CV AUC: {cv_score:.5f}')


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'smoking': test_preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print('Submission file created: submission.csv')

