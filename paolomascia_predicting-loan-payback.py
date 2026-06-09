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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# Load train and test datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# Display basic information
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nFirst rows of train data:")
print(train.head())
print("\nTrain data info:")
print(train.info())
print("\nTest data info:")
print(test.info())


# Check for missing values
print("Missing values in train:\n", train.isnull().sum())
print("\nMissing values in test:\n", test.isnull().sum())

# Descriptive statistics
print("\nDescriptive statistics:")
print(train.describe())

# Target variable distribution
print("\nTarget variable distribution:")
print(train['loan_paid_back'].value_counts())
print("\nTarget variable percentage:")
print(train['loan_paid_back'].value_counts(normalize=True))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Target distribution
axes[0, 0].bar(train['loan_paid_back'].value_counts().index, 
               train['loan_paid_back'].value_counts().values)
axes[0, 0].set_title('Target Distribution')
axes[0, 0].set_xlabel('Loan Paid Back')
axes[0, 0].set_ylabel('Count')

# Numerical features distributions
axes[0, 1].hist(train['credit_score'], bins=30, edgecolor='black')
axes[0, 1].set_title('Credit Score Distribution')
axes[0, 1].set_xlabel('Credit Score')

axes[0, 2].hist(train['loan_amount'], bins=30, edgecolor='black')
axes[0, 2].set_title('Loan Amount Distribution')
axes[0, 2].set_xlabel('Loan Amount')

axes[1, 0].hist(train['annual_income'], bins=30, edgecolor='black')
axes[1, 0].set_title('Annual Income Distribution')
axes[1, 0].set_xlabel('Annual Income')

axes[1, 1].hist(train['interest_rate'], bins=30, edgecolor='black')
axes[1, 1].set_title('Interest Rate Distribution')
axes[1, 1].set_xlabel('Interest Rate')

axes[1, 2].hist(train['debt_to_income_ratio'], bins=30, edgecolor='black')
axes[1, 2].set_title('Debt to Income Ratio Distribution')
axes[1, 2].set_xlabel('Debt to Income Ratio')

plt.tight_layout()
plt.show()

# Correlation with target for numerical features
numerical_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                      'loan_amount', 'interest_rate']
correlation_with_target = train[numerical_features + ['loan_paid_back']].corr()['loan_paid_back'].drop('loan_paid_back')
print("\nCorrelation with target:")
print(correlation_with_target.sort_values(ascending=False))


# Separate features and target in train set
X = train.drop(['loan_paid_back'], axis=1)
y = train['loan_paid_back']

# Save IDs for submission
train_ids = X['id']
test_ids = test['id']

# Remove ID column from features
X = X.drop('id', axis=1)
test_features = test.drop('id', axis=1)

# Identify categorical and numerical columns
categorical_cols = ['gender', 'marital_status', 'education_level', 
                    'employment_status', 'loan_purpose', 'grade_subgrade']
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                  'loan_amount', 'interest_rate']

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)
print("\nData preprocessing setup complete!")


# Fill missing values in numerical columns with median
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        test_features[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in {col} with median: {median_val:.2f}")

# Fill missing values in categorical columns with mode
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        test_features[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in {col} with mode: {mode_val}")

print("\nMissing values handled successfully!")
print("Remaining missing values in train:", X.isnull().sum().sum())
print("Remaining missing values in test:", test_features.isnull().sum().sum())


# Label Encoding for categorical variables
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Fit on all unique values from both train and test combined
    all_values = pd.concat([X[col], test_features[col]]).unique()
    le.fit(all_values)
    
    X[col] = le.transform(X[col])
    test_features[col] = le.transform(test_features[col])
    label_encoders[col] = le
    
    print(f"Encoded {col}: {len(le.classes_)} unique classes")

print("\nEncoding completed successfully!")


# Create new features
print("Creating new features...")

# Income to loan ratio
X['income_to_loan_ratio'] = X['annual_income'] / (X['loan_amount'] + 1)
test_features['income_to_loan_ratio'] = test_features['annual_income'] / (test_features['loan_amount'] + 1)

# Debt-loan interaction
X['debt_loan_interaction'] = X['debt_to_income_ratio'] * X['loan_amount']
test_features['debt_loan_interaction'] = test_features['debt_to_income_ratio'] * test_features['loan_amount']

# Credit score to interest rate ratio
X['credit_to_interest'] = X['credit_score'] / (X['interest_rate'] + 0.01)
test_features['credit_to_interest'] = test_features['credit_score'] / (test_features['interest_rate'] + 0.01)

# Loan to income percentage
X['loan_to_income_pct'] = (X['loan_amount'] / X['annual_income']) * 100
test_features['loan_to_income_pct'] = (test_features['loan_amount'] / test_features['annual_income']) * 100

print(f"Feature engineering completed! New feature count: {X.shape[1]}")
print("New features created:")
print("- income_to_loan_ratio")
print("- debt_loan_interaction")
print("- credit_to_interest")
print("- loan_to_income_pct")


# Standardize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_features)

# Convert back to DataFrame to maintain column names
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
test_scaled = pd.DataFrame(test_scaled, columns=test_features.columns)

print("Feature scaling completed!")
print(f"Scaled train features shape: {X_scaled.shape}")
print(f"Scaled test features shape: {test_scaled.shape}")


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"\nTraining target distribution:\n{y_train.value_counts()}")
print(f"\nValidation target distribution:\n{y_val.value_counts()}")


print("=" * 60)
print("BASELINE MODELS - WITHOUT HYPERPARAMETER TUNING")
print("=" * 60)

# Dictionary to store baseline results
baseline_results = {}

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_val)
lr_pred_proba = lr_model.predict_proba(X_val)[:, 1]
lr_acc = accuracy_score(y_val, lr_pred)
lr_auc = roc_auc_score(y_val, lr_pred_proba)
baseline_results['Logistic Regression'] = {'accuracy': lr_acc, 'auc': lr_auc}
print(f"   Accuracy: {lr_acc:.4f}")
print(f"   ROC-AUC: {lr_auc:.4f}")

# 2. Random Forest
print("\n2. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_pred_proba = rf_model.predict_proba(X_val)[:, 1]
rf_acc = accuracy_score(y_val, rf_pred)
rf_auc = roc_auc_score(y_val, rf_pred_proba)
baseline_results['Random Forest'] = {'accuracy': rf_acc, 'auc': rf_auc}
print(f"   Accuracy: {rf_acc:.4f}")
print(f"   ROC-AUC: {rf_auc:.4f}")

# 3. Gradient Boosting
print("\n3. Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_val)
gb_pred_proba = gb_model.predict_proba(X_val)[:, 1]
gb_acc = accuracy_score(y_val, gb_pred)
gb_auc = roc_auc_score(y_val, gb_pred_proba)
baseline_results['Gradient Boosting'] = {'accuracy': gb_acc, 'auc': gb_auc}
print(f"   Accuracy: {gb_acc:.4f}")
print(f"   ROC-AUC: {gb_auc:.4f}")

# Summary of baseline results
print("\n" + "=" * 60)
print("BASELINE RESULTS SUMMARY")
print("=" * 60)
baseline_df = pd.DataFrame(baseline_results).T
print(baseline_df)


print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# ===== RANDOM FOREST TUNING =====
print("\n1. Tuning Random Forest Classifier...")
print("   This may take several minutes...")

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Use RandomizedSearchCV for efficiency (faster than GridSearchCV)
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=50,  # Number of parameter settings sampled
    cv=3,  # 3-fold cross-validation
    scoring='roc_auc',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_random_search.fit(X_train, y_train)

print(f"\n   Best parameters: {rf_random_search.best_params_}")
print(f"   Best cross-validation score: {rf_random_search.best_score_:.4f}")

# Evaluate on validation set
rf_tuned_pred = rf_random_search.predict(X_val)
rf_tuned_pred_proba = rf_random_search.predict_proba(X_val)[:, 1]
rf_tuned_acc = accuracy_score(y_val, rf_tuned_pred)
rf_tuned_auc = roc_auc_score(y_val, rf_tuned_pred_proba)

print(f"   Validation Accuracy: {rf_tuned_acc:.4f}")
print(f"   Validation ROC-AUC: {rf_tuned_auc:.4f}")

# ===== GRADIENT BOOSTING TUNING =====
print("\n2. Tuning Gradient Boosting Classifier...")
print("   This may take several minutes...")

# Define parameter grid for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# Use RandomizedSearchCV
gb_random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=gb_param_grid,
    n_iter=50,
    cv=3,
    scoring='roc_auc',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

gb_random_search.fit(X_train, y_train)

print(f"\n   Best parameters: {gb_random_search.best_params_}")
print(f"   Best cross-validation score: {gb_random_search.best_score_:.4f}")

# Evaluate on validation set
gb_tuned_pred = gb_random_search.predict(X_val)
gb_tuned_pred_proba = gb_random_search.predict_proba(X_val)[:, 1]
gb_tuned_acc = accuracy_score(y_val, gb_tuned_pred)
gb_tuned_auc = roc_auc_score(y_val, gb_tuned_pred_proba)

print(f"   Validation Accuracy: {gb_tuned_acc:.4f}")
print(f"   Validation ROC-AUC: {gb_tuned_auc:.4f}")

# ===== LOGISTIC REGRESSION TUNING =====
print("\n3. Tuning Logistic Regression...")

# Define parameter grid for Logistic Regression
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000]
}

lr_grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=lr_param_grid,
    cv=3,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

lr_grid_search.fit(X_train, y_train)

print(f"\n   Best parameters: {lr_grid_search.best_params_}")
print(f"   Best cross-validation score: {lr_grid_search.best_score_:.4f}")

# Evaluate on validation set
lr_tuned_pred = lr_grid_search.predict(X_val)
lr_tuned_pred_proba = lr_grid_search.predict_proba(X_val)[:, 1]
lr_tuned_acc = accuracy_score(y_val, lr_tuned_pred)
lr_tuned_auc = roc_auc_score(y_val, lr_tuned_pred_proba)

print(f"   Validation Accuracy: {lr_tuned_acc:.4f}")
print(f"   Validation ROC-AUC: {lr_tuned_auc:.4f}")


print("\n" + "=" * 60)
print("MODEL COMPARISON: BASELINE vs TUNED")
print("=" * 60)

# Create comparison dataframe
comparison_results = {
    'Logistic Regression (Baseline)': {'accuracy': lr_acc, 'auc': lr_auc},
    'Logistic Regression (Tuned)': {'accuracy': lr_tuned_acc, 'auc': lr_tuned_auc},
    'Random Forest (Baseline)': {'accuracy': rf_acc, 'auc': rf_auc},
    'Random Forest (Tuned)': {'accuracy': rf_tuned_acc, 'auc': rf_tuned_auc},
    'Gradient Boosting (Baseline)': {'accuracy': gb_acc, 'auc': gb_auc},
    'Gradient Boosting (Tuned)': {'accuracy': gb_tuned_acc, 'auc': gb_tuned_auc}
}

comparison_df = pd.DataFrame(comparison_results).T
comparison_df = comparison_df.sort_values('auc', ascending=False)
print(comparison_df)

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

comparison_df['accuracy'].plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xlabel('Accuracy')

comparison_df['auc'].plot(kind='barh', ax=axes[1], color='salmon')
axes[1].set_title('Model ROC-AUC Comparison')
axes[1].set_xlabel('ROC-AUC Score')

plt.tight_layout()
plt.show()

# Select best model based on ROC-AUC
best_model_name = comparison_df['auc'].idxmax()
print(f"\nBest Model: {best_model_name}")
print(f"   Accuracy: {comparison_df.loc[best_model_name, 'accuracy']:.4f}")
print(f"   ROC-AUC: {comparison_df.loc[best_model_name, 'auc']:.4f}")

# Assign best model
if 'Random Forest (Tuned)' in best_model_name:
    best_model = rf_random_search.best_estimator_
elif 'Gradient Boosting (Tuned)' in best_model_name:
    best_model = gb_random_search.best_estimator_
else:
    best_model = lr_grid_search.best_estimator_


print("\n" + "=" * 60)
print("DETAILED EVALUATION OF BEST MODEL")
print("=" * 60)

# Get predictions from best model
best_pred = best_model.predict(X_val)
best_pred_proba = best_model.predict_proba(X_val)[:, 1]

# Classification report
print("\nClassification Report:")
print(classification_report(y_val, best_pred, target_names=['Not Paid', 'Paid']))

# Confusion matrix
cm = confusion_matrix(y_val, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Paid', 'Paid'],
            yticklabels=['Not Paid', 'Paid'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    feature_importance.head(15).plot(x='feature', y='importance', kind='barh')
    plt.title('Top 15 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


print("\n" + "=" * 60)
print("TRAINING FINAL MODEL ON COMPLETE DATASET")
print("=" * 60)

# Retrain the best model on the entire training dataset
print(f"Retraining {best_model_name} on full training data...")
final_model = best_model
final_model.fit(X_scaled, y)

print("Final model trained successfully on complete training set!")
print(f"   Training samples: {X_scaled.shape[0]}")
print(f"   Features: {X_scaled.shape[1]}")


print("\n" + "=" * 60)
print("GENERATING PREDICTIONS FOR TEST SET")
print("=" * 60)

# Make predictions on test set
test_predictions = final_model.predict(test_scaled)
test_predictions_proba = final_model.predict_proba(test_scaled)[:, 1]

print(f"Predictions generated: {len(test_predictions)}")
print(f"\nPrediction distribution:")
print(pd.Series(test_predictions).value_counts())
print(f"\nPrediction percentages:")
print(pd.Series(test_predictions).value_counts(normalize=True))


print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid': test_predictions
})

# Verify format
print("\nFirst 10 rows of submission:")
print(submission.head(10))
print(f"\nSubmission shape: {submission.shape}")
print(f"Columns: {submission.columns.tolist()}")
print(f"Data types:\n{submission.dtypes}")

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("\nsubmission.csv file created successfully!")

