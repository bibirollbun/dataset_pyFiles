# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
np.random.seed(42)

print("Libraries loaded successfully!")


# Load datasets
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
except FileNotFoundError:
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# First look at the data
print("\n=== Training Data Preview ===")
display(train_df.head())

print("\n=== Test Data Preview ===")
display(test_df.head())


# Data info
print("\n=== Training Data Info ===")
print(train_df.info())

print("\n=== Basic Statistics ===")
display(train_df.describe())


# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
target_counts = train_df['loan_paid_back'].value_counts()
axes[0].bar(['Not Paid (0)', 'Paid (1)'], target_counts.values, color=['#e74c3c', '#2ecc71'])
axes[0].set_ylabel('Count')
axes[0].set_title('Loan Payback Distribution')
axes[0].set_xlabel('Class')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 1000, str(v), ha='center', fontweight='bold')

# Pie chart
colors = ['#e74c3c', '#2ecc71']
axes[1].pie(target_counts.values, labels=['Not Paid (0)', 'Paid (1)'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Loan Payback Proportion')

plt.tight_layout()
plt.show()

print(f"\nClass Distribution:")
print(f"Not Paid (0): {target_counts[0]} ({target_counts[0]/len(train_df)*100:.2f}%)")
print(f"Paid (1): {target_counts[1]} ({target_counts[1]/len(train_df)*100:.2f}%)")
print(f"\nClass imbalance ratio: {target_counts[0]/target_counts[1]:.2f}")


# Check for missing values
def check_missing(df, name):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Percentage': missing_pct})
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\n=== Missing Values in {name} ===")
        display(missing_df)
    else:
        print(f"\n=== No Missing Values in {name} ===")
    
    return missing_df

train_missing = check_missing(train_df, "Training Set")
test_missing = check_missing(test_df, "Test Set")


# Identify numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove('id')  # Remove ID
if 'loan_paid_back' in numerical_cols:
    numerical_cols.remove('loan_paid_back')  # Remove target

categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")


# Distribution of numerical features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(train_df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].axvline(train_df[col].mean(), color='red', linestyle='--', label='Mean')
    axes[idx].axvline(train_df[col].median(), color='green', linestyle='--', label='Median')
    axes[idx].legend()

plt.tight_layout()
plt.show()


# Numerical features vs Target
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    train_df.boxplot(column=col, by='loan_paid_back', ax=axes[idx])
    axes[idx].set_title(f'{col} by Loan Payback Status')
    axes[idx].set_xlabel('Loan Paid Back')
    axes[idx].set_ylabel(col)

plt.suptitle('')  # Remove the automatic title
plt.tight_layout()
plt.show()


# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = train_df[numerical_cols + ['loan_paid_back']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Show strongest correlations with target
target_corr = correlation_matrix['loan_paid_back'].abs().sort_values(ascending=False)
print("\n=== Correlations with Target Variable ===")
print(target_corr[1:])  # Exclude self-correlation


# Categorical features distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    value_counts = train_df[col].value_counts()
    axes[idx].bar(range(len(value_counts)), value_counts.values)
    axes[idx].set_xticks(range(len(value_counts)))
    axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_ylabel('Count')

plt.tight_layout()
plt.show()


# Categorical features vs Target
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    cross_tab = pd.crosstab(train_df[col], train_df['loan_paid_back'], normalize='index') * 100
    cross_tab.plot(kind='bar', ax=axes[idx], stacked=False, 
                   color=['#e74c3c', '#2ecc71'], alpha=0.8)
    axes[idx].set_title(f'{col} vs Loan Payback Rate')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Percentage')
    axes[idx].legend(['Not Paid (0)', 'Paid (1)'])
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Create copies for processing
train_processed = train_df.copy()
test_processed = test_df.copy()

# If there are missing values, handle them
# For numerical: use median
# For categorical: use mode

for col in numerical_cols:
    if train_processed[col].isnull().sum() > 0:
        median_val = train_processed[col].median()
        train_processed[col].fillna(median_val, inplace=True)
        test_processed[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in {col} with median: {median_val}")

for col in categorical_cols:
    if train_processed[col].isnull().sum() > 0:
        mode_val = train_processed[col].mode()[0]
        train_processed[col].fillna(mode_val, inplace=True)
        test_processed[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in {col} with mode: {mode_val}")

print("\nMissing values handled successfully!")


def feature_engineering(df):
    """
    Create meaningful features from existing ones
    """
    df = df.copy()
    
    # 1. Income to Loan Ratio - measures affordability
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    
    # 2. Credit Utilization Score
    df['credit_utilization'] = df['debt_to_income_ratio'] * (800 - df['credit_score'])
    
    # 3. Loan Burden - estimated monthly payment as % of monthly income
    # Assuming 36-month term (3 years)
    monthly_income = df['annual_income'] / 12
    monthly_payment = (df['loan_amount'] * (1 + df['interest_rate']/100)) / 36
    df['loan_burden'] = (monthly_payment / (monthly_income + 1)) * 100
    
    # 4. Risk Score - composite indicator
    df['risk_score'] = (
        (df['debt_to_income_ratio'] * 100) + 
        ((800 - df['credit_score']) / 8) + 
        (df['interest_rate'])
    )
    
    # 5. Credit Score Tiers
    df['credit_tier'] = pd.cut(df['credit_score'], 
                                bins=[0, 580, 670, 740, 800, 850],
                                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    # 6. Loan Amount Category
    df['loan_size'] = pd.cut(df['loan_amount'],
                              bins=[0, 5000, 10000, 20000, 50000],
                              labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    # 7. Income Category
    df['income_category'] = pd.cut(df['annual_income'],
                                    bins=[0, 30000, 50000, 75000, 100000, 200000],
                                    labels=['Low', 'Medium', 'High', 'Very High', 'Exceptional'])
    
    # 8. High Risk Indicator (interaction)
    df['high_risk'] = ((df['debt_to_income_ratio'] > 0.4) & 
                       (df['credit_score'] < 650)).astype(int)
    
    return df

# Apply feature engineering
train_processed = feature_engineering(train_processed)
test_processed = feature_engineering(test_processed)

print("Feature engineering completed!")
print(f"\nNew shape - Train: {train_processed.shape}, Test: {test_processed.shape}")
print(f"\nNew features created: {[col for col in train_processed.columns if col not in train_df.columns]}")


# Update categorical columns list with new features
new_categorical_cols = ['credit_tier', 'loan_size', 'income_category']
all_categorical_cols = categorical_cols + new_categorical_cols

# Label Encoding for categorical variables
label_encoders = {}

for col in all_categorical_cols:
    le = LabelEncoder()
    
    # Fit on combined data to ensure consistent encoding
    combined = pd.concat([train_processed[col].astype(str), 
                         test_processed[col].astype(str)])
    le.fit(combined)
    
    train_processed[col + '_encoded'] = le.transform(train_processed[col].astype(str))
    test_processed[col + '_encoded'] = le.transform(test_processed[col].astype(str))
    
    label_encoders[col] = le

print("Categorical encoding completed!")


# Select features for modeling
# Numerical features (original + engineered)
feature_cols = numerical_cols + [
    'income_to_loan_ratio', 'credit_utilization', 'loan_burden', 
    'risk_score', 'high_risk'
]

# Add encoded categorical features
encoded_cols = [col + '_encoded' for col in all_categorical_cols]
feature_cols += encoded_cols

# Prepare X and y
X = train_processed[feature_cols]
y = train_processed['loan_paid_back']
X_test = test_processed[feature_cols]

print(f"Final feature set: {len(feature_cols)} features")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")


# Standardize features for models that benefit from scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for convenience
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

print("Feature scaling completed!")


# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"\nClass distribution in training set:")
print(pd.Series(y_train).value_counts())
print(f"\nClass distribution in validation set:")
print(pd.Series(y_val).value_counts())


# Dictionary to store models and results
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
}

results = {}

print("Training models...\n")
print("="*80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print("-" * 80)

print("\nAll models trained successfully!")


# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
}).sort_values('ROC-AUC', ascending=False)

print("\n=== Model Performance Comparison ===")
display(comparison_df)

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
axes[0].barh(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xlim([0.5, 1.0])
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0].text(v + 0.005, i, f'{v:.4f}', va='center')

# ROC-AUC comparison
axes[1].barh(comparison_df['Model'], comparison_df['ROC-AUC'], color='lightcoral')
axes[1].set_xlabel('ROC-AUC Score')
axes[1].set_title('Model ROC-AUC Comparison')
axes[1].set_xlim([0.5, 1.0])
for i, v in enumerate(comparison_df['ROC-AUC']):
    axes[1].text(v + 0.005, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.show()


# Select best model based on ROC-AUC
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"Best Model: {best_model_name}")
print(f"Validation Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Validation ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

# Detailed classification report
print("\n=== Classification Report ===")
print(classification_report(y_val, best_predictions, target_names=['Not Paid', 'Paid']))

# Confusion Matrix
cm = confusion_matrix(y_val, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=['Not Paid', 'Paid'],
            yticklabels=['Not Paid', 'Paid'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.show()


# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n=== Top 15 Most Important Features ===")
    display(feature_importance.head(15))
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print(f"\n{best_model_name} does not provide feature importances.")


# Perform cross-validation on best model
print(f"Performing 5-Fold Cross-Validation on {best_model_name}...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print("Cross-Validation ROC-AUC Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nMean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


print(f"Training final {best_model_name} model on full training data...\n")

# Train on all available training data
final_model = best_model.__class__(**best_model.get_params())
final_model.fit(X_scaled, y)

print("Final model training completed!")


# Generate predictions for test set
test_predictions = final_model.predict(X_test_scaled)
test_probabilities = final_model.predict_proba(X_test_scaled)[:, 1]

print(f"Test predictions generated: {len(test_predictions)} samples")
print(f"\nPrediction distribution:")
print(pd.Series(test_predictions).value_counts())
print(f"\nProportion of predicted paybacks: {test_predictions.mean():.2%}")


# Create submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': test_predictions
})

# Verify format matches sample submission
print("=== Submission File Preview ===")
display(submission.head(10))

print(f"\nSubmission shape: {submission.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")

