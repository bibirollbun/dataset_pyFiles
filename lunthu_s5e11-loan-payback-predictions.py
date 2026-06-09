# Loan Payback Prediction - Kaggle Competition
# Playground Series S5E11

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain columns: {list(train_df.columns)}")


# Basic info
print("\n--- Data Info ---")
print(train_df.info())

print("\n--- Statistical Summary ---")
print(train_df.describe())

# Check for missing values
print("\n--- Missing Values ---")
print("Train missing values:")
print(train_df.isnull().sum())
print("\nTest missing values:")
print(test_df.isnull().sum())


# Target variable distribution
print("\n--- Target Variable Distribution ---")
print(train_df['loan_paid_back'].value_counts())
print(f"Percentage of loans paid back: {train_df['loan_paid_back'].mean() * 100:.2f}%")

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_df['loan_paid_back'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Loan Payback Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Loan Paid Back')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Not Paid (0)', 'Paid (1)'], rotation=0)

train_df['loan_paid_back'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                                colors=['#e74c3c', '#2ecc71'])
axes[1].set_title('Loan Payback Percentage', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Categorical features analysis
print("\n--- Categorical Features ---")
categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 
                   'loan_purpose', 'grade_subgrade']

for col in categorical_cols:
    if col in train_df.columns:
        print(f"\n{col}:")
        print(train_df[col].value_counts())

# Numerical features distribution
print("\n--- Numerical Features ---")
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                 'loan_amount', 'interest_rate']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    train_df[col].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# Correlation analysis
print("\n--- Correlation with Target ---")
numerical_data = train_df[numerical_cols + ['loan_paid_back']]
correlation = numerical_data.corr()['loan_paid_back'].sort_values(ascending=False)
print(correlation)

plt.figure(figsize=(10, 6))
correlation.drop('loan_paid_back').plot(kind='barh', color='steelblue')
plt.title('Feature Correlation with Loan Payback', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()


# Combine train and test for consistent preprocessing
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

# Store target variable and id
if 'loan_paid_back' in train_df.columns:
    y_train = train_df['loan_paid_back'].copy()
    train_df_temp = train_df.drop('loan_paid_back', axis=1)
else:
    train_df_temp = train_df.copy()

# Store test IDs if they exist
test_ids = test_df['id'] if 'id' in test_df.columns else None

# Combine datasets
combined_df = pd.concat([train_df_temp, test_df], axis=0, ignore_index=True)

print(f"Combined shape: {combined_df.shape}")

# Drop ID column for modeling
if 'id' in combined_df.columns:
    combined_df = combined_df.drop('id', axis=1)

# Handle missing values (if any)
for col in combined_df.columns:
    if combined_df[col].isnull().sum() > 0:
        if combined_df[col].dtype in ['float64', 'int64']:
            combined_df[col].fillna(combined_df[col].median(), inplace=True)
        else:
            combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col in combined_df.columns:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")

# Create additional features
print("\n--- Creating Additional Features ---")

# Loan to income ratio
combined_df['loan_to_income_ratio'] = combined_df['loan_amount'] / (combined_df['annual_income'] + 1)

# Total debt estimation
combined_df['estimated_total_debt'] = combined_df['annual_income'] * combined_df['debt_to_income_ratio']

# Monthly payment estimation (assuming 5-year term)
combined_df['estimated_monthly_payment'] = (combined_df['loan_amount'] * 
                                           (1 + combined_df['interest_rate'] * 5 / 100)) / 60

# Payment to income ratio
combined_df['payment_to_income_ratio'] = (combined_df['estimated_monthly_payment'] * 12) / (combined_df['annual_income'] + 1)

# Credit score bins
combined_df['credit_score_bin'] = pd.cut(combined_df['credit_score'], 
                                         bins=[0, 580, 670, 740, 800, 850],
                                         labels=[0, 1, 2, 3, 4])
combined_df['credit_score_bin'] = combined_df['credit_score_bin'].astype(int)

# Interest rate bins
combined_df['interest_rate_bin'] = pd.cut(combined_df['interest_rate'], 
                                          bins=5, labels=[0, 1, 2, 3, 4])
combined_df['interest_rate_bin'] = combined_df['interest_rate_bin'].astype(int)

print(f"Total features after engineering: {combined_df.shape[1] - 1}")  # -1 for dataset column

# Split back into train and test
train_processed = combined_df[combined_df['dataset'] == 'train'].drop('dataset', axis=1)
test_processed = combined_df[combined_df['dataset'] == 'test'].drop('dataset', axis=1)

print(f"\nProcessed train shape: {train_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")

# Scale numerical features
scaler = StandardScaler()
feature_cols = train_processed.columns.tolist()

train_scaled = train_processed.copy()
test_scaled = test_processed.copy()

train_scaled[feature_cols] = scaler.fit_transform(train_processed[feature_cols])
test_scaled[feature_cols] = scaler.transform(test_processed[feature_cols])


# Split data for validation
X_train, X_val, y_train_split, y_val = train_test_split(
    train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                           min_samples_split=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                    max_depth=5, random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Train model
    model.fit(X_train, y_train_split)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Probabilities for ROC-AUC
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    
    # Evaluate
    train_acc = accuracy_score(y_train_split, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    train_auc = roc_auc_score(y_train_split, y_pred_proba_train)
    val_auc = roc_auc_score(y_val, y_pred_proba_val)
    
    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_auc': train_auc,
        'val_auc': val_auc
    }
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred_val)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Classification report
    print(f"\nClassification Report:\n{classification_report(y_val, y_pred_val)}")


comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_acc'] for m in results.keys()],
    'Val Accuracy': [results[m]['val_acc'] for m in results.keys()],
    'Train ROC-AUC': [results[m]['train_auc'] for m in results.keys()],
    'Val ROC-AUC': [results[m]['val_auc'] for m in results.keys()]
})

print(comparison_df.to_string(index=False))

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

comparison_df.set_index('Model')[['Train Accuracy', 'Val Accuracy']].plot(kind='bar', ax=axes[0])
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[0].legend(['Train', 'Validation'])
axes[0].set_ylim([0.5, 1.0])

comparison_df.set_index('Model')[['Train ROC-AUC', 'Val ROC-AUC']].plot(kind='bar', ax=axes[1])
axes[1].set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('ROC-AUC Score')
axes[1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[1].legend(['Train', 'Validation'])
axes[1].set_ylim([0.5, 1.0])

plt.tight_layout()
plt.show()


# Select best model based on validation ROC-AUC
best_model_name = max(results.keys(), key=lambda x: results[x]['val_auc'])
best_model = results[best_model_name]['model']

print(f"\n*** Best Model: {best_model_name} ***")
print(f"Validation ROC-AUC: {results[best_model_name]['val_auc']:.4f}")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print("\n--- Top 15 Feature Importances ---")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    print(feature_importance.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# Train final model on full training data
print(f"Training final {best_model_name} on full training data...")
final_model = models[best_model_name]
final_model.fit(train_scaled, y_train)

# Generate predictions
test_predictions = final_model.predict(test_scaled)
test_predictions_proba = final_model.predict_proba(test_scaled)[:, 1]

print(f"Generated {len(test_predictions)} predictions")
print(f"Predicted distribution:")
print(pd.Series(test_predictions).value_counts())

# Create submission file
submission_df = pd.DataFrame({
    'id': test_ids if test_ids is not None else range(len(test_predictions)),
    'loan_paid_back': test_predictions
})


print(f"Best Model: {best_model_name}")
print(f"Validation Accuracy: {results[best_model_name]['val_acc']:.4f}")
print(f"Validation ROC-AUC: {results[best_model_name]['val_auc']:.4f}")


submission_df.to_csv('submission.csv', index=False)

