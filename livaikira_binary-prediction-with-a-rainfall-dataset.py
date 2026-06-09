# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# For visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


main_dir = '/kaggle/input/playground-series-s5e3'
train_csv = main_dir + '/train.csv'
test_csv = main_dir + '/test.csv'
sample_csv = main_dir + '/sample_submission.csv'


# Load the datasets
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sample_submission = pd.read_csv(sample_csv)

# Display basic information about the datasets
print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print("\nSample submission format:")
sample_submission.head()


# Explore the training data
train_df.head()


# Check data types and missing values
print("Data types:")
print(train_df.dtypes)

print("\nMissing values in training set:")
print(train_df.isnull().sum())

print("\nMissing values in test set:")
print(test_df.isnull().sum())


# Check the target distribution
target_counts = train_df['rainfall'].value_counts(normalize=True) * 100
print(f"Target distribution:\n{target_counts}")

plt.figure(figsize=(10, 6))
sns.countplot(x='rainfall', data=train_df)
plt.title('Distribution of Rainfall Target')
plt.ylabel('Count')
plt.show()


# Basic statistical analysis
train_df.describe()


# Visualize correlations between features
plt.figure(figsize=(16, 12))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Top correlated features with target
target_correlations = correlation_matrix['rainfall'].sort_values(ascending=False)
print("Top correlations with rainfall:")
print(target_correlations[1:11])  # Exclude self-correlation


# Data preprocessing function
def preprocess_data(df, is_train=True):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Separate features and target if training data
    if is_train:
        X = df_processed.drop(['id', 'rainfall'], axis=1)
        y = df_processed['rainfall']
    else:
        X = df_processed.drop(['id'], axis=1)
        y = None
    
    # Handle missing values with median imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    
    return X_scaled, y


# Preprocess the data
X_train, y_train = preprocess_data(train_df, is_train=True)
X_test, _ = preprocess_data(test_df, is_train=False)

# Split training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train
)

print(f"Training set shape: {X_train_split.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")


# Initialize models for evaluation
models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier()
}

# Train and evaluate models with cross-validation
cv_results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    cv_results[name] = cv_scores
    print(f"{name} CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# Visualize model comparison
cv_means = [scores.mean() for scores in cv_results.values()]
cv_stds = [scores.std() for scores in cv_results.values()]
model_names = list(cv_results.keys())

plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=10)
plt.title('Model Performance Comparison (ROC-AUC)')
plt.ylabel('ROC-AUC Score')
plt.ylim(0.7, 1.0)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005, f"{height:.4f}",
             ha='center', va='bottom')
plt.show()


# Select the best model based on CV results
best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"Best model: {best_model_name}")

# Fine-tune the best model with hyperparameter tuning
if best_model_name == 'Random Forest':
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'Gradient Boosting':
    model = GradientBoostingClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
elif best_model_name == 'XGBoost':
    model = XGBClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
else:  # LightGBM
    model = LGBMClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50, 70]
    }

# Random search for hyperparameter tuning
random_search = RandomizedSearchCV(
    model, param_grid, n_iter=10, cv=5,
    scoring='roc_auc', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

# Get the best model
best_model = random_search.best_estimator_


# Train the best model on the full training data
best_model.fit(X_train, y_train)

# Make predictions on validation data
val_preds_proba = best_model.predict_proba(X_val)[:, 1]
val_preds = (val_preds_proba >= 0.5).astype(int)

# Evaluate on validation set
print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_preds_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_preds))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, val_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.show()


# Make predictions on the test set
test_preds_proba = best_model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': test_preds_proba.round().astype(int)
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created!")
submission.head()

