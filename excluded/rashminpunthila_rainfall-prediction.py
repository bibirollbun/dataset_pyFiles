# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load the datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


# Basic dataset exploration
train.head()


# Check for missing values
print("Missing values in training set:")
print(train.isna().sum())
print("\nMissing values in test set:")
print(test.isna().sum())


# Display class distribution
print("Class distribution:")
print(train['rainfall'].value_counts())

# Visualize class imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x='rainfall', data=train)
plt.title('Class Distribution - Rainfall')
plt.show()


def preprocess_data(df):
    """
    Performs preprocessing and feature engineering on the dataset
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    for col in processed_df.columns:
        if processed_df[col].isnull().sum() > 0:
            if processed_df[col].dtype in ['float64', 'int64']:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            else:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    # Feature engineering - add cyclical encoding for winddirection
    if 'winddirection' in processed_df.columns:
        processed_df['winddir_x'] = np.cos(processed_df['winddirection'] * 2 * np.pi / 360)
        processed_df['winddir_y'] = np.sin(processed_df['winddirection'] * 2 * np.pi / 360)
    
    # Create interaction features
    if 'humidity' in processed_df.columns and 'temperature' in processed_df.columns:
        processed_df['humidity_temp'] = processed_df['humidity'] * processed_df['temperature']
    
    if 'windspeed' in processed_df.columns and 'temperature' in processed_df.columns:
        # Wind chill formula
        processed_df['windchill'] = 13.12 + 0.6215 * processed_df['temperature'] - 11.37 * (processed_df['windspeed']**0.16) + 0.3965 * processed_df['temperature'] * (processed_df['windspeed']**0.16)
    
    return processed_df


# Apply preprocessing
train_processed = preprocess_data(train)
test_processed = preprocess_data(test)

# Verify new features
print("New features added:")
print(train_processed.columns.difference(train.columns))


# Prepare features and target
features = train_processed.drop(columns=["rainfall", "id"])
target = train_processed['rainfall']
test_features = test_processed.drop(columns=["id"])


# Split data with stratification to maintain class balance in validation set
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.25, random_state=42, stratify=target)


# Handle class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original training class distribution: {np.bincount(y_train)}")
print(f"Resampled training class distribution: {np.bincount(y_train_resampled)}")

# Visualize the effect of resampling
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train)
plt.title('Original Class Distribution')
plt.subplot(1, 2, 2)
sns.countplot(x=y_train_resampled)
plt.title('Resampled Class Distribution')
plt.tight_layout()
plt.show()


# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_features)


# Import models and metrics
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Initialize XGBoost model with GPU support
xgb_model = XGBClassifier(
    eval_metric='auc', 
    use_label_encoder=False, 
    random_state=42,
    tree_method='gpu_hist'  # Enables GPU acceleration
)

# Parameter grid for XGBoost - focused on handling class imbalance
xgb_param_grid = {
    'n_estimators': [150, 200, 250],
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [5, 7, 9],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'subsample': [0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 3, 5],  # Important for imbalanced classes
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# Grid search for hyperparameter tuning (GPU-enabled model)
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=xgb_param_grid, 
    scoring='roc_auc', 
    cv=3, 
    verbose=2,
    n_jobs=-1  # Use all CPU cores for parallel processing
)




# Grid search with stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=xgb_param_grid, 
    scoring='roc_auc', 
    cv=cv, 
    verbose=1, 
    n_jobs=-1
)

# Fit the grid search
print("Training XGBoost model with grid search...")
grid_search.fit(X_train_scaled, y_train_resampled)


# Get best parameters and model
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best ROC-AUC Score from Grid Search: {best_score}")


# Train best model from grid search
best_xgb_model = grid_search.best_estimator_

# Cross-validation of best model
cv_scores = cross_val_score(best_xgb_model, X_train_scaled, y_train_resampled, cv=cv, scoring='roc_auc')
print(f"Cross-Validation ROC-AUC Scores: {cv_scores}")
print(f"Mean Cross-Validation ROC-AUC Score: {np.mean(cv_scores):.4f}")


# Evaluate on validation set
y_val_pred_proba = best_xgb_model.predict_proba(X_val_scaled)[:, 1]
y_val_pred = best_xgb_model.predict(X_val_scaled)

# Calculate multiple metrics
auc_score = roc_auc_score(y_val, y_val_pred_proba)
avg_precision = average_precision_score(y_val, y_val_pred_proba)
f1 = f1_score(y_val, y_val_pred)

print(f"Validation ROC-AUC Score: {auc_score:.4f}")
print(f"Validation Average Precision Score: {avg_precision:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))


# Visualize confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)  # adding small epsilon to avoid division by zero
optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_idx]

print(f"Optimal threshold to maximize F1: {optimal_threshold:.4f}")

# Plot precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.', label=f'XGBoost (AP={avg_precision:.2f})')
plt.axvline(x=recall[optimal_threshold_idx], color='r', linestyle='--', 
            label=f'Optimal threshold: {optimal_threshold:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()


# Train ensemble model
print("\nTraining ensemble model...")
# Adjust hyperparameters for each model based on your grid search
best_xgb = XGBClassifier(**best_params, random_state=42)
best_lgb = lgb.LGBMClassifier(
    n_estimators=200, 
    learning_rate=0.02, 
    max_depth=7,
    random_state=42
)
best_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgb', best_lgb),
        ('rf', best_rf)
    ],
    voting='soft'  # Use probabilities for voting
)

# Train ensemble
voting_clf.fit(X_train_scaled, y_train_resampled)


# Evaluate ensemble
ensemble_val_pred_proba = voting_clf.predict_proba(X_val_scaled)[:, 1]
# Apply the optimal threshold found earlier
ensemble_val_pred = (ensemble_val_pred_proba >= optimal_threshold).astype(int)

ensemble_auc = roc_auc_score(y_val, ensemble_val_pred_proba)
ensemble_avg_precision = average_precision_score(y_val, ensemble_val_pred_proba)
ensemble_f1 = f1_score(y_val, ensemble_val_pred)

print("\nEnsemble Model Metrics:")
print(f"Ensemble Validation ROC-AUC Score: {ensemble_auc:.4f}")
print(f"Ensemble Validation Average Precision Score: {ensemble_avg_precision:.4f}")
print(f"Ensemble Validation F1 Score: {ensemble_f1:.4f}")
print("\nEnsemble Classification Report:")
print(classification_report(y_val, ensemble_val_pred))


# Compare models with ROC curve
from sklearn.metrics import roc_curve

# Get ROC curve data for XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_val_pred_proba)

# Get ROC curve data for Ensemble
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_val, ensemble_val_pred_proba)

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_score:.3f})')
plt.plot(fpr_ensemble, tpr_ensemble, label=f'Ensemble (AUC = {ensemble_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()


# Choose final model based on performance
if ensemble_auc > auc_score:
    print("\nUsing ensemble model for final predictions")
    final_model = voting_clf
    y_test_pred_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_pred_proba >= optimal_threshold).astype(int)
else:
    print("\nUsing XGBoost model for final predictions")
    final_model = best_xgb_model
    final_model.fit(X_train_scaled, y_train_resampled)  # Refit on full training data
    y_test_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_pred_proba >= optimal_threshold).astype(int)


# Create submission file
submission = pd.DataFrame({'id': test['id'], 'rainfall': y_test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Predictions saved to submission.csv")


# Feature importance analysis
if hasattr(final_model, 'feature_importances_'):
    # For single models like XGBoost or Random Forest
    importances = final_model.feature_importances_
    feature_names = features.columns
elif hasattr(final_model, 'estimators_'):
    # For ensemble with named estimators
    # Use first model's importances (XGBoost in this case)
    importances = final_model.estimators_[0].feature_importances_  
    feature_names = features.columns
else:
    importances = None
    
if importances is not None:
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.show()

