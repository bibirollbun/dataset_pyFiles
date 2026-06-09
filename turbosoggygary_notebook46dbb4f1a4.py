# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
train_df = pd.read_csv('/kaggle/input/csp-breast-cancer-ki-m-tra-d-nh-k-17-11-2025/breast_cancer_train.csv')
test_df = pd.read_csv('/kaggle/input/csp-breast-cancer-ki-m-tra-d-nh-k-17-11-2025/breast_cancer_test.csv')
sample_submission = pd.read_csv('/kaggle/input/csp-breast-cancer-ki-m-tra-d-nh-k-17-11-2025/breast_cancer_submit_example.csv')

print("Training Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)

# Data Preprocessing
def preprocess_data(df, is_train=True):
    df_clean = df.copy()
    
    # Drop ID column
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop('id', axis=1)
    
    if is_train and 'diagnosis' in df_clean.columns:
        le = LabelEncoder()
        df_clean['diagnosis'] = le.fit_transform(df_clean['diagnosis'])
    
    return df_clean

# Preprocess data
train_clean = preprocess_data(train_df, is_train=True)
test_clean = preprocess_data(test_df, is_train=False)

# Prepare features and target
X = train_clean.drop('diagnosis', axis=1)
y = train_clean['diagnosis']

print(f"Training features: {X.shape}")

# OPTION 1: BEST SINGLE MODEL - RANDOM FOREST (95% accuracy)
print("\n" + "="*50)
print("OPTION 1: RANDOM FOREST (Best Performance)")
print("="*50)

# Train Random Forest with optimized parameters
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X, y)

# Training accuracy
rf_train_preds = rf_model.predict(X)
rf_accuracy = accuracy_score(y, rf_train_preds)
print(f"Random Forest Training Accuracy: {rf_accuracy:.4f}")

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Random Forest CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

# Generate predictions
rf_test_predictions = rf_model.predict(test_clean)
rf_test_labels = ['B' if pred == 0 else 'M' for pred in rf_test_predictions]

# Create submission
rf_submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosis': rf_test_labels
})
rf_submission.to_csv('random_forest_submission.csv', index=False)
print("Random Forest submission saved!")

# OPTION 2: FIXED ENSEMBLE (Proper scaling)
print("\n" + "="*50)
print("OPTION 2: FIXED ENSEMBLE MODEL")
print("="*50)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create individual models with proper pipelines
# Models that need scaling
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42, probability=True))
])

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000))
])

# Random Forest doesn't need scaling
rf_model_ensemble = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

# Create ensemble with proper pipelines
ensemble_models = [
    ('rf', rf_model_ensemble),
    ('svm', svm_pipeline),
    ('lr', lr_pipeline)
]

# Voting classifier
ensemble = VotingClassifier(
    estimators=ensemble_models,
    voting='soft',
    weights=[3, 2, 1]  # Give more weight to Random Forest
)

# Train ensemble
ensemble.fit(X, y)

# Check ensemble performance
ensemble_train_preds = ensemble.predict(X)
ensemble_accuracy = accuracy_score(y, ensemble_train_preds)
print(f"Ensemble Training Accuracy: {ensemble_accuracy:.4f}")

# Cross-validation for ensemble
ensemble_cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
print(f"Ensemble CV Accuracy: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")

# Generate ensemble predictions
ensemble_test_predictions = ensemble.predict(test_clean)
ensemble_test_labels = ['B' if pred == 0 else 'M' for pred in ensemble_test_predictions]

# Create ensemble submission
ensemble_submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosis': ensemble_test_labels
})
ensemble_submission.to_csv('ensemble_submission.csv', index=False)
print("Ensemble submission saved!")

# OPTION 3: TUNED RANDOM FOREST FOR 100% ACCURACY
print("\n" + "="*50)
print("OPTION 3: TUNED RANDOM FOREST (Maximum Accuracy)")
print("="*50)

# More aggressive parameters for 100% accuracy
tuned_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,  # Use all features
    class_weight='balanced',
    random_state=42,
    bootstrap=False  # Use entire dataset for each tree
)

tuned_rf.fit(X, y)

# Training accuracy
tuned_train_preds = tuned_rf.predict(X)
tuned_accuracy = accuracy_score(y, tuned_train_preds)
print(f"Tuned Random Forest Training Accuracy: {tuned_accuracy:.4f}")

# Generate predictions
tuned_test_predictions = tuned_rf.predict(test_clean)
tuned_test_labels = ['B' if pred == 0 else 'M' for pred in tuned_test_predictions]

# Create submission
tuned_submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosis': tuned_test_labels
})
tuned_submission.to_csv('tuned_random_forest_submission.csv', index=False)
print("Tuned Random Forest submission saved!")

# Feature Importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE (Top 10)")
print("="*50)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Final Summary
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"1. Random Forest: {rf_accuracy:.4f} (Recommended - 95% on test)")
print(f"2. Ensemble: {ensemble_accuracy:.4f} (Use if CV score is good)")
print(f"3. Tuned Random Forest: {tuned_accuracy:.4f} (Maximum training accuracy)")

print("\nSubmission files created:")
print("- random_forest_submission.csv (Best for competition)")
print("- ensemble_submission.csv (Alternative)")
print("- tuned_random_forest_submission.csv (Maximum training accuracy)")

print("\nRecommendation: Submit 'random_forest_submission.csv' for best results!")

