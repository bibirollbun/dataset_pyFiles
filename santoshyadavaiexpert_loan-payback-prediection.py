# Loan Repayment Prediction - Kaggle Playground Series November 2025
# Goal: Predict the probability that a borrower will pay back their loan
# Evaluation Metric: Area Under ROC Curve (AUC-ROC)

# =============================================================================
# STEP 1: Import Required Libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("Libraries imported successfully!")

# =============================================================================
# STEP 2: Load the Data
# =============================================================================
# Load training and test datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print("\n" + "="*80)
print("DATA LOADING COMPLETE")
print("="*80)
print(f"\nTraining set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"\nTraining columns: {train_df.columns.tolist()}")

# Display first few rows
print("\nFirst few rows of training data:")
print(train_df.head())

# =============================================================================
# STEP 3: Exploratory Data Analysis (EDA)
# =============================================================================
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Check for missing values
print("\nMissing values in training set:")
print(train_df.isnull().sum())

print("\nMissing values in test set:")
print(test_df.isnull().sum())

# Data types
print("\nData types:")
print(train_df.dtypes)

# Statistical summary
print("\nStatistical summary of numerical features:")
print(train_df.describe())

# Target variable distribution
print("\nTarget variable distribution:")
print(train_df['loan_paid_back'].value_counts())
print(f"\nTarget balance: {train_df['loan_paid_back'].value_counts(normalize=True)}")

# Visualize target distribution
plt.figure(figsize=(8, 5))
train_df['loan_paid_back'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Loan Repayment Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Loan Paid Back (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 4: Feature Engineering
# =============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

def engineer_features(df):
    """
    Create new features from existing ones to improve model performance
    """
    df = df.copy()
    
    # 1. Debt-to-income interaction with loan amount
    df['debt_loan_interaction'] = df['debt_to_income_ratio'] * df['loan_amount']
    
    # 2. Credit score to interest rate ratio (higher is better)
    df['credit_interest_ratio'] = df['credit_score'] / (df['interest_rate'] + 1)
    
    # 3. Income to loan ratio (ability to repay)
    df['income_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    
    # 4. Monthly debt burden (assuming monthly income)
    df['monthly_debt_burden'] = (df['annual_income'] / 12) * df['debt_to_income_ratio']
    
    # 5. Loan to income percentage
    df['loan_to_income_pct'] = (df['loan_amount'] / df['annual_income']) * 100
    
    # 6. Risk score (combination of factors)
    df['risk_score'] = (
        df['interest_rate'] * df['debt_to_income_ratio'] / 
        (df['credit_score'] / 100)
    )
    
    # 7. Binned features for non-linear relationships
    df['income_bins'] = pd.qcut(df['annual_income'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    df['credit_bins'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 800, 900], 
                                labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
    
    print(f"Created {len([c for c in df.columns if c not in df.columns[:12]])} new features")
    return df

# Apply feature engineering
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

print("\nNew features created:")
new_features = [c for c in train_df.columns if c not in ['id', 'loan_paid_back']]
print(new_features[-8:])  # Show last 8 features (the new ones)

# =============================================================================
# STEP 5: Data Preprocessing
# =============================================================================
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

def preprocess_data(train, test, target_col='loan_paid_back'):
    """
    Encode categorical variables and scale numerical features
    """
    # Separate features and target
    if target_col in train.columns:
        X = train.drop([target_col, 'id'], axis=1, errors='ignore')
        y = train[target_col]
    else:
        X = train.drop(['id'], axis=1, errors='ignore')
        y = None
    
    X_test = test.drop(['id'], axis=1, errors='ignore')
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {len(numerical_cols)}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to handle unseen categories
        combined_values = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined_values)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print("\nPreprocessing complete!")
    print(f"Final feature shape: {X.shape}")
    
    return X, y, X_test, label_encoders, scaler

# Preprocess the data
X_train, y_train, X_test, encoders, scaler = preprocess_data(train_df, test_df)

# =============================================================================
# STEP 6: Model Training with Multiple Algorithms
# =============================================================================
print("\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

# Split data for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTraining set: {X_tr.shape}")
print(f"Validation set: {X_val.shape}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                           min_samples_split=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                    max_depth=5, random_state=42)
}

# Train and evaluate each model
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_tr, y_tr)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate AUC-ROC
    auc_score = roc_auc_score(y_val, y_pred_proba)
    results[name] = auc_score
    trained_models[name] = model
    
    print(f"{name} - Validation AUC-ROC: {auc_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='roc_auc', n_jobs=-1)
    print(f"{name} - CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Select best model
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\n" + "="*80)
print(f"BEST MODEL: {best_model_name}")
print(f"Validation AUC-ROC: {results[best_model_name]:.4f}")
print("="*80)

# =============================================================================
# STEP 7: Feature Importance Analysis
# =============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

if hasattr(best_model, 'feature_importances_'):
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# STEP 8: Model Evaluation
# =============================================================================
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# ROC Curve
y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {results[best_model_name]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Validation Set', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification report (using optimal threshold)
y_val_pred = (y_val_pred_proba > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# =============================================================================
# STEP 9: Generate Predictions for Test Set
# =============================================================================
print("\n" + "="*80)
print("GENERATING TEST PREDICTIONS")
print("="*80)

# Make predictions on test set
test_predictions = best_model.predict_proba(X_test)[:, 1]

print(f"\nTest predictions generated: {len(test_predictions)}")
print(f"Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
print(f"Mean prediction: {test_predictions.mean():.4f}")

# =============================================================================
# STEP 10: Create Submission File
# =============================================================================
print("\n" + "="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("\nSubmission file created: submission.csv")
print(f"Submission shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head(10))

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"Validation AUC-ROC: {results[best_model_name]:.4f}")
print(f"\nSubmission file ready for upload to Kaggle!")
print("="*80)

# =============================================================================
# STEP 11: Ensemble Model (BONUS - Optional Advanced Approach)
# =============================================================================
print("\n" + "="*80)
print("BONUS: ENSEMBLE PREDICTIONS")
print("="*80)

# Create ensemble predictions by averaging all models
ensemble_val_preds = np.zeros(len(X_val))
ensemble_test_preds = np.zeros(len(X_test))

for name, model in trained_models.items():
    ensemble_val_preds += model.predict_proba(X_val)[:, 1]
    ensemble_test_preds += model.predict_proba(X_test)[:, 1]

ensemble_val_preds /= len(trained_models)
ensemble_test_preds /= len(trained_models)

# Evaluate ensemble
ensemble_auc = roc_auc_score(y_val, ensemble_val_preds)
print(f"\nEnsemble Validation AUC-ROC: {ensemble_auc:.4f}")

# Create ensemble submission
ensemble_submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': ensemble_test_preds
})
ensemble_submission.to_csv('submission_ensemble.csv', index=False)

print("Ensemble submission file created: submission_ensemble.csv")
print("\nComparison:")
print(f"Best Single Model AUC: {results[best_model_name]:.4f}")
print(f"Ensemble Model AUC: {ensemble_auc:.4f}")

