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


# ===========================
# LOAN PAYBACK PREDICTION
# Kaggle Playground Series S5E11
# ===========================

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. LOAD DATA
# ===========================
print("="*60)
print("LOADING DATA")
print("="*60)

train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

target_col = 'loan_paid_back'
id_col = 'id'

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget distribution:")
print(train[target_col].value_counts())
print(f"\nTarget proportion:")
print(train[target_col].value_counts(normalize=True))

# ===========================
# 2. EXPLORATORY DATA ANALYSIS
# ===========================
print("\n" + "="*60)
print("DATA EXPLORATION")
print("="*60)

print("\nTrain columns:")
print(train.columns.tolist())

print("\nData types:")
print(train.dtypes.value_counts())

print("\nMissing values in train:")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\nMissing values in test:")
print(test.isnull().sum()[test.isnull().sum() > 0])

print("\nBasic statistics:")
print(train.describe())

# ===========================
# 3. FEATURE ENGINEERING
# ===========================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

def feature_engineering(df, is_train=True):
    """Create new features from existing ones"""
    df = df.copy()
    
    # Example features (adjust based on your actual columns)
    # Debt-to-income ratio
    if 'monthly_debt' in df.columns and 'monthly_income' in df.columns:
        df['debt_to_income'] = df['monthly_debt'] / (df['monthly_income'] + 1)
    
    # Loan amount to income ratio
    if 'loan_amount' in df.columns and 'monthly_income' in df.columns:
        df['loan_to_income'] = df['loan_amount'] / (df['monthly_income'] * 12 + 1)
    
    # Payment to income ratio
    if 'monthly_payment' in df.columns and 'monthly_income' in df.columns:
        df['payment_to_income'] = df['monthly_payment'] / (df['monthly_income'] + 1)
    
    # Age groups
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                  labels=['young', 'early_career', 'mid_career', 'senior', 'retired'])
    
    # Credit score categories
    if 'credit_score' in df.columns:
        df['credit_category'] = pd.cut(df['credit_score'], 
                                        bins=[0, 580, 670, 740, 800, 850],
                                        labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
    
    return df

# Apply feature engineering
train = feature_engineering(train, is_train=True)
test = feature_engineering(test, is_train=False)

print("✓ Feature engineering complete")

# ===========================
# 4. PREPROCESSING
# ===========================
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

# Separate features and target
X = train.drop([id_col, target_col], axis=1)
y = train[target_col]
X_test = test.drop([id_col], axis=1)

# Identify column types
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

print(f"Numeric columns ({len(num_cols)}): {num_cols[:5]}...")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# ===========================
# Handle missing values
# ===========================

# Numeric columns - fill with median
for col in num_cols:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# Categorical columns - fill with mode
for col in cat_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
        X[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)

print("✓ Missing values handled")

# ===========================
# Encode categorical variables
# ===========================

# Get ALL categorical columns (including newly created ones)
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Categorical columns to encode: {cat_cols}")

label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle unseen categories in test
    X_test[col] = X_test[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    
    label_encoders[col] = le

print("✓ Categorical variables encoded")

# Verify all columns are numeric
print(f"\nData types after encoding:")
print(X.dtypes.value_counts())
print(f"\nAny non-numeric columns: {X.select_dtypes(include='object').columns.tolist()}")

# ===========================
# Scale features
# ===========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X = pd.DataFrame(X_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("✓ Features scaled")

print(f"\nFinal feature shape: {X.shape}")
print(f"Final test shape: {X_test.shape}")

# ===========================
# 5. MODEL TRAINING
# ===========================
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# ===========================
# Try multiple models
# ===========================

# Sometimes simpler is better for generalization
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                           class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Evaluate
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    
    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='accuracy', n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ===========================
# Select best model
# ===========================
print("\n" + "="*60)
print("MODEL SELECTION")
print("="*60)

best_model_name = max(results, key=lambda x: results[x]['val_acc'])
best_model = results[best_model_name]['model']

print(f"Best model: {best_model_name}")
print(f"Validation Accuracy: {results[best_model_name]['val_acc']:.4f}")

# ===========================
# 6. FINAL PREDICTIONS
# ===========================
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

# Retrain on full training data
print("Retraining best model on full training data...")
best_model.fit(X, y)

# Make predictions
predictions = best_model.predict(X_test)

print(f"Predictions generated: {len(predictions)}")
print(f"Prediction distribution:")
print(pd.Series(predictions).value_counts())

# ===========================
# 7. CREATE SUBMISSION
# ===========================
print("\n" + "="*60)
print("CREATING SUBMISSION")
print("="*60)

submission[target_col] = predictions

print(f"\nSubmission shape: {submission.shape}")
print(f"\nSubmission preview:")
print(submission.head(10))

# Save
submission.to_csv('submission.csv', index=False)
print("\n✓ Submission saved to 'submission.csv'")

# ===========================
# 8. FEATURE IMPORTANCE
# ===========================
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)




