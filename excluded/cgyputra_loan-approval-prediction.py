# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# Load datasets
train_data = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")

# Display basic information about the datasets
print("Training Data Shape:", train_data.shape)
print("\nTraining Data Info:")
print(train_data.info())

print("\nFirst few rows of training data:")
print(train_data.head())


# Data Preprocessing

# Function to handle outliers and clean data
def clean_data(df):
    df = df.copy()
    
    # Remove extreme outliers
    df = df[df['person_age'] < 100]
    df = df[df['person_emp_length'] < 50]
    df = df[df['person_income'] < 1000000]
    
    return df

# Clean training data
train_data = clean_data(train_data)

# Check missing values
print("Missing values in training data:")
print(train_data.isnull().sum())



# Feature Engineering

def create_features(df):
    df = df.copy()
    
    # Income to loan ratio
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    
    # Monthly loan payment estimation
    df['estimated_monthly_payment'] = (
        df['loan_amnt'] * (df['loan_int_rate']/1200) / 
        (1 - (1 + df['loan_int_rate']/1200)**-36)
    )
    
    # Debt burden ratio
    df['debt_burden_ratio'] = df['estimated_monthly_payment'] / (df['person_income']/12)
    
    # Credit history score
    df['credit_score'] = df['cb_person_cred_hist_length'] * 2
    df.loc[df['cb_person_default_on_file'] == 'Y', 'credit_score'] *= 0.7
    
    # Age groups
    df['age_group'] = pd.cut(
        df['person_age'], 
        bins=[0, 25, 35, 45, 100], 
        labels=['Young', 'Adult', 'Middle', 'Senior']
    )
    
    # Employment length groups
    df['emp_length_group'] = pd.cut(
        df['person_emp_length'], 
        bins=[-1, 2, 5, 10, 100], 
        labels=['New', 'Established', 'Experienced', 'Veteran']
    )
    
    return df

# Apply feature engineering
train_data = create_features(train_data)
test_data = create_features(test_data)

# Display new features
print("New features added:")
print(train_data[['income_to_loan_ratio', 'estimated_monthly_payment', 
                  'debt_burden_ratio', 'credit_score', 'age_group', 
                  'emp_length_group']].head())



# Encode Categorical Variables

# Identify categorical columns
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 
                      'cb_person_default_on_file', 'age_group', 'emp_length_group']

# Initialize label encoders
label_encoders = {}

# Encode categorical variables
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])
    test_data[column] = label_encoders[column].transform(test_data[column])

print("Encoded categorical columns:")
print(train_data[categorical_columns].head())



# Prepare Features and Split Data

# Select features for modeling
features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
           'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
           'income_to_loan_ratio', 'estimated_monthly_payment', 'debt_burden_ratio',
           'credit_score'] + categorical_columns

# Prepare X and y
X = train_data[features]
y = train_data['loan_status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)



# Train Random Forest Model

# Define Random Forest parameters
rf_params = {
    'n_estimators': [200, 300, 400],
    'max_depth': [15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize and train Random Forest with RandomizedSearchCV
rf_model = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(
    rf_model, 
    rf_params, 
    n_iter=20, 
    cv=3, 
    random_state=42,
    verbose=2
)
rf_random.fit(X_train, y_train)

print("\nBest Random Forest parameters:", rf_random.best_params_)
rf_model = rf_random.best_estimator_



# Train XGBoost Model

# Define XGBoost parameters
xgb_params = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Initialize and train XGBoost with RandomizedSearchCV
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False)
xgb_random = RandomizedSearchCV(
    xgb_model, 
    xgb_params, 
    n_iter=20, 
    cv=3, 
    random_state=42,
    verbose=2
)
xgb_random.fit(X_train, y_train)

print("\nBest XGBoost parameters:", xgb_random.best_params_)
xgb_model = xgb_random.best_estimator_



# Train CatBoost Model

# Define CatBoost parameters
catboost_params = {
    'iterations': [200, 300, 400],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128]
}

# Initialize and train CatBoost with RandomizedSearchCV
cat_model = CatBoostClassifier(
    random_state=42,
    cat_features=categorical_columns,
    verbose=False
)
cat_random = RandomizedSearchCV(
    cat_model, 
    catboost_params, 
    n_iter=20, 
    cv=3, 
    random_state=42,
    verbose=2
)
cat_random.fit(X_train, y_train)

print("\nBest CatBoost parameters:", cat_random.best_params_)
cat_model = cat_random.best_estimator_



# Create and Train Ensemble Model

# Create voting classifier
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('cat', cat_model)
    ],
    voting='soft',
    weights=[1, 1.2, 1.1]  # Giving slightly more weight to XGBoost and CatBoost
)

# Train ensemble
ensemble.fit(X_train, y_train)


# Model Evaluation Function

def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    acc_score = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Performance:")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"Accuracy Score: {acc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.show()



# Evaluate All Models

# Dictionary of models
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'CatBoost': cat_model,
    'Ensemble': ensemble
}

# Evaluate each model
for name, model in models.items():
    evaluate_model(model, X_test, y_test, name)



# Generate Final Predictions

# Make predictions on test set using ensemble model
final_predictions = ensemble.predict_proba(test_data[features])[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'loan_status': final_predictions
})

# Save predictions
submission.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission.csv'")

