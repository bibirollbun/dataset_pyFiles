


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_data(train_path, test_path):
    """Load train and test datasets"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    display(train_df[0:3].T)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    return train_df, test_df

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
def eda_summary(df, is_train=True):
    """Quick EDA summary"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    print("\nNumerical Features Summary:")
    print(df.describe())
    
    if is_train and 'loan_paid_back' in df.columns:
        print("\nTarget Distribution:")
        print(df['loan_paid_back'].value_counts())
        print(f"Payback Rate: {df['loan_paid_back'].mean()*100:.2f}%")
        print(f"Default Rate: {(1 - df['loan_paid_back'].mean())*100:.2f}%")
    
    print("\nCategorical Features:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head())

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
def feature_engineering(df):
    """Add new engineered features"""
    df = df.copy()

    # --- 1. Income & Loan Ratio Features ---
    #df["$loan_to_income_ratio"] = df["loan_amount"] / df["annual_income"]
    #df["$effective_interest_load"] = (df["loan_amount"] * df["interest_rate"]) / df["annual_income"]
    df["$adjusted_dti"] = df["debt_to_income_ratio"] * (1 + df["interest_rate"])

    # --- 2. Credit Score Normalization ---
    df["$normalized_credit_score"] = df["credit_score"] / 850
    #df["$credit_per_income"] = df["credit_score"] / df["annual_income"]

    # --- 3. Interaction Features ---
    df["$interaction_dti_credit"] = df["debt_to_income_ratio"] * (1 - df["$normalized_credit_score"])
    #df["$income_interest_ratio"] = df["annual_income"] / (df["interest_rate"] + 1e-6)
    df["$loan_interest_product"] = df["loan_amount"] * df["interest_rate"]

    # --- 4. Log / Nonlinear Transformations ---
    df["$log_income"] = np.log1p(df["annual_income"])
    df["$sqrt_loan_amount"] = np.sqrt(df["loan_amount"])
    #df["$income_to_loan_gap"] = df["annual_income"] - df["loan_amount"]

    # --- 5. Employment & Education Derived Stability ---
    df["$is_employed_and_married"] = (
        (df["employment_status"] == "Employed") & (df["marital_status"] == "Married")
    ).astype(int)
    df["$employment_education_combo"] = (
        df["employment_status"].astype(str) + "_" + df["education_level"].astype(str)
    )

    # --- 6. Grade and Purpose Encoding (if available) ---
    # Example: extract grade letter (A-G) and subgrade number
    df["$grade_letter"] = df["grade_subgrade"].str[0]
    #df["$subgrade_number"] = df["grade_subgrade"].str[1:].astype(float, errors="ignore")
    
    # Convert grade letter to numeric scale (A=1, ..., G=7)
    grade_map = {g: i for i, g in enumerate("ABCDEFG", start=1)}
    df["$grade_numeric"] = df["$grade_letter"].map(grade_map)

    # --- 7. Grade-adjusted Interest Rate ---
    df["$grade_interest_diff"] = df["interest_rate"] - df.groupby("$grade_letter")["interest_rate"].transform("mean")

    # --- 8. Purpose Risk Approximation (optional: if categorical encoding exists) ---
    df["$purpose_encoded"] = df["loan_purpose"].astype("category").cat.codes

    return df

# ============================================================================
# 4. PREPROCESSING
# ============================================================================
def preprocess_data(train_df, test_df, target_col='loan_paid_back'):
    """Preprocess train and test data"""
    
    # Store IDs
    train_id = train_df['id'].copy()
    test_id = test_df['id'].copy() if 'id' in test_df.columns else None
    
    # Feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Identify categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    print(f"\nCategorical columns: {categorical_cols}")
    
    # Label encoding for categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col, 'id'])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=['id'], errors='ignore')
    
    # Feature scaling
    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, y_train, X_test, test_id, scaler, label_encoders

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
def train_models(X_train, y_train):
    """Train multiple models and return the best one"""
    
    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
    }
    
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_tr, y_tr)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Evaluate
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        print(f"{name} - ROC-AUC: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, digits=4))
        
        results[name] = {
            'model': model,
            'auc': auc_score
        }
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
    print(f"{'='*60}")
    
    # Retrain on full training data
    best_model.fit(X_train, y_train)
    
    return best_model, best_model_name

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
def plot_feature_importance(model, feature_names, top_n=20):
    """Display feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 60)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_names[idx]:35s} {importances[idx]:.4f}")
    elif hasattr(model, 'coef_'):
        # For logistic regression
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\nTop {top_n} Most Important Features (by coefficient magnitude):")
        print("-" * 60)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_names[idx]:35s} {importances[idx]:.4f}")

# ============================================================================
# 7. GENERATE PREDICTIONS
# ============================================================================
def generate_predictions(model, X_test, test_id=None, output_file='predictions.csv'):
    """Generate predictions and save to CSV"""
    
    # Predict probabilities (float values between 0 and 1)
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Create submission dataframe
    if test_id is not None:
        submission = pd.DataFrame({
            'id': test_id,
            'loan_paid_back': predictions
        })
    else:
        submission = pd.DataFrame({
            'loan_paid_back': predictions
        })
    
    # Save to CSV
    submission.to_csv(output_file, index=False)
    print(f"\nPredictions saved to '{output_file}'")
    print(f"Predictions shape: {submission.shape}")
    print(f"\nPrediction Statistics:")
    print(submission['loan_paid_back'].describe())
    
    return submission

# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================
def main(train_path, test_path, output_file='predictions.csv'):
    """Main execution pipeline"""
    
    print("="*60)
    print("LOAN DEFAULT PREDICTION PIPELINE")
    print("="*60)
    
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # EDA
    eda_summary(train_df, is_train=True)
    
    # Preprocess
    X_train, y_train, X_test, test_id, scaler, label_encoders = preprocess_data(
        train_df, test_df
    )
    
    print(f"\nProcessed training features shape: {X_train.shape}")
    print(f"Processed test features shape: {X_test.shape}")
    print(f"Feature names: {list(X_train.columns)}")
    
    # Train models
    best_model, model_name = train_models(X_train, y_train)
    
    # Feature importance
    plot_feature_importance(best_model, X_train.columns)
    
    # Generate predictions
    predictions = generate_predictions(best_model, X_test, test_id, output_file)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return best_model, predictions

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Update these paths to your actual file locations
    pj_path='/kaggle/input/playground-series-s5e11'
    TRAIN_PATH = f'{pj_path}/train.csv'
    TEST_PATH = f'{pj_path}/test.csv'
    OUTPUT_PATH = 'submission.csv'

    # Run the pipeline
    model, predictions = main(TRAIN_PATH, TEST_PATH, OUTPUT_PATH)
    
    # To use the model later:
    # predictions = model.predict_proba(new_data)[:, 1]

