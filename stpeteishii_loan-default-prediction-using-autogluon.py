


# Install uv using pip
!pip install uv
# Install autogluon with uv
!uv pip install autogluon


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("AutoGluon not installed. Will use sklearn models only.")
    print("To install: pip install autogluon")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_data(train_path, test_path):
    """Load train and test datasets"""
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"âœ“ Train shape: {train_df.shape}")
    print(f"âœ“ Test shape: {test_df.shape}")
    return train_df, test_df

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
def eda_summary(df, is_train=True):
    """Comprehensive EDA summary"""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    print("\nğŸ“Š Dataset Info:")
    print(df.info())
    
    print("\nğŸ”� Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_df = pd.DataFrame({
            'Column': missing[missing > 0].index,
            'Missing Count': missing[missing > 0].values,
            'Percentage': (missing[missing > 0].values / len(df) * 100).round(2)
        })
        print(missing_df.to_string(index=False))
    else:
        print("âœ“ No missing values found!")
    
    print("\nğŸ“ˆ Numerical Features Summary:")
    print(df.describe().round(2))
    
    if is_train and 'loan_paid_back' in df.columns:
        print("\nğŸ�¯ Target Variable Distribution:")
        target_counts = df['loan_paid_back'].value_counts()
        print(target_counts)
        payback_rate = df['loan_paid_back'].mean() * 100
        default_rate = (1 - df['loan_paid_back'].mean()) * 100
        print(f"\nâœ“ Payback Rate: {payback_rate:.2f}%")
        print(f"âœ“ Default Rate: {default_rate:.2f}%")
        
        if default_rate < 20 or default_rate > 80:
            print("âš ï¸�  WARNING: Imbalanced dataset detected!")
    
    print("\nğŸ“‹ Categorical Features:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head(10))

def plot_eda(df, output_dir='eda_plots'):
    """Generate EDA visualizations"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nğŸ“Š Generating EDA plots...")
    
    # Target distribution
    if 'loan_paid_back' in df.columns:
        plt.figure(figsize=(8, 6))
        df['loan_paid_back'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'])
        plt.title('Loan Payback Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Loan Paid Back (0=No, 1=Yes)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_distribution.png', dpi=300)
        plt.close()
    
    # Numerical features distribution
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in ['id', 'loan_paid_back']]
    
    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols[:6]):
            df[col].hist(bins=50, ax=axes[idx], color='#3498db', edgecolor='black')
            axes[idx].set_title(f'{col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/numerical_distributions.png', dpi=300)
        plt.close()
    
    print(f"âœ“ Plots saved to '{output_dir}/' directory")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
def feature_engineering(df, verbose=True):
    """Create engineered features"""
    if verbose:
        print("\nğŸ”§ Feature Engineering...")
    
    df = df.copy()
    
    # 1. Income-based features
    df['monthly_income'] = df['annual_income'] / 12
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['log_annual_income'] = np.log1p(df['annual_income'])
    df['log_loan_amount'] = np.log1p(df['loan_amount'])
    
    # 2. Estimated monthly installment using loan payment formula
    # PMT = P * [r(1+r)^n] / [(1+r)^n - 1]
    monthly_rate = df['interest_rate'] / 100 / 12
    n_months = 48  # Average loan term assumption
    
    df['estimated_installment'] = df['loan_amount'] * (
        monthly_rate * (1 + monthly_rate)**n_months
    ) / ((1 + monthly_rate)**n_months - 1)
    
    # Handle edge cases (very low interest rates)
    df['estimated_installment'] = df['estimated_installment'].fillna(
        df['loan_amount'] / n_months
    )
    
    # 3. Payment burden ratios
    df['payment_to_income_ratio'] = df['estimated_installment'] / (df['monthly_income'] + 1)
    df['payment_burden_pct'] = df['payment_to_income_ratio'] * 100
    
    # 4. Debt calculations
    df['existing_debt_estimate'] = df['debt_to_income_ratio'] * df['annual_income']
    df['total_debt_with_loan'] = df['existing_debt_estimate'] + df['loan_amount']
    df['new_dti_ratio'] = df['total_debt_with_loan'] / (df['annual_income'] + 1)
    df['dti_increase'] = df['new_dti_ratio'] - df['debt_to_income_ratio']
    
    # 5. Credit score categories
    df['credit_score_category'] = pd.cut(
        df['credit_score'],
        bins=[0, 580, 670, 740, 800, 850],
        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    )
    
    # 6. Interest rate categories
    df['interest_rate_category'] = pd.cut(
        df['interest_rate'],
        bins=[0, 7, 10, 13, 16, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # 7. Loan amount categories
    df['loan_amount_category'] = pd.cut(
        df['loan_amount'],
        bins=[0, 5000, 10000, 20000, 50000, np.inf],
        labels=['Small', 'Medium', 'Large', 'Very Large', 'Jumbo']
    )
    
    # 8. Risk score (composite metric, normalized 0-1, higher = riskier)
    df['risk_score'] = (
        (1 - (df['credit_score'] - 300) / 550) * 0.35 +
        np.clip(df['debt_to_income_ratio'] / 50, 0, 1) * 0.25 +
        np.clip(df['interest_rate'] / 30, 0, 1) * 0.25 +
        np.clip(df['payment_to_income_ratio'], 0, 1) * 0.15
    )
    
    # 9. Affordability score (higher = more affordable)
    df['affordability_score'] = 1 - df['risk_score']
    
    # 10. Income stability indicator
    df['high_income'] = (df['annual_income'] > df['annual_income'].median()).astype(int)
    df['low_dti'] = (df['debt_to_income_ratio'] < df['debt_to_income_ratio'].median()).astype(int)
    df['financial_health'] = df['high_income'] + df['low_dti']
    
    if verbose:
        print(f"âœ“ Created {df.shape[1] - 13} new features")  # 13 = original columns
    
    return df

# ============================================================================
# 4. PREPROCESSING
# ============================================================================
def preprocess_data(train_df, test_df, target_col='loan_paid_back'):
    """Preprocess train and test data"""
    print("\nâš™ï¸�  Preprocessing data...")
    
    # Store IDs
    train_id = train_df['id'].copy()
    test_id = test_df['id'].copy() if 'id' in test_df.columns else None
    
    # Feature engineering
    train_df = feature_engineering(train_df, verbose=True)
    test_df = feature_engineering(test_df, verbose=False)
    
    # Identify categorical columns
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    print(f"âœ“ Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
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
    
    print(f"âœ“ Training features: {X_train.shape}")
    print(f"âœ“ Test features: {X_test.shape}")
    
    # Feature scaling
    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print("âœ“ Feature scaling completed")
    
    return X_train, y_train, X_test, test_id, scaler, label_encoders

# ============================================================================
# 5. SKLEARN MODEL TRAINING
# ============================================================================
def train_sklearn_models(X_train, y_train):
    """Train multiple sklearn models"""
    
    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced',
            C=0.1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
    }
    
    print("\n" + "="*70)
    print("SKLEARN MODEL TRAINING & EVALUATION")
    print("="*70)
    
    results = {}
    for name, model in models.items():
        print(f"\nğŸ”„ Training {name}...")
        
        # Train
        model.fit(X_tr, y_tr)
        
        # Predict
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Evaluate
        auc_score = roc_auc_score(y_val, y_pred_proba)
        acc_score = accuracy_score(y_val, y_pred)
        
        print(f"âœ“ {name}")
        print(f"  ROC-AUC: {auc_score:.4f}")
        print(f"  Accuracy: {acc_score:.4f}")
        
        results[name] = {
            'model': model,
            'auc': auc_score,
            'accuracy': acc_score
        }
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*70}")
    print(f"ğŸ�† BEST SKLEARN MODEL: {best_model_name}")
    print(f"   ROC-AUC: {results[best_model_name]['auc']:.4f}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"{'='*70}")
    
    # Retrain on full training data
    print("\nğŸ”„ Retraining on full dataset...")
    best_model.fit(X_train, y_train)
    print("âœ“ Training completed")
    
    return best_model, best_model_name, results

# ============================================================================
# 6. AUTOGLUON MODEL TRAINING
# ============================================================================
def train_autogluon_model(train_df, test_df, target_col='loan_paid_back', 
                          time_limit=600, presets='best_quality'):
    """Train AutoGluon model"""
    
    if not AUTOGLUON_AVAILABLE:
        print("\nâ�Œ AutoGluon not available. Skipping.")
        return None, None
    
    print("\n" + "="*70)
    print("AUTOGLUON MODEL TRAINING")
    print("="*70)
    
    # Prepare data
    train_data = train_df.copy()
    test_data = test_df.copy()
    
    # Remove ID column
    if 'id' in train_data.columns:
        train_data = train_data.drop(columns=['id'])
    if 'id' in test_data.columns:
        test_data = test_data.drop(columns=['id'])
    
    print(f"\nâš™ï¸�  AutoGluon Settings:")
    print(f"  Time limit: {time_limit}s ({time_limit/60:.1f} minutes)")
    print(f"  Presets: {presets}")
    print(f"  Evaluation metric: roc_auc")
    
    # Train
    print(f"\nğŸ”„ Training AutoGluon (this may take a while)...")
    predictor = TabularPredictor(
        label=target_col,
        eval_metric='roc_auc',
        problem_type='binary'
    ).fit(
        train_data=train_data,
        time_limit=time_limit,
        presets=presets,
        verbosity=2
    )
    
    # Get leaderboard
    print("\nğŸ“Š Model Leaderboard:")
    leaderboard = predictor.leaderboard(train_data, silent=True)
    print(leaderboard.head(10).to_string())
    
    # Best model info
    try:
        best_model = predictor.model_best
        print(f"\nğŸ�† Best Model: {best_model}")
    except:
        print(f"\nğŸ�† Best Model: {leaderboard.iloc[0]['model']}")
    
    # Feature importance
    print("\nğŸ“ˆ Feature Importance (Top 20):")
    try:
        importance = predictor.feature_importance(train_data)
        print(importance.head(20).to_string())
    except:
        print("  Feature importance not available for this model")
    
    return predictor, leaderboard

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
def plot_feature_importance(model, feature_names, top_n=25, output_file='feature_importance.png'):
    """Display and plot feature importance"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\nğŸ“Š Top {top_n} Most Important Features:")
        print("-" * 70)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices][::-1], color='#3498db')
        plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"âœ“ Feature importance plot saved to '{output_file}'")
        
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\nğŸ“Š Top {top_n} Features (by coefficient magnitude):")
        print("-" * 70)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")

# ============================================================================
# 8. GENERATE PREDICTIONS
# ============================================================================
def generate_predictions(model, X_test, test_id=None, output_file='predictions.csv',
                        model_type='sklearn'):
    """Generate predictions and save to CSV"""
    
    print(f"\nğŸ”® Generating predictions using {model_type} model...")
    
    # Predict probabilities
    if model_type == 'sklearn':
        predictions = model.predict_proba(X_test)[:, 1]
    elif model_type == 'autogluon':
        predictions = model.predict_proba(X_test, as_pandas=False)
        if len(predictions.shape) > 1:
            predictions = predictions[:, 1]
    
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
    
    print(f"âœ“ Predictions saved to '{output_file}'")
    print(f"âœ“ Shape: {submission.shape}")
    print(f"\nğŸ“Š Prediction Statistics:")
    print(submission['loan_paid_back'].describe().to_string())
    
    return submission

# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================
def main(train_path, test_path, output_file='predictions.csv', 
         use_autogluon=True, autogluon_time_limit=600,
         generate_plots=True):
    """Main execution pipeline"""
    
    print("="*70)
    print("ğŸš€ LOAN DEFAULT PREDICTION PIPELINE")
    print("="*70)
    
    # 1. Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # 2. EDA
    eda_summary(train_df, is_train=True)
    
    if generate_plots:
        plot_eda(train_df)
    
    # 3. Train with sklearn
    print("\n" + "="*70)
    print("ğŸ“š SKLEARN APPROACH")
    print("="*70)
    
    X_train, y_train, X_test, test_id, scaler, label_encoders = preprocess_data(
        train_df, test_df
    )
    
    sklearn_model, sklearn_name, sklearn_results = train_sklearn_models(X_train, y_train)
    
    # Feature importance
    plot_feature_importance(sklearn_model, X_train.columns)
    
    # Generate sklearn predictions
    sklearn_predictions = generate_predictions(
        sklearn_model, X_test, test_id, 
        output_file=output_file.replace('.csv', '_sklearn.csv'),
        model_type='sklearn'
    )
    
    # 4. Train with AutoGluon (if available and requested)
    autogluon_predictor = None
    autogluon_predictions = None
    
    if use_autogluon and AUTOGLUON_AVAILABLE:
        print("\n" + "="*70)
        print("ğŸ¤– AUTOGLUON APPROACH")
        print("="*70)
        
        # Prepare data for AutoGluon (original features + engineered)
        train_ag = feature_engineering(train_df.copy(), verbose=False)
        test_ag = feature_engineering(test_df.copy(), verbose=False)
        
        autogluon_predictor, leaderboard = train_autogluon_model(
            train_ag, test_ag,
            time_limit=autogluon_time_limit,
            presets='best_quality'
        )
        
        if autogluon_predictor is not None:
            # Prepare test data
            test_ag_pred = test_ag.drop(columns=['id'], errors='ignore')
            
            autogluon_predictions = generate_predictions(
                autogluon_predictor, test_ag_pred, test_id,
                output_file=output_file.replace('.csv', '_autogluon.csv'),
                model_type='autogluon'
            )
    
    # 5. Ensemble (if both models available)
    if autogluon_predictions is not None:
        print("\n" + "="*70)
        print("ğŸ”€ ENSEMBLE PREDICTIONS")
        print("="*70)
        
        ensemble_preds = (
            sklearn_predictions['loan_paid_back'] * 0.4 +
            autogluon_predictions['loan_paid_back'] * 0.6
        )
        
        ensemble_submission = pd.DataFrame({
            'id': test_id if test_id is not None else range(len(ensemble_preds)),
            'loan_paid_back': ensemble_preds
        })
        
        ensemble_file = output_file.replace('.csv', '_ensemble.csv')
        ensemble_submission.to_csv(ensemble_file, index=False)
        
        print(f"âœ“ Ensemble predictions saved to '{ensemble_file}'")
        print(f"  Weights: sklearn=40%, autogluon=60%")
        print(f"\nğŸ“Š Ensemble Statistics:")
        print(ensemble_submission['loan_paid_back'].describe().to_string())
    
    # Summary
    print("\n" + "="*70)
    print("âœ… PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nğŸ“� Output Files:")
    print(f"  1. {output_file.replace('.csv', '_sklearn.csv')} (sklearn predictions)")
    if autogluon_predictions is not None:
        print(f"  2. {output_file.replace('.csv', '_autogluon.csv')} (autogluon predictions)")
        print(f"  3. {output_file.replace('.csv', '_ensemble.csv')} (ensemble predictions)")
    if generate_plots:
        print(f"  4. eda_plots/ (EDA visualizations)")
        print(f"  5. feature_importance.png")
    
    return {
        'sklearn_model': sklearn_model,
        'sklearn_predictions': sklearn_predictions,
        'autogluon_predictor': autogluon_predictor,
        'autogluon_predictions': autogluon_predictions,
        'X_train': X_train,
        'y_train': y_train
    }

# ============================================================================
# USAGE
# ============================================================================
if __name__ == "__main__":
    # Configuration
    pj_path='/kaggle/input/playground-series-s5e11'
    TRAIN_PATH = f'{pj_path}/train.csv'
    TEST_PATH = f'{pj_path}/test.csv'
    OUTPUT_PATH = 'submission.csv'
    
    # Run pipeline
    results = main(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        output_file=OUTPUT_PATH,
        use_autogluon=True,  # Set to False to skip AutoGluon
        autogluon_time_limit=600,  # 10 minutes
        generate_plots=True
    )
    
    print("\n" + "="*70)
    print("ğŸ�‰ All done! Check the output files for predictions.")
    print("="*70)

