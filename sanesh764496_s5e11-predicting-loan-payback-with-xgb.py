# Import libraries
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from colorama import Fore, Style
import xgboost as xgb
from IPython.core.display import HTML

# Center all plots
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

# Set a consistent default figure size
plt.rcParams['figure.figsize'] = (6, 4)  # width=6, height=4 inches

# Standardized output formatting
def print_step(message):
    """Standardized step printing"""
    print(f"ğŸ“Š {message}")

def print_success(message):
    """Standardized success printing"""
    print(f"âœ… {message}")

def print_warning(message):
    """Standardized warning printing"""
    print(f"âš ï¸�  {message}")

def print_fold_header(fold):
    """Standardized fold header"""
    print(f"\n{Fore.GREEN}ğŸ�¯ {'='*15} FOLD {fold} {'='*15}{Style.RESET_ALL}")

def print_section_header(title):
    """Standardized section header"""
    print(f"\n{Fore.CYAN}{'='*20} {title} {'='*20}{Style.RESET_ALL}")


# Load Data
print_section_header("LOADING DATA")
print_step("Loading training and test data")
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
print_success(f"Training data loaded: {train.shape}")
print_success(f"Test data loaded: {test.shape}")


# Check data structure
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nFirst 5 rows of train data:")
display(train.head())

# Data Dimensions
print_section_header("DATA EXPLORATION")
print_step("Checking data structure")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print_step("First 5 rows of train data:")
display(train.head())

print_step("Statistical summary:")
styled_describe = train.describe().style.format("{:.2f}")
display(styled_describe)


import warnings
warnings.filterwarnings("ignore")

# Target Distribution
plt.figure()
sns.histplot(train['loan_paid_back'], kde=True, bins=50)
plt.title('Distribution of Loan Paid Back')
plt.xlabel('Loan Paid Back')
plt.show()


import warnings
warnings.filterwarnings("ignore")

# Correlation Heatmap
plt.figure()
numeric_cols = train.select_dtypes(include=[np.number]).columns
correlation_matrix = train[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Data Preprocessing
print_section_header("DATA PREPROCESSING")

# Separate target variable
TARGET = 'loan_paid_back'
ID_COL = 'id'

y = train[TARGET]
X = train.drop(TARGET, axis=1)
X_test = test.copy()

# Store IDs for later use
train_ids = X[ID_COL]
test_ids = X_test[ID_COL]

# Drop IDs before training
X = X.drop(ID_COL, axis=1)
X_test = X_test.drop(ID_COL, axis=1)

# Identify categorical and numerical features
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numerical_features = [col for col in X.columns if X[col].dtype != 'object']

print_success(f"Basic preprocessing completed")
print_success(f"Initial features - Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")


# Enhanced Financial Feature Engineering
print_section_header("ENHANCED FEATURE ENGINEERING")

def create_domain_features(df):
    """Create domain-specific financial features like the high-scoring model"""
    # Core affordability metrics
    df['income_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    
    # Debt and payment analysis
    df['total_debt'] = df['debt_to_income_ratio'] * df['annual_income']
    df['available_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    df['monthly_payment'] = df['loan_amount'] * df['interest_rate'] / 1200
    df['payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12 + 1)
    
    # Risk scoring (inspired by credit models)
    df['default_risk'] = (df['debt_to_income_ratio'] * 0.40 + 
                         (850 - df['credit_score']) / 850 * 0.35 + 
                         df['interest_rate'] / 100 * 0.25)
    
    # Credit analysis
    df['credit_interest_product'] = df['credit_score'] * df['interest_rate'] / 100
    
    # Log transformations for skewed amounts
    df['annual_income_log'] = np.log1p(df['annual_income'])
    df['loan_amount_log'] = np.log1p(df['loan_amount'])
    
    # Grade parsing (from grade_subgrade)
    df['grade_letter'] = df['grade_subgrade'].str[0]
    df['grade_number'] = df['grade_subgrade'].str[1].astype(int)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_rank'] = df['grade_letter'].map(grade_map)
    
    return df

print_step("Creating domain-specific financial features")
X = create_domain_features(X)
X_test = create_domain_features(X_test)

# Update feature lists
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numerical_features = [col for col in X.columns if X[col].dtype != 'object']

print_success(f"Created domain features. Now have: {len(numerical_features)} numerical, {len(categorical_features)} categorical")


# Strategic Interaction Features
def create_strategic_interactions(df, numerical_features, categorical_features):
    """Create carefully selected interactions like the high-scoring model"""
    new_features = {}
    
    # Only create the most valuable interactions
    important_pairs = [
        # Financial x Demographic
        ('annual_income', 'employment_status'),
        ('credit_score', 'grade_letter'),
        ('debt_to_income_ratio', 'employment_status'),
        ('interest_rate', 'grade_letter'),
        
        # Demographic x Demographic
        ('employment_status', 'education_level'),
        ('employment_status', 'loan_purpose'),
        ('grade_letter', 'loan_purpose'),
    ]
    
    # Create multiplicative interactions for numerical pairs
    for num1, num2 in [('annual_income', 'credit_score'), 
                       ('loan_amount', 'interest_rate'),
                       ('debt_to_income_ratio', 'credit_score')]:
        if num1 in numerical_features and num2 in numerical_features:
            new_features[f'{num1}_x_{num2}'] = df[num1] * df[num2]
    
    # Create categorical interactions
    for col1, col2 in important_pairs:
        if col1 in df.columns and col2 in df.columns:
            # For categorical x categorical, create concatenation
            if col1 in categorical_features and col2 in categorical_features:
                new_features[f'{col1}_{col2}'] = df[col1].astype(str) + '_' + df[col2].astype(str)
            # For numerical x categorical, create group stats
            elif col1 in numerical_features and col2 in categorical_features:
                group_means = df.groupby(col2)[col1].transform('mean')
                new_features[f'{col1}_mean_by_{col2}'] = group_means
    
    print_success(f"Created {len(new_features)} strategic interaction features")
    
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_df], axis=1)
    else:
        return df.copy()

print_step("Creating strategic interaction features")
X = create_strategic_interactions(X, numerical_features, categorical_features)
X_test = create_strategic_interactions(X_test, numerical_features, categorical_features)

# Update feature lists
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numerical_features = [col for col in X.columns if X[col].dtype != 'object']

print_success(f"Final feature count: {X.shape[1]} total features")


# Memory optimization
def optimize_interaction_memory(df, categorical_features, numerical_features):
    """Optimize memory usage specifically for interaction features"""
    
    # Optimize numerical features (including interactions)
    for col in numerical_features:
        if col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].astype(np.float32)
            elif df[col].dtype in ['int64', 'int32']:
                df[col] = df[col].astype(np.int16)
    
    # Optimize categorical features
    for col in categorical_features:
        if col in df.columns:
            # For high cardinality, use categorical type
            if df[col].nunique() > 100:
                df[col] = df[col].astype('category')
    
    return df

print_step("Optimizing memory usage")
X = optimize_interaction_memory(X, categorical_features, numerical_features)
X_test = optimize_interaction_memory(X_test, categorical_features, numerical_features)
print_success("Memory optimization completed")


# Processing
print_section_header("DATA PREPROCESSING")

print_step("Handling missing values")
imputer_numerical = SimpleImputer(strategy='mean')
imputer_categorical = SimpleImputer(strategy='most_frequent')

X[numerical_features] = imputer_numerical.fit_transform(X[numerical_features])
X_test[numerical_features] = imputer_numerical.transform(X_test[numerical_features])

X[categorical_features] = imputer_categorical.fit_transform(X[categorical_features])
X_test[categorical_features] = imputer_categorical.transform(X_test[categorical_features])

print_step("One-hot encoding categorical features")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X[categorical_features])

X_encoded = encoder.transform(X[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

X_encoded_df = pd.DataFrame(X_encoded, index=X.index, columns=encoder.get_feature_names_out(categorical_features))
X_test_encoded_df = pd.DataFrame(X_test_encoded, index=X_test.index, columns=encoder.get_feature_names_out(categorical_features))

X = pd.concat([X.drop(categorical_features, axis=1), X_encoded_df], axis=1)
X_test = pd.concat([X_test.drop(categorical_features, axis=1), X_test_encoded_df], axis=1)

print_step("Scaling numerical features")
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print_success(f"Preprocessing completed. Final feature count: {X.shape[1]}")


##### XGBoost Model with Cross-Validation
# --- Optimized Hyperparameters ---
HYPERPARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.015,  # Balanced learning rate
    'n_estimators': 8000,    # More trees for complex features
    'max_depth': 7,          # Optimal depth balance
    'subsample': 0.80,
    'colsample_bytree': 0.65,  
    'colsample_bylevel': 0.65,
    'random_state': 42,
    'use_label_encoder': False,
    'early_stopping_rounds': 200,
    'reg_alpha': 1.5,        # Increased L1 for sparsity
    'reg_lambda': 3.0,       # Increased L2 for generalization
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
    'max_leaves': 64,        # Increased for more complex trees
    'min_child_weight': 5,   # Added for regularization
    'max_bin': 256           # Added for precision
}

# Instantiate XGBoost model
model = XGBClassifier(**HYPERPARAMS)

print_section_header("MODEL TRAINING")

# Use 5 folds (balanced between stability and computation)
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

start_time = time.time()
print_step(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_step(f"Using {N_SPLITS}-fold CV with {X.shape[1]} features")

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print_fold_header(fold)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    print_step(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")

    model = XGBClassifier(**HYPERPARAMS)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=1000,  # Less verbose
              callbacks=[xgb.callback.EarlyStopping(rounds=HYPERPARAMS['early_stopping_rounds'], save_best=True)]
              )

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    fold_score = roc_auc_score(y_val, val_preds)
    fold_scores.append(fold_score)
    print_success(f"Fold {fold} AUC: {fold_score:.6f} | Best iteration: {model.best_iteration}")

oof_score = roc_auc_score(y, oof_preds)
total_time = (time.time() - start_time) / 3600

print_success(f"Training completed in {total_time:.2f} hours")
print_success(f"Final OOF ROC AUC: {oof_score:.6f}")
print_step(f"Fold scores: {[f'{s:.6f}' for s in fold_scores]}")
print_step(f"Fold std: {np.std(fold_scores):.6f}")


import warnings
warnings.filterwarnings("ignore")

# Evaluation
print_section_header("MODEL EVALUATION")

print_step("Generating ROC curve")
fpr, tpr, thresholds = roc_curve(y, oof_preds)

plt.figure()
plt.plot(fpr, tpr, label=f'OOF ROC AUC = {oof_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print_step("Plotting feature importance")
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure()
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Feature Importance')
plt.show()

print_step("Analyzing prediction distribution")
plt.figure()
sns.histplot(test_preds, kde=True, bins=50, color='blue')
plt.title('Distribution of Test Set Predictions')
plt.xlabel('Predicted Probability of Loan Paid Back')
plt.show()


# Enhanced Feature Analysis
print_section_header("FEATURE ANALYSIS")

# Get feature importance
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

print_step("Top 20 Most Important Features:")
for i, row in importance_df.head(20).iterrows():
    print(f"  {i+1:2d}. {row['Feature']}: {row['Importance']:.4f}")

# Check if new features are important
new_domain_features = ['income_loan_ratio', 'loan_to_income', 'default_risk', 
                      'credit_interest_product', 'grade_rank', 'annual_income_log']
new_feature_importance = importance_df[importance_df['Feature'].isin(new_domain_features)]

print_step("Domain feature importance:")
for _, row in new_feature_importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} (rank {importance_df.index.get_loc(row.name) + 1})")


# Create submission file
print_section_header("SUBMISSION")

print_step("Creating submission file")
submission = pd.DataFrame({'loan_paid_back': test_preds}, index=test_ids)
submission.index.name = 'id'

print_step("Saving submission file")
submission.to_csv('submission.csv', header=True)

print_success(f"Submission file created: {submission.shape}")
print_step("First 5 rows of submission:")
display(submission.head())

