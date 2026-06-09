import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, GridSearchCV
import re
import joblib
import nltk
from sklearn.metrics import (accuracy_score, classification_report, 
roc_auc_score, roc_curve, f1_score, confusion_matrix, ConfusionMatrixDisplay, 
precision_recall_curve,average_precision_score, precision_score, recall_score)
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


%%time
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train.head()


train.info()


test.info()


train.duplicated().sum()


test.duplicated().sum()


train.describe()


train.describe(include = 'object')


train_numeric = train.select_dtypes(include=['number'])

# Calculate correlation
corrtrain_matrix = train_numeric.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corrtrain_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Correlation")


# --- Target Distribution: Accident Risk ---
plt.figure(figsize=(8,5))
sns.histplot(train["loan_paid_back"], bins=40, kde=True, color="skyblue")
plt.title("Distribution of Loan Paid Back (Target)", fontsize=14)
plt.xlabel("Loan Paid Back")
plt.ylabel("Count")
plt.show()


exclude_cols = ["loan_paid_back", "id"]

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(num_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]   # handle single row
    sns.histplot(train[col], bins=40, kde=True, color="orange", ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

for j in range(len(num_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


# --- Numerical Features vs Target (Scatter Plot) ---
exclude_cols = ["loan_paid_back", "id"]
target_col = "loan_paid_back"

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(num_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]   # handle single row
    
    sns.scatterplot(x=train[col], y=train[target_col], alpha=0.5, color="orange", ax=ax)
    ax.set_title(f"{col} vs {target_col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel(target_col)

for j in range(len(num_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


# --- Categorical Features vs Numeric Target ---
exclude_cols = ["id"]   # contoh exclude
target_col = "loan_paid_back"  # target numerik

cat_cols = train.select_dtypes(include=["object"]).columns
cat_cols = [c for c in cat_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(cat_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(cat_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]
    
    sns.boxplot(x=train[col], y=train[target_col], ax=ax)
    ax.set_title(f"{target_col} by {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel(target_col)
    ax.tick_params(axis="x", rotation=45)

# hapus subplot kosong
for j in range(len(cat_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


# Global variables for storing encoder mappings and scalers
encoder_mappings = {}
scalers = {}

def create_feature_engineering(df, target_col='loan_paid_back', is_training=True):
    """
    Performing comprehensive feature engineering for the loan prediction dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset input
    target_col : str
        Target column name (default: ‘loan_paid_back’)
    is_training : bool
        True if training data, False if test data
    
    Returns:
    --------
    pandas DataFrame with engineered features
    """
    
    # Create a copy of the dataframe
    df_eng = df.copy()
    
    # Save the target if there is one
    if target_col in df_eng.columns and is_training:
        target = df_eng[target_col]
        df_eng = df_eng.drop(columns=[target_col])
    else:
        target = None
    
    print(f"Starting feature engineering for {'training' if is_training else 'test'} data...")
    print(f"First Shape: {df_eng.shape}")
    
    # 1. FEATURE ENGINEERING NUMERIC
    print("1. Processing numerical features...")
    df_eng = create_numerical_features(df_eng, is_training)
    
    # 2. FEATURE ENGINEERING CATEGORICAL
    print("2. Processing categorical features...")
    df_eng = create_categorical_features(df_eng, target, is_training)
    
    # 3. FEATURE ENGINEERING GRADE_SUBGRADE
    print("3. Processing grade_subgrade...")
    df_eng = process_grade_subgrade(df_eng, target, is_training)
    
    # 4. ADVANCED FEATURE ENGINEERING
    print("4. Creating advanced features...")
    df_eng = create_advanced_features(df_eng)
    
    # 5. POLYNOMIAL FEATURES
    print("5. Creating polynomial features...")
    df_eng = create_polynomial_features(df_eng, is_training)
    
    # 6. DOMAIN KNOWLEDGE FEATURES
    print("6. Creating domain knowledge features...")
    df_eng = create_domain_features(df_eng)

    # Delete duplocate columns
    df_eng = remove_duplicate_columns(df_eng)
    
    # Reset the target if training
    if target is not None and is_training:
        df_eng[target_col] = target
    
    print(f"Feature engineering complete! Final shape: {df_eng.shape}")
    return df_eng

def remove_duplicate_columns(df):
    """Removing duplicate columns from a DataFrame"""
    # Check the duplicate column
    duplicate_cols = df.columns[df.columns.duplicated()]
    if len(duplicate_cols) > 0:
        print(f"Menghapus {len(duplicate_cols)} duplicate columns: {list(duplicate_cols)}")
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def create_numerical_features(df, is_training=True):
    """Creating features from numeric variables"""
    df_fe = df.copy()
    
    # Pastikan kolom numerik ada
    num_cols_to_check = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
    existing_num_cols = [col for col in num_cols_to_check if col in df_fe.columns]
    
    if not existing_num_cols:
        return df_fe
    
    # Log transform for features that may be skewed
    if 'annual_income' in df_fe.columns:
        df_fe['log_annual_income'] = np.log1p(df_fe['annual_income'])
    if 'loan_amount' in df_fe.columns:
        df_fe['log_loan_amount'] = np.log1p(df_fe['loan_amount'])
    
    # Ratio and interaction features
    if all(col in df_fe.columns for col in ['annual_income', 'loan_amount']):
        df_fe['income_to_loan_ratio'] = df_fe['annual_income'] / (df_fe['loan_amount'] + 1)
    
    if all(col in df_fe.columns for col in ['annual_income', 'debt_to_income_ratio']):
        df_fe['debt_burden'] = df_fe['annual_income'] * df_fe['debt_to_income_ratio']
    
    if all(col in df_fe.columns for col in ['annual_income', 'loan_amount', 'interest_rate']):
        df_fe['affordability_score'] = df_fe['annual_income'] / (df_fe['loan_amount'] * df_fe['interest_rate'] + 1)
    
    if all(col in df_fe.columns for col in ['credit_score', 'interest_rate']):
        df_fe['credit_interest_ratio'] = df_fe['credit_score'] / (df_fe['interest_rate'] + 1)
    
    # Risk assessment features
    if all(col in df_fe.columns for col in ['debt_to_income_ratio', 'interest_rate']):
        df_fe['risk_score'] = df_fe['debt_to_income_ratio'] * df_fe['interest_rate']
    
    if all(col in df_fe.columns for col in ['credit_score', 'debt_to_income_ratio']):
        df_fe['financial_stability'] = df_fe['credit_score'] - (df_fe['debt_to_income_ratio'] * 100)
    
    return df_fe

def create_categorical_features(df, target=None, is_training=True):
    """Creating features from categorical variables"""
    df_fe = df.copy()
    
    # One-Hot Encoding for categories with few unique values
    categorical_features = ['gender', 'marital_status', 'employment_status']
    
    for col in categorical_features:
        if col in df_fe.columns:
            # Check if the column already exists before creating dummies.
            dummy_prefix = f"{col}_"
            existing_dummy_cols = [c for c in df_fe.columns if c.startswith(dummy_prefix)]
            
            if not existing_dummy_cols:
                dummies = pd.get_dummies(df_fe[col], prefix=col)
                df_fe = pd.concat([df_fe, dummies], axis=1)
    
    # Target Encoding for categories with many unique values
    high_cardinality_features = ['education_level', 'loan_purpose']
    
    for col in high_cardinality_features:
        if col in df_fe.columns:
            encoded_col = f'{col}_encoded'
            if is_training and target is not None:
                # Training mode - fit dan transform
                encoder = TargetEncoder()
                encoded_values = encoder.fit_transform(df_fe[col], target)
                df_fe[encoded_col] = encoded_values
                
                # Save mapping for inference
                mapping = df_fe.groupby(col)[encoded_col].first().to_dict()
                encoder_mappings[encoded_col] = mapping
                
            else:
                # Inference mode - use saved mapping
                if encoded_col in encoder_mappings:
                    mapping = encoder_mappings[encoded_col]
                    df_fe[encoded_col] = df_fe[col].map(mapping)
                    # Fill missing values with the global mean
                    if mapping:
                        global_mean = np.mean(list(mapping.values()))
                        df_fe[encoded_col] = df_fe[encoded_col].fillna(global_mean)
    
    # Frequency encoding
    for col in ['education_level', 'loan_purpose']:
        if col in df_fe.columns:
            freq_col = f'{col}_freq'
            if is_training:
                freq_map = df_fe[col].value_counts().to_dict()
                df_fe[freq_col] = df_fe[col].map(freq_map)
                encoder_mappings[freq_col] = freq_map
            else:
                if freq_col in encoder_mappings:
                    freq_map = encoder_mappings[freq_col]
                    df_fe[freq_col] = df_fe[col].map(freq_map)
    
    return df_fe

def process_grade_subgrade(df, target=None, is_training=True):
    """Processing the grade_subgrade feature"""
    df_fe = df.copy()
    
    if 'grade_subgrade' in df_fe.columns:
        # Split grade and subgrade (format: A1, B2, C3, etc.)
        df_fe['loan_grade'] = df_fe['grade_subgrade'].str[0]  # Take the first character
        
        # Ordinal encoding for loan grade
        grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
        df_fe['loan_grade_ordinal'] = df_fe['loan_grade'].map(grade_mapping).fillna(4)
        
        # Numeric subgrade (A1=1, A2=2, B1=3, etc.)
        subgrade_mapping = {
            'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
            'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10,
            'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15,
            'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
            'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25,
            'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30
        }
        df_fe['loan_subgrade_numeric'] = df_fe['grade_subgrade'].map(subgrade_mapping).fillna(18)  # Default D3
        # Target encoding for subgrade
        encoded_col = 'loan_subgrade_encoded'
        if is_training and target is not None:
            encoder = TargetEncoder()
            df_fe[encoded_col] = encoder.fit_transform(df_fe[['grade_subgrade']], target)
            
            # Save mapping
            mapping = df_fe.groupby('grade_subgrade')[encoded_col].first().to_dict()
            encoder_mappings[encoded_col] = mapping
        else:
            if encoded_col in encoder_mappings:
                mapping = encoder_mappings[encoded_col]
                df_fe[encoded_col] = df_fe['grade_subgrade'].map(mapping)
                df_fe[encoded_col] = df_fe[encoded_col].fillna(df_fe['loan_grade_ordinal'])
    
    return df_fe

def create_advanced_features(df):
    """Membuat features lanjutan"""
    df_fe = df.copy()
    
    # Binning numerical variables with error handling
    try:
        if 'annual_income' in df_fe.columns:
            income_bins = [0, 30000, 50000, 75000, 100000, float('inf')]
            df_fe['income_bin'] = pd.cut(df_fe['annual_income'], bins=income_bins, labels=[1, 2, 3, 4, 5])
        
        if 'credit_score' in df_fe.columns:
            credit_bins = [0, 580, 670, 740, 800, 850]
            df_fe['credit_score_bin'] = pd.cut(df_fe['credit_score'], bins=credit_bins, labels=[1, 2, 3, 4, 5])
        
        if 'loan_amount' in df_fe.columns:
            loan_bins = [0, 5000, 15000, 30000, 50000, float('inf')]
            df_fe['loan_size'] = pd.cut(df_fe['loan_amount'], bins=loan_bins, labels=[1, 2, 3, 4, 5])
        
        if 'interest_rate' in df_fe.columns:
            interest_bins = [0, 8, 12, 15, 18, float('inf')]
            df_fe['interest_rate_bin'] = pd.cut(df_fe['interest_rate'], bins=interest_bins, labels=[1, 2, 3, 4, 5])
        
        # Convert bins to numeric
        for col in ['income_bin', 'credit_score_bin', 'loan_size', 'interest_rate_bin']:
            if col in df_fe.columns:
                df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce').fillna(3)
                
    except Exception as e:
        print(f"Warning dalam advanced features: {e}")
    
    return df_fe

def create_polynomial_features(df, is_training=True):
    """Membuat polynomial dan interaction features"""
    df_fe = df.copy()
    
    # Select key numerical features for interaction
    interaction_features = ['debt_to_income_ratio', 'interest_rate', 'credit_score']
    
    # Ensure features are in the dataframe
    available_features = [f for f in interaction_features if f in df_fe.columns]
    
    if len(available_features) >= 2:
        try:
            # Use only the available features
            available_data = df_fe[available_features].fillna(0)
            
            if is_training:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                interaction_array = poly.fit_transform(available_data)
                feature_names = poly.get_feature_names_out(available_features)
                scalers['poly'] = poly
            else:
                if 'poly' in scalers:
                    poly = scalers['poly']
                    interaction_array = poly.transform(available_data)
                    feature_names = poly.get_feature_names_out(available_features)
                else:
                    return df_fe
            
            # Create a new DataFrame for interaction features
            interaction_df = pd.DataFrame(interaction_array, columns=feature_names, index=df_fe.index)
            
            # Combine with the main dataframe, avoiding duplicate names
            for col in feature_names:
                if col not in df_fe.columns:
                    df_fe[col] = interaction_df[col]
                    
        except Exception as e:
            print(f"Warning dalam polynomial features: {e}")
    
    return df_fe

def create_domain_features(df):
    """Membuat features berdasarkan domain knowledge lending"""
    df_fe = df.copy()
    
    try:
        # Financial ratios with safe division
        if all(col in df_fe.columns for col in ['annual_income', 'debt_to_income_ratio', 'loan_amount']):
            # Use np.where to avoid division by zero
            df_fe['debt_service_ratio'] = np.where(
                df_fe['loan_amount'] == 0, 
                0,
                (df_fe['annual_income'] * df_fe['debt_to_income_ratio']) / df_fe['loan_amount']
            )
            
            df_fe['credit_utilization'] = np.where(
                df_fe['annual_income'] == 0,
                0,
                df_fe['loan_amount'] / df_fe['annual_income']
            )
        
        # Risk assessment flags
        if 'debt_to_income_ratio' in df_fe.columns:
            df_fe['high_risk_dti'] = (df_fe['debt_to_income_ratio'] > 0.43).astype(int)
        
        if 'credit_score' in df_fe.columns:
            df_fe['poor_credit'] = (df_fe['credit_score'] < 580).astype(int)
            df_fe['good_credit'] = (df_fe['credit_score'] >= 670).astype(int)
        
        if 'interest_rate' in df_fe.columns:
            df_fe['high_interest'] = (df_fe['interest_rate'] > 15).astype(int)
        
        # Income to loan ratio categories
        if 'income_to_loan_ratio' in df_fe.columns:
            df_fe['high_income_ratio'] = (df_fe['income_to_loan_ratio'] > 5).astype(int)
            df_fe['low_income_ratio'] = (df_fe['income_to_loan_ratio'] < 1).astype(int)
        
        # Composite risk score
        risk_cols = [col for col in ['high_risk_dti', 'poor_credit', 'high_interest', 'low_income_ratio'] if col in df_fe.columns]
        if risk_cols:
            df_fe['composite_risk_score'] = df_fe[risk_cols].sum(axis=1)
        
        # Loan purpose risk categories
        high_risk_purposes = ['Debt consolidation', 'Other']
        if 'loan_purpose' in df_fe.columns:
            df_fe['high_risk_purpose'] = df_fe['loan_purpose'].isin(high_risk_purposes).astype(int)
        
        # Employment risk
        if 'employment_status' in df_fe.columns:
            df_fe['employment_risk'] = df_fe['employment_status'].isin(['Self-employed', 'Unemployed']).astype(int)
            
    except Exception as e:
        print(f"Warning dalam domain features: {e}")
    
    return df_fe

def get_final_features(df, target_col='loan_paid_back', id_col='id'):
    """
    Mengembalikan features final setelah engineering dengan menghapus kolom yang tidak diperlukan
    """
    # The column to be dropped
    cols_to_drop = [id_col, target_col, 'grade_subgrade', 'loan_grade', 'gender', 'marital_status',
                   'education_level', 'employment_status', 'loan_purpose']
    
    # Delete columns in the dataframe
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    final_df = df.drop(columns=existing_cols_to_drop)
    
    # Delete duplicate columns again to make sure
    final_df = remove_duplicate_columns(final_df)
    
    # Handle missing values for numeric columns
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        final_df[numeric_cols] = final_df[numeric_cols].fillna(0)
    
    # Handle missing values for categorical columns
    categorical_cols = final_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        final_df[col] = final_df[col].fillna('unknown')
    
    return final_df

# MAIN FUNCTIONS FOR USE
def process_datasets(df_train, df_test, target_col='loan_paid_back', id_col='id'):
    """
    Memproses dataset training dan testing secara konsisten
    """
    # Reset global mappings
    global encoder_mappings, scalers
    encoder_mappings = {}
    scalers = {}
    
    print("=== PROCESSING TRAINING DATA ===")
    df_train_engineered = create_feature_engineering(
        df_train, 
        target_col=target_col, 
        is_training=True
    )
    
    print("\n=== PROCESSING TEST DATA ===")
    df_test_engineered = create_feature_engineering(
        df_test,
        target_col=target_col, 
        is_training=False
    )
    
    # Prepare final features
    X_train = get_final_features(df_train_engineered, target_col=target_col, id_col=id_col)
    y_train = df_train_engineered[target_col]
    X_test = get_final_features(df_test_engineered, target_col=target_col, id_col=id_col)
    
    # Align test features with train features
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test


# Split into train and test for demonstration
df_train = train
df_test = test
    
# Process datasets
X_train, y_train, X_test = process_datasets(df_train, df_test)
    
print("\n5 Features teratas:")
print(X_train.columns[:5])
    
print("\n5 Samples pertama dari X_train:")
print(X_train.head())

print("\n5 Samples pertama dari X_test:")
print(X_test.head())


def build_and_evaluate_classification(df, final_features, target_col='loan_paid_back', test_size=0.2, random_state=101):
    # --- Data ---
    X = df[final_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # --- Models ---
    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state, eval_metric='logloss')
    }

    # --- Hyperparameter grids ---
    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 300],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False]
        },
        "XGBoost": {
            "n_estimators": [100, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 0.1],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [1, 2]
        }
    }

    # --- Cross-validation setup ---
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # --- Results storage ---
    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"=== {name} ===")
        print(f"{'='*50}")

        grid = GridSearchCV(
            model, param_grids[name],
            cv=kf, scoring='roc_auc',
            n_jobs=-1, verbose=1)
        
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        
        # Predict probabilities
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability for class 1
        y_pred = best_model.predict(X_test)

        # --- Evaluation metrics for classification ---
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred)

        print(f"Best Parameters for {name}: {grid.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"\nClassification Report:\n{class_report}")

        # --- ROC Curve ---
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({name})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Probability Distribution ---
        plt.figure(figsize=(10, 6))
        
        # Probabilities for each class
        plt.subplot(1, 2, 1)
        sns.histplot(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Class 0', color='red')
        sns.histplot(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Class 1', color='blue')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title(f'Probability Distribution by True Class ({name})')
        plt.legend()
        
        # Confusion Matrix
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix ({name})')
        
        plt.tight_layout()
        plt.show()

        # --- Precision-Recall Curve ---
        plt.figure(figsize=(8, 6))
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        average_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.plot(recall_curve, precision_curve, linewidth=2, 
                label=f'{name} (AP = {average_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({name})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Store results
        results[name] = {
            "model": best_model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "best_params": grid.best_params_,
            "y_pred_proba": y_pred_proba,
            "y_pred": y_pred
        }

        # --- Feature importance ---
        if hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_
            feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importance})
            feat_imp = feat_imp.sort_values("Importance", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feat_imp.head(15))
            plt.title(f"Top 15 Feature Importance ({name})")
            plt.tight_layout()
            plt.show()

            # Print top features
            print(f"\nTop 10 Features for {name}:")
            for i, row in feat_imp.head(10).iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")

        # --- SHAP Analysis ---
        try:
            print(f"\nCalculating SHAP values for {name}...")
            
            # Sample data untuk SHAP (untuk menghindari memory issues)
            X_train_sample = X_train.iloc[:1000]
            X_test_sample = X_test.iloc[:500]
            
            if name == "RandomForest":
                explainer = shap.TreeExplainer(best_model)
            else:
                explainer = shap.TreeExplainer(best_model, X_train_sample)
            
            shap_values = explainer.shap_values(X_test_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # For binary classification, use values for class 1
                shap_values_class1 = shap_values[1]
            else:
                shap_values_class1 = shap_values
            
            print(f"SHAP Summary Plot ({name}):")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_class1, X_test_sample, feature_names=X.columns.tolist(), show=False)
            plt.tight_layout()
            plt.show()
            
            # SHAP Bar Plot
            print(f"SHAP Bar Plot ({name}):")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_class1, X_test_sample, feature_names=X.columns.tolist(), 
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"SHAP calculation failed for {name}: {str(e)}")
            print("Continuing without SHAP analysis...")

    # --- Model Comparison ---
    print(f"\n{'='*50}")
    print("=== MODEL COMPARISON ===")
    print(f"{'='*50}")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[name]['accuracy'] for name in results],
        'Precision': [results[name]['precision'] for name in results],
        'Recall': [results[name]['recall'] for name in results],
        'F1-Score': [results[name]['f1'] for name in results],
        'ROC-AUC': [results[name]['roc_auc'] for name in results],
        'Avg Precision': [results[name]['average_precision'] for name in results]
    }).round(4)
    
    print(comparison_df)
    
    # Plot comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision']
    comparison_df_melted = comparison_df.melt(id_vars='Model', value_vars=metrics_to_plot, 
                                            var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df_melted, x='Metric', y='Score', hue='Model')
    plt.title('Model Comparison Across Metrics')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return results


# FUNGSI UNTUK PREDICTION PADA DATA BARU
def predict_probability(model_results, X_new, model_name=None):
    """
    Memprediksi probability untuk data baru
    
    Parameters:
    -----------
    model_results : dict hasil dari build_and_evaluate_classification
    X_new : DataFrame dengan features yang sama seperti training
    model_name : nama model yang akan digunakan (jika None, gunakan model terbaik berdasarkan ROC-AUC)
    
    Returns:
    --------
    probabilities : array of probabilities untuk class 1
    predictions : binary predictions
    """
    
    if model_name is None:
        # Pilih model dengan ROC-AUC tertinggi
        best_model_name = max(model_results.keys(), 
                            key=lambda x: model_results[x]['roc_auc'])
        print(f"Using best model: {best_model_name} (ROC-AUC: {model_results[best_model_name]['roc_auc']:.4f})")
        model = model_results[best_model_name]['model']
    else:
        model = model_results[model_name]['model']
    
    # Predict probabilities
    probabilities = model.predict_proba(X_new)[:, 1]  # Probability for class 1
    predictions = (probabilities >= 0.5).astype(int)  # Default threshold 0.5
    
    return probabilities, predictions


# 2. Dapatkan final features
final_features = list(X_train.columns)

# 3. Gabungkan X_train dengan y_train untuk modeling
df_modeling = X_train.copy()
df_modeling['loan_paid_back'] = y_train

# 4. Jalankan classification
results = build_and_evaluate_classification(
    df_modeling, 
    final_features=final_features,
    target_col='loan_paid_back',
    test_size=0.2,
    random_state=101
)

# 5. Prediksi pada test data
probabilities, predictions = predict_probability(results, X_test)

print(f"Probabilities: {probabilities[:10]}")
print(f"Predictions: {predictions[:10]}")


df_sub = sample.drop('loan_paid_back', axis=1)
df_sub.head


df_sub['loan_paid_back'] = probabilities
df_sub.to_csv('submission.csv', index=False)
df_sub.value_counts()
df_sub.head()

