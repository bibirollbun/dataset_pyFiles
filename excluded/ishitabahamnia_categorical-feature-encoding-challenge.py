# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd

def load_data():
    base_path = '/kaggle/input/cat-in-the-dat-ii/'
    try:
        train = pd.read_csv(base_path + 'train.csv')
        test = pd.read_csv(base_path + 'test.csv')
        sample_sub = pd.read_csv(base_path + 'sample_submission.csv')
        print("Data loaded successfully!")
        return train, test, sample_sub
    except FileNotFoundError:
        print("Could not find data files. Please ensure:")
        print("1. You've added the 'Cat in the Dat II' dataset to your notebook")
        print("2. The files exist at:", base_path)
        print("Available files in /kaggle/input:")
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in filenames:
                print(os.path.join(dirname, filename))
        return None, None, None

train, test, sample_submission = load_data()

if train is not None:
    # Rest of your notebook code here
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
else:
    print("Cannot proceed without data files")


# Load the data
train = pd.read_csv('/kaggle/input/categorical-feature-encoding-challenge-binary-clas/train.csv')
test = pd.read_csv('/kaggle/input/categorical-feature-encoding-challenge-binary-clas/test.csv')
sample_submission = pd.read_csv('/kaggle/input/categorical-feature-encoding-challenge-binary-clas/sample_submission.csv')

# Separate target from features
y = train['target']
X = train.drop(['id', 'target'], axis=1)
test = test.drop('id', axis=1)

# Combine train and test for consistent encoding
all_data = pd.concat([X, test], axis=0)


# Exploratory Data Analysis (EDA)

# First ensure we have our X and y defined properly
y = train['target']
X = train.drop(['id', 'target'], axis=1)

# Define categorical features (columns with object dtype)
cat_features = [col for col in X.columns if X[col].dtype == 'object']
num_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

print("\n=== Basic Dataset Info ===")
print(f"Training data shape: {train.shape}")
print(f"Number of categorical features: {len(cat_features)}")
print(f"Number of numerical features: {len(num_features)}")
print(f"Target variable distribution:\n{y.value_counts(normalize=True)}")

# 1. Target Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title('Target Variable Distribution (0 vs 1)')
plt.show()

# 2. Missing Values Analysis
print("\n=== Missing Values Analysis ===")
missing_values = X.isnull().sum() / len(X) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if len(missing_values) > 0:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.xticks(rotation=90)
    plt.title('Percentage of Missing Values by Feature')
    plt.ylabel('Percentage Missing')
    plt.show()
    
    print("Features with missing values:")
    print(missing_values.round(2))
else:
    print("No missing values found in any features!")

# 3. Categorical Features Analysis
print("\n=== Categorical Features Analysis ===")
if len(cat_features) > 0:
    for col in cat_features:
        print(f"\nFeature: {col}")
        print(f"Number of unique values: {X[col].nunique()}")
        print("Top 5 most frequent values:")
        print(X[col].value_counts().head())
        
        # Plot for features with reasonable cardinality
        if X[col].nunique() < 20:
            plt.figure(figsize=(10, 5))
            sns.countplot(x=X[col], order=X[col].value_counts().iloc[:20].index)
            plt.title(f'Value Counts for {col}')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Skipping plot for {col} - too many unique values (>20)")
else:
    print("No categorical features found!")

# 4. Numerical Features Analysis (if any exist)
print("\n=== Numerical Features Analysis ===")
if len(num_features) > 0:
    print(X[num_features].describe())
    
    for col in num_features:
        plt.figure(figsize=(10, 5))
        sns.histplot(X[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.show()
else:
    print("No numerical features found!")


# Numerical features summary (if any)
num_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
if num_features:
    print("\nNumerical Features Summary:")
    print(X[num_features].describe())
    
    # Plot distributions for numerical features
    for col in num_features:
        plt.figure(figsize=(10, 5))
        sns.histplot(X[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()


# List of categorical features
cat_features = [col for col in X.columns if X[col].dtype == 'object']

# Function to apply different encoding strategies
def encode_features(strategy='ordinal'):
    encoded_data = all_data.copy()
    
    if strategy == 'ordinal':
        # Simple ordinal encoding
        for col in cat_features:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
            
    elif strategy == 'target':
        # Target encoding with regularization (using training data only)
        for col in cat_features:
            te = TargetEncoder()
            encoded_data[col] = te.fit_transform(encoded_data[col].astype(str), y)
            
    elif strategy == 'frequency':
        # Frequency encoding
        for col in cat_features:
            freq = encoded_data[col].value_counts(normalize=True)
            encoded_data[col] = encoded_data[col].map(freq)
            
    elif strategy == 'onehot':
        # One-hot encoding for low cardinality features
        low_cardinality = [col for col in cat_features if encoded_data[col].nunique() < 10]
        high_cardinality = [col for col in cat_features if encoded_data[col].nunique() >= 10]
        
        # One-hot encode low cardinality
        encoded_data = pd.get_dummies(encoded_data, columns=low_cardinality)
        
        # Ordinal encode high cardinality
        for col in high_cardinality:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
    
    elif strategy == 'catboost':
        # CatBoost encoding
        for col in cat_features:
            cbe = CatBoostEncoder()
            encoded_data[col] = cbe.fit_transform(encoded_data[col].astype(str), y)
    
    # Fill missing values (after encoding)
    imputer = SimpleImputer(strategy='median')
    encoded_data = pd.DataFrame(imputer.fit_transform(encoded_data), columns=encoded_data.columns)
    
    return encoded_data

# Split back into train and test
def split_data(encoded_data):
    X_encoded = encoded_data[:len(X)]
    test_encoded = encoded_data[len(X):]
    return X_encoded, test_encoded


# Exploratory Data Analysis (EDA)

# First ensure we have our X and y defined properly
y = train['target']
X = train.drop(['id', 'target'], axis=1)

# Define categorical features (columns with object dtype)
cat_features = [col for col in X.columns if X[col].dtype == 'object']
num_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

print("\n=== Basic Dataset Info ===")
print(f"Training data shape: {train.shape}")
print(f"Number of categorical features: {len(cat_features)}")
print(f"Number of numerical features: {len(num_features)}")
print(f"Target variable distribution:\n{y.value_counts(normalize=True)}")

# 1. Target Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title('Target Variable Distribution (0 vs 1)')
plt.show()

# 2. Missing Values Analysis
print("\n=== Missing Values Analysis ===")
missing_values = X.isnull().sum() / len(X) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if len(missing_values) > 0:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.xticks(rotation=90)
    plt.title('Percentage of Missing Values by Feature')
    plt.ylabel('Percentage Missing')
    plt.show()
    
    print("Features with missing values:")
    print(missing_values.round(2))
else:
    print("No missing values found in any features!")

# 3. Categorical Features Analysis
print("\n=== Categorical Features Analysis ===")
if len(cat_features) > 0:
    for col in cat_features:
        print(f"\nFeature: {col}")
        print(f"Number of unique values: {X[col].nunique()}")
        print("Top 5 most frequent values:")
        print(X[col].value_counts().head())
        
        # Plot for features with reasonable cardinality
        if X[col].nunique() < 20:
            plt.figure(figsize=(10, 5))
            sns.countplot(x=X[col], order=X[col].value_counts().iloc[:20].index)
            plt.title(f'Value Counts for {col}')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Skipping plot for {col} - too many unique values (>20)")
else:
    print("No categorical features found!")

# 4. Numerical Features Analysis (if any exist)
print("\n=== Numerical Features Analysis ===")
if len(num_features) > 0:
    print(X[num_features].describe())
    
    for col in num_features:
        plt.figure(figsize=(10, 5))
        sns.histplot(X[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.show()
else:
    print("No numerical features found!")


# First ensure all required imports are present
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import xgboost as xgb
from category_encoders import TargetEncoder, CatBoostEncoder

# Define y and X properly (if not already done)
y = train['target']
X = train.drop(['id', 'target'], axis=1)

def encode_features(strategy='ordinal'):
    """Enhanced encoding function with proper data splitting"""
    # Make copy to avoid modifying original
    encoded_data = all_data.copy()
    
    # Handle missing values first
    for col in cat_features:
        encoded_data[col] = encoded_data[col].astype(str).fillna('missing')
    
    # Apply encoding strategy
    if strategy == 'ordinal':
        for col in cat_features:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col])
            
    elif strategy == 'target':
        # Important: Only fit on training portion!
        train_size = len(X)
        for col in cat_features:
            te = TargetEncoder()
            train_encoded = te.fit_transform(encoded_data[col][:train_size], y)
            test_encoded = te.transform(encoded_data[col][train_size:])
            encoded_data[col] = pd.concat([train_encoded, test_encoded])
            
    elif strategy == 'frequency':
        for col in cat_features:
            freq = encoded_data[col][:len(X)].value_counts(normalize=True)
            encoded_data[col] = encoded_data[col].map(freq).fillna(0)
            
    elif strategy == 'onehot':
        low_card = [col for col in cat_features if encoded_data[col].nunique() < 10]
        encoded_data = pd.get_dummies(encoded_data, columns=low_card)
        for col in [c for c in cat_features if c not in low_card]:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col])
            
    elif strategy == 'catboost':
        train_size = len(X)
        for col in cat_features:
            cbe = CatBoostEncoder()
            train_encoded = cbe.fit_transform(encoded_data[col][:train_size], y)
            test_encoded = cbe.transform(encoded_data[col][train_size:])
            encoded_data[col] = pd.concat([train_encoded, test_encoded])
    
    return encoded_data

def split_data(encoded_data):
    """Split back into train and test portions"""
    X_encoded = encoded_data[:len(X)]
    test_encoded = encoded_data[len(X):]
    return X_encoded, test_encoded

def evaluate_encoding(strategy, model_type='lgbm'):
    """Enhanced evaluation function with better handling"""
    print(f"\nEvaluating {strategy} encoding with {model_type}...")
    
    try:
        # Encode data
        encoded_data = encode_features(strategy)
        X_encoded, _ = split_data(encoded_data)
        
        # Model configuration
        if model_type == 'lgbm':
            model = lgb.LGBMClassifier(
                n_estimators=500,  # Reduced for faster testing
                learning_rate=0.05,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                metric='auc'
            )
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='auc',
                tree_method='auto'  # Changed from GPU for wider compatibility
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,  # Reduced for faster testing
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        # Cross-validation with error handling
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for speed
        scores = cross_val_score(model, X_encoded, y, cv=kf, scoring='roc_auc', n_jobs=-1)
        
        print(f"Mean ROC AUC: {np.mean(scores):.5f} (Â±{np.std(scores):.5f})")
        return np.mean(scores)
    
    except Exception as e:
        print(f"Error evaluating {strategy}: {str(e)}")
        return 0  # Return 0 score if failed

# Main evaluation loop
strategies = ['ordinal', 'target', 'frequency', 'onehot', 'catboost']
results = {}

print("=== Starting Evaluation ===")
for strategy in strategies:
    results[strategy] = evaluate_encoding(strategy, 'lgbm')

# Results analysis
if results:
    best_strategy = max(results, key=results.get)
    print("\n=== Results Summary ===")
    for strat, score in results.items():
        print(f"{strat:>10}: {score:.5f}")
    print(f"\nBest encoding strategy: {best_strategy} (AUC = {results[best_strategy]:.5f})")
else:
    print("No strategies were successfully evaluated")


# Final Model Training and Submission - Corrected Version
try:
    # 1. Encode with best strategy
    print(f"\nEncoding data using best strategy: {best_strategy}")
    encoded_data = encode_features(best_strategy)
    X_encoded, test_encoded = split_data(encoded_data)
    
    # 2. Train final model with optimal parameters
    print("\nTraining final model...")
    final_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        metric='auc',
        n_jobs=-1,
        verbose=-1
    )
    
    # Create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Correct way to implement early stopping
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(100)
        ]
    )
    
    # 3. Predict on test set
    print("\nGenerating test predictions...")
    test_preds = final_model.predict_proba(test_encoded)[:, 1]
    
    # 4. Create submission file
    submission = sample_submission.copy()
    submission['target'] = test_preds
    submission['target'] = submission['target'].clip(0.0001, 0.9999)
    
    submission.to_csv('submission.csv', index=False)
    
    print("\n=== Submission Summary ===")
    print(f"Shape: {submission.shape}")
    print(f"Target mean: {submission['target'].mean():.4f}")
    print(f"Target min/max: {submission['target'].min():.4f}/{submission['target'].max():.4f}")
    print("\nSubmission file created successfully!")
    
    # 5. Save model and plot importance
    import joblib
    joblib.dump(final_model, 'lgbm_model.pkl')
    print("Model saved as 'lgbm_model.pkl'")
    
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(final_model, max_num_features=20)
    plt.title('Feature Importance')
    plt.show()

except Exception as e:
    print(f"\nError during final model training/submission: {str(e)}")
    try:
        baseline_sub = sample_submission.copy()
        baseline_sub['target'] = y.mean()
        baseline_sub.to_csv('submission.csv', index=False)
        print("Created baseline submission file using target mean")
    except:
        print("Could not create any submission file")


# Final Model Training and Submission with Kaggle Output Display
import os
from IPython.display import FileLink

try:
    # 1. Encode with best strategy
    print(f"\nEncoding data using best strategy: {best_strategy}")
    encoded_data = encode_features(best_strategy)
    X_encoded, test_encoded = split_data(encoded_data)
    
    # 2. Train final model
    print("\nTraining final model...")
    final_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        metric='auc',
        n_jobs=-1,
        verbose=-1
    )
    
    # Train with validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100)
        ]
    )
    
    # 3. Generate predictions
    print("\nGenerating test predictions...")
    test_preds = final_model.predict_proba(test_encoded)[:, 1]
    
    # 4. Create and save submission
    submission = sample_submission.copy()
    submission['target'] = test_preds.clip(0.0001, 0.9999)  # Clip probabilities
    
    submission_path = '/kaggle/working/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    # 5. Display output files
    print("\n=== Output Files ===")
    print("Files in /kaggle/working directory:")
    print(os.listdir('/kaggle/working'))
    
    # Create downloadable link
    display(FileLink(submission_path))
    
    # 6. Model and feature importance
    import joblib
    model_path = '/kaggle/working/lgbm_model.pkl'
    joblib.dump(final_model, model_path)
    display(FileLink(model_path))
    
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(final_model, max_num_features=20)
    plt.title('Feature Importance')
    plt.show()
    
    print("\n=== Submission Summary ===")
    print(f"Target mean: {submission['target'].mean():.4f}")
    print(f"Target range: [{submission['target'].min():.4f}, {submission['target'].max():.4f}]")

except Exception as e:
    print(f"\nError: {str(e)}")
    # Create baseline submission if error occurs
    baseline_path = '/kaggle/working/submission.csv'
    sample_submission['target'] = y.mean()
    sample_submission.to_csv(baseline_path, index=False)
    display(FileLink(baseline_path))
    print("Created baseline submission using target mean")


# ===== KAGGLE SUBMISSION GENERATOR v2.0 =====
import pandas as pd
import numpy as np
import os
import gzip
from IPython.display import display, HTML

def generate_kaggle_submission(model, test_features, sample_submission_path):
    """
    Generates competition-ready submission files with verification
    Returns paths to generated files
    """
    # 1. Load sample submission template
    sample_sub = pd.read_csv(sample_submission_path)
    
    # 2. Generate predictions (optimized)
    preds = model.predict_proba(test_features)[:, 1].astype(np.float32)
    
    # 3. Create submission DataFrame
    submission = pd.DataFrame({
        'id': sample_sub['id'].values,
        'target': np.clip(preds, 0.0001, 0.9999)  # Competition-safe range
    })
    
    # 4. File paths
    base_path = '/kaggle/working/'
    files = {
        'csv': f'{base_path}submission.csv',
        'gzip': f'{base_path}submission.csv.gz'
    }
    
    # 5. Save files
    # CSV version
    submission.to_csv(files['csv'], index=False, float_format='%.5f')
    
    # GZIP version (faster upload)
    with gzip.open(files['gzip'], 'wb') as f:
        f.write(submission.to_csv(index=False, float_format='%.5f').encode())
    
    # 6. Verification
    print("\nâœ… SUBMISSION FILES CREATED")
    print(f"ğŸ“Š Contains {len(submission):,} rows")
    print(f"ğŸ“¦ File sizes:")
    print(f"- CSV: {os.path.getsize(files['csv'])/1024:.1f} KB")
    print(f"- GZIP: {os.path.getsize(files['gzip'])/1024:.1f} KB")
    
    # 7. Create download buttons
    display(HTML(
        f"""
        <div style='margin:1em 0; font-family:Arial'>
            <h3>â¬‡ï¸� Download Submission Files</h3>
            <a href='{files['gzip']}' download style='background:#20BEFF; color:white; padding:0.5em 1em; border-radius:4px; text-decoration:none; margin-right:1em'>
                Download GZIP (Recommended)
            </a>
            <a href='{files['csv']}' download style='background:#666; color:white; padding:0.5em 1em; border-radius:4px; text-decoration:none'>
                Download CSV
            </a>
            <p style='color:#666; margin-top:0.5em'>GZIP is 3x smaller for faster uploads</p>
        </div>
        """
    ))
    
    return files

# ===== USAGE EXAMPLE ===== 
# files = generate_kaggle_submission(
#     model=final_model,
#     test_features=test_encoded,
#     sample_submission_path='/kaggle/input/competition-data/sample_submission.csv'
# )


# 1. FIRST define these required variables:
# (Replace these examples with your actual variables)

# Your trained model (from earlier in notebook)
final_model = lgb.Booster(model_file='model.txt')  # Example - use your actual model

# Your preprocessed test features (200,000 rows)
test_encoded = pd.read_csv('/kaggle/input/test_data_preprocessed.csv')  # Your test data

# Path to competition's sample submission
sample_path = '/kaggle/input/cat-in-the-dat-ii/sample_submission.csv'  # Actual competition path

# 2. THEN run the submission generator
submission_files = generate_kaggle_submission(
    model=final_model,              # Your actual model variable
    test_features=test_encoded,     # Your actual test data
    sample_submission_path=sample_path  # Actual sample submission path
)


# First train your model
from lightgbm import LGBMClassifier
final_model = LGBMClassifier()
final_model.fit(X_train, y_train)  # Your training data


# Load and preprocess test data
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
test_encoded = preprocess_data(test)  # Your preprocessing function


# Check available files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 1. Load data
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
sample_path = '/kaggle/input/cat-in-the-dat-ii/sample_submission.csv'

# 2. Train model (example)
from lightgbm import LGBMClassifier
final_model = LGBMClassifier()
final_model.fit(train.drop(['id', 'target'], axis=1), train['target'])

# 3. Prepare test data (example preprocessing)
test_encoded = test.drop('id', axis=1)  # Replace with your actual preprocessing

# 4. Generate submission
submission_files = generate_kaggle_submission(
    model=final_model,
    test_features=test_encoded,
    sample_submission_path=sample_path
)

