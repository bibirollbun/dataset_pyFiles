# Import core libraries
import os
import glob
import math
import warnings
import typing as t

# Import data stack
import numpy as np
import pandas as pd

# Import plotting
import matplotlib.pyplot as plt

# Import modeling tools
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

# Global display options
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 140)
warnings.filterwarnings('ignore')

# Paths and patterns
BASE_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
INPUT_GLOB = os.path.join(BASE_DIR, 'input_2023_*.csv')
OUTPUT_GLOB = os.path.join(BASE_DIR, 'output_2023_*.csv')
SUPPLEMENT_PATH = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv'

# User-editable configuration
TARGET_COLUMN = None
ID_COLUMNS = None
JOIN_KEYS = None
TASK_TYPE = None
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
MAX_EDA_PLOTS = 15



# Define a function to load many CSVs by glob
def load_many_csvs(pattern):
    # Collect paths
    paths = sorted(glob.glob(pattern))
    # Return None if no matches
    if len(paths) == 0:
        return None, []
    # Read all dataframes
    frames = []
    # Iterate paths
    for p in paths:
        # Read csv
        df = pd.read_csv(p, low_memory=False)
        # Keep source file name
        df['__source_file'] = os.path.basename(p)
        # Append frame
        frames.append(df)
    # Concatenate frames
    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Return combined and list of paths
    return combined, paths

# Define a function to detect target column and task type
def detect_target_and_task(input_df, output_df):
    # Initialize results
    target_col = None
    task_type = None
    
    # Choose candidates
    candidates = ['target', 'label', 'y', 'outcome']
    
    # Search output first
    if output_df is not None:
        # Intersect with columns
        for c in candidates:
            if c in output_df.columns:
                target_col = c
                break
        # Fallback to last column
        if target_col is None and len(output_df.columns) > 0:
            target_col = output_df.columns[-1]
    
    # If still None, search input
    if target_col is None and input_df is not None:
        for c in candidates:
            if c in input_df.columns:
                target_col = c
                break
    
    # Determine task type
    if target_col is not None:
        # Choose source df
        src = output_df if output_df is not None and target_col in output_df.columns else input_df
        # Compute unique count
        nunique = src[target_col].nunique(dropna=False)
        # Infer type
        if nunique <= 20 and not np.issubdtype(src[target_col].dtype, np.floating):
            task_type = 'classification'
        else:
            task_type = 'regression'
    
    # Return results
    return target_col, task_type

# Define a function to propose join keys by shared columns
def propose_join_keys(df_left, df_right):
    # Return None if unavailable
    if df_left is None or df_right is None:
        return None
    # Compute shared columns
    shared = [c for c in df_left.columns if c in df_right.columns]
    # Exclude non-keys
    exclude = {'target', 'label', 'y', 'outcome', '__source_file'}
    # Filter
    shared = [c for c in shared if c not in exclude]
    # Return top candidates
    if len(shared) == 0:
        return None
    if len(shared) == 1:
        return [shared[0]]
    return shared[:2]

# Define a main loader
def load_all():
    # Load inputs
    df_in, in_paths = load_many_csvs(INPUT_GLOB)
    # Load outputs
    df_out, out_paths = load_many_csvs(OUTPUT_GLOB)
    # Load supplement
    df_sup = pd.read_csv(SUPPLEMENT_PATH, low_memory=False) if os.path.exists(SUPPLEMENT_PATH) else None
    
    # Detect target and task
    detected_target, detected_task = detect_target_and_task(df_in, df_out)
    
    # Propose join keys
    proposed_keys = propose_join_keys(df_in, df_out)
    
    # Apply user overrides
    target_col = TARGET_COLUMN if TARGET_COLUMN else detected_target
    task_type = TASK_TYPE if TASK_TYPE else detected_task
    join_keys = JOIN_KEYS if JOIN_KEYS else proposed_keys
    
    # Print summary
    print('Input files:', len(in_paths))
    print('Output files:', len(out_paths))
    print('Input shape:', None if df_in is None else df_in.shape)
    print('Output shape:', None if df_out is None else df_out.shape)
    print('Supplementary shape:', None if df_sup is None else df_sup.shape)
    print('Detected target:', target_col)
    print('Detected task type:', task_type)
    print('Proposed join keys:', join_keys)
    
    # Return artifacts
    return df_in, df_out, df_sup, target_col, task_type, join_keys

# Execute loader
df_input, df_output, df_supp, target_col, task_type, join_keys = load_all()

# Show head
if df_input is not None:
    display(df_input.head())
if df_output is not None:
    display(df_output.head())
if df_supp is not None:
    display(df_supp.head())



# Define function for basic summaries
def basic_eda(df, name):
    # Print structure
    print(f'[{name}] shape:', df.shape)
    print(f'[{name}] dtypes:')
    print(df.dtypes.head(40))
    # Missingness
    miss = df.isna().sum().sort_values(ascending=False)
    print(f'[{name}] missing values (top 25):')
    print(miss.head(25))

# Define function to plot numeric histograms
def plot_numeric_histograms(df, limit=MAX_EDA_PLOTS):
    # Select numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Truncate to limit
    num_cols = num_cols[:limit]
    # Iterate columns
    for c in num_cols:
        # New figure
        plt.figure()
        # Histogram
        df[c].hist(bins=30)
        # Title
        plt.title(f'Distribution: {c}')
        # Show
        plt.show()

# Define function to plot correlation heatmap
def plot_correlation(df, limit=25):
    # Select numeric subset
    cols = df.select_dtypes(include=[np.number]).columns.tolist()[:limit]
    # Return if not enough columns
    if len(cols) < 2:
        return
    # Compute correlation
    corr = df[cols].corr()
    # New figure
    plt.figure()
    # Show matrix
    plt.imshow(corr.values, aspect='auto')
    # Title
    plt.title('Correlation matrix (numeric subset)')
    # Ticks
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    # Color bar
    plt.colorbar()
    # Show
    plt.show()

# Run EDA on input
if df_input is not None:
    # Summaries
    basic_eda(df_input, 'input')
    # Histograms
    plot_numeric_histograms(df_input)
    # Correlation
    plot_correlation(df_input)

# Target analysis
if target_col is not None:
    # Choose source
    src = df_output if df_output is not None and target_col in df_output.columns else df_input
    # Classification
    if task_type == 'classification':
        # Distribution
        print('Target class distribution:')
        print(src[target_col].value_counts(dropna=False))
    # Regression
    else:
        # Summary
        print('Target distribution summary:')
        print(src[target_col].describe())



# Define a function to assemble the dataset
def assemble_dataset(df_in, df_out, df_sup, target_col, join_keys):
    # Initialize df
    df = df_in.copy() if df_in is not None else None
    # Merge output
    if df is not None and df_out is not None and join_keys is not None:
        df = df.merge(df_out, on=join_keys, how='inner')
    # Merge supplement
    if df is not None and df_sup is not None and join_keys is not None:
        shared = [c for c in join_keys if c in df_sup.columns]
        if len(shared) > 0:
            df = df.merge(df_sup, on=shared, how='left')
    # Drop obvious non-features
    drop_cols = ['__source_file']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    # Return assembled
    return df

# Define a function to split and build preprocessing pipeline
def build_preprocessor(df, target_col):
    # Identify feature columns
    features = [c for c in df.columns if c != target_col]
    # Split by dtype
    num_cols = [c for c in features if np.issubdtype(df[c].dtype, np.number)]
    cat_cols = [c for c in features if c not in num_cols]
    # Numeric pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])
    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=True))
    ])
    # Column transformer
    preproc = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    # Return objects
    return preproc, features, num_cols, cat_cols

# Assemble dataset
dataset = None
if df_input is not None and target_col is not None:
    dataset = assemble_dataset(df_input, df_output, df_supp, target_col, join_keys)

# Train test split
X_train, X_test, y_train, y_test = None, None, None, None
if dataset is not None:
    y = dataset[target_col]
    X = dataset.drop(columns=[target_col])
    if task_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

# Build preprocessor
preproc = None
feature_cols = None
num_cols = None
cat_cols = None
if dataset is not None:
    preproc, feature_cols, num_cols, cat_cols = build_preprocessor(dataset, target_col)
    print('Numeric features:', len(num_cols), 'Categorical features:', len(cat_cols))



# Define helpers to build models
def build_models(task_type):
    # Classification models
    if task_type == 'classification':
        return {
            'Dummy': DummyClassifier(strategy='most_frequent'),
            'LogisticRegression': LogisticRegression(max_iter=200),
            'RandomForest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        }
    # Regression models
    else:
        return {
            'Dummy': DummyRegressor(strategy='mean'),
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
        }

# Define evaluation for classification
def eval_classification(y_true, y_proba, y_pred):
    # Compute metrics
    auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba.shape[1] > 1 else np.nan
    ap = average_precision_score(y_true, y_proba[:, 1]) if y_proba.shape[1] > 1 else np.nan
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    # Print report
    print('Accuracy:', round(acc, 4))
    print('F1 (weighted):', round(f1, 4))
    print('ROC AUC:', round(auc, 4))
    print('PR AUC:', round(ap, 4))
    print('Classification Report:')
    print(classification_report(y_true, y_pred))
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm, aspect='equal')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # ROC curve
    if y_proba.shape[1] > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        plt.figure()
        plt.plot(recall, precision)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

# Define evaluation for regression
def eval_regression(y_true, y_pred):
    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    # Print metrics
    print('MAE:', round(mae, 4))
    print('RMSE:', round(rmse, 4))
    print('R²:', round(r2, 4))

# Train and evaluate models
trained = {}
if dataset is not None:
    # Build candidate models
    models = build_models(task_type)
    # Iterate over models
    for name, base_model in models.items():
        # Create pipeline
        pipe = Pipeline(steps=[('preproc', preproc), ('model', base_model)])
        # Fit
        pipe.fit(X_train, y_train)
        # Predict
        if task_type == 'classification':
            y_pred = pipe.predict(X_test)
            # Predict probabilities with fallback
            if hasattr(pipe.named_steps['model'], 'predict_proba'):
                y_proba = pipe.predict_proba(X_test)
            else:
                proba = pipe.decision_function(X_test)
                proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
                y_proba = np.vstack([1 - proba, proba]).T
            # Print header
            print(f'=== {name} ===')
            # Evaluate
            eval_classification(y_test, y_proba, y_pred)
        else:
            y_pred = pipe.predict(X_test)
            print(f'=== {name} ===')
            eval_regression(y_test, y_pred)
        # Save
        trained[name] = pipe



# Define a function to get top errors or misclassifications
def top_errors(pipe, X_test, y_test, k=25):
    # Classification
    if task_type == 'classification':
        # Predict probabilities
        if hasattr(pipe.named_steps['model'], 'predict_proba'):
            proba = pipe.predict_proba(X_test)
            conf = np.max(proba, axis=1)
            pred = np.argmax(proba, axis=1)
        else:
            decision = pipe.decision_function(X_test)
            decision = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
            conf = np.maximum(decision, 1 - decision)
            pred = (decision > 0.5).astype(int)
        # Convert y_test
        y_true = y_test.values
        # Compute correctness
        correct = (pred == y_true)
        # Compute confidence errors
        order = np.argsort(conf[~correct]) if (~correct).sum() > 0 else np.array([], dtype=int)
        # Return indices
        return np.where(~correct)[0][order][:k]
    # Regression
    else:
        # Predict values
        pred = pipe.predict(X_test)
        # Compute absolute error
        err = np.abs(pred - y_test.values)
        # Sort errors
        order = np.argsort(-err)
        # Return top-k
        return order[:k]

# Define a function to subgroup performance by top categorical columns
def subgroup_eval(pipe, X, y, cat_cols, max_groups=3):
    # Select up to max_groups categorical columns
    cats = cat_cols[:max_groups]
    # Evaluate for each categorical column
    for c in cats:
        # Skip if column not present
        if c not in X.columns:
            continue
        # Value counts
        vc = X[c].astype(str).value_counts().head(5).index.tolist()
        # Iterate groups
        for v in vc:
            # Build mask
            mask = X[c].astype(str) == v
            # Skip tiny groups
            if mask.sum() < 20:
                continue
            # Predict subset
            y_pred = pipe.predict(X[mask])
            # Classification metrics
            if task_type == 'classification':
                acc = accuracy_score(y[mask], y_pred)
                f1 = f1_score(y[mask], y_pred, average='weighted')
                print(f'[Subgroup] {c}={v}  n={mask.sum()}  acc={acc:.3f}  f1={f1:.3f}')
            # Regression metrics
            else:
                mae = mean_absolute_error(y[mask], y_pred)
                rmse = mean_squared_error(y[mask], y_pred, squared=False)
                print(f'[Subgroup] {c}={v}  n={mask.sum()}  MAE={mae:.3f}  RMSE={rmse:.3f}')

# Run error analysis on the RandomForest model if trained
if 'RandomForest' in globals().get('trained', {}):
    # Select model
    rf_pipe = trained['RandomForest']
    # Compute top errors
    idx = top_errors(rf_pipe, X_test, y_test, k=25)
    # Print indices
    print('Top error indices:', idx)
    # Subgroup evaluation
    subgroup_eval(rf_pipe, X_test, y_test, cat_cols)

# Feature importance via RF if available
if 'RandomForest' in globals().get('trained', {}):
    # Extract model
    rf = trained['RandomForest'].named_steps['model']
    # Check attribute
    if hasattr(rf, 'feature_importances_'):
        # Fit preprocessor on full training set
        trained['RandomForest'].named_steps['preproc'].fit(X_train, y_train)
        # Get transformed feature names
        num_features = [f'num__{c}' for c in num_cols]
        cat_features = []
        if len(cat_cols) > 0:
            ohe = trained['RandomForest'].named_steps['preproc'].named_transformers_['cat'].named_steps['onehot']
            cat_features = [f'cat__{i}' for i in range(len(ohe.get_feature_names_out()))]
        # Plot importance
        importances = rf.feature_importances_
        order = np.argsort(-importances)[:20]
        plt.figure()
        plt.bar(range(len(order)), importances[order])
        plt.title('RandomForest Feature Importances (Top 20)')
        plt.xlabel('Feature index')
        plt.ylabel('Importance')
        plt.show()


