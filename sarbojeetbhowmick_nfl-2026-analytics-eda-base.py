# Core libraries
import os
import glob
import math
import warnings
import typing as t

# Data stack and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Modeling tools
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

# Display options
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 140)
warnings.filterwarnings('ignore')

# Input Paths
BASE_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
INPUT_GLOB = os.path.join(BASE_DIR, 'input_2023_*.csv')
OUTPUT_GLOB = os.path.join(BASE_DIR, 'output_2023_*.csv')
SUPPLEMENT_PATH = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv'

# User configuration
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


