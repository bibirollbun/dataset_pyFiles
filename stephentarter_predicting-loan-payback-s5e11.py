!pip install sweetviz
%pip install --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON lightgbm
%pip -q install -U optuna
%pip -q install optuna-integration[lightgbm]
%pip -q install --upgrade ydata_profiling
!pip install catboost


import os
import sys
import math
import random
import warnings
import json
import joblib
from pathlib import Path
from contextlib import contextmanager
from time import time
from typing import Iterable
from IPython.display import display, Markdown, IFrame

# --- Third-party
import numpy as np
import pandas as pd
from IPython.display import display
from ydata_profiling import ProfileReport
import scipy.stats as st
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

# --- Notebook settings
warnings.filterwarnings('ignore')

%matplotlib inline

# Suppress Optuna/LGBM Warnings
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.INFO)


# Define some utilities functions
def configure_notebook(seed: int = 10301, float_precision: int = 3, max_columns: int = 15, max_rows: int = 25) -> int:
    """
    Configure notebook settings:
      - Disables warnings for cleaner output.
      - Sets pandas display options for better table formatting.
      - Returns a seed value for reproducibility.
    
    Parameters:
      seed (int): Random seed (default 548).
      float_precision (int): Number of decimal places for floats (default 3).
      max_columns (int): Maximum number of columns to display (default 15).
      max_rows (int): Maximum number of rows to display (default 25).

    Returns:
      int: The provided seed.
    """
    # Disable all warnings
    warnings.filterwarnings('ignore')
    
    # Set pandas display options for nicer output
    pd.options.display.float_format = f'{{:,.{float_precision}f}}'.format
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)

    # Set seeds for reproducibility in numpy and the standard random module
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    return seed

def running_in_kaggle() -> bool:
    """
    Heuristics that are true in Kaggle notebooks:
    - Special directories exist (/kaggle/input, /kaggle/working)
    - Env var KAGGLE_KERNEL_RUN_TYPE is set
    - The kaggle_secrets module is available
    """
    try:
        if os.path.isdir('/kaggle/input') and os.path.isdir('/kaggle/working'):
            return True
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return True
        import kaggle_secrets  # noqa: F401  (only exists in Kaggle)
        return True
    except Exception:
        return False

# Use a context manager to suppress standard error output
# This is a robust way to handle warnings that bypass Python's warnings module
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr


# Apply configuration and set random seeds for reproducibility
seed = configure_notebook(max_columns = None, max_rows = None)   

# Configurable flag to control whether GPU is used
USE_GPU = False

# Set some other control variables up here
PERFORM_LGB_OPTUNA_TUNING = False
PERFORM_CB_OPTUNA_TUNING = False


DATA_DIR = Path('/kaggle/input/playground-series-s5e11') if running_in_kaggle() else Path('data')

training_df = pd.read_csv(DATA_DIR / 'train.csv')
print(training_df.head(5))


test_df = pd.read_csv(DATA_DIR / 'test.csv')
print(test_df.head(5))


ORIGINAL_DIR = Path('/kaggle/input/loan-prediction-dataset-2025') if running_in_kaggle() else Path('original_data')

original_df = pd.read_csv(ORIGINAL_DIR / 'loan_dataset_20000.csv')
print(original_df.head(5))


# Create the profile report object for the TRAINING data
training_report = ProfileReport(training_df, title='Training Data')

# Create the profile report object for the TEST data
test_report = ProfileReport(test_df, title='Test Data')

# Create the comparison report
# This compares the TEST report *against* the TRAINING report
train_test_comparison_report = training_report.compare(test_report)

# Save the combined report to an HTML file
train_test_comparison_report.to_file('training_vs_test_comparison.html')


# Display the HTML report in an iframe
IFrame(src='training_vs_test_comparison.html', width=1000, height=600)


# Create the profile report object for the ORIGINAL data
original_report = ProfileReport(original_df, title='Original Data')

# Create the comparison report
# This compares the ORIGINAL report *against* the TRAINING report
train_original_comparison_report = training_report.compare(original_report)

# Save the combined report to an HTML file
train_original_comparison_report.to_file('training_vs_original_comparison.html')


# Display the HTML report in an iframe
IFrame(src='training_vs_original_comparison.html', width=1000, height=600)


TARGET = 'loan_paid_back'
IS_CLASSIFICATION = training_df[TARGET].nunique() <= 10 and training_df[TARGET].dtype != float


def split_columns(df: pd.DataFrame, max_cardinality: int = 30):
    """
    Heuristic split into numeric vs categorical, with a cardinality cap for cats.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    
    # treat low-unique integer columns as categorical (IDs will be filtered later)
    for c in list(num_cols):
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= max_cardinality:
            cat_cols.append(c)
            num_cols.remove(c)
            
    # drop high-cardinality cats from categorical plotting by default
    low_card_cats = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_cardinality]
    
    return num_cols, low_card_cats
    
# short axis label helper
def colname(v):
    return str(v).replace('_', ' ').title()


def collapse_rare_categories(s: pd.Series, min_count: int = 50) -> pd.Series:
    vc = s.value_counts()
    keep = vc[vc >= min_count].index
    return s.where(s.isin(keep), other='__OTHER__')


def categorical_vs_target_classification(df: pd.DataFrame, feature, target, min_count = 50, top_n = 20):
    s = collapse_rare_categories(df[feature].astype('object'), min_count=min_count)
    d = pd.DataFrame({feature: s, target: df[target]}).dropna()
    grp = d.groupby(feature)
    counts = grp.size().sort_values(ascending=False).head(top_n)
    cats = counts.index

    rate = grp[target].mean().reindex(cats)

    # counts
    plt.figure(figsize=(8, max(2, 0.35 * len(cats))))
    plt.barh([str(c) for c in cats], counts.values)
    plt.gca().invert_yaxis()
    plt.xlabel('Count'); plt.title(f'{feature} – Top {len(cats)} Counts')
    plt.show()

    # target rates
    plt.figure(figsize=(8, max(2, 0.35 * len(cats))))
    plt.barh([str(c) for c in cats], rate.values)
    plt.gca().invert_yaxis()
    plt.xlabel(f'Mean {target}'); plt.title(f'{feature} – Target Rate (Top {len(cats)})')
    plt.show()


def has_target(df: pd.DataFrame, target:str):
    return target is not None and target in df.columns


def numeric_by_category_trend(df: pd.DataFrame, xcol:str, cat:str, target:str = TARGET, q:int = 15, min_count:int = 2000):
    if not has_target(df, target):
        print(f"Dataframe doesn't have a target column {target}.")
        return
        
    # Keep frequent categories to reduce noise
    keep = df[cat].value_counts()
    keep = keep[keep >= min_count].index
    for k in keep:
        d = df[df[cat] == k][[xcol, target]].dropna()
        if d.empty or d[xcol].nunique() <2 : 
            continue
            
        bins = pd.qcut(d[xcol], q = min(q, d[xcol].nunique()), duplicates = 'drop')
        m = d.groupby(bins)[target].mean()

        idx = m.index
        try:
            centers = idx.mid.to_numphy()
        except AttributeError:
            # fallback for very old pandas
            centers = np.array([(iv.left + iv.right) * 0.5 for iv in idx])
            
    plt.plot(centers, m.values, label = str(k))
    plt.title(f'{xcol} → mean({target}) by {cat}')
    plt.xlabel(xcol); 
    plt.ylabel(f'Mean {target}')
    plt.legend(loc='best')
    plt.show()


def cat_cat_heatmap(df: pd.DataFrame, cat1:str, cat2:str, target:str = TARGET, min_count:int = 500):
    if not has_target(df, target):
        print(f"Dataframe doesn't have a target column {target}.")
        return
        
    d = df[[cat1, cat2, target]].dropna()
    g = d.groupby([cat1, cat2])[target].agg(['mean', 'count']).reset_index()
    g = g[g['count'] >= min_count]
    if g.empty:
        print(f'[info] No {cat1}×{cat2} cells with count >= {min_count}.')
        return

    # Normalize labels ONCE to consistent strings
    s1 = g[cat1].astype('string').fillna('<NA>').astype(str)
    s2 = g[cat2].astype('string').fillna('<NA>').astype(str)

    # Stable ordering (by label) or by frequency if preferred
    rows, r_codes = np.unique(s1, return_inverse = True)
    cols, c_codes = np.unique(s2, return_inverse = True)

    A = np.full((len(rows), len(cols)), np.nan, dtype = float)
    A[r_codes, c_codes] = g['mean'].to_numpy()

    plt.figure(figsize=(1.2 * len(cols) + 2, 1.2 * len(rows) + 2))
    im = plt.imshow(A, aspect = 'auto', origin = 'upper')
    cbar = plt.colorbar(im, fraction = 0.046, pad = 0.04, label = f'Mean {target}')
    plt.xticks(np.arange(len(cols)), cols, rotation = 45, ha = 'right')
    plt.yticks(np.arange(len(rows)), rows)
    plt.title(f'{cat1} × {cat2} → mean({target})')
    plt.tight_layout()
    plt.show()


def numeric_numeric_hex(df: pd.DataFrame, x:str, y:str, target:str = TARGET, gridsize:int = 50):
    if not has_target(df, target):
        print(f"Dataframe doesn't have a target column {target}.")
        return
        
    d = df[[x, y, target]].dropna()
    plt.figure(figsize = (7, 5))
    hb = plt.hexbin(d[x].to_numpy(), d[y].to_numpy(), C=d[target].to_numpy(),
                    gridsize = gridsize, reduce_C_function = np.mean)
    plt.xlabel(x); plt.ylabel(y); plt.title(f'{x} × {y} → mean({target})')
    cb = plt.colorbar(hb)
    cb.set_label(f'Mean {target}')
    plt.show()


# Feature signal ranking without modeling
def show_feature_signal_ranking(df: pd.DataFrame, target:str = TARGET):
    if not has_target(df, target):
        print(f"Dataframe doesn't have a target column {target}.")
        return
        
    # Spearman for numeric; ANOVA-style effect for categoricals
    num_cols = df.select_dtypes(include = [np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target]

    scores = []
    for c in num_cols:
        s = df[[c, target]].dropna()
        if s.empty: continue
        rho, p = st.spearmanr(s[c], s[target])
        scores.append((c, float(rho)))
    scores = sorted(scores, key = lambda x: -abs(x[1]))
    print('Top numeric (|Spearman|):', scores[:10])

    cat_cols = [c for c in df.columns if c not in num_cols + [target]]
    effects = []
    for c in cat_cols:
        g = df.groupby(c, observed = True)[target].mean()
        if g.size >= 2:
            rng = float(g.max() - g.min())
            effects.append((c, rng, int(df[c].nunique())))
            
    effects = sorted(effects, key = lambda x: -x[1])
    print('Top categorical (range of mean risk):', effects[:10])


def plot_pairplot(df: pd.DataFrame, num_features: Iterable[str], target:str = TARGET):
    """
    Generates a pair plot for numerical features, colored by the target variable.
    """
    # Increase the 'height' parameter to make the overall plot larger.
    # A good starting point is around 3 to 5.
    sns.pairplot(df[num_features + [target]], hue=target, palette = 'viridis', height = 6)

    plt.suptitle('Pair Plot of Numerical Features', y=1.02)
    plt.show()
    
def plot_boxplots_grid(df: pd.DataFrame, cat_features: Iterable[str], target: str = TARGET, n_cols: int = 3):
    """
    Generates box plots for each categorical feature against the target variable
    in a n_cols -column grid.
    """
    # Define the grid dimensions
    n_features = len(cat_features)

    # Calculate the number of rows needed, using math.ceil to round up
    n_rows = math.ceil(n_features / n_cols)

    # Create the subplot grid
    # The figsize is increased to accommodate the grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (18, n_rows * 5))

    # Flatten the axes array to make it easy to iterate over
    axes = axes.flatten()

    # Loop through the features and plot on the corresponding subplot
    for i, feature in enumerate(cat_features):
        ax = axes[i]
        sns.boxplot(x=feature, y=target, data=df, palette='viridis', ax = ax)
        ax.set_title(f'Box Plot of {target} by {feature}', fontsize = 12)
        ax.tick_params(axis = 'x', labelrotation = 45)

    # Hide any unused subplots
    # This is necessary if the number of features is not a perfect multiple of n_cols
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    # Adjust the layout to prevent titles and labels from overlapping
    plt.tight_layout()
    plt.show()

def plot_violinplots_grid(df: pd.DataFrame, cat_features: Iterable[str], target:str = TARGET, n_cols: int = 3):
    """
    Generates violin plots for each categorical feature against the target variable.
    in a n_cols -column grid.
    """
    # Define the grid dimensions
    n_features = len(cat_features)

    # Calculate the number of rows needed, using math.ceil to round up
    n_rows = math.ceil(n_features / n_cols)

    # Create the subplot grid
    # The figsize is increased to accommodate the grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (18, n_rows * 5))

    # Flatten the axes array to make it easy to iterate over
    axes = axes.flatten()

    # Loop through the features and plot on the corresponding subplot
    for i, feature in enumerate(cat_features):
        ax = axes[i]
        sns.violinplot(x=feature, y=target, data = df, palette = 'viridis', ax = ax)
        ax.set_title(f'Violin Plot of {target} by {feature}', fontsize = 12)
        ax.tick_params(axis = 'x', labelrotation = 45)

    # Hide any unused subplots
    # This is necessary if the number of features is not a perfect multiple of n_cols
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    # Adjust the layout to prevent titles and labels from overlapping
    plt.tight_layout()
    plt.show()        


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r'[,\\$%]', '', regex=True)
         .str.replace(r'\\s+', '', regex=True)
         .replace({'n/a': np.nan, 'na': np.nan, '-': np.nan, '': np.nan}),
        errors='coerce'
    )


def numeric_vs_target_classification(df: pd.DataFrame, numeric_col: str, target_col: str):
    """
    Plots the distribution of a single numeric feature against a categorical target
    using a boxplot and a violin plot.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        numeric_col (str): The name of the numeric feature column.
        target_col (str): The name of the categorical target column.
    """
    
    # --- Input Validation ---
    if numeric_col not in df.columns:
        print(f'[warn] Numeric column "{numeric_col}" not in DataFrame. Skipping plot.')
        return
    if target_col not in df.columns:
        print(f'[warn] Target column "{target_col}" not in DataFrame. Skipping plot.')
        return

    # --- Data Preparation ---
    
    # Create a working copy
    plot_df = df[[numeric_col, target_col]].copy()
    
    # Coerce the numeric column to float, handling potential errors
    plot_df[numeric_col] = _coerce_numeric_series(plot_df[numeric_col])
    
    # Ensure target is treated as a category for plotting
    plot_df[target_col] = plot_df[target_col].astype('category')
    
    # Drop rows where either column is NaN (e.g., from coercion failure)
    plot_df = plot_df.dropna(subset=[numeric_col, target_col])

    if plot_df.empty:
        print(f'[warn] No valid data to plot for {numeric_col} vs {target_col}. Skipping.')
        return

    # --- Plotting ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Boxplot
        sns.boxplot(x=target_col, y=numeric_col, data=plot_df, ax=axes[0], palette='viridis')
        axes[0].set_title(f'Boxplot of {numeric_col}')
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel(numeric_col)
        
        # Violin Plot
        sns.violinplot(x=target_col, y=numeric_col, data=plot_df, ax=axes[1], palette='viridis')
        axes[1].set_title(f'Violin Plot of {numeric_col}')
        axes[1].set_xlabel(target_col)
        axes[1].set_ylabel(numeric_col)
        
        fig.suptitle(f'{numeric_col} vs. {target_col} Distribution', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f'[warn] Plotting failed for {numeric_col} vs {target_col}: {e}')


def is_discrete_numeric(s: pd.Series, max_unique:int = 20):
    return pd.api.types.is_numeric_dtype(s) and s.nunique(dropna = True) <= max_unique

def top_numeric_by_spearman(df: pd.DataFrame, num_cols: Iterable[str], target, k = 6):
    import scipy.stats as st
    scores = []
    for c in num_cols:
        d = df[[c, target]].dropna()
        if d.empty: 
            continue
            
        # skip all-constant / all-equal
        if d[c].nunique() < 2: 
            continue
            
        rho, _ = st.spearmanr(d[c], d[target])
        if np.isfinite(rho):
            scores.append((c, abs(float(rho))))
            
    scores.sort(key=lambda x: -x[1])
    return [c for c,_ in scores[:k]]

def top_categorical_by_range(df: pd.DataFrame, cat_cols: Iterable[str], target:str = TARGET, min_count:int = 200, k:int = 6):
    effects = []
    for c in cat_cols:
        d = df[[c, target]].dropna()
        if d.empty: 
            continue
            
        vc = d[c].value_counts()
        keep = set(vc[vc >= min_count].index)
        if not keep:
            continue
            
        m = d[d[c].isin(keep)].groupby(c)[target].mean()
        if m.size >= 2:
            effects.append((c, float(m.max() - m.min())))
            
    effects.sort(key=lambda x: -x[1])
    return [c for c,_ in effects[:k]]


def run_data_visualization(df: pd.DataFrame, target: str, max_cat_card: int = 30,
            heavy_sample: int = 150_000, num_count: int = 10, cat_count = 10):
    """
    Compact data_ isualization runner for mixed tabular data.
    - Uses new target-aware visuals when target is present.
    - Skips safely on test sets (no target).
    - Limits heavy plots to top-signal features.
    - Optionally downsamples for hexbin/heatmaps.
    """
    # Separate a version without target (works if target absent)
    df_wo_target = df.drop(columns = [target], errors = 'ignore') if target else df

    # Split columns
    num_cols, cat_cols = split_columns(df_wo_target, max_cardinality = max_cat_card)
    
    # Treat booleans as categorical for plotting
    bool_cols = [c for c in df_wo_target.columns if pd.api.types.is_bool_dtype(df_wo_target[c])]
    cat_cols = sorted(set(cat_cols).union(bool_cols))

    # Target-aware (only if target present)
    if not has_target(df, target):
        print('Target not present → skipping target-aware plots.')
        return

    y = df[target]
    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique(dropna = True) > 20

    # Choose top features to keep plots readable and fast
    if is_regression:
        top_nums = top_numeric_by_spearman(df, [c for c in num_cols if c != target], target, k = num_count) or num_cols[:num_count]
        top_cats = top_categorical_by_range(df, cat_cols, target, min_count = 200, k = cat_count) or cat_cols[:cat_count]

        # Pair plots
        display(Markdown('### Pair Plots'))
        plot_pairplot(df, top_nums, target)

        # Box plots
        # Since violin plots show the same, skipping boxplots for now.
        # display(Markdown('### Box Plots'))
        # plot_boxplots_grid(df, top_cats, target)

        # Violin plots
        display(Markdown('### Violin Plots'))
        plot_violinplots_grid(df, top_cats, target)

        # Interactions (small set)
        # numeric × categorical (trend by category)
        if top_nums and top_cats:
            display(Markdown('### Numeric × Categorical (Trend By Category)'))
            numeric_by_category_trend(df, top_nums[0], top_cats[0], target = target, q = 15, min_count = 3000)

        # categorical × categorical (heatmap)
        if len(top_cats) >= 2:
            display(Markdown('### Categorical × Categorical (Heatmap)'))
            cat_cat_heatmap(df, top_cats[0], top_cats[1], target = target, min_count = 1000)

        # numeric × numeric (hexbin target mean)
        if len(top_nums) >= 2:
            display(Markdown('### Numeric × Numeric (Hexbin Target Mean)'))
            d_hex = df
            if heavy_sample is not None and len(df) > heavy_sample:
                d_hex = df.sample(heavy_sample, random_state = seed)
            numeric_numeric_hex(d_hex, top_nums[0], top_nums[1], target = target, gridsize = 50)

    else:
        # Classification-style (few unique target values)
        top_nums = [c for c in num_cols if c != target][:num_count]
        top_cats = cat_cols[:cat_count]

        for c in top_nums:
            try:
                numeric_vs_target_classification(df, c, target)
            except Exception as e:
                print(f'[warn] numeric(classif) plot failed for {c}: {e}')

        for c in top_cats:
            try:
                categorical_vs_target_classification(df, c, target, min_count = 100, top_n = 20)
            except Exception as e:
                print(f'[warn] categorical(classif) plot failed for {c}: {e}')

        # A couple of interactions
        if len(top_nums) and len(top_cats):
            display(Markdown('### Numeric x Category Trend'))
            numeric_by_category_trend(df, top_nums[0], top_cats[0], target = target, q = 12, min_count = 3000)
        if len(top_cats) >= 2:
            display(Markdown('### Category x Category Trend'))
            cat_cat_heatmap(df, top_cats[0], top_cats[1], target = target, min_count = 1500)

    display(Markdown('### Feature Signal Ranking'))
    show_feature_signal_ranking(df)


run_data_visualization(training_df, target = TARGET, cat_count = 0)


run_data_visualization(test_df, target = TARGET, cat_count = 0)


run_data_visualization(original_df, target = TARGET, cat_count = 0)


print('Combining training dataset with the original dataset.')

print(f'Training dataset shape before: {training_df.shape}')

# Copy the dataframes and add source flags to them
trn_df = training_df.copy()
trn_df['is_generated'] = 1
tst_df = test_df.copy()
tst_df['is_generated'] = 1
orig_df = original_df.copy()
orig_df['is_generated'] = 0

# Remove columns from original copy so that it has the same columns as the training dataset.
for col in orig_df.columns:
    if col not in trn_df.columns:
        orig_df = orig_df.drop(col, axis=1, errors='ignore')

# Concatenate
# Only add original data to TRAIN. Never touch TEST.
trn_df = pd.concat([trn_df, orig_df], axis=0)

# Critical Cleanup
# Reset index to prevent the 'sliding' bug
trn_df = trn_df.reset_index(drop=True)

# Handle Missing Targets
# If the original dataset has missing values in the target column, drop them
trn_df = trn_df.dropna(subset=[TARGET])

print(f'Training dataset shape after: {trn_df.shape}')
print('training_df columns', training_df.columns)
print('trn_df columns', trn_df.columns)


class FeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.cat_cols = []
        self.fitted = False

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal helper to create the raw interaction features
        strictly replicating our 'High Scoring' logic from Version 6.
        """
        df_eng = df.copy()
        
        # Drop ID
        if 'id' in df_eng.columns:
            df_eng = df_eng.drop('id', axis=1)

        # Split Grade/Subgrade
        if 'grade_subgrade' in df_eng.columns:
            df_eng['grade'] = df_eng['grade_subgrade'].str[0]
            df_eng['subgrade'] = df_eng['grade_subgrade'].str[1:]
            df_eng = df_eng.drop('grade_subgrade', axis=1)

        # INTERACTIONS
        
        # Score x Rate
        df_eng['score_x_rate'] = df_eng['credit_score'] * df_eng['interest_rate']
        df_eng['rate_to_score_ratio'] = df_eng['interest_rate'] / (df_eng['credit_score'] + 1)
            
        # emp_status_x_rate (String Interaction)
        # Note: qcut might vary slightly between train/test if not careful, 
        # but we follow our exact previous logic here.
        rate_bins = pd.qcut(df_eng['interest_rate'], 4, labels=False, duplicates='drop').astype(str)
        df_eng['emp_status_x_rate'] = df_eng['employment_status'].astype(str) + '_' + rate_bins

        # Emp Status x Score (String Interaction)
        score_bins = pd.qcut(df_eng['credit_score'], 4, labels=False, duplicates='drop').astype(str)
        df_eng['emp_status_x_score'] = df_eng['employment_status'].astype(str) + '_' + score_bins

        # Estimated Total Interest
        df_eng['estimated_total_interest'] = df_eng['loan_amount'] * df_eng['interest_rate']

        # Score x DTI
        df_eng['score_x_dti'] = df_eng['credit_score'] / (df_eng['debt_to_income_ratio'] + 1)
            
        return df_eng
        
    def fit(self, df: pd.DataFrame, target_col=TARGET):
        """
        Learns the Label Encodings from the Training Data.
        """
        
        # Create a temporary copy to generate the columns we need to learn
        temp_df = self._generate_features(df)
        
        # Define the list of categorical columns based on your old code
        self.cat_cols = [
            'gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose',
            'grade', 'subgrade', 'emp_status_x_score', 'emp_status_x_rate'
        ]
        
        # Learn the mapping for each column
        for col in self.cat_cols:
            if col in temp_df.columns:
                # Convert to string to ensure uniformity
                series = temp_df[col].astype(str)
                
                # We use a dictionary for "Safe" encoding (handles unseen values in test)
                # Map distinct values to 0, 1, 2...
                unique_vals = series.unique()
                self.encoders[col] = {val: i for i, val in enumerate(unique_vals)}
        
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("You must call fit() on the training data first!")

        # Generate the features (Interactions, Splits)
        df_eng = self._generate_features(df)
        
        # Apply Label Encoding
        for col in self.cat_cols:
            if col in df_eng.columns:
                # Map values using the dictionary learned in fit()
                # If a value is not found (unseen in test), map to -1
                mapping = self.encoders[col]
                df_eng[col] = df_eng[col].astype(str).map(mapping).fillna(-1).astype(int)
                
                # Optional: Shift -1 to a positive integer if specific models dislike negatives
                # But XGB/LGBM/CatBoost usually handle -1 fine or we can treat it as a distinct group
                
                # Cast to 'category' so LightGBM knows it's discrete
                df_eng[col] = df_eng[col].astype('category')

        return df_eng
        
    def get_cat_feature_indices(self, df: pd.DataFrame):
        """
        Helper to return the integer indices of categorical columns
        for CatBoost/LightGBM parameters.
        """
        return [i for i, col in enumerate(df.columns) if col in self.cat_cols]        


feature_engineer = FeatureEngineer()

# Fit on training data only
feature_engineer.fit(trn_df, target_col=TARGET)

# Transform both
X = feature_engineer.transform(trn_df)
X_test = feature_engineer.transform(tst_df)

# Prepare for Models
y = X[TARGET].astype(float)
X = X.drop([TARGET], axis=1)

# Get Categorical Indices (Crucial for the models!)
# Since we output integers, the models might think they are numbers (1 < 2 < 3).
# We must explicitly tell them "These are categories".
cat_features_indices = feature_engineer.get_cat_feature_indices(X)

print("Categorical Indices:", cat_features_indices)


print(X.head(5))


print(X_test.head(5))


def resolve_lgb_device(use_gpu: bool, X: pd.DataFrame, y: pd.Series,
                        platform_id: int = -1, device_id: int = -1):
    """
    Returns a dict of LightGBM params for the chosen device.
    Tries OpenCL GPU -> CPU.
    """
    if not use_gpu:
        print('GPU use disabled by flag. Falling back to CPU.')
        return {'device_type': 'cpu'}

    try:
        probe = lgb.LGBMClassifier(
            n_estimators=10,
            device_type='gpu',
            gpu_platform_id=platform_id, 
            gpu_device_id=device_id,
            verbosity=-1,
        )
        
        # tiny probe fit to validate the backend works
        _n = min(len(X), 2000)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            probe.fit(X.iloc[:_n], y.iloc[:_n])
            
        print('LightGBM GPU backend successfully validated.')
        return {'device_type': 'gpu',
                'gpu_platform_id': platform_id,
                'gpu_device_id': device_id}
                
    except lgb.LGBMError as e:
        print(f'LightGBM GPU backend validation failed (Error: {e}). Falling back to CPU.')
    except Exception as e:
        print(f'An unexpected error occurred during GPU probe (Type: {type(e).__name__}, Error: {e}). Falling back to CPU.')

    # Simple CPU fallback
    return {'device_type': 'cpu'}

# Set parameter based on whether a gpu is available and requested.
device_params = resolve_lgb_device(USE_GPU, X, y)
has_gpu = device_params['device_type'] == 'gpu'
print(f'USE_GPU = {USE_GPU}; device_params={device_params}')#


def save_optuna_lgb_params(study):
    output_dir = 'export'
    os.makedirs(output_dir, exist_ok=True)

    optuna_lgb_params = study.best_params.copy()
    
    # Save the Optuna tuning parameter
    filepath = os.path.join(output_dir, 'optuna_lgb_params.json')
    with open(f'{filepath}', 'w') as f:
        json.dump(optuna_lgb_params, f, indent=2)

    print('Saved LGBM Optuna tuning parameters')

    return optuna_lgb_params

def load_optuna_lgb_params():
    output_dir = 'export'
    filepath = os.path.join(output_dir, 'optuna_lgb_params.json')
    try:
        with open(filepath, 'r') as f:
            params = json.load(f)
            print('Loaded LGBM Optuna tuning parameters')
            return params
            
    except (FileNotFoundError, Exception):
        params = {
            'use_class_weight': False,
            'learning_rate': 0.039842959087751775,
            'num_leaves': 36, 'min_child_samples': 476,
            'reg_alpha': 4.664659029228782,
            'reg_lambda': 0.00023937463697762932,
            'cat_smooth': 50.630453275002125,
            'cat_l2': 19.431015971962644
        }
        print(f'Failed to load LGBM Optuna tuning parameters; using pre-calculated:\n{params}')
        return params


pruning_disabled = True

def objective_lgb(trial):
    # Broad search: stick to GBDT for speed
    boosting = 'gbdt'

    # One imbalance strategy only (search-mode: fix it to keep space small)
    # Later, you can toggle between class_weight='balanced' and is_unbalance.
    use_class_weight = trial.suggest_categorical('use_class_weight', [True, False])
    class_weight = 'balanced' if use_class_weight else None
    is_unbalance = False if use_class_weight else True

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': boosting,
        'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.6),
        'num_leaves': trial.suggest_int('num_leaves', 30, 60),
        'min_child_samples': trial.suggest_int('min_child_samples', 350, 650),
        'subsample': 1.0,
        'colsample_bytree': 0.6,
        'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'deterministic': True,
        'force_col_wise': True,
        'verbose': -1,
        'n_jobs': -1,
        'seed': seed,
        'enable_categorical': True,
        'cat_smooth': trial.suggest_float('cat_smooth', 10.0, 80.0),
        'cat_l2': trial.suggest_float('cat_l2', 1.0, 20.0),
        **device_params,
    }
    if class_weight is not None:
        params['class_weight'] = class_weight
        params['is_unbalance'] = False

    # 5-fold CV for search speed
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        # Pass categorical_feature explicitly to the Datase
        dtr = lgb.Dataset(
            X_tr, 
            label=y_tr,
            categorical_feature=cat_features_indices
        )
        
        dva = lgb.Dataset(
            X_va, 
            label=y_va, 
            reference=dtr
        )

        booster = lgb.train(params,
                            dtr,
                            num_boost_round=1500,
                            valid_sets=[dva],
                            valid_names=['valid'],
                            callbacks=[
                                lgb.early_stopping(60, verbose=False),  # tighter ES
                                lgb.log_evaluation(period=0),
                            ]
                           )

        pred = booster.predict(X_va, num_iteration=booster.best_iteration)
        fold_auc = roc_auc_score(y_va, pred)
        aucs.append(roc_auc_score(y_va, pred))

        # We optionally disable pruning to ensure volatile (but potentially high-scoring) 
        # low-regularization trials are not killed after Fold 0.
        if pruning_disabled is False:
            trial.report(fold_auc, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(aucs))


best_lgb_params = None

# If user doesn't want tuning performed, try to load the parameters from disk.
if PERFORM_LGB_OPTUNA_TUNING is False:
    print('\nAttempting to load LGBM Optuna Hyperparameters')
    best_lgb_params = load_optuna_lgb_params();
    print(best_lgb_params)
    
if best_lgb_params is None:
    print('\nStarting LGBM Optuna Hyperparameter Search')
    print('============================================')

    # Create study and optimize (maximize AUC)
    study = optuna.create_study(study_name='lgb_loan_payback_prediction_optuna', 
                                direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True))
        

    study.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

    best_lgb_params = study.best_params
    print(f'Best trial finished with AUC: {study.best_value}')
    print('Best hyperparameters:')
    print(best_lgb_params)
    save_optuna_lgb_params(study)


# Clean up the parameters
# Remove the custom flag and translate it to LGBM syntax
final_params = best_lgb_params
use_cw = final_params.pop('use_class_weight', False) # Remove and capture value

lgb_params_tuned = {
    **final_params,
    'class_weight': 'balanced' if use_cw else None, # Translate to LGBM param
    'n_estimators': 10000, # High cap, controlled by Early Stopping
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'n_jobs': -1,
    'seed': seed,
    'enable_categorical': True,
    **device_params, # Ensure this is CPU
}

print('Training LGBM CV Ensemble (10 Folds)')
print('====================================')

# Setup
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
lgb_models = []
lgb_oof_preds = np.zeros(len(X))
lgb_scores = []

# Loop through folds
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Train model for this fold
    clf = lgb.LGBMClassifier(**lgb_params_tuned)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        categorical_feature=cat_features_indices,
        callbacks=[
            lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(0) # Silence logs
        ]
    )
    
    # Save model
    lgb_models.append(clf)
    
    # Record performance
    val_pred = clf.predict_proba(X_va)[:, 1]
    lgb_oof_preds[val_idx] = val_pred
    score = roc_auc_score(y_va, val_pred)
    lgb_scores.append(score)
    
    print(f'Fold {fold+1} AUC: {score:.5f} | Best Iter: {clf.best_iteration_}')

print(f'\nAverage CV AUC: {np.mean(lgb_scores):.5f}')
print(f'Out-of-Fold (OOF) AUC: {roc_auc_score(y, lgb_oof_preds):.5f}')


def save_optuna_cb_params(study):
    output_dir = 'export'
    os.makedirs(output_dir, exist_ok=True)

    optuna_cb_params = study.best_params.copy()
    
    # Save the Optuna tuning parameter
    filepath = os.path.join(output_dir, 'optuna_cb_params.json')
    with open(f'{filepath}', 'w') as f:
        json.dump(optuna_cb_params, f, indent=2)

    print('Saved CatBoost Optuna tuning parameters')
    return optuna_cb_params

def load_optuna_cb_params():
    output_dir = 'export'
    filepath = os.path.join(output_dir, 'optuna_cb_params.json')
    try:
        with open(filepath, 'r') as f:
            params = json.load(f)
            print('Loaded CatBoost Optuna tuning parameters')
            return params
            
    except (FileNotFoundError, Exception):
        params = {
            'use_balanced': False,
            'learning_rate': 0.055950191324294574,
            'one_hot_max_size': 5,
            'depth': 6,
            'grow_policy': 'Lossguide',
            'l2_leaf_reg': 3.871236839298481,
            'random_strength': 3.9509569056171947,
            'min_data_in_leaf': 86,
            'max_leaves': 21,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.6743003097783667
        }
        print(f'Failed to load CatBoost Optuna tuning parameters; using pre-calculated:\n{params}')
        return params


def objective_cb(trial):
    # Toggles
    # Check if we should use balanced weights (LGBM liked this)
    use_balanced = trial.suggest_categorical('use_balanced', [True, False])
    
    # Hyperparameter Search Space
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'bootstrap_type': 'Bernoulli',
        'task_type': 'GPU' if has_gpu else 'CPU',
        'verbose': 0,
        'random_seed': seed,
        'n_estimators': 5000,
        
        'auto_class_weights': 'Balanced' if use_balanced else None,
        
        # Learning rate: often lower is better with more trees
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08),
        
        # One-Hot Encoding Threshold
        # If a category has <= X unique values, use One-Hot. 
        # This forces OHE for 'grade' (7 values), which is usually better.
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 20),
        
        # Architecture
        'depth': trial.suggest_int('depth', 6, 12),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        
        # Regularization
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
        
        # Complexity
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
    }

    # Conditional Params for Lossguide (The LGBM mimic)
    if params['grow_policy'] == 'Lossguide':
        params['max_leaves'] = trial.suggest_int('max_leaves', 16, 64)
        
    # Bayesian Bootstrap (Often better than subsample on GPU)
    # We toggle between Bernoulli (subsample) and Bayesian
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'Bayesian'])
    params['bootstrap_type'] = bootstrap_type
    
    if bootstrap_type == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.5, 0.95)
    else:
        # Bayesian uses bagging_temperature instead of subsample
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
        
    # Cross-Validation Loop (Fast 5-Fold for Tuning)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_pool = Pool(
            X_tr, 
            y_tr, 
            cat_features=cat_features_indices
        )
        val_pool = Pool(
            X_val,
            y_val,
            cat_features=cat_features_indices
        )
        
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
        
    return np.mean(aucs)


best_cb_params = None

# If user doesn't want tuning performed, try to load the parameters from disk.
if PERFORM_CB_OPTUNA_TUNING is False:
    print('\nAttempting to load CatBoost Optuna Hyperparametersh')
    best_cb_params = load_optuna_cb_params();
    print(best_cb_params)
    
if best_cb_params is None:
    print('\nStarting CatBoost Optuna Hyperparameter Search')
    print('================================================')


    # Run Optimization
    study_cb = optuna.create_study(study_name='cb_loan_payback_prediction_optuna', direction='maximize')
    study_cb.optimize(objective_cb, n_trials=30)

    print('Best CatBoost Params:', study_cb.best_params)


    best_cb_params = study_cb.best_params
    print(f'Best trial finished with AUC: {study_cb.best_value}')
    print('Best hyperparameters:')
    print(best_cb_params)
    
    save_optuna_cb_params(study_cb)


print('Training CatBoost Ensemble')
print('==========================')

cb_params = {
    **best_cb_params,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 15000,
    'border_count': 128,
    'random_seed': seed,
    'verbose': 100,
    'allow_writing_files': False,
    'thread_count': -1,
    'bootstrap_type': 'Bernoulli',
    'task_type': 'GPU' if has_gpu else 'CPU'
}

# Capture the helper variable and remove it from the dict
use_balanced_weights = cb_params.pop('use_balanced', False)

# Translate it into the actual CatBoost parameter
# If True, set auto_class_weights to 'Balanced'. If False, remove/set to None.
if use_balanced_weights:
    cb_params['auto_class_weights'] = 'Balanced'
else:
    # Ensure it's not set if False (or set to None)
    cb_params['auto_class_weights'] = None
    
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cb_models = []
cb_scores = []
cb_oof_preds = np.zeros(len(X))

cat_features_indices = np.where(X.dtypes == 'category')[0]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    train_pool = Pool(
        X_tr,
        y_tr,
        cat_features=cat_features_indices
    )
    val_pool = Pool(
        X_va,
        y_va,
        cat_features=cat_features_indices
    )
    
    model = CatBoostClassifier(**cb_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=200,
        use_best_model=True
    )
    
    cb_models.append(model)
    
    # Score
    preds = model.predict_proba(X_va)[:, 1]
    cb_oof_preds[val_idx] = preds
    score = roc_auc_score(y_va, preds)
    cb_scores.append(score)
    print(f'Fold {fold+1} AUC: {score:.5f} | Best Iter: {model.best_iteration_}')

print(f'Average CatBoost AUC: {np.mean(cb_scores):.5f}')


def find_best_blend(lgb_oof, cb_oof, y_true):
    best_score = 0
    best_ratio = 0
    scores = []
    ratios = np.linspace(0.0, 1.00, 101) # Test 0%, 1%, ... 100%
    
    print('Optimizing Blend Ratio...')
    
    for ratio in ratios:
        # Blend: ratio * LGBM + (1-ratio) * NN
        blend = (lgb_oof * ratio) + (cb_oof * (1 - ratio))
        
        score = roc_auc_score(y_true, blend)
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_ratio = ratio
            
    print(f'\nOptimization Complete')
    print(f'Best Ratio: {best_ratio:.2f} LGBM  / {1-best_ratio:.2f} CatBoost')
    print(f'Best AUC:   {best_score:.5f}')
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 5))
    plt.plot(ratios, scores, marker='.', color='blue')
    plt.title('Ensemble Score vs. LightGBM Weight')
    plt.xlabel('LightGBM Weight (1.0 = All LGBM, 0.0 = All CatBoost)')
    plt.ylabel('ROC AUC Score')
    
    # Mark the best point
    plt.axvline(best_ratio, color='red', linestyle='--', label=f'Best: {best_ratio:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_ratio

lightgbm_ratio = find_best_blend(lgb_oof_preds, cb_oof_preds, y)


# INFERENCE
def infer_ensemble(df: pd.DataFrame) -> np.ndarray:
    global lgb_models, cb_models, lightgbm_ratio, feature_engineer
    
    # Get LightGBM-friendly features. 
    fe_df = feature_engineer.transform(df)
    
    # Check if 'fe_df' has an 'is_generated' column.  If not, add one
    if 'is_generated' not in fe_df.columns:
        fe_df['is_generated'] = 1

    # Get predictions from each of the models in the ensemble and average them
    lgb_preds = []
    for model in lgb_models:
        preds = model.predict_proba(fe_df)[:, 1]
        lgb_preds.append(preds)
    avg_lgb = np.mean(lgb_preds, axis=0)

    cb_preds = []
    for model in cb_models:
        preds = model.predict_proba(fe_df)[:, 1]
        cb_preds.append(preds)
    avg_cb = np.mean(cb_preds, axis=0)

    # Blend using the calculated ratio
    final_blend = (avg_lgb * lightgbm_ratio) + (avg_cb * (1 - lightgbm_ratio))
    
    return final_blend


def analyze_feature_importance(X, y):
    """
    Retrains a model on the full dataset using the best params from Optuna
    and plots Gain vs Split importance.
    """
    print('Retraining model with best hyperparameters...')
    
    # LightGBM native API crashes on 'object' dtype. 
    # We must cast them to 'category'.
    X_train = X.copy() # Operate on a copy to not affect the global dataframe
    
    # Identify object columns (strings)
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Also identify existing category columns (to be safe)
    existing_cat_cols = X_train.select_dtypes(include=['category']).columns.tolist()
    
    all_cat_cols = cat_cols + existing_cat_cols
    
    print(f'Converting the following columns to "category": {cat_cols}')
    
    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        
    # Reconstruct the parameter dictionary
    # We must merge the fixed params with the Optuna best params
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'seed': seed, 
        'device_type': 'cpu', # Force CPU as discussed
        **best_lgb_params
    }
    
    # Handle class weights if they were part of the tuning
    if params.get('use_class_weight'):
        params['class_weight'] = 'balanced'
        params['is_unbalance'] = False
        del params['use_class_weight'] # Remove flag not used by LGBM
    else:
        if 'use_class_weight' in params:
            params['is_unbalance'] = True
            del params['use_class_weight']

    # Prepare Dataset
    # Ensure categorical columns are set correctly before training
    # (LightGBM ignores 'categorical_feature' param when data is 'category' dtype)
    train_data = lgb.Dataset(X_train, label=y)

    # Train Final Model
    # No validation set needed here; we are just extracting structural info
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=1000 # Arbitrary high number, we just need the structure
    )

    # Extract Importance
    features = X_train.columns
    importance_gain = booster.feature_importance(importance_type='gain')
    importance_split = booster.feature_importance(importance_type='split')

    df_imp = pd.DataFrame({
        'feature': features,
        'gain': importance_gain,
        'split': importance_split
    })

    # Normalize for visualization (0-100 scale)
    df_imp['gain_log'] = df_imp['gain'] / df_imp['gain'].sum()
    df_imp['split_log'] = df_imp['split'] / df_imp['split'].sum()

    # Sort by Gain (usually the most critical metric)
    df_imp = df_imp.sort_values('gain', ascending=False).reset_index(drop=True)

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # Plot Gain
    sns.barplot(x='gain', y='feature', data=df_imp.head(20), ax=ax[0], palette='viridis')
    ax[0].set_title('Feature Importance (GAIN)\n(The "Quality" of the split)')
    
    # Plot Split
    # Re-sort by split for the second chart
    df_split = df_imp.sort_values('split', ascending=False).head(20)
    sns.barplot(x='split', y='feature', data=df_split, ax=ax[1], palette='mako')
    ax[1].set_title('Feature Importance (SPLIT)\n(The "Frequency" of use)')

    plt.tight_layout()
    plt.show()

    return df_imp




imp_df = analyze_feature_importance(X, y)
    
# Show raw numbers
print(imp_df.head(10))


tst_df.head(10)


print('\nGenerating Submission File')
print('==========================')

# Predict probabilities on the test set
test_predictions = infer_ensemble(tst_df)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': tst_df['id'],  # Use the original tst_df to get the 'id'
    'loan_paid_back': test_predictions
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

print('Submission.csv created successfully!')


submission_df.head()




