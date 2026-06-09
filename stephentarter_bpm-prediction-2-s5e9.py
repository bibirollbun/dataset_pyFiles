%pip install --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON lightgbm


%pip -q install -U optuna
%pip -q install optuna-integration[lightgbm]


# --- Standard library
import os
import sys
import random
import warnings
from pathlib import Path
from contextlib import contextmanager
from time import time

# --- Third-party
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_edf,
)
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from pandas.api.types import (
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, RandomizedSearchCV
from sklearn import set_config
set_config(transform_output="pandas")
from statsmodels.graphics.gofplots import qqplot
from tqdm import tqdm
import lightgbm as lgbm
import xgboost as xgb

# --- Notebook settings
warnings.filterwarnings("ignore")

# Suppress XGBoost warnings about mismatched devices and tree methods.
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost.core')

%matplotlib inline


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
    warnings.filterwarnings("ignore")
    
    # Set pandas display options for nicer output
    pd.options.display.float_format = f"{{:,.{float_precision}f}}".format
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)

    # Set seeds for reproducibility in numpy and the standard random module
    np.random.seed(seed)
    random.seed(seed)
    
    return seed

def running_in_kaggle() -> bool:
    """
    Heuristics that are true in Kaggle notebooks:
    - Special directories exist (/kaggle/input, /kaggle/working)
    - Env var KAGGLE_KERNEL_RUN_TYPE is set
    - The kaggle_secrets module is available
    """
    try:
        if os.path.isdir("/kaggle/input") and os.path.isdir("/kaggle/working"):
            return True
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
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
seed = configure_notebook()    


# Configurable flag to control whether GPU is used
USE_GPU = True

def resolve_lgbm_device(use_gpu: bool, X: pd.DataFrame, y: pd.Series,
                        platform_id: int = -1, device_id: int = -1):
    """
    Returns a dict of LightGBM params for the chosen device.
    Tries OpenCL GPU -> CPU.
    """
    if not use_gpu:
        return {"device_type": "cpu", "force_col_wise": True}  # CPU-only hint

    try:
        probe = lgbm.LGBMRegressor(
            n_estimators=10,
            device_type="gpu",
            max_bin=63,
            gpu_platform_id=platform_id, 
            gpu_device_id=device_id,
            verbosity=-1,
        )
        # tiny probe fit to validate the backend works
        _n = min(len(X), 2000)
        probe.fit(X.iloc[:_n], y.iloc[:_n])
        return {"device_type": "gpu", "gpu_platform_id": platform_id, "gpu_device_id": device_id}
    except Exception as e:
        print("OpenCL not available, falling back to CPU:", type(e).__name__, e)

    # Fall back to CPU
    return {"device_type": "cpu", "force_col_wise": True}


def eda_summary(df: pd.DataFrame) -> None:
    """
    Produce a concise, notebook-friendly Exploratory Data Analysis (EDA) summary.

    This utility prints and displays a standard set of diagnostics for a single
    DataFrame to help you quickly understand schema, completeness, and basic
    statistics. It is designed for use in Jupyter/IPython environments.

    The report includes:
      1. **First 5 rows** (transposed for vertical readability).
      2. **DataFrame info** (`df.info()`): dtypes and non-null counts.
      3. **Numeric describe** (`df.describe()`): count/mean/std/min/percentiles/max.
      4. **Categorical describe** (`df.select_dtypes(['object','category']).describe()`),
         or a note if none exist.
      5. **Missing values summary**: a table with per-column missing count and
         percentage of total rows.
      6. **Duplicate rows**: total number of duplicated records (`df.duplicated().sum()`).
      7. **Data types count**: frequency of each dtype in `df.dtypes`.
      8. **Correlation matrix** for numeric columns (if more than one numeric
         column exists), computed with Pearson correlation.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to summarize.

    Returns
    -------
    None
        Displays/prints the EDA sections and returns nothing.
    """
    # 1. Display the first few rows
    print("======== First 5 Rows ========")
    display(df.head().T)
    
    # 2. DataFrame information (data types, non-null counts, etc.)
    print("\n======== DataFrame Info ========")
    df.info()
    
    # 3. Descriptive statistics for numeric columns
    print("\n======== Descriptive Statistics (Numeric Columns) ========")
    display(df.describe().T)
    
    # 4. Descriptive statistics for categorical columns (if any)
    categorical_df = df.select_dtypes(include=['object', 'category'])
    print("\n======== Descriptive Statistics (Categorical Columns) ========")
    if not categorical_df.empty:
        display(categorical_df.describe())
    else:
        print("No categorical columns found.")
    
    # 5. Missing values summary
    print("\n======== Missing Values Summary ========")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Count": missing,
        "Percentage": missing_percent
    })
    display(missing_summary)
    
    # 6. Count of duplicated rows
    print("\n======== Duplicated Rows ========")
    print(f"Total duplicated rows: {df.duplicated().sum()}")
    
    # 7. Count of each data type
    print("\n======== Data Types Count ========")
    display(df.dtypes.value_counts())
    
    # 8. Correlation matrix for numeric variables (if more than one exists)
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 1:
        print("\n======== Correlation Matrix (Numeric Columns) ========")
        display(numeric_cols.corr())
    else:
        print("\n======== Correlation Matrix ========")
        print("Not enough numeric columns to compute correlation.")
    
    # 9. Value counts for categorical variables with low cardinality
    print("\n======== Value Counts for Categorical Columns (Low Cardinality) ========")
    if not categorical_df.empty:
        for col in categorical_df.columns:
            if df[col].nunique() <= 20:
                print(f"\nValue Counts for '{col}':")
                display(df[col].value_counts())
    else:
        print("No categorical columns found.")



DATA_DIR = Path('/kaggle/input/playground-series-s5e9') if running_in_kaggle() else Path('data')

training_df = pd.read_csv(DATA_DIR / 'train.csv')
training_df.head()


test_df = pd.read_csv(DATA_DIR / 'test.csv')
test_df.head()


def coerce_numeric(df, cols) -> None:
    """
    Convert selected columns to numeric dtype in-place, coercing invalid values to NaN.

    For each column name in ``cols`` that exists in ``df``, this function:
    1) casts the column to string,
    2) removes thousands separators (commas),
    3) applies ``pd.to_numeric(..., errors="coerce")``.

    Any value that cannot be parsed as a number becomes ``NaN``. Columns listed
    in ``cols`` but not present in ``df`` are silently skipped.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose columns will be normalized.
    cols : Iterable[str]
        Column names to coerce to numeric.

    Returns
    -------
    None
        The function mutates ``df`` in-place and returns nothing.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

coerce_numeric(training_df, ["TrackDurationMs"])
coerce_numeric(test_df, ["TrackDurationMs"])


training_df.drop('id', axis=1, inplace=True)
training_df.head()


# Save a copy with the ID for use later
test_ids = test_df['id'].copy()

test_df.drop('id', axis=1, inplace=True)
test_df.head()


eda_summary(training_df)


eda_summary(test_df)


def detect_target(df: pd.DataFrame):
    """
    Infer the target column name from common candidates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to inspect.

    Returns
    -------
    str
        The name of the detected target column.

    Raises
    ------
    KeyError
        If none of the candidate names are present in ``df`` columns.
    """
    candidates = ["BPM", "bpm", "tempo", "Tempo", "target", "BeatsPerMinute"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("Could not detect target column. Set `target_col` manually.")

def numeric_features(df: pd.DataFrame, exclude=None) -> list[str]:
    """
    Return numeric feature column names, optionally excluding some.

    Uses ``DataFrame.select_dtypes(include="number")`` to find numeric dtypes
    (e.g., int, float; typically excludes booleans), then removes any names
    provided in ``exclude``.

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame.
    exclude : iterable of str or None, optional
        Column names to exclude from the result. Defaults to ``None``.

    Returns
    -------
    list of str
        Ordered list of numeric feature column names present in ``df`` and
        not listed in ``exclude``.
    """
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]

def safe_sample(df: pd.DataFrame, n: int = 200_000, seed: int = seed) -> pd.DataFrame:
    """
    Return a sample of up to ``n`` rows without replacement; if ``df`` has
    ``<= n`` rows, return it unchanged.

    This is a convenience helper for large datasets—useful to keep quick EDA
    cells responsive while remaining deterministic via ``seed``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    n : int, optional
        Maximum number of rows to return. Default is ``200_000``.
    seed : int, optional
        Random seed for reproducibility (passed to ``random_state``). Default ``seed``.

    Returns
    -------
    pandas.DataFrame
        A sampled view of ``df`` with at most ``n`` rows, or ``df`` itself if
        it already has ``<= n`` rows.
    """
    if len(df) > n:
        return df.sample(n, random_state=seed)
    return df

def describe_df(df: pd.DataFrame):
    """
    Summarize a modeling DataFrame as (target, numeric_features).

    Detects the target column via :func:`detect_target`, then returns the target
    name and the list of numeric feature columns excluding that target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame expected to contain a recognizable target column.

    Returns
    -------
    (str, list of str)
        A 2-tuple ``(target_name, numeric_feature_names)``.

    Raises
    ------
    KeyError
        Propagated from :func:`detect_target` if no target column is found.
    """
    target = detect_target(df)
    nums   = numeric_features(df, exclude=[target])
    return target, nums


def plot_target_distribution(df: pd.DataFrame, target:str) :
    """
    Plot a histogram of a target column in a DataFrame.

    Computes the number of bins as ``max(20, floor(sqrt(N)))`` where ``N`` is the
    number of rows, then renders a simple histogram using Matplotlib. The plot is
    titled with the target name and labeled on both axes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the target column.
    target : str
        Name of the target column in ``df``.

    Returns
    -------
    None
        Displays the histogram and returns nothing.

    Raises
    ------
    KeyError
        If ``target`` is not a column in ``df``.
    """
    plt.figure(figsize=(8,5))
    x = df[target].to_numpy()
    bins = max(20, int(np.sqrt(len(x))))
    plt.hist(x, bins=bins, edgecolor="black")
    plt.title(f"{target} Distribution")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, num: list[str]) :
    """
    Plot histograms for a prioritized subset of numeric features.

    Given a list of candidate numeric columns, the function selects features
    using a case-insensitive priority list:
    {"energy", "audioloudness", "vocalscore", "vocalcontent", "instrumentalscore",
     "rhythmscore", "acousticquality", "moodscore", "trackdurationms"}.
    If none of the provided names match the priority set, it falls back to the
    first eight names in ``num``.

    For each selected feature, missing values are dropped, an appropriate number
    of bins is chosen as ``max(20, floor(sqrt(N)))``, and a histogram is rendered
    with axis labels and a title.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the candidate feature columns.
    num : list of str
        Names of numeric columns to consider. The function will plot up to eight
        of these, prioritizing the canonical feature names listed above.

    Returns
    -------
    None
        Displays one histogram per selected feature and returns nothing.

    Raises
    ------
    KeyError
        If any selected column name is not present in ``df``.
    TypeError
        If a selected column is non-numeric and cannot be histogrammed.
    """
    priority = {
        "energy", "audioloudness", "vocalscore", "vocalcontent", "instrumentalscore",
        "rhythmscore", "acousticquality", "moodscore", "trackdurationms"
    }
    
    top_view = [c for c in num if c.lower() in priority]
    if not top_view:
        top_view = num[:8]  # fallback

    for col in top_view:
        vals = df[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
            
        plt.figure(figsize=(8,4))
        bins = max(20, int(np.sqrt(len(vals))))
        plt.hist(vals, bins=bins, edgecolor="black")
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()


def plot_feature_target_relationships(df: pd.DataFrame, target:str, num: list[str]):
    """
    Plot hexbin relationships between selected features and the target.

    The function first subsamples up to 200,000 rows (via ``safe_sample``) from
    the columns ``[target] + num`` for responsiveness. It then checks a
    canonical, case-sensitive list of feature→target pairs:

        ("AudioLoudness", target),
        ("Energy", target),
        ("VocalContent", target),
        ("InstrumentalScore", target),
        ("RhythmScore", target),
        ("TrackDurationMs", target)

    Any pair whose columns are present in the sampled DataFrame is plotted as a
    hexbin density scatter with a count colorbar.

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame containing the target and candidate feature columns.
    target : str
        Name of the target column in ``df`` (must be numeric for hexbin).
    num : list of str
        Candidate numeric feature column names. Used to subset the DataFrame
        before plotting (for performance) and to ensure the target is present.

    Returns
    -------
    None
        Displays one hexbin plot per valid (feature, target) pair and returns nothing.

    Raises
    ------
    KeyError
        If any of the columns in ``[target] + num`` are not present in ``df``.
    TypeError or ValueError
        If a plotted column contains non-numeric data incompatible with hexbin.
    """
    sampled = safe_sample(df[[target] + num], n=200_000)

    pairs_to_check = [
        ("AudioLoudness", target),
        ("Energy", target),
        ("VocalContent", target),
        ("InstrumentalScore", target),
        ("RhythmScore", target),
        ("TrackDurationMs", target),
    ]
    pairs_to_check = [(x,y) for x,y in pairs_to_check if x in sampled.columns and y in sampled.columns]

    for x, y in pairs_to_check:
        plt.figure(figsize=(7,6))
        hb = plt.hexbin(sampled[x], sampled[y], gridsize=40, mincnt=1)
        plt.xlabel(x); plt.ylabel(y); plt.title(f"{x} vs. {y} (hexbin)")
        cb = plt.colorbar(hb); cb.set_label("count")
        plt.show()


def plot_missing_values(df: pd.DataFrame):
    """
    Visualize the count of missing values per column as a horizontal bar chart.

    Computes per-column NA counts, filters out columns with zero missing values,
    and plots a horizontal bar chart sorted by descending count. If the DataFrame
    has no missing values, a message is printed instead of plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze for missing values.

    Returns
    -------
    None
        Displays the plot (if any) and returns nothing.
    """
    na_counts = df.isna().sum().sort_values(ascending=False)
    na_counts = na_counts[na_counts > 0]
    if len(na_counts):
        plt.figure(figsize=(8, max(3, len(na_counts)*0.35)))
        plt.barh(na_counts.index.astype(str), na_counts.values)
        plt.title("Missing Values per Column")
        plt.xlabel("Count")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values detected.")


# Show top non-trivial pairs
def top_pairs(corr: pd.Series, k: int = 10):
    """
    Return the top-k strongest (absolute) pairwise correlations from an unstacked correlation Series.

    Expects a Series whose index is a 2-level MultiIndex ``(col_i, col_j)`` and whose
    values are correlation coefficients (usually absolute values), e.g. produced by
    ``df.corr().abs().unstack()``. Self-pairs and symmetric duplicates are removed
    by keeping only entries where ``col_i < col_j``.

    Parameters
    ----------
    corr : pandas.Series
        Series with MultiIndex pairs and correlation values.
    k : int, optional
        Number of top pairs to return. Default is ``10``.

    Returns
    -------
    list of (Hashable, Hashable, float)
        A list of ``(col_i, col_j, value)`` tuples sorted by descending ``value``,
        containing only non-trivial, unique pairs.
    """
    pairs = [(a,b,v) for (a,b),v in corr.items() if a<b]
    return sorted(pairs, key=lambda x: x[2], reverse=True)[:k]

def show_relationship_rank(df: pd.DataFrame):
    """
    Compute and return the strongest pairwise relationships among numeric columns
    using both Pearson (linear) and Spearman (rank/monotonic) correlations.

    The function:
      1) Selects numeric columns via ``df.select_dtypes(include="number")``.
      2) Computes absolute-valued Pearson and Spearman correlation matrices.
      3) Unstacks each matrix to a long Series of ``(col_i, col_j) -> score``.
      4) Summarizes with ``top_pairs(...)`` to keep only unique, non-trivial
         pairs (no self-pairs, no symmetric duplicates) sorted by descending score.
         By default, ``top_pairs`` returns the top 10 pairs for each metric.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing candidate numeric features.

    Returns
    -------
    Tuple[List[Tuple[Hashable, Hashable, float]], List[Tuple[Hashable, Hashable, float]]]
        ``(pearson_top, spearman_top)`` where each element is a list of
        ``(col_i, col_j, score)`` tuples, sorted by descending ``score``.
    """
    num = df.select_dtypes(include="number")
    pearson_top = num.corr().abs().unstack().sort_values(ascending=False)
    spearman_top = num.corr(method="spearman").abs().unstack().sort_values(ascending=False)
    return top_pairs(pearson_top), top_pairs(spearman_top)



t_target, t_num = describe_df(training_df)
print(f"Detected target: {t_target}")
print(f"Numeric features (excluding target): {len(t_num)} columns")


plot_target_distribution(training_df, t_target)


plot_feature_distributions(training_df, t_num)


plot_feature_target_relationships(training_df, t_target, t_num)


plot_missing_values(training_df)


show_relationship_rank(training_df)


ORIGINAL_DIR = Path("/kaggle/input/bpm-prediction-challenge") if running_in_kaggle() else Path("original_data")

original_df = pd.read_csv(ORIGINAL_DIR / 'Train.csv')
original_df


coerce_numeric(original_df, ["TrackDurationMs"])


eda_summary(original_df)


target_col, num_cols = describe_df(original_df)
o_target, o_num = describe_df(original_df)


plot_target_distribution(original_df, o_target)


plot_feature_distributions(original_df, o_num)


plot_feature_target_relationships(original_df, o_target, o_num)


plot_missing_values(original_df)


show_relationship_rank(original_df)


from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

@dataclass
class PandasPreprocessor:
    """
    Pandas-based preprocessing with train-only winsorization, robust transforms,
    and engineered interaction features. It now includes integrated memory reduction.

    Steps on `fit(df)`:
      1) Reduce memory usage of the DataFrame.
      2) Learn per-feature quantile clip thresholds for ALL numeric columns.
      3) Learn robust stats (median/IQR) for scaling.
      4) Learn top N features for selective polynomial interactions.

    Steps on `transform(df)`:
      1) Clip ALL numeric columns using learned quantiles (winsorize).
      2) Overwrite existing columns with specialized transforms.
      3) Create and append selective polynomial interactions.
      4) Ensure all numeric columns are float32 for XGBoost.
      5) Optionally drop the target column for X.

    Parameters
    ----------
    target_col : str or None
        Target column to drop when `drop_target=True` in `transform`.
    duration_col : str
        Column name for track duration in milliseconds.
    loudness_col : str
        Column name for loudness in dB (typically negative values).
    winsor_q : (float, float)
        Lower/upper quantiles for winsorization across numeric columns.
    eps : float
        Small constant for safe division in ratios.
    add_interactions : bool
        Whether to append engineered interaction features in `transform`.
    long_quantile : float
        Quantile threshold for `long_track` (applied on log1p(duration)).
    high_energy_threshold : float
        Threshold for `high_energy = ReLU(Energy - high_energy_threshold)`.
    low_energy_threshold : float
        Threshold for `low_energy  = ReLU(low_energy_threshold - Energy)`.
    poly_features_degree : int
        Degree for polynomial features (e.g., 2 for squared and interactions).
    poly_features_n : int
        Number of top features to use for generating polynomial interactions.
    use_float16 : bool
        Whether to use float16 for memory reduction (can be less precise).

    Attributes learned on fit
    -------------------------
    numeric_cols_ : list[str]
        Numeric feature columns (excluding `target_col`).
    q_low_, q_high_ : pd.Series
        Per-column winsorization thresholds.
    duration_median_, duration_iqr_ : float
        Robust stats for log1p(duration) scaling.
    loud_median_, loud_iqr_ : float
        Robust stats for -AudioLoudness scaling.
    long_duration_thresh_ : float
        Learned threshold for `long_track`.
    instr_floor_: float
        Learned floor for instrumental score.
    v2i_low_, v2i_high_: float
        Learned clipping thresholds for vocal-to-instrumental ratio.
    top_features_: List[str]
        Column names of the top features identified for polynomial interactions.
    """
    target_col: Optional[str] = None
    duration_col: str = "TrackDurationMs"
    loudness_col: str = "AudioLoudness"
    winsor_q: Tuple[float, float] = (0.002, 0.998)
    eps: float = 1e-9
    ratio_floor_quantile: float = 0.05      # floor denom at this train quantile
    ratio_clip_q: Tuple[float, float] = (0.01, 0.99) # clip raw ratio to these train quantiles

    # engineered-feature controls
    add_interactions: bool = True
    long_quantile: float = 0.95
    high_energy_threshold: float = 0.70
    low_energy_threshold: float = 0.30

    # New: Polynomial Feature controls
    poly_features_degree: int = 2
    poly_features_n: int = 6
    use_float16: bool = False

    # learned parameters (filled on fit)
    numeric_cols_: List[str] = field(default_factory=list)
    q_low_: pd.Series = field(default=None)
    q_high_: pd.Series = field(default=None)
    duration_median_: float = 0.0
    duration_iqr_: float = 1.0
    loud_median_: float = 0.0
    loud_iqr_: float = 1.0
    long_duration_thresh_: float = 0.0
    instr_floor_: float = 0.0
    v2i_low_: float = 0.0
    v2i_high_: float = 0.0
    top_features_: List[str] = field(default_factory=list)

    # ------------------------ internals ------------------------

    def _num_cols(self, df: pd.DataFrame) -> List[str]:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in cols:
            cols.remove(self.target_col)
        return cols

    def _has(self, df: pd.DataFrame, name: str) -> bool:
        return name in df.columns

    def _reduce_mem_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col in df.columns:
            if is_datetime(df[col]) or is_categorical_dtype(df[col]):
                # skip datetime type or categorical type
                continue
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64) 
                else:
                    if self.use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
        return df
        
    def _add_v2i_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add vocal/instrumentation ratio features using *training-learned* stats:
        - self.q_low_, self.q_high_
        - self.instr_floor_, self.v2i_low_, self.v2i_high_
        """
        if not (self._has(df, "VocalContent") and self._has(df, "InstrumentalScore")):
            return df

        # need a copy since we'll be adding columns
        out = df.copy()
    
        v = out["VocalContent"].clip(self.q_low_["VocalContent"], self.q_high_["VocalContent"])
        i = out["InstrumentalScore"].clip(self.q_low_["InstrumentalScore"], self.q_high_["InstrumentalScore"])

        # use training-learned floor/quantiles (already computed in fit)
        denom = np.maximum(i, self.instr_floor_ + self.eps)
        ratio = v / denom
        ratio = ratio.clip(self.v2i_low_, self.v2i_high_)

        out["vocal_to_instr"] = ratio
        out["vocal_to_instr_log1p"] = np.log1p(ratio)
        out["vocal_to_instr_asinh"] = np.arcsinh(ratio)
        return out

    # ------------------------ API ------------------------

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        
        # New: Reduce memory usage first
        df = self._reduce_mem_usage(df)

        self.numeric_cols_ = self._num_cols(df)

        # --- winsorization thresholds on ALL numeric cols ---
        q_low, q_high = self.winsor_q
        self.q_low_ = df[self.numeric_cols_].quantile(q_low)
        self.q_high_ = df[self.numeric_cols_].quantile(q_high)

        # --- robust stats for VocalContent / InstrumentalScore ratio (train-only) ---
        if self._has(df, "VocalContent") and self._has(df, "InstrumentalScore"):
            v = df["VocalContent"].clip(self.q_low_["VocalContent"], self.q_high_["VocalContent"])
            i = df["InstrumentalScore"].clip(self.q_low_["InstrumentalScore"], self.q_high_["InstrumentalScore"])

            # denominator floor at a train quantile (e.g., 5th percentile)
            self.instr_floor_ = float(np.quantile(i, self.ratio_floor_quantile))
            denom = np.maximum(i, self.instr_floor_ + self.eps)

            ratio = v / denom
            ql, qh = self.ratio_clip_q
            self.v2i_low_ = float(np.quantile(ratio, ql))
            self.v2i_high_ = float(np.quantile(ratio, qh))

            # Add more features based on vocal to instrumentation
            df = self._add_v2i_features(df)
        
        # --- robust stats for duration (after log1p of CLIPPED duration) ---
        if self._has(df, self.duration_col):
            dur_clipped = df[self.duration_col].clip(self.q_low_[self.duration_col],
                                                     self.q_high_[self.duration_col])
            dur_log = np.log1p(dur_clipped)
            self.duration_median_ = float(dur_log.median())
            self.duration_iqr_ = float(dur_log.quantile(0.75) - dur_log.quantile(0.25)) or 1.0
            # long-track threshold learned on train
            self.long_duration_thresh_ = float(dur_log.quantile(self.long_quantile))

        # --- robust stats for loudness (after negation of CLIPPED loudness) ---
        if self._has(df, self.loudness_col):
            loud_clipped = df[self.loudness_col].clip(self.q_low_[self.loudness_col],
                                                      self.q_high_[self.loudness_col])
            loud_mag = -loud_clipped
            self.loud_median_ = float(loud_mag.median())
            self.loud_iqr_ = float(loud_mag.quantile(0.75) - loud_mag.quantile(0.25)) or 1.0
            
        # --- NEW: Learn top features for selective polynomial interactions ---
        if self.add_interactions:
            # Drop the target column to train the model
            X_temp = df.drop(columns=[self.target_col])
            y_temp = df[self.target_col].copy()

            # Train a quick, simple model to get feature importances.
            temp_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                tree_method='hist',
                device='cuda' if USE_GPU else 'cpu',
                seed=seed
            )
            temp_model.fit(X_temp, y_temp)

            # Get feature importances and sort them
            importances = temp_model.feature_importances_
            feature_names = X_temp.columns
            sorted_indices = np.argsort(importances)[::-1]
            self.top_features_ = feature_names[sorted_indices][:self.poly_features_n].tolist()

        return self

    def transform(self, df: pd.DataFrame, drop_target: bool = True) -> pd.DataFrame:
        df = df.copy()

        # 1) winsorize all numeric columns in-place
        for c in self.numeric_cols_:
            lo = self.q_low_[c]
            hi = self.q_high_[c]
            df[c] = df[c].clip(lo, hi)

        # 2) specialized transforms (overwrite columns; also create base features)
        # duration: log1p -> robust scale; also expose raw log_duration
        if self._has(df, self.duration_col):
            dur_log_raw = np.log1p(df[self.duration_col])  # after winsorization
            df["log_duration"] = dur_log_raw
            df[self.duration_col] = (dur_log_raw - self.duration_median_) / (self.duration_iqr_ + self.eps)

        # loudness: negate -> robust scale; also expose loud_mag (pre-scale)
        if self._has(df, self.loudness_col):
            loud_mag_raw = -df[self.loudness_col]  # after winsorization
            df["loud_mag"] = loud_mag_raw
            df[self.loudness_col] = (loud_mag_raw - self.loud_median_) / (self.loud_iqr_ + self.eps)

        # 3) engineered features
        if self.add_interactions:
            # helpers
            def has(name: str) -> bool:
                return name in df.columns

            def add_prod(a: str, b: str, name: str):
                if has(a) and has(b):
                    df[name] = df[a] * df[b]

            # simple curvature
            if has("Energy"):
                df["Energy_sq"] = df["Energy"] ** 2

            # pairwise products
            add_prod("Energy", "RhythmScore", "drive")
            add_prod("Energy", "LivePerformanceLikelihood", "stage")
            if has("Energy") and has("loud_mag"):
                df["bright"] = df["Energy"] * df["loud_mag"]
            add_prod("Energy", "MoodScore", "mood_energy")
            add_prod("AcousticQuality", "Energy", "acoustic_energy")
            add_prod("RhythmScore", "MoodScore", "rhythm_mood")

            # contrasts / ratios
            if has("VocalContent") and has("InstrumentalScore"):
                v = df["VocalContent"]
                i = df["InstrumentalScore"]

                # 1) raw ratio with train-based floor + clipping to train quantiles
                denom = np.maximum(i, self.instr_floor_ + self.eps)
                ratio = (v / denom).clip(self.v2i_low_, self.v2i_high_)
                df["vocal_to_instr"] = ratio

                # 2) smooth, symmetric log-ratio (finite at zeros)
                df["vocal_log_ratio"] = np.log1p(v) - np.log1p(i + self.eps)

                # 3) bounded relative difference in [-1, 1]
                df["vocal_rel"] = (v - i) / (v + i + self.eps)
                df["vocal_minus_instr"] = df["VocalContent"] - df["InstrumentalScore"]

                df = self._add_v2i_features(df)
                
            if has("Energy") and has("AcousticQuality"):
                df["energy_minus_acoustic"] = df["Energy"] - df["AcousticQuality"]

            # piecewise (hinges) on Energy
            if has("Energy"):
                df["high_energy"] = np.maximum(df["Energy"] - self.high_energy_threshold, 0.0)
                df["low_energy"]  = np.maximum(self.low_energy_threshold - df["Energy"], 0.0)

            # long-track indicator based on TRAIN-learned threshold
            if has("log_duration") and self.long_duration_thresh_ > 0.0:
                df["long_track"] = (df["log_duration"] > self.long_duration_thresh_).astype("int8")

        # 4) Create Selective Polynomial Features
        if self.add_interactions and len(self.top_features_) > 0:
            # Select only the top features from the original DataFrame
            X_top = df[self.top_features_]

            # Use PolynomialFeatures on this smaller, selected set of features
            poly = PolynomialFeatures(
                degree=self.poly_features_degree, 
                interaction_only=True,   # <- keeps x_i * x_j, drops x_i^2
                include_bias=False
            )
            poly_features = poly.fit_transform(X_top)
            
            # Get the new column names and convert back to a DataFrame
            feature_names_out = poly.get_feature_names_out(input_features=X_top.columns)
            
            # Create a DataFrame from the polynomial features
            X_interactions = pd.DataFrame(poly_features, columns=feature_names_out, index=df.index)

            # Get a list of the new columns that are not duplicates of the original columns
            new_poly_cols = [col for col in X_interactions.columns if col not in df.columns]

            # Combine with the full original feature set
            df = pd.concat([df, X_interactions[new_poly_cols]], axis=1)

        # 5) Ensure all numeric columns are float32 for XGBoost
        # This will now work correctly since we have unique column names.
        for col in df.columns:
            if df[col].dtype.kind in 'fi':
                df[col] = df[col].astype(np.float32)

        # 6) optionally drop target for X
        if drop_target and self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        return df

    def fit_transform(self, df: pd.DataFrame, drop_target: bool = True) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, drop_target=drop_target)



pp = PandasPreprocessor(
    target_col="BeatsPerMinute",
    add_interactions=True,        # turn off if you want a pure baseline
    long_quantile=0.95,
    high_energy_threshold=0.70,
    low_energy_threshold=0.30,
)

X_train = pp.fit_transform(training_df, drop_target=True)

# Try windsorizing the target data ar 0.05 and 99.5 percentiles
y_train = training_df["BeatsPerMinute"].copy()
y_q_low = y_train.quantile(0.05 / 100.0)
y_q_high = y_train.quantile(99.95 / 100.0)
y_train = y_train.clip(y_q_low, y_q_high)

# Create a validation set with the original data
X_valid = pp.transform(original_df, drop_target=True)
y_valid = original_df["BeatsPerMinute"].copy()

# For validation/test, apply the *same* fitted stats:
X_test = pp.transform(test_df, drop_target=False)


eda_summary(X_train)


# Target transform helpers
def to_model_space(y, target_mode: str = "raw"):
    if target_mode == "raw":
        return y
    if target_mode == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    if target_mode == "log1p":
        return np.log1p(np.clip(y, 0, None))
    raise ValueError(f"Unknown target_mode: {target_mode}")

def from_model_space(p, target_mode: str = "raw"):
    if target_mode == "raw":
        return p
    if target_mode == "sqrt":
        return np.clip(p, 0, None) ** 2
    if target_mode == "log1p":
        # inverse of log1p; clip to keep BPM non-negative
        return np.clip(np.expm1(p), 0, None)
    raise ValueError(f"Unknown target_mode: {target_mode}")


def rmse_on_original_scale(preds, dmatrix, target_mode='raw'):
    y_true = dmatrix.get_label()           # ORIGINAL BPM labels
    y_pred_bpm = from_model_space(preds, target_mode)
    rmse = float(np.sqrt(np.mean((y_pred_bpm - y_true) ** 2)))
    return "rmse_bpm", rmse


# ==== Optuna tuning for XGBoost (gbtree) on BPM with your custom RMSE ====
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import numpy as np
import pandas as pd
from time import time

# Using raw target_mode for tuning.
target_mode = "raw"

# Safety: ensure bins exist (same approach you already use)
def _ensure_bins(y, n_bins=12):
    try:
        return pd.qcut(y, q=n_bins, duplicates="drop").astype(str)
    except Exception:
        # fallback to uniform bins if qcut fails in rare edge cases
        return pd.cut(y, bins=n_bins, include_lowest=True).astype(str)

bins_for_cv = bins if 'bins' in globals() else _ensure_bins(y_train, n_bins=12)

N_SPLITS = 5
CV_SEED = 2025  # independent of your training seeds
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

def _objective(trial: optuna.trial.Trial) -> float:
    # --- Suggest hyperparameters (focused, reasonably wide ranges) ---
    params = {
        "objective": "reg:squarederror",
        "disable_default_eval_metric": 1,
        "tree_method": "gpu_hist" if USE_GPU else "hist",
        # Learning / depth
        "eta": trial.suggest_float("eta", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        # Column / row sampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        # Regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 6.0),
        # Split penalty
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        # Node-wise sampling (helps generalization on some datasets)
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        # Use single-precision on GPU to speed up (harmless on CPU, ignored when unsupported)
        "gpu_id": 0,
        "max_bin": trial.suggest_int("max_bin", 128, 512),
    }

    # We’ll let early stopping find best n_estimators per fold
    num_boost_round = 12000
    early_stopping_rounds = 500
    verbose_eval = False

    # CV
    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, bins_for_cv), start=1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        # Train-on transformed labels; VALIDATION LABELS are raw BPM for metric
        dtr = xgb.DMatrix(X_tr, label=to_model_space(y_tr, target_mode))
        dva = xgb.DMatrix(X_va, label=y_va)

        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            evals=[(dva, "valid")],
            feval=rmse_on_original_scale,     # returns ("rmse_bpm", value)
            maximize=False,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        # XGBoost stores custom metric as best_score on the eval set
        fold_scores.append(booster.best_score)

        # Optional: prune if trial is clearly underperforming
        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))

# Study configuration
study_name = "xgb_bpm_optuna"
sampler = TPESampler(seed=CV_SEED)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)

study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name=study_name)
t0 = time()
study.optimize(_objective, n_trials=60, show_progress_bar=True)
t1 = time()

print(f"[Optuna] Completed {len(study.trials)} trials in {t1 - t0:.1f}s")
print("Best value (CV RMSE on BPM):", study.best_value)
print("Best params:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")

# Compose a ready-to-use params dict for your training cell
xgb_best_params = {
    "objective": "reg:squarederror",
    "disable_default_eval_metric": 1,
    "tree_method": "gpu_hist" if USE_GPU else "hist",
    **study.best_trial.params,
}
xgb_best_params


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# inverse lookup
INV = {
    "raw":   lambda p: p,
    "sqrt":  lambda p: np.clip(p, 0, None) ** 2,
    "log1p": lambda p: np.clip(np.expm1(p), 0, None),
}

# See if training with raw, sqrt, or log1p transformations perform best.
def cv_rmse_for_target(target_mode: str) -> float:
    oof = np.zeros(len(y_train))
    inv = INV[target_mode]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    bins = pd.qcut(y_train, q=12, duplicates="drop").astype(str)

    for fold, (tr, va) in enumerate(skf.split(X_train, bins)):
        y_tr_t = to_model_space(y_train.iloc[tr], target_mode).values

        dtr = xgb.DMatrix(X_train.iloc[tr], label=y_tr_t)
        dva = xgb.DMatrix(X_train.iloc[va],  label=y_train.iloc[va])  # ORIGINAL for feval

        booster = xgb.train(
            {
                "objective":"reg:squarederror",
                "disable_default_eval_metric":1,
                "eta":0.05, 
                "max_depth":7, 
                "subsample":0.9, 
                "colsample_bytree":0.9,
                "lambda":1.0, 
                "alpha":0.0, 
                "tree_method":"gpu_hist" if USE_GPU else "hist",
                "seed": 999 + fold
            },
            dtr, 
            20000, 
            evals=[(dva,"valid")],
            feval=lambda p, d: ("rmse_bpm", float(np.sqrt(np.mean((inv(p) - d.get_label())**2)))),
            maximize=False, 
            early_stopping_rounds=800, 
            verbose_eval=False
        )
        oof[va] = inv(booster.predict(dva, iteration_range=(0, booster.best_iteration+1)))

    return np.sqrt(mean_squared_error(y_train.values, oof))
# Uncomment to perform comparisons of scores if the Y is left raw, or transformed by sqrt or log1p
# for target_mode in ["raw","sqrt","log1p"]:
#     print(target_mode, cv_rmse_for_target(target_mode))



target_mode = "raw"


def make_bins(y, n_bins=12):
    # quantile bins
    return pd.qcut(y, q=n_bins, duplicates="drop").astype(str)

bins = make_bins(y_train, n_bins=12)

seeds = [7, 42, 2025, 10301, 19930619]
oof_blend = np.zeros(len(y_train), dtype=np.float64)
test_blend = np.zeros(len(X_test), dtype=np.float64)
fold_models = []          # list[(seed, fold, booster, best_iter)]

for s in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    oof = np.zeros(len(y_train), dtype=np.float64)
    test_fold_preds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, bins)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        dtr = xgb.DMatrix(X_tr, label=to_model_space(y_tr, target_mode))
        # EVAL on ORIGINAL BPM for early stopping
        dva  = xgb.DMatrix(X_va, label=y_va)

        # Use Optuna-selected values
        params = dict(xgb_best_params)

        booster = xgb.train(
            params, 
            dtr, 
            num_boost_round=20000,
            evals=[(dva, "valid")],
            feval=rmse_on_original_scale,
            maximize=False, 
            early_stopping_rounds=800,
            verbose_eval=100
        )

        # OOF in original BPM
        preds_va_model = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        oof[va_idx] = from_model_space(preds_va_model, target_mode)

        fold_models.append((s, fold, booster, booster.best_iteration))
        
        # Keep per-fold test preds for fold-avg
        dte = xgb.DMatrix(X_test)
        test_fold_preds.append(from_model_space(
            booster.predict(dte, iteration_range=(0, booster.best_iteration+1)),
            target_mode
        ))

    oof_rmse = float(np.sqrt(np.mean((oof - y_train.values) ** 2)))
    print(f"[seed {s}] OOF RMSE = {oof_rmse:.5f}")
    oof_blend += oof / len(seeds)
    test_blend += np.mean(test_fold_preds, axis=0) / len(seeds)

# Final blended OOF RMSE
final_oof_rmse = float(np.sqrt(np.mean((oof_blend - y_train.values) ** 2)))
print("OOF (seed-avg) RMSE:", final_oof_rmse)

# Create submission
submission = pd.DataFrame({"id": test_ids, "y": test_blend})


def predict_new(X_new, fold_models, target_mode: str = target_mode):
    """
    fold_models: iterable of (seed, fold, booster, best_iteration[, sqrt_flag])
    target_mode: How the target is encoded: raw, sqrt, or log1p
    """
    dnew = xgb.DMatrix(X_new)
    preds = []
    for item in fold_models:
        # support tuples of len 4 or 5 (with per-model sqrt flag)
        if len(item) == 5:
            _, _, booster, best_it, sqrt_flag = item
        else:
            _, _, booster, best_it = item

        p = booster.predict(dnew, iteration_range=(0, best_it + 1))
        p = from_model_space(p)
        preds.append(p)
    return np.mean(preds, axis=0)


# Use the Original data set as a validation set.
val_preds = predict_new(X_valid, fold_models, target_mode=target_mode)
rmse = np.sqrt(mean_squared_error(y_valid, val_preds))
print(f"Final model RMSE: {rmse:.5f}")


# Build submission
submission = pd.DataFrame({"id": test_ids, "y": test_blend})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
submission.head()




