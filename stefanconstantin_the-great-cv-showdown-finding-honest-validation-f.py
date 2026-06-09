import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np

from scipy.stats import pearsonr
from sklearn.model_selection import BaseCrossValidator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, TimeSeriesSplit, ShuffleSplit
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

import xgboost as xgb
import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

import inspect
import logging
from typing import Iterator, Tuple, Optional


train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')

selected_features = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
]

X_df = train_df[selected_features]
y_df = train_df["label"]

n_samples = y_df.shape[0]
n_samples_per_part = n_samples // 2

X_train, X_test = X_df[:n_samples_per_part], X_df[n_samples_per_part:]
y_train, y_test = y_df[:n_samples_per_part], y_df[n_samples_per_part:]


class TimeSeriesCV(BaseCrossValidator):
    """
    Custom time series cross-validation for sequential data tasks.

    For each fold:
    1. Select a random contiguous window within the data (of at least min_train_test_size samples).
    2. Split the window into two halves: the first half for training, the second half for validation.

    This approach preserves temporal order and prevents lookahead data leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_test_size: int = 100000,
        random_state: Optional[int] = None
    ):
        """
        Initialize the time series cross-validator.

        Args:
            n_splits: Number of cross-validation folds.
            min_train_test_size: Minimum total samples (train + test) required in each random window.
            random_state: Seed for reproducible random windows.
        """
        self.n_splits = n_splits
        self.min_train_test_size = min_train_test_size
        self.random_state = random_state

    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature array of shape (n_samples, ...).
            y: Optional target array (unused).
            groups: Optional group labels (unused).

        Yields:
            train_idx: Indices for training (earlier half of window).
            test_idx: Indices for validation (later half of window).

        Raises:
            ValueError: If total samples less than min_train_test_size.
        """
        n_samples = len(X)
        if n_samples < self.min_train_test_size:
            raise ValueError(
                f"Not enough samples: need at least {self.min_train_test_size}, got {n_samples}."
            )

        rng = np.random.RandomState(self.random_state)

        for fold in range(self.n_splits):
            # Determine random window size at least min size
            window_size = rng.randint(
                low=self.min_train_test_size,
                high=n_samples + 1
            )
            # Randomly pick start index so window fits
            start = rng.randint(0, n_samples - window_size + 1)
            end = start + window_size

            # Split window halfway
            mid = start + window_size // 2
            train_idx = np.arange(start, mid)
            test_idx = np.arange(mid, end)

            logger.debug(
                f"Fold {fold + 1}: train[{start}:{mid}] ({len(train_idx)} samples), "
                f"test[{mid}:{end}] ({len(test_idx)} samples)"
            )
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None
    ) -> int:
        """
        Return the number of folds.

        Args:
            X: not used.
            y: not used.
            groups: not used.

        Returns:
            Number of splitting iterations.
        """
        return self.n_splits


import numpy as np
from typing import Iterator, Tuple
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)

class WalkForwardCV(BaseCrossValidator):
    """
    Walk-forward cross-validation for time series using:
      train_size = floor(L / ((1 - o)*(n_splits - 1) + (1 + f)))
      test_size = floor(f * train_size)
      step_size = floor(train_size * (1 - o)), at least 1

    Parameters
    ----------
    n_splits : int
        Number of folds (n in formula). Must be >= 1.
    overlap_percentage : float
        o in [0.0, 1.0). Fractional overlap between consecutive train windows.
    test_train_ratio : float
        f > 0. Ratio of test_size to train_size.
    """
    def __init__(
        self,
        n_splits: int = 1,
        overlap_percentage: float = 0.8,
        test_train_ratio: float = 1.0
    ):
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("n_splits must be integer >= 1")
        if not (0.0 <= overlap_percentage < 1.0):
            raise ValueError("overlap_percentage must be in [0.0, 1.0)")
        if test_train_ratio <= 0:
            raise ValueError("test_train_ratio must be > 0")
        self.n_splits = n_splits
        self.overlap_percentage = overlap_percentage
        self.test_train_ratio = test_train_ratio

    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        L = len(X)
        if L <= 0:
            return
        o = self.overlap_percentage
        n = self.n_splits
        f = self.test_train_ratio

        # Compute train_size by formula
        denom = (1 - o) * (n - 1) + (1 + f)
        train_size = int(np.floor(L / denom))
        if train_size < 1:
            raise ValueError(
                f"Computed train_size < 1: with L={L}, n_splits={n}, "
                f"overlap={o}, test_train_ratio={f}. Adjust parameters or data size."
            )
        test_size = int(np.floor(f * train_size))
        if test_size < 1:
            test_size = 1
        step_size = int(np.floor(train_size * (1 - o)))
        if step_size < 1:
            step_size = 1

        start = 0
        fold = 0
        while True:
            train_start = start
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size

            if test_end > L:
                break
            if fold >= n:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            logger.debug(
                f"Fold {fold+1}: train[{train_start}:{train_end}] ({len(train_idx)}), "
                f"test[{test_start}:{test_end}] ({len(test_idx)})"
            )
            yield train_idx, test_idx

            start += step_size
            fold += 1

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None
    ) -> int:
        if X is None:
            raise ValueError("X must be provided to compute n_splits")
        L = len(X)
        if L <= 0:
            return 0
        o = self.overlap_percentage
        n = self.n_splits
        f = self.test_train_ratio

        denom = (1 - o) * (n - 1) + (1 + f)
        train_size = int(np.floor(L / denom))
        if train_size < 1:
            return 0
        test_size = int(np.floor(f * train_size))
        if test_size < 1:
            test_size = 1
        step_size = int(np.floor(train_size * (1 - o)))
        if step_size < 1:
            step_size = 1

        # count how many folds fit: for fold i starting at start=i*step_size,
        # require start + train_size + test_size <= L
        max_fits = int(np.floor((L - train_size - test_size) / step_size)) + 1
        return min(max_fits, n)



from sklearn.model_selection import TimeSeriesSplit


from sklearn.model_selection import KFold


from sklearn.model_selection import ShuffleSplit


class WalkForwardShuffleCV(BaseCrossValidator):
    """
    Walk-forward cross-validation with subsampling within each fold.

    For each walk-forward fold (defined by n_splits, overlap_percentage, test_train_ratio),
    we randomly sample a fraction of the training indices and a fraction of the test indices,
    repeating this subsampling multiple times per fold.

    Parameters
    ----------
    n_splits : int >= 1
        Number of walk-forward folds.
    overlap_percentage : float in [0.0, 1.0)
        Fractional overlap between consecutive train windows (o).
    test_train_ratio : float > 0
        Ratio f = test_size / train_size.
    frac : float in (0, 1]
        Fraction of each fold’s train indices and test indices to keep when subsampling.
    n_repeats_per_fold : int >= 1
        How many random subsamples to draw per fold.
    random_state : int or None
        Seed for reproducibility. Different repeats use the same RNG instance.
    """
    def __init__(
        self,
        n_splits: int = 1,
        overlap_percentage: float = 0.8,
        test_train_ratio: float = 1.0,
        frac: float = 0.5,
        n_repeats_per_fold: int = 1,
        random_state: int = None
    ):
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("n_splits must be integer >= 1")
        if not (0.0 <= overlap_percentage < 1.0):
            raise ValueError("overlap_percentage must be in [0.0, 1.0)")
        if test_train_ratio <= 0:
            raise ValueError("test_train_ratio must be > 0")
        if not (0.0 < frac <= 1.0):
            raise ValueError("frac must be in (0, 1]")
        if not isinstance(n_repeats_per_fold, int) or n_repeats_per_fold < 1:
            raise ValueError("n_repeats_per_fold must be integer >= 1")

        self.n_splits = n_splits
        self.overlap_percentage = overlap_percentage
        self.test_train_ratio = test_train_ratio
        self.frac = frac
        self.n_repeats_per_fold = n_repeats_per_fold
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Yield subsampled train/test indices.
        Total yields = (actual_folds) * n_repeats_per_fold.
        """
        L = len(X)
        if L <= 0:
            return
        o = self.overlap_percentage
        n = self.n_splits
        f = self.test_train_ratio
        frac = self.frac
        repeats = self.n_repeats_per_fold

        # Prepare RNG
        rng = np.random.RandomState(self.random_state)

        # Compute base train_size, test_size, step_size as in WalkForwardCV
        denom = (1 - o) * (n - 1) + (1 + f)
        train_size = int(np.floor(L / denom))
        if train_size < 1:
            raise ValueError(
                f"Computed train_size < 1: L={L}, n_splits={n}, "
                f"overlap={o}, test_train_ratio={f}"
            )
        test_size = int(np.floor(f * train_size))
        if test_size < 1:
            test_size = 1
        step_size = int(np.floor(train_size * (1 - o)))
        if step_size < 1:
            step_size = 1

        start = 0
        fold = 0
        # For each walk-forward fold
        while True:
            train_start = start
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size

            if test_end > L or fold >= n:
                break

            train_idx_full = np.arange(train_start, train_end)
            test_idx_full  = np.arange(test_start, test_end)

            n_train_full = len(train_idx_full)
            n_test_full  = len(test_idx_full)
            # Number to subsample
            n_train_sub = max(1, int(np.floor(frac * n_train_full)))
            n_test_sub  = max(1, int(np.floor(frac * n_test_full)))

            for rep in range(repeats):
                # draw without replacement
                train_idx_sub = rng.choice(train_idx_full, size=n_train_sub, replace=False)
                test_idx_sub  = rng.choice(test_idx_full,  size=n_test_sub,  replace=False)
                yield train_idx_sub, test_idx_sub

            start += step_size
            fold += 1

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns total number of splits = actual_walk_folds * n_repeats_per_fold.
        """
        if X is None:
            raise ValueError("X must be provided to compute n_splits")
        L = len(X)
        if L <= 0:
            return 0
        o = self.overlap_percentage
        n = self.n_splits
        f = self.test_train_ratio
        repeats = self.n_repeats_per_fold

        denom = (1 - o) * (n - 1) + (1 + f)
        train_size = int(np.floor(L / denom))
        if train_size < 1:
            return 0
        test_size = int(np.floor(f * train_size))
        if test_size < 1:
            test_size = 1
        step_size = int(np.floor(train_size * (1 - o)))
        if step_size < 1:
            step_size = 1

        # Count how many walk-forward folds fit
        max_fits = int(np.floor((L - train_size - test_size) / step_size)) + 1
        actual_folds = min(max_fits, n)
        return actual_folds * repeats


def plot_cv_indices(cv, X, y, ax, n_splits=10, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    n_samples = len(X)
    
    # Create custom colormap: white for missing, then your original train/test colors
    cmap_cv = plt.cm.coolwarm  # Your original colormap
    train_color = cmap_cv(0.02)  # Your training color
    test_color = cmap_cv(0.8)    # Your testing color
    
    # Custom colormap: white -> train_color -> test_color
    colors = ['white', train_color, test_color]
    cmap_custom = ListedColormap(colors)
    
    # Generate the training/testing visualizations for each CV split
    for ii, (tr_idx, tt_idx) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.zeros(n_samples)  # 0 = white (missing)
        indices[tr_idx] = 1  # 1 = train color  
        indices[tt_idx] = 2  # 2 = test color
        
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_custom,
            vmin=0,            
            vmax=2,            
        )
    
    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 0.2, -0.2],
        xlim=[0, n_samples],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


cvs = [TimeSeriesCV, WalkForwardCV, TimeSeriesSplit, KFold, ShuffleSplit, WalkForwardShuffleCV]
n_splits = 10
cmap_cv = plt.cm.coolwarm

for cv in cvs:
    this_cv = cv(n_splits=n_splits)
    fig, ax = plt.subplots(figsize=(12, 3))
    plot_cv_indices(this_cv, X_train, y_train, ax, n_splits)

    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    # Make the legend fit
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
plt.show()


# Custom scorer: returns Pearson r between y_true and y_pred
def pearsonr_scorer(y_true, y_pred):
    """Pearson correlation scorer with error handling"""
    try:
        r, _ = pearsonr(y_true, y_pred)  # Fixed typo: y*true -> y_true
    except Exception:
        r = 0.0
    if np.isnan(r):
        r = 0.0
    return r

# Create sklearn-compatible scorers
pearson_sklearn_scorer = make_scorer(pearsonr_scorer, greater_is_better=True)
r2_sklearn_scorer = make_scorer(r2_score, greater_is_better=True)
mse_sklearn_scorer = make_scorer(mean_squared_error, greater_is_better=False)


# CV strategies 
cv_strategies = {
    "RandomTimeSeriesSplit": TimeSeriesCV(n_splits=10),
    "KFold": KFold(n_splits=10),
    "TimeSeriesSplit": TimeSeriesSplit(n_splits=10),
    "ShuffleSplit": ShuffleSplit(n_splits=10, test_size=0.5, random_state=42),
    "WalkForwardCV": WalkForwardCV(n_splits=10, overlap_percentage=0.8, test_train_ratio=1.0),
    "WalkForwardShuffleCV": WalkForwardShuffleCV(n_splits=10, overlap_percentage=0.8, test_train_ratio=1.0, frac=0.5, n_repeats_per_fold=1)
}


# Model pipelines
model_pipelines = {
    "Ridge": Pipeline([
        ("quantile", QuantileTransformer(output_distribution='normal', random_state=42)),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ]),
    
    "XGBoost": Pipeline([
        ("quantile", QuantileTransformer(output_distribution='normal', random_state=42)),
        ("xgb", xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
    ]),
    
    "LightGBM": Pipeline([
        ("lgb", lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ))
    ])
}


def evaluate_model_cv_strategy(model_name, model, cv_name, cv, X_train, y_train, X_test, y_test):
    """Evaluate a single model with a single CV strategy"""
    
    # Storage for fold scores
    pearson_scores = []
    r2_scores = []
    mse_scores = []
    
    # Manual loop to capture per-fold scores
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        X_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        y_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        # Fit model on this fold
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        # Compute metrics
        # Pearson correlation
        try:
            r, _ = pearsonr(y_val, y_pred)
        except Exception:
            r = 0.0
        if np.isnan(r):
            r = 0.0
        pearson_scores.append(r)
        
        # R²
        try:
            r2 = r2_score(y_val, y_pred)
        except Exception:
            r2 = 0.0
        if np.isnan(r2):
            r2 = 0.0
        r2_scores.append(r2)
        
        # MSE
        try:
            mse = mean_squared_error(y_val, y_pred)
        except Exception:
            mse = float('inf')
        if np.isnan(mse):
            mse = float('inf')
        mse_scores.append(mse)
    
    # Convert to arrays and compute statistics
    pearson_scores = np.array(pearson_scores)
    r2_scores = np.array(r2_scores)
    mse_scores = np.array(mse_scores)
    
    # Test set evaluation
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    
    test_pearson, _ = pearsonr(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    return {
        "Model": model_name,
        "CV": cv_name,
        "CV_mean_PearsonR": pearson_scores.mean(),
        "CV_std_PearsonR": pearson_scores.std(),
        "CV_mean_R2": r2_scores.mean(),
        "CV_std_R2": r2_scores.std(),
        "CV_mean_MSE": mse_scores.mean(),
        "CV_std_MSE": mse_scores.std(),
        "Test_PearsonR": test_pearson,
        "Test_R2": test_r2,
        "Test_MSE": test_mse,
        "fold_pearson_scores": pearson_scores,
        "fold_r2_scores": r2_scores,
        "fold_mse_scores": mse_scores
    }


def plot_cv_stability_analysis(results_df):
    """Plot CV stability analysis - the main focus"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. CV Standard Deviation heatmap (Lower = More Stable)
    pivot_std_pearson = results_df.pivot(index='Model', columns='CV', values='CV_std_PearsonR')
    sns.heatmap(pivot_std_pearson, annot=True, fmt='.4f', cmap='Reds_r', ax=axes[0])
    axes[0].set_title('CV Stability: Pearson R Standard Deviation (Lower = More Stable)', 
                     fontsize=12, weight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. CV Bias Analysis: |CV_mean - Test_score|
    bias_data = []
    for _, row in results_df.iterrows():
        bias = abs(row['CV_mean_PearsonR'] - row['Test_PearsonR'])
        bias_data.append({
            'Model': row['Model'],
            'CV': row['CV'],
            'Bias': bias
        })
    
    bias_df = pd.DataFrame(bias_data)
    pivot_bias = bias_df.pivot(index='Model', columns='CV', values='Bias')
    sns.heatmap(pivot_bias, annot=True, fmt='.4f', cmap='Reds', ax=axes[1])
    axes[1].set_title('CV Bias: |CV Mean - Test Score| (Lower = Less Biased)', 
                     fontsize=12, weight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print stability rankings
    print("="*60)
    print("CV STABILITY RANKINGS (by Pearson R Std - Lower is Better)")
    print("="*60)
    
    stability_summary = results_df.groupby('CV')['CV_std_PearsonR'].mean().sort_values()
    for i, (cv, std) in enumerate(stability_summary.items(), 1):
        print(f"{i:2d}. {cv:25s}: {std:.4f} (avg std)")
    
    print("="*60)
    print("CV BIAS RANKINGS (by |CV - Test| - Lower is Better)")
    print("="*60)
    
    bias_summary = bias_df.groupby('CV')['Bias'].mean().sort_values()
    for i, (cv, bias) in enumerate(bias_summary.items(), 1):
        print(f"{i:2d}. {cv:25s}: {bias:.4f} (avg bias)")
    
    print("="*60)
    print("COMBINED RANKING: Stability + Bias (Lower Combined Score = Better)")
    print("="*60)
    
    # Normalize both metrics and combine
    stability_norm = (stability_summary - stability_summary.min()) / (stability_summary.max() - stability_summary.min())
    bias_norm = (bias_summary - bias_summary.min()) / (bias_summary.max() - bias_summary.min())
    combined_score = stability_norm + bias_norm
    combined_ranking = combined_score.sort_values()
    
    for i, (cv, score) in enumerate(combined_ranking.items(), 1):
        stability = stability_summary[cv]
        bias = bias_summary[cv]
        print(f"{i:2d}. {cv:25s}: {score:.3f} (std: {stability:.4f}, bias: {bias:.4f})")


def plot_model_cv_detailed_comparison(detailed_scores, results_df):
    """Create 9 plots (3 models × 3 metrics) showing CV stability"""
    
    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    models = list(detailed_scores.keys())
    metrics = ['pearson', 'r2', 'mse']
    metric_names = ['Pearson Correlation', 'R² Score', 'Mean Squared Error']
    cv_names = list(cv_strategies.keys())
    
    # Get test scores for horizontal lines
    test_scores = {}
    for model in models:
        model_data = results_df[results_df['Model'] == model].iloc[0]  # All rows same test score
        test_scores[model] = {
            'pearson': model_data['Test_PearsonR'],
            'r2': model_data['Test_R2'],
            'mse': model_data['Test_MSE']
        }
    
    for model_idx, model in enumerate(models):
        for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[model_idx, metric_idx]
            
            # Prepare data for box plot
            data_for_plot = []
            labels_for_plot = []
            
            for cv_name in cv_names:
                if cv_name in detailed_scores[model]:
                    scores = detailed_scores[model][cv_name][metric]
                    data_for_plot.append(scores)
                    labels_for_plot.append(cv_name)
            
            # Create box plot
            bp = ax.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels_for_plot)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add title
            ax.set_title(f'{model} - {metric_name} Fold Score Distributions', 
                        fontsize=11, weight='bold')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add mean values as red diamonds
            for j, cv_name in enumerate(labels_for_plot):
                if cv_name in detailed_scores[model]:
                    mean_val = detailed_scores[model][cv_name][metric].mean()
                    ax.scatter(j+1, mean_val, color='red', marker='D', s=50, zorder=10)
            
            # Add horizontal dashed line for test performance
            test_score = test_scores[model][metric]
            ax.axhline(y=test_score, color='black', linestyle='--', linewidth=2, 
                      label=f'Test: {test_score:.3f}', alpha=0.8)
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()


# Run comprehensive evaluation
print("Evaluating all combinations of models and CV strategies...")
results = []
detailed_scores = {}

total_combinations = len(model_pipelines) * len(cv_strategies)
current_combination = 0

for model_name, model in model_pipelines.items():
    print(f"=== Model: {model_name} ===")
    detailed_scores[model_name] = {}
    
    for cv_name, cv in cv_strategies.items():
        current_combination += 1
        print(f"[{current_combination}/{total_combinations}] {model_name} + {cv_name}")
        
        try:
            result = evaluate_model_cv_strategy(model_name, model, cv_name, cv, X_train, y_train, X_test, y_test)
            results.append(result)
            
            # Store detailed scores for plotting
            detailed_scores[model_name][cv_name] = {
                'pearson': result['fold_pearson_scores'],
                'r2': result['fold_r2_scores'], 
                'mse': result['fold_mse_scores']
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

# Create results DataFrame
results_df = pd.DataFrame(results)
display_cols = ['Model', 'CV', 'CV_mean_PearsonR', 'CV_std_PearsonR', 'CV_mean_R2', 'CV_std_R2', 
                'CV_mean_MSE', 'CV_std_MSE', 'Test_PearsonR', 'Test_R2', 'Test_MSE']

print("="*80)
print("COMPLETE RESULTS SUMMARY")
print("="*80)
print(results_df[display_cols].round(4))

# Summary by model
print("="*50)
print("SUMMARY BY MODEL (Average across CV strategies)")
print("="*50)
model_summary = results_df.groupby('Model')[['CV_mean_PearsonR', 'CV_std_PearsonR', 'Test_PearsonR']].mean()
print(model_summary.round(4))

# Summary by CV strategy
print("="*50)
print("SUMMARY BY CV STRATEGY (Average across models)")
print("="*50)
cv_summary = results_df.groupby('CV')[['CV_mean_PearsonR', 'CV_std_PearsonR', 'Test_PearsonR']].mean()
print(cv_summary.round(4))

# Create visualizations and analysis
print("Creating detailed CV comparison visualizations...")
plot_model_cv_detailed_comparison(detailed_scores, results_df)

print("Creating stability analysis...")
plot_cv_stability_analysis(results_df)

