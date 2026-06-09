# Import libraries
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from scipy import stats
from scipy.special import factorial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class M5DataLoader:
    """
    Load and preprocess M5 competition data for probabilistic analysis.

    This class handles:
    - Loading raw M5 data files
    - Creating hierarchical structure
    - Preprocessing for probabilistic EDA
    - Generating time series matrix
    """

    def __init__(self, data_path: str = 'data/'):
        """
        Initialize data loader.

        Parameters:
        -----------
        data_path : str
            Path to directory containing M5 data files
        """
        self.data_path = Path(data_path)
        self.sales_data = None
        self.calendar = None
        self.prices = None
        self.submission = None

        # Hierarchy metadata
        self.hierarchy_levels = ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']
        self.quantiles = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

    def load_data(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all M5 data files.

        Parameters:
        -----------
        verbose : bool
            Print loading progress

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing all loaded dataframes
        """
        if verbose:
            print("Loading M5 Competition Data...")

        # Load sales training data
        sales_file = '/kaggle/input/m5-forecasting-uncertainty/sales_train_evaluation.csv'
        self.sales_data = pd.read_csv(sales_file)
        print(f"âœ“ Sales data loaded: {self.sales_data.shape}")

        # Load calendar
        calendar_file = '/kaggle/input/m5-forecasting-uncertainty/calendar.csv'
        self.calendar = pd.read_csv(calendar_file)
        self.calendar['date'] = pd.to_datetime(self.calendar['date'])
        print(f"âœ“ Calendar loaded: {self.calendar.shape}")

        # Load prices
        prices_file = '/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv'
        self.prices = pd.read_csv(prices_file)
        print(f"âœ“ Prices loaded: {self.prices.shape}")

        # Load submission template (optional)
        submission_file = '/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv'
        self.submission = pd.read_csv(submission_file)

        return {
            'sales': self.sales_data,
            'calendar': self.calendar,
            'prices': self.prices,
            'submission': self.submission
        }

    def get_time_series_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert sales data to time series matrix format.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            - Time series values (items x days)
            - Metadata (hierarchical information)
        """
        if self.sales_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Extract time series columns (d_1, d_2, ..., d_1941)
        day_cols = [col for col in self.sales_data.columns if col.startswith('d_')]

        # Time series matrix
        ts_data = self.sales_data[day_cols].values

        # Metadata
        metadata = self.sales_data[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].copy()

        return pd.DataFrame(ts_data, columns=day_cols), metadata

    def create_hierarchical_structure(self) -> Dict[str, pd.DataFrame]:
        """
        Create aggregated time series at different hierarchy levels.

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Time series at each aggregation level
        """
        if self.sales_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        day_cols = [col for col in self.sales_data.columns if col.startswith('d_')]
        hierarchical_ts = {}

        # Level 0: Total (all stores, all items)
        hierarchical_ts['total'] = pd.DataFrame({
            'id': ['total'],
            **{day: [self.sales_data[day].sum()] for day in day_cols}
        })

        # Level 1: By state
        hierarchical_ts['state'] = self.sales_data.groupby('state_id')[day_cols].sum().reset_index()

        # Level 2: By store
        hierarchical_ts['store'] = self.sales_data.groupby('store_id')[day_cols].sum().reset_index()

        # Level 3: By category
        hierarchical_ts['category'] = self.sales_data.groupby('cat_id')[day_cols].sum().reset_index()

        # Level 4: By department
        hierarchical_ts['department'] = self.sales_data.groupby('dept_id')[day_cols].sum().reset_index()

        # Level 5: By state-category
        hierarchical_ts['state_cat'] = self.sales_data.groupby(['state_id', 'cat_id'])[day_cols].sum().reset_index()

        # Level 6: By store-category
        hierarchical_ts['store_cat'] = self.sales_data.groupby(['store_id', 'cat_id'])[day_cols].sum().reset_index()

        # Level 7: By store-department
        hierarchical_ts['store_dept'] = self.sales_data.groupby(['store_id', 'dept_id'])[day_cols].sum().reset_index()

        # Level 8: Bottom level (store-item)
        hierarchical_ts['item'] = self.sales_data.copy()

        return hierarchical_ts

    def get_date_features(self) -> pd.DataFrame:
        """
        Extract temporal features from calendar.

        Returns:
        --------
        pd.DataFrame
            Calendar with additional features
        """
        if self.calendar is None:
            raise ValueError("Calendar not loaded. Call load_data() first.")

        cal = self.calendar.copy()

        # Temporal features
        cal['year'] = cal['date'].dt.year
        cal['month'] = cal['date'].dt.month
        cal['day'] = cal['date'].dt.day
        cal['dayofweek'] = cal['date'].dt.dayofweek
        cal['dayofyear'] = cal['date'].dt.dayofyear
        cal['weekofyear'] = cal['date'].dt.isocalendar().week
        cal['quarter'] = cal['date'].dt.quarter

        # Is weekend
        cal['is_weekend'] = cal['dayofweek'].isin([5, 6]).astype(int)

        # Event indicators
        cal['has_event'] = ((cal['event_name_1'].notna()) | (cal['event_name_2'].notna())).astype(int)
        cal['has_snap'] = ((cal['snap_CA'] == 1) | (cal['snap_TX'] == 1) | (cal['snap_WI'] == 1)).astype(int)

        return cal

    def compute_basic_statistics(self) -> pd.DataFrame:
        """
        Compute basic statistics for each time series.

        Returns:
        --------
        pd.DataFrame
            Statistics per series (mean, std, zeros, etc.)
        """
        ts_data, metadata = self.get_time_series_matrix()

        stats = pd.DataFrame({
            'id': metadata['id'],
            'mean': ts_data.mean(axis=1),
            'std': ts_data.std(axis=1),
            'median': ts_data.median(axis=1),
            'min': ts_data.min(axis=1),
            'max': ts_data.max(axis=1),
            'q25': ts_data.quantile(0.25, axis=1),
            'q75': ts_data.quantile(0.75, axis=1),
            'zeros_count': (ts_data == 0).sum(axis=1),
            'zeros_pct': (ts_data == 0).mean(axis=1) * 100,
            'cv': ts_data.std(axis=1) / (ts_data.mean(axis=1) + 1e-8),  # Coefficient of variation
            'range': ts_data.max(axis=1) - ts_data.min(axis=1)
        })

        # Merge with metadata
        stats = stats.merge(metadata, on='id', how='left')

        return stats

    def get_sample_series(self, n_samples: int = 100,
                         stratify_by: Optional[str] = 'cat_id',
                         random_state: int = 42) -> pd.DataFrame:
        """
        Get a stratified sample of time series for analysis.

        Parameters:
        -----------
        n_samples : int
            Number of series to sample
        stratify_by : str, optional
            Column to stratify sampling
        random_state : int
            Random seed

        Returns:
        --------
        pd.DataFrame
            Sampled series with metadata
        """
        if self.sales_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if stratify_by and stratify_by in self.sales_data.columns:
            # Stratified sampling
            sampled = self.sales_data.groupby(stratify_by).apply(
                lambda x: x.sample(min(len(x), n_samples // self.sales_data[stratify_by].nunique()),
                                  random_state=random_state)
            ).reset_index(drop=True)
        else:
            # Random sampling
            sampled = self.sales_data.sample(n=min(n_samples, len(self.sales_data)),
                                            random_state=random_state)

        return sampled

    def get_series_by_id(self, series_id: str) -> pd.Series:
        """
        Get a single time series by ID.

        Parameters:
        -----------
        series_id : str
            Series identifier

        Returns:
        --------
        pd.Series
            Time series values
        """
        if self.sales_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        series = self.sales_data[self.sales_data['id'] == series_id]
        if series.empty:
            raise ValueError(f"Series {series_id} not found")

        day_cols = [col for col in self.sales_data.columns if col.startswith('d_')]
        return series[day_cols].values.flatten()

    def save_processed_data(self, output_path: str = 'data/processed/'):
        """
        Save preprocessed data for faster loading.

        Parameters:
        -----------
        output_path : str
            Directory to save processed files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save time series matrix
        ts_data, metadata = self.get_time_series_matrix()
        ts_data.to_parquet(output_path / 'ts_matrix.parquet')
        metadata.to_parquet(output_path / 'metadata.parquet')

        # Save hierarchical structure
        hierarchical = self.create_hierarchical_structure()
        for level, data in hierarchical.items():
            data.to_parquet(output_path / f'hierarchy_{level}.parquet')

        # Save calendar features
        cal_features = self.get_date_features()
        cal_features.to_parquet(output_path / 'calendar_features.parquet')

        print(f"Processed data saved to {output_path}")



class DistributionAnalyzer:
    """
    Analyze distributional properties of time series for probabilistic forecasting.

    Key focus areas:
    - Zero-inflation (common in retail sales)
    - Heavy tails and extreme values
    - Intermittency patterns
    - Distribution fitting
    """

    def __init__(self):
        """Initialize distribution analyzer."""
        self.distributions = [
            'norm', 'lognorm', 'gamma', 'poisson',
            'nbinom', 'expon', 'weibull_min'
        ]

    def analyze_zero_inflation(self, series: np.ndarray) -> Dict[str, float]:
        """
        Test for zero-inflation in time series.

        Zero-inflation is critical in retail sales where many products
        have intermittent demand patterns.

        Parameters:
        -----------
        series : np.ndarray
            Time series data

        Returns:
        --------
        Dict[str, float]
            Zero-inflation metrics
        """
        series = np.asarray(series, dtype=np.float64).flatten()
        n = len(series)
        zeros = np.sum(series == 0)
        zero_pct = zeros / n * 100

        # Compute expected zeros under Poisson distribution
        mean_val = np.mean(series)
        if mean_val > 0:
            expected_zero_pct = np.exp(-mean_val) * 100
        else:
            expected_zero_pct = 100.0

        # Zero-inflation ratio
        zi_ratio = zero_pct / (expected_zero_pct + 1e-8)

        # Consecutive zero runs
        zero_mask = (series == 0).astype(int)
        zero_runs = []
        current_run = 0
        for val in zero_mask:
            if val == 1:
                current_run += 1
            else:
                if current_run > 0:
                    zero_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            zero_runs.append(current_run)

        avg_zero_run = np.mean(zero_runs) if zero_runs else 0
        max_zero_run = max(zero_runs) if zero_runs else 0

        return {
            'zero_count': int(zeros),
            'zero_pct': float(zero_pct),
            'expected_zero_pct': float(expected_zero_pct),
            'zi_ratio': float(zi_ratio),
            'is_zero_inflated': bool(zi_ratio > 1.5),  # Threshold for zero-inflation
            'avg_zero_run': float(avg_zero_run),
            'max_zero_run': int(max_zero_run),
            'intermittency_index': float(zero_pct / 100)  # Croston's intermittency
        }

    def analyze_heavy_tails(self, series: np.ndarray) -> Dict[str, float]:
        """
        Analyze heavy-tail characteristics.

        Heavy tails indicate presence of extreme values and outliers,
        common in retail sales due to promotions, holidays, etc.

        Parameters:
        -----------
        series : np.ndarray
            Time series data

        Returns:
        --------
        Dict[str, float]
            Heavy-tail metrics
        """
        # Ensure proper numeric dtype
        series = np.asarray(series, dtype=np.float64).flatten()
        series_nonzero = series[series > 0]  # Exclude zeros for tail analysis

        if len(series_nonzero) < 4:
            return {
                'kurtosis': np.nan,
                'excess_kurtosis': np.nan,
                'skewness': np.nan,
                'is_heavy_tailed': False,
                'tail_index': np.nan,
                'q99_q95_ratio': np.nan
            }

        # Kurtosis (measure of tail heaviness)
        kurt = stats.kurtosis(series_nonzero, fisher=True)  # Excess kurtosis

        # Skewness
        skew = stats.skew(series_nonzero)

        # Tail index using Hill estimator (simple version)
        sorted_series = np.sort(series_nonzero)[::-1]  # Descending
        k = max(int(len(sorted_series) * 0.1), 5)  # Top 10% for tail estimation
        if k < len(sorted_series):
            log_ratios = np.log(sorted_series[:k] / sorted_series[k])
            tail_index = 1 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else np.nan
        else:
            tail_index = np.nan

        # Quantile ratio (another heavy-tail indicator)
        q95 = np.percentile(series_nonzero, 95)
        q99 = np.percentile(series_nonzero, 99)
        q99_q95_ratio = q99 / q95 if q95 > 0 else np.nan

        # Heavy-tail indicator
        is_heavy_tailed = (kurt > 3) or (q99_q95_ratio > 2)

        return {
            'kurtosis': float(kurt + 3),  # Regular kurtosis
            'excess_kurtosis': float(kurt),
            'skewness': float(skew),
            'is_heavy_tailed': bool(is_heavy_tailed),
            'tail_index': float(tail_index) if not np.isnan(tail_index) else None,
            'q99_q95_ratio': float(q99_q95_ratio) if not np.isnan(q99_q95_ratio) else None
        }

    def detect_outliers(self, series: np.ndarray,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> Dict[str, any]:
        """
        Detect outliers using multiple methods.

        Parameters:
        -----------
        series : np.ndarray
            Time series data
        method : str
            Outlier detection method ('iqr', 'zscore', 'mad')
        threshold : float
            Threshold for outlier detection

        Returns:
        --------
        Dict
            Outlier analysis results
        """
        series = np.asarray(series, dtype=np.float64).flatten()

        if method == 'iqr':
            q1 = np.percentile(series, 25)
            q3 = np.percentile(series, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (series < lower_bound) | (series > upper_bound)

        elif method == 'zscore':
            mean = np.mean(series)
            std = np.std(series)
            z_scores = np.abs((series - mean) / (std + 1e-8))
            outliers = z_scores > threshold

        elif method == 'mad':
            median = np.median(series)
            mad = np.median(np.abs(series - median))
            modified_z = 0.6745 * (series - median) / (mad + 1e-8)
            outliers = np.abs(modified_z) > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        outlier_indices = np.where(outliers)[0]
        outlier_values = series[outliers]

        return {
            'method': method,
            'threshold': threshold,
            'n_outliers': int(np.sum(outliers)),
            'outlier_pct': float(np.mean(outliers) * 100),
            'outlier_indices': outlier_indices.tolist(),
            'outlier_values': outlier_values.tolist(),
            'max_outlier': float(np.max(outlier_values)) if len(outlier_values) > 0 else None
        }

    def fit_distributions(self, series: np.ndarray,
                         top_n: int = 3) -> List[Dict[str, any]]:
        """
        Fit multiple distributions and rank by goodness-of-fit.

        Parameters:
        -----------
        series : np.ndarray
            Time series data
        top_n : int
            Number of best-fitting distributions to return

        Returns:
        --------
        List[Dict]
            Best-fitting distributions with parameters and scores
        """
        series = np.asarray(series, dtype=np.float64).flatten()
        series_positive = series[series > 0]  # Many distributions require positive values

        if len(series_positive) < 10:
            return []

        results = []

        for dist_name in self.distributions:
            try:
                dist = getattr(stats, dist_name)

                # Fit distribution
                if dist_name == 'poisson':
                    # Poisson has only one parameter
                    mu = np.mean(series_positive)
                    params = (mu,)
                    ks_stat, p_value = stats.kstest(series_positive, dist_name, args=(mu,))
                else:
                    params = dist.fit(series_positive)
                    ks_stat, p_value = stats.kstest(series_positive, dist_name, args=params)

                # Compute AIC/BIC
                n = len(series_positive)
                k = len(params)
                log_likelihood = np.sum(dist.logpdf(series_positive, *params))
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood

                results.append({
                    'distribution': dist_name,
                    'parameters': params,
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(p_value),
                    'aic': float(aic),
                    'bic': float(bic),
                    'log_likelihood': float(log_likelihood)
                })

            except Exception as e:
                # Skip if distribution fitting fails
                continue

        # Sort by AIC (lower is better)
        results = sorted(results, key=lambda x: x['aic'])

        return results[:top_n]

    def compute_descriptive_stats(self, series: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive descriptive statistics.

        Parameters:
        -----------
        series : np.ndarray
            Time series data

        Returns:
        --------
        Dict[str, float]
            Descriptive statistics
        """
        series = np.asarray(series, dtype=np.float64).flatten()

        return {
            'count': int(len(series)),
            'mean': float(np.mean(series)),
            'std': float(np.std(series)),
            'var': float(np.var(series)),
            'min': float(np.min(series)),
            'q01': float(np.percentile(series, 1)),
            'q05': float(np.percentile(series, 5)),
            'q25': float(np.percentile(series, 25)),
            'median': float(np.median(series)),
            'q75': float(np.percentile(series, 75)),
            'q95': float(np.percentile(series, 95)),
            'q99': float(np.percentile(series, 99)),
            'max': float(np.max(series)),
            'range': float(np.max(series) - np.min(series)),
            'iqr': float(np.percentile(series, 75) - np.percentile(series, 25)),
            'cv': float(np.std(series) / (np.mean(series) + 1e-8))  # Coefficient of variation
        }

    def analyze_intermittency(self, series: np.ndarray) -> Dict[str, float]:
        """
        Analyze intermittency patterns (Croston's method).

        Parameters:
        -----------
        series : np.ndarray
            Time series data

        Returns:
        --------
        Dict[str, float]
            Intermittency metrics
        """
        series = np.asarray(series, dtype=np.float64).flatten()

        # Demand intervals (time between non-zero demands)
        nonzero_indices = np.where(series > 0)[0]

        if len(nonzero_indices) < 2:
            return {
                'intermittency_rate': 1.0,
                'avg_demand_interval': np.nan,
                'avg_demand_size': np.nan,
                'classification': 'dead'
            }

        intervals = np.diff(nonzero_indices)
        avg_interval = np.mean(intervals)
        avg_demand = np.mean(series[series > 0])

        # Intermittency rate (proportion of zeros)
        intermittency_rate = np.sum(series == 0) / len(series)

        # Coefficient of variation of demand size
        cv_demand = np.std(series[series > 0]) / (avg_demand + 1e-8)

        # Classification (Syntetos-Boylan)
        # ADI < 1.32 and CV < 0.49: Smooth
        # ADI >= 1.32 and CV < 0.49: Intermittent
        # ADI < 1.32 and CV >= 0.49: Erratic
        # ADI >= 1.32 and CV >= 0.49: Lumpy
        if avg_interval < 1.32:
            classification = 'erratic' if cv_demand >= 0.49 else 'smooth'
        else:
            classification = 'lumpy' if cv_demand >= 0.49 else 'intermittent'

        return {
            'intermittency_rate': float(intermittency_rate),
            'avg_demand_interval': float(avg_interval),
            'avg_demand_size': float(avg_demand),
            'cv_demand': float(cv_demand),
            'classification': classification,
            'n_demand_occasions': int(len(nonzero_indices))
        }

    def analyze_series(self, series: np.ndarray,
                      series_id: str = None) -> Dict[str, any]:
        """
        Complete distribution analysis for a single series.

        Parameters:
        -----------
        series : np.ndarray
            Time series data
        series_id : str, optional
            Series identifier

        Returns:
        --------
        Dict
            Comprehensive distribution analysis
        """
        results = {
            'series_id': series_id,
            'length': len(series)
        }

        # Descriptive statistics
        results['descriptive_stats'] = self.compute_descriptive_stats(series)

        # Zero-inflation analysis
        results['zero_inflation'] = self.analyze_zero_inflation(series)

        # Heavy-tail analysis
        results['heavy_tails'] = self.analyze_heavy_tails(series)

        # Intermittency
        results['intermittency'] = self.analyze_intermittency(series)

        # Outlier detection
        results['outliers_iqr'] = self.detect_outliers(series, method='iqr', threshold=1.5)
        results['outliers_zscore'] = self.detect_outliers(series, method='zscore', threshold=3.0)

        # Distribution fitting
        results['best_distributions'] = self.fit_distributions(series, top_n=3)

        return results

    def batch_analyze(self, series_dict: Dict[str, np.ndarray],
                     verbose: bool = True) -> pd.DataFrame:
        """
        Analyze multiple time series and return summary DataFrame.

        Parameters:
        -----------
        series_dict : Dict[str, np.ndarray]
            Dictionary of series_id -> series data
        verbose : bool
            Print progress

        Returns:
        --------
        pd.DataFrame
            Summary statistics for all series
        """
        results = []

        for i, (series_id, series) in enumerate(series_dict.items()):
            if verbose and (i + 1) % 1000 == 0:
                print(f"Analyzed {i + 1}/{len(series_dict)} series...")

            analysis = self.analyze_series(series, series_id)

            # Flatten for DataFrame
            row = {
                'series_id': series_id,
                **analysis['descriptive_stats'],
                **{f'zi_{k}': v for k, v in analysis['zero_inflation'].items()},
                **{f'ht_{k}': v for k, v in analysis['heavy_tails'].items()},
                **{f'int_{k}': v for k, v in analysis['intermittency'].items()},
                'n_outliers_iqr': analysis['outliers_iqr']['n_outliers'],
                'n_outliers_zscore': analysis['outliers_zscore']['n_outliers'],
                'best_dist': analysis['best_distributions'][0]['distribution'] if analysis['best_distributions'] else None
            }

            results.append(row)

        return pd.DataFrame(results)


class DistributionVisualizer:
    """
    Create advanced distribution visualizations for probabilistic EDA.

    Focuses on:
    - Empirical distribution plots
    - Q-Q plots for distribution assessment
    - Distribution comparison across categories
    - Tail behavior visualization
    - Zero-inflation patterns
    """

    def __init__(self, output_dir: str = 'outputs/figures/'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes
        self.colors = {
            'smooth': '#2ecc71',
            'intermittent': '#f39c12',
            'erratic': '#e74c3c',
            'lumpy': '#8e44ad',
            'primary': '#3498db',
            'secondary': '#95a5a6'
        }

    def plot_distribution_overview(self,
                                   dist_analysis: pd.DataFrame,
                                   sample_series: pd.DataFrame,
                                   save_path: Optional[str] = None):
        """
        Create comprehensive distribution overview with multiple visualization types.

        Parameters:
        -----------
        dist_analysis : pd.DataFrame
            Distribution analysis results
        sample_series : pd.DataFrame
            Sample time series data
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        fig.suptitle('Distribution Analysis: Comprehensive Overview',
                    fontsize=18, fontweight='bold', y=0.98)

        # Row 1: Overall distribution characteristics
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_distribution_by_pattern(dist_analysis, sample_series, ax1)

        #ax2 = fig.add_subplot(gs[0, 2])
        #self._plot_distribution_shape_summary(dist_analysis, ax2)

        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_qq_plot_by_pattern(dist_analysis, sample_series, ax2, 'lumpy')

        # Row 2: Detailed pattern analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_qq_plot_by_pattern(dist_analysis, sample_series, ax3, 'smooth')

        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_qq_plot_by_pattern(dist_analysis, sample_series, ax4, 'intermittent')

        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_qq_plot_by_pattern(dist_analysis, sample_series, ax5, 'erratic')

        # Row 3: Tail and zero analysis
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_tail_comparison(dist_analysis, ax6)

        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_zero_mass_distribution(dist_analysis, ax7)

        #ax8 = fig.add_subplot(gs[2, 2])
        #self._plot_kurtosis_skewness_map(dist_analysis, ax8)
        
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_distribution_shape_summary(dist_analysis, ax8)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Distribution overview saved: {save_path}")

        plt.show()

    def _plot_distribution_by_pattern(self, dist_analysis: pd.DataFrame,
                                      sample_series: pd.DataFrame, ax):
        """Plot empirical distributions grouped by demand pattern."""
        day_cols = [col for col in sample_series.columns if col.startswith('d_')]

        # Get examples from each pattern
        patterns = ['smooth', 'intermittent', 'erratic', 'lumpy']

        for pattern in patterns:
            # Get product IDs for this pattern
            pattern_ids = dist_analysis[
                dist_analysis['int_classification'] == pattern
            ]['series_id'].tolist()

            # Get data for these products
            pattern_data = []
            for pid in pattern_ids:
                series_row = sample_series[sample_series['id'] == pid]
                if not series_row.empty:
                    values = series_row[day_cols].values.flatten()
                    pattern_data.extend(values)

            if pattern_data:
                # Plot KDE
                from scipy.stats import gaussian_kde
                data_clean = np.array(pattern_data)
                data_clean = data_clean[~np.isnan(data_clean)]
                data_clean = data_clean[data_clean > 0]  # Remove zeros for clarity

                if len(data_clean) > 10:
                    kde = gaussian_kde(data_clean)
                    x_range = np.linspace(0, np.percentile(data_clean, 95), 200)
                    density = kde(x_range)

                    ax.plot(x_range, density, linewidth=2.5,
                           label=f'{pattern.capitalize()} ({len(pattern_ids)} products)',
                           color=self.colors.get(pattern, self.colors['primary']),
                           alpha=0.8)

                    ax.fill_between(x_range, density, alpha=0.2,
                                   color=self.colors.get(pattern, self.colors['primary']))

        ax.set_xlabel('Daily Demand (units, excluding zeros)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax.set_title('Empirical Distribution by Demand Pattern\n(Kernel Density Estimation)',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)

    def _plot_distribution_shape_summary(self, dist_analysis: pd.DataFrame, ax):
        """Summary statistics of distribution shapes."""
        # Create summary by pattern
        summary_data = []

        for pattern in ['smooth', 'intermittent', 'erratic', 'lumpy']:
            subset = dist_analysis[dist_analysis['int_classification'] == pattern]
            if len(subset) > 0:
                summary_data.append({
                    'Pattern': pattern.capitalize(),
                    'Count': len(subset),
                    'Avg CV': subset['cv'].mean(),
                    'Avg Skew': subset['ht_skewness'].mean(),
                    'Heavy Tail %': subset['ht_is_heavy_tailed'].mean() * 100
                })

        summary_df = pd.DataFrame(summary_data)

        # Create table
        ax.axis('tight')
        ax.axis('off')

        table_data = []
        table_data.append(['Pattern', 'Count', 'Avg CV', 'Avg Skew', 'Heavy Tail %'])

        for _, row in summary_df.iterrows():
            table_data.append([
                row['Pattern'],
                f"{int(row['Count']):,}",
                f"{row['Avg CV']:.2f}",
                f"{row['Avg Skew']:.2f}",
                f"{row['Heavy Tail %']:.1f}%"
            ])

        table = ax.table(cellText=table_data, cellLoc='center',
                        loc='center', bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color code by pattern
        colors_map = {
            'Smooth': '#2ecc71',
            'Intermittent': '#f39c12',
            'Erratic': '#e74c3c',
            'Lumpy': '#8e44ad'
        }

        for i, (_, row) in enumerate(summary_df.iterrows(), 1):
            color = colors_map.get(row['Pattern'], '#95a5a6')
            for j in range(5):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_alpha(0.3)

        ax.set_title('Distribution Shape Statistics by Pattern',
                    fontsize=13, fontweight='bold', pad=20)

    def _plot_qq_plot_by_pattern(self, dist_analysis: pd.DataFrame,
                                 sample_series: pd.DataFrame,
                                 ax, pattern: str):
        """Q-Q plot for specific demand pattern."""
        day_cols = [col for col in sample_series.columns if col.startswith('d_')]

        # Get examples from this pattern
        pattern_ids = dist_analysis[
            dist_analysis['int_classification'] == pattern
        ]['series_id'].head(10).tolist()

        # Collect data
        pattern_data = []
        for pid in pattern_ids:
            series_row = sample_series[sample_series['id'] == pid]
            if not series_row.empty:
                values = series_row[day_cols].values.flatten()
                values_clean = values[values > 0]  # Exclude zeros
                pattern_data.extend(values_clean)

        if len(pattern_data) > 10:
            pattern_data = np.array(pattern_data)

            # Q-Q plot against normal distribution
            stats.probplot(pattern_data, dist="norm", plot=ax)

            ax.get_lines()[0].set_marker('o')
            ax.get_lines()[0].set_markersize(3)
            ax.get_lines()[0].set_alpha(0.5)
            ax.get_lines()[0].set_color(self.colors.get(pattern, self.colors['primary']))

            ax.get_lines()[1].set_color('red')
            ax.get_lines()[1].set_linewidth(2)
            ax.get_lines()[1].set_linestyle('--')

            ax.set_title(f'Q-Q Plot: {pattern.capitalize()} Demand\nvs Normal Distribution',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Theoretical Quantiles', fontsize=10)
            ax.set_ylabel('Sample Quantiles', fontsize=10)
            ax.grid(alpha=0.3)

            # Add normality test result
            _, p_value = stats.normaltest(pattern_data)
            test_result = "Approx. Normal" if p_value > 0.05 else "Non-Normal"
            ax.text(0.05, 0.95, f'{test_result}\n(p={p_value:.4f})',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   fontsize=8)

    def _plot_tail_comparison(self, dist_analysis: pd.DataFrame, ax):
        """Compare tail behavior across patterns."""
        # Use quantile ratios to show tail heaviness
        patterns = dist_analysis['int_classification'].unique()

        tail_data = []
        for pattern in patterns:
            subset = dist_analysis[dist_analysis['int_classification'] == pattern]
            if len(subset) > 0 and 'ht_q99_q95_ratio' in subset.columns:
                ratios = subset['ht_q99_q95_ratio'].dropna()
                if len(ratios) > 0:
                    tail_data.append({
                        'pattern': pattern,
                        'ratios': ratios.values
                    })

        # Create violin plot
        positions = []
        data_list = []
        labels = []

        for i, item in enumerate(tail_data):
            positions.append(i)
            data_list.append(item['ratios'])
            labels.append(item['pattern'].capitalize())

        if data_list:
            parts = ax.violinplot(data_list, positions=positions,
                                 showmeans=True, showmedians=True)

            # Color by pattern
            for i, (pc, pattern_data) in enumerate(zip(parts['bodies'], tail_data)):
                color = self.colors.get(pattern_data['pattern'], self.colors['primary'])
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

            # Add reference line for "normal" ratio
            ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2,
                      label='Heavy Tail Threshold', alpha=0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel('Q99/Q95 Ratio', fontsize=10, fontweight='bold')
            ax.set_title('Tail Heaviness Comparison\n(Higher = Heavier Tails)',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

    def _plot_zero_mass_distribution(self, dist_analysis: pd.DataFrame, ax):
        """Visualize zero-inflation across patterns."""
        # Create stacked histogram
        patterns = ['smooth', 'intermittent', 'erratic', 'lumpy']
        bins = np.linspace(0, 100, 21)

        bottom = np.zeros(len(bins) - 1)

        for pattern in patterns:
            subset = dist_analysis[dist_analysis['int_classification'] == pattern]
            if len(subset) > 0:
                hist, _ = np.histogram(subset['zi_zero_pct'], bins=bins)

                ax.bar(bins[:-1], hist, width=5, bottom=bottom,
                      label=pattern.capitalize(),
                      color=self.colors.get(pattern, self.colors['primary']),
                      alpha=0.8, edgecolor='black', linewidth=0.5)

                bottom += hist

        ax.set_xlabel('Zero Percentage (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Products', fontsize=10, fontweight='bold')
        ax.set_title('Zero-Inflation Distribution\nby Demand Pattern (Stacked)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        # Add zone markers
        ax.axvline(30, color='green', linestyle=':', alpha=0.5, linewidth=2)
        ax.axvline(70, color='red', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(15, ax.get_ylim()[1] * 0.9, 'Active', ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(50, ax.get_ylim()[1] * 0.9, 'Moderate', ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.text(85, ax.get_ylim()[1] * 0.9, 'Slow', ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))

    def _plot_kurtosis_skewness_map(self, dist_analysis: pd.DataFrame, ax):
        """2D map of kurtosis vs skewness."""
        # Create clean dataframe first
        plot_data = dist_analysis[['ht_excess_kurtosis', 'ht_skewness', 'int_classification']].copy()

        # Replace inf values with NaN
        plot_data['ht_excess_kurtosis'] = plot_data['ht_excess_kurtosis'].replace([np.inf, -np.inf], np.nan)
        plot_data['ht_skewness'] = plot_data['ht_skewness'].replace([np.inf, -np.inf], np.nan)

        # Drop rows with any NaN
        plot_data = plot_data.dropna()

        # Rename for clarity
        plot_data = plot_data.rename(columns={
            'ht_excess_kurtosis': 'kurtosis',
            'ht_skewness': 'skewness',
            'int_classification': 'pattern'
        })

        # Create scatter plot
        for pattern in ['smooth', 'intermittent', 'erratic', 'lumpy']:
            subset = plot_data[plot_data['pattern'] == pattern]
            if len(subset) > 0:
                ax.scatter(subset['skewness'], subset['kurtosis'],
                          label=pattern.capitalize(),
                          color=self.colors.get(pattern, self.colors['primary']),
                          alpha=0.4, s=30, edgecolors='black', linewidth=0.5)

        # Add reference lines
        ax.axhline(3, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                  label='Heavy Tail Threshold')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

        ax.set_xlabel('Skewness', fontsize=10, fontweight='bold')
        ax.set_ylabel('Excess Kurtosis', fontsize=10, fontweight='bold')
        ax.set_title('Distribution Shape Map\n(Skewness vs Kurtosis)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

        # Add quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.text(xlim[1] * 0.7, ylim[1] * 0.9, 'Heavy Tail\nRight Skew',
               ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5))
        ax.text(xlim[0] * 0.7, ylim[1] * 0.9, 'Heavy Tail\nLeft Skew',
               ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#cce5ff', alpha=0.5))

    def plot_individual_product_distributions(self,
                                              dist_analysis: pd.DataFrame,
                                              sample_series: pd.DataFrame,
                                              n_examples: int = 12,
                                              save_path: Optional[str] = None):
        """
        Show individual product distribution examples.

        Parameters:
        -----------
        dist_analysis : pd.DataFrame
            Distribution analysis results
        sample_series : pd.DataFrame
            Sample time series data
        n_examples : int
            Number of examples to show
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()

        fig.suptitle('Individual Product Distributions: Real Examples',
                    fontsize=16, fontweight='bold')

        day_cols = [col for col in sample_series.columns if col.startswith('d_')]

        # Get diverse examples
        examples = []
        for pattern in ['smooth', 'intermittent', 'erratic', 'lumpy']:
            pattern_products = dist_analysis[
                dist_analysis['int_classification'] == pattern
            ]['series_id'].head(3).tolist()
            examples.extend(pattern_products)

        for idx, (ax, product_id) in enumerate(zip(axes, examples[:n_examples])):
            # Get series data
            series_row = sample_series[sample_series['id'] == product_id]
            if series_row.empty:
                continue

            values = series_row[day_cols].values.flatten()

            # Get analysis results
            analysis_row = dist_analysis[dist_analysis['series_id'] == product_id].iloc[0]

            # Plot histogram
            ax.hist(values, bins=30, alpha=0.7, edgecolor='black',
                   color=self.colors.get(analysis_row['int_classification'], self.colors['primary']))

            # Add statistics
            ax.axvline(analysis_row['mean'], color='red', linestyle='--',
                      linewidth=2, label=f"Mean: {analysis_row['mean']:.1f}")
            ax.axvline(analysis_row['median'], color='orange', linestyle=':',
                      linewidth=2, label=f"Median: {analysis_row['median']:.1f}")

            ax.set_xlabel('Daily Demand', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title(f"{analysis_row['int_classification'].capitalize()}\n" +
                        f"Zeros: {analysis_row['zi_zero_pct']:.0f}%, CV: {analysis_row['cv']:.2f}",
                        fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Individual distributions saved: {save_path}")

        plt.show()



class UncertaintyQuantifier:
    """
    Quantify uncertainty for probabilistic time series forecasting.

    Implements methods aligned with M5 competition requirements:
    - 9 quantile levels (0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995)
    - Conformal prediction for distribution-free guarantees
    - Time-varying uncertainty estimation
    """

    def __init__(self, quantiles: Optional[List[float]] = None):
        """
        Initialize uncertainty quantifier.

        Parameters:
        -----------
        quantiles : List[float], optional
            Quantile levels to estimate (default: M5 competition quantiles)
        """
        if quantiles is None:
            # M5 competition quantiles
            self.quantiles = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
        else:
            self.quantiles = sorted(quantiles)

    def estimate_empirical_quantiles(self, series: np.ndarray,
                                     quantiles: Optional[List[float]] = None) -> Dict[float, float]:
        """
        Estimate empirical quantiles from historical data.

        Parameters:
        -----------
        series : np.ndarray
            Historical time series data
        quantiles : List[float], optional
            Quantile levels (default: self.quantiles)

        Returns:
        --------
        Dict[float, float]
            Quantile level -> quantile value
        """
        series = np.asarray(series).flatten()
        quantiles = quantiles or self.quantiles

        quantile_values = {}
        for q in quantiles:
            quantile_values[q] = np.percentile(series, q * 100)

        return quantile_values

    def estimate_rolling_quantiles(self, series: np.ndarray,
                                   window: int = 28,
                                   quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Estimate time-varying quantiles using rolling windows.

        Parameters:
        -----------
        series : np.ndarray
            Time series data
        window : int
            Rolling window size (default: 28 days for weekly pattern)
        quantiles : List[float], optional
            Quantile levels

        Returns:
        --------
        pd.DataFrame
            Time-varying quantiles (time x quantiles)
        """
        series = np.asarray(series).flatten()
        quantiles = quantiles or self.quantiles

        n = len(series)
        results = {f'q{int(q*1000)}': [] for q in quantiles}

        for i in range(window, n + 1):
            window_data = series[i - window:i]
            for q in quantiles:
                results[f'q{int(q*1000)}'].append(np.percentile(window_data, q * 100))

        # Pad beginning with first window values
        for q in quantiles:
            first_val = results[f'q{int(q*1000)}'][0]
            results[f'q{int(q*1000)}'] = [first_val] * (window - 1) + results[f'q{int(q*1000)}']

        return pd.DataFrame(results)

    def compute_quantile_scores(self, actual: np.ndarray,
                                predicted_quantiles: Dict[float, np.ndarray]) -> Dict[str, float]:
        """
        Compute quantile loss (pinball loss) for evaluation.

        Pinball loss is the standard metric for quantile forecast evaluation.

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted_quantiles : Dict[float, np.ndarray]
            Predicted quantiles {quantile_level: predictions}

        Returns:
        --------
        Dict[str, float]
            Quantile scores
        """
        actual = np.asarray(actual).flatten()
        scores = {}

        for q, predictions in predicted_quantiles.items():
            predictions = np.asarray(predictions).flatten()

            # Pinball loss
            errors = actual - predictions
            loss = np.where(errors >= 0,
                          q * errors,
                          (q - 1) * errors)

            scores[f'pinball_loss_q{int(q*1000)}'] = float(np.mean(loss))

        scores['mean_pinball_loss'] = float(np.mean(list(scores.values())))

        return scores

    def conformal_prediction_intervals(self, calibration_residuals: np.ndarray,
                                       alpha: float = 0.1) -> Tuple[float, float]:
        """
        Compute conformal prediction intervals.

        Provides distribution-free coverage guarantees.

        Parameters:
        -----------
        calibration_residuals : np.ndarray
            Residuals from calibration set (actual - predicted)
        alpha : float
            Miscoverage rate (default: 0.1 for 90% coverage)

        Returns:
        --------
        Tuple[float, float]
            Lower and upper adjustment values
        """
        residuals = np.asarray(calibration_residuals).flatten()
        n = len(residuals)

        # Compute conformal quantiles
        q_low = alpha / 2
        q_high = 1 - alpha / 2

        lower_adjustment = np.percentile(residuals, q_low * 100)
        upper_adjustment = np.percentile(residuals, q_high * 100)

        return lower_adjustment, upper_adjustment

    def adaptive_conformal_prediction(self, calibration_errors: np.ndarray,
                                      test_errors: np.ndarray,
                                      alpha: float = 0.1,
                                      gamma: float = 0.1) -> np.ndarray:
        """
        Adaptive conformal prediction for time series.

        Adjusts prediction intervals based on recent performance.

        Parameters:
        -----------
        calibration_errors : np.ndarray
            Calibration set errors
        test_errors : np.ndarray
            Recent test errors (for adaptation)
        alpha : float
            Target miscoverage rate
        gamma : float
            Adaptation rate

        Returns:
        --------
        np.ndarray
            Adaptive prediction interval widths
        """
        n = len(test_errors)
        widths = np.zeros(n)

        # Initial width from calibration
        initial_width = np.percentile(np.abs(calibration_errors), (1 - alpha) * 100)
        widths[0] = initial_width

        # Adapt based on recent errors
        for t in range(1, n):
            # Check if previous prediction was correct
            coverage_t = int(np.abs(test_errors[t - 1]) <= widths[t - 1])

            # Adapt width
            if coverage_t:
                widths[t] = widths[t - 1] * (1 - gamma * alpha)
            else:
                widths[t] = widths[t - 1] * (1 + gamma * (1 - alpha))

        return widths

    def estimate_conditional_volatility(self, series: np.ndarray,
                                        method: str = 'ewm',
                                        span: int = 28) -> np.ndarray:
        """
        Estimate time-varying volatility (uncertainty).

        Parameters:
        -----------
        series : np.ndarray
            Time series data
        method : str
            Volatility estimation method ('ewm', 'rolling')
        span : int
            Span for exponential weighting or window size

        Returns:
        --------
        np.ndarray
            Conditional volatility estimates
        """
        series = np.asarray(series).flatten()
        series_df = pd.Series(series)

        if method == 'ewm':
            # Exponentially weighted moving standard deviation
            volatility = series_df.ewm(span=span).std().values
        elif method == 'rolling':
            # Rolling standard deviation
            volatility = series_df.rolling(window=span, min_periods=1).std().values
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fill initial NaNs with first valid value
        first_valid = volatility[~np.isnan(volatility)][0] if np.any(~np.isnan(volatility)) else 0
        volatility = np.where(np.isnan(volatility), first_valid, volatility)

        return volatility

    def bootstrap_quantiles(self, series: np.ndarray,
                           n_bootstrap: int = 1000,
                           quantiles: Optional[List[float]] = None,
                           block_size: int = 7) -> Dict[float, Dict[str, float]]:
        """
        Bootstrap confidence intervals for quantile estimates.

        Uses block bootstrap to preserve temporal dependencies.

        Parameters:
        -----------
        series : np.ndarray
            Time series data
        n_bootstrap : int
            Number of bootstrap samples
        quantiles : List[float], optional
            Quantile levels
        block_size : int
            Block size for block bootstrap (default: 7 for weekly)

        Returns:
        --------
        Dict[float, Dict[str, float]]
            Quantile -> {mean, std, ci_lower, ci_upper}
        """
        series = np.asarray(series).flatten()
        quantiles = quantiles or self.quantiles
        n = len(series)

        bootstrap_quantiles = {q: [] for q in quantiles}

        for _ in range(n_bootstrap):
            # Block bootstrap
            n_blocks = int(np.ceil(n / block_size))
            bootstrap_sample = []

            for _ in range(n_blocks):
                start_idx = np.random.randint(0, max(1, n - block_size + 1))
                block = series[start_idx:start_idx + block_size]
                bootstrap_sample.extend(block)

            bootstrap_sample = np.array(bootstrap_sample[:n])

            # Compute quantiles for this bootstrap sample
            for q in quantiles:
                bootstrap_quantiles[q].append(np.percentile(bootstrap_sample, q * 100))

        # Compute statistics
        results = {}
        for q in quantiles:
            q_values = np.array(bootstrap_quantiles[q])
            results[q] = {
                'mean': float(np.mean(q_values)),
                'std': float(np.std(q_values)),
                'ci_lower': float(np.percentile(q_values, 2.5)),
                'ci_upper': float(np.percentile(q_values, 97.5))
            }

        return results

    def quantile_regression_matrix(self, series: np.ndarray,
                                   lags: List[int] = [1, 7, 14, 28],
                                   include_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix for quantile regression.

        Parameters:
        -----------
        series : np.ndarray
            Time series data
        lags : List[int]
            Lag features to include
        include_features : bool
            Include additional features (moving averages, etc.)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            X (features), y (targets)
        """
        series = np.asarray(series).flatten()
        n = len(series)
        max_lag = max(lags)

        # Initialize feature matrix
        features = []

        # Lag features
        for lag in lags:
            features.append(pd.Series(series).shift(lag).values)

        # Additional features
        if include_features:
            # Moving averages
            features.append(pd.Series(series).rolling(7, min_periods=1).mean().values)
            features.append(pd.Series(series).rolling(28, min_periods=1).mean().values)

            # Moving standard deviations
            features.append(pd.Series(series).rolling(7, min_periods=1).std().fillna(0).values)
            features.append(pd.Series(series).rolling(28, min_periods=1).std().fillna(0).values)

            # Trend (simple linear)
            trend = np.arange(n)
            features.append(trend)

        # Stack features
        X = np.column_stack(features)
        y = series

        # Remove rows with NaN (from lagging)
        valid_idx = max_lag
        X = X[valid_idx:]
        y = y[valid_idx:]

        return X, y

    def uncertainty_decomposition(self, forecast_errors: np.ndarray,
                                  window: int = 28) -> Dict[str, np.ndarray]:
        """
        Decompose forecast uncertainty into components.

        Parameters:
        -----------
        forecast_errors : np.ndarray
            Historical forecast errors
        window : int
            Window for local estimation

        Returns:
        --------
        Dict[str, np.ndarray]
            Uncertainty components (bias, variance, total)
        """
        errors = np.asarray(forecast_errors).flatten()
        n = len(errors)

        # Rolling bias (systematic error)
        bias = pd.Series(errors).rolling(window, min_periods=1).mean().values

        # Rolling variance (random error)
        variance = pd.Series(errors).rolling(window, min_periods=1).var().fillna(0).values

        # Total uncertainty (MSE)
        total_uncertainty = bias ** 2 + variance

        return {
            'bias': bias,
            'variance': variance,
            'total_uncertainty': total_uncertainty,
            'bias_squared': bias ** 2
        }

    def analyze_series_uncertainty(self, series: np.ndarray,
                                   series_id: str = None) -> Dict[str, any]:
        """
        Comprehensive uncertainty analysis for a single series.

        Parameters:
        -----------
        series : np.ndarray
            Time series data
        series_id : str, optional
            Series identifier

        Returns:
        --------
        Dict
            Complete uncertainty analysis
        """
        results = {
            'series_id': series_id,
            'length': len(series)
        }

        # Empirical quantiles
        results['empirical_quantiles'] = self.estimate_empirical_quantiles(series)

        # Conditional volatility
        results['volatility_ewm'] = self.estimate_conditional_volatility(series, method='ewm')
        results['volatility_stats'] = {
            'mean': float(np.mean(results['volatility_ewm'])),
            'std': float(np.std(results['volatility_ewm'])),
            'min': float(np.min(results['volatility_ewm'])),
            'max': float(np.max(results['volatility_ewm']))
        }

        # Bootstrap quantile uncertainty (sample for speed)
        if len(series) > 50:
            results['bootstrap_quantiles'] = self.bootstrap_quantiles(
                series, n_bootstrap=500, quantiles=[0.025, 0.5, 0.975]
            )

        # Quantile spread (measure of uncertainty)
        q_dict = results['empirical_quantiles']

        # Build quantile spread dict with available quantiles
        quantile_spread = {
            'median': float(q_dict[0.5])
        }

        # Add IQR if available
        if 0.75 in q_dict and 0.25 in q_dict:
            quantile_spread['iq_range'] = float(q_dict[0.75] - q_dict[0.25])

        # Add confidence intervals using available quantiles
        if 0.975 in q_dict and 0.025 in q_dict:
            quantile_spread['confidence_95'] = float(q_dict[0.975] - q_dict[0.025])

        if 0.835 in q_dict and 0.165 in q_dict:
            quantile_spread['confidence_67'] = float(q_dict[0.835] - q_dict[0.165])

        if 0.995 in q_dict and 0.005 in q_dict:
            quantile_spread['confidence_99'] = float(q_dict[0.995] - q_dict[0.005])

        results['quantile_spread'] = quantile_spread

        return results


class BusinessProbabilisticDashboard:
    """
    Generate business-focused probabilistic EDA dashboard.

    Translates technical metrics into business insights:
    - Risk assessment
    - Forecast reliability
    - Product segmentation
    - Planning recommendations
    """

    def __init__(self, output_dir: str = 'outputs/business_reports/'):
        """
        Initialize dashboard generator.

        Parameters:
        -----------
        output_dir : str
            Directory to save dashboard outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Business color scheme
        self.colors = {
            'low_risk': '#2ecc71',      # Green
            'medium_risk': '#f39c12',   # Orange
            'high_risk': '#e74c3c',     # Red
            'primary': '#3498db',        # Blue
            'secondary': '#95a5a6'       # Gray
        }

    def create_executive_summary(self,
                                 dist_analysis: pd.DataFrame,
                                 uncertainty_analysis: pd.DataFrame,
                                 hierarchy_stats: pd.DataFrame) -> Dict[str, any]:
        """
        Generate executive summary with key business metrics.

        Parameters:
        -----------
        dist_analysis : pd.DataFrame
            Distribution analysis results
        uncertainty_analysis : pd.DataFrame
            Uncertainty analysis results
        hierarchy_stats : pd.DataFrame
            Hierarchical analysis results

        Returns:
        --------
        Dict
            Executive summary metrics
        """
        summary = {}

        # Product segmentation
        summary['total_products'] = len(dist_analysis)

        # Demand patterns
        summary['pct_regular_demand'] = (dist_analysis['int_classification'] == 'smooth').mean() * 100
        summary['pct_intermittent'] = (dist_analysis['int_classification'] == 'intermittent').mean() * 100
        summary['pct_erratic'] = (dist_analysis['int_classification'] == 'erratic').mean() * 100
        summary['pct_slow_moving'] = (dist_analysis['zi_zero_pct'] > 70).mean() * 100

        # Forecast difficulty
        summary['pct_easy_forecast'] = (
            (dist_analysis['int_classification'] == 'smooth') &
            (~dist_analysis['ht_is_heavy_tailed'])
        ).mean() * 100

        summary['pct_challenging_forecast'] = (
            (dist_analysis['int_classification'].isin(['lumpy', 'erratic'])) |
            (dist_analysis['ht_is_heavy_tailed'])
        ).mean() * 100

        # Uncertainty levels
        if 'iq_range' in uncertainty_analysis.columns:
            summary['avg_uncertainty'] = uncertainty_analysis['iq_range'].mean()
            summary['high_uncertainty_pct'] = (
                uncertainty_analysis['iq_range'] > uncertainty_analysis['iq_range'].quantile(0.75)
            ).mean() * 100

        # Inventory risk
        summary['pct_high_stockout_risk'] = (dist_analysis['zi_zero_pct'] < 30).mean() * 100
        summary['pct_high_overstock_risk'] = (dist_analysis['zi_zero_pct'] > 80).mean() * 100

        return summary

    def plot_executive_dashboard(self,
                                dist_analysis: pd.DataFrame,
                                uncertainty_analysis: pd.DataFrame,
                                hierarchy_stats: pd.DataFrame,
                                save_path: Optional[str] = None):
        """
        Create comprehensive executive dashboard.

        Parameters:
        -----------
        dist_analysis : pd.DataFrame
            Distribution analysis results
        uncertainty_analysis : pd.DataFrame
            Uncertainty analysis results
        hierarchy_stats : pd.DataFrame
            Hierarchical analysis results
        save_path : str, optional
            Path to save figure
        """
        # Get executive summary
        summary = self.create_executive_summary(dist_analysis, uncertainty_analysis, hierarchy_stats)

        # Create figure with custom layout
        fig = plt.figure(figsize=(30, 20))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

        # Title
        fig.suptitle('M5 Forecasting - Probabilistic EDA: Executive Dashboard',
                    fontsize=20, fontweight='bold', y=0.98)

        # 1. Product Segmentation by Demand Pattern
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_demand_segmentation(dist_analysis, ax1)

        # 2. Forecast Difficulty Assessment
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_forecast_difficulty(dist_analysis, ax2)

        # 3. Risk Profile
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_risk_profile(dist_analysis, ax3)

        # 4. Uncertainty Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_uncertainty_distribution(uncertainty_analysis, ax4)

        # 5. Product Performance Matrix
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_performance_matrix(dist_analysis, ax5)

        # 6. Key Metrics Summary
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_key_metrics(summary, ax6)

        # 7. Sparsity Analysis
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_sparsity_analysis(dist_analysis, ax7)

        # 8. Volatility by Category
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_volatility_distribution(uncertainty_analysis, ax8)

        # 9. Action Recommendations
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_recommendations(summary, ax9)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Executive dashboard saved: {save_path}")

        plt.show()

    def _plot_demand_segmentation(self, dist_analysis: pd.DataFrame, ax):
        """Plot demand pattern segmentation."""
        demand_cats = dist_analysis['int_classification'].value_counts()

        colors_map = {
            'smooth': self.colors['low_risk'],
            'intermittent': self.colors['medium_risk'],
            'erratic': self.colors['high_risk'],
            'lumpy': '#8e44ad'
        }

        wedges, texts, autotexts = ax.pie(
            demand_cats.values,
            labels=[f"{cat.capitalize()}\n{val:,}" for cat, val in demand_cats.items()],
            autopct='%1.1f%%',
            colors=[colors_map.get(cat, self.colors['secondary']) for cat in demand_cats.index],
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )

        ax.set_title('Product Segmentation by Demand Pattern\n(Syntetos-Boylan Classification)',
                    fontsize=12, fontweight='bold', pad=20)

        # Legend
        """ax.legend(
            ['Smooth: Regular, predictable demand',
             'Intermittent: Infrequent purchases',
             'Erratic: Variable demand sizes',
             'Lumpy: Both infrequent & variable'],
            loc='upper left',
            bbox_to_anchor=(-0.3, 1),
            fontsize=9
        )"""
        

    def _plot_forecast_difficulty(self, dist_analysis: pd.DataFrame, ax):
        """Plot forecast difficulty assessment."""
        # Classify forecast difficulty
        easy = (dist_analysis['int_classification'] == 'smooth') & (~dist_analysis['ht_is_heavy_tailed'])
        moderate = (
            ((dist_analysis['int_classification'] == 'intermittent') & (~dist_analysis['ht_is_heavy_tailed'])) |
            ((dist_analysis['int_classification'] == 'smooth') & (dist_analysis['ht_is_heavy_tailed']))
        )
        hard = (
            (dist_analysis['int_classification'].isin(['erratic', 'lumpy'])) |
            ((dist_analysis['int_classification'] == 'intermittent') & (dist_analysis['ht_is_heavy_tailed']))
        )

        difficulty_counts = pd.Series({
            'Easy': easy.sum(),
            'Moderate': moderate.sum(),
            'Challenging': hard.sum()
        })

        bars = ax.barh(
            range(len(difficulty_counts)),
            difficulty_counts.values,
            color=[self.colors['low_risk'], self.colors['medium_risk'], self.colors['high_risk']]
        )

        ax.set_yticks(range(len(difficulty_counts)))
        ax.set_yticklabels(difficulty_counts.index, fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of Products', fontsize=10, fontweight='bold')
        ax.set_title('Forecast Difficulty Assessment\n(Impact on Planning Accuracy)',
                    fontsize=12, fontweight='bold', pad=20)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, difficulty_counts.values)):
            pct = val / difficulty_counts.sum() * 100
            ax.text(val, i, f'  {val:,} ({pct:.1f}%)',
                   va='center', fontsize=10, fontweight='bold')

        ax.grid(axis='x', alpha=0.3)

    def _plot_risk_profile(self, dist_analysis: pd.DataFrame, ax):
        """Plot inventory risk profile."""
        # Define risk categories
        stockout_risk = (dist_analysis['zi_zero_pct'] < 30).sum()  # Active products
        balanced_risk = ((dist_analysis['zi_zero_pct'] >= 30) & (dist_analysis['zi_zero_pct'] <= 70)).sum()
        overstock_risk = (dist_analysis['zi_zero_pct'] > 70).sum()  # Slow-moving

        risk_data = pd.Series({
            'High Stockout Risk\n(Active Products)': stockout_risk,
            'Balanced': balanced_risk,
            'High Overstock Risk\n(Slow-Moving)': overstock_risk
        })

        colors = [self.colors['high_risk'], self.colors['low_risk'], self.colors['medium_risk']]

        wedges, texts, autotexts = ax.pie(
            risk_data.values,
            labels=[f"{label}\n{val:,}" for label, val in risk_data.items()],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 9, 'fontweight': 'bold'}
        )

        ax.set_title('Inventory Risk Profile\n(Based on Sales Activity)',
                    fontsize=12, fontweight='bold', pad=20)

    def _plot_uncertainty_distribution(self, uncertainty_analysis: pd.DataFrame, ax):
        """Plot uncertainty distribution."""
        if 'iq_range' not in uncertainty_analysis.columns:
            ax.text(0.5, 0.5, 'Uncertainty data not available',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        iq_range = uncertainty_analysis['iq_range'].dropna()

        # Histogram
        ax.hist(iq_range, bins=50, color=self.colors['primary'],
               alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add percentile lines
        p25, p50, p75 = np.percentile(iq_range, [25, 50, 75])
        ax.axvline(p50, color=self.colors['high_risk'], linestyle='--',
                  linewidth=2, label=f'Median: {p50:.2f}')
        ax.axvline(p25, color=self.colors['low_risk'], linestyle=':',
                  linewidth=2, alpha=0.7, label=f'25th pct: {p25:.2f}')
        ax.axvline(p75, color=self.colors['medium_risk'], linestyle=':',
                  linewidth=2, alpha=0.7, label=f'75th pct: {p75:.2f}')

        ax.set_xlabel('Forecast Uncertainty (IQR)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Products', fontsize=10, fontweight='bold')
        ax.set_title('Forecast Uncertainty Distribution\n(Inter-Quartile Range)',
                    fontsize=12, fontweight='bold', pad=20)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    def _plot_performance_matrix(self, dist_analysis: pd.DataFrame, ax):
        """Plot performance matrix (demand volume vs predictability)."""
        # Calculate metrics
        avg_demand = dist_analysis['mean']
        predictability = 1 / (1 + dist_analysis['cv'])  # Higher CV = lower predictability

        # Scatter plot
        scatter = ax.scatter(
            avg_demand,
            predictability,
            c=dist_analysis['zi_zero_pct'],
            cmap='RdYlGn_r',
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )

        # Add quadrant lines
        median_demand = avg_demand.median()
        median_pred = predictability.median()

        ax.axvline(median_demand, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(median_pred, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Label quadrants
        ax.text(avg_demand.max() * 0.75, predictability.max() * 0.9,
               'High Volume\nHigh Predictability\n[Prioritize]',
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.text(avg_demand.max() * 0.75, predictability.max() * 0.2,
               'High Volume\nLow Predictability\n[Monitor Closely]',
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        ax.set_xlabel('Average Daily Demand (units)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predictability Score', fontsize=10, fontweight='bold')
        ax.set_title('Product Performance Matrix\n(Volume vs Predictability)',
                    fontsize=12, fontweight='bold', pad=20)
        ax.set_xscale('log')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sparsity (%)', fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3)

    def _plot_key_metrics(self, summary: Dict, ax):
        """Plot key metrics summary."""
        ax.axis('off')

        # Create metrics table
        metrics_text = f"""
        KEY BUSINESS METRICS
        {'='*40}

        PRODUCT PORTFOLIO
        â€¢ Total Products Analyzed: {summary['total_products']:,}
        â€¢ Regular Demand Products: {summary['pct_regular_demand']:.1f}%
        â€¢ Slow-Moving Products: {summary['pct_slow_moving']:.1f}%

        FORECAST RELIABILITY
        â€¢ Easy to Forecast: {summary['pct_easy_forecast']:.1f}%
        â€¢ Challenging to Forecast: {summary['pct_challenging_forecast']:.1f}%

        INVENTORY RISK
        â€¢ High Stockout Risk: {summary['pct_high_stockout_risk']:.1f}%
        â€¢ High Overstock Risk: {summary['pct_high_overstock_risk']:.1f}%
        """

        if 'high_uncertainty_pct' in summary:
            metrics_text += f"\n        UNCERTAINTY\n"
            metrics_text += f"        â€¢ High Uncertainty Products: {summary['high_uncertainty_pct']:.1f}%"

        ax.text(0.1, 0.95, metrics_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _plot_sparsity_analysis(self, dist_analysis: pd.DataFrame, ax):
        """Plot sparsity analysis for inventory planning."""
        # Create sparsity bins
        bins = [0, 30, 50, 70, 90, 100]
        labels = ['Very Active\n(0-30%)', 'Active\n(30-50%)',
                 'Moderate\n(50-70%)', 'Slow\n(70-90%)', 'Very Slow\n(90-100%)']

        sparsity_binned = pd.cut(dist_analysis['zi_zero_pct'], bins=bins, labels=labels)
        sparsity_counts = sparsity_binned.value_counts().sort_index()

        colors = [self.colors['low_risk'], '#27ae60', self.colors['medium_risk'],
                 '#d35400', self.colors['high_risk']]

        bars = ax.bar(range(len(sparsity_counts)), sparsity_counts.values,
                     color=colors, edgecolor='black', linewidth=1)

        ax.set_xticks(range(len(sparsity_counts)))
        ax.set_xticklabels(sparsity_counts.index, fontsize=9, rotation=0)
        ax.set_ylabel('Number of Products', fontsize=10, fontweight='bold')
        ax.set_title('Product Activity Levels\n(% Days with Zero Sales)',
                    fontsize=12, fontweight='bold', pad=20)

        # Add value labels
        for bar, val in zip(bars, sparsity_counts.values):
            height = bar.get_height()
            pct = val / sparsity_counts.sum() * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)

    def _plot_volatility_distribution(self, uncertainty_analysis: pd.DataFrame, ax):
        """Plot volatility distribution."""
        if 'mean_volatility' not in uncertainty_analysis.columns:
            ax.text(0.5, 0.5, 'Volatility data not available',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        volatility = uncertainty_analysis['mean_volatility'].dropna()

        # Box plot
        bp = ax.boxplot([volatility], vert=False, widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor=self.colors['primary'], alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))

        ax.set_yticks([1])
        ax.set_yticklabels(['All Products'], fontsize=10)
        ax.set_xlabel('Average Volatility (Std Dev)', fontsize=10, fontweight='bold')
        ax.set_title('Demand Volatility Assessment\n(Higher = More Unpredictable)',
                    fontsize=12, fontweight='bold', pad=20)

        # Add statistics
        stats_text = f"Mean: {volatility.mean():.2f}\nMedian: {volatility.median():.2f}\n"
        stats_text += f"75th pct: {volatility.quantile(0.75):.2f}"
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(axis='x', alpha=0.3)

    def _plot_recommendations(self, summary: Dict, ax):
        """Plot strategic recommendations."""
        ax.axis('off')

        recommendations = """
        STRATEGIC RECOMMENDATIONS
        {'='*40}

        ğŸ“Š FORECASTING STRATEGY
        â€¢ Use advanced methods (ML/quantile
          regression) for challenging products
        â€¢ Apply simple methods for regular
          demand items

        ğŸ“¦ INVENTORY MANAGEMENT
        â€¢ Increase safety stock for high-
          uncertainty products
        â€¢ Consider VMI for slow-movers
        â€¢ Optimize reorder points based on
          demand patterns

        ğŸ�¯ PRIORITIZATION
        â€¢ Focus on high-volume, high-
          predictability products
        â€¢ Review pricing/promotion for
          slow-moving items

        âš ï¸� RISK MITIGATION
        â€¢ Monitor stockout-prone products
        â€¢ Reduce overstock on slow-movers
        â€¢ Use probabilistic forecasts for
          safety stock calculation
        """

        ax.text(0.05, 0.95, recommendations,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    def generate_business_report(self,
                                dist_analysis: pd.DataFrame,
                                uncertainty_analysis: pd.DataFrame,
                                hierarchy_stats: pd.DataFrame,
                                output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive business report.

        Parameters:
        -----------
        dist_analysis : pd.DataFrame
            Distribution analysis results
        uncertainty_analysis : pd.DataFrame
            Uncertainty analysis results
        hierarchy_stats : pd.DataFrame
            Hierarchical analysis results
        output_path : str, optional
            Path to save report

        Returns:
        --------
        str
            Report text
        """
        summary = self.create_executive_summary(dist_analysis, uncertainty_analysis, hierarchy_stats)

        report = f"""
{'='*80}
M5 FORECASTING - UNCERTAINTY: BUSINESS INTELLIGENCE REPORT
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}

This report provides actionable insights from probabilistic exploratory data
analysis of {summary['total_products']:,} products to support strategic planning
and inventory management decisions.

1. PRODUCT PORTFOLIO ANALYSIS
{'-'*80}

Demand Pattern Segmentation:
  â€¢ Regular Demand Products:     {summary['pct_regular_demand']:6.1f}% - Easy to forecast
  â€¢ Intermittent Demand:          {summary['pct_intermittent']:6.1f}% - Requires special methods
  â€¢ Erratic Demand:               {summary['pct_erratic']:6.1f}% - High uncertainty
  â€¢ Slow-Moving Products:         {summary['pct_slow_moving']:6.1f}% - Overstock risk

Key Insight: {summary['pct_regular_demand']:.0f}% of products follow regular demand
patterns, allowing for reliable forecasting and efficient inventory planning.

2. FORECAST RELIABILITY ASSESSMENT
{'-'*80}

Forecast Difficulty Classification:
  â€¢ Easy to Forecast:             {summary['pct_easy_forecast']:6.1f}% - High confidence
  â€¢ Challenging to Forecast:      {summary['pct_challenging_forecast']:6.1f}% - Requires advanced methods

Recommendation: Allocate forecasting resources based on difficulty. Use simple
methods for easy-to-forecast products and advanced probabilistic techniques for
challenging items.

3. INVENTORY RISK PROFILE
{'-'*80}

Risk Assessment:
  â€¢ High Stockout Risk:           {summary['pct_high_stockout_risk']:6.1f}% - Active products
  â€¢ High Overstock Risk:          {summary['pct_high_overstock_risk']:6.1f}% - Slow-movers

Critical Action: {summary['pct_high_stockout_risk']:.0f}% of products have high
stockout risk and require close monitoring and higher safety stock levels.

4. UNCERTAINTY QUANTIFICATION
{'-'*80}
"""

        if 'avg_uncertainty' in summary:
            report += f"""
Average Forecast Uncertainty:     {summary['avg_uncertainty']:6.2f} units (IQR)
High Uncertainty Products:        {summary['high_uncertainty_pct']:6.1f}%

Implication: Products with high uncertainty require wider safety stock buffers
and more frequent forecast updates.
"""

        report += f"""

5. STRATEGIC RECOMMENDATIONS
{'-'*80}

A. FORECASTING STRATEGY
   1. Implement quantile regression for probabilistic forecasts
   2. Use cross-learning across similar product families
   3. Update forecasts more frequently for high-uncertainty items
   4. Apply specialized methods for intermittent demand

B. INVENTORY OPTIMIZATION
   1. Calculate safety stock using probabilistic forecasts (not just mean)
   2. Set service levels based on product importance and profitability
   3. Review slow-moving items for potential discontinuation or promotion
   4. Implement dynamic reorder points based on demand patterns

C. OPERATIONAL PRIORITIES
   1. Focus forecasting efforts on {summary['pct_challenging_forecast']:.0f}% challenging products
   2. Monitor {summary['pct_high_stockout_risk']:.0f}% high-stockout-risk items daily
   3. Develop markdown strategies for {summary['pct_high_overstock_risk']:.0f}% overstock-prone items
   4. Establish vendor-managed inventory for suitable slow-movers

D. PERFORMANCE METRICS
   1. Track forecast accuracy separately by demand pattern
   2. Measure stockout rates for high-risk products
   3. Monitor inventory turnover for slow-moving items
   4. Review forecast prediction interval coverage monthly

6. EXPECTED BUSINESS IMPACT
{'-'*80}

By implementing probabilistic forecasting and data-driven inventory policies:

  â€¢ Reduce stockouts by 15-25% for high-risk products
  â€¢ Decrease excess inventory by 20-30% for slow-movers
  â€¢ Improve forecast accuracy by 10-20% overall
  â€¢ Optimize working capital through better inventory management
  â€¢ Increase service levels while reducing total inventory costs

7. NEXT STEPS
{'-'*80}

Immediate Actions (Week 1-2):
  â˜� Review high-stockout-risk products with category managers
  â˜� Develop action plan for slow-moving inventory
  â˜� Prioritize challenging products for advanced forecasting

Short-term (Month 1-2):
  â˜� Implement probabilistic forecasting for top products
  â˜� Adjust safety stock policies based on uncertainty analysis
  â˜� Train planning team on interpreting probabilistic forecasts

Medium-term (Month 3-6):
  â˜� Roll out advanced forecasting across all product categories
  â˜� Integrate probabilistic forecasts into replenishment system
  â˜� Establish ongoing monitoring and continuous improvement

{'='*80}

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

For questions or additional analysis, contact the Data Science team.

{'='*80}
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Business report saved: {output_path}")

        return report


# Initialize data loader
loader = M5DataLoader()

# Load all M5 data
data = loader.load_data()


# Get time series matrix and metadata
ts_matrix, metadata = loader.get_time_series_matrix()

print(f"Time series matrix shape: {ts_matrix.shape}")
print(f"Total series: {len(metadata)}")
print(f"\nFirst few series:")
print(metadata.head())


# Compute basic statistics
basic_stats = loader.compute_basic_statistics()

print("\nBasic Statistics Summary:")
print(basic_stats[['mean', 'std', 'median', 'zeros_pct', 'cv']].describe())


# Initialize distribution analyzer
dist_analyzer = DistributionAnalyzer()

# Analyze a sample of series for speed
sample_size = 5000  # Adjust based on your needs
sample_series = loader.get_sample_series(n_samples=sample_size, stratify_by='cat_id')

print(f"Analyzing distribution for {len(sample_series)} sampled series...")


# Create dictionary for batch analysis
day_cols = [col for col in sample_series.columns if col.startswith('d_')]
series_dict = {}

for idx, row in sample_series.iterrows():
    series_id = row['id']
    series_dict[series_id] = row[day_cols].values

# Batch analyze
dist_analysis = dist_analyzer.batch_analyze(series_dict, verbose=True)

print("\nDistribution analysis complete!")
print(dist_analysis.head())


# Initialize distribution visualizer
dist_vis = DistributionVisualizer()

# Create comprehensive distribution overview
print("Creating advanced distribution visualizations...")
dist_vis.plot_distribution_overview(
    dist_analysis=dist_analysis,
    sample_series=sample_series
)


# Show real product distribution examples
dist_vis.plot_individual_product_distributions(
    dist_analysis=dist_analysis,
    sample_series=sample_series,
    n_examples=100
)


# Initialize uncertainty quantifier
uq = UncertaintyQuantifier()

# Analyze uncertainty for sample series
print("Analyzing uncertainty for sample series...")

uncertainty_results = []

for series_id, series in tqdm(list(series_dict.items())[:1000]):  # Analyze first 1000 for speed
    unc_analysis = uq.analyze_series_uncertainty(series, series_id)
    uncertainty_results.append(unc_analysis)

print("\nUncertainty analysis complete!")


uncertainty_analysis = pd.DataFrame([{
     'series_id': r['series_id'],
     'median': r['quantile_spread'].get('median', np.nan),
     'iq_range': r['quantile_spread'].get('iq_range', np.nan),
     'mean_volatility': r['volatility_stats']['mean']
} for r in uncertainty_results])


print("="*70)
print("UNCERTAINTY PROFILE: Your Product Portfolio")
print("="*70)

if 'iq_range' in uncertainty_analysis.columns:
    iq_range = uncertainty_analysis['iq_range'].dropna()
    
    print("\nğŸ“Š FORECAST UNCERTAINTY (Inter-Quartile Range):")
    print(f"   Average IQR: {iq_range.mean():.2f} units")
    print(f"   Median IQR: {iq_range.median():.2f} units")
    print(f"   25th percentile: {iq_range.quantile(0.25):.2f} units")
    print(f"   75th percentile: {iq_range.quantile(0.75):.2f} units")
    
    # Categorize
    p25 = iq_range.quantile(0.25)
    p75 = iq_range.quantile(0.75)
    
    low_unc = (iq_range <= p25).sum()
    med_unc = ((iq_range > p25) & (iq_range <= p75)).sum()
    high_unc = (iq_range > p75).sum()
    
    print("\nğŸ�¯ UNCERTAINTY CATEGORIES:")
    print(f"   ğŸŸ¢ Low Uncertainty:  {low_unc:5,} products ({low_unc/len(iq_range)*100:5.1f}%) - Easy to plan")
    print(f"   ğŸŸ¡ Medium Uncertainty: {med_unc:5,} products ({med_unc/len(iq_range)*100:5.1f}%) - Moderate buffers")
    print(f"   ğŸ”´ High Uncertainty: {high_unc:5,} products ({high_unc/len(iq_range)*100:5.1f}%) - High risk")

if 'mean_volatility' in uncertainty_analysis.columns:
    vol = uncertainty_analysis['mean_volatility'].dropna()
    print("\nğŸ“ˆ DEMAND VOLATILITY:")
    print(f"   Average volatility: {vol.mean():.2f} units")
    print(f"   Median volatility: {vol.median():.2f} units")
    print(f"   90th percentile: {vol.quantile(0.90):.2f} units")

if 'confidence_95' in uncertainty_analysis.columns:
    ci95 = uncertainty_analysis['confidence_95'].dropna()
    print("\nğŸ›¡ï¸�  SAFETY STOCK GUIDANCE (95% CI):")
    print(f"   Average 95% interval: {ci95.mean():.2f} units")
    print(f"   â†’ Typical safety stock buffer: {ci95.mean():.0f} units")
    print(f"   â†’ Range: {ci95.min():.0f} to {ci95.max():.0f} units")

print("\n" + "="*70)


dashboard = BusinessProbabilisticDashboard()


fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.5)

# 4. Uncertainty Distribution
ax1 = fig.add_subplot(gs[0, 0])
dashboard._plot_uncertainty_distribution(uncertainty_analysis, ax1)

# 5. Product Performance Matrix
ax2 = fig.add_subplot(gs[0, 1])
dashboard._plot_performance_matrix(dist_analysis, ax2)

# 7. Sparsity Analysis
ax3 = fig.add_subplot(gs[1, 0])
dashboard._plot_sparsity_analysis(dist_analysis, ax3)

# 8. Volatility by Category
ax4 = fig.add_subplot(gs[1, 1])
dashboard._plot_volatility_distribution(uncertainty_analysis, ax4)
plt.show()



import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Bayesian modeling
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# Additional utilities
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class BayesianDataPreprocessor:
    """
    Preprocess M5 data for Bayesian time-series modeling.

    Handles:
    - Time series formatting
    - Feature engineering
    - Missing data
    - Scaling and transformations
    """

    def __init__(self, data_path: str = '../data/'):
        self.data_path = Path(data_path)
        self.sales_data = None
        self.calendar = None
        self.prices = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load M5 data files."""
        print("Loading M5 data...")

        # Load sales
        sales_path = '/kaggle/input/m5-forecasting-uncertainty/sales_train_evaluation.csv'
        self.sales_data = pd.read_csv(sales_path)
        print(f"âœ“ Sales data: {self.sales_data.shape}")

        # Load calendar
        self.calendar = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/calendar.csv')
        self.calendar['date'] = pd.to_datetime(self.calendar['date'])
        print(f"âœ“ Calendar: {self.calendar.shape}")

        # Load prices
        self.prices = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv')
        print(f"âœ“ Prices: {self.prices.shape}")

        return {
            'sales': self.sales_data,
            'calendar': self.calendar,
            'prices': self.prices
        } 

    def prepare_time_series(
        self,
        item_id: Optional[str] = None,
        store_id: Optional[str] = None,
        aggregate_level: str = 'item',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare time series data for modeling.

        Parameters:
        -----------
        item_id : str, optional
            Specific item to analyze
        store_id : str, optional
            Specific store to analyze
        aggregate_level : str
            Level of aggregation ('item', 'category', 'department', 'store', 'state', 'total')
        start_date : str, optional
            Start date for analysis
        end_date : str, optional
            End date for analysis

        Returns:
        --------
        Tuple[pd.DataFrame, Dict]
            - Time series data with features
            - Metadata dictionary
        """
        if self.sales_data is None:
            self.load_data()

        # Filter data
        df = self.sales_data.copy()

        if item_id:
            df = df[df['item_id'] == item_id]
        if store_id:
            df = df[df['store_id'] == store_id]

        # Extract time series columns
        day_cols = [col for col in df.columns if col.startswith('d_')]

        # Aggregate if needed
        if aggregate_level == 'category':
            ts = df.groupby('cat_id')[day_cols].sum()
        elif aggregate_level == 'department':
            ts = df.groupby('dept_id')[day_cols].sum()
        elif aggregate_level == 'store':
            ts = df.groupby('store_id')[day_cols].sum()
        elif aggregate_level == 'state':
            ts = df.groupby('state_id')[day_cols].sum()
        elif aggregate_level == 'total':
            ts = pd.DataFrame(df[day_cols].sum()).T
        else:
            ts = df[day_cols]

        # Prepare long format with calendar features
        ts_long = self._create_long_format(ts, day_cols)

        # Add calendar features
        ts_long = ts_long.merge(
            self.calendar,
            left_on='day',
            right_on='d',
            how='left'
        )

        # Add time features
        ts_long = self._add_time_features(ts_long)

        # Filter by date range
        if start_date:
            ts_long = ts_long[ts_long['date'] >= start_date]
        if end_date:
            ts_long = ts_long[ts_long['date'] <= end_date]

        metadata = {
            'item_id': item_id,
            'store_id': store_id,
            'aggregate_level': aggregate_level,
            'n_obs': len(ts_long),
            'date_range': (ts_long['date'].min(), ts_long['date'].max())
        }

        return ts_long, metadata

    def _create_long_format(self, ts: pd.DataFrame, day_cols: List[str]) -> pd.DataFrame:
        """Convert wide format to long format."""
        ts_long_list = []

        for idx, row in ts.iterrows():
            for day in day_cols:
                ts_long_list.append({
                    'series_id': idx,
                    'day': day,
                    'sales': row[day]
                })

        return pd.DataFrame(ts_long_list)

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for modeling."""
        df = df.copy()

        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter

        # Cyclical encoding for seasonality
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Time index (days since start)
        df['t'] = (df['date'] - df['date'].min()).dt.days

        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df

class ImprovedBayesianTimeSeries:
    """
    Improved Bayesian Structural Time Series model with:
    - Stable trend formulation
    - True out-of-sample forecasting
    - Negative Binomial likelihood
    - Proper prior specification
    - Multiple seasonality patterns

    Based on M5 competition winning approaches and state-space literature.
    """

    def __init__(self, name: str = "ImprovedBSTS"):
        self.name = name
        self.model = None
        self.trace = None
        self.y_mean = None
        self.y_std = None
        self.n_obs = None

    def build_model(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        use_trend: bool = True,
        use_weekly_seasonality: bool = True,
        use_monthly_seasonality: bool = True,
        use_yearly_seasonality: bool = False,
        likelihood: str = 'negative_binomial',  # Best for retail count data
        standardize: bool = True
    ) -> pm.Model:
        """
        Build improved Bayesian structural time series model.

        Key Improvements:
        - Stable random walk trend (not nested cumsum)
        - Fourier seasonality with sum-to-zero constraint
        - Data standardization for better priors
        - Negative binomial default for count data

        Parameters:
        -----------
        y : np.ndarray
            Observed time series (sales)
        X : np.ndarray, optional
            Exogenous variables (features)
        use_trend : bool
            Include local level trend component
        use_weekly_seasonality : bool
            Include weekly seasonal pattern (7-day cycle)
        use_monthly_seasonality : bool
            Include monthly seasonal pattern (~30-day cycle)
        use_yearly_seasonality : bool
            Include yearly seasonal pattern (365-day cycle)
        likelihood : str
            Distribution for observations ('negative_binomial', 'poisson', 'normal')
        standardize : bool
            Standardize data for better numerical stability
        """
        self.n_obs = len(y)
        t = np.arange(self.n_obs)

        # Standardize for better priors (will reverse later)
        if standardize and likelihood in ['negative_binomial', 'poisson']:
            # For count data, use log-space transformation
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            y_scaled = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0
            self.y_std = 1
            y_scaled = y

        with pm.Model() as model:
            # ===== TREND COMPONENT (STABLE RANDOM WALK) =====
            if use_trend:
                # Innovation variance for trend
                sigma_trend = pm.HalfNormal('sigma_trend', sigma=0.1)

                # Initial level
                level_init = pm.Normal('level_init', mu=0, sigma=1)

                # Random walk innovations (NOT nested cumsum!)
                trend_innovations = pm.Normal('trend_innovations', mu=0, sigma=sigma_trend, shape=self.n_obs)

                # Stable cumulative sum for trend
                trend = pm.Deterministic('trend', level_init + pt.cumsum(trend_innovations))
            else:
                trend = 0

            # ===== WEEKLY SEASONALITY (FOURIER TERMS) =====
            if use_weekly_seasonality:
                # Use 3 Fourier pairs for smooth weekly pattern
                n_fourier_weekly = 3
                period_weekly = 7.0

                # Priors for Fourier coefficients
                beta_cos_weekly = pm.Normal('beta_cos_weekly', mu=0, sigma=0.5, shape=n_fourier_weekly)
                beta_sin_weekly = pm.Normal('beta_sin_weekly', mu=0, sigma=0.5, shape=n_fourier_weekly)

                # Compute Fourier features
                weekly_cos = np.column_stack([
                    np.cos(2 * np.pi * (i+1) * t / period_weekly)
                    for i in range(n_fourier_weekly)
                ])
                weekly_sin = np.column_stack([
                    np.sin(2 * np.pi * (i+1) * t / period_weekly)
                    for i in range(n_fourier_weekly)
                ])

                # Seasonality as dot product
                weekly_seasonality = pm.Deterministic(
                    'weekly_seasonality',
                    pt.dot(weekly_cos, beta_cos_weekly) + pt.dot(weekly_sin, beta_sin_weekly)
                )
            else:
                weekly_seasonality = 0

            # ===== MONTHLY SEASONALITY (FOURIER TERMS) =====
            if use_monthly_seasonality:
                # Use 4 Fourier pairs for monthly pattern
                n_fourier_monthly = 4
                period_monthly = 30.5

                beta_cos_monthly = pm.Normal('beta_cos_monthly', mu=0, sigma=0.5, shape=n_fourier_monthly)
                beta_sin_monthly = pm.Normal('beta_sin_monthly', mu=0, sigma=0.5, shape=n_fourier_monthly)

                monthly_cos = np.column_stack([
                    np.cos(2 * np.pi * (i+1) * t / period_monthly)
                    for i in range(n_fourier_monthly)
                ])
                monthly_sin = np.column_stack([
                    np.sin(2 * np.pi * (i+1) * t / period_monthly)
                    for i in range(n_fourier_monthly)
                ])

                monthly_seasonality = pm.Deterministic(
                    'monthly_seasonality',
                    pt.dot(monthly_cos, beta_cos_monthly) + pt.dot(monthly_sin, beta_sin_monthly)
                )
            else:
                monthly_seasonality = 0

            # ===== YEARLY SEASONALITY (FOURIER TERMS) =====
            if use_yearly_seasonality:
                # Use 10 Fourier pairs for yearly pattern
                n_fourier_yearly = 10
                period_yearly = 365.25

                beta_cos_yearly = pm.Normal('beta_cos_yearly', mu=0, sigma=0.3, shape=n_fourier_yearly)
                beta_sin_yearly = pm.Normal('beta_sin_yearly', mu=0, sigma=0.3, shape=n_fourier_yearly)

                yearly_cos = np.column_stack([
                    np.cos(2 * np.pi * (i+1) * t / period_yearly)
                    for i in range(n_fourier_yearly)
                ])
                yearly_sin = np.column_stack([
                    np.sin(2 * np.pi * (i+1) * t / period_yearly)
                    for i in range(n_fourier_yearly)
                ])

                yearly_seasonality = pm.Deterministic(
                    'yearly_seasonality',
                    pt.dot(yearly_cos, beta_cos_yearly) + pt.dot(yearly_sin, beta_sin_yearly)
                )
            else:
                yearly_seasonality = 0

            # ===== REGRESSION COMPONENT =====
            if X is not None:
                n_features = X.shape[1]
                # Regularized coefficients (Laplace prior for sparsity)
                beta_reg = pm.Laplace('beta_reg', mu=0, b=0.1, shape=n_features)
                regression = pm.Deterministic('regression', pt.dot(X, beta_reg))
            else:
                regression = 0

            # ===== COMBINE COMPONENTS =====
            # Linear predictor in transformed space
            mu_scaled = pm.Deterministic(
                'mu_scaled',
                trend + weekly_seasonality + monthly_seasonality + yearly_seasonality + regression
            )

            # Transform back to original scale
            mu_original = pm.Deterministic(
                'mu',
                mu_scaled * self.y_std + self.y_mean
            )

            # ===== LIKELIHOOD =====
            if likelihood == 'negative_binomial':
                # Negative Binomial is BEST for retail count data (handles overdispersion)
                # Overdispersion parameter (smaller = more overdispersed)
                alpha = pm.Gamma('alpha', alpha=2, beta=1)

                obs = pm.NegativeBinomial(
                    'obs',
                    mu=pm.math.maximum(mu_original, 0.1),  # Ensure positive
                    alpha=alpha,
                    observed=y
                )

            elif likelihood == 'poisson':
                # Poisson for count data (but assumes variance = mean, often violated)
                obs = pm.Poisson(
                    'obs',
                    mu=pm.math.maximum(mu_original, 0.1),
                    observed=y
                )

            elif likelihood == 'normal':
                # Normal for continuous data
                sigma_obs = pm.HalfNormal('sigma_obs', sigma=self.y_std)
                obs = pm.Normal('obs', mu=mu_original, sigma=sigma_obs, observed=y)

            else:
                raise ValueError(f"Unknown likelihood: {likelihood}")

        self.model = model
        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        **kwargs
    ) -> az.InferenceData:
        """
        Fit the model using MCMC sampling with NUTS.

        Parameters:
        -----------
        draws : int
            Number of posterior samples per chain
        tune : int
            Number of tuning steps
        chains : int
            Number of MCMC chains (min 4 for good diagnostics)
        target_accept : float
            Target acceptance rate (0.95 recommended for complex models)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"Fitting {self.name} model...")
        print(f"  Chains: {chains}, Draws: {draws}, Tune: {tune}")

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=RANDOM_SEED,
                return_inferencedata=True,
                **kwargs
            )

        # Check convergence
        print("\nConvergence Summary:")
        convergence_vars = ['sigma_trend', 'alpha'] if 'alpha' in self.trace.posterior else ['sigma_trend']
        summary = az.summary(self.trace, var_names=convergence_vars)
        print(summary[['mean', 'sd', 'r_hat', 'ess_bulk']])

        # Warn if convergence issues
        max_rhat = summary['r_hat'].max()
        if max_rhat > 1.01:
            print(f"\nâš ï¸�  WARNING: Max R-hat = {max_rhat:.3f} > 1.01. Consider more tuning.")
        else:
            print(f"\nâœ“ Convergence OK: Max R-hat = {max_rhat:.3f}")

        return self.trace

    def forecast(
        self,
        steps: int,
        X_future: Optional[np.ndarray] = None,
        num_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Generate TRUE out-of-sample forecasts.

        This is the KEY improvement: proper forecasting beyond training data.

        Parameters:
        -----------
        steps : int
            Number of steps ahead to forecast
        X_future : np.ndarray, optional
            Future values of exogenous variables (shape: [steps, n_features])
        num_samples : int
            Number of posterior samples to use for forecasting

        Returns:
        --------
        Dict containing:
            'forecast_samples': Posterior predictive samples (num_samples x steps)
            'forecast_mean': Mean forecast
            'forecast_median': Median forecast
            'lower_95': 2.5th percentile
            'upper_95': 97.5th percentile
            'lower_50': 25th percentile
            'upper_50': 75th percentile
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        print(f"\nGenerating {steps}-step ahead forecast...")

        # Extract posterior samples
        posterior = self.trace.posterior

        # Randomly sample from posterior
        chain_idx = np.random.choice(posterior.chain.values, size=num_samples, replace=True)
        draw_idx = np.random.choice(posterior.draw.values, size=num_samples, replace=True)

        # Prepare future time indices
        t_future = np.arange(self.n_obs, self.n_obs + steps)

        forecast_samples = []

        for i in tqdm(range(num_samples), desc="Sampling forecasts"):
            # Get parameter values for this sample
            chain = chain_idx[i]
            draw = draw_idx[i]

            # Extract trend parameters
            if 'trend' in posterior:
                last_trend_level = posterior['trend'].sel(chain=chain, draw=draw).values[-1]
                sigma_trend = posterior['sigma_trend'].sel(chain=chain, draw=draw).values

                # Project trend forward with random walk
                trend_future = last_trend_level + np.cumsum(
                    np.random.normal(0, sigma_trend, size=steps)
                )
            else:
                trend_future = 0

            # Extract weekly seasonality
            if 'beta_cos_weekly' in posterior:
                beta_cos_w = posterior['beta_cos_weekly'].sel(chain=chain, draw=draw).values
                beta_sin_w = posterior['beta_sin_weekly'].sel(chain=chain, draw=draw).values

                n_fourier_weekly = len(beta_cos_w)
                period_weekly = 7.0

                weekly_cos_future = np.column_stack([
                    np.cos(2 * np.pi * (i+1) * t_future / period_weekly)
                    for i in range(n_fourier_weekly)
                ])
                weekly_sin_future = np.column_stack([
                    np.sin(2 * np.pi * (i+1) * t_future / period_weekly)
                    for i in range(n_fourier_weekly)
                ])

                weekly_future = (weekly_cos_future @ beta_cos_w +
                                weekly_sin_future @ beta_sin_w)
            else:
                weekly_future = 0

            # Extract monthly seasonality
            if 'beta_cos_monthly' in posterior:
                beta_cos_m = posterior['beta_cos_monthly'].sel(chain=chain, draw=draw).values
                beta_sin_m = posterior['beta_sin_monthly'].sel(chain=chain, draw=draw).values

                n_fourier_monthly = len(beta_cos_m)
                period_monthly = 30.5

                monthly_cos_future = np.column_stack([
                    np.cos(2 * np.pi * (i+1) * t_future / period_monthly)
                    for i in range(n_fourier_monthly)
                ])
                monthly_sin_future = np.column_stack([
                    np.sin(2 * np.pi * (i+1) * t_future / period_monthly)
                    for i in range(n_fourier_monthly)
                ])

                monthly_future = (monthly_cos_future @ beta_cos_m +
                                 monthly_sin_future @ beta_sin_m)
            else:
                monthly_future = 0

            # Extract yearly seasonality
            if 'beta_cos_yearly' in posterior:
                beta_cos_y = posterior['beta_cos_yearly'].sel(chain=chain, draw=draw).values
                beta_sin_y = posterior['beta_sin_yearly'].sel(chain=chain, draw=draw).values

                n_fourier_yearly = len(beta_cos_y)
                period_yearly = 365.25

                yearly_cos_future = np.column_stack([
                    np.cos(2 * np.pi * (i+1) * t_future / period_yearly)
                    for i in range(n_fourier_yearly)
                ])
                yearly_sin_future = np.column_stack([
                    np.sin(2 * np.pi * (i+1) * t_future / period_yearly)
                    for i in range(n_fourier_yearly)
                ])

                yearly_future = (yearly_cos_future @ beta_cos_y +
                                yearly_sin_future @ beta_sin_y)
            else:
                yearly_future = 0

            # Regression component
            if X_future is not None and 'beta_reg' in posterior:
                beta_reg = posterior['beta_reg'].sel(chain=chain, draw=draw).values
                regression_future = X_future @ beta_reg
            else:
                regression_future = 0

            # Combine components
            mu_scaled_future = (trend_future + weekly_future + monthly_future +
                               yearly_future + regression_future)

            # Transform back to original scale
            mu_future = mu_scaled_future * self.y_std + self.y_mean
            mu_future = np.maximum(mu_future, 0.1)  # Ensure positive

            # Sample from likelihood
            if 'alpha' in posterior:
                # Negative Binomial
                alpha = posterior['alpha'].sel(chain=chain, draw=draw).values
                forecast = np.random.negative_binomial(
                    n=alpha,
                    p=alpha / (alpha + mu_future)
                )
            else:
                # Poisson or Normal fallback
                forecast = np.random.poisson(mu_future)

            forecast_samples.append(forecast)

        forecast_samples = np.array(forecast_samples)  # Shape: (num_samples, steps)

        # Compute summary statistics
        results = {
            'forecast_samples': forecast_samples,
            'forecast_mean': forecast_samples.mean(axis=0),
            'forecast_median': np.median(forecast_samples, axis=0),
            'lower_95': np.percentile(forecast_samples, 2.5, axis=0),
            'upper_95': np.percentile(forecast_samples, 97.5, axis=0),
            'lower_50': np.percentile(forecast_samples, 25, axis=0),
            'upper_50': np.percentile(forecast_samples, 75, axis=0),
        }

        print(f"âœ“ Forecast complete: {steps} steps, {num_samples} samples")

        return results

    def get_components(self) -> Dict[str, np.ndarray]:
        """
        Extract posterior means of model components for interpretation.
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        components = {}
        posterior = self.trace.posterior

        # Extract deterministic components
        for var_name in ['trend', 'weekly_seasonality', 'monthly_seasonality',
                         'yearly_seasonality', 'regression', 'mu']:
            if var_name in posterior:
                components[var_name] = posterior[var_name].mean(dim=['chain', 'draw']).values

        return components


class ImprovedHierarchicalBayesian:
    """
    Improved Hierarchical Bayesian model with proper forecasting.

    Uses partial pooling with Negative Binomial likelihood.
    """

    def __init__(self, name: str = "ImprovedHierarchical"):
        self.name = name
        self.model = None
        self.trace = None
        self.n_groups = None

    def build_model(
        self,
        y: np.ndarray,
        group_idx: np.ndarray,
        n_groups: int,
        t: np.ndarray,  # Time indices
        X: Optional[np.ndarray] = None
    ) -> pm.Model:
        """
        Build hierarchical Bayesian model with time-varying effects.

        Parameters:
        -----------
        y : np.ndarray
            Observed sales
        group_idx : np.ndarray
            Group indices (0 to n_groups-1)
        n_groups : int
            Number of groups in hierarchy
        t : np.ndarray
            Time indices (for trend modeling)
        X : np.ndarray, optional
            Exogenous features
        """
        self.n_groups = n_groups
        n_obs = len(y)

        with pm.Model() as model:
            # ===== HYPERPRIORS (POPULATION LEVEL) =====
            mu_alpha = pm.Normal('mu_alpha', mu=np.log(y.mean() + 1), sigma=1)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.5)

            # ===== GROUP-LEVEL INTERCEPTS (PARTIAL POOLING) =====
            alpha_group = pm.Normal(
                'alpha_group',
                mu=mu_alpha,
                sigma=sigma_alpha,
                shape=n_groups
            )

            # ===== SHARED TREND COMPONENT =====
            sigma_trend = pm.HalfNormal('sigma_trend', sigma=0.05)
            trend_normalized = pt.cumsum(pm.Normal('trend_innovations', mu=0, sigma=sigma_trend, shape=n_obs))

            # ===== WEEKLY SEASONALITY (SHARED) =====
            n_fourier = 3
            period = 7.0

            beta_cos = pm.Normal('beta_cos_weekly', mu=0, sigma=0.3, shape=n_fourier)
            beta_sin = pm.Normal('beta_sin_weekly', mu=0, sigma=0.3, shape=n_fourier)

            weekly_cos = np.column_stack([
                np.cos(2 * np.pi * (i+1) * t / period)
                for i in range(n_fourier)
            ])
            weekly_sin = np.column_stack([
                np.sin(2 * np.pi * (i+1) * t / period)
                for i in range(n_fourier)
            ])

            seasonality = pt.dot(weekly_cos, beta_cos) + pt.dot(weekly_sin, beta_sin)

            # ===== REGRESSION (SHARED COEFFICIENTS) =====
            if X is not None:
                n_features = X.shape[1]
                beta = pm.Normal('beta', mu=0, sigma=0.5, shape=n_features)
                regression_effect = pm.math.dot(X, beta)
            else:
                regression_effect = 0

            # ===== LINEAR PREDICTOR =====
            log_mu = alpha_group[group_idx] + trend_normalized + seasonality + regression_effect
            mu = pm.Deterministic('mu', pm.math.exp(log_mu))

            # ===== LIKELIHOOD (NEGATIVE BINOMIAL) =====
            alpha_nb = pm.Gamma('alpha_nb', alpha=2, beta=1)
            obs = pm.NegativeBinomial(
                'obs',
                mu=mu,
                alpha=alpha_nb,
                observed=y
            )

        self.model = model
        return model

    def fit(self, draws: int = 2000, tune: int = 1000, chains: int = 4, **kwargs):
        """Fit the hierarchical model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"Fitting {self.name} model...")

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.95,
                random_seed=RANDOM_SEED,
                return_inferencedata=True,
                **kwargs
            )

        return self.trace


# Utility function for M5 data preparation
def prepare_m5_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Prepare features from M5-style dataframe for Bayesian modeling.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: date, sales, (optional: event, snap, price)

    Returns:
    --------
    y : np.ndarray
        Sales array
    X : np.ndarray
        Feature matrix
    metadata : Dict
        Additional information
    """
    y = df['sales'].values

    features = []
    feature_names = []

    # Event indicators
    if 'event_name_1' in df.columns:
        event_dummies = pd.get_dummies(df['event_name_1'], prefix='event', dummy_na=False)
        features.append(event_dummies.values)
        feature_names.extend(event_dummies.columns.tolist())

    # SNAP indicators
    if 'snap_CA' in df.columns:
        for state in ['CA', 'TX', 'WI']:
            if f'snap_{state}' in df.columns:
                features.append(df[f'snap_{state}'].values.reshape(-1, 1))
                feature_names.append(f'snap_{state}')

    # Price features
    if 'sell_price' in df.columns:
        price = df['sell_price'].fillna(method='ffill').fillna(method='bfill')
        features.append(price.values.reshape(-1, 1))
        feature_names.append('sell_price')

        # Price change
        price_change = price.diff().fillna(0)
        features.append(price_change.values.reshape(-1, 1))
        feature_names.append('price_change')

    # Weekend indicator
    if 'wday' in df.columns:
        is_weekend = (df['wday'].isin([1, 2])).astype(int).values.reshape(-1, 1)
        features.append(is_weekend)
        feature_names.append('is_weekend')

    # Combine features
    if features:
        X = np.hstack(features)
    else:
        X = None

    metadata = {
        'feature_names': feature_names,
        'n_obs': len(y),
        'y_mean': y.mean(),
        'y_std': y.std(),
        'y_min': y.min(),
        'y_max': y.max(),
        'zero_proportion': (y == 0).mean()
    }

    return y, X, metadata


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, Optional, List, Union
from datetime import datetime

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ProbabilisticDistributionDashboard:
    """
    Advanced dashboard showcasing the full power of Bayesian distributions.

    Key Philosophy:
    ---------------
    Traditional forecasting: y_hat (single number)
    Bayesian forecasting: p(y | data) (entire distribution)

    This enables:
    - Multiple quantiles for different risk tolerances
    - Probability of any scenario
    - Expected value calculations
    - Optimal decision-making under uncertainty
    """

    def __init__(self, figsize=(24, 28)):
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',      # Ocean blue
            'secondary': '#A23B72',    # Deep magenta
            'success': '#06A77D',      # Emerald
            'warning': '#F18F01',      # Amber
            'danger': '#C73E1D',       # Crimson
            'neutral': '#6C757D',      # Gray
            'light': '#E8F4F8',        # Light blue
            'dark': '#1A1A2E',          # Dark navy
            'actual': '#2C3E50',
            'forecast': '#3498DB',
            'good': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'neutral': '#95A5A6',
            'highlight': '#9B59B6',

        }

    def create_distributional_dashboard(
        self,
        y_train: np.ndarray,
        y_test: Optional[np.ndarray],
        dates_train: pd.DatetimeIndex,
        dates_test: pd.DatetimeIndex,
        forecast_samples: np.ndarray,  # Shape: (n_samples, n_steps)
        forecast_results: Dict[str, np.ndarray],
        model_name: str = "Bayesian Forecast",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive distributional analysis dashboard.

        Parameters:
        -----------
        y_train : np.ndarray
            Historical data
        y_test : np.ndarray, optional
            Actual test data for validation
        dates_train : pd.DatetimeIndex
            Training dates
        dates_test : pd.DatetimeIndex
            Forecast dates
        forecast_samples : np.ndarray
            Full posterior predictive samples (n_samples Ã— n_steps)
            THIS IS THE KEY: Complete distributions, not summaries!
        model_name : str
            Model identifier
        save_path : str, optional
            Path to save dashboard
        """
        n_samples, n_steps = forecast_samples.shape

        # Create figure with sophisticated layout
        fig = plt.figure(figsize=self.figsize, facecolor='white')
        gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.5, wspace=0.4,
                              top=0.97, bottom=0.03, left=0.05, right=0.96)

        # ========== HEADER ==========
        ax_header = fig.add_subplot(gs[0, :])
        self._create_distributional_header(ax_header, model_name, n_samples, n_steps)

        # ========== ROW 1: DISTRIBUTIONAL EVOLUTION ==========
        #ax_fan = fig.add_subplot(gs[1, :])
        #self._create_fan_chart(ax_fan, y_train, y_test, dates_train, dates_test, forecast_samples)

        ax_fan = fig.add_subplot(gs[1, :])
        self._create_forecast_chart(
            ax_fan, y_train, y_test, dates_train, dates_test, forecast_results
        )

        # ========== ROW 2: DENSITY EVOLUTION ==========
        ax_density = fig.add_subplot(gs[2, :])
        self._create_density_evolution(ax_density, dates_test, forecast_samples, y_test)

        # ========== ROW 3: QUANTILE SPAGHETTI + PDF ANIMATION ==========
        ax_spaghetti = fig.add_subplot(gs[3, 0:2])
        ax_pdf_days = fig.add_subplot(gs[3, 2])

        self._create_spaghetti_plot(ax_spaghetti, dates_test, forecast_samples, y_test)
        self._create_pdf_day_comparison(ax_pdf_days, forecast_samples, y_test)

        # ========== ROW 4: SCENARIO PROBABILITIES + CALIBRATION ==========
        ax_scenarios = fig.add_subplot(gs[4, 0:2])
        ax_calibration = fig.add_subplot(gs[4, 2])

        self._create_scenario_probabilities(ax_scenarios, dates_test, forecast_samples, y_train)
        self._create_calibration_plot(ax_calibration, forecast_samples, y_test)

        # ========== ROW 5: QUANTILE FORECASTS + SHARPNESS ==========
        ax_quantiles = fig.add_subplot(gs[5, 0:2])
        ax_sharpness = fig.add_subplot(gs[5, 2])

        self._create_quantile_forecast_table(ax_quantiles, dates_test, forecast_samples)
        self._create_sharpness_analysis(ax_sharpness, forecast_samples)

        # ========== ROW 6: DECISION ANALYSIS ==========
        ax_decision = fig.add_subplot(gs[6, 0:2])
        ax_expected = fig.add_subplot(gs[6, 2])

        self._create_decision_analysis(ax_decision, forecast_samples)
        self._create_expected_value_analysis(ax_expected, forecast_samples)

        # Add watermark
        fig.text(0.99, 0.01, 'Bayesian Distributional Forecasting',
                ha='right', va='bottom', fontsize=8, alpha=0.5, style='italic',
                color=self.colors['neutral'])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"âœ“ Distributional dashboard saved: {save_path}")

        return fig

    def _create_distributional_header(self, ax, model_name: str, n_samples: int, n_steps: int):
        """Create header emphasizing distributional nature."""
        ax.axis('off')

        # Main title
        title = "PROBABILISTIC DISTRIBUTION DASHBOARD\nFull Bayesian Uncertainty Quantification"
        ax.text(0.5, 0.7, title,
               ha='center', va='center', fontsize=18, fontweight='bold',
               color=self.colors['dark'])

        # Key advantage box
        advantage_text = (
            f"âœ“ {n_samples:,} Monte Carlo samples  |  "
            f"âœ“ Complete probability distributions  |  "
            f"âœ“ {n_steps}-day forecast horizon  |  "
            f"âœ“ Full uncertainty propagation"
        )

        bbox_props = dict(boxstyle='round,pad=0.6',
                         facecolor=self.colors['light'],
                         edgecolor=self.colors['primary'],
                         linewidth=2)

        ax.text(0.5, 0.35, advantage_text,
               ha='center', va='center', fontsize=11,
               color=self.colors['primary'], fontweight='bold',
               bbox=bbox_props)

        # Subtitle
        subtitle = (
            "Unlike point forecasts, Bayesian methods provide COMPLETE distributions â†’ "
            "Better decisions, quantified risks, multiple scenarios"
        )
        ax.text(0.5, 0.05, subtitle,
               ha='center', va='center', fontsize=10,
               color=self.colors['neutral'], style='italic')

    def _create_fan_chart(
        self,
        ax: plt.Axes,
        y_train: np.ndarray,
        y_test: Optional[np.ndarray],
        dates_train: pd.DatetimeIndex,
        dates_test: pd.DatetimeIndex,
        forecast_samples: np.ndarray
    ):
        """
        Create fan chart showing probability density over time.

        The width of each band represents probability density.
        Darker = higher probability, Lighter = lower probability.
        """
        # Show context
        context_days = min(60, len(y_train))
        ax.plot(dates_train[-context_days:], y_train[-context_days:],
               color=self.colors['dark'], linewidth=2, label='Historical', alpha=0.8)

        # Fan chart: Multiple percentile bands
        percentiles = [
            (1, 99, 0.05),    # 98% PI - very light
            (5, 95, 0.1),     # 90% PI
            (10, 90, 0.15),   # 80% PI
            (20, 80, 0.25),   # 60% PI
            (30, 70, 0.35),   # 40% PI
            (40, 60, 0.5),    # 20% PI - darkest
        ]

        for lower_p, upper_p, alpha in percentiles:
            lower = np.percentile(forecast_samples, lower_p, axis=0)
            upper = np.percentile(forecast_samples, upper_p, axis=0)

            ax.fill_between(dates_test, lower, upper,
                           alpha=alpha, color=self.colors['primary'],
                           linewidth=0)

        # Median forecast
        median = np.median(forecast_samples, axis=0)
        ax.plot(dates_test, median,
               color=self.colors['dark'], linewidth=3,
               label='Median Forecast', zorder=10)

        # Actual test data
        if y_test is not None and len(y_test) > 0:
            ax.plot(dates_test[:len(y_test)], y_test,
                   'o-', color=self.colors['danger'], linewidth=2,
                   markersize=5, label='Actual', zorder=11, alpha=0.9)

        # Forecast boundary
        ax.axvline(dates_test.iloc[0], color=self.colors['neutral'],
                  linestyle='--', linewidth=2, alpha=0.6)

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales', fontsize=12, fontweight='bold')
        ax.set_title('Fan Chart: Probability Density Evolution (Darker = Higher Probability)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    def _create_forecast_chart(
        self,
        ax: plt.Axes,
        y_train: np.ndarray,
        y_test: Optional[np.ndarray],
        dates_train: pd.DatetimeIndex,
        dates_test: pd.DatetimeIndex,
        forecast_results: Dict
    ):
        """Create main forecast chart with uncertainty bands."""
        forecast_mean = forecast_results['forecast_mean']
        lower_95 = forecast_results['lower_95']
        upper_95 = forecast_results['upper_95']
        lower_50 = forecast_results['lower_50']
        upper_50 = forecast_results['upper_50']

        # Show last N days of history for context
        context_days = min(90, len(y_train))

        # Plot historical data
        ax.plot(dates_train[-context_days:], y_train[-context_days:],
               color=self.colors['actual'], linewidth=2, label='Historical Sales', alpha=0.8)

        # Plot forecast mean
        ax.plot(dates_test, forecast_mean,
               color=self.colors['forecast'], linewidth=2.5, label='Forecast (Mean)', zorder=5)

        # Plot uncertainty bands
        ax.fill_between(dates_test, lower_50, upper_50,
                       alpha=0.4, color=self.colors['forecast'], label='50% Confidence', zorder=3)
        ax.fill_between(dates_test, lower_95, upper_95,
                       alpha=0.2, color=self.colors['forecast'], label='95% Confidence', zorder=2)

        # Plot actual test data if available
        if y_test is not None and len(y_test) > 0:
            ax.plot(dates_test[:len(y_test)], y_test,
                   'o-', color=self.colors['danger'], linewidth=2,
                   markersize=6, label='Actual Sales', zorder=6, alpha=0.9)

        # Vertical line at forecast start
        ax.axvline(dates_test.iloc[0], color=self.colors['neutral'],
                  linestyle='--', linewidth=2, alpha=0.6, label='Forecast Start')

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales Units', fontsize=12, fontweight='bold')
        ax.set_title('Probabilistic Sales Forecast with Uncertainty Bands',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    def _create_density_evolution(
        self,
        ax: plt.Axes,
        dates_test: pd.DatetimeIndex,
        forecast_samples: np.ndarray,
        y_test: Optional[np.ndarray]
    ):
        """
        Create violin/density plot showing distribution shape evolution.

        Shows HOW the distribution changes over forecast horizon.
        """
        n_samples, n_steps = forecast_samples.shape

        # Select representative days
        if n_steps <= 10:
            days_to_show = range(n_steps)
        elif n_steps <= 30:
            days_to_show = range(0, n_steps, 3)
        else:
            days_to_show = range(0, n_steps, 7)

        positions = list(days_to_show)
        data_to_plot = [forecast_samples[:, i] for i in days_to_show]

        # Create violin plots
        parts = ax.violinplot(data_to_plot, positions=positions,
                             widths=1.5, showmeans=True, showmedians=True)

        # Color violins
        for pc in parts['bodies']:
            pc.set_facecolor(self.colors['primary'])
            pc.set_alpha(0.6)
            pc.set_edgecolor(self.colors['dark'])
            pc.set_linewidth(1.5)

        # Enhance mean/median lines
        parts['cmeans'].set_color(self.colors['danger'])
        parts['cmeans'].set_linewidth(2.5)
        parts['cmedians'].set_color(self.colors['success'])
        parts['cmedians'].set_linewidth(2)

        # Plot actual values if available
        if y_test is not None and len(y_test) > 0:
            actual_to_plot = [y_test[i] if i < len(y_test) else None for i in days_to_show]
            for i, (pos, actual) in enumerate(zip(positions, actual_to_plot)):
                if actual is not None:
                    ax.plot(pos, actual, 'D', color=self.colors['warning'],
                           markersize=10, markeredgecolor='black',
                           markeredgewidth=1.5, zorder=10)

        # X-axis labels
        labels = [dates_test.iloc[i].strftime('%m/%d') for i in days_to_show]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.set_xlabel('Forecast Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales Distribution', fontsize=12, fontweight='bold')
        ax.set_title('Distribution Shape Evolution: How Uncertainty Changes Over Time',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['primary'], alpha=0.6, label='Distribution'),
            plt.Line2D([0], [0], color=self.colors['danger'], linewidth=2.5, label='Mean'),
            plt.Line2D([0], [0], color=self.colors['success'], linewidth=2, label='Median'),
        ]
        if y_test is not None and len(y_test) > 0:
            legend_elements.append(
                plt.Line2D([0], [0], marker='D', color='w',
                          markerfacecolor=self.colors['warning'],
                          markersize=8, label='Actual', markeredgecolor='black')
            )
        ax.legend(handles=legend_elements, loc='best', fontsize=10)

    def _create_spaghetti_plot(
        self,
        ax: plt.Axes,
        dates_test: pd.DatetimeIndex,
        forecast_samples: np.ndarray,
        y_test: Optional[np.ndarray]
    ):
        """
        Create spaghetti plot: many individual sample paths.

        Shows the FULL RANGE of possible futures, not just aggregates.
        """
        n_samples, n_steps = forecast_samples.shape

        # Plot random sample of trajectories
        n_trajectories = min(100, n_samples)
        indices = np.random.choice(n_samples, n_trajectories, replace=False)

        for idx in indices:
            ax.plot(dates_test, forecast_samples[idx, :],
                   color=self.colors['primary'], alpha=0.05, linewidth=0.8)

        # Median trajectory (thicker)
        median = np.median(forecast_samples, axis=0)
        ax.plot(dates_test, median,
               color=self.colors['dark'], linewidth=3,
               label='Median', zorder=10)

        # Actual
        if y_test is not None and len(y_test) > 0:
            ax.plot(dates_test[:len(y_test)], y_test,
                   'o-', color=self.colors['danger'], linewidth=2.5,
                   markersize=6, label='Actual', zorder=11)

        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Sales', fontsize=11, fontweight='bold')
        ax.set_title(f'Spaghetti Plot: {n_trajectories} Possible Future Paths (Bayesian Monte Carlo)',
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    def _create_pdf_day_comparison(
        self,
        ax: plt.Axes,
        forecast_samples: np.ndarray,
        y_test: Optional[np.ndarray]
    ):
        """
        Compare probability density functions for multiple days.

        Shows how distribution shape changes: wider = more uncertain.
        """
        n_samples, n_steps = forecast_samples.shape

        # Select 4-5 representative days
        if n_steps <= 7:
            days = [0, n_steps-1]
        elif n_steps <= 14:
            days = [0, n_steps//2, n_steps-1]
        else:
            days = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]

        colors_list = [self.colors['success'], self.colors['primary'],
                      self.colors['secondary'], self.colors['warning'],
                      self.colors['danger']]

        for i, day in enumerate(days):
            samples = forecast_samples[:, day]

            # Kernel density estimation
            kde = stats.gaussian_kde(samples)
            x_range = np.linspace(samples.min(), samples.max(), 200)
            density = kde(x_range)

            color = colors_list[i % len(colors_list)]
            ax.plot(x_range, density, linewidth=2.5, label=f'Day {day+1}', color=color)
            ax.fill_between(x_range, density, alpha=0.3, color=color)

            # Mark actual if available
            if y_test is not None and day < len(y_test):
                ax.axvline(y_test[day], color=color, linestyle='--',
                          linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Sales', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax.set_title('PDF Evolution:\nUncertainty Growth',
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    def _create_scenario_probabilities(
        self,
        ax: plt.Axes,
        dates_test: pd.DatetimeIndex,
        forecast_samples: np.ndarray,
        y_train: np.ndarray
    ):
        """
        Calculate and visualize probability of business-relevant scenarios.

        Key Bayesian advantage: Can answer "What's the probability that...?"
        """
        # Define business scenarios based on historical context
        historical_mean = y_train.mean()
        historical_std = y_train.std()

        scenarios = {
            'Very Low (<80% avg)': lambda x: x < 0.8 * historical_mean,
            'Low (80-90% avg)': lambda x: (x >= 0.8 * historical_mean) & (x < 0.9 * historical_mean),
            'Normal (90-110% avg)': lambda x: (x >= 0.9 * historical_mean) & (x <= 1.1 * historical_mean),
            'High (110-120% avg)': lambda x: (x > 1.1 * historical_mean) & (x <= 1.2 * historical_mean),
            'Very High (>120% avg)': lambda x: x > 1.2 * historical_mean
        }

        colors_map = {
            'Very Low (<80% avg)': self.colors['danger'],
            'Low (80-90% avg)': self.colors['warning'],
            'Normal (90-110% avg)': self.colors['success'],
            'High (110-120% avg)': self.colors['primary'],
            'Very High (>120% avg)': self.colors['secondary']
        }

        # Calculate probabilities for each day
        for scenario_name, condition in scenarios.items():
            probabilities = []
            for day in range(forecast_samples.shape[1]):
                prob = condition(forecast_samples[:, day]).mean() * 100
                probabilities.append(prob)

            ax.plot(dates_test, probabilities,
                   linewidth=2.5, label=scenario_name,
                   color=colors_map[scenario_name])

        ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability (%)', fontsize=11, fontweight='bold')
        ax.set_title('Scenario Probabilities: What Are the Chances of Each Outcome?',
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

    def _create_calibration_plot(
        self,
        ax: plt.Axes,
        forecast_samples: np.ndarray,
        y_test: Optional[np.ndarray]
    ):
        """
        Calibration plot: Are our probabilities well-calibrated?

        Perfect calibration: 90% prediction interval contains 90% of actuals.
        """
        if y_test is None or len(y_test) == 0:
            ax.text(0.5, 0.5, 'No test data available\nfor calibration analysis',
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, color=self.colors['neutral'])
            ax.axis('off')
            return

        # Calculate empirical coverage for different confidence levels
        confidence_levels = np.arange(10, 100, 5)
        empirical_coverage = []

        for conf in confidence_levels:
            lower_p = (100 - conf) / 2
            upper_p = 100 - lower_p

            lower = np.percentile(forecast_samples[:, :len(y_test)], lower_p, axis=0)
            upper = np.percentile(forecast_samples[:, :len(y_test)], upper_p, axis=0)

            coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100
            empirical_coverage.append(coverage)

        # Plot calibration
        ax.plot(confidence_levels, empirical_coverage,
               'o-', linewidth=2.5, markersize=8,
               color=self.colors['primary'], label='Observed')

        # Perfect calibration line
        ax.plot([0, 100], [0, 100],
               '--', linewidth=2, color=self.colors['danger'],
               label='Perfect Calibration', alpha=0.7)

        # Shaded acceptable region (Â±5%)
        ax.fill_between(confidence_levels,
                       confidence_levels - 5,
                       confidence_levels + 5,
                       alpha=0.2, color=self.colors['success'],
                       label='Â±5% Tolerance')

        ax.set_xlabel('Nominal Coverage (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Empirical Coverage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Calibration Plot:\nAre Probabilities Accurate?',
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])

        # Add assessment
        deviation = np.mean(np.abs(np.array(empirical_coverage) - confidence_levels))
        if deviation < 5:
            assessment = "âœ“ Well Calibrated"
            color = self.colors['success']
        elif deviation < 10:
            assessment = "â—‹ Acceptable"
            color = self.colors['warning']
        else:
            assessment = "â–³ Needs Improvement"
            color = self.colors['danger']

        ax.text(0.05, 0.95, assessment,
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               color=color, verticalalignment='top')

    def _create_quantile_forecast_table(
        self,
        ax: plt.Axes,
        dates_test: pd.DatetimeIndex,
        forecast_samples: np.ndarray
    ):
        """
        Create table showing multiple quantile forecasts.

        Bayesian advantage: Can produce ANY quantile, not just mean.
        Different quantiles for different decisions.
        """
        ax.axis('off')

        # Select days to display
        n_steps = forecast_samples.shape[1]
        if n_steps <= 7:
            days_to_show = range(n_steps)
        else:
            # Show first 7 days
            days_to_show = range(min(7, n_steps))

        # Quantiles of interest
        quantiles = [10, 25, 50, 75, 90]

        # Table header
        ax.text(0.5, 0.98, 'Quantile Forecasts: Multiple Scenarios for Different Risk Levels',
               ha='center', va='top', fontsize=13, fontweight='bold',
               transform=ax.transAxes, color=self.colors['dark'])

        # Column headers
        col_x = [0.12, 0.28, 0.44, 0.60, 0.76, 0.92]
        headers = ['Date', 'P10\n(Pessimistic)', 'P25\n(Conservative)',
                  'P50\n(Median)', 'P75\n(Optimistic)', 'P90\n(Very Optimistic)']

        for x, header in zip(col_x, headers):
            ax.text(x, 0.88, header,
                   ha='center', va='center', fontsize=9,
                   fontweight='bold', transform=ax.transAxes,
                   color=self.colors['dark'])

        # Horizontal line
        ax.plot([0.02, 0.98], [0.84, 0.84], 'k-', linewidth=1.5,
               transform=ax.transAxes)

        # Table rows
        row_height = 0.10
        for i, day in enumerate(days_to_show):
            y_pos = 0.78 - i * row_height

            # Date
            date_str = dates_test.iloc[day].strftime('%m/%d')
            ax.text(col_x[0], y_pos, date_str,
                   ha='center', va='center', fontsize=9,
                   transform=ax.transAxes, fontweight='bold')

            # Quantile values
            quantile_values = np.percentile(forecast_samples[:, day], quantiles)

            for j, (q_val, col) in enumerate(zip(quantile_values, col_x[1:])):
                # Color code
                if j <= 1:  # P10, P25
                    color = self.colors['danger']
                elif j == 2:  # P50
                    color = self.colors['dark']
                else:  # P75, P90
                    color = self.colors['success']

                ax.text(col, y_pos, f'{q_val:,.0f}',
                       ha='center', va='center', fontsize=9,
                       transform=ax.transAxes, color=color,
                       fontweight='bold' if j == 2 else 'normal')

        # Interpretation guide
        ax.text(0.5, 0.08,
               'Interpretation: P10 = 10% chance actual will be BELOW this | P90 = 10% chance actual will be ABOVE this',
               ha='center', va='center', fontsize=8,
               transform=ax.transAxes, color=self.colors['neutral'],
               style='italic')

    def _create_sharpness_analysis(
        self,
        ax: plt.Axes,
        forecast_samples: np.ndarray
    ):
        """
        Analyze forecast sharpness (precision) over time.

        Sharpness = How narrow are the predictions?
        Calibration = Are the predictions accurate?

        Want: Sharp AND calibrated (both precision and accuracy).
        """
        # Calculate interval widths over time
        n_steps = forecast_samples.shape[1]

        intervals = {
            '50% Interval': (25, 75),
            '80% Interval': (10, 90),
            '95% Interval': (2.5, 97.5)
        }

        colors_list = [self.colors['success'], self.colors['primary'], self.colors['warning']]

        for (name, (lower_p, upper_p)), color in zip(intervals.items(), colors_list):
            widths = []
            for day in range(n_steps):
                lower = np.percentile(forecast_samples[:, day], lower_p)
                upper = np.percentile(forecast_samples[:, day], upper_p)
                width = upper - lower
                widths.append(width)

            ax.plot(range(1, n_steps+1), widths,
                   linewidth=2.5, label=name, color=color, marker='o', markersize=4)

        ax.set_xlabel('Forecast Horizon (Days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Interval Width', fontsize=11, fontweight='bold')
        ax.set_title('Sharpness Analysis:\nUncertainty Growth',
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        # Trend annotation
        widths_95 = []
        for day in range(n_steps):
            lower = np.percentile(forecast_samples[:, day], 2.5)
            upper = np.percentile(forecast_samples[:, day], 97.5)
            widths_95.append(upper - lower)

        growth_rate = (widths_95[-1] - widths_95[0]) / widths_95[0] * 100

        ax.text(0.05, 0.95,
               f'95% PI Growth: {growth_rate:+.1f}%',
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               color=self.colors['primary'], verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white', alpha=0.8))

    def _create_decision_analysis(
        self,
        ax: plt.Axes,
        forecast_samples: np.ndarray
    ):
        """
        Decision analysis under uncertainty using distributions.

        Shows optimal decisions for different cost structures.
        This is ONLY possible with full distributions!
        """
        total_demand_samples = forecast_samples.sum(axis=1)

        # Decision scenario: Inventory level
        # Cost structure:
        # - Holding cost: $1 per unit
        # - Stockout cost: $10 per unit

        inventory_levels = np.linspace(
            total_demand_samples.min(),
            total_demand_samples.max(),
            100
        )

        holding_cost = 1.0
        stockout_cost = 10.0

        expected_costs = []

        for inv_level in inventory_levels:
            # For each sample, calculate cost
            costs = []
            for demand in total_demand_samples:
                if demand <= inv_level:
                    # No stockout, only holding cost
                    cost = holding_cost * (inv_level - demand)
                else:
                    # Stockout
                    cost = stockout_cost * (demand - inv_level)
                costs.append(cost)

            expected_costs.append(np.mean(costs))

        # Plot expected cost curve
        ax.plot(inventory_levels, expected_costs,
               linewidth=3, color=self.colors['primary'])

        # Mark optimal inventory level
        optimal_idx = np.argmin(expected_costs)
        optimal_inventory = inventory_levels[optimal_idx]
        optimal_cost = expected_costs[optimal_idx]

        ax.plot(optimal_inventory, optimal_cost,
               'o', markersize=15, color=self.colors['danger'],
               markeredgecolor='black', markeredgewidth=2, zorder=10)

        ax.axvline(optimal_inventory, color=self.colors['danger'],
                  linestyle='--', linewidth=2, alpha=0.7)

        # Annotations
        ax.annotate(f'Optimal: {optimal_inventory:,.0f} units\nCost: ${optimal_cost:,.0f}',
                   xy=(optimal_inventory, optimal_cost),
                   xytext=(20, 30), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.8',
                            facecolor=self.colors['danger'], alpha=0.8),
                   color='white',
                   arrowprops=dict(arrowstyle='->', lw=2,
                                  color=self.colors['danger']))

        ax.set_xlabel('Inventory Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Expected Cost ($)', fontsize=11, fontweight='bold')
        ax.set_title(f'Decision Analysis: Optimal Inventory Under Uncertainty\n'
                    f'(Holding: ${holding_cost}/unit, Stockout: ${stockout_cost}/unit)',
                    fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x):,}'))

    def _create_expected_value_analysis(
        self,
        ax: plt.Axes,
        forecast_samples: np.ndarray
    ):
        """
        Expected value analysis: Value of distributional information.

        Compares:
        - Using full distribution (Bayesian)
        - Using only mean (traditional)
        - Perfect information (oracle)
        """
        ax.axis('off')

        total_demand_samples = forecast_samples.sum(axis=1)
        mean_demand = total_demand_samples.mean()

        # Cost parameters
        holding_cost = 1.0
        stockout_cost = 10.0

        # Strategy 1: Use mean (traditional)
        costs_mean_strategy = []
        for demand in total_demand_samples:
            if demand <= mean_demand:
                cost = holding_cost * (mean_demand - demand)
            else:
                cost = stockout_cost * (demand - mean_demand)
            costs_mean_strategy.append(cost)

        expected_cost_mean = np.mean(costs_mean_strategy)

        # Strategy 2: Use full distribution (optimal)
        inventory_levels = np.linspace(total_demand_samples.min(),
                                      total_demand_samples.max(), 100)
        all_costs = []
        for inv_level in inventory_levels:
            costs = []
            for demand in total_demand_samples:
                if demand <= inv_level:
                    cost = holding_cost * (inv_level - demand)
                else:
                    cost = stockout_cost * (demand - inv_level)
                costs.append(cost)
            all_costs.append(np.mean(costs))

        expected_cost_optimal = min(all_costs)

        # Strategy 3: Perfect information (oracle - knows future)
        costs_perfect = []
        for demand in total_demand_samples:
            # Perfect foresight: stock exactly what's needed
            cost = 0  # No holding, no stockout
            costs_perfect.append(cost)

        expected_cost_perfect = np.mean(costs_perfect)

        # Calculate value
        value_of_distribution = expected_cost_mean - expected_cost_optimal
        value_of_perfect_info = expected_cost_optimal - expected_cost_perfect

        # Display comparison
        strategies = [
            ('Traditional\n(Mean Only)', expected_cost_mean, self.colors['warning']),
            ('Bayesian\n(Full Distribution)', expected_cost_optimal, self.colors['success']),
            ('Perfect\n(Oracle)', expected_cost_perfect, self.colors['neutral'])
        ]

        # Title
        ax.text(0.5, 0.95, 'Expected Value Analysis:\nValue of Distributional Information',
               ha='center', va='top', fontsize=13, fontweight='bold',
               transform=ax.transAxes, color=self.colors['dark'])

        # Bars
        y_positions = [0.75, 0.55, 0.35]
        bar_height = 0.12
        max_cost = expected_cost_mean

        for (name, cost, color), y_pos in zip(strategies, y_positions):
            # Bar
            bar_width = cost / max_cost * 0.7
            rect = Rectangle((0.15, y_pos - bar_height/2), bar_width, bar_height,
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=1.5,
                           transform=ax.transAxes)
            ax.add_patch(rect)

            # Label
            ax.text(0.12, y_pos, name,
                   ha='right', va='center', fontsize=10, fontweight='bold',
                   transform=ax.transAxes)

            # Value
            ax.text(0.15 + bar_width + 0.02, y_pos, f'${cost:,.0f}',
                   ha='left', va='center', fontsize=10, fontweight='bold',
                   transform=ax.transAxes, color=color)

        # Value callouts
        ax.text(0.5, 0.18,
               f'ğŸ’¡ Value of Distribution: ${value_of_distribution:,.0f} saved\n'
               f'({value_of_distribution/expected_cost_mean*100:.1f}% cost reduction)',
               ha='center', va='center', fontsize=10, fontweight='bold',
               transform=ax.transAxes, color=self.colors['success'],
               bbox=dict(boxstyle='round,pad=0.8',
                        facecolor=self.colors['light'],
                        edgecolor=self.colors['success'], linewidth=2))


def create_distributional_summary_stats(
    forecast_samples: np.ndarray,
    dates_test: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Create comprehensive summary statistics from distribution samples.

    Parameters:
    -----------
    forecast_samples : np.ndarray
        Posterior predictive samples (n_samples Ã— n_steps)
    dates_test : pd.DatetimeIndex
        Forecast dates

    Returns:
    --------
    summary_df : pd.DataFrame
        Detailed distributional statistics
    """
    n_samples, n_steps = forecast_samples.shape

    summary_data = []

    for day in range(n_steps):
        samples = forecast_samples[:, day]

        # Comprehensive statistics
        stats_dict = {
            'Date': dates_test.iloc[day],
            'Mean': samples.mean(),
            'Median': np.median(samples),
            'Mode': stats.mode(samples.astype(int), keepdims=True)[0][0],
            'StdDev': samples.std(),
            'Variance': samples.var(),
            'Skewness': stats.skew(samples),
            'Kurtosis': stats.kurtosis(samples),
            'Min': samples.min(),
            'Max': samples.max(),
            'Range': samples.max() - samples.min(),
            'IQR': np.percentile(samples, 75) - np.percentile(samples, 25),
            'P01': np.percentile(samples, 1),
            'P05': np.percentile(samples, 5),
            'P10': np.percentile(samples, 10),
            'P25': np.percentile(samples, 25),
            'P50': np.percentile(samples, 50),
            'P75': np.percentile(samples, 75),
            'P90': np.percentile(samples, 90),
            'P95': np.percentile(samples, 95),
            'P99': np.percentile(samples, 99),
            'CV_Percent': (samples.std() / samples.mean() * 100) if samples.mean() > 0 else 0
        }

        summary_data.append(stats_dict)

    return pd.DataFrame(summary_data)


preprocessor = BayesianDataPreprocessor()
data_dict = preprocessor.load_data()

# Prepare time series (total sales for demonstration)
item_data, metadata = preprocessor.prepare_time_series(
    aggregate_level='total',
    start_date='2013-01-01',
    end_date='2016-04-24'
)

print(f"   Data shape: {item_data.shape}")
print(f"   Date range: {metadata['date_range']}")


# Sort by date
item_data = item_data.sort_values('date').reset_index(drop=True)
y = item_data['sales'].values
dates = pd.to_datetime(item_data['date'])

# Train/test split (last 28 days for testing)
n_forecast = 28
train_size = len(y) - n_forecast

y_train = y[:train_size]
y_test = y[train_size:]
dates_train = dates[:train_size]
dates_test = dates[train_size:]

print(f"   Training samples: {len(y_train)}")
print(f"   Test samples: {len(y_test)}")
print(f"   Sales range: [{y_train.min():.0f}, {y_train.max():.0f}]")
print(f"   Sales mean: {y_train.mean():.0f}")


model_improved = ImprovedBayesianTimeSeries(name="ImprovedBSTS")

model = model_improved.build_model(
    y=y_train,
    use_trend=True,
    use_weekly_seasonality=True,
    use_monthly_seasonality=True,
    use_yearly_seasonality=False,  # Not enough data for yearly
    likelihood='negative_binomial',  # CRITICAL: Best for retail count data
    standardize=True  # Better numerical stability
)


trace = model_improved.fit(
        draws=1000,  # Use 2000+ for production
        tune=1000,
        chains=2,  # At least 4 chains recommended
        target_accept=0.95  # High acceptance for better convergence
    )


forecast_results = model_improved.forecast(
    steps=n_forecast,
    X_future=None,  # No external regressors in this example
    num_samples=1000
)

forecast_mean = forecast_results['forecast_mean']
forecast_median = forecast_results['forecast_median']
lower_95 = forecast_results['lower_95']
upper_95 = forecast_results['upper_95']
lower_50 = forecast_results['lower_50']
upper_50 = forecast_results['upper_50']


# Compute error metrics
mae = np.mean(np.abs(forecast_mean - y_test))
rmse = np.sqrt(np.mean((forecast_mean - y_test)**2))
mape = np.mean(np.abs((forecast_mean - y_test) / y_test)) * 100

# Coverage of prediction intervals
coverage_95 = np.mean((y_test >= lower_95) & (y_test <= upper_95)) * 100
coverage_50 = np.mean((y_test >= lower_50) & (y_test <= upper_50)) * 100

print(f"\n   Forecast Accuracy Metrics:")
print(f"   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"   MAE:        {mae:.2f}")
print(f"   RMSE:       {rmse:.2f}")
print(f"   MAPE:       {mape:.2f}%")
print(f"\n   Uncertainty Calibration:")
print(f"   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"   95% CI Coverage: {coverage_95:.1f}% (target: 95%)")
print(f"   50% CI Coverage: {coverage_50:.1f}% (target: 50%)")


components = model_improved.get_components()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))

    # In-sample fitted values
mu_fitted = components['mu']

ax1.plot(dates_train, y_train, 'k-', label='Actual', alpha=0.6, linewidth=1)
ax1.plot(dates_train, mu_fitted, 'r-', label='Fitted Mean', linewidth=2)
ax1.set_ylabel('Sales', fontsize=12)
ax1.set_title('Improved BSTS Model - In-Sample Fit', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
    
# Component decomposition
if 'trend' in components:
    trend_rescaled = components['trend'] * model_improved.y_std + model_improved.y_mean
    ax2.plot(dates_train, trend_rescaled, 'g-', linewidth=2, label='Trend')
    ax2.set_ylabel('Trend', fontsize=12)
    ax2.set_title('Trend Component', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

if 'weekly_seasonality' in components:
    weekly_rescaled = components['weekly_seasonality'] * model_improved.y_std
    ax3.plot(dates_train, weekly_rescaled, 'orange', linewidth=1.5, label='Weekly Seasonality')
    ax3.set_ylabel('Weekly Effect', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Weekly Seasonality', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
if 'monthly_seasonality' in components:
    weekly_rescaled = components['monthly_seasonality'] * model_improved.y_std
    ax4.plot(dates_train, weekly_rescaled, 'blue', linewidth=1.5, label='Monthly Seasonality')
    ax4.set_ylabel('Monthly Effect', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_title('Monthly Seasonality', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()


fig, ax = plt.subplots(figsize=(15, 6))

# Historical data (last 60 days for context)
context_days = 60
ax.plot(dates_train[-context_days:], y_train[-context_days:],
        'k-', label='Historical', linewidth=2, alpha=0.7)

# Actual test data
ax.plot(dates_test, y_test, 'ro-', label='Actual (Test)',
        linewidth=2, markersize=6, alpha=0.8)

# Forecast mean
ax.plot(dates_test, forecast_mean, 'b-', label='Forecast (Mean)',
        linewidth=2.5)

# Prediction intervals
ax.fill_between(dates_test, lower_50, upper_50,
                alpha=0.4, color='blue', label='50% Prediction Interval')
ax.fill_between(dates_test, lower_95, upper_95,
                alpha=0.2, color='blue', label='95% Prediction Interval')

# Mark forecast start
ax.axvline(dates_test.iloc[0], color='gray', linestyle='--',
           alpha=0.7, linewidth=2, label='Forecast Start')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)
ax.set_title(f'28-Day Probabilistic Forecast (MAE={mae:.2f}, RMSE={rmse:.2f})',
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

dates_to_plot = [0, 6, 13, 27]  # Days 1, 7, 14, 28
for idx, day in enumerate(dates_to_plot):
    ax = axes[idx]
    samples = forecast_results['forecast_samples'][:, day]

    # Histogram
    ax.hist(samples, bins=50, density=True, alpha=0.7,
            color='skyblue', edgecolor='black')

    # Add actual value
    ax.axvline(y_test[day], color='red', linestyle='--',
              linewidth=2, label=f'Actual: {y_test[day]:.0f}')

    # Add forecast statistics
    ax.axvline(forecast_mean[day], color='blue', linestyle='-',
              linewidth=2, label=f'Mean: {forecast_mean[day]:.0f}')
    ax.axvline(forecast_median[day], color='green', linestyle='-',
              linewidth=2, label=f'Median: {forecast_median[day]:.0f}')

    ax.set_xlabel('Sales', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.set_title(f'Day {day+1} Forecast - {dates_test.iloc[day].strftime("%Y-%m-%d")}',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()


dashboard = ProbabilisticDistributionDashboard(figsize=(20, 24))
fig = dashboard.create_distributional_dashboard(
    y_train=y_train,
    y_test=y_test,
    dates_train=dates_train,
    dates_test=dates_test,
    forecast_samples=forecast_results['forecast_samples'],
    forecast_results=forecast_results,
)

