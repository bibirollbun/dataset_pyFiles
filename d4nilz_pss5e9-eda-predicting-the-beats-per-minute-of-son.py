from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance, entropy
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from colorama import Fore, Style, init
from IPython.display import display, HTML
import warnings
import time
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Dict, List, Tuple
import contextlib
import os

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# Initialize colorama
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



class StyleManager:
    """Unified styling system with light/dark mode support"""

    def __init__(self, mode: str = 'auto'):
        """
        Initialize style manager
        
        Args:
            mode: 'light', 'dark', or 'auto' (detects from environment)
        """
        self.mode = self._detect_mode(mode)
        self._setup_colors()
        self._setup_matplotlib_style()

    def _detect_mode(self, mode: str) -> str:
        """Detect or set the color mode"""
        if mode == 'auto':
            # Try to detect from environment or Jupyter theme
            try:
                # Check if running in Jupyter and try to detect theme
                from IPython import get_ipython
                if get_ipython() is not None:
                    # Default to dark for Jupyter (most common for data science)
                    return 'dark'
                else:
                    # Terminal - check environment or default to light
                    return os.environ.get('EDA_THEME', 'light')
            except:
                return 'light'
        return mode

    def _setup_colors(self):
        """Setup color palettes based on mode"""
        if self.mode == 'dark':
            self.colors = {
                # Primary colors
                'primary': '#00D4AA',      # Teal/cyan - main accent
                'secondary': '#FF6B9D',    # Pink - secondary accent
                'success': '#00E676',      # Green - success/positive
                'warning': '#FFB74D',      # Orange - warnings
                'danger': '#FF5252',       # Red - errors/critical
                'info': '#64B5F6',         # Blue - information

                # Chart colors
                'train': '#00D4AA',        # Teal for training data
                'test': '#FF6B9D',         # Pink for test data
                'target': '#FFB74D',       # Orange for target
                'neutral': '#78909C',      # Gray for neutral elements

                # Background and text
                'background': '#1E1E1E',   # Dark background
                'surface': '#2D2D2D',      # Slightly lighter surface
                'text_primary': '#FFFFFF', # White text
                'text_secondary': '#B0BEC5', # Light gray text
                'grid': '#404040',         # Grid lines

                # Gradient colors for heatmaps
                'gradient_positive': '#00E676',
                'gradient_negative': '#FF5252',
                'gradient_neutral': '#78909C',
            }

            # Color palettes for multiple categories
            self.palette_categorical = ['#00D4AA', '#FF6B9D', '#FFB74D', '#64B5F6',
                                        '#AB47BC', '#26A69A', '#FFA726', '#EF5350']
            self.palette_sequential = ['#1A237E', '#3949AB', '#5C6BC0', '#7986CB', '#9FA8DA']
            self.palette_diverging = ['#FF5252', '#FF8A80', '#FFCDD2', '#E8F5E8', '#A5D6A7', '#4CAF50', '#2E7D32']

        else:  # light mode
            self.colors = {
                # Primary colors
                'primary': '#1976D2',      # Blue - main accent
                'secondary': '#7B1FA2',    # Purple - secondary accent
                'success': '#388E3C',      # Green - success/positive
                'warning': '#F57C00',      # Orange - warnings
                'danger': '#D32F2F',       # Red - errors/critical
                'info': '#1976D2',         # Blue - information

                # Chart colors
                'train': '#1976D2',        # Blue for training data
                'test': '#7B1FA2',         # Purple for test data
                'target': '#F57C00',       # Orange for target
                'neutral': '#757575',      # Gray for neutral elements

                # Background and text
                'background': '#FFFFFF',   # White background
                'surface': '#F5F5F5',      # Light gray surface
                'text_primary': '#212121', # Dark text
                'text_secondary': '#757575', # Gray text
                'grid': '#E0E0E0',         # Light grid lines

                # Gradient colors for heatmaps
                'gradient_positive': '#4CAF50',
                'gradient_negative': '#F44336',
                'gradient_neutral': '#9E9E9E',
            }

            # Color palettes for multiple categories
            self.palette_categorical = ['#1976D2', '#7B1FA2', '#F57C00', '#388E3C',
                                        '#D32F2F', '#0097A7', '#5D4037', '#455A64']
            self.palette_sequential = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1976D2']
            self.palette_diverging = ['#D32F2F', '#F44336', '#FFCDD2', '#E8F5E8', '#A5D6A7', '#4CAF50', '#2E7D32']

    def _setup_matplotlib_style(self):
        """Configure matplotlib and seaborn styling"""
        # Set the style based on mode
        if self.mode == 'dark':
            plt.style.use('dark_background')
            sns.set_style("darkgrid", {
                'axes.facecolor': self.colors['surface'],
                'figure.facecolor': self.colors['background'],
                'grid.color': self.colors['grid'],
                'text.color': self.colors['text_primary'],
                'axes.edgecolor': self.colors['text_secondary'],
                'axes.labelcolor': self.colors['text_primary'],
                'xtick.color': self.colors['text_primary'],
                'ytick.color': self.colors['text_primary']
            })
        else:
            plt.style.use('default')
            sns.set_style("whitegrid", {
                'axes.facecolor': self.colors['background'],
                'figure.facecolor': self.colors['background'],
                'grid.color': self.colors['grid'],
                'text.color': self.colors['text_primary'],
                'axes.edgecolor': self.colors['text_secondary'],
                'axes.labelcolor': self.colors['text_primary'],
                'xtick.color': self.colors['text_primary'],
                'ytick.color': self.colors['text_primary']
            })

        # Set default color cycle
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.palette_categorical)
        plt.rcParams['figure.facecolor'] = self.colors['background']
        plt.rcParams['axes.facecolor'] = self.colors['surface']
        plt.rcParams['text.color'] = self.colors['text_primary']
        plt.rcParams['axes.labelcolor'] = self.colors['text_primary']
        plt.rcParams['xtick.color'] = self.colors['text_primary']
        plt.rcParams['ytick.color'] = self.colors['text_primary']
        plt.rcParams['grid.color'] = self.colors['grid']
        plt.rcParams['grid.alpha'] = 0.3

    def get_color(self, color_name: str) -> str:
        """Get a color by name"""
        return self.colors.get(color_name, self.colors['primary'])

    def get_train_test_colors(self) -> Tuple[str, str]:
        """Get consistent train/test colors"""
        return self.colors['train'], self.colors['test']

    def get_categorical_palette(self, n_colors: int = None) -> List[str]:
        """Get categorical color palette"""
        if n_colors is None:
            return self.palette_categorical
        if n_colors <= len(self.palette_categorical):
            return self.palette_categorical[:n_colors]
        # Extend palette if needed
        extended = self.palette_categorical * (n_colors // len(self.palette_categorical) + 1)
        return extended[:n_colors]

    def get_severity_color(self, severity: str) -> str:
        """Get color based on severity level"""
        severity_map = {
            'LOW': self.colors['success'],
            'MEDIUM': self.colors['warning'],
            'HIGH': self.colors['danger'],
            'CRITICAL': self.colors['danger']
        }
        return severity_map.get(severity.upper(), self.colors['info'])

    def get_heatmap_colormap(self, center_zero: bool = True) -> str:
        """Get appropriate colormap for heatmaps"""
        if center_zero:
            if self.mode == 'dark':
                return 'RdBu_r'  # Red-Blue reversed
            else:
                return 'RdBu_r'  # Red-Blue reversed
        else:
            if self.mode == 'dark':
                return 'viridis'
            else:
                return 'Blues'

    def style_text(self, text: str, level: str = 'info') -> str:
        """Style text for terminal output"""
        level_colors = {
            'critical': Fore.RED + Style.BRIGHT,
            'warning': Fore.YELLOW + Style.BRIGHT,
            'info': Fore.CYAN + Style.BRIGHT,
            'success': Fore.GREEN + Style.BRIGHT,
            'header': Fore.MAGENTA + Style.BRIGHT,
            'subheader': Fore.BLUE + Style.BRIGHT,
        }
        color = level_colors.get(level, Fore.WHITE)
        return f"{color}{text}{Style.RESET_ALL}"

    def create_figure(self, figsize: Tuple[int, int] = (12, 8), **kwargs) -> plt.Figure:
        """Create a properly styled figure"""
        fig = plt.figure(figsize=figsize, facecolor=self.colors['background'], **kwargs)
        return fig


class DetailedTimer:
    """Context manager for detailed timing tracking"""
    def __init__(self, eda_instance, category: str, operation: str):
        self.eda = eda_instance
        self.category = category
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.eda._record_detailed_time(self.category, self.operation, duration)


class Config:
    state = 42
    n_splits = 10
    early_stop = 100

    target = 'BeatsPerMinute'
    train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

    target_categorical = False
    labels = list(train[target].unique()) if target in train.columns else []

    # PERFORMANCE OPTIMIZATION SETTINGS
    sample_size_pct: float = 100.0  # Percentage of data to use (1-100)
    fast_mode: bool = False  # Skip computationally expensive tests
    max_bootstrap_samples: int = 100  # Reduce for speed
    max_outlier_methods: int = 3  # Number of outlier detection methods (1-3: iqr, zscore, isolation_forest)
    max_distribution_methods: int = 5  # NEW: Number of distribution comparison methods (1-5)

    # STYLING SETTINGS
    style_mode: str = 'light'  # 'light', 'dark', or 'auto'

    # PROBLEM-SPECIFIC SETTINGS
    is_timeseries: bool = False
    datetime_column: Optional[str] = None  # Specify datetime column for time series
    is_high_cardinality: bool = False  # >100k unique values in categorical features
    enable_advanced_outlier_detection: bool = True
    enable_distribution_comparison: bool = True
    enable_predictive_power_analysis: bool = True
    enable_correlation_analysis: bool = True
    enable_feature_interactions: bool = True

    # COMPUTATIONAL LIMITS
    max_categories_to_plot: int = 20
    max_features_for_correlation: int = 50
    min_samples_for_statistical_tests: int = 30

class EDA(Config):
    def __init__(self):
        super().__init__()

        # Initialize styling first
        self.style = StyleManager(self.style_mode)

        # Performance tracking
        self.analysis_times = {}  # High-level times
        self.detailed_times = {}  # Detailed times: {category: {operation: duration}}
        self.total_start_time = time.time()

        # Sample data if needed FIRST
        self._prepare_data()

        # Initialize feature lists AFTER data preparation
        self.cat_features = self.train_sample.drop(self.target, axis=1).select_dtypes(include=['object']).columns.tolist()
        self.num_features = self.train_sample.drop(self.target, axis=1).select_dtypes(exclude=['object']).columns.tolist()

        # Store all issues found
        self.issues = {
            'critical': [],
            'warning': [],
            'info': []
        }

        # Configurations for outlier and distribution handling
        self.outlier_methods = ['iqr', 'zscore', 'isolation_forest']
        self.distribution_methods = ['ks_test', 'psi', 'js_divergence', 'wasserstein', 'robust_comparison']
        self.robust_comparison = True
        self.bootstrap_baseline = True

        # Initialize progress tracking AFTER feature lists are ready
        self.analysis_steps = self._get_analysis_steps()
        self.progress_bar = tqdm(total=len(self.analysis_steps),
                                 desc="EDA Analysis Progress",
                                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        print(self.style.style_text("ğŸ”� DATA QUALITY ASSESSMENT", 'header'))
        print(f"Dataset size: Train {self.train.shape}, Test {self.test.shape}")
        if self.sample_size_pct < 100:
            print(f"Using {self.sample_size_pct}% sample: Train {self.train_sample.shape}, Test {self.test_sample.shape}")
        print(f"Style mode: {self.style.mode.upper()}")
        print(f"Fast mode: {'ON' if self.fast_mode else 'OFF'}")
        print(f"Max outlier methods: {self.max_outlier_methods}")
        print(f"Max distribution methods: {self.max_distribution_methods}")
        if self.is_timeseries and self.datetime_column:
            print(f"Time series mode: ON (datetime column: '{self.datetime_column}')")
        print("=" * 70)

        # Run all checks in logical order with progress tracking
        self._run_analysis()

        # Final summary
        self._print_final_summary()

    def _record_detailed_time(self, category: str, operation: str, duration: float):
        """Record detailed timing for operations"""
        if category not in self.detailed_times:
            self.detailed_times[category] = {}
        self.detailed_times[category][operation] = duration

    def timer(self, category: str, operation: str):
        """Get a detailed timer context manager"""
        return DetailedTimer(self, category, operation)

    def _get_analysis_steps(self) -> List[Tuple[str, str]]:
        """Define analysis steps based on configuration"""
        steps = [
            ("basic_data_info", "Basic Data Info"),
            ("data_quality_checks", "Data Quality Checks"),
            ("feature_quality_checks", "Feature Quality Checks"),
        ]

        if len(self.num_features) > 0:
            steps.append(("numerical_feature_analysis", "Numerical Analysis"))

        if len(self.cat_features) > 0:
            steps.append(("categorical_feature_analysis", "Categorical Analysis"))

        steps.append(("target_analysis", "Target Analysis"))
        steps.append(("advanced_checks", "Advanced Checks"))

        if self.enable_predictive_power_analysis:
            steps.append(("predictive_power_analysis", "Predictive Power Analysis"))

        if self.enable_correlation_analysis and len(self.num_features) > 1:
            steps.append(("correlation_analysis", "Correlation Analysis"))

        if self.is_timeseries:
            steps.append(("timeseries_analysis", "Time Series Analysis"))

        return steps

    def _prepare_data(self):
        """Prepare data samples for analysis"""
        if self.sample_size_pct >= 100:
            self.train_sample = self.train.copy()
            self.test_sample = self.test.copy()
            return

        sample_ratio = self.sample_size_pct / 100.0

        # Smart sampling strategy
        if self.target_categorical and self.target in self.train.columns:
            # Stratified sampling for classification
            try:
                self.train_sample, _ = train_test_split(
                    self.train,
                    test_size=1-sample_ratio,
                    stratify=self.train[self.target],
                    random_state=self.state
                )
            except ValueError:
                # Fallback to random sampling if stratification fails
                self.train_sample = self.train.sample(frac=sample_ratio, random_state=self.state)
        else:
            # Random sampling for regression or when target is not available
            if self.is_timeseries:
                # For time series, take most recent data
                n_samples = int(len(self.train) * sample_ratio)
                self.train_sample = self.train.tail(n_samples)
            else:
                self.train_sample = self.train.sample(frac=sample_ratio, random_state=self.state)

        # Sample test set
        if self.is_timeseries:
            n_test_samples = int(len(self.test) * sample_ratio)
            self.test_sample = self.test.tail(n_test_samples)
        else:
            self.test_sample = self.test.sample(frac=sample_ratio, random_state=self.state)

    def _run_analysis(self):
        """Run analysis with progress tracking"""
        for method_name, description in self.analysis_steps:
            step_start_time = time.time()

            try:
                method = getattr(self, method_name)
                method()

                step_time = time.time() - step_start_time
                self.analysis_times[description] = step_time

                self.progress_bar.set_postfix_str(f"âœ“ {description} ({step_time:.2f}s)")
                self.progress_bar.update(1)

            except Exception as e:
                self.progress_bar.set_postfix_str(f"âœ— {description} failed: {str(e)}")
                self.progress_bar.update(1)
                print(self.style.style_text(f"Error in {description}: {str(e)}", 'critical'))

    def _print_final_summary(self):
        """Print timing and performance summary with detailed breakdown"""
        self.progress_bar.close()

        total_time = time.time() - self.total_start_time

        print(self.style.style_text("â�±ï¸� PERFORMANCE SUMMARY", 'header'))
        print("=" * 80)

        # Calculate total time for percentages
        for category_name, category_time in self.analysis_times.items():
            percentage = (category_time / total_time) * 100
            print(f"\n{self.style.style_text(f'ğŸ“Š {category_name}:', 'subheader')} {category_time:>8.2f}s ({percentage:>5.1f}%)")

            # Show detailed breakdown if available
            if category_name in self.detailed_times:
                detailed_ops = self.detailed_times[category_name]

                # Sort operations by time (descending)
                sorted_ops = sorted(detailed_ops.items(), key=lambda x: x[1], reverse=True)

                for operation, op_time in sorted_ops:
                    op_percentage = (op_time / category_time) * 100 if category_time > 0 else 0
                    print(f"   â””â”€ {operation:<35} {op_time:>8.2f}s ({op_percentage:>5.1f}%)")

        print("-" * 80)
        print(f"{'Total analysis time:':<30} {total_time:>8.2f}s")

        if self.sample_size_pct < 100:
            estimated_full_time = total_time * (100 / self.sample_size_pct)
            print(f"{'Estimated full dataset time:':<30} {estimated_full_time:>8.2f}s")

        # Summary of all issues
        self.print_issue_summary()

    def _add_issue(self, level: str, category: str, message: str, recommendation: str = ""):
        """Add an issue to the tracking system"""
        issue = {
            'category': category,
            'message': message,
            'recommendation': recommendation
        }
        self.issues[level].append(issue)

    def _smart_sample_for_analysis(self, data: pd.Series, max_samples: int = 10000) -> pd.Series:
        """Intelligently sample data for expensive operations"""
        if len(data) <= max_samples or not self.fast_mode:
            return data

        if self.is_timeseries:
            # For time series, take recent samples
            return data.tail(max_samples)
        else:
            # Random sampling
            return data.sample(n=max_samples, random_state=self.state)

    def _detect_outliers_multiple_methods(self, data, methods=['iqr', 'zscore', 'isolation_forest']):
        """Detect outliers using multiple methods and return consensus"""
        # Always respect max_outlier_methods limit
        methods = methods[:self.max_outlier_methods]

        # Sample data for expensive operations
        data_sample = self._smart_sample_for_analysis(data)

        outlier_indices = {}

        # Method 1: IQR
        if 'iqr' in methods:
            Q1 = data_sample.quantile(0.25)
            Q3 = data_sample.quantile(0.75)
            IQR = Q3 - Q1
            outlier_indices['iqr'] = data_sample[(data_sample < (Q1 - 1.5 * IQR)) | (data_sample > (Q3 + 1.5 * IQR))].index

        # Method 2: Z-score
        if 'zscore' in methods:
            try:
                z_scores = np.abs(stats.zscore(data_sample.dropna()))
                outlier_indices['zscore'] = data_sample.iloc[z_scores > 3].index
            except:
                outlier_indices['zscore'] = pd.Index([])

        # Method 3: Isolation Forest
        if 'isolation_forest' in methods and len(data_sample.dropna()) > 10:
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=self.state)
                outlier_mask = iso_forest.fit_predict(data_sample.dropna().values.reshape(-1, 1)) == -1
                outlier_indices['isolation_forest'] = data_sample.dropna().iloc[outlier_mask].index
            except:
                outlier_indices['isolation_forest'] = pd.Index([])

        # Consensus: outliers detected by at least 2 methods (or 1 if only 1 method available)
        all_outliers = set()
        for method_outliers in outlier_indices.values():
            all_outliers.update(method_outliers)

        consensus_outliers = []
        min_consensus = min(2, len(methods))  # Require 2 methods, but 1 if only 1 method available

        for outlier_idx in all_outliers:
            count = sum(1 for method_outliers in outlier_indices.values() if outlier_idx in method_outliers)
            if count >= min_consensus:
                consensus_outliers.append(outlier_idx)

        return pd.Index(consensus_outliers), outlier_indices

    def _calculate_psi(self, train_data, test_data, bins=10):
        """Calculate Population Stability Index (PSI)"""
        def psi_score(expected, actual, bins):
            try:
                # Sample data in fast mode
                if self.fast_mode:
                    expected = self._smart_sample_for_analysis(expected, 5000)
                    actual = self._smart_sample_for_analysis(actual, 5000)

                # Create bins based on expected (train) data
                _, bin_edges = np.histogram(expected, bins=bins)

                # Calculate expected percentages
                expected_percents, _ = np.histogram(expected, bins=bin_edges)
                expected_percents = expected_percents / len(expected)

                # Calculate actual percentages
                actual_percents, _ = np.histogram(actual, bins=bin_edges)
                actual_percents = actual_percents / len(actual)

                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                expected_percents = expected_percents + epsilon
                actual_percents = actual_percents + epsilon

                # Calculate PSI
                psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
                return psi
            except:
                return np.nan

        return psi_score(train_data, test_data, bins)

    def _jensen_shannon_divergence(self, p, q, bins=50):
        """Calculate Jensen-Shannon divergence between two distributions"""
        try:
            # Sample data in fast mode
            if self.fast_mode:
                p = self._smart_sample_for_analysis(p, 5000)
                q = self._smart_sample_for_analysis(q, 5000)

            # Create common bins
            combined_data = np.concatenate([p, q])
            bin_edges = np.linspace(combined_data.min(), combined_data.max(), bins + 1)

            # Calculate histograms
            p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
            q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

            # Normalize to probability distributions
            p_hist = p_hist / np.sum(p_hist) + 1e-10
            q_hist = q_hist / np.sum(q_hist) + 1e-10

            # Calculate JS divergence
            m = 0.5 * (p_hist + q_hist)
            js_div = 0.5 * entropy(p_hist, m) + 0.5 * entropy(q_hist, m)
            return js_div
        except:
            return np.nan

    def _calculate_wasserstein_distance(self, train_data, test_data):
        """Calculate Wasserstein (Earth Mover's) distance"""
        try:
            # Sample data in fast mode
            if self.fast_mode:
                train_data = self._smart_sample_for_analysis(train_data, 5000)
                test_data = self._smart_sample_for_analysis(test_data, 5000)

            distance = wasserstein_distance(train_data, test_data)
            return distance
        except:
            return np.nan

    def _create_bootstrap_baseline(self, data, n_bootstraps=None, sample_size=None):
        """Create bootstrap baseline for natural variance estimation"""
        if n_bootstraps is None:
            n_bootstraps = self.max_bootstrap_samples if self.fast_mode else 100

        if sample_size is None:
            sample_size = len(data) // 2

        bootstrap_stats = []
        for _ in range(n_bootstraps):
            # Create two random samples from the same distribution
            sample1 = np.random.choice(data, size=sample_size, replace=True)
            sample2 = np.random.choice(data, size=sample_size, replace=True)

            # Calculate KS statistic
            ks_stat, _ = ks_2samp(sample1, sample2)
            bootstrap_stats.append(ks_stat)

        return {
            'mean': np.mean(bootstrap_stats),
            'std': np.std(bootstrap_stats),
            'p95': np.percentile(bootstrap_stats, 95),
            'p99': np.percentile(bootstrap_stats, 99)
        }

    def _smart_distribution_comparison(self, train_data, test_data, feature_name):
        """Distribution comparison with configurable methods"""
        results = {}

        # Limit methods based on configuration
        available_methods = self.distribution_methods[:self.max_distribution_methods]

        # Original data comparison with configurable methods
        comparison_results = {}

        if 'ks_test' in available_methods:
            ks_stat, p_value = ks_2samp(train_data, test_data)
            comparison_results['ks_stat'] = ks_stat
            comparison_results['p_value'] = p_value

        if 'psi' in available_methods:
            psi = self._calculate_psi(train_data, test_data)
            comparison_results['psi'] = psi

        if 'js_divergence' in available_methods:
            js_div = self._jensen_shannon_divergence(train_data, test_data)
            comparison_results['js_divergence'] = js_div

        if 'wasserstein' in available_methods:
            wasserstein_dist = self._calculate_wasserstein_distance(train_data, test_data)
            comparison_results['wasserstein_distance'] = wasserstein_dist

        results['original'] = comparison_results

        # Robust comparison (without outliers) if enabled
        if 'robust_comparison' in available_methods:
            train_outliers, _ = self._detect_outliers_multiple_methods(train_data)
            test_outliers, _ = self._detect_outliers_multiple_methods(test_data)

            train_robust = train_data.drop(train_outliers, errors='ignore')
            test_robust = test_data.drop(test_outliers, errors='ignore')

            if len(train_robust) > 10 and len(test_robust) > 10:
                robust_results = {}

                if 'ks_test' in available_methods:
                    ks_stat_robust, p_value_robust = ks_2samp(train_robust, test_robust)
                    robust_results['ks_stat'] = ks_stat_robust
                    robust_results['p_value'] = p_value_robust

                if 'psi' in available_methods:
                    psi_robust = self._calculate_psi(train_robust, test_robust)
                    robust_results['psi'] = psi_robust

                if 'js_divergence' in available_methods:
                    js_div_robust = self._jensen_shannon_divergence(train_robust, test_robust)
                    robust_results['js_divergence'] = js_div_robust

                results['robust'] = robust_results

        # Bootstrap baseline for natural variance
        if self.bootstrap_baseline and len(train_data) > 50:
            baseline = self._create_bootstrap_baseline(train_data)
            results['baseline'] = baseline

        # Smart severity assessment
        severity = self._assess_distribution_shift_severity(results, feature_name, available_methods)
        results['severity'] = severity
        results['methods_used'] = available_methods

        return results

    def _assess_distribution_shift_severity(self, comparison_results, feature_name, methods_used):
        """Smart severity assessment considering multiple factors and used methods"""
        original = comparison_results['original']
        robust = comparison_results.get('robust', {})
        baseline = comparison_results.get('baseline', {})

        severity_scores = []

        # PSI-based severity (if available)
        if 'psi' in methods_used:
            psi = original.get('psi', 0)
            if psi < 0.1:
                severity_scores.append(1)  # LOW
            elif psi < 0.2:
                severity_scores.append(2)  # MEDIUM
            else:
                severity_scores.append(3)  # HIGH

        # KS test with baseline consideration (if available)
        if 'ks_test' in methods_used:
            ks_stat = original.get('ks_stat', 0)
            if baseline:
                # Compare against natural variance
                if ks_stat < baseline['p95']:
                    severity_scores.append(1)  # LOW
                elif ks_stat < baseline['p99']:
                    severity_scores.append(2)  # MEDIUM
                else:
                    severity_scores.append(3)  # HIGH
            else:
                # Traditional thresholds
                if ks_stat < 0.1:
                    severity_scores.append(1)  # LOW
                elif ks_stat < 0.3:
                    severity_scores.append(2)  # MEDIUM
                else:
                    severity_scores.append(3)  # HIGH

        # JS Divergence (if available)
        if 'js_divergence' in methods_used:
            js_div = original.get('js_divergence', 0)
            if js_div < 0.1:
                severity_scores.append(1)  # LOW
            elif js_div < 0.3:
                severity_scores.append(2)  # MEDIUM
            else:
                severity_scores.append(3)  # HIGH

        # Wasserstein distance (if available) - normalized by data scale
        if 'wasserstein' in methods_used:
            wasserstein_dist = original.get('wasserstein_distance', 0)
            # Normalize by data range for better interpretation
            data_range = abs(original.get('max', 1) - original.get('min', 0)) or 1
            normalized_wasserstein = wasserstein_dist / data_range

            if normalized_wasserstein < 0.1:
                severity_scores.append(1)  # LOW
            elif normalized_wasserstein < 0.3:
                severity_scores.append(2)  # MEDIUM
            else:
                severity_scores.append(3)  # HIGH

        # Robust vs original comparison (if available)
        improvement_factor = 1.0
        if robust and 'ks_test' in methods_used:
            ks_improvement = (original['ks_stat'] - robust['ks_stat']) / original['ks_stat']
            if ks_improvement > 0.5:  # Significant improvement without outliers
                improvement_factor = 0.5

        # Calculate average severity
        if severity_scores:
            combined_score = np.mean(severity_scores) * improvement_factor
        else:
            combined_score = 1.0  # Default to LOW if no methods available

        if combined_score <= 1.5:
            return 'LOW'
        elif combined_score <= 2.5:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _calculate_mutual_information(self, feature_data, target_data):
        """Calculate mutual information between feature and target"""
        try:
            # Sample data in fast mode
            if self.fast_mode and len(feature_data) > 5000:
                mask = np.random.choice(len(feature_data), 5000, replace=False)
                feature_sample = feature_data.iloc[mask]
                target_sample = target_data.iloc[mask]
            else:
                feature_sample = feature_data
                target_sample = target_data

            # Handle missing values
            mask = ~(pd.isna(feature_sample) | pd.isna(target_sample))
            if mask.sum() < 10:
                return np.nan

            feature_clean = feature_sample[mask]
            target_clean = target_sample[mask]

            if self.target_categorical:
                # For classification
                if feature_data.dtype == 'object':
                    # Categorical feature
                    le = LabelEncoder()
                    feature_encoded = le.fit_transform(feature_clean.astype(str))
                    mi = mutual_info_classif(feature_encoded.reshape(-1, 1), target_clean)[0]
                else:
                    # Numerical feature
                    mi = mutual_info_classif(feature_clean.values.reshape(-1, 1), target_clean)[0]
            else:
                # For regression
                if feature_data.dtype == 'object':
                    # Categorical feature
                    le = LabelEncoder()
                    feature_encoded = le.fit_transform(feature_clean.astype(str))
                    mi = mutual_info_regression(feature_encoded.reshape(-1, 1), target_clean)[0]
                else:
                    # Numerical feature
                    mi = mutual_info_regression(feature_clean.values.reshape(-1, 1), target_clean)[0]

            return mi
        except Exception as e:
            return np.nan

    def _calculate_cramers_v(self, x, y):
        """Calculate Cramer's V for categorical variables"""
        try:
            # Create contingency table
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        except:
            return np.nan

    def _calculate_feature_importance_quick(self, feature_data, target_data):
        """Quick feature importance using RÂ² or accuracy score instead of tree importance"""
        try:
            # Handle missing values
            mask = ~(pd.isna(feature_data) | pd.isna(target_data))
            if mask.sum() < 10:
                return np.nan
    
            feature_clean = feature_data[mask]
            target_clean = target_data[mask]
    
            # For categorical features, encode them
            if feature_data.dtype == 'object':
                try:
                    le = LabelEncoder()
                    feature_encoded = le.fit_transform(feature_clean.astype(str))
                except:
                    return np.nan
            else:
                feature_encoded = feature_clean.values
    
            feature_reshaped = feature_encoded.reshape(-1, 1)
    
            if self.target_categorical:
                # For classification: use cross-validated accuracy
                model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=self.state)
                try:
                    # Use simple train/validation split for speed
                    if len(feature_reshaped) > 100:
                        X_train, X_val, y_train, y_val = train_test_split(
                            feature_reshaped, target_clean, test_size=0.3,
                            random_state=self.state, stratify=target_clean
                        )
                        model.fit(X_train, y_train)
                        score = model.score(X_val, y_val)
                    else:
                        # For small datasets, use the full data
                        model.fit(feature_reshaped, target_clean)
                        score = model.score(feature_reshaped, target_clean)
    
                    # Convert accuracy to a more meaningful scale (subtract baseline)
                    baseline_accuracy = max(target_clean.value_counts()) / len(target_clean)
                    importance_score = max(0, score - baseline_accuracy)
                    return importance_score
    
                except Exception as e:
                    return np.nan
            else:
                # For regression: use RÂ² score
                model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=self.state)
                try:
                    if len(feature_reshaped) > 100:
                        X_train, X_val, y_train, y_val = train_test_split(
                            feature_reshaped, target_clean, test_size=0.3, random_state=self.state
                        )
                        model.fit(X_train, y_train)
                        score = model.score(X_val, y_val)
                    else:
                        model.fit(feature_reshaped, target_clean)
                        score = model.score(feature_reshaped, target_clean)
    
                    # Ensure non-negative scores
                    return max(0, score)
    
                except Exception as e:
                    return np.nan
    
        except Exception as e:
            return np.nan


    def basic_data_info(self):
        """Basic dataset information and structure"""
        with self.timer("Basic Data Info", "Dataset structure analysis"):
            print(self.style.style_text("ğŸ“Š BASIC DATASET INFORMATION", 'success'))

            # Dataset shapes and basic info
            print(f"Train shape: {self.train.shape}")
            print(f"Test shape: {self.test.shape}")
            if self.sample_size_pct < 100:
                print(f"Sample shapes - Train: {self.train_sample.shape}, Test: {self.test_sample.shape}")
            print(f"Target column: {self.target}")
            print(f"Target type: {'Classification' if self.target_categorical else 'Regression'}")

        with self.timer("Basic Data Info", "Size validation checks"):
            # Check for basic structural issues
            train_size = len(self.train_sample) if hasattr(self, 'train_sample') else len(self.train)
            if train_size < 100:
                self._add_issue('warning', 'Sample Size',
                                f"Very small training set ({train_size} samples)",
                                "Consider gathering more data or using appropriate validation strategies")

            test_size = len(self.test_sample) if hasattr(self, 'test_sample') else len(self.test)
            if test_size / train_size > 0.5:
                self._add_issue('warning', 'Sample Size',
                                f"Test set is large relative to train ({test_size}/{train_size})",
                                "Ensure this split makes sense for your use case")

        with self.timer("Basic Data Info", "Feature counting"):
            # Feature counts
            print(f"Numerical features: {len(self.num_features)}")
            print(f"Categorical features: {len(self.cat_features)}")

        with self.timer("Basic Data Info", "Data preview generation"):
            # Display basic info
            for data, label in zip([self.train_sample, self.test_sample], ['Train', 'Test']):
                print(f"\n{self.style.style_text(f'{label} Dataset Info:', 'subheader')}")
                display(data.head(3))
                missing_summary = data.isnull().sum()
                if missing_summary.sum() > 0:
                    print(f"Missing values:\n{missing_summary[missing_summary > 0]}")

    def data_quality_checks(self):
        """Comprehensive data quality checks"""
        print(self.style.style_text("ğŸ”� DATA QUALITY CHECKS", 'success'))

        # Use sample data for checks
        train_data = self.train_sample
        test_data = self.test_sample

        with self.timer("Data Quality Checks", "Duplicate detection"):
            # 1. Check for duplicate rows
            train_dups = train_data.duplicated().sum()
            test_dups = test_data.duplicated().sum()

            if train_dups > 0:
                self._add_issue('warning', 'Data Quality',
                                f"Found {train_dups} duplicate rows in training set",
                                "Consider removing duplicates or investigating why they exist")

            if test_dups > 0:
                self._add_issue('warning', 'Data Quality',
                                f"Found {test_dups} duplicate rows in test set",
                                "Consider removing duplicates")

        with self.timer("Data Quality Checks", "Index validation"):
            # 2. Check for index/ID issues
            if train_data.index.duplicated().any():
                self._add_issue('critical', 'Data Quality',
                                "Duplicate IDs found in training set",
                                "Fix ID duplicates before proceeding")

            if test_data.index.duplicated().any():
                self._add_issue('critical', 'Data Quality',
                                "Duplicate IDs found in test set",
                                "Fix ID duplicates before proceeding")

        with self.timer("Data Quality Checks", "Data leakage detection"):
            # 3. Check for potential data leakage in IDs
            common_ids = set(train_data.index) & set(test_data.index)
            if common_ids:
                self._add_issue('critical', 'Data Leakage',
                                f"Found {len(common_ids)} common IDs between train and test",
                                "Remove overlapping samples to prevent data leakage")

        with self.timer("Data Quality Checks", "Missing data analysis"):
            # 4. Missing data patterns
            self._analyze_missing_patterns()

        with self.timer("Data Quality Checks", "Data type consistency"):
            # 5. Data type consistency
            self._check_data_type_consistency()

        print("âœ“ Data quality checks completed")

    def _analyze_missing_patterns(self):
        """Analyze missing data patterns"""
        train_missing = self.train_sample.isnull().sum()
        test_missing = self.test_sample.isnull().sum()

        # Features with high missing rates
        high_missing_train = train_missing[train_missing > 0.5 * len(self.train_sample)]
        high_missing_test = test_missing[test_missing > 0.5 * len(self.test_sample)]

        for col in high_missing_train.index:
            self._add_issue('warning', 'Missing Data',
                            f"Feature '{col}' has {high_missing_train[col]/len(self.train_sample)*100:.1f}% missing in train",
                            "Consider dropping this feature or advanced imputation")

        for col in high_missing_test.index:
            self._add_issue('warning', 'Missing Data',
                            f"Feature '{col}' has {high_missing_test[col]/len(self.test_sample)*100:.1f}% missing in test",
                            "Consider dropping this feature or advanced imputation")

        # Different missing patterns between train/test
        for col in self.train_sample.columns:
            if col == self.target:
                continue
            train_missing_rate = train_missing[col] / len(self.train_sample)
            test_missing_rate = test_missing[col] / len(self.test_sample)

            if abs(train_missing_rate - test_missing_rate) > 0.1:
                self._add_issue('warning', 'Missing Data',
                                f"Feature '{col}' has different missing rates: train {train_missing_rate*100:.1f}%, test {test_missing_rate*100:.1f}%",
                                "Investigate why missing patterns differ")

    def _check_data_type_consistency(self):
        """Check for data type issues"""
        train_features = set(self.train_sample.columns) - {self.target}
        test_features = set(self.test_sample.columns)

        # Check for missing features
        missing_in_test = train_features - test_features
        missing_in_train = test_features - train_features

        if missing_in_test:
            self._add_issue('critical', 'Feature Mismatch',
                            f"Features in train but not test: {list(missing_in_test)}",
                            "Ensure feature consistency between datasets")

        if missing_in_train:
            self._add_issue('critical', 'Feature Mismatch',
                            f"Features in test but not train: {list(missing_in_train)}",
                            "Ensure feature consistency between datasets")

        # Check data type consistency
        common_features = train_features & test_features
        for col in common_features:
            if self.train_sample[col].dtype != self.test_sample[col].dtype:
                self._add_issue('warning', 'Data Types',
                                f"Feature '{col}' has different types: train({self.train_sample[col].dtype}) vs test({self.test_sample[col].dtype})",
                                "Ensure consistent data types")

    def feature_quality_checks(self):
        """Check feature quality issues"""
        print(self.style.style_text("âš™ï¸� FEATURE QUALITY CHECKS", 'success'))

        all_features = self.num_features + self.cat_features

        # Sample features for expensive checks if needed
        if self.fast_mode and len(all_features) > 50:
            features_to_check = all_features[:50]
            print(f"Fast mode: Checking first {len(features_to_check)} features")
        else:
            features_to_check = all_features

        with self.timer("Feature Quality Checks", "Constant feature detection"):
            # 1. Constant features
            constant_features = []
            for col in features_to_check:
                if self.train_sample[col].nunique() <= 1:
                    constant_features.append(col)
                    self._add_issue('warning', 'Feature Quality',
                                    f"Feature '{col}' is constant",
                                    "Consider removing constant features")

        with self.timer("Feature Quality Checks", "Near-constant feature detection"):
            # 2. Near-constant features (>95% same value)
            near_constant = []
            for col in features_to_check:
                if self.train_sample[col].value_counts().iloc[0] / len(self.train_sample) > 0.95:
                    near_constant.append(col)
                    self._add_issue('warning', 'Feature Quality',
                                    f"Feature '{col}' is near-constant (>95% same value)",
                                    "Consider removing or investigating this feature")

        with self.timer("Feature Quality Checks", "High cardinality detection"):
            # 3. High cardinality categorical features
            high_cardinality = []
            for col in self.cat_features:
                if col in features_to_check:
                    cardinality = self.train_sample[col].nunique()
                    if cardinality > 0.1 * len(self.train_sample):  # More than 10% of samples
                        high_cardinality.append((col, cardinality))
                        self._add_issue('warning', 'Feature Quality',
                                        f"Feature '{col}' has high cardinality ({cardinality} unique values)",
                                        "Consider grouping rare categories or using encoding techniques")

        with self.timer("Feature Quality Checks", "ID feature detection"):
            # 4. Potential ID features
            potential_ids = []
            for col in self.cat_features + self.num_features:
                if col in features_to_check and self.train_sample[col].nunique() == len(self.train_sample):
                    potential_ids.append(col)
                    self._add_issue('warning', 'Feature Quality',
                                    f"Feature '{col}' appears to be an ID (unique for each row)",
                                    "Remove if it's truly an ID, or investigate if it contains useful information")

        with self.timer("Feature Quality Checks", "Leakage keyword detection"):
            # 5. Features with suspicious names (potential leakage)
            suspicious_names = []
            leakage_keywords = ['target', 'label', 'result', 'outcome', 'prediction', 'forecast', 'future']
            for col in self.train_sample.columns:
                if col != self.target and any(keyword in col.lower() for keyword in leakage_keywords):
                    suspicious_names.append(col)
                    self._add_issue('critical', 'Potential Leakage',
                                    f"Feature '{col}' has suspicious name suggesting potential leakage",
                                    "Investigate if this feature contains future information")

        print("âœ“ Feature quality checks completed")

    def numerical_feature_analysis(self):
        """Numerical feature analysis with detailed timing"""
        if not self.num_features:
            print(self.style.style_text("ğŸ“ˆ No numerical features found", 'warning'))
            return

        print(self.style.style_text("ğŸ“ˆ NUMERICAL FEATURE ANALYSIS", 'success'))

        # Limit features in fast mode
        features_to_analyze = self.num_features
        if self.fast_mode and len(features_to_analyze) > 20:
            features_to_analyze = features_to_analyze[:20]
            print(f"Fast mode: Analyzing first {len(features_to_analyze)} numerical features")

        with self.timer("Numerical Analysis", "Individual feature analysis"):
            for col in features_to_analyze:
                self._analyze_numerical_feature(col)

        # Plot distribution comparisons
        if not self.fast_mode:
            with self.timer("Numerical Analysis", "Distribution plotting"):
                self._plot_numerical_distributions()

    def _analyze_numerical_feature(self, col):
        """Numerical feature analysis with robust methods"""
        train_data = self.train_sample[col].dropna()
        test_data = self.test_sample[col].dropna()

        if len(train_data) == 0 or len(test_data) == 0:
            self._add_issue('warning', 'Data Quality',
                            f"Feature '{col}' has no valid data in train or test",
                            "Investigate missing data patterns")
            return

        # Outlier detection with detailed timing
        if self.enable_advanced_outlier_detection:
            # Respect max_outlier_methods setting
            available_methods = ['iqr', 'zscore', 'isolation_forest']
            methods_to_use = available_methods[:self.max_outlier_methods]

            # Run individual methods for detailed timing
            for method in methods_to_use:
                with self.timer("Numerical Analysis", f"Outlier analysis: {method}"):
                    # Run individual method (result not used, just for timing)
                    _, _ = self._detect_outliers_multiple_methods(train_data, [method])

            # Get consensus result using the limited methods
            with self.timer("Numerical Analysis", f"Outlier consensus ({len(methods_to_use)} methods)"):
                consensus_outliers, outlier_methods_result = self._detect_outliers_multiple_methods(train_data, methods_to_use)

            outlier_percentage = len(consensus_outliers) / len(train_data) * 100
            if outlier_percentage > 5:
                self._add_issue('warning', 'Data Quality',
                                f"Feature '{col}' has {len(consensus_outliers)} consensus outliers ({outlier_percentage:.1f}%) using {len(methods_to_use)} methods",
                                "Consider robust methods or outlier treatment")

        # Smart distribution comparison with detailed timing
        if self.enable_distribution_comparison:
            with self.timer("Numerical Analysis", f"Distribution comparison: {col}"):
                comparison_results = self._smart_distribution_comparison(train_data, test_data, col)

                severity = comparison_results['severity']
                methods_used = comparison_results.get('methods_used', [])
                psi = comparison_results['original'].get('psi', 0)

                if severity == 'HIGH':
                    self._add_issue('warning', 'Distribution Shift',
                                    f"Feature '{col}' shows significant distribution shift (Severity: {severity}, Methods: {len(methods_used)})",
                                    "Investigate distribution differences - may impact model performance")
                elif severity == 'MEDIUM':
                    self._add_issue('info', 'Distribution Shift',
                                    f"Feature '{col}' shows moderate distribution shift (Severity: {severity}, Methods: {len(methods_used)})",
                                    "Monitor model performance on this feature")

        # Statistical checks with timing
        with self.timer("Numerical Analysis", f"Statistical validation: {col}"):
            if train_data.std() > 1000 * abs(train_data.mean()) and train_data.mean() != 0:
                self._add_issue('info', 'Feature Scaling',
                                f"Feature '{col}' has very large variance relative to mean",
                                "Consider scaling this feature")

            # Domain-aware checks
            if (train_data < 0).any() and any(keyword in col.lower() for keyword in ['count', 'size', 'age', 'duration']):
                self._add_issue('warning', 'Data Quality',
                                f"Feature '{col}' has negative values but seems to be a {col.lower()}",
                                "Investigate negative values")

    def _plot_numerical_distributions(self):
        """Plot numerical feature distributions with styled colors"""
        if not self.num_features:
            return

        print(self.style.style_text("Distribution Analysis with Robust Metrics", 'success'))

        # Print metrics explanation
        self._print_metrics_legend()

        # Limit features for plotting
        features_to_plot = self.num_features
        if len(features_to_plot) > 10:
            features_to_plot = features_to_plot[:10]
            print(f"Showing first {len(features_to_plot)} features")

        # Get styled colors
        train_color, test_color = self.style.get_train_test_colors()

        # Combine datasets for plotting
        df = pd.concat([
            self.train_sample[features_to_plot].assign(Source='Train'),
            self.test_sample[features_to_plot].assign(Source='Test')
        ], axis=0, ignore_index=True)

        fig = self.style.create_figure(figsize=(24, len(features_to_plot) * 6))

        gs = fig.add_gridspec(len(features_to_plot), 4,
                              hspace=0.4, wspace=0.3,
                              width_ratios=[0.35, 0.20, 0.25, 0.20])

        for i, col in enumerate(features_to_plot):
            train_data = self.train_sample[col].dropna()
            test_data = self.test_sample[col].dropna()

            # Plot 1: KDE comparison
            ax = fig.add_subplot(gs[i, 0])
            sns.kdeplot(data=df[[col, 'Source']], x=col, hue='Source',
                        palette=[train_color, test_color], ax=ax, linewidth=2)

            # Add quartile lines with styled colors
            for percentile, linestyle in [(25, '--'), (50, '-'), (75, '--')]:
                train_q = np.percentile(train_data, percentile)
                test_q = np.percentile(test_data, percentile)
                ax.axvline(train_q, color=train_color, linestyle=linestyle, alpha=0.7)
                ax.axvline(test_q, color=test_color, linestyle=linestyle, alpha=0.7)

            ax.set_title(f"{col} - Distribution\n(Train: {len(train_data):,}, Test: {len(test_data):,})")
            ax.grid(alpha=0.3, color=self.style.get_color('grid'))

            # Plot 2: Boxplot with outlier highlighting
            ax = fig.add_subplot(gs[i, 1])
            box_data = [train_data, test_data]
            bp = ax.boxplot(box_data, labels=['Train', 'Test'], patch_artist=True)
            bp['boxes'][0].set_facecolor(train_color)
            bp['boxes'][1].set_facecolor(test_color)

            # Highlight consensus outliers
            consensus_outliers, _ = self._detect_outliers_multiple_methods(train_data)
            if len(consensus_outliers) > 0:
                outlier_values = train_data[consensus_outliers]
                ax.scatter([1] * len(outlier_values), outlier_values,
                           c=self.style.get_color('warning'), s=30, alpha=0.7,
                           label=f'Consensus Outliers ({len(outlier_values)})')

            ax.set_title(f"{col} - Boxplot with Outliers")
            if len(consensus_outliers) > 0:
                ax.legend()

            # Plot 3: Distribution difference with multiple metrics
            ax = fig.add_subplot(gs[i, 2])
            comparison_results = self._smart_distribution_comparison(train_data, test_data, col)
            self._plot_distribution_difference(ax, comparison_results, col)

            # Plot 4: Outlier analysis
            ax = fig.add_subplot(gs[i, 3])
            self._plot_outlier_analysis(ax, train_data, test_data, col)

        plt.tight_layout()
        plt.show()

    def _plot_distribution_difference(self, ax, comparison_results, col):
        """Plot distribution difference with multiple metrics using styled colors"""
        original = comparison_results['original']
        robust = comparison_results.get('robust', {})
        baseline = comparison_results.get('baseline', {})
        severity = comparison_results['severity']
        methods_used = comparison_results.get('methods_used', [])

        # Create metrics summary
        metrics = []
        values = []
        colors = []

        # PSI (if available)
        if 'psi' in methods_used:
            psi = original.get('psi', 0)
            metrics.append('PSI')
            values.append(psi)
            if psi < 0.1:
                colors.append(self.style.get_color('success'))
            elif psi < 0.2:
                colors.append(self.style.get_color('warning'))
            else:
                colors.append(self.style.get_color('danger'))

        # KS Statistic (if available)
        if 'ks_test' in methods_used:
            ks_stat = original.get('ks_stat', 0)
            metrics.append('KS Stat')
            values.append(ks_stat)
            if baseline:
                threshold = baseline['p95']
                if ks_stat < threshold:
                    colors.append(self.style.get_color('success'))
                elif ks_stat < baseline['p99']:
                    colors.append(self.style.get_color('warning'))
                else:
                    colors.append(self.style.get_color('danger'))
            else:
                colors.append(self.style.get_color('info'))

        # JS Divergence (if available)
        if 'js_divergence' in methods_used:
            js_div = original.get('js_divergence', 0)
            metrics.append('JS Div')
            values.append(js_div)
            colors.append(self.style.get_color('secondary'))

        # Wasserstein Distance (if available)
        if 'wasserstein' in methods_used:
            wasserstein_dist = original.get('wasserstein_distance', 0)
            metrics.append('Wasserstein')
            values.append(wasserstein_dist)
            colors.append(self.style.get_color('neutral'))

        # Robust KS (if available)
        if robust and 'ks_test' in methods_used:
            ks_robust = robust.get('ks_stat', 0)
            metrics.append('KS Robust')
            values.append(ks_robust)
            colors.append(self.style.get_color('primary'))

        # Bar plot
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_title(f"{col} - Multiple Metrics\nSeverity: {severity}")
        ax.set_ylabel('Metric Value')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # Add baseline reference if available
        if baseline and 'ks_test' in methods_used:
            ax.axhline(y=baseline['p95'], color=self.style.get_color('text_secondary'),
                       linestyle='--', alpha=0.5, label='95% baseline')
            ax.axhline(y=baseline['p99'], color=self.style.get_color('text_secondary'),
                       linestyle=':', alpha=0.5, label='99% baseline')
            ax.legend(fontsize=8)

        ax.grid(axis='y', alpha=0.3, color=self.style.get_color('grid'))

    def _plot_outlier_analysis(self, ax, train_data, test_data, col):
        """Plot outlier analysis comparison using styled colors"""
        # Use only the configured number of methods
        available_methods = ['iqr', 'zscore', 'isolation_forest']
        methods_to_use = available_methods[:self.max_outlier_methods]

        # Detect outliers in both datasets using limited methods
        train_outliers, train_methods = self._detect_outliers_multiple_methods(train_data, methods_to_use)
        test_outliers, test_methods = self._detect_outliers_multiple_methods(test_data, methods_to_use)

        # Count outliers by method
        train_counts = []
        test_counts = []

        for method in methods_to_use:
            train_count = len(train_methods.get(method, []))
            test_count = len(test_methods.get(method, []))
            train_counts.append(train_count)
            test_counts.append(test_count)

        x = np.arange(len(methods_to_use))
        width = 0.35

        train_color, test_color = self.style.get_train_test_colors()

        bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color=train_color, alpha=0.7)
        bars2 = ax.bar(x + width/2, test_counts, width, label='Test', color=test_color, alpha=0.7)

        ax.set_xlabel('Outlier Detection Method')
        ax.set_ylabel('Number of Outliers')
        ax.set_title(f'{col} - Outlier Comparison ({len(methods_to_use)} methods)')
        ax.set_xticks(x)
        ax.set_xticklabels(methods_to_use)
        ax.legend()
        ax.grid(axis='y', alpha=0.3, color=self.style.get_color('grid'))

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

    def _print_metrics_legend(self):
        """Print metrics explanation with styled text"""
        legend_text = """
        ğŸ“Š DISTRIBUTION ANALYSIS METRICS:
        
        ğŸ”� ROBUST NUMERICAL ANALYSIS:
        â€¢ PSI (Population Stability Index): Industry standard for distribution drift
          â””â”€ <0.1: Stable âœ… | 0.1-0.2: Monitor âš ï¸� | >0.2: Significant drift â�Œ
        
        â€¢ KS with Bootstrap Baseline: Kolmogorov-Smirnov test with natural variance baseline
          â””â”€ Compares against expected natural variation from same distribution
        
        â€¢ JS Divergence: Jensen-Shannon divergence (symmetric, bounded measure)
          â””â”€ 0: Identical | 1: Completely different
        
        â€¢ Wasserstein Distance: Earth Mover's distance between distributions
          â””â”€ Measures cost of transforming one distribution to another
        
        â€¢ Robust Analysis: Metrics calculated after removing consensus outliers
          â””â”€ Shows if differences are due to bulk distribution or extreme values
        
        ğŸ�¯ CONSENSUS OUTLIER DETECTION:
        â€¢ Combines IQR, Z-score, and Isolation Forest methods
        â€¢ Flags points identified by multiple methods (more reliable)
        """
        print(self.style.style_text(legend_text, 'info'))

    def categorical_feature_analysis(self):
        """ Categorical feature analysis"""
        if not self.cat_features:
            print(self.style.style_text("ğŸ“Š No categorical features found", 'warning'))
            return

        print(self.style.style_text("ğŸ“Š CATEGORICAL FEATURE ANALYSIS", 'success'))

        # Limit features in fast mode
        features_to_analyze = self.cat_features
        if self.fast_mode and len(features_to_analyze) > 15:
            features_to_analyze = features_to_analyze[:15]
            print(f"Fast mode: Analyzing first {len(features_to_analyze)} categorical features")

        with self.timer("Categorical Analysis", "Category validation"):
            for col in features_to_analyze:
                # print(f"\n{self.style.style_text(f'Analyzing: {col}', 'subheader')}")

                # Get value counts
                train_values = set(self.train_sample[col].dropna().unique())
                test_values = set(self.test_sample[col].dropna().unique())

                # Check for unseen categories
                unseen_in_test = test_values - train_values
                unseen_in_train = train_values - test_values

                if unseen_in_test:
                    self._add_issue('critical', 'Categorical Mismatch',
                                    f"Feature '{col}' has unseen categories in test: {list(unseen_in_test)[:5]}",
                                    "Handle unseen categories with appropriate encoding/imputation")

                if unseen_in_train:
                    self._add_issue('warning', 'Categorical Mismatch',
                                    f"Feature '{col}' has categories only in train: {list(unseen_in_train)[:5]}",
                                    "These categories won't be useful for prediction")

                # Check for rare categories (appear in <1% of data)
                train_counts = self.train_sample[col].value_counts()
                rare_categories = train_counts[train_counts < 0.01 * len(self.train_sample)]

                if len(rare_categories) > 0:
                    self._add_issue('info', 'Categorical Quality',
                                    f"Feature '{col}' has {len(rare_categories)} rare categories (<1% frequency)",
                                    "Consider grouping rare categories")

        # Statistical comparison
        if not self.fast_mode:
            with self.timer("Categorical Analysis", "Distribution plotting"):
                for col in features_to_analyze[:5]:  # Limit plotting in detailed timing
                    self._plot_categorical_comparison(col)

    def _plot_categorical_comparison(self, col):
        """Plot categorical feature comparison between train and test with styled colors"""
        fig = self.style.create_figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

        # Get styled colors
        train_color, test_color = self.style.get_train_test_colors()

        # Get common categories for comparison
        train_counts = self.train_sample[col].value_counts().head(10)
        test_counts = self.test_sample[col].value_counts().head(10)

        # Plot 1: Train distribution
        ax = fig.add_subplot(gs[0, 0])
        train_counts.plot(kind='bar', ax=ax, color=train_color)
        ax.set_title(f'{col} - Train Distribution')
        ax.tick_params(axis='x', rotation=45)

        # Plot 2: Test distribution
        ax = fig.add_subplot(gs[0, 1])
        test_counts.plot(kind='bar', ax=ax, color=test_color)
        ax.set_title(f'{col} - Test Distribution')
        ax.tick_params(axis='x', rotation=45)

        # Plot 3: Proportion differences
        ax = fig.add_subplot(gs[0, 2])
        all_cats = list(set(train_counts.index) | set(test_counts.index))

        train_props = []
        test_props = []

        for cat in all_cats:
            train_prop = train_counts.get(cat, 0) / len(self.train_sample)
            test_prop = test_counts.get(cat, 0) / len(self.test_sample)
            train_props.append(train_prop)
            test_props.append(test_prop)

        diff = np.array(train_props) - np.array(test_props)
        colors = [train_color if x >= 0 else test_color for x in diff]

        ax.bar(range(len(all_cats)), diff * 100, color=colors, alpha=0.7)
        ax.set_xticks(range(len(all_cats)))
        ax.set_xticklabels(all_cats, rotation=45, ha='right')
        ax.set_ylabel('Proportion Difference (%)')
        ax.set_title(f'{col} - Train vs Test Proportions')
        ax.axhline(y=0, color=self.style.get_color('text_primary'), linestyle='-')
        ax.grid(axis='y', alpha=0.3, color=self.style.get_color('grid'))

        plt.tight_layout()
        plt.show()

    def target_analysis(self):
        """ Target variable analysis"""
        print(self.style.style_text("ğŸ�¯ TARGET VARIABLE ANALYSIS", 'success'))

        target_data = self.train_sample[self.target]

        with self.timer("Target Analysis", "Missing value check"):
            # Check for missing targets
            if target_data.isnull().sum() > 0:
                self._add_issue('critical', 'Target Quality',
                                f"Target has {target_data.isnull().sum()} missing values",
                                "Handle missing targets before modeling")

        with self.timer("Target Analysis", "Target distribution analysis"):
            if self.target_categorical:
                self._analyze_classification_target(target_data)
            else:
                self._analyze_regression_target(target_data)

        with self.timer("Target Analysis", "Leakage detection"):
            # Check for potential target leakage
            self._check_target_leakage()

    def _analyze_classification_target(self, target_data):
        """Analyze classification target with styled visualizations"""
        value_counts = target_data.value_counts()

        # Class imbalance check
        min_class_ratio = value_counts.min() / value_counts.max()

        if min_class_ratio < 0.1:
            self._add_issue('warning', 'Class Imbalance',
                            f"Severe class imbalance detected (ratio: {min_class_ratio:.3f})",
                            "Consider resampling techniques or appropriate metrics")
        elif min_class_ratio < 0.3:
            self._add_issue('info', 'Class Imbalance',
                            f"Moderate class imbalance detected (ratio: {min_class_ratio:.3f})",
                            "Monitor model performance on minority classes")

        # Plot target distribution
        if not self.fast_mode:
            fig = self.style.create_figure(figsize=(12, 6))
            gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

            # Get categorical palette
            n_classes = len(value_counts)
            colors = self.style.get_categorical_palette(n_classes)

            # Bar plot
            ax = fig.add_subplot(gs[0, 0])
            value_counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_title('Target Distribution (Counts)')
            ax.tick_params(axis='x', rotation=45)

            # Pie chart
            ax = fig.add_subplot(gs[0, 1])
            wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index,
                                              autopct='%1.1f%%', colors=colors)
            ax.set_title('Target Distribution (Proportions)')

            plt.tight_layout()
            plt.show()

    def _analyze_regression_target(self, target_data):
        """Analyze regression target with styled visualizations"""
        # Check for outliers
        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = target_data[(target_data < (Q1 - 3*IQR)) | (target_data > (Q3 + 3*IQR))]

        if len(outliers) > 0.05 * len(target_data):
            self._add_issue('warning', 'Target Quality',
                            f"Target has {len(outliers)} outliers ({len(outliers)/len(target_data)*100:.1f}%)",
                            "Consider robust regression methods or outlier treatment")

        # Check for skewness
        skewness = stats.skew(target_data.dropna())
        if abs(skewness) > 1:
            self._add_issue('info', 'Target Distribution',
                            f"Target is highly skewed (skewness: {skewness:.3f})",
                            "Consider log transformation or other normalization techniques")

        # Check for negative values where they might not make sense
        if (target_data < 0).any():
            self._add_issue('info', 'Target Values',
                            f"Target has {(target_data < 0).sum()} negative values",
                            "Ensure negative values are expected for your use case")

        # Plot target distribution
        if not self.fast_mode:
            fig = self.style.create_figure(figsize=(15, 5))
            gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

            primary_color = self.style.get_color('primary')

            # Histogram
            ax = fig.add_subplot(gs[0, 0])
            ax.hist(target_data.dropna(), bins=50, color=primary_color, alpha=0.7)
            ax.set_title('Target Distribution')
            ax.set_xlabel(self.target)
            ax.grid(alpha=0.3, color=self.style.get_color('grid'))

            # Box plot
            ax = fig.add_subplot(gs[0, 1])
            bp = ax.boxplot(target_data.dropna(), patch_artist=True)
            bp['boxes'][0].set_facecolor(primary_color)
            ax.set_title('Target Boxplot')
            ax.set_ylabel(self.target)
            ax.grid(alpha=0.3, color=self.style.get_color('grid'))

            # Q-Q plot for normality
            ax = fig.add_subplot(gs[0, 2])
            stats.probplot(target_data.dropna(), dist="norm", plot=ax)
            ax.set_title('Q-Q Plot (Normality Check)')
            ax.grid(alpha=0.3, color=self.style.get_color('grid'))

            plt.tight_layout()
            plt.show()

    def _check_target_leakage(self):
        """Check for potential target leakage in features"""
        if self.target_categorical:
            # For classification, check correlation with encoded target
            target_encoded_array = pd.factorize(self.train_sample[self.target])[0]
            target_encoded = pd.Series(target_encoded_array, index=self.train_sample.index)
        else:
            target_encoded = self.train_sample[self.target]

        high_corr_features = []

        features_to_check = self.num_features
        if self.fast_mode and len(features_to_check) > 20:
            features_to_check = features_to_check[:20]

        for col in features_to_check:
            if self.train_sample[col].dtype in ['int64', 'float64']:
                try:
                    correlation = abs(self.train_sample[col].corr(target_encoded))
                    if pd.notna(correlation) and correlation > 0.9:
                        high_corr_features.append((col, correlation))
                        self._add_issue('critical', 'Potential Leakage',
                                        f"Feature '{col}' has very high correlation with target ({correlation:.3f})",
                                        "Investigate if this feature contains future information")
                except Exception:
                    continue

    def advanced_checks(self):
        """Advanced data quality checks"""
        print(self.style.style_text("ğŸ”¬ ADVANCED QUALITY CHECKS", 'success'))

        with self.timer("Advanced Checks", "Temporal pattern analysis"):
            # 1. Check for temporal patterns (using specified datetime column if available)
            self._check_temporal_patterns()

        with self.timer("Advanced Checks", "Logical consistency validation"):
            # 2. Check for logical inconsistencies
            self._check_logical_consistency()

        # 3. Feature interaction analysis
        if self.enable_feature_interactions and not self.fast_mode:
            with self.timer("Advanced Checks", "Feature interaction analysis"):
                self._check_feature_interactions()

        print("âœ“ Advanced checks completed")

    def _check_temporal_patterns(self):
        """Check for temporal patterns using specified datetime column or keyword search"""
        date_like_cols = []

        # First, check if datetime_column is specified and exists
        if self.is_timeseries and self.datetime_column:
            if self.datetime_column in self.train_sample.columns:
                date_like_cols.append(self.datetime_column)
                print(f"Using specified datetime column: '{self.datetime_column}'")
            else:
                self._add_issue('critical', 'Time Series Configuration',
                                f"Specified datetime column '{self.datetime_column}' not found in data",
                                "Check datetime_column configuration or column name")

        # If no datetime column specified or not in time series mode, fall back to keyword search
        if not date_like_cols:
            # Look for date-like column names
            date_keywords = ['date', 'time', 'timestamp', 'created', 'modified', 'updated']

            for col in self.train_sample.columns:
                if any(keyword in col.lower() for keyword in date_keywords):
                    date_like_cols.append(col)

            # Also check for columns that might be dates but are numeric
            for col in self.num_features:
                if col in self.train_sample.columns:
                    # Check if values look like timestamps
                    sample_values = self.train_sample[col].dropna().head(100)
                    if len(sample_values) > 0:
                        # Check if values are in typical timestamp ranges
                        if sample_values.min() > 946684800 and sample_values.max() < 2147483647:  # 2000-2038 in unix time
                            date_like_cols.append(col)

        for col in date_like_cols:
            self._add_issue('warning', 'Temporal Features',
                            f"Feature '{col}' appears to be temporal",
                            "Ensure no future information is leaked and consider time-based splits")

    def _check_logical_consistency(self):
        """Check for logical inconsistencies in the data"""
        # Example checks - customize based on domain
        pass

    def _check_feature_interactions(self):
        """Check for problematic feature interactions"""
        # Look for highly correlated feature pairs
        if len(self.num_features) > 1:
            # Limit features for correlation analysis
            features_for_corr = self.num_features
            if len(features_for_corr) > self.max_features_for_correlation:
                features_for_corr = features_for_corr[:self.max_features_for_correlation]

            corr_matrix = self.train_sample[features_for_corr].corr()

            # Find high correlations (excluding diagonal)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    correlation = abs(corr_matrix.iloc[i, j])
                    if correlation > 0.95:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            correlation
                        ))

            for col1, col2, corr in high_corr_pairs:
                self._add_issue('warning', 'Feature Correlation',
                                f"Features '{col1}' and '{col2}' are highly correlated ({corr:.3f})",
                                "Consider removing one of these features to reduce multicollinearity")

    def _calculate_correlation_smart(self, feature_data, target_data):
        """Smart correlation calculation that handles different feature/target type combinations"""
        try:
            # Handle missing values
            mask = ~(pd.isna(feature_data) | pd.isna(target_data))
            if mask.sum() < 10:
                return np.nan
    
            feature_clean = feature_data[mask]
            target_clean = target_data[mask]
    
            # Determine feature and target types
            is_feature_categorical = feature_data.dtype == 'object'
            is_target_categorical = self.target_categorical
    
            if not is_feature_categorical and not is_target_categorical:
                # Numerical feature + Numerical target: Pearson correlation
                correlation = abs(feature_clean.corr(target_clean))
                return correlation if not pd.isna(correlation) else 0.0
    
            elif not is_feature_categorical and is_target_categorical:
                # Numerical feature + Categorical target: Point-biserial or eta-squared
                if target_clean.nunique() == 2:
                    # Binary target: point-biserial correlation
                    target_binary = pd.factorize(target_clean)[0]
                    correlation = abs(np.corrcoef(feature_clean, target_binary)[0, 1])
                    return correlation if not pd.isna(correlation) else 0.0
                else:
                    # Multi-class: use eta-squared (ANOVA-based)
                    try:
                        groups = [feature_clean[target_clean == cat] for cat in target_clean.unique()]
                        groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                        if len(groups) > 1:
                            f_stat, p_val = stats.f_oneway(*groups)
                            # Convert F-statistic to eta-squared approximation
                            eta_squared = f_stat / (f_stat + len(feature_clean) - len(groups))
                            return eta_squared if not pd.isna(eta_squared) else 0.0
                        else:
                            return 0.0
                    except:
                        return 0.0
    
            elif is_feature_categorical and not is_target_categorical:
                # Categorical feature + Numerical target: eta-squared
                try:
                    groups = [target_clean[feature_clean == cat] for cat in feature_clean.unique()]
                    groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                    if len(groups) > 1:
                        f_stat, p_val = stats.f_oneway(*groups)
                        eta_squared = f_stat / (f_stat + len(target_clean) - len(groups))
                        return eta_squared if not pd.isna(eta_squared) else 0.0
                    else:
                        return 0.0
                except:
                    return 0.0
    
            elif is_feature_categorical and is_target_categorical:
                # Categorical feature + Categorical target: Cramer's V
                return self._calculate_cramers_v(feature_clean, target_clean)
    
            else:
                return np.nan
    
        except Exception as e:
            return np.nan
    
    def predictive_power_analysis(self):
        """Analysis of predictive power using multiple metrics"""
        print(self.style.style_text("ğŸ�¯ PREDICTIVE POWER ANALYSIS", 'success'))
    
        target_data = self.train_sample[self.target]
    
        # Limit features in fast mode
        all_features = self.num_features + self.cat_features
        if self.fast_mode and len(all_features) > 30:
            features_to_analyze = all_features[:30]
            print(f"Fast mode: Analyzing predictive power for first {len(features_to_analyze)} features")
        else:
            features_to_analyze = all_features
    
        predictive_scores = []
    
        with self.timer("Predictive Power Analysis", "Multiple metrics calculation"):
            for feature in features_to_analyze:
                feature_data = self.train_sample[feature]
    
                # Calculate multiple metrics with fixed methods
                mutual_info = self._calculate_mutual_information(feature_data, target_data)
                tree_importance = self._calculate_feature_importance_quick(feature_data, target_data)
                correlation = self._calculate_correlation_smart(feature_data, target_data)
    
                # Composite predictive power score
                scores = [s for s in [mutual_info, tree_importance, correlation] if not pd.isna(s) and s > 0]
                if scores:
                    composite_score = np.mean(scores)
                else:
                    composite_score = 0
    
                predictive_scores.append({
                    'feature': feature,
                    'mutual_info': mutual_info,
                    'tree_importance': tree_importance,
                    'correlation': correlation,
                    'composite_score': composite_score,
                    'feature_type': 'numerical' if feature in self.num_features else 'categorical'
                })
    
        with self.timer("Predictive Power Analysis", "Results analysis and plotting"):
            # Convert to DataFrame and sort
            predictive_df = pd.DataFrame(predictive_scores).sort_values('composite_score', ascending=False)
    
            # Display top features
            print(f"\n{self.style.style_text('Top 10 Features by Predictive Power:', 'subheader')}")
            display(predictive_df.head(10)[['feature', 'composite_score', 'mutual_info', 'tree_importance', 'correlation']])
    
            # Flag potential issues
            low_predictive_features = predictive_df[predictive_df['composite_score'] < 0.01]
            if len(low_predictive_features) > 0:
                self._add_issue('info', 'Feature Selection',
                                f"{len(low_predictive_features)} features have very low predictive power",
                                "Consider removing these features to reduce noise")
    
            high_predictive_features = predictive_df[predictive_df['composite_score'] > 0.8]
            if len(high_predictive_features) > 0:
                for _, feature_info in high_predictive_features.iterrows():
                    self._add_issue('warning', 'Potential Leakage',
                                    f"Feature '{feature_info['feature']}' has very high predictive power ({feature_info['composite_score']:.3f})",
                                    "Investigate if this feature contains future information")
    
            # Plot predictive power comparison
            if not self.fast_mode:
                self._plot_predictive_power(predictive_df)
    
        return predictive_df

    def _plot_predictive_power(self, predictive_df):
        """Plot predictive power analysis with styled colors"""
        fig = self.style.create_figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Get colors for numerical vs categorical
        num_color = self.style.get_color('primary')
        cat_color = self.style.get_color('secondary')

        # Top features bar plot
        top_features = predictive_df.head(15)
        ax = fig.add_subplot(gs[0, 0])
        colors = [num_color if ft == 'numerical' else cat_color for ft in top_features['feature_type']]
        bars = ax.barh(range(len(top_features)), top_features['composite_score'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Composite Predictive Score')
        ax.set_title('Top 15 Features by Predictive Power')
        ax.grid(axis='x', alpha=0.3, color=self.style.get_color('grid'))

        # Mutual Information vs Tree Importance
        ax = fig.add_subplot(gs[0, 1])
        scatter_colors = [num_color if ft == 'numerical' else cat_color for ft in predictive_df['feature_type']]
        ax.scatter(predictive_df['mutual_info'], predictive_df['tree_importance'],
                   c=scatter_colors, alpha=0.6)
        ax.set_xlabel('Mutual Information')
        ax.set_ylabel('Tree Importance')
        ax.set_title('Mutual Information vs Tree Importance')
        ax.grid(alpha=0.3, color=self.style.get_color('grid'))

        # Distribution of predictive scores
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(predictive_df['composite_score'], bins=20, color=self.style.get_color('primary'),
                alpha=0.7, edgecolor=self.style.get_color('text_primary'))
        ax.set_xlabel('Composite Predictive Score')
        ax.set_ylabel('Number of Features')
        ax.set_title('Distribution of Predictive Scores')
        ax.grid(axis='y', alpha=0.3, color=self.style.get_color('grid'))

        # Feature type comparison
        ax = fig.add_subplot(gs[1, 1])
        type_scores = predictive_df.groupby('feature_type')['composite_score'].agg(['mean', 'std'])
        type_scores['mean'].plot(kind='bar', ax=ax, color=[num_color, cat_color],
                                 yerr=type_scores['std'], capsize=5)
        ax.set_xlabel('Feature Type')
        ax.set_ylabel('Average Predictive Score')
        ax.set_title('Predictive Power by Feature Type')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3, color=self.style.get_color('grid'))

        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        """Correlation analysis with styled heatmap - fixed to include all meaningful correlations"""
        print(self.style.style_text("ğŸ”— CORRELATION ANALYSIS", 'success'))
    
        with self.timer("Correlation Analysis", "Correlation matrix calculation"):
            # Include both numerical features and the target for correlation
            features_for_corr = self.num_features.copy()
    
            # Add target if it's numerical or if we can convert it meaningfully
            if not self.target_categorical:
                features_for_corr.append(self.target)
            else:
                # For categorical targets, create a numeric version for correlation
                target_numeric = pd.factorize(self.train_sample[self.target])[0]
                correlation_data = self.train_sample[features_for_corr].copy()
                correlation_data[f'{self.target}_encoded'] = target_numeric
                features_for_corr.append(f'{self.target}_encoded')
    
            if len(features_for_corr) > self.max_features_for_correlation:
                features_for_corr = features_for_corr[:self.max_features_for_correlation]
                print(f"Limiting correlation analysis to first {len(features_for_corr)} features")
    
        with self.timer("Correlation Analysis", "Correlation visualization"):
            if not self.fast_mode and len(features_for_corr) > 1:
                fig = self.style.create_figure(figsize=(12, 10))
    
                if self.target_categorical and f'{self.target}_encoded' in features_for_corr:
                    corr_matrix = correlation_data[features_for_corr].corr()
                else:
                    corr_matrix = self.train_sample[features_for_corr].corr()
    
                # Create a mask for the upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
                # Generate heatmap with styled colors
                cmap = self.style.get_heatmap_colormap(center_zero=True)
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, center=0,
                            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                plt.show()
            else:
                print("Skipping correlation visualization (fast mode or insufficient features)")

    def timeseries_analysis(self):
        """Time series specific analysis using specified datetime column"""
        if not self.is_timeseries:
            return

        print(self.style.style_text("ğŸ“ˆ TIME SERIES ANALYSIS", 'success'))

        with self.timer("Time Series Analysis", "Datetime column validation"):
            # Use specified datetime column if available
            if self.datetime_column:
                if self.datetime_column not in self.train_sample.columns:
                    self._add_issue('critical', 'Time Series Configuration',
                                    f"Specified datetime column '{self.datetime_column}' not found in training data",
                                    "Check datetime_column configuration")
                    return

                date_cols = [self.datetime_column]
                print(f"Using specified datetime column: '{self.datetime_column}'")
            else:
                # Fall back to looking for date columns
                date_cols = []
                for col in self.train_sample.columns:
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                        date_cols.append(col)

                if not date_cols:
                    self._add_issue('warning', 'Time Series',
                                    "No datetime column specified and no obvious date columns found",
                                    "Specify datetime_column in config or ensure proper time indexing")
                    return

        with self.timer("Time Series Analysis", "Temporal ordering validation"):
            # Check for temporal ordering
            for col in date_cols[:2]:  # Check first 2 date columns
                try:
                    date_series = pd.to_datetime(self.train_sample[col])
                    if not date_series.is_monotonic_increasing:
                        self._add_issue('warning', 'Time Series',
                                        f"Date column '{col}' is not monotonically increasing",
                                        "Ensure proper temporal ordering for time series analysis")
                    else:
                        print(f"âœ“ Date column '{col}' is properly ordered")
                except Exception as e:
                    self._add_issue('warning', 'Time Series',
                                    f"Could not parse date column '{col}': {str(e)}",
                                    "Check date format and ensure proper datetime parsing")

        with self.timer("Time Series Analysis", "Time-based data leakage check"):
            # Additional time series specific checks
            if self.datetime_column and self.datetime_column in self.train_sample.columns:
                try:
                    train_dates = pd.to_datetime(self.train_sample[self.datetime_column])
                    test_dates = pd.to_datetime(self.test_sample[self.datetime_column]) if self.datetime_column in self.test_sample.columns else None

                    # Check for data leakage (test dates before train dates)
                    if test_dates is not None:
                        max_train_date = train_dates.max()
                        min_test_date = test_dates.min()

                        if min_test_date < max_train_date:
                            overlap_count = (test_dates < max_train_date).sum()
                            self._add_issue('critical', 'Temporal Data Leakage',
                                            f"Found {overlap_count} test samples with dates before latest training date",
                                            "Ensure proper temporal split - test data should be after training data")
                        else:
                            print(f"âœ“ Proper temporal split: test data ({min_test_date}) after training data ({max_train_date})")

                except Exception as e:
                    self._add_issue('warning', 'Time Series',
                                    f"Error validating temporal ordering: {str(e)}",
                                    "Check datetime column format and data quality")

        print("âœ“ Time series analysis completed")

    def print_issue_summary(self):
        """Print comprehensive summary of all issues found"""
        print(f"\n{Style.BRIGHT}{Fore.CYAN}ğŸ“‹ DATA QUALITY ASSESSMENT SUMMARY{Style.RESET_ALL}")
        print("=" * 70)

        total_issues = len(self.issues['critical']) + len(self.issues['warning']) + len(self.issues['info'])

        if total_issues == 0:
            print(f"{Style.BRIGHT}{Fore.GREEN}âœ… No significant issues detected! Data appears to be of excellent quality.{Style.RESET_ALL}")
            return

        # Critical issues
        if self.issues['critical']:
            print(f"\n{Style.BRIGHT}{Fore.RED}ğŸš¨ CRITICAL ISSUES ({len(self.issues['critical'])}):ğŸš¨{Style.RESET_ALL}")
            print(f"{Fore.RED}These issues MUST be addressed before modeling:{Style.RESET_ALL}")
            for i, issue in enumerate(self.issues['critical'], 1):
                print(f"{Fore.RED}{i}. [{issue['category']}] {issue['message']}{Style.RESET_ALL}")
                print(f"   ğŸ’¡ Recommendation: {issue['recommendation']}\n")

        # Warning issues
        if self.issues['warning']:
            print(f"\n{Style.BRIGHT}{Fore.YELLOW}âš ï¸� WARNING ISSUES ({len(self.issues['warning'])}):âš ï¸�{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}These issues should be investigated and potentially addressed:{Style.RESET_ALL}")
            for i, issue in enumerate(self.issues['warning'][:10], 1):  # Show first 10
                print(f"{Fore.YELLOW}{i}. [{issue['category']}] {issue['message']}{Style.RESET_ALL}")
                print(f"   ğŸ’¡ Recommendation: {issue['recommendation']}\n")

            if len(self.issues['warning']) > 10:
                print(f"{Fore.YELLOW}... and {len(self.issues['warning']) - 10} more warnings{Style.RESET_ALL}")

        # Info issues
        if self.issues['info']:
            print(f"\n{Style.BRIGHT}{Fore.BLUE}â„¹ï¸� INFORMATIONAL NOTES ({len(self.issues['info'])}):â„¹ï¸�{Style.RESET_ALL}")
            print(f"{Fore.BLUE}These are observations that might be relevant:{Style.RESET_ALL}")
            for i, issue in enumerate(self.issues['info'][:5], 1):  # Show first 5
                print(f"{Fore.BLUE}{i}. [{issue['category']}] {issue['message']}{Style.RESET_ALL}")
                print(f"   ğŸ’¡ Recommendation: {issue['recommendation']}\n")

            if len(self.issues['info']) > 5:
                print(f"{Fore.BLUE}... and {len(self.issues['info']) - 5} more info items{Style.RESET_ALL}")

        # Summary statistics
        print(f"\n{Style.BRIGHT}ğŸ“Š SUMMARY STATISTICS:{Style.RESET_ALL}")
        print(f"Total issues found: {total_issues}")
        print(f"Critical: {len(self.issues['critical'])}")
        print(f"Warnings: {len(self.issues['warning'])}")
        print(f"Info: {len(self.issues['info'])}")

        # Priority action plan
        print(f"\n{Style.BRIGHT}{Fore.MAGENTA}ğŸ�¯ PRIORITY ACTION PLAN:{Style.RESET_ALL}")
        if self.issues['critical']:
            print(f"{Fore.RED}1. Address all CRITICAL issues immediately - these will break your model{Style.RESET_ALL}")
        if self.issues['warning']:
            print(f"{Fore.YELLOW}2. Investigate WARNING issues - these may significantly impact performance{Style.RESET_ALL}")
        if self.issues['info']:
            print(f"{Fore.BLUE}3. Consider INFO items for model optimization{Style.RESET_ALL}")

        print(f"\n{Style.BRIGHT}âœ¨ Ready to proceed with modeling once critical and warning issues are resolved!{Style.RESET_ALL}")


eda = EDA()


def analyze_numerical_regression(data: pd.DataFrame, target: str, feature_col: str,
                                 n_bins: int = 8, binning_method: str = 'quantile') -> pd.DataFrame:
    """
    Analyze numerical feature relationships with continuous target using binning.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target : str
        Target column name (continuous)
    feature_col : str
        Numerical feature column name
    n_bins : int
        Number of bins to create
    binning_method : str
        'quantile' (equal frequency) or 'uniform' (equal width)
        
    Returns:
    --------
    pd.DataFrame
        Analysis results with means, deviations, effect sizes, and predictive strength
    """
    # Remove missing values
    clean_data = data[[feature_col, target]].dropna()

    if len(clean_data) == 0:
        raise ValueError(f"No valid data remaining after removing NAs")

    # Get baseline statistics
    baseline_mean = clean_data[target].mean()
    baseline_std = clean_data[target].std()
    baseline_median = clean_data[target].median()

    # Create bins
    try:
        if binning_method == 'quantile':
            bins = pd.qcut(clean_data[feature_col], q=n_bins, duplicates='drop', precision=2)
        else:  # uniform
            bins = pd.cut(clean_data[feature_col], bins=n_bins, precision=2)
    except Exception as e:
        raise ValueError(f"Binning failed: {str(e)}. Try reducing n_bins or changing binning_method.")

    if bins.isna().all():
        raise ValueError(f"All bins are NaN - unable to create valid bins")

    clean_data = clean_data.copy()
    clean_data['bin'] = bins

    # Remove any remaining NaN bins
    clean_data = clean_data.dropna(subset=['bin'])

    if len(clean_data) == 0:
        raise ValueError(f"No valid data remaining after binning")

    # Calculate stats for each bin
    results = []
    bin_sample_sizes = []

    for bin_label in clean_data['bin'].cat.categories:
        mask = clean_data['bin'] == bin_label
        bin_data = clean_data[mask]

        if len(bin_data) == 0:
            continue

        bin_sample_sizes.append(len(bin_data))

        # Basic statistics
        total_count = len(bin_data)
        bin_values = bin_data[target]

        bin_mean = bin_values.mean()
        bin_std = bin_values.std()
        bin_median = bin_values.median()

        # Deviations and effect sizes
        mean_deviation = bin_mean - baseline_mean
        median_deviation = bin_median - baseline_median

        # Cohen's d effect size
        pooled_std = np.sqrt(((len(bin_data) - 1) * bin_std**2 +
                              (len(clean_data) - len(bin_data) - 1) * baseline_std**2) /
                             (len(clean_data) - 2))
        cohens_d = mean_deviation / pooled_std if pooled_std > 0 else 0

        # Percentage deviation from baseline
        pct_deviation = (mean_deviation / baseline_mean * 100) if baseline_mean != 0 else 0

        # Statistical significance (t-test against rest of data)
        other_data = clean_data[~mask][target]
        if len(other_data) > 0 and len(bin_data) > 1:
            t_stat, p_value = stats.ttest_ind(bin_values, other_data)
        else:
            t_stat, p_value = 0, 1

        # Predictive strength classification based on Cohen's d
        abs_cohens_d = abs(cohens_d)
        if abs_cohens_d > 0.8:
            strength = 'High'
        elif abs_cohens_d > 0.5:
            strength = 'Medium'
        elif abs_cohens_d > 0.2:
            strength = 'Low'
        else:
            strength = 'Very Low'

        # Statistical significance classification
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'

        # Bin range information
        bin_range = f"[{bin_label.left:.2f}, {bin_label.right:.2f}]"
        bin_center = (bin_label.left + bin_label.right) / 2
        bin_width = bin_label.right - bin_label.left

        results.append({
            'bin_range': bin_range,
            'bin_center': round(bin_center, 3),
            'bin_width': round(bin_width, 3),
            'total_samples': total_count,
            'mean': round(bin_mean, 4),
            'median': round(bin_median, 4),
            'std': round(bin_std, 4),
            'mean_deviation': round(mean_deviation, 4),
            'median_deviation': round(median_deviation, 4),
            'pct_deviation': round(pct_deviation, 2),
            'cohens_d': round(cohens_d, 3),
            'abs_cohens_d': round(abs_cohens_d, 3),
            't_statistic': round(t_stat, 3),
            'p_value': round(p_value, 4),
            'predictive_strength': strength,
            'significance': significance
        })

    # Check for sample size imbalances and warn user
    if bin_sample_sizes:
        min_samples = min(bin_sample_sizes)
        max_samples = max(bin_sample_sizes)
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')

        if imbalance_ratio > 10:
            print(f"WARNING: Large sample size imbalance in '{feature_col}' bins!")
            print(f"Min samples: {min_samples:,}, Max samples: {max_samples:,} (ratio: {imbalance_ratio:.1f}x)")
            print("Consider using 'uniform' binning method or adjusting n_bins.")

    result_df = pd.DataFrame(results)
    return result_df.sort_values('bin_center').reset_index(drop=True)

def calculate_numerical_correlations_regression(data: pd.DataFrame, target: str, feature_col: str) -> Dict:
    """Calculate various correlation metrics for numerical feature with continuous target."""
    clean_data = data[[feature_col, target]].dropna()

    # Pearson correlation (linear relationship)
    pearson_r, pearson_p = stats.pearsonr(clean_data[feature_col], clean_data[target])

    # Spearman correlation (monotonic relationship)
    spearman_r, spearman_p = stats.spearmanr(clean_data[feature_col], clean_data[target])

    # Kendall's tau (rank-based correlation, robust to outliers)
    kendall_tau, kendall_p = stats.kendalltau(clean_data[feature_col], clean_data[target])

    return {
        'pearson_r': round(pearson_r, 4),
        'pearson_p': round(pearson_p, 4),
        'spearman_r': round(spearman_r, 4),
        'spearman_p': round(spearman_p, 4),
        'kendall_tau': round(kendall_tau, 4),
        'kendall_p': round(kendall_p, 4)
    }

def detect_regression_trend(analysis_result: pd.DataFrame) -> Dict:
    """Detect trends in the binned regression analysis."""
    target_means = analysis_result['mean'].values
    bin_centers = analysis_result['bin_center'].values

    # Calculate Spearman correlation between bin centers and target means
    spearman_r, spearman_p = stats.spearmanr(bin_centers, target_means)

    # Calculate Pearson correlation for linear trend
    pearson_r, pearson_p = stats.pearsonr(bin_centers, target_means)

    # Determine trend direction and strength (use stronger of the two correlations)
    stronger_r = spearman_r if abs(spearman_r) > abs(pearson_r) else pearson_r

    if abs(stronger_r) > 0.8:
        trend_strength = 'Strong'
    elif abs(stronger_r) > 0.6:
        trend_strength = 'Moderate'
    elif abs(stronger_r) > 0.3:
        trend_strength = 'Weak'
    else:
        trend_strength = 'None'

    trend_direction = 'Positive' if stronger_r > 0 else 'Negative' if stronger_r < 0 else 'None'

    # Linearity assessment
    linearity_diff = abs(abs(pearson_r) - abs(spearman_r))
    if linearity_diff < 0.1:
        linearity = 'Linear'
    elif linearity_diff < 0.3:
        linearity = 'Mostly Linear'
    else:
        linearity = 'Non-Linear'

    return {
        'trend_correlation_spearman': round(spearman_r, 4),
        'trend_p_value_spearman': round(spearman_p, 4),
        'trend_correlation_pearson': round(pearson_r, 4),
        'trend_p_value_pearson': round(pearson_p, 4),
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'linearity': linearity,
        'is_monotonic': trend_strength in ['Strong', 'Moderate'],
        'stronger_correlation': round(stronger_r, 4)
    }

def calculate_binned_r2(analysis_result: pd.DataFrame, baseline_mean: float) -> float:
    """Calculate RÂ² for binned means vs bin centers."""
    if len(analysis_result) < 2:
        return 0.0

    bin_centers = analysis_result['bin_center'].values.reshape(-1, 1)
    target_means = analysis_result['mean'].values

    # Weight by sample sizes
    weights = analysis_result['total_samples'].values

    # Simple linear regression
    model = LinearRegression()
    model.fit(bin_centers, target_means, sample_weight=weights)
    r2 = model.score(bin_centers, target_means, sample_weight=weights)

    return max(0, r2)  # Ensure non-negative

def _create_target_distribution_plot(ax, analysis_result, baseline_mean, feature_name):
    """Create bar chart showing target means by bins with baseline reference."""
    bin_labels = [f"Bin {i+1}" for i in range(len(analysis_result))]
    target_means = analysis_result['mean']

    # Color bars based on deviation from baseline
    colors = ['green' if x > baseline_mean else 'red' for x in target_means]

    bars = ax.bar(bin_labels, target_means, color=colors, alpha=0.7)
    ax.axhline(y=baseline_mean, color='blue', linestyle='--', linewidth=2,
               label=f'Baseline Mean: {baseline_mean:.3f}')

    # Better y-axis scaling to show variation
    y_min, y_max = target_means.min(), target_means.max()
    y_range = y_max - y_min
    if y_range > 0:
        padding = y_range * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_ylabel('Target Mean Value')
    ax.set_title(f'{feature_name} - Target Means by Bins')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, val in zip(bars, target_means):
        height = bar.get_height()
        y_offset = (y_max - y_min) * 0.02 if y_range > 0 else 0.02
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

def _create_trend_plot_regression(ax, analysis_result, feature_name):
    """Create trend plot showing target mean vs bin centers."""
    bin_centers = analysis_result['bin_center']
    target_means = analysis_result['mean']
    sample_sizes = analysis_result['total_samples']

    # Normalize sample sizes for better scatter plot scaling
    max_samples = sample_sizes.max()
    normalized_sizes = (sample_sizes / max_samples) * 100 + 20  # Scale between 20-120

    # Create scatter plot with sizes proportional to sample sizes
    scatter = ax.scatter(bin_centers, target_means, s=normalized_sizes,
                         alpha=0.6, c='blue', edgecolors='darkblue', linewidth=1)

    # Add trend line (no fill)
    ax.plot(bin_centers, target_means, color='red', linewidth=2, marker='o',
            markersize=4, alpha=0.8)

    ax.set_xlabel('Feature Value (Bin Centers)')
    ax.set_ylabel('Target Mean')
    ax.set_title(f'{feature_name} - Target Mean Trend')
    ax.grid(True, alpha=0.3)

    # Add sample size legend
    ax.text(0.02, 0.98, f'Bubble size âˆ� Sample size\nMax samples: {max_samples:,}',
            transform=ax.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def _create_deviation_plot_regression(ax, analysis_result, feature_name):
    """Create deviation from baseline bar chart for regression."""
    bin_labels = [f"Bin {i+1}" for i in range(len(analysis_result))]
    mean_deviation = analysis_result['mean_deviation']

    colors = ['green' if x > 0 else 'red' for x in mean_deviation]
    bars = ax.bar(bin_labels, mean_deviation, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Better y-axis scaling to show variation
    if len(mean_deviation) > 0:
        y_min, y_max = mean_deviation.min(), mean_deviation.max()
        y_range = y_max - y_min
        if y_range > 0:
            padding = y_range * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_ylabel('Mean Deviation from Baseline')
    ax.set_title(f'{feature_name} - Mean Deviation by Bin')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, val in zip(bars, mean_deviation):
        height = bar.get_height()
        y_range = mean_deviation.max() - mean_deviation.min() if len(mean_deviation) > 0 else 1
        y_offset = y_range * 0.02 if y_range > 0 else 0.02

        if height >= 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height - y_offset,
                    f'{val:.3f}', ha='center', va='top', fontsize=9)

def _create_effect_size_plot_regression(ax, analysis_result, feature_name):
    """Create Cohen's d effect size plot for regression."""
    bin_labels = [f"Bin {i+1}" for i in range(len(analysis_result))]
    cohens_d = analysis_result['cohens_d']

    colors = ['green' if x > 0 else 'red' for x in cohens_d]
    bars = ax.bar(bin_labels, cohens_d, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Adaptive reference lines based on actual data range
    y_min, y_max = cohens_d.min(), cohens_d.max()
    y_range = y_max - y_min
    max_abs = max(abs(y_min), abs(y_max))

    # Only show reference lines if they're in a reasonable range
    if max_abs < 2.0:  # If effect sizes are reasonable
        if max_abs > 0.15:  # Only show small effect line if data warrants it
            ax.axhline(y=0.2, color='orange', linestyle=':', alpha=0.7, label='Small Effect (0.2)')
            ax.axhline(y=-0.2, color='orange', linestyle=':', alpha=0.7)
        if max_abs > 0.4:  # Only show medium effect line if data warrants it
            ax.axhline(y=0.5, color='blue', linestyle=':', alpha=0.7, label='Medium Effect (0.5)')
            ax.axhline(y=-0.5, color='blue', linestyle=':', alpha=0.7)
        if max_abs > 0.7:  # Only show large effect line if data warrants it
            ax.axhline(y=0.8, color='purple', linestyle=':', alpha=0.7, label='Large Effect (0.8)')
            ax.axhline(y=-0.8, color='purple', linestyle=':', alpha=0.7)

    # Better y-axis scaling
    if y_range > 0:
        padding = max(y_range * 0.1, 0.05)  # At least 0.05 padding
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_ylabel("Cohen's d")
    ax.set_title(f'{feature_name} - Effect Size by Bin')
    if max_abs > 0.15:  # Only show legend if reference lines are shown
        ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)

    # Add value labels for better readability
    for bar, val in zip(bars, cohens_d):
        height = bar.get_height()
        y_offset = (y_max - y_min) * 0.03 if y_range > 0 else 0.01
        if height >= 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height - y_offset,
                    f'{val:.3f}', ha='center', va='top', fontsize=8)

def _create_sample_size_plot_regression(ax, analysis_result, feature_name):
    """Create sample size bar chart for bins."""
    bin_labels = [f"Bin {i+1}" for i in range(len(analysis_result))]
    sample_sizes = analysis_result['total_samples']

    ax.bar(bin_labels, sample_sizes, alpha=0.7, color='orange')
    ax.set_ylabel('Sample Count')
    ax.set_title(f'{feature_name} - Sample Sizes per Bin')
    ax.tick_params(axis='x', rotation=45)

def _create_correlation_summary_regression(ax, correlations, trend_info, binned_r2, feature_name):
    """Create correlation and trend summary table for regression."""
    ax.axis('tight')
    ax.axis('off')

    def format_pvalue(p):
        """Format p-value for better readability."""
        if p < 0.0001:
            return "p<0.0001"
        else:
            return f"p={p:.4f}"

    summary_data = [
        ['Pearson r', f"{correlations['pearson_r']:.4f}", format_pvalue(correlations['pearson_p'])],
        ['Spearman r', f"{correlations['spearman_r']:.4f}", format_pvalue(correlations['spearman_p'])],
        ['Kendall Ï„', f"{correlations['kendall_tau']:.4f}", format_pvalue(correlations['kendall_p'])],
        ['Binned RÂ²', f"{binned_r2:.4f}", ''],
        ['Trend Direction', trend_info['trend_direction'], ''],
        ['Trend Strength', trend_info['trend_strength'], ''],
        ['Linearity', trend_info['linearity'], ''],
        ['Monotonic?', 'Yes' if trend_info['is_monotonic'] else 'No', '']
    ]

    table = ax.table(cellText=summary_data,
                     colLabels=['Metric', 'Value', 'P-Value'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)
    ax.set_title(f'{feature_name} - Correlation & Trend Summary')

def plot_numerical_regression_analysis(data: pd.DataFrame, analysis_result: pd.DataFrame,
                                       baseline_mean: float, feature_name: str, target_col: str,
                                       figsize: Tuple[int, int] = (18, 12)):
    """Create comprehensive visualization for a numerical feature in regression context."""
    try:
        # Calculate correlations, trend info, and binned RÂ²
        correlations = calculate_numerical_correlations_regression(data, target_col, feature_name)
        trend_info = detect_regression_trend(analysis_result)
        binned_r2 = calculate_binned_r2(analysis_result, baseline_mean)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

        # Create all subplots with individual error handling
        plot_configs = [
            (gs[0, 0], _create_target_distribution_plot, (analysis_result, baseline_mean, feature_name)),
            (gs[0, 1], _create_trend_plot_regression, (analysis_result, feature_name)),
            (gs[0, 2], _create_deviation_plot_regression, (analysis_result, feature_name)),
            (gs[1, 0], _create_effect_size_plot_regression, (analysis_result, feature_name)),
            (gs[1, 1], _create_sample_size_plot_regression, (analysis_result, feature_name)),
            (gs[1, 2], _create_correlation_summary_regression, (correlations, trend_info, binned_r2, feature_name))
        ]

        for position, plot_func, args in plot_configs:
            try:
                ax = fig.add_subplot(position)
                plot_func(ax, *args)
            except Exception as e:
                print(f"Warning: Error creating {plot_func.__name__} for {feature_name}: {str(e)}")
                ax = fig.add_subplot(position)
                ax.text(0.5, 0.5, f'Error in {plot_func.__name__}\n{str(e)}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{feature_name} - Plot Error')

        plt.suptitle(f'Comprehensive Numerical Regression Analysis: {feature_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Critical error in plot_numerical_regression_analysis for {feature_name}: {str(e)}")
        # Show a simple fallback plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Critical Error for {feature_name}\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Error analyzing {feature_name}')
        plt.show()

def calculate_numerical_regression_importance(analysis_result: pd.DataFrame,
                                              correlations: Dict, trend_info: Dict,
                                              binned_r2: float) -> float:
    """
    Calculate weighted importance score for a numerical feature in regression.
    Combines binned analysis effect sizes with correlation strength and explained variance.
    """
    # Effect size component (weighted by sample size)
    weighted_effects = analysis_result['abs_cohens_d'] * analysis_result['total_samples']
    total_samples = analysis_result['total_samples'].sum()
    effect_score = weighted_effects.sum() / total_samples if total_samples > 0 else 0

    # Correlation component (use strongest correlation)
    correlation_score = max(abs(correlations['pearson_r']),
                            abs(correlations['spearman_r']),
                            abs(correlations['kendall_tau']))

    # RÂ² component (explained variance)
    r2_score = binned_r2

    # Trend component (bonus for strong trends)
    trend_bonus = 0.1 if trend_info['is_monotonic'] else 0
    linearity_bonus = 0.05 if trend_info['linearity'] == 'Linear' else 0

    # Combined score (weighted combination + bonuses)
    # Effect size gets highest weight as it's most interpretable
    combined_score = (effect_score * 0.4 +
                      correlation_score * 0.3 +
                      r2_score * 0.3) + trend_bonus + linearity_bonus

    return round(combined_score, 4)

def get_numerical_regression_summary(data: pd.DataFrame, target_col: str,
                                     analysis_result: pd.DataFrame, feature_name: str) -> Dict:
    """Generate comprehensive summary statistics for a numerical feature in regression."""
    baseline_mean = data[target_col].mean()
    baseline_std = data[target_col].std()

    # Calculate correlations, trends, and RÂ²
    correlations = calculate_numerical_correlations_regression(data, target_col, feature_name)
    trend_info = detect_regression_trend(analysis_result)
    binned_r2 = calculate_binned_r2(analysis_result, baseline_mean)

    # Calculate importance
    importance = calculate_numerical_regression_importance(analysis_result, correlations,
                                                           trend_info, binned_r2)

    # Feature statistics
    feature_stats = data[feature_name].describe()

    return {
        'feature': feature_name,
        'num_bins': len(analysis_result),
        'total_samples': analysis_result['total_samples'].sum(),
        'max_effect_size': analysis_result['abs_cohens_d'].max(),
        'avg_effect_size': analysis_result['abs_cohens_d'].mean(),
        'importance_score': importance,
        'significant_bins': (analysis_result['significance'] != 'ns').sum(),
        'high_effect_bins': (analysis_result['predictive_strength'] == 'High').sum(),
        'baseline_mean': round(baseline_mean, 4),
        'baseline_std': round(baseline_std, 4),
        'pearson_r': correlations['pearson_r'],
        'spearman_r': correlations['spearman_r'],
        'kendall_tau': correlations['kendall_tau'],
        'binned_r2': round(binned_r2, 4),
        'trend_direction': trend_info['trend_direction'],
        'trend_strength': trend_info['trend_strength'],
        'linearity': trend_info['linearity'],
        'is_monotonic': trend_info['is_monotonic'],
        'feature_mean': round(feature_stats['mean'], 3),
        'feature_std': round(feature_stats['std'], 3),
        'feature_min': round(feature_stats['min'], 3),
        'feature_max': round(feature_stats['max'], 3)
    }

def analyze_all_numerical_regression_features(data: pd.DataFrame, target_col: str,
                                              n_bins: int = 8, binning_method: str = 'quantile',
                                              min_samples_per_bin: int = 10) -> pd.DataFrame:
    """
    Analyze all numerical features for regression and return ranking by importance.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_col : str
        Target column name (continuous)
    n_bins : int
        Number of bins for analysis
    binning_method : str
        'quantile' or 'uniform' binning
    min_samples_per_bin : int
        Minimum samples required per bin
        
    Returns:
    --------
    pd.DataFrame
        Feature ranking by importance
    """
    # Get numerical features (exclude target)
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_features:
        numerical_features.remove(target_col)

    baseline_mean = data[target_col].mean()
    baseline_std = data[target_col].std()
    feature_summaries = []

    for feature in numerical_features:
        try:
            # Check if feature has enough variation and valid data
            feature_data = data[feature].dropna()
            if len(feature_data) < 50:
                print(f"Skipping '{feature}': insufficient data after removing NAs ({len(feature_data)} samples)")
                continue

            if feature_data.nunique() < 3:
                print(f"Skipping '{feature}': insufficient variation (< 3 unique values)")
                continue

            # Check for extreme values that might cause binning issues
            if feature_data.std() == 0:
                print(f"Skipping '{feature}': no variation (std = 0)")
                continue

            # Analyze feature
            analysis_result = analyze_numerical_regression(data, target_col, feature,
                                                           n_bins, binning_method)

            # Validate analysis results
            if analysis_result.empty or len(analysis_result) < 2:
                print(f"Skipping '{feature}': insufficient bins created ({len(analysis_result)} bins)")
                continue

            # Check if bins have sufficient samples
            min_samples = analysis_result['total_samples'].min()
            if min_samples < min_samples_per_bin:
                print(f"Warning: Some bins in '{feature}' have < {min_samples_per_bin} samples (min: {min_samples})")

            # Create plots
            plot_numerical_regression_analysis(data, analysis_result, baseline_mean, feature, target_col)

            # Store summary
            summary = get_numerical_regression_summary(data, target_col, analysis_result, feature)
            feature_summaries.append(summary)

        except Exception as e:
            print(f"Error analyzing feature '{feature}': {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            continue

    # Create and sort summary DataFrame
    if feature_summaries:
        summary_df = pd.DataFrame(feature_summaries)
        summary_df = summary_df.sort_values('importance_score', ascending=False).reset_index(drop=True)
        return summary_df
    else:
        return pd.DataFrame()

def run_numerical_regression_analysis(data: pd.DataFrame, target_col: str,
                                      n_bins: int = 8, binning_method: str = 'quantile',
                                      min_samples_per_bin: int = 10) -> pd.DataFrame:
    """
    Main function to run complete numerical feature analysis for regression.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_col : str
        Target column name (continuous)
    n_bins : int
        Number of bins for analysis
    binning_method : str
        'quantile' (equal frequency) or 'uniform' (equal width)
    min_samples_per_bin : int
        Minimum samples per bin
        
    Usage:
    ------
    # Basic usage:
    train_data = your_dataframe
    target_column = 'price'  # or any continuous target
    feature_ranking = run_numerical_regression_analysis(train_data, target_column)
    
    # With custom parameters:
    feature_ranking = run_numerical_regression_analysis(train_data, target_column, 
                                                       n_bins=10, binning_method='uniform')
    """
    baseline_mean = data[target_col].mean()
    baseline_std = data[target_col].std()
    baseline_median = data[target_col].median()

    print(f"Dataset baseline statistics:")
    print(f"Mean: {baseline_mean:.4f}")
    print(f"Std:  {baseline_std:.4f}")
    print(f"Median: {baseline_median:.4f}")
    print(f"Binning method: {binning_method} with {n_bins} bins")
    print("="*70)

    # Analyze all features
    summary = analyze_all_numerical_regression_features(data, target_col, n_bins,
                                                        binning_method, min_samples_per_bin)

    if not summary.empty:
        print("\nNUMERICAL FEATURE RANKING BY PREDICTIVE POWER:")
        display_cols = ['feature', 'importance_score', 'max_effect_size', 'pearson_r',
                        'binned_r2', 'trend_strength', 'linearity']
        print(summary[display_cols].to_string(index=False))

        # Additional insights
        top_features = summary.head(3)['feature'].tolist()
        monotonic_features = summary[summary['is_monotonic'] == True]['feature'].tolist()
        linear_features = summary[summary['linearity'] == 'Linear']['feature'].tolist()
        strong_corr_features = summary[abs(summary['pearson_r']) > 0.5]['feature'].tolist()
        high_r2_features = summary[summary['binned_r2'] > 0.25]['feature'].tolist()

        print(f"\nTOP 3 MOST PREDICTIVE NUMERICAL FEATURES: {', '.join(top_features)}")
        if monotonic_features:
            print(f"FEATURES WITH MONOTONIC TRENDS: {', '.join(monotonic_features)}")
        if linear_features:
            print(f"FEATURES WITH LINEAR RELATIONSHIPS: {', '.join(linear_features)}")
        if strong_corr_features:
            print(f"FEATURES WITH STRONG CORRELATIONS (|r| > 0.5): {', '.join(strong_corr_features)}")
        if high_r2_features:
            print(f"FEATURES WITH HIGH BINNED RÂ² (> 0.25): {', '.join(high_r2_features)}")

        print(f"\nLEGEND:")
        print(f"- importance_score: Combined predictive power score")
        print(f"- max_effect_size: Maximum Cohen's d across bins")
        print(f"- binned_r2: RÂ² from binned means vs bin centers")
        print(f"- linearity: Linear vs Non-Linear relationship")

    else:
        print("No numerical features found or analyzed.")

    return summary

feature_ranking = run_numerical_regression_analysis(Config.train, Config.target)




