# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# =====================================================
# COMPREHENSIVE DATA EXPLORATION AND VISUALIZATION TOOLKIT
# =====================================================
# This module provides thorough analysis of the Student Bag Price dataset
# with advanced visualizations and automated insights generation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats
from matplotlib.gridspec import GridSpec
from IPython.display import display, HTML
import matplotlib.ticker as mtick

# Set visualization styles and parameters
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# Color palette for consistent visualizations
COLORS = {
    'primary': '#1f77b4',    # Main color for key metrics
    'secondary': '#ff7f0e',  # Complementary color
    'tertiary': '#2ca02c',   # Third accent color
    'alert': '#d62728',      # For highlighting issues/outliers
    'neutral': '#7f7f7f',    # For background elements
    'highlight': '#bcbd22',  # For emphasizing important patterns
    'categorical': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
}

class BagPriceExplorer:
    """
    Comprehensive tool for exploring and visualizing the Student Bag Price dataset.
    
    This class provides a suite of analysis functions to understand patterns, 
    relationships, and quality issues in the dataset before modeling.
    """
    
    def __init__(self, base_path='/kaggle/input/'):
        """
        Initialize the explorer with default paths and settings.
        
        Parameters:
        -----------
        base_path : str
            Base directory where datasets are stored
        """
        self.base_path = base_path
        self.competition_path = os.path.join(base_path, 'playground-series-s5e2')
        self.original_path = os.path.join(base_path, 'student-bag-price-prediction-dataset')
        self.train_data = None
        self.test_data = None
        self.original_data = None
        self.categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                                    'Waterproof', 'Style', 'Color']
        self.numerical_features = ['Weight Capacity (kg)', 'Compartments']
        self.target = 'Price'
        self.insights = []
        
    def load_datasets(self, verbose=True):
        """
        Load all datasets from the specified paths.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print information about loaded datasets
            
        Returns:
        --------
        tuple
            (train_data, test_data, original_data)
        """
        try:
            print("ğŸ”� Loading datasets...")
            
            # Load the datasets
            self.train_data = pd.read_csv(os.path.join(self.competition_path, 'train.csv'))
            self.test_data = pd.read_csv(os.path.join(self.competition_path, 'test.csv'))
            
            try:
                self.original_data = pd.read_csv(os.path.join(self.original_path, 
                                               'Noisy_Student_Bag_Price_Prediction_Dataset.csv'))
            except FileNotFoundError:
                print("âš ï¸� Original dataset not found. Analysis will continue with training data only.")
                self.original_data = None
            
            if verbose:
                print(f"âœ… Train data: {self.train_data.shape[0]:,} rows, {self.train_data.shape[1]} columns")
                print(f"âœ… Test data: {self.test_data.shape[0]:,} rows, {self.test_data.shape[1]} columns")
                if self.original_data is not None:
                    print(f"âœ… Original data: {self.original_data.shape[0]:,} rows, {self.original_data.shape[1]} columns")
            
            # Validate the expected columns are present
            self._validate_columns()
            
            return self.train_data, self.test_data, self.original_data
            
        except Exception as e:
            print(f"â�Œ Error loading datasets: {str(e)}")
            return None, None, None
    
    def _validate_columns(self):
        """
        Validate that expected columns are present in the datasets.
        Updates categorical and numerical features lists based on available columns.
        """
        # Update categorical features to only include those in the dataset
        if self.train_data is not None:
            self.categorical_features = [col for col in self.categorical_features 
                                        if col in self.train_data.columns]
            self.numerical_features = [col for col in self.numerical_features 
                                      if col in self.train_data.columns]
            
            # Add insight about missing expected columns
            all_expected = self.categorical_features + self.numerical_features + [self.target]
            missing_cols = [col for col in ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                                           'Waterproof', 'Style', 'Color', 
                                           'Weight Capacity (kg)', 'Compartments', 'Price'] 
                           if col not in all_expected]
            
            if missing_cols:
                self.insights.append(f"The dataset is missing these expected columns: {', '.join(missing_cols)}")
    
    def summarize_data(self):
        """
        Display a comprehensive summary of the datasets with key statistics.
        """
        if self.train_data is None:
            print("â�Œ No data loaded. Please run load_datasets() first.")
            return
        
        print("\n" + "="*80)
        print("ğŸ“Š DATASET SUMMARY")
        print("="*80)
        
        # Basic info and shape
        print("\nğŸ“‹ BASIC INFORMATION")
        print(f"Training set: {self.train_data.shape[0]:,} rows, {self.train_data.shape[1]} columns")
        print(f"Test set: {self.test_data.shape[0]:,} rows, {self.test_data.shape[1]} columns")
        if self.original_data is not None:
            print(f"Original dataset: {self.original_data.shape[0]:,} rows, {self.original_data.shape[1]} columns")
        
        # Duplicate analysis
        print("\nğŸ”� DUPLICATE ROWS")
        train_dupes = self.train_data.duplicated().sum()
        test_dupes = self.test_data.duplicated().sum()
        print(f"Training set duplicates: {train_dupes:,} ({train_dupes/len(self.train_data):.2%} of data)")
        print(f"Test set duplicates: {test_dupes:,} ({test_dupes/len(self.test_data):.2%} of data)")
        
        if train_dupes > 0:
            self.insights.append(f"The training set contains {train_dupes} duplicate rows which may affect model generalization.")
        
        # Data types overview
        print("\nğŸ“Š DATA TYPES")
        dtypes_df = pd.DataFrame({
            'Column': self.train_data.columns,
            'Type': self.train_data.dtypes,
            'Train Unique Values': self.train_data.nunique(),
            'Test Unique Values': [self.test_data[col].nunique() 
                                  if col in self.test_data.columns else 'N/A' 
                                  for col in self.train_data.columns]
        })
        display(dtypes_df)
        
        # Missing values analysis
        self.analyze_missing_values(summary_only=True)
        
        # Target variable summary (if present)
        if self.target in self.train_data.columns:
            print("\nğŸ�¯ TARGET VARIABLE SUMMARY")
            target_stats = self.train_data[self.target].describe()
            print(f"Min: {target_stats['min']:.2f}, Max: {target_stats['max']:.2f}, Mean: {target_stats['mean']:.2f}")
            print(f"25th percentile: {target_stats['25%']:.2f}, Median: {target_stats['50%']:.2f}, 75th percentile: {target_stats['75%']:.2f}")
            print(f"Standard deviation: {target_stats['std']:.2f}")
            
            skewness = self.train_data[self.target].skew()
            kurtosis = self.train_data[self.target].kurt()
            print(f"Skewness: {skewness:.2f} ({'Highly skewed' if abs(skewness) > 1 else 'Moderately skewed' if abs(skewness) > 0.5 else 'Approximately symmetric'})")
            print(f"Kurtosis: {kurtosis:.2f} ({'Heavy-tailed' if kurtosis > 3 else 'Light-tailed'})")
            
            if abs(skewness) > 0.5:
                self.insights.append(f"The target variable '{self.target}' is {('positively' if skewness > 0 else 'negatively')} skewed ({skewness:.2f}), which may require transformation before modeling.")
        
        # Categorical features overview
        if self.categorical_features:
            print("\nğŸ“‹ CATEGORICAL FEATURES SUMMARY")
            for col in self.categorical_features:
                value_counts = self.train_data[col].value_counts()
                print(f"{col}: {value_counts.shape[0]} unique values")
                print(f"  Top 3: {', '.join([f'{v} ({c:,})' for v, c in value_counts.head(3).items()])}")
                
                # Check for severe class imbalance
                top_class_pct = value_counts.iloc[0] / value_counts.sum()
                if top_class_pct > 0.8:
                    print(f"  âš ï¸� Severe imbalance: '{value_counts.index[0]}' represents {top_class_pct:.1%} of values")
                    self.insights.append(f"'{col}' is highly imbalanced with one class representing {top_class_pct:.1%} of the data.")
                print()
        
        # Numerical features overview
        if self.numerical_features:
            print("\nğŸ“Š NUMERICAL FEATURES SUMMARY")
            num_stats = self.train_data[self.numerical_features].describe().T
            num_stats['skew'] = self.train_data[self.numerical_features].skew()
            num_stats['kurt'] = self.train_data[self.numerical_features].kurtosis()
            display(num_stats.round(2))
            
            for col in self.numerical_features:
                skewness = self.train_data[col].skew()
                if abs(skewness) > 1:
                    self.insights.append(f"'{col}' is highly skewed ({skewness:.2f}) and may benefit from transformation.")
    
    def analyze_missing_values(self, summary_only=False):
        """
        Analyze missing values in all datasets with visualizations.
        
        Parameters:
        -----------
        summary_only : bool
            If True, only show summarized text output without visualizations
        """
        if self.train_data is None:
            print("â�Œ No data loaded. Please run load_datasets() first.")
            return
        
        print("\nğŸ”� MISSING VALUES ANALYSIS")
        
        # Create missing value statistics
        missing_stats = pd.DataFrame({
            'Feature': self.train_data.columns,
            'Train_Missing': self.train_data.isnull().sum(),
            'Train_Missing_Pct': (self.train_data.isnull().sum() / len(self.train_data) * 100).round(2),
            'Test_Missing': self.test_data.isnull().sum(),
            'Test_Missing_Pct': (self.test_data.isnull().sum() / len(self.test_data) * 100).round(2),
        })
        
        if self.original_data is not None:
            missing_stats['Original_Missing'] = self.original_data.isnull().sum()
            missing_stats['Original_Missing_Pct'] = (self.original_data.isnull().sum() / 
                                                    len(self.original_data) * 100).round(2)
        
        # Filter to features with missing values
        has_missing = (missing_stats['Train_Missing'] > 0) | (missing_stats['Test_Missing'] > 0)
        if self.original_data is not None:
            has_missing = has_missing | (missing_stats['Original_Missing'] > 0)
        
        missing_stats_filtered = missing_stats[has_missing]
        
        if len(missing_stats_filtered) == 0:
            print("âœ… No missing values found in any dataset!")
            return
        
        # Display summary table
        print("\nColumns with missing values:")
        display(missing_stats_filtered)
        
        # Add insights about missing values
        for _, row in missing_stats_filtered.iterrows():
            feature = row['Feature']
            train_pct = row['Train_Missing_Pct']
            test_pct = row['Test_Missing_Pct']
            
            if train_pct > 0 and test_pct > 0:
                self.insights.append(f"'{feature}' has missing values in both train ({train_pct:.1f}%) and test ({test_pct:.1f}%) datasets.")
            elif train_pct > 0:
                self.insights.append(f"'{feature}' has missing values in train data ({train_pct:.1f}%) but not in test data.")
            elif test_pct > 0:
                self.insights.append(f"'{feature}' has missing values in test data ({test_pct:.1f}%) but not in train data.")
            
            if train_pct > 20 or test_pct > 20:
                self.insights.append(f"âš ï¸� '{feature}' has a high percentage of missing values, which may require special handling.")
        
        # Skip visualizations if summary_only is True
        if summary_only:
            return
            
        # Visualize missing values
        print("\nMissing Values Visualization:")
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        features_with_missing = missing_stats_filtered['Feature'].tolist()
        train_missing_pct = missing_stats_filtered['Train_Missing_Pct'].values
        test_missing_pct = missing_stats_filtered['Test_Missing_Pct'].values
        
        # Create positions for the bars
        x = np.arange(len(features_with_missing))
        width = 0.35
        
        # Create the bar chart
        ax = plt.subplot(111)
        train_bars = ax.bar(x - width/2, train_missing_pct, width, label='Train', color=COLORS['primary'])
        test_bars = ax.bar(x + width/2, test_missing_pct, width, label='Test', color=COLORS['secondary'])
        
        # Add labels and formatting
        ax.set_title('Percentage of Missing Values by Feature', fontsize=14)
        ax.set_ylabel('Percentage Missing (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(features_with_missing, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend()
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
        
        add_labels(train_bars)
        add_labels(test_bars)
        
        plt.tight_layout()
        plt.show()
    
    def explore_target_distribution(self):
        """
        Explore the distribution of the target variable with multiple visualizations.
        """
        if self.train_data is None or self.target not in self.train_data.columns:
            print(f"â�Œ Target variable '{self.target}' not found in data. Please run load_datasets() first.")
            return
        
        print("\n" + "="*80)
        print(f"ğŸ�¯ TARGET VARIABLE ANALYSIS: {self.target}")
        print("="*80)
        
        # Create a figure with multiple plots
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. Histogram with KDE
        ax1 = fig.add_subplot(gs[0, 0:2])
        sns.histplot(self.train_data[self.target], kde=True, ax=ax1, color=COLORS['primary'])
        ax1.set_title(f'Distribution of {self.target}', fontsize=14)
        ax1.set_xlabel(self.target, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        
        # Add vertical lines for key statistics
        mean_val = self.train_data[self.target].mean()
        median_val = self.train_data[self.target].median()
        ax1.axvline(mean_val, color=COLORS['highlight'], linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color=COLORS['tertiary'], linestyle=':', linewidth=2, 
                   label=f'Median: {median_val:.2f}')
        ax1.legend()
        
        # 2. Boxplot
        ax2 = fig.add_subplot(gs[0, 2])
        sns.boxplot(y=self.train_data[self.target], ax=ax2, color=COLORS['primary'])
        ax2.set_title(f'Boxplot of {self.target}', fontsize=14)
        ax2.set_ylabel(self.target, fontsize=12)
        
        # 3. QQ plot to assess normality
        ax3 = fig.add_subplot(gs[1, 0])
        stats.probplot(self.train_data[self.target], plot=ax3)
        ax3.set_title('QQ Plot - Testing for Normality', fontsize=14)
        
        # 4. Log-transformed histogram
        ax4 = fig.add_subplot(gs[1, 1])
        log_target = np.log1p(self.train_data[self.target])
        sns.histplot(log_target, kde=True, ax=ax4, color=COLORS['secondary'])
        ax4.set_title(f'Log-Transformed {self.target} Distribution', fontsize=14)
        ax4.set_xlabel(f'Log({self.target} + 1)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        
        # 5. CDF plot
        ax5 = fig.add_subplot(gs[1, 2])
        sorted_data = np.sort(self.train_data[self.target])
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax5.plot(sorted_data, yvals, color=COLORS['tertiary'], linewidth=2)
        ax5.set_title('Cumulative Distribution Function', fontsize=14)
        ax5.set_xlabel(self.target, fontsize=12)
        ax5.set_ylabel('Cumulative Probability', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print key statistics
        print("\nğŸ“Š TARGET VARIABLE STATISTICS")
        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Min', 'Max', 'Range', 'Mean', 'Median', 'Mode', 
                         'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis'],
            'Value': [
                self.train_data[self.target].count(),
                self.train_data[self.target].min(),
                self.train_data[self.target].max(),
                self.train_data[self.target].max() - self.train_data[self.target].min(),
                self.train_data[self.target].mean(),
                self.train_data[self.target].median(),
                self.train_data[self.target].mode().iloc[0],
                self.train_data[self.target].std(),
                self.train_data[self.target].var(),
                self.train_data[self.target].skew(),
                self.train_data[self.target].kurt()
            ]
        })
        display(stats_df)
        
        # Normality tests
        print("\nğŸ“‹ NORMALITY TESTS")
        shapiro_test = stats.shapiro(self.train_data[self.target].sample(min(5000, len(self.train_data))))
        ks_test = stats.kstest(self.train_data[self.target], 'norm', args=(self.train_data[self.target].mean(), 
                                                                          self.train_data[self.target].std()))
        
        normality_df = pd.DataFrame({
            'Test': ['Shapiro-Wilk', 'Kolmogorov-Smirnov'],
            'Statistic': [shapiro_test.statistic, ks_test.statistic],
            'p-value': [shapiro_test.pvalue, ks_test.pvalue],
            'Interpretation': [
                'Not normally distributed' if shapiro_test.pvalue < 0.05 else 'Normally distributed',
                'Not normally distributed' if ks_test.pvalue < 0.05 else 'Normally distributed'
            ]
        })
        display(normality_df)
        
        # Add insights about target distribution
        skewness = self.train_data[self.target].skew()
        kurtosis = self.train_data[self.target].kurt()
        
        if abs(skewness) > 1:
            skew_direction = "positively" if skewness > 0 else "negatively"
            self.insights.append(f"Target variable is {skew_direction} skewed ({skewness:.2f}), suggesting a log transformation may be beneficial.")
        
        if shapiro_test.pvalue < 0.05:
            self.insights.append("Target variable is not normally distributed, which may impact some modeling assumptions.")
        
        if kurtosis > 3:
            self.insights.append(f"Target has high kurtosis ({kurtosis:.2f}), indicating heavy tails with potential outliers.")
            
        # Identify and report potential outliers
        q1 = self.train_data[self.target].quantile(0.25)
        q3 = self.train_data[self.target].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = self.train_data[(self.train_data[self.target] < lower_bound) | 
                                  (self.train_data[self.target] > upper_bound)]
        
        outlier_pct = len(outliers) / len(self.train_data) * 100
        if outlier_pct > 0:
            self.insights.append(f"{outlier_pct:.1f}% of target values are potential outliers, which should be considered during preprocessing.")
    
    def explore_categorical_features(self):
        """
        Comprehensive analysis of categorical features and their relationship with the target.
        """
        if self.train_data is None:
            print("â�Œ No data loaded. Please run load_datasets() first.")
            return
        
        if not self.categorical_features:
            print("â�Œ No categorical features found in the data.")
            return
            
        print("\n" + "="*80)
        print("ğŸ“Š CATEGORICAL FEATURES ANALYSIS")
        print("="*80)
        
        for feature in self.categorical_features:
            print(f"\nğŸ“‹ Analysis of '{feature}':")
            
            # Get value counts
            value_counts = self.train_data[feature].value_counts()
            value_pcts = self.train_data[feature].value_counts(normalize=True) * 100
            
            print(f"â€¢ {len(value_counts):,} unique values")
            print(f"â€¢ Most common: {value_counts.index[0]} ({value_counts.iloc[0]:,} occurrences, {value_pcts.iloc[0]:.1f}%)")
            
            if len(value_counts) > 10:
                print(f"â€¢ Top 10 values represent {value_pcts.iloc[:10].sum():.1f}% of the data")
            
            # Check for cardinality issues
            if len(value_counts) > 50:
                self.insights.append(f"'{feature}' has high cardinality ({len(value_counts)} categories), which may require encoding strategies beyond one-hot encoding.")
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            
            # 1. Bar plot of counts
            plt.subplot(2, 2, 1)
            top_n = min(10, len(value_counts))
            ax = value_counts.head(top_n).plot(kind='bar', color=COLORS['categorical'][:top_n])
            plt.title(f'Top {top_n} Categories by Count', fontsize=12)
            plt.xlabel(feature, fontsize=10)
            plt.ylabel('Count', fontsize=10)
            plt.xticks(rotation=45, ha='right')
            
            # Add count labels on bars
            for i, v in enumerate(value_counts.head(top_n)):
                ax.text(i, v + 0.1, f"{v:,}", ha='center', fontsize=9)
            
            # 2. Pie chart of percentages
            plt.subplot(2, 2, 2)
            if len(value_counts) > 6:
                # Group smaller categories into "Other"
                top_5 = value_pcts.head(5)
                other_pct = 100 - top_5.sum()
                pie_data = pd.concat([top_5, pd.Series([other_pct], index=['Other'])])
                pie_colors = COLORS['categorical'][:5] + [COLORS['neutral']]
            else:
                pie_data = value_pcts
                pie_colors = COLORS['categorical'][:len(value_pcts)]
                
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', 
                   colors=pie_colors, startangle=90, wedgeprops={'edgecolor': 'white'})
            plt.title(f'Percentage Distribution', fontsize=12)
            plt.axis('equal')
            
            # 3. Relationship with target (if target exists)
            if self.target in self.train_data.columns:
                plt.subplot(2, 1, 2)
                
                # For features with many categories, show only top categories
                if len(value_counts) > 15:
                    top_cats = value_counts.head(15).index
                    df_subset = self.train_data[self.train_data[feature].isin(top_cats)]
                    title_suffix = " (Top 15 Categories)"
                else:
                    df_subset = self.train_data
                    title_suffix = ""
                
                # Create boxplot
                sns.boxplot(x=feature, y=self.target, data=df_subset, palette=COLORS['categorical'])
                plt.title(f'Distribution of {self.target} by {feature}' + title_suffix, fontsize=12)
                plt.xlabel(feature, fontsize=10)
                plt.ylabel(self.target, fontsize=10)
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                
                # Get statistical insights
                try:
                    aov_result = stats.f_oneway(
                        *[self.train_data[self.train_data[feature] == cat][self.target].values 
                        for cat in value_counts.index[:min(10, len(value_counts))]]
                    )
                    
                    if aov_result.pvalue < 0.05:
                        print(f"â€¢ âœ“ Statistically significant relationship with target (p-value: {aov_result.pvalue:.6f})")
                        self.insights.append(f"'{feature}' shows a statistically significant relationship with the target variable.")
                    else:
                        print(f"â€¢ âœ— No statistically significant relationship with target (p-value: {aov_result.pvalue:.6f})")
                except Exception as e:
                    print(f"â€¢ âš ï¸� Could not perform statistical test: {str(e)}")
                
                # Calculate mean target value for each category
                category_means = self.train_data.groupby(feature)[self.target].mean().sort_values(ascending=False)
                print(f"â€¢ Categories with highest average {self.target}:")
                for cat, mean_val in category_means.head(3).items():
                    print(f"  - {cat}: {mean_val:.2f}")
                
                print(f"â€¢ Categories with lowest average {self.target}:")
                for cat, mean_val in category_means.tail(3).items():
                    print(f"  - {cat}: {mean_val:.2f}")
            
            plt.tight_layout()
            plt.show()

            # Calculate Cramer's V for categorical features if target is categorical
            if self.target in self.train_data.columns and self.train_data[self.target].nunique() < 20:
                cramers_v = self._calculate_cramers_v(feature, self.target)
                if cramers_v is not None:
                    print(f"â€¢ Cramer's V (association strength): {cramers_v:.4f} ({self._interpret_cramers_v(cramers_v)})")
    
    def explore_numerical_features(self):
        """
        Comprehensive analysis of numerical features and their relationship with the target.
        """
        if self.train_data is None:
            print("â�Œ No data loaded. Please run load_datasets() first.")
            return
        
        if not self.numerical_features:
            print("â�Œ No numerical features found in the data.")
            return
            
        print("\n" + "="*80)
        print("ğŸ“Š NUMERICAL FEATURES ANALYSIS")
        print("="*80)
        
        for feature in self.numerical_features:
            print(f"\nğŸ“‹ Analysis of '{feature}':")
            
            # Skip if the feature doesn't exist
            if feature not in self.train_data.columns:
                print(f"âš ï¸� Feature '{feature}' not found in data.")
                continue
                
            # Get basic statistics
            stats_series = self.train_data[feature].describe()
            skewness = self.train_data[feature].skew()
            kurtosis = self.train_data[feature].kurt()
            
            print(f"â€¢ Range: {stats_series['min']:.2f} to {stats_series['max']:.2f}")
            print(f"â€¢ Mean: {stats_series['mean']:.2f}, Median: {stats_series['50%']:.2f}")
            print(f"â€¢ Standard Deviation: {stats_series['std']:.2f}")
            print(f"â€¢ Skewness: {skewness:.2f} ({'Highly skewed' if abs(skewness) > 1 else 'Moderately skewed' if abs(skewness) > 0.5 else 'Approximately symmetric'})")
            print(f"â€¢ Kurtosis: {kurtosis:.2f} ({'Heavy-tailed' if kurtosis > 3 else 'Light-tailed'})")
            
            # Check for potential issues
            if self.train_data[feature].isnull().sum() > 0:
                print(f"â€¢ âš ï¸� Contains {self.train_data[feature].isnull().sum()} missing values")
            
            if np.isinf(self.train_data[feature]).any():
                print(f"â€¢ âš ï¸� Contains infinite values")
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            
            # 1. Histogram with KDE
            plt.subplot(2, 2, 1)
            sns.histplot(self.train_data[feature], kde=True, color=COLORS['primary'])
            plt.title(f'Distribution of {feature}', fontsize=12)
            plt.xlabel(feature, fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            
            # Add lines for mean and median
            plt.axvline(stats_series['mean'], color=COLORS['highlight'], linestyle='--', 
                       label=f'Mean: {stats_series["mean"]:.2f}')
            plt.axvline(stats_series['50%'], color=COLORS['tertiary'], linestyle=':', 
                       label=f'Median: {stats_series["50%"]:.2f}')
            plt.legend()
            
            # 2. Boxplot
            plt.subplot(2, 2, 2)
            sns.boxplot(y=self.train_data[feature], color=COLORS['primary'])
            plt.title(f'Boxplot of {feature}', fontsize=12)
            plt.ylabel(feature, fontsize=10)
            
            # 3. Relationship with target (if target exists)
            if self.target in self.train_data.columns:
                plt.subplot(2, 1, 2)
                
                # Detect if target is categorical or numerical
                if self.train_data[self.target].nunique() < 10:
                    # Target is categorical
                    sns.boxplot(x=self.target, y=feature, data=self.train_data, palette=COLORS['categorical'])
                    plt.title(f'Distribution of {feature} by {self.target}', fontsize=12)
                    plt.xlabel(self.target, fontsize=10)
                    plt.ylabel(feature, fontsize=10)
                else:
                    # Target is numerical - scatterplot
                    # Filter out any NaN or infinite values to avoid SVD errors
                    valid_data = self.train_data[[feature, self.target]].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # Check if we have enough valid data
                    if len(valid_data) > 10:
                        # Create scatter plot
                        plt.scatter(
                            valid_data[feature], 
                            valid_data[self.target],
                            alpha=0.5, 
                            color=COLORS['primary'],
                            edgecolors='none'
                        )
                        
                        # Calculate correlation
                        correlation = valid_data[[feature, self.target]].corr().iloc[0, 1]
                        
                        # Try to add trend line with error handling
                        try:
                            # Use a random sample if dataset is very large to avoid performance issues
                            if len(valid_data) > 10000:
                                sample_data = valid_data.sample(10000, random_state=42)
                            else:
                                sample_data = valid_data
                                
                            z = np.polyfit(sample_data[feature], sample_data[self.target], 1)
                            p = np.poly1d(z)
                            
                            plt.plot(
                                [valid_data[feature].min(), valid_data[feature].max()],
                                [p(valid_data[feature].min()), p(valid_data[feature].max())],
                                color=COLORS['alert'],
                                linestyle='--',
                                label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}'
                            )
                            plt.legend()
                        except Exception as e:
                            print(f"â€¢ âš ï¸� Could not fit trend line: {str(e)}")
                            # Add text annotation about correlation instead
                            plt.annotate(
                                f"Correlation: {correlation:.4f}",
                                xy=(0.05, 0.95),
                                xycoords='axes fraction',
                                fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                            )
                        
                        plt.title(f'Relationship between {feature} and {self.target}', fontsize=12)
                        plt.xlabel(feature, fontsize=10)
                        plt.ylabel(self.target, fontsize=10)
                        
                        print(f"â€¢ Correlation with target: {correlation:.4f} ({self._interpret_correlation(correlation)})")
                        
                        if abs(correlation) > 0.5:
                            self.insights.append(f"'{feature}' has a {self._interpret_correlation(correlation)} correlation ({correlation:.2f}) with the target variable.")
                    else:
                        plt.text(0.5, 0.5, "Insufficient valid data for visualization", 
                                ha='center', va='center', fontsize=12)
                        plt.axis('off')
                        
            plt.tight_layout()
            plt.show()
            
            # Check for outliers
            q1 = self.train_data[feature].quantile(0.25)
            q3 = self.train_data[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.train_data[(self.train_data[feature] < lower_bound) | 
                                     (self.train_data[feature] > upper_bound)]
            
            outlier_pct = len(outliers) / len(self.train_data) * 100
            print(f"â€¢ Outliers: {len(outliers):,} values ({outlier_pct:.2f}% of data)")
            
            if outlier_pct > 5:
                self.insights.append(f"'{feature}' has {outlier_pct:.1f}% outliers which may impact model performance.")
                
            # Add skewness insights
            if abs(skewness) > 1:
                self.insights.append(f"'{feature}' is highly skewed ({skewness:.2f}) and may benefit from transformation.")
                
                # Show log transformation effect
                if skewness > 1 and self.train_data[feature].min() >= 0:
                    print("\nğŸ“ˆ Log Transformation Effect:")
                    log_feature = np.log1p(self.train_data[feature])
                    log_skewness = log_feature.skew()
                    print(f"â€¢ Original skewness: {skewness:.2f}")
                    print(f"â€¢ After log(x+1) transform: {log_skewness:.2f}")
                    
                    if abs(log_skewness) < abs(skewness):
                        self.insights.append(f"A log transformation reduces the skewness of '{feature}' from {skewness:.2f} to {log_skewness:.2f}.")
    
    def correlation_analysis(self):
        """
        Analyze correlations between features and with the target.
        """
        if self.train_data is None:
            print("â�Œ No data loaded. Please run load_datasets() first.")
            return
        
        print("\n" + "="*80)
        print("ğŸ“Š CORRELATION ANALYSIS")
        print("="*80)
        
        # Create a dataframe with numeric and encoded categorical features
        analysis_df = self.train_data.copy()
        
        # Encode binary categorical variables for correlation analysis
        for col in self.categorical_features:
            if analysis_df[col].nunique() == 2:
                # Binary encoding
                analysis_df[col] = analysis_df[col].astype('category').cat.codes
        
        # Select numeric columns for correlation
        numeric_cols = analysis_df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("â�Œ Insufficient numerical features for correlation analysis.")
            return
            
        # Create correlation matrix
        corr_matrix = analysis_df[numeric_cols].corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            cmap='viridis',
            vmin=-1, vmax=1, 
            center=0,
            annot=True, 
            fmt='.2f',
            square=True,
            linewidths=.5
        )
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Display correlation with target (if exists)
        if self.target in numeric_cols:
            print("\nğŸ“‹ CORRELATION WITH TARGET VARIABLE")
            target_corr = corr_matrix[self.target].drop(self.target).sort_values(ascending=False)
            
            # Display as a horizontal bar chart
            plt.figure(figsize=(10, max(6, len(target_corr) * 0.3)))
            sns.barplot(
                x=target_corr.values,
                y=target_corr.index,
                palette=[COLORS['primary'] if x > 0 else COLORS['alert'] for x in target_corr.values]
            )
            plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
            plt.title(f'Correlation with {self.target}', fontsize=14)
            plt.xlabel('Correlation Coefficient', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Print key findings
            print("\nStrongest positive correlations:")
            for feature, corr in target_corr.head(3).items():
                print(f"â€¢ {feature}: {corr:.4f} ({self._interpret_correlation(corr)})")
                
            print("\nStrongest negative correlations:")
            for feature, corr in target_corr.tail(3).items():
                print(f"â€¢ {feature}: {corr:.4f} ({self._interpret_correlation(corr)})")
                
            # Add insights about correlation
            strong_corr_features = target_corr[abs(target_corr) > 0.5]
            if not strong_corr_features.empty:
                for feature, corr in strong_corr_features.items():
                    self.insights.append(f"'{feature}' has a {self._interpret_correlation(corr)} correlation ({corr:.2f}) with the target.")
            else:
                self.insights.append("No features show strong correlation (>0.5) with the target variable.")
                
        # Check for multicollinearity
        print("\nğŸ“‹ MULTICOLLINEARITY CHECK")
        high_corr_pairs = []
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                if col1 != self.target and col2 != self.target:  # Skip target variable
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((col1, col2, corr_val))
        
        if high_corr_pairs:
            print("Highly correlated feature pairs (|r| > 0.7):")
            for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"â€¢ {col1} & {col2}: {corr:.4f}")
                self.insights.append(f"High correlation ({corr:.2f}) between '{col1}' and '{col2}' suggests multicollinearity.")
        else:
            print("No high multicollinearity detected (no pairs with |r| > 0.7).")
    
    def train_test_distribution_comparison(self):
        """
        Compare feature distributions between train and test sets to detect distribution shifts.
        """
        if self.train_data is None or self.test_data is None:
            print("â�Œ No data loaded. Please run load_datasets() first.")
            return
        
        print("\n" + "="*80)
        print("ğŸ“Š TRAIN VS TEST DISTRIBUTION COMPARISON")
        print("="*80)
        
        print("Comparing feature distributions to detect potential data leakage or distribution shifts...")
        
        # Compare numerical features
        for feature in self.numerical_features:
            if feature in self.test_data.columns:  # Ensure feature exists in test data
                print(f"\nğŸ“‹ '{feature}' Distribution Comparison:")
                
                # Get basic stats for comparison
                train_mean = self.train_data[feature].mean()
                test_mean = self.test_data[feature].mean()
                train_std = self.train_data[feature].std()
                test_std = self.test_data[feature].std()
                
                print(f"â€¢ Train mean: {train_mean:.4f}, Test mean: {test_mean:.4f}")
                print(f"â€¢ Train std: {train_std:.4f}, Test std: {test_std:.4f}")
                
                # Create visualization
                plt.figure(figsize=(12, 5))
                
                # KDE plot for distribution comparison
                plt.subplot(1, 2, 1)
                sns.kdeplot(self.train_data[feature], label='Train', color=COLORS['primary'])
                sns.kdeplot(self.test_data[feature], label='Test', color=COLORS['secondary'])
                plt.title(f'Distribution of {feature}', fontsize=12)
                plt.xlabel(feature, fontsize=10)
                plt.ylabel('Density', fontsize=10)
                plt.legend()
                
                # Run statistical test with error handling
                try:
                    # Get samples for statistical test (limit for performance)
                    train_sample = self.train_data[feature].sample(min(5000, len(self.train_data)), random_state=42)
                    test_sample = self.test_data[feature].sample(min(5000, len(self.test_data)), random_state=42)
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pval = stats.ks_2samp(train_sample, test_sample)
                    print(f"â€¢ Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
                    
                    if ks_pval < 0.05:
                        print("â€¢ âš ï¸� Significant difference detected between train and test distributions")
                        self.insights.append(f"'{feature}' shows significantly different distributions in train and test sets (p={ks_pval:.4f}), which may impact model generalization.")
                    else:
                        print("â€¢ âœ“ No significant difference between train and test distributions")
                        
                    # QQ plot to compare distributions
                    plt.subplot(1, 2, 2)
                    
                    # Sort samples
                    train_sample_sorted = np.sort(train_sample)
                    test_sample_sorted = np.sort(test_sample)
                    
                    # If samples have different sizes, interpolate to match
                    if len(train_sample_sorted) != len(test_sample_sorted):
                        if len(train_sample_sorted) > len(test_sample_sorted):
                            # Interpolate test sample to match train sample size
                            train_quantiles = np.linspace(0, 1, len(train_sample_sorted))
                            test_quantiles = np.linspace(0, 1, len(test_sample_sorted))
                            test_interpolated = np.interp(train_quantiles, test_quantiles, test_sample_sorted)
                            x_data, y_data = train_sample_sorted, test_interpolated
                        else:
                            # Interpolate train sample to match test sample size
                            train_quantiles = np.linspace(0, 1, len(train_sample_sorted))
                            test_quantiles = np.linspace(0, 1, len(test_sample_sorted))
                            train_interpolated = np.interp(test_quantiles, train_quantiles, train_sample_sorted)
                            x_data, y_data = train_interpolated, test_sample_sorted
                    else:
                        x_data, y_data = train_sample_sorted, test_sample_sorted
                    
                    # Plot QQ
                    plt.scatter(x_data, y_data, alpha=0.5, color=COLORS['primary'])
                    
                    # Add diagonal line
                    min_val = min(x_data.min(), y_data.min())
                    max_val = max(x_data.max(), y_data.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                    
                    plt.title(f'QQ Plot: Train vs Test', fontsize=12)
                    plt.xlabel('Train Quantiles', fontsize=10)
                    plt.ylabel('Test Quantiles', fontsize=10)
                    
                except Exception as e:
                    plt.subplot(1, 2, 2)
                    plt.text(0.5, 0.5, f"Could not create QQ plot:\n{str(e)}", 
                            ha='center', va='center', fontsize=10)
                    plt.axis('off')
                    print(f"â€¢ âš ï¸� Error in statistical comparison: {str(e)}")
                
                plt.tight_layout()
                plt.show()
                
        # Compare categorical features
        for feature in self.categorical_features:
            if feature in self.test_data.columns:  # Ensure feature exists in test data
                print(f"\nğŸ“‹ '{feature}' Distribution Comparison:")
                
                # Get category proportions
                train_props = self.train_data[feature].value_counts(normalize=True)
                test_props = self.test_data[feature].value_counts(normalize=True)
                
                # Merge proportions
                props_df = pd.DataFrame({
                    'Train': train_props,
                    'Test': test_props
                }).fillna(0)
                
                # Keep only top 10 categories if there are many
                if len(props_df) > 10:
                    # Get top categories by combined proportion
                    props_df['Combined'] = props_df['Train'] + props_df['Test']
                    top_cats = props_df.sort_values('Combined', ascending=False).head(10).index
                    props_df = props_df.loc[top_cats, ['Train', 'Test']]
                
                # Create visualization
                plt.figure(figsize=(14, 6))
                
                # Proportion comparison
                plt.subplot(1, 2, 1)
                props_df.plot(kind='bar', ax=plt.gca(), color=[COLORS['primary'], COLORS['secondary']])
                plt.title(f'Category Distribution: {feature}', fontsize=12)
                plt.xlabel('Category', fontsize=10)
                plt.ylabel('Proportion', fontsize=10)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Dataset')
                
                # Scatter plot of proportions
                plt.subplot(1, 2, 2)
                plt.scatter(props_df['Train'], props_df['Test'], alpha=0.7, color=COLORS['primary'])
                
                # Add diagonal line for reference
                max_prop = max(props_df['Train'].max(), props_df['Test'].max())
                plt.plot([0, max_prop], [0, max_prop], 'r--')
                
                # Add category labels
                for i, category in enumerate(props_df.index):
                    plt.annotate(
                        category, 
                        (props_df['Train'].iloc[i], props_df['Test'].iloc[i]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha='left',
                        fontsize=8
                    )
                
                plt.title(f'Train vs Test Proportions', fontsize=12)
                plt.xlabel('Train Proportion', fontsize=10)
                plt.ylabel('Test Proportion', fontsize=10)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
                # Calculate chi-square test to compare distributions
                try:
                    # Create contingency table with actual counts
                    categories = list(set(train_props.index) | set(test_props.index))
                    train_counts = [self.train_data[feature].value_counts().get(cat, 0) for cat in categories]
                    test_counts = [self.test_data[feature].value_counts().get(cat, 0) for cat in categories]
                    
                    # Only perform test if we have sufficient counts
                    min_count = min([c for c in train_counts + test_counts if c > 0])
                    if min_count >= 5 and len(categories) > 1:
                        # Create contingency table
                        contingency = np.array([train_counts, test_counts])
                        chi2_stat, chi2_pval = stats.chi2_contingency(contingency)[:2]
                        
                        print(f"â€¢ Chi-square test: statistic={chi2_stat:.4f}, p-value={chi2_pval:.4f}")
                        
                        if chi2_pval < 0.05:
                            print("â€¢ âš ï¸� Significant difference detected between train and test distributions")
                            self.insights.append(f"'{feature}' shows significantly different distributions in train and test sets (p={chi2_pval:.4f}).")
                        else:
                            print("â€¢ âœ“ No significant difference between train and test distributions")
                    else:
                        print("â€¢ âš ï¸� Insufficient data for statistical testing")
                except Exception as e:
                    print(f"â€¢ âš ï¸� Error in statistical comparison: {str(e)}")
    
    def summarize_insights(self):
        """
        Summarize all insights gathered during the analysis.
        """
        if not self.insights:
            print("No insights gathered yet. Run some analysis functions first.")
            return
        
        print("\n" + "="*80)
        print("ğŸ“‹ KEY INSIGHTS FROM DATA EXPLORATION")
        print("="*80)
        
        # Group insights by category
        data_quality = []
        feature_insights = []
        target_insights = []
        modeling_suggestions = []
        
        for insight in self.insights:
            if any(kw in insight.lower() for kw in ['missing', 'duplicate', 'outlier', 'distribution']):
                data_quality.append(insight)
            elif any(kw in insight.lower() for kw in ['target']):
                target_insights.append(insight)
            elif any(kw in insight.lower() for kw in ['transform', 'encoding', 'correlation', 'multicollinearity']):
                modeling_suggestions.append(insight)
            else:
                feature_insights.append(insight)
        
        # Print insights by category
        if data_quality:
            print("\nğŸ“Š DATA QUALITY INSIGHTS:")
            for i, insight in enumerate(data_quality, 1):
                print(f"{i}. {insight}")
        
        if target_insights:
            print("\nğŸ�¯ TARGET VARIABLE INSIGHTS:")
            for i, insight in enumerate(target_insights, 1):
                print(f"{i}. {insight}")
        
        if feature_insights:
            print("\nğŸ“‹ FEATURE INSIGHTS:")
            for i, insight in enumerate(feature_insights, 1):
                print(f"{i}. {insight}")
        
        if modeling_suggestions:
            print("\nğŸ”§ MODELING SUGGESTIONS:")
            for i, insight in enumerate(modeling_suggestions, 1):
                print(f"{i}. {insight}")
    
    def _calculate_cramers_v(self, feat1, feat2):
        """
        Calculate Cramer's V statistic for categorical-categorical association.
        
        Parameters:
        -----------
        feat1 : str
            First feature name
        feat2 : str
            Second feature name
            
        Returns:
        --------
        float
            Cramer's V statistic
        """
        try:
            contingency = pd.crosstab(self.train_data[feat1], self.train_data[feat2])
            chi2 = stats.chi2_contingency(contingency)[0]
            n = contingency.sum().sum()
            phi2 = chi2 / n
            r, k = contingency.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        except Exception:
            return None
    
    def _interpret_cramers_v(self, v):
        """
        Interpret the strength of Cramer's V statistic.
        
        Parameters:
        -----------
        v : float
            Cramer's V statistic
            
        Returns:
        --------
        str
            Interpretation of association strength
        """
        if v < 0.1:
            return "negligible association"
        elif v < 0.2:
            return "weak association"
        elif v < 0.3:
            return "moderate association"
        elif v < 0.4:
            return "relatively strong association"
        elif v < 0.5:
            return "strong association"
        else:
            return "very strong association"
    
    def _interpret_correlation(self, corr):
        """
        Interpret the strength of a correlation coefficient.
        
        Parameters:
        -----------
        corr : float
            Correlation coefficient
            
        Returns:
        --------
        str
            Interpretation of correlation strength
        """
        corr = abs(corr)
        if corr < 0.1:
            return "negligible"
        elif corr < 0.3:
            return "weak"
        elif corr < 0.5:
            return "moderate"
        elif corr < 0.7:
            return "strong"
        else:
            return "very strong"
    
    def feature_importance_preview(self):
        """
        Preview feature importance using a simple RandomForest model.
        """
        if self.train_data is None or self.target not in self.train_data.columns:
            print(f"â�Œ Target variable '{self.target}' not found. Please run load_datasets() first.")
            return
        
        print("\n" + "="*80)
        print("ğŸ”� FEATURE IMPORTANCE PREVIEW")
        print("="*80)
        
        print("Training a simple RandomForest model to estimate feature importance...")
        
        try:
            # Prepare the data - handle missing values first
            X = self.train_data.drop(self.target, axis=1)
            y = self.train_data[self.target]
            
            # Handle categorical features
            X = pd.get_dummies(X, columns=self.categorical_features, drop_first=True)
            
            # Keep track of original feature names
            feature_mapping = {}
            for col in X.columns:
                # Extract original feature name from dummy variables
                if '_' in col:
                    original_feature = col.split('_')[0]
                    feature_mapping[col] = original_feature
                else:
                    feature_mapping[col] = col
            
            # Train a simple RandomForest model
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Train on data, handling any remaining issues
            X_clean = X.select_dtypes(include=['number']).fillna(0)  # Use only numeric columns
            y_clean = y.fillna(y.mean())  # Handle any missing target values
            
            if X_clean.shape[1] == 0:
                print("âš ï¸� No numeric features available for importance analysis after preprocessing.")
                return
            
            rf_model.fit(X_clean, y_clean)
            
            # Get feature importances
            importances = rf_model.feature_importances_
            
            # Map to original feature names
            orig_features = [feature_mapping.get(col, col) for col in X_clean.columns]
            
            # Group importances by original feature
            grouped_importances = {}
            for i, feat in enumerate(orig_features):
                if feat in grouped_importances:
                    grouped_importances[feat] += importances[i]
                else:
                    grouped_importances[feat] = importances[i]
            
            # Convert to DataFrame and sort
            importance_df = pd.DataFrame({
                'Feature': list(grouped_importances.keys()),
                'Importance': list(grouped_importances.values())
            }).sort_values('Importance', ascending=False)
            
            # Display top 15 features (or fewer if there aren't that many)
            top_n = min(15, len(importance_df))
            print(f"\nğŸ“‹ TOP {top_n} FEATURES BY IMPORTANCE")
            display(importance_df.head(top_n))
            
            # Visualize feature importance
            plt.figure(figsize=(12, max(6, len(importance_df.head(top_n)) * 0.3)))
            sns.barplot(
                x='Importance',
                y='Feature',
                data=importance_df.head(top_n),
                palette=sns.color_palette('viridis', len(importance_df.head(top_n)))
            )
            plt.title('Feature Importance from RandomForest Model', fontsize=14)
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Add insights about feature importance
            top_features = importance_df.head(5)['Feature'].tolist()
            self.insights.append(f"Top predictive features are likely: {', '.join(top_features)}")
            
            # Check for features with very low importance
            low_importance = importance_df[importance_df['Importance'] < 0.01]
            if len(low_importance) > 0:
                low_features = low_importance['Feature'].tolist()
                if len(low_features) > 5:
                    low_count = len(low_features)
                    low_features = low_features[:5]
                    self.insights.append(f"{low_count} features show very low importance (<1%), including: {', '.join(low_features)}...")
                else:
                    self.insights.append(f"Features with very low importance (<1%): {', '.join(low_features)}")
        
        except Exception as e:
            print(f"â�Œ Error in feature importance calculation: {str(e)}")
            print("This is just a preview and not critical for the analysis.")

# Initialize the explorer
explorer = BagPriceExplorer()

# Load the datasets
explorer.load_datasets()

# Run specific analyses you're interested in
explorer.summarize_data()
explorer.explore_target_distribution()
explorer.explore_categorical_features()
explorer.explore_numerical_features()
explorer.correlation_analysis()
explorer.train_test_distribution_comparison()
explorer.feature_importance_preview()

# Get a summary of all insights
explorer.summarize_insights()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge, SGDRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_regression, VarianceThreshold
from sklearn.cluster import KMeans
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import os
import warnings
import time
import joblib
from tqdm.notebook import tqdm
from functools import partial
import copy
import itertools
import shap
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class EnhancedBagPriceModeler:
    """
    Advanced modeling pipeline with sophisticated feature engineering and ensemble techniques
    for the Student Bag Price Prediction competition.
    """
    
    def __init__(self, base_path='/kaggle/input/', output_path='/kaggle/working/'):
        """Initialize the advanced modeler with paths and settings."""
        self.base_path = base_path
        self.output_path = output_path
        self.train_data = None
        self.test_data = None
        self.original_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.test_ids = None
        self.categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                                    'Waterproof', 'Style', 'Color']
        self.numerical_features = ['Weight Capacity (kg)', 'Compartments']
        self.target = 'Price'
        self.model_results = []
        self.predictions = None
        self.submission = None
        self.best_features = None
        self.best_models = {}
        self.feature_importance = None
        self.target_encoding_maps = {}
        self.feature_metadata = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Create subdirectories for organization
        self.models_path = os.path.join(output_path, 'models')
        self.plots_path = os.path.join(output_path, 'plots')
        self.features_path = os.path.join(output_path, 'features')
        
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(self.features_path, exist_ok=True)
    
    def load_data(self, train_data=None, test_data=None, original_data=None):
        """Load or set datasets for modeling."""
        # If data is provided directly, use it
        if train_data is not None and test_data is not None:
            self.train_data = train_data.copy()
            self.test_data = test_data.copy()
            if original_data is not None:
                self.original_data = original_data.copy()
            
            print(f"âœ… Using provided datasets")
            print(f"Train data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            if self.original_data is not None:
                print(f"Original data shape: {self.original_data.shape}")
            
            return self
        
        # Otherwise, load from files
        try:
            print("ğŸ”� Loading datasets from files...")
            
            # Define paths
            competition_path = os.path.join(self.base_path, 'playground-series-s5e2')
            original_path = os.path.join(self.base_path, 'student-bag-price-prediction-dataset')
            
            # Load the datasets
            self.train_data = pd.read_csv(os.path.join(competition_path, 'train.csv'))
            self.test_data = pd.read_csv(os.path.join(competition_path, 'test.csv'))
            
            try:
                self.original_data = pd.read_csv(os.path.join(original_path, 
                                                'Noisy_Student_Bag_Price_Prediction_Dataset.csv'))
            except FileNotFoundError:
                print("âš ï¸� Original dataset not found.")
                self.original_data = None
            
            print(f"âœ… Train data: {self.train_data.shape}")
            print(f"âœ… Test data: {self.test_data.shape}")
            if self.original_data is not None:
                print(f"âœ… Original data: {self.original_data.shape}")
            
            return self
            
        except Exception as e:
            print(f"â�Œ Error loading datasets: {str(e)}")
            return self
    
    def analyze_data_structure(self):
        """Perform deep structural analysis of the dataset to guide feature engineering strategy."""
        print("\n" + "="*80)
        print("ğŸ”� DATASET STRUCTURE ANALYSIS")
        print("="*80)
        
        if self.train_data is None:
            print("â�Œ No data loaded. Please load data first.")
            return self
        
        # Analyze data types and basic stats
        print("\nğŸ“Š Data Types and Basic Statistics:")
        
        data_types = self.train_data.dtypes
        missing_values = self.train_data.isnull().sum()
        unique_counts = self.train_data.nunique()
        
        summary = pd.DataFrame({
            'Data Type': data_types,
            'Unique Values': unique_counts,
            'Missing Values': missing_values,
            'Missing Percentage': (missing_values / len(self.train_data)) * 100
        })
        
        print(summary)
        
        # Analyze categorical cardinality
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print("\nğŸ“Š Categorical Feature Cardinality:")
            
            cat_cardinality = {}
            for col in categorical_cols:
                value_counts = self.train_data[col].value_counts()
                cat_cardinality[col] = {
                    'cardinality': len(value_counts),
                    'top_values': value_counts.head(5).to_dict(),
                    'rare_values': sum(value_counts < len(self.train_data) * 0.01),
                    'rare_percentage': sum(value_counts[value_counts < len(self.train_data) * 0.01]) / len(self.train_data) * 100
                }
                
                # Store metadata for feature engineering
                self.feature_metadata[col] = {
                    'type': 'categorical',
                    'cardinality': len(value_counts),
                    'rare_threshold': len(self.train_data) * 0.01,
                    'rare_values': [val for val, count in value_counts.items() if count < len(self.train_data) * 0.01],
                    'frequent_values': [val for val, count in value_counts.items() if count >= len(self.train_data) * 0.01],
                }
            
            for col, stats in cat_cardinality.items():
                print(f"\n{col}:")
                print(f"  Unique values: {stats['cardinality']}")
                print(f"  Rare values (< 1% occurrence): {stats['rare_values']} ({stats['rare_percentage']:.2f}% of data)")
                print("  Top values:")
                for val, count in stats['top_values'].items():
                    print(f"    - {val}: {count} ({count/len(self.train_data)*100:.2f}%)")
        
        # Analyze numerical feature distributions
        numerical_cols = self.train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'id' and col != self.target]
        
        if numerical_cols:
            print("\nğŸ“Š Numerical Feature Statistics:")
            
            num_stats = self.train_data[numerical_cols].describe().T
            num_stats['skewness'] = self.train_data[numerical_cols].skew()
            num_stats['kurtosis'] = self.train_data[numerical_cols].kurtosis()
            
            print(num_stats)
            
            # Store metadata for numerical features
            for col in numerical_cols:
                self.feature_metadata[col] = {
                    'type': 'numerical',
                    'min': self.train_data[col].min(),
                    'max': self.train_data[col].max(),
                    'median': self.train_data[col].median(),
                    'mean': self.train_data[col].mean(),
                    'std': self.train_data[col].std(),
                    'skewness': self.train_data[col].skew(),
                    'kurtosis': self.train_data[col].kurtosis(),
                    'is_highly_skewed': abs(self.train_data[col].skew()) > 1.0,
                    'zero_count': (self.train_data[col] == 0).sum(),
                    'zero_percentage': (self.train_data[col] == 0).mean() * 100
                }
        
        # Analyze potential date features
        date_cols = []
        for col in self.train_data.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                date_cols.append(col)
                
        if date_cols:
            print("\nğŸ“Š Potential Date Features:")
            for col in date_cols:
                print(f"\n{col}:")
                try:
                    # Try to convert to datetime
                    dt_series = pd.to_datetime(self.train_data[col], errors='coerce')
                    valid_dates = dt_series.notna().sum()
                    
                    if valid_dates > 0:
                        print(f"  Valid dates: {valid_dates} ({valid_dates/len(self.train_data)*100:.2f}%)")
                        print(f"  Min date: {dt_series.min()}")
                        print(f"  Max date: {dt_series.max()}")
                        
                        # Store metadata for temporal features
                        self.feature_metadata[col] = {
                            'type': 'temporal',
                            'valid_dates': valid_dates,
                            'min_date': dt_series.min(),
                            'max_date': dt_series.max(),
                            'range_days': (dt_series.max() - dt_series.min()).days if pd.notna(dt_series.max()) and pd.notna(dt_series.min()) else None
                        }
                    else:
                        print("  Not a valid date column")
                except Exception as e:
                    print(f"  Error parsing dates: {str(e)}")
        
        # Calculate and store feature relationships
        self._analyze_feature_relationships()
        
        # Identify potential data leakage
        self._check_for_data_leakage()
        
        return self
    
    def _analyze_feature_relationships(self):
        """Analyze relationships between features to guide feature engineering."""
        if self.train_data is None:
            return
        
        print("\nğŸ“Š Analyzing Feature Relationships:")
        
        # Calculate correlations for numerical features
        numerical_cols = self.train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'id']
        
        if len(numerical_cols) > 1:
            correlation_matrix = self.train_data[numerical_cols].corr()
            
            # Save correlation matrix visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_path, 'correlation_matrix.png'))
            plt.close()
            
            # Find highly correlated pairs
            corr_pairs = []
            for i in range(len(numerical_cols)):
                for j in range(i+1, len(numerical_cols)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.5:  # Adjust threshold as needed
                        corr_pairs.append((numerical_cols[i], numerical_cols[j], corr))
            
            if corr_pairs:
                print("\nHighly correlated feature pairs:")
                for feat1, feat2, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                    print(f"  â€¢ {feat1} and {feat2}: correlation = {corr:.3f}")
                    
                    # Store relationship metadata
                    if feat1 not in self.feature_metadata:
                        self.feature_metadata[feat1] = {'type': 'numerical', 'relationships': []}
                    if 'relationships' not in self.feature_metadata[feat1]:
                        self.feature_metadata[feat1]['relationships'] = []
                    
                    if feat2 not in self.feature_metadata:
                        self.feature_metadata[feat2] = {'type': 'numerical', 'relationships': []}
                    if 'relationships' not in self.feature_metadata[feat2]:
                        self.feature_metadata[feat2]['relationships'] = []
                    
                    self.feature_metadata[feat1]['relationships'].append({
                        'feature': feat2,
                        'correlation': corr,
                        'type': 'linear' if abs(corr) > 0.7 else 'moderate'
                    })
                    
                    self.feature_metadata[feat2]['relationships'].append({
                        'feature': feat1,
                        'correlation': corr,
                        'type': 'linear' if abs(corr) > 0.7 else 'moderate'
                    })
            
            # Create hierarchical clustering of features based on correlation
            if len(numerical_cols) > 2:
                try:
                    # Calculate the correlation distance matrix
                    corr_condensed = pdist(correlation_matrix, metric='correlation')
                    
                    # Perform hierarchical clustering
                    z = hierarchy.linkage(corr_condensed, method='average')
                    
                    # Plot dendrogram
                    plt.figure(figsize=(12, 8))
                    hierarchy.dendrogram(
                        z,
                        labels=correlation_matrix.index,
                        orientation='right',
                        leaf_font_size=10
                    )
                    plt.title('Hierarchical Clustering of Features')
                    plt.xlabel('Distance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.plots_path, 'feature_dendrogram.png'))
                    plt.close()
                    
                    # Find clusters at a certain distance threshold
                    cluster_ids = hierarchy.fcluster(z, t=0.5, criterion='distance')
                    clusters = {}
                    for i, cluster_id in enumerate(cluster_ids):
                        if cluster_id not in clusters:
                            clusters[cluster_id] = []
                        clusters[cluster_id].append(numerical_cols[i])
                    
                    # Store cluster information for feature engineering
                    for cluster_id, features in clusters.items():
                        if len(features) > 1:
                            print(f"\nFeature cluster {cluster_id}: {features}")
                            for feat in features:
                                if feat not in self.feature_metadata:
                                    self.feature_metadata[feat] = {'type': 'numerical'}
                                self.feature_metadata[feat]['cluster_id'] = cluster_id
                                self.feature_metadata[feat]['cluster_members'] = [f for f in features if f != feat]
                except Exception as e:
                    print(f"Error in hierarchical clustering: {str(e)}")
    
    def _check_for_data_leakage(self):
        """Check for potential data leakage issues in the dataset."""
        if self.train_data is None or self.target not in self.train_data.columns:
            return
        
        print("\nğŸ”� Checking for Potential Data Leakage:")
        
        # Check for suspiciously high correlations with target
        numerical_cols = self.train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'id' and col != self.target]
        
        if numerical_cols:
            target_correlations = self.train_data[numerical_cols + [self.target]].corr()[self.target].drop(self.target)
            
            # Check for very high correlations that might indicate leakage
            high_corr_features = target_correlations[abs(target_correlations) > 0.85].index.tolist()
            
            if high_corr_features:
                print("\nâš ï¸� Features with suspiciously high correlation to target:")
                for feat in high_corr_features:
                    corr = target_correlations[feat]
                    print(f"  â€¢ {feat}: correlation = {corr:.3f}")
                    
                    # Flag in metadata
                    if feat in self.feature_metadata:
                        self.feature_metadata[feat]['potential_leakage'] = True
                        self.feature_metadata[feat]['target_correlation'] = corr
        
        # Check for duplicate or near-duplicate rows
        duplicate_count = self.train_data.duplicated().sum()
        if duplicate_count > 0:
            print(f"\nâš ï¸� Found {duplicate_count} duplicate rows in training data")
        
        # Check for unusual patterns in IDs
        if 'id' in self.train_data.columns:
            try:
                # Check if IDs are sequential
                ids = self.train_data['id'].sort_values().values
                if np.all(np.diff(ids) == 1):
                    print("\nğŸ”� IDs appear to be sequential")
                
                # Check if target has pattern by ID
                if self.target in self.train_data.columns:
                    id_target_corr = self.train_data[['id', self.target]].corr().iloc[0, 1]
                    if abs(id_target_corr) > 0.1:
                        print(f"\nâš ï¸� ID has correlation of {id_target_corr:.3f} with target - potential data ordering issue")
            except Exception as e:
                print(f"Error analyzing IDs: {str(e)}")
    
    def analyze_target_variable(self):
        """
        Perform deep analysis of the target variable to guide transformation strategy.
        """
        print("\n" + "="*80)
        print("ğŸ”� TARGET VARIABLE ANALYSIS")
        print("="*80)
        
        if self.train_data is None or self.target not in self.train_data.columns:
            print("â�Œ Target variable not found. Please load data first.")
            return self
        
        # Analyze target distribution
        target_values = self.train_data[self.target].values
        
        # Basic statistics
        print(f"\nTarget Variable: {self.target}")
        print(f"Min: {np.min(target_values):.4f}")
        print(f"Max: {np.max(target_values):.4f}")
        print(f"Mean: {np.mean(target_values):.4f}")
        print(f"Median: {np.median(target_values):.4f}")
        print(f"Std Dev: {np.std(target_values):.4f}")
        print(f"Skewness: {stats.skew(target_values):.4f}")
        print(f"Kurtosis: {stats.kurtosis(target_values):.4f}")
        
        # Check for outliers using IQR method
        q1 = np.percentile(target_values, 25)
        q3 = np.percentile(target_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = target_values[(target_values < lower_bound) | (target_values > upper_bound)]
        print(f"Outliers: {len(outliers)} ({len(outliers)/len(target_values)*100:.2f}% of data)")
        
        # Advanced distribution analysis
        # 1. Test for normality
        _, p_value = stats.shapiro(target_values[:5000])  # Limit to 5000 samples for Shapiro-Wilk test
        print(f"Shapiro-Wilk p-value: {p_value:.6f} ({'Normal' if p_value > 0.05 else 'Not normal'})")
        
        # 2. Test for different distributions
        distributions = ['norm', 'lognorm', 'gamma', 'expon', 'weibull_min']
        dist_results = {}
        
        for dist_name in distributions:
            try:
                # Fit distribution
                distribution = getattr(stats, dist_name)
                params = distribution.fit(target_values)
                
                # Calculate Kolmogorov-Smirnov statistic
                _, p_value = stats.kstest(target_values, dist_name, params)
                dist_results[dist_name] = p_value
            except Exception as e:
                print(f"Error fitting {dist_name}: {str(e)}")
        
        # Find best-fit distribution
        best_dist = max(dist_results.items(), key=lambda x: x[1]) if dist_results else None
        print("\nDistribution Fitting Results:")
        for dist_name, p_value in dist_results.items():
            print(f"  {dist_name}: p-value = {p_value:.6f}")
        
        if best_dist:
            print(f"\nBest fitting distribution: {best_dist[0]} (p-value: {best_dist[1]:.6f})")
        
        # Visualization
        plt.figure(figsize=(15, 12))
        
        # Original distribution
        plt.subplot(3, 2, 1)
        sns.histplot(target_values, kde=True)
        plt.title("Original Target Distribution")
        
        # Log transformation
        plt.subplot(3, 2, 2)
        log_target = np.log1p(target_values)
        sns.histplot(log_target, kde=True)
        plt.title(f"Log-Transformed Target (Skew: {stats.skew(log_target):.4f})")
        
        # Square root transformation
        plt.subplot(3, 2, 3)
        sqrt_target = np.sqrt(target_values)
        sns.histplot(sqrt_target, kde=True)
        plt.title(f"Sqrt-Transformed Target (Skew: {stats.skew(sqrt_target):.4f})")
        
        # Box-Cox transformation
        plt.subplot(3, 2, 4)
        try:
            boxcox_target, lambda_param = stats.boxcox(target_values)
            sns.histplot(boxcox_target, kde=True)
            plt.title(f"Box-Cox-Transformed Target (Î»={lambda_param:.4f}, Skew: {stats.skew(boxcox_target):.4f})")
            print(f"Box-Cox Lambda: {lambda_param:.4f}")
            # We'll store lambda for later use
            self.boxcox_lambda = lambda_param
        except Exception as e:
            plt.title("Box-Cox Transformation Failed")
            print(f"Box-Cox transformation failed: {str(e)}")
            self.boxcox_lambda = None
        
        # Yeo-Johnson transformation
        plt.subplot(3, 2, 5)
        try:
            pt = PowerTransformer(method='yeo-johnson')
            yj_target = pt.fit_transform(target_values.reshape(-1, 1)).flatten()
            sns.histplot(yj_target, kde=True)
            plt.title(f"Yeo-Johnson-Transformed Target (Skew: {stats.skew(yj_target):.4f})")
            self.yj_transformer = pt
        except Exception as e:
            plt.title("Yeo-Johnson Transformation Failed")
            print(f"Yeo-Johnson transformation failed: {str(e)}")
            self.yj_transformer = None
        
        # Quantile transformation
        plt.subplot(3, 2, 6)
        try:
            qt = QuantileTransformer(n_quantiles=min(1000, len(target_values)), output_distribution='normal')
            q_target = qt.fit_transform(target_values.reshape(-1, 1)).flatten()
            sns.histplot(q_target, kde=True)
            plt.title(f"Quantile-Transformed Target (Skew: {stats.skew(q_target):.4f})")
            self.q_transformer = qt
        except Exception as e:
            plt.title("Quantile Transformation Failed")
            print(f"Quantile transformation failed: {str(e)}")
            self.q_transformer = None
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'target_transformations.png'))
        plt.close()
        
        # Determine best transformation strategy based on skewness
        skewness_original = stats.skew(target_values)
        skewness_log = stats.skew(log_target)
        skewness_sqrt = stats.skew(sqrt_target)
        
        transformations = [
            ("Original", skewness_original),
            ("Log", skewness_log),
            ("Square Root", skewness_sqrt)
        ]
        
        if self.boxcox_lambda is not None:
            transformations.append(("Box-Cox", stats.skew(boxcox_target)))
        
        if hasattr(self, 'yj_transformer') and self.yj_transformer is not None:
            transformations.append(("Yeo-Johnson", stats.skew(yj_target)))
        
        if hasattr(self, 'q_transformer') and self.q_transformer is not None:
            transformations.append(("Quantile", stats.skew(q_target)))
        
        best_transform = min(transformations, key=lambda x: abs(x[1]))
        print(f"\nBest transformation based on skewness: {best_transform[0]} (Skewness: {best_transform[1]:.4f})")
        
        # Additional test: Check normality after transformation
        transformation_normality = []
        for name, _ in transformations:
            if name == "Original":
                transformed = target_values
            elif name == "Log":
                transformed = log_target
            elif name == "Square Root":
                transformed = sqrt_target
            elif name == "Box-Cox":
                transformed = boxcox_target
            elif name == "Yeo-Johnson":
                transformed = yj_target
            elif name == "Quantile":
                transformed = q_target
            
            # Use Shapiro-Wilk test on a sample
            _, p_value = stats.shapiro(transformed[:5000])
            transformation_normality.append((name, p_value))
        
        print("\nNormality test after transformation (higher p-value is better):")
        for name, p_value in sorted(transformation_normality, key=lambda x: x[1], reverse=True):
            print(f"  {name}: p-value = {p_value:.6f}")
        
        # Store the best transformation based on both skewness and normality
        self.target_transform = best_transform[0]
        
        # Analyze target variable by categorical features
        self._analyze_target_by_categories()
        
        return self
    
    def _analyze_target_by_categories(self):
        """Analyze target variable distribution within categorical features."""
        if self.train_data is None or self.target not in self.train_data.columns:
            return
        
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        if not categorical_cols:
            return
        
        print("\nğŸ“Š Target by Categorical Features:")
        
        # Limit to top 5 categorical features with most distinct values
        top_categorical = sorted(categorical_cols, 
                               key=lambda x: self.train_data[x].nunique(), 
                               reverse=True)[:5]
        
        for col in top_categorical:
            # Calculate target statistics by category
            target_stats = self.train_data.groupby(col)[self.target].agg(['mean', 'median', 'std', 'count'])
            target_stats = target_stats.sort_values('mean', ascending=False)
            
            print(f"\nTarget statistics by {col}:")
            print(target_stats.head())
            
            # Create visualization for top categories
            plt.figure(figsize=(12, 6))
            
            # Only include top categories by frequency to avoid cluttered plot
            top_cats = self.train_data[col].value_counts().nlargest(10).index
            subset = self.train_data[self.train_data[col].isin(top_cats)]
            
            # Boxplot
            sns.boxplot(x=col, y=self.target, data=subset)
            plt.xticks(rotation=45)
            plt.title(f'{self.target} Distribution by {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_path, f'target_by_{col}.png'))
            plt.close()
            
            # Calculate discriminative power
            overall_variance = self.train_data[self.target].var()
            group_variance_weighted = 0
            
            for category, group in self.train_data.groupby(col):
                weight = len(group) / len(self.train_data)
                group_variance_weighted += weight * group[self.target].var()
            
            discrimination_ratio = 1 - (group_variance_weighted / overall_variance)
            
            # Store target encoding statistics for feature engineering
            # Only encode categories with meaningful sample size
            min_samples = max(5, len(self.train_data) * 0.01)
            category_stats = {}
            
            for category, group in self.train_data.groupby(col):
                if len(group) >= min_samples:
                    category_stats[category] = {
                        'mean': group[self.target].mean(),
                        'median': group[self.target].median(),
                        'std': group[self.target].std(),
                        'count': len(group)
                    }
            
            self.target_encoding_maps[col] = category_stats
            
            print(f"  Discrimination ratio: {discrimination_ratio:.4f}")
            print(f"  Target encoding stats created for {len(category_stats)} categories")
            
            # Store feature metadata
            if col in self.feature_metadata:
                self.feature_metadata[col]['discrimination_ratio'] = discrimination_ratio
                self.feature_metadata[col]['target_encoding'] = True
            else:
                self.feature_metadata[col] = {
                    'type': 'categorical',
                    'discrimination_ratio': discrimination_ratio,
                    'target_encoding': True
                }
    
    def preprocess_data(self, handle_outliers=True, use_original_data=True):
        """
        Enhanced preprocessing with multiple strategies for handling missing values and outliers.
        """
        print("\n" + "="*80)
        print("ğŸ”§ ENHANCED DATA PREPROCESSING")
        print("="*80)
        
        if self.train_data is None or self.test_data is None:
            print("â�Œ No data loaded. Please run load_data() first.")
            return self
        
        print("\nğŸ“‹ Handling missing values and merging datasets...")
        
        # Save test IDs before any processing
        self.test_ids = self.test_data['id'].values
        
        # Make copies to avoid modifying the original data
        train_df = self.train_data.copy()
        test_df = self.test_data.copy()
        
        # Drop rows with missing target from original data if available
        if use_original_data and self.original_data is not None:
            original_df = self.original_data.copy()
            original_df = original_df.dropna(subset=[self.target])
            
            # Combine original data with train data to increase training dataset size
            train_df = pd.concat([train_df, original_df], axis=0).reset_index(drop=True)
            print(f"Combined training data shape: {train_df.shape}")
        
        # Update categorical and numerical features based on available columns
        self.categorical_features = [col for col in self.categorical_features if col in train_df.columns]
        self.numerical_features = [col for col in self.numerical_features if col in train_df.columns]
        
        # Advanced missing value imputation with multiple strategies
        print("\nğŸ“‹ Advanced missing value imputation...")
        
        # 1. For categorical features
        for feature in self.categorical_features:
            if feature in train_df.columns:
                missing_count = train_df[feature].isnull().sum()
                if missing_count > 0:
                    print(f"  Imputing {missing_count} missing values in {feature}")
                    
                    # Use mode (most frequent value)
                    mode_value = train_df[feature].mode()[0]
                    
                    # Create missing indicator
                    train_df[f'{feature}_missing'] = train_df[feature].isnull().astype(int)
                    if feature in test_df.columns:
                        test_df[f'{feature}_missing'] = test_df[feature].isnull().astype(int)
                    
                    # Fill missing values
                    train_df[feature].fillna(mode_value, inplace=True)
                    if feature in test_df.columns:
                        test_df[feature].fillna(mode_value, inplace=True)
        
        # 2. For numerical features, use multiple strategies
        for feature in self.numerical_features:
            if feature in train_df.columns:
                missing_count = train_df[feature].isnull().sum()
                if missing_count > 0:
                    print(f"  Imputing {missing_count} missing values in {feature}")
                    
                    # Create missing indicator
                    train_df[f'{feature}_missing'] = train_df[feature].isnull().astype(int)
                    if feature in test_df.columns:
                        test_df[f'{feature}_missing'] = test_df[feature].isnull().astype(int)
                    
                    # For numerical features, check if feature has many zeros
                    zero_percent = (train_df[feature] == 0).mean()
                    
                    if zero_percent > 0.2:
                        # If many zeros, use median to avoid zero-inflation
                        impute_value = train_df[feature].median()
                    else:
                        # Otherwise use mean for better central tendency
                        impute_value = train_df[feature].mean()
                    
                    # Fill missing values
                    train_df[feature].fillna(impute_value, inplace=True)
                    if feature in test_df.columns:
                        test_df[feature].fillna(impute_value, inplace=True)
        
        print("âœ… Missing values handled")
        
        # Handle outliers in training data using multiple strategies
        if handle_outliers:
            print("\nğŸ“‹ Advanced outlier handling...")
            
            initial_train_rows = len(train_df)
            
            # 1. For numerical features, use winsorization with different thresholds
            for feature in self.numerical_features:
                if feature in train_df.columns:
                    # Check if feature has high skewness
                    skewness = train_df[feature].skew()
                    kurtosis = train_df[feature].kurtosis()
                    
                    # Adjust strategy based on distribution
                    if abs(skewness) > 3 or kurtosis > 10:
                        # Highly skewed or heavy-tailed - use more aggressive winsorization
                        q_low, q_high = 0.01, 0.99
                        print(f"  Using aggressive winsorization for {feature} (skew={skewness:.2f}, kurtosis={kurtosis:.2f})")
                    else:
                        # More normally distributed - use standard IQR method
                        q_low, q_high = 0.05, 0.95
                    
                    # Get quantile values
                    q1 = train_df[feature].quantile(q_low)
                    q3 = train_df[feature].quantile(q_high)
                    
                    # Apply winsorization
                    train_df[feature] = train_df[feature].clip(lower=q1, upper=q3)
                    
                    # Store information about clipping for later reference
                    if feature in self.feature_metadata:
                        self.feature_metadata[feature]['winsorized'] = True
                        self.feature_metadata[feature]['winsor_lower'] = q1
                        self.feature_metadata[feature]['winsor_upper'] = q3
            
            # 2. Handle target outliers separately with a more conservative approach
            if self.target in train_df.columns:
                # Use standard IQR method with wider bounds
                q1 = train_df[self.target].quantile(0.005)
                q3 = train_df[self.target].quantile(0.995)
                iqr = q3 - q1
                
                lower_bound = q1 - 2.0 * iqr  # More conservative lower bound
                upper_bound = q3 + 2.0 * iqr  # More conservative upper bound
                
                # Count outliers before clipping
                outliers_count = ((train_df[self.target] < lower_bound) | 
                                 (train_df[self.target] > upper_bound)).sum()
                
                if outliers_count > 0:
                    print(f"  Winsorizing {outliers_count} outliers in target variable")
                    train_df[self.target] = train_df[self.target].clip(lower=lower_bound, upper=upper_bound)
            
            print(f"âœ… Applied winsorization to handle outliers while preserving data")
            print(f"Final training shape: {train_df.shape}")
        
        # Update the processed data
        self.train_data = train_df
        self.test_data = test_df
        
        return self
    
    def transform_target(self):
        """
        Apply the best transformation to the target variable.
        """
        print("\nğŸ“‹ Transforming target variable...")
        
        if not hasattr(self, 'target_transform'):
            print("âš ï¸� No target transformation analysis performed. Running analysis...")
            self.analyze_target_variable()
            
        target_values = self.train_data[self.target].values
            
        # Apply the selected transformation
        if self.target_transform == "Log":
            print("Applying log transformation to target")
            self.train_data[f"{self.target}_original"] = self.train_data[self.target]
            self.train_data[self.target] = np.log1p(self.train_data[self.target])
            self.inverse_transform_fn = lambda x: np.expm1(x)
        
        elif self.target_transform == "Square Root":
            print("Applying square root transformation to target")
            self.train_data[f"{self.target}_original"] = self.train_data[self.target]
            self.train_data[self.target] = np.sqrt(self.train_data[self.target])
            self.inverse_transform_fn = lambda x: x**2
        
        elif self.target_transform == "Box-Cox" and hasattr(self, 'boxcox_lambda'):
            print(f"Applying Box-Cox transformation to target (lambda={self.boxcox_lambda:.4f})")
            self.train_data[f"{self.target}_original"] = self.train_data[self.target]
            self.train_data[self.target] = stats.boxcox(self.train_data[self.target], self.boxcox_lambda)
            # For back-transformation
            self.inverse_transform_fn = lambda x: stats.inv_boxcox(x, self.boxcox_lambda)
        
        elif self.target_transform == "Yeo-Johnson" and hasattr(self, 'yj_transformer'):
            print("Applying Yeo-Johnson transformation to target")
            self.train_data[f"{self.target}_original"] = self.train_data[self.target]
            self.train_data[self.target] = self.yj_transformer.transform(
                self.train_data[self.target].values.reshape(-1, 1)).flatten()
            # For back-transformation
            self.inverse_transform_fn = lambda x: self.yj_transformer.inverse_transform(
                np.array(x).reshape(-1, 1)).flatten()
        
        elif self.target_transform == "Quantile" and hasattr(self, 'q_transformer'):
            print("Applying Quantile transformation to target")
            self.train_data[f"{self.target}_original"] = self.train_data[self.target]
            self.train_data[self.target] = self.q_transformer.transform(
                self.train_data[self.target].values.reshape(-1, 1)).flatten()
            # For back-transformation
            self.inverse_transform_fn = lambda x: self.q_transformer.inverse_transform(
                np.array(x).reshape(-1, 1)).flatten()
            
        else:
            print("Using original target values (no transformation)")
            self.inverse_transform_fn = lambda x: x
            
        return self
    
    def engineer_features(self):
def engineer_features(self):
    """
    Enhanced feature engineering with advanced techniques based on data analysis.
    """
    print("\n" + "="*80)
    print("ğŸ”§ ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    if self.train_data is None or self.test_data is None:
        print("â�Œ No data loaded. Please run preprocess_data() first.")
        return self
    
    print("\nğŸ“‹ Creating enhanced features...")
    
    # Make copies to avoid modifying the input data
    train_data = self.train_data.copy()
    test_data = self.test_data.copy()
    
    # Check what columns are available before creating features
    available_columns = set(train_data.columns)
    
    # Keep track of created features for reporting
    created_features = []
    
    # Group 1: Basic transformations for numerical features
    print("\nğŸ“Š Creating numerical transformations...")
    numerical_features = [col for col in self.numerical_features if col in available_columns]
    
    for feature in numerical_features:
        # Get feature metadata if available
        feature_meta = self.feature_metadata.get(feature, {})
        is_skewed = feature_meta.get('is_highly_skewed', False)
        
        # 1. Log transformation for positive skewed features
        if train_data[feature].min() >= 0:  # Only for non-negative features
            train_data[f'Log_{feature}'] = np.log1p(train_data[feature])
            test_data[f'Log_{feature}'] = np.log1p(test_data[feature])
            created_features.append(f'Log_{feature}')
        
        # 2. Square root transformation (for moderately skewed positive data)
        if train_data[feature].min() >= 0:
            train_data[f'Sqrt_{feature}'] = np.sqrt(train_data[feature])
            test_data[f'Sqrt_{feature}'] = np.sqrt(test_data[feature])
            created_features.append(f'Sqrt_{feature}')
        
        # 3. Square and higher powers
        train_data[f'{feature}_Squared'] = train_data[feature] ** 2
        test_data[f'{feature}_Squared'] = test_data[feature] ** 2
        created_features.append(f'{feature}_Squared')
        
        train_data[f'{feature}_Cubed'] = train_data[feature] ** 3
        test_data[f'{feature}_Cubed'] = test_data[feature] ** 3
        created_features.append(f'{feature}_Cubed')
        
        # 4. Binning (create categorical variables from continuous)
        # Use quantile binning to ensure even distribution
        try:
            bins = 5
            train_data[f'{feature}_Bin'] = pd.qcut(
                train_data[feature], q=bins, labels=False, duplicates='drop')
            
            # Get the bin edges for consistent application to test set
            bin_edges = pd.qcut(train_data[feature], q=bins, retbins=True, duplicates='drop')[1]
            
            # Apply same binning to test data
            test_data[f'{feature}_Bin'] = pd.cut(
                test_data[feature], bins=bin_edges, labels=False, include_lowest=True)
            
            # Handle edge cases in test set
            test_data[f'{feature}_Bin'].fillna(0, inplace=True)
            
            created_features.append(f'{feature}_Bin')
            
            # Also create one-hot encoded bins for direct model use
            for i in range(min(bins, len(bin_edges)-1)):
                train_data[f'{feature}_Bin_{i}'] = (train_data[f'{feature}_Bin'] == i).astype(int)
                test_data[f'{feature}_Bin_{i}'] = (test_data[f'{feature}_Bin'] == i).astype(int)
                created_features.append(f'{feature}_Bin_{i}')
        except Exception as e:
            print(f"  Error creating bins for {feature}: {str(e)}")
    
    # Group 2: Feature interactions
    print("\nğŸ“Š Creating feature interactions...")
    
    # 1. Numerical-Numerical interactions
    # Generate pairwise interactions between numerical features
    num_features_for_interactions = numerical_features[:min(5, len(numerical_features))]
    
    for i, feat1 in enumerate(num_features_for_interactions):
        for feat2 in num_features_for_interactions[i+1:]:
            # Multiplication (interaction term)
            train_data[f'{feat1}_x_{feat2}'] = train_data[feat1] * train_data[feat2]
            test_data[f'{feat1}_x_{feat2}'] = test_data[feat1] * test_data[feat2]
            created_features.append(f'{feat1}_x_{feat2}')
            
            # Division (ratio features) - with safeguard against division by zero
            train_data[f'{feat1}_div_{feat2}'] = train_data[feat1] / (train_data[feat2] + 1e-5)
            test_data[f'{feat1}_div_{feat2}'] = test_data[feat1] / (test_data[feat2] + 1e-5)
            created_features.append(f'{feat1}_div_{feat2}')
            
            # Sum
            train_data[f'{feat1}_plus_{feat2}'] = train_data[feat1] + train_data[feat2]
            test_data[f'{feat1}_plus_{feat2}'] = test_data[feat1] + test_data[feat2]
            created_features.append(f'{feat1}_plus_{feat2}')
            
            # Difference
            train_data[f'{feat1}_minus_{feat2}'] = train_data[feat1] - train_data[feat2]
            test_data[f'{feat1}_minus_{feat2}'] = test_data[feat1] - test_data[feat2]
            created_features.append(f'{feat1}_minus_{feat2}')
    
    # 2. Categorical interactions
    print("\nğŸ“Š Creating categorical interactions...")
    categorical_features = [col for col in self.categorical_features if col in available_columns]
    
    # Get top categorical features by discrimination ratio (if available)
    if hasattr(self, 'feature_metadata') and self.feature_metadata:
        categoricals_with_meta = [(col, self.feature_metadata.get(col, {}).get('discrimination_ratio', 0)) 
                               for col in categorical_features]
        top_categoricals = [col for col, _ in sorted(categoricals_with_meta, 
                                                  key=lambda x: x[1], reverse=True)][:min(4, len(categorical_features))]
    else:
        top_categoricals = categorical_features[:min(4, len(categorical_features))]
    
    # Create interactions between top categorical features
    for i, cat1 in enumerate(top_categoricals):
        for cat2 in top_categoricals[i+1:]:
            # Combined categorical feature
            train_data[f'{cat1}_{cat2}'] = train_data[cat1].astype(str) + "_" + train_data[cat2].astype(str)
            test_data[f'{cat1}_{cat2}'] = test_data[cat1].astype(str) + "_" + test_data[cat2].astype(str)
            created_features.append(f'{cat1}_{cat2}')
    
    # 3. Mixed numerical-categorical interactions
    print("\nğŸ“Š Creating numerical-categorical interactions...")
    
    for num_feat in num_features_for_interactions[:min(3, len(num_features_for_interactions))]:
        for cat_feat in top_categoricals[:min(3, len(top_categoricals))]:
            # Group statistics
            cat_means = train_data.groupby(cat_feat)[num_feat].mean()
            cat_stds = train_data.groupby(cat_feat)[num_feat].std().fillna(0)
            
            # Create features showing the deviation from group mean
            train_data[f'{num_feat}_dev_by_{cat_feat}'] = train_data.apply(
                lambda x: (x[num_feat] - cat_means.get(x[cat_feat], 0)) 
                / (cat_stds.get(x[cat_feat], 1) + 1e-5), axis=1)
            
            test_data[f'{num_feat}_dev_by_{cat_feat}'] = test_data.apply(
                lambda x: (x[num_feat] - cat_means.get(x[cat_feat], 0)) 
                / (cat_stds.get(x[cat_feat], 1) + 1e-5), axis=1)
            
            created_features.append(f'{num_feat}_dev_by_{cat_feat}')
            
            # Create group rank features
            train_data[f'{num_feat}_rank_in_{cat_feat}'] = train_data.groupby(cat_feat)[num_feat].rank(pct=True)
            # For test data, we need a mapping function
            train_ranks = train_data.groupby(cat_feat)[num_feat].rank(pct=True)
            
            # Create a mapping function for test data ranks (approximate)
            def get_percentile_rank(row, feature, category_col, category_values):
                category = row[category_col]
                value = row[feature]
                
                # Get values for this category from training data
                if category in category_values:
                    cat_values = category_values[category]
                    # Find percentile rank of this value
                    return pd.Series(cat_values).rank(pct=True).iloc[-1]
                else:
                    return 0.5  # Default to middle rank for unknown categories
            
            # Get values by category
            category_values = {cat: train_data[train_data[cat_feat] == cat][num_feat].values 
                             for cat in train_data[cat_feat].unique()}
            
            # Apply to test data
            test_data[f'{num_feat}_rank_in_{cat_feat}'] = test_data.apply(
                lambda row: get_percentile_rank(row, num_feat, cat_feat, category_values), axis=1)
            
            created_features.append(f'{num_feat}_rank_in_{cat_feat}')
    
    # Group 4: Target encoding with cross-validation
    print("\nğŸ“Š Creating target encoding features...")
    
    if hasattr(self, 'target_encoding_maps') and self.target_encoding_maps and self.target in train_data.columns:
        # Identify categorical features for target encoding
        categorical_for_encoding = [col for col in categorical_features 
                                  if col in self.target_encoding_maps]
        
        if categorical_for_encoding:
            # Function for smooth target encoding with regularization
            def get_smooth_target_encoding(x, encoding_map, global_mean, k=20):
                if x in encoding_map and encoding_map[x]['count'] > 0:
                    # Smooth encoding using additive smoothing
                    count = encoding_map[x]['count']
                    category_mean = encoding_map[x]['mean']
                    # More weight to category mean for larger groups
                    weight = count / (count + k)
                    return weight * category_mean + (1 - weight) * global_mean
                else:
                    return global_mean
            
            # Perform target encoding
            for col in categorical_for_encoding:
                encoding_map = self.target_encoding_maps[col]
                global_mean = train_data[self.target].mean()
                
                # Apply target encoding with smoothing
                train_data[f'{col}_TargetEncoded'] = train_data[col].apply(
                    lambda x: get_smooth_target_encoding(x, encoding_map, global_mean))
                
                test_data[f'{col}_TargetEncoded'] = test_data[col].apply(
                    lambda x: get_smooth_target_encoding(x, encoding_map, global_mean))
                
                created_features.append(f'{col}_TargetEncoded')
                
                # Also create high/low binary flags based on target encoding
                threshold = np.percentile(train_data[f'{col}_TargetEncoded'], 75)
                train_data[f'{col}_HighValue'] = (train_data[f'{col}_TargetEncoded'] > threshold).astype(int)
                test_data[f'{col}_HighValue'] = (test_data[f'{col}_TargetEncoded'] > threshold).astype(int)
                
                created_features.append(f'{col}_HighValue')
            
            print(f"  Created target encoding features for {len(categorical_for_encoding)} categorical variables")
    
    # Group 5: Dimensionality reduction and clustering features
    print("\nğŸ“Š Creating dimensionality reduction features...")
    
    # 1. PCA on numerical features
    all_numerical = [col for col in train_data.columns 
                    if train_data[col].dtype in ['int64', 'float64'] 
                    and col != 'id' and col != self.target 
                    and (col.startswith('Log_') or col in numerical_features)]
    
    if len(all_numerical) > 5:  # Only do PCA if we have enough numerical features
        try:
            # Scale the data for PCA
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(train_data[all_numerical].fillna(0))
            scaled_test = scaler.transform(test_data[all_numerical].fillna(0))
            
            # Apply PCA
            n_components = min(5, len(all_numerical) - 1)
            pca = PCA(n_components=n_components)
            
            pca_result_train = pca.fit_transform(scaled_train)
            pca_result_test = pca.transform(scaled_test)
            
            # Add PCA components as features
            for i in range(n_components):
                train_data[f'PCA_{i+1}'] = pca_result_train[:, i]
                test_data[f'PCA_{i+1}'] = pca_result_test[:, i]
                created_features.append(f'PCA_{i+1}')
            
            # Save explained variance for reference
            explained_variance = pca.explained_variance_ratio_
            print(f"  PCA components explain {sum(explained_variance) * 100:.2f}% of variance")
            
            # Create compound feature from top PCA components
            train_data['PCA_Combined'] = 0
            test_data['PCA_Combined'] = 0
            
            for i in range(min(3, n_components)):
                train_data['PCA_Combined'] += train_data[f'PCA_{i+1}'] * explained_variance[i]
                test_data['PCA_Combined'] += test_data[f'PCA_{i+1}'] * explained_variance[i]
            
            created_features.append('PCA_Combined')
        except Exception as e:
            print(f"  Error creating PCA features: {str(e)}")
    
    # 2. Clustering features based on numerical data
    if len(all_numerical) > 3:  # Only do clustering if we have enough features
        try:
            print("\nğŸ“Š Creating clustering features...")
            
            # Scale data for clustering
            if 'scaled_train' not in locals():
                scaler = StandardScaler()
                scaled_train = scaler.fit_transform(train_data[all_numerical].fillna(0))
                scaled_test = scaler.transform(test_data[all_numerical].fillna(0))
            
            # Apply K-Means clustering with different numbers of clusters
            for n_clusters in [3, 5, 8]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
                train_clusters = kmeans.fit_predict(scaled_train)
                test_clusters = kmeans.predict(scaled_test)
                
                # Add cluster labels as features
                train_data[f'Cluster_{n_clusters}'] = train_clusters
                test_data[f'Cluster_{n_clusters}'] = test_clusters
                created_features.append(f'Cluster_{n_clusters}')
                
                # One-hot encode the clusters
                for i in range(n_clusters):
                    train_data[f'Cluster_{n_clusters}_{i}'] = (train_clusters == i).astype(int)
                    test_data[f'Cluster_{n_clusters}_{i}'] = (test_clusters == i).astype(int)
                    created_features.append(f'Cluster_{n_clusters}_{i}')
                
                # Create distance features to each cluster center
                for i in range(n_clusters):
                    # Calculate Euclidean distance to cluster center
                    center = kmeans.cluster_centers_[i]
                    
                    # Calculate distances for train data
                    distances_train = np.sqrt(np.sum((scaled_train - center) ** 2, axis=1))
                    train_data[f'Dist_to_Cluster_{n_clusters}_{i}'] = distances_train
                    
                    # Calculate distances for test data
                    distances_test = np.sqrt(np.sum((scaled_test - center) ** 2, axis=1))
                    test_data[f'Dist_to_Cluster_{n_clusters}_{i}'] = distances_test
                    
                    created_features.append(f'Dist_to_Cluster_{n_clusters}_{i}')
            
            print(f"  Created clustering features with K={', '.join(map(str, [3, 5, 8]))}")
        except Exception as e:
            print(f"  Error creating clustering features: {str(e)}")
    
    # Group 6: Advanced polynomial features (limited to most important features)
    if len(all_numerical) > 2:
        print("\nğŸ“Š Creating advanced polynomial features...")
        
        # Use top 3 numerical features
        important_num_features = all_numerical[:min(3, len(all_numerical))]
        
        for i, feat1 in enumerate(important_num_features):
            for j, feat2 in enumerate(important_num_features):
                if i <= j:  # Include i==j for squared terms and i<j for interactions
                    # Create quadratic terms (squared and interactions)
                    feature_name = f'{feat1}_x_{feat2}_poly2'
                    train_data[feature_name] = train_data[feat1] * train_data[feat2]
                    test_data[feature_name] = test_data[feat1] * test_data[feat2]
                    created_features.append(feature_name)
                    
                    # For cubics, only consider i<j to avoid too many features
                    if i < j:
                        for k, feat3 in enumerate(important_num_features):
                            if j <= k:  # Avoid duplicates
                                # Create cubic terms
                                feature_name = f'{feat1}_x_{feat2}_x_{feat3}_poly3'
                                train_data[feature_name] = train_data[feat1] * train_data[feat2] * train_data[feat3]
                                test_data[feature_name] = test_data[feat1] * test_data[feat2] * test_data[feat3]
                                created_features.append(feature_name)
        
        print(f"  Created polynomial features up to degree 3")
    
    # Group 7: Feature frequency encoding
    print("\nğŸ“Š Creating frequency encoding features...")
    
    for col in categorical_features:
        if col in train_data.columns:
            # Calculate frequency encodings
            frequency_map = train_data[col].value_counts(normalize=True).to_dict()
            
            # Apply to both datasets
            train_data[f'{col}_Frequency'] = train_data[col].map(frequency_map)
            test_data[f'{col}_Frequency'] = test_data[col].map(frequency_map)
            
            # For test data, handle unseen categories
            test_data[f'{col}_Frequency'].fillna(min(frequency_map.values()) if frequency_map else 0, inplace=True)
            
            created_features.append(f'{col}_Frequency')
    
    # Group 8: Count encoding
    print("\nğŸ“Š Creating count encoding features...")
    
    for col in categorical_features:
        if col in train_data.columns:
            # Calculate count encodings
            count_map = train_data[col].value_counts().to_dict()
            
            # Apply to both datasets
            train_data[f'{col}_Count'] = train_data[col].map(count_map)
            test_data[f'{col}_Count'] = test_data[col].map(count_map)
            
            # For test data, handle unseen categories
            test_data[f'{col}_Count'].fillna(1, inplace=True)
            
            created_features.append(f'{col}_Count')
    
    # Group 9: Create boolean flags for rare categories
    print("\nğŸ“Š Creating rare category flags...")
    
    for col in categorical_features:
        if col in train_data.columns:
            # Identify rare categories (less than 5% of data)
            value_counts = train_data[col].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < 0.05].index.tolist()
            
            if rare_categories:
                # Create a flag for rare categories
                train_data[f'{col}_is_rare'] = train_data[col].isin(rare_categories).astype(int)
                test_data[f'{col}_is_rare'] = test_data[col].isin(rare_categories).astype(int)
                
                created_features.append(f'{col}_is_rare')
    
    print(f"\nâœ… Created {len(created_features)} new features")
    
    # Update the processed data
    self.train_data = train_data
    self.test_data = test_data
    
    return self

    def select_features(self, method='mutual_info', n_features=None, threshold=None):
        """
        Advanced feature selection to identify the most important features.
        
        Args:
            method: Feature selection method ('mutual_info', 'forest', 'recursive', 'shap', 'combined')
            n_features: Number of features to select (None for auto)
            threshold: Importance threshold for selection (overrides n_features if provided)
        """
        print("\n" + "="*80)
        print("ğŸ”� ADVANCED FEATURE SELECTION")
        print("="*80)
        
        if self.X_train is None or self.y_train is None:
            print("â�Œ No prepared data found. Please run prepare_for_modeling() first.")
            return self
        
        # Calculate automatic n_features if not provided
        if n_features is None:
            n_features = max(20, min(self.X_train.shape[1] // 3, 100))
        
        print(f"\nğŸ“‹ Selecting top features using method: {method}")
        
        # Store feature importance for all methods
        feature_importance_dict = {}
        
        if method == 'mutual_info' or method == 'combined':
            print(f"\nğŸ“Š Using mutual information...")
            
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(self.X_train, self.y_train, random_state=RANDOM_STATE)
            
            # Create a dataframe of features and their scores
            mi_df = pd.DataFrame({
                'Feature': self.X_train.columns,
                'MI_Score': mi_scores
            }).sort_values('MI_Score', ascending=False)
            
            # Normalize importance scores to [0, 1]
            if mi_df['MI_Score'].max() > 0:
                mi_df['MI_Score_Norm'] = mi_df['MI_Score'] / mi_df['MI_Score'].max()
            else:
                mi_df['MI_Score_Norm'] = 0
            
            # Store normalized importance scores
            feature_importance_dict['mutual_info'] = dict(zip(mi_df['Feature'], mi_df['MI_Score_Norm']))
            
            # Visualize feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='MI_Score', y='Feature', data=mi_df.head(min(20, n_features)))
            plt.title(f'Top Features by Mutual Information Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_path, 'mutual_info_features.png'))
            plt.close()
        
        if method == 'forest' or method == 'combined':
            print(f"\nğŸ“Š Using Random Forest importance...")
            
            # Train a Random Forest model
            forest = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
            forest.fit(self.X_train, self.y_train)
            
            # Get feature importances
            importances = forest.feature_importances_
            
            # Create a dataframe of features and their importances
            forest_df = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Normalize importance scores
            forest_df['Importance_Norm'] = forest_df['Importance'] / forest_df['Importance'].max()
            
            # Store normalized importance scores
            feature_importance_dict['forest'] = dict(zip(forest_df['Feature'], forest_df['Importance_Norm']))
            
            # Visualize feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=forest_df.head(min(20, n_features)))
            plt.title(f'Top Features by Random Forest Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_path, 'forest_feature_importance.png'))
            plt.close()
            
            # Save the trained model for later use
            self.best_models['feature_selector'] = forest
        
        if method == 'shap' or method == 'combined':
            print(f"\nğŸ“Š Using SHAP values for feature importance...")
            
            try:
                # Train a LightGBM model for SHAP values
                model = lgb.LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE)
                model.fit(self.X_train, self.y_train)
                
                # Calculate SHAP values on a sample of data (for speed)
                sample_size = min(1000, len(self.X_train))
                X_sample = self.X_train.sample(sample_size, random_state=RANDOM_STATE)
                
                # Create explainer and calculate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Calculate mean absolute SHAP values per feature
                shap_importance = np.abs(shap_values).mean(axis=0)
                
                # Create dataframe of features and SHAP importance
                shap_df = pd.DataFrame({
                    'Feature': self.X_train.columns,
                    'SHAP_Importance': shap_importance
                }).sort_values('SHAP_Importance', ascending=False)
                
                # Normalize SHAP importance scores
                shap_df['SHAP_Importance_Norm'] = shap_df['SHAP_Importance'] / shap_df['SHAP_Importance'].max()
                
                # Store normalized importance scores
                feature_importance_dict['shap'] = dict(zip(shap_df['Feature'], shap_df['SHAP_Importance_Norm']))
                
                # Create SHAP summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                plt.title('Feature Importance using SHAP Values')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_path, 'shap_feature_importance.png'))
                plt.close()
                
            except Exception as e:
                print(f"  Error calculating SHAP values: {str(e)}")
                print("  Skipping SHAP-based feature selection")
        
        if method == 'recursive' or method == 'combined':
            print(f"\nğŸ“Š Using Recursive Feature Elimination...")
            
            try:
                # Use a fast estimator for RFE (Ridge regression)
                estimator = Ridge(alpha=1.0, random_state=RANDOM_STATE)
                
                # Initialize RFECV
                selector = RFECV(
                    estimator=estimator,
                    step=max(1, self.X_train.shape[1] // 20),  # Remove 5% of features at a time
                    cv=5,
                    scoring='neg_mean_absolute_error',
                    min_features_to_select=min(5, self.X_train.shape[1] // 10),
                    n_jobs=-1
                )
                
                # Fit the selector
                selector.fit(self.X_train, self.y_train)
                
                # Get selected features
                selected_mask = selector.support_
                rfe_selected = self.X_train.columns[selected_mask].tolist()
                
                # Create importance scores based on ranking
                rfe_importance = 1 / (selector.ranking_ + 1e-10)  # Lower rank = higher importance
                
                # Create dataframe of features and RFE importance
                rfe_df = pd.DataFrame({
                    'Feature': self.X_train.columns,
                    'RFE_Importance': rfe_importance
                }).sort_values('RFE_Importance', ascending=False)
                
                # Normalize RFE importance scores
                rfe_df['RFE_Importance_Norm'] = rfe_df['RFE_Importance'] / rfe_df['RFE_Importance'].max()
                
                # Store normalized importance scores
                feature_importance_dict['recursive'] = dict(zip(rfe_df['Feature'], rfe_df['RFE_Importance_Norm']))
                
                # Visualize number of features vs. performance
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
                plt.xlabel('Number of Features')
                plt.ylabel('Negative MAE')
                plt.title('Feature Selection using RFE with Cross-Validation')
                plt.grid(True)
                plt.savefig(os.path.join(self.plots_path, 'rfecv_feature_selection.png'))
                plt.close()
                
                print(f"  Selected {len(rfe_selected)} features using RFECV")
                print(f"  Optimal number of features: {selector.n_features_}")
                
            except Exception as e:
                print(f"  Error in recursive feature elimination: {str(e)}")
                print("  Skipping RFE-based feature selection")
        
        # Combine importance scores for multiple methods
        if method == 'combined':
            print("\nğŸ“Š Combining feature importance scores from multiple methods...")
            
            all_features = set()
            for method_scores in feature_importance_dict.values():
                all_features.update(method_scores.keys())
            
            # Create dataframe with combined scores
            combined_scores = {}
            
            for feature in all_features:
                # Average the scores from all methods that have a score for this feature
                scores = [method_scores.get(feature, 0) for method_scores in feature_importance_dict.values()]
                # Weighted combination: higher weight to Forest and SHAP if available
                weights = []
                for method_name in feature_importance_dict.keys():
                    if method_name == 'forest':
                        weights.append(1.5)  # Higher weight to Random Forest
                    elif method_name == 'shap':
                        weights.append(2.0)  # Higher weight to SHAP
                    else:
                        weights.append(1.0)  # Normal weight to others
                
                # Calculate weighted average if we have weights, otherwise simple average
                if len(scores) == len(weights):
                    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                else:
                    weighted_score = sum(scores) / len(scores)
                
                combined_scores[feature] = weighted_score
            
            # Create sorted dataframe
            combined_df = pd.DataFrame({
                'Feature': list(combined_scores.keys()),
                'Combined_Importance': list(combined_scores.values())
            }).sort_values('Combined_Importance', ascending=False)
            
            # Store for later use
            self.feature_importance = combined_df
            
            # Visualize combined importance
            plt.figure(figsize=(12, 10))
            sns.barplot(x='Combined_Importance', y='Feature', data=combined_df.head(min(30, n_features)))
            plt.title('Feature Importance (Combined Methods)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_path, 'combined_feature_importance.png'))
            plt.close()
            
            # Select top features based on combined importance
            if threshold is not None:
                top_features = combined_df[combined_df['Combined_Importance'] >= threshold]['Feature'].tolist()
            else:
                top_features = combined_df.head(n_features)['Feature'].tolist()
            
        else:
            # Use the individual method's results
            if method == 'mutual_info':
                importance_df = mi_df
                importance_col = 'MI_Score'
            elif method == 'forest':
                importance_df = forest_df
                importance_col = 'Importance'
            elif method == 'shap' and 'shap_df' in locals():
                importance_df = shap_df
                importance_col = 'SHAP_Importance'
            elif method == 'recursive' and 'rfe_df' in locals():
                importance_df = rfe_df
                importance_col = 'RFE_Importance'
            else:
                print("âš ï¸� Fallback to Random Forest method for feature selection")
                # Train a forest if we don't have one
                if 'forest_df' not in locals():
                    forest = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
                    forest.fit(self.X_train, self.y_train)
                    importances = forest.feature_importances_
                    forest_df = pd.DataFrame({
                        'Feature': self.X_train.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                
                importance_df = forest_df
                importance_col = 'Importance'
            
            # Store for later use
            self.feature_importance = importance_df
            
            # Select top features based on importance
            if threshold is not None:
                top_features = importance_df[importance_df[importance_col] >= threshold]['Feature'].tolist()
            else:
                top_features = importance_df.head(n_features)['Feature'].tolist()
    
    # Additional step: Remove highly correlated features among the selected ones
    if len(top_features) > 1:
        print("\nğŸ“Š Checking for correlation among selected features...")
        
        # Calculate correlation matrix for selected features
        selected_corr = self.X_train[top_features].corr().abs()
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                if selected_corr.iloc[i, j] > 0.95:  # Very high correlation threshold
                    feat_i = top_features[i]
                    feat_j = top_features[j]
                    high_corr_pairs.append((feat_i, feat_j, selected_corr.iloc[i, j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat_i, feat_j, corr in high_corr_pairs:
            print(f"  High correlation ({corr:.3f}) between '{feat_i}' and '{feat_j}'")
            
            # Keep the feature with higher importance
            if feat_i in importance_df['Feature'].values and feat_j in importance_df['Feature'].values:
                imp_i = importance_df[importance_df['Feature'] == feat_i][importance_col].values[0]
                imp_j = importance_df[importance_df['Feature'] == feat_j][importance_col].values[0]
                
                if imp_i >= imp_j:
                    features_to_remove.add(feat_j)
                    print(f"    Removing '{feat_j}' (lower importance)")
                else:
                    features_to_remove.add(feat_i)
                    print(f"    Removing '{feat_i}' (lower importance)")
        
        # Update selected features
        top_features = [f for f in top_features if f not in features_to_remove]
    
    # Store final selected features
    self.best_features = top_features
    
    print(f"\nâœ… Selected {len(top_features)} features")
    
    # Display top selected features
    print("\nTop 15 selected features (or all if fewer):")
    for i, feature in enumerate(top_features[:min(15, len(top_features))]):
        importance = importance_df[importance_df['Feature'] == feature][importance_col].values[0] \
            if feature in importance_df['Feature'].values else 'N/A'
        print(f"{i+1}. {feature} (importance: {importance})")
    
    # Save selected features to file
    with open(os.path.join(self.features_path, 'selected_features.txt'), 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
    
    return self

def prepare_for_modeling(self):
    """
    Prepare datasets for modeling with advanced preprocessing.
    """
    print("\n" + "="*80)
    print("ğŸ”§ PREPARING DATA FOR MODELING")
    print("="*80)
    
    if self.train_data is None or self.test_data is None:
        print("â�Œ No data loaded. Please run engineer_features() first.")
        return self
    
    print("\nğŸ“‹ Encoding and scaling features...")
    
    # Separate target variable
    if self.target in self.train_data.columns:
        self.y_train = self.train_data[self.target].values
        self.X_train = self.train_data.drop(self.target, axis=1)
        
        # Also drop the original target if we transformed it
        if f"{self.target}_original" in self.X_train.columns:
            self.X_train = self.X_train.drop(f"{self.target}_original", axis=1)
    else:
        print("âš ï¸� Warning: Target variable not found in training data")
        self.X_train = self.train_data.copy()
        self.y_train = None
    
    self.X_test = self.test_data.copy()
    
    # Drop ID column if present
    if 'id' in self.X_train.columns:
        self.X_train = self.X_train.drop('id', axis=1)
    
    if 'id' in self.X_test.columns:
        self.X_test = self.X_test.drop('id', axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    
    # Get dummies for categorical variables (with improved handling)
    print("\nğŸ“‹ Encoding categorical features...")
    categorical_columns_to_encode = []
    
    for col in categorical_cols:
        # Only encode if cardinality is not too high relative to dataset size
        cardinality = self.X_train[col].nunique()
        
        if cardinality < min(1000, len(self.X_train) // 10):  # Practical limit
            categorical_columns_to_encode.append(col)
        else:
            print(f"  âš ï¸� Skipping encoding of '{col}' due to high cardinality ({cardinality} categories)")
            # Remove high-cardinality column to avoid memory issues
            self.X_train = self.X_train.drop(col, axis=1)
            if col in self.X_test.columns:
                self.X_test = self.X_test.drop(col, axis=1)
    
    # Use get_dummies with handling for categorical columns to encode
    if categorical_columns_to_encode:
        print(f"  Encoding {len(categorical_columns_to_encode)} categorical features...")
        self.X_train = pd.get_dummies(self.X_train, columns=categorical_columns_to_encode, drop_first=True)
        self.X_test = pd.get_dummies(self.X_test, columns=categorical_columns_to_encode, drop_first=True)
    
    # Align the encoded datasets to have the same columns
    self.X_train, self.X_test = self.X_train.align(self.X_test, join='left', axis=1, fill_value=0)
    
    # Handle any missing columns in test set
    missing_cols = set(self.X_train.columns) - set(self.X_test.columns)
    for col in missing_cols:
        self.X_test[col] = 0
    
    # Ensure the test set has the same columns in the same order
    self.X_test = self.X_test[self.X_train.columns]
    
    # Advanced scaling for numerical features with outlier handling
    numerical_cols = [col for col in numerical_cols if col in self.X_train.columns]
    if len(numerical_cols) > 0:
        print("\nğŸ“‹ Scaling numerical features...")
        
        # Use RobustScaler for better handling of outliers
        scaler = RobustScaler()
        
        # Apply separately to each column to preserve other columns
        for col in numerical_cols:
            # Only scale if the column has variability
            if self.X_train[col].std() > 0:
                col_values = self.X_train[col].values.reshape(-1, 1)
                self.X_train[col] = scaler.fit_transform(col_values).flatten()
                
                # Apply same transformation to test data
                test_col_values = self.X_test[col].values.reshape(-1, 1)
                self.X_test[col] = scaler.transform(test_col_values).flatten()
        
        print(f"âœ… Scaled {len(numerical_cols)} numerical features using RobustScaler")
    
    # Apply feature selection if available
    if hasattr(self, 'best_features') and self.best_features:
        print(f"\nğŸ“‹ Applying feature selection ({len(self.best_features)} features)...")
        
        # Check that all best features are actually in X_train
        available_features = [f for f in self.best_features if f in self.X_train.columns]
        
        if len(available_features) < len(self.best_features):
            print(f"  âš ï¸� {len(self.best_features) - len(available_features)} selected features are not in the dataset")
            self.best_features = available_features
        
        self.X_train = self.X_train[self.best_features]
        self.X_test = self.X_test[self.best_features]
    
    # Verify data looks good
    print(f"\nFinal training data shape: {self.X_train.shape}")
    print(f"Final test data shape: {self.X_test.shape}")
    
    # Check for NaN values
    train_nans = self.X_train.isna().sum().sum()
    test_nans = self.X_test.isna().sum().sum()
    
    if train_nans > 0 or test_nans > 0:
        print(f"âš ï¸� Found NaN values: {train_nans} in training, {test_nans} in test")
        print("Filling NaN values with 0")
        self.X_train.fillna(0, inplace=True)
        self.X_test.fillna(0, inplace=True)
    
    # Save prepared data for reference
    self.X_train.head(1000).to_csv(os.path.join(self.features_path, 'prepared_train_sample.csv'), index=False)
    
    return self

def optimize_model_parameters(self, model_type, n_trials=50):
    """
    Optimize model hyperparameters using Optuna.
    
    Args:
        model_type: Type of model to optimize ('lightgbm', 'xgboost', 'catboost', etc.)
        n_trials: Number of Optuna trials
    
    Returns:
        dict: Optimized hyperparameters
    """
    print("\n" + "="*80)
    print(f"ğŸ”§ OPTIMIZING {model_type.upper()} HYPERPARAMETERS")
    print("="*80)
    
    if self.X_train is None or self.y_train is None:
        print("â�Œ No prepared data found. Please run prepare_for_modeling() first.")
        return {}
    
    # Define optimization functions for different model types
    
    def optimize_lightgbm(trial):
        param = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', 
                                               ['gbdt', 'dart', 'goss']),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'early_stopping_rounds': 50,
            'random_state': RANDOM_STATE
        }
        
        return evaluate_model_cv(lgb.LGBMRegressor(**param))
    
    def optimize_xgboost(trial):
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'verbosity': 0,
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'early_stopping_rounds': 50,
            'random_state': RANDOM_STATE
        }
        
        return evaluate_model_cv(xgb.XGBRegressor(**param))
    
    def optimize_catboost(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', 
                                                   ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            'verbose': False,
            'early_stopping_rounds': 50,
            'random_seed': RANDOM_STATE
        }
        
        return evaluate_model_cv(cb.CatBoostRegressor(**param))
    
    def optimize_extra_trees(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.5, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        
        return evaluate_model_cv(ExtraTreesRegressor(**param))
    
    def optimize_gradient_boosting(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_float('max_features', 0.5, 1.0),
            'alpha': trial.suggest_float('alpha', 0.1, 0.9),
            'random_state': RANDOM_STATE
        }
        
        return evaluate_model_cv(GradientBoostingRegressor(**param))
    
    def optimize_elastic_net(trial):
        param = {
            'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': 10000,
            'tol': 1e-4,
            'random_state': RANDOM_STATE
        }
        
        return evaluate_model_cv(ElasticNet(**param))
    
    def optimize_svr(trial):
        param = {
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf']),
            'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-5, 1.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        
        return evaluate_model_cv(SVR(**param))
    
    # Helper function for cross-validation
    def evaluate_model_cv(model, cv=5):
        # K-fold cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        
        for train_idx, valid_idx in kf.split(self.X_train):
            X_train_fold, X_valid_fold = self.X_train.iloc[train_idx], self.X_train.iloc[valid_idx]
            y_train_fold, y_valid_fold = self.y_train[train_idx], self.y_train[valid_idx]
            
            # Special handling for LightGBM and other models with early stopping
            if isinstance(model, lgb.LGBMRegressor):
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_valid_fold, y_valid_fold)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                    eval_metric='mae'
                )
                
            elif isinstance(model, xgb.XGBRegressor):
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_valid_fold, y_valid_fold)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
            elif isinstance(model, cb.CatBoostRegressor):
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_valid_fold, y_valid_fold)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
            else:
                # Standard sklearn API
                model.fit(X_train_fold, y_train_fold)
            
            # Predict on validation set
            try:
                if hasattr(model, 'predict_proba'):
                    val_preds = model.predict_proba(X_valid_fold)[:, 1]
                else:
                    val_preds = model.predict(X_valid_fold)
                
                # Calculate metrics - negative MAE for maximization in Optuna
                score = -mean_absolute_error(y_valid_fold, val_preds)
                scores.append(score)
            except Exception as e:
                print(f"Error in model evaluation: {str(e)}")
                return float('-inf')  # Return worst possible score
        
        # Return mean score across folds
        return np.mean(scores)
    
    # Select the appropriate optimization function
    if model_type == 'lightgbm':
        objective = optimize_lightgbm
    elif model_type == 'xgboost':
        objective = optimize_xgboost
    elif model_type == 'catboost':
        objective = optimize_catboost
    elif model_type == 'extra_trees':
        objective = optimize_extra_trees
    elif model_type == 'gradient_boosting':
        objective = optimize_gradient_boosting
    elif model_type == 'elastic_net':
        objective = optimize_elastic_net
    elif model_type == 'svr':
        objective = optimize_svr
    else:
        print(f"â�Œ Unknown model type: {model_type}")
        return {}
    
    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_score = -study.best_value  # Convert back to MAE (positive)
    
    print(f"\nâœ… Best MAE: {best_score:.4f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Visualize optimization history
    try:
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title(f'{model_type} Optimization History')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, f'{model_type}_optimization.png'))
        plt.close()
        
        # Plot parameter importances if there are enough trials
        if n_trials >= 20:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title(f'{model_type} Parameter Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_path, f'{model_type}_param_importance.png'))
            plt.close()
    except Exception as e:
        print(f"Error creating optimization plots: {str(e)}")
    
    return best_params

def train_advanced_models(self, use_optuna=True, n_trials=50, cv_folds=5, ensemble_type='blending'):
   """
   Train multiple models with advanced configurations and ensembling.
   
   Args:
       use_optuna: Whether to use Optuna for hyperparameter optimization
       n_trials: Number of Optuna trials
       cv_folds: Number of cross-validation folds
       ensemble_type: Type of ensemble ('stacking', 'blending', 'voting')
   """
   print("\n" + "="*80)
   print("ğŸ”§ TRAINING ADVANCED MODELS")
   print("="*80)
   
   # Check if we have data prepared
   if self.X_train is None or self.y_train is None:
       print("â�Œ No prepared data found. Please run prepare_for_modeling() first.")
       return self
   
   # Define cross-validation strategy
   kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
   
   # Optimize hyperparameters if requested
   if use_optuna:
       print("\nğŸ“‹ Optimizing hyperparameters with Optuna...")
       
       # Use balanced trial allocation with more for top models
       lgb_trials = max(10, n_trials // 4)
       xgb_trials = max(10, n_trials // 4)
       cb_trials = max(10, n_trials // 4)
       et_trials = max(5, n_trials // 10)
       gb_trials = max(5, n_trials // 10)
       
       # Run optimizations with progress tracking
       print(f"Optimizing LightGBM ({lgb_trials} trials)...")
       lgb_params = self.optimize_model_parameters('lightgbm', n_trials=lgb_trials)
       
       print(f"Optimizing XGBoost ({xgb_trials} trials)...")
       xgb_params = self.optimize_model_parameters('xgboost', n_trials=xgb_trials)
       
       print(f"Optimizing CatBoost ({cb_trials} trials)...")
       cb_params = self.optimize_model_parameters('catboost', n_trials=cb_trials)
       
       print(f"Optimizing Extra Trees ({et_trials} trials)...")
       et_params = self.optimize_model_parameters('extra_trees', n_trials=et_trials)
       
       print(f"Optimizing Gradient Boosting ({gb_trials} trials)...")
       gb_params = self.optimize_model_parameters('gradient_boosting', n_trials=gb_trials)
   else:
       # Use default hyperparameters
       print("\nğŸ“‹ Using default hyperparameters...")
       
       lgb_params = {
           'n_estimators': 1000,
           'learning_rate': 0.05,
           'num_leaves': 31,
           'max_depth': 6,
           'min_child_samples': 20,
           'subsample': 0.8,
           'colsample_bytree': 0.8,
           'random_state': RANDOM_STATE,
           'verbose': -1
       }
       
       xgb_params = {
           'n_estimators': 1000,
           'learning_rate': 0.05,
           'max_depth': 6,
           'subsample': 0.8,
           'colsample_bytree': 0.8,
           'min_child_weight': 1,
           'random_state': RANDOM_STATE,
           'verbosity': 0
       }
       
       cb_params = {
           'iterations': 1000,
           'learning_rate': 0.05,
           'depth': 6,
           'l2_leaf_reg': 3,
           'random_strength': 1,
           'verbose': 0,
           'random_seed': RANDOM_STATE
       }
       
       et_params = {
           'n_estimators': 500,
           'max_depth': 10,
           'min_samples_split': 5,
           'min_samples_leaf': 2,
           'random_state': RANDOM_STATE,
           'n_jobs': -1
       }
       
       gb_params = {
           'n_estimators': 500,
           'learning_rate': 0.05,
           'max_depth': 6,
           'min_samples_split': 5,
           'min_samples_leaf': 2,
           'random_state': RANDOM_STATE
       }
   
   # Models to train with their parameters
   print("\nğŸ“‹ Preparing advanced models...")
   models = {
       'lightgbm': {
           'model': lgb.LGBMRegressor(**lgb_params),
           'params': lgb_params,
           'weight': 1.5  # Higher weight for strong models
       },
       'xgboost': {
           'model': xgb.XGBRegressor(**xgb_params),
           'params': xgb_params,
           'weight': 1.5
       },
       'catboost': {
           'model': cb.CatBoostRegressor(**cb_params),
           'params': cb_params,
           'weight': 1.5
       },
       'extra_trees': {
           'model': ExtraTreesRegressor(**et_params),
           'params': et_params,
           'weight': 1.0
       },
       'gradient_boosting': {
           'model': GradientBoostingRegressor(**gb_params),
           'params': gb_params,
           'weight': 1.0
       },
       'huber': {
           'model': HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000),
           'params': {},
           'weight': 0.7
       },
       'ridge': {
           'model': Ridge(alpha=1.0, random_state=RANDOM_STATE),
           'params': {},
           'weight': 0.7
       },
       'elastic_net': {
           'model': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
           'params': {},
           'weight': 0.7
       }
   }
   
   # Storage for model results
   self.model_results = []
   
   # Out-of-fold predictions for each model
   oof_preds = {}
   test_preds = {}
   
   # Timing information
   timing_info = {}
   
   # Train each model
   for model_name, model_info in models.items():
       print(f"\nğŸ”§ Training {model_name}...")
       start_time = time.time()
       
       # Lists to store fold results
       rmse_scores = []
       mae_scores = []
       r2_scores = []
       oof_predictions = np.zeros(len(self.X_train))
       test_predictions = np.zeros(len(self.X_test))
       
       # Save best model for each fold
       fold_models = []
       
       # Perform K-Fold Cross Validation
       for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
           print(f"  Fold {fold+1}/{cv_folds}...", end='')
           
           # Split data
           X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
           y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
           
           # Train model
           model = copy.deepcopy(model_info['model'])
           
           try:
               if model_name == 'lightgbm':
                   model.fit(
                       X_fold_train, y_fold_train,
                       eval_set=[(X_fold_val, y_fold_val)],
                       callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                       eval_metric='mae'
                   )
               elif model_name == 'xgboost':
                   model.fit(
                       X_fold_train, y_fold_train,
                       eval_set=[(X_fold_val, y_fold_val)],
                       early_stopping_rounds=50,
                       verbose=False
                   )
               elif model_name == 'catboost':
                   model.fit(
                       X_fold_train, y_fold_train,
                       eval_set=[(X_fold_val, y_fold_val)],
                       early_stopping_rounds=50,
                       verbose=False
                   )
               else:
                   # Standard sklearn API
                   model.fit(X_fold_train, y_fold_train)
               
               # Store the trained model for this fold
               fold_models.append(model)
               
               # Predict on validation set
               val_preds = model.predict(X_fold_val)
               oof_predictions[val_idx] = val_preds
               
               # Calculate metrics
               rmse = np.sqrt(mean_squared_error(y_fold_val, val_preds))
               mae = mean_absolute_error(y_fold_val, val_preds)
               r2 = r2_score(y_fold_val, val_preds)
               
               rmse_scores.append(rmse)
               mae_scores.append(mae)
               r2_scores.append(r2)
               
               print(f" RMSE: {rmse:.4f}, MAE: {mae:.4f}, RÂ²: {r2:.4f}")
               
               # Predict on test data (accumulate predictions for later averaging)
               test_predictions += model.predict(self.X_test) / cv_folds
               
           except Exception as e:
               print(f"\nâš ï¸� Error in {model_name} training: {str(e)}")
               continue
       
       # Record timing
       end_time = time.time()
       training_time = end_time - start_time
       timing_info[model_name] = training_time
       
       # Check if we have any successful folds
       if len(rmse_scores) > 0:
           # Calculate average metrics
           mean_rmse = np.mean(rmse_scores)
           mean_mae = np.mean(mae_scores)
           mean_r2 = np.mean(r2_scores)
           
           print(f"\nğŸ“Š {model_name} Cross-Validation Results:")
           print(f"  Mean RMSE: {mean_rmse:.4f}")
           print(f"  Mean MAE: {mean_mae:.4f}")
           print(f"  Mean RÂ²: {mean_r2:.4f}")
           print(f"  Training time: {training_time:.2f} seconds")
           
           # Store results
           self.model_results.append({
               'model': model_name,
               'rmse': mean_rmse,
               'mae': mean_mae,
               'r2': mean_r2,
               'predictions': test_predictions,
               'oof_predictions': oof_predictions,
               'fold_models': fold_models,
               'training_time': training_time,
               'weight': model_info['weight']
           })
           
           oof_preds[model_name] = oof_predictions
           test_preds[model_name] = test_predictions
           
           # Save model for later use
           try:
               # For ensembling, save a "full" model trained on all data
               full_model = copy.deepcopy(model_info['model'])
               
               if model_name == 'lightgbm':
                   full_model.fit(
                       self.X_train, self.y_train,
                       eval_metric='mae'
                   )
               elif model_name == 'xgboost':
                   full_model.fit(
                       self.X_train, self.y_train
                   )
               elif model_name == 'catboost':
                   full_model.fit(
                       self.X_train, self.y_train,
                       verbose=False
                   )
               else:
                   # Standard sklearn API
                   full_model.fit(self.X_train, self.y_train)
               
               # Save trained model
               model_path = os.path.join(self.models_path, f'{model_name}_model.pkl')
               joblib.dump(full_model, model_path)
               
               # Store in best_models dictionary
               self.best_models[model_name] = full_model
               
           except Exception as e:
               print(f"âš ï¸� Error saving {model_name} model: {str(e)}")
       else:
           print(f"\nâš ï¸� No successful folds for {model_name}.")
   
   # Create advanced ensemble model based on the specified strategy
   if len(self.model_results) >= 2:
       print("\nğŸ“‹ Building ensemble model...")
       
       if ensemble_type == 'stacking':
           # Stacking: Train a meta-model using OOF predictions as features
           print("\nğŸ”§ Creating stacking ensemble...")
           
           # Create a dataframe of oof predictions
           if oof_preds:
               meta_train = np.column_stack([oof_preds[result['model']] for result in self.model_results])
               meta_test = np.column_stack([test_preds[result['model']] for result in self.model_results])
               
               # Train meta-models with different algorithms
               meta_models = {
                   'ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
                   'huber': HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000),
                   'gbm': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
               }
               
               meta_predictions = {}
               
               for meta_name, meta_model in meta_models.items():
                   print(f"  Training {meta_name} meta-model...")
                   
                   # K-fold validation for the meta-model
                   meta_oof = np.zeros(len(meta_train))
                   meta_test_preds = np.zeros(len(meta_test))
                   
                   for train_idx, val_idx in kf.split(meta_train):
                       # Split data
                       X_meta_train, X_meta_val = meta_train[train_idx], meta_train[val_idx]
                       y_meta_train, y_meta_val = self.y_train[train_idx], self.y_train[val_idx]
                       
                       # Train meta-model
                       meta_model.fit(X_meta_train, y_meta_train)
                       
                       # Make predictions
                       meta_oof[val_idx] = meta_model.predict(X_meta_val)
                       meta_test_preds += meta_model.predict(meta_test) / cv_folds
                   
                   # Evaluate meta-model
                   meta_mae = mean_absolute_error(self.y_train, meta_oof)
                   print(f"    Meta-model MAE: {meta_mae:.4f}")
                   
                   # Store predictions
                   meta_predictions[meta_name] = meta_test_preds
               
               # Final meta-model trained on all data
               best_meta_name = 'ridge'  # Default
               
               # Find best meta-model based on validation performance
               meta_maes = {}
               for meta_name in meta_models:
                   meta_oof = meta_oof = np.zeros(len(meta_train))
                   
                   for train_idx, val_idx in kf.split(meta_train):
                       X_meta_train, X_meta_val = meta_train[train_idx], meta_train[val_idx]
                       y_meta_train, y_meta_val = self.y_train[train_idx], self.y_train[val_idx]
                       
                       meta_models[meta_name].fit(X_meta_train, y_meta_train)
                       meta_oof[val_idx] = meta_models[meta_name].predict(X_meta_val)
                   
                   meta_maes[meta_name] = mean_absolute_error(self.y_train, meta_oof)
               
               best_meta_name = min(meta_maes, key=meta_maes.get)
               print(f"  Best meta-model: {best_meta_name} (MAE: {meta_maes[best_meta_name]:.4f})")
               
               # Train the best meta-model on all data
               final_meta_model = copy.deepcopy(meta_models[best_meta_name])
               final_meta_model.fit(meta_train, self.y_train)
               
               # Make final meta-predictions
               meta_predictions = final_meta_model.predict(meta_test)
               
               # Store the meta-model
               self.best_models['meta_model'] = final_meta_model
               
               # Save feature names for the meta-model
               self.best_models['meta_features'] = [result['model'] for result in self.model_results]
               
               # Store stacking ensemble predictions
               self.model_results.append({
                   'model': 'stacking_ensemble',
                   'rmse': 0.0,  # Not directly available
                   'mae': meta_maes[best_meta_name],
                   'r2': 1.0,    # Give it the highest weight
                   'predictions': meta_predictions,
                   'training_time': 0.0,
                   'weight': 2.0  # Higher weight for ensemble
               })
               
               print("\nâœ… Stacking ensemble created successfully")
           else:
               print("âš ï¸� Cannot create stacking ensemble: no OOF predictions available")
       
       elif ensemble_type == 'blending':
           # Blending: Weighted average of base model predictions
           print("\nğŸ”§ Creating blending ensemble...")
           
           # Get weights for each model
           weights = []
           model_names = []
           
           # Use RÂ² scores to determine weights (avoid negative weights)
           for result in self.model_results:
               # Adjust weight by model performance and predefined model importance
               r2_weight = max(0, result['r2']) if result['r2'] < 1.0 else 0.5
               mae_weight = 1 / (result['mae'] + 1e-5)  # Lower MAE = higher weight
               
               # Combine metrics with model's base weight
               weight = (r2_weight + mae_weight) * result['weight']
               weights.append(weight)
               model_names.append(result['model'])
           
           # Normalize weights
           if sum(weights) > 0:
               weights = [w / sum(weights) for w in weights]
           else:
               # If all weights are 0, use equal weights
               weights = [1.0 / len(self.model_results)] * len(self.model_results)
           
           # Print blending weights
           print("  Blending weights:")
           for model_name, weight in zip(model_names, weights):
               print(f"    {model_name}: {weight:.4f}")
           
           # Create blended predictions
           blend_predictions = np.zeros(len(self.X_test))
           for i, result in enumerate(self.model_results):
               blend_predictions += result['predictions'] * weights[i]
           
           # Store blending ensemble predictions
           self.model_results.append({
               'model': 'blending_ensemble',
               'rmse': 0.0,  # Not directly available
               'mae': 0.0,   # Not directly available
               'r2': 1.0,    # Give it the highest weight
               'predictions': blend_predictions,
               'training_time': 0.0,
               'weight': 2.0  # Higher weight for ensemble
           })
           
           print("\nâœ… Blending ensemble created successfully")
       
       elif ensemble_type == 'voting':
           # Voting: Simple average of base model predictions
           print("\nğŸ”§ Creating voting ensemble...")
           
           # Calculate simple average of predictions
           vote_predictions = np.zeros(len(self.X_test))
           for result in self.model_results:
               vote_predictions += result['predictions'] / len(self.model_results)
           
           # Store voting ensemble predictions
           self.model_results.append({
               'model': 'voting_ensemble',
               'rmse': 0.0,  # Not directly available
               'mae': 0.0,   # Not directly available
               'r2': 1.0,    # Give it the highest weight
               'predictions': vote_predictions,
               'training_time': 0.0,
               'weight': 2.0  # Higher weight for ensemble
           })
           
           print("\nâœ… Voting ensemble created successfully")
       
       else:
           print(f"âš ï¸� Unknown ensemble type: {ensemble_type}. Using blending as fallback.")
           # Implement fallback to blending
   
   # Set predictions based on best ensemble or model
   if self.model_results:
       # Prioritize ensemble predictions if available
       ensemble_results = [result for result in self.model_results 
                         if 'ensemble' in result['model']]
       
       if ensemble_results:
           # Use the first ensemble method
           self.predictions = ensemble_results[0]['predictions']
           print(f"\nâœ… Using {ensemble_results[0]['model']} predictions")
       else:
           # Use best single model based on MAE
           best_model = min(self.model_results, key=lambda x: x['mae'])
           self.predictions = best_model['predictions']
           print(f"\nâœ… Using {best_model['model']} predictions (best MAE: {best_model['mae']:.4f})")
       
       # If we applied a transformation to the target, inverse transform the predictions
       if hasattr(self, 'inverse_transform_fn'):
           print("\nğŸ“‹ Applying inverse transformation to predictions...")
           self.predictions = self.inverse_transform_fn(self.predictions)
       
       print("\nâœ… Predictions created successfully")
   else:
       print("\nâš ï¸� No models were successfully trained. Cannot create predictions.")
   
   return self

def create_submission(self, filename='submission.csv'):
   """
   Create a submission file with the predictions.
   """
   print("\n" + "="*80)
   print("ğŸ“Š CREATING SUBMISSION")
   print("="*80)
   
   if self.predictions is None:
       print("â�Œ No predictions found. Please run train_advanced_models() first.")
       return self
   
   if self.test_ids is None:
       print("â�Œ No test IDs found. Please run preprocess_data() first.")
       return self
   
   # Create submission DataFrame
   self.submission = pd.DataFrame({
       'id': self.test_ids,
       'Price': self.predictions
   })
   
   # Check for invalid predictions
   invalid_count = np.sum(~np.isfinite(self.submission['Price']))
   if invalid_count > 0:
       print(f"âš ï¸� Found {invalid_count} invalid predictions (NaN/Inf). Replacing with mean.")
       valid_mean = np.mean(self.submission['Price'][np.isfinite(self.submission['Price'])])
       self.submission['Price'] = np.where(np.isfinite(self.submission['Price']), 
                                      self.submission['Price'], valid_mean)
   
   # Check for negative prices
   negative_count = np.sum(self.submission['Price'] < 0)
   if negative_count > 0:
       print(f"âš ï¸� Found {negative_count} negative price predictions. Replacing with abs value.")
       self.submission['Price'] = np.abs(self.submission['Price'])
   
   # Save submission
   submission_path = os.path.join(self.output_path, filename)
   self.submission.to_csv(submission_path, index=False)
   
   print(f"âœ… Submission saved to {submission_path}")
   print("\nSubmission Preview:")
   print(self.submission.head())
   
   # Basic statistics
   print("\nğŸ“Š Prediction Statistics:")
   print(f"Count: {len(self.submission)}")
   print(f"Min: {self.submission['Price'].min():.4f}")
   print(f"Max: {self.submission['Price'].max():.4f}")
   print(f"Mean: {self.submission['Price'].mean():.4f}")
   print(f"Median: {self.submission['Price'].median():.4f}")
   print(f"Std Dev: {self.submission['Price'].std():.4f}")
   
   return self

def analyze_feature_importance(self):
   """
   Analyze and visualize feature importance from the trained models.
   """
   print("\n" + "="*80)
   print("ğŸ“Š FEATURE IMPORTANCE ANALYSIS")
   print("="*80)
   
   if not hasattr(self, 'best_models') or not self.best_models:
       print("â�Œ No trained models found. Please run train_advanced_models() first.")
       return self
   
   # Check which models have feature importance
   models_with_importance = {}
   
   for model_name, model in self.best_models.items():
       if hasattr(model, 'feature_importances_'):
           models_with_importance[model_name] = model
   
   if not models_with_importance:
       print("â�Œ No models with feature importance found.")
       return self
   
   print(f"\nğŸ“‹ Analyzing feature importance from {len(models_with_importance)} models...")
   
   # Combined feature importance across models
   feature_importance_dict = {}
   
   for model_name, model in models_with_importance.items():
       # For tree-based models
       if hasattr(model, 'feature_importances_'):
           if hasattr(model, 'feature_name_'):
               # For LightGBM
               features = model.feature_name_
           else:
               # For other models
               features = self.X_train.columns
           
           importances = model.feature_importances_
           
           # Create dataframe for this model
           imp_df = pd.DataFrame({
               'Feature': features,
               'Importance': importances
           }).sort_values('Importance', ascending=False)
           
           # Normalize importance scores
           imp_df['Importance_Norm'] = imp_df['Importance'] / imp_df['Importance'].max()
           
           # Store normalized importance
           feature_importance_dict[model_name] = dict(zip(imp_df['Feature'], imp_df['Importance_Norm']))
           
           # Plot feature importance for this model
           plt.figure(figsize=(12, 10))
           sns.barplot(x='Importance', y='Feature', data=imp_df.head(20))
           plt.title(f'Feature Importance - {model_name}')
           plt.tight_layout()
           plt.savefig(os.path.join(self.plots_path, f'{model_name}_feature_importance.png'))
           plt.close()
   
   # Calculate combined feature importance across models
   combined_importance = {}
   
   for feature in self.X_train.columns:
       # Get importance from each model that has this feature
       importances = [model_imps.get(feature, 0) for model_imps in feature_importance_dict.values()]
       
       # Average importance across models
       if importances:
           combined_importance[feature] = np.mean(importances)
   
   # Create dataframe of combined importance
   combined_df = pd.DataFrame({
       'Feature': list(combined_importance.keys()),
       'Importance': list(combined_importance.values())
   }).sort_values('Importance', ascending=False)
   
   # Plot combined feature importance
   plt.figure(figsize=(12, 10))
   sns.barplot(x='Importance', y='Feature', data=combined_df.head(20))
   plt.title('Combined Feature Importance')
   plt.tight_layout()
   plt.savefig(os.path.join(self.plots_path, 'combined_feature_importance.png'))
   plt.close()
   
   # SHAP analysis for the most important model
   try:
       print("\nğŸ“‹ Performing SHAP analysis...")
       
       # Choose a model for SHAP analysis (pick the first tree-based model)
       shap_model_name = next(iter(models_with_importance))
       shap_model = models_with_importance[shap_model_name]
       
       # Create SHAP explainer
       explainer = shap.TreeExplainer(shap_model)
       
       # Sample from the dataset for faster computation
       sample_size = min(1000, len(self.X_train))
       X_sample = self.X_train.sample(sample_size, random_state=RANDOM_STATE)
       
       # Calculate SHAP values
       shap_values = explainer.shap_values(X_sample)
       
       # Summary plot
       plt.figure(figsize=(12, 10))
       shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
       plt.title(f'SHAP Feature Importance - {shap_model_name}')
       plt.tight_layout()
       plt.savefig(os.path.join(self.plots_path, 'shap_feature_importance.png'))
       plt.close()
       
       # SHAP dependence plots for top features
       top_features = combined_df.head(5)['Feature'].tolist()
       
       for feature in top_features:
           if feature in X_sample.columns:
               plt.figure(figsize=(10, 6))
               shap.dependence_plot(feature, shap_values, X_sample, show=False)
               plt.title(f'SHAP Dependence Plot - {feature}')
               plt.tight_layout()
               plt.savefig(os.path.join(self.plots_path, f'shap_dependence_{feature}.png'))
               plt.close()
   
   except Exception as e:
       print(f"âš ï¸� Error in SHAP analysis: {str(e)}")
   
   # Store combined feature importance
   self.feature_importance = combined_df
   
   # Print top features
   print("\nTop 20 most important features:")
   for i, (feature, importance) in enumerate(zip(combined_df['Feature'].head(20), combined_df['Importance'].head(20))):
       print(f"{i+1}. {feature}: {importance:.4f}")
   
   return self

def run_full_pipeline(self, train_data=None, test_data=None, original_data=None, optimize=True, n_trials=50):
   """
   Run the complete enhanced modeling pipeline.
   """
   print("\n" + "="*80)
   print("ğŸš€ RUNNING ENHANCED MODELING PIPELINE")
   print("="*80)
   
   # Check package versions for compatibility
   try:
       import pkg_resources
       
       def check_version(package, min_version=None):
           version = pkg_resources.get_distribution(package).version
           print(f"Using {package} version: {version}")
           if min_version:
               version_tuple = tuple(map(int, version.split('.')))
               min_version_tuple = tuple(map(int, min_version.split('.')))
               if version_tuple < min_version_tuple:
                   print(f"âš ï¸� Warning: {package} version {version} is older than recommended {min_version}")
           return version
       
       # Check versions of key packages
       check_version('lightgbm', '3.3.0')
       check_version('xgboost', '1.5.0')
       check_version('catboost', '1.0.0')
       check_version('scikit-learn', '1.0.0')
       check_version('optuna', '2.10.0')
       check_version('numpy', '1.20.0')
       check_version('pandas', '1.3.0')
   except Exception as e:
       print(f"âš ï¸� Warning: Could not check package versions: {str(e)}")
   
   try:
       print("\nğŸ“‹ Executing pipeline steps in sequence...")
       
       # Step 1: Load data
       self.load_data(train_data, test_data, original_data)
       
       # Step 2: Analyze data structure
       self.analyze_data_structure()
       
       # Step 3: Analyze target variable
       self.analyze_target_variable()
       
       # Step 4: Transform target variable
       self.transform_target()
       
       # Step 5: Preprocess data
       self.preprocess_data(handle_outliers=True, use_original_data=True)
       
       # Step 6: Engineer features
       self.engineer_features()
       
       # Step 7: Prepare for initial modeling
       self.prepare_for_modeling()
       
       # Step 8: Feature selection
       self.select_features(method='combined', n_features=60)
       
       # Step 9: Prepare for modeling again with selected features
       self.prepare_for_modeling()
       
       # Step 10: Train models
       try:
           self.train_advanced_models(use_optuna=optimize, n_trials=n_trials, ensemble_type='blending')
       except Exception as model_error:
           print(f"\nâ�Œ Error in model training: {str(model_error)}")
           print("âš ï¸� Attempting to continue with a simplified model approach...")
           
           # Fallback to a simple model approach
           try:
               from sklearn.ensemble import GradientBoostingRegressor
               
               print("\nğŸ“‹ Training fallback Gradient Boosting model...")
               model = GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE)
               model.fit(self.X_train, self.y_train)
               
               # Generate predictions
               self.predictions = model.predict(self.X_test)
               
               # If we applied a transformation to the target, inverse transform the predictions
               if hasattr(self, 'inverse_transform_fn'):
                   print("ğŸ“‹ Applying inverse transformation to predictions...")
                   self.predictions = self.inverse_transform_fn(self.predictions)
               
               print("âœ… Fallback model trained successfully")
           except Exception as fallback_error:
               print(f"\nâ�Œ Fallback model also failed: {str(fallback_error)}")
               # Use a very simple approach - mean prediction
               if hasattr(self, 'y_train') and self.y_train is not None:
                   mean_target = np.mean(self.y_train)
                   if hasattr(self, 'inverse_transform_fn'):
                       mean_target = self.inverse_transform_fn(mean_target)
                   self.predictions = np.ones(len(self.X_test)) * mean_target
                   print(f"âš ï¸� Using constant mean prediction: {mean_target:.4f}")
               else:
                   # Hard-coded fallback if nothing else works
                   self.predictions = np.ones(len(self.X_test)) * 80.0
                   print("âš ï¸� Using hard-coded constant prediction")
       
       # Step 11: Create submission
       self.create_submission()
       
       # Step 12: Analyze feature importance
       if hasattr(self, 'best_models') and self.best_models:
           self.analyze_feature_importance()
       
       print("\nâœ… Enhanced pipeline completed successfully!")
       
       return self.submission
       
   except Exception as e:
       print(f"\nâ�Œ Error in pipeline: {str(e)}")
       
       # Create fallback submission if possible
       try:
           if hasattr(self, 'test_ids') and self.test_ids is not None:
               # Use mean of training data or a constant value
               if hasattr(self, 'train_data') and self.target in self.train_data.columns:
                   mean_price = self.train_data[self.target].mean()
                   if hasattr(self, 'inverse_transform_fn'):
                       mean_price = self.inverse_transform_fn(mean_price)
               else:
                   mean_price = 80.0  # Fallback constant
               
               fallback_submission = pd.DataFrame({
                   'id': self.test_ids,
                   'Price': np.ones(len(self.test_ids)) * mean_price
               })
               
               fallback_path = os.path.join(self.output_path, 'fallback_submission.csv')
               fallback_submission.to_csv(fallback_path, index=False)
               
               print(f"\nâš ï¸� Created fallback submission with mean price: {mean_price:.2f}")
               print(f"Saved to: {fallback_path}")
               
               return fallback_submission
           else:
               print("\nâ�Œ Could not create fallback submission. No test IDs available.")
               return None
       except Exception as fallback_error:
           print(f"\nâ�Œ Could not create fallback submission: {str(fallback_error)}")
           return None

# Example usage
if __name__ == "__main__":
   # Create the enhanced modeler
   modeler = EnhancedBagPriceModeler()
   
   # Run the full advanced pipeline
   submission = modeler.run_full_pipeline(optimize=True, n_trials=30)


# # =====================================================
# # ADVANCED BAG PRICE PREDICTION - OPTIMIZATION STRATEGIES
# # =====================================================
# # This module provides enhanced modeling techniques to improve prediction performance

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
# from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
# from sklearn.svm import SVR
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_regression
# import lightgbm as lgb
# import catboost as cb
# import xgboost as xgb
# import optuna
# from scipy import stats
# import os
# import warnings
# import time
# from tqdm.notebook import tqdm

# # Suppress warnings
# warnings.filterwarnings('ignore')

# # Set random seed for reproducibility
# RANDOM_STATE = 42
# np.random.seed(RANDOM_STATE)

# class AdvancedBagPriceModeler:
#     """
#     Enhanced modeling pipeline with advanced optimization techniques
#     for the Student Bag Price Prediction competition.
#     """
    
#     def __init__(self, base_path='/kaggle/input/', output_path='/kaggle/working/'):
#         """Initialize the advanced modeler with paths and settings."""
#         self.base_path = base_path
#         self.output_path = output_path
#         self.train_data = None
#         self.test_data = None
#         self.original_data = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.test_ids = None
#         self.categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
#                                     'Waterproof', 'Style', 'Color']
#         self.numerical_features = ['Weight Capacity (kg)', 'Compartments']
#         self.target = 'Price'
#         self.model_results = []
#         self.predictions = None
#         self.submission = None
#         self.best_features = None
        
#         # Create output directory if it doesn't exist
#         os.makedirs(output_path, exist_ok=True)
    
#     def load_data(self, train_data=None, test_data=None, original_data=None):
#         """Load or set datasets for modeling."""
#         # If data is provided directly, use it
#         if train_data is not None and test_data is not None:
#             self.train_data = train_data.copy()
#             self.test_data = test_data.copy()
#             if original_data is not None:
#                 self.original_data = original_data.copy()
            
#             print(f"âœ… Using provided datasets")
#             print(f"Train data shape: {self.train_data.shape}")
#             print(f"Test data shape: {self.test_data.shape}")
#             if self.original_data is not None:
#                 print(f"Original data shape: {self.original_data.shape}")
            
#             return self
        
#         # Otherwise, load from files
#         try:
#             print("ğŸ”� Loading datasets from files...")
            
#             # Define paths
#             competition_path = os.path.join(self.base_path, 'playground-series-s5e2')
#             original_path = os.path.join(self.base_path, 'student-bag-price-prediction-dataset')
            
#             # Load the datasets
#             self.train_data = pd.read_csv(os.path.join(competition_path, 'train.csv'))
#             self.test_data = pd.read_csv(os.path.join(competition_path, 'test.csv'))
            
#             try:
#                 self.original_data = pd.read_csv(os.path.join(original_path, 
#                                                 'Noisy_Student_Bag_Price_Prediction_Dataset.csv'))
#             except FileNotFoundError:
#                 print("âš ï¸� Original dataset not found.")
#                 self.original_data = None
            
#             print(f"âœ… Train data: {self.train_data.shape}")
#             print(f"âœ… Test data: {self.test_data.shape}")
#             if self.original_data is not None:
#                 print(f"âœ… Original data: {self.original_data.shape}")
            
#             return self
            
#         except Exception as e:
#             print(f"â�Œ Error loading datasets: {str(e)}")
#             return self
    
#     def analyze_target_variable(self):
#         """
#         Perform deep analysis of the target variable to guide transformation strategy.
#         """
#         print("\n" + "="*80)
#         print("ğŸ”� TARGET VARIABLE ANALYSIS")
#         print("="*80)
        
#         if self.train_data is None or self.target not in self.train_data.columns:
#             print("â�Œ Target variable not found. Please load data first.")
#             return self
        
#         # Analyze target distribution
#         target_values = self.train_data[self.target].values
        
#         # Basic statistics
#         print(f"\nTarget Variable: {self.target}")
#         print(f"Min: {np.min(target_values):.4f}")
#         print(f"Max: {np.max(target_values):.4f}")
#         print(f"Mean: {np.mean(target_values):.4f}")
#         print(f"Median: {np.median(target_values):.4f}")
#         print(f"Std Dev: {np.std(target_values):.4f}")
#         print(f"Skewness: {stats.skew(target_values):.4f}")
#         print(f"Kurtosis: {stats.kurtosis(target_values):.4f}")
        
#         # Visualization
#         plt.figure(figsize=(12, 8))
        
#         # Original distribution
#         plt.subplot(2, 2, 1)
#         sns.histplot(target_values, kde=True)
#         plt.title("Original Target Distribution")
        
#         # Log transformation
#         plt.subplot(2, 2, 2)
#         log_target = np.log1p(target_values)
#         sns.histplot(log_target, kde=True)
#         plt.title(f"Log-Transformed Target (Skew: {stats.skew(log_target):.4f})")
        
#         # Square root transformation
#         plt.subplot(2, 2, 3)
#         sqrt_target = np.sqrt(target_values)
#         sns.histplot(sqrt_target, kde=True)
#         plt.title(f"Sqrt-Transformed Target (Skew: {stats.skew(sqrt_target):.4f})")
        
#         # Box-Cox transformation
#         plt.subplot(2, 2, 4)
#         try:
#             boxcox_target, lambda_param = stats.boxcox(target_values)
#             sns.histplot(boxcox_target, kde=True)
#             plt.title(f"Box-Cox-Transformed Target (Î»={lambda_param:.4f}, Skew: {stats.skew(boxcox_target):.4f})")
#             print(f"Box-Cox Lambda: {lambda_param:.4f}")
#             # We'll store lambda for later use
#             self.boxcox_lambda = lambda_param
#         except Exception as e:
#             plt.title("Box-Cox Transformation Failed")
#             print(f"Box-Cox transformation failed: {str(e)}")
#             self.boxcox_lambda = None
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_path, 'target_transformations.png'))
#         plt.show()
        
#         # Determine best transformation strategy
#         skewness_original = stats.skew(target_values)
#         skewness_log = stats.skew(log_target)
#         skewness_sqrt = stats.skew(sqrt_target)
        
#         transformations = [
#             ("Original", skewness_original),
#             ("Log", skewness_log),
#             ("Square Root", skewness_sqrt)
#         ]
        
#         if self.boxcox_lambda is not None:
#             transformations.append(("Box-Cox", stats.skew(boxcox_target)))
        
#         best_transform = min(transformations, key=lambda x: abs(x[1]))
#         print(f"\nBest transformation: {best_transform[0]} (Skewness: {best_transform[1]:.4f})")
        
#         self.target_transform = best_transform[0]
#         return self
    
#     def preprocess_data(self, handle_outliers=True, use_original_data=True):
#         """
#         Enhanced preprocessing with multiple strategies for handling missing values and outliers.
#         """
#         print("\n" + "="*80)
#         print("ğŸ”§ ENHANCED DATA PREPROCESSING")
#         print("="*80)
        
#         if self.train_data is None or self.test_data is None:
#             print("â�Œ No data loaded. Please run load_data() first.")
#             return self
        
#         print("\nğŸ“‹ Handling missing values and merging datasets...")
        
#         # Save test IDs before any processing
#         self.test_ids = self.test_data['id'].values
        
#         # Make copies to avoid modifying the original data
#         train_df = self.train_data.copy()
#         test_df = self.test_data.copy()
        
#         # Drop rows with missing target from original data if available
#         if use_original_data and self.original_data is not None:
#             original_df = self.original_data.copy()
#             original_df = original_df.dropna(subset=[self.target])
            
#             # Combine original data with train data to increase training dataset size
#             train_df = pd.concat([train_df, original_df], axis=0).reset_index(drop=True)
#             print(f"Combined training data shape: {train_df.shape}")
        
#         # Update categorical and numerical features based on available columns
#         self.categorical_features = [col for col in self.categorical_features if col in train_df.columns]
#         self.numerical_features = [col for col in self.numerical_features if col in train_df.columns]
        
#         # More sophisticated missing value imputation
#         print("\nğŸ“‹ Advanced missing value imputation...")
        
#         # For categorical features, impute using mode
#         for feature in self.categorical_features:
#             if feature in train_df.columns:
#                 mode_value = train_df[feature].mode()[0]
#                 train_df[feature].fillna(mode_value, inplace=True)
#                 if feature in test_df.columns:
#                     test_df[feature].fillna(mode_value, inplace=True)
        
#         # For numerical features, use median (more robust than mean)
#         for feature in self.numerical_features:
#             if feature in train_df.columns:
#                 median_value = train_df[feature].median()
#                 train_df[feature].fillna(median_value, inplace=True)
#                 if feature in test_df.columns:
#                     test_df[feature].fillna(median_value, inplace=True)
        
#         print("âœ… Missing values handled")
        
#         # Handle outliers in training data using multiple strategies
#         if handle_outliers:
#             print("\nğŸ“‹ Advanced outlier handling...")
            
#             initial_train_rows = len(train_df)
            
#             # Strategy 1: Remove outliers from numerical features using Winsorization
#             # This clips extreme values rather than removing entire rows
#             for feature in self.numerical_features:
#                 if feature in train_df.columns:
#                     q1 = train_df[feature].quantile(0.01)  # Less aggressive than standard 0.25
#                     q3 = train_df[feature].quantile(0.99)  # Less aggressive than standard 0.75
#                     iqr = q3 - q1
                    
#                     lower_bound = q1 - 1.5 * iqr
#                     upper_bound = q3 + 1.5 * iqr
                    
#                     # Apply Winsorization instead of removing (clip instead of filter)
#                     train_df[feature] = train_df[feature].clip(lower=lower_bound, upper=upper_bound)
            
#             # Also handle price outliers - use winsorization instead of removal
#             if self.target in train_df.columns:
#                 q1 = train_df[self.target].quantile(0.01)
#                 q3 = train_df[self.target].quantile(0.99)
#                 iqr = q3 - q1
                
#                 lower_bound = q1 - 1.5 * iqr
#                 upper_bound = q3 + 1.5 * iqr
                
#                 train_df[self.target] = train_df[self.target].clip(lower=lower_bound, upper=upper_bound)
            
#             print(f"âœ… Applied Winsorization to handle outliers while preserving data")
#             print(f"Final training shape: {train_df.shape}")
        
#         # Update the processed data
#         self.train_data = train_df
#         self.test_data = test_df
        
#         return self
    
#     def transform_target(self):
#         """
#         Apply the best transformation to the target variable.
#         """
#         print("\nğŸ“‹ Transforming target variable...")
        
#         if not hasattr(self, 'target_transform'):
#             print("âš ï¸� No target transformation analysis performed. Running analysis...")
#             self.analyze_target_variable()
            
#         target_values = self.train_data[self.target].values
            
#         # Apply the selected transformation
#         if self.target_transform == "Log":
#             print("Applying log transformation to target")
#             self.train_data[f"{self.target}_original"] = self.train_data[self.target]
#             self.train_data[self.target] = np.log1p(self.train_data[self.target])
#             self.inverse_transform_fn = lambda x: np.expm1(x)
        
#         elif self.target_transform == "Square Root":
#             print("Applying square root transformation to target")
#             self.train_data[f"{self.target}_original"] = self.train_data[self.target]
#             self.train_data[self.target] = np.sqrt(self.train_data[self.target])
#             self.inverse_transform_fn = lambda x: x**2
        
#         elif self.target_transform == "Box-Cox" and hasattr(self, 'boxcox_lambda'):
#             print(f"Applying Box-Cox transformation to target (lambda={self.boxcox_lambda:.4f})")
#             self.train_data[f"{self.target}_original"] = self.train_data[self.target]
#             self.train_data[self.target] = stats.boxcox(self.train_data[self.target], self.boxcox_lambda)
#             # For back-transformation
#             self.inverse_transform_fn = lambda x: stats.inv_boxcox(x, self.boxcox_lambda)
        
#         else:
#             print("Using original target values (no transformation)")
#             self.inverse_transform_fn = lambda x: x
            
#         return self
    
#     def engineer_features(self):
#         """
#         Enhanced feature engineering with advanced techniques.
#         """
#         print("\n" + "="*80)
#         print("ğŸ”§ ADVANCED FEATURE ENGINEERING")
#         print("="*80)
        
#         if self.train_data is None or self.test_data is None:
#             print("â�Œ No data loaded. Please run preprocess_data() first.")
#             return self
        
#         print("\nğŸ“‹ Creating enhanced features...")
        
#         # Make copies to avoid modifying the input data
#         train_data = self.train_data.copy()
#         test_data = self.test_data.copy()
        
#         # Check what columns are available before creating features
#         has_brand = 'Brand' in train_data.columns
#         has_material = 'Material' in train_data.columns
#         has_size = 'Size' in train_data.columns
#         has_laptop = 'Laptop Compartment' in train_data.columns
#         has_waterproof = 'Waterproof' in train_data.columns
#         has_style = 'Style' in train_data.columns
#         has_color = 'Color' in train_data.columns
#         has_compartments = 'Compartments' in train_data.columns
#         has_weight = 'Weight Capacity (kg)' in train_data.columns
        
#         # Track created features for logging
#         created_features = []
        
#         # Group 1: Basic features and transformations
        
#         # 1. Binary Features
#         if has_laptop:
#             train_data['Has_Laptop_Compartment'] = train_data['Laptop Compartment'].map({'Yes': 1, 'No': 0})
#             test_data['Has_Laptop_Compartment'] = test_data['Laptop Compartment'].map({'Yes': 1, 'No': 0})
#             created_features.append('Has_Laptop_Compartment')
        
#         if has_waterproof:
#             train_data['Is_Waterproof'] = train_data['Waterproof'].map({'Yes': 1, 'No': 0})
#             test_data['Is_Waterproof'] = test_data['Waterproof'].map({'Yes': 1, 'No': 0})
#             created_features.append('Is_Waterproof')
        
#         # 2. Numerical transformations
#         if has_compartments:
#             # Log transform
#             train_data['Log_Compartments'] = np.log1p(train_data['Compartments'])
#             test_data['Log_Compartments'] = np.log1p(test_data['Compartments'])
#             created_features.append('Log_Compartments')
            
#             # Square and cube (polynomial features)
#             train_data['Compartments_Squared'] = train_data['Compartments'] ** 2
#             test_data['Compartments_Squared'] = test_data['Compartments'] ** 2
#             created_features.append('Compartments_Squared')
            
#             train_data['Compartments_Cubed'] = train_data['Compartments'] ** 3
#             test_data['Compartments_Cubed'] = test_data['Compartments'] ** 3
#             created_features.append('Compartments_Cubed')
        
#         if has_weight:
#             # Log transform
#             train_data['Log_Weight'] = np.log1p(train_data['Weight Capacity (kg)'])
#             test_data['Log_Weight'] = np.log1p(test_data['Weight Capacity (kg)'])
#             created_features.append('Log_Weight')
            
#             # Square and cube (polynomial features)
#             train_data['Weight_Squared'] = train_data['Weight Capacity (kg)'] ** 2
#             test_data['Weight_Squared'] = test_data['Weight Capacity (kg)'] ** 2
#             created_features.append('Weight_Squared')
            
#             train_data['Weight_Cubed'] = train_data['Weight Capacity (kg)'] ** 3
#             test_data['Weight_Cubed'] = test_data['Weight Capacity (kg)'] ** 3
#             created_features.append('Weight_Cubed')
            
#             # Square root transform
#             train_data['Sqrt_Weight'] = np.sqrt(train_data['Weight Capacity (kg)'])
#             test_data['Sqrt_Weight'] = np.sqrt(test_data['Weight Capacity (kg)'])
#             created_features.append('Sqrt_Weight')
        
#         # Group 2: Interaction Features
        
#         # 3. Categorical Interactions
#         if has_brand and has_material:
#             train_data['Brand_Material'] = train_data['Brand'] + '_' + train_data['Material']
#             test_data['Brand_Material'] = test_data['Brand'] + '_' + test_data['Material']
#             created_features.append('Brand_Material')
        
#         if has_brand and has_size:
#             train_data['Brand_Size'] = train_data['Brand'] + '_' + train_data['Size']
#             test_data['Brand_Size'] = test_data['Brand'] + '_' + test_data['Size']
#             created_features.append('Brand_Size')
        
#         if has_style and has_size:
#             train_data['Style_Size'] = train_data['Style'] + '_' + train_data['Size']
#             test_data['Style_Size'] = test_data['Style'] + '_' + test_data['Size']
#             created_features.append('Style_Size')
        
#         if has_color and has_style:
#             train_data['Color_Style'] = train_data['Color'] + '_' + train_data['Style']
#             test_data['Color_Style'] = test_data['Color'] + '_' + test_data['Style']
#             created_features.append('Color_Style')
        
#         if has_material and has_style:
#             train_data['Material_Style'] = train_data['Material'] + '_' + train_data['Style']
#             test_data['Material_Style'] = test_data['Material'] + '_' + test_data['Style']
#             created_features.append('Material_Style')
        
#         if has_material and has_waterproof:
#             train_data['Material_Waterproof'] = train_data['Material'] + '_' + train_data['Waterproof']
#             test_data['Material_Waterproof'] = test_data['Material'] + '_' + test_data['Waterproof']
#             created_features.append('Material_Waterproof')
        
#         # 4. Numeric-Numeric Interactions
#         if has_weight and has_compartments:
#             # Weight to compartments ratio
#             train_data['Weight_to_Compartments'] = train_data['Weight Capacity (kg)'] / (train_data['Compartments'] + 1)
#             test_data['Weight_to_Compartments'] = test_data['Weight Capacity (kg)'] / (test_data['Compartments'] + 1)
#             created_features.append('Weight_to_Compartments')
            
#             # Product (interaction)
#             train_data['Weight_Compartments_Interaction'] = train_data['Weight Capacity (kg)'] * train_data['Compartments']
#             test_data['Weight_Compartments_Interaction'] = test_data['Weight Capacity (kg)'] * test_data['Compartments']
#             created_features.append('Weight_Compartments_Interaction')
            
#             # Compartments per kg
#             train_data['Compartments_per_kg'] = train_data['Compartments'] / (train_data['Weight Capacity (kg)'] + 0.1)
#             test_data['Compartments_per_kg'] = test_data['Compartments'] / (test_data['Weight Capacity (kg)'] + 0.1)
#             created_features.append('Compartments_per_kg')
        
#         # Group 3: Target Encoding Features (with k-fold to avoid leakage)
#         if self.target in train_data.columns:
#             print("\nğŸ“‹ Creating target encoding features...")
            
#             # Define k-fold for target encoding
#             kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            
#             # List of categorical features to encode
#             categorical_for_encoding = [col for col in self.categorical_features 
#                                         if col in train_data.columns]
            
#             # Loop through each categorical column
#             for col in categorical_for_encoding:
#                 # Initialize the encoded column in train dataset
#                 train_data[f'{col}_Target_Mean'] = np.nan
                
#                 # Perform k-fold target encoding
#                 for train_idx, test_idx in kf.split(train_data):
#                     # Get target mean for each category in train fold
#                     target_means = train_data.iloc[train_idx].groupby(col)[self.target].mean()
#                     # Map these means to the test fold
#                     train_data.iloc[test_idx, train_data.columns.get_loc(f'{col}_Target_Mean')] = \
#                         train_data.iloc[test_idx][col].map(target_means)
                
#                 # Fill any missing values with the global mean
#                 global_mean = train_data[self.target].mean()
#                 train_data[f'{col}_Target_Mean'].fillna(global_mean, inplace=True)
                
#                 # For the test set, use the entire training data to compute target means
#                 target_means = train_data.groupby(col)[self.target].mean()
#                 test_data[f'{col}_Target_Mean'] = test_data[col].map(target_means)
#                 test_data[f'{col}_Target_Mean'].fillna(global_mean, inplace=True)
                
#                 created_features.append(f'{col}_Target_Mean')
                
#                 # Also add target standard deviation as a feature
#                 train_data[f'{col}_Target_Std'] = np.nan
                
#                 # Perform k-fold target encoding for standard deviation
#                 for train_idx, test_idx in kf.split(train_data):
#                     target_stds = train_data.iloc[train_idx].groupby(col)[self.target].std()
#                     train_data.iloc[test_idx, train_data.columns.get_loc(f'{col}_Target_Std')] = \
#                         train_data.iloc[test_idx][col].map(target_stds)
                
#                 # Fill any missing values with the global std
#                 global_std = train_data[self.target].std()
#                 train_data[f'{col}_Target_Std'].fillna(global_std, inplace=True)
                
#                 # For the test set, use the entire training data to compute target stds
#                 target_stds = train_data.groupby(col)[self.target].std()
#                 test_data[f'{col}_Target_Std'] = test_data[col].map(target_stds)
#                 test_data[f'{col}_Target_Std'].fillna(global_std, inplace=True)
                
#                 created_features.append(f'{col}_Target_Std')
        
#         # Group 4: Count and frequency features
#         print("\nğŸ“‹ Creating count and frequency features...")
        
#         for col in self.categorical_features:
#             if col in train_data.columns:
#                 # Value counts
#                 counts = train_data[col].value_counts()
#                 train_data[f'{col}_Count'] = train_data[col].map(counts)
#                 test_data[f'{col}_Count'] = test_data[col].map(counts)
#                 test_data[f'{col}_Count'].fillna(1, inplace=True)  # For categories not in train
#                 created_features.append(f'{col}_Count')
                
#                 # Convert to frequency
#                 train_data[f'{col}_Frequency'] = train_data[f'{col}_Count'] / len(train_data)
#                 test_data[f'{col}_Frequency'] = test_data[f'{col}_Count'] / len(train_data)
#                 created_features.append(f'{col}_Frequency')
        
#         # Group 5: Grouping and aggregation features
#         if 'Compartments' in train_data.columns:
#             # Convert to categorical bins
#             train_data['Compartments_Category'] = pd.cut(
#                 train_data['Compartments'], 
#                 bins=[0, 2, 5, 10, float('inf')], 
#                 labels=['Few', 'Moderate', 'Many', 'Very Many']
#             )
#             test_data['Compartments_Category'] = pd.cut(
#                 test_data['Compartments'], 
#                 bins=[0, 2, 5, 10, float('inf')], 
#                 labels=['Few', 'Moderate', 'Many', 'Very Many']
#             )
#             created_features.append('Compartments_Category')
        
#         if 'Weight Capacity (kg)' in train_data.columns:
#             # Create weight capacity bins
#             train_data['Weight_Category'] = pd.qcut(
#                 train_data['Weight Capacity (kg)'], 
#                 q=5, 
#                 labels=['Very Light', 'Light', 'Medium', 'Heavy', 'Very Heavy']
#             )
            
#             # Get bin edges for test data
#             qcut_bins = pd.qcut(train_data['Weight Capacity (kg)'], q=5, retbins=True)[1]
#             # Apply these bin edges to test data
#             test_data['Weight_Category'] = pd.cut(
#                 test_data['Weight Capacity (kg)'], 
#                 bins=qcut_bins, 
#                 labels=['Very Light', 'Light', 'Medium', 'Heavy', 'Very Heavy'],
#                 include_lowest=True
#             )
#             created_features.append('Weight_Category')
            
#             # Create weight capacity ratio
#             train_data['Weight_Capacity_Ratio'] = train_data['Weight Capacity (kg)'] / train_data['Weight Capacity (kg)'].max()
#             test_data['Weight_Capacity_Ratio'] = test_data['Weight Capacity (kg)'] / train_data['Weight Capacity (kg)'].max()
#             created_features.append('Weight_Capacity_Ratio')
        
#         print(f"âœ… Created {len(created_features)} new features")
        
#         # Update the processed data
#         self.train_data = train_data
#         self.test_data = test_data
        
#         return self
    
#     def select_features(self, method='mutual_info', n_features=None):
#         """
#         Perform feature selection to identify the most important features.
        
#         Args:
#             method: str, feature selection method ('mutual_info', 'forest', or 'recursive')
#             n_features: int, number of features to select (None for auto)
#         """
#         print("\n" + "="*80)
#         print("ğŸ”� FEATURE SELECTION")
#         print("="*80)
        
#         if self.X_train is None or self.y_train is None:
#             print("â�Œ No prepared data found. Please run prepare_for_modeling() first.")
#             return self
        
#         n_features = n_features or max(20, self.X_train.shape[1] // 4)
        
#         if method == 'mutual_info':
#             print(f"\nğŸ“‹ Using mutual information to select top {n_features} features...")
            
#             # Calculate mutual information scores
#             mi_scores = mutual_info_regression(self.X_train, self.y_train, random_state=RANDOM_STATE)
            
#             # Create a dataframe of features and their scores
#             mi_df = pd.DataFrame({
#                 'Feature': self.X_train.columns,
#                 'MI_Score': mi_scores
#             }).sort_values('MI_Score', ascending=False)
            
#             # Select top features
#             top_features = mi_df.head(n_features)['Feature'].tolist()
            
#             # Visualize feature importance
#             plt.figure(figsize=(10, 8))
#             sns.barplot(x='MI_Score', y='Feature', data=mi_df.head(min(20, n_features)))
#             plt.title(f'Top Features by Mutual Information Score')
#             plt.tight_layout()
#             plt.savefig(os.path.join(self.output_path, 'mutual_info_features.png'))
            
#         elif method == 'forest':
#             print(f"\nğŸ“‹ Using Random Forest to select top {n_features} features...")
            
#             # Train a Random Forest model
#             forest = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
#             forest.fit(self.X_train, self.y_train)
            
#             # Get feature importances
#             importances = forest.feature_importances_
            
#             # Create a dataframe of features and their importances
#             forest_df = pd.DataFrame({
#                 'Feature': self.X_train.columns,
#                 'Importance': importances
#             }).sort_values('Importance', ascending=False)
            
#             # Select top features
#             top_features = forest_df.head(n_features)['Feature'].tolist()
            
#             # Visualize feature importance
#             plt.figure(figsize=(10, 8))
#             sns.barplot(x='Importance', y='Feature', data=forest_df.head(min(20, n_features)))
#             plt.title(f'Top Features by Random Forest Importance')
#             plt.tight_layout()
#             plt.savefig(os.path.join(self.output_path, 'forest_feature_importance.png'))
            
#         elif method == 'recursive':
#             print(f"\nğŸ“‹ Using Recursive Feature Elimination...")
            
#             # Use a fast estimator for RFE
#             estimator = Ridge(alpha=1.0)
            
#             # Initialize RFECV
#             selector = RFECV(
#                 estimator=estimator,
#                 step=1,
#                 cv=5,
#                 scoring='neg_mean_absolute_error',
#                 min_features_to_select=5
#             )
            
#             # Fit the selector
#             selector.fit(self.X_train, self.y_train)
            
#             # Get selected features
#             selected_mask = selector.support_
#             top_features = self.X_train.columns[selected_mask].tolist()
            
#             # Visualize number of features vs. performance
#             plt.figure(figsize=(10, 6))
#             plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
#             plt.xlabel('Number of Features')
#             plt.ylabel('Negative MAE')
#             plt.title('Feature Selection using RFE with Cross-Validation')
#             plt.grid(True)
#             plt.savefig(os.path.join(self.output_path, 'rfecv_feature_selection.png'))
            
#             print(f"âœ… Selected {len(top_features)} features using RFECV")
            
#         else:
#             print(f"â�Œ Unknown feature selection method: {method}")
#             return self
        
#         # Store selected features
#         self.best_features = top_features
        
#         print(f"âœ… Selected {len(top_features)} features")
        
#         # Display top 10 features
#         print("\nTop 10 selected features:")
#         for i, feature in enumerate(top_features[:10]):
#             print(f"{i+1}. {feature}")
        
#         return self
    
#     def prepare_for_modeling(self):
#         """
#         Prepare datasets for modeling with advanced preprocessing.
#         """
#         print("\n" + "="*80)
#         print("ğŸ”§ PREPARING DATA FOR MODELING")
#         print("="*80)
        
#         if self.train_data is None or self.test_data is None:
#             print("â�Œ No data loaded. Please run engineer_features() first.")
#             return self
        
#         print("\nğŸ“‹ Encoding and scaling features...")
        
#         # Separate target variable
#         if self.target in self.train_data.columns:
#             self.y_train = self.train_data[self.target].values
#             self.X_train = self.train_data.drop(self.target, axis=1)
            
#             # Also drop the original target if we transformed it
#             if f"{self.target}_original" in self.X_train.columns:
#                 self.X_train = self.X_train.drop(f"{self.target}_original", axis=1)
#         else:
#             print("âš ï¸� Warning: Target variable not found in training data")
#             self.X_train = self.train_data.copy()
#             self.y_train = None
        
#         self.X_test = self.test_data.copy()
        
#         # Drop ID column if present
#         if 'id' in self.X_train.columns:
#             self.X_train = self.X_train.drop('id', axis=1)
        
#         if 'id' in self.X_test.columns:
#             self.X_test = self.X_test.drop('id', axis=1)
        
#         # Identify categorical and numerical columns
#         categorical_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
#         numerical_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
#         print(f"Categorical features: {len(categorical_cols)}")
#         print(f"Numerical features: {len(numerical_cols)}")
        
#         # Get dummies for categorical variables
#         self.X_train = pd.get_dummies(self.X_train, columns=categorical_cols, drop_first=True)
#         self.X_test = pd.get_dummies(self.X_test, columns=categorical_cols, drop_first=True)
        
#         # Align the encoded datasets to have the same columns
#         self.X_train, self.X_test = self.X_train.align(self.X_test, join='left', axis=1, fill_value=0)
        
#         # Handle any missing columns in test set
#         missing_cols = set(self.X_train.columns) - set(self.X_test.columns)
#         for col in missing_cols:
#             self.X_test[col] = 0
        
#         # Ensure the test set has the same columns in the same order
#         self.X_test = self.X_test[self.X_train.columns]
        
#         # Scale numerical features
#         numerical_cols = [col for col in numerical_cols if col in self.X_train.columns]
#         if len(numerical_cols) > 0:
#             # Use RobustScaler for better handling of outliers
#             scaler = RobustScaler()
#             self.X_train[numerical_cols] = scaler.fit_transform(self.X_train[numerical_cols])
#             self.X_test[numerical_cols] = scaler.transform(self.X_test[numerical_cols])
#             print(f"âœ… Scaled {len(numerical_cols)} numerical features using RobustScaler")
        
#         # Apply feature selection if available
#         if hasattr(self, 'best_features') and self.best_features:
#             print(f"\nğŸ“‹ Applying feature selection ({len(self.best_features)} features)...")
#             self.X_train = self.X_train[self.best_features]
#             self.X_test = self.X_test[self.best_features]
        
#         # Verify data looks good
#         print(f"\nFinal training data shape: {self.X_train.shape}")
#         print(f"Final test data shape: {self.X_test.shape}")
        
#         # Check for NaN values
#         train_nans = self.X_train.isna().sum().sum()
#         test_nans = self.X_test.isna().sum().sum()
        
#         if train_nans > 0 or test_nans > 0:
#             print(f"âš ï¸� Found NaN values: {train_nans} in training, {test_nans} in test")
#             print("Filling NaN values with 0")
#             self.X_train.fillna(0, inplace=True)
#             self.X_test.fillna(0, inplace=True)
        
#         return self
    
#     def optimize_catboost(self, n_trials=50):
#         """
#         Optimize CatBoost hyperparameters using Optuna.
        
#         Args:
#             n_trials: Number of Optuna trials
        
#         Returns:
#             dict: Optimized hyperparameters
#         """
#         print("\n" + "="*80)
#         print("ğŸ”§ OPTIMIZING CATBOOST HYPERPARAMETERS")
#         print("="*80)
        
#         if self.X_train is None or self.y_train is None:
#             print("â�Œ No prepared data found. Please run prepare_for_modeling() first.")
#             return {}
        
#         def objective(trial):
#             param = {
#                 'iterations': trial.suggest_int('iterations', 500, 3000),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#                 'depth': trial.suggest_int('depth', 4, 10),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
#                 'border_count': trial.suggest_int('border_count', 32, 255),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#                 'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
#                 'grow_policy': trial.suggest_categorical('grow_policy', 
#                                                        ['SymmetricTree', 'Depthwise', 'Lossguide']),
#                 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
#                 'verbose': False,
#                 'random_seed': RANDOM_STATE
#             }
            
#             # K-fold cross-validation
#             kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#             scores = []
            
#             for train_idx, valid_idx in kf.split(self.X_train):
#                 X_train_fold, X_valid_fold = self.X_train.iloc[train_idx], self.X_train.iloc[valid_idx]
#                 y_train_fold, y_valid_fold = self.y_train[train_idx], self.y_train[valid_idx]
                
#                 # Train model
#                 model = cb.CatBoostRegressor(**param)
#                 model.fit(
#                     X_train_fold, y_train_fold,
#                     eval_set=[(X_valid_fold, y_valid_fold)],
#                     early_stopping_rounds=50,
#                     verbose=False
#                 )
                
#                 # Predict and score
#                 y_pred = model.predict(X_valid_fold)
#                 score = mean_absolute_error(y_valid_fold, y_pred)
#                 scores.append(score)
            
#             # Return mean score
#             return np.mean(scores)
        
#         # Create study
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=n_trials)
        
#         # Get best parameters
#         best_params = study.best_params
#         best_score = study.best_value
        
#         print(f"\nâœ… Best MAE: {best_score:.4f}")
#         print("\nBest parameters:")
#         for param, value in best_params.items():
#             print(f"  {param}: {value}")
        
#         return best_params
    
#     def optimize_lightgbm(self, n_trials=50):
#         """
#         Optimize LightGBM hyperparameters using Optuna.
        
#         Args:
#             n_trials: Number of Optuna trials
        
#         Returns:
#             dict: Optimized hyperparameters
#         """
#         print("\n" + "="*80)
#         print("ğŸ”§ OPTIMIZING LIGHTGBM HYPERPARAMETERS")
#         print("="*80)
        
#         if self.X_train is None or self.y_train is None:
#             print("â�Œ No prepared data found. Please run prepare_for_modeling() first.")
#             return {}
        
#         def objective(trial):
#             param = {
#                 'objective': 'regression',
#                 'metric': 'mae',
#                 'verbosity': -1,
#                 'boosting_type': 'gbdt',
#                 'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#                 'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#                 'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#                 'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
#                 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
#                 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#                 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
#                 'max_depth': trial.suggest_int('max_depth', 3, 12),
#                 'random_state': RANDOM_STATE
#             }
            
#             # K-fold cross-validation
#             kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#             scores = []
            
#             for train_idx, valid_idx in kf.split(self.X_train):
#                 X_train_fold, X_valid_fold = self.X_train.iloc[train_idx], self.X_train.iloc[valid_idx]
#                 y_train_fold, y_valid_fold = self.y_train[train_idx], self.y_train[valid_idx]
                
#                 # Train model
#                 model = lgb.LGBMRegressor(**param)
#                 model.fit(
#                     X_train_fold, y_train_fold,
#                     eval_set=[(X_valid_fold, y_valid_fold)],
#                     callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
#                     eval_metric='mae'
#                     # Removed verbose parameter that was causing the error
#                 )
                
#                 # Predict and score
#                 y_pred = model.predict(X_valid_fold)
#                 score = mean_absolute_error(y_valid_fold, y_pred)
#                 scores.append(score)
            
#             # Return mean score
#             return np.mean(scores)
        
#         # Create study
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=n_trials)
        
#         # Get best parameters
#         best_params = study.best_params
#         best_score = study.best_value
        
#         print(f"\nâœ… Best MAE: {best_score:.4f}")
#         print("\nBest parameters:")
#         for param, value in best_params.items():
#             print(f"  {param}: {value}")
        
#         return best_params
    
#     def optimize_xgboost(self, n_trials=50):
#         """
#         Optimize XGBoost hyperparameters using Optuna.
        
#         Args:
#             n_trials: Number of Optuna trials
        
#         Returns:
#             dict: Optimized hyperparameters
#         """
#         print("\n" + "="*80)
#         print("ğŸ”§ OPTIMIZING XGBOOST HYPERPARAMETERS")
#         print("="*80)
        
#         if self.X_train is None or self.y_train is None:
#             print("â�Œ No prepared data found. Please run prepare_for_modeling() first.")
#             return {}
        
#         def objective(trial):
#             param = {
#                 'objective': 'reg:squarederror',
#                 'eval_metric': 'mae',
#                 'verbosity': 0,
#                 'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
#                 'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
#                 'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#                 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
#                 'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
#                 'max_depth': trial.suggest_int('max_depth', 3, 12),
#                 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#                 'random_state': RANDOM_STATE
#             }
            
#             # K-fold cross-validation
#             kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#             scores = []
            
#             for train_idx, valid_idx in kf.split(self.X_train):
#                 X_train_fold, X_valid_fold = self.X_train.iloc[train_idx], self.X_train.iloc[valid_idx]
#                 y_train_fold, y_valid_fold = self.y_train[train_idx], self.y_train[valid_idx]
                
#                 # Train model
#                 model = xgb.XGBRegressor(**param)
#                 model.fit(
#                     X_train_fold, y_train_fold,
#                     eval_set=[(X_valid_fold, y_valid_fold)],
#                     early_stopping_rounds=50,
#                     verbose=False
#                 )
                
#                 # Predict and score
#                 y_pred = model.predict(X_valid_fold)
#                 score = mean_absolute_error(y_valid_fold, y_pred)
#                 scores.append(score)
            
#             # Return mean score
#             return np.mean(scores)
        
#         # Create study
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=n_trials)
        
#         # Get best parameters
#         best_params = study.best_params
#         best_score = study.best_value
        
#         print(f"\nâœ… Best MAE: {best_score:.4f}")
#         print("\nBest parameters:")
#         for param, value in best_params.items():
#             print(f"  {param}: {value}")
        
#         return best_params
    
#     def train_advanced_models(self, use_optuna=True, n_trials=50, cv_folds=5):
#         """
#         Train multiple models with advanced configurations and stacking.
        
#         Args:
#             use_optuna: Whether to use Optuna for hyperparameter optimization
#             n_trials: Number of Optuna trials
#             cv_folds: Number of cross-validation folds
#         """
#         print("\n" + "="*80)
#         print("ğŸ”§ TRAINING ADVANCED MODELS")
#         print("="*80)
        
#         # Check if we have data prepared
#         if self.X_train is None or self.y_train is None:
#             print("â�Œ No prepared data found. Please run prepare_for_modeling() first.")
#             return self
        
#         # Define cross-validation strategy
#         kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
#         # Optimize hyperparameters if requested
#         if use_optuna:
#             print("\nğŸ“‹ Optimizing hyperparameters with Optuna...")
#             cb_params = self.optimize_catboost(n_trials=max(10, n_trials // 3))
#             lgb_params = self.optimize_lightgbm(n_trials=max(10, n_trials // 3))
#             xgb_params = self.optimize_xgboost(n_trials=max(10, n_trials // 3))
#         else:
#             # Use default hyperparameters
#             cb_params = {
#                 'iterations': 1000,
#                 'learning_rate': 0.05,
#                 'depth': 6,
#                 'l2_leaf_reg': 3,
#                 'verbose': 0,
#                 'random_seed': RANDOM_STATE
#             }
            
#             lgb_params = {
#                 'n_estimators': 1000,
#                 'learning_rate': 0.05,
#                 'num_leaves': 31,
#                 'subsample': 0.8,
#                 'colsample_bytree': 0.8,
#                 'random_state': RANDOM_STATE,
#                 'verbosity': -1
#             }
            
#             xgb_params = {
#                 'n_estimators': 1000,
#                 'learning_rate': 0.05,
#                 'max_depth': 6,
#                 'subsample': 0.8,
#                 'colsample_bytree': 0.8,
#                 'random_state': RANDOM_STATE,
#                 'verbosity': 0
#             }
        
#         # Models to train
#         print("\nğŸ“‹ Preparing advanced models...")
#         models = {
#             'catboost': {
#                 'model': cb.CatBoostRegressor(**cb_params),
#                 'params': cb_params
#             },
#             'lightgbm': {
#                 'model': lgb.LGBMRegressor(**lgb_params),
#                 'params': lgb_params
#             },
#             'xgboost': {
#                 'model': xgb.XGBRegressor(**xgb_params),
#                 'params': xgb_params
#             },
#             'huber': {
#                 'model': HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000),
#                 'params': {}
#             },
#             'ridge': {
#                 'model': Ridge(alpha=1.0, random_state=RANDOM_STATE),
#                 'params': {}
#             }
#         }
        
#         # Storage for model results
#         self.model_results = []
        
#         # Out-of-fold predictions for each model
#         oof_preds = {}
#         test_preds = {}
        
#         # Timing information
#         timing_info = {}
        
#         # Train each model
#         for model_name, model_info in models.items():
#             print(f"\nğŸ”§ Training {model_name}...")
#             start_time = time.time()
            
#             # Lists to store fold results
#             rmse_scores = []
#             mae_scores = []
#             r2_scores = []
#             oof_predictions = np.zeros(len(self.X_train))
#             test_predictions = np.zeros(len(self.X_test))
            
#             # Perform K-Fold Cross Validation
#             for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
#                 print(f"  Fold {fold+1}/{cv_folds}...", end='')
                
#                 # Split data
#                 X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
#                 y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
                
#                 # Train model
#                 model = model_info['model']
                
#                 try:
#                     if model_name == 'catboost':
#                         model.fit(
#                             X_fold_train, y_fold_train,
#                             eval_set=[(X_fold_val, y_fold_val)],
#                             early_stopping_rounds=50,
#                             verbose=False
#                         )
#                     elif model_name == 'lightgbm':
#                         model.fit(
#                             X_fold_train, y_fold_train,
#                             eval_set=[(X_fold_val, y_fold_val)],
#                             callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
#                             eval_metric='mae'
#                             # Removed verbose parameter that was causing the error
#                         )
#                     elif model_name == 'xgboost':
#                         model.fit(
#                             X_fold_train, y_fold_train,
#                             eval_set=[(X_fold_val, y_fold_val)],
#                             early_stopping_rounds=50,
#                             verbose=False
#                         )
#                     else:
#                         # Standard sklearn API
#                         model.fit(X_fold_train, y_fold_train)
                    
#                     # Predict on validation set
#                     val_preds = model.predict(X_fold_val)
#                     oof_predictions[val_idx] = val_preds
                    
#                     # Calculate metrics
#                     rmse = np.sqrt(mean_squared_error(y_fold_val, val_preds))
#                     mae = mean_absolute_error(y_fold_val, val_preds)
#                     r2 = r2_score(y_fold_val, val_preds)
                    
#                     rmse_scores.append(rmse)
#                     mae_scores.append(mae)
#                     r2_scores.append(r2)
                    
#                     print(f" RMSE: {rmse:.4f}, MAE: {mae:.4f}, RÂ²: {r2:.4f}")
                    
#                     # Predict on test data
#                     test_predictions += model.predict(self.X_test) / cv_folds
                    
#                 except Exception as e:
#                     print(f"\nâš ï¸� Error in {model_name} training: {str(e)}")
#                     continue
            
#             # Record timing
#             end_time = time.time()
#             training_time = end_time - start_time
#             timing_info[model_name] = training_time
            
#             # Check if we have any successful folds
#             if len(rmse_scores) > 0:
#                 # Calculate average metrics
#                 mean_rmse = np.mean(rmse_scores)
#                 mean_mae = np.mean(mae_scores)
#                 mean_r2 = np.mean(r2_scores)
                
#                 print(f"\nğŸ“Š {model_name} Cross-Validation Results:")
#                 print(f"  Mean RMSE: {mean_rmse:.4f}")
#                 print(f"  Mean MAE: {mean_mae:.4f}")
#                 print(f"  Mean RÂ²: {mean_r2:.4f}")
#                 print(f"  Training time: {training_time:.2f} seconds")
                
#                 # Store results
#                 self.model_results.append({
#                     'model': model_name,
#                     'rmse': mean_rmse,
#                     'mae': mean_mae,
#                     'r2': mean_r2,
#                     'predictions': test_predictions,
#                     'training_time': training_time
#                 })
                
#                 oof_preds[model_name] = oof_predictions
#                 test_preds[model_name] = test_predictions
#             else:
#                 print(f"\nâš ï¸� No successful folds for {model_name}.")
        
#         # Train second level meta-model using oof predictions
#         if len(oof_preds) >= 2:
#             print("\nğŸ“‹ Training meta-model...")
            
#             # Create a dataframe of oof predictions
#             meta_train = np.column_stack([oof_preds[model_name] for model_name in oof_preds.keys()])
#             meta_test = np.column_stack([test_preds[model_name] for model_name in test_preds.keys()])
            
#             # Train a meta-model (ridge regression for stability)
#             meta_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
#             meta_model.fit(meta_train, self.y_train)
            
#             # Get coefficients
#             meta_coefs = meta_model.coef_
#             meta_intercept = meta_model.intercept_
            
#             # Normalize coefficients
#             meta_coefs = meta_coefs / np.sum(np.abs(meta_coefs))
            
#             # Print meta-model weights
#             print("\nMeta-model weights:")
#             for i, model_name in enumerate(oof_preds.keys()):
#                 print(f"  {model_name}: {meta_coefs[i]:.4f}")
            
#             # Make meta-model predictions
#             meta_predictions = meta_model.predict(meta_test)
            
#             # Store meta-model results
#             self.model_results.append({
#                 'model': 'meta_ensemble',
#                 'rmse': 0.0,  # We don't have a direct RMSE for the meta-model
#                 'mae': 0.0,   # We don't have a direct MAE for the meta-model
#                 'r2': 1.0,    # Giving it the highest weight
#                 'predictions': meta_predictions,
#                 'training_time': 0.0
#             })
            
#             print("\nâœ… Meta-model trained successfully")
        
#         # Generate ensemble predictions
#         if len(self.model_results) > 0:
#             # Calculate weights based on RÂ² scores
#             r2_scores = [result['r2'] for result in self.model_results]
            
#             # If all RÂ² scores are negative or very low, use equal weights
#             if all(r2 < 0.01 for r2 in r2_scores):
#                 print("\nâš ï¸� All models have very low RÂ² scores. Using equal weights for ensemble.")
#                 ensemble_weights = [1.0 / len(self.model_results)] * len(self.model_results)
#             else:
#                 # If we have the meta-ensemble, give it more weight
#                 meta_idx = next((i for i, x in enumerate(self.model_results) if x['model'] == 'meta_ensemble'), None)
#                 if meta_idx is not None:
#                     # Adjust weights to give meta-model more importance
#                     adjusted_r2 = [max(0, r2) for r2 in r2_scores]  # Handle negative RÂ²
#                     adjusted_r2[meta_idx] = max(0.5, adjusted_r2[meta_idx]) * 2  # Boost meta-model weight
#                     ensemble_weights = [score / sum(adjusted_r2) for score in adjusted_r2]
#                 else:
#                     # Convert RÂ² to weights, handling negative values
#                     adjusted_r2 = [max(0, r2) for r2 in r2_scores]
#                     if sum(adjusted_r2) > 0:
#                         ensemble_weights = [score / sum(adjusted_r2) for score in adjusted_r2]
#                     else:
#                         ensemble_weights = [1.0 / len(self.model_results)] * len(self.model_results)
            
#             # Print ensemble weights
#             print("\nEnsemble weights:")
#             for i, result in enumerate(self.model_results):
#                 print(f"  {result['model']}: {ensemble_weights[i]:.4f}")
            
#             # Generate ensemble predictions
#             self.predictions = np.zeros(len(self.X_test))
#             for i, result in enumerate(self.model_results):
#                 self.predictions += result['predictions'] * ensemble_weights[i]
            
#             # If we applied a transformation to the target, inverse transform the predictions
#             if hasattr(self, 'inverse_transform_fn'):
#                 print("\nğŸ“‹ Applying inverse transformation to predictions...")
#                 self.predictions = self.inverse_transform_fn(self.predictions)
            
#             print("\nâœ… Ensemble predictions created successfully")
#         else:
#             print("\nâš ï¸� No models were successfully trained. Cannot create predictions.")
        
#         return self
    
#     def create_submission(self, filename='submission.csv'):
#         """
#         Create a submission file with the predictions.
#         """
#         print("\n" + "="*80)
#         print("ğŸ“Š CREATING SUBMISSION")
#         print("="*80)
        
#         if self.predictions is None:
#             print("â�Œ No predictions found. Please run train_advanced_models() first.")
#             return self
        
#         if self.test_ids is None:
#             print("â�Œ No test IDs found. Please run preprocess_data() first.")
#             return self
        
#         # Create submission DataFrame
#         self.submission = pd.DataFrame({
#             'id': self.test_ids,
#             'Price': self.predictions
#         })
        
#         # Save submission
#         submission_path = os.path.join(self.output_path, filename)
#         self.submission.to_csv(submission_path, index=False)
        
#         print(f"âœ… Submission saved to {submission_path}")
#         print("\nSubmission Preview:")
#         print(self.submission.head())
        
#         # Basic statistics
#         print("\nğŸ“Š Prediction Statistics:")
#         print(f"Count: {len(self.submission)}")
#         print(f"Min: {self.submission['Price'].min():.4f}")
#         print(f"Max: {self.submission['Price'].max():.4f}")
#         print(f"Mean: {self.submission['Price'].mean():.4f}")
#         print(f"Median: {self.submission['Price'].median():.4f}")
        
#         return self
    
#     def run_full_pipeline(self, train_data=None, test_data=None, original_data=None, optimize=True, n_trials=50):
#         """
#         Run the complete advanced modeling pipeline.
#         """
#         print("\n" + "="*80)
#         print("ğŸš€ RUNNING ADVANCED MODELING PIPELINE")
#         print("="*80)
        
#         # Check package versions for compatibility
#         try:
#             import pkg_resources
            
#             def check_version(package, min_version=None):
#                 version = pkg_resources.get_distribution(package).version
#                 print(f"Using {package} version: {version}")
#                 if min_version:
#                     version_tuple = tuple(map(int, version.split('.')))
#                     min_version_tuple = tuple(map(int, min_version.split('.')))
#                     if version_tuple < min_version_tuple:
#                         print(f"âš ï¸� Warning: {package} version {version} is older than recommended {min_version}")
#                 return version
            
#             # Check versions of key packages
#             check_version('lightgbm', '3.3.0')
#             check_version('xgboost', '1.5.0')
#             check_version('catboost', '1.0.0')
#             check_version('scikit-learn', '1.0.0')
#             check_version('optuna', '2.10.0')
#         except Exception as e:
#             print(f"âš ï¸� Warning: Could not check package versions: {str(e)}")
        
#         try:
#             # Step 1: Load data
#             self.load_data(train_data, test_data, original_data)
            
#             # Step 2: Analyze target variable
#             self.analyze_target_variable()
            
#             # Step 3: Transform target variable
#             self.transform_target()
            
#             # Step 4: Preprocess data
#             self.preprocess_data(handle_outliers=True, use_original_data=True)
            
#             # Step 5: Engineer features
#             self.engineer_features()
            
#             # Step 6: Prepare for modeling
#             self.prepare_for_modeling()
            
#             # Step 7: Feature selection
#             self.select_features(method='mutual_info', n_features=40)
            
#             # Step 8: Prepare for modeling again with selected features
#             self.prepare_for_modeling()
            
#             # Step 9: Train models with improved error handling
#             try:
#                 self.train_advanced_models(use_optuna=optimize, n_trials=n_trials)
#             except Exception as model_error:
#                 print(f"\nâ�Œ Error in model training: {str(model_error)}")
#                 print("âš ï¸� Attempting to continue with a simplified model approach...")
                
#                 # Fallback to a simple model approach
#                 try:
#                     from sklearn.ensemble import GradientBoostingRegressor
                    
#                     print("\nğŸ“‹ Training fallback Gradient Boosting model...")
#                     model = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
#                     model.fit(self.X_train, self.y_train)
                    
#                     # Generate predictions
#                     self.predictions = model.predict(self.X_test)
                    
#                     # If we applied a transformation to the target, inverse transform the predictions
#                     if hasattr(self, 'inverse_transform_fn'):
#                         print("ğŸ“‹ Applying inverse transformation to predictions...")
#                         self.predictions = self.inverse_transform_fn(self.predictions)
                    
#                     print("âœ… Fallback model trained successfully")
#                 except Exception as fallback_error:
#                     print(f"\nâ�Œ Fallback model also failed: {str(fallback_error)}")
#                     # Use a very simple approach - mean prediction
#                     if hasattr(self, 'y_train') and self.y_train is not None:
#                         mean_target = np.mean(self.y_train)
#                         if hasattr(self, 'inverse_transform_fn'):
#                             mean_target = self.inverse_transform_fn(mean_target)
#                         self.predictions = np.ones(len(self.X_test)) * mean_target
#                         print(f"âš ï¸� Using constant mean prediction: {mean_target:.4f}")
#                     else:
#                         # Hard-coded fallback if nothing else works
#                         self.predictions = np.ones(len(self.X_test)) * 81.5
#                         print("âš ï¸� Using hard-coded constant prediction")
            
#             # Step 10: Create submission
#             self.create_submission()
            
#             print("\nâœ… Advanced pipeline completed successfully!")
            
#             return self.submission
            
#         except Exception as e:
#             print(f"\nâ�Œ Error in pipeline: {str(e)}")
            
#             # Create fallback submission if possible
#             try:
#                 if hasattr(self, 'test_ids') and self.test_ids is not None:
#                     # Use mean of training data or a constant value
#                     if hasattr(self, 'train_data') and 'Price' in self.train_data.columns:
#                         mean_price = self.train_data['Price'].mean()
#                     else:
#                         mean_price = 81.5  # Based on your results
                    
#                     fallback_submission = pd.DataFrame({
#                         'id': self.test_ids,
#                         'Price': np.ones(len(self.test_ids)) * mean_price
#                     })
                    
#                     fallback_path = os.path.join(self.output_path, 'fallback_submission.csv')
#                     fallback_submission.to_csv(fallback_path, index=False)
                    
#                     print(f"\nâš ï¸� Created fallback submission with mean price: {mean_price:.2f}")
#                     print(f"Saved to: {fallback_path}")
                    
#                     return fallback_submission
#                 else:
#                     print("\nâ�Œ Could not create fallback submission. No test IDs available.")
#                     return None
#             except Exception as fallback_error:
#                 print(f"\nâ�Œ Could not create fallback submission: {str(fallback_error)}")
#                 return None

# # Example usage
# if __name__ == "__main__":
#     # Create the advanced modeler
#     modeler = AdvancedBagPriceModeler()
    
#     # Run the full advanced pipeline
#     submission = modeler.run_full_pipeline(optimize=True, n_trials=30)

