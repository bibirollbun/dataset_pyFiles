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


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# COMPREHENSIVE EDA SCRIPT FOR DIABETES PREDICTION CHALLENGE
# ============================================================================

class DiabetesEDA:
    """
    Comprehensive EDA class for Kaggle Diabetes Prediction Challenge
    """
    
    def __init__(self, train, test, orig):
        self.train = train.copy()
        self.test = test.copy()
        self.orig = orig.copy()
        self.target = 'diagnosed_diabetes'
        
        print("="*80)
        print("DIABETES PREDICTION CHALLENGE - COMPREHENSIVE EDA")
        print("="*80)
    
    def basic_info(self):
        """Display basic information about all datasets"""
        print("\n" + "="*80)
        print("1. BASIC DATASET INFORMATION")
        print("="*80)
        
        for name, df in [('TRAIN', self.train), ('TEST', self.test), ('ORIGINAL', self.orig)]:
            print(f"\n{'='*40}")
            print(f"{name} DATASET")
            print(f"{'='*40}")
            print(f"Shape: {df.shape}")
            print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"\nData Types:")
            print(df.dtypes.value_counts())
            print(f"\nColumn Names:")
            print(df.columns.tolist())
            print(f"\nInfo:")
            df.info()
            print("\n")
    
    def missing_values_analysis(self):
        """Analyze missing values"""
        print("\n" + "="*80)
        print("2. MISSING VALUES ANALYSIS")
        print("="*80)
        
        for name, df in [('TRAIN', self.train), ('TEST', self.test), ('ORIGINAL', self.orig)]:
            print(f"\n{name} Dataset:")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing_Count': missing,
                'Percentage': missing_pct
            })
            missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
            
            if len(missing_df) > 0:
                print(missing_df)
            else:
                print("No missing values found!")
    
    def descriptive_statistics(self):
        """Comprehensive descriptive statistics"""
        print("\n" + "="*80)
        print("3. DESCRIPTIVE STATISTICS")
        print("="*80)
        
        for name, df in [('TRAIN', self.train), ('TEST', self.test), ('ORIGINAL', self.orig)]:
            print(f"\n{'='*40}")
            print(f"{name} DATASET")
            print(f"{'='*40}")
            print("\nNumerical Columns Description:")
            print(df.describe().T)
            
            # Additional statistics
            print("\nAdditional Statistics:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats_df = pd.DataFrame({
                'Skewness': df[numeric_cols].skew(),
                'Kurtosis': df[numeric_cols].kurtosis(),
                'Range': df[numeric_cols].max() - df[numeric_cols].min(),
                'IQR': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
            })
            print(stats_df)
    
    def column_classification(self):
        """Classify columns into numerical and categorical"""
        print("\n" + "="*80)
        print("4. COLUMN CLASSIFICATION")
        print("="*80)
        
        for name, df in [('TRAIN', self.train), ('TEST', self.test), ('ORIGINAL', self.orig)]:
            print(f"\n{name} Dataset:")
            
            # Numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'id' in numerical_cols:
                numerical_cols.remove('id')
            
            print(f"\nNumerical Columns ({len(numerical_cols)}):")
            print(numerical_cols)
            
            # Check for potentially categorical numerical columns
            potential_categorical = []
            for col in numerical_cols:
                unique_count = df[col].nunique()
                if unique_count <= 20:  # Threshold for categorical
                    potential_categorical.append((col, unique_count))
            
            if potential_categorical:
                print(f"\nPotentially Categorical Numerical Columns:")
                for col, count in potential_categorical:
                    print(f"  - {col}: {count} unique values")
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            print(f"\nCategorical Columns ({len(categorical_cols)}):")
            print(categorical_cols)
    
    def unique_values_analysis(self):
        """Analyze unique values in all columns"""
        print("\n" + "="*80)
        print("5. UNIQUE VALUES ANALYSIS")
        print("="*80)
        
        for name, df in [('TRAIN', self.train), ('TEST', self.test), ('ORIGINAL', self.orig)]:
            print(f"\n{name} Dataset:")
            
            unique_df = pd.DataFrame({
                'Column': df.columns,
                'Unique_Count': [df[col].nunique() for col in df.columns],
                'Unique_Percentage': [df[col].nunique() / len(df) * 100 for col in df.columns],
                'Most_Common_Value': [df[col].mode()[0] if len(df[col].mode()) > 0 else None for col in df.columns],
                'Most_Common_Frequency': [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0 for col in df.columns]
            })
            
            print(unique_df.sort_values('Unique_Count', ascending=False).to_string())
            
            # Show value distributions for low-cardinality columns
            print(f"\n{name} - Value Distributions for Categorical/Binary Columns:")
            for col in df.columns:
                if df[col].nunique() <= 10 and col != 'id':
                    print(f"\n{col}:")
                    print(df[col].value_counts().sort_index())
                    print(f"Percentages:")
                    print(df[col].value_counts(normalize=True).sort_index() * 100)
    
    def target_distribution(self):
        """Analyze target variable distribution"""
        print("\n" + "="*80)
        print("6. TARGET VARIABLE DISTRIBUTION")
        print("="*80)
        
        for name, df in [('TRAIN', self.train), ('ORIGINAL', self.orig)]:
            if self.target in df.columns:
                print(f"\n{name} Dataset:")
                print(f"\nTarget Distribution:")
                print(df[self.target].value_counts())
                print(f"\nTarget Percentage:")
                print(df[self.target].value_counts(normalize=True) * 100)
                
                # Class imbalance ratio
                target_counts = df[self.target].value_counts()
                if len(target_counts) == 2:
                    imbalance_ratio = target_counts.max() / target_counts.min()
                    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    def numerical_distribution_analysis(self):
        """Analyze distribution of numerical columns"""
        print("\n" + "="*80)
        print("7. NUMERICAL DISTRIBUTION ANALYSIS")
        print("="*80)
        
        numerical_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numerical_cols:
            numerical_cols.remove('id')
        
        print("\nDistribution Characteristics:")
        
        for col in numerical_cols:
            print(f"\n{col}:")
            print(f"  Min: {self.train[col].min()}")
            print(f"  Max: {self.train[col].max()}")
            print(f"  Mean: {self.train[col].mean():.2f}")
            print(f"  Median: {self.train[col].median():.2f}")
            print(f"  Std: {self.train[col].std():.2f}")
            print(f"  Skewness: {self.train[col].skew():.2f}")
            print(f"  Kurtosis: {self.train[col].kurtosis():.2f}")
            
            # Check for outliers using IQR method
            Q1 = self.train[col].quantile(0.25)
            Q3 = self.train[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.train[col] < (Q1 - 1.5 * IQR)) | (self.train[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"  Outliers (IQR method): {outliers} ({outliers/len(self.train)*100:.2f}%)")
    
    def correlation_analysis(self):
        """Comprehensive correlation analysis"""
        print("\n" + "="*80)
        print("8. CORRELATION ANALYSIS")
        print("="*80)
        
        numerical_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numerical_cols:
            numerical_cols.remove('id')
        
        # Pearson correlation
        print("\nPearson Correlation with Target:")
        if self.target in self.train.columns:
            corr_with_target = self.train[numerical_cols].corrwith(self.train[self.target]).sort_values(ascending=False)
            print(corr_with_target)
        
        # Full correlation matrix
        print("\nTop 20 Highest Correlations (excluding diagonal):")
        corr_matrix = self.train[numerical_cols].corr()
        
        # Get upper triangle
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find correlations
        high_corr = []
        for column in upper_tri.columns:
            for index in upper_tri.index:
                value = upper_tri.loc[index, column]
                if not pd.isna(value):
                    high_corr.append((index, column, abs(value), value))
        
        high_corr_df = pd.DataFrame(high_corr, columns=['Feature_1', 'Feature_2', 'Abs_Correlation', 'Correlation'])
        high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False).head(20)
        print(high_corr_df)
        
        # Spearman correlation for monotonic relationships
        print("\nSpearman Correlation with Target (for non-linear relationships):")
        if self.target in self.train.columns:
            spearman_corr = pd.DataFrame({col: [self.train[col].corr(self.train[self.target], method='spearman')] 
                                         for col in numerical_cols}).T
            spearman_corr.columns = ['Spearman_Correlation']
            print(spearman_corr.sort_values('Spearman_Correlation', ascending=False))
    
    def mutual_information_analysis(self):
        """Calculate mutual information for feature importance"""
        print("\n" + "="*80)
        print("9. MUTUAL INFORMATION ANALYSIS")
        print("="*80)
        
        if self.target not in self.train.columns:
            print("Target not found in training data. Skipping MI analysis.")
            return
        
        feature_cols = [col for col in self.train.columns if col not in ['id', self.target]]
        X = self.train[feature_cols].copy()
        y = self.train[self.target]
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            print(f"\nEncoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Calculate mutual information
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
            mi_df = pd.DataFrame({
                'Feature': feature_cols,
                'MI_Score': mi_scores
            }).sort_values('MI_Score', ascending=False)
            
            print("\nMutual Information Scores (sorted):")
            print(mi_df)
            
            print("\nTop 10 Most Important Features by MI:")
            print(mi_df.head(10))
        except Exception as e:
            print(f"\nError calculating mutual information: {str(e)}")
            print("Skipping MI analysis.")
    
    def train_vs_orig_comparison(self):
        """Compare train and original datasets"""
        print("\n" + "="*80)
        print("10. TRAIN VS ORIGINAL DATASET COMPARISON")
        print("="*80)
        
        # Column comparison
        print("\nColumn Comparison:")
        train_cols = set(self.train.columns)
        orig_cols = set(self.orig.columns)
        
        common_cols = train_cols & orig_cols
        only_train = train_cols - orig_cols
        only_orig = orig_cols - train_cols
        
        print(f"Common columns: {len(common_cols)}")
        print(f"Columns only in train: {only_train}")
        print(f"Columns only in original: {only_orig}")
        
        # Statistical comparison for common columns
        print("\nStatistical Comparison for Common Numerical Columns:")
        
        common_numerical = []
        for col in common_cols:
            if col in self.train.select_dtypes(include=[np.number]).columns and col != 'id':
                common_numerical.append(col)
        
        comparison_stats = []
        for col in common_numerical:
            ks_stat, ks_pval = ks_2samp(self.train[col].dropna(), self.orig[col].dropna())
            
            comparison_stats.append({
                'Feature': col,
                'Train_Mean': self.train[col].mean(),
                'Orig_Mean': self.orig[col].mean(),
                'Mean_Diff_%': ((self.train[col].mean() - self.orig[col].mean()) / self.orig[col].mean() * 100) if self.orig[col].mean() != 0 else 0,
                'Train_Std': self.train[col].std(),
                'Orig_Std': self.orig[col].std(),
                'KS_Statistic': ks_stat,
                'KS_P_Value': ks_pval,
                'Distributions_Similar': 'Yes' if ks_pval > 0.05 else 'No'
            })
        
        comparison_df = pd.DataFrame(comparison_stats)
        print(comparison_df.to_string())
        
        # Target distribution comparison
        if self.target in common_cols:
            print(f"\nTarget Distribution Comparison:")
            print(f"\nTrain Target Distribution:")
            print(self.train[self.target].value_counts(normalize=True))
            print(f"\nOriginal Target Distribution:")
            print(self.orig[self.target].value_counts(normalize=True))
    
    def train_vs_test_comparison(self):
        """Compare train and test datasets for distribution shift"""
        print("\n" + "="*80)
        print("11. TRAIN VS TEST DATASET COMPARISON")
        print("="*80)
        
        common_cols = set(self.train.columns) & set(self.test.columns)
        common_cols.discard('id')
        if self.target in common_cols:
            common_cols.discard(self.target)
        
        print(f"\nComparing {len(common_cols)} common features")
        
        comparison_stats = []
        for col in common_cols:
            if col in self.train.select_dtypes(include=[np.number]).columns:
                ks_stat, ks_pval = ks_2samp(self.train[col].dropna(), self.test[col].dropna())
                
                comparison_stats.append({
                    'Feature': col,
                    'Train_Mean': self.train[col].mean(),
                    'Test_Mean': self.test[col].mean(),
                    'Mean_Diff_%': ((self.test[col].mean() - self.train[col].mean()) / self.train[col].mean() * 100) if self.train[col].mean() != 0 else 0,
                    'KS_Statistic': ks_stat,
                    'KS_P_Value': ks_pval,
                    'Distributions_Similar': 'Yes' if ks_pval > 0.05 else 'No'
                })
        
        comparison_df = pd.DataFrame(comparison_stats)
        print(comparison_df.sort_values('KS_Statistic', ascending=False).to_string())
        
        # Flag potential distribution shifts
        shifted_features = comparison_df[comparison_df['Distributions_Similar'] == 'No']
        if len(shifted_features) > 0:
            print(f"\n⚠️  WARNING: {len(shifted_features)} features show distribution shift:")
            print(shifted_features[['Feature', 'KS_Statistic', 'KS_P_Value']])
    
    def feature_engineering_insights(self):
        """Generate insights for feature engineering"""
        print("\n" + "="*80)
        print("12. FEATURE ENGINEERING INSIGHTS")
        print("="*80)
        
        insights = []
        
        # 1. Binary features
        binary_features = []
        for col in self.train.columns:
            if col not in ['id', self.target] and self.train[col].nunique() == 2:
                binary_features.append(col)
        
        if binary_features:
            insights.append(f"✓ Found {len(binary_features)} binary features: {binary_features}")
            insights.append("  → Consider: Interaction terms between binary features")
        
        # 2. Categorical features (low cardinality)
        categorical_features = []
        for col in self.train.columns:
            if col not in ['id', self.target]:
                unique_count = self.train[col].nunique()
                if 3 <= unique_count <= 20:
                    categorical_features.append((col, unique_count))
        
        if categorical_features:
            insights.append(f"\n✓ Found {len(categorical_features)} categorical features:")
            for feat, count in categorical_features:
                insights.append(f"  - {feat}: {count} categories")
            insights.append("  → Consider: One-hot encoding, target encoding, frequency encoding")
        
        # 3. Continuous features
        numerical_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numerical_cols:
            numerical_cols.remove('id')
        if self.target in numerical_cols:
            numerical_cols.remove(self.target)
        
        continuous_features = [col for col in numerical_cols 
                              if self.train[col].nunique() > 20]
        
        if continuous_features:
            insights.append(f"\n✓ Found {len(continuous_features)} continuous features")
            insights.append("  → Consider: Polynomial features, binning, normalization, log transform")
        
        # 4. Skewed features
        skewed_features = []
        for col in continuous_features:
            skew = self.train[col].skew()
            if abs(skew) > 1:
                skewed_features.append((col, skew))
        
        if skewed_features:
            insights.append(f"\n✓ Found {len(skewed_features)} highly skewed features:")
            for feat, skew in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
                insights.append(f"  - {feat}: skewness = {skew:.2f}")
            insights.append("  → Consider: Log transform, Box-Cox transform, Yeo-Johnson transform")
        
        # 5. High correlation pairs
        numerical_cols_for_corr = [col for col in numerical_cols if col in self.train.columns]
        if len(numerical_cols_for_corr) > 1:
            corr_matrix = self.train[numerical_cols_for_corr].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append((corr_matrix.columns[i], 
                                              corr_matrix.columns[j], 
                                              corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                insights.append(f"\n✓ Found {len(high_corr_pairs)} highly correlated feature pairs:")
                for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
                    insights.append(f"  - {feat1} <-> {feat2}: {corr:.3f}")
                insights.append("  → Consider: Feature selection, PCA, creating ratio features")
        
        # 6. Potential interaction features
        if self.target in self.train.columns and len(binary_features) >= 2:
            insights.append(f"\n✓ Potential interaction features from {len(binary_features)} binary features")
            insights.append(f"  → Consider: Creating {len(binary_features) * (len(binary_features)-1) // 2} pairwise interactions")
        
        # Print all insights
        for insight in insights:
            print(insight)
        
        # Additional recommendations
        print("\n" + "="*40)
        print("GENERAL RECOMMENDATIONS:")
        print("="*40)
        print("1. Feature Scaling: StandardScaler or RobustScaler for tree-based models")
        print("2. Handle Imbalance: SMOTE, class weights, or stratified sampling")
        print("3. Cross-Validation: Stratified K-Fold for reliable validation")
        print("4. Feature Selection: Use mutual information, SHAP values, or permutation importance")
        print("5. Ensemble Methods: Combine synthetic and original data for training")
        print("6. Model Selection: Try XGBoost, LightGBM, CatBoost, and Neural Networks")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*80)
        print("13. GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Create figure for target distribution
        if self.target in self.train.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Target distribution in train
            self.train[self.target].value_counts().plot(kind='bar', ax=axes[0])
            axes[0].set_title('Target Distribution (Train)')
            axes[0].set_xlabel('Diagnosed Diabetes')
            axes[0].set_ylabel('Count')
            
            # Target distribution in original
            if self.target in self.orig.columns:
                self.orig[self.target].value_counts().plot(kind='bar', ax=axes[1])
                axes[1].set_title('Target Distribution (Original)')
                axes[1].set_xlabel('Diagnosed Diabetes')
                axes[1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: target_distribution.png")
            plt.close()
        
        # Correlation heatmap
        numerical_cols = [col for col in self.train.select_dtypes(include=[np.number]).columns 
                         if col != 'id']
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(20, 16))
            corr_matrix = self.train[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: correlation_heatmap.png")
            plt.close()
        
        # Distribution plots for top features
        if self.target in self.train.columns:
            # Get top features by correlation
            corr_with_target = self.train[numerical_cols].corrwith(self.train[self.target]).abs().sort_values(ascending=False)
            top_features = corr_with_target.head(9).index.tolist()
            
            if self.target in top_features:
                top_features.remove(self.target)
            top_features = top_features[:8]
            
            if len(top_features) > 0:
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.flatten()
                
                for idx, col in enumerate(top_features):
                    if self.train[col].nunique() <= 10:
                        # Bar plot for categorical
                        self.train.groupby([col, self.target]).size().unstack().plot(kind='bar', ax=axes[idx])
                    else:
                        # Histogram for continuous
                        for target_val in sorted(self.train[self.target].unique()):
                            axes[idx].hist(self.train[self.train[self.target] == target_val][col], 
                                         alpha=0.5, label=f'Target={target_val}', bins=30)
                        axes[idx].legend()
                    
                    axes[idx].set_title(f'{col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Count')
                
                plt.tight_layout()
                plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
                print("✓ Saved: feature_distributions.png")
                plt.close()
        
        print("\nVisualization generation complete!")
    
    def run_complete_eda(self):
        """Run all EDA functions"""
        self.basic_info()
        self.missing_values_analysis()
        self.descriptive_statistics()
        self.column_classification()
        self.unique_values_analysis()
        self.target_distribution()
        self.numerical_distribution_analysis()
        self.correlation_analysis()
        self.mutual_information_analysis()
        self.train_vs_orig_comparison()
        self.train_vs_test_comparison()
        self.feature_engineering_insights()
        self.generate_visualizations()
        
        print("\n" + "="*80)
        print("EDA COMPLETE!")
        print("="*80)
        print("\nNext Steps:")
        print("1. Review the insights for feature engineering")
        print("2. Check the generated visualizations")
        print("3. Plan your feature engineering strategy")
        print("4. Consider combining synthetic and original data")
        print("5. Start building your models!")


# ============================================================================
# USAGE
# ============================================================================

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')

# Initialize and run EDA
eda = DiabetesEDA(train, test, orig)
eda.run_complete_eda()

# Optional: Access individual analyses
# eda.correlation_analysis()
# eda.mutual_information_analysis()
# eda.feature_engineering_insights()

