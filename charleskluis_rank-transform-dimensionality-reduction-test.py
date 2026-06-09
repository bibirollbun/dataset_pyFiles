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


#!/usr/bin/env python3
"""
DRW Crypto Market Prediction - Comprehensive Dimensionality Reduction Analysis
============================================================================
Testing 25+ dimensionality reduction techniques on rank-transformed financial data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import warnings
import os
import gc
import time
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import all dimensionality reduction methods
from sklearn.decomposition import (
    PCA, FastICA, NMF, FactorAnalysis, TruncatedSVD,
    SparsePCA, MiniBatchSparsePCA, KernelPCA,
    IncrementalPCA, DictionaryLearning
)
from sklearn.manifold import (
    TSNE, Isomap, LocallyLinearEmbedding, 
    SpectralEmbedding, MDS
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel, VarianceThreshold,
    chi2, SelectPercentile
)
from sklearn.random_projection import (
    GaussianRandomProjection, SparseRandomProjection
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
import xgboost as xgb
import lightgbm as lgb

# Try importing additional methods
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.neural_network import MLPRegressor
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

# Set random seed
np.random.seed(42)

# Check if we're in Kaggle environment
KAGGLE_INPUT_PATH = '/kaggle/input/drw-crypto-market-prediction'
if os.path.exists(KAGGLE_INPUT_PATH):
    DATA_PATH = KAGGLE_INPUT_PATH
else:
    DATA_PATH = '.'

class FinancialDimensionalityReducer:
    """
    Comprehensive dimensionality reduction for financial data
    """
    
    def __init__(self, n_components=50, sample_size=10000, n_jobs=4):
        self.n_components = n_components
        self.sample_size = sample_size
        self.n_jobs = n_jobs
        self.results = {}
        self.best_method = None
        self.reducers = {}
        
    def optimize_dtypes(self, df):
        """Optimize dataframe dtypes to save memory"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.float32)
        
        return df
    
    def handle_infinities(self, X):
        """Replace inf/-inf with appropriate finite values"""
        X = X.copy()
        
        for i in range(X.shape[1]):
            col = X[:, i]
            finite_mask = np.isfinite(col)
            
            if np.any(~finite_mask):
                finite_vals = col[finite_mask]
                
                if len(finite_vals) > 0:
                    # Replace inf with max + 1 std, -inf with min - 1 std
                    col_max = np.max(finite_vals)
                    col_min = np.min(finite_vals)
                    col_std = np.std(finite_vals)
                    
                    col[col == np.inf] = col_max + col_std
                    col[col == -np.inf] = col_min - col_std
                    
                    # Replace any remaining non-finite with median
                    col[~np.isfinite(col)] = np.median(finite_vals)
                else:
                    # If all values are non-finite, replace with 0
                    col[:] = 0
        
        return X
    
    def remove_constant_features(self, X, feature_names, variance_threshold=1e-10):
        """Remove features with near-zero variance"""
        print(f"Removing constant features (variance < {variance_threshold})...")
        
        valid_indices = []
        
        for i in range(X.shape[1]):
            col = X[:, i]
            finite_mask = np.isfinite(col)
            
            if np.any(finite_mask):
                finite_vals = col[finite_mask]
                if len(finite_vals) > 1 and np.var(finite_vals) > variance_threshold:
                    valid_indices.append(i)
        
        valid_indices = np.array(valid_indices)
        print(f"Kept {len(valid_indices)} / {X.shape[1]} features")
        
        return X[:, valid_indices], np.array(feature_names)[valid_indices]
    
    def rank_transform(self, X):
        """
        Apply rank transformation - best practice for financial data
        Converts each feature to its percentile rank
        """
        print("Applying rank transformation...")
        X_ranked = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            col = X[:, i]
            finite_mask = np.isfinite(col)
            
            if np.any(finite_mask):
                # Rank transform only finite values
                finite_vals = col[finite_mask]
                ranks = stats.rankdata(finite_vals, method='average')
                # Convert to percentiles (0-1 range)
                percentiles = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else ranks
                
                # Fill in the ranked values
                X_ranked[finite_mask, i] = percentiles
                # Non-finite values get 0.5 (median rank)
                X_ranked[~finite_mask, i] = 0.5
            else:
                # If all values are non-finite, use 0.5
                X_ranked[:, i] = 0.5
        
        return X_ranked.astype(np.float32)
    
    def get_dimensionality_reducers(self):
        """Get all dimensionality reduction methods to test"""
        reducers = {}
        
        # 1. Linear decomposition methods
        reducers['PCA'] = PCA(n_components=self.n_components, random_state=42)
        reducers['IncrementalPCA'] = IncrementalPCA(n_components=self.n_components, batch_size=1000)
        reducers['TruncatedSVD'] = TruncatedSVD(n_components=self.n_components, random_state=42)
        reducers['FastICA'] = FastICA(n_components=min(self.n_components, 50), random_state=42, max_iter=500)
        reducers['FactorAnalysis'] = FactorAnalysis(n_components=self.n_components, random_state=42)
        
        # 2. Sparse methods
        reducers['SparsePCA'] = SparsePCA(n_components=20, random_state=42, max_iter=100)
        reducers['MiniBatchSparsePCA'] = MiniBatchSparsePCA(n_components=20, random_state=42, batch_size=100)
        reducers['DictionaryLearning'] = DictionaryLearning(n_components=30, random_state=42, max_iter=100)
        
        # 3. Non-negative matrix factorization (for rank-transformed data)
        reducers['NMF'] = NMF(n_components=self.n_components, random_state=42, max_iter=500)
        
        # 4. Kernel methods
        reducers['KernelPCA_rbf'] = KernelPCA(n_components=self.n_components, kernel='rbf', gamma=0.01, random_state=42)
        reducers['KernelPCA_sigmoid'] = KernelPCA(n_components=self.n_components, kernel='sigmoid', random_state=42)
        
        # 5. Random projections
        reducers['GaussianRandomProjection'] = GaussianRandomProjection(n_components=self.n_components, random_state=42)
        reducers['SparseRandomProjection'] = SparseRandomProjection(n_components=self.n_components, random_state=42)
        
        # 6. Manifold learning (use fewer components due to computational cost)
        manifold_components = min(10, self.n_components)
        reducers['Isomap'] = Isomap(n_components=manifold_components, n_neighbors=10)
        reducers['LocallyLinearEmbedding'] = LocallyLinearEmbedding(n_components=manifold_components, n_neighbors=10, random_state=42)
        reducers['SpectralEmbedding'] = SpectralEmbedding(n_components=manifold_components, random_state=42)
        reducers['MDS'] = MDS(n_components=manifold_components, random_state=42, max_iter=100)
        
        # Only include t-SNE for very small samples due to computational cost
        if self.sample_size <= 5000:
            reducers['TSNE'] = TSNE(n_components=2, random_state=42, perplexity=30)
        
        # 7. UMAP if available
        if UMAP_AVAILABLE:
            reducers['UMAP'] = umap.UMAP(n_components=manifold_components, random_state=42)
        
        # 8. Supervised methods (these need the target variable)
        # Will be handled separately
        
        # 9. Feature selection methods
        # Will be handled separately
        
        return reducers
    
    def get_feature_selection_methods(self, X, y):
        """Get feature selection methods (these need target variable)"""
        selectors = {}
        
        # Statistical methods
        selectors['SelectKBest_f_regression'] = SelectKBest(f_regression, k=self.n_components)
        selectors['SelectKBest_mutual_info'] = SelectKBest(mutual_info_regression, k=self.n_components)
        selectors['SelectPercentile_f_regression'] = SelectPercentile(f_regression, percentile=20)
        
        # Variance threshold
        selectors['VarianceThreshold_0.01'] = VarianceThreshold(threshold=0.01)
        selectors['VarianceThreshold_0.05'] = VarianceThreshold(threshold=0.05)
        
        # Model-based selection
        selectors['L1_Lasso'] = SelectFromModel(LassoCV(cv=3, random_state=42, max_iter=1000), threshold='median')
        selectors['L1_ElasticNet'] = SelectFromModel(ElasticNetCV(cv=3, random_state=42, max_iter=1000), threshold='median')
        
        # Tree-based selection
        selectors['RandomForest'] = SelectFromModel(
            RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=self.n_jobs),
            threshold='median'
        )
        selectors['ExtraTrees'] = SelectFromModel(
            ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=self.n_jobs),
            threshold='median'
        )
        
        # Recursive feature elimination (use faster models)
        if X.shape[1] > 100:
            # For many features, use simpler models in RFE
            base_estimator = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
            selectors['RFE_XGBoost'] = RFE(base_estimator, n_features_to_select=self.n_components, step=10)
        
        return selectors
    
    def get_supervised_methods(self):
        """Get supervised dimensionality reduction methods"""
        supervised = {}
        
        # PLS methods
        supervised['PLSRegression'] = PLSRegression(n_components=min(self.n_components, 20))
        
        # CCA (Canonical Correlation Analysis)
        supervised['CCA'] = CCA(n_components=min(self.n_components, 20))
        
        return supervised
    
    def evaluate_method(self, X_train_reduced, X_val_reduced, y_train, y_val, method_name):
        """Evaluate a dimensionality reduction method using XGBoost"""
        # Train a simple XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=self.n_jobs
        )
        
        # Handle edge cases
        if X_train_reduced.shape[1] == 0:
            return {
                'method': method_name,
                'n_features': 0,
                'rmse': np.inf,
                'r2': -np.inf,
                'pearson': 0,
                'train_time': 0,
                'error': 'No features selected'
            }
        
        try:
            start_time = time.time()
            model.fit(X_train_reduced, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_val_reduced)
            
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            pearson_corr, _ = stats.pearsonr(y_val, y_pred)
            
            return {
                'method': method_name,
                'n_features': X_train_reduced.shape[1],
                'rmse': rmse,
                'r2': r2,
                'pearson': pearson_corr,
                'train_time': train_time,
                'error': None
            }
        except Exception as e:
            return {
                'method': method_name,
                'n_features': X_train_reduced.shape[1],
                'rmse': np.inf,
                'r2': -np.inf,
                'pearson': 0,
                'train_time': 0,
                'error': str(e)
            }
    
    def run_experiments(self, df, feature_cols):
        """Run all dimensionality reduction experiments"""
        print("="*60)
        print("FINANCIAL DIMENSIONALITY REDUCTION EXPERIMENTS")
        print(f"Sample size: {self.sample_size}")
        print(f"Target dimensions: {self.n_components}")
        print("="*60)
        
        # Sample data
        if len(df) > self.sample_size:
            # Use most recent samples
            df_sample = df.iloc[-self.sample_size:].copy()
        else:
            df_sample = df.copy()
        
        print(f"\nUsing {len(df_sample)} samples")
        
        # Extract features and target
        X = df_sample[feature_cols].values.astype(np.float32)
        y = df_sample['label'].values.astype(np.float32)
        
        # Clean data
        print("\n1. Data preprocessing...")
        X = self.handle_infinities(X)
        X, feature_names = self.remove_constant_features(X, feature_cols)
        
        # Rank transform - crucial for financial data
        X_ranked = self.rank_transform(X)
        print(f"Data shape after preprocessing: {X_ranked.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_ranked, y, test_size=0.2, random_state=42
        )
        
        # Results storage
        all_results = []
        
        print("\n2. Testing unsupervised methods...")
        print("-" * 40)
        
        # Test unsupervised methods
        reducers = self.get_dimensionality_reducers()
        
        for name, reducer in tqdm(reducers.items(), desc="Unsupervised methods"):
            try:
                # Fit and transform
                start_time = time.time()
                X_train_reduced = reducer.fit_transform(X_train)
                X_val_reduced = reducer.transform(X_val)
                reduction_time = time.time() - start_time
                
                # Evaluate
                result = self.evaluate_method(X_train_reduced, X_val_reduced, y_train, y_val, name)
                result['reduction_time'] = reduction_time
                result['method_type'] = 'unsupervised'
                all_results.append(result)
                
                # Store reducer for later use
                self.reducers[name] = reducer
                
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                all_results.append({
                    'method': name,
                    'method_type': 'unsupervised',
                    'error': str(e),
                    'rmse': np.inf,
                    'r2': -np.inf
                })
        
        print("\n3. Testing feature selection methods...")
        print("-" * 40)
        
        # Test feature selection methods
        selectors = self.get_feature_selection_methods(X_train, y_train)
        
        for name, selector in tqdm(selectors.items(), desc="Feature selection"):
            try:
                # Fit and transform
                start_time = time.time()
                X_train_reduced = selector.fit_transform(X_train, y_train)
                X_val_reduced = selector.transform(X_val)
                reduction_time = time.time() - start_time
                
                # For feature selection, we can identify which features were selected
                if hasattr(selector, 'get_support'):
                    selected_features = feature_names[selector.get_support()]
                else:
                    selected_features = None
                
                # Evaluate
                result = self.evaluate_method(X_train_reduced, X_val_reduced, y_train, y_val, name)
                result['reduction_time'] = reduction_time
                result['method_type'] = 'feature_selection'
                result['selected_features'] = selected_features
                all_results.append(result)
                
                # Store selector
                self.reducers[name] = selector
                
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                all_results.append({
                    'method': name,
                    'method_type': 'feature_selection',
                    'error': str(e),
                    'rmse': np.inf,
                    'r2': -np.inf
                })
        
        print("\n4. Testing supervised methods...")
        print("-" * 40)
        
        # Test supervised methods
        supervised_methods = self.get_supervised_methods()
        
        for name, method in tqdm(supervised_methods.items(), desc="Supervised methods"):
            try:
                # Fit and transform
                start_time = time.time()
                method.fit(X_train, y_train)
                X_train_reduced = method.transform(X_train)
                X_val_reduced = method.transform(X_val)
                reduction_time = time.time() - start_time
                
                # Evaluate
                result = self.evaluate_method(X_train_reduced, X_val_reduced, y_train, y_val, name)
                result['reduction_time'] = reduction_time
                result['method_type'] = 'supervised'
                all_results.append(result)
                
                # Store method
                self.reducers[name] = method
                
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                all_results.append({
                    'method': name,
                    'method_type': 'supervised',
                    'error': str(e),
                    'rmse': np.inf,
                    'r2': -np.inf
                })
        
        # Convert to DataFrame and sort by performance
        self.results = pd.DataFrame(all_results)
        self.results = self.results[self.results['error'].isna()].sort_values('pearson', ascending=False)
        
        # Identify best method
        if len(self.results) > 0:
            self.best_method = self.results.iloc[0]['method']
        
        return self.results
    
    def create_comprehensive_visualization(self):
        """Create comprehensive visualization of results"""
        if self.results is None or len(self.results) == 0:
            print("No results to visualize")
            return
        
        # Set up the plot
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Performance comparison (Pearson correlation)
        ax1 = plt.subplot(3, 3, 1)
        top_methods = self.results.head(20)
        colors = ['green' if x == 'feature_selection' else 'blue' if x == 'unsupervised' else 'red' 
                  for x in top_methods['method_type']]
        ax1.barh(range(len(top_methods)), top_methods['pearson'], color=colors)
        ax1.set_yticks(range(len(top_methods)))
        ax1.set_yticklabels(top_methods['method'], fontsize=8)
        ax1.set_xlabel('Pearson Correlation')
        ax1.set_title('Top 20 Methods by Pearson Correlation')
        ax1.invert_yaxis()
        
        # 2. R² comparison
        ax2 = plt.subplot(3, 3, 2)
        ax2.barh(range(len(top_methods)), top_methods['r2'], color=colors)
        ax2.set_yticks(range(len(top_methods)))
        ax2.set_yticklabels(top_methods['method'], fontsize=8)
        ax2.set_xlabel('R² Score')
        ax2.set_title('Top 20 Methods by R²')
        ax2.invert_yaxis()
        
        # 3. RMSE comparison (lower is better)
        ax3 = plt.subplot(3, 3, 3)
        top_methods_rmse = self.results.sort_values('rmse').head(20)
        colors_rmse = ['green' if x == 'feature_selection' else 'blue' if x == 'unsupervised' else 'red' 
                       for x in top_methods_rmse['method_type']]
        ax3.barh(range(len(top_methods_rmse)), top_methods_rmse['rmse'], color=colors_rmse)
        ax3.set_yticks(range(len(top_methods_rmse)))
        ax3.set_yticklabels(top_methods_rmse['method'], fontsize=8)
        ax3.set_xlabel('RMSE')
        ax3.set_title('Top 20 Methods by RMSE (Lower is Better)')
        ax3.invert_yaxis()
        
        # 4. Method type distribution
        ax4 = plt.subplot(3, 3, 4)
        method_counts = self.results['method_type'].value_counts()
        ax4.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
        ax4.set_title('Distribution of Method Types')
        
        # 5. Performance vs Features scatter
        ax5 = plt.subplot(3, 3, 5)
        for method_type in self.results['method_type'].unique():
            mask = self.results['method_type'] == method_type
            ax5.scatter(self.results[mask]['n_features'], 
                       self.results[mask]['pearson'],
                       label=method_type, alpha=0.6, s=50)
        ax5.set_xlabel('Number of Features')
        ax5.set_ylabel('Pearson Correlation')
        ax5.set_title('Performance vs Number of Features')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Computation time comparison
        ax6 = plt.subplot(3, 3, 6)
        top_methods_time = self.results.sort_values('pearson', ascending=False).head(20)
        total_time = top_methods_time['reduction_time'] + top_methods_time['train_time']
        ax6.barh(range(len(top_methods_time)), total_time)
        ax6.set_yticks(range(len(top_methods_time)))
        ax6.set_yticklabels(top_methods_time['method'], fontsize=8)
        ax6.set_xlabel('Total Time (seconds)')
        ax6.set_title('Computation Time for Top 20 Methods')
        ax6.invert_yaxis()
        
        # 7. Performance heatmap by method type
        ax7 = plt.subplot(3, 3, 7)
        pivot_data = self.results.pivot_table(
            values=['pearson', 'r2', 'rmse'],
            index='method_type',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data.T, annot=True, fmt='.3f', cmap='coolwarm', ax=ax7)
        ax7.set_title('Average Performance by Method Type')
        
        # 8. Top methods summary
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        top_5 = self.results.head(5)
        summary_text = "TOP 5 METHODS:\n" + "="*40 + "\n"
        for idx, row in top_5.iterrows():
            summary_text += f"{row['method']}\n"
            summary_text += f"  Pearson: {row['pearson']:.4f}\n"
            summary_text += f"  R²: {row['r2']:.4f}\n"
            summary_text += f"  RMSE: {row['rmse']:.4f}\n"
            summary_text += f"  Features: {row['n_features']}\n"
            summary_text += "-"*40 + "\n"
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace')
        
        # 9. Method type performance comparison
        ax9 = plt.subplot(3, 3, 9)
        self.results.boxplot(column='pearson', by='method_type', ax=ax9)
        ax9.set_title('Performance Distribution by Method Type')
        ax9.set_xlabel('Method Type')
        ax9.set_ylabel('Pearson Correlation')
        
        plt.suptitle('Comprehensive Dimensionality Reduction Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig('dimensionality_reduction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional detailed plots
        self.create_detailed_comparisons()
    
    def create_detailed_comparisons(self):
        """Create additional detailed comparison plots"""
        # Plot 1: Feature selection methods comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Feature selection only
        fs_results = self.results[self.results['method_type'] == 'feature_selection'].copy()
        
        if len(fs_results) > 0:
            ax1 = axes[0, 0]
            fs_results = fs_results.sort_values('pearson', ascending=False)
            ax1.barh(range(len(fs_results)), fs_results['pearson'])
            ax1.set_yticks(range(len(fs_results)))
            ax1.set_yticklabels(fs_results['method'], fontsize=10)
            ax1.set_xlabel('Pearson Correlation')
            ax1.set_title('Feature Selection Methods Comparison')
            ax1.invert_yaxis()
        
        # Unsupervised methods
        unsup_results = self.results[self.results['method_type'] == 'unsupervised'].copy()
        
        if len(unsup_results) > 0:
            ax2 = axes[0, 1]
            unsup_results = unsup_results.sort_values('pearson', ascending=False)
            ax2.barh(range(len(unsup_results)), unsup_results['pearson'])
            ax2.set_yticks(range(len(unsup_results)))
            ax2.set_yticklabels(unsup_results['method'], fontsize=10)
            ax2.set_xlabel('Pearson Correlation')
            ax2.set_title('Unsupervised Methods Comparison')
            ax2.invert_yaxis()
        
        # Performance vs Reduction Time
        ax3 = axes[1, 0]
        ax3.scatter(self.results['reduction_time'], self.results['pearson'], alpha=0.6)
        for idx, row in self.results.head(5).iterrows():
            ax3.annotate(row['method'], (row['reduction_time'], row['pearson']), fontsize=8)
        ax3.set_xlabel('Reduction Time (seconds)')
        ax3.set_ylabel('Pearson Correlation')
        ax3.set_title('Performance vs Computation Time')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_stats = f"""
SUMMARY STATISTICS:
{'='*40}
Total methods tested: {len(self.results)}
Best method: {self.best_method}
Best Pearson: {self.results.iloc[0]['pearson']:.4f}
Best R²: {self.results['r2'].max():.4f}
Best RMSE: {self.results['rmse'].min():.4f}

Average by Type:
{'-'*40}
"""
        for method_type in self.results['method_type'].unique():
            type_results = self.results[self.results['method_type'] == method_type]
            summary_stats += f"{method_type}:\n"
            summary_stats += f"  Avg Pearson: {type_results['pearson'].mean():.4f}\n"
            summary_stats += f"  Avg Features: {type_results['n_features'].mean():.1f}\n"
        
        ax4.text(0.1, 0.9, summary_stats, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        plt.savefig('detailed_method_comparison.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def export_results(self):
        """Export results to CSV files"""
        # Save main results
        self.results.to_csv('dimensionality_reduction_results.csv', index=False)
        
        # Save top methods
        self.results.head(10).to_csv('top_10_methods.csv', index=False)
        
        # Save method type summary
        summary = self.results.groupby('method_type').agg({
            'pearson': ['mean', 'std', 'max'],
            'r2': ['mean', 'std', 'max'],
            'rmse': ['mean', 'std', 'min'],
            'n_features': ['mean', 'std'],
            'reduction_time': ['mean', 'std']
        }).round(4)
        summary.to_csv('method_type_summary.csv')
        
        print("\nResults saved:")
        print("  - dimensionality_reduction_results.csv")
        print("  - top_10_methods.csv")
        print("  - method_type_summary.csv")

def main():
    """Main execution"""
    print("="*60)
    print("FINANCIAL DATA DIMENSIONALITY REDUCTION EXPERIMENTS")
    print(f"Started at: {datetime.now()}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_path = os.path.join(DATA_PATH, 'train.parquet')
    df = pd.read_parquet(train_path)
    print(f"Total dataset size: {len(df):,} rows")
    
    # Initialize reducer
    reducer = FinancialDimensionalityReducer(
        n_components=50,  # Target dimensions
        sample_size=10000,  # Use 10k samples as requested
        n_jobs=4
    )
    
    # Optimize memory
    print("\nOptimizing memory...")
    df = reducer.optimize_dtypes(df)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
    print(f"Number of features: {len(feature_cols)}")
    
    # Run experiments
    start_time = time.time()
    results = reducer.run_experiments(df, feature_cols)
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS")
    print('='*60)
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Methods tested: {len(results)}")
    print(f"Best method: {reducer.best_method}")
    
    # Show top 10 results
    print("\nTop 10 Methods:")
    print("-"*80)
    print(f"{'Method':<30} {'Type':<15} {'Pearson':<10} {'R²':<10} {'RMSE':<10} {'Features':<10}")
    print("-"*80)
    
    for _, row in results.head(10).iterrows():
        print(f"{row['method']:<30} {row['method_type']:<15} "
              f"{row['pearson']:<10.4f} {row['r2']:<10.4f} "
              f"{row['rmse']:<10.4f} {int(row['n_features']):<10}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    reducer.create_comprehensive_visualization()
    
    # Export results
    reducer.export_results()
    
    # Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    top_3 = results.head(3)
    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"\n{idx}. {row['method']} ({row['method_type']})")
        print(f"   - Pearson: {row['pearson']:.4f}")
        print(f"   - Features: {int(row['n_features'])}")
        print(f"   - Reduction time: {row['reduction_time']:.2f}s")
    
    print(f"\nCompleted at: {datetime.now()}")
    print("="*60)
    
    return reducer, results

if __name__ == "__main__":
    reducer, results = main()

