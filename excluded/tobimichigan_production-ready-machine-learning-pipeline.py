!pip install torch-geometric


#!/usr/bin/env python3
"""
Complete Cheating Detection Pipeline
Production-ready implementation for Mercor Cheating Detection Challenge
FIXED: Target column name mismatch - using 'is_cheating' instead of 'is_cheat'
FIXED: Handling labeled and unlabeled training data
FIXED: Memory optimization with proper chunking
FIXED: Comprehensive EDA and visualization
FIXED: ReduceLROnPlateau verbose parameter issue
"""

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from collections import defaultdict, Counter
import math

# Machine Learning & Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, EdgeConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GraphDataLoader

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px

# Progress bars
from tqdm.auto import tqdm
import time

# Memory management
import psutil
import tracemalloc

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Memory configuration
MEMORY_LIMIT_MB = 1024 * 4  # 4GB limit
BATCH_SIZE = 256
CHUNK_SIZE = 10000  # Conservative chunk size

# Set non-interactive backend for matplotlib to reduce memory
plt.switch_backend('Agg')

# Initialize memory monitoring
tracemalloc.start()

# ============================================================================
# MEMORY MANAGEMENT & SAFETY
# ============================================================================

class MemoryMonitor:
    """Real-time memory usage tracking and safety checks"""
    
    def __init__(self, memory_limit_mb: int = MEMORY_LIMIT_MB):
        self.memory_limit_mb = memory_limit_mb
        self.peak_memory = 0
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, memory_mb)
        return memory_mb
    
    def check_memory_safe(self, buffer_mb: int = 200) -> bool:
        """Check if memory usage is within safe limits"""
        current_mb = self.get_memory_usage()
        return current_mb < (self.memory_limit_mb - buffer_mb)
    
    def force_cleanup(self):
        """Force garbage collection and clear caches"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.get_memory_usage()
    
    def log_memory_status(self, operation: str = ""):
        """Log current memory status"""
        current = self.get_memory_usage()
        print(f"[Memory] {operation}: Current={current:.1f}MB, Peak={self.peak_memory:.1f}MB")

memory_monitor = MemoryMonitor()

# ============================================================================
# DATA LOADING WITH MEMORY SAFETY
# ============================================================================

def load_data_chunked(file_path: str, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """
    Ultra-conservative chunked data loading with memory safety
    """
    print(f"Loading {file_path} in chunks...")
    
    chunks = []
    chunk_counter = 0
    max_chunks = 5  # Increased limit for better sampling
    
    try:
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), 
                         desc="Loading chunks"):
            chunks.append(chunk)
            chunk_counter += 1
            
            # Memory safety check
            if not memory_monitor.check_memory_safe(500):
                print(f"Warning: Memory approaching limit after {chunk_counter} chunks")
                break
                
            if chunk_counter >= max_chunks:
                print(f"Reached maximum chunk limit ({max_chunks})")
                break
                
            # Force cleanup between chunks
            memory_monitor.force_cleanup()
            
    except Exception as e:
        print(f"Error loading chunks: {e}")
        if chunks:
            print("Returning partial data...")
        else:
            raise
    
    if chunks:
        data = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(data)} rows from {chunk_counter} chunks")
        return data
    else:
        return pd.DataFrame()

def load_social_graph_safe(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load social graph with memory-efficient processing
    Returns: DataFrame and adjacency dictionary
    """
    print("Loading social graph with memory optimization...")
    
    # Load in chunks
    graph_df = load_data_chunked(file_path, chunk_size=50000)
    
    # Build adjacency list in memory-efficient way
    adjacency_dict = defaultdict(set)
    
    for _, row in tqdm(graph_df.iterrows(), total=len(graph_df), desc="Building adjacency"):
        user_a = row['user_a']
        user_b = row['user_b']
        adjacency_dict[user_a].add(user_b)
        adjacency_dict[user_b].add(user_a)
        
        # Periodic memory check
        if _ % 100000 == 0:
            if not memory_monitor.check_memory_safe(300):
                print("Memory limit reached during graph building")
                break
    
    # Convert sets to lists for easier handling
    adjacency_dict = {k: list(v) for k, v in adjacency_dict.items()}
    
    print(f"Graph loaded: {len(graph_df)} edges, {len(adjacency_dict)} nodes")
    return graph_df, adjacency_dict

# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

class EDAAnalyzer:
    """Comprehensive EDA with visualization"""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict, target_col: str = 'is_cheating'):
        self.df = df
        self.metadata = metadata
        self.target_col = target_col
        self.numeric_cols = []
        self.binary_cols = []
        self.cat_cols = []
        
        # Classify columns
        self._classify_columns()
    
    def _classify_columns(self):
        """Classify columns based on metadata"""
        for col in self.df.columns:
            if col.startswith('feature_'):
                if col in self.metadata:
                    if self.metadata[col]['type'] == 'binary':
                        self.binary_cols.append(col)
                    else:
                        self.numeric_cols.append(col)
            elif col not in ['user_hash', self.target_col, 'high_conf_clean']:
                self.cat_cols.append(col)
    
    def analyze_missing_values(self):
        """Analyze missing values pattern"""
        print("\n=== Missing Values Analysis ===")
        missing_stats = self.df.isnull().sum()
        missing_percent = (missing_stats / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing_stats,
            'missing_percent': missing_percent
        }).sort_values('missing_percent', ascending=False)
        
        # Plot missing values
        plt.figure(figsize=(15, 8))
        missing_df[missing_df['missing_percent'] > 0]['missing_percent'].plot(
            kind='bar', color='coral'
        )
        plt.title('Missing Values Percentage by Column', fontsize=16)
        plt.xlabel('Columns')
        plt.ylabel('Percentage Missing')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('missing_values_analysis.png')
        plt.show()
        
        print(missing_df[missing_df['missing_percent'] > 0])
        return missing_df
    
    def analyze_target_distribution(self):
        """Analyze target variable distribution"""
        print("\n=== Target Variable Analysis ===")
        
        if self.target_col in self.df.columns:
            target_counts = self.df[self.target_col].value_counts(dropna=False)
            target_percent = (target_counts / len(self.df)) * 100
            
            print(f"Target distribution:\n{target_counts}")
            print(f"Target percentages:\n{target_percent}")
            
            # Plot target distribution
            plt.figure(figsize=(10, 6))
            colors = ['lightgreen', 'lightcoral', 'lightblue']
            target_counts.plot(kind='bar', color=colors[:len(target_counts)])
            plt.title('Target Variable Distribution', fontsize=16)
            plt.xlabel('Target Value')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig('target_distribution.png')
            plt.show()
            
            return target_counts
        else:
            print(f"Target column '{self.target_col}' not found in data")
            return None
    
    def analyze_feature_distributions(self):
        """Analyze distributions of all features"""
        print("\n=== Feature Distributions Analysis ===")
        
        # Numeric features
        if self.numeric_cols:
            # Select subset for visualization
            plot_cols = self.numeric_cols[:min(12, len(self.numeric_cols))]
            
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            axes = axes.flatten()
            
            for idx, col in enumerate(plot_cols):
                if idx < len(axes):
                    # Plot distribution
                    axes[idx].hist(self.df[col].dropna(), bins=50, alpha=0.7, color='steelblue')
                    axes[idx].set_title(f'{col} Distribution', fontsize=10)
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                    
                    # Add statistics
                    mean_val = self.df[col].mean()
                    median_val = self.df[col].median()
                    axes[idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                    axes[idx].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                    axes[idx].legend(fontsize=8)
            
            plt.tight_layout()
            plt.savefig('numeric_features_distribution.png')
            plt.show()
        
        # Binary features
        if self.binary_cols:
            plot_cols = self.binary_cols[:min(8, len(self.binary_cols))]
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(plot_cols):
                if idx < len(axes):
                    value_counts = self.df[col].value_counts().sort_index()
                    axes[idx].bar(value_counts.index.astype(str), value_counts.values, 
                                 color=['lightblue', 'lightcoral'])
                    axes[idx].set_title(f'{col} Distribution', fontsize=10)
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Count')
                    
                    # Add percentages
                    total = len(self.df[col].dropna())
                    for i, v in enumerate(value_counts.values):
                        percentage = (v / total) * 100
                        axes[idx].text(i, v + total * 0.01, f'{percentage:.1f}%', 
                                      ha='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('binary_features_distribution.png')
            plt.show()
    
    def analyze_correlations(self):
        """Analyze correlations between features and target"""
        print("\n=== Correlation Analysis ===")
        
        # Select numeric columns only
        analysis_cols = self.numeric_cols + self.binary_cols
        if self.target_col in self.df.columns:
            analysis_cols = [self.target_col] + analysis_cols
        
        # Compute correlation matrix
        corr_matrix = self.df[analysis_cols].corr(numeric_only=True)
        
        # Plot heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.show()
        
        # Get correlations with target
        if self.target_col in self.df.columns and self.target_col in corr_matrix.columns:
            target_correlations = corr_matrix[self.target_col].drop(self.target_col)
            target_correlations = target_correlations.sort_values(key=abs, ascending=False)
            
            print("\nTop 20 features correlated with target:")
            print(target_correlations.head(20))
            
            # Plot top correlations
            plt.figure(figsize=(12, 8))
            target_correlations.head(20).plot(kind='barh', color='steelblue')
            plt.title('Top 20 Feature Correlations with Target', fontsize=16)
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            plt.savefig('target_correlations.png')
            plt.show()
            
            return target_correlations
        
        return None
    
    def analyze_outliers(self):
        """Analyze outliers in numeric features"""
        print("\n=== Outlier Analysis ===")
        
        outlier_stats = []
        for col in self.numeric_cols:
            if col in self.df.columns:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_percent = (len(outliers) / len(self.df)) * 100
                
                outlier_stats.append({
                    'feature': col,
                    'outlier_count': len(outliers),
                    'outlier_percent': outlier_percent,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
        
        outlier_df = pd.DataFrame(outlier_stats).sort_values('outlier_percent', ascending=False)
        
        print("Features with most outliers:")
        print(outlier_df.head(10))
        
        return outlier_df
    
    def perform_comprehensive_eda(self, output_dir: str = './eda_results'):
        """Perform comprehensive EDA with all analyses"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("COMPREHENSIVE EDA ANALYSIS")
        print("="*80)
        
        # 1. Basic info
        print("\n1. Dataset Information:")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes.value_counts()}")
        
        # 2. Missing values analysis
        missing_df = self.analyze_missing_values()
        
        # 3. Target analysis
        target_stats = self.analyze_target_distribution()
        
        # 4. Feature distributions
        self.analyze_feature_distributions()
        
        # 5. Correlation analysis
        correlations = self.analyze_correlations()
        
        # 6. Outlier analysis
        outliers = self.analyze_outliers()
        
        # 7. Statistical summary
        print("\n7. Statistical Summary:")
        print(self.df.describe())
        
        # Save EDA report
        self._save_eda_report(missing_df, target_stats, correlations, outliers, output_dir)
        
        print("\n" + "="*80)
        print("EDA COMPLETE")
        print("="*80)
    
    def _save_eda_report(self, missing_df, target_stats, correlations, outliers, output_dir):
        """Save comprehensive EDA report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE EDA REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nAnalysis Date: {datetime.now()}")
        report_lines.append(f"Dataset Shape: {self.df.shape}")
        
        if target_stats is not None:
            report_lines.append("\nTarget Distribution:")
            for val, count in target_stats.items():
                report_lines.append(f"  {val}: {count} ({count/len(self.df)*100:.2f}%)")
        
        if missing_df is not None:
            report_lines.append("\nMissing Values Summary:")
            high_missing = missing_df[missing_df['missing_percent'] > 20]
            if len(high_missing) > 0:
                report_lines.append("Columns with >20% missing values:")
                for idx, row in high_missing.iterrows():
                    report_lines.append(f"  {idx}: {row['missing_percent']:.2f}%")
        
        if correlations is not None:
            report_lines.append("\nTop Feature Correlations with Target:")
            top_corr = correlations.head(10)
            for feat, corr in top_corr.items():
                report_lines.append(f"  {feat}: {corr:.4f}")
        
        if outliers is not None:
            report_lines.append("\nOutlier Analysis (Top 5):")
            top_outliers = outliers.head()
            for _, row in top_outliers.iterrows():
                report_lines.append(f"  {row['feature']}: {row['outlier_percent']:.2f}% outliers")
        
        # Write report to file
        report_path = os.path.join(output_dir, 'eda_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"EDA report saved to: {report_path}")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Comprehensive data preprocessing with missing value handling"""
    
    def __init__(self, metadata_path: str, is_training: bool = True):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_columns = [f'feature_{i:03d}' for i in range(1, 19)]
        self.numeric_features = []
        self.binary_features = []
        self.imputers = {}
        self.scalers = {}
        self.is_training = is_training
        
        # Classify features
        for feat in self.feature_columns:
            if feat in self.metadata:
                if self.metadata[feat]['type'] == 'binary':
                    self.binary_features.append(feat)
                else:
                    self.numeric_features.append(feat)
    
    def create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create indicators for missing values"""
        for feat in self.feature_columns:
            if feat in df.columns:
                df[f'{feat}_missing'] = df[feat].isna().astype(int)
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'iterative') -> pd.DataFrame:
        """
        Advanced missing value imputation with strategy selection
        """
        df = df.copy()
        
        # First, create missing indicators
        df = self.create_missing_indicators(df)
        
        # Different strategies for different feature types
        for feat in self.feature_columns:
            if feat not in df.columns:
                continue
                
            missing_mask = df[feat].isna()
            
            if missing_mask.sum() == 0:
                continue
            
            if feat in self.binary_features:
                # Binary features: impute with mode
                if strategy == 'advanced':
                    # Use KNN for binary features with similar patterns
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[feat] = imputer.fit_transform(df[[feat]])
                else:
                    df[feat] = df[feat].fillna(df[feat].mode()[0] if not df[feat].mode().empty else 0)
            
            else:
                # Numeric features
                if strategy == 'iterative':
                    # Use iterative imputer for complex patterns
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    
                    imputer = IterativeImputer(max_iter=10, random_state=SEED)
                    other_features = [f for f in self.feature_columns if f != feat and f in df.columns]
                    if len(other_features) > 0:
                        temp_df = df[other_features + [feat]].copy()
                        for col in other_features:
                            temp_df[col] = temp_df[col].fillna(temp_df[col].median())
                        
                        imputed = imputer.fit_transform(temp_df)
                        df[feat] = imputed[:, -1]
                
                elif strategy == 'knn':
                    # KNN imputation
                    imputer = KNNImputer(n_neighbors=5)
                    other_features = [f for f in self.feature_columns if f != feat and f in df.columns]
                    if len(other_features) > 0:
                        temp_df = df[other_features + [feat]].copy()
                        temp_df[other_features] = temp_df[other_features].fillna(
                            temp_df[other_features].median()
                        )
                        imputed = imputer.fit_transform(temp_df)
                        df[feat] = imputed[:, -1]
                
                else:
                    # Simple median imputation with feature-specific adjustments
                    if feat == 'feature_010':  # Large range feature
                        df[feat] = df[feat].fillna(0)  # Often missing means 0 for this feature
                    else:
                        df[feat] = df[feat].fillna(df[feat].median())
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Robust feature scaling"""
        df = df.copy()
        
        for feat in self.numeric_features:
            if feat not in df.columns:
                continue
            
            if fit:
                # Use RobustScaler for numeric features (less sensitive to outliers)
                scaler = RobustScaler()
                df[feat] = scaler.fit_transform(df[[feat]])
                self.scalers[feat] = scaler
            else:
                if feat in self.scalers:
                    df[feat] = self.scalers[feat].transform(df[[feat]])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new engineered features"""
        df = df.copy()
        
        # Interaction features
        if all(f in df.columns for f in ['feature_001', 'feature_002']):
            df['feat_01_02_interaction'] = df['feature_001'] * df['feature_002']
        
        if all(f in df.columns for f in ['feature_015', 'feature_016']):
            df['feat_15_16_ratio'] = df['feature_015'] / (df['feature_016'] + 1e-6)
        
        # Statistical features
        numeric_cols = [f for f in self.numeric_features if f in df.columns]
        if numeric_cols:
            df['feat_numeric_mean'] = df[numeric_cols].mean(axis=1)
            df['feat_numeric_std'] = df[numeric_cols].std(axis=1)
            df['feat_numeric_skew'] = df[numeric_cols].skew(axis=1)
            df['feat_numeric_min'] = df[numeric_cols].min(axis=1)
            df['feat_numeric_max'] = df[numeric_cols].max(axis=1)
        
        # Missing value patterns
        missing_cols = [c for c in df.columns if 'missing' in c]
        if missing_cols:
            df['missing_count'] = df[missing_cols].sum(axis=1)
            df['missing_ratio'] = df['missing_count'] / len(missing_cols)
        
        # Outlier detection features
        for col in numeric_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df[f'{col}_is_outlier'] = ((df[col] < (q1 - 1.5 * iqr)) | 
                                          (df[col] > (q3 + 1.5 * iqr))).astype(int)
        
        return df
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline - FIXED to preserve target column"""
        memory_monitor.log_memory_status("Preprocessing start")
        
        # Store target column if it exists
        target_col = None
        if 'is_cheating' in df.columns:
            target_col = df['is_cheating'].copy()
        
        # Store high_conf_clean if it exists
        high_conf_col = None
        if 'high_conf_clean' in df.columns:
            high_conf_col = df['high_conf_clean'].copy()
        
        # Store user_hash if it exists
        user_hash_col = None
        if 'user_hash' in df.columns:
            user_hash_col = df['user_hash'].copy()
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy='knn')
        
        # Scale features
        df = self.scale_features(df, fit=fit)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Restore columns if they existed
        if target_col is not None:
            df['is_cheating'] = target_col.values
        if high_conf_col is not None:
            df['high_conf_clean'] = high_conf_col.values
        if user_hash_col is not None:
            df['user_hash'] = user_hash_col.values
        
        memory_monitor.force_cleanup()
        memory_monitor.log_memory_status("Preprocessing end")
        
        return df

# ============================================================================
# GRAPH FEATURE ENGINEERING
# ============================================================================

class GraphFeatureEngineer:
    """Extract features from social graph"""
    
    def __init__(self, adjacency_dict: Dict):
        self.adjacency_dict = adjacency_dict
        self.node_features = {}
        
    def extract_basic_features(self, user_hash: str) -> Dict:
        """Extract basic graph features for a node"""
        neighbors = self.adjacency_dict.get(user_hash, [])
        
        features = {
            'graph_degree': len(neighbors),
            'graph_has_connections': int(len(neighbors) > 0),
            'graph_log_degree': np.log1p(len(neighbors)),
        }
        
        return features
    
    def extract_egonet_features(self, user_hash: str, max_depth: int = 2) -> Dict:
        """Extract ego-network features"""
        if user_hash not in self.adjacency_dict:
            return {}
        
        # BFS to get nodes within max_depth
        visited = set([user_hash])
        queue = [(user_hash, 0)]
        nodes_at_depth = defaultdict(set)
        
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
                
            for neighbor in self.adjacency_dict.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_at_depth[depth + 1].add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        features = {}
        for depth in range(1, max_depth + 1):
            features[f'graph_neighbors_depth_{depth}'] = len(nodes_at_depth[depth])
        
        features['graph_total_reachable'] = len(visited) - 1
        features['graph_avg_degree_egonet'] = np.mean([
            len(self.adjacency_dict.get(n, [])) 
            for n in visited
        ]) if visited else 0
        
        # Calculate clustering coefficient
        neighbors = self.adjacency_dict.get(user_hash, [])
        if len(neighbors) >= 2:
            # Count edges between neighbors
            neighbor_edges = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if n2 in self.adjacency_dict.get(n1, []):
                        neighbor_edges += 1
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            features['graph_clustering_coefficient'] = neighbor_edges / possible_edges
        else:
            features['graph_clustering_coefficient'] = 0
        
        return features
    
    def extract_community_features(self, user_hash: str, 
                                  labeled_users: set = None) -> Dict:
        """Extract community-based features"""
        if user_hash not in self.adjacency_dict:
            return {}
        
        neighbors = self.adjacency_dict[user_hash]
        
        features = {
            'graph_neighbor_count': len(neighbors),
        }
        
        if labeled_users:
            # Calculate fraction of neighbors that are labeled
            labeled_neighbors = [n for n in neighbors if n in labeled_users]
            features['graph_labeled_neighbor_ratio'] = len(labeled_neighbors) / max(len(neighbors), 1)
            features['graph_labeled_neighbor_count'] = len(labeled_neighbors)
        
        return features
    
    def extract_all_features(self, user_hashes: List[str], 
                            labeled_users: set = None) -> pd.DataFrame:
        """Extract all graph features for a list of users"""
        print("Extracting graph features...")
        
        features_list = []
        
        for user_hash in tqdm(user_hashes, desc="Graph features"):
            feat_dict = {}
            
            # Basic features
            basic_feats = self.extract_basic_features(user_hash)
            feat_dict.update(basic_feats)
            
            # Egonet features (compute only if has connections)
            if basic_feats.get('graph_has_connections', 0):
                ego_feats = self.extract_egonet_features(user_hash, max_depth=2)
                feat_dict.update(ego_feats)
            
            # Community features
            if labeled_users is not None:
                comm_feats = self.extract_community_features(user_hash, labeled_users)
                feat_dict.update(comm_feats)
            
            feat_dict['user_hash'] = user_hash
            features_list.append(feat_dict)
        
        features_df = pd.DataFrame(features_list)
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        return features_df

# ============================================================================
# ENSEMBLE MODEL WITH ATTENTION
# ============================================================================

class AttentionBlock(nn.Module):
    """Self-attention block for tabular features"""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class TabularNN(nn.Module):
    """Neural network for tabular data with attention mechanism"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3,
                 use_attention: bool = True):
        super(TabularNN, self).__init__()
        
        # Input layer
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Hidden layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(current_dim, num_heads=4, dropout=dropout)
            self.attention_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(current_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Attention mechanism
        if self.use_attention:
            # Reshape for attention: add sequence dimension
            x_attn = x.unsqueeze(1)  # (batch, 1, features)
            x_attn = self.attention(x_attn)
            # Pool across sequence dimension
            x = x_attn.squeeze(1)
        
        # Output
        x = self.output(x)
        
        return torch.sigmoid(x)

class HybridCheatingDetector(nn.Module):
    """
    Hybrid ensemble model combining multiple architectures
    """
    
    def __init__(self, 
                 tabular_input_dim: int,
                 graph_input_dim: Optional[int] = None,
                 use_graph: bool = False):
        super(HybridCheatingDetector, self).__init__()
        
        self.use_graph = use_graph
        
        # Tabular model
        self.tabular_nn = TabularNN(
            input_dim=tabular_input_dim,
            hidden_dims=[128, 64, 32],
            dropout=0.3,
            use_attention=True
        )
        
        # Graph model (if available)
        if use_graph and graph_input_dim:
            self.gnn = nn.Sequential(
                nn.Linear(graph_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        # LightGBM-like feature extractor
        self.lgbm_features = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        
        # Calculate fusion dimension
        fusion_dim = 1 + (1 if use_graph and graph_input_dim else 0) + 8
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, tabular_features, graph_features=None):
        # Tabular features
        tabular_out = self.tabular_nn(tabular_features)
        
        # LightGBM-like features
        lgbm_feats = self.lgbm_features(tabular_features)
        
        # Graph features (if available)
        if self.use_graph and graph_features is not None:
            graph_out = self.gnn(graph_features)
            combined = torch.cat([tabular_out, graph_out, lgbm_feats], dim=1)
        else:
            combined = torch.cat([tabular_out, lgbm_feats], dim=1)
        
        # Final prediction
        output = self.fusion(combined)
        
        return torch.sigmoid(output)

# ============================================================================
# DATA LOADERS & DATASETS
# ============================================================================

class CheatingDataset(Dataset):
    """PyTorch Dataset for cheating detection"""
    
    def __init__(self, 
                 features: np.ndarray, 
                 targets: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None,
                 graph_features: Optional[np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.weights = torch.FloatTensor(weights) if weights is not None else None
        self.graph_features = torch.FloatTensor(graph_features) if graph_features is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            if self.weights is not None and self.graph_features is not None:
                return (self.features[idx], self.graph_features[idx]), self.targets[idx], self.weights[idx]
            elif self.weights is not None:
                return self.features[idx], self.targets[idx], self.weights[idx]
            elif self.graph_features is not None:
                return (self.features[idx], self.graph_features[idx]), self.targets[idx]
            else:
                return self.features[idx], self.targets[idx]
        else:
            if self.graph_features is not None:
                return (self.features[idx], self.graph_features[idx])
            else:
                return self.features[idx]

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = model.state_dict().copy()

def train_model(model, 
                train_loader, 
                val_loader,
                epochs: int = 50,
                lr: float = 1e-3,
                device: str = 'cpu',
                use_graph: bool = False):
    """Train model with early stopping"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping(patience=10)
    
    model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            if use_graph:
                (features, graph_features), targets, weights = batch
                features = features.to(device)
                graph_features = graph_features.to(device)
            else:
                if len(batch) == 3:
                    features, targets, weights = batch
                else:
                    features, targets = batch
                    weights = None
                features = features.to(device)
                graph_features = None
            
            targets = targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass
            if use_graph:
                outputs = model(features, graph_features)
            else:
                outputs = model(features)
            
            # Compute loss
            if weights is not None:
                weights = weights.to(device)
                loss = (criterion(outputs, targets) * weights).mean()
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                if use_graph:
                    (features, graph_features), targets = batch
                    features = features.to(device)
                    graph_features = graph_features.to(device)
                else:
                    features, targets = batch
                    features = features.to(device)
                    graph_features = None
                
                targets = targets.to(device).unsqueeze(1)
                
                if use_graph:
                    outputs = model(features, graph_features)
                else:
                    outputs = model(features)
                
                loss = criterion(outputs, targets)
                
                val_losses.append(loss.item())
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        
        avg_val_loss = np.mean(val_losses)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_preds_binary = [1 if p > 0.5 else 0 for p in val_preds]
        val_f1 = f1_score(val_targets, val_preds_binary)
        val_precision = precision_score(val_targets, val_preds_binary)
        val_recall = recall_score(val_targets, val_preds_binary)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val AUC={val_auc:.4f}, "
              f"F1={val_f1:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'val_auc': best_val_auc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'best_model.pth')
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Early stopping
        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            if early_stopping.best_model:
                model.load_state_dict(early_stopping.best_model)
            break
        
        # Memory cleanup
        memory_monitor.force_cleanup()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history

# ============================================================================
# MODEL EVALUATION AND VISUALIZATION
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation with visualization"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def evaluate(self, data_loader, use_graph=False):
        """Evaluate model on given data loader"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                if use_graph:
                    (features, graph_features), targets = batch
                    features = features.to(self.device)
                    graph_features = graph_features.to(self.device)
                    outputs = self.model(features, graph_features)
                else:
                    features, targets = batch
                    features = features.to(self.device)
                    outputs = self.model(features)
                
                all_probs.extend(outputs.cpu().numpy().flatten())
                all_preds.extend([1 if p > 0.5 else 0 for p in outputs.cpu().numpy().flatten()])
                all_targets.extend(targets.numpy().flatten())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds),
            'recall': recall_score(all_targets, all_preds),
            'f1': f1_score(all_targets, all_preds),
            'auc': roc_auc_score(all_targets, all_probs),
            'ap': average_precision_score(all_targets, all_probs)
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        return metrics, cm, all_probs, all_targets
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Cheating', 'Cheating'],
                   yticklabels=['Not Cheating', 'Cheating'])
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    def plot_roc_curve(self, y_true, y_probs, save_path='roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        auc_score = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_probs, save_path='precision_recall_curve.png'):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        ap_score = average_precision_score(y_true, y_probs)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    def plot_training_history(self, history, save_path='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC curve
        axes[0, 1].plot(history['val_auc'], label='Val AUC', color='green')
        axes[0, 1].set_title('Validation AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 score curve
        axes[0, 2].plot(history['val_f1'], label='Val F1', color='orange')
        axes[0, 2].set_title('Validation F1 Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Precision curve
        axes[1, 0].plot(history['val_precision'], label='Val Precision', color='red')
        axes[1, 0].set_title('Validation Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall curve
        axes[1, 1].plot(history['val_recall'], label='Val Recall', color='purple')
        axes[1, 1].set_title('Validation Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Best values
        best_epoch = np.argmax(history['val_auc'])
        axes[1, 2].axis('off')
        axes[1, 2].text(0.1, 0.9, 'Best Performance:', fontsize=12, fontweight='bold')
        axes[1, 2].text(0.1, 0.7, f'Epoch: {best_epoch + 1}', fontsize=10)
        axes[1, 2].text(0.1, 0.6, f'Best AUC: {history["val_auc"][best_epoch]:.4f}', fontsize=10)
        axes[1, 2].text(0.1, 0.5, f'Best F1: {history["val_f1"][best_epoch]:.4f}', fontsize=10)
        axes[1, 2].text(0.1, 0.4, f'Best Precision: {history["val_precision"][best_epoch]:.4f}', fontsize=10)
        axes[1, 2].text(0.1, 0.3, f'Best Recall: {history["val_recall"][best_epoch]:.4f}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    def plot_prediction_distribution(self, y_probs, y_true=None, save_path='prediction_distribution.png'):
        """Plot prediction distribution"""
        plt.figure(figsize=(12, 6))
        
        if y_true is not None:
            # Separate predictions by true class
            cheating_probs = [p for p, t in zip(y_probs, y_true) if t == 1]
            not_cheating_probs = [p for p, t in zip(y_probs, y_true) if t == 0]
            
            plt.hist(not_cheating_probs, bins=50, alpha=0.5, label='Not Cheating', color='green')
            plt.hist(cheating_probs, bins=50, alpha=0.5, label='Cheating', color='red')
            plt.legend()
        else:
            plt.hist(y_probs, bins=50, alpha=0.7, color='steelblue')
        
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("="*80)
    print("CHEATING DETECTION PIPELINE - PRODUCTION VERSION")
    print("="*80)
    
    memory_monitor.log_memory_status("Start")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    DATA_DIR = "/kaggle/input/mercor-cheating-detection"
    METADATA_PATH = os.path.join(DATA_DIR, "feature_metadata.json")
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    GRAPH_PATH = os.path.join(DATA_DIR, "social_graph.csv")
    
    USE_GRAPH = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Load training data
    train_df = load_data_chunked(TRAIN_PATH)
    print(f"Training data shape: {train_df.shape}")
    
    # Load test data
    test_df = load_data_chunked(TEST_PATH)
    print(f"Test data shape: {test_df.shape}")
    
    # Load social graph
    if USE_GRAPH:
        graph_df, adjacency_dict = load_social_graph_safe(GRAPH_PATH)
    else:
        adjacency_dict = None
    
    memory_monitor.log_memory_status("Data loaded")
    
    # ========================================================================
    # EXPLORATORY DATA ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Perform EDA on training data
    eda_analyzer = EDAAnalyzer(train_df, metadata, target_col='is_cheating')
    eda_analyzer.perform_comprehensive_eda()
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    
    print("\n" + "="*80)
    print("PREPROCESSING")
    print("="*80)
    
    # Separate labeled and unlabeled data
    labeled_train = train_df[train_df['is_cheating'].notna()].copy()
    unlabeled_train = train_df[train_df['is_cheating'].isna()].copy()
    
    print(f"Labeled training samples: {len(labeled_train)}")
    print(f"Unlabeled training samples: {len(unlabeled_train)}")
    
    # Initialize preprocessors
    train_preprocessor = DataPreprocessor(METADATA_PATH, is_training=True)
    test_preprocessor = DataPreprocessor(METADATA_PATH, is_training=False)
    
    # Preprocess labeled training data
    labeled_train_processed = train_preprocessor.preprocess(labeled_train, fit=True)
    
    # Preprocess test data
    test_processed = test_preprocessor.preprocess(test_df, fit=False)
    
    memory_monitor.log_memory_status("Preprocessing complete")
    
    # ========================================================================
    # GRAPH FEATURES
    # ========================================================================
    
    if USE_GRAPH and adjacency_dict:
        print("\n" + "="*80)
        print("EXTRACTING GRAPH FEATURES")
        print("="*80)
        
        graph_engineer = GraphFeatureEngineer(adjacency_dict)
        
        # Extract for labeled training data
        train_graph_feats = graph_engineer.extract_all_features(
            labeled_train_processed['user_hash'].values,
            labeled_users=set(labeled_train_processed['user_hash'].values)
        )
        
        # Extract for test data
        test_graph_feats = graph_engineer.extract_all_features(
            test_processed['user_hash'].values,
            labeled_users=set(labeled_train_processed['user_hash'].values)
        )
        
        # Merge with main data
        labeled_train_processed = pd.merge(labeled_train_processed, train_graph_feats, on='user_hash', how='left')
        test_processed = pd.merge(test_processed, test_graph_feats, on='user_hash', how='left')
        
        # Fill missing graph features with 0
        graph_cols = [c for c in train_graph_feats.columns if c != 'user_hash']
        labeled_train_processed[graph_cols] = labeled_train_processed[graph_cols].fillna(0)
        test_processed[graph_cols] = test_processed[graph_cols].fillna(0)
        
        memory_monitor.log_memory_status("Graph features extracted")
    
    # ========================================================================
    # PREPARE FEATURES
    # ========================================================================
    
    print("\n" + "="*80)
    print("PREPARING FEATURES")
    print("="*80)
    
    # Define feature columns - exclude non-feature columns
    exclude_cols = ['user_hash', 'is_cheating', 'high_conf_clean']
    feature_cols = [c for c in labeled_train_processed.columns if c not in exclude_cols]
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    
    # Prepare labeled training data
    X = labeled_train_processed[feature_cols].values
    y = labeled_train_processed['is_cheating'].values
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Prepare test data
    X_test = test_processed[feature_cols].values
    
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"Class distribution - Val: {np.bincount(y_val.astype(int))}")
    
    # ========================================================================
    # CREATE DATA LOADERS
    # ========================================================================
    
    print("\n" + "="*80)
    print("CREATING DATA LOADERS")
    print("="*80)
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train.astype(int))
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train.astype(int)]
    
    # Create datasets
    train_dataset = CheatingDataset(X_train, y_train, sample_weights)
    val_dataset = CheatingDataset(X_val, y_val)
    test_dataset = CheatingDataset(X_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    # ========================================================================
    # BUILD MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    
    input_dim = X_train.shape[1]
    
    model = HybridCheatingDetector(
        tabular_input_dim=input_dim,
        use_graph=False
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=1e-3,
        device=DEVICE,
        use_graph=False
    )
    
    # ========================================================================
    # EVALUATE MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("EVALUATING MODEL")
    print("="*80)
    
    evaluator = ModelEvaluator(model, device=DEVICE)
    
    # Evaluate on validation set
    val_metrics, val_cm, val_probs, val_targets = evaluator.evaluate(val_loader, use_graph=False)
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot evaluation results
    evaluator.plot_confusion_matrix(val_cm)
    evaluator.plot_roc_curve(val_targets, val_probs)
    evaluator.plot_precision_recall_curve(val_targets, val_probs)
    evaluator.plot_training_history(history)
    evaluator.plot_prediction_distribution(val_probs, val_targets)
    
    # ========================================================================
    # GENERATE PREDICTIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            features = batch.to(DEVICE)
            outputs = model(features)
            test_preds.extend(outputs.cpu().numpy().flatten())
    
    # ========================================================================
    # CREATE SUBMISSION
    # ========================================================================
    
    print("\n" + "="*80)
    print("CREATING SUBMISSION")
    print("="*80)
    
    # Ensure we have the correct number of predictions
    if len(test_preds) != len(test_processed):
        print(f"Warning: Mismatch between predictions ({len(test_preds)}) and test samples ({len(test_processed)})")
        # Align predictions with test data
        test_preds = test_preds[:len(test_processed)]
    
    submission = pd.DataFrame({
        'user_hash': test_processed['user_hash'],
        'prediction': test_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print(f"Submission saved: {len(submission)} predictions")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {np.mean(test_preds):.4f}")
    print(f"  Median: {np.median(test_preds):.4f}")
    print(f"  Std: {np.std(test_preds):.4f}")
    print(f"  Min: {np.min(test_preds):.4f}")
    print(f"  Max: {np.max(test_preds):.4f}")
    
    # Plot prediction distribution
    plt.figure(figsize=(12, 6))
    plt.hist(test_preds, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title('Test Set Prediction Distribution', fontsize=16)
    plt.xlabel('Predicted Probability of Cheating')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_predictions_distribution.png')
    plt.show()
    
    # ========================================================================
    # FINAL CLEANUP AND REPORT
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL PERFORMANCE REPORT")
    print("="*80)
    
    print(f"\nBest Validation Performance:")
    best_epoch = np.argmax(history['val_auc'])
    print(f"  Epoch: {best_epoch + 1}")
    print(f"  AUC: {history['val_auc'][best_epoch]:.4f}")
    print(f"  F1 Score: {history['val_f1'][best_epoch]:.4f}")
    print(f"  Precision: {history['val_precision'][best_epoch]:.4f}")
    print(f"  Recall: {history['val_recall'][best_epoch]:.4f}")
    
    # Generate comprehensive report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("CHEATING DETECTION PIPELINE - FINAL REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nAnalysis Date: {datetime.now()}")
    report_lines.append(f"Device Used: {DEVICE}")
    report_lines.append(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    report_lines.append(f"\nDataset Statistics:")
    report_lines.append(f"  Total Training Samples: {len(train_df)}")
    report_lines.append(f"  Labeled Training Samples: {len(labeled_train)}")
    report_lines.append(f"  Unlabeled Training Samples: {len(unlabeled_train)}")
    report_lines.append(f"  Test Samples: {len(test_df)}")
    report_lines.append(f"\nTraining Details:")
    report_lines.append(f"  Features Used: {len(feature_cols)}")
    report_lines.append(f"  Batch Size: {BATCH_SIZE}")
    report_lines.append(f"  Epochs Trained: {len(history['train_loss'])}")
    report_lines.append(f"\nBest Validation Metrics:")
    report_lines.append(f"  AUC: {history['val_auc'][best_epoch]:.4f}")
    report_lines.append(f"  F1 Score: {history['val_f1'][best_epoch]:.4f}")
    report_lines.append(f"  Precision: {history['val_precision'][best_epoch]:.4f}")
    report_lines.append(f"  Recall: {history['val_recall'][best_epoch]:.4f}")
    report_lines.append(f"\nTest Predictions:")
    report_lines.append(f"  Mean Prediction: {np.mean(test_preds):.4f}")
    report_lines.append(f"  Median Prediction: {np.median(test_preds):.4f}")
    report_lines.append(f"  Std Prediction: {np.std(test_preds):.4f}")
    report_lines.append(f"  Min Prediction: {np.min(test_preds):.4f}")
    report_lines.append(f"  Max Prediction: {np.max(test_preds):.4f}")
    
    # Save report
    with open('pipeline_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nPipeline report saved to: pipeline_report.txt")
    
    memory_monitor.log_memory_status("Complete")
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SUBMISSION.CSV GENERATED")
    print("="*80)

if __name__ == "__main__":
    main()

