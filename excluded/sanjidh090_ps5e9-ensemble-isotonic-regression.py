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


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


train


test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test


print(f"{train.describe()}, {test.describe()}")



"""
ğŸ�µ COMPREHENSIVE MUSIC DATASET ANALYSIS - PYTHON TOOLKIT
===========================================================
Complete data science analysis with advanced visualizations for music audio features
Designed for Kaggle Playground Series S5E9 dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MusicDataAnalyzer:
    """
    ğŸ�¼ Advanced Music Dataset Analyzer for Kaggle Competition
    Comprehensive toolkit for exploring audio features and musical characteristics
    """
    
    def __init__(self, file_path="/kaggle/input/playground-series-s5e9/train.csv"):
        """Initialize with the Kaggle dataset"""
        print("ğŸ�µ Loading Kaggle Playground Series S5E9 Dataset...")
        
        try:
            self.df = pd.read_csv(file_path)
            print(f"âœ… Successfully loaded dataset!")
        except Exception as e:
            print(f"â�Œ Error loading dataset: {e}")
            print("ğŸ“� Creating sample data for demonstration...")
            self.df = self._create_sample_data()
        
        # Automatically detect numeric features
        self.numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in self.numeric_features:
            self.numeric_features.remove('id')
        
        # Identify likely target column(s)
        self.target_cols = []
        potential_targets = ['target', 'label', 'class', 'rating', 'score']
        for col in self.df.columns:
            if any(target in col.lower() for target in potential_targets):
                self.target_cols.append(col)
        
        print(f"ğŸ�µ Dataset loaded: {len(self.df)} tracks, {len(self.df.columns)} features")
        print(f"ğŸ“Š Numeric features: {len(self.numeric_features)}")
        print(f"ğŸ�¯ Target columns detected: {self.target_cols}")
        print(f"ğŸ“‹ All columns: {list(self.df.columns)}")
    
    def _create_sample_data(self, n_samples=1000):
        """Create sample data if file loading fails"""
        np.random.seed(42)
        
        data = {
            'id': range(n_samples),
            'RhythmScore': np.random.beta(2, 2, n_samples) * 0.8 + 0.1,
            'AudioLoudness': -np.random.exponential(8, n_samples) - 1,
            'VocalContent': np.random.beta(1, 5, n_samples) * 0.2,
            'AcousticQuality': np.random.beta(1.5, 3, n_samples) * 0.6,
            'InstrumentalScore': np.random.beta(1, 3, n_samples) * 0.5,
            'LivePerformanceLikelihood': np.random.beta(1, 4, n_samples) * 0.4,
            'MoodScore': np.random.beta(2, 2, n_samples) * 0.8 + 0.1,
            'TrackDurationMs': np.random.normal(200000, 50000, n_samples),
            'Energy': np.random.beta(2, 2, n_samples) * 0.9 + 0.05,
            'BeatsPerMinute': np.random.normal(120, 30, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def comprehensive_eda(self):
        """ğŸ”� Comprehensive Exploratory Data Analysis"""
        print("\n" + "="*80)
        print("ğŸ”� COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Dataset Overview
        print(f"ğŸ“Š Dataset Shape: {self.df.shape}")
        print(f"ğŸ�µ Total Records: {len(self.df):,}")
        print(f"ğŸ“ˆ Features: {len(self.df.columns)}")
        print(f"ğŸ”¢ Numeric Features: {len(self.numeric_features)}")
        
        # Data types and missing values
        print("\nğŸ“‹ DATA TYPES & MISSING VALUES:")
        info_df = pd.DataFrame({
            'Data Type': self.df.dtypes,
            'Missing Values': self.df.isnull().sum(),
            'Missing %': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'Unique Values': self.df.nunique()
        })
        print(info_df)
        
        # Statistical Summary for numeric features
        if self.numeric_features:
            print(f"\nğŸ“Š STATISTICAL SUMMARY (Top {min(10, len(self.numeric_features))} numeric features):")
            desc_stats = self.df[self.numeric_features[:10]].describe()
            print(desc_stats.round(4))
            
            # Advanced statistics
            print("\nğŸ�¯ ADVANCED STATISTICS:")
            for col in self.numeric_features[:8]:
                skewness = stats.skew(self.df[col].dropna())
                kurtosis = stats.kurtosis(self.df[col].dropna())
                print(f"{col:25} | Skew: {skewness:7.3f} | Kurt: {kurtosis:7.3f}")
        
        return info_df
    
    def create_mega_dashboard(self):
        """ğŸ�¨ Create the ultimate visualization dashboard"""
        # Determine layout based on available features
        n_numeric = len(self.numeric_features)
        
        if n_numeric < 4:
            print("âš ï¸� Limited numeric features available. Creating simplified dashboard.")
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        else:
            fig = plt.figure(figsize=(24, 18))
        
        fig.suptitle('ğŸ�µ ULTIMATE MUSIC DATASET ANALYSIS DASHBOARD ğŸ�µ', 
                     fontsize=26, fontweight='bold', y=0.98)
        
        # 1. Distribution Analysis
        if n_numeric >= 4:
            for i, feature in enumerate(self.numeric_features[:4]):
                plt.subplot(4, 5, i+1)
                sns.histplot(self.df[feature], bins=30, kde=True, alpha=0.7)
                plt.title(f'ğŸ“Š {feature} Distribution', fontweight='bold', fontsize=10)
                plt.xlabel(feature, fontsize=9)
        else:
            for i, feature in enumerate(self.numeric_features[:2]):
                sns.histplot(self.df[feature], bins=30, kde=True, alpha=0.7, ax=axes[i])
                axes[i].set_title(f'ğŸ“Š {feature} Distribution', fontweight='bold')
        
        if n_numeric >= 4:
            # 2. Correlation Heatmap
            plt.subplot(4, 5, 5)
            if len(self.numeric_features) > 1:
                corr_features = self.numeric_features[:min(8, len(self.numeric_features))]
                corr_matrix = self.df[corr_features].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                            square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
                plt.title('ğŸ”¥ Feature Correlations', fontweight='bold', fontsize=10)
            
            # 3. Box plots for feature comparison
            plt.subplot(4, 5, 6)
            features_to_box = self.numeric_features[:min(6, len(self.numeric_features))]
            df_normalized = self.df[features_to_box].copy()
            for col in features_to_box:
                df_normalized[col] = (df_normalized[col] - df_normalized[col].mean()) / df_normalized[col].std()
            
            sns.boxplot(data=df_normalized, palette='Set2')
            plt.title('ğŸ“¦ Normalized Feature Distributions', fontweight='bold', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
        
        if n_numeric >= 2:
            # 4. Scatter plots for relationships
            if n_numeric >= 4:
                plt.subplot(4, 5, 7)
                plt.scatter(self.df[self.numeric_features[0]], self.df[self.numeric_features[1]], 
                            alpha=0.6, s=20, c=self.df[self.numeric_features[2]] if n_numeric > 2 else 'blue')
                if n_numeric > 2:
                    plt.colorbar(label=self.numeric_features[2])
                plt.xlabel(self.numeric_features[0])
                plt.ylabel(self.numeric_features[1])
                plt.title(f'âš¡ {self.numeric_features[0]} vs {self.numeric_features[1]}', 
                          fontweight='bold', fontsize=10)
            else:
                axes[2].scatter(self.df[self.numeric_features[0]], self.df[self.numeric_features[1]], 
                                alpha=0.6, s=30)
                axes[2].set_xlabel(self.numeric_features[0])
                axes[2].set_ylabel(self.numeric_features[1])
                axes[2].set_title(f'âš¡ {self.numeric_features[0]} vs {self.numeric_features[1]}', fontweight='bold')
        
        # More advanced visualizations if we have enough features
        if n_numeric >= 4:
            # 5. Feature importance analysis
            plt.subplot(4, 5, 8)
            self._plot_feature_importance_analysis()
            
            # 6-10. Individual feature analysis
            for i in range(5):
                if i < len(self.numeric_features):
                    plt.subplot(4, 5, 9+i)
                    feature = self.numeric_features[i]
                    sample_data = self.df[feature].iloc[:min(100, len(self.df))]
                    plt.plot(sample_data.index, sample_data, marker='o', 
                             markersize=3, linewidth=1, alpha=0.8)
                    plt.title(f'ğŸ“ˆ {feature} Pattern', fontweight='bold', fontsize=10)
                    plt.xlabel('Index', fontsize=8)
                    plt.ylabel(feature, fontsize=8)
                    plt.grid(True, alpha=0.3)
            
            # 11-15. Distribution comparisons
            for i in range(5):
                if i < len(self.numeric_features):
                    plt.subplot(4, 5, 14+i)
                    feature = self.numeric_features[i]
                    
                    # Create categories based on quartiles
                    quartiles = self.df[feature].quantile([0.25, 0.75])
                    categories = ['Low', 'Medium', 'High']
                    
                    low_data = self.df[self.df[feature] <= quartiles[0.25]][feature]
                    high_data = self.df[self.df[feature] >= quartiles[0.75]][feature]
                    
                    plt.hist([low_data, high_data], bins=15, alpha=0.7, 
                             label=['Low', 'High'], color=['lightblue', 'salmon'])
                    plt.title(f'ğŸ“Š {feature} Categories', fontweight='bold', fontsize=10)
                    plt.legend(fontsize=8)
                    plt.xlabel(feature, fontsize=8)
                    
            # 16-20. Advanced analysis
            remaining_plots = min(5, len(self.numeric_features) - 15)
            for i in range(remaining_plots):
                if 15 + i < len(self.numeric_features):
                    plt.subplot(4, 5, 19+i)
                    feature = self.numeric_features[15+i]
                    
                    # Density plot
                    data = self.df[feature].dropna()
                    sns.kdeplot(data, fill=True, alpha=0.7)
                    plt.title(f'ğŸŒŠ {feature} Density', fontweight='bold', fontsize=10)
                    plt.xlabel(feature, fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _plot_feature_importance_analysis(self):
        """Advanced feature importance analysis"""
        if len(self.numeric_features) < 2:
            plt.text(0.5, 0.5, 'Need more features\nfor importance analysis', 
                     ha='center', va='center', fontsize=10)
            plt.title('ğŸ�¯ Feature Importance', fontweight='bold', fontsize=10)
            return
        
        try:
            # Use the first numeric feature as target for importance analysis
            target_feature = self.numeric_features[0]
            feature_cols = self.numeric_features[1:min(8, len(self.numeric_features))]
            
            X = self.df[feature_cols].fillna(self.df[feature_cols].mean())
            y = self.df[target_feature].fillna(self.df[target_feature].mean())
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:6]  # Top 6 features
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(indices)))
            bars = plt.bar(range(len(indices)), importances[indices], color=colors, alpha=0.8)
            
            plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], 
                       rotation=45, fontsize=8)
            plt.title(f'ğŸ�¯ Feature Importance\n(Predicting {target_feature})', 
                      fontweight='bold', fontsize=10)
            plt.ylabel('Importance', fontsize=8)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{importances[indices[i]]:.3f}', 
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
                         
        except Exception as e:
            plt.text(0.5, 0.5, f'Feature Analysis\n{str(e)[:20]}...', 
                     ha='center', va='center', fontsize=10)
            plt.title('ğŸ�¯ Feature Importance', fontweight='bold', fontsize=10)
    
    def advanced_correlation_analysis(self):
        """ğŸ”¥ Deep correlation analysis with multiple visualizations"""
        if len(self.numeric_features) < 2:
            print("âš ï¸� Need at least 2 numeric features for correlation analysis")
            return None
            
        print("\n" + "="*80)
        print("ğŸ”¥ ADVANCED CORRELATION ANALYSIS")
        print("="*80)
        
        # Select features for correlation analysis
        corr_features = self.numeric_features[:min(12, len(self.numeric_features))]
        corr_matrix = self.df[corr_features].corr()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('ğŸ”¥ COMPREHENSIVE CORRELATION ANALYSIS ğŸ”¥', 
                     fontsize=20, fontweight='bold')
        
        # 1. Full correlation heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                    square=True, fmt='.2f', ax=axes[0,0],
                    cbar_kws={'label': 'Correlation Coefficient'})
        axes[0,0].set_title('ğŸŒ¡ï¸� Complete Correlation Matrix', fontweight='bold')
        
        # 2. Strong correlations only (|r| > 0.3)
        strong_corr = corr_matrix.copy()
        strong_corr[np.abs(strong_corr) < 0.3] = 0
        sns.heatmap(strong_corr, annot=True, cmap='viridis', center=0, 
                    square=True, fmt='.2f', ax=axes[0,1],
                    cbar_kws={'label': 'Strong Correlations'})
        axes[0,1].set_title('âš¡ Strong Correlations (|r| â‰¥ 0.3)', fontweight='bold')
        
        # 3. Correlation distribution
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)]
        axes[0,2].hist(corr_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,2].set_title('ğŸ“Š Correlation Distribution', fontweight='bold')
        axes[0,2].set_xlabel('Correlation Coefficient')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Find and display top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # 4-6. Plot top 3 correlations
        for idx, (feat1, feat2, corr_val) in enumerate(corr_pairs[:3]):
            ax = axes[1, idx]
            
            # Create scatter plot with regression line
            x_data = self.df[feat1].dropna()
            y_data = self.df[feat2].dropna()
            
            # Ensure same length
            min_len = min(len(x_data), len(y_data))
            x_data = x_data.iloc[:min_len]
            y_data = y_data.iloc[:min_len]
            
            ax.scatter(x_data, y_data, alpha=0.6, s=20, color='blue')
            
            # Add regression line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.set_title(f'{feat1} vs {feat2}\nr = {corr_val:.3f}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print correlation insights
        print("\nğŸ”¥ TOP CORRELATION INSIGHTS:")
        print("="*60)
        for i, (feat1, feat2, corr_val) in enumerate(corr_pairs[:10]):
            strength = ("Very Strong" if abs(corr_val) > 0.7 else 
                        "Strong" if abs(corr_val) > 0.5 else 
                        "Moderate" if abs(corr_val) > 0.3 else "Weak")
            direction = "Positive" if corr_val > 0 else "Negative"
            print(f"{i+1:2d}. {feat1:20} â†” {feat2:20} | {corr_val:6.3f} | {strength} {direction}")
        
        return corr_matrix, corr_pairs
    
    def clustering_analysis(self):
        """ğŸ�¯ Advanced clustering analysis"""
        if len(self.numeric_features) < 2:
            print("âš ï¸� Need at least 2 numeric features for clustering")
            return None
            
        print("\n" + "="*80)
        print("ğŸ�¯ ADVANCED CLUSTERING ANALYSIS")
        print("="*80)
        
        # Prepare data
        cluster_features = self.numeric_features[:min(8, len(self.numeric_features))]
        X = self.df[cluster_features].fillna(self.df[cluster_features].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('ğŸ�¯ COMPREHENSIVE CLUSTERING ANALYSIS ğŸ�¯', 
                     fontsize=18, fontweight='bold')
        
        # 1. Elbow method
        inertias = []
        k_range = range(2, min(11, len(X)//10 + 2))  # Ensure reasonable k range
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        axes[0,0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0,0].set_xlabel('Number of Clusters (k)')
        axes[0,0].set_ylabel('Inertia')
        axes[0,0].set_title('ğŸ“Š Elbow Method for Optimal k', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Find elbow point (simple method)
        if len(inertias) > 2:
            differences = np.diff(inertias)
            elbow_k = k_range[np.argmax(differences[:-1] - differences[1:]) + 1]
        else:
            elbow_k = k_range[0] if k_range else 3
            
        axes[0,0].axvline(x=elbow_k, color='red', linestyle='--', alpha=0.7, 
                          label=f'Suggested k={elbow_k}')
        axes[0,0].legend()
        
        # 2. K-means clustering with optimal k
        optimal_k = min(elbow_k, 5)  # Cap at 5 for visualization
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                                    cmap='viridis', alpha=0.7, s=30)
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0,1].set_title(f'ğŸ�¨ K-Means Clusters (k={optimal_k})', fontweight='bold')
        plt.colorbar(scatter, ax=axes[0,1], label='Cluster')
        
        # 3. Cluster sizes
        unique, counts = np.unique(clusters, return_counts=True)
        axes[0,2].bar(unique, counts, color='lightblue', alpha=0.8, edgecolor='black')
        axes[0,2].set_xlabel('Cluster')
        axes[0,2].set_ylabel('Number of Samples')
        axes[0,2].set_title('ğŸ“Š Cluster Sizes', fontweight='bold')
        
        # Add percentage labels
        for i, count in enumerate(counts):
            percentage = count / len(clusters) * 100
            axes[0,2].text(i, count + max(counts)*0.01, f'{percentage:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
        
        # 4. Cluster characteristics heatmap
        cluster_df = pd.DataFrame(X, columns=cluster_features)
        cluster_df['Cluster'] = clusters
        cluster_means = cluster_df.groupby('Cluster').mean()
        
        sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', ax=axes[1,0])
        axes[1,0].set_title('ğŸ�ª Cluster Characteristics', fontweight='bold')
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Features (Standardized)')
        
        # 5. PCA explained variance
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        
        axes[1,1].plot(range(1, len(cumvar) + 1), cumvar, 'ro-', linewidth=2, markersize=6)
        axes[1,1].axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% Variance')
        axes[1,1].axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
        axes[1,1].set_xlabel('Number of Components')
        axes[1,1].set_ylabel('Cumulative Explained Variance')
        axes[1,1].set_title('ğŸ“ˆ PCA Explained Variance', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Feature contributions to first 2 PCs
        if len(cluster_features) > 2:
            feature_contrib = np.abs(pca.components_[:2]).mean(axis=0)
            sorted_idx = np.argsort(feature_contrib)[::-1]
            
            axes[1,2].barh(range(len(feature_contrib)), feature_contrib[sorted_idx], 
                           color='lightgreen', alpha=0.8)
            axes[1,2].set_yticks(range(len(feature_contrib)))
            axes[1,2].set_yticklabels([cluster_features[i] for i in sorted_idx])
            axes[1,2].set_title('ğŸ�¯ Feature Contributions to PC1&2', fontweight='bold')
            axes[1,2].set_xlabel('Average Absolute Contribution')
        else:
            axes[1,2].text(0.5, 0.5, 'Need more features\nfor contribution analysis', 
                           ha='center', va='center', fontsize=12)
            axes[1,2].set_title('ğŸ�¯ Feature Contributions', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print cluster analysis results
        print(f"\nğŸ�ª CLUSTERING RESULTS:")
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Explained variance (PC1+PC2): {sum(pca.explained_variance_ratio_[:2]):.1%}")
        print(f"Total explained variance needed for 95%: {np.argmax(cumvar >= 0.95) + 1} components")
        
        # Cluster descriptions
        for i in range(optimal_k):
            cluster_size = np.sum(clusters == i)
            print(f"\nğŸ“Š Cluster {i}: {cluster_size:,} samples ({cluster_size/len(clusters)*100:.1f}%)")
            
            # Find most distinctive characteristics
            cluster_mean = cluster_means.loc[i]
            overall_mean = cluster_df[cluster_features].mean()
            
            # Calculate z-scores for cluster characteristics
            differences = (cluster_mean - overall_mean) / cluster_df[cluster_features].std()
            significant_features = differences[np.abs(differences) > 0.5].sort_values(key=abs, ascending=False)
            
            print("    Key characteristics (>0.5 std dev from overall mean):")
            for feature, diff in significant_features.head(3).items():
                direction = "higher" if diff > 0 else "lower"
                print(f"    - {feature}: {abs(diff):.1f}Ïƒ {direction}")
        
        return clusters, cluster_means
    
    def predictive_modeling(self):
        """ğŸ¤– Advanced predictive modeling"""
        if len(self.numeric_features) < 2:
            print("âš ï¸� Need at least 2 numeric features for predictive modeling")
            return None
            
        print("\n" + "="*80)
        print("ğŸ¤– ADVANCED PREDICTIVE MODELING")
        print("="*80)
        
        # Select target and features
        target_candidates = self.numeric_features[:4]  # Try first 4 as potential targets
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ğŸ¤– PREDICTIVE MODELING RESULTS ğŸ¤–', fontsize=18, fontweight='bold')
        
        results = {}
        
        for idx, target in enumerate(target_candidates):
            if idx >= 4:  # Limit to 4 models for visualization
                break
                
            # Prepare features (exclude target)
            features = [col for col in self.numeric_features if col != target][:8]  # Limit features
            
            if not features:
                continue
                
            X = self.df[features].fillna(self.df[features].mean())
            y = self.df[target].fillna(self.df[target].mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Model Training
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
            rf.fit(X_train, y_train)
            
            # Predictions and Evaluation
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            results[target] = {'r2': r2, 'mse': mse}
            
            # Visualization
            ax = axes[idx // 2, idx % 2]
            ax.scatter(y_test, y_pred, alpha=0.5, s=25, edgecolors='k', c='cyan')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'ğŸ�¯ Predicting: {target}\n$R^2 = {r2:.3f}$ | MSE = {mse:.3f}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Feature Importance inset plot
            inset_ax = ax.inset_axes([0.05, 0.6, 0.35, 0.35])
            importances = rf.feature_importances_
            indices = np.argsort(importances)[-5:] # Top 5
            inset_ax.barh(range(len(indices)), importances[indices], color='lightgreen', align='center')
            inset_ax.set_yticks(range(len(indices)))
            inset_ax.set_yticklabels([features[i] for i in indices], fontsize=8)
            inset_ax.set_xlabel('Importance', fontsize=8)
            inset_ax.set_title('Top Features', fontsize=10, fontweight='bold')
            inset_ax.tick_params(axis='x', labelsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        # Print summary
        print("\nğŸ¤– PREDICTIVE MODELING SUMMARY:")
        print("="*60)
        print(f"{'Target Feature':<25} | {'R-squared (RÂ²)':<20} | {'Mean Squared Error (MSE)':<20}")
        print("-"*60)
        for target, metrics in results.items():
            print(f"{target:<25} | {metrics['r2']:<20.4f} | {metrics['mse']:<20.4f}")
            
        return results

# Main execution block to demonstrate the class
if __name__ == '__main__':
    # Initialize the analyzer
    # If you are not in a Kaggle environment, change the file_path to your local CSV file
    # or leave it empty to use the sample data generator.
    # IMPORTANT: Update the file path to your dataset location.
    try:
        analyzer = MusicDataAnalyzer(file_path="train.csv") # <--- CHANGE THIS PATH if needed
    except FileNotFoundError:
        print("\n--- train.csv not found. Running with sample data. ---\n")
        analyzer = MusicDataAnalyzer(file_path=None)

    # 1. Run Comprehensive Exploratory Data Analysis
    analyzer.comprehensive_eda()
    
    # 2. Generate the Ultimate Visualization Dashboard
    # Note: This might generate a large plot with many subplots
    analyzer.create_mega_dashboard()
    
    # 3. Perform Advanced Correlation Analysis
    analyzer.advanced_correlation_analysis()
    
    # 4. Conduct Clustering Analysis to find music segments
    analyzer.clustering_analysis()
    
    # 5. Run Predictive Modeling
    analyzer.predictive_modeling()
    
    print("\nâœ… Full music data analysis complete!")



print(f"{train.info()}, {test.info()}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for the plots for better aesthetics
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-deep")

# --- 1. Load the data ---
try:
    train_df = train
    print("Data loaded successfully. Here are the first 5 rows:\n")
    print(train_df.head())
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please make sure the file is in the same directory.")
    exit()

# --- 2. Analyze the Target Variable: BeatsPerMinute ---
print("\n--- Analyzing the distribution of BeatsPerMinute ---")
plt.figure(figsize=(10, 6))
sns.histplot(train_df['BeatsPerMinute'], kde=True, bins=50)
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')
plt.show()

print("\nObservation: The target variable is not a perfect bell curve; it has a long tail, which explains the need for a transformation like the one in your notebook.")

# --- 3. Feature Correlation Analysis ---
print("\n--- Visualizing Feature Correlations ---")
# Select only the numerical features
numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = train_df[numerical_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of All Numerical Features')
plt.show()

print("\nObservation: This heatmap is a goldmine. Look for features with high correlation to 'BeatsPerMinute'. 'RhythmScore' and 'Energy' seem to be the most correlated.")

# --- 4. Relationship between key features and target ---
print("\n--- Plotting relationships with the target variable ---")
# Scatter plot for the most correlated feature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RhythmScore', y='BeatsPerMinute', data=train_df, alpha=0.6)
plt.title('RhythmScore vs. BeatsPerMinute')
plt.xlabel('RhythmScore')
plt.ylabel('BeatsPerMinute')
plt.show()

# Scatter plot for another key feature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Energy', y='BeatsPerMinute', data=train_df, alpha=0.6)
plt.title('Energy vs. BeatsPerMinute')
plt.xlabel('Energy')
plt.ylabel('BeatsPerMinute')
plt.show()

print("\nObservation: The scatter plots show non-linear relationships, which is a key reason why tree-based models (like LightGBM) are a great choice for this dataset.")

# --- 5. Outlier Detection with Box Plots ---
print("\n--- Checking for outliers using box plots ---")
features_to_check = ['AudioLoudness', 'TrackDurationMs', 'AcousticQuality']
plt.figure(figsize=(15, 5))

for i, feature in enumerate(features_to_check, 1):
    plt.subplot(1, len(features_to_check), i)
    sns.boxplot(y=train_df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

print("\nObservation: Box plots reveal outliers in some features, especially 'TrackDurationMs'. This is another reason your notebook included data transformation and robust models.")



import matplotlib.pyplot as plt

train.hist(bins=30, figsize=(15, 10))
plt.suptitle("Train Data Distributions", fontsize=16)
plt.show()



Train = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')


import matplotlib.pyplot as plt

Train.hist(bins=30,figsize=(15, 10),color= 'blue')
plt.suptitle("Train() Data Distributions", fontsize=16)
plt.show()



import matplotlib.pyplot as plt

test.hist(bins=30,figsize=(15, 10),color= 'red')
plt.suptitle("Test Data Distributions", fontsize=16)
plt.show()



import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(train.corr(), annot=False, cmap='viridis')
plt.title("Correlation Heatmap (Train)", fontsize=16)
plt.show()



print("Train ID range:", train['id'].min(), "to", train['id'].max())
print("Test ID range:", test['id'].min(), "to", test['id'].max())

print("\nTrain Summary:\n", train.describe().T[['mean','std','min','max']])
print("\nTest Summary:\n", test.describe().T[['mean','std','min','max']])



X = train.drop(columns=['id', 'BeatsPerMinute'])
y = train['BeatsPerMinute']



import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(y, bins=30, kde=True,color = 'red')
plt.title("Target Distribution: BeatsPerMinute")
plt.show()



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)



train


X_train_scaled


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# lr = LinearRegression()
# lr.fit(X_train_scaled, y_train)
# y_pred = lr.predict(X_val_scaled)

# print("Linear Regression RÂ²:", r2_score(y_val, y_pred))
# print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))



# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForest+Regressor(n_estimators=200, random_state=42, n_jobs=-1)
# rf.fit(X_train, y_train)   # trees donâ€™t need scaling
# y_pred_rf = rf.predict(X_val)

# print("Random Forest RÂ²:", r2_score(y_val, y_pred_rf))
# print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_rf)))



import seaborn as sns
import matplotlib.pyplot as plt

corr = train.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr[['BeatsPerMinute']].sort_values(by='BeatsPerMinute', ascending=False), 
            annot=True, cmap="coolwarm")
plt.title("Correlation with BPM")
plt.show()



# from xgboost import XGBRegressor

# xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
# xgb.fit(X_train, y_train)
# y_pred_xgb = xgb.predict(X_val)

# print("XGB RÂ²:", r2_score(y_val, y_pred_xgb))
# print("XGB RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_xgb)))




import seaborn as sns
import matplotlib.pyplot as plt

corr = train.corr()['BeatsPerMinute'].sort_values(ascending=False)
print(corr)

plt.figure(figsize=(8,4))
sns.barplot(x=corr.index, y=corr.values)
plt.xticks(rotation=45)
plt.title("Feature Correlation with BPM")
plt.show()



# Import the LightGBM model
import lightgbm as lgb


# # 1. Define the features (X) and target (y) using the FULL training dataset
# features = ['RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']
# X_full_train = train[features]
# y_full_train = train['BeatsPerMinute']

# # 2. Prepare the final test data from test.csv
# X_final_test = test[features]

# # 3. Initialize and train the LightGBM Regressor model
# # These parameters are a good starting point. They can be tuned for better performance.
# lgbm = lgb.LGBMRegressor(
#     n_estimators=1000,      # Number of boosting rounds (trees)
#     learning_rate=0.05,     # Controls the step size at each iteration
#     num_leaves=31,          # Number of leaves in one tree
#     random_state=42,        # Seed for reproducibility
#     n_jobs=-1               # Use all available CPU cores
# )

# lgbm.fit(X_full_train, y_full_train)

# # 4. Make predictions on the final test data
# final_predictions_lgbm = lgbm.predict(X_final_test)

# # 5. Create the new submission DataFrame
# submission_lgbm = pd.DataFrame({
#     'id': test['id'],
#     'BeatsPerMinute': final_predictions_lgbm
# })

# # 6. Save the DataFrame to a new CSV file
# submission_lgbm.to_csv('submission_lgbm.csv', index=False)

# print("LGBM submission file created successfully!")
# print("Here's a preview of the new submission file:")
# print(submission_lgbm.head())


#not a good score by that 
from sklearn import metrics


# from sklearn.model_selection import KFold
# import numpy as np

# # Define the features and target again
# features = ['RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']
# X_full_train = train[features].values # Using .values to get numpy array
# y_full_train = train['BeatsPerMinute'].values
# X_final_test = test[features].values

# # Set up the K-Fold cross-validation
# n_splits = 7
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Initialize arrays to store predictions
# oof_predictions = np.zeros(X_full_train.shape[0])
# test_predictions = np.zeros(X_final_test.shape[0])

# # Loop through each fold
# for fold, (train_index, val_index) in enumerate(kf.split(X_full_train, y_full_train)):
#     print(f"===== Fold {fold+1} =====")
    
#     # Split the data for this fold
#     X_train, X_val = X_full_train[train_index], X_full_train[val_index]
#     y_train, y_val = y_full_train[train_index], y_full_train[val_index]
    
#     # Initialize and train the LightGBM model
#     lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)
#     lgbm.fit(X_train, y_train, 
#              eval_set=[(X_val, y_val)],
#              eval_metric='rmse',
#              callbacks=[lgb.early_stopping(100, verbose=False)]) # Stop if validation score doesn't improve for 100 rounds
    
#     # Make predictions on the validation data and the final test data
#     oof_predictions[val_index] = lgbm.predict(X_val)
#     test_predictions += lgbm.predict(X_final_test) / n_splits # Add this fold's prediction, averaged

# # Calculate overall validation score
# oof_rmse = np.sqrt(metrics.mean_squared_error(y_full_train, oof_predictions))
# print(f"\nOverall Out-of-Fold RMSE: {oof_rmse}")

# # Create the submission file from the averaged test predictions
# submission_df_cv = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': test_predictions})
# submission_df_cv.to_csv('submission_cv_7fold.csv', index=False)

# print("\nCross-validated submission file created successfully!")
# print(submission_df_cv.head())


# from sklearn.model_selection import KFold
# import numpy as np

# # Define the features and target again
# features = ['RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']
# X_full_train = train[features].values # Using .values to get numpy array
# y_full_train = train['BeatsPerMinute'].values
# X_final_test = test[features].values

# # Set up the K-Fold cross-validation
# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Initialize arrays to store predictions
# oof_predictions = np.zeros(X_full_train.shape[0])
# test_predictions = np.zeros(X_final_test.shape[0])

# # Loop through each fold
# for fold, (train_index, val_index) in enumerate(kf.split(X_full_train, y_full_train)):
#     print(f"===== Fold {fold+1} =====")
    
#     # Split the data for this fold
#     X_train, X_val = X_full_train[train_index], X_full_train[val_index]
#     y_train, y_val = y_full_train[train_index], y_full_train[val_index]
    
#     # Initialize and train the LightGBM model
#     lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)
#     lgbm.fit(X_train, y_train, 
#              eval_set=[(X_val, y_val)],
#              eval_metric='rmse',
#              callbacks=[lgb.early_stopping(100, verbose=False)]) # Stop if validation score doesn't improve for 100 rounds
    
#     # Make predictions on the validation data and the final test data
#     oof_predictions[val_index] = lgbm.predict(X_val)
#     test_predictions += lgbm.predict(X_final_test) / n_splits # Add this fold's prediction, averaged

# # Calculate overall validation score
# oof_rmse = np.sqrt(metrics.mean_squared_error(y_full_train, oof_predictions))
# print(f"\nOverall Out-of-Fold RMSE: {oof_rmse}")

# # Create the submission file from the averaged test predictions
# submission_df_cv = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': test_predictions})
# submission_df_cv.to_csv('submission_cv_5fold.csv', index=False)

# print("\nCross-validated submission file created successfully!")
# print(submission_df_cv.head())


from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import lightgbm as lgb

## --- 1. Feature Engineering ---
# We'll work with copies so the original dataframes remain unchanged
train_fe = train.copy()
test_fe = test.copy()

# Create new interaction features
train_fe['mood_energy_interaction'] = train_fe['MoodScore'] * train_fe['Energy']
test_fe['mood_energy_interaction'] = test_fe['MoodScore'] * test_fe['Energy']

train_fe['loudness_vocal_interaction'] = train_fe['AudioLoudness'] * train_fe['VocalContent']
test_fe['loudness_vocal_interaction'] = test_fe['AudioLoudness'] * test_fe['VocalContent']

# Create a new ratio feature (add a small number to prevent division by zero)
train_fe['acoustic_instrumental_ratio'] = train_fe['AcousticQuality'] / (train_fe['InstrumentalScore'] + 1e-6)
test_fe['acoustic_instrumental_ratio'] = test_fe['AcousticQuality'] / (test_fe['InstrumentalScore'] + 1e-6)

print("New features have been created.\n")


## --- 2. Cross-Validation with New Features ---
# Update the list of features to include our new creations
original_features = ['RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']
new_features = ['mood_energy_interaction', 'loudness_vocal_interaction', 'acoustic_instrumental_ratio']
features_fe = original_features + new_features

# Use the new dataframes (train_fe, test_fe)
X_full_train = train_fe[features_fe].values
y_full_train = train_fe['BeatsPerMinute'].values
X_final_test = test_fe[features_fe].values

# Set up K-Fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize arrays for predictions
oof_predictions = np.zeros(X_full_train.shape[0])
test_predictions = np.zeros(X_final_test.shape[0])

# Loop through each fold
for fold, (train_index, val_index) in enumerate(kf.split(X_full_train, y_full_train)):
    print(f"===== Fold {fold+1} =====")
    
    X_train, X_val = X_full_train[train_index], X_full_train[val_index]
    y_train, y_val = y_full_train[train_index], y_full_train[val_index]
    
    lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)
    lgbm.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)],
             eval_metric='rmse',
             callbacks=[lgb.early_stopping(100, verbose=False)])
    
    oof_predictions[val_index] = lgbm.predict(X_val)
    test_predictions += lgbm.predict(X_final_test) / n_splits

# Calculate and print the new overall validation score
oof_rmse_fe = np.sqrt(metrics.mean_squared_error(y_full_train, oof_predictions))
print(f"\nOverall Out-of-Fold RMSE with new features: {oof_rmse_fe}")

# Create the new submission file
submission_df_fe = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': test_predictions})
submission_df_fe.to_csv('submission_fe.csv', index=False)

print("\nSubmission file with new features created successfully!")
print(submission_df_fe.head())


# from sklearn.model_selection import KFold
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans # Import KMeans
# from sklearn import metrics
# import numpy as np
# import lightgbm as lgb

# # --- 1. Feature Engineering (keeping the ones we made) ---
# train_fe = train.copy()
# test_fe = test.copy()

# # Interaction and Ratio features
# train_fe['mood_energy_interaction'] = train_fe['MoodScore'] * train_fe['Energy']
# test_fe['mood_energy_interaction'] = test_fe['MoodScore'] * test_fe['Energy']
# train_fe['loudness_vocal_interaction'] = train_fe['AudioLoudness'] * train_fe['VocalContent']
# test_fe['loudness_vocal_interaction'] = test_fe['AudioLoudness'] * test_fe['VocalContent']
# train_fe['acoustic_instrumental_ratio'] = train_fe['AcousticQuality'] / (train_fe['InstrumentalScore'] + 1e-6)
# test_fe['acoustic_instrumental_ratio'] = test_fe['AcousticQuality'] / (test_fe['InstrumentalScore'] + 1e-6)

# features_fe = [
#     'RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality',
#     'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
#     'TrackDurationMs','Energy', 'mood_energy_interaction', 
#     'loudness_vocal_interaction', 'acoustic_instrumental_ratio'
# ]

# # --- 2. Scaling ---
# scaler = MinMaxScaler()
# # We scale the combined data to create consistent clusters
# combined_data = pd.concat([train_fe[features_fe], test_fe[features_fe]], axis=0)
# scaled_combined_data = scaler.fit_transform(combined_data)

# # --- 3. K-Means Clustering Feature ---
# n_clusters = 5 # This is a parameter you can experiment with!
# kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
# # Create cluster labels for the whole dataset
# cluster_labels = kmeans.fit_predict(scaled_combined_data)

# # Add the new cluster feature back to the dataframes
# train_fe['cluster'] = cluster_labels[:len(train_fe)]
# test_fe['cluster'] = cluster_labels[len(train_fe):]

# print(f"Created {n_clusters} clusters. Added 'cluster' feature.\n")
# features_final = features_fe + ['cluster']


# ## --- 4. Cross-Validation with Cluster Feature ---
# X_full_train = train_fe[features_final]
# y_full_train = train_fe['BeatsPerMinute'].values
# X_final_test = test_fe[features_final]

# params = {
#     'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
#     'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
#     'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
#     'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
#     'boosting_type': 'gbdt',
# }

# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# oof_predictions = np.zeros(X_full_train.shape[0])
# test_predictions = np.zeros(X_final_test.shape[0])

# for fold, (train_index, val_index) in enumerate(kf.split(X_full_train, y_full_train)):
#     print(f"===== Fold {fold+1} =====")
#     X_train, X_val = X_full_train.iloc[train_index], X_full_train.iloc[val_index]
#     y_train, y_val = y_full_train[train_index], y_full_train[val_index]
    
#     lgbm = lgb.LGBMRegressor(**params)
#     # Tell LightGBM that 'cluster' is a categorical feature
#     lgbm.fit(X_train, y_train, 
#              eval_set=[(X_val, y_val)],
#              eval_metric='rmse',
#              callbacks=[lgb.early_stopping(100, verbose=False)],
#              categorical_feature=['cluster'])
    
#     oof_predictions[val_index] = lgbm.predict(X_val)
#     test_predictions += lgbm.predict(X_final_test) / n_splits

# oof_rmse_cluster = np.sqrt(metrics.mean_squared_error(y_full_train, oof_predictions))
# print(f"\nOverall Out-of-Fold RMSE with cluster features: {oof_rmse_cluster}")

# submission_df_cluster = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': test_predictions})
# submission_df_cluster.to_csv('submission_cluster.csv', index=False)

# print("\nSubmission file with cluster features created successfully!")
# print(submission_df_cluster.head())


import joblib


# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import joblib # Library for saving and loading models

# # --- 1. Load the Data ---
# # This assumes train_df is already loaded from your previous cells.
# # If not, you can add the pd.read_csv lines here.
# train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


# # --- 2. Feature Engineering (Using the best-performing set) ---
# train_fe = train.copy()

# # Interaction features
# train_fe['mood_energy_interaction'] = train_fe['MoodScore'] * train_fe['Energy']
# train_fe['loudness_vocal_interaction'] = train_fe['AudioLoudness'] * train_fe['VocalContent']

# # Ratio feature
# train_fe['acoustic_instrumental_ratio'] = train_fe['AcousticQuality'] / (train_fe['InstrumentalScore'] + 1e-6)

# print("New features have been created for the final model.\n")


# # --- 3. Define Final Feature Set ---
# original_features = ['RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']
# new_features = ['mood_energy_interaction', 'loudness_vocal_interaction', 'acoustic_instrumental_ratio']
# features_fe = original_features + new_features

# X_final_train = train_fe[features_fe]
# y_final_train = train_fe['BeatsPerMinute']

# # --- 4. Train the Final LightGBM Model on ALL Data ---
# # We use the same parameters from your best submission
# lgbm_final_model = lgb.LGBMRegressor(
#     n_estimators=1000, 
#     learning_rate=0.05, 
#     num_leaves=31, 
#     random_state=42, 
#     n_jobs=-1
# )

# print("Training the final model on the full dataset...")
# # Note: No validation set or early stopping here, as we want the model to learn from all data.
# lgbm_final_model.fit(X_final_train, y_final_train)
# print("Training complete.")

# # --- 5. Save the Trained Model to a File ---
# model_filename = 'best_bpm_model.joblib'
# joblib.dump(lgbm_final_model, model_filename)

# print(f"\nModel successfully saved to '{model_filename}'!")
# print("This file is now available in your output directory and is ready for fine-tuning.")


# import pandas as pd
# import lightgbm as lgb
# import joblib

# # --- For Future Use: Load and Fine-Tune ---

# # 1. Load your new dataset from its Kaggle path
# new_data_df = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")

# # 2. **IMPORTANT**: Create the EXACT SAME engineered features on the new data
# # Interaction features
# new_data_df['mood_energy_interaction'] = new_data_df['MoodScore'] * new_data_df['Energy']
# new_data_df['loudness_vocal_interaction'] = new_data_df['AudioLoudness'] * new_data_df['VocalContent']
# # Ratio feature
# new_data_df['acoustic_instrumental_ratio'] = new_data_df['AcousticQuality'] / (new_data_df['InstrumentalScore'] + 1e-6)

# # 3. Load the model you saved earlier
# # Make sure 'best_bpm_model.joblib' is in your notebook's output directory
# loaded_model = joblib.load('best_bpm_model.joblib')

# # 4. Define the features and target from your NEW data
# # This is the complete list of features the model was trained on
# features_fe = [
#     'RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality',
#     'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
#     'TrackDurationMs','Energy', 'mood_energy_interaction', 
#     'loudness_vocal_interaction', 'acoustic_instrumental_ratio'
# ]
# X_new = new_data_df[features_fe]
# y_new = new_data_df['BeatsPerMinute']

# # 5. Fine-tune the model by continuing the training on the new data
# # The 'init_model' parameter tells LightGBM to start from where the loaded model left off
# print("Fine-tuning the model on new data...")
# loaded_model.fit(X_new, y_new, init_model=loaded_model)
# print("Fine-tuning complete.")

# # 6. You can now use the fine-tuned model for predictions or save it again
# joblib.dump(loaded_model, 'finetuned_bpm_model.joblib')
# print("Fine-tuned model saved as 'finetuned_bpm_model.joblib'")


# import pandas as pd
# import lightgbm as lgb
# import joblib

# # --- 1. Load the Test Data ---
# # This assumes test_df is already loaded. If not, add the pd.read_csv line.
# test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# # --- 2. Feature Engineering on the Test Data ---
# # It is CRITICAL to perform the same feature engineering on the test set
# test_fe = test.copy()

# # Interaction features
# test_fe['mood_energy_interaction'] = test_fe['MoodScore'] * test_fe['Energy']
# test_fe['loudness_vocal_interaction'] = test_fe['AudioLoudness'] * test_fe['VocalContent']

# # Ratio feature
# test_fe['acoustic_instrumental_ratio'] = test_fe['AcousticQuality'] / (test_fe['InstrumentalScore'] + 1e-6)

# print("Engineered features created on the test data.\n")

# # --- 3. Define the Full Feature Set ---
# features_fe = [
#     'RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality',
#     'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
#     'TrackDurationMs','Energy', 'mood_energy_interaction', 
#     'loudness_vocal_interaction', 'acoustic_instrumental_ratio'
# ]
# X_final_test = test_fe[features_fe]

# # --- 4. Load the Saved Model ---
# model_filename = '/kaggle/working/best_bpm_model.joblib'
# print(f"Loading model from {model_filename}...")
# loaded_model = joblib.load(model_filename)
# print("Model loaded successfully.")

# # --- 5. Make Predictions on the Final Test Data ---
# print("Generating predictions...")
# final_predictions = loaded_model.predict(X_final_test)
# print("Predictions generated.")

# # --- 6. Create the Submission DataFrame ---
# submission_df_final = pd.DataFrame({
#     'id': test['id'],
#     'BeatsPerMinute': final_predictions
# })

# # Save the DataFrame to a CSV file
# submission_df_final.to_csv('submission_from_saved_model.csv', index=False)

# print("\nFinal submission file 'submission_from_saved_model.csv' created successfully!")
# print(submission_df_final.head())
# #we don't have RMSE We got 26.42 bad


# import pandas as pd
# import lightgbm as lgb
# import joblib

# # --- 1. Load the Test Data ---
# # This assumes test_df is already loaded. If not, add the pd.read_csv line.
# test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# # --- 2. Feature Engineering on the Test Data ---
# # It is CRITICAL to perform the same feature engineering on the test set
# test_fe = test.copy()

# # Interaction features
# test_fe['mood_energy_interaction'] = test_fe['MoodScore'] * test_fe['Energy']
# test_fe['loudness_vocal_interaction'] = test_fe['AudioLoudness'] * test_fe['VocalContent']

# # Ratio feature
# test_fe['acoustic_instrumental_ratio'] = test_fe['AcousticQuality'] / (test_fe['InstrumentalScore'] + 1e-6)

# print("Engineered features created on the test data.\n")

# # --- 3. Define the Full Feature Set ---
# features_fe = [
#     'RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality',
#     'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
#     'TrackDurationMs','Energy', 'mood_energy_interaction', 
#     'loudness_vocal_interaction', 'acoustic_instrumental_ratio'
# ]
# X_final_test = test_fe[features_fe]

# # --- 4. Load the Saved Model ---
# model_filename = '/kaggle/working/finetuned_bpm_model.joblib'
# print(f"Loading model from {model_filename}...")
# loaded_model = joblib.load(model_filename)
# print("Model loaded successfully.")

# # --- 5. Make Predictions on the Final Test Data ---
# print("Generating predictions...")
# final_predictions = loaded_model.predict(X_final_test)
# print("Predictions generated.")

# # --- 6. Create the Submission DataFrame ---
# submission_df_final = pd.DataFrame({
#     'id': test['id'],
#     'BeatsPerMinute': final_predictions
# })

# # Save the DataFrame to a CSV file
# submission_df_final.to_csv('submission_from_finetuned_model.csv', index=False)

# print("\nFinal submission file 'submission_from_finetuned_model.csv' created successfully!")
# print(submission_df_final.head())
# #we don't have RMSE We got 27.29 bad


# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import joblib

# # --- 1. Load All Datasets ---
# # Load the original competition data
# original_train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
# original_test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# # Load the new, external training data
# new_train_df = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")

# print(f"Original training data shape: {original_train_df.shape}")
# print(f"New training data shape: {new_train_df.shape}")


# # --- 2. Combine the Training Data ---
# # We'll concatenate the two training dataframes.
# # ignore_index=True resets the index for the new combined dataframe.
# combined_train_df = pd.concat([original_train_df, new_train_df], ignore_index=True)
# print(f"Combined training data shape: {combined_train_df.shape}\n")


# # --- 3. Feature Engineering on ALL Data ---
# # Apply the same feature creation to both the combined training set and the test set
# def create_features(df):
#     df_copy = df.copy()
#     # Interaction features
#     df_copy['mood_energy_interaction'] = df_copy['MoodScore'] * df_copy['Energy']
#     df_copy['loudness_vocal_interaction'] = df_copy['AudioLoudness'] * df_copy['VocalContent']
#     # Ratio feature
#     df_copy['acoustic_instrumental_ratio'] = df_copy['AcousticQuality'] / (df_copy['InstrumentalScore'] + 1e-6)
#     return df_copy

# train_fe = create_features(combined_train_df)
# test_fe = create_features(original_test_df)

# print("Engineered features created on combined training and test data.\n")


# # --- 4. Train the Final Model on the Merged Dataset ---
# # Define the full feature set
# features_fe = [
#     'RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality',
#     'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
#     'TrackDurationMs','Energy', 'mood_energy_interaction', 
#     'loudness_vocal_interaction', 'acoustic_instrumental_ratio'
# ]

# X_train_final = train_fe[features_fe]
# y_train_final = train_fe['BeatsPerMinute']
# X_test_final = test_fe[features_fe]

# # Use the parameters from our best-performing model
# lgbm_final_model = lgb.LGBMRegressor(
#     n_estimators=1000, 
#     learning_rate=0.05, 
#     num_leaves=31, 
#     random_state=42, 
#     n_jobs=-1
# )

# print("Training the final model on the full MERGED dataset...")
# lgbm_final_model.fit(X_train_final, y_train_final)
# print("Training complete.")


# # --- 5. Make Predictions and Save Submission File ---
# print("Generating final predictions...")
# final_predictions = lgbm_final_model.predict(X_test_final)

# submission_df_merged = pd.DataFrame({
#     'id': original_test_df['id'],
#     'BeatsPerMinute': final_predictions
# })

# submission_df_merged.to_csv('submission_merged.csv', index=False)
# print("\nSubmission file 'submission_merged.csv' from merged data created successfully!")
# print(submission_df_merged.head())

# # (Optional) Save this powerful new model for future use
# joblib.dump(lgbm_final_model, 'lgbm_merged_data_model.joblib')
# print("\nFinal model saved to 'lgbm_merged_data_model.joblib'")
# #score 27.19


# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import optuna
# import joblib

# # --- 1. Load All Datasets ---
# original_train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
# original_test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
# new_train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")

# # --- 2. Combine the Training Data ---
# combined_train_df = original_train_df

# # --- 3. Feature Engineering ---
# def create_features(df):
#     df_copy = df.copy()
#     df_copy['mood_energy_interaction'] = df_copy['MoodScore'] * df_copy['Energy']
#     df_copy['loudness_vocal_interaction'] = df_copy['AudioLoudness'] * df_copy['VocalContent']
#     df_copy['acoustic_instrumental_ratio'] = df_copy['AcousticQuality'] / (df_copy['InstrumentalScore'] + 1e-6)
#     return df_copy

# train_fe = create_features(combined_train_df)
# test_fe = create_features(original_test_df)

# # --- 4. Define Feature Set and Prepare Data ---
# features_fe = [
#     'RhythmScore','AudioLoudness', 'VocalContent', 'AcousticQuality',
#     'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
#     'TrackDurationMs','Energy', 'mood_energy_interaction', 
#     'loudness_vocal_interaction', 'acoustic_instrumental_ratio'
# ]
# X = train_fe[features_fe]
# y = train_fe['BeatsPerMinute']
# X_test = test_fe[features_fe]

# # --- 5. Optuna Objective Function ---
# # This function takes a "trial" and returns a score for Optuna to minimize
# def objective(trial):
#     # Define the hyperparameters for Optuna to search over
#     params = {
#         'objective': 'regression_l1',
#         'metric': 'rmse',
#         'n_estimators': 1000,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'verbose': -1,
#         'n_jobs': -1,
#         'seed': 42,
#         'boosting_type': 'gbdt',
#     }
    
#     n_splits = 5
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     oof_rmse_scores = []

#     # Using cross-validation to get a robust score for each trial
#     for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
#         X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#         y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
#         model = lgb.LGBMRegressor(**params)
#         model.fit(X_train, y_train,
#                   eval_set=[(X_val, y_val)],
#                   eval_metric='rmse',
#                   callbacks=[lgb.early_stopping(100, verbose=False)])
        
#         preds = model.predict(X_val)
#         rmse = np.sqrt(mean_squared_error(y_val, preds))
#         oof_rmse_scores.append(rmse)
    
#     return np.mean(oof_rmse_scores)

# # --- 6. Run the Optuna Study ---
# # n_trials=30 means Optuna will test 30 different combinations of hyperparameters.
# # You can increase this for a more thorough search, but it will take longer.
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=30)

# print(f"\nBest trial score (RMSE): {study.best_value}")
# print("Best hyperparameters:")
# for key, value in study.best_params.items():
#     print(f"    {key}: {value}")
# best_params = study.best_params
# best_params['objective'] = 'regression_l1' # Add back the fixed parameters
# best_params['metric'] = 'rmse'
# best_params['n_jobs'] = -1
# best_params['seed'] = 42

# # --- 7. Train Final Model and Create Submission ---
# print("\nTraining final model with best hyperparameters...")
# final_model = lgb.LGBMRegressor(**best_params, n_estimators=2000) # Use more estimators for the final model
# final_model.fit(X, y)

# print("Generating final predictions...")
# final_predictions = final_model.predict(X_test)

# submission_df_optuna = pd.DataFrame({'id': original_test_df['id'], 'BeatsPerMinute': final_predictions})
# submission_df_optuna.to_csv('submission_optuna-only-train.csv', index=False)

# print("\nSubmission file 'submission_optuna.csv' from tuned model created successfully!")
# print(submission_df_optuna.head())

# # (Optional) Save this hyper-tuned model
# joblib.dump(final_model, 'lgbm_optuna_model.joblib')
# print("\nOptuna-tuned model saved to 'lgbm_optuna_model.joblib'")
# #26.40 no improvement


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import KFold
from sklearn import metrics

# --- 1. Load Data (Modify paths as needed) ---
try:
    train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
except FileNotFoundError:
    print("Please make sure 'train.csv' and 'test.csv' are in the same directory.")
    exit()

# --- 2. Feature Engineering ---
train_fe = train.copy()
test_fe = test.copy()

train_fe['mood_energy_interaction'] = train_fe['MoodScore'] * train_fe['Energy']
test_fe['mood_energy_interaction'] = test_fe['MoodScore'] * test_fe['Energy']
train_fe['loudness_vocal_interaction'] = train_fe['AudioLoudness'] * train_fe['VocalContent']
test_fe['loudness_vocal_interaction'] = test_fe['AudioLoudness'] * test_fe['VocalContent']
train_fe['acoustic_instrumental_ratio'] = train_fe['AcousticQuality'] / (train_fe['InstrumentalScore'] + 1e-6)
test_fe['acoustic_instrumental_ratio'] = test_fe['AcousticQuality'] / (test_fe['InstrumentalScore'] + 1e-6)

original_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                    'TrackDurationMs', 'Energy']
new_features = ['mood_energy_interaction', 'loudness_vocal_interaction', 'acoustic_instrumental_ratio']
features = original_features + new_features

X_full_train = train_fe[features].values
y_full_train = train_fe['BeatsPerMinute'].values
X_final_test = test_fe[features].values

# --- 3. Cross-Validation and Model Training ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize arrays to store predictions
oof_lgbm = np.zeros(X_full_train.shape[0])
oof_xgb = np.zeros(X_full_train.shape[0])
oof_cat = np.zeros(X_full_train.shape[0])
test_preds_lgbm = np.zeros(X_final_test.shape[0])
test_preds_xgb = np.zeros(X_final_test.shape[0])
test_preds_cat = np.zeros(X_final_test.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X_full_train, y_full_train)):
    print(f"===== Fold {fold+1} =====")
    X_train, X_val = X_full_train[train_index], X_full_train[val_index]
    y_train, y_val = y_full_train[train_index], y_full_train[val_index]

    # LightGBM Model
    lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgbm[val_index] = lgbm.predict(X_val)
    test_preds_lgbm += lgbm.predict(X_final_test) / n_splits

    # XGBoost Model
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1,
                                 tree_method='hist', early_stopping_rounds=100)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_index] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_final_test) / n_splits

    # CatBoost Model
    cat_model = cat.CatBoostRegressor(n_estimators=1000, learning_rate=0.05, random_state=42,
                                     early_stopping_rounds=100, verbose=0)
    cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    oof_cat[val_index] = cat_model.predict(X_val)
    test_preds_cat += cat_model.predict(X_final_test) / n_splits

# --- 4. Blending and Performance Evaluation ---

# Calculate individual model scores
rmse_lgbm = np.sqrt(metrics.mean_squared_error(y_full_train, oof_lgbm))
rmse_xgb = np.sqrt(metrics.mean_squared_error(y_full_train, oof_xgb))
rmse_cat = np.sqrt(metrics.mean_squared_error(y_full_train, oof_cat))
print(f"\nIndividual Model RMSEs:\nLGBM: {rmse_lgbm:.4f}\nXGB: {rmse_xgb:.4f}\nCatBoost: {rmse_cat:.4f}")

# Simple Even Blend
print("\n--- Running Simple Even Blend ---")
blended_oof_even = (oof_lgbm + oof_xgb + oof_cat) / 3
blended_test_even = (test_preds_lgbm + test_preds_xgb + test_preds_cat) / 3

rmse_even = np.sqrt(metrics.mean_squared_error(y_full_train, blended_oof_even))
print(f"Simple Even Blend RMSE: {rmse_even:.4f}")
submission_even = pd.DataFrame({'id': test_fe['id'], 'BeatsPerMinute': blended_test_even})
submission_even.to_csv('submission_even_blend.csv', index=False)
print("Saved submission_even_blend.csv")

# Weighted Blend
print("\n--- Running Weighted Blend ---")
# Adjust these weights. Tip: Use 1 / RMSE for a simple start.
w_lgbm = 1 / rmse_lgbm
w_xgb = 1 / rmse_xgb
w_cat = 1 / rmse_cat
total_weight = w_lgbm + w_xgb + w_cat
w_lgbm /= total_weight
w_xgb /= total_weight
w_cat /= total_weight

blended_oof_weighted = (w_lgbm * oof_lgbm) + (w_xgb * oof_xgb) + (w_cat * oof_cat)
blended_test_weighted = (w_lgbm * test_preds_lgbm) + (w_xgb * test_preds_xgb) + (w_cat * test_preds_cat)

rmse_weighted = np.sqrt(metrics.mean_squared_error(y_full_train, blended_oof_weighted))
print(f"Weighted Blend RMSE: {rmse_weighted:.4f}")
submission_weighted = pd.DataFrame({'id': test_fe['id'], 'BeatsPerMinute': blended_test_weighted})
submission_weighted.to_csv('submission_weighted_blend.csv', index=False)
print("Saved submission_weighted_blend.csv")


# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error

# # --- Place this code after your cross-validation loop ---

# def two_stage_weight_search(oof_preds, target, stage1_steps=20, stage2_steps=100, clip_lo=0, clip_hi=1):
#     """
#     Finds the optimal blending weights for OOF predictions.
    
#     This function performs a grid search in two stages to find the best weights
#     that minimize the RMSE.
    
#     Args:
#         oof_preds (list of np.array): A list containing the out-of-fold predictions
#                                        from each model.
#         target (np.array): The true target values (y_full_train).
#         stage1_steps (int): The number of steps for the coarse search.
#         stage2_steps (int): The number of steps for the fine-grained search.
#         clip_lo (float): The lower bound to clip the final predictions.
#         clip_hi (float): The upper bound to clip the final predictions.
        
#     Returns:
#         tuple: A tuple containing the optimal weights and the best RMSE.
#     """
    
#     best_loss = 999999.9
#     best_weights = None
    
#     # Stage 1: Coarse search
#     for w1 in np.linspace(0, 1, stage1_steps):
#         for w2 in np.linspace(0, 1, stage1_steps):
#             w3 = 1 - w1 - w2
#             if w3 < 0: continue
            
#             blended_oof = (w1 * oof_preds[0] + w2 * oof_preds[1] + w3 * oof_preds[2])
#             blended_oof = np.clip(blended_oof, clip_lo, clip_hi)
            
#             loss = np.sqrt(mean_squared_error(target, blended_oof))
#             if loss < best_loss:
#                 best_loss = loss
#                 best_weights = [w1, w2, w3]

#     # Stage 2: Fine-grained search around the best weights from Stage 1
#     if best_weights:
#         w1_range = np.linspace(max(0, best_weights[0] - 0.1), min(1, best_weights[0] + 0.1), stage2_steps)
#         w2_range = np.linspace(max(0, best_weights[1] - 0.1), min(1, best_weights[1] + 0.1), stage2_steps)
        
#         for w1 in w1_range:
#             for w2 in w2_range:
#                 w3 = 1 - w1 - w2
#                 if w3 < 0: continue
                
#                 blended_oof = (w1 * oof_preds[0] + w2 * oof_preds[1] + w3 * oof_preds[2])
#                 blended_oof = np.clip(blended_oof, clip_lo, clip_hi)
                
#                 loss = np.sqrt(mean_squared_error(target, blended_oof))
#                 if loss < best_loss:
#                     best_loss = loss
#                     best_weights = [w1, w2, w3]
    
#     return best_weights, best_loss

# # --- Example Usage ---
# # Assuming you have the OOF and test predictions from your 3 models
# # after the cross-validation loop.
# # Example:
# # oof_lgbm, oof_xgb, oof_cat
# # test_preds_lgbm, test_preds_xgb, test_preds_cat

# oof_list = [oof_lgbm, oof_xgb, oof_cat]
# test_list = [test_preds_lgbm, test_preds_xgb, test_preds_cat]

# # Find the optimal weights
# best_weights, best_loss = two_stage_weight_search(oof_list, y_full_train)

# print(f"Optimal Weights: {best_weights}")
# print(f"Blended OOF RMSE: {best_loss}")

# # Apply the optimal weights to the test predictions
# blended_test_final = (best_weights[0] * test_list[0] + 
#                       best_weights[1] * test_list[1] + 
#                       best_weights[2] * test_list[2])

# # Save the final submission file
# submission = pd.DataFrame({'id': test_fe['id'], 'BeatsPerMinute': blended_test_final})
# submission.to_csv('submission_best_blend.csv', index=False)
# print("Saved submission_best_blend.csv with optimal weights.")


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import QuantileTransformer
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

# --- 1. Constants and Data Loading ---
ID_COL = 'id'
TARGET_COL = 'BeatsPerMinute'

try:
    train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
except FileNotFoundError:
    print("Please make sure 'train.csv' and 'test.csv' are in the same directory.")
    exit()

# --- 2. Feature Engineering ---
train_fe = train.copy()
test_fe = test.copy()

# New interaction features
train_fe['mood_energy_interaction'] = train_fe['MoodScore'] * train_fe['Energy']
test_fe['mood_energy_interaction'] = test_fe['MoodScore'] * test_fe['Energy']
train_fe['loudness_vocal_interaction'] = train_fe['AudioLoudness'] * train_fe['VocalContent']
test_fe['loudness_vocal_interaction'] = test_fe['AudioLoudness'] * test_fe['VocalContent']
train_fe['acoustic_instrumental_ratio'] = train_fe['AcousticQuality'] / (train_fe['InstrumentalScore'] + 1e-6)
test_fe['acoustic_instrumental_ratio'] = test_fe['AcousticQuality'] / (test_fe['InstrumentalScore'] + 1e-6)

# Select features for the models
features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
            'TrackDurationMs', 'Energy', 'mood_energy_interaction',
            'loudness_vocal_interaction', 'acoustic_instrumental_ratio']

X_full_train = train_fe[features].values
y_full_train = train_fe[TARGET_COL].values
X_final_test = test_fe[features].values

# --- 3. Cross-Validation and Model Training ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgbm = np.zeros(X_full_train.shape[0])
oof_xgb = np.zeros(X_full_train.shape[0])
oof_cat = np.zeros(X_full_train.shape[0])
test_preds_lgbm = np.zeros(X_final_test.shape[0])
test_preds_xgb = np.zeros(X_final_test.shape[0])
test_preds_cat = np.zeros(X_final_test.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X_full_train, y_full_train)):
    print(f"===== Fold {fold+1} =====")
    X_train, X_val = X_full_train[train_index], X_full_train[val_index]
    y_train, y_val = y_full_train[train_index], y_full_train[val_index]

    # LightGBM Model
    lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgbm[val_index] = lgbm.predict(X_val)
    test_preds_lgbm += lgbm.predict(X_final_test) / n_splits

    # XGBoost Model
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1,
                                 tree_method='hist', early_stopping_rounds=100)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_index] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_final_test) / n_splits

    # CatBoost Model
    cat_model = cat.CatBoostRegressor(n_estimators=1000, learning_rate=0.05, random_state=42,
                                     early_stopping_rounds=100, verbose=0)
    cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    oof_cat[val_index] = cat_model.predict(X_val)
    test_preds_cat += cat_model.predict(X_final_test) / n_splits

# --- 4. Optimal Blending with Weight Search ---
def rmse(weights, preds, target):
    blended_preds = np.average(preds, axis=0, weights=weights)
    return np.sqrt(mean_squared_error(target, blended_preds))

oof_list = np.array([oof_lgbm, oof_xgb, oof_cat])
test_list = np.array([test_preds_lgbm, test_preds_xgb, test_preds_cat])

# Find the optimal weights using an optimization function
initial_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1) for _ in range(len(initial_weights))]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

res = minimize(rmse, initial_weights, args=(oof_list, y_full_train), method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print(f"\nOptimal Blending Weights: {best_weights}")

# Apply the optimal weights to the test predictions
blended_test_final = np.average(test_list, axis=0, weights=best_weights)
blended_oof_final = np.average(oof_list, axis=0, weights=best_weights)

# --- 5. Final Isotonic Regression Calibration ---
ir = IsotonicRegression(y_min=train_fe[TARGET_COL].min(), y_max=train_fe[TARGET_COL].max(), out_of_bounds="clip")

# Fit the calibrator on the blended OOF predictions
ir.fit(blended_oof_final, y_full_train)

# Calibrate the final test predictions
calibrated_test_preds = ir.transform(blended_test_final)

# --- 6. Save the Final Submission File ---
submission = pd.DataFrame({'id': test_fe[ID_COL], TARGET_COL: calibrated_test_preds})
submission.to_csv('submission_final_calibrated.csv', index=False)
print("Saved submission_final_calibrated.csv with optimal weights and calibration.")


# -*- coding: utf-8 -*-
"""
Combined & Optimized BPM Prediction Pipeline
-------------------------------------------
This script merges the best of both previous approaches:
1.  Comprehensive EDA and feature engineering from the first script.
2.  Robust K-Fold CV, optimal blending, and calibration from the second script.
3.  Adds an automated feature selection step for improved performance.
"""

# --- 0. IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# --- 1. DATA LOADING & INITIAL EXPLORATION ---
print("ğŸ�µ Loading Data...")
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5/sample_submission.csv")
except FileNotFoundError:
    print("âš ï¸� Local dev: Using placeholder paths. Update for Kaggle environment.")
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


ID_COL = 'id'
TARGET_COL = 'BeatsPerMinute'

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# EDA: Target Distribution
plt.figure(figsize=(12, 6))
sns.histplot(train_df[TARGET_COL], bins=50, kde=True)
plt.title('ğŸ�¯ Distribution of BeatsPerMinute (BPM)', fontsize=16)
plt.xlabel('BPM')
plt.ylabel('Frequency')
plt.show()

# --- 2. FEATURE ENGINEERING ---
print("\nğŸ§® Generating a wide range of features...")
def create_features(df):
    df_copy = df.copy()
    # Interactions
    df_copy['mood_energy_interaction'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['loudness_vocal_interaction'] = df_copy['AudioLoudness'] * df_copy['VocalContent']
    # Ratios
    df_copy['acoustic_instrumental_ratio'] = df_copy['AcousticQuality'] / (df_copy['InstrumentalScore'] + 1e-6)
    df_copy['rhythm_energy_ratio'] = df_copy['RhythmScore'] / (df_copy['Energy'] + 1e-6)
    # Polynomial features
    df_copy['Energy_sq'] = df_copy['Energy']**2
    df_copy['RhythmScore_sq'] = df_copy['RhythmScore']**2
    return df_copy

train_fe = create_features(train_df)
test_fe = create_features(test_df)

print(f"New shape after FE: Train={train_fe.shape}, Test={test_fe.shape}")

# --- 3. FEATURE SELECTION ---
print("\nğŸ”� Selecting the most impactful features...")

# Define all potential features
all_features = [col for col in train_fe.columns if col not in [ID_COL, TARGET_COL]]

# Use a fast LightGBM model to rank features by importance
selector_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
selector_model.fit(train_fe[all_features], train_fe[TARGET_COL])

# Create a DataFrame of feature importances
feature_importances = pd.DataFrame({
    'feature': all_features,
    'importance': selector_model.feature_importances_
}).sort_values('importance', ascending=False)

# Select the top N features (e.g., top 15)
N_FEATURES = 15
selected_features = feature_importances.head(N_FEATURES)['feature'].tolist()

print(f"Selected the top {N_FEATURES} features: {selected_features}")

# Prepare final data
X = train_fe[selected_features].values
y = train_fe[TARGET_COL].values
X_test = test_fe[selected_features].values

# --- 4. CROSS-VALIDATION & MODEL TRAINING ---
print("\nğŸ¤– Training models with 5-Fold Cross-Validation...")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize prediction arrays
oof_lgbm, oof_xgb, oof_cat = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
test_preds_lgbm, test_preds_xgb, test_preds_cat = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))
fold_importances = pd.DataFrame(index=selected_features)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"===== Fold {fold+1} =====")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # LightGBM
    lgbm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, random_state=42+fold, n_jobs=-1)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgbm[val_idx] = lgbm.predict(X_val)
    test_preds_lgbm += lgbm.predict(X_test) / n_splits
    fold_importances[f'lgbm_fold_{fold+1}'] = lgbm.feature_importances_

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, random_state=42+fold, n_jobs=-1,
                                 tree_method='hist', early_stopping_rounds=100)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test) / n_splits

    # CatBoost
    cat_model = cat.CatBoostRegressor(n_estimators=2000, learning_rate=0.03, random_state=42+fold,
                                      early_stopping_rounds=100, verbose=0)
    cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_preds_cat += cat_model.predict(X_test) / n_splits

# --- 5. OPTIMAL BLENDING & CALIBRATION ---
print("\nâš–ï¸� Finding optimal blend weights and calibrating...")

# Define the RMSE objective function for the optimizer
def rmse_objective(weights, preds, target):
    blended_preds = np.average(preds, axis=0, weights=weights)
    return np.sqrt(mean_squared_error(target, blended_preds))

oof_list = np.array([oof_lgbm, oof_xgb, oof_cat])
test_list = np.array([test_preds_lgbm, test_preds_xgb, test_preds_cat])

# Find optimal weights
initial_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1) for _ in range(len(initial_weights))]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
res = minimize(rmse_objective, initial_weights, args=(oof_list, y),
               method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print(f"Optimal Weights: LGBM={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, CAT={best_weights[2]:.4f}")

# Blend OOF and test predictions
blended_oof = np.average(oof_list, axis=0, weights=best_weights)
blended_test = np.average(test_list, axis=0, weights=best_weights)

# Isotonic Regression Calibration
ir = IsotonicRegression(y_min=y.min(), y_max=y.max(), out_of_bounds="clip")
ir.fit(blended_oof, y)
calibrated_preds = ir.transform(blended_test)

final_oof_rmse = np.sqrt(mean_squared_error(y, blended_oof))
print(f"Final Blended OOF RMSE: {final_oof_rmse:.5f}")

# --- 6. FINAL ANALYSIS & DIAGNOSTICS ---
print("\nğŸ“Š Final analysis...")

# Feature Importance Plot (averaged over folds for LGBM)
fold_importances['mean'] = fold_importances.mean(axis=1)
fold_importances.sort_values('mean', ascending=False, inplace=True)
plt.figure(figsize=(10, 8))
sns.barplot(x='mean', y=fold_importances.index, data=fold_importances, palette='viridis')
plt.title('ğŸ”¥ Top Feature Importances (Averaged Across Folds)', fontsize=16)
plt.xlabel('Mean Importance Score')
plt.ylabel('Features')
plt.show()

# Diagnostic Plot: Predictions vs Actuals
plt.figure(figsize=(8, 8))
plt.scatter(y, blended_oof, alpha=0.3, s=5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.title(f'ğŸ�¯ Predictions vs Actuals (OOF)\nRMSE: {final_oof_rmse:.4f}', fontsize=16)
plt.xlabel('Actual BPM')
plt.ylabel('Predicted BPM')
plt.legend()
plt.grid(True)
plt.show()

# --- 7. SUBMISSION ---
print("\nğŸ“„ Creating submission file...")
submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: calibrated_preds})
submission_df.to_csv('submission.csv', index=False)
print("âœ… Submission file 'submission.csv' created successfully!")


# --- 0. IMPORTS ---
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# --- 1. DATA LOADING & CONSTANTS ---
print("ğŸ�µ Loading Data...")
ID_COL = 'id'
TARGET_COL = 'BeatsPerMinute'
try:
    train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
except FileNotFoundError:
    train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# --- 2. FEATURE ENGINEERING (Based on your best script) ---
print("ğŸ§® Applying the proven feature engineering strategy...")
def engineer_features(df):
    df_fe = df.copy()
    df_fe['mood_energy_interaction'] = df_fe['MoodScore'] * df_fe['Energy']
    df_fe['loudness_vocal_interaction'] = df_fe['AudioLoudness'] * df_fe['VocalContent']
    df_fe['acoustic_instrumental_ratio'] = df_fe['AcousticQuality'] / (df_fe['InstrumentalScore'] + 1e-6)
    return df_fe

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Use the exact feature list from your high-performing script
features = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy', 'mood_energy_interaction',
    'loudness_vocal_interaction', 'acoustic_instrumental_ratio'
]

X = train_fe[features].values
y = train[TARGET_COL].values
X_test = test_fe[features].values

# --- 3. CROSS-VALIDATION & DIVERSE MODEL TRAINING ---
print("ğŸ¤– Training a diverse set of models with 5-Fold Cross-Validation...")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize OOF and test predictions for all 4 models
oof_preds = np.zeros((len(train), 4))
test_preds = np.zeros((len(test), 4))

# Model parameters from your best script
lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1}
xgb_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1, 'tree_method': 'hist'}
cat_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'random_state': 42, 'verbose': 0}
ridge_params = {'alpha': 50, 'random_state': 42}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"===== Fold {fold+1} =====")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Scale data for Ridge model (important!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 1. LightGBM
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds[val_idx, 0] = lgbm.predict(X_val)
    test_preds[:, 0] += lgbm.predict(X_test) / n_splits

    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds[val_idx, 1] = xgb_model.predict(X_val)
    test_preds[:, 1] += xgb_model.predict(X_test) / n_splits

    # 3. CatBoost
    cat_model = cat.CatBoostRegressor(**cat_params, early_stopping_rounds=100)
    cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    oof_preds[val_idx, 2] = cat_model.predict(X_val)
    test_preds[:, 2] += cat_model.predict(X_test) / n_splits
    
    # 4. Ridge (Diverse Model)
    ridge = Ridge(**ridge_params)
    ridge.fit(X_train_scaled, y_train)
    oof_preds[val_idx, 3] = ridge.predict(X_val_scaled)
    X_test_scaled = scaler.transform(X_test)
    test_preds[:, 3] += ridge.predict(X_test_scaled) / n_splits

# --- 4. OPTIMAL BLENDING & CALIBRATION ---
print("\nâš–ï¸� Finding optimal blend weights for the 4-model ensemble...")

def rmse_objective(weights, oof_preds, target):
    blended_preds = np.dot(oof_preds, weights)
    return np.sqrt(mean_squared_error(target, blended_preds))

# Find optimal weights for the 4 models
initial_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1) for _ in range(4)]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
res = minimize(rmse_objective, initial_weights, args=(oof_preds, y),
               method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print(f"Optimal Weights: LGBM={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, CAT={best_weights[2]:.4f}, RIDGE={best_weights[3]:.4f}")

# Blend OOF and test predictions
blended_oof = np.dot(oof_preds, best_weights)
blended_test = np.dot(test_preds, best_weights)
final_oof_rmse = np.sqrt(mean_squared_error(y, blended_oof))
print(f"Final Blended OOF RMSE: {final_oof_rmse:.5f}")

# Isotonic Regression Calibration
ir = IsotonicRegression(y_min=y.min(), y_max=y.max(), out_of_bounds="clip")
ir.fit(blended_oof, y)
calibrated_preds = ir.transform(blended_test)

# --- 5. SUBMISSION ---
print("\nğŸ“„ Creating final submission file...")
submission = pd.DataFrame({'id': test[ID_COL], TARGET_COL: calibrated_preds})
submission.to_csv('submission_diverse_ensemble.csv', index=False)
print("âœ… Submission file 'submission_diverse_ensemble.csv' created successfully!")


import numpy as np
import pandas as pd

dtrain = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
sub_sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv') 

sub_sample['BeatsPerMinute'] = np.mean(dtrain['BeatsPerMinute'])   
sub_sample.to_csv('submission_mean.csv', index=False)    # Public Score: 26.39492

sub_sample['BeatsPerMinute'] = np.median(dtrain['BeatsPerMinute'])   
sub_sample.to_csv('submission_median.csv', index=False)    # Public Score: 26.39822

sub_sample      


#Diverse enemble our best shot yet so far!


sub_import = pd.read_csv('/kaggle/working/submission_diverse_ensemble.csv') 

# ..................................................................
per = sub_import['BeatsPerMinute'].values

min_per = np.min(per)
max_per = np.max(per)
mean_per = np.mean(per)

min_per, max_per, mean_per


per1 = [f for f in per if f < (min_per+7)]
per2 = [f for f in per if f > (max_per-9)]

len(per1), len(per2)


for i in range(len(per)):
        
    if per[i] < (min_per+10):
        per[i] = per[i] - 0.6

    if per[i] > (max_per-10):
        per[i] = per[i] + 0.5

# ......................................................
sub_sample['BeatsPerMinute'] = per
sub_sample.to_csv('submission_value.csv', index=False)   
sub_sample   


sub_import = pd.read_csv('/kaggle/input/data-is-all-you-need-to-care-about/final_weighted_blended_predictions.csv') 

# ............................................................................
per = sub_import['prediction'].values

min_per = np.min(per)
max_per = np.max(per)
mean_per = np.mean(per)

min_per, max_per, mean_per


R = 0.0    # Adjusting the R value can increase the accuracy of the guide.
guide = mean_per - R

# ....................................
per1 = [f for f in per if f < guide]
per2 = [f for f in per if f > guide]

len(per1), len(per2)


for i in range(len(per)):
        
    if per[i] < guide:
        per[i] = (per[i]* 1.10) - (guide* 0.10)

    if per[i] > guide:
        per[i] = (per[i]* 1.05) - (guide* 0.05)

# ...................................................
sub_sample['BeatsPerMinute'] = per
sub_sample.to_csv('submission_***1.csv', index=False) 
sub_sample 


# --- IMPORTS ---
import pandas as pd 
import numpy as np 
import os 
import time
import logging 
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from itertools import combinations
import warnings
warnings.simplefilter('ignore')

# --- DATA LOADING ---
print("ğŸ�µ Loading Data...")
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
df_external_1 = pd.read_csv("/kaggle/input/beats-per-minute-xgb-lgb-cat/submission.csv")

# --- FEATURE ENGINEERING ---
print("ğŸ§® Applying feature engineering...")

def bin_column(df, column, bins, bin_names=None):
    if bin_names is None:
        bin_names = [f'{b:.1f}_to_{b_next:.1f}' for b, b_next in zip(bins[:-1], bins[1:])]
    df[column + '_binned'] = pd.cut(df[column], bins=bins, labels=bin_names, include_lowest=True)
    return df

# Binning
bins = [0.025, 0.1, 0.15, 0.2]
train = bin_column(train, 'VocalContent', bins)
test = bin_column(test, 'VocalContent', bins)
bins = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
train = bin_column(train, 'AcousticQuality', bins)
test = bin_column(test, 'AcousticQuality', bins)
bins = [0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
train = bin_column(train, 'InstrumentalScore', bins)
test = bin_column(test, 'InstrumentalScore', bins)
bins = [0.05, 0.2, 0.4]
train = bin_column(train, 'LivePerformanceLikelihood', bins)
test = bin_column(test, 'LivePerformanceLikelihood', bins)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
train = bin_column(train, 'MoodScore', bins)
test = bin_column(test, 'MoodScore', bins)

# Interaction and Ratio Features
numerical_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000 
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    
    df_new['acoustic_instrumental_ratio'] = df_new['AcousticQuality'] / (df_new['InstrumentalScore'] + 1e-6)
    df_new['RhythmEnergyRatio'] = df_new['RhythmScore'] / (df_new['Energy'] + 1e-8)
    df_new['VocalInstrumentalRatio'] = df_new['VocalContent'] / (df_new['InstrumentalScore'] + 1e-8)
    df['EnergyBin'] = pd.cut(df['Energy'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    df['RhythmBin'] = pd.cut(df['RhythmScore'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

# Polynomial Features
def add_feature_sq_terms(df, numerical_features):
    for feature in numerical_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
    return df
    
train = add_feature_sq_terms(train, numerical_features)
test = add_feature_sq_terms(test, numerical_features)

# --- DATA PREPARATION ---
print("Preparing data for modeling...")
X = train.drop(columns=["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]
X_test = test.drop(columns=["id"])

# Align columns after feature engineering
train_cols = X.columns
test_cols = X_test.columns
shared_cols = list(set(train_cols) & set(test_cols))
X = X[shared_cols]
X_test = X_test[shared_cols]

# --- MODEL 1: XGBOOST ---
print("ğŸ¤– Training XGBoost model...")
FOLDS = 10
FEATURES = X.columns.tolist()
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"===== XGB Fold {i+1} =====")
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    model = XGBRegressor(
        device="cuda",
        max_depth=6, colsample_bytree=0.9, subsample=0.9,
        n_estimators=2000, learning_rate=0.02, gamma=10.0, 
        max_delta_step=2, early_stopping_rounds=100,
        eval_metric="rmse", enable_categorical=True
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
    oof_xgb[valid_idx] = model.predict(x_valid)
    pred_xgb += model.predict(X_test) / FOLDS

full_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
print(f"\nXGB Final CV RMSE: {full_rmse:.4f}")

# --- MODEL 2: LIGHTGBM ---
print("\nğŸ¤– Training LightGBM model...")
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=2025)
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"===== LGBM Fold {i+1} =====")
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    model = LGBMRegressor(
        device="gpu", max_depth=6, colsample_bytree=0.9, subsample=0.9,
        n_estimators=2000, learning_rate=0.03, reg_alpha=0.8, 
        reg_lambda=4.0, metric="rmse", verbose=-1
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[valid_idx] = model.predict(x_valid)
    pred_lgb += model.predict(X_test) / FOLDS

full_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
print(f"\nLGBM Final CV RMSE: {full_rmse:.4f}")

# --- MODEL 3: CATBOOST ---
print("\nğŸ¤– Training CatBoost model...")
cat_features = [col for col in FEATURES if X[col].dtype == 'category']
for col in cat_features:
    X[col] = X[col].cat.add_categories(['missing']).fillna('missing')
    X_test[col] = X_test[col].cat.add_categories(['missing']).fillna('missing')

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=2026)
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"===== CAT Fold {i+1} =====")
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    model = CatBoostRegressor(
        task_type="GPU",
        max_depth=6,
        colsample_bylevel=0.9,
        subsample=0.9,
        bootstrap_type='MVS', # <<< FIX: Add this line
        n_estimators=2000,
        learning_rate=0.08,
        random_strength=0.1, 
        early_stopping_rounds=100,
        loss_function="RMSE",
        verbose=0
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], cat_features=cat_features, use_best_model=True)
    oof_cat[valid_idx] = model.predict(x_valid)
    pred_cat += model.predict(X_test) / FOLDS

full_rmse = np.sqrt(mean_squared_error(y, oof_cat))
print(f"\nCAT Final CV RMSE: {full_rmse:.4f}")

# --- ENSEMBLING & SUBMISSION ---
print("\nğŸ“„ Creating final submission file...")
pred = df_external_1["BeatsPerMinute"] * .85 + pred_xgb * 0.05 + pred_lgb * 0.05 + pred_cat * 0.05
y_pred_after = np.clip(pred, 46.718, 206.037)

submission["BeatsPerMinute"] = y_pred_after
submission.to_csv("submission_finallll.csv", index=False)

print("âœ… Submission file 'submission.csv' created successfully!")
print("Final submission head:")
print(submission.head())







