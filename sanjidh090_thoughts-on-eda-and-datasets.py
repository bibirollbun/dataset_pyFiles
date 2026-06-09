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
        analyzer = MusicDataAnalyzer(file_path="/kaggle/input/bpm-prediction-challenge/Train.csv") # <--- CHANGE THIS PATH if needed
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
        analyzer = MusicDataAnalyzer(file_path="/kaggle/input/playground-series-s5e9/train.csv") # <--- CHANGE THIS PATH if needed
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



# # --- 0. IMPORTS ---
# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import xgboost as xgb
# import catboost as cat
# import itertools
# import warnings
# from category_encoders import TargetEncoder
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.isotonic import IsotonicRegression

# # Configure warnings
# warnings.filterwarnings('ignore')

# # --- Constants ---
# ID_COL = 'id'
# TARGET_COL = 'BeatsPerMinute'
# N_SPLITS = 10        # Number of folds for cross-validation
# RANDOM_STATE = 42
# TOP_N_FEATURES = 200 # Number of features to select after engineering

# # --- 1. DATA LOADING ---
# print("ğŸ�µ Loading Data...")
# try:
#     # Adjust path if necessary
#     train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
#     test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
# except FileNotFoundError:
#     print("Error: Data files not found. Please check paths.")
#     # Dummy data for script execution if files are missing (for testing logic)
#     # train_df = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
#     # train_df[TARGET_COL] = np.random.rand(100) * 100 + 50
#     # test_df = pd.DataFrame(np.random.rand(50, 10), columns=[f'f{i}' for i in range(10)])
#     # train_df[ID_COL] = range(100)
#     # test_df[ID_COL] = range(100, 150)

# # --- 2. FEATURE ENGINEERING (Combined Strategy) ---
# print("ğŸ§® Engineering features by combining strategies...")

# def feature_engineer(train_data, test_data):
#     """Applies combined feature engineering from both notebooks."""
#     # Combine train and test for consistent processing
#     combined_df = pd.concat([train_data.drop(TARGET_COL, axis=1), test_data], ignore_index=True)
    
#     # --- Strategy 1: Manual Interactions (from Script 2) ---
#     combined_df['mood_energy_interaction'] = combined_df['MoodScore'] * combined_df['Energy']
#     combined_df['loudness_vocal_interaction'] = combined_df['AudioLoudness'] * combined_df['VocalContent']
#     combined_df['acoustic_instrumental_ratio'] = combined_df['AcousticQuality'] / (combined_df['InstrumentalScore'] + 1e-6)
    
#     # --- Strategy 2: Binning and Combinatorial Features (from Script 1) ---
#     # Binning continuous features
#     combined_df['EnergyBin'] = pd.cut(combined_df['Energy'], bins=10, labels=False)
#     combined_df['RhythmBin'] = pd.cut(combined_df['RhythmScore'], bins=10, labels=False)
#     combined_df['MoodBin'] = pd.cut(combined_df['MoodScore'], bins=10, labels=False)

#     # Combinatorial feature creation logic
#     combinatorial_features_to_create = []
#     base_combinatorial_cols = [
#         'RhythmScore', 'Energy', 'MoodScore', 'VocalContent', 
#         'AcousticQuality', 'EnergyBin', 'RhythmBin'
#     ]
    
#     # Create combinations of size 2 and 3
#     for combo_size in [2, 3]:
#         for combo in itertools.combinations(base_combinatorial_cols, combo_size):
#             new_feature_name = '||'.join(combo)
#             combinatorial_features_to_create.append(new_feature_name)
            
#             # Create feature value by concatenating string representations
#             combined_df[new_feature_name] = combined_df[list(combo)].astype(str).agg('_'.join, axis=1)

#     return combined_df, combinatorial_features_to_create

# # Apply feature engineering
# df_processed, combinatorial_features = feature_engineer(train_df, test_df)

# # --- 3. PREPROCESSING & FEATURE SELECTION ---
# print("ğŸ”’ Encoding features and selecting top performers...")

# # Separate train and test again
# X = df_processed.iloc[:len(train_df)]
# X_test = df_processed.iloc[len(train_df):]
# y = train_df[TARGET_COL]

# # Target Encoding for high-cardinality combinatorial features
# # This converts string categories into meaningful numeric values based on target average
# encoder = TargetEncoder(target_type='continuous', random_state=RANDOM_STATE)
# X[combinatorial_features] = encoder.fit_transform(X[combinatorial_features], y)
# X_test[combinatorial_features] = encoder.transform(X_test[combinatorial_features])

# # Identify all features (original + engineered)
# original_features = test_df.columns.drop(ID_COL).tolist()
# manual_features = ['mood_energy_interaction', 'loudness_vocal_interaction', 'acoustic_instrumental_ratio']
# all_features = original_features + manual_features + combinatorial_features

# # Clean up data for modeling (fill NaNs that might result from encoding unseen categories)
# X[all_features] = X[all_features].fillna(X[all_features].mean())
# X_test[all_features] = X_test[all_features].fillna(X[all_features].mean())

# # Feature Selection using LightGBM feature importance
# print(f"Selecting top {TOP_N_FEATURES} features from {len(all_features)} total candidates...")
# lgbm_selector = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
# lgbm_selector.fit(X[all_features], y)

# importances = pd.Series(lgbm_selector.feature_importances_, index=all_features)
# selected_features = importances.sort_values(ascending=False).index[:TOP_N_FEATURES].tolist()

# # Final data for modeling
# X_final = X[selected_features]
# X_test_final = X_test[selected_features]

# print(f"Data ready for modeling with {len(selected_features)} features.")

# # --- 4. LEVEL 1 MODELING: CROSS-VALIDATION ENSEMBLE ---
# print("ğŸ¤– Training Level 1 models (LGBM, XGB, CatBoost, Ridge)...")

# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# # Initialize OOF (Out-of-Fold) predictions for meta-model training
# oof_preds = np.zeros((len(X_final), 4)) 
# # Initialize test predictions for meta-model inference
# test_preds = np.zeros((len(X_test_final), 4))

# # Model Parameters
# lgbm_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'objective': 'regression_l1',
#                'metric': 'rmse', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
#                'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1}

# xgb_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'objective': 'reg:squarederror',
#               'eval_metric': 'rmse', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist',
#               'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1}

# cat_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'loss_function': 'RMSE',
#               'random_seed': RANDOM_STATE, 'verbose': 0, 'early_stopping_rounds': 100}

# ridge_params = {'alpha': 20, 'random_state': RANDOM_STATE}

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_final, y)):
#     print(f"===== Fold {fold+1}/{N_SPLITS} =====")
#     X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # Scale data for Ridge model (Tree models don't require scaling)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test_final)

#     # --- Model 1: LightGBM ---
#     model_lgbm = lgb.LGBMRegressor(**lgbm_params)
#     model_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
#     oof_preds[val_idx, 0] = model_lgbm.predict(X_val)
#     test_preds[:, 0] += model_lgbm.predict(X_test_final) / N_SPLITS

#     # --- Model 2: XGBoost ---
#     model_xgb = xgb.XGBRegressor(**xgb_params)
#     model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
#     oof_preds[val_idx, 1] = model_xgb.predict(X_val)
#     test_preds[:, 1] += model_xgb.predict(X_test_final) / N_SPLITS

#     # --- Model 3: CatBoost ---
#     model_cat = cat.CatBoostRegressor(**cat_params)
#     model_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)])
#     oof_preds[val_idx, 2] = model_cat.predict(X_val)
#     test_preds[:, 2] += model_cat.predict(X_test_final) / N_SPLITS
    
#     # --- Model 4: Ridge Regression ---
#     model_ridge = Ridge(**ridge_params)
#     model_ridge.fit(X_train_scaled, y_train)
#     oof_preds[val_idx, 3] = model_ridge.predict(X_val_scaled)
#     test_preds[:, 3] += model_ridge.predict(X_test_scaled) / N_SPLITS

# # Display OOF RMSE for each base model
# print("\n--- Level 1 Model Performance (OOF RMSE) ---")
# print(f"LGBM RMSE:   {np.sqrt(mean_squared_error(y, oof_preds[:, 0])):.5f}")
# print(f"XGBoost RMSE:{np.sqrt(mean_squared_error(y, oof_preds[:, 1])):.5f}")
# print(f"CatBoost RMSE:{np.sqrt(mean_squared_error(y, oof_preds[:, 2])):.5f}")
# print(f"Ridge RMSE:  {np.sqrt(mean_squared_error(y, oof_preds[:, 3])):.5f}")

# # --- 5. LEVEL 2 STACKING & CALIBRATION ---
# print("\nğŸ§  Training Level 2 Meta-Model (Stacking)...")

# # Level 2 training data uses OOF predictions from Level 1 models
# X_meta_train = oof_preds
# y_meta_train = y

# # Level 2 test data uses averaged test predictions from Level 1 models
# X_meta_test = test_preds

# # Train a simple linear model as the meta-model. It's robust against overfitting.
# meta_model = Ridge(alpha=0.5, fit_intercept=True)
# meta_model.fit(X_meta_train, y_meta_train)

# # Calculate final blended OOF RMSE
# blended_oof_preds = meta_model.predict(X_meta_train)
# final_oof_rmse = np.sqrt(mean_squared_error(y_meta_train, blended_oof_preds))
# print(f"\nFinal Stacked OOF RMSE: {final_oof_rmse:.5f}")

# # Generate final predictions on test data using the meta-model
# stacked_test_preds = meta_model.predict(X_meta_test)

# # --- 6. POST-PROCESSING CALIBRATION ---
# print("calibrating final predictions...")
# ir = IsotonicRegression(y_min=y.min(), y_max=y.max(), out_of_bounds="clip")

# # Fit calibration model on the OOF predictions from the stacking layer
# ir.fit(blended_oof_preds, y_meta_train)
# calibrated_preds = ir.transform(stacked_test_preds)

# # --- 7. SUBMISSION ---
# print("\nğŸ“„ Creating final submission file...")
# submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: calibrated_preds})
# submission_df.to_csv('submission_checkin-in.csv', index=False)
# print("âœ… Submission file 'submission_checkjik.csv' created successfully!")

# # Display prediction distribution comparison
# print("\n--- Prediction Statistics ---")
# print("Target Statistics (Train):")
# print(f"  Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}, Std: {y.std():.2f}")
# print("\nFinal Prediction Statistics (Test):")
# print(f"  Min: {calibrated_preds.min():.2f}, Max: {calibrated_preds.max():.2f}, Mean: {calibrated_preds.mean():.2f}, Std: {calibrated_preds.std():.2f}")


# # --- 0. INSTALL DEPENDENCIES (only needed once, uncomment if in Kaggle/Colab) ---
# # !pip install --upgrade --force-reinstall scikit-learn lightgbm xgboost catboost category_encoders

# # --- 0. IMPORTS ---
# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import xgboost as xgb
# import catboost as cat
# import itertools
# import warnings
# from category_encoders import TargetEncoder
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.isotonic import IsotonicRegression

# # Configure warnings
# warnings.filterwarnings('ignore')

# # --- Constants ---
# ID_COL = 'id'
# TARGET_COL = 'BeatsPerMinute'
# N_SPLITS = 10        # Number of folds for cross-validation
# RANDOM_STATE = 42
# TOP_N_FEATURES = 200 # Number of features to select after engineering

# # --- 1. DATA LOADING ---
# print("ğŸ�µ Loading Data...")
# try:
#     train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
#     test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
# except FileNotFoundError:
#     print("Error: Data files not found. Using dummy data for testing...")
#     train_df = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
#     train_df[TARGET_COL] = np.random.rand(100) * 100 + 50
#     test_df = pd.DataFrame(np.random.rand(50, 10), columns=[f'f{i}' for i in range(10)])
#     train_df[ID_COL] = range(100)
#     test_df[ID_COL] = range(100, 150)

# # --- 2. FEATURE ENGINEERING ---
# print("ğŸ§® Engineering features by combining strategies...")

# def feature_engineer(train_data, test_data):
#     combined_df = pd.concat([train_data.drop(TARGET_COL, axis=1, errors='ignore'), test_data], ignore_index=True)

#     # Manual interactions
#     if all(col in combined_df for col in ['MoodScore', 'Energy']):
#         combined_df['mood_energy_interaction'] = combined_df['MoodScore'] * combined_df['Energy']
#     if all(col in combined_df for col in ['AudioLoudness', 'VocalContent']):
#         combined_df['loudness_vocal_interaction'] = combined_df['AudioLoudness'] * combined_df['VocalContent']
#     if all(col in combined_df for col in ['AcousticQuality', 'InstrumentalScore']):
#         combined_df['acoustic_instrumental_ratio'] = combined_df['AcousticQuality'] / (combined_df['InstrumentalScore'] + 1e-6)

#     # Binning
#     for col in ['Energy', 'RhythmScore', 'MoodScore']:
#         if col in combined_df:
#             combined_df[f"{col}Bin"] = pd.cut(combined_df[col], bins=10, labels=False)

#     # Combinatorial features
#     combinatorial_features_to_create = []
#     base_cols = [c for c in ['RhythmScore', 'Energy', 'MoodScore', 'VocalContent', 'AcousticQuality', 'EnergyBin', 'RhythmBin'] if c in combined_df]

#     for combo_size in [2, 3]:
#         for combo in itertools.combinations(base_cols, combo_size):
#             new_feature_name = '||'.join(combo)
#             combinatorial_features_to_create.append(new_feature_name)
#             combined_df[new_feature_name] = combined_df[list(combo)].astype(str).agg('_'.join, axis=1)

#     return combined_df, combinatorial_features_to_create

# # Apply feature engineering
# df_processed, combinatorial_features = feature_engineer(train_df, test_df)

# # --- 3. PREPROCESSING & FEATURE SELECTION ---
# print("ğŸ”’ Encoding features and selecting top performers...")

# X = df_processed.iloc[:len(train_df)]
# X_test = df_processed.iloc[len(train_df):]
# y = train_df[TARGET_COL]

# # Target Encoding
# encoder = TargetEncoder()
# X[combinatorial_features] = encoder.fit_transform(X[combinatorial_features], y)
# X_test[combinatorial_features] = encoder.transform(X_test[combinatorial_features])

# original_features = [c for c in test_df.columns if c != ID_COL]
# manual_features = [f for f in ['mood_energy_interaction', 'loudness_vocal_interaction', 'acoustic_instrumental_ratio'] if f in df_processed]
# all_features = original_features + manual_features + combinatorial_features

# X[all_features] = X[all_features].fillna(X[all_features].mean())
# X_test[all_features] = X_test[all_features].fillna(X[all_features].mean())

# print(f"Selecting top {TOP_N_FEATURES} features...")
# lgbm_selector = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
# lgbm_selector.fit(X[all_features], y)

# importances = pd.Series(lgbm_selector.feature_importances_, index=all_features)
# selected_features = importances.sort_values(ascending=False).index[:TOP_N_FEATURES].tolist()

# X_final = X[selected_features]
# X_test_final = X_test[selected_features]

# print(f"Data ready for modeling with {len(selected_features)} features.")

# # --- 4. LEVEL 1 MODELING ---
# print("ğŸ¤– Training Level 1 models...")

# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
# oof_preds = np.zeros((len(X_final), 4)) 
# test_preds = np.zeros((len(X_test_final), 4))

# lgbm_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'objective': 'regression_l1', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1, 'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1}

# xgb_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist', 'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1}

# cat_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'loss_function': 'RMSE', 'random_seed': RANDOM_STATE, 'verbose': 0, 'early_stopping_rounds': 100}

# ridge_params = {'alpha': 20, 'random_state': RANDOM_STATE}

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_final, y)):
#     print(f"===== Fold {fold+1}/{N_SPLITS} =====")
#     X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test_final)

#     # LightGBM
#     model_lgbm = lgb.LGBMRegressor(**lgbm_params)
#     model_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
#     oof_preds[val_idx, 0] = model_lgbm.predict(X_val)
#     test_preds[:, 0] += model_lgbm.predict(X_test_final) / N_SPLITS

#     # XGBoost
#     model_xgb = xgb.XGBRegressor(**xgb_params)
#     model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
#     oof_preds[val_idx, 1] = model_xgb.predict(X_val)
#     test_preds[:, 1] += model_xgb.predict(X_test_final) / N_SPLITS

#     # CatBoost
#     model_cat = cat.CatBoostRegressor(**cat_params)
#     model_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)])
#     oof_preds[val_idx, 2] = model_cat.predict(X_val)
#     test_preds[:, 2] += model_cat.predict(X_test_final) / N_SPLITS

#     # Ridge
#     model_ridge = Ridge(**ridge_params)
#     model_ridge.fit(X_train_scaled, y_train)
#     oof_preds[val_idx, 3] = model_ridge.predict(X_val_scaled)
#     test_preds[:, 3] += model_ridge.predict(X_test_scaled) / N_SPLITS

# print("\n--- Level 1 Model Performance (OOF RMSE) ---")
# print(f"LGBM RMSE:   {np.sqrt(mean_squared_error(y, oof_preds[:, 0])):.5f}")
# print(f"XGBoost RMSE:{np.sqrt(mean_squared_error(y, oof_preds[:, 1])):.5f}")
# print(f"CatBoost RMSE:{np.sqrt(mean_squared_error(y, oof_preds[:, 2])):.5f}")
# print(f"Ridge RMSE:  {np.sqrt(mean_squared_error(y, oof_preds[:, 3])):.5f}")

# # --- 5. LEVEL 2 STACKING ---
# print("\nğŸ§  Training Level 2 Meta-Model...")

# X_meta_train = oof_preds
# y_meta_train = y
# X_meta_test = test_preds

# meta_model = Ridge(alpha=0.5, fit_intercept=True)
# meta_model.fit(X_meta_train, y_meta_train)

# blended_oof_preds = meta_model.predict(X_meta_train)
# final_oof_rmse = np.sqrt(mean_squared_error(y_meta_train, blended_oof_preds))
# print(f"\nFinal Stacked OOF RMSE: {final_oof_rmse:.5f}")

# stacked_test_preds = meta_model.predict(X_meta_test)

# # --- 6. CALIBRATION ---
# print("Calibrating final predictions...")
# ir = IsotonicRegression(y_min=y.min(), y_max=y.max(), out_of_bounds="clip")
# ir.fit(blended_oof_preds, y_meta_train)
# calibrated_preds = ir.transform(stacked_test_preds)

# # --- 7. SUBMISSION ---
# print("\nğŸ“„ Creating final submission file...")
# submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: calibrated_preds})
# submission_df.to_csv('submission_final.csv', index=False)
# print("âœ… Submission file 'submission_final.csv' created successfully!")

# # --- STATS ---
# print("\n--- Prediction Statistics ---")
# print("Target Statistics (Train):")
# print(f"  Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}, Std: {y.std():.2f}")
# print("\nFinal Prediction Statistics (Test):")
# print(f"  Min: {calibrated_preds.min():.2f}, Max: {calibrated_preds.max():.2f}, Mean: {calibrated_preds.mean():.2f}, Std: {calibrated_preds.std():.2f}")



pip install scikit-learn==1.2.2


pip install --upgrade feature-engine scikeras


# --- 0. IMPORTS ---
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import itertools
import warnings
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge  # âœ… CORRECTED: Use public API
from sklearn.isotonic import IsotonicRegression

# Configure warnings
warnings.filterwarnings('ignore')

# --- Constants ---
ID_COL = 'id'
TARGET_COL = 'BeatsPerMinute'
N_SPLITS = 5
RANDOM_STATE = 42
TOP_N_FEATURES = 200

# --- 1. DATA LOADING ---
print("ğŸ�µ Loading Data...")
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
except FileNotFoundError:
    print("Error: Data files not found. Using dummy data for testing...")
    train_df = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
    train_df[TARGET_COL] = np.random.rand(100) * 100 + 50
    test_df = pd.DataFrame(np.random.rand(50, 10), columns=[f'f{i}' for i in range(10)])
    train_df[ID_COL] = range(100)
    test_df[ID_COL] = range(100, 150)

# --- 2. FEATURE ENGINEERING ---
print("ğŸ§® Engineering features by combining strategies...")

def feature_engineer(train_data, test_data):
    combined_df = pd.concat([train_data.drop(TARGET_COL, axis=1, errors='ignore'), test_data], ignore_index=True)

    # Manual interactions
    if all(col in combined_df for col in ['MoodScore', 'Energy']):
        combined_df['mood_energy_interaction'] = combined_df['MoodScore'] * combined_df['Energy']
    if all(col in combined_df for col in ['AudioLoudness', 'VocalContent']):
        combined_df['loudness_vocal_interaction'] = combined_df['AudioLoudness'] * combined_df['VocalContent']
    if all(col in combined_df for col in ['AcousticQuality', 'InstrumentalScore']):
        combined_df['acoustic_instrumental_ratio'] = combined_df['AcousticQuality'] / (combined_df['InstrumentalScore'] + 1e-6)

    # Binning
    for col in ['Energy', 'RhythmScore', 'MoodScore']:
        if col in combined_df:
            combined_df[f"{col}Bin"] = pd.cut(combined_df[col], bins=10, labels=False)

    # Combinatorial features
    combinatorial_features_to_create = []
    base_cols = [c for c in ['RhythmScore', 'Energy', 'MoodScore', 'VocalContent', 'AcousticQuality', 'EnergyBin', 'RhythmBin'] if c in combined_df]

    for combo_size in [2, 3]:
        for combo in itertools.combinations(base_cols, combo_size):
            new_feature_name = '||'.join(combo)
            combinatorial_features_to_create.append(new_feature_name)
            combined_df[new_feature_name] = combined_df[list(combo)].astype(str).agg('_'.join, axis=1)

    return combined_df, combinatorial_features_to_create

# Apply feature engineering
df_processed, combinatorial_features = feature_engineer(train_df, test_df)

# --- 3. PREPROCESSING & FEATURE SELECTION ---
print("ğŸ”’ Encoding features and selecting top performers...")

X = df_processed.iloc[:len(train_df)]
X_test = df_processed.iloc[len(train_df):]
y = train_df[TARGET_COL]

# Target Encoding
encoder = TargetEncoder()
X_encoded = encoder.fit_transform(X[combinatorial_features], y)
X_test_encoded = encoder.transform(X_test[combinatorial_features])
X[combinatorial_features] = X_encoded
X_test[combinatorial_features] = X_test_encoded

original_features = [c for c in test_df.columns if c != ID_COL]
manual_features = [f for f in ['mood_energy_interaction', 'loudness_vocal_interaction', 'acoustic_instrumental_ratio'] if f in df_processed]
all_features = original_features + manual_features + combinatorial_features

X[all_features] = X[all_features].fillna(X[all_features].mean())
X_test[all_features] = X_test[all_features].fillna(X[all_features].mean())

print(f"Selecting top {TOP_N_FEATURES} features...")
lgbm_selector = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
lgbm_selector.fit(X[all_features], y)

importances = pd.Series(lgbm_selector.feature_importances_, index=all_features)
selected_features = importances.sort_values(ascending=False).index[:TOP_N_FEATURES].tolist()

X_final = X[selected_features]
X_test_final = X_test[selected_features]

print(f"Data ready for modeling with {len(selected_features)} features.")

# --- 4. LEVEL 1 MODELING ---
print("ğŸ¤– Training Level 1 models...")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(X_final), 4))
test_preds = np.zeros((len(X_test_final), 4))

lgbm_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'objective': 'regression_l1', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1, 'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
xgb_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist', 'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
cat_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'loss_function': 'RMSE', 'random_seed': RANDOM_STATE, 'verbose': 0, 'early_stopping_rounds': 100}
ridge_params = {'alpha': 20, 'random_state': RANDOM_STATE}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_final, y)):
    print(f"===== Fold {fold+1}/{N_SPLITS} =====")
    X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_final)

    # LightGBM
    model_lgbm = lgb.LGBMRegressor(**lgbm_params)
    model_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds[val_idx, 0] = model_lgbm.predict(X_val)
    test_preds[:, 0] += model_lgbm.predict(X_test_final) / N_SPLITS

    # XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_preds[val_idx, 1] = model_xgb.predict(X_val)
    test_preds[:, 1] += model_xgb.predict(X_test_final) / N_SPLITS

    # CatBoost
    model_cat = cat.CatBoostRegressor(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    oof_preds[val_idx, 2] = model_cat.predict(X_val)
    test_preds[:, 2] += model_cat.predict(X_test_final) / N_SPLITS

    # Ridge
    model_ridge = Ridge(**ridge_params)
    model_ridge.fit(X_train_scaled, y_train)
    oof_preds[val_idx, 3] = model_ridge.predict(X_val_scaled)
    test_preds[:, 3] += model_ridge.predict(X_test_scaled) / N_SPLITS

print("\n--- Level 1 Model Performance (OOF RMSE) ---")
print(f"LGBM RMSE:   {np.sqrt(mean_squared_error(y, oof_preds[:, 0])):.5f}")
print(f"XGBoost RMSE:{np.sqrt(mean_squared_error(y, oof_preds[:, 1])):.5f}")
print(f"CatBoost RMSE:{np.sqrt(mean_squared_error(y, oof_preds[:, 2])):.5f}")
print(f"Ridge RMSE:  {np.sqrt(mean_squared_error(y, oof_preds[:, 3])):.5f}")

# --- 5. LEVEL 2 STACKING ---
print("\nğŸ§  Training Level 2 Meta-Model...")

X_meta_train = oof_preds
y_meta_train = y
X_meta_test = test_preds

meta_model = Ridge(alpha=0.5, fit_intercept=True)
meta_model.fit(X_meta_train, y_meta_train)

blended_oof_preds = meta_model.predict(X_meta_train)
final_oof_rmse = np.sqrt(mean_squared_error(y_meta_train, blended_oof_preds))
print(f"\nFinal Stacked OOF RMSE: {final_oof_rmse:.5f}")

stacked_test_preds = meta_model.predict(X_meta_test)

# --- 6. CALIBRATION ---
print("Calibrating final predictions...")
ir = IsotonicRegression(y_min=y.min(), y_max=y.max(), out_of_bounds="clip")
ir.fit(blended_oof_preds, y_meta_train)
calibrated_preds = ir.transform(stacked_test_preds)

# --- 7. SUBMISSION ---
print("\nğŸ“„ Creating final submission file...")
submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: calibrated_preds})
submission_df.to_csv('submission_final.csv', index=False)
print("âœ… Submission file 'submission_final.csv' created successfully!")

# --- STATS ---
print("\n--- Prediction Statistics ---")
print("Target Statistics (Train):")
print(f"  Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}, Std: {y.std():.2f}")
print("\nFinal Prediction Statistics (Test):")
print(f"  Min: {calibrated_preds.min():.2f}, Max: {calibrated_preds.max():.2f}, Mean: {calibrated_preds.mean():.2f}, Std: {calibrated_preds.std():.2f}")




