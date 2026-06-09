# Clustering-Based Crypto Price Prediction
# Advanced Non-Parametric Approach using Cluster Representatives

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("=== CLUSTERING-BASED CRYPTO PRICE PREDICTION ===")
print("Non-Parametric Approach using Cluster Representatives")
print("="*60)


def load_and_preprocess_for_clustering():
    """Load and preprocess data specifically for clustering approach"""
    print("\n1. Loading and Preprocessing Data for Clustering...")
    
    # Load data
    train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
    test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

    start_date = "2023-12-01 00:00:00"
    end_date = "2024-02-29 23:59:00"

    # Filter the DataFrame and update train_df with the subset
    train_df = train_df.loc[start_date:end_date]
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    for col in train_df.columns:
        if train_df[col].dtype != object:
            if train_df[col].dtype == 'float64':
                train_df[col] = train_df[col].astype(np.float32)
            elif df[col].dtype == 'int64' :
                train_df[col] = train_df[col].astype(np.int32)
                
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col.startswith('X')]
    public_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    all_feature_cols = feature_cols + public_cols
    
    print(f"Total features for clustering: {len(all_feature_cols)}")
    
    # Handle missing values
    train_df[all_feature_cols] = train_df[all_feature_cols].fillna(method='ffill').fillna(0)
    test_df[all_feature_cols] = test_df[all_feature_cols].fillna(method='ffill').fillna(0)
    
    # Remove infinite values
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return train_df, test_df, all_feature_cols


def create_clustering_features(df):
    """Create features specifically useful for clustering"""
    print("Creating clustering-specific features...")
    
    df = df.copy()
    
    # Market microstructure features
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['net_flow'] = df['buy_qty'] - df['sell_qty']
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    
    # Intensity features
    df['buy_intensity'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['sell_intensity'] = df['sell_qty'] / (df['volume'] + 1e-8)
    
    # Volatility proxies
    for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
        df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5).mean()
        df[f'{col}_rolling_std_5'] = df[col].rolling(window=5).std()
        df[f'{col}_rolling_mean_15'] = df[col].rolling(window=15).mean()
        df[f'{col}_z_score'] = (df[col] - df[f'{col}_rolling_mean_15']) / (df[f'{col}_rolling_std_5'] + 1e-8)
    
    # Remove columns with NaN values after feature creation
    df = df.fillna(method='ffill').fillna(0)
    
    return df


class ClusteringPredictor:
    """
    Clustering-based predictor that assigns labels based on cluster representatives
    """
    
    def __init__(self, n_clusters=1000, clustering_method='kmeans', scaler_type='standard'):
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.scaler_type = scaler_type
        self.scaler = None
        self.clusterer = None
        self.cluster_representatives = None
        self.cluster_labels = None
        self.is_fitted = False
        
    def _initialize_scaler(self):
        """Initialize scaler based on type"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard', 'robust', or 'minmax'")
    
    def _initialize_clusterer(self):
        """Initialize clustering algorithm"""
        if self.clustering_method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.clustering_method == 'minibatch_kmeans':
            self.clusterer = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1000)
        else:
            raise ValueError("clustering_method must be 'kmeans' or 'minibatch_kmeans'")
    
    def fit(self, X, y):
        """
        Fit the clustering model
        X: feature matrix (n_samples, n_features)
        y: target values (n_samples,)
        """
        print(f"\nFitting Clustering Predictor:")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Method: {self.clustering_method}")
        
        # Initialize components
        self._initialize_scaler()
        self._initialize_clusterer()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        print("Performing clustering...")
        cluster_assignments = self.clusterer.fit_predict(X_scaled)
        
        # Calculate cluster representatives and their labels
        print("Calculating cluster representatives...")
        self.cluster_representatives = {}
        self.cluster_labels = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            
            if cluster_mask.sum() > 0:  # Check if cluster has any points
                # Representative is the centroid (mean of points in cluster)
                cluster_points = X_scaled[cluster_mask]
                representative = np.mean(cluster_points, axis=0)
                
                # Label is the mean of target values in the cluster
                cluster_target_values = y[cluster_mask]
                representative_label = np.mean(cluster_target_values)
                
                self.cluster_representatives[cluster_id] = representative
                self.cluster_labels[cluster_id] = representative_label
        
        print(f"Created {len(self.cluster_representatives)} non-empty clusters")
        
        # Calculate clustering quality metrics
        if len(self.cluster_representatives) > 1:
            try:
                silhouette_avg = silhouette_score(X_scaled, cluster_assignments)
                calinski_score = calinski_harabasz_score(X_scaled, cluster_assignments)
                print(f"Silhouette Score: {silhouette_avg:.4f}")
                print(f"Calinski-Harabasz Score: {calinski_score:.4f}")
            except:
                print("Could not calculate clustering quality metrics")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using cluster representatives
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        print(f"Making predictions for {X.shape[0]} samples...")
        
        # Scale test features
        X_scaled = self.scaler.transform(X)
        
        # Find closest cluster representative for each test point
        predictions = np.zeros(X.shape[0])
        
        # Convert cluster representatives to array for efficient distance calculation
        if len(self.cluster_representatives) == 0:
            return predictions
        
        cluster_ids = list(self.cluster_representatives.keys())
        representatives_array = np.array([self.cluster_representatives[cid] for cid in cluster_ids])
        
        # Calculate distances and find closest clusters
        distances = cdist(X_scaled, representatives_array, metric='euclidean')
        closest_cluster_indices = np.argmin(distances, axis=1)
        
        # Assign predictions based on closest cluster representatives
        for i, closest_idx in enumerate(closest_cluster_indices):
            closest_cluster_id = cluster_ids[closest_idx]
            predictions[i] = self.cluster_labels[closest_cluster_id]
        
        return predictions
    
    def get_cluster_statistics(self):
        """Get statistics about the clusters"""
        if not self.is_fitted:
            return None
        
        cluster_stats = {
            'n_clusters': len(self.cluster_representatives),
            'label_stats': {
                'mean': np.mean(list(self.cluster_labels.values())),
                'std': np.std(list(self.cluster_labels.values())),
                'min': np.min(list(self.cluster_labels.values())),
                'max': np.max(list(self.cluster_labels.values()))
            }
        }
        
        return cluster_stats


class EnsembleClusteringPredictor:
    """
    Ensemble of multiple clustering approaches
    """
    
    def __init__(self):
        self.predictors = {}
        self.weights = {}
        self.is_fitted = False
    
    def fit(self, X, y, validation_split=0.2):
        """Fit ensemble of clustering predictors"""
        print("\nFitting Ensemble Clustering Predictors...")
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Different clustering configurations
        configs = [
            {'n_clusters': 500, 'method': 'kmeans', 'scaler': 'standard'},
            {'n_clusters': 1000, 'method': 'kmeans', 'scaler': 'standard'},
            {'n_clusters': 1500, 'method': 'kmeans', 'scaler': 'standard'},
            {'n_clusters': 1000, 'method': 'kmeans', 'scaler': 'robust'},
            {'n_clusters': 1000, 'method': 'minibatch_kmeans', 'scaler': 'standard'},
            {'n_clusters': 2000, 'method': 'minibatch_kmeans', 'scaler': 'standard'},
        ]
        
        val_scores = {}
        
        for i, config in enumerate(configs):
            config_name = f"config_{i+1}_{config['n_clusters']}_{config['method']}_{config['scaler']}"
            print(f"\nTraining {config_name}...")
            
            predictor = ClusteringPredictor(
                n_clusters=config['n_clusters'],
                clustering_method=config['method'],
                scaler_type=config['scaler']
            )
            
            try:
                predictor.fit(X_train, y_train)
                val_pred = predictor.predict(X_val)
                
                # Calculate validation correlation
                val_corr = pearsonr(y_val, val_pred)[0]
                if np.isnan(val_corr):
                    val_corr = 0
                
                val_scores[config_name] = max(0, val_corr)  # Only positive correlations
                self.predictors[config_name] = predictor
                
                print(f"Validation correlation: {val_corr:.4f}")
                
            except Exception as e:
                print(f"Failed to train {config_name}: {e}")
                val_scores[config_name] = 0
        
        # Calculate ensemble weights
        total_score = sum(val_scores.values())
        if total_score > 0:
            self.weights = {name: score / total_score for name, score in val_scores.items()}
        else:
            # Equal weights if no positive correlations
            self.weights = {name: 1.0 / len(self.predictors) for name in self.predictors.keys()}
        
        print(f"\nEnsemble weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        print(f"\nMaking ensemble predictions...")
        predictions = np.zeros(X.shape[0])
        
        for name, predictor in self.predictors.items():
            if name in self.weights and self.weights[name] > 0:
                try:
                    pred = predictor.predict(X)
                    predictions += self.weights[name] * pred
                    print(f"Added predictions from {name} (weight: {self.weights[name]:.4f})")
                except Exception as e:
                    print(f"Failed to get predictions from {name}: {e}")
        
        return predictions


def select_features_for_clustering(X, y, n_features=200):
    """Select most informative features for clustering"""
    print(f"\nSelecting top {n_features} features for clustering...")
    
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.ensemble import RandomForestRegressor
    
    # Method 1: Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
    mi_selector.fit(X, y)
    mi_features = mi_selector.get_support()
    
    # Method 2: F-regression
    f_selector = SelectKBest(score_func=f_regression, k=n_features)
    f_selector.fit(X, y)
    f_features = f_selector.get_support()
    
    # Method 3: Random Forest Feature Importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    rf_top_indices = np.argsort(rf_importance)[-n_features:]
    rf_features = np.zeros(X.shape[1], dtype=bool)
    rf_features[rf_top_indices] = True
    
    # Combine features (union of all methods)
    combined_features = mi_features | f_features | rf_features
    
    print(f"Selected {combined_features.sum()} features total")
    print(f"  Mutual Info: {mi_features.sum()}")
    print(f"  F-regression: {f_features.sum()}")
    print(f"  Random Forest: {rf_features.sum()}")
    
    return combined_features


def main_clustering_pipeline():
    """Main clustering-based prediction pipeline"""
    print("Starting Clustering-Based Prediction Pipeline...")
    
    # Load and preprocess data
    train_df, test_df, base_feature_cols = load_and_preprocess_for_clustering()
    
    # Create clustering-specific features
    train_df = create_clustering_features(train_df)
    test_df = create_clustering_features(test_df)
    
    # Get all available features
    all_feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'label']]
    print(f"Total features available: {len(all_feature_cols)}")
    
    # Prepare data
    X = train_df[all_feature_cols].values
    y = train_df['label'].values
    X_test = test_df[all_feature_cols].values
    
    # Feature selection for clustering
    selected_features = select_features_for_clustering(X, y, n_features=50)
    X_selected = X[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    print(f"Using {X_selected.shape[1]} selected features for clustering")
    
    # Use recent data as suggested in competition
    if 'timestamp' in train_df.columns:
        recent_cutoff = train_df.index.max() - pd.DateOffset(months=2)
        recent_mask = train_df.index >= recent_cutoff
        
        X_recent = X_selected[recent_mask]
        y_recent = y[recent_mask]
        
        print(f"Using recent 6 months: {len(X_recent)} samples")
    else:
        # Use all data if no timestamp
        X_recent = X_selected
        y_recent = y
        print(f"Using all available data: {len(X_recent)} samples")
    
    # Train ensemble clustering predictor
    ensemble_predictor = EnsembleClusteringPredictor()
    ensemble_predictor.fit(X_recent, y_recent, validation_split=0.2)
    
    # Make predictions
    predictions = ensemble_predictor.predict(X_test_selected)
    
    # Post-process predictions
    # Remove extreme outliers
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    predictions = np.clip(predictions, 
                         pred_mean - 3*pred_std, 
                         pred_mean + 3*pred_std)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_df['timestamp'] if 'timestamp' in test_df.columns else range(len(test_df)),
        'label': predictions
    })
    
    submission.to_csv('/kaggle/working/clustering_submission.csv', index=False)
    
    print(f"\n" + "="*50)
    print("CLUSTERING PREDICTION RESULTS")
    print("="*50)
    print(f"Submission file: clustering_submission.csv")
    print(f"Predictions statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Median: {np.median(predictions):.6f}")
    
    # Compare with target statistics
    print(f"\nTarget statistics (for reference):")
    print(f"  Mean: {y.mean():.6f}")
    print(f"  Std: {y.std():.6f}")
    print(f"  Min: {y.min():.6f}")
    print(f"  Max: {y.max():.6f}")
    print(f"  Median: {np.median(y):.6f}")
    
    return ensemble_predictor, predictions, submission


def validate_clustering_approach(train_df, all_feature_cols, n_folds=3):
    """Validate clustering approach using time series splits"""
    print(f"\nValidating Clustering Approach with {n_folds} folds...")
    
    X = train_df[all_feature_cols].values
    y = train_df['label'].values
    
    # Feature selection
    selected_features = select_features_for_clustering(X, y, n_features=50)
    X_selected = X[:, selected_features]
    
    # Time series validation
    fold_size = len(X_selected) // (n_folds + 1)
    correlations = []
    
    for fold in range(n_folds):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Create train/val split
        train_end = (fold + 1) * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, len(X_selected))
        
        X_train_fold = X_selected[:train_end]
        y_train_fold = y[:train_end]
        X_val_fold = X_selected[val_start:val_end]
        y_val_fold = y[val_start:val_end]
        
        print(f"Train: {len(X_train_fold)}, Val: {len(X_val_fold)}")
        
        # Train clustering predictor
        predictor = ClusteringPredictor(n_clusters=1000, clustering_method='kmeans')
        predictor.fit(X_train_fold, y_train_fold)
        
        # Predict
        val_pred = predictor.predict(X_val_fold)
        
        # Calculate correlation
        corr = pearsonr(y_val_fold, val_pred)[0]
        if not np.isnan(corr):
            correlations.append(corr)
            print(f"Fold {fold + 1} correlation: {corr:.4f}")
        else:
            print(f"Fold {fold + 1} correlation: NaN (skipped)")
    
    if correlations:
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        print(f"\nValidation Results:")
        print(f"Mean correlation: {mean_corr:.4f} Â± {std_corr:.4f}")
        print(f"Individual fold correlations: {[f'{c:.4f}' for c in correlations]}")
        return mean_corr, std_corr
    else:
        print("No valid correlations calculated")
        return 0, 0


if __name__ == "__main__":
    # Run main pipeline
    print("Executing Clustering-Based Crypto Price Prediction...")
    
    try:
        predictor, predictions, submission = main_clustering_pipeline()
        print("\nâœ… Clustering pipeline completed successfully!")
        print(f"ğŸ“Š Check 'clustering_submission.csv' for results")
        
    except Exception as e:
        print(f"\nâ�Œ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("CLUSTERING APPROACH SUMMARY")
    print("="*60)


