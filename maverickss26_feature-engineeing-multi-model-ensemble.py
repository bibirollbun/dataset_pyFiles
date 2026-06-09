# Kaggle Playground Series S5E7: Introvert vs Extrovert Prediction
# Advanced Solution - Push Beyond 0.975708 to >0.978
# Previous: 0.974898 â†’ 0.975708 | Target: >0.978

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel, VarianceThreshold
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("ðŸš€ ADVANCED High-Performance Introvert/Extrovert Prediction")
print("Previous: 0.975708 | Target: >0.978 | Focus: Advanced Techniques")
print("=" * 70)

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

X = train_df.drop(['id', 'Personality'], axis=1, errors='ignore')
y = train_df['Personality']
X_test = test_df.drop(['id'], axis=1, errors='ignore')

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"Class distribution: {np.bincount(y_encoded)/len(y_encoded)}")

# Ultra-Advanced Data Processing
class UltraDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.numeric_transformers = {}
        
    def detect_feature_types_advanced(self, df):
        """Enhanced feature type detection"""
        numeric_features = []
        binary_features = []
        categorical_features = []
        ordinal_features = []
        
        for col in df.columns:
            unique_vals = df[col].nunique()
            sample_vals = df[col].dropna().head(100)
            
            # Try numeric conversion
            try:
                numeric_converted = pd.to_numeric(df[col], errors='coerce')
                non_null_ratio = numeric_converted.notna().sum() / len(df)
                
                if non_null_ratio > 0.95:  # Mostly numeric
                    if unique_vals == 2:
                        binary_features.append(col)
                    elif unique_vals <= 15 and all(isinstance(x, (int, float)) and x == int(x) for x in sample_vals if pd.notna(x)):
                        ordinal_features.append(col)
                    else:
                        numeric_features.append(col)
                else:
                    if unique_vals <= 25:
                        categorical_features.append(col)
                    else:
                        categorical_features.append(col)
            except:
                if unique_vals == 2:
                    binary_features.append(col)
                else:
                    categorical_features.append(col)
        
        return numeric_features, binary_features, ordinal_features, categorical_features
    
    def advanced_categorical_encoding(self, X_train, X_test, y_train, col, method='comprehensive'):
        """Multiple categorical encoding strategies"""
        X_train_col = X_train[col].astype(str).fillna('MISSING')
        X_test_col = X_test[col].astype(str).fillna('MISSING')
        
        encoded_features = {}
        
        # 1. Label Encoding
        le = LabelEncoder()
        combined = pd.concat([X_train_col, X_test_col])
        le.fit(combined)
        encoded_features[f'{col}_label'] = (
            le.transform(X_train_col),
            le.transform(X_test_col)
        )
        
        # 2. Frequency Encoding
        freq_map = X_train_col.value_counts().to_dict()
        encoded_features[f'{col}_freq'] = (
            X_train_col.map(freq_map).fillna(0).values,
            X_test_col.map(freq_map).fillna(0).values
        )
        
        # 3. Target Encoding with Multiple Smoothing
        overall_mean = y_train.mean()
        
        # Regular target encoding
        target_map = {}
        for val in X_train_col.unique():
            mask = X_train_col == val
            if mask.sum() > 5:  # At least 5 samples
                val_mean = y_train[mask].mean()
                count = mask.sum()
                # Smoothing
                smoothed = (val_mean * count + overall_mean * 10) / (count + 10)
                target_map[val] = smoothed
            else:
                target_map[val] = overall_mean
        
        encoded_features[f'{col}_target'] = (
            X_train_col.map(target_map).fillna(overall_mean).values,
            X_test_col.map(target_map).fillna(overall_mean).values
        )
        
        # 4. Likelihood Encoding
        pos_rates = {}
        neg_rates = {}
        for val in X_train_col.unique():
            mask = X_train_col == val
            if mask.sum() > 3:
                pos_rate = y_train[mask].sum() / mask.sum()
                neg_rate = 1 - pos_rate
                pos_rates[val] = pos_rate
                neg_rates[val] = neg_rate
            else:
                pos_rates[val] = overall_mean
                neg_rates[val] = 1 - overall_mean
        
        encoded_features[f'{col}_pos_rate'] = (
            X_train_col.map(pos_rates).fillna(overall_mean).values,
            X_test_col.map(pos_rates).fillna(overall_mean).values
        )
        
        # 5. Rank Encoding
        rank_map = X_train_col.value_counts().rank(method='dense').to_dict()
        encoded_features[f'{col}_rank'] = (
            X_train_col.map(rank_map).fillna(0).values,
            X_test_col.map(rank_map).fillna(0).values
        )
        
        return encoded_features
    
    def process_numeric_features(self, X_train, X_test, numeric_cols):
        """Advanced numeric feature processing"""
        processed_train = {}
        processed_test = {}
        
        for col in numeric_cols:
            if col not in X_train.columns:
                continue
                
            train_vals = pd.to_numeric(X_train[col], errors='coerce')
            test_vals = pd.to_numeric(X_test[col], errors='coerce')
            
            # Original values
            processed_train[col] = train_vals.values
            processed_test[col] = test_vals.values
            
            # Log transform (if positive)
            if train_vals.min() > 0:
                processed_train[f'{col}_log'] = np.log1p(train_vals).values
                processed_test[f'{col}_log'] = np.log1p(test_vals).values
            
            # Square root transform
            if train_vals.min() >= 0:
                processed_train[f'{col}_sqrt'] = np.sqrt(train_vals.fillna(0)).values
                processed_test[f'{col}_sqrt'] = np.sqrt(test_vals.fillna(0)).values
            
            # Quantile transform
            try:
                from sklearn.preprocessing import QuantileTransformer
                qt = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
                qt_train = qt.fit_transform(train_vals.fillna(train_vals.median()).values.reshape(-1, 1)).flatten()
                qt_test = qt.transform(test_vals.fillna(train_vals.median()).values.reshape(-1, 1)).flatten()
                processed_train[f'{col}_quantile'] = qt_train
                processed_test[f'{col}_quantile'] = qt_test
            except:
                pass
        
        return processed_train, processed_test
    
    def fit_transform(self, X_train, X_test, y_train):
        print("ðŸ”§ Ultra-advanced data processing...")
        
        # Detect feature types
        numeric_features, binary_features, ordinal_features, categorical_features = self.detect_feature_types_advanced(X_train)
        
        print(f"Feature distribution:")
        print(f"  Numeric: {len(numeric_features)}")
        print(f"  Binary: {len(binary_features)}")
        print(f"  Ordinal: {len(ordinal_features)}")
        print(f"  Categorical: {len(categorical_features)}")
        
        # Process numeric features
        numeric_processed_train, numeric_processed_test = self.process_numeric_features(
            X_train, X_test, numeric_features + ordinal_features
        )
        
        # Process categorical features
        categorical_processed_train = {}
        categorical_processed_test = {}
        
        for col in categorical_features + binary_features:
            if col in X_train.columns:
                encoded_features = self.advanced_categorical_encoding(X_train, X_test, y_train, col)
                for feature_name, (train_vals, test_vals) in encoded_features.items():
                    categorical_processed_train[feature_name] = train_vals
                    categorical_processed_test[feature_name] = test_vals
        
        # Combine all features
        all_train_features = {**numeric_processed_train, **categorical_processed_train}
        all_test_features = {**numeric_processed_test, **categorical_processed_test}
        
        # Convert to arrays
        feature_names = list(all_train_features.keys())
        X_train_processed = np.column_stack([all_train_features[name] for name in feature_names])
        X_test_processed = np.column_stack([all_test_features[name] for name in feature_names])
        
        # Handle missing values with advanced imputation
        imputer = KNNImputer(n_neighbors=7, weights='distance')
        X_train_processed = imputer.fit_transform(X_train_processed)
        X_test_processed = imputer.transform(X_test_processed)
        
        print(f"Processed features: {X_train_processed.shape[1]}")
        return X_train_processed, X_test_processed

# Advanced Feature Engineering
class AdvancedFeatureEngineer:
    def __init__(self):
        self.pca = None
        self.ica = None
        self.kmeans = None
        
    def create_personality_features_v2(self, X):
        """Enhanced personality-specific features"""
        if X.shape[1] == 0:
            return np.array([]).reshape(X.shape[0], 0)
        
        features = []
        
        # Response pattern analysis
        features.append(np.std(X, axis=1))  # Response variability
        features.append(np.mean(X, axis=1))  # Average response
        features.append(np.median(X, axis=1))  # Median response
        features.append(np.max(X, axis=1) - np.min(X, axis=1))  # Response range
        
        # Extreme response tendency
        q25 = np.percentile(X, 25, axis=1)
        q75 = np.percentile(X, 75, axis=1)
        extreme_low = np.sum(X <= q25[:, np.newaxis], axis=1)
        extreme_high = np.sum(X >= q75[:, np.newaxis], axis=1)
        features.extend([extreme_low, extreme_high])
        
        # Response consistency across segments
        n_segments = 4
        segment_size = X.shape[1] // n_segments
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else X.shape[1]
            segment_mean = np.mean(X[:, start_idx:end_idx], axis=1)
            features.append(segment_mean)
        
        # Statistical moments
        try:
            features.append(stats.skew(X, axis=1))
            features.append(stats.kurtosis(X, axis=1))
        except:
            features.extend([np.zeros(X.shape[0]), np.zeros(X.shape[0])])
        
        # Response entropy (measure of randomness)
        def row_entropy(row):
            _, counts = np.unique(row, return_counts=True)
            probs = counts / len(row)
            return -np.sum(probs * np.log2(probs + 1e-10))
        
        entropy_vals = np.array([row_entropy(X[i]) for i in range(X.shape[0])])
        features.append(entropy_vals)
        
        return np.column_stack(features)
    
    def create_advanced_interactions(self, X, max_features=100):
        """Create sophisticated feature interactions"""
        if X.shape[1] < 2:
            return np.array([]).reshape(X.shape[0], 0)
        
        # Select most important features for interactions
        feature_importance = np.var(X, axis=0)
        top_indices = np.argsort(feature_importance)[-min(15, X.shape[1]):]
        
        interactions = []
        count = 0
        
        # Pairwise interactions
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                if count >= max_features:
                    break
                
                idx1, idx2 = top_indices[i], top_indices[j]
                
                # Multiplicative
                interactions.append(X[:, idx1] * X[:, idx2])
                count += 1
                
                # Additive
                if count < max_features:
                    interactions.append(X[:, idx1] + X[:, idx2])
                    count += 1
                
                # Ratio (safe division)
                if count < max_features and np.all(X[:, idx2] != 0):
                    interactions.append(X[:, idx1] / (X[:, idx2] + 1e-8))
                    count += 1
                
                # Difference
                if count < max_features:
                    interactions.append(X[:, idx1] - X[:, idx2])
                    count += 1
                
                if count >= max_features:
                    break
            if count >= max_features:
                break
        
        return np.column_stack(interactions) if interactions else np.array([]).reshape(X.shape[0], 0)
    
    def create_clustering_features(self, X_train, X_test, n_clusters_list=[3, 5, 8, 12]):
        """Advanced clustering features"""
        if X_train.shape[1] == 0:
            return np.array([]).reshape(X_train.shape[0], 0), np.array([]).reshape(X_test.shape[0], 0)
        
        cluster_features_train = []
        cluster_features_test = []
        
        for n_clusters in n_clusters_list:
            if n_clusters <= X_train.shape[0]:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
                    
                    # Fit on training data
                    train_clusters = kmeans.fit_predict(X_train)
                    test_clusters = kmeans.predict(X_test)
                    
                    cluster_features_train.append(train_clusters)
                    cluster_features_test.append(test_clusters)
                    
                    # Distance to centroids
                    train_distances = kmeans.transform(X_train)
                    test_distances = kmeans.transform(X_test)
                    
                    # Min, max, mean distances
                    cluster_features_train.append(np.min(train_distances, axis=1))
                    cluster_features_train.append(np.max(train_distances, axis=1))
                    cluster_features_train.append(np.mean(train_distances, axis=1))
                    
                    cluster_features_test.append(np.min(test_distances, axis=1))
                    cluster_features_test.append(np.max(test_distances, axis=1))
                    cluster_features_test.append(np.mean(test_distances, axis=1))
                    
                except:
                    continue
        
        if cluster_features_train:
            return np.column_stack(cluster_features_train), np.column_stack(cluster_features_test)
        else:
            return np.array([]).reshape(X_train.shape[0], 0), np.array([]).reshape(X_test.shape[0], 0)
    
    def create_decomposition_features(self, X_train, X_test):
        """Multiple decomposition techniques"""
        decomp_train = []
        decomp_test = []
        
        # PCA
        try:
            n_components = min(12, X_train.shape[1], X_train.shape[0])
            if n_components >= 2:
                self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
                pca_train = self.pca.fit_transform(X_train)
                pca_test = self.pca.transform(X_test)
                decomp_train.append(pca_train)
                decomp_test.append(pca_test)
        except:
            pass
        
        # ICA
        try:
            n_components = min(8, X_train.shape[1], X_train.shape[0])
            if n_components >= 2:
                self.ica = FastICA(n_components=n_components, random_state=RANDOM_STATE, max_iter=500)
                ica_train = self.ica.fit_transform(X_train)
                ica_test = self.ica.transform(X_test)
                decomp_train.append(ica_train)
                decomp_test.append(ica_test)
        except:
            pass
        
        # SVD
        try:
            n_components = min(10, X_train.shape[1] - 1, X_train.shape[0])
            if n_components >= 1:
                svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
                svd_train = svd.fit_transform(X_train)
                svd_test = svd.transform(X_test)
                decomp_train.append(svd_train)
                decomp_test.append(svd_test)
        except:
            pass
        
        if decomp_train:
            return np.hstack(decomp_train), np.hstack(decomp_test)
        else:
            return np.array([]).reshape(X_train.shape[0], 0), np.array([]).reshape(X_test.shape[0], 0)
    
    def fit_transform(self, X_train, X_test):
        print("ðŸ§  Advanced feature engineering...")
        
        # Personality features
        personality_train = self.create_personality_features_v2(X_train)
        personality_test = self.create_personality_features_v2(X_test)
        
        # Advanced interactions
        interaction_train = self.create_advanced_interactions(X_train)
        interaction_test = self.create_advanced_interactions(X_test)
        
        # Clustering features
        cluster_train, cluster_test = self.create_clustering_features(X_train, X_test)
        
        # Decomposition features
        decomp_train, decomp_test = self.create_decomposition_features(X_train, X_test)
        
        # Combine all features
        all_features_train = [X_train]
        all_features_test = [X_test]
        
        if personality_train.shape[1] > 0:
            all_features_train.append(personality_train)
            all_features_test.append(personality_test)
        
        if interaction_train.shape[1] > 0:
            all_features_train.append(interaction_train)
            all_features_test.append(interaction_test)
        
        if cluster_train.shape[1] > 0:
            all_features_train.append(cluster_train)
            all_features_test.append(cluster_test)
        
        if decomp_train.shape[1] > 0:
            all_features_train.append(decomp_train)
            all_features_test.append(decomp_test)
        
        X_train_enhanced = np.hstack(all_features_train)
        X_test_enhanced = np.hstack(all_features_test)
        
        print(f"Enhanced features: {X_train_enhanced.shape[1]} (from {X_train.shape[1]})")
        return X_train_enhanced, X_test_enhanced

# Advanced Multi-Level Ensemble
class AdvancedEnsemble:
    def __init__(self):
        self.level1_models = {}
        self.level2_model = None
        self.feature_selectors = {}
        
    def create_diverse_models(self):
        """Create highly diverse model collection"""
        
        # Gradient boosting variants
        self.level1_models['xgb1'] = xgb.XGBClassifier(
            n_estimators=600, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, eval_metric='logloss'
        )
        
        self.level1_models['xgb2'] = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=0.2,
            random_state=RANDOM_STATE+1, eval_metric='logloss'
        )
        
        self.level1_models['lgb1'] = lgb.LGBMClassifier(
            n_estimators=600, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, objective='binary', verbosity=-1
        )
        
        self.level1_models['lgb2'] = lgb.LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=0.2,
            random_state=RANDOM_STATE+2, objective='binary', verbosity=-1
        )
        
        self.level1_models['cat'] = CatBoostClassifier(
            iterations=500, depth=7, learning_rate=0.05,
            l2_leaf_reg=3, random_seed=RANDOM_STATE, verbose=False
        )
        
        # Tree-based ensembles
        self.level1_models['rf1'] = RandomForestClassifier(
            n_estimators=400, max_depth=20, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', random_state=RANDOM_STATE, class_weight='balanced'
        )
        
        self.level1_models['rf2'] = RandomForestClassifier(
            n_estimators=300, max_depth=25, min_samples_split=3, min_samples_leaf=1,
            max_features='log2', random_state=RANDOM_STATE+3, class_weight='balanced'
        )
        
        self.level1_models['et'] = ExtraTreesClassifier(
            n_estimators=400, max_depth=25, min_samples_split=4, min_samples_leaf=1,
            max_features='sqrt', random_state=RANDOM_STATE, class_weight='balanced'
        )
        
        self.level1_models['gb'] = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.8, random_state=RANDOM_STATE
        )
        
        # Linear models
        self.level1_models['lr1'] = LogisticRegression(
            C=0.1, penalty='elasticnet', l1_ratio=0.5, solver='saga',
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        )
        
        self.level1_models['lr2'] = LogisticRegression(
            C=1.0, penalty='l2', solver='lbfgs',
            max_iter=1000, random_state=RANDOM_STATE+4, class_weight='balanced'
        )
        
        # Other models
        self.level1_models['nb'] = GaussianNB()
        self.level1_models['knn'] = KNeighborsClassifier(n_neighbors=7, weights='distance')
        self.level1_models['lda'] = LinearDiscriminantAnalysis()
        
        # Neural network
        self.level1_models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50), activation='relu', solver='adam',
            alpha=0.001, learning_rate='adaptive', max_iter=300,
            random_state=RANDOM_STATE
        )
    
    def train_stacked_ensemble(self, X, y):
        """Advanced stacked ensemble with feature selection per model"""
        print("ðŸŽ¯ Training advanced stacked ensemble...")
        
        # Create different feature subsets for model diversity
        n_features = X.shape[1]
        
        # Different feature selection strategies
        self.feature_selectors['all'] = slice(None)  # All features
        
        # Top statistical features
        selector_stat = SelectKBest(score_func=f_classif, k=min(300, n_features))
        selector_stat.fit(X, y)
        self.feature_selectors['statistical'] = selector_stat.get_support()
        
        # Top mutual information features
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(250, n_features))
        selector_mi.fit(X, y)
        self.feature_selectors['mutual_info'] = selector_mi.get_support()
        
        # Random subsets for diversity
        np.random.seed(RANDOM_STATE)
        self.feature_selectors['random1'] = np.random.choice(n_features, min(200, n_features), replace=False)
        self.feature_selectors['random2'] = np.random.choice(n_features, min(180, n_features), replace=False)
        
        # Assign feature selectors to models
        model_features = {
            'xgb1': 'all', 'xgb2': 'statistical', 'lgb1': 'all', 'lgb2': 'mutual_info',
            'cat': 'statistical', 'rf1': 'all', 'rf2': 'random1', 'et': 'random2',
            'gb': 'mutual_info', 'lr1': 'statistical', 'lr2': 'mutual_info',
            'nb': 'statistical', 'knn': 'random1', 'lda': 'mutual_info', 'mlp': 'all'
        }
        
        # Cross-validation for level 1
        kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE)
        level1_predictions = np.zeros((X.shape[0], len(self.level1_models)))
        model_scores = {}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"  Fold {fold + 1}/7")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            for i, (name, model) in enumerate(self.level1_models.items()):
                # Select features for this model
                feature_selector = model_features[name]
                if feature_selector == 'all':
                    X_train_model = X_train_fold
                    X_val_model = X_val_fold
                else:
                    selector = self.feature_selectors[feature_selector]
                    X_train_model = X_train_fold[:, selector]
                    X_val_model = X_val_fold[:, selector]
                
                # Train model
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_train_model, y_train_fold)
                
                # Predict
                if hasattr(fold_model, 'predict_proba'):
                    val_pred = fold_model.predict_proba(X_val_model)[:, 1]
                else:
                    val_pred = fold_model.decision_function(X_val_model)
                    val_pred = 1 / (1 + np.exp(-val_pred))  # Sigmoid
                
                level1_predictions[val_idx, i] = val_pred
                
                # Track performance
                val_score = accuracy_score(y_val_fold, (val_pred > 0.5).astype(int))
                if name not in model_scores:
                    model_scores[name] = []
                model_scores[name].append(val_score)
        
        # Print model performances
        print("\nLevel 1 Model Performances:")
        avg_scores = {}
        for name, scores in model_scores.items():
            avg_score = np.mean(scores)
            avg_scores[name] = avg_score
            print(f"  {name:>8}: {avg_score:.6f} (+/- {np.std(scores)*2:.6f})")
        
        # Train level 1 models on full data
        print("\nTraining level 1 models on full data...")
        for name, model in self.level1_models.items():
            feature_selector = model_features[name]
            if feature_selector == 'all':
                X_model = X
            else:
                selector = self.feature_selectors[feature_selector]
                X_model = X[:, selector]
            
            model.fit(X_model, y)
        
        # Train level 2 meta-learner
        print("Training level 2 meta-learner...")
        
        # Try multiple meta-learners and pick the best
        meta_models = {
            'lr': LogisticRegression(C=1.0, random_state=RANDOM_STATE),
            'xgb': xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
        }
        
        best_meta_score = 0
        best_meta_name = 'lr'
        
        for meta_name, meta_model in meta_models.items():
            meta_cv = cross_val_score(meta_model, level1_predictions, y, cv=5, scoring='accuracy')
            meta_score = meta_cv.mean()
            print(f"  Meta-{meta_name}: {meta_score:.6f}")
            
            if meta_score > best_meta_score:
                best_meta_score = meta_score
                best_meta_name = meta_name
        
        print(f"Selected meta-learner: {best_meta_name} (score: {best_meta_score:.6f})")
        
        self.level2_model = meta_models[best_meta_name]
        self.level2_model.fit(level1_predictions, y)
        self.model_features = model_features
        
        return avg_scores, best_meta_score
    
    def predict_proba(self, X):
        """Predict using stacked ensemble"""
        # Get level 1 predictions
        level1_predictions = np.zeros((X.shape[0], len(self.level1_models)))
        
        for i, (name, model) in enumerate(self.level1_models.items()):
            feature_selector = self.model_features[name]
            if feature_selector == 'all':
                X_model = X
            else:
                selector = self.feature_selectors[feature_selector]
                X_model = X[:, selector]
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_model)[:, 1]
            else:
                pred = model.decision_function(X_model)
                pred = 1 / (1 + np.exp(-pred))  # Sigmoid
            
            level1_predictions[:, i] = pred
        
        # Level 2 prediction
        final_proba = self.level2_model.predict_proba(level1_predictions)
        return final_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

# Main Pipeline
print("\nðŸš€ ADVANCED PIPELINE EXECUTION")
print("="*70)

# Step 1: Ultra data processing
processor = UltraDataProcessor()
X_processed, X_test_processed = processor.fit_transform(X, X_test, y_encoded)

# Step 2: Advanced feature engineering
feature_engineer = AdvancedFeatureEngineer()
X_enhanced, X_test_enhanced = feature_engineer.fit_transform(X_processed, X_test_processed)

# Step 3: Advanced preprocessing and scaling
print("ðŸ”§ Advanced preprocessing...")

# Remove zero variance features
variance_selector = VarianceThreshold(threshold=0.001)
X_variance = variance_selector.fit_transform(X_enhanced)
X_test_variance = variance_selector.transform(X_test_enhanced)

# Multiple scaling strategies combined
scaler1 = RobustScaler()
X_scaled1 = scaler1.fit_transform(X_variance)
X_test_scaled1 = scaler1.transform(X_test_variance)

scaler2 = PowerTransformer(method='yeo-johnson', standardize=True)
X_scaled2 = scaler2.fit_transform(X_variance)
X_test_scaled2 = scaler2.transform(X_test_variance)

# Combine scaled versions
X_final = np.hstack([X_scaled1, X_scaled2])
X_test_final = np.hstack([X_test_scaled1, X_test_scaled2])

print(f"Final feature count: {X_final.shape[1]}")

# Step 4: Advanced ensemble training
ensemble = AdvancedEnsemble()
ensemble.create_diverse_models()
model_scores, meta_score = ensemble.train_stacked_ensemble(X_final, y_encoded)

# Step 5: Alternative approaches for comparison
print("\nðŸ”„ Training alternative approaches...")

# Voting ensemble
voting_models = [
    ('xgb', ensemble.level1_models['xgb1']),
    ('lgb', ensemble.level1_models['lgb1']),
    ('cat', ensemble.level1_models['cat']),
    ('rf', ensemble.level1_models['rf1']),
    ('et', ensemble.level1_models['et'])
]

voting_ensemble = VotingClassifier(estimators=voting_models, voting='soft')
voting_cv = cross_val_score(voting_ensemble, X_final, y_encoded, cv=5, scoring='accuracy')
voting_score = voting_cv.mean()

# Bagging ensemble of best models
best_model_name = max(model_scores, key=model_scores.get)
best_model = ensemble.level1_models[best_model_name]

bagging_ensemble = BaggingClassifier(
    base_estimator=best_model.__class__(**best_model.get_params()),
    n_estimators=10,
    random_state=RANDOM_STATE
)
bagging_cv = cross_val_score(bagging_ensemble, X_final, y_encoded, cv=5, scoring='accuracy')
bagging_score = bagging_cv.mean()

print(f"Stacked ensemble CV: {meta_score:.6f}")
print(f"Voting ensemble CV: {voting_score:.6f}")
print(f"Bagging ensemble CV: {bagging_score:.6f}")

# Step 6: Model selection and prediction
approaches = {
    'stacked': (ensemble, meta_score),
    'voting': (voting_ensemble, voting_score),
    'bagging': (bagging_ensemble, bagging_score)
}

best_approach_name = max(approaches, key=lambda x: approaches[x][1])
best_approach, best_score = approaches[best_approach_name]

print(f"\nSelected approach: {best_approach_name} (CV: {best_score:.6f})")

# Train best approach on full data if needed
if best_approach_name != 'stacked':
    best_approach.fit(X_final, y_encoded)

# Make final predictions
print("\nðŸ”® Making final predictions...")
if best_approach_name == 'stacked':
    final_pred_proba = ensemble.predict_proba(X_test_final)
    final_pred = (final_pred_proba[:, 1] > 0.5).astype(int)
else:
    final_pred_proba = best_approach.predict_proba(X_test_final)
    final_pred = (final_pred_proba[:, 1] > 0.5).astype(int)

# Apply threshold optimization
print("ðŸŽ¯ Optimizing prediction threshold...")

# Use validation set to find optimal threshold
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
    X_final, y_encoded, test_size=0.15, stratify=y_encoded, random_state=RANDOM_STATE
)

if best_approach_name == 'stacked':
    # Retrain on validation set
    ensemble_val = AdvancedEnsemble()
    ensemble_val.create_diverse_models()
    ensemble_val.train_stacked_ensemble(X_train_val, y_train_val)
    holdout_proba = ensemble_val.predict_proba(X_holdout)[:, 1]
else:
    best_approach.fit(X_train_val, y_train_val)
    holdout_proba = best_approach.predict_proba(X_holdout)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.3, 0.8, 0.01)
best_threshold = 0.5
best_threshold_score = 0

for threshold in thresholds:
    threshold_pred = (holdout_proba >= threshold).astype(int)
    threshold_score = accuracy_score(y_holdout, threshold_pred)
    if threshold_score > best_threshold_score:
        best_threshold_score = threshold_score
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.3f} (validation accuracy: {best_threshold_score:.6f})")

# Apply optimal threshold to final predictions
final_pred_optimized = (final_pred_proba[:, 1] >= best_threshold).astype(int)

# Convert back to original labels
final_pred_labels = target_encoder.inverse_transform(final_pred_optimized)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_pred_labels
})

submission.to_csv('submission.csv', index=False)

# Results summary
print("\n" + "="*70)
print("ðŸŽ¯ ADVANCED SOLUTION COMPLETE!")
print("="*70)
print("ðŸš€ ADVANCED TECHNIQUES APPLIED:")
print("âœ… Ultra-advanced categorical encoding (5 methods per feature)")
print("âœ… Numeric feature transformations (log, sqrt, quantile)")
print("âœ… Enhanced personality response patterns")
print("âœ… Sophisticated feature interactions (4 types)")
print("âœ… Multiple clustering approaches")
print("âœ… Triple decomposition (PCA + ICA + SVD)")
print("âœ… 15-model stacked ensemble with feature diversity")
print("âœ… Multiple meta-learners with selection")
print("âœ… Combined scaling strategies")
print("âœ… Threshold optimization")
print("âœ… Multiple ensemble approaches comparison")

print(f"\nðŸŽ¯ PERFORMANCE:")
print(f"Best approach: {best_approach_name}")
print(f"Expected accuracy: {best_score:.6f}")
print(f"Threshold optimized: {best_threshold:.3f}")
print(f"Previous: 0.975708 | Target: >0.978")

if best_score > 0.978:
    print("ðŸŽ‰ TARGET ACHIEVED!")
elif best_score > 0.977:
    print("ðŸ”¥ VERY CLOSE TO TARGET!")
elif best_score > 0.976:
    print("ðŸ“ˆ SIGNIFICANT IMPROVEMENT!")
else:
    print("âš¡ PERFORMANCE BOOST EXPECTED")

print(f"\nðŸ’¾ Submission: {submission.shape}")
print(f"Distribution:\n{submission['Personality'].value_counts(normalize=True)}")
print("="*70)




