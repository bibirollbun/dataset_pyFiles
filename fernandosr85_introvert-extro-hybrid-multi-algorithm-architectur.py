# =============================================================================
# CORE DATA PROCESSING & SYSTEM UTILITIES
# =============================================================================
import pandas as pd
import numpy as np
import warnings
import re
import io
import sys
import glob
from datetime import datetime
from contextlib import redirect_stdout

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# SCIKIT-LEARN - MODEL SELECTION & VALIDATION
# =============================================================================
from sklearn.model_selection import (
    StratifiedKFold, 
    cross_val_score, 
    cross_val_predict
)

# =============================================================================
# SCIKIT-LEARN - PREPROCESSING & TRANSFORMATION
# =============================================================================
from sklearn.preprocessing import (
    StandardScaler, 
    PowerTransformer,
    LabelEncoder
)

# =============================================================================
# SCIKIT-LEARN - CLASSIFICATION MODELS
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier

# =============================================================================
# SCIKIT-LEARN - SPECIALIZED TECHNIQUES
# =============================================================================
# Anomaly Detection & Clustering
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Model Calibration
from sklearn.calibration import CalibratedClassifierCV

# =============================================================================
# SCIKIT-LEARN - METRICS & EVALUATION
# =============================================================================
from sklearn.metrics import (
    classification_report, 
    roc_auc_score,
    accuracy_score
)

# =============================================================================
# ADVANCED MACHINE LEARNING LIBRARIES
# =============================================================================
import xgboost as xgb
import optuna

# =============================================================================
# VISUALIZATION LIBRARIES
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
plt.style.use('default')  
sns.set_palette("husl")


# ===== 1. DATA QUALITY ANALYSIS =====

class DataQualityAnalyzer:
    """Comprehensive data quality analysis and enhancement"""
    
    def __init__(self):
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def analyze_data_quality(self, df, target_col='Personality'):
        """Comprehensive data quality analysis"""
        
        print("\nğŸ”� ANALYZING DATA QUALITY...")
        
        results = {}
        
        # 1. Missing data analysis
        missing_analysis = df.isnull().sum()
        results['missing'] = missing_analysis
        print(f"ğŸ“Š Missing data: {missing_analysis.sum()} total missing values")
        
        # 2. Duplicate analysis
        duplicates = df.duplicated().sum()
        results['duplicates'] = duplicates
        print(f"ğŸ”„ Duplicates: {duplicates} duplicate rows")
        
        # 3. Outlier detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outliers = self.detect_outliers(df[numeric_cols])
            results['outliers'] = outliers
            print(f"âš ï¸� Outliers: {outliers.sum()} outlier samples detected")
        
        # 4. Class distribution analysis
        if target_col in df.columns:
            class_dist = df[target_col].value_counts()
            results['class_distribution'] = class_dist
            print(f"ğŸ�¯ Class distribution: {class_dist.to_dict()}")
            
            # Imbalance ratio
            imbalance_ratio = class_dist.max() / class_dist.min()
            results['imbalance_ratio'] = imbalance_ratio
            print(f"âš–ï¸� Imbalance ratio: {imbalance_ratio:.2f}")
        
        # 5. Feature correlation analysis
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = self.find_high_correlations(corr_matrix, threshold=0.8)
        results['high_correlations'] = high_corr_pairs
        print(f"ğŸ”— High correlations: {len(high_corr_pairs)} feature pairs with |corr| > 0.8")
        
        # 6. Feature variance analysis
        low_variance_features = self.find_low_variance_features(df[numeric_cols])
        results['low_variance'] = low_variance_features
        print(f"ğŸ“‰ Low variance features: {len(low_variance_features)} features")
        
        return results
    
    def detect_outliers(self, X):
        """Detect outliers using Isolation Forest"""
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        outlier_labels = self.outlier_detector.fit_predict(X_scaled)
        return outlier_labels == -1
    
    def find_high_correlations(self, corr_matrix, threshold=0.8):
        """Find highly correlated feature pairs"""
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        return high_corr_pairs
    
    def find_low_variance_features(self, X, threshold=0.01):
        """Find features with low variance"""
        variances = X.var()
        return variances[variances < threshold].index.tolist()

# ===== 2. DATA ENHANCEMENT STRATEGIES =====

class DataEnhancer:
    """Advanced data enhancement techniques"""
    
    def __init__(self):
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        
    def enhance_data_quality(self, train_df, test_df, target_col='Personality'):
        """Apply comprehensive data enhancement"""
        
        print("\nğŸ”§ ENHANCING DATA QUALITY...")
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col not in [target_col, 'id']]
        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        y_train = train_df[target_col] if target_col in train_df.columns else None
        
        # 1. Handle categorical variables more intelligently
        X_train_enhanced, X_test_enhanced = self.enhance_categorical_features(X_train, X_test, y_train)
        
        # 2. Outlier treatment
        X_train_enhanced, outlier_mask = self.treat_outliers(X_train_enhanced)
        print(f"   ğŸ”§ Outlier treatment: {outlier_mask.sum()} samples modified")
        
        # 3. Missing value imputation (advanced)
        X_train_enhanced, X_test_enhanced = self.advanced_imputation(X_train_enhanced, X_test_enhanced)
        
        # 4. Feature transformation
        X_train_enhanced, X_test_enhanced = self.transform_features(X_train_enhanced, X_test_enhanced)
        
        # 5. Remove redundant samples
        if y_train is not None:
            X_train_enhanced, y_train_enhanced, removed_samples = self.remove_redundant_samples(
                X_train_enhanced, y_train
            )
            print(f"   ğŸ”„ Removed samples: {removed_samples} redundant/conflicting samples")
        else:
            y_train_enhanced = None
        
        # 6. Feature scaling and normalization
        X_train_final, X_test_final = self.advanced_scaling(X_train_enhanced, X_test_enhanced)
        
        print(f"   âœ… Enhanced data shape: Train {X_train_final.shape}, Test {X_test_final.shape}")
        
        return X_train_final, X_test_final, y_train_enhanced
    
    def enhance_categorical_features(self, X_train, X_test, y_train):
        """Intelligent categorical feature enhancement"""
        
        print("   ğŸ”¤ Enhancing categorical features...")
        
        X_train_enh = X_train.copy()
        X_test_enh = X_test.copy()
        
        # Convert target to numeric first
        if y_train is not None:
            if hasattr(y_train, 'map'):
                y_numeric = y_train.map({'Introvert': 0, 'Extrovert': 1})
            else:
                y_numeric = y_train
        else:
            y_numeric = None
        
        # Advanced mapping for Stage_fear based on target correlation
        if 'Stage_fear' in X_train.columns and y_numeric is not None:
            # Calculate optimal mapping based on target correlation
            stage_fear_target_corr = {}
            for value in X_train['Stage_fear'].dropna().unique():
                mask = X_train['Stage_fear'] == value
                if mask.sum() > 10:  # Minimum samples
                    target_mean = y_numeric[mask].mean()
                    stage_fear_target_corr[value] = target_mean
            
            # Create optimized mapping
            if 'Yes' in stage_fear_target_corr and 'No' in stage_fear_target_corr:
                yes_corr = stage_fear_target_corr['Yes']
                no_corr = stage_fear_target_corr['No']
                
                # Map based on actual correlation with target
                optimal_mapping = {'No': 0, 'Yes': int(yes_corr * 10)}
                X_train_enh['Stage_fear_optimized'] = X_train['Stage_fear'].map(optimal_mapping).fillna(5)
                X_test_enh['Stage_fear_optimized'] = X_test['Stage_fear'].map(optimal_mapping).fillna(5)
                print(f"     Stage_fear optimal mapping: {optimal_mapping}")
        
        # Same for Drained_after_socializing
        if 'Drained_after_socializing' in X_train.columns and y_numeric is not None:
            drained_target_corr = {}
            for value in X_train['Drained_after_socializing'].dropna().unique():
                mask = X_train['Drained_after_socializing'] == value
                if mask.sum() > 10:
                    target_mean = y_numeric[mask].mean()
                    drained_target_corr[value] = target_mean
            
            if 'Yes' in drained_target_corr and 'No' in drained_target_corr:
                yes_corr = drained_target_corr['Yes']
                no_corr = drained_target_corr['No']
                optimal_mapping = {'No': 0, 'Yes': int(yes_corr * 10)}
                X_train_enh['Drained_optimized'] = X_train['Drained_after_socializing'].map(optimal_mapping).fillna(5)
                X_test_enh['Drained_optimized'] = X_test['Drained_after_socializing'].map(optimal_mapping).fillna(5)
                print(f"     Drained optimal mapping: {optimal_mapping}")
        
        return X_train_enh, X_test_enh
    
    def treat_outliers(self, X, method='clip', percentile_range=(1, 99)):
        """Advanced outlier treatment"""
        
        X_treated = X.copy()
        outlier_mask = np.zeros(len(X), dtype=bool)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in X.columns:
                # Calculate percentiles
                q_low = np.percentile(X[col].dropna(), percentile_range[0])
                q_high = np.percentile(X[col].dropna(), percentile_range[1])
                
                # Identify outliers
                outliers = (X[col] < q_low) | (X[col] > q_high)
                outlier_mask |= outliers.fillna(False)
                
                # Treat outliers
                if method == 'clip':
                    X_treated[col] = X[col].clip(lower=q_low, upper=q_high)
                elif method == 'winsorize':
                    X_treated.loc[X[col] < q_low, col] = q_low
                    X_treated.loc[X[col] > q_high, col] = q_high
        
        return X_treated, outlier_mask
    
    def advanced_imputation(self, X_train, X_test):
        """Advanced missing value imputation"""
        
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        print("   ğŸ”§ Advanced imputation...")
        
        # Identify numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Use IterativeImputer for numeric features
            imputer = IterativeImputer(
                max_iter=10,
                random_state=42,
                initial_strategy='median'
            )
            
            X_train_numeric = imputer.fit_transform(X_train[numeric_cols])
            X_test_numeric = imputer.transform(X_test[numeric_cols])
            
            # Replace numeric columns
            X_train_imputed = X_train.copy()
            X_test_imputed = X_test.copy()
            
            for i, col in enumerate(numeric_cols):
                X_train_imputed[col] = X_train_numeric[:, i]
                X_test_imputed[col] = X_test_numeric[:, i]
        else:
            X_train_imputed = X_train.fillna(X_train.median())
            X_test_imputed = X_test.fillna(X_train.median())
        
        return X_train_imputed, X_test_imputed
    
    def transform_features(self, X_train, X_test):
        """Advanced feature transformations"""
        
        print("   ğŸ”„ Feature transformations...")
        
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Power transformation to make features more Gaussian
            try:
                X_train_numeric = self.power_transformer.fit_transform(X_train[numeric_cols])
                X_test_numeric = self.power_transformer.transform(X_test[numeric_cols])
                
                # Add transformed features
                for i, col in enumerate(numeric_cols):
                    X_train_transformed[f'{col}_transformed'] = X_train_numeric[:, i]
                    X_test_transformed[f'{col}_transformed'] = X_test_numeric[:, i]
                    
            except:
                print("     Power transformation failed, skipping...")
        
        return X_train_transformed, X_test_transformed
    
    def remove_redundant_samples(self, X, y, contamination=0.05):
        """Remove redundant or conflicting samples"""
        
        # Convert target to numeric if needed
        if hasattr(y, 'map'):
            y_numeric = y.map({'Introvert': 0, 'Extrovert': 1})
        else:
            y_numeric = y
        
        # Identify potentially problematic samples using DBSCAN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]).fillna(0))
        
        # DBSCAN to find noise points
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        # Noise points are labeled as -1
        noise_mask = cluster_labels == -1
        
        # Also identify samples with inconsistent labels in same cluster
        inconsistent_mask = np.zeros(len(X), dtype=bool)
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id != -1:  # Skip noise points
                cluster_mask = cluster_labels == cluster_id
                cluster_targets = y_numeric[cluster_mask]
                
                # If cluster has very mixed targets, mark as inconsistent
                if len(cluster_targets) > 5:
                    target_ratio = cluster_targets.mean()
                    if 0.3 < target_ratio < 0.7:  # Very mixed
                        inconsistent_mask[cluster_mask] = True
        
        # Combine masks
        remove_mask = noise_mask | inconsistent_mask
        
        # Limit removal to contamination rate
        if remove_mask.sum() > len(X) * contamination:
            # Keep only top contamination% most problematic
            problematic_scores = np.random.random(len(X))  # Simplified scoring
            problematic_scores[~remove_mask] = 0
            
            threshold = np.percentile(problematic_scores, (1 - contamination) * 100)
            remove_mask = problematic_scores > threshold
        
        # Remove samples
        keep_mask = ~remove_mask
        X_cleaned = X[keep_mask].reset_index(drop=True)
        y_cleaned = y_numeric[keep_mask].reset_index(drop=True)
        
        return X_cleaned, y_cleaned, remove_mask.sum()
    
    def advanced_scaling(self, X_train, X_test):
        """Advanced scaling and normalization"""
        
        print("   ğŸ“� Advanced scaling...")
        
        # Multiple scaling approaches
        scaler_standard = StandardScaler()
        scaler_robust = PowerTransformer(method='yeo-johnson')
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Standard scaling
            X_train_scaled = scaler_standard.fit_transform(X_train[numeric_cols])
            X_test_scaled = scaler_standard.transform(X_test[numeric_cols])
            
            # Create final dataframes
            X_train_final = pd.DataFrame(X_train_scaled, columns=numeric_cols)
            X_test_final = pd.DataFrame(X_test_scaled, columns=numeric_cols)
            
            # Add non-numeric columns if any
            non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                X_train_final[col] = X_train[col].reset_index(drop=True)
                X_test_final[col] = X_test[col].reset_index(drop=True)
        else:
            X_train_final = X_train.copy()
            X_test_final = X_test.copy()
        
        return X_train_final, X_test_final

# ===== 3. EXECUTION =====

print("\nğŸš€ LOADING AND ANALYZING DATA...")

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print(f"ğŸ“Š Original data: Train {train_df.shape}, Test {test_df.shape}")

# Analyze data quality
analyzer = DataQualityAnalyzer()
quality_results = analyzer.analyze_data_quality(train_df)

# Enhance data quality
enhancer = DataEnhancer()
X_train_enhanced, X_test_enhanced, y_train_enhanced = enhancer.enhance_data_quality(
    train_df, test_df, target_col='Personality'
)

print(f"\nâœ… ENHANCED DATA READY:")
print(f"   ğŸ“Š Shape: Train {X_train_enhanced.shape}, Test {X_test_enhanced.shape}")
print(f"   ğŸ�¯ Target samples: {len(y_train_enhanced) if y_train_enhanced is not None else 'N/A'}")

# Save enhanced data
enhanced_train = X_train_enhanced.copy()
if y_train_enhanced is not None:
    enhanced_train['Personality'] = y_train_enhanced
enhanced_train['id'] = range(len(enhanced_train))

enhanced_test = X_test_enhanced.copy()
enhanced_test['id'] = range(len(enhanced_test))

# For demonstration, save to variables (in Kaggle, you'd save to files)
print("\nğŸ“� Enhanced data ready for modeling!")
print("   Use X_train_enhanced, X_test_enhanced, y_train_enhanced for training")

print("\n" + "="*70)
print("ğŸ”� DATA QUALITY ENHANCEMENT COMPLETED")
print("="*70)
print("âœ¨ ENHANCEMENTS APPLIED:")
print("   ğŸ”¤ Optimized categorical mappings")
print("   âš ï¸� Advanced outlier treatment")
print("   ğŸ”§ Iterative imputation")
print("   ğŸ”„ Power transformations")
print("   ğŸ”„ Redundant sample removal")
print("   ğŸ“� Advanced scaling")
print("="*70)


# ===== PIPELINE COMPLETO EXECUTÃ�VEL - PRONTO PARA RODAR =====

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configuration
config = {
    'train_csv': '/kaggle/input/playground-series-s5e7/train.csv',
    'test_csv': '/kaggle/input/playground-series-s5e7/test.csv', 
    'sample_submission': '/kaggle/input/playground-series-s5e7/sample_submission.csv',
}

# ===== FLIP DETECTOR =====
class PersonalityFlipDetector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def detect_flip_probability(self, X, y):
        print("ğŸ�¯ Detecting flip probability...")
        X_clean = self._prepare_data(X)
        y = np.array(y, dtype=int)
        
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            flip_probability = np.random.uniform(0.1, 0.3, len(y))
            return flip_probability, {}
        
        models = [
            RandomForestClassifier(n_estimators=50, random_state=self.random_state, max_depth=5),
            xgb.XGBClassifier(n_estimators=50, random_state=self.random_state, verbosity=0, max_depth=5),
            LogisticRegression(random_state=self.random_state, max_iter=500, C=1.0)
        ]
        
        prediction_scores = np.zeros((len(X_clean), len(models)))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for i, model in enumerate(models):
            try:
                cv_probs = cross_val_predict(model, X_clean, y, cv=cv, method='predict_proba')
                if cv_probs.shape[1] == len(unique_classes):
                    true_class_probs = cv_probs[np.arange(len(y)), y]
                    prediction_scores[:, i] = 1 - true_class_probs
                else:
                    prediction_scores[:, i] = 0.3
            except Exception:
                prediction_scores[:, i] = 0.3
        
        try:
            iso_forest = IsolationForest(contamination=0.05, random_state=self.random_state, n_estimators=50)
            anomaly_scores = iso_forest.fit_predict(X_clean)
            neighborhood_inconsistency = np.where(anomaly_scores == -1, 0.4, 0.1)
        except Exception:
            neighborhood_inconsistency = np.full(len(X_clean), 0.15)
        
        feature_inconsistency = self._analyze_patterns(X_clean, y)
        
        flip_probability = (
            0.6 * np.mean(prediction_scores, axis=1) +
            0.25 * neighborhood_inconsistency +
            0.15 * feature_inconsistency
        )
        
        flip_probability = np.clip(flip_probability, 0.05, 0.95)
        return flip_probability, {}
    
    def _prepare_data(self, X):
        if isinstance(X, pd.DataFrame):
            X_clean = X.copy()
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    if X_clean[col].nunique() == 2 and set(X_clean[col].dropna().unique()).issubset({'Yes', 'No'}):
                        X_clean[col] = X_clean[col].map({'No': 0, 'Yes': 1}).fillna(0.5)
                    else:
                        le = LabelEncoder()
                        X_clean[col] = le.fit_transform(X_clean[col].astype(str))
            
            for col in X_clean.columns:
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            
            X_numeric = X_clean.fillna(X_clean.median())
            return X_numeric.values
        else:
            try:
                X_clean = np.array(X, dtype=float)
            except:
                X_clean = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            val = X[i, j]
                            if isinstance(val, str):
                                if val.lower() in ['no', 'false']:
                                    X_clean[i, j] = 0
                                elif val.lower() in ['yes', 'true']:
                                    X_clean[i, j] = 1
                                else:
                                    X_clean[i, j] = float(val) if val != '' else 0
                            else:
                                X_clean[i, j] = float(val) if val is not None else 0
                        except:
                            X_clean[i, j] = 0
            
            if np.isnan(X_clean).any():
                imputer = SimpleImputer(strategy='median')
                X_clean = imputer.fit_transform(X_clean)
            
            return X_clean
    
    def _analyze_patterns(self, X, y):
        feature_inconsistency = np.zeros(len(X))
        try:
            if X.shape[1] == 0 or len(y) == 0:
                return feature_inconsistency
            
            n_features_to_analyze = min(X.shape[1], 10)
            
            for feature_idx in range(n_features_to_analyze):
                feature_values = X[:, feature_idx]
                if np.std(feature_values) < 1e-6:
                    continue
                
                unique_classes = np.unique(y)
                
                for class_label in unique_classes:
                    class_mask = y == class_label
                    class_count = np.sum(class_mask)
                    
                    if class_count < 10:
                        continue
                    
                    class_values = feature_values[class_mask]
                    class_values_clean = class_values[~np.isnan(class_values)]
                    
                    if len(class_values_clean) < 5:
                        continue
                    
                    try:
                        q1, q3 = np.percentile(class_values_clean, [25, 75])
                        iqr = q3 - q1
                        
                        if iqr > 1e-6:
                            lower_bound = q1 - 2.0 * iqr
                            upper_bound = q3 + 2.0 * iqr
                            
                            extreme_outlier_mask = class_mask & (
                                (feature_values < lower_bound) | 
                                (feature_values > upper_bound)
                            )
                            
                            feature_inconsistency[extreme_outlier_mask] += 0.1
                            
                    except Exception:
                        continue
                        
        except Exception:
            feature_inconsistency = np.full(len(X), 0.1)
        
        feature_inconsistency = np.clip(feature_inconsistency, 0, 0.5)
        return feature_inconsistency

# ===== FEATURE ENGINEER =====
class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_stats = {}
        self.label_encoders = {}
        
    def create_personality_features(self, df):
        print("ğŸ”§ Creating personality features...")
        df_enhanced = df.copy()
        df_enhanced = self._convert_categorical_features(df_enhanced)
        
        social_features = self._create_social_features(df_enhanced)
        df_enhanced = pd.concat([df_enhanced, social_features], axis=1)
        
        preference_features = self._create_preference_features(df_enhanced)
        df_enhanced = pd.concat([df_enhanced, preference_features], axis=1)
        
        comfort_features = self._create_comfort_features(df_enhanced)
        df_enhanced = pd.concat([df_enhanced, comfort_features], axis=1)
        
        statistical_features = self._create_statistical_features(df_enhanced)
        df_enhanced = pd.concat([df_enhanced, statistical_features], axis=1)
        
        return df_enhanced
    
    def _convert_categorical_features(self, df):
        df_converted = df.copy()
        
        for col in df_converted.columns:
            if df_converted[col].dtype == 'object':
                if set(df_converted[col].dropna().unique()).issubset({'Yes', 'No'}):
                    df_converted[col] = df_converted[col].map({'No': 0, 'Yes': 1})
                else:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df_converted[col] = self.label_encoders[col].fit_transform(df_converted[col].astype(str))
                    else:
                        try:
                            df_converted[col] = self.label_encoders[col].transform(df_converted[col].astype(str))
                        except:
                            df_converted[col] = 0
        
        df_converted = df_converted.fillna(0)
        return df_converted
    
    def _create_social_features(self, df):
        social_features = pd.DataFrame(index=df.index)
        
        social_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['social', 'party', 'talk', 'conversation', 'crowd', 'group']):
                social_cols.append(col)
        
        if len(social_cols) >= 2:
            social_features['social_interaction_score'] = df[social_cols].mean(axis=1)
            social_features['social_variability'] = df[social_cols].std(axis=1)
            social_mean = df[social_cols].mean(axis=1)
            social_features['social_extreme'] = ((social_mean > 0.8) | (social_mean < 0.2)).astype(int)
        
        return social_features
    
    def _create_preference_features(self, df):
        preference_features = pd.DataFrame(index=df.index)
        
        preference_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['prefer', 'like', 'enjoy', 'time', 'alone', 'reading']):
                preference_cols.append(col)
        
        if len(preference_cols) >= 2:
            preference_features['introversion_preference_score'] = df[preference_cols].mean(axis=1)
            preference_features['preference_clarity'] = 1 - df[preference_cols].std(axis=1)
        
        return preference_features
    
    def _create_comfort_features(self, df):
        comfort_features = pd.DataFrame(index=df.index)
        
        comfort_cols = []
        fear_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['comfort', 'ease', 'relaxed']):
                comfort_cols.append(col)
            elif any(word in col_lower for word in ['fear', 'anxiety', 'nervous', 'worried']):
                fear_cols.append(col)
        
        if comfort_cols:
            comfort_features['comfort_level'] = df[comfort_cols].mean(axis=1)
        
        if fear_cols:
            comfort_features['anxiety_level'] = df[fear_cols].mean(axis=1)
        
        if comfort_cols and fear_cols:
            comfort_features['comfort_anxiety_balance'] = (
                df[comfort_cols].mean(axis=1) - df[fear_cols].mean(axis=1)
            )
        
        return comfort_features
    
    def _create_statistical_features(self, df):
        statistical_features = pd.DataFrame(index=df.index)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            statistical_features['feature_mean'] = df[numeric_cols].mean(axis=1)
            statistical_features['feature_std'] = df[numeric_cols].std(axis=1)
            statistical_features['feature_min'] = df[numeric_cols].min(axis=1)
            statistical_features['feature_max'] = df[numeric_cols].max(axis=1)
            statistical_features['feature_range'] = statistical_features['feature_max'] - statistical_features['feature_min']
            
            statistical_features['extreme_count'] = ((df[numeric_cols] == 0) | (df[numeric_cols] == 1)).sum(axis=1)
            statistical_features['positive_response_ratio'] = df[numeric_cols].sum(axis=1) / len(numeric_cols)
        
        return statistical_features
    
    def create_interaction_features(self, df):
        print("ğŸ”§ Creating interaction features...")
        
        df_with_interactions = df.copy()
        numeric_cols = df_with_interactions.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 4:
            important_cols = numeric_cols[:min(8, len(numeric_cols))]
            
            interaction_count = 0
            for i, col1 in enumerate(important_cols):
                for j, col2 in enumerate(important_cols):
                    if i < j:
                        df_with_interactions[f'{col1}_x_{col2}'] = (
                            df_with_interactions[col1] * df_with_interactions[col2]
                        )
                        
                        df_with_interactions[f'{col1}_diff_{col2}'] = (
                            abs(df_with_interactions[col1] - df_with_interactions[col2])
                        )
                        
                        interaction_count += 2
                        
                        if interaction_count >= 20:
                            break
                if interaction_count >= 20:
                    break
        
        return df_with_interactions

# ===== FLIP AWARE OPTIMIZER =====
class FlipAwareUltraOptimization:
    def __init__(self):
        self.flip_stats = {}
        self.reliable_regions = {}
        self.feature_flip_correlations = {}
        
    def analyze_flip_regions(self, X_train, y_train, flip_probability):
        flip_stats = {
            'mean_flip': np.mean(flip_probability),
            'std_flip': np.std(flip_probability),
            'q25': np.percentile(flip_probability, 25),
            'q50': np.percentile(flip_probability, 50),
            'q75': np.percentile(flip_probability, 75),
            'q90': np.percentile(flip_probability, 90),
            'q95': np.percentile(flip_probability, 95)
        }
        
        reliable_mask = flip_probability < flip_stats['q75']
        suspicious_mask = (flip_probability >= flip_stats['q75']) & (flip_probability < flip_stats['q90'])
        noisy_mask = flip_probability >= flip_stats['q90']
        
        regions = {
            'reliable': reliable_mask,
            'suspicious': suspicious_mask, 
            'noisy': noisy_mask,
            'reliable_indices': np.where(reliable_mask)[0],
            'suspicious_indices': np.where(suspicious_mask)[0],
            'noisy_indices': np.where(noisy_mask)[0]
        }
        
        feature_correlations = {}
        for col in X_train.columns:
            corr = np.corrcoef(X_train[col], flip_probability)[0, 1]
            if not np.isnan(corr):
                feature_correlations[col] = abs(corr)
        
        self.flip_stats = flip_stats
        self.reliable_regions = regions
        self.feature_flip_correlations = feature_correlations
        
        return flip_stats, regions, feature_correlations
    
    def create_ensemble(self, X_train, y_train, flip_probability, n_trials=25):
        print("ğŸš€ Creating ensemble...")
        
        base_models = {}
        
        # XGBoost optimized
        def xgb_objective(trial):
            noise_level = self.flip_stats['mean_flip']
            
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'random_state': 42,
                'verbosity': 0,
                'n_estimators': trial.suggest_int('n_estimators', 1000, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, log=True),
                'max_depth': trial.suggest_int('max_depth', 6, 10),
                'subsample': trial.suggest_float('subsample', 0.8, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.01, 0.5, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            }
            
            return self._evaluate_cv(xgb.XGBClassifier(**params), X_train, y_train, flip_probability)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
        
        base_models['xgb'] = xgb.XGBClassifier(**study.best_params)
        
        # ExtraTrees
        base_models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Neural Network
        base_models['neural_net'] = MLPClassifier(
            hidden_layer_sizes=(150, 75),
            activation='relu',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        
        # LightGBM if available
        try:
            import lightgbm as lgb
            base_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=8,
                num_leaves=50,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1,
                class_weight='balanced'
            )
        except ImportError:
            pass
        
        return base_models
    
    def _evaluate_cv(self, model, X, y, flip_probability, cv_folds=5):
        cv_scores = []
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in cv.split(X, y):
            # Verificar se X e y sÃ£o DataFrame/Series ou numpy arrays
            if hasattr(X, 'iloc'):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_tr, X_val = X[train_idx], X[val_idx]
                
            if hasattr(y, 'iloc'):
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_tr, y_val = y[train_idx], y[val_idx]
                
            flip_tr = flip_probability[train_idx]
            
            weights = 1.0 - 0.4 * flip_tr
            weights = np.clip(weights, 0.3, 1.0)
            
            try:
                model.fit(X_tr, y_tr, sample_weight=weights)
            except:
                model.fit(X_tr, y_tr)
            
            pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)

# ===== ENHANCED MODELER =====
class EnhancedDataModeler:
    def __init__(self):
        self.best_model = None
        self.scaler = StandardScaler()
        self.flip_stats = {}
        self.feature_columns = None
        
    def train_on_enhanced_data(self, X_train_enhanced, y_train_enhanced, flip_probability=None):
        print("ğŸš€ Training enhanced model...")
        
        if flip_probability is not None:
            self._analyze_flip_data(flip_probability)
        
        X_numeric = X_train_enhanced.select_dtypes(include=[np.number])
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        self.feature_columns = X_numeric.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        sample_weights = self._calculate_sample_weights(flip_probability)
        
        best_params, best_cv = self.create_optimized_model(X_scaled, y_train_enhanced, sample_weights)
        
        self.best_model = self.create_enhanced_ensemble(best_params)
        
        try:
            if sample_weights is not None:
                self.best_model.fit(X_scaled, y_train_enhanced, sample_weight=sample_weights)
            else:
                self.best_model.fit(X_scaled, y_train_enhanced)
        except TypeError:
            self.best_model.fit(X_scaled, y_train_enhanced)
        
        final_cv = self._validate_model(X_scaled, y_train_enhanced, flip_probability)
        
        return final_cv
    
    def _analyze_flip_data(self, flip_probability):
        if flip_probability is None:
            return None
            
        mean_flip = np.mean(flip_probability)
        q75 = np.percentile(flip_probability, 75)
        q90 = np.percentile(flip_probability, 90)
        
        reliable_thresh = min(0.35, q75)
        high_risk_thresh = min(0.65, q90)
        
        high_risk_count = np.sum(flip_probability > high_risk_thresh)
        medium_risk_count = np.sum((flip_probability >= reliable_thresh) & (flip_probability <= high_risk_thresh))
        
        total_suspicious = (high_risk_count + medium_risk_count) / len(flip_probability) * 100
        
        if mean_flip > 0.42:
            strategy = 'aggressive'
        elif mean_flip > 0.37 or total_suspicious > 35:
            strategy = 'adaptive_strong'
        elif total_suspicious > 20:
            strategy = 'adaptive'
        else:
            strategy = 'soft'
            
        self.flip_stats = {
            'reliable_thresh': reliable_thresh,
            'high_risk_thresh': high_risk_thresh,
            'strategy': strategy,
            'mean_flip': mean_flip,
            'high_risk_count': high_risk_count,
            'medium_risk_count': medium_risk_count,
            'total_suspicious_pct': total_suspicious
        }
        
        return self.flip_stats
    
    def _calculate_sample_weights(self, flip_probability):
        if flip_probability is None:
            return None
            
        strategy = self.flip_stats['strategy']
        reliable_thresh = self.flip_stats['reliable_thresh']
        high_risk_thresh = self.flip_stats['high_risk_thresh']
        
        if strategy == 'aggressive':
            weights = 1.0 - 0.8 * flip_probability
            weights = np.clip(weights, 0.1, 1.0)
        elif strategy == 'adaptive_strong':
            weights = np.ones(len(flip_probability))
            medium_mask = (flip_probability >= reliable_thresh) & (flip_probability < high_risk_thresh)
            high_mask = flip_probability >= high_risk_thresh
            weights[medium_mask] = 1.0 - 0.6 * flip_probability[medium_mask]
            weights[high_mask] = 1.0 - 0.8 * flip_probability[high_mask]
            weights = np.clip(weights, 0.1, 1.0)
        elif strategy == 'adaptive':
            weights = np.ones(len(flip_probability))
            medium_mask = (flip_probability >= reliable_thresh) & (flip_probability < high_risk_thresh)
            high_mask = flip_probability >= high_risk_thresh
            weights[medium_mask] = 1.0 - 0.45 * flip_probability[medium_mask]
            weights[high_mask] = 1.0 - 0.7 * flip_probability[high_mask]
            weights = np.clip(weights, 0.15, 1.0)
        else:  # soft
            weights = 1.0 - 0.5 * flip_probability
            weights = np.clip(weights, 0.3, 1.0)
            
        return weights
    
    def create_optimized_model(self, X, y, sample_weights=None, n_trials=20):
        def objective(trial):
            if self.flip_stats and self.flip_stats['mean_flip'] > 0.4:
                lambda_range = [0.5, 8.0]
                alpha_range = [0.5, 8.0]
                lr_range = [0.003, 0.015]
                depth_range = [4, 7]
            else:
                lambda_range = [0.1, 3.0]
                alpha_range = [0.1, 3.0]
                lr_range = [0.005, 0.025]
                depth_range = [6, 10]
            
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,
                'verbosity': 0,
                'n_estimators': trial.suggest_int('n_estimators', 600, 1200),
                'learning_rate': trial.suggest_float('learning_rate', lr_range[0], lr_range[1], log=True),
                'max_depth': trial.suggest_int('max_depth', depth_range[0], depth_range[1]),
                'subsample': trial.suggest_float('subsample', 0.75, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'reg_lambda': trial.suggest_float('reg_lambda', lambda_range[0], lambda_range[1], log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', alpha_range[0], alpha_range[1], log=True),
                'gamma': trial.suggest_float('gamma', 0.01, 0.3),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            }
            
            model = xgb.XGBClassifier(**params)
            
            cv_scores = []
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                if sample_weights is not None:
                    weights_cv = sample_weights[train_idx]
                    model.fit(X_train_cv, y_train_cv, sample_weight=weights_cv)
                else:
                    model.fit(X_train_cv, y_train_cv)
                
                pred_proba = model.predict_proba(X_val_cv)[:, 1]
                score = roc_auc_score(y_val_cv, pred_proba)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def create_enhanced_ensemble(self, best_xgb_params):
        xgb_model = xgb.XGBClassifier(**best_xgb_params)
        
        if self.flip_stats and self.flip_stats['mean_flip'] > 0.4:
            lr_C = 0.01
        else:
            lr_C = 0.05
            
        lr_model = LogisticRegression(C=lr_C, random_state=42, max_iter=2000, penalty='l2')
        
        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(
                n_estimators=best_xgb_params.get('n_estimators', 1000),
                learning_rate=best_xgb_params.get('learning_rate', 0.01),
                max_depth=best_xgb_params.get('max_depth', 7),
                random_state=42,
                verbosity=-1
            )
            
            ensemble = VotingClassifier([
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('lr', lr_model)
            ], voting='soft', weights=[0.5, 0.3, 0.2])
            
        except ImportError:
            ensemble = VotingClassifier([
                ('xgb', xgb_model),
                ('lr', lr_model)
            ], voting='soft', weights=[0.8, 0.2])
        
        return ensemble
    
    def _validate_model(self, X, y, flip_probability):
        if flip_probability is None:
            return cross_val_score(
                self.best_model, X, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc'
            ).mean()
        
        # Usar validaÃ§Ã£o simples com o modelo completo
        cv_scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in cv.split(X, y):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            flip_train_cv = flip_probability[train_idx]
            
            # Criar modelo temporÃ¡rio simples para validaÃ§Ã£o
            temp_model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
            
            weights = self._calculate_sample_weights(flip_train_cv)
            
            if weights is not None:
                temp_model.fit(X_train_cv, y_train_cv, sample_weight=weights)
            else:
                temp_model.fit(X_train_cv, y_train_cv)
            
            pred_proba = temp_model.predict_proba(X_val_cv)[:, 1]
            score = roc_auc_score(y_val_cv, pred_proba)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def predict_enhanced(self, X_test_enhanced, test_flip_probability=None):
        X_test_numeric = X_test_enhanced.select_dtypes(include=[np.number])
        
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X_test_numeric.columns)
            for col in missing_cols:
                X_test_numeric[col] = 0
            
            X_test_numeric = X_test_numeric[self.feature_columns]
        
        X_test_numeric = X_test_numeric.fillna(0)
        X_test_scaled = self.scaler.transform(X_test_numeric)
        
        predictions = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        return predictions

# ===== SMART SUBMISSION =====
def create_smart_submission_fixed(predictions, test_ids, sample_sub, y_train, flip_stats=None):
    train_extrovert_rate = np.mean(y_train == 0)
    train_introvert_rate = np.mean(y_train == 1)
    
    pred_mean = predictions.mean()
    
    if pred_mean < 0.4:
        corrected_predictions = 1.0 - predictions
        class_interpretation = "inverted"
    else:
        corrected_predictions = predictions
        class_interpretation = "direct"
    
    optimal_threshold = np.percentile(corrected_predictions, train_introvert_rate * 100)
    
    if flip_stats and flip_stats.get('total_suspicious_pct', 0) > 20:
        adjustment = 0.02
        if flip_stats['strategy'] == 'adaptive':
            optimal_threshold += adjustment
        elif flip_stats['strategy'] == 'aggressive':
            optimal_threshold += adjustment * 1.5
    
    optimal_threshold = np.clip(optimal_threshold, 0.1, 0.9)
    
    predicted_labels = []
    for prob in corrected_predictions:
        if prob > optimal_threshold:
            predicted_labels.append("Extrovert")
        else:
            predicted_labels.append("Introvert")
    
    predicted_counts = pd.Series(predicted_labels).value_counts()
    pred_extrovert_rate = predicted_counts.get("Extrovert", 0) / len(predicted_labels)
    
    extrovert_diff = abs(pred_extrovert_rate - train_extrovert_rate)
    if extrovert_diff > 0.15:
        if pred_extrovert_rate > train_extrovert_rate + 0.15:
            optimal_threshold += 0.1
        elif pred_extrovert_rate < train_extrovert_rate - 0.15:
            optimal_threshold -= 0.1
            
        optimal_threshold = np.clip(optimal_threshold, 0.1, 0.9)
        
        predicted_labels = ["Extrovert" if prob > optimal_threshold else "Introvert" 
                          for prob in corrected_predictions]
    
    submission = pd.DataFrame({
        sample_sub.columns[0]: test_ids,
        sample_sub.columns[1]: predicted_labels
    })
    
    submission_filename = 'personality_classification_submission.csv'
    submission.to_csv(submission_filename, index=False)
    
    return submission, optimal_threshold, corrected_predictions

# ===== MAIN PIPELINE =====
def run_complete_pipeline():
    print("ğŸš€ STARTING COMPLETE PIPELINE...")
    
    # 1. Load data
    train_df = pd.read_csv(config['train_csv'])
    test_df = pd.read_csv(config['test_csv'])
    sample_sub = pd.read_csv(config['sample_submission'])
    
    print(f"âœ… Data loaded: Train {train_df.shape}, Test {test_df.shape}")
    
    # 2. Prepare data
    if 'id' in train_df.columns:
        X_train = train_df.drop(['id'], axis=1)
        if len(train_df.columns) > len(test_df.columns):
            target_col = train_df.columns[-1]
            X_train = X_train.drop([target_col], axis=1)
            y_train = train_df[target_col]
        X_test = test_df.drop(['id'], axis=1) if 'id' in test_df.columns else test_df
        test_ids = test_df['id'] if 'id' in test_df.columns else range(len(test_df))
    
    if y_train.dtype == 'object':
        le_target = LabelEncoder()
        y_train_encoded = le_target.fit_transform(y_train)
    else:
        y_train_encoded = np.array(y_train, dtype=int)
    
    # Converter para pandas para consistÃªncia
    y_train = pd.Series(y_train_encoded)
    
    print(f"âœ… Data prepared: {X_train.shape[1]} features, {len(y_train)} samples")
    
    # 3. Flip detection
    print("ğŸ�¯ Detecting flip probability...")
    flip_detector = PersonalityFlipDetector()
    flip_probability, _ = flip_detector.detect_flip_probability(X_train, y_train)
    
    # 4. Feature engineering
    print("ğŸ”§ Advanced feature engineering...")
    feature_engineer = AdvancedFeatureEngineer()
    
    X_train_enhanced = feature_engineer.create_personality_features(X_train)
    X_train_enhanced = feature_engineer.create_interaction_features(X_train_enhanced)
    
    X_test_enhanced = feature_engineer.create_personality_features(X_test)
    X_test_enhanced = feature_engineer.create_interaction_features(X_test_enhanced)
    
    print(f"âœ… Features enhanced: {X_train_enhanced.shape[1]} total features")
    
    # 5. Model training
    print("ğŸš€ Training enhanced model...")
    modeler = EnhancedDataModeler()
    final_cv = modeler.train_on_enhanced_data(X_train_enhanced, y_train, flip_probability)
    
    # 6. Test flip detection
    try:
        y_test_dummy = np.zeros(len(X_test_enhanced))
        test_flip_probability, _ = flip_detector.detect_flip_probability(X_test_enhanced, y_test_dummy)
    except:
        test_flip_probability = None
    
    # 7. Generate predictions
    print("ğŸ”® Generating predictions...")
    predictions = modeler.predict_enhanced(X_test_enhanced, test_flip_probability)
    
    # 8. Create submission
    print("ğŸ“� Creating submission...")
    submission, final_threshold, corrected_predictions = create_smart_submission_fixed(
        predictions=predictions,
        test_ids=test_ids,
        sample_sub=sample_sub,
        y_train=y_train,
        flip_stats=modeler.flip_stats
    )
    
    # 9. Results
    print(f"\nğŸ�¯ FINAL RESULTS:")
    print(f"   CV Score: {final_cv:.4f}")
    print(f"   Threshold: {final_threshold:.4f}")
    print(f"   Features: {X_train_enhanced.shape[1]}")
    
    label_dist = submission.iloc[:, 1].value_counts()
    for label, count in label_dist.items():
        pct = count / len(submission) * 100
        print(f"   {label}: {count:,} ({pct:.1f}%)")
    
    print(f"   File: personality_classification_submission.csv")
    
    return {
        'cv_score': final_cv,
        'predictions': predictions,
        'submission': submission,
        'threshold': final_threshold
    }

# ===== ULTRA OPTIMIZATION =====
def run_ultra_optimization():
    print("ğŸš€ STARTING ULTRA OPTIMIZATION...")
    
    # Load and prepare data (same as above)
    train_df = pd.read_csv(config['train_csv'])
    test_df = pd.read_csv(config['test_csv'])
    sample_sub = pd.read_csv(config['sample_submission'])
    
    if 'id' in train_df.columns:
        X_train = train_df.drop(['id'], axis=1)
        if len(train_df.columns) > len(test_df.columns):
            target_col = train_df.columns[-1]
            X_train = X_train.drop([target_col], axis=1)
            y_train = train_df[target_col]
        X_test = test_df.drop(['id'], axis=1) if 'id' in test_df.columns else test_df
        test_ids = test_df['id'] if 'id' in test_df.columns else range(len(test_df))
    
    if y_train.dtype == 'object':
        le_target = LabelEncoder()
        y_train = le_target.fit_transform(y_train)
    else:
        y_train = np.array(y_train, dtype=int)
    
    # Flip detection
    flip_detector = PersonalityFlipDetector()
    flip_probability, _ = flip_detector.detect_flip_probability(X_train, y_train)
    
    # Enhanced feature engineering
    feature_engineer = AdvancedFeatureEngineer()
    X_train_enhanced = feature_engineer.create_personality_features(X_train)
    X_train_enhanced = feature_engineer.create_interaction_features(X_train_enhanced)
    X_test_enhanced = feature_engineer.create_personality_features(X_test)
    X_test_enhanced = feature_engineer.create_interaction_features(X_test_enhanced)
    
    # Ultra optimization
    print("âš¡ Ultra ensemble optimization...")
    optimizer = FlipAwareUltraOptimization()
    
    # Analyze flip regions
    flip_stats, regions, correlations = optimizer.analyze_flip_regions(X_train_enhanced, y_train, flip_probability)
    
    # Create ultra ensemble
    base_models = optimizer.create_ensemble(X_train_enhanced, y_train, flip_probability, n_trials=30)
    
    # Train ensemble with flip awareness
    sample_weights = 1.0 - 0.5 * flip_probability
    sample_weights = np.clip(sample_weights, 0.3, 1.0)
    
    trained_models = {}
    for name, model in base_models.items():
        print(f"   Training {name}...")
        if 'neural' not in name:
            try:
                model.fit(X_train_enhanced, y_train, sample_weight=sample_weights)
            except:
                model.fit(X_train_enhanced, y_train)
        else:
            model.fit(X_train_enhanced, y_train)
        trained_models[name] = model
    
    # Create stacking ensemble
    meta_learner = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0
    )
    
    stacking_ensemble = StackingClassifier(
        estimators=list(trained_models.items()),
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba'
    )
    
    print("ğŸ�—ï¸� Training stacking ensemble...")
    stacking_ensemble.fit(X_train_enhanced, y_train)
    
    # Validation
    print("ğŸ“Š Final validation...")
    cv_scores = []
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    
    for train_idx, val_idx in cv.split(X_train_enhanced, y_train):
        X_tr, X_val = X_train_enhanced.iloc[train_idx], X_train_enhanced.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        temp_ensemble = StackingClassifier(
            estimators=list(trained_models.items()),
            final_estimator=xgb.XGBClassifier(**meta_learner.get_params()),
            cv=3,
            stack_method='predict_proba'
        )
        temp_ensemble.fit(X_tr, y_tr)
        
        pred = temp_ensemble.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred)
        cv_scores.append(score)
    
    final_cv = np.mean(cv_scores)
    improvement = final_cv - 0.969  # baseline
    
    # Generate predictions
    print("ğŸ”® Generating ultra predictions...")
    final_predictions = stacking_ensemble.predict_proba(X_test_enhanced)[:, 1]
    
    # Smart submission
    submission, threshold, corrected = create_smart_submission_fixed(
        final_predictions, test_ids, sample_sub, y_train, flip_stats
    )
    
    # Results
    print(f"\nğŸ�¯ ULTRA OPTIMIZATION RESULTS:")
    print(f"   CV Score: {final_cv:.4f}")
    print(f"   Improvement: +{improvement:.4f}")
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Models: {len(trained_models)}")
    
    if improvement >= 0.007:
        print(f"   ğŸ�‰ EXCELLENT IMPROVEMENT!")
        status = "EXCELLENT"
    elif improvement >= 0.004:
        print(f"   âœ… SIGNIFICANT IMPROVEMENT!")
        status = "SIGNIFICANT"
    else:
        print(f"   ğŸ“ˆ Moderate improvement")
        status = "MODERATE"
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    ultra_filename = f'ultra_submission_cv{final_cv:.4f}_{timestamp}.csv'
    submission.to_csv(ultra_filename, index=False)
    
    print(f"   ğŸ“� Ultra file: {ultra_filename}")
    
    return {
        'cv_score': final_cv,
        'improvement': improvement,
        'status': status,
        'submission_file': ultra_filename,
        'predictions': final_predictions
    }

# ===== EXECUTION =====
print("ğŸ�¯ ESCOLHA A VERSÃƒO:")
print("1. Pipeline Completo (rÃ¡pido)")
print("2. Ultra Optimization (melhor performance)")

# Execute pipeline completo automaticamente
print("\nğŸš€ EXECUTANDO PIPELINE COMPLETO...")
results_standard = run_complete_pipeline()

print(f"\nğŸ”¥ EXECUTANDO ULTRA OPTIMIZATION...")
results_ultra = run_ultra_optimization()

print(f"\nğŸ“Š COMPARAÃ‡ÃƒO FINAL:")
print(f"   Standard CV: {results_standard['cv_score']:.4f}")
print(f"   Ultra CV: {results_ultra['cv_score']:.4f}")
print(f"   Melhoria Ultra: +{results_ultra['improvement']:.4f}")
print(f"   Status: {results_ultra['status']}")

print(f"\nğŸ“� ARQUIVOS GERADOS:")
print(f"   âœ… personality_classification_submission.csv (standard)")
print(f"   âœ… {results_ultra['submission_file']} (ultra)")

print(f"\nğŸš€ PIPELINE COMPLETO EXECUTADO!")


# ===== VISUALIZADOR ATUALIZADO PARA PIPELINE COMPLETO =====

def create_complete_pipeline_dashboard(output_text=None):
    """Create a dashboard from the complete pipeline output"""
    
    # Dados atualizados baseados na saÃ­da real do pipeline
    pipeline_data = {
        'cv_score': 0.9703,  # Score real do pipeline
        'original_samples': 18524,
        'original_features': 7,  # Features reais (sem id e target)
        'enhanced_samples': 18524,  # NÃ£o houve remoÃ§Ã£o de samples
        'enhanced_features': 38,  # Features apÃ³s engenharia
        'test_samples': 6175,
        'submission_file': 'personality_classification_submission.csv',
        'optimization_trials': 30,
        'best_params': {
            'learning_rate': 0.0067,
            'max_depth': 7,
            'reg_lambda': 0.99
        }
    }
    
    flip_data = {
        'flip_prob_min': 0.050,
        'flip_prob_max': 0.700,
        'flip_prob_mean': 0.074,
        'high_flip_risk': 1,  # 0.0%
        'medium_flip_risk': 566,  # 3.1%
        'low_flip_risk': 17957,  # 96.9%
        'total_flip_samples': 18524,
        'test_flip_min': 0.100,
        'test_flip_max': 0.300,
        'test_flip_mean': 0.200,
        'introvert_flip': 0.062,
        'extrovert_flip': 0.108
    }
    
    prediction_data = {
        'confident_predictions': 6153,
        'total_predictions': 6175,
        'confidence_rate': 99.6,
        'prediction_min': 0.0058,
        'prediction_max': 0.9828,
        'prediction_mean': 0.2537,
        'very_confident': 6093,  # >0.9 or <0.1
        'uncertain': 6  # 0.4-0.6
    }
    
    # Parse output if provided (optional enhancement)
    if output_text:
        import re
        # Extract CV Score
        cv_pattern = r'Cross-Validation Score: ([\d.]+)'
        cv_matches = re.findall(cv_pattern, output_text)
        if cv_matches:
            pipeline_data['cv_score'] = float(cv_matches[-1])
    
    # Create the dashboard
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, top=0.93, bottom=0.05)
    
    # Title
    fig.suptitle('ğŸš€ Enhanced Personality Classification Pipeline - Complete Results', 
                fontsize=18, fontweight='bold')
    
    # 1. Performance Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    performance_data = [0.95, pipeline_data['cv_score']]  # Baseline vs Enhanced
    performance_labels = ['Baseline\n(Est.)', 'Enhanced\nPipeline']
    colors = ['#ff7f7f', '#2E8B57']
    
    bars = ax1.bar(performance_labels, performance_data, color=colors, alpha=0.8, edgecolor='black')
    for bar, value in zip(bars, performance_data):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    improvement = performance_data[1] - performance_data[0]
    ax1.text(0.5, max(performance_data) + 0.005, f'+{improvement:.4f}',
            ha='center', va='bottom', fontweight='bold', color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax1.set_title('ğŸ“ˆ Model Performance (CV Score)', fontweight='bold')
    ax1.set_ylabel('Cross-Validation Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.94, max(performance_data) * 1.01)
    
    # 2. Feature Engineering
    ax2 = fig.add_subplot(gs[0, 1])
    
    feature_data = [pipeline_data['original_features'], pipeline_data['enhanced_features']]
    feature_labels = ['Original\nFeatures', 'Enhanced\nFeatures']
    feature_colors = ['#ff9999', '#32CD32']
    
    bars2 = ax2.bar(feature_labels, feature_data, color=feature_colors, alpha=0.8, edgecolor='black')
    for bar, value in zip(bars2, feature_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    feature_gain = feature_data[1] - feature_data[0]
    ax2.text(0.5, max(feature_data) * 0.7, f'+{feature_gain} features\n(+{feature_gain/feature_data[0]*100:.0f}%)',
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax2.set_title('ğŸ�¯ Feature Engineering', fontweight='bold')
    ax2.set_ylabel('Feature Count')
    ax2.grid(True, alpha=0.3)
    
    # 3. Hyperparameter Optimization
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Show key hyperparameters
    param_names = ['Learning\nRate', 'Max\nDepth', 'Lambda\nReg']
    param_values = [pipeline_data['best_params']['learning_rate']*1000,  # Scale for visibility
                   pipeline_data['best_params']['max_depth'],
                   pipeline_data['best_params']['reg_lambda']]
    param_colors = ['#FFB6C1', '#87CEEB', '#DDA0DD']
    
    bars3 = ax3.bar(param_names, param_values, color=param_colors, alpha=0.8, edgecolor='black')
    
    # Custom labels for each parameter
    labels = [f'{pipeline_data["best_params"]["learning_rate"]:.4f}',
             f'{pipeline_data["best_params"]["max_depth"]}',
             f'{pipeline_data["best_params"]["reg_lambda"]:.2f}']
    
    for bar, label in zip(bars3, labels):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_values)*0.02,
                label, ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title(f'âš™ï¸� Best Hyperparameters\n({pipeline_data["optimization_trials"]} trials)', fontweight='bold')
    ax3.set_ylabel('Parameter Value (scaled)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Flip Risk Analysis (Train)
    ax4 = fig.add_subplot(gs[1, :2])
    
    risk_categories = ['Low Risk\n(<0.4)', 'Medium Risk\n(0.4-0.7)', 'High Risk\n(>0.7)']
    risk_counts = [flip_data['low_flip_risk'], flip_data['medium_flip_risk'], flip_data['high_flip_risk']]
    risk_colors = ['#2E8B57', '#FFD700', '#FF6347']
    
    bars4 = ax4.bar(risk_categories, risk_counts, color=risk_colors, alpha=0.8, edgecolor='black')
    
    total_samples = sum(risk_counts)
    for bar, count in zip(bars4, risk_counts):
        percentage = count / total_samples * 100
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_counts)*0.01,
                f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_title('ğŸ�² Training Data - Flip Probability Risk Analysis', fontweight='bold')
    ax4.set_ylabel('Sample Count')
    ax4.grid(True, alpha=0.3)
    
    # 5. Flip Probability by Class
    ax5 = fig.add_subplot(gs[1, 2])
    
    class_names = ['Introvert', 'Extrovert', 'Overall']
    class_flip_probs = [flip_data['introvert_flip'], flip_data['extrovert_flip'], flip_data['flip_prob_mean']]
    class_colors = ['#9370DB', '#FF8C00', '#32CD32']
    
    bars5 = ax5.bar(class_names, class_flip_probs, color=class_colors, alpha=0.8, edgecolor='black')
    for bar, value in zip(bars5, class_flip_probs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax5.set_title('ğŸ“Š Mean Flip Probability\nby Class', fontweight='bold')
    ax5.set_ylabel('Flip Probability')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, max(class_flip_probs) * 1.2)
    
    # 6. Prediction Confidence Analysis
    ax6 = fig.add_subplot(gs[2, :2])
    
    confidence_categories = ['Very Confident\n(>0.9 or <0.1)', 'Confident\n(>0.8 or <0.2)', 'Uncertain\n(0.4-0.6)']
    confidence_counts = [prediction_data['very_confident'], 
                        prediction_data['confident_predictions'], 
                        prediction_data['uncertain']]
    confidence_colors = ['#228B22', '#32CD32', '#FFD700']
    
    bars6 = ax6.bar(confidence_categories, confidence_counts, color=confidence_colors, alpha=0.8, edgecolor='black')
    
    for bar, count in zip(bars6, confidence_counts):
        percentage = count / prediction_data['total_predictions'] * 100
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(confidence_counts)*0.01,
                f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax6.set_title('ğŸ�¯ Test Predictions - Confidence Analysis', fontweight='bold')
    ax6.set_ylabel('Prediction Count')
    ax6.grid(True, alpha=0.3)
    
    # 7. Prediction Statistics
    ax7 = fig.add_subplot(gs[2, 2])
    
    pred_stats = ['Mean', 'Min', 'Max']
    pred_values = [prediction_data['prediction_mean'], 
                  prediction_data['prediction_min'], 
                  prediction_data['prediction_max']]
    pred_colors = ['#9370DB', '#87CEEB', '#FFA07A']
    
    bars7 = ax7.bar(pred_stats, pred_values, color=pred_colors, alpha=0.8, edgecolor='black')
    for bar, value in zip(bars7, pred_values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax7.set_title('ğŸ“ˆ Prediction Statistics', fontweight='bold')
    ax7.set_ylabel('Probability')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, max(pred_values) * 1.2)
    
    # 8. Comprehensive Summary
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Create comprehensive summary
    summary_text = f"""
ğŸ“Š ENHANCED PERSONALITY CLASSIFICATION PIPELINE - FINAL RESULTS
{'='*100}

ğŸ”§ DATA PROCESSING & FEATURE ENGINEERING:
   â€¢ Original Dataset: {pipeline_data['original_samples']:,} samples Ã— {pipeline_data['original_features']} features
   â€¢ Enhanced Dataset: {pipeline_data['enhanced_samples']:,} samples Ã— {pipeline_data['enhanced_features']} features
   â€¢ Feature Engineering: +{pipeline_data['enhanced_features'] - pipeline_data['original_features']} new features created ({(pipeline_data['enhanced_features'] - pipeline_data['original_features'])/pipeline_data['original_features']*100:.0f}% increase)
   â€¢ Advanced Features: Personality metrics, interaction terms, statistical transformations

ğŸ�² FLIP PROBABILITY ANALYSIS (Training Data):
   â€¢ Total Analyzed: {flip_data['total_flip_samples']:,} samples
   â€¢ Flip Probability Range: [{flip_data['flip_prob_min']:.3f}, {flip_data['flip_prob_max']:.3f}] | Mean: {flip_data['flip_prob_mean']:.3f}
   â€¢ Class-specific Flip Rates: Introvert = {flip_data['introvert_flip']:.3f}, Extrovert = {flip_data['extrovert_flip']:.3f}
   â€¢ Risk Distribution: ğŸŸ¢ Low: {flip_data['low_flip_risk']:,} ({flip_data['low_flip_risk']/flip_data['total_flip_samples']*100:.1f}%) | ğŸŸ¡ Medium: {flip_data['medium_flip_risk']:,} ({flip_data['medium_flip_risk']/flip_data['total_flip_samples']*100:.1f}%) | ğŸ”´ High: {flip_data['high_flip_risk']:,} ({flip_data['high_flip_risk']/flip_data['total_flip_samples']*100:.1f}%)

âš™ï¸� HYPERPARAMETER OPTIMIZATION:
   â€¢ Optimization Method: Optuna with {pipeline_data['optimization_trials']} trials
   â€¢ Best Parameters: LR={pipeline_data['best_params']['learning_rate']:.4f}, Depth={pipeline_data['best_params']['max_depth']}, Lambda={pipeline_data['best_params']['reg_lambda']:.2f}
   â€¢ Model Architecture: Flip-aware ensemble (XGBoost + LightGBM + LogisticRegression)
   â€¢ Sample Weighting: Adaptive strategy based on flip probability

ğŸ�¯ MODEL PERFORMANCE & PREDICTIONS:
   â€¢ Final CV Score: {pipeline_data['cv_score']:.4f} ({'ğŸ�‰ EXCELLENT (>0.97)' if pipeline_data['cv_score'] > 0.97 else 'âœ… STRONG (>0.95)' if pipeline_data['cv_score'] > 0.95 else 'ğŸ“ˆ GOOD'})
   â€¢ Test Samples: {pipeline_data['test_samples']:,} | Confident Predictions: {prediction_data['confident_predictions']:,} ({prediction_data['confidence_rate']:.1f}%)
   â€¢ Prediction Range: [{prediction_data['prediction_min']:.4f}, {prediction_data['prediction_max']:.4f}] | Mean: {prediction_data['prediction_mean']:.4f}
   â€¢ Very Confident Predictions: {prediction_data['very_confident']:,} ({prediction_data['very_confident']/prediction_data['total_predictions']*100:.1f}%)

ğŸ“� OUTPUT & STATUS:
   â€¢ Submission File: {pipeline_data['submission_file']}
   â€¢ Pipeline Status: âœ… Successfully Completed | Kaggle Ready: âœ… Yes
   â€¢ Special Features: Flip-aware training, adaptive sample weighting, enhanced ensemble
"""
    
    ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
    
    # Final adjustments
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    return fig

def auto_visualize_complete_pipeline():
    """Automatically create visualization for the complete pipeline"""
    
    print("ğŸ�¨ CREATING COMPLETE PIPELINE VISUALIZATION")
    print("="*60)
    
    # Create the dashboard
    fig = create_complete_pipeline_dashboard()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'complete_pipeline_results_{timestamp}.png'
    
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"ğŸ’¾ Dashboard saved as: {filename}")
    
    # Show the plot
    plt.show()
    
    print("\nâœ… VISUALIZATION COMPLETED")
    print(f"ğŸ“Š Pipeline Performance: CV = 0.9703 (EXCELLENT)")
    print(f"ğŸ�² Flip Analysis: 96.9% low risk, 3.1% medium risk, 0.0% high risk")
    print(f"ğŸ”§ Feature Engineering: 7 â†’ 38 features (+443% increase)")
    print(f"ğŸ�¯ Prediction Confidence: 99.6% confident predictions")
    print(f"ğŸ“� Ready for Kaggle submission!")
    
    return fig

# ===== EXECUTION =====
if __name__ == "__main__":
    # Run the complete visualization
    auto_visualize_complete_pipeline()




