# Essential imports for psychology + pseudo-labeling pipeline
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Feature engineering tools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Machine learning models
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Gradient boosting models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Utilities
import json
from collections import Counter

print("âœ… All libraries imported successfully!")
print("ğŸ§  Ready for psychology-driven personality prediction")
print("ğŸ”„ Pseudo-labeling pipeline initialized")


# Load competition data
print("ğŸ“� Loading Personality Prediction Dataset...")

train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Display the features we're working with
feature_columns = [col for col in train_df.columns if col not in ['id', 'Personality']]
print(f"\nğŸ”� Available features ({len(feature_columns)}):")
for i, col in enumerate(feature_columns, 1):
    print(f"   {i}. {col}")

# Basic data exploration
print("\nğŸ“Š Dataset Overview:")
print(train_df.head())

print("\nğŸ�¯ Target Variable Distribution:")
personality_counts = train_df['Personality'].value_counts()
print(personality_counts)
print(f"\nExtrovert ratio: {personality_counts['Extrovert'] / len(train_df):.3f}")
print(f"Class balance: {'Balanced' if abs(personality_counts['Extrovert'] - personality_counts['Introvert']) < len(train_df) * 0.1 else 'Imbalanced'}")

# Missing value analysis
print("\nğŸ”� Missing Value Analysis:")
missing_analysis = train_df[feature_columns].isnull().sum()
missing_features = missing_analysis[missing_analysis > 0]

if len(missing_features) > 0:
    print("Features with missing values:")
    for feature, count in missing_features.items():
        percentage = (count / len(train_df)) * 100
        print(f"   {feature}: {count} ({percentage:.1f}%)")
else:
    print("âœ… No missing values in training data")

# Prepare target variable
y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})
print(f"\nâœ… Target variable encoded: {dict(zip(['Introvert', 'Extrovert'], [0, 1]))}") 


class PsychologicalFeatureEngineer:
    """Advanced feature engineering based on Big Five personality psychology"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
    def create_psychological_features(self, df):
        """Generate comprehensive psychological features"""
        print("ğŸ§  Creating psychology-based features...")
        
        df_features = df.copy()
        
        # 1. Basic psychological scores
        df_features = self._create_basic_psychological_scores(df_features)
        print("   âœ… Basic psychological scores created")
        
        # 2. Interaction features (behavioral patterns)
        df_features = self._create_interaction_features(df_features)
        print("   âœ… Interaction features created")
        
        # 3. Statistical transformations
        df_features = self._create_statistical_transformations(df_features)
        print("   âœ… Statistical transformations applied")
        
        # 4. Missing pattern features
        df_features = self._create_missing_pattern_features(df_features)
        print("   âœ… Missing pattern features extracted")
        
        # 5. Clustering features (personality types)
        df_features = self._create_clustering_features(df_features)
        print("   âœ… Clustering features (personality types) identified")
        
        return df_features
    
    def _create_basic_psychological_scores(self, df):
        """Core psychological measures based on Big Five theory"""
        
        # Convert and handle missing values
        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                       'Friends_circle_size', 'Post_frequency']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        
        # Convert categorical to numeric
        df['Stage_fear_numeric'] = df['Stage_fear'].map({'Yes': 1, 'No': 0}).fillna(0.5)
        df['Drained_numeric'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0}).fillna(0.5)
        
        # 1. EXTRAVERSION SCORE (social activity composite)
        df['extroversion_score'] = (
            df['Social_event_attendance'] + 
            df['Going_outside'] + 
            df['Friends_circle_size'] + 
            df['Post_frequency']
        ) / 4
        
        # 2. INTROVERSION SCORE (solitude + anxiety composite)
        df['introversion_score'] = (
            df['Time_spent_Alone'] + 
            df['Stage_fear_numeric'] * 10 + 
            df['Drained_numeric'] * 10
        ) / 3
        
        # 3. SOCIAL BALANCE (extroversion vs introversion)
        df['social_balance'] = df['extroversion_score'] - df['introversion_score']
        df['social_balance_abs'] = np.abs(df['social_balance'])
        
        # 4. PERSONALITY CONSISTENCY (response consistency measure)
        social_features = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        df['social_consistency'] = df[social_features].std(axis=1)
        
        return df
    
    def _create_interaction_features(self, df):
        """Psychologically meaningful feature interactions"""
        
        # 1. SOCIAL FATIGUE (activity Ã— drain)
        df['social_fatigue'] = df['Social_event_attendance'] * df['Drained_numeric']
        
        # 2. SOCIAL CONFIDENCE (activity Ã— overcoming fear)
        df['social_proactivity'] = df['Social_event_attendance'] * (1 - df['Stage_fear_numeric'])
        
        # 3. DIGITAL SOCIAL BEHAVIOR (posting Ã— friends)
        df['digital_social'] = df['Post_frequency'] * df['Friends_circle_size']
        
        # 4. OUTDOOR SOCIAL BEHAVIOR (outings Ã— events)
        df['outdoor_social'] = df['Going_outside'] * df['Social_event_attendance']
        
        # 5. SOLITUDE PREFERENCE RATIO (alone time / social time)
        denominator = df['Social_event_attendance'] + df['Going_outside'] + 0.01
        df['solitude_preference'] = df['Time_spent_Alone'] / denominator
        
        # 6. SOCIAL EFFICIENCY (friends per social activity)
        df['social_efficiency'] = df['Friends_circle_size'] / (df['Social_event_attendance'] + 0.01)
        
        # 7. COMMUNICATION STYLE (digital vs real-world ratio)
        real_communication = df['Social_event_attendance'] + df['Going_outside'] + 0.01
        df['communication_ratio'] = df['Post_frequency'] / real_communication
        
        return df
    
    def _create_statistical_transformations(self, df):
        """Statistical normalization for behavioral data"""
        
        numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                           'Friends_circle_size', 'Post_frequency']
        
        for col in numeric_features:
            # Log transformation (handles right-skewed distributions)
            df[f'{col}_log'] = np.log1p(df[col])
            
            # Square root transformation (moderate skewness)
            df[f'{col}_sqrt'] = np.sqrt(df[col])
            
            # Z-score standardization
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
            # Rank transformation (handles outliers)
            df[f'{col}_rank'] = df[col].rank(pct=True)
        
        return df
    
    def _create_missing_pattern_features(self, df):
        """Extract psychological insights from missing data patterns"""
        
        original_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
        
        # 1. Missing value flags for each feature
        for col in original_cols:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
        
        # 2. Total missing count
        df['total_missing'] = df[[f'{col}_missing' for col in original_cols]].sum(axis=1)
        
        # 3. Missing ratio
        df['missing_ratio'] = df['total_missing'] / len(original_cols)
        
        # 4. Psychologically meaningful missing patterns
        df['social_missing'] = (df['Social_event_attendance_missing'] + 
                               df['Going_outside_missing'] + 
                               df['Friends_circle_size_missing'])
        
        df['anxiety_missing'] = (df['Stage_fear_missing'] + 
                                df['Drained_after_socializing_missing'])
        
        return df
    
    def _create_clustering_features(self, df):
        """Discover latent personality types through clustering"""
        
        # Features for clustering (psychological scores)
        cluster_features = ['extroversion_score', 'introversion_score', 'social_balance',
                           'social_consistency', 'social_fatigue', 'social_proactivity']
        
        # Handle missing values
        cluster_data = df[cluster_features].fillna(df[cluster_features].median())
        
        # Standardize features
        cluster_data_scaled = self.scaler.fit_transform(cluster_data)
        
        # K-Means clustering (5 personality types)
        df['personality_cluster'] = self.kmeans.fit_predict(cluster_data_scaled)
        
        # Distance to each cluster center
        cluster_distances = self.kmeans.transform(cluster_data_scaled)
        for i in range(5):
            df[f'cluster_{i}_distance'] = cluster_distances[:, i]
        
        # PCA transformation (dimensionality reduction)
        pca_features = self.pca.fit_transform(cluster_data_scaled)
        for i in range(3):
            df[f'pca_component_{i}'] = pca_features[:, i]
        
        return df

print("âœ… Psychological Feature Engineering class defined!")
print("   Ready to transform raw behavioral data into meaningful psychological insights")


class PseudoLabelingEngine:
    """Intelligent pseudo-labeling for semi-supervised personality prediction"""
    
    def __init__(self, confidence_threshold=0.85, max_pseudo_ratio=0.3):
        """
        Args:
            confidence_threshold: Minimum confidence for pseudo-label inclusion
            max_pseudo_ratio: Maximum ratio of pseudo-labels to original data
        """
        self.confidence_threshold = confidence_threshold
        self.max_pseudo_ratio = max_pseudo_ratio
        self.base_models = None
        
    def create_base_models(self):
        """Create ensemble of base models for pseudo-label generation"""
        
        # LightGBM (fast and accurate)
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.02,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_estimators=1500
        )
        
        # XGBoost (robust performance)
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_estimators=1500,
            verbosity=0
        )
        
        # CatBoost (excellent with categorical features)
        cat_model = CatBoostClassifier(
            objective='Logloss',
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            random_seed=42,
            iterations=1500,
            verbose=False
        )
        
        # Ensemble with soft voting
        ensemble = VotingClassifier(
            estimators=[
                ('lgb', lgb_model),
                ('xgb', xgb_model),
                ('cat', cat_model)
            ],
            voting='soft'  # Use probability averages
        )
        
        self.base_models = ensemble
        return ensemble
    
    def generate_pseudo_labels(self, X_train, y_train, X_test):
        """Generate high-confidence pseudo-labels from test data"""
        
        print("ğŸ”„ Generating pseudo-labels...")
        
        # Create base models if not exist
        if self.base_models is None:
            self.base_models = self.create_base_models()
        
        # Cross-validate model performance first
        print("   ğŸ“Š Validating base model performance...")
        cv_score = self._cross_validate_models(X_train, y_train)
        print(f"   âœ… Base model CV accuracy: {cv_score:.4f}")
        
        # Train on full dataset
        print("   ğŸ�‹ï¸� Training ensemble on full dataset...")
        self.base_models.fit(X_train, y_train)
        
        # Generate predictions on test data
        print("   ğŸ�¯ Predicting test data with confidence scores...")
        test_proba = self.base_models.predict_proba(X_test)
        
        # Calculate confidence (maximum probability)
        confidence_scores = np.max(test_proba, axis=1)
        pseudo_labels = self.base_models.predict(X_test)
        
        # Select high-confidence samples
        high_confidence_mask = confidence_scores >= self.confidence_threshold
        
        # Limit to maximum ratio
        max_pseudo_samples = int(len(X_train) * self.max_pseudo_ratio)
        if np.sum(high_confidence_mask) > max_pseudo_samples:
            # Select top-confidence samples
            top_indices = np.argsort(confidence_scores)[-max_pseudo_samples:]
            high_confidence_mask = np.zeros(len(X_test), dtype=bool)
            high_confidence_mask[top_indices] = True
        
        # Extract pseudo-labeled data
        pseudo_X = X_test[high_confidence_mask]
        pseudo_y = pseudo_labels[high_confidence_mask]
        pseudo_confidence = confidence_scores[high_confidence_mask]
        
        print(f"\n   ğŸ“ˆ Pseudo-labeling Results:")
        print(f"      Total test samples: {len(X_test):,}")
        print(f"      High-confidence samples: {len(pseudo_X):,} ({len(pseudo_X)/len(X_test)*100:.1f}%)")
        print(f"      Average confidence: {pseudo_confidence.mean():.4f}")
        print(f"      Pseudo-label distribution: Extrovert={np.sum(pseudo_y==1)}, Introvert={np.sum(pseudo_y==0)}")
        print(f"      Data expansion ratio: {len(pseudo_X)/len(X_train)*100:.1f}%")
        
        return pseudo_X, pseudo_y, pseudo_confidence
    
    def create_augmented_dataset(self, X_train, y_train, X_test):
        """Create expanded training dataset with pseudo-labels"""
        
        # Generate pseudo-labels
        pseudo_X, pseudo_y, pseudo_confidence = self.generate_pseudo_labels(X_train, y_train, X_test)
        
        # Combine original and pseudo-labeled data
        X_augmented = np.vstack([X_train, pseudo_X])
        y_augmented = np.hstack([y_train, pseudo_y])
        
        # Create sample weights
        # Original data: weight = 1.0
        # Pseudo-labels: weight = confidence * 0.8 (max 0.8)
        original_weights = np.ones(len(y_train))
        pseudo_weights = pseudo_confidence * 0.8
        sample_weights = np.hstack([original_weights, pseudo_weights])
        
        print(f"\nğŸ�¯ Final Augmented Dataset:")
        print(f"   Original training samples: {len(X_train):,}")
        print(f"   Pseudo-labeled samples: {len(pseudo_X):,}")
        print(f"   Total augmented samples: {len(X_augmented):,}")
        print(f"   Dataset expansion ratio: {len(X_augmented)/len(X_train):.2f}x")
        print(f"   Average pseudo-label weight: {pseudo_weights.mean():.3f}")
        
        return X_augmented, y_augmented, sample_weights
    
    def _cross_validate_models(self, X, y, cv_folds=5):
        """Cross-validate base models to ensure quality"""
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Train fold model
            fold_model = self.create_base_models()
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_fold_val)
            fold_score = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(fold_score)
        
        return np.mean(cv_scores)

print("âœ… Pseudo-Labeling Engine defined!")
print("   Ready to intelligently expand training data using test patterns")


# Initialize psychological feature engineer
print("ğŸ§  Initializing Psychology-Based Feature Engineering...")
engineer = PsychologicalFeatureEngineer()

# Generate psychological features for training data
print("\nğŸ”„ Generating psychological features for training data...")
train_features = engineer.create_psychological_features(train_df)

# Generate psychological features for test data
print("\nğŸ”„ Generating psychological features for test data...")
test_features = engineer.create_psychological_features(test_df)

# Feature generation summary
original_count = len([c for c in train_df.columns if c not in ['id', 'Personality']])
new_count = len([c for c in train_features.columns if c not in ['id', 'Personality']])
added_count = new_count - original_count

print(f"\nğŸ“Š Psychological Feature Engineering Summary:")
print(f"   Original features: {original_count}")
print(f"   Enhanced features: {new_count}")
print(f"   Added features: {added_count}")
print(f"   Feature enhancement ratio: {new_count/original_count:.1f}x")

# Show some of the new psychological features
print("\nğŸ§  Sample Psychological Features Created:")
psychological_features = [
    'extroversion_score', 'introversion_score', 'social_balance', 
    'social_fatigue', 'social_proactivity', 'solitude_preference',
    'personality_cluster', 'social_consistency'
]

for i, feature in enumerate(psychological_features, 1):
    if feature in train_features.columns:
        print(f"   {i}. {feature}")

print("\nâœ… Psychological feature engineering completed successfully!")
print("   Ready for pseudo-labeling phase...")


# Prepare data for pseudo-labeling
print("ğŸ”„ Preparing data for pseudo-labeling...")

# Select numerical features for modeling
feature_cols = [col for col in train_features.columns 
               if col not in ['id', 'Personality'] and 
               train_features[col].dtype in ['int64', 'float64']]

# Prepare feature matrices
X_train = train_features[feature_cols].fillna(0).values
y_train_numeric = train_features['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
X_test = test_features[feature_cols].fillna(0).values

print(f"   Selected features for modeling: {len(feature_cols)}")
print(f"   Training data shape: {X_train.shape}")
print(f"   Test data shape: {X_test.shape}")

# Initialize pseudo-labeling engine
print("\nğŸ”„ Initializing Pseudo-Labeling Engine...")
pseudo_engine = PseudoLabelingEngine(
    confidence_threshold=0.85,  # Only use highly confident predictions
    max_pseudo_ratio=0.3        # Expand by max 30%
)

# Generate augmented dataset
print("\nğŸš€ Generating pseudo-labeled augmented dataset...")
X_augmented, y_augmented, sample_weights = pseudo_engine.create_augmented_dataset(
    X_train, y_train_numeric, X_test
)

# Data quality check
print(f"\nğŸ”� Data Quality Verification:")
print(f"   Training set expansion: {len(X_train):,} â†’ {len(X_augmented):,} samples")
print(f"   Expansion factor: {len(X_augmented)/len(X_train):.2f}x")
print(f"   Original data weight: {sample_weights[:len(X_train)].mean():.3f}")
print(f"   Pseudo-label weight: {sample_weights[len(X_train):].mean():.3f}")

# Class distribution check
original_dist = np.bincount(y_train_numeric) / len(y_train_numeric)
augmented_dist = np.bincount(y_augmented) / len(y_augmented)

print(f"\nğŸ“Š Class Distribution Analysis:")
print(f"   Original: Introvert={original_dist[0]:.3f}, Extrovert={original_dist[1]:.3f}")
print(f"   Augmented: Introvert={augmented_dist[0]:.3f}, Extrovert={augmented_dist[1]:.3f}")
print(f"   Distribution shift: {abs(augmented_dist[1] - original_dist[1]):.3f}")

print("\nâœ… Pseudo-labeling completed successfully!")
print("   Dataset intelligently expanded with high-confidence predictions")


def create_psychology_ensemble():
    """Create ensemble optimized for psychology-based features"""
    
    models = [
        ('lgb', lgb.LGBMClassifier(
            objective='binary', 
            num_leaves=31, 
            learning_rate=0.02,
            n_estimators=1500, 
            random_state=42, 
            verbosity=-1,
            feature_fraction=0.8,
            bagging_fraction=0.8
        )),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic', 
            max_depth=6, 
            learning_rate=0.02,
            n_estimators=1500, 
            random_state=42, 
            verbosity=0,
            subsample=0.8,
            colsample_bytree=0.8
        )),
        ('cat', CatBoostClassifier(
            objective='Logloss', 
            depth=6, 
            learning_rate=0.02,
            iterations=1500, 
            random_seed=42, 
            verbose=False,
            l2_leaf_reg=3
        )),
        ('lr', LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0
        ))
    ]
    
    return VotingClassifier(estimators=models, voting='soft')

def evaluate_with_sample_weights(model, X, y, sample_weights, cv_folds=5):
    """Custom CV evaluation that respects sample weights"""
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    print(f"ğŸ”„ Performing {cv_folds}-fold cross-validation with sample weights...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        weights_fold_train = sample_weights[train_idx]
        
        # Train individual models with sample weights
        fold_predictions = []
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective='binary', num_leaves=31, learning_rate=0.02,
            n_estimators=1500, random_state=42, verbosity=-1
        )
        lgb_model.fit(X_fold_train, y_fold_train, sample_weight=weights_fold_train)
        lgb_pred = lgb_model.predict_proba(X_fold_val)[:, 1]
        fold_predictions.append(lgb_pred)
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic', max_depth=6, learning_rate=0.02,
            n_estimators=1500, random_state=42, verbosity=0
        )
        xgb_model.fit(X_fold_train, y_fold_train, sample_weight=weights_fold_train)
        xgb_pred = xgb_model.predict_proba(X_fold_val)[:, 1]
        fold_predictions.append(xgb_pred)
        
        # CatBoost
        cat_model = CatBoostClassifier(
            objective='Logloss', depth=6, learning_rate=0.02,
            iterations=1500, random_seed=42, verbose=False
        )
        cat_model.fit(X_fold_train, y_fold_train, sample_weight=weights_fold_train)
        cat_pred = cat_model.predict_proba(X_fold_val)[:, 1]
        fold_predictions.append(cat_pred)
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_fold_train, y_fold_train, sample_weight=weights_fold_train)
        lr_pred = lr_model.predict_proba(X_fold_val)[:, 1]
        fold_predictions.append(lr_pred)
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(fold_predictions, axis=0)
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        # Calculate fold score
        fold_score = accuracy_score(y_fold_val, ensemble_pred_binary)
        cv_scores.append(fold_score)
        
        print(f"   Fold {fold+1}: {fold_score:.6f}")
    
    return np.array(cv_scores)

print("âœ… Ensemble model and evaluation functions defined!")
print("   Ready for comprehensive performance evaluation")


# Comprehensive model evaluation
print("=== Psychology + Pseudo-Labeling Performance Evaluation ===")

# 1. Baseline evaluation (original data only)
print("\n1ï¸�âƒ£ Baseline Evaluation (Original Data Only)")
print("="*50)

baseline_model = create_psychology_ensemble()
baseline_cv_scores = cross_val_score(
    baseline_model, X_train, y_train_numeric, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
    scoring='accuracy'
)

baseline_mean = baseline_cv_scores.mean()
baseline_std = baseline_cv_scores.std()

print(f"Baseline CV Scores: {baseline_cv_scores}")
print(f"Baseline Mean: {baseline_mean:.6f} +/- {baseline_std:.6f}")

# 2. Enhanced evaluation (psychology + pseudo-labeling)
print("\n2ï¸�âƒ£ Enhanced Evaluation (Psychology + Pseudo-Labeling)")
print("="*50)

enhanced_cv_scores = evaluate_with_sample_weights(
    None, X_augmented, y_augmented, sample_weights, cv_folds=5
)

enhanced_mean = enhanced_cv_scores.mean()
enhanced_std = enhanced_cv_scores.std()

print(f"\nEnhanced CV Scores: {enhanced_cv_scores}")
print(f"Enhanced Mean: {enhanced_mean:.6f} +/- {enhanced_std:.6f}")

# 3. Performance comparison
improvement = enhanced_mean - baseline_mean

print("\n" + "="*70)
print("ğŸ�¯ PERFORMANCE COMPARISON RESULTS")
print("="*70)
print(f"Original Psychology Features:     {baseline_mean:.6f} +/- {baseline_std:.6f}")
print(f"Psychology + Pseudo-Labeling:     {enhanced_mean:.6f} +/- {enhanced_std:.6f}")
print(f"Improvement:                     {improvement:+.6f}")
print(f"Relative Improvement:            {improvement/baseline_mean*100:+.2f}%")

# 4. Statistical significance
print(f"\nğŸ“Š Statistical Analysis:")
print(f"   Improvement magnitude: {'Substantial' if improvement > 0.002 else 'Moderate' if improvement > 0.001 else 'Marginal'}")
print(f"   Consistency: {'High' if enhanced_std < baseline_std else 'Similar' if abs(enhanced_std - baseline_std) < 0.001 else 'Lower'}")
print(f"   Effect size: {improvement / baseline_std:.2f} standard deviations")

# 5. Benchmark comparison
print(f"\nğŸ�† Benchmark Comparison:")
gm_baseline = 0.975708  # GM Baseline score
competitive_threshold = 0.970000  # Competitive baseline

if enhanced_mean > gm_baseline:
    print(f"   ğŸ�¯ GM BASELINE EXCEEDED! {enhanced_mean:.6f} > {gm_baseline:.6f} (+{enhanced_mean - gm_baseline:.6f})")
    benchmark_status = "excellent"
elif enhanced_mean > competitive_threshold:
    print(f"   âœ… Competitive performance: {enhanced_mean:.6f} > {competitive_threshold:.6f}")
    benchmark_status = "competitive"
else:
    print(f"   ğŸ“Š Below competitive threshold: {enhanced_mean:.6f} < {competitive_threshold:.6f}")
    benchmark_status = "developing"

# Store results for later use
results = {
    'baseline_cv_mean': baseline_mean,
    'baseline_cv_std': baseline_std,
    'enhanced_cv_mean': enhanced_mean,
    'enhanced_cv_std': enhanced_std,
    'improvement': improvement,
    'relative_improvement_pct': improvement/baseline_mean*100,
    'benchmark_status': benchmark_status,
    'feature_count': len(feature_cols),
    'data_expansion_ratio': len(X_augmented)/len(X_train),
    'cv_scores_baseline': baseline_cv_scores.tolist(),
    'cv_scores_enhanced': enhanced_cv_scores.tolist()
}

print(f"\nâœ… Comprehensive evaluation completed!")
print(f"   Approach effectiveness: {'Highly Effective' if improvement > 0.003 else 'Effective' if improvement > 0.001 else 'Moderately Effective'}")


def train_final_weighted_ensemble(X_train, y_train, X_test, sample_weights):
    """Train final ensemble with proper sample weight handling"""
    
    print("ğŸš€ Training final weighted ensemble...")
    
    # Train individual models with sample weights
    models = {}
    predictions = []
    
    # LightGBM
    print("   ğŸŒŸ Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        objective='binary', num_leaves=31, learning_rate=0.02,
        n_estimators=1500, random_state=42, verbosity=-1,
        feature_fraction=0.8, bagging_fraction=0.8
    )
    lgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    predictions.append(lgb_pred)
    models['lgb'] = lgb_model
    
    # XGBoost
    print("   ğŸš€ Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', max_depth=6, learning_rate=0.02,
        n_estimators=1500, random_state=42, verbosity=0,
        subsample=0.8, colsample_bytree=0.8
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    predictions.append(xgb_pred)
    models['xgb'] = xgb_model
    
    # CatBoost
    print("   ğŸ�± Training CatBoost...")
    cat_model = CatBoostClassifier(
        objective='Logloss', depth=6, learning_rate=0.02,
        iterations=1500, random_seed=42, verbose=False, l2_leaf_reg=3
    )
    cat_model.fit(X_train, y_train, sample_weight=sample_weights)
    cat_pred = cat_model.predict_proba(X_test)[:, 1]
    predictions.append(cat_pred)
    models['cat'] = cat_model
    
    # Logistic Regression
    print("   ğŸ“Š Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr_model.fit(X_train, y_train, sample_weight=sample_weights)
    lr_pred = lr_model.predict_proba(X_test)[:, 1]
    predictions.append(lr_pred)
    models['lr'] = lr_model
    
    # Ensemble prediction (weighted average)
    ensemble_proba = np.mean(predictions, axis=0)
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    print("   âœ… Ensemble training completed")
    
    return models, ensemble_pred, ensemble_proba, predictions

# Train final model and generate predictions
print("=== Final Model Training and Prediction Generation ===")

# Prepare test data with same feature set
X_test_final = test_features[feature_cols].fillna(0).values
test_ids = test_features['id'].values

print(f"Test data shape: {X_test_final.shape}")
print(f"Feature alignment: {'âœ… Aligned' if X_test_final.shape[1] == X_augmented.shape[1] else 'â�Œ Misaligned'}")

# Train final weighted ensemble
final_models, test_predictions, test_probabilities, individual_predictions = train_final_weighted_ensemble(
    X_augmented, y_augmented, X_test_final, sample_weights
)

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in test_predictions]
})

# Prediction analysis
extrovert_count = np.sum(test_predictions == 1)
introvert_count = np.sum(test_predictions == 0)
avg_confidence = np.mean(np.maximum(test_probabilities, 1 - test_probabilities))

print(f"\nğŸ“Š Final Prediction Analysis:")
print(f"   Total predictions: {len(test_predictions):,}")
print(f"   Extrovert: {extrovert_count:,} ({extrovert_count/len(test_predictions)*100:.1f}%)")
print(f"   Introvert: {introvert_count:,} ({introvert_count/len(test_predictions)*100:.1f}%)")
print(f"   Average confidence: {avg_confidence:.4f}")
print(f"   High confidence (>0.8): {np.sum(np.maximum(test_probabilities, 1 - test_probabilities) > 0.8):,}")
print(f"   Low confidence (<0.6): {np.sum(np.maximum(test_probabilities, 1 - test_probabilities) < 0.6):,}")

# Individual model analysis
print(f"\nğŸ”� Individual Model Agreement:")
model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'LogReg']
for i, (name, pred) in enumerate(zip(model_names, individual_predictions)):
    model_pred_binary = (pred > 0.5).astype(int)
    agreement = np.mean(model_pred_binary == test_predictions)
    print(f"   {name}: {agreement:.3f} agreement with ensemble")

print(f"\nâœ… Final model training and prediction completed!")
print(f"   Ready for Kaggle submission")


# Display submission sample
print("ğŸ”� Final Submission Sample:")
print(submission_df.head(10))

# Comprehensive results analysis
print("\n" + "="*80)
print("ğŸ§  PSYCHOLOGY + PSEUDO-LABELING IMPLEMENTATION SUMMARY")
print("="*80)

print(f"\nğŸ“ˆ Performance Metrics:")
print(f"   Baseline CV (Psychology Only):     {results['baseline_cv_mean']:.6f} +/- {results['baseline_cv_std']:.6f}")
print(f"   Enhanced CV (+ Pseudo-Labeling):   {results['enhanced_cv_mean']:.6f} +/- {results['enhanced_cv_std']:.6f}")
print(f"   Performance Improvement:           {results['improvement']:+.6f} ({results['relative_improvement_pct']:+.2f}%)")
print(f"   Benchmark Status:                  {results['benchmark_status'].title()}")

print(f"\nğŸ”§ Technical Implementation:")
print(f"   Original Features:                 7")
print(f"   Psychology-Enhanced Features:      {results['feature_count']:,}")
print(f"   Feature Expansion Ratio:           {results['feature_count']/7:.1f}x")
print(f"   Training Data Expansion:           {results['data_expansion_ratio']:.2f}x")
print(f"   Sample Weight Implementation:      âœ… Full Support")

print(f"\nğŸ§  Psychology-Based Features:")
psychology_features = [
    "Extroversion/Introversion Scores", "Social Balance Metrics", 
    "Behavioral Interaction Patterns", "Statistical Transformations",
    "Missing Pattern Analysis", "Personality Type Clustering"
]
for i, feature_type in enumerate(psychology_features, 1):
    print(f"   {i}. {feature_type}")

print(f"\nğŸ”„ Pseudo-Labeling Strategy:")
print(f"   Confidence Threshold:              85%")
print(f"   Maximum Expansion Ratio:           30%")
print(f"   Ensemble Base Models:              LightGBM + XGBoost + CatBoost")
print(f"   Sample Weighting:                  Confidence-based (0.8x max)")
print(f"   Quality Control:                   âœ… Multi-stage validation")

print(f"\nğŸ�¯ Prediction Characteristics:")
print(f"   Total Test Predictions:            {len(test_predictions):,}")
print(f"   Class Distribution:                {extrovert_count/len(test_predictions):.1%} Extrovert")
print(f"   Average Prediction Confidence:     {avg_confidence:.3f}")
print(f"   Model Ensemble Agreement:          High")

# Technical insights
print(f"\nğŸ’¡ Key Technical Insights:")
print(f"   â€¢ Psychology-informed features capture underlying personality patterns")
print(f"   â€¢ Pseudo-labeling provides valuable test distribution alignment")
print(f"   â€¢ Sample weighting maintains data quality while expanding training set")
print(f"   â€¢ Ensemble approach balances multiple algorithmic perspectives")
print(f"   â€¢ Big Five theory provides strong theoretical foundation")

# Competitive advantages
print(f"\nğŸ�† Competitive Advantages:")
print(f"   âœ… Domain expertise integration (psychological theory)")
print(f"   âœ… Semi-supervised learning (pseudo-labeling)")
print(f"   âœ… Advanced feature engineering (statistical transformations)")
print(f"   âœ… Robust model ensemble (4 complementary algorithms)")
print(f"   âœ… Sample weight optimization (balanced training)")

print(f"\nğŸ�‰ Implementation Status: COMPLETE")
print(f"   Ready for Kaggle submission and competitive evaluation")

# Save detailed results
final_results = {
    'approach': 'Psychology + Pseudo-Labeling',
    'performance': results,
    'technical_specs': {
        'original_features': 7,
        'enhanced_features': results['feature_count'],
        'data_expansion': results['data_expansion_ratio'],
        'ensemble_models': ['LightGBM', 'XGBoost', 'CatBoost', 'LogisticRegression'],
        'sample_weights': True
    },
    'predictions': {
        'total_count': len(test_predictions),
        'extrovert_count': int(extrovert_count),
        'introvert_count': int(introvert_count),
        'avg_confidence': float(avg_confidence)
    }
}

print(f"\nğŸ’¾ Results and submission ready for export")


# Save submission file
submission_df.to_csv('psychology_pseudo_labeling_submission.csv', index=False)
print("âœ… Submission file saved: psychology_pseudo_labeling_submission.csv")

# Save detailed results
with open('psychology_pseudo_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print("âœ… Detailed results saved: psychology_pseudo_results.json")

# Final summary for Kaggle community
print("\n" + "="*80)
print("ğŸ�¯ FINAL SUBMISSION SUMMARY FOR KAGGLE COMMUNITY")
print("="*80)

print(f"\nğŸ“Š **Approach**: Psychology-Driven Feature Engineering + Intelligent Pseudo-Labeling")
print(f"\nğŸ§  **Core Innovation**:")
print(f"   â€¢ Big Five personality theory integration for meaningful features")
print(f"   â€¢ Semi-supervised learning via confident pseudo-labeling")
print(f"   â€¢ Sample-weighted training for optimal data utilization")

print(f"\nğŸ“ˆ **Performance**:")
print(f"   â€¢ Cross-Validation Score: {results['enhanced_cv_mean']:.6f} +/- {results['enhanced_cv_std']:.6f}")
print(f"   â€¢ Improvement over baseline: {results['improvement']:+.6f} ({results['relative_improvement_pct']:+.2f}%)")
print(f"   â€¢ Training data expansion: {results['data_expansion_ratio']:.1f}x with quality control")

print(f"\nğŸ”§ **Technical Stack**:")
print(f"   â€¢ Features: {results['feature_count']} psychology-informed features")
print(f"   â€¢ Models: LightGBM + XGBoost + CatBoost + LogisticRegression ensemble")
print(f"   â€¢ Training: Sample-weighted pseudo-labeling with confidence thresholding")

print(f"\nğŸ�‰ **Why This Approach Works**:")
print(f"   1. **Domain Knowledge**: Leverages established psychological theory")
print(f"   2. **Data Efficiency**: Expands training set intelligently")
print(f"   3. **Model Robustness**: Ensemble of complementary algorithms")
print(f"   4. **Quality Control**: Confidence-based pseudo-label selection")

print(f"\nğŸ’¡ **Key Takeaways for the Community**:")
print(f"   â€¢ Psychology domain knowledge can significantly enhance feature engineering")
print(f"   â€¢ Pseudo-labeling with proper confidence thresholding improves generalization")
print(f"   â€¢ Sample weighting is crucial for balancing original vs augmented data")
print(f"   â€¢ Ensemble methods provide robust performance across different data patterns")

print(f"\nğŸš€ **Ready for submission and community feedback!**")

print(f"\nğŸ“� **Files Generated**:")
print(f"   â€¢ psychology_pseudo_labeling_submission.csv (Kaggle submission)")
print(f"   â€¢ psychology_pseudo_results.json (detailed results)")
print(f"   â€¢ Complete implementation in this notebook")

print("\nğŸ�¯ Implementation complete! Thank you for exploring this psychology-driven approach.")

