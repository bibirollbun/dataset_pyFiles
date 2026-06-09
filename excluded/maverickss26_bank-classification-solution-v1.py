# Kaggle Playground Series S5E8 - Advanced Binary Classification Solution
# Target: Beat 0.97568 accuracy with state-of-the-art techniques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy.stats import skew, kurtosis
from scipy.special import boxcox1p
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("ğŸš€ Starting Kaggle S5E8 Advanced Solution")
print("Target: Achieve >97.5% accuracy through advanced ensemble techniques")

# ===============================
# 1. DATA LOADING & EXPLORATION
# ===============================

def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    
    # Load data
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
    
    print(f"ğŸ“Š Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"ğŸ�¯ Target distribution:\n{train_df['y'].value_counts(normalize=True)}")
    
    # Display basic info
    print("\nğŸ“‹ Dataset Info:")
    print(train_df.info())
    print("\nğŸ“ˆ Statistical Summary:")
    print(train_df.describe())
    
    # Check for missing values
    print(f"\nâ�“ Missing values in train: {train_df.isnull().sum().sum()}")
    print(f"â�“ Missing values in test: {test_df.isnull().sum().sum()}")
    
    return train_df, test_df

# ===============================
# 2. ADVANCED FEATURE ENGINEERING
# ===============================

class AdvancedFeatureEngineer:
    """Advanced feature engineering for bank marketing dataset"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        
    def create_interaction_features(self, df):
        """Create sophisticated interaction features"""
        df_new = df.copy()
        
        # Age-based interactions
        if 'age' in df.columns:
            df_new['age_squared'] = df['age'] ** 2
            df_new['age_cubed'] = df['age'] ** 3
            df_new['age_log'] = np.log1p(df['age'])
            
            # Age groups
            df_new['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                                       labels=['young', 'adult', 'middle', 'senior', 'elderly'])
        
        # Duration interactions (if exists)
        if 'duration' in df.columns:
            df_new['duration_log'] = np.log1p(df['duration'])
            df_new['duration_squared'] = df['duration'] ** 2
            
        # Campaign interactions
        if 'campaign' in df.columns:
            df_new['campaign_log'] = np.log1p(df['campaign'])
            df_new['is_first_contact'] = (df['campaign'] == 1).astype(int)
            df_new['high_campaign'] = (df['campaign'] > 3).astype(int)
        
        # Previous outcome interactions
        if 'previous' in df.columns:
            df_new['has_previous'] = (df['previous'] > 0).astype(int)
            df_new['previous_log'] = np.log1p(df['previous'])
        
        # Economic indicators interactions
        economic_cols = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        available_econ_cols = [col for col in economic_cols if col in df.columns]
        
        if len(available_econ_cols) >= 2:
            # Economic sentiment score
            df_new['economic_sentiment'] = 0
            for col in available_econ_cols:
                if 'conf' in col or 'var' in col:
                    df_new['economic_sentiment'] += df[col]
                    
            # Employment vs confidence ratio
            if 'nr.employed' in df.columns and 'cons.conf.idx' in df.columns:
                df_new['employment_confidence_ratio'] = df['nr.employed'] / (df['cons.conf.idx'] + 1e-8)
        
        return df_new
    
    def create_statistical_features(self, df):
        """Create statistical features from numerical columns"""
        df_new = df.copy()
        
        # Get numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'y' in numeric_cols:
            numeric_cols.remove('y')
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
            
        if len(numeric_cols) >= 2:
            # Statistical aggregations
            df_new['numeric_mean'] = df[numeric_cols].mean(axis=1)
            df_new['numeric_std'] = df[numeric_cols].std(axis=1)
            df_new['numeric_median'] = df[numeric_cols].median(axis=1)
            df_new['numeric_max'] = df[numeric_cols].max(axis=1)
            df_new['numeric_min'] = df[numeric_cols].min(axis=1)
            df_new['numeric_range'] = df_new['numeric_max'] - df_new['numeric_min']
            df_new['numeric_skew'] = df[numeric_cols].skew(axis=1)
            df_new['numeric_kurt'] = df[numeric_cols].kurtosis(axis=1)
            
            # Feature interactions
            for i, col1 in enumerate(numeric_cols[:3]):  # Limit to avoid explosion
                for col2 in numeric_cols[i+1:4]:
                    df_new[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
                    df_new[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
                    df_new[f'{col1}_{col2}_sum'] = df[col1] + df[col2]
        
        return df_new
    
    def handle_categorical_advanced(self, df, target=None, is_train=True):
        """Advanced categorical encoding techniques"""
        df_new = df.copy()
        
        # Get categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in cat_cols:
            # Basic label encoding
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_new[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_new[col].astype(str))
            else:
                # Handle unseen categories
                categories = self.label_encoders[col].classes_
                df_new[f'{col}_encoded'] = df_new[col].apply(
                    lambda x: self.label_encoders[col].transform([str(x)])[0] 
                    if str(x) in categories else -1
                )
            
            # Frequency encoding - store frequency map for test data
            if is_train:
                freq_map = df_new[col].value_counts().to_dict()
                self.freq_maps = getattr(self, 'freq_maps', {})
                self.freq_maps[col] = freq_map
                df_new[f'{col}_freq'] = df_new[col].map(freq_map)
            else:
                # Use stored frequency map from training
                freq_map = getattr(self, 'freq_maps', {}).get(col, {})
                df_new[f'{col}_freq'] = df_new[col].map(freq_map).fillna(0)
            
            # Target encoding (only for training data)
            if target is not None and is_train:
                target_mean = df_new.groupby(col)[target].mean()
                df_new[f'{col}_target_mean'] = df_new[col].map(target_mean)
                
                # Store target encodings for test data
                self.target_encodings = getattr(self, 'target_encodings', {})
                self.target_encodings[f'{col}_target_mean'] = target_mean.to_dict()
                
                # Smoothed target encoding
                global_mean = df_new[target].mean()
                counts = df_new[col].value_counts()
                smooth_factor = 10
                
                smoothed_target = {}
                for category in df_new[col].unique():
                    cat_mean = target_mean.get(category, global_mean)
                    cat_count = counts.get(category, 0)
                    smoothed_target[category] = (cat_mean * cat_count + global_mean * smooth_factor) / (cat_count + smooth_factor)
                
                df_new[f'{col}_target_smooth'] = df_new[col].map(smoothed_target)
                self.target_encodings[f'{col}_target_smooth'] = smoothed_target
                
            elif not is_train:
                # Apply stored target encodings to test data
                target_encodings = getattr(self, 'target_encodings', {})
                if f'{col}_target_mean' in target_encodings:
                    df_new[f'{col}_target_mean'] = df_new[col].map(target_encodings[f'{col}_target_mean']).fillna(0.5)
                if f'{col}_target_smooth' in target_encodings:
                    df_new[f'{col}_target_smooth'] = df_new[col].map(target_encodings[f'{col}_target_smooth']).fillna(0.5)
        
        return df_new
    
    def feature_selection_advanced(self, X, y, method='hybrid'):
        """Elite feature selection for maximum performance"""
        
        # Remove non-numeric columns for feature selection
        X_numeric = X.select_dtypes(include=[np.number])
        
        selected_features = set()
        
        # Method 1: Univariate selection
        selector = SelectKBest(f_classif, k=min(60, X_numeric.shape[1]))
        selector.fit(X_numeric, y)
        mask = selector.get_support()
        selected_features.update(X_numeric.columns[mask])
        
        # Method 2: Tree-based selection with XGBoost
        xgb_selector = xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        selector = SelectFromModel(xgb_selector, threshold='0.75*median')
        selector.fit(X_numeric, y)
        mask = selector.get_support()
        selected_features.update(X_numeric.columns[mask])
        
        # Method 3: LightGBM-based selection
        lgb_selector = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        selector = SelectFromModel(lgb_selector, threshold='0.75*median')
        selector.fit(X_numeric, y)
        mask = selector.get_support()
        selected_features.update(X_numeric.columns[mask])
        
        # Ensure minimum features
        if len(selected_features) < 30:
            # Add top univariate features
            selector = SelectKBest(f_classif, k=30)
            selector.fit(X_numeric, y)
            mask = selector.get_support()
            selected_features.update(X_numeric.columns[mask])
        
        return list(selected_features)

# ===============================
# 3. ADVANCED MODELS & HYPERPARAMETER OPTIMIZATION
# ===============================

class EliteModelOptimizer:
    """Elite optimization for XGBoost and LightGBM targeting 0.97586+"""
    
    def __init__(self):
        self.best_params = {}
        self.best_scores = {}
    
    def optimize_xgboost(self, X, y, n_trials=50):
        """Elite XGBoost optimization targeting 0.97586+"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1200),
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample': trial.suggest_float('subsample', 0.75, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.75, 0.95),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.75, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 15),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'tree_method': 'hist'
            }
            
            model = xgb.XGBClassifier(**params)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params['xgboost'] = study.best_params
        return study.best_params, study.best_value
    
    def optimize_lightgbm(self, X, y, n_trials=50):
        """Elite LightGBM optimization targeting 0.97586+"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1200),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample': trial.suggest_float('subsample', 0.75, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 15),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.75, 0.95),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.75, 0.95),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_split_gain': trial.suggest_float('min_split_gain', 0, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 50, 300),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                'objective': 'binary',
                'metric': 'binary_logloss'
            }
            
            model = lgb.LGBMClassifier(**params)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params['lightgbm'] = study.best_params
        return study.best_params, study.best_value

# ===============================
# 4. ADVANCED ENSEMBLE TECHNIQUES
# ===============================

class EliteEnsemble:
    """Elite two-model ensemble for beating 0.97586"""
    
    def __init__(self):
        self.models = {}
        self.weights = None
        self.ensemble_score = None
        
    def create_elite_models(self, best_params):
        """Create elite XGBoost and LightGBM models"""
        
        models = {}
        
        # Elite XGBoost
        if 'xgboost' in best_params:
            models['xgb'] = xgb.XGBClassifier(**best_params['xgboost'])
        else:
            # Champion-level default parameters
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=800, 
                max_depth=8, 
                learning_rate=0.08,
                subsample=0.85, 
                colsample_bytree=0.85,
                colsample_bylevel=0.85,
                reg_alpha=5,
                reg_lambda=8,
                min_child_weight=3,
                gamma=0.2,
                random_state=42, 
                n_jobs=-1,
                tree_method='hist'
            )
        
        # Elite LightGBM
        if 'lightgbm' in best_params:
            models['lgb'] = lgb.LGBMClassifier(**best_params['lightgbm'])
        else:
            # Champion-level default parameters
            models['lgb'] = lgb.LGBMClassifier(
                n_estimators=800,
                max_depth=10,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=5,
                reg_lambda=8,
                min_child_samples=20,
                num_leaves=150,
                feature_fraction=0.85,
                bagging_fraction=0.85,
                bagging_freq=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        return models
    
    def elite_ensemble(self, models, X_train, y_train, X_test):
        """Elite ensemble of XGBoost and LightGBM with optimized blending"""
        
        # Store individual predictions and scores
        predictions = {}
        test_predictions = {}
        
        for name, model in models.items():
            print(f"Training elite {name}...")
            model.fit(X_train, y_train)
            
            # 5-fold CV for accurate weighting
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            predictions[name] = cv_scores.mean()
            print(f"{name} CV accuracy: {cv_scores.mean():.6f}")
            
            # Test predictions
            test_predictions[name] = model.predict_proba(X_test)[:, 1]
        
        # Advanced weighting - exponential weighting favors better models
        scores = np.array(list(predictions.values()))
        # Amplify differences between models
        exp_scores = np.exp((scores - scores.min()) * 10)
        weights_array = exp_scores / exp_scores.sum()
        
        weights = dict(zip(predictions.keys(), weights_array))
        print(f"ğŸ“Š Model weights: {weights}")
        
        # Weighted ensemble
        final_predictions = np.zeros(X_test.shape[0])
        for name, weight in weights.items():
            final_predictions += weight * test_predictions[name]
        
        # Additional blending strategies
        # Strategy 1: Simple average
        avg_predictions = np.mean(list(test_predictions.values()), axis=0)
        
        # Strategy 2: Rank averaging (more robust)
        from scipy.stats import rankdata
        rank_predictions = np.zeros(X_test.shape[0])
        for pred in test_predictions.values():
            rank_predictions += rankdata(pred) / len(pred)
        rank_predictions = rank_predictions / len(test_predictions)
        
        # Final blend: 50% weighted, 30% average, 20% rank
        final_blend = (0.5 * final_predictions + 
                      0.3 * avg_predictions + 
                      0.2 * rank_predictions)
        
        self.weights = weights
        self.ensemble_score = np.mean(list(predictions.values()))
        
        return final_blend, predictions

# ===============================
# 5. MAIN EXECUTION PIPELINE
# ===============================

def main_pipeline():
    """Main execution pipeline"""
    
    print("ğŸ“� Loading data...")
    train_df, test_df = load_and_explore_data()
    
    # Prepare target
    y = train_df['y']
    
    # Initialize feature engineer
    fe = AdvancedFeatureEngineer()
    
    print("ğŸ”§ Advanced feature engineering...")
    
    # Apply feature engineering to training data
    train_processed = fe.create_interaction_features(train_df)
    train_processed = fe.create_statistical_features(train_processed)
    train_processed = fe.handle_categorical_advanced(train_processed, target='y', is_train=True)
    
    # Apply same transformations to test data
    test_processed = fe.create_interaction_features(test_df)
    test_processed = fe.create_statistical_features(test_processed)
    test_processed = fe.handle_categorical_advanced(test_processed, is_train=False)
    
    # Remove non-feature columns and get common features
    train_feature_cols = [col for col in train_processed.columns 
                         if col not in ['id', 'y'] and train_processed[col].dtype in [np.number]]
    test_feature_cols = [col for col in test_processed.columns 
                        if col not in ['id'] and test_processed[col].dtype in [np.number]]
    
    X_train = train_processed[train_feature_cols]
    X_test = test_processed[test_feature_cols]
    
    # Ensure both train and test have same columns
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    # Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test
    
    print(f"ğŸ�¯ Final feature count: {X_train.shape[1]}")
    
    # Feature selection
    print("ğŸ”� Quick feature selection...")
    selected_features = fe.feature_selection_advanced(X_train, y, method='univariate')
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Elite feature selection
    print("ğŸ”� Elite feature selection...")
    selected_features = fe.feature_selection_advanced(X_train, y, method='hybrid')
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"ğŸ“Š Selected features: {len(selected_features)}")
    
    # Elite model optimization
    print("âš¡ Elite hyperparameter optimization (targeting 0.97586+)...")
    optimizer = EliteModelOptimizer()
    
    # Comprehensive optimization for both models
    print("Optimizing XGBoost (50 trials)...")
    xgb_params, xgb_score = optimizer.optimize_xgboost(X_train_selected, y, n_trials=50)
    print(f"XGBoost best CV score: {xgb_score:.6f}")
    
    print("Optimizing LightGBM (50 trials)...")
    lgb_params, lgb_score = optimizer.optimize_lightgbm(X_train_selected, y, n_trials=50)
    print(f"LightGBM best CV score: {lgb_score:.6f}")
    
    # Create and train elite ensemble
    print("ğŸš€ Creating elite ensemble...")
    elite_ensemble = EliteEnsemble()
    models = elite_ensemble.create_elite_models(optimizer.best_params)
    
    # Elite ensemble with advanced blending
    print("Building elite ensemble with advanced blending...")
    final_predictions, individual_scores = elite_ensemble.elite_ensemble(
        models, X_train_selected, y, X_test_selected
    )
    
    print(f"ğŸ�¯ Ensemble average CV: {elite_ensemble.ensemble_score:.6f}")
    print(f"ğŸ�¯ Target: Beat 0.97586")
    
    # Convert to binary predictions with optimized threshold
    # Find optimal threshold using cross-validation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    
    # Quick threshold optimization
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    # Use one model for threshold tuning
    xgb_model = models['xgb']
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for threshold in thresholds:
        scores = []
        for train_idx, val_idx in skf.split(X_train_selected, y):
            X_fold_train, X_fold_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            xgb_model.fit(X_fold_train, y_fold_train)
            val_preds = xgb_model.predict_proba(X_fold_val)[:, 1]
            val_binary = (val_preds > threshold).astype(int)
            scores.append(accuracy_score(y_fold_val, val_binary))
        
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_threshold = threshold
    
    print(f"ğŸ�¯ Optimal threshold: {best_threshold:.3f} (CV accuracy: {best_score:.6f})")
    
    binary_predictions = (final_predictions > best_threshold).astype(int)
    
    print(f"ğŸ�† Elite Ensemble CV Score: {elite_ensemble.ensemble_score:.6f}")
    print(f"ğŸ�¯ Individual model scores: {individual_scores}")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'y': final_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("âœ… Submission saved as 'submission.csv'")
    
    # Display feature importance from best model
    print("\nğŸ”� Top 20 Feature Importances (XGBoost):")
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': models['xgb'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20))
    print(f"ğŸ�¯ Target accuracy: >97.5% (Current best: 97.568%)")
    
    return submission, feature_importance

# Run the pipeline
if __name__ == "__main__":
    submission, feature_imp = main_pipeline()
    
    print("\nğŸ�† ELITE SOLUTION TARGETING 0.97586+ COMPLETE!")
    print("Championship-level approach:")
    print("âœ“ XGBoost + LightGBM elite duo")
    print("âœ“ 50 trials each with comprehensive parameter space")
    print("âœ“ Advanced 3-strategy ensemble blending")
    print("âœ“ Hybrid feature selection (univariate + tree-based)")
    print("âœ“ Optimized prediction threshold")
    print("âœ“ 5-fold CV for robust validation")
    print("âœ“ TARGET: Beat 0.97586 â†’ Achieve 0.976+")
    print(f"âœ“ Model weights: {elite_ensemble.weights}")
    print(f"âœ“ Ensemble score: {elite_ensemble.ensemble_score:.6f}")

