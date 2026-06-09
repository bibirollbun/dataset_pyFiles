# ========================================
# COMPLETE ONE-CELL BPM IMPROVEMENT SOLUTION
# ========================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Essential imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, BayesianRidge
import matplotlib.pyplot as plt

# ML libraries with fallbacks
try:
    import xgboost as xgb
    print("âœ… XGBoost available")
except ImportError:
    print("â�Œ XGBoost not available - installing...")
    !pip install xgboost -q
    import xgboost as xgb

try:
    import lightgbm as lgb
    print("âœ… LightGBM available")
except ImportError:
    print("â�Œ LightGBM not available - installing...")
    !pip install lightgbm -q
    import lightgbm as lgb

try:
    import catboost as cb
    print("âœ… CatBoost available")
except ImportError:
    print("â�Œ CatBoost not available - installing...")
    !pip install catboost -q
    import catboost as cb

# ========================================
# 1. DATA LOADING
# ========================================

def load_data():
    """Load competition data automatically"""
    import os
    
    print("ğŸ“‚ Loading data...")
    
    # Find the data path
    data_path = None
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'train.csv' in files:
            data_path = root + '/'
            break
    
    if data_path is None:
        raise FileNotFoundError("Cannot find train.csv")
    
    print(f"âœ… Found data at: {data_path}")
    
    train = pd.read_csv(data_path + 'train.csv')
    test = pd.read_csv(data_path + 'test.csv')
    
    print(f"ğŸ“Š Train: {train.shape}, Test: {test.shape}")
    return train, test

# ========================================
# 2. FEATURE ENGINEERING
# ========================================

def create_music_features(df):
    """Create music domain features"""
    df = df.copy()
    
    print("ğŸ�µ Creating music features...")
    
    # Energy-based features
    df['EnergyDensity'] = df['Energy'] / (df['TrackDurationMs'] / 60000 + 1)
    df['EnergyMomentum'] = df['Energy'] * df['RhythmScore']
    df['EnergyBalance'] = df['Energy'] / (df['MoodScore'] + 0.1)
    
    # Acoustic features
    df['AcousticComplexity'] = df['AcousticQuality'] * df['InstrumentalScore']
    df['VocalDominance'] = df['VocalContent'] / (df['InstrumentalScore'] + df['VocalContent'] + 0.01)
    df['InstrumentalDominance'] = df['InstrumentalScore'] / (df['InstrumentalScore'] + df['VocalContent'] + 0.01)
    
    # Loudness features
    df['LoudnessRange'] = np.abs(df['AudioLoudness'])
    df['LoudnessIntensity'] = np.abs(df['AudioLoudness']) * df['Energy']
    df['QuietIndex'] = 1 / (np.abs(df['AudioLoudness']) + 1)
    
    # Duration features
    df['TrackLength_Min'] = df['TrackDurationMs'] / 60000
    df['IsShortTrack'] = (df['TrackDurationMs'] < 180000).astype(int)
    df['IsLongTrack'] = (df['TrackDurationMs'] > 300000).astype(int)
    df['DurationEnergyRatio'] = df['TrackLength_Min'] / (df['Energy'] + 0.1)
    
    # Performance features
    df['LiveEnergyProduct'] = df['LivePerformanceLikelihood'] * df['Energy']
    df['LiveAcousticProduct'] = df['LivePerformanceLikelihood'] * df['AcousticQuality']
    df['StudioPolish'] = (1 - df['LivePerformanceLikelihood']) * df['AcousticQuality']
    
    # Mood & Rhythm
    df['MoodRhythmHarmony'] = df['MoodScore'] * df['RhythmScore']
    df['EmotionalIntensity'] = df['MoodScore'] * df['Energy']
    df['RhythmicPower'] = df['RhythmScore'] ** 2
    df['MoodVariability'] = np.abs(df['MoodScore'] - 0.5)
    
    # Mathematical transformations
    df['LogDuration'] = np.log1p(df['TrackDurationMs'])
    df['LogLoudness'] = np.log1p(np.abs(df['AudioLoudness']))
    df['SqrtEnergy'] = np.sqrt(df['Energy'])
    df['Energy_squared'] = df['Energy'] ** 2
    df['RhythmScore_squared'] = df['RhythmScore'] ** 2
    
    # Key interactions
    key_features = ['Energy', 'RhythmScore', 'MoodScore', 'AcousticQuality', 'VocalContent']
    for i, feat1 in enumerate(key_features):
        for feat2 in key_features[i+1:]:
            df[f'{feat1}_{feat2}_product'] = df[feat1] * df[feat2]
            df[f'{feat1}_{feat2}_ratio'] = df[feat1] / (df[feat2] + 0.01)
    
    return df

def add_clustering(df):
    """Add clustering features"""
    df = df.copy()
    
    print("ğŸ�¯ Adding clustering...")
    
    cluster_features = ['Energy', 'RhythmScore', 'MoodScore', 'AcousticQuality', 
                       'VocalContent', 'InstrumentalScore', 'LivePerformanceLikelihood']
    
    X_cluster = df[cluster_features].fillna(0)
    
    # Create clusters
    for n_clusters in [5, 8]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_cluster)
        df[f'MusicStyle_Cluster_{n_clusters}'] = clusters
        
        distances = kmeans.transform(X_cluster)
        df[f'MinClusterDistance_{n_clusters}'] = np.min(distances, axis=1)
    
    # Create dummies for main clustering
    cluster_dummies = pd.get_dummies(df['MusicStyle_Cluster_8'], prefix='Style')
    df = pd.concat([df, cluster_dummies], axis=1)
    
    return df

# ========================================
# 3. SMART MODEL CREATION
# ========================================

def test_gpu():
    """Test if GPU is working"""
    try:
        # Test XGBoost GPU
        test_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=1)
        X_test = np.random.random((100, 5))
        y_test = np.random.random(100)
        test_model.fit(X_test, y_test)
        return True
    except:
        return False

def create_models():
    """Create models with automatic GPU/CPU detection"""
    
    gpu_available = test_gpu()
    
    if gpu_available:
        print("ğŸš€ GPU available - using acceleration")
        tree_method = 'gpu_hist'
        lgb_device = 'gpu'
        cat_task = 'GPU'
    else:
        print("ğŸ’» GPU not available - using CPU")
        tree_method = 'hist'
        lgb_device = 'cpu'
        cat_task = 'CPU'
    
    models = {}
    
    # XGBoost models
    models['XGB_1'] = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method=tree_method, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    
    models['XGB_2'] = xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        tree_method=tree_method, subsample=0.9, colsample_bytree=0.9,
        random_state=123, n_jobs=-1
    )
    
    # LightGBM models
    models['LGB_1'] = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        device=lgb_device, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    
    models['LGB_2'] = lgb.LGBMRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        device=lgb_device, subsample=0.9, colsample_bytree=0.9,
        num_leaves=50, random_state=123, verbose=-1
    )
    
    # CatBoost
    try:
        models['CAT'] = cb.CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.05,
            task_type=cat_task, random_seed=42, verbose=False
        )
    except:
        models['CAT'] = cb.CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.05,
            task_type='CPU', random_seed=42, verbose=False
        )
    
    # Tree models
    models['RF'] = RandomForestRegressor(
        n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
    )
    
    models['ET'] = ExtraTreesRegressor(
        n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
    )
    
    # Linear models
    models['Ridge'] = Ridge(alpha=1.0)
    models['BayesRidge'] = BayesianRidge()
    
    return models

# ========================================
# 4. EVALUATION & ENSEMBLE
# ========================================

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, RÂ²={r2:.4f}")
    return rmse

def create_ensemble(models, X_train, y_train, X_val, y_val, X_test):
    """Create optimized ensemble"""
    
    print("âš–ï¸� Creating ensemble...")
    
    model_scores = {}
    test_predictions = {}
    
    for name, model in models.items():
        try:
            print(f"  ğŸ”„ Training {name}...")
            model.fit(X_train, y_train)
            
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            rmse = evaluate_model(y_val, val_pred, name)
            
            model_scores[name] = rmse
            test_predictions[name] = test_pred
            
        except Exception as e:
            print(f"    â�Œ {name} failed: {str(e)[:50]}...")
            model_scores[name] = np.inf
    
    # Select top models
    valid_models = {k: v for k, v in model_scores.items() if v != np.inf}
    sorted_models = sorted(valid_models.items(), key=lambda x: x[1])
    
    print(f"\nğŸ“Š Model Rankings:")
    for name, score in sorted_models[:6]:
        print(f"   {name}: {score:.4f}")
    
    # Create weighted ensemble from top 5 models
    top_5_models = [name for name, score in sorted_models[:5]]
    
    if len(top_5_models) == 0:
        raise Exception("No models succeeded!")
    
    # Calculate weights (inverse of RMSE)
    weights = []
    for name in top_5_models:
        weight = 1 / (model_scores[name] + 1e-6)
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"\nğŸ“Š Ensemble weights:")
    for name, weight in zip(top_5_models, weights):
        print(f"   {name}: {weight:.3f}")
    
    # Create final prediction
    final_prediction = np.zeros(len(X_test))
    for name, weight in zip(top_5_models, weights):
        final_prediction += weight * test_predictions[name]
    
    return final_prediction, sorted_models

def create_submission(predictions, test_ids, filename='improved_submission.csv'):
    """Create submission file"""
    
    # Apply BPM constraints
    predictions = np.clip(predictions, 45, 200)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'BeatsPerMinute': predictions
    })
    
    submission.to_csv(filename, index=False)
    
    print(f"âœ… Submission saved: {filename}")
    print(f"ğŸ“Š BPM range: {predictions.min():.1f} - {predictions.max():.1f}")
    print(f"ğŸ“Š Mean BPM: {predictions.mean():.1f}")
    
    return submission

# ========================================
# 5. MAIN PIPELINE
# ========================================

def complete_improvement_pipeline():
    """Complete improvement pipeline - everything in one function"""
    
    print("ğŸš€ STARTING COMPLETE BPM IMPROVEMENT PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    print("\nğŸ“‚ PHASE 1: Data Loading")
    train, test = load_data()
    
    # 2. Feature engineering
    print("\nğŸ”§ PHASE 2: Feature Engineering")
    train_features = create_music_features(train)
    test_features = create_music_features(test)
    
    train_features = add_clustering(train_features)
    test_features = add_clustering(test_features)
    
    # Prepare features
    feature_cols = [col for col in train_features.columns 
                   if col not in ['id', 'BeatsPerMinute']]
    
    X = train_features[feature_cols].fillna(0)
    y = train_features['BeatsPerMinute']
    X_test = test_features[feature_cols].fillna(0)
    
    print(f"âœ… Created {len(feature_cols)} features")
    
    # 3. Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"ğŸ“Š Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # 4. Create models
    print("\nğŸ¤– PHASE 3: Model Creation")
    models = create_models()
    
    # 5. Create ensemble
    print("\nğŸ�—ï¸� PHASE 4: Ensemble Training")
    final_prediction, model_rankings = create_ensemble(
        models, X_train.values, y_train, X_val.values, y_val, X_test.values
    )
    
    # 6. Create submission
    print("\nğŸ“„ PHASE 5: Submission Creation")
    submission = create_submission(final_prediction, test_features['id'])
    
    print(f"\nğŸ�‰ PIPELINE COMPLETE!")
    print(f"ğŸ�¯ Features: {len(feature_cols)}")
    print(f"ğŸ¤– Models trained: {len([m for m in model_rankings if m[1] != np.inf])}")
    print(f"ğŸ“„ Submission ready: improved_submission.csv")
    
    return {
        'submission': submission,
        'model_rankings': model_rankings,
        'feature_count': len(feature_cols)
    }

# ========================================
# 6. EXECUTE PIPELINE
# ========================================

print("ğŸ�¶ COMPLETE BPM IMPROVEMENT PIPELINE READY!")
print("=" * 60)
print("ğŸš€ Running pipeline automatically...")
print("â�±ï¸� Estimated time: 5-10 minutes")

# Run the complete pipeline
try:
    results = complete_improvement_pipeline()
    print("\nğŸ�‰ SUCCESS! Check your improved_submission.csv file!")
    
except Exception as e:
    print(f"\nâ�Œ Pipeline failed: {e}")
    print("Please check the error and try again")

