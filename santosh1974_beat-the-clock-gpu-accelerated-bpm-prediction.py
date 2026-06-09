import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train and test data"""
    train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    return train, test, sample_submission

def create_engineered_features(df):
    """Create new features based on EDA insights"""
    df = df.copy()
    
    # Ratio features (based on musical domain knowledge)
    df['EnergyLoudnessRatio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 1e-6)
    df['VocalInstrumentalRatio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-6)
    df['RhythmEnergyProduct'] = df['RhythmScore'] * df['Energy']
    df['MoodEnergyProduct'] = df['MoodScore'] * df['Energy']
    
    # Duration-based features
    df['TrackDurationSec'] = df['TrackDurationMs'] / 1000
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
    
    # Log transformations for skewed features
    df['LogTrackDuration'] = np.log1p(df['TrackDurationMs'])
    df['LogAudioLoudness'] = np.log1p(np.abs(df['AudioLoudness']) + 1)
    
    # Interaction features (experiment with top correlated features from EDA)
    df['RhythmMoodInteraction'] = df['RhythmScore'] * df['MoodScore']
    df['EnergyAcousticInteraction'] = df['Energy'] * df['AcousticQuality']
    
    # Binning features (if helpful from EDA)
    df['EnergyBin'] = pd.cut(df['Energy'], bins=5, labels=['VeryLow', 'Low', 'Med', 'High', 'VeryHigh'])
    df['LoudnessBin'] = pd.cut(df['AudioLoudness'], bins=5, labels=['VeryQuiet', 'Quiet', 'Med', 'Loud', 'VeryLoud'])
    
    # One-hot encode categorical bins
    energy_dummies = pd.get_dummies(df['EnergyBin'], prefix='Energy')
    loudness_dummies = pd.get_dummies(df['LoudnessBin'], prefix='Loudness')
    
    df = pd.concat([df, energy_dummies, loudness_dummies], axis=1)
    df.drop(['EnergyBin', 'LoudnessBin'], axis=1, inplace=True)
    
    # Polynomial features for top predictors (use sparingly!)
    df['RhythmScore_squared'] = df['RhythmScore'] ** 2
    df['Energy_squared'] = df['Energy'] ** 2
    
    print(f"Created {df.shape[1] - len(['id', 'BeatsPerMinute'])} features total")
    
    return df

def handle_outliers(df, target_col='BeatsPerMinute', method='clip'):
    """Handle outliers in features and target"""
    df = df.copy()
    
    # Handle target outliers
    if target_col in df.columns:
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'clip':
            df[target_col] = df[target_col].clip(lower_bound, upper_bound)
        elif method == 'remove':
            df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
    
    # Handle feature outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['id', target_col]]
    
    for col in feature_cols:
        Q1 = df[col].quantile(0.05)  # More conservative
        Q3 = df[col].quantile(0.95)
        df[col] = df[col].clip(Q1, Q3)
    
    return df

def scale_features(X_train, X_test, method='standard'):
    """Scale features using different methods"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def select_features(X_train, y_train, X_test, method='mutual_info', k=20):
    """Feature selection"""
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    if hasattr(X_train, 'columns'):
        selected_features = X_train.columns[selector.get_support()].tolist()
        print(f"Selected features: {selected_features}")
    
    return X_train_selected, X_test_selected, selector

def prepare_data_for_modeling(train, test):
    """Complete data preparation pipeline"""
    print("Starting data preparation...")
    
    # Create engineered features
    train_eng = create_engineered_features(train)
    test_eng = create_engineered_features(test)
    
    # Handle outliers
    train_clean = handle_outliers(train_eng)
    
    # Prepare features and target
    feature_cols = [col for col in train_clean.columns if col not in ['id', 'BeatsPerMinute']]
    
    X = train_clean[feature_cols]
    y = train_clean['BeatsPerMinute']
    X_test = test_eng[feature_cols]
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val, method='robust')
    X_test_scaled = scaler.transform(X_test)
    
    # Optional: Feature selection
    # X_train_selected, X_val_selected, selector = select_features(
    #     pd.DataFrame(X_train_scaled, columns=feature_cols), y_train, 
    #     pd.DataFrame(X_val_scaled, columns=feature_cols), k=15
    # )
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled, 
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'test_ids': test_eng['id']
    }

# Example usage
if __name__ == "__main__":
    # Load data
    train, test, sample_submission = load_data()
    
    # Prepare data
    data = prepare_data_for_modeling(train, test)
    
    print("âœ… Data preparation complete!")
    print(f"Ready for modeling with {data['X_train'].shape[1]} features")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def check_gpu_availability():
    """Check if GPU is available for XGBoost and LightGBM"""
    print("ğŸ”� Checking GPU availability...")
    
    # Check XGBoost GPU
    try:
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
        print(f"XGBoost GPU support: {xgb.rabit.get_processor_name()}")
    except:
        print("XGBoost GPU check failed")
    
    # Check LightGBM GPU
    try:
        import lightgbm as lgb
        print(f"LightGBM version: {lgb.__version__}")
        # Test GPU device
        lgb_test = lgb.LGBMRegressor(device='gpu', objective='regression', verbose=-1)
        print("âœ… LightGBM GPU support available")
    except Exception as e:
        print(f"â�Œ LightGBM GPU not available: {e}")
        print("Will use CPU version")
    
    return True

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Comprehensive model evaluation"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RÂ²:   {r2:.4f}")
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def gpu_optimized_model_comparison(X_train, y_train, X_val, y_val):
    """GPU-optimized model comparison"""
    
    print("=" * 60)
    print("GPU-OPTIMIZED MODEL COMPARISON")
    print("=" * 60)
    
    # Check GPU availability first
    check_gpu_availability()
    
    models = {}
    
    # GPU-accelerated XGBoost
    try:
        models['XGBoost_GPU'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method='gpu_hist',  # GPU acceleration
            gpu_id=0,
            random_state=42,
            n_jobs=-1
        )
        print("âœ… XGBoost GPU model added")
    except:
        models['XGBoost_CPU'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',  # CPU fallback
            random_state=42,
            n_jobs=-1
        )
        print("âš ï¸� XGBoost using CPU fallback")
    
    # GPU-accelerated LightGBM
    try:
        models['LightGBM_GPU'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            device='gpu',  # GPU acceleration
            gpu_platform_id=0,
            gpu_device_id=0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        print("âœ… LightGBM GPU model added")
    except:
        models['LightGBM_CPU'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            device='cpu',  # CPU fallback
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        print("âš ï¸� LightGBM using CPU fallback")
    
    # Fast CPU models (still good performers)
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15,
        random_state=42, 
        n_jobs=-1
    )
    
    models['ExtraTrees'] = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=15, 
        random_state=42, 
        n_jobs=-1
    )
    
    models['Ridge'] = Ridge(alpha=1.0)
    models['ElasticNet'] = ElasticNet(alpha=1.0, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nğŸ”„ Training {name}...")
        try:
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Evaluate
            metrics = evaluate_model(y_val, y_pred, name)
            results[name] = metrics
            
        except Exception as e:
            print(f"\nâ�Œ {name} failed: {e}")
            results[name] = {'MAE': np.inf, 'MSE': np.inf, 'RMSE': np.inf, 'R2': -np.inf}
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T.sort_values('RMSE')
    
    print(f"\nğŸ“Š MODEL RANKING (by RMSE):")
    print("=" * 50)
    print(results_df.round(4))
    
    return results_df, models

def gpu_hyperparameter_tuning(X_train, y_train, X_val, y_val, top_models):
    """GPU-accelerated hyperparameter tuning"""
    
    print("\n" + "=" * 60)
    print("GPU-ACCELERATED HYPERPARAMETER TUNING")
    print("=" * 60)
    
    tuned_models = {}
    
    # XGBoost GPU tuning
    if any('XGBoost' in model for model in top_models):
        print("\nğŸ”§ Tuning XGBoost with GPU...")
        
        try:
            xgb_params = {
                'n_estimators': [200, 400, 600],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
            
            xgb_model = xgb.XGBRegressor(
                tree_method='gpu_hist',
                gpu_id=0,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_search = RandomizedSearchCV(
                xgb_model, xgb_params, n_iter=15, cv=3,
                scoring='neg_mean_squared_error', random_state=42, n_jobs=1  # n_jobs=1 for GPU
            )
            xgb_search.fit(X_train, y_train)
            tuned_models['XGBoost_GPU_Tuned'] = xgb_search.best_estimator_
            print(f"âœ… Best XGB params: {xgb_search.best_params_}")
            
        except Exception as e:
            print(f"âš ï¸� XGBoost GPU tuning failed: {e}")
            # CPU fallback
            xgb_model = xgb.XGBRegressor(tree_method='hist', random_state=42, n_jobs=-1)
            xgb_search = RandomizedSearchCV(
                xgb_model, xgb_params, n_iter=10, cv=3,
                scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
            )
            xgb_search.fit(X_train, y_train)
            tuned_models['XGBoost_CPU_Tuned'] = xgb_search.best_estimator_
    
    # LightGBM GPU tuning
    if any('LightGBM' in model for model in top_models):
        print("\nğŸ”§ Tuning LightGBM with GPU...")
        
        try:
            lgb_params = {
                'n_estimators': [200, 400, 600],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'num_leaves': [31, 50, 70]
            }
            
            lgb_model = lgb.LGBMRegressor(
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                random_state=42,
                verbose=-1
            )
            
            lgb_search = RandomizedSearchCV(
                lgb_model, lgb_params, n_iter=15, cv=3,
                scoring='neg_mean_squared_error', random_state=42, n_jobs=1  # n_jobs=1 for GPU
            )
            lgb_search.fit(X_train, y_train)
            tuned_models['LightGBM_GPU_Tuned'] = lgb_search.best_estimator_
            print(f"âœ… Best LGB params: {lgb_search.best_params_}")
            
        except Exception as e:
            print(f"âš ï¸� LightGBM GPU tuning failed: {e}")
            # CPU fallback
            lgb_model = lgb.LGBMRegressor(device='cpu', random_state=42, verbose=-1)
            lgb_search = RandomizedSearchCV(
                lgb_model, lgb_params, n_iter=10, cv=3,
                scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
            )
            lgb_search.fit(X_train, y_train)
            tuned_models['LightGBM_CPU_Tuned'] = lgb_search.best_estimator_
    
    # Evaluate tuned models
    print(f"\nğŸ“ˆ TUNED MODEL PERFORMANCE:")
    print("=" * 40)
    
    tuned_results = {}
    for name, model in tuned_models.items():
        y_pred = model.predict(X_val)
        metrics = evaluate_model(y_val, y_pred, name)
        tuned_results[name] = metrics
    
    return tuned_models, tuned_results

def create_fast_ensemble(models, X_train, y_train, X_val, y_val, X_test):
    """Fast ensemble creation with GPU models"""
    
    print("\n" + "=" * 60)
    print("FAST GPU ENSEMBLE")
    print("=" * 60)
    
    val_predictions = []
    test_predictions = []
    model_weights = []
    
    for name, model in models.items():
        print(f"ğŸ”„ Getting predictions from {name}...")
        
        # Ensure model is fitted
        model.fit(X_train, y_train)
        
        # Get predictions
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate weight based on performance (inverse of RMSE)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        weight = 1 / (rmse + 1e-6)
        
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        model_weights.append(weight)
        
        print(f"   RMSE: {rmse:.4f}, Weight: {weight:.4f}")
    
    # Normalize weights
    model_weights = np.array(model_weights)
    model_weights = model_weights / model_weights.sum()
    
    # Weighted ensemble
    ensemble_val = np.average(val_predictions, axis=0, weights=model_weights)
    ensemble_test = np.average(test_predictions, axis=0, weights=model_weights)
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_model(y_val, ensemble_val, "Weighted GPU Ensemble")
    
    print(f"\nğŸ�¯ Model weights: {dict(zip(models.keys(), model_weights.round(3)))}")
    
    return ensemble_test, ensemble_val

def create_submission(predictions, test_ids, filename='gpu_submission.csv'):
    """Create submission file"""
    submission = pd.DataFrame({
        'id': test_ids,
        'BeatsPerMinute': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"âœ… Submission saved as '{filename}'")
    print(f"Submission shape: {submission.shape}")
    print(f"BPM range: {submission['BeatsPerMinute'].min():.2f} - {submission['BeatsPerMinute'].max():.2f}")
    
    return submission

def gpu_modeling_pipeline(data):
    """Complete GPU-optimized modeling pipeline"""
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    test_ids = data['test_ids']
    
    print("ğŸš€ Starting GPU-accelerated modeling pipeline...")
    
    # 1. GPU model comparison
    results_df, models = gpu_optimized_model_comparison(X_train, y_train, X_val, y_val)
    
    # 2. Get top 2 models for tuning (faster)
    top_2_models = results_df.head(2).index.tolist()
    print(f"\nğŸ�† Top 2 models for tuning: {top_2_models}")
    
    # 3. GPU hyperparameter tuning
    tuned_models, tuned_results = gpu_hyperparameter_tuning(X_train, y_train, X_val, y_val, top_2_models)
    
    # 4. Create ensemble with best models
    best_models = {}
    
    # Add original best models
    for name in top_2_models:
        if name in models:
            best_models[name] = models[name]
    
    # Add tuned models
    for name, model in tuned_models.items():
        best_models[name] = model
    
    # 5. Fast ensemble
    ensemble_test, ensemble_val = create_fast_ensemble(best_models, X_train, y_train, X_val, y_val, X_test)
    
    # 6. Create submission
    create_submission(ensemble_test, test_ids, 'gpu_ensemble_submission.csv')
    
    return {
        'models': models,
        'tuned_models': tuned_models,
        'results': results_df,
        'ensemble_predictions': ensemble_test,
        'best_models': best_models
    }

if __name__ == "__main__":
    print("ğŸš€ GPU-Optimized modeling ready!")
    print("Run: gpu_results = gpu_modeling_pipeline(data)")


gpu_results = gpu_modeling_pipeline(data)




