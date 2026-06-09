import scipy as sp
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from concurrent.futures import ThreadPoolExecutor
import warnings
import time
import gc
import threading
import scipy as sp
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import time
import gc
import threading
USE_COL = 'FFV'
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


Train  = pd.read_pickle('/kaggle/input/polymers-prepare-data/train.pkl')
Train.index=Train['SMILES'].values
X=Train[~Train[USE_COL].isna()]
y=X[USE_COL]
X = X.iloc[:,7:]
X=X.replace(-np.inf,np.nan)
X=X.replace(np.inf,np.nan)
X=X.fillna(X.mean())
X.shape,y.shape


import scipy as sp
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import time
import gc

warnings.filterwarnings('ignore')

def calculate_zscore(hits, trials):
    """Calculate Z-score for binomial test."""
    expected = trials * 0.5
    std = np.sqrt(trials * 0.25)
    return (hits - expected) / std

def choose_features_zscore(feature_zscores, z_threshold=1.96):
    """Classify features based on Z-scores."""
    green_zone = [key for key, z in feature_zscores.items() if z > z_threshold]
    blue_zone = [key for key, z in feature_zscores.items() if 0 < z <= z_threshold]
    return green_zone, blue_zone

class BorutaShapLGBMConfig:
    """Configuration class for Boruta-SHAP with LightGBM GPU."""
    
    def __init__(self, 
                 trials=20,                  # Number of Boruta trials
                 cv_folds=3,                 # Number of cross-validation folds
                 z_threshold=1.96,           # Z-score threshold for feature selection
                 seed=2024,                  # Random seed
                 test_size=0.2,              # Hold-out test size
                 # LightGBM parameters
                 n_estimators=1000,          # Max number of trees (early stopping will decide actual)
                 max_depth=6,                # Max depth of trees
                 learning_rate=0.05,         # Lower for early stopping
                 subsample=0.8,              # Row subsampling
                 colsample_bytree=0.8,       # Column subsampling
                 reg_alpha=0.1,              # L1 regularization
                 reg_lambda=0.1,             # L2 regularization
                 min_child_samples=20,       # Minimum samples in leaf
                 # Early stopping parameters
                 early_stopping_rounds=50,   # Patience for early stopping
                 eval_metric='mae',          # Metric for early stopping
                 # GPU parameters
                 gpu_device_id=0,            # GPU device ID
                 gpu_use_dp=False,           # Use double precision (False = float32)
                 verbose=True):              # Print progress
        
        self.trials = trials
        self.cv_folds = cv_folds
        self.z_threshold = z_threshold
        self.seed = seed
        self.test_size = test_size
        # LightGBM params
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        # Early stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        # GPU
        self.gpu_device_id = gpu_device_id
        self.gpu_use_dp = gpu_use_dp
        self.verbose = verbose
        
        # Validate inputs
        assert 0 < self.test_size < 1, "test_size must be between 0 and 1"
        assert self.trials > 0, "trials must be positive"
        assert self.cv_folds >= 2, "cv_folds must be at least 2"

def _compute_shap_importance_cv(X_train, X_test, y_train, y_test, model_params, cv_folds, 
                                early_stopping_rounds, verbose):
    """Compute SHAP-based feature importance using cross-validation."""
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=model_params['random_state'])
    shap_values_list = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # Split data for this fold
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train model with early stopping
        model = lgb.train(

            model_params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds,verbose=False),
                lgb.log_evaluation(0)  # Suppress iteration logs
            ]
        )
        
        # Get SHAP values using native LightGBM method
        shap_values_with_base = model.predict(X_test, pred_contrib=True)
        # Remove base value (last column)
        shap_values = shap_values_with_base[:, :-1]
        shap_values_list.append(shap_values)
        
        if verbose and fold_idx == 0:
            print(f"    Fold {fold_idx+1}: Stopped at iteration {model.current_iteration()}")
    
    # Average SHAP values across folds
    shap_values_avg = np.mean(np.array(shap_values_list), axis=0)
    # Calculate feature importance as mean absolute SHAP value
    feature_importance = np.abs(shap_values_avg).mean(axis=0)
    
    return feature_importance

def _compute_feature_hits_worker(trial, config, X_train, X_test, y_train, y_test, 
                                shuffled_col_names, base_model_params, features_hits_keys):
    """Worker function for computing feature hits in a single trial."""
    
    # Initialize local features_hits for this trial
    local_features_hits = {feature: 0 for feature in features_hits_keys}
    local_shadow_hits = {feature: 0 for feature in shuffled_col_names}
    
    trial_seed = config.seed + trial
    np.random.seed(trial_seed)
    
    # Create shuffled features
    X_shuffle_train = X_train.apply(np.random.permutation, axis=0)
    X_shuffle_train.columns = shuffled_col_names
    
    X_shuffle_test = X_test.apply(np.random.permutation, axis=0)
    X_shuffle_test.columns = shuffled_col_names
    
    # Combine original and shuffled features
    X_boruta_train = pd.concat([X_train, X_shuffle_train], axis=1)
    X_boruta_test = pd.concat([X_test, X_shuffle_test], axis=1)
    
    # Update model params with trial seed
    trial_params = base_model_params.copy()
    trial_params['random_state'] = trial_seed
    
    # Compute feature importance using CV
    feature_importance = _compute_shap_importance_cv(
        X_boruta_train, X_boruta_test, y_train, y_test,
        trial_params, config.cv_folds, config.early_stopping_rounds,
        verbose=False
    )
    
    # Separate original and shuffled feature importances
    original_importance = feature_importance[:len(X_train.columns)]
    shuffled_importance = feature_importance[len(X_train.columns):]
    
    # Update hits for features better than best shuffled feature
    max_shuffled_importance = shuffled_importance.max()
    
    # Track original feature hits
    for idx, (feature, importance) in enumerate(zip(X_train.columns, original_importance)):
        if importance > max_shuffled_importance:
            local_features_hits[feature] += 1
    
    # Track shadow feature hits against max shadow
    for idx, (feature, importance) in enumerate(zip(shuffled_col_names, shuffled_importance)):
        if importance > max_shuffled_importance:
            local_shadow_hits[feature] += 1
    
    # Clean up memory
    gc.collect()
    
    return local_features_hits, local_shadow_hits

def plot_zscore_histogram(feature_zscores, shadow_zscores, z_threshold=1.96, max_shadow_z=None):
    """Plot Z-score histogram showing feature distributions."""
    plt.figure(figsize=(12, 8))
    
    # Combine all Z-scores for range
    all_zscores = list(feature_zscores.values()) + list(shadow_zscores.values())
    min_z, max_z = min(all_zscores) - 0.5, max(all_zscores) + 0.5
    
    # Create bins
    bins = np.linspace(min_z, max_z, 40)
    
    # Separate features by zone
    green_z = [z for feat, z in feature_zscores.items() if z > z_threshold]
    blue_z = [z for feat, z in feature_zscores.items() if 0 < z <= z_threshold]
    rejected_z = [z for feat, z in feature_zscores.items() if z <= 0]
    shadow_z = list(shadow_zscores.values())
    
    # Plot histograms
    plt.hist(shadow_z, bins=bins, alpha=0.6, label='Shadow features', color='gray', edgecolor='black')
    plt.hist(rejected_z, bins=bins, alpha=0.6, label='Rejected features', color='red', edgecolor='black')
    plt.hist(blue_z, bins=bins, alpha=0.6, label='Tentative features', color='blue', edgecolor='black')
    plt.hist(green_z, bins=bins, alpha=0.6, label='Confirmed features', color='green', edgecolor='black')
    
    # Add threshold lines
    plt.axvline(x=z_threshold, color='black', linestyle='--', linewidth=2, label=f'Z-threshold ({z_threshold})')
    plt.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    if max_shadow_z is not None:
        plt.axvline(x=max_shadow_z, color='red', linestyle='--', linewidth=2, 
                   label=f'Max shadow Z ({max_shadow_z:.2f})')
    
    # Add normal distribution overlay for shadows
    x = np.linspace(min_z, max_z, 100)
    shadow_mean = np.mean(shadow_z)
    shadow_std = np.std(shadow_z)
    normal_curve = stats.norm.pdf(x, shadow_mean, shadow_std) * len(shadow_z) * (bins[1] - bins[0])
    plt.plot(x, normal_curve, 'k-', linewidth=2, label='Expected normal dist.')
    
    plt.xlabel('Z-score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Boruta-SHAP Feature Importance: Z-score Distribution (LightGBM GPU)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Features:\nConfirmed: {len(green_z)}\nTentative: {len(blue_z)}\nRejected: {len(rejected_z)}\nShadow: {len(shadow_z)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return plt.gcf()

def boruta_shap_lgbm(X, y, config=None):
    """
    Boruta-SHAP algorithm using LightGBM with GPU acceleration, CV, and early stopping.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series or np.array
        Target variable (continuous for regression)
    config : BorutaShapLGBMConfig
        Configuration object with all parameters
    
    Returns:
    --------
    dict
        Dictionary with selected features, rankings, and visualizations
    """
    
    # Use default config if none provided
    if config is None:
        config = BorutaShapLGBMConfig()
    
    # Set the seed
    np.random.seed(config.seed)
    
    # Convert y to pandas Series if needed
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Validate inputs
    assert X.shape[0] == y.shape[0], "X and y dimensions don't match"
    
    if config.verbose:
        print(f"Starting Boruta-SHAP with LightGBM GPU:")
        print(f"  Trials: {config.trials}")
        print(f"  CV Folds: {config.cv_folds}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Early Stopping Rounds: {config.early_stopping_rounds}")
        print(f"  GPU Device: {config.gpu_device_id}")
    
    # Initialize feature hits counter
    features_hits = {feature: 0 for feature in X.columns}
    shadow_hits = {}
    
    # Create shuffled column names
    shuffled_col_names = [f"{column}_shuffle" for column in X.columns]
    for col in shuffled_col_names:
        shadow_hits[col] = 0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.seed
    )
    
    # Configure LightGBM parameters for GPU
    base_model_params = {
        'objective': 'regression',
        'metric': config.eval_metric,
        'boosting_type': 'gbdt',
        'num_leaves': 2 ** config.max_depth - 1,
        'max_depth': config.max_depth,
        'learning_rate': config.learning_rate,
        'n_estimators': config.n_estimators,
        'subsample': config.subsample,
        'colsample_bytree': config.colsample_bytree,
        'reg_alpha': config.reg_alpha,
        'reg_lambda': config.reg_lambda,
        'min_child_samples': config.min_child_samples,
        'random_state': config.seed,
        'n_jobs': -1,
        'device': 'gpu',
        'gpu_device_id': config.gpu_device_id,
        'gpu_use_dp': config.gpu_use_dp,
        'verbosity': -1,
        'verbose_eval' : -1
    }
    
    if config.verbose:
        print(f"Running {config.trials} trials with {config.cv_folds}-fold CV...")
    
    start_time = time.time()
    
    # Run trials
    results = []
    for trial in tqdm(range(config.trials)):
        #if config.verbose and (trial + 1) % 5 == 0:
        #    print(f"  Completed {trial + 1}/{config.trials} trials...")
        
        feature_result, shadow_result = _compute_feature_hits_worker(
            trial, config, X_train, X_test, y_train, y_test,
            shuffled_col_names, base_model_params, list(features_hits.keys())
        )
        results.append((feature_result, shadow_result))
    
    # Aggregate results from all trials
    for feature_result, shadow_result in results:
        for feature, hits in feature_result.items():
            features_hits[feature] += hits
        for shadow, hits in shadow_result.items():
            shadow_hits[shadow] += hits
    
    end_time = time.time()
    
    if config.verbose:
        print(f"Completed in {end_time - start_time:.2f} seconds")
    
    # Calculate Z-scores
    feature_zscores = {feat: calculate_zscore(hits, config.trials) for feat, hits in features_hits.items()}
    shadow_zscores = {feat: calculate_zscore(hits, config.trials) for feat, hits in shadow_hits.items()}
    
    # Find max shadow Z-score
    max_shadow_z = max(shadow_zscores.values()) if shadow_zscores else 0
    
    # Classify features based on Z-scores
    green, blue = choose_features_zscore(feature_zscores, config.z_threshold)
    
    # Print results
    if config.verbose:
        print(f"\nFeature selection results:")
        print(f"  Green (accepted) features: {len(green)}")
        print(f"  Blue (tentative) features: {len(blue)}")
        print(f"  Z-score threshold: {config.z_threshold}")
        print(f"  Max shadow Z-score: {max_shadow_z:.3f}")
    
    # Prepare selected features
    if len(green) > 0:
        selected_features = green
        selection_type = 'green'
    elif len(blue) > 0:
        selected_features = blue
        selection_type = 'blue'
    else:
        selected_features = X.columns.tolist()
        selection_type = 'all'
    
    # Create ranked DataFrame with all features
    feature_data = []
    for feature, hits in features_hits.items():
        # Calculate scores
        hit_rate = hits / config.trials
        z_score = feature_zscores[feature]
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Determine zone
        if feature in green:
            zone = 'green'
        elif feature in blue:
            zone = 'blue'
        else:
            zone = 'rejected'
        
        # Determine if selected
        is_selected = feature in selected_features
        
        feature_data.append({
            'feature': feature,
            'hits': hits,
            'trials': config.trials,
            'hit_rate': hit_rate,
            'z_score': z_score,
            'p_value': p_value,
            'zone': zone,
            'selected': is_selected
        })
    
    # Create DataFrame and sort by Z-score (descending)
    feature_ranking_df = pd.DataFrame(feature_data)
    feature_ranking_df = feature_ranking_df.sort_values('z_score', ascending=False).reset_index(drop=True)
    feature_ranking_df['rank'] = feature_ranking_df.index + 1
    
    # Reorder columns
    feature_ranking_df = feature_ranking_df[['rank', 'feature', 'hits', 'trials', 'hit_rate', 
                                             'z_score', 'p_value', 'zone', 'selected']]
    
    if config.verbose:
        print(f"\nFeature Ranking Summary (Top 10):")
        print(feature_ranking_df.head(10).to_string(index=False))
    
    # Create Z-score histogram
    zscore_plot = plot_zscore_histogram(feature_zscores, shadow_zscores, 
                                       z_threshold=config.z_threshold,
                                       max_shadow_z=max_shadow_z)
    
    results = {
        'selected_features': selected_features,
        'selection_type': selection_type,
        'feature_hits': features_hits,
        'feature_zscores': feature_zscores,
        'shadow_zscores': shadow_zscores,
        'max_shadow_zscore': max_shadow_z,
        'green_features': green,
        'blue_features': blue,
        'feature_ranking': feature_ranking_df,
        'zscore_plot': zscore_plot,
        'config': config
    }
    
    return results





seed=0
# Configure Boruta-SHAP with LightGBM GPU
config = BorutaShapLGBMConfig(
    trials=500,                    # Number of Boruta trials
    cv_folds=3,                   # 3-fold CV for each trial
    z_threshold=1.96,             # Standard significance
    n_estimators=1000,            # Max iterations (early stopping will decide)
    learning_rate=0.05,           # Lower for early stopping
    early_stopping_rounds=5,      # Patience
    eval_metric='mae',            # Use MAE for early stopping
    gpu_device_id=0,              # Use first GPU
    verbose=True,
    seed=seed
)

# Run the algorithm
results = boruta_shap_lgbm(X, y, config)

# Display results
print(f"\nSelected {len(results['selected_features'])} features:")
print(results['selected_features'])

# Save outputs
results['feature_ranking'].to_csv(f'boruta_lgbm_gpu_ranking_{seed}_{USE_COL}.csv', index=False)
results['zscore_plot'].savefig(f'boruta_lgbm_gpu_histogram_{seed}_{USE_COL}.png', dpi=300, bbox_inches='tight')
plt.show()


seed=33
# Configure Boruta-SHAP with LightGBM GPU
config = BorutaShapLGBMConfig(
    trials=500,                    # Number of Boruta trials
    cv_folds=3,                   # 3-fold CV for each trial
    z_threshold=1.96,             # Standard significance
    n_estimators=1000,            # Max iterations (early stopping will decide)
    learning_rate=0.05,           # Lower for early stopping
    early_stopping_rounds=5,      # Patience
    eval_metric='mae',            # Use MAE for early stopping
    gpu_device_id=0,              # Use first GPU
    verbose=True,
    seed=seed
)

# Run the algorithm
results = boruta_shap_lgbm(X, y, config)

# Display results
print(f"\nSelected {len(results['selected_features'])} features:")
print(results['selected_features'])

# Save outputs
results['feature_ranking'].to_csv(f'boruta_lgbm_gpu_ranking_{seed}_{USE_COL}.csv', index=False)
results['zscore_plot'].savefig(f'boruta_lgbm_gpu_histogram_{seed}_{USE_COL}.png', dpi=300, bbox_inches='tight')
plt.show()


seed=2018
# Configure Boruta-SHAP with LightGBM GPU
config = BorutaShapLGBMConfig(
    trials=500,                    # Number of Boruta trials
    cv_folds=3,                   # 3-fold CV for each trial
    z_threshold=1.96,             # Standard significance
    n_estimators=1000,            # Max iterations (early stopping will decide)
    learning_rate=0.05,           # Lower for early stopping
    early_stopping_rounds=5,      # Patience
    eval_metric='mae',            # Use MAE for early stopping
    gpu_device_id=0,              # Use first GPU
    verbose=True,
    seed=seed
)

# Run the algorithm
results = boruta_shap_lgbm(X, y, config)

# Display results
print(f"\nSelected {len(results['selected_features'])} features:")
print(results['selected_features'])

# Save outputs
results['feature_ranking'].to_csv(f'boruta_lgbm_gpu_ranking_{seed}_{USE_COL}.csv', index=False)
results['zscore_plot'].savefig(f'boruta_lgbm_gpu_histogram_{seed}_{USE_COL}.png', dpi=300, bbox_inches='tight')
plt.show()


seed=65
# Configure Boruta-SHAP with LightGBM GPU
config = BorutaShapLGBMConfig(
    trials=500,                    # Number of Boruta trials
    cv_folds=3,                   # 3-fold CV for each trial
    z_threshold=1.96,             # Standard significance
    n_estimators=1000,            # Max iterations (early stopping will decide)
    learning_rate=0.05,           # Lower for early stopping
    early_stopping_rounds=5,      # Patience
    eval_metric='mae',            # Use MAE for early stopping
    gpu_device_id=0,              # Use first GPU
    verbose=True,
    seed=seed
)

# Run the algorithm
results = boruta_shap_lgbm(X, y, config)

# Display results
print(f"\nSelected {len(results['selected_features'])} features:")
print(results['selected_features'])

# Save outputs
results['feature_ranking'].to_csv(f'boruta_lgbm_gpu_ranking_{seed}_{USE_COL}.csv', index=False)
results['zscore_plot'].savefig(f'boruta_lgbm_gpu_histogram_{seed}_{USE_COL}.png', dpi=300, bbox_inches='tight')
plt.show()

