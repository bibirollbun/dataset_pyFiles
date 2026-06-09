# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def load_data():
    """
    Load training and testing datasets.
    
    The data consists of time series with 133 features and a target variable to predict.
    Training data spans from 2008 to 2019, while test data covers 2019 to 2024.
    
    Returns:
        tuple: (train_df, test_df) with parsed datetime columns
    """
    train_file_path = r"/kaggle/input/become-a-kaggle-master-2025-hw-2/train.csv"
    train_df = pd.read_csv(train_file_path, parse_dates=['Dates'])
    
    test_file_path = r"/kaggle/input/become-a-kaggle-master-2025-hw-2/test.csv"
    test_df = pd.read_csv(test_file_path, parse_dates=['Dates'])
    
    # Store test dates for final submission
    test_dates = test_df['Dates'].copy()
    
    return train_df, test_df, test_dates


def feature_engineering(train_df, test_df):
    """
    Creates a few features.
    
    Feature engineering strategy:
    1. Time-based features: Capture seasonal patterns identified in EDA
       - Month: Direct seasonal indicator (March high, May low pattern)
       - Quarter: Business cycle patterns
    
    2. Statistical aggregations: Create robust representations of 133 features
       - Mean: Central tendency of all features
       - Std: Variability measure
       - Median: Robust central tendency
    
    3. Interaction features: Model relationships between statistics
       - Mean/Std ratio: Coefficient of variation
       - Median-Mean diff: Distribution skewness indicator
    
    4. Feature selection: Use SelectKBest with k=100 to reduce dimensionality
       - Reduces to 100 most informative features
       - Prevents overfitting while retaining signal
    """
    # Extract target variable
    y = train_df['ToPredict']
    
    # Create feature matrices without target and date
    X = train_df.drop(['ToPredict', 'Dates'], axis=1)
    X_test = test_df.drop(['Dates'], axis=1)
    
    # Time-based features
    X['month'] = train_df['Dates'].dt.month
    X['quarter'] = train_df['Dates'].dt.quarter
    X_test['month'] = test_df['Dates'].dt.month
    X_test['quarter'] = test_df['Dates'].dt.quarter
    
    # Statistical features from the 133 original features
    feature_cols = [col for col in X.columns if col.startswith('Features_')]
    
    # Calculate statistical aggregations
    X['features_mean'] = X[feature_cols].mean(axis=1)
    X['features_std'] = X[feature_cols].std(axis=1)
    X['features_median'] = X[feature_cols].median(axis=1)
    
    X_test['features_mean'] = X_test[feature_cols].mean(axis=1)
    X_test['features_std'] = X_test[feature_cols].std(axis=1)
    X_test['features_median'] = X_test[feature_cols].median(axis=1)
    
    # Interaction features
    X['mean_std_ratio'] = X['features_mean'] / (X['features_std'] + 1e-6)
    X['median_mean_diff'] = X['features_median'] - X['features_mean']
    
    X_test['mean_std_ratio'] = X_test['features_mean'] / (X_test['features_std'] + 1e-6)
    X_test['median_mean_diff'] = X_test['features_median'] - X_test['features_mean']
    
    # Feature selection using F-statistic
    selector = SelectKBest(f_regression, k=100)
    X_selected = pd.DataFrame(selector.fit_transform(X, y))
    X_test_selected = pd.DataFrame(selector.transform(X_test))
    
    # Preserve feature names for interpretability
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected.columns = selected_features
    X_test_selected.columns = selected_features
    
    # Handle any potential missing values
    X_selected.fillna(X_selected.median(), inplace=True)
    X_test_selected.fillna(X_test_selected.median(), inplace=True)
    
    return X_selected, X_test_selected, y


# Define search space for Bayesian optimization
dimensions = [
    # Configuration 1: Balanced model
    Real(0.052, 0.057, name='lr1'),      # Learning rate around 0.0545
    Real(13, 17, name='leaves1'),        # Number of leaves around 15
    Real(0.13, 0.17, name='lambda1'),    # L2 regularization around 0.15
    
    # Configuration 2: Complex model
    Real(0.057, 0.062, name='lr2'),      # Higher learning rate around 0.0595
    Real(14, 18, name='leaves2'),        # More leaves around 16
    Real(0.08, 0.12, name='lambda2'),    # Less regularization around 0.1
    
    # Configuration 3: Conservative model
    Real(0.055, 0.060, name='lr3'),      # Moderate learning rate around 0.0575
    Real(15, 19, name='leaves3'),        # Medium complexity around 17
    Real(0.06, 0.10, name='lambda3'),    # Light regularization around 0.08
    
    # Post-processing parameter
    Real(0.45, 0.51, name='smooth_thresh') # Smoothing threshold around 0.48
]

# Global variables for optimization
global_X = None
global_y = None

@use_named_args(dimensions)
def objective(**params):
    """
    Objective function for Bayesian optimization.
    
    This function evaluates a set of hyperparameters by:
    1. Training 3 different model configurations
    2. Using 3-fold cross-validation with 2 random seeds
    3. Computing weighted ensemble performance
    4. Returning MSE to minimize
    """
    global global_X, global_y
    
    # Extract parameters for three configurations
    configs = [
        {
            'num_leaves': int(params['leaves1']),
            'learning_rate': params['lr1'],
            'reg_lambda': params['lambda1'],
            'num_boost_round': 130
        },
        {
            'num_leaves': int(params['leaves2']),
            'learning_rate': params['lr2'],
            'reg_lambda': params['lambda2'],
            'num_boost_round': 120
        },
        {
            'num_leaves': int(params['leaves3']),
            'learning_rate': params['lr3'],
            'reg_lambda': params['lambda3'],
            'num_boost_round': 125
        }
    ]
    
    # Base LightGBM parameters (fixed across all configurations)
    base_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'verbose': -1,
        'force_col_wise': True,
        'max_depth': 4
    }
    
    # Evaluate each configuration
    random_seeds = [42, 43]
    all_scores = []
    
    for config in configs:
        config_params = base_params.copy()
        config_params.update({
            'num_leaves': config['num_leaves'],
            'learning_rate': config['learning_rate'],
            'reg_lambda': config['reg_lambda']
        })
        
        config_scores = []
        for seed in random_seeds:
            config_params['random_state'] = seed
            
            # 3-fold cross-validation
            kf = KFold(n_splits=3, shuffle=True, random_state=seed)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(global_X):
                X_train, X_val = global_X.iloc[train_idx], global_X.iloc[val_idx]
                y_train, y_val = global_y.iloc[train_idx], global_y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    config_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=config['num_boost_round'],
                    callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
                )
                
                pred = model.predict(X_val, num_iteration=model.best_iteration)
                score = mean_squared_error(y_val, pred)
                cv_scores.append(score)
            
            config_scores.append(np.mean(cv_scores))
        
        all_scores.append(np.mean(config_scores))
    
    # Compute weighted ensemble score
    weights = [1.0 / s for s in all_scores]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    final_score = sum(w * s for w, s in zip(weights, all_scores))
    
    return final_score

def run_bayesian_optimization(X, y):
    """
    Run Bayesian optimization to find optimal hyperparameters.
    
    Optimization strategy:
    1. Use Gaussian Process to model the objective function
    2. Balance exploration vs exploitation with gp_hedge acquisition
    3. Run 25 iterations to find global optimum
    4. Return best parameters found
    """
    global global_X, global_y
    global_X = X
    global_y = y
    
    # Run optimization
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=25,
        random_state=42,
        acq_func='gp_hedge'
    )
    
    # Extract best parameters
    best_params = {
        'lr1': result.x[0], 'leaves1': result.x[1], 'lambda1': result.x[2],
        'lr2': result.x[3], 'leaves2': result.x[4], 'lambda2': result.x[5],
        'lr3': result.x[6], 'leaves3': result.x[7], 'lambda3': result.x[8],
        'smooth_thresh': result.x[9]
    }
    
    return best_params, result.fun


def train_final_model(X, y, X_test, best_params):
    """
    Train final ensemble model using optimized hyperparameters.
    
    Training strategy:
    1. Use best parameters from Bayesian optimization
    2. Train 3 model configurations with different complexity levels
    3. Use 4 random seeds for robustness
    4. Create weighted ensemble based on validation performance
    """
    # Define configurations with optimized parameters
    configs = [
        {
            'num_leaves': int(best_params['leaves1']),
            'learning_rate': best_params['lr1'],
            'reg_lambda': best_params['lambda1'],
            'num_boost_round': 130
        },
        {
            'num_leaves': int(best_params['leaves2']),
            'learning_rate': best_params['lr2'],
            'reg_lambda': best_params['lambda2'],
            'num_boost_round': 120
        },
        {
            'num_leaves': int(best_params['leaves3']),
            'learning_rate': best_params['lr3'],
            'reg_lambda': best_params['lambda3'],
            'num_boost_round': 125
        }
    ]
    
    base_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'verbose': -1,
        'force_col_wise': True,
        'max_depth': 4
    }
    
    # Train each configuration
    random_seeds = [42, 43, 44, 45]
    all_predictions = []
    all_scores = []
    
    for config in configs:
        config_params = base_params.copy()
        config_params.update({
            'num_leaves': config['num_leaves'],
            'learning_rate': config['learning_rate'],
            'reg_lambda': config['reg_lambda']
        })
        
        seed_predictions = []
        seed_scores = []
        
        for seed in random_seeds:
            config_params['random_state'] = seed
            
            kf = KFold(n_splits=3, shuffle=True, random_state=seed)
            test_predictions = []
            oof_preds = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    config_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=config['num_boost_round'],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                
                oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
                test_pred = model.predict(X_test, num_iteration=model.best_iteration)
                test_predictions.append(test_pred)
            
            seed_prediction = np.mean(test_predictions, axis=0)
            seed_predictions.append(seed_prediction)
            seed_score = mean_squared_error(y, oof_preds)
            seed_scores.append(seed_score)
        
        config_prediction = np.mean(seed_predictions, axis=0)
        config_score = np.mean(seed_scores)
        all_predictions.append(config_prediction)
        all_scores.append(config_score)
    
    # Create weighted ensemble
    weights = [1.0 / s for s in all_scores]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    final_prediction = np.zeros_like(all_predictions[0])
    for i, pred in enumerate(all_predictions):
        final_prediction += weights[i] * pred
    
    return final_prediction


def post_process_predictions(predictions, y_train, smooth_thresh):
    """
    Apply post-processing to improve prediction quality.
    
    Post-processing strategy:
    1. Clip predictions to training data range (with 1% buffer)
    2. Apply adaptive smoothing based on local variability
    3. Use optimized smoothing threshold from Bayesian optimization
    """
    # Range clipping
    y_min, y_max = y_train.min(), y_train.max()
    predictions = np.clip(predictions, y_min * 0.99, y_max * 1.01)
    
    # Adaptive smoothing
    smoothed = np.copy(predictions)
    window_size = 5
    
    for i in range(window_size//2, len(predictions) - window_size//2):
        local_window = predictions[i-window_size//2:i+window_size//2+1]
        local_std = np.std(local_window)
        
        if local_std > np.std(predictions) * smooth_thresh:
            # High variability: use mean smoothing
            smoothed[i] = np.mean(local_window)
        else:
            # Low variability: use weighted average
            smoothed[i] = 0.3 * predictions[i-1] + 0.4 * predictions[i] + 0.3 * predictions[i+1]
    
    return smoothed


# Step 1: Load data
train_df, test_df, test_dates = load_data()
    
# Step 2: Feature engineering
X, X_test, y = feature_engineering(train_df, test_df)
    
# Step 3: Bayesian hyperparameter optimization
best_params, best_score = run_bayesian_optimization(X, y)
    
# Step 4: Train final model
predictions = train_final_model(X, y, X_test, best_params)
    
# Step 5: Post-process predictions
final_predictions = post_process_predictions(predictions, y, best_params['smooth_thresh'])
    
# Step 6: Create submission
submission = pd.DataFrame({
    'ID': test_dates.dt.strftime('%Y-%m-%d'),
    'y': final_predictions
})
    
submission.to_csv('final_submission.csv', index=False)

