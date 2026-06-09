import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import pickle
import joblib

# Machine Learning Libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor

# Advanced Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Check GPU availability
import torch
print(f"âœ… CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")

print("âœ… All libraries imported successfully!")
print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {train.columns.tolist()}")
print(f"\nFirst few rows:")
print(train.head())

# Save IDs
train_id = train['id']
test_id = test['id']

# Separate features and target
y = train['accident_risk']
X = train.drop(['id', 'accident_risk'], axis=1)
X_test = test.drop(['id'], axis=1)

print(f"\nğŸ“Š Target Statistics:")
print(y.describe())



def plot_target_distribution(y):
    """Visualize target distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram
    axes[0].hist(y, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Accident Risk Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Accident Risk')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.3f}')
    axes[0].axvline(y.median(), color='green', linestyle='--', label=f'Median: {y.median():.3f}')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(y, vert=True)
    axes[1].set_title('Box Plot of Accident Risk', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accident Risk')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(y, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

plot_target_distribution(y)

# Feature types analysis
print("\nğŸ“‹ Feature Types:")
print(f"Numerical features: {X.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"Categorical features: {X.select_dtypes(include=['object']).columns.tolist()}")



def advanced_feature_engineering(df, is_train=True):
    """
    Create sophisticated features for better model performance
    """
    df = df.copy()
    
    # Identify feature types
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical features: {len(cat_features)}")
    print(f"Numerical features: {len(num_features)}")
    
    # 1. Encode categorical features
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # 2. Interaction features (for numerical features)
    if len(num_features) >= 2:
        # Create polynomial interactions for top features
        for i in range(min(3, len(num_features))):
            for j in range(i+1, min(4, len(num_features))):
                col1, col2 = num_features[i], num_features[j]
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    # 3. Statistical aggregations
    if len(num_features) >= 3:
        num_data = df[num_features]
        df['num_mean'] = num_data.mean(axis=1)
        df['num_std'] = num_data.std(axis=1)
        df['num_max'] = num_data.max(axis=1)
        df['num_min'] = num_data.min(axis=1)
        df['num_range'] = df['num_max'] - df['num_min']
        df['num_median'] = num_data.median(axis=1)
    
    # 4. Frequency encoding for categorical features (if train)
    if is_train:
        freq_dict = {}
        for col in cat_features:
            freq = df[col].value_counts(normalize=True).to_dict()
            df[f'{col}_freq'] = df[col].map(freq)
            freq_dict[col] = freq
        return df, label_encoders, freq_dict
    else:
        return df
    
print("ğŸ”§ Applying Advanced Feature Engineering...")
X_enhanced, label_encoders, freq_dict = advanced_feature_engineering(X, is_train=True)

# Apply to test set
X_test_enhanced = advanced_feature_engineering(X_test, is_train=False)

# Apply frequency encoding to test using train frequencies
cat_features = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_features:
    X_test_enhanced[f'{col}_freq'] = X_test_enhanced[col].map(freq_dict[col])
    X_test_enhanced[f'{col}_freq'].fillna(X_test_enhanced[f'{col}_freq'].mean(), inplace=True)

print(f"âœ… Enhanced feature count: {X_enhanced.shape[1]} (from {X.shape[1]})")



def objective_xgb(trial, X, y):
    """Optuna objective function for XGBoost with GPU support"""
    # GPU parameters
    tree_method = 'gpu_hist' if torch.cuda.is_available() else 'hist'
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),  # Reduced range
        'max_depth': trial.suggest_int('max_depth', 4, 10),  # Reduced range
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0.01, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
        'random_state': 42,
        'tree_method': tree_method,
        'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor',
        'enable_categorical': False
    }
    
    # 3-Fold CV (reduced from 5 for speed)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rmse_scores = []
    
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    for train_idx, val_idx in kf.split(X, y_binned):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold, 
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False)
        
        preds = model.predict(X_val_fold)
        rmse = mean_squared_error(y_val_fold, preds, squared=False)
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

print("ğŸ”� Starting Optuna Optimization for XGBoost (GPU-Accelerated)...")
print("This will be faster with GPU enabled!")

study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
# REDUCED: 30 â†’ 15 trials for faster optimization
study_xgb.optimize(lambda trial: objective_xgb(trial, X_enhanced, y), 
                   n_trials=15, show_progress_bar=True)

print(f"\nâœ… Best XGBoost RMSE: {study_xgb.best_value:.6f}")
print(f"Best parameters: {study_xgb.best_params}")

best_xgb_params = study_xgb.best_params
best_xgb_params['random_state'] = 42
best_xgb_params['tree_method'] = 'gpu_hist' if torch.cuda.is_available() else 'hist'
best_xgb_params['predictor'] = 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'


def objective_lgb(trial, X, y):
    """Optuna objective function for LightGBM with GPU support"""
    # GPU device type
    device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),  # Reduced
        'max_depth': trial.suggest_int('max_depth', 4, 10),  # Reduced
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
        'random_state': 42,
        'device': device_type,
        'verbose': -1
    }
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced to 3
    rmse_scores = []
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    for train_idx, val_idx in kf.split(X, y_binned):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        
        preds = model.predict(X_val_fold)
        rmse = mean_squared_error(y_val_fold, preds, squared=False)
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

print("ğŸ”� Starting Optuna Optimization for LightGBM (GPU-Accelerated)...")
study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_lgb.optimize(lambda trial: objective_lgb(trial, X_enhanced, y), 
                   n_trials=15, show_progress_bar=True)  # Reduced to 15

print(f"\nâœ… Best LightGBM RMSE: {study_lgb.best_value:.6f}")
print(f"Best parameters: {study_lgb.best_params}")

best_lgb_params = study_lgb.best_params
best_lgb_params['random_state'] = 42
best_lgb_params['device'] = 'gpu' if torch.cuda.is_available() else 'cpu'
best_lgb_params['verbose'] = -1


def objective_cat(trial, X, y):
    """Optuna objective function for CatBoost with GPU support"""
    task_type = 'GPU' if torch.cuda.is_available() else 'CPU'
    
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),  # Reduced
        'depth': trial.suggest_int('depth', 4, 8),  # Reduced
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 7),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 0.8),
        'random_strength': trial.suggest_float('random_strength', 0, 5),
        'border_count': trial.suggest_int('border_count', 64, 255),
        'random_state': 42,
        'task_type': task_type,
        'verbose': False
    }
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced to 3
    rmse_scores = []
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    for train_idx, val_idx in kf.split(X, y_binned):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=(X_val_fold, y_val_fold),
                  early_stopping_rounds=30,
                  verbose=False)
        
        preds = model.predict(X_val_fold)
        rmse = mean_squared_error(y_val_fold, preds, squared=False)
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

print("ğŸ”� Starting Optuna Optimization for CatBoost (GPU-Accelerated)...")
study_cat = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_cat.optimize(lambda trial: objective_cat(trial, X_enhanced, y), 
                   n_trials=15, show_progress_bar=True)  # Reduced to 15

print(f"\nâœ… Best CatBoost RMSE: {study_cat.best_value:.6f}")
print(f"Best parameters: {study_cat.best_params}")

best_cat_params = study_cat.best_params
best_cat_params['random_state'] = 42
best_cat_params['task_type'] = 'GPU' if torch.cuda.is_available() else 'CPU'
best_cat_params['verbose'] = False


from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

def create_stacking_model(X, y):
    """Create advanced stacking ensemble"""
    
    # Base models with optimized parameters
    base_models = [
        ('xgb', xgb.XGBRegressor(**best_xgb_params)),
        ('lgb', lgb.LGBMRegressor(**best_lgb_params)),
        ('cat', CatBoostRegressor(**best_cat_params)),
        ('hgb', HistGradientBoostingRegressor(
            max_iter=1000,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )),
        ('et', ExtraTreesRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ]
    
    # Meta-learner: Ridge regression
    meta_model = Ridge(alpha=1.0)
    
    # Create stacking regressor
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    return stacking_model

print("ğŸ�—ï¸� Building Advanced Stacking Ensemble...")
stacking_model = create_stacking_model(X_enhanced, y)


def train_with_cv(model, X, y, n_splits=5):
    """
    Train model with Stratified K-Fold Cross Validation safely
    (prevents GPU conflicts with CatBoost/XGBoost inside stacking)
    """
    
    # Bin continuous target for stratified splits
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test_enhanced))
    rmse_scores = []

    print("\nğŸš€ Training Stacking Ensemble with 5-Fold CV...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_binned), 1):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold}/{n_splits}")
        print(f"{'='*50}")

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # âš™ï¸� Fit model sequentially (avoid parallel GPU conflict)
        if isinstance(model, StackingRegressor):
            # Temporarily set model to run single-threaded
            model.n_jobs = 1
            model.fit(X_train_fold, y_train_fold)
        else:
            model.fit(X_train_fold, y_train_fold)

        # ğŸ”® Validation predictions
        val_preds = model.predict(X_val_fold)
        oof_predictions[val_idx] = val_preds

        # ğŸ“‰ Compute RMSE for fold
        fold_rmse = mean_squared_error(y_val_fold, val_preds, squared=False)
        rmse_scores.append(fold_rmse)
        print(f"âœ… Fold {fold} RMSE: {fold_rmse:.6f}")

        # ğŸ§  Test predictions
        test_preds = model.predict(X_test_enhanced)
        test_predictions += test_preds / n_splits

    # ğŸ“Š Overall CV performance
    overall_rmse = mean_squared_error(y, oof_predictions, squared=False)
    print(f"\n{'='*50}")
    print(f"ğŸ“Š Overall CV RMSE: {overall_rmse:.6f} (+/- {np.std(rmse_scores):.6f})")
    print(f"{'='*50}")

    return oof_predictions, test_predictions, rmse_scores


# âœ… Train safely (no GPU race conditions)
oof_preds, test_preds_stack, cv_scores = train_with_cv(stacking_model, X_enhanced, y, n_splits=5)



print("\nğŸ“¦ Training Individual Models for Additional Blending...")

# Train individual models
models = {
    'XGBoost': xgb.XGBRegressor(**best_xgb_params),
    'LightGBM': lgb.LGBMRegressor(**best_lgb_params),
    'CatBoost': CatBoostRegressor(**best_cat_params)
}

individual_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    _, test_preds, _ = train_with_cv(model, X_enhanced, y, n_splits=3)  # Reduced to 3
    individual_predictions[name] = test_preds


print("\nğŸ�¯ Creating Final Blended Predictions...")

# Weighted average of all predictions
final_predictions = (
    0.50 * test_preds_stack +  # Stacking ensemble (highest weight)
    0.20 * individual_predictions['XGBoost'] +
    0.15 * individual_predictions['LightGBM'] +
    0.15 * individual_predictions['CatBoost']
)

# Ensure predictions are in valid range [0, 1]
final_predictions = np.clip(final_predictions, 0, 1)

print(f"Final predictions range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print(f"Final predictions mean: {final_predictions.mean():.4f}")
print(f"Final predictions std: {final_predictions.std():.4f}")


print("\nğŸ’¾ Saving Models and Artifacts...")

# Create models directory
Path('models').mkdir(exist_ok=True)

# Save best parameters
with open('models/best_params.pkl', 'wb') as f:
    pickle.dump({
        'xgb': best_xgb_params,
        'lgb': best_lgb_params,
        'cat': best_cat_params
    }, f)

# Save label encoders and frequency dict
with open('models/feature_engineering.pkl', 'wb') as f:
    pickle.dump({
        'label_encoders': label_encoders,
        'freq_dict': freq_dict
    }, f)

# Save the stacking model
joblib.dump(stacking_model, 'models/stacking_model.pkl')

# Save individual trained models
for name, model in models.items():
    joblib.dump(model, f'models/{name.lower()}_model.pkl')

# Save OOF predictions
np.save('models/oof_predictions.npy', oof_preds)

# Save test predictions
np.save('models/test_predictions_stack.npy', test_preds_stack)
for name, preds in individual_predictions.items():
    np.save(f'models/test_predictions_{name.lower()}.npy', preds)

print("âœ… All models and artifacts saved successfully!")
print("\nğŸ“� Saved files:")
print("   - models/best_params.pkl")
print("   - models/feature_engineering.pkl")
print("   - models/stacking_model.pkl")
print("   - models/xgboost_model.pkl")
print("   - models/lightgbm_model.pkl")
print("   - models/catboost_model.pkl")
print("   - models/oof_predictions.npy")
print("   - models/test_predictions_*.npy")


submission['accident_risk'] = final_predictions

# Display sample predictions
print("\nğŸ“‹ Sample Submission:")
print(submission.head(20))

# Save submission file
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file saved successfully!")

# Visualization of predictions
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(final_predictions, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_title('Distribution of Final Predictions', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Accident Risk')
axes[0].set_ylabel('Frequency')
axes[0].axvline(final_predictions.mean(), color='red', linestyle='--', 
                label=f'Mean: {final_predictions.mean():.3f}')
axes[0].legend()

axes[1].hist(y, bins=50, alpha=0.5, label='Train', edgecolor='black')
axes[1].hist(final_predictions, bins=50, alpha=0.5, label='Test Predictions', edgecolor='black')
axes[1].set_title('Train vs Test Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Accident Risk')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ğŸ�‰ NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"ğŸ“Š Cross-Validation RMSE: {mean_squared_error(y, oof_preds, squared=False):.6f}")
print(f"ğŸ“� Submission saved as: submission.csv")
print(f"ğŸš€ Ready for submission to Kaggle!")
print("="*70)


