# Install required packages (most are pre-installed on Kaggle, but optuna might not be)
# Run this cell first if you get ModuleNotFoundError
# You can comment out this cell after first successful run

!pip install -q optuna xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn scipy

print("âœ… All packages installed successfully!")
print("\nğŸ’¡ You can comment out this cell after first run")


# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Gradient Boosting Models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Hyperparameter optimization
import optuna

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("\nğŸš€ Playground Series S5E10 - Road Accident Risk Prediction")
print(f"ğŸ“… Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nâœ… All libraries imported successfully!")


# CUDA Auto-Detection
print("ğŸ”� Detecting hardware configuration...")

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        DEVICE = 'cuda'
        print(f"âœ… CUDA detected: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # XGBoost GPU config
        XGB_TREE_METHOD = 'gpu_hist'
        XGB_DEVICE = 'cuda'
        # LightGBM GPU config
        LGB_DEVICE = 'gpu'
        LGB_GPU_PLATFORM = 0
        LGB_GPU_DEVICE = 0
        # CatBoost GPU config
        CB_TASK_TYPE = 'GPU'
        CB_DEVICES = '0'
    else:
        DEVICE = 'cpu'
        print("âš ï¸�  CUDA not available, using CPU")
        # XGBoost CPU config
        XGB_TREE_METHOD = 'hist'
        XGB_DEVICE = 'cpu'
        # LightGBM CPU config
        LGB_DEVICE = 'cpu'
        LGB_GPU_PLATFORM = None
        LGB_GPU_DEVICE = None
        # CatBoost CPU config
        CB_TASK_TYPE = 'CPU'
        CB_DEVICES = None
except ImportError:
    print("âš ï¸�  PyTorch not installed, defaulting to CPU")
    CUDA_AVAILABLE = False
    DEVICE = 'cpu'
    # XGBoost CPU config
    XGB_TREE_METHOD = 'hist'
    XGB_DEVICE = 'cpu'
    # LightGBM CPU config
    LGB_DEVICE = 'cpu'
    LGB_GPU_PLATFORM = None
    LGB_GPU_DEVICE = None
    # CatBoost CPU config
    CB_TASK_TYPE = 'CPU'
    CB_DEVICES = None

print(f"Training mode: {DEVICE.upper()}")
if DEVICE == 'cpu':
    print("ğŸ’¡ Tip: GPU training is 10-20x faster. Expected runtime: 2-4 hours on CPU")
else:
    print("âš¡ GPU acceleration enabled. Expected runtime: 30-60 minutes")



# Load datasets
print("ğŸ“Š Loading datasets...")

# Auto-detect environment (Kaggle vs local)
if os.path.exists('/kaggle/input'):
    # Running on Kaggle
    DATA_PATH = '/kaggle/input/playground-series-s5e10'
    print("âœ… Detected Kaggle environment")
else:
    # Running locally
    DATA_PATH = 'data'
    print("âœ… Detected local environment")

train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
test_df = pd.read_csv(f'{DATA_PATH}/test.csv')
sample_submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')

print(f"\nğŸ“ˆ Dataset Summary:")
print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")
print(f"Features: {len(train_df.columns) - 2}")  # Exclude id and target
print(f"Target: accident_risk")

# Display first few rows
print("\nğŸ“‹ Training data sample:")
display(train_df.head())

print("\nğŸ“‹ Test data sample:")
display(test_df.head())

# Basic info
print("\nğŸ“Š Training data info:")
print(train_df.info())

# Target statistics
print("\nğŸ�¯ Target (accident_risk) statistics:")
print(train_df['accident_risk'].describe())


# Missing values analysis
print("â�“ Missing values analysis:")
missing = train_df.isnull().sum()
if missing.sum() == 0:
    print("âœ“ No missing values found")
else:
    print(missing[missing > 0])

# Data types analysis
print("\nğŸ”¤ Feature types:")
print(f"Numerical features: {train_df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"Categorical features: {train_df.select_dtypes(include=['object', 'bool']).columns.tolist()}")

# Unique values for categorical features
print("\nğŸ“Š Categorical feature cardinality:")
categorical_cols = train_df.select_dtypes(include=['object', 'bool']).columns
for col in categorical_cols:
    if col != 'id':
        unique_count = train_df[col].nunique()
        print(f"{col:25s}: {unique_count:3d} unique values - {train_df[col].unique()[:5]}")


# Target distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
axes[0].hist(train_df['accident_risk'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
axes[0].set_title('Accident Risk Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Accident Risk', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(train_df['accident_risk'].mean(), color='red', linestyle='--', label=f"Mean: {train_df['accident_risk'].mean():.3f}")
axes[0].axvline(train_df['accident_risk'].median(), color='green', linestyle='--', label=f"Median: {train_df['accident_risk'].median():.3f}")
axes[0].legend()

# Box plot
axes[1].boxplot(train_df['accident_risk'])
axes[1].set_title('Accident Risk Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accident Risk', fontsize=12)
axes[1].grid(True, alpha=0.3)

# QQ plot for normality check
from scipy import stats
stats.probplot(train_df['accident_risk'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical tests
print(f"\nğŸ“Š Target Distribution Statistics:")
print(f"Mean: {train_df['accident_risk'].mean():.4f}")
print(f"Median: {train_df['accident_risk'].median():.4f}")
print(f"Std Dev: {train_df['accident_risk'].std():.4f}")
print(f"Skewness: {train_df['accident_risk'].skew():.4f}")
print(f"Kurtosis: {train_df['accident_risk'].kurtosis():.4f}")
print(f"Min: {train_df['accident_risk'].min():.4f}")
print(f"Max: {train_df['accident_risk'].max():.4f}")


# Numerical features analysis
numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    axes[idx].scatter(train_df[col], train_df['accident_risk'], alpha=0.3, s=10)
    axes[idx].set_xlabel(col, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('accident_risk', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'{col} vs Accident Risk', fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = train_df[col].corr(train_df['accident_risk'])
    axes[idx].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                   transform=axes[idx].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=11, fontweight='bold')

plt.suptitle('Numerical Features vs Accident Risk', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = train_df[numerical_cols + ['accident_risk']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, square=True, 
            fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Categorical features analysis
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    # Calculate mean accident risk by category
    category_risk = train_df.groupby(col)['accident_risk'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
    
    # Bar plot
    axes[idx].bar(range(len(category_risk)), category_risk['mean'], 
                  alpha=0.7, color='skyblue', edgecolor='black')
    axes[idx].errorbar(range(len(category_risk)), category_risk['mean'], 
                       yerr=category_risk['std'], fmt='none', color='red', alpha=0.5)
    axes[idx].set_xticks(range(len(category_risk)))
    axes[idx].set_xticklabels(category_risk.index, rotation=45, ha='right')
    axes[idx].set_xlabel(col, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Mean Accident Risk', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'{col} - Mean Accident Risk', fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Add sample counts
    for i, (idx_val, row) in enumerate(category_risk.iterrows()):
        axes[idx].text(i, row['mean'] + 0.01, f"n={int(row['count'])}", 
                      ha='center', va='bottom', fontsize=9)

plt.suptitle('Categorical Features - Mean Accident Risk Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Boolean features analysis
boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(boolean_cols):
    # Box plot by boolean value
    train_df.boxplot(column='accident_risk', by=col, ax=axes[idx])
    axes[idx].set_title(f'Accident Risk by {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Accident Risk', fontsize=11)
    
    # Add mean values
    means = train_df.groupby(col)['accident_risk'].mean()
    for i, (val, mean) in enumerate(means.items()):
        axes[idx].text(i+1, mean, f'Î¼={mean:.3f}', 
                      ha='center', va='bottom', fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.suptitle('Boolean Features vs Accident Risk', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Statistical comparison
print("\nğŸ“Š Boolean Features - Statistical Summary:")
for col in boolean_cols:
    print(f"\n{col}:")
    print(train_df.groupby(col)['accident_risk'].agg(['mean', 'std', 'count']))


def engineer_features(df, is_train=True):
    """
    Comprehensive feature engineering pipeline
    
    Args:
        df: Input dataframe
        is_train: Whether this is training data (affects target handling)
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # 1. Interaction Features
    print("ğŸ”§ Creating interaction features...")
    
    # Speed and curvature interaction (dangerous combination)
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    
    # Lanes and speed interaction
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
    
    # Speed per lane (traffic density proxy)
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)
    
    # Accidents per lane
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    
    # 2. Risk Score Features
    print("ğŸ�¯ Creating risk score features...")
    
    # Weather risk (ordinal encoding of weather severity)
    weather_risk_map = {'clear': 0, 'rainy': 1, 'foggy': 2, 'snowy': 3}
    df['weather_risk'] = df['weather'].map(weather_risk_map).fillna(0)
    
    # Lighting risk (ordinal encoding of visibility)
    lighting_risk_map = {'daylight': 0, 'dim': 1, 'night': 2}
    df['lighting_risk'] = df['lighting'].map(lighting_risk_map).fillna(0)
    
    # Combined visibility risk
    df['visibility_risk'] = df['weather_risk'] + df['lighting_risk']
    
    # Road complexity score
    df['road_complexity'] = df['curvature'] * df['speed_limit'] / (df['num_lanes'] + 1)
    
    # 3. Binary Aggregations
    print("â�• Creating binary aggregation features...")
    
    # Total risk factors (sum of boolean risks)
    df['total_risk_factors'] = (
        (~df['road_signs_present']).astype(int) +  # No signs = risk
        (~df['public_road']).astype(int) +         # Private road = risk
        df['holiday'].astype(int) +                # Holiday = risk
        df['school_season'].astype(int)            # School season = risk
    )
    
    # Poor conditions (weather + lighting)
    df['poor_conditions'] = ((df['weather'] != 'clear') | (df['lighting'] != 'daylight')).astype(int)
    
    # High risk scenario
    df['high_risk_scenario'] = (
        (df['speed_limit'] > 45) & 
        (df['curvature'] > 0.5) & 
        (df['weather'] != 'clear')
    ).astype(int)
    
    # 4. Time-based Features
    print("â�° Creating time-based features...")
    
    # Time of day risk (ordinal)
    time_risk_map = {'morning': 1, 'afternoon': 0, 'evening': 2, 'night': 3}
    df['time_risk'] = df['time_of_day'].map(time_risk_map).fillna(0)
    
    # Rush hour indicator (morning/evening)
    df['rush_hour'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    
    # Peak risk time (night + poor weather)
    df['peak_risk_time'] = (
        (df['time_of_day'] == 'night') & 
        (df['weather'] != 'clear')
    ).astype(int)
    
    # 5. Road Type Features
    print("ğŸ›£ï¸� Creating road type features...")
    
    # Urban vs highway indicator
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    
    # 6. Statistical Features
    print("ğŸ“Š Creating statistical features...")
    
    # Normalized curvature by road type
    if is_train:
        global curvature_by_road_type
        curvature_by_road_type = df.groupby('road_type')['curvature'].mean().to_dict()
    df['curvature_vs_road_avg'] = df['curvature'] - df['road_type'].map(curvature_by_road_type)
    
    # Normalized speed by road type
    if is_train:
        global speed_by_road_type
        speed_by_road_type = df.groupby('road_type')['speed_limit'].mean().to_dict()
    df['speed_vs_road_avg'] = df['speed_limit'] - df['road_type'].map(speed_by_road_type)
    
    print(f"âœ… Feature engineering completed: {len(df.columns)} total features")
    return df

# Apply feature engineering
print("\nğŸ”¨ Applying feature engineering to datasets...\n")
train_engineered = engineer_features(train_df, is_train=True)
test_engineered = engineer_features(test_df, is_train=False)

print(f"\nğŸ“Š New features created:")
new_features = [col for col in train_engineered.columns if col not in train_df.columns]
print(f"Total new features: {len(new_features)}")
print(f"Feature list: {new_features}")


# Prepare features and target
print("ğŸ”§ Preparing features and target...\n")

# Identify feature types
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']
id_col = 'id'
target_col = 'accident_risk'

# Get all feature columns (exclude id and target)
feature_cols = [col for col in train_engineered.columns if col not in [id_col, target_col]]

print(f"ğŸ“Š Feature Summary:")
print(f"Total features: {len(feature_cols)}")
print(f"Categorical: {len(categorical_features)}")
print(f"Boolean: {len(boolean_features)}")
print(f"Numerical: {len(feature_cols) - len(categorical_features) - len(boolean_features)}")

# Encode categorical features for XGBoost/LightGBM
print("\nğŸ”¤ Encoding categorical features...")

# Label encoding for tree-based models
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train_engineered[f'{col}_encoded'] = le.fit_transform(train_engineered[col])
    test_engineered[f'{col}_encoded'] = le.transform(test_engineered[col])
    label_encoders[col] = le

# Convert boolean to int
for col in boolean_features:
    train_engineered[col] = train_engineered[col].astype(int)
    test_engineered[col] = test_engineered[col].astype(int)

# Define features for different models
# For XGBoost/LightGBM: use encoded features
encoded_categorical = [f'{col}_encoded' for col in categorical_features]
xgb_features = [col for col in feature_cols if col not in categorical_features] + encoded_categorical

# For CatBoost: use original categorical features
catboost_features = feature_cols
catboost_cat_features = [feature_cols.index(col) for col in categorical_features]

print(f"\nâœ… Encoding completed")
print(f"XGBoost/LightGBM features: {len(xgb_features)}")
print(f"CatBoost features: {len(catboost_features)} (with {len(catboost_cat_features)} categorical)")


# Create train/validation split
print("ğŸ�¯ Creating train/validation split...\n")

# Prepare data
X = train_engineered[xgb_features]
y = train_engineered[target_col]

# Split data: 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

print(f"ğŸ“Š Split Summary:")
print(f"Training samples: {len(X_train):,} ({len(X_train)/len(X):.1%})")
print(f"Validation samples: {len(X_val):,} ({len(X_val)/len(X):.1%})")
print(f"Features: {len(xgb_features)}")

print(f"\nğŸ�¯ Target distribution:")
print(f"Train - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
print(f"Val   - Mean: {y_val.mean():.4f}, Std: {y_val.std():.4f}")

# Prepare test data
X_test = test_engineered[xgb_features]

print(f"\nâœ… Data preparation completed!")
print(f"Ready for model training with {len(X_train):,} samples")


print("ğŸš€ XGBoost Model Training with Optuna Hyperparameter Optimization\n")

# Define XGBoost objective function for Optuna
def xgb_objective(trial):
    """
    Optuna objective function for XGBoost hyperparameter optimization
    """
    # Suggest hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': XGB_TREE_METHOD,  # Use GPU if available
        'device': XGB_DEVICE,
        
        # Hyperparameters to optimize
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

# Run Optuna optimization
print("ğŸ”� Starting hyperparameter optimization...")
print("â�±ï¸�  This may take several minutes...\n")

xgb_study = optuna.create_study(
    direction='minimize',
    study_name='xgboost_optimization',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)

xgb_study.optimize(
    xgb_objective,
    n_trials=50,  # Increase for better results
    show_progress_bar=True,
    n_jobs=1  # Parallel trials (set to 1 if using GPU)
)

print(f"\nâœ… Optimization completed!")
print(f"\nğŸ�† Best hyperparameters:")
for key, value in xgb_study.best_params.items():
    print(f"  {key:20s}: {value}")
print(f"\nğŸ“Š Best RMSE: {xgb_study.best_value:.6f}")


# Visualize optimization results
try:
    # Try to use plotly visualizations (will display in notebook)
    from optuna.visualization import plot_optimization_history, plot_param_importances
    
    fig1 = plot_optimization_history(xgb_study)
    fig1.update_layout(title='XGBoost Optimization History')
    fig1.show()
    
    fig2 = plot_param_importances(xgb_study)
    fig2.update_layout(title='XGBoost Hyperparameter Importance')
    fig2.show()
    
except Exception as e:
    # Fallback: create simple matplotlib plots manually
    print(f"âš ï¸�  Optuna visualization not available, creating basic plots: {e}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: Optimization history (trial values over time)
    trials = xgb_study.trials
    trial_numbers = [t.number for t in trials]
    trial_values = [t.value for t in trials]
    best_values = [min(trial_values[:i+1]) for i in range(len(trial_values))]
    
    axes[0].plot(trial_numbers, trial_values, 'o-', alpha=0.6, label='Trial Value')
    axes[0].plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
    axes[0].set_xlabel('Trial Number', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
    axes[0].set_title('XGBoost Optimization History', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Parameter importance (using best trial params)
    axes[1].text(0.5, 0.5, 'Parameter importance\nrequires plotly', 
                ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
    axes[1].set_title('XGBoost Hyperparameter Importance', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


# Train final XGBoost model with best parameters
print("ğŸ�¯ Training final XGBoost model with best parameters...\n")

best_xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': XGB_TREE_METHOD,
    'device': XGB_DEVICE,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    **xgb_study.best_params
}

xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

# Make predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)

# Evaluate
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
xgb_val_r2 = r2_score(y_val, xgb_val_pred)

print(f"\nğŸ“Š XGBoost Performance:")
print(f"Train RMSE: {xgb_train_rmse:.6f}")
print(f"Val RMSE:   {xgb_val_rmse:.6f}")
print(f"Val MAE:    {xgb_val_mae:.6f}")
print(f"Val RÂ²:     {xgb_val_r2:.6f}")
print(f"\nâœ… XGBoost model training completed!")


print("ğŸš€ LightGBM Model Training with Optuna Hyperparameter Optimization\n")

# Define LightGBM objective function for Optuna
def lgb_objective(trial):
    """
    Optuna objective function for LightGBM hyperparameter optimization
    """
    # Suggest hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'device': LGB_DEVICE,
        'gpu_platform_id': LGB_GPU_PLATFORM,
        'gpu_device_id': LGB_GPU_DEVICE,
        'verbosity': -1,
        
        # Hyperparameters to optimize
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # Train model
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

# Run Optuna optimization
print("ğŸ”� Starting hyperparameter optimization...")
print("â�±ï¸�  This may take several minutes...\n")

lgb_study = optuna.create_study(
    direction='minimize',
    study_name='lightgbm_optimization',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)

lgb_study.optimize(
    lgb_objective,
    n_trials=50,
    show_progress_bar=True,
    n_jobs=1
)

print(f"\nâœ… Optimization completed!")
print(f"\nğŸ�† Best hyperparameters:")
for key, value in lgb_study.best_params.items():
    print(f"  {key:20s}: {value}")
print(f"\nğŸ“Š Best RMSE: {lgb_study.best_value:.6f}")


# Train final LightGBM model with best parameters
print("ğŸ�¯ Training final LightGBM model with best parameters...\n")

best_lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'device': LGB_DEVICE,
    'gpu_platform_id': LGB_GPU_PLATFORM,
    'gpu_device_id': LGB_GPU_DEVICE,
    'verbosity': -1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    **lgb_study.best_params
}

lgb_model = lgb.LGBMRegressor(**best_lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[lgb.log_evaluation(50)]
)

# Make predictions
lgb_train_pred = lgb_model.predict(X_train)
lgb_val_pred = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)

# Evaluate
lgb_train_rmse = np.sqrt(mean_squared_error(y_train, lgb_train_pred))
lgb_val_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
lgb_val_mae = mean_absolute_error(y_val, lgb_val_pred)
lgb_val_r2 = r2_score(y_val, lgb_val_pred)

print(f"\nğŸ“Š LightGBM Performance:")
print(f"Train RMSE: {lgb_train_rmse:.6f}")
print(f"Val RMSE:   {lgb_val_rmse:.6f}")
print(f"Val MAE:    {lgb_val_mae:.6f}")
print(f"Val RÂ²:     {lgb_val_r2:.6f}")
print(f"\nâœ… LightGBM model training completed!")


print("ğŸš€ CatBoost Model Training with Optuna Hyperparameter Optimization\n")

# Prepare CatBoost data (using original categorical features)
X_train_cb = train_engineered.loc[X_train.index, catboost_features]
X_val_cb = train_engineered.loc[X_val.index, catboost_features]
X_test_cb = test_engineered[catboost_features]

# Define CatBoost objective function for Optuna
def cb_objective(trial):
    """
    Optuna objective function for CatBoost hyperparameter optimization
    """
    # Suggest hyperparameters
    params = {
        'loss_function': 'RMSE',
        'task_type': CB_TASK_TYPE,
        'devices': CB_DEVICES,
        'verbose': False,
        
        # Hyperparameters to optimize
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        
        'random_state': RANDOM_STATE,
        'thread_count': -1
    }
    
    # Train model
    model = cb.CatBoostRegressor(**params)
    model.fit(
        X_train_cb, y_train,
        eval_set=(X_val_cb, y_val),
        cat_features=categorical_features,
        verbose=False
    )
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_val_cb)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

# Run Optuna optimization
print("ğŸ”� Starting hyperparameter optimization...")
print("â�±ï¸�  This may take several minutes...\n")

cb_study = optuna.create_study(
    direction='minimize',
    study_name='catboost_optimization',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)

cb_study.optimize(
    cb_objective,
    n_trials=50,
    show_progress_bar=True,
    n_jobs=1
)

print(f"\nâœ… Optimization completed!")
print(f"\nğŸ�† Best hyperparameters:")
for key, value in cb_study.best_params.items():
    print(f"  {key:20s}: {value}")
print(f"\nğŸ“Š Best RMSE: {cb_study.best_value:.6f}")


# Train final CatBoost model with best parameters
print("ğŸ�¯ Training final CatBoost model with best parameters...\n")

best_cb_params = {
    'loss_function': 'RMSE',
    'task_type': CB_TASK_TYPE,
    'devices': CB_DEVICES,
    'random_state': RANDOM_STATE,
    'thread_count': -1,
    **cb_study.best_params
}

cb_model = cb.CatBoostRegressor(**best_cb_params)
cb_model.fit(
    X_train_cb, y_train,
    eval_set=(X_val_cb, y_val),
    cat_features=categorical_features,
    verbose=50
)

# Make predictions
cb_train_pred = cb_model.predict(X_train_cb)
cb_val_pred = cb_model.predict(X_val_cb)
cb_test_pred = cb_model.predict(X_test_cb)

# Evaluate
cb_train_rmse = np.sqrt(mean_squared_error(y_train, cb_train_pred))
cb_val_rmse = np.sqrt(mean_squared_error(y_val, cb_val_pred))
cb_val_mae = mean_absolute_error(y_val, cb_val_pred)
cb_val_r2 = r2_score(y_val, cb_val_pred)

print(f"\nğŸ“Š CatBoost Performance:")
print(f"Train RMSE: {cb_train_rmse:.6f}")
print(f"Val RMSE:   {cb_val_rmse:.6f}")
print(f"Val MAE:    {cb_val_mae:.6f}")
print(f"Val RÂ²:     {cb_val_r2:.6f}")
print(f"\nâœ… CatBoost model training completed!")


# Compare all models
print("ğŸ“Š Model Performance Comparison")
print("=" * 80)

models_performance = {
    'XGBoost': {'train_rmse': xgb_train_rmse, 'val_rmse': xgb_val_rmse, 
                'val_mae': xgb_val_mae, 'val_r2': xgb_val_r2},
    'LightGBM': {'train_rmse': lgb_train_rmse, 'val_rmse': lgb_val_rmse,
                 'val_mae': lgb_val_mae, 'val_r2': lgb_val_r2},
    'CatBoost': {'train_rmse': cb_train_rmse, 'val_rmse': cb_val_rmse,
                 'val_mae': cb_val_mae, 'val_r2': cb_val_r2}
}

comparison_df = pd.DataFrame(models_performance).T
comparison_df = comparison_df.round(6)
comparison_df['overfit_gap'] = comparison_df['val_rmse'] - comparison_df['train_rmse']

print("\nğŸ“ˆ Performance Metrics:")
display(comparison_df)

# Find best model
best_model_name = comparison_df['val_rmse'].idxmin()
best_rmse = comparison_df['val_rmse'].min()

print(f"\nğŸ�† Best Single Model: {best_model_name}")
print(f"   Validation RMSE: {best_rmse:.6f}")
print(f"   Validation MAE:  {comparison_df.loc[best_model_name, 'val_mae']:.6f}")
print(f"   Validation RÂ²:   {comparison_df.loc[best_model_name, 'val_r2']:.6f}")


# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE comparison
x = np.arange(len(models_performance))
width = 0.35
axes[0].bar(x - width/2, comparison_df['train_rmse'], width, label='Train', alpha=0.8, color='skyblue')
axes[0].bar(x + width/2, comparison_df['val_rmse'], width, label='Validation', alpha=0.8, color='salmon')
axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
axes[0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df.index)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# MAE comparison
axes[1].bar(comparison_df.index, comparison_df['val_mae'], alpha=0.8, color='lightgreen', edgecolor='black')
axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MAE', fontsize=12, fontweight='bold')
axes[1].set_title('Validation MAE Comparison', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# RÂ² comparison
axes[2].bar(comparison_df.index, comparison_df['val_r2'], alpha=0.8, color='gold', edgecolor='black')
axes[2].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[2].set_ylabel('RÂ² Score', fontsize=12, fontweight='bold')
axes[2].set_title('Validation RÂ² Comparison', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


print("ğŸ�¼ Creating Weighted Ensemble Model...\n")

# Calculate ensemble weights based on inverse RMSE
inv_rmse = 1 / comparison_df['val_rmse']
weights = inv_rmse / inv_rmse.sum()

print("ğŸ“Š Ensemble Weights (based on validation RMSE):")
for model_name, weight in weights.items():
    print(f"  {model_name:15s}: {weight:.4f} ({weight*100:.2f}%)")

# Create ensemble predictions
ensemble_val_pred = (
    weights['XGBoost'] * xgb_val_pred +
    weights['LightGBM'] * lgb_val_pred +
    weights['CatBoost'] * cb_val_pred
)

ensemble_test_pred = (
    weights['XGBoost'] * xgb_test_pred +
    weights['LightGBM'] * lgb_test_pred +
    weights['CatBoost'] * cb_test_pred
)

# Evaluate ensemble
ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
ensemble_val_mae = mean_absolute_error(y_val, ensemble_val_pred)
ensemble_val_r2 = r2_score(y_val, ensemble_val_pred)

print(f"\nğŸ�† Ensemble Performance:")
print(f"Validation RMSE: {ensemble_val_rmse:.6f}")
print(f"Validation MAE:  {ensemble_val_mae:.6f}")
print(f"Validation RÂ²:   {ensemble_val_r2:.6f}")

# Compare with best single model
improvement = best_rmse - ensemble_val_rmse
improvement_pct = (improvement / best_rmse) * 100

print(f"\nğŸ“ˆ Ensemble vs Best Single Model ({best_model_name}):")
print(f"RMSE Improvement: {improvement:.6f} ({improvement_pct:+.2f}%)")

if ensemble_val_rmse < best_rmse:
    print("âœ… Ensemble outperforms best single model!")
else:
    print("âš ï¸�  Best single model performs better than ensemble")


# Visualize predictions
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

predictions = {
    'XGBoost': xgb_val_pred,
    'LightGBM': lgb_val_pred,
    'CatBoost': cb_val_pred,
    'Ensemble': ensemble_val_pred
}

for idx, (name, preds) in enumerate(predictions.items()):
    axes[idx].scatter(y_val, preds, alpha=0.3, s=10)
    axes[idx].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[idx].set_xlabel('True Values', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Predictions', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'{name} Predictions', fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()
    
    # Add metrics
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    axes[idx].text(0.05, 0.95, f'RMSE: {rmse:.4f}\nRÂ²: {r2:.4f}',
                  transform=axes[idx].transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  fontsize=11, fontweight='bold')

plt.suptitle('Model Predictions vs True Values', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Get feature importances from each model
xgb_importance = pd.DataFrame({
    'feature': xgb_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

lgb_importance = pd.DataFrame({
    'feature': xgb_features,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

cb_importance = pd.DataFrame({
    'feature': catboost_features,
    'importance': cb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 20 features for each model
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# XGBoost
top_n = 20
axes[0].barh(range(top_n), xgb_importance['importance'].head(top_n), alpha=0.8, color='skyblue')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(xgb_importance['feature'].head(top_n), fontsize=9)
axes[0].set_xlabel('Importance', fontsize=11, fontweight='bold')
axes[0].set_title('XGBoost Top 20 Features', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# LightGBM
axes[1].barh(range(top_n), lgb_importance['importance'].head(top_n), alpha=0.8, color='lightgreen')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(lgb_importance['feature'].head(top_n), fontsize=9)
axes[1].set_xlabel('Importance', fontsize=11, fontweight='bold')
axes[1].set_title('LightGBM Top 20 Features', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

# CatBoost
axes[2].barh(range(top_n), cb_importance['importance'].head(top_n), alpha=0.8, color='salmon')
axes[2].set_yticks(range(top_n))
axes[2].set_yticklabels(cb_importance['feature'].head(top_n), fontsize=9)
axes[2].set_xlabel('Importance', fontsize=11, fontweight='bold')
axes[2].set_title('CatBoost Top 20 Features', fontsize=13, fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(True, alpha=0.3, axis='x')

plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print top 10 features for ensemble
print("\nğŸ�¯ Top 10 Most Important Features (XGBoost):")
for idx, row in xgb_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")


print("ğŸš€ Generating Final Competition Submissions...\n")

# Create submissions directory if it doesn't exist
os.makedirs('submissions', exist_ok=True)

# Generate timestamp for versioning
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 1. Ensemble submission (recommended)
ensemble_submission = sample_submission.copy()
ensemble_submission['accident_risk'] = ensemble_test_pred
ensemble_filename = f'submissions/ensemble_{timestamp}.csv'
ensemble_submission.to_csv(ensemble_filename, index=False)

print(f"âœ… Ensemble submission saved: {ensemble_filename}")
print(f"   Validation RMSE: {ensemble_val_rmse:.6f}")

# 2. Individual model submissions for comparison
xgb_submission = sample_submission.copy()
xgb_submission['accident_risk'] = xgb_test_pred
xgb_filename = f'submissions/xgboost_{timestamp}.csv'
xgb_submission.to_csv(xgb_filename, index=False)
print(f"\nâœ… XGBoost submission saved: {xgb_filename}")
print(f"   Validation RMSE: {xgb_val_rmse:.6f}")

lgb_submission = sample_submission.copy()
lgb_submission['accident_risk'] = lgb_test_pred
lgb_filename = f'submissions/lightgbm_{timestamp}.csv'
lgb_submission.to_csv(lgb_filename, index=False)
print(f"\nâœ… LightGBM submission saved: {lgb_filename}")
print(f"   Validation RMSE: {lgb_val_rmse:.6f}")

cb_submission = sample_submission.copy()
cb_submission['accident_risk'] = cb_test_pred
cb_filename = f'submissions/catboost_{timestamp}.csv'
cb_submission.to_csv(cb_filename, index=False)
print(f"\nâœ… CatBoost submission saved: {cb_filename}")
print(f"   Validation RMSE: {cb_val_rmse:.6f}")

# Submission validation
print("\n" + "="*80)
print("ğŸ“Š Submission Validation")
print("="*80)

for name, submission_df in [
    ('Ensemble', ensemble_submission),
    ('XGBoost', xgb_submission),
    ('LightGBM', lgb_submission),
    ('CatBoost', cb_submission)
]:
    print(f"\n{name}:")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Missing values: {submission_df['accident_risk'].isna().sum()}")
    print(f"  Min: {submission_df['accident_risk'].min():.6f}")
    print(f"  Max: {submission_df['accident_risk'].max():.6f}")
    print(f"  Mean: {submission_df['accident_risk'].mean():.6f}")
    print(f"  Std: {submission_df['accident_risk'].std():.6f}")

print("\n" + "="*80)
print("ğŸ�‰ All submissions ready for competition!")
print("="*80)


# Upload submission to Kaggle
print("ğŸ“¤ Uploading submission to Kaggle...\n")

# Submission message
submission_message = f"""
Ensemble model (XGBoost + LightGBM + CatBoost)
- Validation RMSE: {ensemble_val_rmse:.6f}
- Features: {len(xgb_features)} engineered features
- Optuna optimization: 50 trials per model
- GPU-accelerated training
- Timestamp: {timestamp}
"""

print("To submit this to Kaggle, run the following command in your terminal:")
print("\n" + "="*80)
print(f"kaggle competitions submit -c playground-series-s5e10 -f {ensemble_filename} -m \"{submission_message.strip()}\"")
print("="*80)

# Uncomment the line below to automatically submit
# !kaggle competitions submit -c playground-series-s5e10 -f {ensemble_filename} -m "{submission_message.strip()}"


print("="*80)
print("ğŸ“‹ Competition Summary")
print("="*80)

print(f"\nğŸ�¯ Competition: Playground Series S5E10 - Predicting Road Accident Risk")
print(f"ğŸ“… Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"â�° Deadline: October 31, 2025")

print(f"\nğŸ“Š Dataset:")
print(f"  Training samples: {len(train_df):,}")
print(f"  Test samples: {len(test_df):,}")
print(f"  Features: {len(feature_cols)} (original + engineered)")

print(f"\nğŸ¤– Models Trained:")
print(f"  1. XGBoost    - RMSE: {xgb_val_rmse:.6f}")
print(f"  2. LightGBM   - RMSE: {lgb_val_rmse:.6f}")
print(f"  3. CatBoost   - RMSE: {cb_val_rmse:.6f}")
print(f"  4. Ensemble   - RMSE: {ensemble_val_rmse:.6f} â­�")

print(f"\nğŸ“� Submissions Generated:")
print(f"  â€¢ {ensemble_filename} (recommended)")
print(f"  â€¢ {xgb_filename}")
print(f"  â€¢ {lgb_filename}")
print(f"  â€¢ {cb_filename}")

print(f"\nğŸ”§ Optimization:")
print(f"  â€¢ Optuna hyperparameter tuning: 50 trials per model")
print(f"  â€¢ GPU acceleration enabled")
print(f"  â€¢ Bayesian optimization (TPE sampler)")

print(f"\nğŸ’¡ Next Steps:")
print(f"  1. Submit ensemble prediction to Kaggle")
print(f"  2. Monitor leaderboard performance")
print(f"  3. Iterate on feature engineering based on results")
print(f"  4. Try additional models (Neural Networks, Stacking)")
print(f"  5. Increase Optuna trials for fine-tuning (100-200 trials)")
print(f"  6. Explore cross-validation ensemble")
print(f"  7. Build web application for Stack Overflow challenge")

print("\n" + "="*80)
print("âœ… Pipeline execution completed successfully!")
print("ğŸš€ Good luck in the competition!")
print("="*80)

