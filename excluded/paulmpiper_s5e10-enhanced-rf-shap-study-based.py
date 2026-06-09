# Install required packages (most are pre-installed on Kaggle, but optuna might not be)
# Run this cell first if you get ModuleNotFoundError
# You can comment out this cell after first successful run

# Pin scikit-learn to avoid conflicts with category-encoders
!pip install -q optuna xgboost lightgbm catboost 'scikit-learn<1.6.0' pandas numpy matplotlib seaborn scipy imbalanced-learn shap

print("âœ… All packages installed successfully!")
print("ğŸ’¡ You can comment out this cell after first run")


# ============================================================================
# CONFIGURATION
# ============================================================================
# Set to False to skip visualization generation and speed up execution

# ============================================================================
# OPTUNA OPTIMIZATION CONFIGURATION
# ============================================================================
# Control which models run Optuna optimization vs use pre-set parameters
RUN_OPTUNA = {
    'xgboost': False,  # Using best params (RMSE: 0.056223)
    'lightgbm': False,  # Using best params (RMSE: 0.056168)
    'catboost': False,  # Using best params (RMSE: 0.056181)
}

# ============================================================================
# MODEL SELECTION CONFIGURATION
# ============================================================================
# Choose which models to train and ensemble
# Set to False to skip a model entirely
# If only 1 model is True, it will be used directly (no ensemble)
# If 2+ models are True, Optuna will optimize ensemble weights
TRAIN_MODELS = {
    'xgboost': True,      # Gradient boosting (best: 0.056223)
    'lightgbm': True,     # Gradient boosting (best: 0.056168) â­� Best single model
    'catboost': True,     # Gradient boosting (best: 0.056181)
    'randomforest': False  # Tree ensemble (research shows it beats boosting!)
}

# ============================================================================
# SHAP FEATURE SELECTION CONFIGURATION
# ============================================================================
# Enable SHAP-based feature selection (NZ research: 5-8% boost with top 15 features)
USE_SHAP_FEATURES = False  # Set to True to use only top SHAP features
SHAP_TOP_K = 15            # Number of top features to keep (15-20 recommended)
USE_FEATURE_SCALING = True   # Apply StandardScaler to features (helps with numerical stability)



# Number of Optuna trials (reduce for faster execution)
# Number of Optuna trials per model (RF is much slower!)
OPTUNA_TRIALS = {
    'xgboost': 50,
    'lightgbm': 50,
    'catboost': 50,
}

# Pre-found best parameters (from previous Optuna runs)
# Use these when RUN_OPTUNA[model] = False
BEST_PARAMS = {
    'xgboost': {
        'max_depth': 12,
        'learning_rate': 0.07292200104610734,
        'n_estimators': 842,
        'min_child_weight': 5,
        'subsample': 0.9945419898530093,
        'colsample_bytree': 0.8456525428279127,
        'gamma': 0.007357032677461465,
        'reg_alpha': 1.9356337308345917,
        'reg_lambda': 2.3063700096255024
    },  # RMSE: 0.056223
    'lightgbm': {
        'num_leaves': 169,
        'max_depth': 11,
        'learning_rate': 0.018519858214832784,
        'n_estimators': 751,
        'min_child_samples': 5,
        'subsample': 0.6061425878179438,
        'colsample_bytree': 0.669979815175806,
        'reg_alpha': 0.06889053197526089,
        'reg_lambda': 8.34338254469409
    },  # RMSE: 0.056168
    'catboost': {
        'depth': 8,
        'learning_rate': 0.04756379799190705,
        'iterations': 1314,
        'l2_leaf_reg': 7.3707415558508576,
        'border_count': 246,
        'bagging_temperature': 0.35286217097756356
    },  # RMSE: 0.056181
}
# ============================================================================


GENERATE_VISUALIZATIONS = False  # Set to True for local analysis  # Set to False to skip plots on Kaggle
# ============================================================================

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
from sklearn.ensemble import RandomForestRegressor

# SMOTE for handling imbalanced data
from imblearn.over_sampling import SMOTE

# Gradient Boosting Models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Hyperparameter optimization
import optuna

# SHAP for explainable ML
import shap

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("\nğŸš€ Playground Series S5E10 - Road Accident Risk Prediction")
print(f"ğŸ“… Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nâœ… All libraries imported successfully!")
print("ğŸ“¦ Enhanced with: SMOTE, SHAP analysis")


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


# ============================================================================
# LOAD EXTERNAL DATASET FOR TARGET ENCODING (v24)
# ============================================================================
# The original synthetic dataset (100k rows) provides "external knowledge"
# We'll use it to create target-encoded features (mean accident_risk per category)

print("\n" + "=" * 80)
print("ğŸ“¦ Loading External Dataset for Target Encoding")
print("=" * 80)

try:
    # Load original synthetic dataset (100k rows)
    orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
    print(f"\nâœ… Original dataset loaded: {orig.shape}")
    print(f"   Columns: {list(orig.columns)}")

    # Verify it has accident_risk column
    if 'accident_risk' in orig.columns:
        print(f"   Target variable present: accident_risk")
        print(f"   Mean accident_risk: {orig['accident_risk'].mean():.6f}")
        print(f"   Std accident_risk:  {orig['accident_risk'].std():.6f}")
    else:
        print("   âš ï¸�  WARNING: 'accident_risk' column not found!")
        orig = None

except FileNotFoundError:
    print("\nâš ï¸�  WARNING: External dataset not found!")
    print("   Expected: /kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")
    print("   Will proceed without external target encoding features")
    orig = None
except Exception as e:
    print(f"\nâš ï¸�  ERROR loading external dataset: {e}")
    orig = None

print("=" * 80)



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
    Enhanced with findings from NZ road accident study + external dataset

    Args:
        df: Input dataframe
        is_train: Whether this is training data (affects target handling)

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    print(f"Engineering features for {len(df)} rows...")

    # ========================================================================
    # EXTERNAL DATASET TARGET ENCODING (v24) - TabM Winning Approach
    # Use original 100k dataset to create mean target per category
    # This provides "external knowledge" not available in train/test split
    # ========================================================================

    if orig is not None:
        # Base features to encode (categorical and some numerical)
        encode_features = [
            "road_type", "num_lanes", "curvature", "speed_limit",
            "lighting", "weather", "road_signs_present", "public_road",
            "time_of_day", "holiday", "school_season", "num_reported_accidents"
        ]

        for col in encode_features:
            if col in df.columns and col in orig.columns:
                # Calculate mean accident_risk per category in original dataset
                target_encoding = orig.groupby(col)["accident_risk"].mean()

                # Map to current dataframe
                df[f"orig_{col}"] = df[col].map(target_encoding)

                # Fill missing values with global mean from original dataset
                if df[f"orig_{col}"].isna().any():
                    global_mean = orig["accident_risk"].mean()
                    df[f"orig_{col}"].fillna(global_mean, inplace=True)

        print(f"   âœ… Added {len(encode_features)} target-encoded features from external dataset")
    else:
        print(f"   â�­ï¸�  Skipping external target encoding (dataset not loaded)")


    # ========================================================================
    # 1. INTERACTION FEATURES
    # ========================================================================
    print("ğŸ”§ Creating interaction features...")

    # Speed and curvature interaction (dangerous combination)
    df['speed_curvature'] = df['speed_limit'] * df['curvature']

    # Lanes and speed interaction
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']

    # Speed per lane (traffic density proxy)
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)

    # Accidents per lane
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)

    # Curvature Ã— accidents (from study: critical for fatal crashes)
    df['curvature_accidents'] = df['curvature'] * (df['num_reported_accidents'] + 1)

    # ========================================================================
    # 2. RISK SCORE FEATURES
    # ========================================================================
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

    # ========================================================================
    # 3. BINARY AGGREGATIONS
    # ========================================================================
    print("â�• Creating binary aggregation features...")

    # Total risk factors (sum of boolean risks)
    df['total_risk_factors'] = (
        (~df['road_signs_present']).astype(int) +
        (~df['public_road']).astype(int) +
        df['holiday'].astype(int) +
        df['school_season'].astype(int)
    )

    # Poor conditions (weather + lighting)
    df['poor_conditions'] = ((df['weather'] != 'clear') | (df['lighting'] != 'daylight')).astype(int)

    # High risk scenario
    df['high_risk_scenario'] = (
        (df['speed_limit'] > 45) &
        (df['curvature'] > 0.5) &
        (df['weather'] != 'clear')
    ).astype(int)

    # ========================================================================
    # 4. TIME-BASED FEATURES
    # ========================================================================
    print("â�° Creating time-based features...")

    # Time of day risk (ordinal)
    time_risk_map = {'morning': 1, 'afternoon': 0, 'evening': 2, 'night': 3}
    df['time_risk'] = df['time_of_day'].map(time_risk_map).fillna(0)

    # Rush hour indicator
    df['rush_hour'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)

    # Peak risk time (night + poor weather)
    df['peak_risk_time'] = (
        (df['time_of_day'] == 'night') &
        (df['weather'] != 'clear')
    ).astype(int)

    # ========================================================================
    # 5. ROAD TYPE FEATURES
    # ========================================================================
    print("ğŸ›£ï¸� Creating road type features...")

    df['is_urban'] = (df['road_type'] == 'urban').astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_rural'] = (df['road_type'] == 'rural').astype(int)

    # ========================================================================
    # 6. STUDY-BASED INTERACTION FEATURES
    # ========================================================================
    print("ğŸ”¬ Creating study-based interaction features...")

    # Weather Ã— Lighting interaction
    df['weather_lighting'] = df['weather_risk'] * df['lighting_risk']

    # Speed Ã— Weather interaction
    df['speed_weather'] = df['speed_limit'] * df['weather_risk']

    # Road type Ã— Speed interaction
    df['highway_speed'] = df['is_highway'] * df['speed_limit']
    df['urban_speed'] = df['is_urban'] * df['speed_limit']

    # Night Ã— No signs interaction
    df['night_no_signs'] = (
        (df['lighting'] == 'night') &
        (~df['road_signs_present'])
    ).astype(int)

    # Curvature Ã— Lighting
    df['curve_night_risk'] = df['curvature'] * df['lighting_risk']

    # Accidents Ã— Road type
    df['accidents_road_type'] = df['num_reported_accidents'] * (df['is_highway'] + 1)

    # ========================================================================
    # 7. STATISTICAL FEATURES
    # ========================================================================
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

    # ========================================================================
    # 8. BASE RISK FORMULA
    # ========================================================================
    print("ğŸ�¯ Creating base risk formula feature...")

    df["base_risk"] = (
        0.3 * df["curvature"] +
        0.2 * (df["lighting"] == "night").astype(int) +
        0.1 * (df["weather"] != "clear").astype(int) +
        0.2 * (df["speed_limit"] >= 60).astype(int) +
        0.1 * (df["num_reported_accidents"] > 2).astype(int)
    )

    # ========================================================================
    # 9. POWER TRANSFORMATIONS
    # ========================================================================
    print("ğŸ“� Creating power transformation features...")

    # Curvature transformations
    df["curvature_sqrt"] = np.sqrt(df["curvature"] + 1e-6)
    df["curvature_squared"] = df["curvature"] ** 2
    df["curvature_cubed"] = df["curvature"] ** 3
    df["curvature_log"] = np.log1p(df["curvature"] + 1e-6)

    # Speed transformations
    df["speed_limit_log"] = np.log1p(df["speed_limit"])
    df["speed_limit_squared"] = df["speed_limit"] ** 2

    # Accidents transformations
    df["num_reported_accidents_sqrt"] = np.sqrt(df["num_reported_accidents"] + 1e-6)
    df["num_reported_accidents_log"] = np.log1p(df["num_reported_accidents"])

    # ========================================================================
    # 10. EXTREME CASE DETECTION
    # ========================================================================
    print("âš ï¸�  Creating extreme case features...")

    df["extreme_risk_1"] = (
        (df["curvature"] > df["curvature"].quantile(0.9)) &
        (df["speed_limit"] > 60) &
        (df["lighting"] == "night")
    ).astype(int)

    df["extreme_risk_2"] = (
        (df["num_reported_accidents"] >= 3) &
        (df["weather"] != "clear") &
        (df["visibility_risk"] > df["visibility_risk"].quantile(0.8))
    ).astype(int)

    # ========================================================================
    # 11. DIGIT EXTRACTION
    # ========================================================================
    print("ğŸ”� Extracting digits from numerical features...")

    for col in ['curvature', 'speed_limit']:
        for power in range(-2, 3):
            digit_col = f'{col}_digit_{power}'
            df[digit_col] = ((df[col] * (10 ** power)) % 10).astype(np.int8)

            # Remove if constant
            if df[digit_col].nunique() == 1:
                df.drop(columns=[digit_col], inplace=True)

    # ========================================================================
    # 12. ADVANCED ENGINEERED FEATURES (v23) - Based on SHAP Insights
    # base_risk was 75% more important - create more features like it
    # ========================================================================
    print("ğŸŒŸ Creating advanced risk features (v23)...")

    # Momentum risk: mass Ã— velocity / braking ability
    df["momentum_risk"] = (
        df["num_lanes"] * df["num_reported_accidents"] * df["speed_limit"] /
        (1 + df["curvature"] + 1e-6)
    )

    # Visibility-adjusted risk
    df["visibility_adjusted_risk"] = (
        df["base_risk"] * (1 + df["lighting_risk"] + df["weather_risk"])
    )

    # Terrain complexity
    df["terrain_complexity"] = (
        df["curvature"] * df["speed_limit"] / 100
    )

    # Triple interactions
    df["wet_speed_curve"] = (
        df["weather_risk"] * df["speed_curvature"] * df["lighting_risk"]
    )

    # Polynomial features for base_risk
    df["base_risk_squared"] = df["base_risk"] ** 2
    df["base_risk_sqrt"] = np.sqrt(df["base_risk"] + 1e-6)
    df["base_risk_log"] = np.log1p(df["base_risk"])

    # Binned categories
    df["base_risk_category"] = pd.cut(
        df["base_risk"],
        bins=[-np.inf, 0.3, 0.6, 1.0, 2.0, np.inf],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

    df["speed_category"] = pd.cut(
        df["speed_limit"],
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # Combined risk scores
    df["combined_risk_1"] = (
        df["base_risk"] * df["lighting_risk"] * df["weather_risk"]
    )

    # Advanced curvature features
    df["curve_lanes"] = df["curvature"] * df["num_lanes"]
    df["curve_gradient"] = df["curvature"] ** 2  # gradient column not available

    # Speed features
    df["speed_density"] = df["speed_limit"] * df["num_reported_accidents"] / 1000

    print(f"   âœ… Added 20+ advanced engineered features")

    return df



# Apply feature engineering
print("Applying enhanced feature engineering to datasets...")
print("Based on 1st place techniques + competition discussions")
print()

train_engineered = engineer_features(train_df, is_train=True)
test_engineered = engineer_features(test_df, is_train=False)

print()
print("New features created:")
new_features = [col for col in train_engineered.columns if col not in train_df.columns]
print(f"Total new features: {len(new_features)}")
print(f"Feature list (first 20): {new_features[:20]}")
if len(new_features) > 20:
    print(f"... and {len(new_features) - 20} more")


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


# Check if we should run Optuna or use pre-set params
if RUN_OPTUNA['xgboost']:
    # Run Optuna optimization
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
        n_trials=OPTUNA_TRIALS['xgboost'],  # Increase for better results
        show_progress_bar=True,
        n_jobs=1  # Parallel trials (set to 1 if using GPU)
    )
    
    print(f"\nâœ… Optimization completed!")
    print(f"\nğŸ�† Best hyperparameters:")
    for key, value in xgb_study.best_params.items():
        print(f"  {key:20s}: {value}")
    print(f"\nğŸ“Š Best RMSE: {xgb_study.best_value:.6f}")
else:
    # Use pre-set parameters
    print(f"âš¡ Skipping Optuna for XGBOOST - using pre-set parameters")
    xgboost_study = type('Study', (), {'best_params': BEST_PARAMS['xgboost'], 'best_value': 0.0})()



# Only show visualization if Optuna was run for xgboost
if RUN_OPTUNA["xgboost"]:
    # Only show visualization if Optuna was run
    if RUN_OPTUNA["xgboost"]:
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
    else:
        print("âš¡ Skipped XGBoost Optuna visualization (using pre-set params)")
else:
    print("âš¡ Skipped XGBOOST Optuna visualization (using pre-set params)")


# Train final XGBoost model with best parameters
print("ğŸ�¯ Training final XGBoost model with best parameters...\n")

best_xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': XGB_TREE_METHOD,
    'device': XGB_DEVICE,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    **(xgb_study.best_params if RUN_OPTUNA["xgboost"] else BEST_PARAMS["xgboost"])
}

xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

# Make predictions and CLIP to [0, 1] range
xgb_train_pred = np.clip(xgb_model.predict(X_train), 0, 1)
xgb_val_pred = np.clip(xgb_model.predict(X_val), 0, 1)
xgb_test_pred = np.clip(xgb_model.predict(X_test), 0, 1)

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
print(f"âœ… Predictions clipped to [0, 1] range")


# Check if we should run Optuna or use pre-set params
if RUN_OPTUNA['lightgbm']:
    # Run Optuna optimization
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
        n_trials=OPTUNA_TRIALS['lightgbm'],
        show_progress_bar=True,
        n_jobs=1
    )
    
    print(f"\nâœ… Optimization completed!")
    print(f"\nğŸ�† Best hyperparameters:")
    for key, value in lgb_study.best_params.items():
        print(f"  {key:20s}: {value}")
    print(f"\nğŸ“Š Best RMSE: {lgb_study.best_value:.6f}")
else:
    # Use pre-set parameters
    print(f"âš¡ Skipping Optuna for LIGHTGBM - using pre-set parameters")
    lightgbm_study = type('Study', (), {'best_params': BEST_PARAMS['lightgbm'], 'best_value': 0.0})()



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
    **(lgb_study.best_params if RUN_OPTUNA["lightgbm"] else BEST_PARAMS["lightgbm"])
}

lgb_model = lgb.LGBMRegressor(**best_lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[lgb.log_evaluation(50)]
)

# Make predictions and CLIP to [0, 1] range
lgb_train_pred = np.clip(lgb_model.predict(X_train), 0, 1)
lgb_val_pred = np.clip(lgb_model.predict(X_val), 0, 1)
lgb_test_pred = np.clip(lgb_model.predict(X_test), 0, 1)

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
print(f"âœ… Predictions clipped to [0, 1] range")


print("ğŸš€ CatBoost Model Training with Optuna Hyperparameter Optimization\n")

# Prepare CatBoost data (using original categorical features)
# This ALWAYS runs, regardless of RUN_OPTUNA setting
X_train_cb = train_engineered.loc[X_train.index, catboost_features]
X_val_cb = train_engineered.loc[X_val.index, catboost_features]
X_test_cb = test_engineered[catboost_features]

# Check if we should run Optuna or use pre-set params
if RUN_OPTUNA['catboost']:
    # Run Optuna optimization

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
        n_trials=OPTUNA_TRIALS['catboost'],
        show_progress_bar=True,
        n_jobs=1
    )

    print(f"\nâœ… Optimization completed!")
    print(f"\nğŸ�† Best hyperparameters:")
    for key, value in cb_study.best_params.items():
        print(f"  {key:20s}: {value}")
    print(f"\nğŸ“Š Best RMSE: {cb_study.best_value:.6f}")

else:
    # Use pre-set parameters
    print(f"âš¡ Skipping Optuna for CATBOOST - using pre-set parameters")
    cb_study = type('Study', (), {'best_params': BEST_PARAMS['catboost'], 'best_value': 0.0})()



# Train final CatBoost model with best parameters
print("ğŸ�¯ Training final CatBoost model with best parameters...\n")

best_cb_params = {
    'loss_function': 'RMSE',
    'task_type': CB_TASK_TYPE,
    'devices': CB_DEVICES,
    'random_state': RANDOM_STATE,
    'thread_count': -1,
    **(cb_study.best_params if RUN_OPTUNA["catboost"] else BEST_PARAMS["catboost"])
}

cb_model = cb.CatBoostRegressor(**best_cb_params)
cb_model.fit(
    X_train_cb, y_train,
    eval_set=(X_val_cb, y_val),
    cat_features=categorical_features,
    verbose=50
)

# Make predictions and CLIP to [0, 1] range
cb_train_pred = np.clip(cb_model.predict(X_train_cb), 0, 1)
cb_val_pred = np.clip(cb_model.predict(X_val_cb), 0, 1)
cb_test_pred = np.clip(cb_model.predict(X_test_cb), 0, 1)

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
print(f"âœ… Predictions clipped to [0, 1] range")


print("\n" + "=" * 80)
print("ğŸ”� SHAP Feature Importance Analysis")
print("=" * 80)
print("\nğŸ“Š Calculating SHAP values for all models...")
print("   (This may take 3-5 minutes)\n")

# Sample data for SHAP (use 500 samples for speed)
shap_sample_size = min(500, len(X_train))
sample_indices = X_train.sample(n=shap_sample_size, random_state=RANDOM_STATE).index

# Create sample datasets
X_shap_sample = X_train.loc[sample_indices]

# Dictionary to store SHAP values
shap_values_dict = {}
feature_importance_dict = {}

# 1. XGBoost SHAP values
if TRAIN_MODELS['xgboost']:
    print("â�±ï¸�  Calculating XGBoost SHAP values...")
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_shap_sample)

    # Calculate mean absolute SHAP values for each feature
    shap_importance_xgb = np.abs(shap_values_xgb).mean(axis=0)
    feature_importance_dict['XGBoost'] = shap_importance_xgb
    shap_values_dict['XGBoost'] = shap_values_xgb
    print(f"   âœ… XGBoost SHAP values calculated")

# 2. LightGBM SHAP values
if TRAIN_MODELS['lightgbm']:
    print("â�±ï¸�  Calculating LightGBM SHAP values...")
    explainer_lgb = shap.TreeExplainer(lgb_model)
    shap_values_lgb = explainer_lgb.shap_values(X_shap_sample)

    shap_importance_lgb = np.abs(shap_values_lgb).mean(axis=0)
    feature_importance_dict['LightGBM'] = shap_importance_lgb
    shap_values_dict['LightGBM'] = shap_values_lgb
    print(f"   âœ… LightGBM SHAP values calculated")

# 3. CatBoost SHAP values (needs original categorical features)
if TRAIN_MODELS['catboost']:
    print("â�±ï¸�  Calculating CatBoost SHAP values...")

    # Use original categorical data for CatBoost
    X_shap_sample_cb = train_engineered.loc[sample_indices, catboost_features]

    # Create CatBoost Pool with proper categorical features
    import catboost as cb_module
    shap_pool = cb_module.Pool(
        X_shap_sample_cb,
        cat_features=categorical_features
    )

    # Calculate SHAP values using CatBoost's native method
    shap_values_cb = cb_model.get_feature_importance(
        data=shap_pool,
        type='ShapValues'
    )

    # Remove the last column (it's the expected value)
    shap_values_cb = shap_values_cb[:, :-1]

    shap_importance_cb = np.abs(shap_values_cb).mean(axis=0)
    feature_importance_dict['CatBoost'] = shap_importance_cb
    shap_values_dict['CatBoost'] = shap_values_cb
    print(f"   âœ… CatBoost SHAP values calculated")

# Aggregate SHAP importance across all models (average)
print("\nğŸ“Š Aggregating feature importance across models...")
all_importances = np.array(list(feature_importance_dict.values()))
avg_importance = all_importances.mean(axis=0)

# Create feature importance dataframe
feature_names = X_train.columns.tolist()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_importance
})

# Add individual model importances
for model_name, importance in feature_importance_dict.items():
    importance_df[f'{model_name}_Importance'] = importance

# Sort by average importance
importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

print(f"\nğŸ�† Top 20 Most Important Features (Averaged across models):")
print("=" * 80)
for idx, row in importance_df.head(20).iterrows():
    print(f"{idx+1:2d}. {row['Feature']:30s} | Importance: {row['Importance']:.6f}")

print("\n" + "=" * 80)
print(f"âœ… SHAP analysis complete!")
print(f"   Total features: {len(feature_names)}")
print(f"   Top {SHAP_TOP_K} features selected for optional retraining")
print("=" * 80)

# Store top features for potential use
top_features = importance_df.head(SHAP_TOP_K)['Feature'].tolist()
print(f"\nğŸ“‹ Top {SHAP_TOP_K} features: {top_features}")



if TRAIN_MODELS['randomforest']:
    print("ğŸŒ² Random Forest Model Training...\n")
    print("ğŸ“Š Using hyperparameters from NZ Road Accident Research")
    print("   (Random Forest: 81.45% accuracy vs XGBoost: 78.52%)\n")

    # Random Forest with research-based hyperparameters
    # Source: Ahmed et al., 2023 NZ Road Accident Study
    rf_model = RandomForestRegressor(
        n_estimators=500,        # More trees = more stable
        max_depth=15,            # Prevent overfitting
        min_samples_split=20,    # Conservative splitting
        min_samples_leaf=10,     # Minimum samples per leaf
        max_features='sqrt',     # Feature randomness for diversity
        random_state=RANDOM_STATE,
        n_jobs=-1,               # Use all CPU cores
        verbose=0
    )

    print("â�±ï¸�  Training Random Forest (this may take a few minutes)...")
    rf_model.fit(X_train, y_train)

    # Make predictions and CLIP to [0, 1] range
    rf_train_pred = np.clip(rf_model.predict(X_train), 0, 1)
    rf_val_pred = np.clip(rf_model.predict(X_val), 0, 1)
    rf_test_pred = np.clip(rf_model.predict(X_test), 0, 1)

    # Evaluate
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_pred))
    rf_val_mae = mean_absolute_error(y_val, rf_val_pred)
    rf_val_r2 = r2_score(y_val, rf_val_pred)

    print(f"\nğŸ“Š Random Forest Performance:")
    print(f"Train RMSE: {rf_train_rmse:.6f}")
    print(f"Val RMSE:   {rf_val_rmse:.6f}")
    print(f"Val MAE:    {rf_val_mae:.6f}")
    print(f"Val RÂ²:     {rf_val_r2:.6f}")
    print(f"\nâœ… Random Forest model training completed!")
    print(f"âœ… Predictions clipped to [0, 1] range")

    # Compare with boosting models
    print(f"\nğŸ”� Diversity Check:")
    print(f"Research insight: RF learns different patterns than gradient boosting")
else:
    print("â�­ï¸�  Skipping Random Forest (TRAIN_MODELS['randomforest'] = False)")



if USE_SHAP_FEATURES:
    print("\n" + "=" * 80)
    print("ğŸŒŸ RETRAINING WITH TOP SHAP FEATURES")
    print("=" * 80)
    print(f"\nğŸ“Š Using only top {SHAP_TOP_K} features (NZ research: 5-8% boost)")
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   Reduced features:  {len(top_features)}")
    print(f"   Feature scaling:   {'Enabled' if USE_FEATURE_SCALING else 'Disabled'}\n")

    # Create reduced feature datasets
    X_train_reduced = X_train[top_features].copy()
    X_val_reduced = X_val[top_features].copy()
    X_test_reduced = X_test[top_features].copy()

    # Apply feature scaling if enabled
    if USE_FEATURE_SCALING:
        print("ğŸ“� Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
        X_val_reduced_scaled = scaler.transform(X_val_reduced)
        X_test_reduced_scaled = scaler.transform(X_test_reduced)

        # Convert back to DataFrames with same column names
        X_train_reduced = pd.DataFrame(X_train_reduced_scaled, columns=top_features, index=X_train_reduced.index)
        X_val_reduced = pd.DataFrame(X_val_reduced_scaled, columns=top_features, index=X_val_reduced.index)
        X_test_reduced = pd.DataFrame(X_test_reduced_scaled, columns=top_features, index=X_test_reduced.index)
        print("   âœ… Features scaled (mean=0, std=1)\n")

    # Store original predictions for comparison
    xgb_val_pred_original = xgb_val_pred.copy() if TRAIN_MODELS['xgboost'] else None
    lgb_val_pred_original = lgb_val_pred.copy() if TRAIN_MODELS['lightgbm'] else None
    cb_val_pred_original = cb_val_pred.copy() if TRAIN_MODELS['catboost'] else None

    xgb_val_rmse_original = xgb_val_rmse if TRAIN_MODELS['xgboost'] else None
    lgb_val_rmse_original = lgb_val_rmse if TRAIN_MODELS['lightgbm'] else None
    cb_val_rmse_original = cb_val_rmse if TRAIN_MODELS['catboost'] else None

    # ========================================================================
    # RETRAIN XGBoost with reduced features
    # ========================================================================
    if TRAIN_MODELS['xgboost']:
        print("ğŸš€ Retraining XGBoost with top features...")

        xgb_model_reduced = xgb.XGBRegressor(**best_xgb_params)
        xgb_model_reduced.fit(
            X_train_reduced, y_train,
            eval_set=[(X_val_reduced, y_val)],
            verbose=False
        )

        xgb_val_pred = np.clip(xgb_model_reduced.predict(X_val_reduced), 0, 1)
        xgb_test_pred = np.clip(xgb_model_reduced.predict(X_test_reduced), 0, 1)

        xgb_train_rmse = np.sqrt(mean_squared_error(y_train, np.clip(xgb_model_reduced.predict(X_train_reduced), 0, 1)))
        xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
        xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
        xgb_val_r2 = r2_score(y_val, xgb_val_pred)

        improvement = xgb_val_rmse_original - xgb_val_rmse
        print(f"   XGBoost: {xgb_val_rmse_original:.6f} â†’ {xgb_val_rmse:.6f} ({improvement:+.6f})")

    # ========================================================================
    # RETRAIN LightGBM with reduced features
    # ========================================================================
    if TRAIN_MODELS['lightgbm']:
        print("ğŸš€ Retraining LightGBM with top features...")

        lgb_model_reduced = lgb.LGBMRegressor(**best_lgb_params)
        lgb_model_reduced.fit(
            X_train_reduced, y_train,
            eval_set=[(X_val_reduced, y_val)],
            callbacks=[lgb.log_evaluation(0)]
        )

        lgb_val_pred = np.clip(lgb_model_reduced.predict(X_val_reduced), 0, 1)
        lgb_test_pred = np.clip(lgb_model_reduced.predict(X_test_reduced), 0, 1)

        lgb_train_rmse = np.sqrt(mean_squared_error(y_train, np.clip(lgb_model_reduced.predict(X_train_reduced), 0, 1)))
        lgb_val_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
        lgb_val_mae = mean_absolute_error(y_val, lgb_val_pred)
        lgb_val_r2 = r2_score(y_val, lgb_val_pred)

        improvement = lgb_val_rmse_original - lgb_val_rmse
        print(f"   LightGBM: {lgb_val_rmse_original:.6f} â†’ {lgb_val_rmse:.6f} ({improvement:+.6f})")

    # ========================================================================
    # RETRAIN CatBoost with reduced features
    # ========================================================================
    if TRAIN_MODELS['catboost']:
        print("ğŸš€ Retraining CatBoost with top features...")

        # CatBoost needs original categorical data
        # First, get the categorical features that are in top_features
        categorical_features_reduced = [f for f in categorical_features if f in top_features]

        if USE_FEATURE_SCALING and len(categorical_features_reduced) > 0:
            # If scaling is enabled, we need to handle categorical features specially
            # For simplicity, use the scaled data but tell CatBoost which features were categorical
            print(f"   Note: {len(categorical_features_reduced)} categorical features in top {SHAP_TOP_K}")
            print(f"   CatBoost will treat them as numerical after scaling")

            cb_model_reduced = cb.CatBoostRegressor(**best_cb_params)
            cb_model_reduced.fit(
                X_train_reduced, y_train,
                eval_set=(X_val_reduced, y_val),
                verbose=False
            )

            cb_val_pred = np.clip(cb_model_reduced.predict(X_val_reduced), 0, 1)
            cb_test_pred = np.clip(cb_model_reduced.predict(X_test_reduced), 0, 1)

        else:
            # No scaling or no categorical features - use original method
            X_train_cb_reduced = train_engineered.loc[X_train.index, top_features]
            X_val_cb_reduced = train_engineered.loc[X_val.index, top_features]
            X_test_cb_reduced = test_engineered[top_features]

            cb_model_reduced = cb.CatBoostRegressor(**best_cb_params)
            cb_model_reduced.fit(
                X_train_cb_reduced, y_train,
                eval_set=(X_val_cb_reduced, y_val),
                cat_features=categorical_features_reduced,
                verbose=False
            )

            cb_val_pred = np.clip(cb_model_reduced.predict(X_val_cb_reduced), 0, 1)
            cb_test_pred = np.clip(cb_model_reduced.predict(X_test_cb_reduced), 0, 1)

        cb_train_rmse = np.sqrt(mean_squared_error(y_train, np.clip(cb_model_reduced.predict(X_train_reduced if USE_FEATURE_SCALING else train_engineered.loc[X_train.index, top_features]), 0, 1)))
        cb_val_rmse = np.sqrt(mean_squared_error(y_val, cb_val_pred))
        cb_val_mae = mean_absolute_error(y_val, cb_val_pred)
        cb_val_r2 = r2_score(y_val, cb_val_pred)

        improvement = cb_val_rmse_original - cb_val_rmse
        print(f"   CatBoost: {cb_val_rmse_original:.6f} â†’ {cb_val_rmse:.6f} ({improvement:+.6f})")

    print("\n" + "=" * 80)
    print("âœ… Retraining with reduced features complete!")
    print("=" * 80)
    print("\nğŸ“Š Summary of Improvements:")
    if TRAIN_MODELS['xgboost']:
        print(f"   XGBoost:  {xgb_val_rmse_original:.6f} â†’ {xgb_val_rmse:.6f} ({(xgb_val_rmse_original-xgb_val_rmse):+.6f})")
    if TRAIN_MODELS['lightgbm']:
        print(f"   LightGBM: {lgb_val_rmse_original:.6f} â†’ {lgb_val_rmse:.6f} ({(lgb_val_rmse_original-lgb_val_rmse):+.6f})")
    if TRAIN_MODELS['catboost']:
        print(f"   CatBoost: {cb_val_rmse_original:.6f} â†’ {cb_val_rmse:.6f} ({(cb_val_rmse_original-cb_val_rmse):+.6f})")

    print("\nğŸ’¡ Note: Models above have been updated with reduced features.")
    print("   Ensemble optimization will use these new predictions.\n")

else:
    print("\nâ�­ï¸�  Skipping SHAP feature selection (USE_SHAP_FEATURES = False)")
    print("   Using all features for training\n")



# Compare all TRAINED models
print("Model Performance Comparison")
print("=" * 80)

models_performance = {}

if TRAIN_MODELS['xgboost']:
    models_performance['XGBoost'] = {
        'Train RMSE': xgb_train_rmse,
        'Val RMSE': xgb_val_rmse,
        'Val MAE': xgb_val_mae,
        'Val RÂ²': xgb_val_r2
    }

if TRAIN_MODELS['lightgbm']:
    models_performance['LightGBM'] = {
        'Train RMSE': lgb_train_rmse,
        'Val RMSE': lgb_val_rmse,
        'Val MAE': lgb_val_mae,
        'Val RÂ²': lgb_val_r2
    }

if TRAIN_MODELS['catboost']:
    models_performance['CatBoost'] = {
        'Train RMSE': cb_train_rmse,
        'Val RMSE': cb_val_rmse,
        'Val MAE': cb_val_mae,
        'Val RÂ²': cb_val_r2
    }

if TRAIN_MODELS['randomforest']:
    models_performance['RandomForest'] = {
        'Train RMSE': rf_train_rmse,
        'Val RMSE': rf_val_rmse,
        'Val MAE': rf_val_mae,
        'Val RÂ²': rf_val_r2
    }

# Display comparison
for model_name, metrics in models_performance.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:12s}: {value:.6f}")

# Find best single model
val_rmses = {name: metrics['Val RMSE'] for name, metrics in models_performance.items()}
best_model_name = min(val_rmses, key=val_rmses.get)
best_rmse = val_rmses[best_model_name]

print("\n" + "=" * 80)
print(f"ğŸ�† Best Single Model: {best_model_name} (RMSE: {best_rmse:.6f})")
print("=" * 80)

# Check how many models were trained
trained_model_count = sum(TRAIN_MODELS.values())
print(f"\nğŸ“Š Trained {trained_model_count} model(s)")

if trained_model_count == 1:
    print("âš¡ Using single model (no ensemble needed)")
elif trained_model_count >= 2:
    print("ğŸ�¼ Will create ensemble with Optuna-optimized weights")



# Check how many models were trained
trained_model_count = sum(TRAIN_MODELS.values())

if trained_model_count == 1:
    # Use single model directly
    print("=" * 80)
    print("âš¡ SINGLE MODEL MODE")
    print("=" * 80)
    print(f"\nğŸ�¯ Only ONE model trained: {best_model_name}")
    print(f"   Validation RMSE: {best_rmse:.6f}")
    print(f"   Using single model predictions (no ensemble needed)\n")

    # Map to appropriate predictions
    if TRAIN_MODELS['xgboost'] and best_model_name == 'XGBoost':
        ensemble_val_pred = xgb_val_pred
        ensemble_test_pred = xgb_test_pred
    elif TRAIN_MODELS['lightgbm'] and best_model_name == 'LightGBM':
        ensemble_val_pred = lgb_val_pred
        ensemble_test_pred = lgb_test_pred
    elif TRAIN_MODELS['catboost'] and best_model_name == 'CatBoost':
        ensemble_val_pred = cb_val_pred
        ensemble_test_pred = cb_test_pred
    elif TRAIN_MODELS['randomforest'] and best_model_name == 'RandomForest':
        ensemble_val_pred = rf_val_pred
        ensemble_test_pred = rf_test_pred

    ensemble_val_rmse = best_rmse
    ensemble_val_mae = models_performance[best_model_name]['Val MAE']
    ensemble_val_r2 = models_performance[best_model_name]['Val RÂ²']

    print(f"âœ… Single model ready for submission")

elif trained_model_count >= 2:
    # Create ensemble with Optuna optimization
    print("=" * 80)
    print(f"ğŸ�¼ ENSEMBLE MODE ({trained_model_count} models)")
    print("=" * 80)
    print(f"\nğŸ”� Optimizing Ensemble Weights with Optuna...\n")

    # Collect predictions from trained models only
    model_predictions = {}
    if TRAIN_MODELS['xgboost']:
        model_predictions['xgb'] = (xgb_val_pred, xgb_test_pred)
    if TRAIN_MODELS['lightgbm']:
        model_predictions['lgb'] = (lgb_val_pred, lgb_test_pred)
    if TRAIN_MODELS['catboost']:
        model_predictions['cb'] = (cb_val_pred, cb_test_pred)
    if TRAIN_MODELS['randomforest']:
        model_predictions['rf'] = (rf_val_pred, rf_test_pred)

    # Define objective function for ensemble weight optimization
    def ensemble_objective(trial):
        """Find optimal weights for ensemble"""
        weights = {}
        for model_name in model_predictions.keys():
            weights[model_name] = trial.suggest_float(f'w_{model_name}', 0.0, 1.0)

        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total == 0:
            return 1.0  # Avoid division by zero

        weights = {k: v/total for k, v in weights.items()}

        # Create ensemble with these weights
        ensemble_pred = np.zeros_like(list(model_predictions.values())[0][0])
        for model_name, (val_pred, _) in model_predictions.items():
            ensemble_pred += weights[model_name] * val_pred

        ensemble_pred = np.clip(ensemble_pred, 0, 1)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        return rmse

    # Run optimization
    ensemble_study = optuna.create_study(
        direction='minimize',
        study_name='ensemble_weights',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    ensemble_study.optimize(
        ensemble_objective,
        n_trials=100,  # Quick optimization
        show_progress_bar=True,
        n_jobs=1
    )

    print(f"\nâœ… Optimal weights found!")
    print(f"Best RMSE: {ensemble_study.best_value:.6f}")

    # Get optimized weights
    best_weights = ensemble_study.best_params
    total = sum(best_weights.values())
    optimized_weights = {k: v/total for k, v in best_weights.items()}

    print(f"\nğŸ“Š Optimized Ensemble Weights:")
    for model_name, weight in optimized_weights.items():
        display_name = {
            'w_xgb': 'XGBoost',
            'w_lgb': 'LightGBM',
            'w_cb': 'CatBoost',
            'w_rf': 'RandomForest'
        }.get(model_name, model_name)
        print(f"  {display_name:12s}: {weight:.4f} ({weight*100:.2f}%)")

    # Create optimized ensemble predictions
    ensemble_val_pred = np.zeros_like(y_val, dtype=float)
    ensemble_test_pred = np.zeros_like(X_test.index, dtype=float)

    for model_name, (val_pred, test_pred) in model_predictions.items():
        weight = optimized_weights[f'w_{model_name}']
        ensemble_val_pred += weight * val_pred
        ensemble_test_pred += weight * test_pred

    # CLIP ensemble predictions to [0, 1] range
    ensemble_val_pred = np.clip(ensemble_val_pred, 0, 1)
    ensemble_test_pred = np.clip(ensemble_test_pred, 0, 1)

    # Evaluate ensemble
    ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
    ensemble_val_mae = mean_absolute_error(y_val, ensemble_val_pred)
    ensemble_val_r2 = r2_score(y_val, ensemble_val_pred)

    print(f"\nğŸ�† Optimized Ensemble Performance:")
    print(f"Validation RMSE: {ensemble_val_rmse:.6f}")
    print(f"Validation MAE:  {ensemble_val_mae:.6f}")
    print(f"Validation RÂ²:   {ensemble_val_r2:.6f}")

    # Compare with best single model
    improvement = best_rmse - ensemble_val_rmse
    improvement_pct = (improvement / best_rmse) * 100

    print(f"\nğŸ“ˆ Ensemble vs Best Single Model ({best_model_name}):")
    print(f"Best Single:    {best_rmse:.6f}")
    print(f"Ensemble:       {ensemble_val_rmse:.6f}")
    print(f"Improvement:    {improvement:.6f} ({improvement_pct:+.2f}%)")

    if ensemble_val_rmse < best_rmse:
        print("âœ… Ensemble outperforms best single model!")
    else:
        print("âš ï¸�  Best single model performs better than ensemble")
        print("   (This suggests models are learning similar patterns)")

else:
    print("\nâš ï¸�  ERROR: No models trained! Please enable at least one model in TRAIN_MODELS")



if trained_model_count >= 2:
    # Ridge Stacking Meta-Learner (only for ensembles)
    print("\nğŸ�—ï¸� Creating Ridge Stacking Meta-Learner...")

    from sklearn.linear_model import Ridge

    # Stack validation predictions
    stack_features_val = []
    if TRAIN_MODELS['xgboost']:
        stack_features_val.append(xgb_val_pred.reshape(-1, 1))
    if TRAIN_MODELS['lightgbm']:
        stack_features_val.append(lgb_val_pred.reshape(-1, 1))
    if TRAIN_MODELS['catboost']:
        stack_features_val.append(cb_val_pred.reshape(-1, 1))
    if TRAIN_MODELS['randomforest']:
        stack_features_val.append(rf_val_pred.reshape(-1, 1))

    X_stack_val = np.hstack(stack_features_val)

    # Stack test predictions
    stack_features_test = []
    if TRAIN_MODELS['xgboost']:
        stack_features_test.append(xgb_test_pred.reshape(-1, 1))
    if TRAIN_MODELS['lightgbm']:
        stack_features_test.append(lgb_test_pred.reshape(-1, 1))
    if TRAIN_MODELS['catboost']:
        stack_features_test.append(cb_test_pred.reshape(-1, 1))
    if TRAIN_MODELS['randomforest']:
        stack_features_test.append(rf_test_pred.reshape(-1, 1))

    X_stack_test = np.hstack(stack_features_test)

    # Train Ridge meta-learner
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_stack_val, y_val)

    # Make predictions
    stack_val_pred = np.clip(ridge.predict(X_stack_val), 0, 1)
    stack_test_pred = np.clip(ridge.predict(X_stack_test), 0, 1)

    # Evaluate
    stack_val_rmse = np.sqrt(mean_squared_error(y_val, stack_val_pred))
    stack_val_mae = mean_absolute_error(y_val, stack_val_pred)
    stack_val_r2 = r2_score(y_val, stack_val_pred)

    print(f"ğŸ�† Ridge Stacking Performance:")
    print(f"Validation RMSE: {stack_val_rmse:.6f}")
    print(f"Validation MAE:  {stack_val_mae:.6f}")
    print(f"Validation RÂ²:   {stack_val_r2:.6f}")

    print(f"\nğŸ“ˆ Stacking vs Simple Ensemble:")
    print(f"Simple Ensemble RMSE: {ensemble_val_rmse:.6f}")
    print(f"Ridge Stacking RMSE:  {stack_val_rmse:.6f}")
    improvement = ensemble_val_rmse - stack_val_rmse
    improvement_pct = (improvement / ensemble_val_rmse) * 100
    print(f"Improvement: {improvement:.6f} ({improvement_pct:+.2f}%)")

    if stack_val_rmse < ensemble_val_rmse:
        print("âœ… Ridge stacking improves ensemble!")
    else:
        print("âš ï¸�  Simple ensemble performs better")
else:
    print("\nâ�­ï¸�  Skipping Ridge Stacking (only 1 model trained)")
    # Set stacking predictions equal to ensemble for downstream logic
    stack_val_pred = ensemble_val_pred
    stack_test_pred = ensemble_test_pred
    stack_val_rmse = ensemble_val_rmse



if trained_model_count >= 2:
    # Rank Averaging Ensemble (Kaggle Grandmaster Technique)
    print("\nğŸ�–ï¸� Creating Rank Averaging Ensemble...\n")

    # Convert predictions to ranks
    from scipy.stats import rankdata

    # Collect ranks for trained models
    val_ranks = []
    test_ranks = []
    weights_list = []

    if TRAIN_MODELS['xgboost']:
        val_ranks.append(rankdata(xgb_val_pred))
        test_ranks.append(rankdata(xgb_test_pred))
        weights_list.append(optimized_weights.get('w_xgb', 0))

    if TRAIN_MODELS['lightgbm']:
        val_ranks.append(rankdata(lgb_val_pred))
        test_ranks.append(rankdata(lgb_test_pred))
        weights_list.append(optimized_weights.get('w_lgb', 0))

    if TRAIN_MODELS['catboost']:
        val_ranks.append(rankdata(cb_val_pred))
        test_ranks.append(rankdata(cb_test_pred))
        weights_list.append(optimized_weights.get('w_cb', 0))

    if TRAIN_MODELS['randomforest']:
        val_ranks.append(rankdata(rf_val_pred))
        test_ranks.append(rankdata(rf_test_pred))
        weights_list.append(optimized_weights.get('w_rf', 0))

    # Average ranks with optimized weights
    rank_val_avg = np.zeros_like(y_val, dtype=float)
    rank_test_avg = np.zeros_like(X_test.index, dtype=float)

    for rank_val, rank_test, weight in zip(val_ranks, test_ranks, weights_list):
        rank_val_avg += weight * rank_val
        rank_test_avg += weight * rank_test

    # Convert averaged ranks back to predictions by matching target distribution
    # Use the ensemble predictions as the "template" distribution
    rank_val_pred_scaled = np.zeros_like(rank_val_avg)
    rank_test_pred_scaled = np.zeros_like(rank_test_avg)

    # Sort rank indices
    val_rank_order = np.argsort(rank_val_avg)
    test_rank_order = np.argsort(rank_test_avg)

    # Sort ensemble predictions
    val_pred_sorted = np.sort(ensemble_val_pred)
    test_pred_sorted = np.sort(ensemble_test_pred)

    # Map ranks back to prediction values using ensemble distribution
    rank_val_pred_scaled[val_rank_order] = val_pred_sorted
    rank_test_pred_scaled[test_rank_order] = test_pred_sorted

    # Ensure [0,1] range
    rank_val_pred_scaled = np.clip(rank_val_pred_scaled, 0, 1)
    rank_test_pred_scaled = np.clip(rank_test_pred_scaled, 0, 1)

    # Evaluate rank averaging
    rank_val_rmse = np.sqrt(mean_squared_error(y_val, rank_val_pred_scaled))
    rank_val_mae = mean_absolute_error(y_val, rank_val_pred_scaled)
    rank_val_r2 = r2_score(y_val, rank_val_pred_scaled)

    print(f"ğŸ�† Rank Averaging Performance:")
    print(f"Validation RMSE: {rank_val_rmse:.6f}")
    print(f"Validation MAE:  {rank_val_mae:.6f}")
    print(f"Validation RÂ²:   {rank_val_r2:.6f}")

    print(f"\nğŸ“ˆ Comparison:")
    print(f"Optimized Ensemble: {ensemble_val_rmse:.6f}")
    print(f"Ridge Stacking:     {stack_val_rmse:.6f}")
    print(f"Rank Averaging:     {rank_val_rmse:.6f}")

    # Find the best method
    best_methods = [
        ('Optimized Ensemble', ensemble_val_rmse, ensemble_test_pred, ensemble_val_pred),
        ('Ridge Stacking', stack_val_rmse, stack_test_pred, stack_val_pred),
        ('Rank Averaging', rank_val_rmse, rank_test_pred_scaled, rank_val_pred_scaled)
    ]

    best_method = min(best_methods, key=lambda x: x[1])
    print(f"\nğŸ�¯ Best Method: {best_method[0]} (RMSE: {best_method[1]:.6f})")

    final_test_pred = best_method[2]
    final_val_pred = best_method[3]
    final_val_rmse = best_method[1]

else:
    print("\nâ�­ï¸�  Skipping Rank Averaging (only 1 model trained)")
    # Set rank averaging predictions equal to ensemble for downstream logic
    rank_val_pred_scaled = ensemble_val_pred
    rank_test_pred_scaled = ensemble_test_pred
    rank_val_rmse = ensemble_val_rmse

    # For single model, set final predictions
    final_test_pred = ensemble_test_pred
    final_val_pred = ensemble_val_pred
    final_val_rmse = ensemble_val_rmse



if GENERATE_VISUALIZATIONS:
    # Visualization-only cell
    # Visualize predictions
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    predictions = {
        'XGBoost': xgb_val_pred,
        'LightGBM': lgb_val_pred,
        'CatBoost': cb_val_pred,
        'Ensemble': ensemble_val_pred
    }
    
    for idx, (name, preds) in enumerate(predictions.items()):
        axes[idx].scatter(y_val, preds, alpha=0.3, s=10)
        axes[idx].plot([y_val.min(), y_val.min()], [y_val.min(), y_val.max()], 
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
    
    # Hide the 6th subplot (we only have 3 models)
    axes[3].axis('off')
    
    plt.suptitle('Model Predictions vs True Values (3-Model Ensemble)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if GENERATE_VISUALIZATIONS:
    # Visualization-only cell
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
    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
    axes = axes.flatten()
    
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
    
    
    plt.suptitle('Feature Importance Comparison (All Models)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print top 10 features for ensemble


if GENERATE_VISUALIZATIONS:
    # Visualization-only cell
    # SHAP Dependency Plots - Feature Interactions
    print("ğŸ”— SHAP Dependency Analysis - Feature Interactions\n")
    
    # Plot dependency for top 4 features
    top_4_features = shap_importance.head(4)['feature'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_4_features):
        feature_idx = xgb_features.index(feature)
        
        plt.sca(axes[idx])
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_train_sample,
            feature_names=xgb_features,
            show=False,
            ax=axes[idx]
        )
        axes[idx].set_title(f'SHAP Dependency: {feature}', fontsize=12, fontweight='bold')
    
    plt.suptitle('SHAP Dependency Plots - Top 4 Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nâœ… Dependency plots show:")
    print("   - How feature values affect predictions (X-axis)")
    print("   - SHAP value impact (Y-axis)")
    print("   - Interactions with other features (color coding)")


print("ğŸš€ Generating Final Competition Submission...\n")

# Create submission directly (no post-processing - it overfits!)
submission = sample_submission.copy()
submission['accident_risk'] = final_test_pred

# Save to CSV
submission.to_csv('submission.csv', index=False)

print(f"âœ… Submission saved: submission.csv")
print(f"   Validation RMSE: {final_val_rmse:.6f}")
print(f"   Models: 3-model weighted ensemble (XGBoost + LightGBM + CatBoost)")

# Submission validation
print("\n" + "="*80)
print("ğŸ“Š Submission Validation")
print("="*80)
print(f"\nColumns: {list(submission.columns)}")
print(f"Shape: {submission.shape}")
print(f"Missing values: {submission['accident_risk'].isna().sum()}")
print(f"Min: {submission['accident_risk'].min():.6f}")
print(f"Max: {submission['accident_risk'].max():.6f}")
print(f"Mean: {submission['accident_risk'].mean():.6f}")
print(f"Std: {submission['accident_risk'].std():.6f}")

# Verify range
if submission['accident_risk'].min() >= 0 and submission['accident_risk'].max() <= 1:
    print("\nâœ… All predictions are in valid range [0, 1]")
else:
    print(f"\nâš ï¸�  WARNING: Predictions outside [0, 1] range!")

print("\nFirst 5 rows:")
print(submission.head())

print("\n" + "="*80)
print("ğŸ�‰ Submission ready for competition!")
print("="*80)


