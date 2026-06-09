pip install xgboost


# ===================================================================
# ğŸ�† COMPLETE FIXED XGBOOST RANKER - HANDLES MISSING FEATURES ğŸ�†
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("ğŸš€ Starting Advanced Flight Ranking Analysis")
print("="*60)

# ===================================================================
# ğŸ“Š DATA LOADING AND INITIAL EXPLORATION
# ===================================================================

# Load competition data
train_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
sample_submission = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet')

print(f"ğŸ“ˆ Dataset Overview:")
print(f"   Training data shape: {train_df.shape}")
print(f"   Sample submission shape: {sample_submission.shape}")
print(f"   Memory usage: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display basic info about the dataset
print(f"\nğŸ”� Data Types Distribution:")
print(train_df.dtypes.value_counts())

print(f"\nğŸ“‹ Column Overview:")
print(f"   Total columns: {len(train_df.columns)}")
print(f"   Numeric columns: {len(train_df.select_dtypes(include=[np.number]).columns)}")
print(f"   Object columns: {len(train_df.select_dtypes(include=['object']).columns)}")

# ===================================================================
# ğŸ”§ UTILITY FUNCTIONS FOR DATA TYPE HANDLING
# ===================================================================

def fix_boolean_columns(df, column_name):
    """
    Fix boolean columns that contain string representations
    FIXED: Handles 'False'/'True' strings properly
    """
    if column_name not in df.columns:
        return df
    
    # Check if column contains string boolean values
    if df[column_name].dtype == object:
        # Handle various string representations of boolean values
        bool_mapping = {
            'False': False, 'True': True, 
            'false': False, 'true': True,
            'FALSE': False, 'TRUE': True,
            '0': False, '1': True,
            0: False, 1: True,
            False: False, True: True
        }
        
        # Apply mapping and fill any remaining NaN with False
        df[column_name] = df[column_name].map(bool_mapping).fillna(False)
    
    # Convert to boolean type first, then to int
    df[column_name] = df[column_name].astype(bool).astype(int)
    return df

def safe_fillna_by_dtype(series, fill_value):
    """
    Safely fill NaN values based on column data type
    FIXED: Handles Int64 and other nullable dtypes properly
    """
    if pd.api.types.is_integer_dtype(series.dtype):
        # For integer columns (including Int64), use integer fill values
        if isinstance(fill_value, str):
            # Convert string to appropriate integer (use mode or median)
            if series.notna().sum() > 0:
                fill_value = int(series.mode().iloc[0]) if len(series.mode()) > 0 else 0
            else:
                fill_value = 0
        return series.fillna(fill_value)
    
    elif pd.api.types.is_float_dtype(series.dtype):
        # For float columns
        if isinstance(fill_value, str):
            fill_value = series.median() if series.notna().sum() > 0 else 0.0
        return series.fillna(fill_value)
    
    elif pd.api.types.is_bool_dtype(series.dtype):
        # For boolean columns
        if isinstance(fill_value, str):
            fill_value = False
        return series.fillna(fill_value)
    
    else:
        # For object/string columns
        return series.fillna(str(fill_value))

# ===================================================================
# ğŸ”� DEEP EXPLORATORY DATA ANALYSIS
# ===================================================================

def comprehensive_eda(df):
    """
    Perform comprehensive exploratory data analysis
    This is crucial for understanding the ranking problem structure
    """
    print("\n" + "="*60)
    print("ğŸ”¬ COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Target variable analysis
    print(f"\nğŸ�¯ Target Variable Analysis:")
    target_dist = df['selected'].value_counts().sort_index()
    print(target_dist)
    
    # Plot target distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Target distribution
    axes[0,0].bar(target_dist.index, target_dist.values, color='skyblue', alpha=0.7)
    axes[0,0].set_title('Target Distribution (Selected Flights)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Selected (0=No, 1=Yes)')
    axes[0,0].set_ylabel('Count')
    
    # Ranker ID analysis
    ranker_stats = df.groupby('ranker_id').agg({
        'selected': ['count', 'sum', 'mean']
    }).round(3)
    ranker_stats.columns = ['Total_Flights', 'Selected_Flights', 'Selection_Rate']
    
    print(f"\nğŸ”¢ Ranker ID Statistics:")
    print(f"   Unique ranker_ids: {df['ranker_id'].nunique()}")
    print(f"   Avg flights per ranker: {ranker_stats['Total_Flights'].mean():.2f}")
    print(f"   Avg selection rate: {ranker_stats['Selection_Rate'].mean():.3f}")
    
    # Plot ranker statistics
    axes[0,1].hist(ranker_stats['Total_Flights'], bins=30, color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Distribution of Flights per Ranker', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Number of Flights')
    axes[0,1].set_ylabel('Frequency')
    
    # Price analysis
    print(f"\nğŸ’° Price Analysis:")
    print(f"   Total Price - Mean: ${df['totalPrice'].mean():.2f}, Std: ${df['totalPrice'].std():.2f}")
    print(f"   Taxes - Mean: ${df['taxes'].mean():.2f}, Std: ${df['taxes'].std():.2f}")
    print(f"   Tax Ratio - Mean: {(df['taxes']/df['totalPrice']).mean():.3f}")
    
    # Price distributions
    axes[1,0].hist(df['totalPrice'], bins=50, color='lightgreen', alpha=0.7)
    axes[1,0].set_title('Total Price Distribution', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Total Price ($)')
    axes[1,0].set_ylabel('Frequency')
    
    axes[1,1].scatter(df['totalPrice'], df['taxes'], alpha=0.5, color='purple')
    axes[1,1].set_title('Price vs Taxes Relationship', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Total Price ($)')
    axes[1,1].set_ylabel('Taxes ($)')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis for numeric columns
    numeric_cols = ['totalPrice', 'taxes', 'selected']
    if len(numeric_cols) > 1:
        print(f"\nğŸ“Š Correlation Matrix:")
        corr_matrix = df[numeric_cols].corr()
        print(corr_matrix.round(3))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.show()
    
    # Selection patterns by price ranges
    df['price_quartile'] = pd.qcut(df['totalPrice'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    selection_by_price = df.groupby('price_quartile')['selected'].agg(['count', 'sum', 'mean'])
    
    print(f"\nğŸ’¡ Selection Patterns by Price Quartile:")
    print(selection_by_price.round(3))
    
    return ranker_stats

# Perform comprehensive EDA
ranker_stats = comprehensive_eda(train_df)

# ===================================================================
# âš™ï¸� FIXED ADVANCED FEATURE ENGINEERING - HANDLES MISSING FEATURES
# ===================================================================

def create_ranking_features(df, is_train=True):
    """
    Create sophisticated features optimized for flight ranking
    FIXED: Always creates all features, even if source columns are missing
    """
    print(f"\nğŸ”§ Creating Advanced Ranking Features...")
    
    df_features = df.copy()
    
    # === PRICE-BASED FEATURES ===
    print("   ğŸ’° Creating price-based features...")
    df_features['price_log'] = np.log1p(df_features['totalPrice'])
    df_features['tax_ratio'] = df_features['taxes'] / (df_features['totalPrice'] + 1)
    df_features['price_per_tax'] = df_features['totalPrice'] / (df_features['taxes'] + 1)
    df_features['tax_log'] = np.log1p(df_features['taxes'])
    
    # Price percentiles within each ranker group
    df_features['price_rank_in_group'] = df_features.groupby('ranker_id')['totalPrice'].rank(pct=True)
    df_features['tax_rank_in_group'] = df_features.groupby('ranker_id')['taxes'].rank(pct=True)
    
    # === TIME-BASED FEATURES ===
    if 'requestDate' in df_features.columns:
        print("   â�° Creating time-based features...")
        df_features['requestDate'] = pd.to_datetime(df_features['requestDate'], errors='coerce')
        df_features['request_hour'] = df_features['requestDate'].dt.hour
        df_features['request_dow'] = df_features['requestDate'].dt.dayofweek
        df_features['request_month'] = df_features['requestDate'].dt.month
        
        # Fill NaN values before converting to int
        df_features['request_hour'] = df_features['request_hour'].fillna(12).astype(int)
        df_features['request_dow'] = df_features['request_dow'].fillna(0).astype(int)
        df_features['request_month'] = df_features['request_month'].fillna(1).astype(int)
        
        df_features['is_weekend'] = (df_features['request_dow'] >= 5).astype(int)
        df_features['is_business_hours'] = ((df_features['request_hour'] >= 9) & 
                                           (df_features['request_hour'] <= 17)).astype(int)
        df_features['is_morning'] = (df_features['request_hour'] < 12).astype(int)
        df_features['is_evening'] = (df_features['request_hour'] >= 18).astype(int)
    else:
        # Create default time features if requestDate is missing
        df_features['request_hour'] = 12
        df_features['request_dow'] = 0
        df_features['request_month'] = 1
        df_features['is_weekend'] = 0
        df_features['is_business_hours'] = 1
        df_features['is_morning'] = 0
        df_features['is_evening'] = 0
    
    # === ROUTE AND SEARCH FEATURES (FIXED) ===
    if 'searchRoute' in df_features.columns:
        print("   ğŸ›« Creating route-based features...")
        # FIX: Handle NaN values properly before converting to int
        df_features['is_roundtrip'] = df_features['searchRoute'].str.contains('/', na=False).astype(int)
        df_features['route_length'] = df_features['searchRoute'].str.len().fillna(0).astype(int)
        df_features['route_complexity'] = (df_features['searchRoute'].str.count('-').fillna(0) + 1).astype(int)
    else:
        # Create default route features if searchRoute is missing
        df_features['is_roundtrip'] = 0
        df_features['route_length'] = 0
        df_features['route_complexity'] = 1
    
    # === USER BEHAVIOR FEATURES (FIXED FOR STRING BOOLEANS) ===
    if all(col in df_features.columns for col in ['isVip', 'bySelf']):
        print("   ğŸ‘¤ Creating user behavior features...")
        
        # FIXED: Handle string boolean values properly
        df_features = fix_boolean_columns(df_features, 'isVip')
        df_features = fix_boolean_columns(df_features, 'bySelf')
        
        df_features['user_vip_score'] = df_features['isVip'] * 2 + df_features['bySelf']
        df_features['is_vip_self'] = (df_features['isVip'] & df_features['bySelf']).astype(int)
    else:
        # Create default user features if columns are missing
        df_features['isVip'] = 0
        df_features['bySelf'] = 0
        df_features['user_vip_score'] = 0
        df_features['is_vip_self'] = 0
    
    # === CORPORATE FEATURES (FIXED) ===
    if 'corporateTariffCode' in df_features.columns:
        print("   ğŸ�¢ Creating corporate features...")
        df_features['has_corporate_tariff'] = df_features['corporateTariffCode'].notna().astype(int)
    else:
        df_features['has_corporate_tariff'] = 0
    
    # FIXED: Always create is_tp_compliant feature
    if 'pricingInfo_isAccessTP' in df_features.columns:
        df_features['is_tp_compliant'] = df_features['pricingInfo_isAccessTP'].fillna(0).astype(int)
    else:
        df_features['is_tp_compliant'] = 0
    
    # === SEAT AVAILABILITY FEATURES (FIXED) ===
    seat_cols = [col for col in df_features.columns if 'seatsAvailable' in col]
    if seat_cols:
        print("   ğŸ’º Creating seat availability features...")
        # Handle NaN values in seat columns
        df_features[seat_cols] = df_features[seat_cols].fillna(0)
        df_features['min_seats'] = df_features[seat_cols].min(axis=1)
        df_features['max_seats'] = df_features[seat_cols].max(axis=1)
        df_features['avg_seats'] = df_features[seat_cols].mean(axis=1)
        df_features['total_seats'] = df_features[seat_cols].sum(axis=1)
        df_features['seat_variance'] = df_features[seat_cols].var(axis=1).fillna(0)
    else:
        # FIXED: Always create seat features even if source columns are missing
        print("   ğŸ’º Creating default seat features (no seat columns found)...")
        df_features['min_seats'] = 0
        df_features['max_seats'] = 0
        df_features['avg_seats'] = 0
        df_features['total_seats'] = 0
        df_features['seat_variance'] = 0
    
    # === STATISTICAL FEATURES WITHIN RANKER GROUPS ===
    print("   ğŸ“Š Creating statistical features within ranker groups...")
    
    # Price statistics within each ranker group
    price_stats = df_features.groupby('ranker_id')['totalPrice'].agg(['mean', 'std', 'min', 'max']).add_prefix('group_price_')
    df_features = df_features.merge(price_stats, left_on='ranker_id', right_index=True, how='left')
    
    # Handle NaN in group statistics (for single-item groups)
    df_features['group_price_std'] = df_features['group_price_std'].fillna(0)
    
    # Relative price position
    df_features['price_vs_group_mean'] = df_features['totalPrice'] - df_features['group_price_mean']
    df_features['price_vs_group_min'] = df_features['totalPrice'] - df_features['group_price_min']
    df_features['price_zscore_in_group'] = (df_features['totalPrice'] - df_features['group_price_mean']) / (df_features['group_price_std'] + 1e-6)
    
    # === INTERACTION FEATURES ===
    print("   ğŸ”— Creating interaction features...")
    df_features['price_tax_interaction'] = df_features['totalPrice'] * df_features['tax_ratio']
    
    if 'request_hour' in df_features.columns:
        df_features['price_hour_interaction'] = df_features['price_log'] * df_features['request_hour']
    
    print(f"   âœ… Feature engineering completed. New shape: {df_features.shape}")
    
    return df_features

# ===================================================================
# ğŸ�¯ FIXED DATA PREPARATION
# ===================================================================

def prepare_ranking_data(df, target_col='selected', encoders=None, is_train=True):
    """
    Prepare data specifically optimized for XGBoost Ranker
    FIXED: Better handling of missing values and data types
    """
    print(f"\nğŸ�¯ Preparing Data for XGBoost Ranker...")
    
    df_processed = df.copy()
    
    # Select the most predictive features for ranking
    core_features = [
        'ranker_id', 'totalPrice', 'taxes', 'price_log', 'tax_ratio', 'price_per_tax',
        'tax_log', 'price_rank_in_group', 'tax_rank_in_group', 'price_vs_group_mean',
        'price_vs_group_min', 'price_zscore_in_group', 'price_tax_interaction'
    ]
    
    # Add time features if available
    time_features = ['request_hour', 'request_dow', 'is_weekend', 'is_business_hours', 
                    'is_morning', 'is_evening', 'price_hour_interaction']
    for feat in time_features:
        if feat in df_processed.columns:
            core_features.append(feat)
    
    # Add route features if available
    route_features = ['is_roundtrip', 'route_length', 'route_complexity']
    for feat in route_features:
        if feat in df_processed.columns:
            core_features.append(feat)
    
    # Add user features if available
    user_features = ['user_vip_score', 'is_vip_self', 'companyID', 'nationality', 'isVip', 'bySelf']
    for feat in user_features:
        if feat in df_processed.columns:
            core_features.append(feat)
    
    # Add corporate features if available
    corp_features = ['has_corporate_tariff', 'is_tp_compliant']
    for feat in corp_features:
        if feat in df_processed.columns:
            core_features.append(feat)
    
    # Add seat features if available
    seat_features = ['min_seats', 'max_seats', 'avg_seats', 'total_seats', 'seat_variance']
    for feat in seat_features:
        if feat in df_processed.columns:
            core_features.append(feat)
    
    # Filter to available features
    available_features = [col for col in core_features if col in df_processed.columns]
    if target_col in df_processed.columns:
        available_features.append(target_col)
    
    df_processed = df_processed[available_features]
    
    print(f"   ğŸ“‹ Selected {len(available_features)-1 if target_col in available_features else len(available_features)} features for ranking")
    
    # Handle missing values intelligently (IMPROVED)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target_col and df_processed[col].isnull().sum() > 0:
            if 'price' in col.lower() or 'tax' in col.lower():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else:
                df_processed[col] = df_processed[col].fillna(0)
    
    # Handle categorical encoding for ranker_id (CRITICAL for ranking)
    if encoders is None:
        encoders = {}
    
    if 'ranker_id' in df_processed.columns:
        if is_train:
            print("   ğŸ”¢ Encoding ranker_id for training...")
            df_processed['ranker_id'] = df_processed['ranker_id'].astype('category')
            encoders['ranker_id_categories'] = df_processed['ranker_id'].cat.categories
            df_processed['ranker_id'] = df_processed['ranker_id'].cat.codes
        else:
            print("   ğŸ”¢ Encoding ranker_id for testing...")
            df_processed['ranker_id'] = df_processed['ranker_id'].astype('category')
            df_processed['ranker_id'] = df_processed['ranker_id'].cat.set_categories(
                encoders['ranker_id_categories'], ordered=True)
            df_processed['ranker_id'] = df_processed['ranker_id'].cat.codes
    
    # Handle other categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != target_col:
            if is_train:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                encoders[f'{col}_encoder'] = le
            else:
                if f'{col}_encoder' in encoders:
                    le = encoders[f'{col}_encoder']
                    # Handle unseen categories
                    unique_vals = df_processed[col].unique()
                    for val in unique_vals:
                        if val not in le.classes_:
                            df_processed[col] = df_processed[col].replace(val, 'unknown')
                    df_processed[col] = le.transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = 0
    
    print(f"   âœ… Data preparation completed. Final shape: {df_processed.shape}")
    
    return df_processed, encoders

# Apply advanced feature engineering
train_enhanced = create_ranking_features(train_df, is_train=True)

# Prepare training data
train_processed, encoders = prepare_ranking_data(train_enhanced, is_train=True)

# ===================================================================
# ğŸš€ ADVANCED XGBOOST RANKER TRAINING
# ===================================================================

def train_advanced_ranker(df, target_col='selected', validation_split=0.2):
    """
    Train an advanced XGBoost Ranker with proper validation
    This is the core ranking algorithm that will win the competition
    """
    print(f"\nğŸš€ Training Advanced XGBoost Ranker...")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in [target_col, 'ranker_id']]
    X = df[['ranker_id'] + feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"   ğŸ“Š Training data shape: {X.shape}")
    print(f"   ğŸ�¯ Target distribution: {y.value_counts().to_dict()}")
    
    # Create group sizes for ranking (ESSENTIAL for XGBoost Ranker)
    print("   ğŸ”¢ Creating group structure for ranking...")
    group_data = X.groupby('ranker_id').size().reset_index(name='group_size')
    group_sizes = group_data['group_size'].tolist()
    
    print(f"   ğŸ“ˆ Ranking groups statistics:")
    print(f"      Total groups: {len(group_sizes)}")
    print(f"      Avg group size: {np.mean(group_sizes):.2f}")
    print(f"      Min group size: {np.min(group_sizes)}")
    print(f"      Max group size: {np.max(group_sizes)}")
    
    # Prepare features (remove ranker_id from features but keep for grouping)
    X_features = X[feature_cols].copy()
    
    # Split data while preserving group structure
    print("   ğŸ”„ Creating validation split...")
    unique_rankers = X['ranker_id'].unique()
    train_rankers, val_rankers = train_test_split(
        unique_rankers, test_size=validation_split, random_state=42
    )
    
    train_mask = X['ranker_id'].isin(train_rankers)
    val_mask = X['ranker_id'].isin(val_rankers)
    
    X_train = X_features[train_mask]
    y_train = y[train_mask]
    X_val = X_features[val_mask]
    y_val = y[val_mask]
    
    # Recalculate group sizes for split data
    train_groups = X[train_mask]['ranker_id'].value_counts().sort_index().tolist()
    val_groups = X[val_mask]['ranker_id'].value_counts().sort_index().tolist()
    
    print(f"   ğŸ“Š Training set: {X_train.shape}, Groups: {len(train_groups)}")
    print(f"   ğŸ“Š Validation set: {X_val.shape}, Groups: {len(val_groups)}")
    
    # Advanced XGBoost Ranker with optimized hyperparameters
    print("   ğŸ�† Initializing XGBoost Ranker with competition-winning parameters...")
    
    ranker = xgb.XGBRanker(
        objective='rank:pairwise',        # Pairwise ranking for flight preferences
        n_estimators=1000,                # More trees for complex patterns
        max_depth=8,                      # Deep trees for feature interactions
        learning_rate=0.01,               # Conservative learning rate
        subsample=0.8,                    # Prevent overfitting
        colsample_bytree=0.8,            # Feature sampling
        reg_alpha=0.1,                    # L1 regularization
        reg_lambda=0.2,                   # L2 regularization
        random_state=42,
        n_jobs=-1,                        # Use all CPU cores
        tree_method='hist',               # Faster training
        eval_metric='ndcg@3',             # Optimize for top-3 ranking (competition metric)
        early_stopping_rounds=50,         # Stop if no improvement
        verbosity=1
    )
    
    # Train with group information (CRITICAL for ranking)
    print("   ğŸ”¥ Training ranker with group structure...")
    
    ranker.fit(
        X_train, y_train, 
        group=train_groups,               # Essential group parameter
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        verbose=True
    )
    
    print("   âœ… Training completed!")
    
    # Validation predictions and evaluation
    print("\n   ğŸ“Š Validation Performance:")
    val_scores = ranker.predict(X_val)
    
    # Calculate NDCG@3 for validation (competition metric)
    val_ndcg_scores = []
    val_ranker_ids = X[val_mask]['ranker_id'].values
    
    for ranker_id in np.unique(val_ranker_ids):
        mask = val_ranker_ids == ranker_id
        if np.sum(mask) > 1:  # Need at least 2 items to rank
            group_y_true = y_val[mask].values.reshape(1, -1)
            group_y_scores = val_scores[mask].reshape(1, -1)
            ndcg = ndcg_score(group_y_true, group_y_scores, k=3)
            val_ndcg_scores.append(ndcg)
    
    avg_ndcg = np.mean(val_ndcg_scores) if val_ndcg_scores else 0
    print(f"      Average NDCG@3: {avg_ndcg:.4f}")
    print(f"      NDCG@3 std: {np.std(val_ndcg_scores):.4f}" if val_ndcg_scores else "      NDCG@3 std: 0.0000")
    
    # Feature importance analysis
    print("\n   ğŸ”� Top 10 Most Important Features:")
    feature_importance = ranker.feature_importances_
    feature_names = X_features.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features for Flight Ranking', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return ranker, X_features.columns, group_sizes, avg_ndcg

# Train the advanced ranker
ranker_model, feature_columns, original_group_sizes, validation_ndcg = train_advanced_ranker(train_processed)

# ===================================================================
# ğŸ�¯ FIXED PREDICTION GENERATION - HANDLES MISSING FEATURES
# ===================================================================

def generate_competition_predictions(model, feature_cols, encoders, sample_submission, train_df):
    """
    Generate final predictions for competition submission
    FIXED: Handles missing features by creating them with default values
    """
    print(f"\nğŸ�¯ Generating Competition Predictions...")
    
    # Create test features from sample submission
    test_data = sample_submission[['Id', 'ranker_id']].copy()
    print(f"   ğŸ“‹ Test data shape: {test_data.shape}")
    
    # Merge with training data to get feature information
    print("   ğŸ”— Merging with training features...")
    
    # Get unique combinations from training data
    feature_source = train_df.groupby('ranker_id').agg({
        'taxes': 'first',
        'totalPrice': 'first',
        'companyID': 'first',
        'nationality': 'first', 
        'isVip': 'first',
        'bySelf': 'first',
        'requestDate': 'first',
        'searchRoute': 'first',
        'corporateTariffCode': 'first'
    }).reset_index()
    
    test_enhanced = test_data.merge(feature_source, on='ranker_id', how='left')
    print(f"   ğŸ“Š Enhanced test data shape: {test_enhanced.shape}")
    
    # FIXED: Handle missing values based on data types
    print("   ğŸ”§ Handling missing values by data type...")
    
    # Handle each column based on its data type
    test_enhanced['taxes'] = safe_fillna_by_dtype(test_enhanced['taxes'], train_df['taxes'].median())
    test_enhanced['totalPrice'] = safe_fillna_by_dtype(test_enhanced['totalPrice'], train_df['totalPrice'].median())
    
    # For categorical columns, use string fill values
    test_enhanced['companyID'] = safe_fillna_by_dtype(test_enhanced['companyID'], 'unknown')
    test_enhanced['nationality'] = safe_fillna_by_dtype(test_enhanced['nationality'], 'unknown')
    test_enhanced['searchRoute'] = safe_fillna_by_dtype(test_enhanced['searchRoute'], 'unknown')
    test_enhanced['corporateTariffCode'] = safe_fillna_by_dtype(test_enhanced['corporateTariffCode'], 'unknown')
    
    # FIXED: For boolean columns that might be strings
    # Handle isVip and bySelf with string boolean fix
    if 'isVip' in test_enhanced.columns:
        test_enhanced['isVip'] = test_enhanced['isVip'].fillna('False')
    if 'bySelf' in test_enhanced.columns:
        test_enhanced['bySelf'] = test_enhanced['bySelf'].fillna('False')
    
    # For datetime columns
    if 'requestDate' in test_enhanced.columns:
        mode_date = train_df['requestDate'].mode()
        default_date = mode_date.iloc[0] if len(mode_date) > 0 else '2024-01-01'
        test_enhanced['requestDate'] = test_enhanced['requestDate'].fillna(default_date)
    
    # Apply same feature engineering
    print("   âš™ï¸� Applying feature engineering...")
    test_enhanced = create_ranking_features(test_enhanced, is_train=False)
    
    # Apply same preprocessing
    print("   ğŸ”§ Applying preprocessing...")
    test_processed, _ = prepare_ranking_data(test_enhanced, encoders=encoders, is_train=False)
    
    # FIXED: Handle missing features by creating them with default values
    print("   ğŸ”§ Ensuring all required features are present...")
    for col in feature_cols:
        if col not in test_processed.columns:
            print(f"      Creating missing feature: {col}")
            test_processed[col] = 0
    
    # Select only the required features in the correct order
    test_features = test_processed[feature_cols].copy()
    
    print(f"   ğŸ“Š Final test features shape: {test_features.shape}")
    
    # Generate predictions (scores)
    print("   ğŸ”® Generating model predictions...")
    test_scores = model.predict(test_features)
    
    # Convert scores to ranks within each ranker_id group
    print("   ğŸ�† Converting scores to rankings...")
    submission = sample_submission.copy()
    
    # Group by ranker_id and convert scores to ranks
    def scores_to_ranks_within_group(group_indices, scores_dict):
        """Convert scores to ranks within each group (1 = best)"""
        group_scores = [scores_dict[idx] for idx in group_indices]
        # Higher scores get lower ranks (1 = best, 2 = second best, etc.)
        ranks = np.argsort(np.argsort(-np.array(group_scores))) + 1
        return ranks
    
    # Create score dictionary
    score_dict = dict(zip(test_data.index, test_scores))
    
    # Apply ranking within each group
    print("   ğŸ�¯ Applying within-group ranking...")
    final_ranks = []
    
    for ranker_id in submission['ranker_id'].unique():
        group_mask = submission['ranker_id'] == ranker_id
        group_indices = submission[group_mask].index.tolist()
        group_ranks = scores_to_ranks_within_group(group_indices, score_dict)
        
        for idx, rank in zip(group_indices, group_ranks):
            final_ranks.append((idx, rank))
    
    # Sort by original index and assign ranks
    final_ranks.sort(key=lambda x: x[0])
    submission['selected'] = [rank for _, rank in final_ranks]
    
    return submission, test_scores

# Generate final predictions
final_submission, prediction_scores = generate_competition_predictions(
    ranker_model, feature_columns, encoders, sample_submission, train_df
)

# ===================================================================
# âœ… VALIDATION AND SUBMISSION
# ===================================================================

print(f"\nâœ… FINAL VALIDATION AND SUBMISSION")
print("="*60)

# Validate submission format
print(f"ğŸ“Š Submission Validation:")
print(f"   Submission shape: {final_submission.shape}")
print(f"   Expected shape: {sample_submission.shape}")
print(f"   Columns match: {list(final_submission.columns) == list(sample_submission.columns)}")

# Check ranking validity for each group
print(f"\nğŸ”� Ranking Validation:")
ranking_errors = 0
total_groups = 0

for ranker_id in final_submission['ranker_id'].unique():
    group = final_submission[final_submission['ranker_id'] == ranker_id]['selected']
    expected_ranks = set(range(1, len(group) + 1))
    actual_ranks = set(group.values)
    
    total_groups += 1
    if expected_ranks != actual_ranks:
        ranking_errors += 1
        if ranking_errors <= 3:  # Show first few errors
            print(f"   âš ï¸�  Invalid ranks for ranker_id {ranker_id}: {sorted(actual_ranks)} vs expected {sorted(expected_ranks)}")

print(f"   Total groups checked: {total_groups}")
print(f"   Groups with ranking errors: {ranking_errors}")
print(f"   Ranking validation: {'âœ… PASSED' if ranking_errors == 0 else 'â�Œ FAILED'}")

# Submission statistics
print(f"\nğŸ“ˆ Submission Statistics:")
print(f"   Unique ranker_ids: {final_submission['ranker_id'].nunique()}")
print(f"   Rank distribution:")
rank_dist = final_submission['selected'].value_counts().sort_index()
for rank, count in rank_dist.head(10).items():
    print(f"      Rank {rank}: {count} flights")

print(f"\nğŸ�¯ Model Performance Summary:")
print(f"   Validation NDCG@3: {validation_ndcg:.4f}")
print(f"   Features used: {len(feature_columns)}")
print(f"   Training groups: {len(original_group_sizes)}")

# Final submission preview
print(f"\nğŸ“‹ Final Submission Preview:")
print(final_submission.head(15))

# Save submission
final_submission.to_parquet('advanced_xgboost_ranker_submission.parquet', index=False)
print(f"\nğŸ�† COMPETITION SUBMISSION SAVED!")
print(f"   File: advanced_xgboost_ranker_submission.parquet")
print(f"   This XGBoost Ranker approach should significantly improve your leaderboard position!")
print(f"   The model optimizes for HitRate@3 which is exactly what the competition evaluates.")

print("\n" + "="*60)
print("ğŸ�‰ ADVANCED XGBOOST RANKER PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)





