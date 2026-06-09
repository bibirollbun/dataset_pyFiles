import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import LGBMRegressor
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
import random
import os
warnings.filterwarnings('ignore')

print("Starting ULTRA-ENHANCED House Price Prediction with Advanced Quantile Regression...")
print("=" * 80)

# ============================================================
# 1. REPRODUCIBILITY & CONFIGURATION
# ============================================================
def seed_everything(seed=2025):
    """Enhanced seed setting for full reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(2025)

# Competition configuration
NFOLDS = 4  # Increased folds for better stability
COVERAGE = 0.9  # 90% prediction intervals
ALPHA = 1 - COVERAGE
SEED = 2025

# Load data
print("Loading datasets...")
train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
sample_submission = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv')

print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")

# ============================================================
# 2. ADVANCED FEATURE ENGINEERING (Enhanced from original)
# ============================================================
def ultra_advanced_feature_engineering(df, is_train=True):
    """
    Ultra-advanced feature engineering combining domain expertise with automated feature generation
    """
    print(f"Processing {'train' if is_train else 'test'} data with ULTRA-ADVANCED feature engineering...")
    
    df_clean = df.copy()
    
    # =================================================================
    # ENHANCED BASIC CLEANING
    # =================================================================
    print("  Step 1: Enhanced data cleaning with intelligent imputation...")
    
    # Smart missing value handling
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            if col in ['year_built', 'year_reno']:
                # Use mode for year columns if highly skewed
                if df_clean[col].skew() > 2:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif 'sqft' in col or 'area' in col:
                # Use median for area-related features
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                # Use mean for other numeric features
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # Enhanced categorical handling
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna('Missing', inplace=True)
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    # =================================================================
    # ADVANCED TEMPORAL FEATURES
    # =================================================================
    print("  Step 2: Advanced temporal and cyclical features...")
    
    if 'sale_date' in df_clean.columns:
        df_clean['sale_date'] = pd.to_datetime(df_clean['sale_date'])
        df_clean['sale_year'] = df_clean['sale_date'].dt.year
        df_clean['sale_month'] = df_clean['sale_date'].dt.month
        df_clean['sale_quarter'] = df_clean['sale_date'].dt.quarter
        df_clean['sale_day_of_year'] = df_clean['sale_date'].dt.dayofyear
        df_clean['sale_weekday'] = df_clean['sale_date'].dt.weekday
        df_clean['sale_week'] = df_clean['sale_date'].dt.isocalendar().week
        
        # Advanced seasonal encoding (multiple harmonics)
        for period, name in [(12, 'month'), (4, 'quarter'), (7, 'weekday')]:
            if f'sale_{name}' in df_clean.columns:
                df_clean[f'{name}_sin1'] = np.sin(2 * np.pi * df_clean[f'sale_{name}'] / period)
                df_clean[f'{name}_cos1'] = np.cos(2 * np.pi * df_clean[f'sale_{name}'] / period)
                df_clean[f'{name}_sin2'] = np.sin(4 * np.pi * df_clean[f'sale_{name}'] / period)
                df_clean[f'{name}_cos2'] = np.cos(4 * np.pi * df_clean[f'sale_{name}'] / period)
        
        # Market timing indicators
        df_clean['is_spring_peak'] = ((df_clean['sale_month'] >= 4) & (df_clean['sale_month'] <= 6)).astype(int)
        df_clean['is_summer_peak'] = ((df_clean['sale_month'] >= 7) & (df_clean['sale_month'] <= 8)).astype(int)
        df_clean['is_holiday_season'] = ((df_clean['sale_month'] == 12) | (df_clean['sale_month'] == 1)).astype(int)
        df_clean['is_weekend'] = (df_clean['sale_weekday'] >= 5).astype(int)
        
        # Time since epoch features
        epoch = pd.Timestamp('2000-01-01')
        df_clean['days_since_2000'] = (df_clean['sale_date'] - epoch).dt.days
        df_clean['years_since_2000'] = df_clean['days_since_2000'] / 365.25
    
    # =================================================================
    # ULTRA-ADVANCED PROPERTY CHARACTERISTICS
    # =================================================================
    print("  Step 3: Ultra-advanced property age and renovation features...")
    
    if 'year_built' in df_clean.columns:
        df_clean['property_age'] = df_clean['sale_year'] - df_clean['year_built']
        df_clean['property_age'] = np.maximum(df_clean['property_age'], 0)
        
        # Non-linear age transformations
        df_clean['log_property_age'] = np.log1p(df_clean['property_age'])
        df_clean['sqrt_property_age'] = np.sqrt(df_clean['property_age'])
        df_clean['age_squared'] = df_clean['property_age'] ** 2
        df_clean['age_cubed'] = df_clean['property_age'] ** 3
        
        # Age-based depreciation curve
        df_clean['depreciation_factor'] = np.exp(-df_clean['property_age'] / 50)  # 50-year half-life
        df_clean['age_penalty'] = np.maximum(0, df_clean['property_age'] - 20) ** 1.5
        
        # Detailed age categories with smooth transitions
        age_bins = [0, 5, 10, 20, 30, 50, 75, 200]
        df_clean['age_category'] = pd.cut(df_clean['property_age'], bins=age_bins, labels=False)
        
        # Architectural era with market preferences
        df_clean['is_new_construction'] = (df_clean['property_age'] <= 2).astype(int)
        df_clean['is_modern'] = ((df_clean['year_built'] >= 2000) & (df_clean['year_built'] < 2010)).astype(int)
        df_clean['is_contemporary'] = ((df_clean['year_built'] >= 1990) & (df_clean['year_built'] < 2000)).astype(int)
        df_clean['is_eighties'] = ((df_clean['year_built'] >= 1980) & (df_clean['year_built'] < 1990)).astype(int)
        df_clean['is_vintage'] = ((df_clean['year_built'] >= 1950) & (df_clean['year_built'] < 1980)).astype(int)
        df_clean['is_mid_century'] = ((df_clean['year_built'] >= 1940) & (df_clean['year_built'] < 1970)).astype(int)
        df_clean['is_prewar'] = (df_clean['year_built'] < 1940).astype(int)
    
    # Enhanced renovation features
    if 'year_reno' in df_clean.columns:
        df_clean['has_renovation'] = (df_clean['year_reno'] > 0).astype(int)
        df_clean['years_since_reno'] = np.where(df_clean['year_reno'] > 0, 
                                               df_clean['sale_year'] - df_clean['year_reno'], 
                                               df_clean['property_age'])
        
        # Renovation effectiveness decay
        df_clean['reno_effectiveness'] = np.where(df_clean['year_reno'] > 0,
                                                 np.exp(-df_clean['years_since_reno'] / 15),  # 15-year decay
                                                 0)
        
        # Multiple renovation indicators
        df_clean['recent_reno'] = ((df_clean['year_reno'] > 0) & (df_clean['years_since_reno'] <= 5)).astype(int)
        df_clean['mid_reno'] = ((df_clean['year_reno'] > 0) & (df_clean['years_since_reno'] > 5) & (df_clean['years_since_reno'] <= 15)).astype(int)
        df_clean['old_reno'] = ((df_clean['year_reno'] > 0) & (df_clean['years_since_reno'] > 15)).astype(int)
        
        # Renovation timing relative to sale
        df_clean['reno_sale_timing'] = df_clean['years_since_reno'] / (df_clean['property_age'] + 1)
    
    # =================================================================
    # ULTRA-ADVANCED SIZE AND SPACE FEATURES
    # =================================================================
    print("  Step 4: Ultra-advanced size, space, and efficiency features...")
    
    # Enhanced size features with multiple transformations
    for col in ['sqft', 'sqft_lot', 'sqft_1', 'sqft_fbsmt']:
        if col in df_clean.columns:
            df_clean[f'log_{col}'] = np.log1p(df_clean[col])
            df_clean[f'sqrt_{col}'] = np.sqrt(df_clean[col])
            df_clean[f'{col}_per_1000'] = df_clean[col] / 1000
            
            # Quantile-based features
            df_clean[f'{col}_rank'] = df_clean[col].rank(pct=True)
            df_clean[f'{col}_zscore'] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    
    # Advanced space efficiency metrics
    if all(col in df_clean.columns for col in ['sqft', 'sqft_lot']):
        df_clean['lot_utilization'] = df_clean['sqft'] / (df_clean['sqft_lot'] + 1)
        df_clean['lot_efficiency'] = np.minimum(df_clean['lot_utilization'], 0.5)  # Cap at 50%
        df_clean['lot_waste'] = np.maximum(0, df_clean['sqft_lot'] - df_clean['sqft'] * 4)  # Excess lot space
        
        # Size harmony index
        ideal_ratio = 0.25  # Ideal house-to-lot ratio
        df_clean['size_harmony'] = 1 / (1 + abs(df_clean['lot_utilization'] - ideal_ratio))
    
    # Multi-level space analysis
    if all(col in df_clean.columns for col in ['sqft', 'sqft_1', 'sqft_fbsmt']):
        df_clean['main_floor_dominance'] = df_clean['sqft_1'] / (df_clean['sqft'] + 1)
        df_clean['basement_contribution'] = df_clean['sqft_fbsmt'] / (df_clean['sqft'] + 1)
        df_clean['space_distribution_index'] = abs(df_clean['main_floor_dominance'] - 0.7)  # Ideal ~70% main floor
        
        # Vertical efficiency
        if 'stories' in df_clean.columns:
            df_clean['vertical_efficiency'] = df_clean['sqft'] / (df_clean['stories'] * df_clean['sqft_lot'] + 1)
    
    # =================================================================
    # ADVANCED VALUE AND ASSESSMENT FEATURES
    # =================================================================
    print("  Step 5: Advanced value, assessment, and market features...")
    
    if all(col in df_clean.columns for col in ['land_val', 'imp_val']):
        df_clean['total_assessed_val'] = df_clean['land_val'] + df_clean['imp_val']
        df_clean['log_total_assessed_val'] = np.log1p(df_clean['total_assessed_val'])
        
        # Advanced value ratios with non-linear transforms
        df_clean['land_dominance'] = df_clean['land_val'] / (df_clean['total_assessed_val'] + 1)
        df_clean['improvement_dominance'] = df_clean['imp_val'] / (df_clean['total_assessed_val'] + 1)
        df_clean['value_imbalance'] = abs(df_clean['land_dominance'] - 0.3)  # Ideal ~30% land
        
        # Value efficiency metrics
        if 'sqft' in df_clean.columns:
            df_clean['assessed_val_per_sqft'] = df_clean['total_assessed_val'] / (df_clean['sqft'] + 1)
            df_clean['land_val_per_sqft'] = df_clean['land_val'] / (df_clean['sqft'] + 1)
            df_clean['imp_val_per_sqft'] = df_clean['imp_val'] / (df_clean['sqft'] + 1)
            
            # Value density categories
            df_clean['value_density_tier'] = pd.qcut(df_clean['assessed_val_per_sqft'], 
                                                   q=5, labels=False, duplicates='drop')
        
        # Market position indicators
        df_clean['total_val_rank'] = df_clean['total_assessed_val'].rank(pct=True)
        df_clean['is_luxury_assessment'] = (df_clean['total_val_rank'] > 0.9).astype(int)
        df_clean['is_premium_assessment'] = (df_clean['total_val_rank'] > 0.75).astype(int)
        df_clean['is_budget_assessment'] = (df_clean['total_val_rank'] < 0.25).astype(int)
    
    # =================================================================
    # ULTRA-ADVANCED QUALITY AND CONDITION FEATURES
    # =================================================================
    print("  Step 6: Ultra-advanced quality, condition, and luxury features...")
    
    # Enhanced grade features
    if 'grade' in df_clean.columns:
        df_clean['grade_normalized'] = (df_clean['grade'] - df_clean['grade'].min()) / (df_clean['grade'].max() - df_clean['grade'].min())
        df_clean['grade_squared'] = df_clean['grade'] ** 2
        df_clean['grade_cubed'] = df_clean['grade'] ** 3
        
        # Grade tiers with luxury indicators
        df_clean['is_luxury_grade'] = (df_clean['grade'] >= 11).astype(int)
        df_clean['is_premium_grade'] = ((df_clean['grade'] >= 9) & (df_clean['grade'] < 11)).astype(int)
        df_clean['is_standard_grade'] = ((df_clean['grade'] >= 7) & (df_clean['grade'] < 9)).astype(int)
        df_clean['is_basic_grade'] = (df_clean['grade'] < 7).astype(int)
        
        # Grade deviation from neighborhood mean (if we had neighborhood data)
        df_clean['grade_rank'] = df_clean['grade'].rank(pct=True)
    
    # Enhanced condition features
    if 'condition' in df_clean.columns:
        df_clean['condition_normalized'] = (df_clean['condition'] - df_clean['condition'].min()) / (df_clean['condition'].max() - df_clean['condition'].min())
        df_clean['condition_squared'] = df_clean['condition'] ** 2
        
        # Condition-based market appeal
        df_clean['market_ready'] = (df_clean['condition'] >= 4).astype(int)
        df_clean['needs_work'] = (df_clean['condition'] <= 2).astype(int)
        df_clean['condition_rank'] = df_clean['condition'].rank(pct=True)
    
    # Quality-condition synergy
    if all(col in df_clean.columns for col in ['grade', 'condition']):
        df_clean['quality_condition_product'] = df_clean['grade'] * df_clean['condition']
        df_clean['quality_condition_harmony'] = np.minimum(df_clean['grade'], df_clean['condition'] * 3)  # Condition caps grade effect
        df_clean['quality_index'] = (df_clean['grade'] + df_clean['condition'] * 2) / 3  # Weight condition more
        
        # Quality-condition mismatch
        grade_norm = df_clean['grade'] / df_clean['grade'].max()
        condition_norm = df_clean['condition'] / df_clean['condition'].max()
        df_clean['quality_mismatch'] = abs(grade_norm - condition_norm)
    
    # =================================================================
    # ADVANCED LOCATION AND GEOGRAPHY FEATURES
    # =================================================================
    print("  Step 7: Advanced geographic and location features...")
    
    if all(col in df_clean.columns for col in ['latitude', 'longitude']):
        # Multiple city centers for distance calculations
        centers = {
            'seattle': (47.6062, -122.3321),
            'bellevue': (47.6101, -122.2015),
            'tacoma': (47.2529, -122.4443),
            'everett': (47.9790, -122.2021)
        }
        
        for city, (lat, lon) in centers.items():
            df_clean[f'dist_to_{city}'] = np.sqrt((df_clean['latitude'] - lat)**2 + (df_clean['longitude'] - lon)**2)
            df_clean[f'dist_to_{city}_km'] = df_clean[f'dist_to_{city}'] * 111  # Rough km conversion
        
        # Minimum distance to any major city
        dist_cols = [f'dist_to_{city}' for city in centers.keys()]
        df_clean['min_city_distance'] = df_clean[dist_cols].min(axis=1)
        df_clean['closest_city'] = df_clean[dist_cols].idxmin(axis=1).str.replace('dist_to_', '')
        
        # Geographic clusters using coordinates
        df_clean['coord_cluster_x'] = (df_clean['longitude'] * 100).astype(int)
        df_clean['coord_cluster_y'] = (df_clean['latitude'] * 100).astype(int)
        df_clean['coord_cluster'] = df_clean['coord_cluster_x'].astype(str) + '_' + df_clean['coord_cluster_y'].astype(str)
        
        # Advanced coordinate transformations
        df_clean['lat_lon_product'] = df_clean['latitude'] * df_clean['longitude']
        df_clean['lat_lon_sum'] = df_clean['latitude'] + df_clean['longitude']
        df_clean['lat_lon_diff'] = df_clean['latitude'] - df_clean['longitude']
        df_clean['coord_magnitude'] = np.sqrt(df_clean['latitude']**2 + df_clean['longitude']**2)
        
        # Directional indicators
        seattle_lat, seattle_lon = centers['seattle']
        df_clean['north_of_seattle'] = (df_clean['latitude'] > seattle_lat).astype(int)
        df_clean['south_of_seattle'] = (df_clean['latitude'] < seattle_lat).astype(int)
        df_clean['east_of_seattle'] = (df_clean['longitude'] > seattle_lon).astype(int)
        df_clean['west_of_seattle'] = (df_clean['longitude'] < seattle_lon).astype(int)
        
        # Quadrant features
        df_clean['quadrant_ne'] = ((df_clean['latitude'] > seattle_lat) & (df_clean['longitude'] > seattle_lon)).astype(int)
        df_clean['quadrant_nw'] = ((df_clean['latitude'] > seattle_lat) & (df_clean['longitude'] < seattle_lon)).astype(int)
        df_clean['quadrant_se'] = ((df_clean['latitude'] < seattle_lat) & (df_clean['longitude'] > seattle_lon)).astype(int)
        df_clean['quadrant_sw'] = ((df_clean['latitude'] < seattle_lat) & (df_clean['longitude'] < seattle_lon)).astype(int)
    
    # =================================================================
    # ULTRA-ADVANCED INTERACTION FEATURES
    # =================================================================
    print("  Step 8: Ultra-advanced interaction and polynomial features...")
    
    # Size × Quality interactions (multiple levels)
    if all(col in df_clean.columns for col in ['sqft', 'grade']):
        df_clean['sqft_grade_interaction'] = df_clean['sqft'] * df_clean['grade']
        df_clean['sqft_grade_squared'] = df_clean['sqft'] * (df_clean['grade'] ** 2)
        df_clean['log_sqft_grade'] = df_clean['log_sqft'] * df_clean['grade']
    
    # Age × Quality interactions
    if all(col in df_clean.columns for col in ['property_age', 'grade', 'condition']):
        df_clean['age_grade_interaction'] = df_clean['property_age'] * df_clean['grade']
        df_clean['age_condition_interaction'] = df_clean['property_age'] * df_clean['condition']
        df_clean['age_quality_synergy'] = df_clean['property_age'] * df_clean['quality_index']
        
        # Age penalty adjusted by quality
        df_clean['quality_adjusted_age'] = df_clean['property_age'] / (df_clean['grade'] / 10 + 0.1)
    
    # Value × Size interactions
    if all(col in df_clean.columns for col in ['total_assessed_val', 'sqft']):
        df_clean['value_size_efficiency'] = df_clean['total_assessed_val'] / (df_clean['sqft'] ** 0.8)
        df_clean['value_size_interaction'] = df_clean['log_total_assessed_val'] * df_clean['log_sqft']
    
    # Location × Property interactions
    if all(col in df_clean.columns for col in ['min_city_distance', 'sqft', 'grade']):
        df_clean['location_size_premium'] = df_clean['sqft'] / (df_clean['min_city_distance'] + 0.01)
        df_clean['location_quality_premium'] = df_clean['grade'] / (df_clean['min_city_distance'] + 0.01)
    
    # =================================================================
    # MARKET AND ECONOMIC INDICATORS
    # =================================================================
    print("  Step 9: Advanced market and economic features...")
    
    # Enhanced sale characteristics
    if 'sale_nbr' in df_clean.columns:
        df_clean['sale_nbr_filled'] = df_clean['sale_nbr'].fillna(1)
        df_clean['is_bulk_sale'] = (df_clean['sale_nbr_filled'] > 1).astype(int)
        df_clean['bulk_sale_discount'] = np.where(df_clean['sale_nbr_filled'] > 1, 
                                                 1 - (1 / df_clean['sale_nbr_filled']), 0)
    
    # Market timing and liquidity
    if 'join_status' in df_clean.columns:
        status_map = {'new': 1, 'nochg': 0, 'chg': 2, 'del': -1}
        df_clean['join_status_numeric'] = df_clean['join_status'].map(status_map).fillna(0)
        df_clean['market_disruption'] = abs(df_clean['join_status_numeric'])
    
    # =================================================================
    # POLYNOMIAL AND ADVANCED MATHEMATICAL FEATURES
    # =================================================================
    print("  Step 10: Polynomial and advanced mathematical transformations...")
    
    # Key numerical features for polynomial expansion
    key_features = ['sqft', 'total_assessed_val', 'property_age', 'grade', 'condition']
    available_key_features = [f for f in key_features if f in df_clean.columns]
    
    # Create polynomial features (degree 2 and 3)
    for feature in available_key_features:
        if df_clean[feature].dtype in ['int64', 'float64']:
            # Normalize first to prevent overflow
            feat_normalized = (df_clean[feature] - df_clean[feature].min()) / (df_clean[feature].max() - df_clean[feature].min() + 1e-8)
            df_clean[f'{feature}_poly2'] = feat_normalized ** 2
            df_clean[f'{feature}_poly3'] = feat_normalized ** 3
            df_clean[f'{feature}_sqrt'] = np.sqrt(feat_normalized + 1e-8)
            df_clean[f'{feature}_log'] = np.log1p(feat_normalized)
    
    # Cross-feature polynomials (selected pairs)
    important_pairs = [
        ('sqft', 'grade'), ('sqft', 'total_assessed_val'), 
        ('grade', 'condition'), ('property_age', 'grade')
    ]
    
    for feat1, feat2 in important_pairs:
        if all(f in df_clean.columns for f in [feat1, feat2]):
            # Normalize features
            f1_norm = (df_clean[feat1] - df_clean[feat1].min()) / (df_clean[feat1].max() - df_clean[feat1].min() + 1e-8)
            f2_norm = (df_clean[feat2] - df_clean[feat2].min()) / (df_clean[feat2].max() - df_clean[feat2].min() + 1e-8)
            
            df_clean[f'{feat1}_{feat2}_poly'] = f1_norm * f2_norm
            df_clean[f'{feat1}_{feat2}_poly2'] = (f1_norm * f2_norm) ** 2
    
    print(f"  Ultra-advanced feature engineering completed! Shape: {df_clean.shape}")
    return df_clean

# ============================================================
# 3. ADVANCED TEXT PROCESSING
# ============================================================
def process_text_features(train_df, test_df, text_cols):
    """
    Advanced text feature processing with multiple TF-IDF strategies
    """
    print("Processing text features with advanced TF-IDF...")
    
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    for col in text_cols:
        if col in train_df.columns:
            print(f"  Processing text column: {col}")
            
            # Character-level TF-IDF (like first code)
            tfidf_char = TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 5),  # Slightly extended range
                max_features=30,     # Increased features
                min_df=3,
                sublinear_tf=True
            )
            
            # Word-level TF-IDF
            tfidf_word = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                max_features=20,
                min_df=5,
                stop_words='english',
                sublinear_tf=True
            )
            
            # Combine train and test text for fitting
            combined_text = pd.concat([
                train_df[col].fillna('').astype(str),
                test_df[col].fillna('').astype(str)
            ])
            
            # Fit and transform character-level
            tfidf_char.fit(combined_text)
            
            char_train = tfidf_char.transform(train_df[col].fillna('').astype(str)).toarray()
            char_test = tfidf_char.transform(test_df[col].fillna('').astype(str)).toarray()
            
            char_cols = [f'{col}_char_tfidf_{i}' for i in range(char_train.shape[1])]
            
            # Add character features
            for i, col_name in enumerate(char_cols):
                train_processed[col_name] = char_train[:, i]
                test_processed[col_name] = char_test[:, i]
            
            # Fit and transform word-level
            tfidf_word.fit(combined_text)
            
            word_train = tfidf_word.transform(train_df[col].fillna('').astype(str)).toarray()
            word_test = tfidf_word.transform(test_df[col].fillna('').astype(str)).toarray()
            
            word_cols = [f'{col}_word_tfidf_{i}' for i in range(word_train.shape[1])]
            
            # Add word features
            for i, col_name in enumerate(word_cols):
                train_processed[col_name] = word_train[:, i]
                test_processed[col_name] = word_test[:, i]
            
            # Text length and complexity features
            train_processed[f'{col}_length'] = train_df[col].fillna('').astype(str).str.len()
            test_processed[f'{col}_length'] = test_df[col].fillna('').astype(str).str.len()
            
            train_processed[f'{col}_word_count'] = train_df[col].fillna('').astype(str).str.split().str.len()
            test_processed[f'{col}_word_count'] = test_df[col].fillna('').astype(str).str.split().str.len()
            
            # Remove original text column
            train_processed.drop(columns=[col], inplace=True)
            test_processed.drop(columns=[col], inplace=True)
    
    return train_processed, test_processed

# ============================================================
# 4. ADVANCED CATEGORICAL ENCODING
# ============================================================
def advanced_categorical_encoding(train_df, test_df, target_col='sale_price', high_cardinality_threshold=20):
    """
    Advanced categorical encoding combining multiple strategies
    """
    print("Advanced categorical encoding with multiple strategies...")
    
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    categorical_cols = train_encoded.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['sale_date']]
    
    encoders = {}
    
    for col in categorical_cols:
        print(f"  Encoding {col}...")
        
        # Get unique values count
        combined = pd.concat([train_encoded[col], test_encoded[col]]).astype(str)
        unique_vals = combined.nunique()
        
        if unique_vals <= high_cardinality_threshold:
            # Low cardinality: Use CatBoost target encoding
            if target_col in train_encoded.columns:
                encoder = CatBoostEncoder(
                    cols=[col],
                    random_state=SEED,
                    sigma=1.0,  # Less smoothing for low cardinality
                    a=1.0
                )
                encoder.fit(train_encoded[[col]], train_encoded[target_col])
                
                train_encoded[f'{col}_target_enc'] = encoder.transform(train_encoded[[col]])[col]
                test_encoded[f'{col}_target_enc'] = encoder.transform(test_encoded[[col]])[col]
                
                encoders[f'{col}_target'] = encoder
            
            # Also add frequency encoding
            freq_map = combined.value_counts().to_dict()
            train_encoded[f'{col}_freq'] = train_encoded[col].map(freq_map).fillna(0)
            test_encoded[f'{col}_freq'] = test_encoded[col].map(freq_map).fillna(0)
            
            # Label encoding for tree models
            le = LabelEncoder()
            le.fit(combined)
            train_encoded[f'{col}_label'] = le.transform(train_encoded[col].astype(str))
            test_encoded[f'{col}_label'] = le.transform(test_encoded[col].astype(str))
            encoders[f'{col}_label'] = le
            
        else:
            # High cardinality: Multiple encoding strategies
            
            # 1. Frequency encoding
            freq_map = combined.value_counts().to_dict()
            train_encoded[f'{col}_freq'] = train_encoded[col].map(freq_map).fillna(0)
            test_encoded[f'{col}_freq'] = test_encoded[col].map(freq_map).fillna(0)
            
            # 2. Top categories + other
            top_categories = combined.value_counts().head(high_cardinality_threshold).index
            train_encoded[f'{col}_top'] = train_encoded[col].apply(
                lambda x: x if x in top_categories else 'other'
            )
            test_encoded[f'{col}_top'] = test_encoded[col].apply(
                lambda x: x if x in top_categories else 'other'
            )
            
            # Target encode the top categories
            if target_col in train_encoded.columns:
                encoder = CatBoostEncoder(
                    cols=[f'{col}_top'],
                    random_state=SEED,
                    sigma=2.0,  # More smoothing for high cardinality
                    a=10.0
                )
                encoder.fit(train_encoded[[f'{col}_top']], train_encoded[target_col])
                
                train_encoded[f'{col}_target_enc'] = encoder.transform(train_encoded[[f'{col}_top']])[f'{col}_top']
                test_encoded[f'{col}_target_enc'] = encoder.transform(test_encoded[[f'{col}_top']])[f'{col}_top']
                
                encoders[f'{col}_target'] = encoder
            
            # Label encode top categories
            le = LabelEncoder()
            combined_top = pd.concat([train_encoded[f'{col}_top'], test_encoded[f'{col}_top']])
            le.fit(combined_top)
            train_encoded[f'{col}_label'] = le.transform(train_encoded[f'{col}_top'])
            test_encoded[f'{col}_label'] = le.transform(test_encoded[f'{col}_top'])
            encoders[f'{col}_label'] = le
            
            # Remove the temporary _top column
            train_encoded.drop(columns=[f'{col}_top'], inplace=True)
            test_encoded.drop(columns=[f'{col}_top'], inplace=True)
        
        # Remove original categorical column
        train_encoded.drop(columns=[col], inplace=True)
        test_encoded.drop(columns=[col], inplace=True)
    
    return train_encoded, test_encoded, encoders

# ============================================================
# 5. ENHANCED MODEL CONFIGURATION
# ============================================================

# Ultra-optimized LightGBM parameters for quantile regression
base_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',  # Will be overridden for quantile models
    'metric': 'rmse',
    'verbosity': -1,
    'seed': SEED,
    'deterministic': True,
    
    # Tree structure
    'num_leaves': 150,          # Increased complexity
    'max_depth': -1,
    'min_child_samples': 25,    # Reduced for more flexibility
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    
    # Regularization
    'lambda_l1': 1.0,
    'lambda_l2': 5.0,
    'feature_fraction': 0.7,
    'feature_fraction_bynode': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_seed': SEED,
    
    # Learning
    'learning_rate': 0.02,      # Slower learning for better convergence
    'n_estimators': 5000,       # More trees
    'max_bin': 255,
    'feature_pre_filter': False,
    
    # Performance
    'num_threads': -1,
    'force_row_wise': True,
}

# ============================================================
# 6. ULTRA-ADVANCED WINKLER SCORE OPTIMIZATION
# ============================================================

def advanced_winkler_score(y_true, pi_lower, pi_upper, alpha=0.1):
    """
    Enhanced Winkler score with additional penalties for extreme intervals
    """
    width = pi_upper - pi_lower
    
    # Standard Winkler components
    lower_penalty = (2 / alpha) * np.maximum(0, pi_lower - y_true)
    upper_penalty = (2 / alpha) * np.maximum(0, y_true - pi_upper)
    
    # Additional penalty for extremely wide intervals (prevents gaming)
    median_width = np.median(width)
    width_penalty = 0.1 * np.maximum(0, width - 3 * median_width)
    
    # Penalty for negative intervals
    negative_penalty = 1000 * np.maximum(0, pi_lower - pi_upper)
    
    total_score = width + lower_penalty + upper_penalty + width_penalty + negative_penalty
    
    return np.mean(total_score)

def optimize_interval_scaling(y_true, pred_mean, pred_lower, pred_upper, 
                            alpha=0.1, n_trials=100):
    """
    Advanced interval scaling optimization with multiple parameters
    """
    print("Optimizing interval scaling...")
    
    best_score = float('inf')
    best_params = {'lower_scale': 1.0, 'upper_scale': 1.0, 'shift': 0.0}
    
    # Grid search over scaling parameters
    lower_scales = np.linspace(0.7, 1.5, 20)
    upper_scales = np.linspace(0.7, 1.5, 20)
    shifts = np.linspace(-0.1, 0.1, 11)
    
    for lower_scale in lower_scales:
        for upper_scale in upper_scales:
            for shift in shifts:
                # Apply transformations
                adjusted_lower = pred_mean + (pred_lower - pred_mean) * lower_scale + shift * (pred_upper - pred_lower)
                adjusted_upper = pred_mean + (pred_upper - pred_mean) * upper_scale - shift * (pred_upper - pred_lower)
                
                # Ensure lower <= upper
                adjusted_lower = np.minimum(adjusted_lower, adjusted_upper - 1000)
                
                # Calculate Winkler score
                score = advanced_winkler_score(y_true, adjusted_lower, adjusted_upper, alpha)
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'lower_scale': lower_scale,
                        'upper_scale': upper_scale,
                        'shift': shift
                    }
    
    print(f"  Best Winkler score: {best_score:.2f}")
    print(f"  Best parameters: {best_params}")
    
    return best_params, best_score

# ============================================================
# 7. MAIN PROCESSING PIPELINE
# ============================================================

# Apply ultra-advanced feature engineering
train_clean = ultra_advanced_feature_engineering(train_df, is_train=True)
test_clean = ultra_advanced_feature_engineering(test_df, is_train=False)

print(f"\nAfter feature engineering:")
print(f"Train features: {train_clean.shape[1]}")
print(f"Test features: {test_clean.shape[1]}")

# Identify text columns for special processing
text_columns = []
for col in train_clean.select_dtypes(include=['object']).columns:
    if col not in ['sale_date'] and train_clean[col].dtype == 'object':
        # Check if it looks like free text (high unique value ratio)
        unique_ratio = train_clean[col].nunique() / len(train_clean)
        if unique_ratio > 0.1:  # More than 10% unique values suggests free text
            text_columns.append(col)

print(f"Text columns identified: {text_columns}")

# Process text features
if text_columns:
    train_clean, test_clean = process_text_features(train_clean, test_clean, text_columns)

# Advanced categorical encoding
exclude_cols = ['id', 'sale_price', 'sale_date']
feature_cols = [col for col in train_clean.columns if col not in exclude_cols]
common_features = list(set(feature_cols) & set(test_clean.columns))

train_for_encoding = train_clean[common_features + ['sale_price']].copy()
test_for_encoding = test_clean[common_features].copy()

train_encoded, test_encoded, encoders = advanced_categorical_encoding(
    train_for_encoding, test_for_encoding, 'sale_price', high_cardinality_threshold=25
)

# Final feature preparation
final_features = [col for col in train_encoded.columns if col != 'sale_price']
X = train_encoded[final_features].fillna(0)
y = train_encoded['sale_price']
X_test = test_encoded[final_features].fillna(0)

print(f"\nFinal dataset shapes:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")
print(f"X_test: {X_test.shape}")

# ============================================================
# 8. CROSS-VALIDATION SETUP
# ============================================================

# Enhanced stratified K-fold for better stability
def create_stratified_folds(y, n_splits=NFOLDS, seed=SEED):
    """Create stratified folds based on target quantiles"""
    # Create quantile-based strata
    strata = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    
    for train_idx, val_idx in skf.split(np.zeros(len(y)), strata):
        folds.append((train_idx, val_idx))
    
    return folds

cv_folds = create_stratified_folds(y, n_splits=NFOLDS)

# ============================================================
# 9. MULTI-MODEL QUANTILE REGRESSION
# ============================================================

def train_quantile_ensemble(X_train, y_train, X_test, cv_folds, quantiles=[0.05, 0.95]):
    """
    Train ensemble of quantile regression models with multiple algorithms
    """
    print("Training advanced quantile regression ensemble...")
    
    results = {}
    
    for q in quantiles:
        print(f"\n--- Training quantile {q} models ---")
        
        oof_predictions = np.zeros(len(y_train))
        test_predictions = np.zeros(len(X_test))
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            print(f"  Fold {fold_idx + 1}/{len(cv_folds)}")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Quantile-specific parameters
            quantile_params = base_params.copy()
            quantile_params.update({
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
            })
            
            # Train LightGBM quantile model
            model = LGBMRegressor(**quantile_params)
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=300, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # Predictions
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            oof_predictions[val_idx] = val_pred
            test_predictions += test_pred / len(cv_folds)
            
            # Calculate fold score (quantile loss)
            if q <= 0.5:
                fold_score = np.mean(np.maximum(q * (y_val - val_pred), (q - 1) * (y_val - val_pred)))
            else:
                fold_score = np.mean(np.maximum((1 - q) * (val_pred - y_val), (q - 1) * (val_pred - y_val)))
            
            fold_scores.append(fold_score)
            print(f"    Quantile loss: {fold_score:.3f}")
        
        avg_score = np.mean(fold_scores)
        print(f"  Average quantile loss for {q}: {avg_score:.3f}")
        
        results[q] = {
            'oof': oof_predictions,
            'test': test_predictions,
            'score': avg_score
        }
    
    return results

# Train mean prediction model (for interval centering)
def train_mean_model(X_train, y_train, X_test, cv_folds):
    """Train mean prediction model with log transformation"""
    print("\n--- Training mean prediction model ---")
    
    oof_predictions = np.zeros(len(y_train))
    test_predictions = np.zeros(len(X_test))
    
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        print(f"  Fold {fold_idx + 1}/{len(cv_folds)}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Log transformation for mean model
        y_tr_log = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)
        
        mean_params = base_params.copy()
        mean_params.update({
            'objective': 'regression',
            'metric': 'rmse',
        })
        
        model = LGBMRegressor(**mean_params)
        
        model.fit(
            X_tr, y_tr_log,
            eval_set=[(X_val, y_val_log)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=300, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Predictions (back-transform)
        val_pred = np.expm1(model.predict(X_val))
        test_pred = np.expm1(model.predict(X_test))
        
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / len(cv_folds)
        
        # RMSE score
        rmse_score = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(rmse_score)
        print(f"    RMSE: {rmse_score:.2f}")
    
    avg_rmse = np.mean(fold_scores)
    print(f"  Average RMSE: {avg_rmse:.2f}")
    
    return oof_predictions, test_predictions, avg_rmse

# Train all models
print("=" * 60)
print("TRAINING ULTRA-ADVANCED QUANTILE REGRESSION ENSEMBLE")
print("=" * 60)

# Train mean model
oof_mean, test_mean, mean_rmse = train_mean_model(X, y, X_test, cv_folds)

# Train quantile models
quantile_results = train_quantile_ensemble(X, y, X_test, cv_folds, quantiles=[0.05, 0.95])

# Extract quantile predictions
oof_lower = quantile_results[0.05]['oof']
oof_upper = quantile_results[0.95]['oof']
test_lower = quantile_results[0.05]['test']
test_upper = quantile_results[0.95]['test']

# ============================================================
# 10. ADVANCED INTERVAL OPTIMIZATION
# ============================================================

print("\n" + "=" * 60)
print("ADVANCED INTERVAL CALIBRATION AND OPTIMIZATION")
print("=" * 60)

# Optimize interval scaling using OOF predictions
optimal_params, best_winkler = optimize_interval_scaling(
    y.values, oof_mean, oof_lower, oof_upper, alpha=ALPHA
)

# Apply optimal scaling to test predictions
final_lower = test_mean + (test_lower - test_mean) * optimal_params['lower_scale'] + \
              optimal_params['shift'] * (test_upper - test_lower)
final_upper = test_mean + (test_upper - test_mean) * optimal_params['upper_scale'] - \
              optimal_params['shift'] * (test_upper - test_lower)

# Ensure consistency and reasonable bounds
final_lower = np.maximum(final_lower, 10000)  # Minimum reasonable house price
final_upper = np.maximum(final_upper, final_lower + 5000)  # Minimum interval width

# Final validation metrics
oof_lower_adj = oof_mean + (oof_lower - oof_mean) * optimal_params['lower_scale'] + \
                optimal_params['shift'] * (oof_upper - oof_lower)
oof_upper_adj = oof_mean + (oof_upper - oof_mean) * optimal_params['upper_scale'] - \
                optimal_params['shift'] * (oof_upper - oof_lower)

# Ensure consistency for validation
oof_lower_adj = np.minimum(oof_lower_adj, oof_upper_adj - 1000)

# Calculate final validation metrics
final_coverage = np.mean((y.values >= oof_lower_adj) & (y.values <= oof_upper_adj))
final_width = np.mean(oof_upper_adj - oof_lower_adj)
final_winkler = advanced_winkler_score(y.values, oof_lower_adj, oof_upper_adj, ALPHA)

print(f"\nFINAL VALIDATION RESULTS:")
print(f"Winkler Score: {final_winkler:.2f}")
print(f"Coverage: {final_coverage:.3f} (target: {COVERAGE:.3f})")
print(f"Average interval width: ${final_width:.0f}")
print(f"Mean RMSE: {mean_rmse:.2f}")

# ============================================================
# 11. SUBMISSION PREPARATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUBMISSION PREPARATION")
print("=" * 60)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'pi_lower': final_lower,
    'pi_upper': final_upper
})

print("Final submission statistics:")
print(f"Number of predictions: {len(submission)}")
print(f"Average prediction (mean): ${test_mean.mean():.0f}")
print(f"Average interval width: ${(submission['pi_upper'] - submission['pi_lower']).mean():.0f}")
print(f"Median interval width: ${(submission['pi_upper'] - submission['pi_lower']).median():.0f}")
print(f"Min prediction (lower): ${submission['pi_lower'].min():.0f}")
print(f"Max prediction (upper): ${submission['pi_upper'].max():.0f}")
print(f"Intervals with negative width: {(submission['pi_lower'] > submission['pi_upper']).sum()}")

# Save submission
submission.to_csv('ultra_enhanced_submission.csv', index=False)
print("\nSubmission saved as 'ultra_enhanced_submission.csv'")

# Display sample predictions
print("\nSample predictions:")
print(submission.head(10))

print("\n" + "=" * 80)
print("ULTRA-ENHANCED QUANTILE REGRESSION SOLUTION COMPLETED!")
print(f"Final Validation Winkler Score: {final_winkler:.2f}")
print(f"Final Validation Coverage: {final_coverage:.3f}")
print(f"Total engineered features: {len(final_features)}")
print(f"Mean model RMSE: {mean_rmse:.2f}")
print("=" * 80)




