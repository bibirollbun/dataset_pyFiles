# Import standard libraries 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

#  Scikit-learn core utilities 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_regression

#  Models 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

#  Metrics & Stats 
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
from scipy import stats

# Ignore all warnings

import warnings
warnings.filterwarnings('ignore')


# Load datasets and drop unused columns
train_data = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
train_data.drop(columns=['id', 'clean_title', 'fuel_type','accident'], inplace=True)
test_data = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
test_data.drop(columns=['id', 'clean_title', 'fuel_type','accident'], inplace=True)


# Function to apply Box-Cox transformation to specified columns
def apply_boxcox_transform(df, columns):
    df_boxcox = df.copy()
    shifts = {}
    lambdas = {}
    for col in columns:
        min_val = df_boxcox[col].min()
        shift_amount = min_val - 1 if min_val <= 0 else 0
        shifts[col] = shift_amount
        shifted_data = df_boxcox[col] - shift_amount
        boxcox_data, lambda_value = stats.boxcox(shifted_data)
        lambdas[col] = lambda_value
        df_boxcox[col] = boxcox_data
    return df_boxcox, shifts, lambdas


# Function to invert Box-Cox transformation
def inverse_boxcox(y_trans, lambda_val=-0.03353684426278687, shift=0):
    if np.isclose(lambda_val, 0):
        y_original = np.exp(y_trans) + shift
    else:
        y_original = np.power(lambda_val * y_trans + 1, 1/lambda_val) + shift
    return y_original


# 1. Normalize raw strings
for df in (train_data, test_data):
    # 1) Fill NaNs with a placeholder, 2) cast every entry to str, 3) then do your .str ops
    df['transmission'] = (
        df['transmission']
          .fillna('')              # replace NaN with empty string
          .astype(str)             # ensure everything is string
          .str.strip()             # now safe to use .str
          .str.lower()
    )


# 2. Map to canonical categories
category_map = {
    # AUTOMATIC variants
    '6-speed a/t':                       'automatic',
    '8-speed automatic':                 'automatic',
    'automatic':                         'automatic',
    '7-speed a/t':                       'automatic',
    'a/t':                               'automatic',
    '8-speed a/t':                       'automatic',
    'transmission w/dual shift mode':    'automatic',
    '9-speed automatic':                 'automatic',
    '10-speed automatic':                'automatic',
    '1-speed a/t':                       'automatic',
    '2-speed a/t':                       'automatic',
    '2-speed automatic':                 'automatic',
    '4-speed a/t':                       'automatic',
    '5-speed automatic':                 'automatic',
    '4-speed automatic':                 'automatic',
    '6-speed automatic':                 'automatic',
    '9-speed a/t':                       'automatic',
    '10-speed a/t':                      'automatic',
    '7-speed automatic':                 'automatic',
    '6-speed electronically controlled automatic with o': 'automatic',
    'single-speed fixed gear':           'automatic',
    '7-speed dct automatic':             'automatic',
    '10-speed automatic with overdrive': 'automatic',
    'automatic, 9-spd 9g-tronic':        'automatic',
    'automatic, 8-spd':                  'automatic',
    'automatic, 8-spd sport w/sport & manual modes':     'automatic',
    'automatic, 8-spd pdk dual-clutch':  'automatic',
    'automatic, 8-spd m steptronic w/drivelogic, sport & manual modes': 'automatic',
    'automatic, 8-spd dual-clutch':      'automatic',
    'transmission overdrive switch':     'automatic',

    # TIPTRONIC variants
    '7-speed automatic with auto-shift': 'tiptronic',
    '5-speed a/t':                       'tiptronic',
    '7-speed a/t tiptronic':             'tiptronic',
    '8-speed at':                        'tiptronic',
    '8-speed a/t':                       'tiptronic',

    # MANUAL variants
    '6-speed m/t':                       'manual',
    '7-speed m/t':                       'manual',
    '6-speed manual':                    'manual',
    '5-speed m/t':                       'manual',
    'manual':                            'manual',
    '7-speed manual':                    'manual',
    '8-speed manual':                    'manual',
    'm/t':                               'manual',
    '6 speed at/mt':                     'manual',
    '6 speed mt':                        'manual',

    # CVT / VARIATOR variants
    'automatic cvt':                     'variator',
    'cvt transmission':                  'variator',
    'cvt-f':                             'variator',
    'variable':                          'variator',
}

for df in (train_data, test_data):
    df['transmission_cat'] = (
        df['transmission']
          .map(category_map)              # maps known keys → cat
          .fillna('other')                # unknown → other
    )

# 3. Map categories to integers
numeric_map = {
    'automatic': 1,
    'tiptronic': 2,
    'manual':    3,
    'variator':  4,
    'other':     5
}

for df in (train_data, test_data):
    df['transmission'] = (
        df['transmission_cat']
          .map(numeric_map)
          .fillna(numeric_map['other'])  # just in case
          .astype(int)
    )
    df.drop(columns=['transmission_cat'], inplace=True)

# 4. Verify
print("Train transmission distribution:")
print(train_data['transmission'].value_counts())
print("Dtype:", train_data['transmission'].dtype)





class OutlierClipper(BaseEstimator, TransformerMixin):
    """Analyze and visualize outlier distributions for numeric features only."""
    
    def __init__(self, methods=['iqr', 'zscore', 'percentile'], 
                 contamination=0.02, visualize=True):
        self.methods = methods
        self.contamination = contamination
        self.visualize = visualize
        self.bounds = {}
        self.method_chosen = {}
    
    def _analyze_distribution(self, series):
        s = series.dropna()
        return {
            'skewness': stats.skew(s),
            'kurtosis': stats.kurtosis(s),
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            'min': s.min(),
            'max': s.max(),
            'is_skewed': abs(stats.skew(s)) > 1.0,
            'is_heavy_tailed': stats.kurtosis(s) > 2.0,
        }
    
    def _get_bounds_iqr(self, s, multiplier):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - multiplier*iqr, q3 + multiplier*iqr
    
    def _get_bounds_zscore(self, s, threshold):
        m, st = s.mean(), s.std()
        return m - threshold*st, m + threshold*st
    
    def _get_bounds_percentile(self, s):
        lo_p = self.contamination/2
        hi_p = 1 - lo_p
        return s.quantile(lo_p), s.quantile(hi_p)
    
    def _get_best_bounds(self, series, name):
        info = self._analyze_distribution(series)
        candidates = {}
        # IQR
        if 'iqr' in self.methods:
            mult = 2.5 if info['is_heavy_tailed'] else (2.0 if info['is_skewed'] else 1.5)
            candidates['iqr'] = self._get_bounds_iqr(series, mult)
        # Z-score
        if 'zscore' in self.methods:
            thr = 4.0 if info['is_skewed'] else 3.0
            candidates['zscore'] = self._get_bounds_zscore(series, thr)
        # Percentile
        if 'percentile' in self.methods:
            candidates['percentile'] = self._get_bounds_percentile(series)
        # Choose method
        if info['is_skewed'] and info['is_heavy_tailed']:
            method = 'iqr'
        elif info['is_skewed']:
            method = 'iqr' if 'iqr' in candidates else 'percentile'
        elif abs(info['kurtosis']) < 1.0:
            method = 'zscore' if 'zscore' in candidates else 'percentile'
        else:
            method = 'percentile'
        method = method if method in candidates else next(iter(candidates))
        self.method_chosen[name] = method
        return candidates[method]
    
    def _visualize_bounds(self, series, name, lower, upper):
        if not self.visualize:
            return
        plt.figure(figsize=(12,5))
        # Histogram + bounds
        plt.subplot(1,2,1)
        series.hist(bins=40, alpha=0.7)
        plt.axvline(lower, color='red', ls='--', label=f'Lower: {lower:.2f}')
        plt.axvline(upper, color='red', ls='--', label=f'Upper: {upper:.2f}')
        plt.title(f"{name} ({self.method_chosen[name]})")
        plt.legend()
        # Boxplot + bounds
        plt.subplot(1,2,2)
        plt.boxplot(series.dropna(), vert=False)
        plt.axvline(lower, color='red', ls='--')
        plt.axvline(upper, color='red', ls='--')
        plt.title("Boxplot with bounds")
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def fit(self, X, y=None):
        # Only keep numeric columns
        numeric_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
        for col in numeric_cols:
            s = X[col].dropna()
            if s.nunique() <= 1:
                continue
            lo, hi = self._get_best_bounds(s, col)
            self.bounds[col] = (lo, hi)
            self._visualize_bounds(s, col, lo, hi)
        # Summary
        summary = pd.Series(self.method_chosen).value_counts()
        print("\nOutlier detection methods chosen:")
        for m, cnt in summary.items():
            print(f"  {m}: {cnt} feature(s)")
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, (lo, hi) in self.bounds.items():
            if col not in X_copy or not is_numeric_dtype(X_copy[col]) :
                continue
            
            col_data = X_copy[col]
            
            # Create a mask of finite (non-NaN, non-inf) values
            valid_mask = col_data.notna() & np.isfinite(col_data)
    
            # Count outliers
            below = (col_data[valid_mask] < lo).sum()
            above = (col_data[valid_mask] > hi).sum()
            total = below + above
    
            # Clip only valid numeric values
            X_copy.loc[valid_mask, col] = col_data[valid_mask].clip(lower=lo, upper=hi)
    
            if total:
                pct = 100 * total / valid_mask.sum()
                method = self.method_chosen.get(col, 'unknown')
                print(f"{col}: {total} outliers ({pct:.2f}%) clipped using {method}")
        
            return X_copy
clipper=OutlierClipper()
train_data=clipper.fit_transform(train_data)
test_data=clipper.transform(test_data)



class DataFrameKeeper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X  # just returns the DataFrame as-is



#  Feature Engineering Functions 

def classify_model_segment(model_name, brand=None):
    """Enhanced model segment classification"""
    if pd.isna(model_name):
        return 'unknown'
    
    model_lower = str(model_name).lower()
    
    # Comprehensive keyword lists
    sedan_keywords = [
        'sedan', 'saloon', 'camry', 'accord', 'malibu', 'altima', 'civic', 'corolla', 
        'passat', 'jetta', 'sentra', 'sonata', 'elantra', 'fusion', 'taurus', 'avalon', 
        'maxima', 'cts', 'a3', 'a4', 'a6', 'a8', '328i', '330i', '335i', '528i', '530i', 
        '535i', '320', '325', '335', '340', '520', '525', '530', '535', '540',
        'impala', 'cruze', 'legacy', 'es350', 'gs350', 'charger', 'challenger', 'mazda3',
        'mazda6', 'optima', 'forte', 'g37', 'q50', 'tlx', 'rl', 'tsx'
    ]
    
    suv_keywords = [
        'suv', 'crossover', 'explorer', 'highlander', 'cr-v', 'crv', 'rav4', 'equinox', 
        'terrain', 'edge', 'escape', 'expedition', 'tahoe', 'suburban', 'yukon', 
        'escalade', 'pilot', 'passport', 'pathfinder', 'armada', 'murano', 'rogue',
        'forester', 'outback', 'crosstrek', 'santa', 'tucson', 'sorento', 'sportage',
        'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'gla', 'glb', 'glc', 'gle', 'gls',
        'q3', 'q5', 'q7', 'q8', 'traverse', 'acadia', 'enclave', 'envision', 'encore',
        'rx', 'nx', 'gx', 'lx', 'mdx', 'rdx', 'telluride', 'atlas', 'tiguan', 'cherokee',
        'wrangler', 'compass', 'renegade', 'blazer', 'trailblazer', 'bronco', 
        'land cruiser', 'land rover', 'range rover', 'defender', 'discovery', 'velar',
        'evoque'
    ]
    
    truck_keywords = [
        'truck', 'pickup', 'f-150', 'f150', 'f-250', 'f250', 'f-350', 'f350',
        'silverado', 'sierra', 'ram', '1500', '2500', '3500', 'ranger', 'tacoma', 
        'tundra', 'ridgeline', 'colorado', 'canyon', 'frontier', 'titan', 'gladiator'
    ]
    
    luxury_keywords = [
        'luxury', 'premium', 'mercedes', 'bmw', 'audi', 'lexus', 'cadillac', 'infiniti',
        'acura', 'lincoln', 'genesis', 'volvo', 'porsche', 'jaguar', 'land rover',
        'range rover', 'maserati', 'bentley', 'rolls-royce', 'maybach', 'e-class', 
        's-class', 'c-class', 'a-class', '5-series', '7-series', '3-series'
    ]
    
    sports_keywords = [
        'sports', 'coupe', 'convertible', 'roadster', 'mustang', 'camaro', 'corvette', 
        'miata', 'mx-5', 'supra', 'gr86', '86', 'brz', 'z4', 'tt', 'r8', 'nsx',
        'cayman', 'boxster', '911', 'gt-r', 'gtr', 'challenger', 'charger', 'wrx', 'sti'
    ]
    
    minivan_keywords = [
        'minivan', 'van', 'sienna', 'odyssey', 'pacifica', 'carnival', 'sedona', 
        'grand caravan', 'voyager', 'quest', 'transit', 'sprinter'
    ]
    
    hybrid_ev_keywords = [
        'hybrid', 'phev', 'electric', 'ev', 'tesla', 'leaf', 'bolt', 'volt', 'prius',
        'prime', 'clarity', 'ioniq', 'niro', 'id.4', 'mach-e', 'model s', 'model 3',
        'model x', 'model y'
    ]
    
    # Check brand info for luxury classification
    is_luxury_brand = False
    if brand is not None:
        luxury_brands = ["mercedes", "bmw", "audi", "lexus", "acura", "jaguar", 
                         "infiniti", "cadillac", "land rover", "lincoln", "volvo", 
                         "tesla", "maserati", "bentley", "genesis", "ferrari", 
                         "rolls-royce", "aston", "mclaren", "porsche"]
        
        is_luxury_brand = any(lux_brand in str(brand).lower() for lux_brand in luxury_brands)
    
    # First check for electric/hybrid as it crosses segments
    if any(kw in model_lower for kw in hybrid_ev_keywords):
        return 'electric_hybrid'
    
    # Primary segment checks
    if any(kw in model_lower for kw in truck_keywords):
        return 'truck'
    elif any(kw in model_lower for kw in suv_keywords):
        if is_luxury_brand:
            return 'luxury_suv'
        return 'suv'
    elif any(kw in model_lower for kw in minivan_keywords):
        return 'minivan'
    elif any(kw in model_lower for kw in sports_keywords):
        if is_luxury_brand:
            return 'luxury_sports'
        return 'sports'
    elif any(kw in model_lower for kw in sedan_keywords):
        if is_luxury_brand:
            return 'luxury_sedan'
        return 'sedan'
    elif is_luxury_brand:
        return 'luxury'
    else:
        return 'other'

def extract_engine_features(df):
    """Enhanced engine feature extraction"""
    result = df.copy()
    
    # Create empty columns for features
    result['horsepower'] = np.nan
    result['displacement'] = np.nan
    result['cylinders'] = np.nan
    result['is_turbo'] = 0
    result['is_supercharged'] = 0
    result['engine_config'] = 'unknown'
    
    # Process non-null engine strings
    mask = result['engine'].notna()
    
    # Extract horsepower
    hp_pattern = r"([\d\.]+)\s*HP"
    hp_extracted = result.loc[mask, 'engine'].str.extract(hp_pattern, flags=re.IGNORECASE)
    if not hp_extracted.empty and not hp_extracted[0].empty:
        result.loc[mask, 'horsepower'] = pd.to_numeric(hp_extracted[0], errors='coerce')
    
    # Extract displacement
    disp_pattern = r"([\d\.]+)\s*L(?:iter)?(?:/\d+)?"
    disp_extracted = result.loc[mask, 'engine'].str.extract(disp_pattern, flags=re.IGNORECASE)
    if not disp_extracted.empty and not disp_extracted[0].empty:
        result.loc[mask, 'displacement'] = pd.to_numeric(disp_extracted[0], errors='coerce')
    
    # Extract cylinders and engine configuration
    eng_config_pattern = r"([VHI])[- ]?(\d+)|(\d+)[ -]?([VHI])|(\d+)[ -]?(Cylinder|Cyl)"
    config_extracted = result.loc[mask, 'engine'].str.extract(eng_config_pattern, flags=re.IGNORECASE)
    
    # Process complex cylinder info
    if not config_extracted.empty:
        # Format: V8, V-6, etc.
        v_config_mask = config_extracted[0].notna() & config_extracted[1].notna()
        if not v_config_mask.empty:
            result.loc[mask & v_config_mask, 'engine_config'] = config_extracted[0].str.upper()
            result.loc[mask & v_config_mask, 'cylinders'] = pd.to_numeric(config_extracted[1], errors='coerce')
        
        # Format: 8V, 6-I, etc.
        alt_config_mask = config_extracted[2].notna() & config_extracted[3].notna()
        if not alt_config_mask.empty:
            result.loc[mask & alt_config_mask, 'engine_config'] = config_extracted[3].str.upper()
            result.loc[mask & alt_config_mask, 'cylinders'] = pd.to_numeric(config_extracted[2], errors='coerce')
        
        # Format: 8 Cylinder, 6-Cylinder, etc.
        cyl_only_mask = config_extracted[4].notna()
        if not cyl_only_mask.empty:
            result.loc[mask & cyl_only_mask, 'cylinders'] = pd.to_numeric(config_extracted[4], errors='coerce')
    
    # Extract turbo/supercharged
    result.loc[mask & result['engine'].str.contains('turbo', case=False, na=False), 'is_turbo'] = 1
    result.loc[mask & result['engine'].str.contains('supercharged', case=False, na=False), 'is_supercharged'] = 1
    
    return result


def create_interaction_features(df):
    """Create advanced interaction features with proper encoding"""
    df = df.copy()
    
    # Power density: Horsepower per liter
    if 'horsepower' in df.columns and 'displacement' in df.columns:
        df['hp_per_liter'] = df['horsepower'] / df['displacement']
    
    # Power-to-age ratio
    if 'horsepower' in df.columns and 'car_age' in df.columns:
        df['hp_age_ratio'] = df['horsepower'] / df['car_age']
    
    # Mileage relative to age
    if 'milage' in df.columns and 'car_age' in df.columns:
        avg_miles_per_year = 12000
        expected_mileage = df['car_age'] * avg_miles_per_year
        df['mileage_ratio'] = df['milage'] / expected_mileage
        df['mileage_score'] = 1 / df['mileage_ratio']
    
    # Price depreciation nonlinearities
    if 'model_year' in df.columns:
        df['model_year_squared'] = df['model_year'] ** 2
        df['recent_model'] = (df['model_year'] >= 2018).astype(int)
    
    # Power transformations
    if 'horsepower' in df.columns:
        df['horsepower_log'] = np.log1p(df['horsepower'])
        df['horsepower_sqrt'] = np.sqrt(df['horsepower'])
    
    if 'milage' in df.columns:
        df['milage_log'] = np.log1p(df['milage'])
    
    # Luxury-specific interactions
    if 'is_luxury' in df.columns:
        if 'horsepower' in df.columns:
            df['luxury_hp'] = df['is_luxury'] * df['horsepower']
        if 'car_age' in df.columns:
            df['luxury_age_effect'] = df['is_luxury'] * np.log1p(df['car_age'])
    
    # Cylinder count optimization
    if 'cylinders' in df.columns:
        # Create cylinder categories without one-hot encoding
        cylinder_conditions = [
            (df['cylinders'] <= 4),
            (df['cylinders'] > 4) & (df['cylinders'] <= 6),
            (df['cylinders'] > 6) & (df['cylinders'] <= 8),
            (df['cylinders'] > 8)
        ]
        cylinder_values = [1, 2, 3, 4]  # Numeric values instead of one-hot
        df['cylinder_category'] = np.select(cylinder_conditions, cylinder_values, default=0)
    
    # Create age categories without one-hot encoding
    if 'car_age' in df.columns:
        age_conditions = [
            (df['car_age'] <= 3),
            (df['car_age'] > 3) & (df['car_age'] <= 6),
            (df['car_age'] > 6) & (df['car_age'] <= 10),
            (df['car_age'] > 10) & (df['car_age'] <= 15),
            (df['car_age'] > 15)
        ]
        age_values = [1, 2, 3, 4, 5]  # Numeric values instead of one-hot
        df['age_category'] = np.select(age_conditions, age_values, default=0)
    
    return df


def add_brand_pricing_features(train_df, test_df=None, target='price'):
    """Add brand-specific pricing features"""
    train_df = train_df.copy()
    
    
    if target in train_df.columns and 'brand' in train_df.columns:
        # Calculate brand price statistics
        brand_stats = train_df.groupby('brand')[target].agg(['mean', 'median', 'std', 'count']).reset_index()
        brand_stats.columns = ['brand', 'brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count']
        
        # Remove brands with too few examples
        brand_stats = brand_stats[brand_stats['brand_count'] >= 5]
        
        # Calculate global statistics
        global_mean = train_df[target].mean()
        global_median = train_df[target].median()
        global_std = train_df[target].std()
        
        # Create relative metrics
        brand_stats['brand_price_ratio'] = brand_stats['brand_price_mean'] / global_mean
        brand_stats['brand_premium'] = (brand_stats['brand_price_mean'] - global_mean) / global_std
        
        # Merge stats back to training data
        train_df = train_df.merge(brand_stats, on='brand', how='left')
        
        # Fill missing values with global statistics
        for col in ['brand_price_mean', 'brand_price_median', 'brand_price_std']:
            if col == 'brand_price_mean':
                train_df[col] = train_df[col].fillna(global_mean)
            elif col == 'brand_price_median':
                train_df[col] = train_df[col].fillna(global_median)
            elif col == 'brand_price_std':
                train_df[col] = train_df[col].fillna(global_std)
        
        train_df['brand_price_ratio'] = train_df['brand_price_ratio'].fillna(1.0)
        train_df['brand_premium'] = train_df['brand_premium'].fillna(0.0)
        
        # Handle test data if provided
        if test_df is not None:
            test_df = test_df.copy()
            test_df = test_df.merge(brand_stats, on='brand', how='left')
            
            # Fill missing values with global statistics
            for col in ['brand_price_mean', 'brand_price_median', 'brand_price_std']:
                if col == 'brand_price_mean':
                    test_df[col] = test_df[col].fillna(global_mean)
                elif col == 'brand_price_median':
                    test_df[col] = test_df[col].fillna(global_median)
                elif col == 'brand_price_std':
                    test_df[col] = test_df[col].fillna(global_std)
            
            test_df['brand_price_ratio'] = test_df['brand_price_ratio'].fillna(1.0)
            test_df['brand_premium'] = test_df['brand_premium'].fillna(0.0)
            
            return train_df, test_df
    
    return train_df if test_df is None else (train_df, test_df)

def group_model_names(df):
    """Group car models into market segments"""
    df = df.copy()
    
    if 'model' in df.columns:
        # Create model segment feature
        df['model_segment'] = df.apply(
            lambda row: classify_model_segment(
                row['model'], 
                brand=row.get('brand', None)
            ),
            axis=1
        )
        
        # One-hot encode the segment
        segment_dummies = pd.get_dummies(df['model_segment'], prefix='segment')
        df = pd.concat([df, segment_dummies], axis=1)
    
    return df
def preprocess_data(df):
    """Complete preprocessing pipeline with enhanced features"""
    # Step 1: Extract engine features
    df_processed = extract_engine_features(df)
    
    
    # Step 2: Add car age and related features
    df_processed['car_age'] = 2024 - df_processed['model_year']
    df_processed['car_age'] = df_processed['car_age'].clip(lower=1)  # Better than replace
    df_processed['miles_per_year'] = df_processed['milage'] / df_processed['car_age']
    
    # # Step 3: Map accident values
    # df_processed['accident_binary'] = df_processed['accident'].apply(
    #     lambda x: 1 if isinstance(x, str) and ('accident' in x.lower() or 'damage' in x.lower()) else 0
    # )
    
    # Step 4: Add luxury brand flag
    luxury_brands = ["Mercedes-Benz", "BMW", "Audi", "Lexus", "Acura", "Jaguar", 
                     "INFINITI", "Cadillac", "Land", "Lincoln", "Volvo", "Tesla",
                     "Maserati", "Bentley", "Genesis", "Ferrari", "Rolls-Royce",
                     "Aston", "McLaren", "Porsche"]
    
    df_processed['is_luxury'] = df_processed['brand'].apply(
        lambda x: 1 if isinstance(x, str) and any(brand.lower() in x.lower() for brand in luxury_brands) else 0
    )
    
    # Step 5: Process model into segments
    if 'model' in df_processed.columns:
        # Create model segment feature
        df_processed['model_segment'] = df_processed.apply(
            lambda row: classify_model_segment(
                row['model'], 
                brand=row.get('brand', None)
            ),
            axis=1
        )
        
        # Important: Encode model_segment as numbers instead of one-hot
        le = LabelEncoder()
        df_processed['model_segment'] = le.fit_transform(df_processed['model_segment'])
    
    # Step 6: Process categorical columns
    categorical_config = {
        'brand': {'top_n': 20, 'method': 'label'},
        'model': {'top_n': 600, 'method': 'label'},
        'ext_col': {'top_n': 10, 'method': 'label'},
        'int_col': {'top_n': 5, 'method': 'label'},
        'engine_config': {'top_n': 5, 'method': 'label'}
    }
    
    # Process model to extract base model
    if 'model' in df_processed.columns:
        df_processed['model'] = df_processed['model'].apply(lambda x: 
            str(x).split()[0] if not pd.isna(x) else "missing")
    
    # Process each categorical column
    for col, config in categorical_config.items():
        if col not in df_processed.columns:
            continue
            
        # Fill missing values
        df_processed[col] = df_processed[col].fillna('missing')
        
        # Apply TopN selection
        top_n = config['top_n']
        counts = df_processed[col].value_counts()
        top_categories = set(counts.nlargest(top_n).index)
        df_processed[col] = df_processed[col].apply(lambda x: x if x in top_categories else 'other')
        
        # Apply encoding
        if config['method'] == 'label':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    
    # Step 7: Fill missing values in numeric columns
    numeric_cols = ['horsepower', 'displacement', 'cylinders', 'model_year', 
                   'milage', 'car_age', 'miles_per_year']
    
    for col in numeric_cols:
        if col in df_processed.columns:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
    
    # Step 8: Add interaction features
    df_processed = create_interaction_features(df_processed)
    
    # Drop original columns that have been processed
  #  columns_to_drop = ['engine', 'accident']
    columns_to_drop = ['engine']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
    # Step 9: Final check - make sure all columns are numeric
    non_numeric_cols = df_processed.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Converting non-numeric columns to numeric: {non_numeric_cols}")
        for col in non_numeric_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])

    
    return df_processed


def create_stacked_regressor(random_state=42):
    """Create a stacked regressor ensemble with XGBoost, GBRT, Ridge, and LightGBM."""
    
    # Base models
    base_models = [
        (
            'xgb',
            xgb.XGBRegressor(
                tree_method="hist",
                device='cuda',
                n_estimators=1000,
                learning_rate=0.10,
                max_depth=12,
                min_child_weight=80,
                subsample=0.80,
                colsample_bytree=0.70,
                gamma=1.0,
                reg_alpha=4.0,
                random_state=random_state
            )
        ),
        (
            'gbr',
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=20,
                subsample=0.80,
                random_state=random_state
            )
        ),
        (
            'ridge',
            Ridge(alpha=1.0, random_state=random_state)
        ),
        (
            'lgbm',
            lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.80,
                colsample_bytree=0.70,
                random_state=random_state
            )
        )
    ]
    
    # Meta-model
    meta_model = Ridge(alpha=1.0, random_state=random_state)
    
    # Assemble Stacking Regressor
    stacked = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    
    return stacked

#  Main Pipeline Creation 

def create_optimized_pipeline(use_stacking=False):
    """Create an optimized pipeline with smart feature selection"""
    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('df_keeper', DataFrameKeeper()),
        ('preprocessor', FunctionTransformer(preprocess_data)),
    ])
    
    # Choose regressor
    if use_stacking:
        regressor = create_stacked_regressor()
    else:
       
        
        regressor = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=12,
            min_child_weight=80,
            subsample=0.8,
            colsample_bytree=0.7,
            gamma=1.0,
            alpha=4.0,
            random_state=42,
            tree_method="hist", 
            device="cuda",
        )

    
    # Create complete pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessing_pipeline),
        ('imputer', SimpleImputer(strategy='mean')), 
        ('feature_selection', SelectKBest(score_func=mutual_info_regression, k=12)),
        ('regressor', regressor)
    ])
    
    return pipeline

#  Hyperparameter Optimization 

def optimize_hyperparameters(pipeline, X, y, n_iter=20, cv=5):
    """Run hyperparameter optimization for the pipeline with proper parameter handling"""
    # Check if we're using stacked regressor or direct XGBoost
    is_stacked = isinstance(pipeline.named_steps['regressor'], StackingRegressor)
    
    if is_stacked:
        # For stacked regressor, we'll only optimize feature selection parameters
        # since the stack has multiple models with different parameters
        param_dist = {
            'feature_selection__k': randint(10, 15),
        }
    else:
        # For XGBoost only, we can optimize all parameters
        param_dist = {
            'feature_selection__k': randint(10, 15),
            'regressor__n_estimators': randint(800, 1200),
            'regressor__max_depth': randint(10, 15),
            'regressor__learning_rate': uniform(0.05, 0.15),
            'regressor__subsample': uniform(0.7, 0.3),
            'regressor__colsample_bytree': uniform(0.6, 0.3),
            'regressor__min_child_weight': randint(60, 100),
            'regressor__gamma': uniform(0.5, 2.0),
            'regressor__lambda': uniform(0.1, 0.5),
            'regressor__alpha': uniform(3.0, 5.0)
        }
    
    # Set up search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        random_state=42,
        error_score='raise' ,
        n_jobs=-1
    )
    
    # Fit search
    search.fit(X, y)
    
    # Print results
    print("Best parameters:", search.best_params_)
    print("Best RMSE:", -search.best_score_)
    
    return search.best_estimator_, search.best_params_

#  Final Execution Function 

def build_and_train_model(train_data, test_data, use_stacking=False, optimize=False, n_iter=20):
    """Complete pipeline from data to predictions"""
    print("Starting car price prediction pipeline...")
   
    original_train_price = None
    if 'price' in train_data.columns:
        original_train_price = train_data['price'].copy()
    
    # Step 1: Add brand pricing features (if price data is available)
    if 'price' in train_data.columns:
        print("Adding brand pricing features...")
        train_data, test_data = add_brand_pricing_features(train_data, test_data)
    
    # Step 2: Apply Box-Cox transformation if needed
    try:
        print("Applying Box-Cox transformation...")
        # Only transform price in training data, as it doesn't exist in test data
        train_cols = ['milage']
        test_cols = ['milage']
        if 'price' in train_data.columns:
            train_cols.append('price')
        
        train_data, shift_dict, lambda_dict = apply_boxcox_transform(train_data, train_cols)
        test_data, _, _ = apply_boxcox_transform(test_data, test_cols)
        
        # Store transformed values for later
        transformed_train_price = train_data['price'].copy() if 'price' in train_data.columns else None
    except Exception as e:
        print(f"Box-Cox transformation failed: {e}")
        print("Continuing without transformation...")
        shift_dict = {}
        lambda_dict = {}
        if 'price' in train_data.columns:
            transformed_train_price = train_data['price'].copy()
    
    # Step 3: Prepare data for modeling
    print("Preparing data...")
    X = train_data.drop(columns=['price']) if 'price' in train_data.columns else train_data
    y = train_data['price'] if 'price' in train_data.columns else None

    
    
    # Step 4: Create pipeline
    pipeline = create_optimized_pipeline(use_stacking=use_stacking)
    
    # Step 5: Optimize or train directly
    if optimize:
        print(f"Optimizing hyperparameters with {n_iter} iterations...")
        best_model, best_params = optimize_hyperparameters(pipeline, X, y, n_iter=n_iter)
        print("Optimization complete.")
    else:
        print("Training model with default hyperparameters...")
        best_model = pipeline
        best_model.fit(X, y)
        print("Training complete.")
    
    # Step 6: Make predictions
    print("Making predictions...")
    test_predictions = best_model.predict(test_data)
    
    # Step 7: Inverse transform predictions if Box-Cox was applied
    if 'price' in lambda_dict:
        try:
            print(f"Inverting Box-Cox transformation with lambda={lambda_dict['price']:.4f}")
            price_predictions = inverse_boxcox(test_predictions, lambda_dict['price'])
            if 'price' in shift_dict and shift_dict['price'] > 0:
                price_predictions -= shift_dict['price']
        except Exception as e:
            print(f"Failed to invert Box-Cox transformation: {e}")
            price_predictions = test_predictions
    else:
        price_predictions = test_predictions
    
    # Return model and predictions for further analysis
    return best_model, price_predictions
#  run with stacking 
# best_model, predictions = build_and_train_model(
#     train_data=train_data,
#     test_data=test_data,
#     use_stacking=True,  # Set to False to use only XGBoost
#     optimize=True,      # Set to True to run hyperparameter optimization
#     n_iter=10           # Number of optimization iterations (reduced to 10 for faster execution)
# )

# Alternative: Skip optimization for stacked regressor
# best_model, predictions = build_and_train_model(
#     train_data=train_data,
#     test_data=test_data,
#     use_stacking=True,
#     optimize=False  # Skip optimization for stacked regressor
# )


# Build and train the stacking regressor pipeline
best_model, predictions = build_and_train_model(
    train_data=train_data,
    test_data=test_data,
    use_stacking=False,
    optimize=True,
    n_iter=10
)

