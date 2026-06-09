# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer, SimpleImputer

import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler


def Test_Submission(submitting, splitratio=0.8):

    if submitting: 
        train_df = pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_train.csv')
        test_df = pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_test.csv')
        return train_df, test_df
    else:
        train = pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_train.csv')
        splitindex = int(splitratio * len(train))
        train_df = train[:splitindex]
        test_df = train[splitindex:]
        return train_df, test_df
        
submitting = False
if submitting:
    train_df, test_df = Test_Submission(submitting)
else:
    train_df, test_df = Test_Submission(submitting)

# train_df[train_df.select_dtypes(['object']).columns] = train_df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
# test_df[test_df.select_dtypes(['object']).columns] = test_df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
origin_col = train_df.columns


def advanced_preprocessing(train_df, test_df):
    """Advanced preprocessing with multiple imputation strategies"""
    train = train_df.copy()
    test = test_df.copy()
    
    # Handle damage-related columns
    if 'damage' in train.columns and 'damage_cost' in train.columns:
        # For non-damaged cars, damage_cost should be 0
        train.loc[train['damage'] == 0, 'damage_cost'] = 0
        test.loc[test['damage'] == 0, 'damage_cost'] = 0
        
        # For damaged cars with missing damage_cost, use KNN imputation
        if train['damage'].sum() > 0:
            damaged_mask = train['damage'] == 1
            if damaged_mask.sum() > 5:  # Need enough samples for KNN
                # Create features for KNN imputation
                impute_features = ['horsepower', 'year', 'mileage'] if all(col in train.columns for col in ['horsepower', 'year', 'mileage']) else []
                if impute_features:
                    knn_imputer = KNNImputer(n_neighbors=5)
                    damaged_data = train.loc[damaged_mask, impute_features + ['damage_cost']]
                    imputed_data = knn_imputer.fit_transform(damaged_data)
                    train.loc[damaged_mask, 'damage_cost'] = imputed_data[:, -1]
                    
                    # Apply to test
                    test_damaged_mask = test['damage'] == 1
                    if test_damaged_mask.sum() > 0:
                        test_damaged_data = test.loc[test_damaged_mask, impute_features + ['damage_cost']]
                        test_imputed = knn_imputer.transform(test_damaged_data)
                        test.loc[test_damaged_mask, 'damage_cost'] = test_imputed[:, -1]
            else:
                # Fallback to median
                damage_median = train[train['damage'] == 1]['damage_cost'].median()
                if pd.isna(damage_median):
                    damage_median = 10000
                train['damage_cost'].fillna(damage_median, inplace=True)
                test['damage_cost'].fillna(damage_median, inplace=True)
    
    if 'damage' in train.columns and 'damage_type' in train.columns:
        # Handle damage_type
        train.loc[train['damage'] == 0, 'damage_type'] = 'none'
        test.loc[test['damage'] == 0, 'damage_type'] = 'none'
        train['damage_type'].fillna('minor', inplace=True)
        test['damage_type'].fillna('minor', inplace=True)
    
    # Fill other missing values with advanced strategies
    for col in train.columns:
        if train[col].isnull().any():
            if train[col].dtype in ['float64', 'int64']:
                # Use different strategies based on distribution
                skewness = train[col].skew()
                if abs(skewness) > 1:  # Highly skewed
                    fill_value = train[col].median()
                else:  # Normal-ish distribution
                    fill_value = train[col].mean()
            else:
                # For categorical, use mode with fallback
                mode_values = train[col].mode()
                fill_value = mode_values[0] if len(mode_values) > 0 else 'unknown'
            
            train[col].fillna(fill_value, inplace=True)
            if col in test.columns:
                test[col].fillna(fill_value, inplace=True)
    
    return train, test

# Apply preprocessing
train_df, test_df = advanced_preprocessing(train_df, test_df)

target_col = 'price'


def Ultra_High_Dataframe(train_df, test_df, ultra_brands):

    if submitting:
        ultra_train_mask = train_df['model'].isin(ultra_brands)
        ultra_test_mask = test_df['model'].isin(ultra_brands)
    
        ultra_train_df = train_df[ultra_train_mask]
        high_train_df = train_df[~ultra_train_mask]
    
        ultra_test_df = test_df[ultra_test_mask]
        high_test_df = test_df[~ultra_test_mask]
    
        ultra_train_price = ultra_train_df[['id', 'price']]
        high_train_price = high_train_df[['id', 'price']]
        
        ultra_test_price = None
        high_test_price = None
        
    else:
        ultra_train_mask = train_df['model'].isin(ultra_brands)
        ultra_test_mask = test_df['model'].isin(ultra_brands)
    
        ultra_train_df = train_df[ultra_train_mask]
        high_train_df = train_df[~ultra_train_mask]
    
        ultra_test_df = test_df[ultra_test_mask]
        high_test_df = test_df[~ultra_test_mask]
    
        ultra_train_price = ultra_train_df[['id', 'price']]
        high_train_price = high_train_df[['id', 'price']]
        
        ultra_test_price = ultra_test_df[['id', 'price']]
        high_test_price = high_test_df[['id', 'price']]

    return ultra_train_df, high_train_df, ultra_test_df, high_test_df, ultra_train_price, high_train_price, ultra_test_price, high_test_price

submitting = False
ultra_brands = ['Jesko', 'Chiron', 'Zonda', 'Huayra', 'Veyron', 'Agera', 'Regera']

ultra_train_df, high_train_df, ultra_test_df, high_test_df, ultra_train_price, high_train_price, \
ultra_test_price, high_test_price = Ultra_High_Dataframe(train_df, test_df, ultra_brands)


print("\n3. Advanced Feature Engineering...")

class UltimateFeatureEngineer:
    """Ultimate feature engineering with multiple strategies"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
        self.embedding_dims = {}
        
    def create_statistical_features(self, df, numeric_cols):
        """Create statistical aggregation features"""
        if len(numeric_cols) > 3:
            df['row_mean'] = df[numeric_cols].mean(axis=1)
            df['row_std'] = df[numeric_cols].std(axis=1)
            df['row_max'] = df[numeric_cols].max(axis=1)
            df['row_min'] = df[numeric_cols].min(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            df['row_kurtosis'] = df[numeric_cols].kurtosis(axis=1)
            df['row_skew'] = df[numeric_cols].skew(axis=1)
            df['row_median'] = df[numeric_cols].median(axis=1)
            df['row_q25'] = df[numeric_cols].quantile(0.25, axis=1)
            df['row_q75'] = df[numeric_cols].quantile(0.75, axis=1)
            df['row_iqr'] = df['row_q75'] - df['row_q25']
        return df
    
    def create_polynomial_features(self, df, important_features, degree=2):
        """Create polynomial and interaction features"""
        if len(important_features) > 0:
            # Create more interaction features
            for i in range(len(important_features)):
                for j in range(i+1, len(important_features)):
                    col1, col2 = important_features[i], important_features[j]
                    if col1 in df.columns and col2 in df.columns:
                        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1)
                        df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                        df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                    
        return df
    
    def create_cluster_features(self, df, features):
        """Create cluster-based features"""
        if len(features) > 5:
            # KMeans clustering
            for n_clusters in [3, 5, 7]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df[f'kmeans_{n_clusters}'] = kmeans.fit_predict(df[features[:10]])
            
            # DBSCAN for outlier detection
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            df['dbscan_cluster'] = dbscan.fit_predict(StandardScaler().fit_transform(df[features[:10]]))
            df['is_outlier'] = (df['dbscan_cluster'] == -1).astype(int)
                
        return df
    
    def create_advanced_features(self, df, train_df=None):
        """Create all advanced features"""
        df = df.copy()
        ref_df = train_df if train_df is not None else df
        
        # Performance metrics
        if 'horsepower' in df.columns and 'weight_kg' in df.columns:
            df['power_to_weight'] = df['horsepower'] / (df['weight_kg'] + 1)
            df['power_density'] = df['horsepower'] / (df['weight_kg'] / 1000)
            df['power_weight_ratio_log'] = np.log1p(df['power_to_weight'])
            df['power_weight_ratio_sqrt'] = np.sqrt(df['power_to_weight'])
            df['power_weight_ratio_squared'] = df['power_to_weight'] ** 2
            
        if 'torque' in df.columns and 'weight_kg' in df.columns:
            df['torque_to_weight'] = df['torque'] / (df['weight_kg'] + 1)
            df['torque_per_hp'] = df['torque'] / (df['horsepower'] + 1) if 'horsepower' in df.columns else 0
            df['torque_hp_ratio'] = df['torque'] / (df['horsepower'] + 1) if 'horsepower' in df.columns else 0
            
        # Advanced performance scores
        if all(col in df.columns for col in ['horsepower', 'top_speed_mph', 'zero_to_60_s']):
            df['performance_score'] = (df['horsepower'] * df['top_speed_mph']) / (df['zero_to_60_s'] + 0.1)
            df['acceleration_efficiency'] = df['horsepower'] / (df['zero_to_60_s'] + 0.1)
            df['speed_efficiency'] = df['top_speed_mph'] / df['horsepower']
            df['performance_index'] = np.log1p(df['performance_score'])
            df['performance_score_sqrt'] = np.sqrt(df['performance_score'])
            df['performance_score_squared'] = df['performance_score'] ** 2
            
            # Advanced performance ratios
            df['speed_acceleration_ratio'] = df['top_speed_mph'] / (df['zero_to_60_s'] + 0.1)
            df['power_speed_acceleration'] = df['horsepower'] * df['top_speed_mph'] * df['zero_to_60_s']
            
            # Performance categories
            df['is_hypercar'] = ((df['top_speed_mph'] > 200) & (df['zero_to_60_s'] < 3)).astype(int)
            df['is_superfast'] = (df['zero_to_60_s'] < 3.5).astype(int)
            df['is_high_speed'] = (df['top_speed_mph'] > 180).astype(int)
            df['is_quick'] = (df['zero_to_60_s'] < 4).astype(int)
            
        # Age and depreciation features
        if 'year' in df.columns:
            df['age'] = 2025 - df['year']
            df['age_squared'] = df['age'] ** 2
            df['age_cubed'] = df['age'] ** 3
            df['age_sqrt'] = np.sqrt(df['age'])
            df['age_log'] = np.log1p(df['age'])
            df['age_exp'] = np.exp(-df['age'] / 10)
            
            # Advanced depreciation models
            df['linear_depreciation'] = np.clip(1 - (df['age'] * 0.1), 0, 1)
            df['exponential_depreciation'] = np.exp(-df['age'] * 0.15)
            df['logarithmic_depreciation'] = 1 / (1 + np.log1p(df['age']))
            df['sigmoid_depreciation'] = 1 / (1 + np.exp(df['age'] - 5))
            df['power_depreciation'] = 1 / (1 + df['age'] ** 1.5)
            
            # Age categories
            df['is_new'] = (df['age'] <= 1).astype(int)
            df['is_nearly_new'] = (df['age'] <= 2).astype(int)
            df['is_used'] = ((df['age'] > 2) & (df['age'] <= 5)).astype(int)
            df['is_old'] = (df['age'] > 5).astype(int)
            df['is_vintage'] = (df['age'] > 10).astype(int)
            
        # Mileage features
        if 'mileage' in df.columns and 'age' in df.columns:
            df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
            df['low_mileage'] = (df['mileage_per_year'] < 1000).astype(int)
            df['high_mileage'] = (df['mileage_per_year'] > 5000).astype(int)
            df['very_high_mileage'] = (df['mileage_per_year'] > 10000).astype(int)
            
            # Mileage transformations
            df['mileage_log'] = np.log1p(df['mileage'])
            df['mileage_sqrt'] = np.sqrt(df['mileage'])
            df['mileage_squared'] = df['mileage'] ** 2
            df['mileage_per_hp'] = df['mileage'] / (df['horsepower'] + 1) if 'horsepower' in df.columns else 0
            
            # Mileage categories
            df['mileage_category'] = pd.cut(df['mileage_per_year'], 
                                           bins=[0, 1000, 3000, 5000, 10000, float('inf')],
                                           labels=['very_low', 'low', 'medium', 'high', 'very_high'])
            
        # Advanced condition score
        condition_components = []
        
        if 'damage' in df.columns:
            df['no_damage_score'] = (1 - df['damage']) * 50
            condition_components.append(df['no_damage_score'])
            
        if 'has_warranty' in df.columns:
            df['warranty_score'] = df['has_warranty'] * 30
            condition_components.append(df['warranty_score'])
            
        if 'warranty_years' in df.columns:
            df['warranty_years_score'] = df['warranty_years'] * 10
            df['warranty_years_squared'] = df['warranty_years'] ** 2
            condition_components.append(df['warranty_years_score'])
            
        if 'non_original_parts' in df.columns:
            df['original_parts_score'] = (df['non_original_parts'] == 0).astype(int) * 25
            condition_components.append(df['original_parts_score'])
            
        if 'service_history' in df.columns:
            service_map = {'authorized': 15, 'independent': 8, 'none': 0}
            df['service_history_score'] = df['service_history'].map(service_map).fillna(0)
            condition_components.append(df['service_history_score'])
            
        if condition_components:
            df['condition_score'] = sum(condition_components)
            df['condition_score_squared'] = df['condition_score'] ** 2
            df['condition_score_log'] = np.log1p(df['condition_score'])
            df['condition_score_sqrt'] = np.sqrt(df['condition_score'])
            
        # Luxury and exclusivity features
        luxury_components = []
        
        if 'limited_edition' in df.columns:
            df['limited_score'] = df['limited_edition'] * 5
            luxury_components.append(df['limited_score'])
            
        if 'carbon_fiber_body' in df.columns:
            df['carbon_score'] = df['carbon_fiber_body'] * 4
            luxury_components.append(df['carbon_score'])
            
        if 'aero_package' in df.columns:
            df['aero_score'] = df['aero_package'] * 3
            luxury_components.append(df['aero_score'])
            
        if 'brake_type' in df.columns:
            df['ceramic_brakes_score'] = (df['brake_type'] == 'carbon-ceramic').astype(int) * 3
            luxury_components.append(df['ceramic_brakes_score'])
            
        if luxury_components:
            df['luxury_score'] = sum(luxury_components)
            df['ultra_luxury'] = (df['luxury_score'] >= 10).astype(int)
            df['luxury_score_squared'] = df['luxury_score'] ** 2
            df['luxury_score_log'] = np.log1p(df['luxury_score'])
            
        # Brand and model statistics
        if 'brand' in df.columns and target_col in ref_df.columns:
            # Brand statistics
            brand_stats = ref_df.groupby('brand')[target_col].agg([
                'mean', 'std', 'median', 'min', 'max', 'count', 'skew', 'sem'
            ])
            
            df['brand_avg_price'] = df['brand'].map(brand_stats['mean'])
            df['brand_std_price'] = df['brand'].map(brand_stats['std'])
            df['brand_median_price'] = df['brand'].map(brand_stats['median'])
            df['brand_min_price'] = df['brand'].map(brand_stats['min'])
            df['brand_max_price'] = df['brand'].map(brand_stats['max'])
            df['brand_count'] = df['brand'].map(brand_stats['count'])
            df['brand_price_range'] = df['brand_max_price'] - df['brand_min_price']
            df['brand_price_skew'] = df['brand'].map(brand_stats['skew'])
            df['brand_price_sem'] = df['brand'].map(brand_stats['sem'])
            
            # Fill missing values
            for col in ['brand_avg_price', 'brand_median_price', 'brand_min_price', 'brand_max_price']:
                df[col].fillna(ref_df[target_col].mean(), inplace=True)
            df['brand_std_price'].fillna(ref_df[target_col].std(), inplace=True)
            df['brand_count'].fillna(1, inplace=True)
            df['brand_price_skew'].fillna(0, inplace=True)
            df['brand_price_sem'].fillna(ref_df[target_col].sem(), inplace=True)
            
            # Brand premium indicator
            avg_price = ref_df[target_col].mean()
            df['is_premium_brand'] = (df['brand_avg_price'] > avg_price * 1.5).astype(int)
            df['is_budget_brand'] = (df['brand_avg_price'] < avg_price * 0.5).astype(int)
            df['is_luxury_brand'] = (df['brand_avg_price'] > avg_price * 2).astype(int)
            
        # Model statistics
        if 'model' in df.columns and target_col in ref_df.columns:
            model_stats = ref_df.groupby('model')[target_col].agg(['mean', 'count', 'std', 'min', 'max'])
            
            df['model_avg_price'] = df['model'].map(model_stats['mean'])
            df['model_avg_price'].fillna(ref_df[target_col].mean(), inplace=True)
                
            df['model_count'] = df['model'].map(model_stats['count']).fillna(1)
            df['model_rarity'] = 1 / df['model_count']
            df['model_rarity_log'] = np.log1p(df['model_rarity'])
            df['is_rare_model'] = (df['model_count'] <= 3).astype(int)
            df['is_common_model'] = (df['model_count'] >= 20).astype(int)
            df['is_unique_model'] = (df['model_count'] == 1).astype(int)
            
            df['model_std_price'] = df['model'].map(model_stats['std']).fillna(0)
            df['model_min_price'] = df['model'].map(model_stats['min']).fillna(ref_df[target_col].min())
            df['model_max_price'] = df['model'].map(model_stats['max']).fillna(ref_df[target_col].max())
            df['model_price_range'] = df['model_max_price'] - df['model_min_price']
            
        # Engine type features
        if 'engine_config' in df.columns:
            df['is_electric'] = (df['engine_config'] == 'Electric').astype(int)
            df['is_hybrid'] = (df['engine_config'] == 'Hybrid').astype(int)
            df['is_v12'] = (df['engine_config'] == 'V12').astype(int)
            df['is_v10'] = (df['engine_config'] == 'V10').astype(int)
            df['is_v8'] = (df['engine_config'] == 'V8').astype(int)
            df['is_v6'] = (df['engine_config'] == 'V6').astype(int)
            df['is_traditional'] = (~df['engine_config'].isin(['Electric', 'Hybrid'])).astype(int)
            
            # Engine complexity score
            engine_complexity = {'V12': 5, 'V10': 4, 'V8': 3, 'V6': 2, 'Hybrid': 4, 'Electric': 3}
            df['engine_complexity'] = df['engine_config'].map(engine_complexity).fillna(1)
            
        # Complex interactions
        if 'luxury_score' in df.columns and 'condition_score' in df.columns:
            df['quality_index'] = df['luxury_score'] * df['condition_score']
            df['quality_index_log'] = np.log1p(df['quality_index'])
            df['quality_index_sqrt'] = np.sqrt(df['quality_index'])
            df['quality_per_age'] = df['quality_index'] / (df['age'] + 1) if 'age' in df.columns else df['quality_index']
            
        if 'performance_score' in df.columns and 'age' in df.columns:
            df['performance_retention'] = df['performance_score'] / (df['age'] + 1)
            df['performance_depreciation'] = df['performance_score'] * df['exponential_depreciation'] if 'exponential_depreciation' in df.columns else df['performance_score']
            
        if 'horsepower' in df.columns and 'luxury_score' in df.columns:
            df['hp_luxury_ratio'] = df['horsepower'] * df['luxury_score']
            df['hp_per_luxury'] = df['horsepower'] / (df['luxury_score'] + 1)
            
        # Market positioning features
        if all(col in df.columns for col in ['performance_score', 'luxury_score', 'condition_score']):
            df['market_position'] = df['performance_score'] * 0.4 + df['luxury_score'] * 0.3 + df['condition_score'] * 0.3
            df['market_position_log'] = np.log1p(df['market_position'])
            
        # Color features
        if 'color' in df.columns:
            # Color popularity
            color_counts = ref_df['color'].value_counts()
            df['color_popularity'] = df['color'].map(color_counts).fillna(1)
            df['color_rarity'] = 1 / df['color_popularity']
            df['is_rare_color'] = (df['color_popularity'] <= 10).astype(int)
            
            # Color categories
            premium_colors = ['black', 'silver', 'white', 'grey']
            exotic_colors = ['orange', 'yellow', 'green', 'purple', 'lime']
            classic_colors = ['red', 'blue', 'black', 'white']
            
            df['has_premium_color'] = df['color'].str.lower().isin(premium_colors).astype(int)
            df['has_exotic_color'] = df['color'].str.lower().isin(exotic_colors).astype(int)
            df['has_classic_color'] = df['color'].str.lower().isin(classic_colors).astype(int)
            
        return df

# Initialize feature engineer
feature_engineer = UltimateFeatureEngineer()

# Apply feature engineering
ultra_train_df = feature_engineer.create_advanced_features(ultra_train_df)
high_train_df = feature_engineer.create_advanced_features(high_train_df)
ultra_test_df = feature_engineer.create_advanced_features(ultra_test_df, ultra_train_df)
high_test_df = feature_engineer.create_advanced_features(high_test_df, high_train_df)

# Create additional statistical features
numeric_cols = ultra_train_df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
    
ultra_train_df = feature_engineer.create_statistical_features(ultra_train_df, numeric_cols[:20])
high_train_df = feature_engineer.create_statistical_features(high_train_df, numeric_cols[:20])
ultra_test_df = feature_engineer.create_statistical_features(ultra_test_df, numeric_cols[:20])
high_test_df = feature_engineer.create_statistical_features(high_test_df, numeric_cols[:20])

# Create polynomial features for important numeric features
important_features = ['horsepower', 'torque', 'top_speed_mph', 'age', 'mileage', 'performance_score', 'condition_score']
important_features = [f for f in important_features if f in ultra_train_df.columns]

ultra_train_df = feature_engineer.create_polynomial_features(ultra_train_df, important_features[:5])
high_train_df = feature_engineer.create_polynomial_features(high_train_df, important_features[:5])
ultra_test_df = feature_engineer.create_polynomial_features(ultra_test_df, important_features[:5])
high_test_df = feature_engineer.create_polynomial_features(high_test_df, important_features[:5])

# Create cluster features
numeric_features = [col for col in train_df.columns if train_df[col].dtype in ['float64', 'int64'] and col != target_col]
if len(numeric_features) > 10:
    ultra_train_df = feature_engineer.create_cluster_features(ultra_train_df, numeric_features[:15])
    high_train_df = feature_engineer.create_cluster_features(high_train_df, numeric_features[:15])
    ultra_test_df = feature_engineer.create_cluster_features(ultra_test_df, numeric_features[:15])
    high_test_df = feature_engineer.create_cluster_features(high_test_df, numeric_features[:15])


# =====================================================
# 4. ADVANCED ENCODING - FIXED VERSION
# =====================================================
print("\n4. Advanced Encoding...")

def process_dataset_encoding(train_df, test_df, target_col, dataset_name=""):
    """Process encoding for a single dataset pair"""
    print(f"Processing {dataset_name} dataset...")
    
    # Make copies to avoid modifying original data
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Target encoding with cross-validation
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in ['ID', 'id', target_col]]
    
    # Label encoding for high cardinality features
    label_encoders = {}
    for col in ['brand', 'model']:
        if col in cat_cols:
            le = LabelEncoder()
            # Fit on all unique values from both train and test
            all_values = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
            le.fit(all_values)
            
            train_df[f'{col}_label'] = le.transform(train_df[col].astype(str))
            test_df[f'{col}_label'] = le.transform(test_df[col].astype(str))
            label_encoders[col] = le
            
            # Store embedding dimensions for neural networks
            if 'feature_engineer' in globals():
                feature_engineer.embedding_dims[col] = len(all_values)
    
    # Target encoding with smoothing and cross-validation
    from sklearn.model_selection import KFold
    
    def target_encode_cv(train_df, test_df, col, target_col, n_splits=5, smoothing=1.0):
        """Target encoding with cross-validation to prevent overfitting"""
        if target_col not in train_df.columns:
            print(f"Warning: target column '{target_col}' not found in training data")
            return np.zeros(len(train_df)), np.zeros(len(test_df))
            
        train_encoded = np.zeros(len(train_df))
        
        # Calculate global mean for fallback
        global_mean = train_df[target_col].mean()
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
            
            # Calculate mean target for each category with smoothing
            category_counts = X_tr.groupby(col).size()
            category_means = X_tr.groupby(col)[target_col].mean()
            
            # Apply smoothing: (count * category_mean + smoothing * global_mean) / (count + smoothing)
            smoothed_means = (category_counts * category_means + smoothing * global_mean) / (category_counts + smoothing)
            
            # Map to validation fold
            mapped_values = X_val[col].map(smoothed_means)
            train_encoded[val_idx] = mapped_values.fillna(global_mean)
        
        # For test set, use overall smoothed means
        overall_counts = train_df.groupby(col).size()
        overall_means = train_df.groupby(col)[target_col].mean()
        overall_smoothed = (overall_counts * overall_means + smoothing * global_mean) / (overall_counts + smoothing)
        
        test_encoded = test_df[col].map(overall_smoothed)
        
        # Fill any remaining NaN values with global mean
        train_encoded = pd.Series(train_encoded).fillna(global_mean)
        test_encoded = test_encoded.fillna(global_mean)
        
        print(f"  Target encoding for {col}: train NaNs = {pd.Series(train_encoded).isna().sum()}, test NaNs = {test_encoded.isna().sum()}")
        
        return train_encoded.values, test_encoded.values
    
    # Apply target encoding with CV - only if target column exists
    if target_col in train_df.columns:
        for col in ['brand', 'model']:
            if col in cat_cols:
                print(f"  Applying target encoding to {col}...")
                train_encoded, test_encoded = target_encode_cv(train_df, test_df, col, target_col)
                train_df[f'{col}_target_cv'] = train_encoded
                test_df[f'{col}_target_cv'] = test_encoded
    else:
        print(f"Warning: Target column '{target_col}' not found. Skipping target encoding.")
    
    # One-hot encoding for remaining categorical features
    remaining_cats = [col for col in cat_cols if col not in ['brand', 'model', 'mileage_category']]
    if remaining_cats:
        print(f"  One-hot encoding: {remaining_cats}")
        train_df = pd.get_dummies(train_df, columns=remaining_cats, drop_first=True, dummy_na=False)
        test_df = pd.get_dummies(test_df, columns=remaining_cats, drop_first=True, dummy_na=False)
    
    # Handle mileage_category if it exists
    if 'mileage_category' in train_df.columns:
        print("  One-hot encoding mileage_category...")
        train_df = pd.get_dummies(train_df, columns=['mileage_category'], prefix='mileage_cat', drop_first=True)
        test_df = pd.get_dummies(test_df, columns=['mileage_category'], prefix='mileage_cat', drop_first=True)
    
    # Align columns between train and test
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # Add missing columns to test set
    for col in train_cols - test_cols:
        if col != target_col:  # Don't add target column to test set
            test_df[col] = 0
    
    # Remove extra columns from test set
    for col in test_cols - train_cols:
        test_df = test_df.drop(col, axis=1)
    
    # Ensure column order matches
    if target_col in train_df.columns:
        common_cols = [col for col in train_df.columns if col != target_col]
        test_df = test_df[common_cols]
    
    return train_df, test_df, label_encoders

# Usage example - you need to specify the target column name
# Replace 'price' with your actual target column name
TARGET_COLUMN = 'price'  # Change this to your actual target column

# Check for NaN values before encoding
print("Checking for NaN values before encoding...")
print("Ultra train NaNs:", ultra_train_df.isnull().sum().sum())
print("Ultra test NaNs:", ultra_test_df.isnull().sum().sum())
print("High train NaNs:", high_train_df.isnull().sum().sum())
print("High test NaNs:", high_test_df.isnull().sum().sum())

# Process ultra dataset
print("Processing Ultra dataset...")
ultra_train_df, ultra_test_df, ultra_label_encoders = process_dataset_encoding(
    ultra_train_df, ultra_test_df, TARGET_COLUMN, "Ultra"
)

# Process high dataset
print("Processing High dataset...")
high_train_df, high_test_df, high_label_encoders = process_dataset_encoding(
    high_train_df, high_test_df, TARGET_COLUMN, "High"
)

print("Advanced encoding completed for both datasets!")
print(f"Ultra train shape: {ultra_train_df.shape}")
print(f"Ultra test shape: {ultra_test_df.shape}")
print(f"High train shape: {high_train_df.shape}")
print(f"High test shape: {high_test_df.shape}")

# Check for NaN values after encoding
print("\nChecking for NaN values after encoding...")
print("Ultra train NaNs:", ultra_train_df.isnull().sum().sum())
print("Ultra test NaNs:", ultra_test_df.isnull().sum().sum())
print("High train NaNs:", high_train_df.isnull().sum().sum())
print("High test NaNs:", high_test_df.isnull().sum().sum())

# Check specific target encoding columns
target_encoding_cols = [col for col in ultra_train_df.columns if col.endswith('_target_cv')]
if target_encoding_cols:
    print(f"\nTarget encoding columns: {target_encoding_cols}")
    for col in target_encoding_cols:
        if col in ultra_train_df.columns:
            print(f"Ultra {col} NaNs: train={ultra_train_df[col].isnull().sum()}, test={ultra_test_df[col].isnull().sum()}")
        if col in high_train_df.columns:
            print(f"High {col} NaNs: train={high_train_df[col].isnull().sum()}, test={high_test_df[col].isnull().sum()}")

# Store encoders for later use if needed
encoders = {
    'ultra': ultra_label_encoders,
    'high': high_label_encoders
}


from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import pandas as pd
import numpy as np

def multi_strategy_feature_selection(train_df, test_df, target_col, cat_cols, dataset_label="dataset", top_k=60):
    print(f"\nFeature Selection for {dataset_label}...")

    # Get numeric feature columns (exclude IDs, target, categoricals)
    feature_cols = [col for col in train_df.columns
                    if col not in ['ID', 'id', 'price', target_col] + cat_cols
                    and train_df[col].dtype in ['float64', 'int64', 'uint8', 'float32', 'int32']]

    # Remove constant features
    constant_features = [col for col in feature_cols if train_df[col].std() == 0]
    feature_cols = [col for col in feature_cols if col not in constant_features]

    print(f"  Removed {len(constant_features)} constant features")
    print(f"  Remaining features: {len(feature_cols)}")

    # Prepare arrays
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    X_test = test_df[feature_cols].values

    # Initialize score DataFrame
    selection_scores = pd.DataFrame(index=feature_cols)

    # 1. Mutual Information
    print("  Calculating mutual information...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    selection_scores['mi_score'] = mi_scores

    # 2. Random Forest Importance
    print("  Calculating RF importance...")
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X, y)
    selection_scores['rf_importance'] = rf_selector.feature_importances_

    # 3. XGBoost Importance
    print("  Calculating XGBoost importance...")
    xgb_selector = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb_selector.fit(X, y)
    selection_scores['xgb_importance'] = xgb_selector.feature_importances_

    # 4. LASSO Coefficients
    print("  Calculating LASSO coefficients...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso_selector = Lasso(alpha=100, random_state=42)
    lasso_selector.fit(X_scaled, y)
    selection_scores['lasso_coef'] = np.abs(lasso_selector.coef_)

    # 5. Permutation Importance (on a sample)
    print("  Calculating permutation importance...")
    perm_importance = permutation_importance(rf_selector, X[:1000], y[:1000], n_repeats=10, random_state=42)
    selection_scores['perm_importance'] = perm_importance.importances_mean

    # Rank and aggregate
    for col in ['mi_score', 'rf_importance', 'xgb_importance', 'lasso_coef', 'perm_importance']:
        selection_scores[f'rank_{col}'] = selection_scores[col].rank(ascending=False)

    selection_scores['avg_rank'] = selection_scores[[f'rank_{col}' for col in 
                                                     ['mi_score', 'rf_importance', 'xgb_importance', 'lasso_coef', 'perm_importance']]].mean(axis=1)

    # Sort by rank and select top features
    selection_scores = selection_scores.sort_values('avg_rank')
    selected_features = selection_scores.head(top_k).index.tolist()

    print(f"\nSelected {len(selected_features)} features")
    print(f"Top 10: {selected_features[:10]}")

    # Filter train/test data
    train_selected = train_df[selected_features].copy()
    test_selected = test_df[selected_features].copy()

    return selected_features, train_selected, test_selected, selection_scores
    
cat_cols = ultra_train_df.select_dtypes(include=['object', 'category']).columns.tolist()

# For ultra
ultra_selected, ultra_train_df, ultra_test_df, ultra_scores = multi_strategy_feature_selection(
    train_df=ultra_train_df,
    test_df=ultra_test_df,
    target_col='price',
    cat_cols=cat_cols,
    dataset_label="Ultra",
    top_k=60
)

cat_cols = high_train_df.select_dtypes(include=['object', 'category']).columns.tolist()

# For high
high_selected, high_train_df, high_test_df, high_scores = multi_strategy_feature_selection(
    train_df=high_train_df,
    test_df=high_test_df,
    target_col='price',
    cat_cols=cat_cols,
    dataset_label="High",
    top_k=60
)


high_scores


from sklearn.dummy import DummyRegressor
model = DummyRegressor(strategy='mean')
model.fit(ultra_train_df, ultra_train_price['price'])
print("Baseline Test RÂ²:", model.score(ultra_test_df, ultra_test_price['price']))



ultra_train_df.describe().T.sort_values("std")


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

def detect_data_issues(X_train, y_train, X_test, y_test):
    """Enhanced data quality analysis"""
    print("=== DATA QUALITY ANALYSIS ===")
    
    # Check target distribution
    print(f"Training target - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"Training target - Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
    print(f"Test target - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    print(f"Test target - Min: {y_test.min():.2f}, Max: {y_test.max():.2f}")
    
    # Check for skewness
    from scipy import stats
    train_skew = stats.skew(y_train)
    test_skew = stats.skew(y_test)
    print(f"Target skewness - Train: {train_skew:.3f}, Test: {test_skew:.3f}")
    
    if abs(train_skew) > 2:
        print("âš ï¸�  WARNING: Highly skewed target variable detected!")
        print("ğŸ’¡ Consider log transformation or using models robust to skewness")
    
    # Check for outliers using IQR
    Q1_train, Q3_train = np.percentile(y_train, [25, 75])
    IQR_train = Q3_train - Q1_train
    outliers_train = len(y_train[(y_train < Q1_train - 1.5*IQR_train) | 
                                 (y_train > Q3_train + 1.5*IQR_train)])
    print(f"Training outliers (IQR method): {outliers_train} ({outliers_train/len(y_train)*100:.1f}%)")
    
    print("="*50)

def smart_preprocessing(X_train, y_train, X_val, X_test, use_log_transform=True):
    """Smarter preprocessing for non-linear data"""
    print("=== SMART PREPROCESSING ===")
    original_features = X_train.shape[1]
    
    # Remove constant features
    feature_variance = X_train.var()
    non_constant_features = feature_variance[feature_variance > 1e-8].index
    X_train = X_train[non_constant_features]
    X_val = X_val[non_constant_features] 
    X_test = X_test[non_constant_features]
    print(f"Removed {original_features - len(non_constant_features)} constant features")
    
    # More intelligent correlation removal - keep some redundancy for tree models
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Higher threshold - tree models can handle some correlation
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    
    remaining_features = [col for col in X_train.columns if col not in high_corr_features]
    X_train = X_train[remaining_features]
    X_val = X_val[remaining_features]
    X_test = X_test[remaining_features]
    print(f"Removed {len(high_corr_features)} extremely correlated features (>0.95)")
    
    # More generous feature limit for tree models
    n_samples = len(X_train)
    max_features = min(50, max(10, n_samples // 10))  # More features allowed
    
    if X_train.shape[1] > max_features:
        # Use mutual information or tree-based importance for feature selection
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X_train.columns)
        top_features = mi_scores.nlargest(max_features).index
        
        X_train = X_train[top_features]
        X_val = X_val[top_features]
        X_test = X_test[top_features]
        print(f"Selected top {max_features} features by mutual information")
    
    # Optional log transformation for highly skewed targets
    y_train_transformed = y_train.copy()
    y_val_transformed = y_train.copy() if hasattr(y_train, 'copy') else y_train
    transform_applied = False
    
    if use_log_transform:
        from scipy import stats
        if stats.skew(y_train) > 2 and y_train.min() > 0:
            y_train_transformed = np.log1p(y_train)
            transform_applied = True
            print("Applied log1p transformation to target variable")
    
    print(f"Final feature count: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train_transformed, transform_applied

def build_smart_tree_model(X_train, y_train, X_test, y_test, model_name='RandomForest', 
                          n_trials=50, random_state=42, use_log_transform=True):
    """Smart tree model building with better hyperparameter ranges"""
    
    # Larger validation split for better generalization estimation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state
    )
    
    # Use RobustScaler for features (helps with outliers)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_split), 
        columns=X_train_split.columns, 
        index=X_train_split.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X_val.columns, 
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # Smart preprocessing
    X_train_final, X_val_final, X_test_final, y_train_transformed, log_transform = smart_preprocessing(
        X_train_scaled, y_train_split, X_val_scaled, X_test_scaled, use_log_transform
    )
    
    n_samples = len(X_train_final)
    n_features = X_train_final.shape[1]
    
    def objective(trial):

        max_features_opts = min(n_features, max(3, int(n_features * 0.7)))
        
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 10, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 4, max(10, max(10, n_samples // 50))),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, max(10, n_samples // 100)),
                'bootstrap': True,
                'oob_score': False,
                'random_state': random_state,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            
        elif model_name == 'ExtraTrees':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 4, 18),
                'min_samples_split': trial.suggest_int('min_samples_split', 4, max(10, n_samples // 50)),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, max(10, n_samples // 100)),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', max_features_opts]),
                'bootstrap': False,
                'random_state': random_state,
                'n_jobs': -1
            }
            model = ExtraTreesRegressor(**params)
     
        elif model_name == 'GradientBoosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 4, max(20, n_samples // 50)),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, max(10, n_samples // 100)),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', max_features_opts]),
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'random_state': random_state
            }
            model = GradientBoostingRegressor(**params)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),  # limits feature dominance
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),  # L2 regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True),    # L1 regularization
                'random_state': random_state
            }
            model = LGBMRegressor(**params)

            
        elif model_name == 'XGBoost':
            try:
                from xgboost import XGBRegressor
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                    'random_state': random_state,
                    'tree_method': 'hist',
                    'verbosity': 0
                }
                model = XGBRegressor(**params)
            except ImportError:
                print("XGBoost not available, skipping...")
                return float('inf')
            
        elif model_name == 'CatBoost':
            try:
                from catboost import CatBoostRegressor
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                    'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                    'random_state': random_state,
                    'verbose': 0
                }
                model = CatBoostRegressor(**params)
            except ImportError:
                print("CatBoost not available, skipping...")
                return float('inf')
        
        # Use 3-fold CV for better estimates while still being conservative
        kfold = KFold(n_splits=3, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train_final, y_train_transformed, 
                                   cv=kfold, scoring='neg_root_mean_squared_error')
        return -np.mean(cv_scores)
    
    # Optimize with more trials for better results
    print(f"Optimizing {model_name} with smart parameter ranges...")
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Build final model
    best_params = study.best_params
    
    if model_name == 'RandomForest':
        final_model = RandomForestRegressor(**best_params)
    elif model_name == 'ExtraTrees':
        final_model = ExtraTreesRegressor(**best_params)
    elif model_name == 'GradientBoosting':
        final_model = GradientBoostingRegressor(**best_params)
    elif model_name == 'LightGBM':
        final_model = LGBMRegressor(**best_params)
    elif model_name == 'XGBoost':
        from xgboost import XGBRegressor
        final_model = XGBRegressor(**best_params)
    elif model_name == 'CatBoost':
        from catboost import CatBoostRegressor
        final_model = CatBoostRegressor(**best_params)
    
    # Train and predict
    final_model.fit(X_train_final, y_train_transformed)
    
    y_train_pred = final_model.predict(X_train_final)
    y_val_pred = final_model.predict(X_val_final)
    y_test_pred = final_model.predict(X_test_final)
    
    # Transform predictions back if log transform was used
    if log_transform:
        y_train_pred = np.expm1(y_train_pred)
        y_val_pred = np.expm1(y_val_pred)
        y_test_pred = np.expm1(y_test_pred)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Handle MAPE
        y_true_nonzero = y_true[y_true != 0]
        y_pred_nonzero = y_pred[y_true != 0]
        
        if len(y_true_nonzero) > 0:
            mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
        else:
            mape = float('inf')
        
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae, 
            'mape': mape,
            'r2': r2
        }
    
    train_metrics = calculate_metrics(y_train_split, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Improved generalization analysis
    val_test_rmse_ratio = test_metrics['rmse'] / val_metrics['rmse'] if val_metrics['rmse'] > 0 else float('inf')
    val_test_r2_diff = val_metrics['r2'] - test_metrics['r2']
    
    # More nuanced generalization quality assessment
    if test_metrics['r2'] < 0:
        gen_quality = 'Failed'
    elif val_test_rmse_ratio > 2.0 or val_test_r2_diff > 0.3:
        gen_quality = 'Poor'
    elif val_test_rmse_ratio > 1.5 or val_test_r2_diff > 0.2:
        gen_quality = 'Fair'
    elif val_test_rmse_ratio > 1.2 or val_test_r2_diff > 0.1:
        gen_quality = 'Good'
    else:
        gen_quality = 'Excellent'
    
    # Feature importance (for tree models)
    feature_importance = None
    if hasattr(final_model, 'feature_importances_'):
        feature_importance = pd.Series(
            final_model.feature_importances_, 
            index=X_train_final.columns
        ).sort_values(ascending=False)
    
    results = {
        'model': final_model,
        'model_name': model_name,
        'best_params': best_params,
        'best_cv_score': study.best_value,
        'scaler': scaler,
        'log_transform_used': log_transform,
        'n_features_used': X_train_final.shape[1],
        'feature_importance': feature_importance,
        'training_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'generalization_analysis': {
            'val_test_rmse_ratio': val_test_rmse_ratio,
            'val_test_r2_diff': val_test_r2_diff,
            'generalization_quality': gen_quality
        }
    }
    
    return results

def compare_smart_tree_models(X_train, y_train, X_test, y_test, 
                             models_to_test=None, n_trials=50, random_state=42):
    """Compare tree-based models with smart tuning"""
    
    detect_data_issues(X_train, y_train, X_test, y_test)
    
    if models_to_test is None:
        # Focus on tree-based models that work well with non-linear data
        models_to_test = ['RandomForest', 'LightGBM', 'XGBoost', 'GradientBoosting', 'CatBoost']
    
    results_list = []
    
    print(f"Dataset info: Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Target range: {y_train.min():.2f} to {y_train.max():.2f}")
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_name} with SMART TUNING")
        print(f"{'='*60}")
        
        try:
            result = build_smart_tree_model(
                X_train, y_train, X_test, y_test, 
                model_name=model_name, 
                n_trials=n_trials,
                random_state=random_state
            )
            
            summary = {
                'Model': model_name,
                'Features_Used': result['n_features_used'],
                'CV_Score': result['best_cv_score'],
                'Train_RMSE': result['training_metrics']['rmse'],
                'Val_RMSE': result['validation_metrics']['rmse'],
                'Test_RMSE': result['test_metrics']['rmse'],
                'Train_R2': result['training_metrics']['r2'],
                'Val_R2': result['validation_metrics']['r2'],
                'Test_R2': result['test_metrics']['r2'],
                'Test_MAPE': result['test_metrics']['mape'],
                'RMSE_Ratio': result['generalization_analysis']['val_test_rmse_ratio'],
                'R2_Diff': result['generalization_analysis']['val_test_r2_diff'],
                'Generalization': result['generalization_analysis']['generalization_quality'],
                'Log_Transform': result['log_transform_used']
            }
            
            results_list.append(summary)
            
            # Print detailed results
            print(f"\n{model_name} Results:")
            print(f"Features used:        {result['n_features_used']}")
            print(f"Log transform used:   {result['log_transform_used']}")
            print(f"Cross-validation RMSE: {result['best_cv_score']:.2f}")
            print(f"Train RMSE:           {result['training_metrics']['rmse']:.2f}")
            print(f"Validation RMSE:      {result['validation_metrics']['rmse']:.2f}")
            print(f"Test RMSE:            {result['test_metrics']['rmse']:.2f}")
            print(f"Train RÂ²:             {result['training_metrics']['r2']:.4f}")
            print(f"Val RÂ²:               {result['validation_metrics']['r2']:.4f}")
            print(f"Test RÂ²:              {result['test_metrics']['r2']:.4f}")
            print(f"Test MAPE:            {result['test_metrics']['mape']:.2f}%")
            print(f"RMSE Ratio (T/V):     {result['generalization_analysis']['val_test_rmse_ratio']:.3f}")
            print(f"Generalization:       {result['generalization_analysis']['generalization_quality']}")
            
            # Show top feature importances
            if result['feature_importance'] is not None:
                print(f"\nTop 5 Important Features:")
                for i, (feature, importance) in enumerate(result['feature_importance'].head().items()):
                    print(f"  {i+1}. {feature}: {importance:.4f}")
            
            print(f"Best params: {result['best_params']}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    # Create results DataFrame
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # Sort by test RÂ² (descending) first, then by generalization quality
        results_df = results_df.sort_values(['Test_R2', 'RMSE_Ratio'], 
                                          ascending=[False, True]).reset_index(drop=True)
        results_df['Rank'] = range(1, len(results_df) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Model', 'Features_Used', 'Test_R2', 'Test_RMSE', 'Test_MAPE', 
                'Generalization', 'RMSE_Ratio', 'Val_R2', 'Log_Transform', 'CV_Score']
        results_df = results_df[cols]
        
        return results_df
    else:
        return pd.DataFrame()

def run_smart_tree_comparison(train_features, train_target, test_features, test_target, 
                             feature_set_name="Feature Set", n_trials=50, target_column=None):
    """Enhanced wrapper function"""
    
    # Clean data
    if isinstance(train_features, pd.DataFrame):
        train_features_clean = train_features.drop(['id'], axis=1, errors='ignore')
    else:
        train_features_clean = train_features
        
    if isinstance(test_features, pd.DataFrame):
        test_features_clean = test_features.drop(['id'], axis=1, errors='ignore')
    else:
        test_features_clean = test_features
    
    # Handle targets
    if isinstance(train_target, pd.DataFrame):
        if train_target.shape[1] == 1:
            train_target = train_target.iloc[:, 0]
        elif target_column and target_column in train_target.columns:
            train_target = train_target[target_column]
    
    if isinstance(test_target, pd.DataFrame):
        if test_target.shape[1] == 1:
            test_target = test_target.iloc[:, 0]
        elif target_column and target_column in test_target.columns:
            test_target = test_target[target_column]
    
    print(f"\n{'='*80}")
    print(f"SMART TREE-BASED MODELING: {feature_set_name}")
    print(f"{'='*80}")
    
    results = compare_smart_tree_models(
        train_features_clean, train_target, 
        test_features_clean, test_target, 
        n_trials=n_trials
    )
    
    print(f"\n{feature_set_name} FINAL RESULTS (Ranked by Test RÂ²):")
    print("="*80)
    print(results.to_string(index=False))
    
    return results

# Usage examples with your data:
if 'price' in ultra_train_df.columns:
    ultra_train_df.drop('price', axis=1, inplace=True)
if 'price' in ultra_test_df.columns:
    ultra_test_df.drop('price', axis=1, inplace=True)
if 'price' in high_train_df.columns:
    high_train_df.drop('price', axis=1, inplace=True)
if 'price' in high_test_df.columns:
    high_test_df.drop('price', axis=1, inplace=True)


# For your ultra feature set:
print("Testing ULTRA Feature Set with Smart Tree Models...")
ultra_results = run_smart_tree_comparison(
    ultra_train_df, ultra_train_price['price'], 
    ultra_test_df, ultra_test_price['price'],
    feature_set_name="Ultra Feature Set (Smart Trees)",
    n_trials=50  # More trials for better optimization
)

# For your high feature set:
print("\nTesting HIGH Feature Set with Smart Tree Models...")
high_results = run_smart_tree_comparison(
    high_train_df, high_train_price['price'],
    high_test_df, high_test_price['price'],
    feature_set_name="High Feature Set (Smart Trees)", 
    n_trials=50
)




