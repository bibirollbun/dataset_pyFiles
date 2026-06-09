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


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy import stats
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv')
train,val = train_test_split(df,test_size=0.2,random_state=42)
test_df = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv')



train.head()


train_23 = train[train['sale_year']==2023]


train_23


print(f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test_df.shape}")




# Global dictionary to store statistics
global_stats = {}



def advanced_feature_engineering(data, is_train=True):
    """Comprehensive feature engineering for house prices"""
    df = data.copy()
    
    # === TEMPORAL FEATURES ===
    # Create date features
    df['sale_date'] = pd.to_datetime(df['sale_year'].astype(str) + '-' + df['sale_month'].astype(str) + '-01')
    df['days_since_start'] = (df['sale_date'] - pd.Timestamp('2010-01-01')).dt.days
    df['quarter'] = df['sale_month'].apply(lambda x: (x-1)//3 + 1)
    df['is_spring'] = df['sale_month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['sale_month'].isin([6, 7, 8]).astype(int)
    df['is_autumn'] = df['sale_month'].isin([9, 10, 11]).astype(int)
    df['is_winter'] = df['sale_month'].isin([12, 1, 2]).astype(int)
    
    # Year-based features
    df['years_since_2010'] = df['sale_year'] - 2010
    df['is_recent'] = (df['sale_year'] >= 2020).astype(int)
    
    # === LOCATION FEATURES ===
    # Extract postcode district (first part)
    df['postcode_district'] = df['postcode'].str.extract(r'^([A-Z]+)')[0].fillna('UNKNOWN')
    df['postcode_area'] = df['postcode'].str.extract(r'^([A-Z]+\d+)')[0].fillna('UNKNOWN')
    
    # Radial distance from central London (approximate: 51.5074, -0.1278)
    central_lat, central_lon = 51.5074, -0.1278
    df['dist_from_center'] = np.sqrt(
        (df['latitude'] - central_lat)**2 + (df['longitude'] - central_lon)**2
    )
    
    # Location clusters
    df['lat_rounded'] = np.round(df['latitude'], 2)
    df['lon_rounded'] = np.round(df['longitude'], 2)
    df['location_cluster'] = df['lat_rounded'].astype(str) + '_' + df['lon_rounded'].astype(str)
    
    # Geographic quadrants
    df['is_north'] = (df['latitude'] > central_lat).astype(int)
    df['is_east'] = (df['longitude'] > central_lon).astype(int)
    df['quadrant'] = df['is_north'].astype(str) + '_' + df['is_east'].astype(str)
    
    # === PROPERTY FEATURES ===
    # Fill missing values intelligently
    df['bathrooms_filled'] = df['bathrooms'].fillna(df['bedrooms'] * 0.7)
    df['livingRooms_filled'] = df['livingRooms'].fillna(1)
    
    # Fill floor area by property type median
    for prop_type in df['propertyType'].unique():
        mask = (df['propertyType'] == prop_type) & (df['floorAreaSqM'].isna())
        median_val = df[df['propertyType'] == prop_type]['floorAreaSqM'].median()
        if pd.notna(median_val):
            df.loc[mask, 'floorAreaSqM'] = median_val
    
    # Fill any remaining with overall median
    df['floorAreaSqM'] = df['floorAreaSqM'].fillna(df['floorAreaSqM'].median())
    
    # Total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms_filled'] + df['livingRooms_filled']
    
    # Room ratios (add small constant to avoid division by zero)
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms_filled'] + 0.1)
    df['rooms_per_sqm'] = df['total_rooms'] / (df['floorAreaSqM'] + 1)
    df['sqm_per_bedroom'] = df['floorAreaSqM'] / (df['bedrooms'] + 1)
    df['sqm_per_room'] = df['floorAreaSqM'] / (df['total_rooms'] + 1)
    
    # Property size categories
    df['is_studio'] = (df['bedrooms'] <= 1).astype(int)
    df['is_small'] = (df['bedrooms'] == 2).astype(int)
    df['is_medium'] = (df['bedrooms'] == 3).astype(int)
    df['is_large'] = (df['bedrooms'] >= 4).astype(int)
    
    # Floor area categories (handle NaN properly)
    df['floor_area_category'] = pd.cut(
        df['floorAreaSqM'], 
        bins=[0, 50, 80, 120, 200, 1000],
        labels=['tiny', 'small', 'medium', 'large', 'xlarge'],
        include_lowest=True
    ).astype(str).fillna('unknown')
    
    # === ENERGY RATING ===
    energy_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    df['energy_rating_numeric'] = df['currentEnergyRating'].map(energy_map).fillna(0)
    df['has_energy_rating'] = (df['currentEnergyRating'].isin(energy_map.keys())).astype(int)
    
    # === TENURE ===
    df['is_freehold'] = (df['tenure'] == 'Freehold').astype(int)
    df['is_leasehold'] = (df['tenure'] == 'Leasehold').astype(int)
    
    # === PROPERTY TYPE ===
    df['is_flat'] = df['propertyType'].str.contains('Flat', case=False, na=False).astype(int)
    df['is_house'] = df['propertyType'].str.contains('House', case=False, na=False).astype(int)
    df['is_detached'] = df['propertyType'].str.contains('Detached', case=False, na=False).astype(int)
    df['is_terraced'] = df['propertyType'].str.contains('Terraced', case=False, na=False).astype(int)
    
    # === TARGET ENCODING (for train only) ===
    if is_train and 'price' in df.columns:
        # Location-based statistics
        global_stats['location_stats'] = df.groupby('location_cluster')['price'].agg(['mean', 'median', 'std']).reset_index()
        global_stats['location_stats'].columns = ['location_cluster', 'loc_price_mean', 'loc_price_median', 'loc_price_std']
        
        global_stats['outcode_stats'] = df.groupby('outcode')['price'].agg(['mean', 'median', 'std']).reset_index()
        global_stats['outcode_stats'].columns = ['outcode', 'outcode_price_mean', 'outcode_price_median', 'outcode_price_std']
        
        # Property type statistics
        global_stats['proptype_stats'] = df.groupby('propertyType')['price'].agg(['mean', 'median']).reset_index()
        global_stats['proptype_stats'].columns = ['propertyType', 'proptype_price_mean', 'proptype_price_median']
        
        # Time-based statistics
        global_stats['year_stats'] = df.groupby('sale_year')['price'].agg(['mean', 'median']).reset_index()
        global_stats['year_stats'].columns = ['sale_year', 'year_price_mean', 'year_price_median']
        
        # Merge
        df = df.merge(global_stats['location_stats'], on='location_cluster', how='left')
        df = df.merge(global_stats['outcode_stats'], on='outcode', how='left')
        df = df.merge(global_stats['proptype_stats'], on='propertyType', how='left')
        df = df.merge(global_stats['year_stats'], on='sale_year', how='left')
    
    elif not is_train:
        # Use pre-computed statistics for validation/test
        if 'location_stats' in global_stats:
            df = df.merge(global_stats['location_stats'], on='location_cluster', how='left')
            df = df.merge(global_stats['outcode_stats'], on='outcode', how='left')
            df = df.merge(global_stats['proptype_stats'], on='propertyType', how='left')
            df = df.merge(global_stats['year_stats'], on='sale_year', how='left')
    
    # Fill missing statistics with overall mean
    stat_cols = ['loc_price_mean', 'loc_price_median', 'loc_price_std',
                 'outcode_price_mean', 'outcode_price_median', 'outcode_price_std',
                 'proptype_price_mean', 'proptype_price_median',
                 'year_price_mean', 'year_price_median']
    
    for col in stat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # === INTERACTION FEATURES ===
    if 'loc_price_mean' in df.columns:
        df['price_per_sqm_expected'] = df['loc_price_mean'] / (df['floorAreaSqM'] + 1)
        df['price_per_room_expected'] = df['loc_price_mean'] / (df['total_rooms'] + 1)
    
    df['bedrooms_x_sqm'] = df['bedrooms'] * df['floorAreaSqM']
    df['dist_x_year'] = df['dist_from_center'] * df['sale_year']
    
    return df

# Apply feature engineering
print("Applying feature engineering...")
train_fe = advanced_feature_engineering(train, is_train=True)
val_fe = advanced_feature_engineering(val, is_train=False)
test_fe = advanced_feature_engineering(test_df, is_train=False)

print("Feature engineering complete!")


val_fe



# Select features
exclude_cols = ['ID', 'price', 'fullAddress', 'postcode', 'country', 'sale_date', 
                'bathrooms', 'livingRooms', 'lat_rounded', 'lon_rounded']
cat_features = ['propertyType', 'tenure', 'currentEnergyRating', 'postcode_district', 
                'postcode_area', 'location_cluster', 'quadrant', 'floor_area_category', 'outcode']




# Get all features
feature_cols = [col for col in train_fe.columns if col not in exclude_cols]
feature_cols = [col for col in feature_cols if train_fe[col].dtype in ['int64', 'float64', 'object', 'int32', 'float32']]




# Prepare data
X_train = train_fe[feature_cols].copy()
y_train = train_fe['price'].values
X_val = val_fe[feature_cols].copy()
y_val = val_fe['price'].values
X_test = test_fe[feature_cols].copy()



# Encode categorical features
le_dict = {}
for col in cat_features:
    if col in X_train.columns:
        le = LabelEncoder()
        # Fit on combined data to handle unseen categories
        combined = pd.concat([X_train[col].astype(str), X_val[col].astype(str), X_test[col].astype(str)])
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        le_dict[col] = le

print(f"\nNumber of features: {len(feature_cols)}")
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")



# ============= hyperparam tuning =============

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd



# Test configs around the "Aggressive" params that worked best
fine_tune_configs = [
    {
        'name': 'Aggressive (Current Best)',
        'params': {
            'learning_rate': 0.025,
            'max_depth': 8,
            'min_child_weight': 2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'gamma': 0,
        }
    },
    {
        'name': 'Slightly Higher LR',
        'params': {
            'learning_rate': 0.027,
            'max_depth': 8,
            'min_child_weight': 2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'gamma': 0,
        }
    },
    {
        'name': 'Deeper Trees',
        'params': {
            'learning_rate': 0.025,
            'max_depth': 9,
            'min_child_weight': 2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'gamma': 0,
        }
    },
    {
        'name': 'Lower Reg Alpha',
        'params': {
            'learning_rate': 0.025,
            'max_depth': 8,
            'min_child_weight': 2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.02,
            'reg_lambda': 0.5,
            'gamma': 0,
        }
    },
    {
        'name': 'More Subsample',
        'params': {
            'learning_rate': 0.025,
            'max_depth': 8,
            'min_child_weight': 2,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'gamma': 0,
        }
    },
    {
        'name': 'Min Child Weight = 1',
        'params': {
            'learning_rate': 0.025,
            'max_depth': 8,
            'min_child_weight': 1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'gamma': 0,
        }
    },
]

y_train_log = np.log1p(y_train)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_overall_mae = float('inf')
best_config = None
all_results = []

print("\n[Fine-tuning hyperparameters around current best]\n")

for config in fine_tune_configs:
    config_name = config['name']
    config_params = config['params']
    
    print(f"Testing: {config_name}")
    
    oof_pred = np.zeros(len(X_train))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_tr = y_train_log[train_idx]
        y_val_fold = y_train_log[val_idx]
        
        params = {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1,
            **config_params
        }
        
        model = xgb.XGBRegressor(**params, n_estimators=5000)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=100,
            verbose=0
        )
        
        oof_pred[val_idx] = model.predict(X_val_fold)
        fold_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(model.predict(X_val_fold)))
        fold_scores.append(fold_mae)
    
    cv_mae = mean_absolute_error(y_train, np.expm1(oof_pred))
    cv_std = np.std(fold_scores)
    
    print(f"  MAE = Â£{cv_mae:,.2f} (Â±Â£{cv_std:,.2f})")
    
    all_results.append({
        'config': config_name,
        'mae': cv_mae,
        'std': cv_std,
        'params': config_params
    })
    
    if cv_mae < best_overall_mae:
        best_overall_mae = cv_mae
        best_config = config
        print(f"  âœ“ NEW BEST!")
    
    print()

# ============= DISPLAY RESULTS =============
print("=" * 70)
print("FINE-TUNE RESULTS")
print("=" * 70)

results_df = pd.DataFrame(all_results).sort_values('mae')
print("\n" + results_df[['config', 'mae', 'std']].to_string(index=False))

print(f"\nğŸ�† BEST CONFIG: {best_config['name']}")
print(f"   MAE: Â£{best_overall_mae:,.2f}")
print(f"\nTarget (current Kaggle): Â£164,280")
if best_overall_mae < 140200:  # Target roughly translates to this CV MAE
    print(f"âœ“ Expected improvement!")
else:
    print(f"Note: May need multiple tries or ensemble")

# ============= TRAIN FINAL MODEL WITH BEST CONFIG =============
print("\n" + "=" * 70)
print("TRAINING FINAL MODEL WITH BEST CONFIG")
print("=" * 70)

best_params = best_config['params'].copy()
best_params.update({
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae',
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
})

print(f"\n[Best Params]:")
for k, v in best_params.items():
    if k not in ['objective', 'eval_metric', 'random_state', 'tree_method', 'n_jobs']:
        print(f"  {k}: {v}")

oof_pred_final = np.zeros(len(X_train))
test_pred_final_arr = np.zeros(len(X_test))

fold_scores_final = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold+1}/5...")
    
    X_tr = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_tr = y_train_log[train_idx]
    y_val_fold = y_train_log[val_idx]
    
    model = xgb.XGBRegressor(**best_params, n_estimators=5000)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=100,
        verbose=0
    )
    
    oof_pred_final[val_idx] = model.predict(X_val_fold)
    test_pred_final_arr += model.predict(X_test) / kf.n_splits
    
    fold_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(model.predict(X_val_fold)))
    fold_scores_final.append(fold_mae)
    print(f"  Fold MAE: Â£{fold_mae:,.2f}")

# ============= FINAL METRICS =============
print("\n" + "=" * 70)
print("FINAL METRICS")
print("=" * 70)

final_cv_mae = mean_absolute_error(y_train, np.expm1(oof_pred_final))
final_std = np.std(fold_scores_final)

print(f"\nâœ“ Final CV MAE: Â£{final_cv_mae:,.2f}")
print(f"âœ“ Final Std Dev: Â£{final_std:,.2f}")
print(f"âœ“ Fold Scores: {[f'Â£{s:,.0f}' for s in fold_scores_final]}")

# ============= CREATE SUBMISSION WITH HARD CLIPPING =============
print("\n[Creating Submission with Hard Clipping]")

test_pred_exp = np.expm1(test_pred_final_arr)

# Define clipping bounds from training data
y_train_sorted = np.sort(y_train)
y_min = y_train_sorted[int(len(y_train) * 0.001)]
y_max = y_train_sorted[int(len(y_train) * 0.999)]

test_pred_clipped = np.clip(test_pred_exp, y_min, y_max)
test_pred_safe = np.maximum(test_pred_clipped, 0)

submission = pd.DataFrame({
    'ID': test_df['ID'].values,
    'price': test_pred_safe
})

submission.to_csv('submission_fine_tuned.csv', index=False)

print(f"\nâœ“ Submission saved as 'submission_fine_tuned.csv'")
print(f"\nSubmission Stats:")
print(f"  Mean: Â£{submission['price'].mean():,.2f}")
print(f"  Median: Â£{submission['price'].median():,.2f}")
print(f"  Min: Â£{submission['price'].min():,.2f}")
print(f"  Max: Â£{submission['price'].max():,.2f}")

print("\n" + "=" * 70)
print("âœ“ READY TO SUBMIT - submission_fine_tuned.csv")
print("=" * 70)


# ============= REFINING  BEST MODEL =============

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

print("=" * 70)
print("REFINEMENT: HIGHER ESTIMATORS + DIFFERENT CV SEEDS")
print("=" * 70)


best_params = {
    'learning_rate': 0.025,
    'max_depth': 9,
    'min_child_weight': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'gamma': 0,
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae',
    'random_state': 42,
    
    # ğŸš€ Enable GPU acceleration
    'tree_method': 'cuda',           # modern GPU tree method
    'predictor': 'cuda_predictor',   # GPU prediction
    'device': 'cuda',                # ensure all ops run on GPU
    'n_jobs': -1
}



# Your best params (Deeper Trees config from fine-tuning)
best_params = {
    'learning_rate': 0.025,
    'max_depth': 9,
    'min_child_weight': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'gamma': 0,
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae',
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

# Test different configurations
test_configs = [
    {
        'name': 'Current: 5000 est, seed=42',
        'n_estimators': 5000,
        'early_stopping': 100,
        'kfold_seed': 42
    },
    {
        'name': 'Higher: 7000 est, seed=42',
        'n_estimators': 7000,
        'early_stopping': 150,
        'kfold_seed': 42
    },
    {
        'name': 'High: 8000 est, seed=42',
        'n_estimators': 8000,
        'early_stopping': 200,
        'kfold_seed': 42
    },
    {
        'name': 'Current: 5000 est, seed=123',
        'n_estimators': 5000,
        'early_stopping': 100,
        'kfold_seed': 123
    },
    {
        'name': 'Current: 5000 est, seed=456',
        'n_estimators': 5000,
        'early_stopping': 100,
        'kfold_seed': 456
    },
]

y_train_log = np.log1p(y_train)

results = []
best_overall_mae = float('inf')
best_config_result = None

print("\n[Testing different configurations]\n")

for config in test_configs:
    config_name = config['name']
    n_est = config['n_estimators']
    early_stop = config['early_stopping']
    kfold_seed = config['kfold_seed']
    
    print(f"Testing: {config_name}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
    
    oof_pred = np.zeros(len(X_train))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_tr = y_train_log[train_idx]
        y_val_fold = y_train_log[val_idx]
        
        model = xgb.XGBRegressor(**best_params, n_estimators=n_est)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=early_stop,
            verbose=0
        )
        
        oof_pred[val_idx] = model.predict(X_val_fold)
        fold_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(model.predict(X_val_fold)))
        fold_scores.append(fold_mae)
    
    cv_mae = mean_absolute_error(y_train, np.expm1(oof_pred))
    cv_std = np.std(fold_scores)
    
    print(f"  CV MAE: Â£{cv_mae:,.2f} (Â±Â£{cv_std:,.2f})")
    
    results.append({
        'config': config_name,
        'mae': cv_mae,
        'std': cv_std,
        'n_estimators': n_est,
        'kfold_seed': kfold_seed
    })
    
    if cv_mae < best_overall_mae:
        best_overall_mae = cv_mae
        best_config_result = config
        print(f"  âœ“ NEW BEST!")
    
    print()

# ============= DISPLAY RESULTS =============
print("=" * 70)
print("CONFIGURATION RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results).sort_values('mae')
print("\n" + results_df[['config', 'mae', 'std']].to_string(index=False))

print(f"\nğŸ�† BEST: {best_config_result['name']}")
print(f"   MAE: Â£{best_overall_mae:,.2f}")

# ============= TRAIN FINAL WITH BEST CONFIG =============
print("\n" + "=" * 70)
print("TRAINING FINAL MODEL WITH BEST CONFIG")
print("=" * 70)

n_est_final = best_config_result['n_estimators']
early_stop_final = best_config_result['early_stopping']
kfold_seed_final = best_config_result['kfold_seed']

print(f"\n[Config]: {n_est_final} estimators, early_stopping={early_stop_final}, KFold seed={kfold_seed_final}")

kf_final = KFold(n_splits=5, shuffle=True, random_state=kfold_seed_final)

oof_pred_final = np.zeros(len(X_train))
test_pred_final = np.zeros(len(X_test))
fold_scores_final = []

for fold, (train_idx, val_idx) in enumerate(kf_final.split(X_train)):
    print(f"\nFold {fold+1}/5...")
    
    X_tr = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_tr = y_train_log[train_idx]
    y_val_fold = y_train_log[val_idx]
    
    model = xgb.XGBRegressor(**best_params, n_estimators=n_est_final)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=early_stop_final,
        verbose=0
    )
    
    oof_pred_final[val_idx] = model.predict(X_val_fold)
    test_pred_final += model.predict(X_test) / kf_final.n_splits
    
    fold_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(model.predict(X_val_fold)))
    fold_scores_final.append(fold_mae)
    print(f"  Fold MAE: Â£{fold_mae:,.2f}")

# ============= FINAL METRICS =============
print("\n" + "=" * 70)
print("FINAL METRICS")
print("=" * 70)

final_cv_mae = mean_absolute_error(y_train, np.expm1(oof_pred_final))
final_std = np.std(fold_scores_final)

print(f"\nâœ“ Final CV MAE: Â£{final_cv_mae:,.2f}")
print(f"âœ“ Final Std Dev: Â£{final_std:,.2f}")
print(f"âœ“ Fold Scores: {[f'Â£{s:,.0f}' for s in fold_scores_final]}")

# ============= CREATE SUBMISSION - RAW PREDICTIONS =============
print("\n[Creating Submission]")

test_pred_exp = np.expm1(test_pred_final)
test_pred_safe = np.maximum(test_pred_exp, 0)

submission = pd.DataFrame({
    'ID': test_df['ID'].values,
    'price': test_pred_safe
})

submission.to_csv('submission_refined.csv', index=False)

print(f"\nâœ“ Submission saved as 'submission_refined.csv'")

print(f"\nSubmission Stats:")
print(f"  Mean: Â£{submission['price'].mean():,.2f}")
print(f"  Median: Â£{submission['price'].median():,.2f}")
print(f"  Min: Â£{submission['price'].min():,.2f}")
print(f"  Max: Â£{submission['price'].max():,.2f}")

print("\n" + "=" * 70)
print("âœ“ READY TO SUBMIT - submission_refined.csv")
print("=" * 70)

print("\nNext steps if this doesn't improve:")
print("1. Try LightGBM with same CV strategy")
print("2. Try CatBoost with same CV strategy")
print("3. Analyze residuals - which property types fail most?")
print("4. Add targeted features for those property types")


# ============= 10000 ESTIMATORS =============

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

print("=" * 70)
print("FINAL PUSH: BEST MODEL WITH 10000 ESTIMATORS")
print("=" * 70)

# Your proven best params
best_params = {
    'learning_rate': 0.025,
    'max_depth': 9,
    'min_child_weight': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'gamma': 0,
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae',
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

y_train_log = np.log1p(y_train)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n[Training with 10000 estimators, early_stopping=250]\n")

oof_pred_final = np.zeros(len(X_train))
test_pred_final = np.zeros(len(X_test))
fold_scores_final = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}/5...")
    
    X_tr = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_tr = y_train_log[train_idx]
    y_val_fold = y_train_log[val_idx]
    
    model = xgb.XGBRegressor(**best_params, n_estimators=10000)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=250,
        verbose=0
    )
    
    oof_pred_final[val_idx] = model.predict(X_val_fold)
    test_pred_final += model.predict(X_test) / kf.n_splits
    
    fold_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(model.predict(X_val_fold)))
    fold_scores_final.append(fold_mae)
    print(f"  Fold MAE: Â£{fold_mae:,.2f}")

# ============= FINAL METRICS =============
print("\n" + "=" * 70)
print("FINAL METRICS")
print("=" * 70)

final_cv_mae = mean_absolute_error(y_train, np.expm1(oof_pred_final))
final_std = np.std(fold_scores_final)

print(f"\nâœ“ Final CV MAE: Â£{final_cv_mae:,.2f}")
print(f"âœ“ Final Std Dev: Â£{final_std:,.2f}")
print(f"âœ“ Fold Scores: {[f'Â£{s:,.0f}' for s in fold_scores_final]}")

# ============= CREATE SUBMISSION =============
print("\n[Creating Submission - Raw Predictions]")

test_pred_exp = np.expm1(test_pred_final)
test_pred_safe = np.maximum(test_pred_exp, 0)

submission = pd.DataFrame({
    'ID': test_df['ID'].values,
    'price': test_pred_safe
})

submission.to_csv('submission_final_10k.csv', index=False)

print(f"\nâœ“ Submission saved as 'submission_final_10k.csv'")

print(f"\nSubmission Stats:")
print(f"  Mean: Â£{submission['price'].mean():,.2f}")
print(f"  Median: Â£{submission['price'].median():,.2f}")
print(f"  Min: Â£{submission['price'].min():,.2f}")
print(f"  Max: Â£{submission['price'].max():,.2f}")

print("\n" + "=" * 70)
print("âœ“ SUBMIT: submission_final_10k.csv")
print("=" * 70)


# # ============= LIGHTGBM WITH XGBOOST BEST PARAMS =============

# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_absolute_error
# import numpy as np
# import pandas as pd

# print("=" * 70)
# print("LIGHTGBM: SAME HYPERPARAMS AS BEST XGBOOST")
# print("=" * 70)

# # XGBoost best params converted to LightGBM equivalents
# lgb_params = {
#     'objective': 'mae',
#     'metric': 'mae',
#     'learning_rate': 0.025,  # Same as XGB
#     'num_leaves': 255,  # Equivalent to max_depth=9 (2^9-1 = 511, so use high)
#     'max_depth': 9,  # Direct equivalent
#     'min_child_samples': 20,  # Equivalent to min_child_weight=2
#     'subsample': 0.85,  # Same as XGB
#     'colsample_bytree': 0.85,  # Same as XGB
#     'reg_alpha': 0.05,  # Same L1
#     'reg_lambda': 0.5,  # Same L2
#     'random_state': 42,
#     'verbose': -1,
#     'n_jobs': -1
# }

# y_train_log = np.log1p(y_train)
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# print("\n[Training LightGBM with 8000 rounds, early_stopping=200]\n")

# oof_pred_final = np.zeros(len(X_train))
# test_pred_final = np.zeros(len(X_test))
# fold_scores_final = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
#     print(f"Fold {fold+1}/5...")
    
#     X_tr = X_train.iloc[train_idx]
#     X_val_fold = X_train.iloc[val_idx]
#     y_tr = y_train_log[train_idx]
#     y_val_fold = y_train_log[val_idx]
    
#     lgb_train = lgb.Dataset(X_tr, y_tr)
#     lgb_valid = lgb.Dataset(X_val_fold, y_val_fold, reference=lgb_train)
    
#     model = lgb.train(
#         lgb_params,
#         lgb_train,
#         num_boost_round=8000,
#         valid_sets=[lgb_valid],
#         callbacks=[
#             lgb.early_stopping(200),
#             lgb.log_evaluation(0)
#         ]
#     )
    
#     oof_pred_final[val_idx] = model.predict(X_val_fold)
#     test_pred_final += model.predict(X_test) / kf.n_splits
    
#     fold_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(model.predict(X_val_fold)))
#     fold_scores_final.append(fold_mae)
#     print(f"  Fold MAE: Â£{fold_mae:,.2f}")

# # ============= FINAL METRICS =============
# print("\n" + "=" * 70)
# print("FINAL METRICS")
# print("=" * 70)

# final_cv_mae = mean_absolute_error(y_train, np.expm1(oof_pred_final))
# final_std = np.std(fold_scores_final)

# print(f"\nâœ“ Final CV MAE: Â£{final_cv_mae:,.2f}")
# print(f"âœ“ Final Std Dev: Â£{final_std:,.2f}")
# print(f"âœ“ Fold Scores: {[f'Â£{s:,.0f}' for s in fold_scores_final]}")

# # ============= CREATE SUBMISSION =============
# print("\n[Creating Submission - Raw Predictions]")

# test_pred_exp = np.expm1(test_pred_final)
# test_pred_safe = np.maximum(test_pred_exp, 0)

# submission = pd.DataFrame({
#     'ID': test_df['ID'].values,
#     'price': test_pred_safe
# })

# submission.to_csv('submission_lightgbm.csv', index=False)

# print(f"\nâœ“ Submission saved as 'submission_lightgbm.csv'")

# print(f"\nSubmission Stats:")
# print(f"  Mean: Â£{submission['price'].mean():,.2f}")
# print(f"  Median: Â£{submission['price'].median():,.2f}")
# print(f"  Min: Â£{submission['price'].min():,.2f}")
# print(f"  Max: Â£{submission['price'].max():,.2f}")

# print("\n" + "=" * 70)
# print("âœ“ SUBMIT: submission_lightgbm.csv")
# print("=" * 70)

# print("\nComparison:")
# print("  Compare this LightGBM CV MAE to your XGBoost CV MAE")
# print("  Submit whichever has lower MAE to Kaggle!")

