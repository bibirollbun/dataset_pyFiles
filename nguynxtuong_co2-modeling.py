%%capture
# Install relevant libraries
!pip install geopandas folium 


# Import libraries
import pandas as pd
import numpy as np
import random
import os
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
import geopandas as gpd
from shapely.geometry import Point
import folium
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm  import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_rows = None

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# You can ignore the Shapely GEOS warning :-)


# Set seed for reproducability
SEED = 2023
random.seed(SEED)
np.random.seed(SEED)


DATA_PATH = '/kaggle/input/playground-series-s3e20'
# Load files
train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
samplesubmission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

# Preview train dataset
train.head()


# Drop columns with >90% missing values
missing_ratio = train.isna().mean()
columns_to_drop = missing_ratio[missing_ratio > 0.9].index
train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)


# Create date column from year and week_no
train['date'] = pd.to_datetime(train['year'].astype(str) + '-' + train['week_no'].astype(str) + '-1', format='%Y-%W-%w')
test['date'] = pd.to_datetime(test['year'].astype(str) + '-' + test['week_no'].astype(str) + '-1', format='%Y-%W-%w')

# Create season column based on month
train['season'] = train['date'].dt.month.apply(lambda x: 1 if 3 <= x <= 5 else 2 if 6 <= x <= 8 else 3 if 9 <= x <= 11 else 4)
test['season'] = test['date'].dt.month.apply(lambda x: 1 if 3 <= x <= 5 else 2 if 6 <= x <= 8 else 3 if 9 <= x <= 11 else 4)
# Create holidays column based on specific weeks
train['holidays'] = train['week_no'].isin([0, 51, 12, 30])
test['holidays'] = test['week_no'].isin([0, 51, 12, 30])
# Create rotated coordinates features
train['rot_15_x'] = (np.cos(np.radians(15)) * train['longitude']) + (np.sin(np.radians(15)) * train['latitude'])
train['rot_15_y'] = (np.cos(np.radians(15)) * train['latitude']) + (np.sin(np.radians(15)) * train['longitude'])
train['rot_30_x'] = (np.cos(np.radians(30)) * train['longitude']) + (np.sin(np.radians(30)) * train['latitude'])
train['rot_30_y'] = (np.cos(np.radians(30)) * train['latitude']) + (np.sin(np.radians(30)) * train['longitude'])

test['rot_15_x'] = (np.cos(np.radians(15)) * test['longitude']) + (np.sin(np.radians(15)) * test['latitude'])
test['rot_15_y'] = (np.cos(np.radians(15)) * test['latitude']) + (np.sin(np.radians(15)) * test['longitude'])
test['rot_30_x'] = (np.cos(np.radians(30)) * test['longitude']) + (np.sin(np.radians(30)) * test['latitude'])
test['rot_30_y'] = (np.cos(np.radians(30)) * test['latitude']) + (np.sin(np.radians(30)) * test['longitude'])


# Adjust emissions for 2020 (COVID year) to align with non-COVID years (2019 and 2021)
# Calculate the average weekly emissions for non-virus years (2019 and 2021)
avg_emission_non_virus = train[train['year'].isin((2019, 2021))].groupby('week_no')['emission'].mean()

# Calculate the average weekly emissions for virus year (2020)
avg_emission_virus = train[train['year'] == 2020].groupby('week_no')['emission'].mean()

# Calculate the ratios for each week
ratios_for_weeks = avg_emission_non_virus / avg_emission_virus

# Multiply the emission column for each row in 2020 by the corresponding ratio for the week of that row
train.loc[train['year'] == 2020, 'emission'] *= train['week_no'].map(ratios_for_weeks)

# Fix the large spike in the last week of 2020 (week 52) as an outlier
train.loc[(train['week_no'] == 52) & (train['year'] == 2020), 'emission'] = np.power(
    train.loc[(train['week_no'] == 52) & (train['year'] == 2020), 'emission'], 1/1.5)


#for col in train.columns:
#    if train[col].isna().any():
  #      train[col].fillna(train[col].median(), inplace=True)
  #     if col in test.columns:
   #         test[col].fillna(train[col].median(), inplace=True)


train['emission'] = np.log1p(train['emission'])  # log1p handles zero values


sin_week = np.sin(2 * np.pi * train['week_no'] / 53)
cos_week = np.cos(2 * np.pi * train['week_no'] / 53)
location = [f"{x}_{y}" for x, y in zip(train.latitude, train.longitude)]

emission_idx = train.columns.get_loc('emission')
train.insert(emission_idx, 'sin_week', sin_week)
train.insert(emission_idx + 1, 'cos_week', cos_week)
train.insert(emission_idx + 2, 'location', location)

sin_week_test = np.sin(2 * np.pi * test['week_no'] / 53)
cos_week_test = np.cos(2 * np.pi * test['week_no'] / 53)
location_test = [f"{x}_{y}" for x, y in zip(test.latitude, test.longitude)]

test.insert(emission_idx, 'sin_week', sin_week_test)
test.insert(emission_idx + 1, 'cos_week', cos_week_test)
test.insert(emission_idx + 2, 'location', location_test)



train_eng = train.sort_values(by=['location', 'year', 'week_no'], ignore_index=True)
test_eng = test.sort_values(by=['location', 'year', 'week_no'], ignore_index=True)


original_features = [
    col for col in train_eng.columns 
    if not col.endswith('_roll_mean') and col not in ['emission', 'ID_LAT_LON_YEAR_WEEK', 'location','date']
]

corrs = abs(train_eng[original_features + ['emission']].corr()['emission']).sort_values(ascending=False)
top20_original = corrs[corrs.index.isin(original_features)].head(10).index.tolist()



top20_original


roll_mean_cols = top20_original
train_roll_mean = train.sort_values(by=['location', 'year', 'week_no']) \
    .groupby('location')[roll_mean_cols] \
    .rolling(window=7).mean().reset_index()
train_roll_mean.drop(['level_1', 'location'], axis=1, inplace=True)
train_roll_mean.columns = [col + '_roll_mean' for col in train_roll_mean.columns]

test_roll_mean = test.sort_values(by=['location', 'year', 'week_no']) \
    .groupby('location')[roll_mean_cols] \
    .rolling(window=7).mean().reset_index()
test_roll_mean.drop(['level_1', 'location'], axis=1, inplace=True)
test_roll_mean.columns = [col + '_roll_mean' for col in test_roll_mean.columns]


train_eng = train_eng.merge(train_roll_mean, how='left', left_index=True, right_index=True)
test_eng = test_eng.merge(test_roll_mean, how='left', left_index=True, right_index=True)


# Fill NaN values in rolling mean columns with median
for col in train_eng.columns:
    if train_eng[col].isna().any():
        train_eng[col].fillna(train_eng[col].median(), inplace=True)
        if col in test_eng.columns:
            test_eng[col].fillna(train_eng[col].median(), inplace=True)


selected_features = [col + '_roll_mean' for col in top20_original 
                     if col + '_roll_mean' in train_eng.columns and col + '_roll_mean' in test_eng.columns]

# Replace longitude_roll_mean with original longitude
if 'longitude_roll_mean' in selected_features:
    selected_features.remove('longitude_roll_mean')
    selected_features.append('longitude')

# Add sine and cosine of week and latitude
for feat in ['sin_week', 'cos_week', 'latitude','season','holidays','rot_15_x', 'rot_15_y', 'rot_30_x', 'rot_30_y','year']:
    if feat not in selected_features:
        selected_features.append(feat)


# Remove redundant rotated coordinate rolling means
selected_features = [feat for feat in selected_features if feat not in [
    'rot_15_x_roll_mean', 'rot_30_x_roll_mean', 'rot_30_y_roll_mean', 'rot_15_y_roll_mean'
]]


train_eng.drop(columns=['ID_LAT_LON_YEAR_WEEK', 'date'], inplace=True)
test_eng.drop(columns=['ID_LAT_LON_YEAR_WEEK', 'date'], inplace=True)


selected_features


selected_features = [col for col in selected_features if col in train_eng.columns and col in test_eng.columns]


selected_features


X = train_eng[selected_features]
y = train_eng['emission']
X_test = test_eng[selected_features]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2023)


# # Define parameter grids for GridSearchCV
# param_grid_rf = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
param_grid_lgbm = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [15, 31, 63],
    'max_depth': [-1, 5, 10]
}


# # Initialize models
# rf_model = RandomForestRegressor(random_state=2023, n_jobs=-1)
lgbm_model = LGBMRegressor()


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Perform GridSearchCV for RandomForest
# grid_search_rf = GridSearchCV(
#     estimator=rf_model,
#     param_grid=param_grid_rf,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
#     verbose=1
# )
# grid_search_rf.fit(X, y)
# print("Best parameters for RandomForest:", grid_search_rf.best_params_)
# print("Best RMSE (thang log) for RandomForest:", np.sqrt(-grid_search_rf.best_score_))

grid_search_lgbm = GridSearchCV(
    estimator=lgbm_model,
    param_grid=param_grid_lgbm,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search_lgbm.fit(X, y)



print("Best parameters for LGBM:", grid_search_lgbm.best_params_)
print("Best RMSE (thang log) for LGBM:", np.sqrt(-grid_search_lgbm.best_score_))


# best_model = grid_search_rf.best_estimator_
# model_type = "RandomForest"
# print(f"Selected model: {model_type} with best parameters")
best_model = grid_search_lgbm.best_estimator_
model_type = "LGBM"
print(f"Selected model: {model_type} with best parameters")


# Apply 5-fold cross-validation with the best model
kf = KFold(n_splits=5, shuffle=True, random_state=2023)

print("Káº¿t quáº£ RMSE cho tá»«ng fold:")
rmse_scores_train_log = []
rmse_scores_val_log = []
rmse_scores_train_orig = []
rmse_scores_val_orig = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train the model
    best_model.fit(X_train, y_train)
    
    # Predict on training and validation fold
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    
    # Calculate RMSE on log scale
    rmse_train_log = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val_log = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_scores_train_log.append(rmse_train_log)
    rmse_scores_val_log.append(rmse_val_log)
    
    # Calculate RMSE on original scale
    y_pred_train_orig = np.expm1(y_pred_train)
    y_train_orig = np.expm1(y_train)
    rmse_train_orig = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
    rmse_scores_train_orig.append(rmse_train_orig)
    
    y_pred_val_orig = np.expm1(y_pred_val)
    y_val_orig = np.expm1(y_val)
    rmse_val_orig = np.sqrt(mean_squared_error(y_val_orig, y_pred_val_orig))
    rmse_scores_val_orig.append(rmse_val_orig)
    
    print(f"Fold {fold}: Train RMSE (thang log) = {rmse_train_log:.5f}, Val RMSE (thang log) = {rmse_val_log:.5f}")
    print(f"Fold {fold}: Train RMSE (thang gá»‘c) = {rmse_train_orig:.5f}, Val RMSE (thang gá»‘c) = {rmse_val_orig:.5f}")


# Print average RMSE scores
print(f'ğŸ“‰ RMSE trung bÃ¬nh tá»« 5-fold CV (thang log): {np.mean(rmse_scores_val_log):.5f} Â± {np.std(rmse_scores_val_log):.5f}')
print(f'RMSE trung bÃ¬nh tá»« 5-fold CV (thang gá»‘c): {np.mean(rmse_scores_val_orig):.5f} Â± {np.std(rmse_scores_val_orig):.5f}')


import xgboost as xgb


param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}


xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)


grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_xgb.fit(X, y)


print("Best parameters for XGBoost:", grid_search_xgb.best_params_)
print("Best RMSE (thang log) for XGBoost:", np.sqrt(-grid_search_xgb.best_score_))


# best_model = grid_search_rf.best_estimator_
# model_type = "RandomForest"
# print(f"Selected model: {model_type} with best parameters")
best_model = grid_search_xgb.best_estimator_
model_type = "XGBoost"
print(f"Selected model: {model_type} with best parameters")


# Apply 5-fold cross-validation with the best model
kf = KFold(n_splits=5, shuffle=True, random_state=2023)

print("Káº¿t quáº£ RMSE cho tá»«ng fold:")
rmse_scores_train_log = []
rmse_scores_val_log = []
rmse_scores_train_orig = []
rmse_scores_val_orig = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train the model
    best_model.fit(X_train, y_train)
    
    # Predict on training and validation fold
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    
    # Calculate RMSE on log scale
    rmse_train_log = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val_log = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_scores_train_log.append(rmse_train_log)
    rmse_scores_val_log.append(rmse_val_log)
    
    # Calculate RMSE on original scale
    y_pred_train_orig = np.expm1(y_pred_train)
    y_train_orig = np.expm1(y_train)
    rmse_train_orig = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
    rmse_scores_train_orig.append(rmse_train_orig)
    
    y_pred_val_orig = np.expm1(y_pred_val)
    y_val_orig = np.expm1(y_val)
    rmse_val_orig = np.sqrt(mean_squared_error(y_val_orig, y_pred_val_orig))
    rmse_scores_val_orig.append(rmse_val_orig)
    
    print(f"Fold {fold}: Train RMSE (thang log) = {rmse_train_log:.5f}, Val RMSE (thang log) = {rmse_val_log:.5f}")
    print(f"Fold {fold}: Train RMSE (thang gá»‘c) = {rmse_train_orig:.5f}, Val RMSE (thang gá»‘c) = {rmse_val_orig:.5f}")


# Print average RMSE scores
print(f'ğŸ“‰ RMSE trung bÃ¬nh tá»« 5-fold CV (thang log): {np.mean(rmse_scores_val_log):.5f} Â± {np.std(rmse_scores_val_log):.5f}')
print(f'RMSE trung bÃ¬nh tá»« 5-fold CV (thang gá»‘c): {np.mean(rmse_scores_val_orig):.5f} Â± {np.std(rmse_scores_val_orig):.5f}')




