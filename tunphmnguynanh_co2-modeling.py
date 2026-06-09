%%capture
# Install relevant libraries
!pip install geopandas folium 


# Import libraries
import pandas as pd
import numpy as np
import random
import os
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score,RandomizedSearchCV,LeaveOneGroupOut,GroupKFold
import geopandas as gpd
from shapely.geometry import Point
import folium
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
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


train['emission_original'] = train['emission'].copy()


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
    if not col.endswith('_roll_mean') and col not in ['emission','emission_original', 'ID_LAT_LON_YEAR_WEEK', 'location','date']
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


train_eng['y_transformed'] = np.log1p(train_eng['emission'])


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


# def optimize_model(trial, model_name, X, y):
#     kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
#     scores = []
    
#     if model_name == 'rf':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 200, 500),  # TÄƒng pháº¡m vi Ä‘á»ƒ há»�c sÃ¢u hÆ¡n
#             'max_depth': trial.suggest_categorical('max_depth', [30, 50, 70, None]),  # Má»Ÿ rá»™ng cÃ¡c lá»±a chá»�n
#             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),  # Giá»¯ nhá»� Ä‘á»ƒ há»�c chi tiáº¿t
#             'random_state': SEED
#         }
#         model = RandomForestRegressor(**params)
#     elif model_name == 'xgb':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 200, 500),  # TÄƒng pháº¡m vi
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # Giáº£m pháº¡m vi Ä‘á»ƒ kiá»ƒm soÃ¡t tá»‘t hÆ¡n
#             'max_depth': trial.suggest_int('max_depth', 5, 10),  # Má»Ÿ rá»™ng pháº¡m vi
#             'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # ThÃªm tham sá»‘ Ä‘á»ƒ kiá»ƒm soÃ¡t overfitting
#             'random_state': SEED
#         }
#         model = XGBRegressor(**params)
#     elif model_name == 'lgb':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 200, 500),  # TÄƒng pháº¡m vi
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # Giáº£m pháº¡m vi
#             'num_leaves': trial.suggest_int('num_leaves', 30, 70),  # Má»Ÿ rá»™ng pháº¡m vi
#             'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 20),  # Giáº£m Ä‘á»ƒ há»�c chi tiáº¿t hÆ¡n
#             'random_state': SEED,
#             'verbose': -1
#         }
#         model = LGBMRegressor(**params)
#     else:  # cb
#         params = {
#             'iterations': trial.suggest_int('iterations', 200, 500),  # TÄƒng pháº¡m vi
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # Giáº£m pháº¡m vi
#             'depth': trial.suggest_int('depth', 6, 12),  # Má»Ÿ rá»™ng pháº¡m vi
#             'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 20),  # ThÃªm Ä‘á»ƒ kiá»ƒm soÃ¡t overfitting
#             'random_state': SEED,
#             'verbose': 0
#         }
#         model = CatBoostRegressor(**params)
    
#     for train_idx, val_idx in kfold.split(X):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#         model.fit(X_train, y_train)
#         preds = model.predict(X_val)
#         score = mean_squared_error(y_val, preds, squared=False)
#         scores.append(score)
    
#     return np.mean(scores)


# def optimize_model(trial, model_name, X, y):
#     kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
#     scores = []
    
#     if model_name == 'rf':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#             'max_depth': trial.suggest_categorical('max_depth', [20, 40, None]),
#             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),
#             'random_state': SEED
#         }
#         model = RandomForestRegressor(**params)
#     elif model_name == 'xgb':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#             'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
#             'max_depth': trial.suggest_int('max_depth', 3, 7),
#             'random_state': SEED
#         }
#         model = XGBRegressor(**params)
#     elif model_name == 'lgb':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#             'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 50),
#             'random_state': SEED,
#             'verbose': -1
#         }
#         model = LGBMRegressor(**params)
#     else:  # cb
#         params = {
#             'iterations': trial.suggest_int('iterations', 100, 300),
#             'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
#             'depth': trial.suggest_int('depth', 4, 8),
#             'random_state': SEED,
#             'verbose': 0
#         }
#         model = CatBoostRegressor(**params)
    
#     for train_idx, val_idx in kfold.split(X):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#         model.fit(X_train, y_train)
#         preds = model.predict(X_val)
#         score = mean_squared_error(y_val, preds, squared=False)
#         scores.append(score)
    
#     return np.mean(scores)


# best_models = []
# model_names = ['rf', 'xgb', 'lgb', 'cb']
# for model_name in model_names:
#     print(f"Tuning {model_name}...")
#     study = optuna.create_study(direction='minimize')
#     study.optimize(lambda trial: optimize_model(trial, model_name, X, y), n_trials=10)
#     print(f"Best parameters for {model_name}: {study.best_params}")
#     print(f"Best RMSE: {study.best_value:.5f}")
    
#     # Initialize model with best parameters
#     if model_name == 'rf':
#         model = RandomForestRegressor(**study.best_params, random_state=SEED)
#     elif model_name == 'xgb':
#         model = XGBRegressor(**study.best_params, random_state=SEED)
#     elif model_name == 'lgb':
#         model = LGBMRegressor(**study.best_params, random_state=SEED, verbose=-1)
#     else:
#         model = CatBoostRegressor(**study.best_params, random_state=SEED, verbose=0)
#     best_models.append((model_name, model))


k = KFold(n_splits=5, shuffle=True, random_state=SEED)


def cross_val_score(model, cv=k, label=''):
    val_predictions = np.zeros((len(X)))
    train_scores, val_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        val_predictions[val_idx] += val_preds
        train_score = mean_squared_error(y_train, train_preds, squared=False)
        val_score = mean_squared_error(y_val, val_preds, squared=False)
        train_scores.append(train_score)
        val_scores.append(val_score)
    print(f'Val Score: {np.mean(val_scores):.5f} Â± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} Â± {np.std(train_scores):.5f} | {label}')
    return val_scores, val_predictions


models = [
    ('rf', RandomForestRegressor(n_estimators= 421, max_depth= 50, min_samples_leaf= 2)),
    ('xgb', XGBRegressor(n_estimators= 377, learning_rate= 0.08829642040426344, max_depth= 10, min_child_weight = 6)),
    ('lgb', LGBMRegressor(n_estimators= 475, learning_rate= 0.06740618868536545, num_leaves= 67,min_data_in_leaf = 15,force_row_wise=True)),
    ('cb', CatBoostRegressor(iterations= 432, learning_rate= 0.07570625345059977, depth= 10,min_data_in_leaf= 10))
]


models = [
    ('rf', RandomForestRegressor(n_estimators= 109, max_depth= None, min_samples_leaf= 2)),
    ('xgb', XGBRegressor(n_estimators= 177, learning_rate= 0.14347260931294858, max_depth= 7)),
    ('lgb', LGBMRegressor(n_estimators= 266, learning_rate= 0.15416009912090234, num_leaves= 34,force_row_wise=True)),
    ('cb', CatBoostRegressor(iterations= 247, learning_rate= 0.14254245134507781, depth= 8))
]


# Evaluate models
score_list = pd.DataFrame()
oof_list = pd.DataFrame()

for label, model in models:
    val_scores, val_predictions = cross_val_score(model, cv=k, label=label)
    score_list[label] = val_scores
    oof_list[label] = val_predictions


# Evaluate models
score_list = pd.DataFrame()
oof_list = pd.DataFrame()

for label, model in models:
    val_scores, val_predictions = cross_val_score(model, cv=k, label=label)
    score_list[label] = val_scores
    oof_list[label] = val_predictions


# Summarize mean RMSE
print("\nMean RMSE Summary:")
for label in score_list.columns:
    print(f"{label}: {np.mean(score_list[label]):.5f} Â± {np.std(score_list[label]):.5f}")


plt.figure(figsize = (8, 4), dpi = 300)
sns.barplot(data = score_list.reindex((score_list).mean().sort_values().index, axis = 1), palette = 'viridis', orient = 'h')
plt.title('Score Comparison', weight = 'bold', size = 20)
plt.show()


# weights = Ridge(random_state = 2023).fit(oof_list, y).coef_

# pd.DataFrame(weights, index = oof_list.columns, columns = ['weight per model'])


# voter = VotingRegressor(models, weights = weights)

# _ = cross_val_score(voter)


best_md = RandomForestRegressor(n_estimators= 109, max_depth= None, min_samples_leaf= 2)


best_md.fit(X, y)


negative_count = np.sum(y_pred_test < 0)
print(f"Sá»‘ lÆ°á»£ng giÃ¡ trá»‹ Ã¢m trong y_pred_test: {negative_count}")


# voter.fit(X, y)


# # Predict on test set
# y_pred_test = voter.predict(X_test)


# Chuáº©n bá»‹ dá»¯ liá»‡u train_plot theo date
train_plot = train.copy()
train_plot = train.groupby(['date'])['emission'].sum().reset_index()

# Chuáº©n bá»‹ dá»¯ liá»‡u pred_plot theo date
pred_plot = test.copy()
pred_plot['emission'] = y_pred_test  # GÃ¡n giÃ¡ trá»‹ dá»± Ä‘oÃ¡n
pred_plot = pred_plot.groupby(['date'])['emission'].sum().reset_index()

# Váº½ biá»ƒu Ä‘á»“
plt.figure(figsize=(20, 7))

# Váº½ Ä‘Æ°á»�ng cho train_plot (dá»¯ liá»‡u thá»±c táº¿)
train_plot.groupby(['date'])['emission'].sum().plot(kind='line', label='Actual Emission', color='blue')

# Váº½ Ä‘Æ°á»�ng cho pred_plot (dá»± Ä‘oÃ¡n)
pred_plot.groupby(['date'])['emission'].sum().plot(kind='line', label='Predicted Emission', color='orange')

# ThÃªm vÃ¹ng tÃ´ mÃ u vÃ  Ä‘Æ°á»�ng káº»
# Ä�áº·t ranh giá»›i dá»± Ä‘oÃ¡n tá»« cuá»‘i 2021 Ä‘áº¿n Ä‘áº§u 2022
plt.axvspan(pd.Timestamp('2021-12-27'), pd.Timestamp('2022-01-01'), color='green', alpha=0.1)  # VÃ¹ng dá»± Ä‘oÃ¡n
plt.axvline(pd.Timestamp('2021-12-27'), linestyle="--", color='green')  # Ä�Æ°á»�ng cháº¥m táº¡i cuá»‘i 2021
plt.axvline(pd.Timestamp('2022-01-01'), linestyle="--", color='green')  # Ä�Æ°á»�ng cháº¥m táº¡i Ä‘áº§u 2022

# ThÃªm nhÃ£n cho vÃ¹ng dá»± Ä‘oÃ¡n
plt.text(pd.Timestamp('2022-06-01'), 50000, "Predictions", size=17)  # Ä�iá»�u chá»‰nh vá»‹ trÃ­ nhÃ£n

# ThÃªm tiÃªu Ä‘á»� vÃ  nhÃ£n
plt.title('Emission by Date', size=15, pad=10)
plt.xlabel('Date')
plt.ylabel('Total Emission')
plt.legend()

# Hiá»ƒn thá»‹ biá»ƒu Ä‘á»“
plt.show()


# Visualization
pal = sns.color_palette("husl", 5)

plt.figure(figsize=(20, 10), dpi=300)

# Plot predicted emission for 2022 (test set)
sns.lineplot(x=test.week_no, y=y_pred_test, errorbar=None, label='2022 (Predicted)', color=pal[0])

# Plot actual emission for previous years (train set, adjusted data)
sns.lineplot(x=train.week_no, y=train.emission, hue=train.year, errorbar=None, 
             palette=[pal[1], pal[2], pal[3]], legend=False)

# Plot original emission for 2020
sns.lineplot(x=train[train['year'] == 2020].week_no, y=train[train['year'] == 2020].emission_original, 
             errorbar=None, label='2020 (Original)', color=pal[4], linestyle='--')

plt.legend(['2022 (Predicted)', 2019, '2020 (Adjusted)', 2021, '2020 (Original)'], loc='upper right')
plt.title('Predicted Emission in 2022 vs Previous Years Emission (Including Original 2020)', fontsize=24, fontweight='bold')
plt.xlabel('Week Number')
plt.ylabel('Emission')
plt.grid(True)
plt.show()


submission_df = test[['longitude', 'latitude', 'year', 'week_no']].copy()

# Add the predicted emission column
submission_df['emission'] = y_pred_test  # test_pred from Final Prediction (voter.predict(X_test))

# Save to CSV
submission_df.to_csv('submission_with_coords.csv', index=False)
print("Custom submission file created: submission_with_coords.csv")
print(submission_df.head())  # Preview the first few rows


# submission_train_df = train[['longitude', 'latitude', 'year', 'week_no']].copy()

# # Add the predicted emission column
# submission_train_df['emission'] = train['emission']  # test_pred from Final Prediction (voter.predict(X_test))

# # Save to CSV
# submission_train_df.to_csv('submission_train.csv', index=False)
# print("Custom submission file created: submission_train.csv")
# print(submission_train_df.head())  # Preview the first few rows

