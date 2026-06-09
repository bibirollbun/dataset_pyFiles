import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# LightGBM and Sklearn
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# Check shapes
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)


# Check for nulls
print("Missing values in train:\n", train.isnull().sum())
print("Missing values in test:\n", test.isnull().sum())

# Dtypes
print("Dtypes:\n", train.dtypes)

# Target column stats
print(train['accident_risk'].describe())


TARGET = 'accident_risk'
ID = 'id'

# Set features (exclude ID and target)
FEATURES = [col for col in train.columns if col not in [ID, TARGET]]
print("Features used:", FEATURES)

# Sanity check: train/test column consistency
train_cols = set(FEATURES)
test_cols = set([col for col in test.columns if col != ID])
assert train_cols == test_cols, "Train and test columns do not match"


numeric_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    sns.histplot(train[col], kde=True, bins=30, ax=axes[i], color='skyblue')
    axes[i].set_title(f'{col} Distribution')

plt.tight_layout()
plt.show()


corrs = train[numeric_features + [TARGET]].corr()
print("Correlation with accident_risk:")
print(corrs[TARGET].sort_values(ascending=False))


cat_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'holiday', 'school_season', 'road_signs_present', 'public_road']

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    order = train.groupby(col)[TARGET].mean().sort_values().index
    sns.barplot(data=train, x=col, y=TARGET, ax=axes[i], order=order, palette='viridis')
    axes[i].set_title(f'Mean accident_risk by {col}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


from sklearn.preprocessing import OrdinalEncoder

# Copy
train_fe = train.copy()
test_fe = test.copy()

# Encode object-type categoricals
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
encoder = OrdinalEncoder()

train_fe[cat_cols] = encoder.fit_transform(train_fe[cat_cols])
test_fe[cat_cols] = encoder.transform(test_fe[cat_cols])

# Convert boolean to int
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    train_fe[col] = train_fe[col].astype(int)
    test_fe[col] = test_fe[col].astype(int)


# curvature Ã— speed
train_fe['curvature_speed'] = train_fe['curvature'] * train_fe['speed_limit']
test_fe['curvature_speed'] = test_fe['curvature'] * test_fe['speed_limit']

# speed Ã— num_lanes
train_fe['speed_num_lanes'] = train_fe['speed_limit'] * train_fe['num_lanes']
test_fe['speed_num_lanes'] = test_fe['speed_limit'] * test_fe['num_lanes']

# road_type + weather interaction (ordinal encoding)
train_fe['road_weather'] = train_fe['road_type'].astype(str) + '_' + train_fe['weather'].astype(str)
test_fe['road_weather'] = test_fe['road_type'].astype(str) + '_' + test_fe['weather'].astype(str)

rw_enc = OrdinalEncoder()
train_fe['road_weather_enc'] = rw_enc.fit_transform(train_fe[['road_weather']])
test_fe['road_weather_enc'] = rw_enc.transform(test_fe[['road_weather']])

train_fe.drop(columns=['road_weather'], inplace=True)
test_fe.drop(columns=['road_weather'], inplace=True)

# Cyclical encoding for time_of_day (ordinal: 0,1,2)
for df in [train_fe, test_fe]:
    time_val = df['time_of_day'].astype(int)
    df['time_of_day_sin'] = np.sin(2 * np.pi * time_val / 3)
    df['time_of_day_cos'] = np.cos(2 * np.pi * time_val / 3)


# Mean accident risk per road_type and weather
road_type_risk = train_fe.groupby('road_type')['accident_risk'].mean()
weather_risk = train_fe.groupby('weather')['accident_risk'].mean()

train_fe['road_type_mean_risk'] = train_fe['road_type'].map(road_type_risk)
test_fe['road_type_mean_risk'] = test_fe['road_type'].map(road_type_risk)

train_fe['weather_mean_risk'] = train_fe['weather'].map(weather_risk)
test_fe['weather_mean_risk'] = test_fe['weather'].map(weather_risk)


print("Sample of engineered train data:")
display(train_fe.head())


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Define feature list
feature_cols = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 'weather',
    'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season',
    'num_reported_accidents', 'curvature_speed', 'speed_num_lanes',
    'road_weather_enc', 'time_of_day_sin', 'time_of_day_cos',
    'road_type_mean_risk', 'weather_mean_risk'
]

TARGET = 'accident_risk'

# Features/Target
X = train_fe[feature_cols]
y = train_fe[TARGET]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Prepare datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Params
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1
}

print("ğŸš§ Training LightGBM...")
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)


val_preds = model.predict(X_val, num_iteration=model.best_iteration)
rmse = mean_squared_error(y_val, val_preds, squared=False)

print(f"\nğŸ“Š Validation RMSE: {rmse:.5f}")


import pandas as pd
import matplotlib.pyplot as plt

# Feature importance
importance_df = pd.DataFrame({
    'feature': model.feature_name(),
    'importance': model.feature_importance(importance_type='gain')
}).sort_values(by='importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title('LightGBM Feature Importance (Gain)')
plt.gca().invert_yaxis()
plt.show()

print(importance_df)


import lightgbm as lgb
import pandas as pd

# Step 1: Prepare full dataset
features = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
    'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday',
    'school_season', 'num_reported_accidents',
    'curvature_speed', 'speed_num_lanes', 'weather_mean_risk', 'road_weather_enc',
    'time_of_day_sin', 'time_of_day_cos', 'road_type_mean_risk'
]


X_full = train_fe[features]  
y_full = train_fe['accident_risk']

# Step 2: Create LightGBM dataset
lgb_train_full = lgb.Dataset(X_full, y_full)

# Step 3: Use best params found earlier (replace with your tuned params if any)
best_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'seed': 42,
}

# Step 4: Train final model on full data
final_model = lgb.train(
    best_params,
    lgb_train_full,
    num_boost_round=1000,
    valid_sets=[lgb_train_full],
    callbacks=[
        lgb.callback.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# Step 5: Predict on test set
X_test = test_fe[features]
test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Step 6: Prepare submission file
submission = sample_submission.copy()
submission['accident_risk'] = test_preds


# Step 7: Save submission
submission.to_csv('submission.csv', index=False)

# Display submission head
print(submission.head())

