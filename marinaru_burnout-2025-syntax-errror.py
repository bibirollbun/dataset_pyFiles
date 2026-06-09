import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



import pandas as pd

train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')




train.info()


train.describe()


import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogram of Lap_Time_Seconds (target)
plt.figure(figsize=(8,5))
sns.histplot(train['Lap_Time_Seconds'], kde=True)
plt.title('Distribution of Lap Time')
plt.show()



train.isnull().sum().sort_values(ascending=False)




train['Penalty'].fillna('No Penalty', inplace=True)



from sklearn.preprocessing import LabelEncoder

categorical_cols = [
    'category_x', 'Track_Condition', 'Tire_Compound_Front', 'Tire_Compound_Rear', 
    'Penalty', 'Session', 'shortname', 'circuit_name', 'rider_name', 
    'team_name', 'bike_name', 'weather', 'track'
]

le = LabelEncoder()

for col in categorical_cols:
    train[col] = le.fit_transform(train[col].astype(str))



# Take just 100k rows to test
train_small = train.sample(100000, random_state=42)

# Prepare features and target again
X_small = train_small.drop(['Lap_Time_Seconds', 'Unique ID'], axis=1)
y_small = train_small['Lap_Time_Seconds']



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Split
X_train, X_valid, y_train, y_valid = train_test_split(X_small, y_small, test_size=0.2, random_state=42)

# Model (lower estimators for faster training)
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_valid)

# RMSE
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f'Validation RMSE: {rmse}')



pip install lightgbm



import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
val = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')

# Subsample train data to 100k rows
train_small = train.sample(100000, random_state=42)

# Handle missing Penalty column
train_small['Penalty'] = train_small['Penalty'].fillna('NoPenalty')
val['Penalty'] = val['Penalty'].fillna('NoPenalty')

# Convert all object columns to category (this is KEY fix)
for col in train_small.columns:
    if train_small[col].dtype == 'object':
        train_small[col] = train_small[col].astype('category')
        val[col] = val[col].astype('category')

# Separate features & target
y_train = train_small['Lap_Time_Seconds']
X_train = train_small.drop(['Lap_Time_Seconds', 'Unique ID'], axis=1)

y_val = val['Lap_Time_Seconds']
X_val = val.drop(['Lap_Time_Seconds', 'Unique ID'], axis=1)

# Detect categorical columns automatically
cat_cols = X_train.select_dtypes(include='category').columns.tolist()

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=lgb_train)

# LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

# Train model with early stopping (LightGBM 4.x compatible)
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    callbacks=[early_stopping(stopping_rounds=20)]
)

# Predict & evaluate
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Validation RMSE: {rmse}')



train.head()



import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
val = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')

# Sample 100k rows for faster experimentation
train_small = train.sample(100000, random_state=42)

# Fill missing penalties
train_small['Penalty'] = train_small['Penalty'].fillna('NoPenalty')
val['Penalty'] = val['Penalty'].fillna('NoPenalty')

# Drop useless ID-like columns
drop_cols = [
    'Unique ID', 'Rider_ID', 'rider', 'team', 'bike',
    'shortname', 'circuit_name', 'rider_name',
    'team_name', 'bike_name', 'Session', 'category_x',
    'min_year', 'max_year', 'years_active',
    'starts', 'finishes', 'with_points', 'podiums', 'wins', 
    'Championship_Points', 'Championship_Position', 'Penalty',
    'sequence', 'track', 'weather'
]
train_small = train_small.drop(columns=drop_cols)
val = val.drop(columns=drop_cols)

# Feature Engineering (filtered stronger features only)
for df in [train_small, val]:
    df['Temp_Diff'] = df['Track_Temperature_Celsius'] - df['Ambient_Temperature_Celsius']
    df['Pit_Per_Lap'] = df['Pit_Stop_Duration_Seconds'] / (df['Laps'] + 1)  # avoid division by zero
    df['Deg_x_Laps'] = df['Tire_Degradation_Factor_per_Lap'] * df['Laps']
    df['Pit_x_Deg'] = df['Pit_Stop_Duration_Seconds'] * df['Tire_Degradation_Factor_per_Lap']
    df['Temp_x_Deg'] = df['Temp_Diff'] * df['Tire_Degradation_Factor_per_Lap']

# Separate target
y_train = train_small['Lap_Time_Seconds']
X_train = train_small.drop(['Lap_Time_Seconds'], axis=1)

y_val = val['Lap_Time_Seconds']
X_val = val.drop(['Lap_Time_Seconds'], axis=1)

### RANDOM FOREST — FINAL TUNING ###

X_train_rf = pd.get_dummies(X_train)
X_val_rf = pd.get_dummies(X_val)
X_train_rf, X_val_rf = X_train_rf.align(X_val_rf, join='left', axis=1, fill_value=0)

rf = RandomForestRegressor(
    n_estimators=700, 
    max_depth=28, 
    min_samples_split=4, 
    min_samples_leaf=3,
    random_state=42, 
    n_jobs=-1
)
rf.fit(X_train_rf, y_train)
rf_preds = rf.predict(X_val_rf)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
print(f'Random Forest Validation RMSE: {rf_rmse}')

# Feature Importance Random Forest
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X_train_rf.columns).sort_values(ascending=False)
print("\nTop 15 important features (Random Forest):")
print(feat_imp.head(15))

### LIGHTGBM FINAL TUNING ###

# Handle categorical features
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')

cat_cols = X_train.select_dtypes(include='category').columns.tolist()

lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.015,
    'num_leaves': 120,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'verbose': -1
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=7000,
    callbacks=[early_stopping(stopping_rounds=100)]
)

lgb_preds = model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_preds))
print(f'LightGBM Validation RMSE: {lgb_rmse}')

# LightGBM Feature Importance
lgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.show()

### BLENDING PREDICTIONS ###

# Simple weighted average blending (currently 50-50, you can adjust)
blend_preds = 0.5 * rf_preds + 0.5 * lgb_preds
blend_rmse = np.sqrt(mean_squared_error(y_val, blend_preds))
print(f'Blended Model Validation RMSE: {blend_rmse}')



train.head()



import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
val = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')

# Use small sample (speed)
train_small = train.sample(100000, random_state=42)

# Fill missing penalties
for df in [train_small, val, test]:
    df['Penalty'] = df['Penalty'].fillna('NoPenalty')

# Drop useless ID-like columns
drop_cols = [
    'Rider_ID', 'rider', 'team', 'bike',
    'shortname', 'circuit_name', 'rider_name',
    'team_name', 'bike_name', 'Session', 'category_x',
    'min_year', 'max_year', 'years_active',
    'starts', 'finishes', 'with_points', 'podiums', 'wins', 
    'Championship_Points', 'Championship_Position', 'Penalty',
    'sequence', 'track', 'weather'
]

train_small = train_small.drop(columns=drop_cols + ['Unique ID'])
val = val.drop(columns=drop_cols + ['Unique ID'])
test_features = test.drop(columns=drop_cols)

# Feature Engineering (more powerful now)
for df in [train_small, val, test_features]:
    df['Temp_Diff'] = df['Track_Temperature_Celsius'] - df['Ambient_Temperature_Celsius']
    df['Pit_Per_Lap'] = df['Pit_Stop_Duration_Seconds'] / (df['Laps'] + 1)
    df['Deg_x_Laps'] = df['Tire_Degradation_Factor_per_Lap'] * df['Laps']
    df['Pit_x_Deg'] = df['Pit_Stop_Duration_Seconds'] * df['Tire_Degradation_Factor_per_Lap']
    df['Temp_x_Deg'] = df['Temp_Diff'] * df['Tire_Degradation_Factor_per_Lap']
    df['Laps_x_Temp'] = df['Laps'] * df['Track_Temperature_Celsius']
    df['Pit_Deg_Temp'] = df['Pit_Per_Lap'] * df['Temp_Diff'] * df['Tire_Degradation_Factor_per_Lap']
    # Additional features
    df['Temp_Ratio'] = df['Track_Temperature_Celsius'] / (df['Ambient_Temperature_Celsius'] + 1)
    df['PitSquared'] = df['Pit_Stop_Duration_Seconds'] ** 2
    df['DegSquared'] = df['Tire_Degradation_Factor_per_Lap'] ** 2
    df['LapsSquared'] = df['Laps'] ** 2

# Log-transform target
y_train = np.log1p(train_small['Lap_Time_Seconds'])
X_train = train_small.drop(['Lap_Time_Seconds'], axis=1)
y_val = np.log1p(val['Lap_Time_Seconds'])
X_val = val.drop(['Lap_Time_Seconds'], axis=1)

# Categorical handling for LightGBM
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        test_features[col] = test_features[col].astype('category')

cat_cols = X_train.select_dtypes(include='category').columns.tolist()

# LightGBM Dataset
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=lgb_train)

# Hyperparameters (maxed for small dataset)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.005,
    'num_leaves': 300,
    'min_data_in_leaf': 10,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.9,
    'bagging_freq': 3,
    'lambda_l1': 5.0,
    'lambda_l2': 5.0,
    'max_bin': 1023,
    'verbose': -1
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=20000,
    callbacks=[early_stopping(stopping_rounds=400)]
)

# Evaluate (inverse log transform)
lgb_preds = model.predict(X_val, num_iteration=model.best_iteration)
lgb_preds_actual = np.expm1(lgb_preds)
y_val_actual = np.expm1(y_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val_actual, lgb_preds_actual))
print(f'LightGBM Validation RMSE: {lgb_rmse:.4f}')

# Feature Importance
lgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.show()

# Predict on test set
test_preds = model.predict(test_features.drop(columns=['Unique ID']), num_iteration=model.best_iteration)
test_preds_actual = np.expm1(test_preds)

# Prepare submission
submission = pd.DataFrame({
    'Unique ID': test['Unique ID'],
    
    'Lap_Time_Seconds': test_preds_actual
})
submission.to_csv('submission_v3.csv', index=False)
print("Submission file created successfully!")





