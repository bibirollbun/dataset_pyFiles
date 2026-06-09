# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col=False)

df.columns


def split_data(df, N):
    data_blocks = []

    segment_size = len(df) // N
    
    for i in range(N):
        l = segment_size * i
        r = min(segment_size * (i + 1), len(df))

        data_blocks.append(df[l:r])

    return data_blocks

df_splits = split_data(df, 3)

df_1 = df_splits[0]

pd.plotting.scatter_matrix(df_1[["num_lanes", "curvature"]])


from pandas.plotting import scatter_matrix


df_1.columns


def partition_columns_via_numeracy(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_features.remove("id")

    return numeric_features, non_numeric_features

numeric_features, non_numeric_features = partition_columns_via_numeracy(df_1)


scatter_matrix(df_1[numeric_features], figsize=(12, 12))


df_1[numeric_features].corr()



features = ['num_reported_accidents', 'speed_limit', 'num_lanes']

for feature in features:
  fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                           gridspec_kw={'height_ratios': [1, 3]})

  df[feature].value_counts().sort_index().plot(kind='bar', ax=axes[0],
                                                 edgecolor='black')
  axes[0].set_ylabel('Count')
  axes[0].set_title(f'Distribution of {feature}')
  axes[0].set_xlabel('')

  # Box plot below
  sns.boxplot(data=df, x=feature, y='accident_risk', ax=axes[1])
  axes[1].set_title(f'Accident Risk by {feature}')

  plt.tight_layout()
  plt.show()


def add_indicator_features(df):
    df = df.copy()
    df['lots_of_accidents'] = (df['num_reported_accidents'] >= 3).astype(int)
    df['high_speed_limit'] = (df['speed_limit'] >= 60).astype(int)
    df['curveball_speed'] = df['speed_limit'] * df['curvature']

    return df

df_1_indicatored = add_indicator_features(df_1)


scatter_matrix(df_1[['curvature', 'accident_risk']], figsize=(14, 14), alpha=0.5)


sns.jointplot(data=df_1, x='curvature', y='accident_risk',
                kind='hex', cmap='YlOrRd', height=10)
plt.show()



def elasticnet_first_pass(df):
    """fits linear model using the features so far, returns residual"""
    
    X = df[['curvature', 'lots_of_accidents', 'high_speed_limit']]
    y = df['accident_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

    # Coefficients
    model = pipeline.named_steps['elasticnet']
    print("\nCoefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
        
    print(f"Intercept: {model.intercept_:.4f}")

    return y - pipeline.predict(X)

residuals_1 = elasticnet_first_pass(df_1_indicatored)


sns.jointplot( x=df_1['curvature'], y=residuals_1,
                kind='hex', cmap='YlOrRd', height=10)
plt.show()




df_1['curvature'].corr(residuals_1) # interesting, much less explanative power than we would have thought before!


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have categorical columns
numeric_cols, categorical_cols = partition_columns_via_numeracy(df_1_indicatored)

# One-hot encode categoricals
X = pd.get_dummies(df_1_indicatored.drop('accident_risk', axis=1),
                 columns=categorical_cols, drop_first=True)
y = residuals_1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")


importances = sorted(zip(X.columns, rf.feature_importances_),
                       key=lambda x: x[1], reverse=True)
s = 0
for feature, importance in importances:
  s += importance
  print(f"{feature}: {importance:.4f}")

effect_sizes = []

for feature in X.columns:
  # Get predictions when feature is at different values
  X_temp = X.copy()

  # For binary features (0/1)
  mask_1 = X[feature] == 1
  mask_0 = X[feature] == 0

  if mask_1.sum() > 0 and mask_0.sum() > 0:
      pred_when_1 = rf.predict(X[mask_1]).mean()
      pred_when_0 = rf.predict(X[mask_0]).mean()
      effect_size_sq = (pred_when_1 - pred_when_0) ** 2
  else:
      effect_size_sq = 0

  effect_sizes.append((feature, effect_size_sq))

effect_sizes_sorted = sorted(effect_sizes, key=lambda x: x[1], reverse=True)

for feature, effect_sq in effect_sizes_sorted:
  print(f"{feature}: {effect_sq:.4f}") 


X = pd.get_dummies(df_1_indicatored.drop('accident_risk', axis=1),drop_first=True)

# not clear that all of these do anything

X = X[['lighting_night', 'lighting_dim', 'curvature', 'weather_foggy', 'weather_rainy']]
y = residuals_1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")


 print(df_1_indicatored.columns.tolist()) 





for feature, effect_sq in effect_sizes_sorted:
  print(f"{feature}: {effect_sq:.6f}")


df_1_indicatored['lighting'].value_counts()


# main is lighting_night, weather (rainy/foggy)



# Create temp dataframe with residuals
df_temp = df_1_indicatored.copy()
df_temp['residuals'] = residuals_1

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Lighting
df_temp.boxplot(column='residuals', by='lighting', ax=axes[0])
axes[0].set_title('Residuals by Lighting')
axes[0].set_xlabel('lighting')
axes[0].set_ylabel('residuals')

# Weather
df_temp.boxplot(column='residuals', by='weather', ax=axes[1])
axes[1].set_title('Residuals by Weather')
axes[1].set_xlabel('weather')
axes[1].set_ylabel('residuals')

# Combined
df_temp['lighting_weather'] = df_temp['lighting'] + '_' + df_temp['weather']
df_temp.boxplot(column='residuals', by='lighting_weather', ax=axes[2])
axes[2].set_title('Residuals by Lighting + Weather')
axes[2].set_xlabel('lighting_weather')
axes[2].set_ylabel('residuals')
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle('')
plt.tight_layout()
plt.show()



def add_categorical_indicators(df):
    df = df.copy()
    df['is_night'] = (df['lighting']=='night').astype(int)
    df['poor_weather'] = (df['weather'] != 'clear').astype(int)

    return df

df_1_featured = add_categorical_indicators(df_1_indicatored)


def elasticnet_second_pass(df):
    """fits linear model using the features so far, returns residual"""
    
    X = df[['curvature', 'lots_of_accidents', 'high_speed_limit', 'is_night', 'poor_weather']]
    y = df['accident_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

    # Coefficients
    model = pipeline.named_steps['elasticnet']
    print("\nCoefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
        
    print(f"Intercept: {model.intercept_:.4f}")

    return y - pipeline.predict(X)

residuals_1 = elasticnet_second_pass(df_1_featured)


residuals_1.hist(bins = 30)


X = pd.get_dummies(df_1_featured.drop('accident_risk', axis=1),drop_first=True)

# not clear that all of these do anything

X = X
y = residuals_1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")


X = pd.get_dummies(df_1_featured.drop('accident_risk', axis=1),drop_first=True)

# not clear that all of these do anything

X = X[['lighting_night', 'lighting_dim', 'curvature', 'weather_foggy', 'weather_rainy']]
y = residuals_1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")


import optuna
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def train_ridge_with_optuna(df, features, target_col='accident_risk',
n_trials=50):
    """Fits Ridge regression with Optuna hyperparameter tuning, returns model
and residuals"""

    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

    # Optuna objective
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])

        cv_score = cross_val_score(pipeline, X_train, y_train, cv=5,
scoring='r2').mean()
        return cv_score

    # Optimize
    study = optuna.create_study(direction='maximize',
sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_alpha = study.best_params['alpha']
    print(f"Best alpha: {best_alpha:.4f}")
    print(f"Best CV R²: {study.best_value:.4f}")

    # Train final model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=best_alpha, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    print(f"\nTrain R²: {r2_score(y_train, y_pred_train):.4f}")
    print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_pred_test, squared=False):.4f}")

    # Coefficients
    model = pipeline.named_steps['ridge']
    print("\nCoefficients:")
    coef_sorted = sorted(zip(features, model.coef_), key=lambda x: abs(x[1]),
reverse=True)
    for feature, coef in coef_sorted:
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Quick plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Optuna optimization history
    axes[0].plot([trial.value for trial in study.trials])
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('CV R²')
    axes[0].set_title('Optimization History')
    axes[0].grid(True, alpha=0.3)

    # Predicted vs Actual
    axes[1].scatter(y_test, y_pred_test, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
'r--')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Predicted vs Actual (Test)')
    axes[1].grid(True, alpha=0.3)

    # Residuals distribution
    residuals_test = y_test - y_pred_test
    axes[2].hist(residuals_test, bins=30, edgecolor='black')
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Residual Distribution (Test)')
    axes[2].axvline(0, color='r', linestyle='--')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Return full predictions on all data and residuals
    y_pred_all = pipeline.predict(X)
    residuals = y - y_pred_all

    return pipeline, residuals


def train_rf_on_residuals(df, residuals, features, max_depth=5,
n_estimators=100):
    """Trains RF on residuals with specified features"""

    # One-hot encode if needed
    X_encoded = pd.get_dummies(df.drop('accident_risk', axis=1),
drop_first=True)
    X = X_encoded[features]
    y = residuals

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=20,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = np.clip(rf.predict(X_test), 0, 1)

    print(f"RF on Residuals:")
    print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

    # Feature importances
    print("\nFeature Importances:")
    importances = sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    for feature, importance in importances:
        print(f"  {feature}: {importance:.4f}")

    return rf


def benchmark_model(model, df, features, target_col='accident_risk'):
    """Benchmarks a trained model on a dataframe"""

    X = df[features]
    y = df[target_col]

    y_pred = np.clip(model.predict(X), 0, 1)

    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = np.mean(np.abs(y - y_pred))

    print(f"Benchmark Results:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # Quick residual plot
    residuals = y - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {'r2': r2, 'rmse': rmse, 'mae': mae}

df_featured = add_categorical_indicators(add_indicator_features(df))

# (1) Ridge with Optuna
features_ridge = ['curvature', 'lots_of_accidents', 'high_speed_limit',
'is_night', 'poor_weather']
ridge_model, residuals = train_ridge_with_optuna(df_featured, features_ridge)

# (2) RF on residuals
#features_rf = ['lighting_night', 'lighting_dim', 'curvature','weather_foggy', 'weather_rainy']
#rf_model = train_rf_on_residuals(df_1_featured, residuals, features_rf, max_depth=10)

# Benchmark
benchmark_model(ridge_model, df_featured, features_ridge)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# Apply same feature engineering you did on training data
# (you'll need to replicate whatever transformations you did to create df_1_featured)

test_df = add_categorical_indicators(add_indicator_features(test_df))

# Then predict
features_ridge = ['curvature', 'lots_of_accidents', 'high_speed_limit', 'is_night', 'poor_weather']
predictions = np.clip(ridge_model.predict(test_df[features_ridge]), 0, 1)

# Create submission
submission = pd.DataFrame({
  'id': test_df['id'],
  'accident_risk': predictions
})
submission.to_csv('submission.csv', index=False)


df_featured.columns




