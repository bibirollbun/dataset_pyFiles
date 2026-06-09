import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  
plt.style.use('ggplot')

import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import optuna

from math import sqrt




train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
train_df.head()


train_df.info()


train_df.describe()


missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()
print("Missing Values in Train:\n", missing_train[missing_train > 0])
print("Missing Values in Test:\n", missing_test[missing_test > 0])


numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
categorical_cols = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']
target = 'accident_risk'


def plot_target(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    values = df[target].values
    hist, bins = np.histogram(values, bins=20)
    width = np.diff(bins)
    ax.bar(bins[:-1], hist, width=width, color=cm.plasma(np.linspace(0, 1, len(hist))))  # Plasma for uniqueness
    ax.set_title('Distribution of Accident Risk')
    ax.set_xlabel('Accident Risk')
    ax.set_ylabel('Frequency')
    plt.show()

plot_target(train_df)


def plot_univariate_num(col, df, title_prefix="Train"):
    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(df[col], bins=20, edgecolor='black')
    
    
    fracs = n / n.max()
    norm = plt.Normalize(fracs.min(), fracs.max())
    for frac, patch in zip(fracs, patches):
        color = cm.hot(norm(frac))  
        patch.set_facecolor(color)
    
    ax.set_title(f'{title_prefix} Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    plt.show()


for col in numerical_cols:
    plot_univariate_num(col, train_df)




def plot_univariate_cat(col, df, title_prefix="Train"):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[col].value_counts().sort_values(ascending=False)
    ax.bar(counts.index, counts.values, color=cm.viridis(np.linspace(0, 1, len(counts))))  # Gradient colors
    ax.set_title(f'{title_prefix} Counts of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


for col in categorical_cols:
    plot_univariate_cat(col, train_df)


def plot_bivariate_scatter(x_col, y_col, df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_col], df[y_col], alpha=0.5, c='blue', edgecolor='black')  
   
    x = df[x_col].values
    y = df[y_col].values
    mask = ~np.isnan(x) & ~np.isnan(y)  # Handle any NaNs
    z = np.polyfit(x[mask], y[mask], 1)
    xp = np.linspace(x.min(), x.max(), 100)
    yp = np.polyval(z, xp)
    ax.plot(xp, yp, 'r--', label='Trend Line')
    ax.set_title(f'{x_col} vs {y_col} with Trend Line')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    plt.show()





plot_bivariate_scatter('speed_limit', target, train_df)

plot_bivariate_scatter('num_reported_accidents', target, train_df)

plot_bivariate_scatter('curvature', target, train_df)


def plot_bivariate_box(cat_col, num_col, df):
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = df[cat_col].unique()
    data = [df[df[cat_col] == cat][num_col] for cat in categories]
    ax.boxplot(data, labels=categories, patch_artist=True, 
               boxprops=dict(facecolor='lightblue', color='blue'),
               whiskerprops=dict(color='red'))  # Custom colors for uniqueness
    ax.set_title(f'{num_col} by {cat_col}')
    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    plt.xticks(rotation=45)
    plt.show()


for cat in ['lighting', 'weather', 'time_of_day', 'holiday']:
    plot_bivariate_box(cat, target, train_df)



num_df = train_df[[target] + numerical_cols ]
corr = num_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='coolwarm')  
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticklabels(corr.columns)


for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                       ha="center", va="center", color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")  # Adaptive text color

plt.colorbar(im)
ax.set_title('Enhanced Correlation Heatmap with Values')
plt.show()


top_corr = corr[target].abs().sort_values(ascending=False)[1:4]  
print("Top Features Correlated with Accident Risk:\n", top_corr)


def compare_distributions(col, train, test):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train[col], bins=20, alpha=0.5, label='Train', color='blue')
    ax.hist(test[col], bins=20, alpha=0.5, label='Test', color='orange')
    ax.set_title(f'Distribution Comparison: {col}')
    ax.legend()
    plt.show()

for col in numerical_cols:
    compare_distributions(col, train_df, test_df)




for col in categorical_cols:
    train_counts = train_df[col].value_counts(normalize=True)
    test_counts = test_df[col].value_counts(normalize=True)
    
   
    categories = sorted(set(train_counts.index) | set(test_counts.index))
    
    
    train_values = [train_counts.get(cat, 0) for cat in categories]
    test_values = [test_counts.get(cat, 0) for cat in categories]
    
    
    x = np.arange(len(categories))  
    width = 0.3  
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, train_values, width, label='Train', color='blue')
    ax.bar(x + width/2, test_values, width, label='Test', color='orange')
    
    ax.set_ylabel('Normalized Counts')
    ax.set_title(f'Normalized Counts Comparison: {col}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()
    
    plt.show()


test_ids = test_df['id']
train_df.drop('id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)

target = 'accident_risk'
X_train = train_df.drop(target, axis=1)
y_train = train_df[target]
X_test = test_df.copy()  


# Convert booleans to int
binary_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in binary_cols:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# Ordinal mapping for lighting (assuming dim < daylight for visibility)
lighting_map = {'dim': 0, 'daylight': 1}
X_train['lighting_ord'] = X_train['lighting'].map(lighting_map)
X_test['lighting_ord'] = X_test['lighting'].map(lighting_map)

# Ordinal for weather based on risk (clear=0, rainy=1, foggy=2)
weather_map = {'clear': 0, 'rainy': 1, 'foggy': 2}
X_train['weather_ord'] = X_train['weather'].map(weather_map)
X_test['weather_ord'] = X_test['weather'].map(weather_map)

# Cyclical encoding for time_of_day (morning, afternoon, evening) - treat as cycle
time_map = {'morning': 0, 'afternoon': 1, 'evening': 2}
X_train['time_ord'] = X_train['time_of_day'].map(time_map)
X_test['time_ord'] = X_test['time_of_day'].map(time_map)
# Add sin/cos for cyclical
X_train['time_sin'] = np.sin(2 * np.pi * X_train['time_ord'] / 3)
X_train['time_cos'] = np.cos(2 * np.pi * X_train['time_ord'] / 3)
X_test['time_sin'] = np.sin(2 * np.pi * X_test['time_ord'] / 3)
X_test['time_cos'] = np.cos(2 * np.pi * X_test['time_ord'] / 3)

# One-hot encoding for road_type (nominal)
road_type_dummies = pd.get_dummies(pd.concat([X_train['road_type'], X_test['road_type']]), prefix='road_type')
X_train = pd.concat([X_train, road_type_dummies.iloc[:len(X_train)]], axis=1)
X_test = pd.concat([X_test, road_type_dummies.iloc[len(X_train):]], axis=1)

# Drop original categoricals after encoding
drop_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)


# Log transform for num_reported_accidents (counts, skewed likely)
X_train['num_accidents_log'] = np.log1p(X_train['num_reported_accidents'])
X_test['num_accidents_log'] = np.log1p(X_test['num_reported_accidents'])

# Polynomial features for curvature and speed_limit
X_train['curvature_sq'] = X_train['curvature'] ** 2
X_test['curvature_sq'] = X_test['curvature'] ** 2
X_train['speed_sq'] = X_train['speed_limit'] ** 2
X_test['speed_sq'] = X_test['speed_limit'] ** 2

# Binning curvature into low, med, high (assuming from EDA, thresholds based on sample)
bins = [0, 0.3, 0.6, 1.0]
labels = ['low_curve', 'med_curve', 'high_curve']
X_train['curvature_bin'] = pd.cut(X_train['curvature'], bins=bins, labels=labels)
X_test['curvature_bin'] = pd.cut(X_test['curvature'], bins=bins, labels=labels)
# One-hot the bins
curve_dummies = pd.get_dummies(pd.concat([X_train['curvature_bin'], X_test['curvature_bin']]), prefix='curve')
X_train = pd.concat([X_train, curve_dummies.iloc[:len(X_train)]], axis=1)
X_test = pd.concat([X_test, curve_dummies.iloc[len(X_train):]], axis=1)
X_train.drop('curvature_bin', axis=1, inplace=True)
X_test.drop('curvature_bin', axis=1, inplace=True)


# Interaction: curvature * speed_limit (higher speed on curves = risk)
X_train['curve_speed_interact'] = X_train['curvature'] * X_train['speed_limit']
X_test['curve_speed_interact'] = X_test['curvature'] * X_test['speed_limit']

# Bad weather flag (rainy or foggy)
X_train['bad_weather'] = np.where(X_train['weather_ord'] > 0, 1, 0)
X_test['bad_weather'] = np.where(X_test['weather_ord'] > 0, 1, 0)

# Dim lighting flag
X_train['dim_lighting'] = np.where(X_train['lighting_ord'] == 0, 1, 0)
X_test['dim_lighting'] = np.where(X_test['lighting_ord'] == 0, 1, 0)

# Rush hour: morning or evening
X_train['rush_hour'] = np.where(X_train['time_ord'].isin([0, 2]), 1, 0)
X_test['rush_hour'] = np.where(X_test['time_ord'].isin([0, 2]), 1, 0)

# Busy period: holiday or school season
X_train['busy_period'] = X_train['holiday'] | X_train['school_season']
X_test['busy_period'] = X_test['holiday'] | X_test['school_season']

# No signs flag
X_train['no_signs'] = 1 - X_train['road_signs_present']
X_test['no_signs'] = 1 - X_test['road_signs_present']

# Complex interaction: bad conditions index (bad_weather * dim_lighting * rush_hour * curvature)
X_train['bad_cond_index'] = X_train['bad_weather'] * X_train['dim_lighting'] * X_train['rush_hour'] * X_train['curvature']
X_test['bad_cond_index'] = X_test['bad_weather'] * X_test['dim_lighting'] * X_test['rush_hour'] * X_test['curvature']

# Lanes per speed (density proxy)
X_train['lanes_per_speed'] = X_train['num_lanes'] / (X_train['speed_limit'] + 1)  # +1 avoid divide by 0
X_test['lanes_per_speed'] = X_test['num_lanes'] / (X_test['speed_limit'] + 1)


print("New Train Columns:", X_train.columns.tolist())


try:
    default_tree_method
except NameError:
    default_tree_method = 'gpu_hist'  

try:
    use_device_param
except NameError:
    use_device_param = False
    device_value = None


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, shuffle=True
)


import optuna
import lightgbm as lgb
import numpy as np
import warnings
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
    }

    
    try:
        import torch
        if torch.cuda.is_available():
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
    except:
        pass

    n_estimators = trial.suggest_int('n_estimators', 500, 2000)

    model = lgb.LGBMRegressor(**params, n_estimators=n_estimators)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(0)
        ]
    )

    val_preds = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, val_preds))
    
    
    complexity_penalty = 1 + 0.001 * params['num_leaves']
    return rmse * complexity_penalty


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("âœ… Optuna finished")
print("Best trial value (RMSE on val):", study.best_value)
print("Best params:", study.best_params)


best_params = study.best_params.copy()
n_estimators_best = best_params.pop('n_estimators')

lgb1_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'n_jobs': -1,
    **best_params
}


lgb2_params = lgb1_params.copy()

if 'learning_rate' in lgb2_params:
    lgb2_params['learning_rate'] = lgb2_params['learning_rate'] * 1.1
if 'num_leaves' in lgb2_params:
    lgb2_params['num_leaves'] = max(31, int(lgb2_params['num_leaves'] * 0.9))
if 'subsample' in lgb2_params:
    lgb2_params['subsample'] = min(1.0, lgb2_params['subsample'] + 0.05)


lgb3_params = lgb1_params.copy()
lgb3_params['boosting_type'] = 'dart' 

print("Training multiple LightGBM models...")


models = []
val_predictions = []
test_predictions = []

for i, params in enumerate([lgb1_params, lgb2_params, lgb3_params]):
    print(f"Training model {i+1}...")
    
    model = lgb.LGBMRegressor(**params, n_estimators=n_estimators_best)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    rmse = sqrt(mean_squared_error(y_val, val_pred))
    print(f"Model {i+1} RMSE on val: {rmse:.6f}")
    
    models.append(model)
    val_predictions.append(val_pred)
    test_predictions.append(test_pred)

# Optimisation des poids d'ensemble avec recherche plus sophistiquÃ©e
print("Optimizing ensemble weights...")

best_rmse = float('inf')
best_weights = None
best_ensemble_pred = None

# Recherche avec contrainte de somme = 1
n_models = len(val_predictions)
for w1 in np.linspace(0, 1, 51):
    for w2 in np.linspace(0, 1 - w1, 51):
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
            
        val_ensemble = (w1 * val_predictions[0] + 
                       w2 * val_predictions[1] + 
                       w3 * val_predictions[2])
        
        rmse = sqrt(mean_squared_error(y_val, val_ensemble))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w1, w2, w3)
            best_ensemble_pred = (w1 * test_predictions[0] + 
                                 w2 * test_predictions[1] + 
                                 w3 * test_predictions[2])

print("=" * 60)
print(f"Best weights -> Model1: {best_weights[0]:.3f}, Model2: {best_weights[1]:.3f}, Model3: {best_weights[2]:.3f}")
print(f"Ensemble RMSE on val: {best_rmse:.6f}")
print("=" * 60)

# Validation croisÃ©e supplÃ©mentaire avec les meilleurs paramÃ¨tres
print("Performing cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X_tr):
    X_train_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
    y_train_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
    
    model_cv = lgb.LGBMRegressor(**lgb1_params, n_estimators=n_estimators_best)
    model_cv.fit(X_train_fold, y_train_fold)
    
    preds_cv = model_cv.predict(X_val_fold)
    rmse_cv = sqrt(mean_squared_error(y_val_fold, preds_cv))
    cv_scores.append(rmse_cv)

print(f"CV RMSE: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

# Final prediction
final_test_pred = best_ensemble_pred


best_test = np.clip(final_test_pred, 0.0, 1.0)

submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': best_test
})

assert submission.shape[0] == X_test.shape[0]
assert submission['accident_risk'].isna().sum() == 0
assert (submission['accident_risk'] >= 0).all()
assert (submission['accident_risk'] <= 1).all()

submission.to_csv('/kaggle/working/submission.csv', index=False)

print("âœ… Submission Created Successfully")
print(f"Shape: {submission.shape}")
print(f"Prediction Mean: {submission['accident_risk'].mean():.4f}")
print(f"Prediction Std: {submission['accident_risk'].std():.4f}")
print(f"Prediction Min: {submission['accident_risk'].min():.4f}")
print(f"Prediction Max: {submission['accident_risk'].max():.4f}")
print("\nFirst 10 predictions:")
print(submission.head(10))


