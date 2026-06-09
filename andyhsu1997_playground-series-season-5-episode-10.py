# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import optuna # Import Optuna

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cb
from sklearn.model_selection import KFold, train_test_split # Import KFold and train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# df_100k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
# df_10k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
# df_2k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')

# Remove original ID columns
df_train = df_train.drop('id', axis = 1)
df_test =df_test.drop('id', axis = 1)

# Concatenate all training data
# df_combined = pd.concat([df_train, df_100k, df_10k, df_2k], ignore_index=True)


print("\n--- Combined Dataset Info ---")
df_train.info()

print("\n--- Numerical Features Statistics ---")
print(df_train.describe())

print("\n--- Missing Values Check ---")
print(df_train.isnull().sum())


# Target variable distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_train['accident_risk'], kde=True)
plt.title('Distribution of Accident Risk')
plt.subplot(1, 2, 2)
sns.boxplot(y=df_train['accident_risk'])
plt.title('Box Plot of Accident Risk')
plt.show()

# Select numerical and boolean features for correlation matrix
corr_features_df = df_train.select_dtypes(include=['number', 'bool'])
plt.figure(figsize=(12, 10))
correlation_matrix = corr_features_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


def road_risk(X):
    return (
        0.3 * X["curvature"] +
        0.2 * (X["lighting"] == "night").astype(int) +
        0.1 * (X["weather"] != "clear").astype(int) +
        0.2 * (X["speed_limit"] >= 60).astype(int) +
        0.1 * (X["num_reported_accidents"] > 2).astype(int)
    )

def clipped(func):
    def clip_f(X):
        mu = func(X)
        sigma = 0.05 
        a, b = -mu / sigma, (1 - mu) / sigma
        Phi_a, Phi_b = stats.norm.cdf(a), stats.norm.cdf(b)
        phi_a, phi_b = stats.norm.pdf(a), stats.norm.pdf(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
    return clip_f

df_train["base_prior"] = clipped(road_risk)(df_train)
df_test["base_prior"] = clipped(road_risk)(df_test)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the accident risk prediction dataset.
    """
    df_new = df.copy()

    # Check if 'num_reported_accidents' exists
    df_new['log_accidents'] = np.log1p(df_new['num_reported_accidents'])

    df_new['curvature_squared'] = df_new['curvature'] ** 2
    df_new['speed_x_curvature'] = df_new['speed_limit'] * df_new['curvature']
    df_new["lanes_x_speed"] = df_new["num_lanes"] * df_new["speed_limit"]
    df_new['lanes_curvature_ratio'] = df_new['curvature'] / df_new["num_lanes"]
    df_new["curv_speed_ratio"] = df_new["curvature"] / (df_new["speed_limit"] + 1e-6)
    df_new["speed_sq_over_curv"] = (df_new["speed_limit"] ** 2) / (df_new["curvature"] + 1e-6)
    df_new["curv_factor"] = df_new["curvature"] * (df_new["num_lanes"] / (df_new["speed_limit"] + 1.0))

    df_new['poor_visibility'] = (df_new['lighting'].isin(['night', 'dim']) | (df_new['weather'] == 'foggy'))
    df_new['adverse_weather'] = df_new['weather'].isin(['rainy', 'foggy'])
    df_new['school_traffic_time'] = ((df_new['school_season'] == True) & (df_new['time_of_day'].isin(['morning', 'afternoon'])))
    df_new['rush_hour_no_holiday'] = ((df_new['holiday'] == False) & (df_new['time_of_day'].isin(['morning', 'afternoon'])))
    df_new["weekend_or_hol"] = ((df_new["holiday"].astype(bool)) | (df_new["time_of_day"].isin(["evening", "night"])))
    df_new['dangerous_curve_no_signs'] = ((df_new['curvature'] > 0.5) & (df_new['road_signs_present'] == False))

    df_new['speed_in_poor_visibility'] = df_new['speed_limit'] * df_new['poor_visibility'].astype(int)
    df_new['curve_in_poor_visibility'] = df_new['curvature'] * df_new['poor_visibility'].astype(int)
    df_new['speed_in_adverse_weather'] = df_new['speed_limit'] * df_new['adverse_weather'].astype(int)

    weather_ord_map = {"clear": 0, "foggy": 1, "rainy": 2}
    lighting_ord_map = {"daylight": 0, "dim": 1, "night": 2}
    df_new["weather_ord"] = df_new["weather"].map(weather_ord_map).fillna(0).astype(int)
    df_new["lighting_ord"] = df_new["lighting"].map(lighting_ord_map).fillna(0).astype(int)
    df_new.drop('weather_ord', axis = 1)
    df_new["light_weather_code"] = df_new["lighting_ord"] * 10 + df_new["weather_ord"]

    df_new["acc_density"] = df_new["num_reported_accidents"].astype(float) / (df_new["speed_limit"] + 1.0)
    df_new["acc_rate_per_lane"] = df_new["num_reported_accidents"].astype(float) / (df_new["num_lanes"] + 0.01)
    df_new["hotspot_flag"] = (df_new["num_reported_accidents"] >= 2)


    # 1. 基礎風險 vs 曲率
    # 這個特徵告訴模型："base_prior" 的影響力在不同的 "curvature" 下是不同的。
    # 由於 curvature=0.6 是漂移區，模型會學到當 curvature=0.6 時，這個特徵的值對風險有特定影響。
    df_new['prior_x_curvature'] = df_new['base_prior'] * df_new['curvature']
    
    # 2. 基礎風險 vs 速限 (使用 one-hot)
    # 告訴模型："base_prior" 在 "speed_limit" = 45 (我們知道的漂移值) 時，
    # 其影響力與在其他速限時不同。
    df_new['prior_x_speed_45'] = df_new['base_prior'] * (df_new['speed_limit'] == 45).astype(int)
    df_new['prior_x_speed_35'] = df_new['base_prior'] * (df_new['speed_limit'] == 35).astype(int)
    
    # 3. 基礎風險 vs 事故數
    # 告訴模型："base_prior" 在 "num_reported_accidents" > 2 (我們知道的漂移區) 時，
    # 其影響力與其他時候不同。
    df_new['prior_x_accidents_gt_2'] = df_new['base_prior'] * (df_new['num_reported_accidents'] > 2).astype(int)

    print("Feature engineering complete!")
    return df_new


df_train = create_features(df_train)
df_test = create_features(df_test)

target_variable = 'accident_risk'

X = df_train.drop([target_variable], axis = 1)
y = df_train[target_variable]

categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Data preparation complete")
print("\nCategorical features list:")
print(categorical_cols)

df_test = df_test.reindex(columns=X.columns, fill_value=0)


# --- Control Switches ---
USE_OPTUNA = True # Set to True to enable Optuna, False to use default params
N_TRIALS = 50     # Number of Optuna trials (if enabled)
OPTUNA_TIMEOUT = 60 * 600 # Max time for Optuna search (in seconds), e.g., 30 minutes
N_SPLITS = 10

# --- Optuna Objective Function (with internal K-Fold) ---
def objective(trial):
    params = {
        'task_type': 'GPU',
        'devices': '0',
        'bootstrap_type': 'Bayesian',
        'iterations': trial.suggest_int('iterations', 1000, 3000, 50), # Iterations for Optuna trial
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'cat_features': categorical_cols,
        'verbose': 0,
        'random_seed': 42,
        'early_stopping_rounds': 100
    }
    
    # Perform cross-validation within the Optuna trial
    kf_opt = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1337) # Use a different random state for optuna CV
    rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf_opt.split(X, y)):
        X_train_opt, X_val_opt = X.iloc[train_idx], X.iloc[val_idx]
        y_train_opt, y_val_opt = y.iloc[train_idx], y.iloc[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(X_train_opt, y_train_opt, eval_set=(X_val_opt, y_val_opt), 
                  early_stopping_rounds=params['early_stopping_rounds'], verbose=0)
        
        y_pred_val = model.predict(X_val_opt)
        rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred_val))
        rmses.append(rmse)
        
    # Return the mean RMSE of the 5 folds
    return np.mean(rmses)

# --- Parameter Selection ---
if USE_OPTUNA:
    print("\nStarting Optuna hyperparameter search (using 5-Fold CV internally)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)

    print(f"Optuna search complete! Best average RMSE: {study.best_value:.4f}")
    print("Best parameters found:")
    print(study.best_params)
    final_params = study.best_params
else:
    print("\nUsing default parameters...")
    final_params = {
        'learning_rate': 0.04,
        'depth': 8,
        'l2_leaf_reg': 0.0144,
        'bagging_temperature': 0.0965,
    }

# --- Final Cross-Validation Training ---
print(f"\nStarting final {N_SPLITS}-Fold training with best parameters...")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_predictions = np.zeros(X.shape[0])
test_predictions_list = []
models = []

base_params = {
    'task_type': 'GPU',
    'devices': '0',
    'bootstrap_type': 'Bayesian',
    'iterations': 4000, # Final training iterations
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'cat_features': categorical_cols,
    'verbose': 500,
    'random_seed': 42,
    'early_stopping_rounds': 200
}

# Update base params with the core parameters found by Optuna/default
base_params.update(final_params)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = cb.CatBoostRegressor(**base_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Store validation predictions (OOF)
    oof_predictions[val_idx] = model.predict(X_val)
    
    # Store test set predictions
    test_predictions_list.append(model.predict(df_test))
    
    # Store model
    models.append(model)

print("\nCross-validation training complete!")


print("\nOverall Cross-Validation Evaluation (OOF)...")
oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
oof_r2 = r2_score(y, oof_predictions)

print(f"OOF RMSE: {oof_rmse:.4f}")
print(f"OOF R² Score: {oof_r2:.4f}")


print("\nGenerating Average Feature Importance plot...")
feature_importance_df = pd.DataFrame()
feature_importance_df['Feature'] = X.columns
total_importance = np.zeros(X.shape[1])

for model in models:
    total_importance += model.get_feature_importance()

feature_importance_df['Importance'] = total_importance / N_SPLITS
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Average Feature Importance (5-Fold CV)', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()


print("\nAveraging test predictions...")
avg_test_predictions = np.mean(test_predictions_list, axis=0)

print("\nGenerating submission file...")
submission_df = pd.DataFrame({
    'id': submission['id'],
    'accident_risk': avg_test_predictions
})

# Clip predictions to the [0, 1] range
submission_df['accident_risk'] = np.clip(submission_df['accident_risk'], 0, 1)

submission_df.to_csv('submission.csv', index=False)
print("Submission file submission.csv has been generated.")

