import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna
import shap
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
train_ = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")
train


train = train.drop(["id"],axis =1)
train_



train = pd.concat([train, train_], ignore_index=True)
train


epsilon = 1e-6 
train['Acoustic_to_Instrumental_Ratio'] = train['AcousticQuality'] / (train['InstrumentalScore'] + 0.001)
train['Energy_x_Rhythm'] = train['Energy'] * train['RhythmScore']
train['Loudness_per_Second'] = train['AudioLoudness'] / (train['TrackDurationMs'] / 1000)
train['Danceability_Proxy'] = train['Energy'] * train['RhythmScore'] * (train['AudioLoudness'] - train['AudioLoudness'].min())
train['Vocal_Prominence'] = train['VocalContent'] / (train['InstrumentalScore'] + 0.001)
train['Energy_Acoustic_Ratio'] = train['Energy'] / (train['AcousticQuality'] + epsilon)
train['MoodRhythm'] = train['MoodScore'] * train['RhythmScore']
train['PerformanceIntensity'] = train['LivePerformanceLikelihood'] * train['AudioLoudness']
train['MoodAcoustic'] = train['MoodScore'] * train['AcousticQuality']


test['Acoustic_to_Instrumental_Ratio'] = test['AcousticQuality'] / (test['InstrumentalScore'] + 0.001)
test['Energy_x_Rhythm'] = test['Energy'] * test['RhythmScore']
test['Loudness_per_Second'] = test['AudioLoudness'] / (test['TrackDurationMs'] / 1000)
test['Danceability_Proxy'] = test['Energy'] * test['RhythmScore'] * (test['AudioLoudness'] - test['AudioLoudness'].min())
test['Vocal_Prominence'] = test['VocalContent'] / (test['InstrumentalScore'] + 0.001)
test['Energy_Acoustic_Ratio'] = test['Energy'] / (test['AcousticQuality'] + epsilon)
test['MoodRhythm'] = test['MoodScore'] * test['RhythmScore']
test['PerformanceIntensity'] = test['LivePerformanceLikelihood'] * test['AudioLoudness']
test['MoodAcoustic'] = test['MoodScore'] * test['AcousticQuality']


train.info()


train.isna().sum()


train.describe()


sns.set(style="whitegrid")
colors = sns.color_palette("husl", len(train.columns))

plt.figure(figsize=(25, 20))
for i, (col, color) in enumerate(zip(train.columns, colors), 1):
    plt.subplot(len(train.columns) // 3 + 1, 3, i)
    sns.histplot(train[col], bins=15, kde=True, color=color)
    plt.title(f'Distribution of {col}', color=color)
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 15))
for i, col in enumerate(train.columns):
    plt.subplot(len(train.columns) // 2 + 1, 2, i + 1)
    color = 'purple' if i % 2 == 0 else 'orange'
    sns.boxplot(x=train[col], color=color)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


corr_matrix = train.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


correlation_matrix = train.corr()
correlation_with_response = correlation_matrix['BeatsPerMinute'].sort_values(ascending=False)
print(correlation_with_response)


# --- Logarithmic Transformation ---
skewed_features = [ 'Danceability_Proxy','Vocal_Prominence','Energy_Acoustic_Ratio','RhythmScore',
                   'Loudness_per_Second']

# Apply the log1p transformation on train
for feature in skewed_features:
    train[feature] = np.log1p(train[feature])

# Apply the log1p transformation on test
for feature in skewed_features:
    test[feature] = np.log1p(test[feature])

print("Applied log transformation features.")


# --- Power Transformation ---
power_features = [ 'LivePerformanceLikelihood','MoodAcoustic','TrackDurationMs','InstrumentalScore',
                   'Energy_x_Rhythm','Energy']

power_transformer = PowerTransformer(method='yeo-johnson')

# # Fit the transformer on the training data and then transform it
train[power_features] = power_transformer.fit_transform(train[power_features])

# # Apply the transformation on test
test[power_features] = power_transformer.transform(test[power_features])

print("Applied power transformation features.")


train.columns


numerical_features= [ 'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 
                     'LivePerformanceLikelihood', 'MoodScore','TrackDurationMs', 'Energy',
                     'Acoustic_to_Instrumental_Ratio', 'Energy_x_Rhythm','Loudness_per_Second', 'Danceability_Proxy',
                     'Vocal_Prominence','Energy_Acoustic_Ratio', 'MoodRhythm', 'PerformanceIntensity','MoodAcoustic']
# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('robust_scale', RobustScaler(), numerical_features)
    ],
    remainder='passthrough'
)


X_raw =  train.drop(['BeatsPerMinute'], axis =1)
y = train['BeatsPerMinute']


X_test_raw = test.drop(['id'],axis =1)
X_test_raw


pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
X_processed = pipeline.fit_transform(X_raw)
X_test_processed = pipeline.transform(X_test_raw)


X = pd.DataFrame(X_processed, columns =numerical_features )
X_test = pd.DataFrame(X_test_processed, columns =numerical_features )
X


lgbm_best_params={'n_estimators': 1307,          
    'learning_rate': 0.06748646663694965,
    'num_leaves': 9,
    'max_depth': 8,
    'min_child_samples': 460,
    'subsample': 0.8278422593438073,
    'colsample_bytree': 0.9945980359117047,
    'reg_alpha': 0.00042155616855236246,
    'reg_lambda': 0.02788082464431462,
    'verbose': -1,
    'n_jobs': -1,
    
            
}


xg_best_params={'n_estimators': 1254,            #26.45879631057674
 'learning_rate': 0.014319749589313719,
 'max_depth': 6,
 'min_child_weight': 6,
 'subsample': 0.7355672133971793,
 'colsample_bytree': 0.9793951859907187,
 'lambda': 73.76086293476641,
 'alpha': 0.00010667965771233884}


cat_best_params={'bootstrap_type': 'Poisson',
 'iterations': 551,
 'learning_rate': 0.067306222661126,
 'depth': 4,
 'subsample': 0.9992492093977058,
 'l2_leaf_reg': 6.815365092983756,
 'min_data_in_leaf': 33}


# Load sample submission
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# Initialize KFold
kf = KFold(n_splits=15, shuffle=True, random_state=67)

# Arrays to store predictions & validation scores
test_pred_cat = np.zeros(len(X_test))
test_pred_xgb = np.zeros(len(X_test))
test_pred_lgb = np.zeros(len(X_test))

rmse_cat, rmse_xgb, rmse_lgb = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- CatBoost ---
    model_cat = cb.CatBoostRegressor(**cat_best_params, random_seed=67, verbose=0, task_type="GPU")
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=0)
    y_val_pred = model_cat.predict(X_val)
    rmse_cat.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    test_pred_cat += model_cat.predict(X_test) / kf.get_n_splits()

    # --- XGBoost ---
    model_xgb = xgb.XGBRegressor(**xg_best_params, random_state=67, tree_method="gpu_hist")
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=0)
    y_val_pred = model_xgb.predict(X_val)
    rmse_xgb.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    test_pred_xgb += model_xgb.predict(X_test) / kf.get_n_splits()

    # --- LightGBM ---
    model_lgb = lgb.LGBMRegressor(**lgbm_best_params, random_state=67, device="gpu")
    model_lgb.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    y_val_pred = model_lgb.predict(X_val)
    rmse_lgb.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    test_pred_lgb += model_lgb.predict(X_test) / kf.get_n_splits()

# --- Compute average CV RMSE for each model ---
mean_rmse_cat = np.mean(rmse_cat)
mean_rmse_xgb = np.mean(rmse_xgb)
mean_rmse_lgb = np.mean(rmse_lgb)

print("\nValidation RMSE:")
print(f"CatBoost: {mean_rmse_cat:.5f}")
print(f"XGBoost:  {mean_rmse_xgb:.5f}")
print(f"LightGBM: {mean_rmse_lgb:.5f}")

# --- Convert RMSE to weights (lower RMSE â†’ higher weight) ---
inv_errors = np.array([1/mean_rmse_cat, 1/mean_rmse_xgb, 1/mean_rmse_lgb])
weights = inv_errors / inv_errors.sum()

print("\nEnsemble Weights:")
print(f"CatBoost: {weights[0]:.3f}, XGBoost: {weights[1]:.3f}, LightGBM: {weights[2]:.3f}")

# --- Weighted average of test predictions ---
final_test_pred = (
    weights[0] * test_pred_cat +
    weights[1] * test_pred_xgb +
    weights[2] * test_pred_lgb
)

# --- Create and save submission ---
submission_df = pd.DataFrame({
    "id": sample_submission["id"],
    "BeatsPerMinute": final_test_pred
})

submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file created with weighted ensemble.")
submission_df.head()


