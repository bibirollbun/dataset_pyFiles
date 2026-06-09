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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
import optuna

import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set_style("darkgrid")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

print("-" * 50)
display(train_df.head())

display(test_df.head())

display(train_df.describe().T)


train_df.info()


plt.figure(figsize=(12, 6))
sns.histplot(train_df['BeatsPerMinute'], kde=True, bins=50)
plt.title('Distribution of BeatsPerMinute (BPM)', fontsize=15)
plt.xlabel('BPM')
plt.ylabel('Frequency')
plt.show()


features = train_df.drop(columns=['id', 'BeatsPerMinute']).columns
plt.figure(figsize=(16, 12))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1) # Creating a 3x3 grid of plots
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 10))
# Calculate the correlation matrix
correlation_matrix = train_df.drop(columns=['id']).corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix of Features', fontsize=15)
plt.show()

# To see the exact correlation values with the target
print("\n--- Correlation with BeatsPerMinute ---")
print(correlation_matrix['BeatsPerMinute'].sort_values(ascending=False))


features = train_df.drop(columns=['id', 'BeatsPerMinute'])
target = train_df['BeatsPerMinute']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

lgbm = lgb.LGBMRegressor(random_state=42)

print("\nTraining the LightGBM model...")
lgbm.fit(X_train, y_train)

print("Making predictions on the validation data...")
predictions = lgbm.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\n" + "="*50)
print(f"Validation RMSE: {rmse:.4f}")
print("="*50)


train_featured_df = train_df.copy()

print("Creating new features...")
train_featured_df['MoodEnergy'] = train_featured_df['MoodScore'] * train_featured_df['Energy']
train_featured_df['LoudnessQuality'] = train_featured_df['AudioLoudness'] * train_featured_df['AcousticQuality']
epsilon = 1e-6 
train_featured_df['VocalInstrumentalRatio'] = train_featured_df['VocalContent'] / (train_featured_df['InstrumentalScore'] + epsilon)

print("New features created:")
print(train_featured_df[['MoodEnergy', 'LoudnessQuality', 'VocalInstrumentalRatio']].head())

features = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
target = train_featured_df['BeatsPerMinute']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"\nNew training data shape: {X_train.shape}")

lgbm = lgb.LGBMRegressor(random_state=42)

print("Training the LightGBM model with new features...")
lgbm.fit(X_train, y_train)

print("Making predictions on the validation data...")
predictions = lgbm.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\n" + "="*50)
print(f"Validation RMSE with new features: {rmse:.4f}")
print("="*50)


train_featured_df = train_df.copy()
train_featured_df['MoodEnergy'] = train_featured_df['MoodScore'] * train_featured_df['Energy']
train_featured_df['LoudnessQuality'] = train_featured_df['AudioLoudness'] * train_featured_df['AcousticQuality']
epsilon = 1e-6 
train_featured_df['VocalInstrumentalRatio'] = train_featured_df['VocalContent'] / (train_featured_df['InstrumentalScore'] + epsilon)

features = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
target = train_featured_df['BeatsPerMinute']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'objective': 'regression_l1',  # MAE is often more robust to outliers
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)]) # Stop if no improvement after 100 rounds
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10) 

print("Optimization finished.")
print("Best trial's RMSE:", study.best_value)
print("Best hyperparameters:", study.best_params)

print("\nTraining final model with the best hyperparameters...")
best_params = study.best_params
best_params['n_estimators'] = 2000 # Increase estimators for the final model
best_params['random_state'] = 42
best_params['verbose'] = -1
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'rmse'

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(100, verbose=False)])

predictions = final_model.predict(X_val)
final_rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\n" + "="*50)
print(f"Final Validation RMSE after tuning: {final_rmse:.4f}")
print("="*50)


print("Loading data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

def create_features(df):
    df_copy = df.copy()
    df_copy['MoodEnergy'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['LoudnessQuality'] = df_copy['AudioLoudness'] * df_copy['AcousticQuality']
    epsilon = 1e-6 
    df_copy['VocalInstrumentalRatio'] = df_copy['VocalContent'] / (df_copy['InstrumentalScore'] + epsilon)
    return df_copy

print("Engineering features for train and test sets...")
train_featured_df = create_features(train_df)
test_featured_df = create_features(test_df)

X_full = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
y_full = train_featured_df['BeatsPerMinute']

X_test = test_featured_df.drop(columns=['id'])

X_test = X_test[X_full.columns]

# best_params = {
#     'learning_rate': 0.018440012649975284, 
#     'num_leaves': 35, 
#     'max_depth': 3, 
#     'min_child_samples': 9, 
#     'feature_fraction': 0.970664260192119, 
#     'bagging_fraction': 0.7517661116460606, 
#     'bagging_freq': 7, 
#     'lambda_l1': 2.691050348593101e-08, 
#     'lambda_l2': 0.1023360977484737,
#     'objective': 'regression_l1',
#     'metric': 'rmse',
#     'n_estimators': 2000, # Using a generous number of estimators
#     'random_state': 42,
#     'verbose': -1,
#     'n_jobs': -1
# }

best_params = {
    'learning_rate': 0.07627230220048577, 
    'num_leaves': 33, 
    'max_depth': 11, 
    'min_child_samples': 70, 
    'feature_fraction': 0.5955369534915669, 
    'bagging_fraction': 0.9391797953854437, 
    'bagging_freq': 4, 
    'lambda_l1': 0.006182201954782162, 
    'lambda_l2': 1.8952662924963336e-08
}

print("Training final model on all data...")
final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(X_full, y_full)

print("Making predictions on the test set...")
predictions = final_model.predict(X_test)

submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': predictions})

submission_df.to_csv('submission.csv', index=False)

print("\n'submission.csv' file created successfully!")
print("Head of the submission file:")
print(submission_df.head())

