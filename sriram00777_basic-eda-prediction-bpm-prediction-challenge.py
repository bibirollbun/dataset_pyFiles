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
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")



train.head()


test.head()


train.drop(columns=['id'], inplace=True)



train.columns


train.dtypes


train.info()



train.isnull().sum()



test.isnull().sum()


train.shape


test.shape


train.describe()


# List of numerical columns for visualization (excluding 'id' and non-numeric if any)
num_cols = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy', 'BeatsPerMinute'
]

plt.figure(figsize=(16, 30))
for i, col in enumerate(num_cols):
    plt.subplot(len(num_cols), 2, 2*i+1)
    sns.boxplot(x=train[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
    plt.subplot(len(num_cols), 2, 2*i+2)
    sns.histplot(train[col], kde=True, color='orange')
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()



def create_features(df):    
    # Interaction between RhythmScore and Energy
    df['Rhythm_Energy_Interaction'] = df['RhythmScore'] * df['Energy']
    
    # Normalize TrackDurationMs to minutes
    df['TrackDuration_Minutes'] = df['TrackDurationMs'] / 60000
    
    # Loudness per unit of Energy (handle divide by zero)
    df['Loudness_per_Energy'] = df['AudioLoudness'] / (df['Energy'] + 1e-6)
    
    # Vocal to Instrumental ratio (handle divide by zero)
    df['Vocal_to_Instrumental_Ratio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-6)
    
    # Mood and Energy product
    df['Mood_Energy_Product'] = df['MoodScore'] * df['Energy']
    
    # LivePerformanceLikelihood to Energy ratio (handle divide by zero)
    df['LivePerformance_to_Energy_Ratio'] = df['LivePerformanceLikelihood'] / (df['Energy'] + 1e-6)
    
    # Log transformation of TrackDurationMs
    df['Log_TrackDuration'] = np.log1p(df['TrackDurationMs'])
    
    # Sum of VocalContent and InstrumentalScore
    df['Vocal_Instrumental_Sum'] = df['VocalContent'] + df['InstrumentalScore']
    
    # Bin AudioLoudness into categories
    loud_bins = [-np.inf, -20, -10, 0, np.inf]
    loud_labels = ['quiet', 'soft', 'moderate', 'loud']
    df['AudioLoudness_Bin'] = pd.cut(df['AudioLoudness'], bins=loud_bins, labels=loud_labels)
    
    return df



from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_new_features(train_df, test_df, cols, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Choose 'minmax' or 'zscore'.")

    # Fit scaler on train, transform train and test
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    return train_df, test_df



from sklearn.preprocessing import LabelEncoder

def encode_audio_loudness_bin(train_df, test_df):
    encoder = LabelEncoder()
    # Fit on train and transform
    train_df['AudioLoudness_Bin'] = encoder.fit_transform(train_df['AudioLoudness_Bin'].astype(str))
    # Transform test using same encoder
    test_df['AudioLoudness_Bin'] = encoder.transform(test_df['AudioLoudness_Bin'].astype(str))
    return train_df, test_df



train = create_features(train)
test = create_features(test)


train.describe()


new_cols = [
    'Rhythm_Energy_Interaction', 'TrackDuration_Minutes', 'Loudness_per_Energy',
    'Vocal_to_Instrumental_Ratio', 'Mood_Energy_Product', 'LivePerformance_to_Energy_Ratio',
    'Log_TrackDuration', 'Vocal_Instrumental_Sum'
]

# Apply after feature creation
train, test = normalize_new_features(train, test, new_cols, method='zscore')
train, test = encode_audio_loudness_bin(train, test)



train.describe()


test.describe()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb


SEED = 42
FOLDS = 5
target_col = 'BeatsPerMinute'

# Features excluding target
features = [col for col in train.columns if col != target_col]

X = train[features]
y = train[target_col]
X_test = test[features]

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"Training fold {fold}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': SEED,
        'verbosity': 0,
    }

    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist,
                      early_stopping_rounds=50, verbose_eval=50)

    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration)) / FOLDS

rmse_score = mean_squared_error(y, oof_preds, squared=False)
print(f"OOF RMSE: {rmse_score:.4f}")

def create_submission_kfold():
    # Prepare submission
    submission = pd.DataFrame({
        'id': test['id'],  # Assuming test has id column
        'BeatsPerMinute': test_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved as submission.csv")


# create_submission_kfold()


train.columns


test.columns


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Prepare features and target
target_col = 'BeatsPerMinute'
drop_cols = [target_col]  # Drop only those not necessary for training

X = train.drop(columns=drop_cols, errors='ignore')
y = train[target_col]

# Use all features except id and target for test data
X_test = test.drop(columns=['id'], errors='ignore').copy()

# Split out validation set for tuning
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'verbosity': 0,
        'seed': 42
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, 'eval')],
                      early_stopping_rounds=50, verbose_eval=False)
    preds = model.predict(dval)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse




# Run Optuna study
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# print('Best trial RMSE:', study.best_trial.value)
# print('Best hyperparameters:', study.best_trial.params)



# optuna.visualization.plot_parallel_coordinate(study)




# optuna.visualization.plot_optimization_history(study)


# optuna.visualization.plot_param_importances(study)



# study.best_trial.params







# # Train final model on full train data
# final_params = study.best_trial.params.copy()
# final_params.update({
#     'objective': 'reg:squarederror',
#     'eval_metric': 'rmse',
#     'verbosity': 0,
#     'seed': 42,
# })

# dtrain_full = xgb.DMatrix(X, label=y)
# final_model = xgb.train(final_params, dtrain_full, num_boost_round=study.best_trial.user_attrs.get('best_iteration', 1000))

# # Predict on test dataset
# dtest = xgb.DMatrix(X_test)
# test_preds = final_model.predict(dtest)

# # Prepare submission dataframe
# submission = pd.DataFrame({
#     'id': test['id'],
#     'BeatsPerMinute': test_preds
# })

# submission.to_csv('submission.csv', index=False)
# print('submission.csv created!')



import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEED = 42
FOLDS = 5
N_TRIALS = 50
target_col = 'BeatsPerMinute'

features = [col for col in train.columns if col != target_col and col != 'id']
X = train[features]
y = train[target_col]
X_test = test[features]

print(f"Training data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',  # to use gpu
        'verbosity': 0,
        'seed': SEED,
        # Hyperparameters to optimize
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    cv_scores = []  # Initialize cv_scores list
    folds = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train the model - Fixed variable name conflict
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Fixed variable names
        y_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        cv_scores.append(rmse)
    
    mean_cv_score = np.mean(cv_scores)
    print(f"Trial {trial.number}: CV Scores: {cv_scores}")
    print(f"Trial {trial.number}: Mean CV Score: {mean_cv_score:.4f}")
    
    return mean_cv_score




# Create and run the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

# Print best parameters
print("\n" + "="*50)
print("OPTIMIZATION COMPLETE")
print("="*50)
print(f"Best trial: {study.best_trial.number}")
print(f"Best CV score: {study.best_value:.4f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best parameters
print("\n" + "="*50)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("="*50)

best_params = study.best_params.copy()
best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'verbosity': 0,
    'seed': SEED
})

# Final K-fold training with best parameters
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"Training final fold {fold}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(best_params, dtrain, num_boost_round=1000, 
                      evals=[(dtrain, 'train'), (dval, 'eval')],
                      early_stopping_rounds=50, verbose_eval=50)

    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration)) / FOLDS

final_rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"\nFinal OOF RMSE: {final_rmse:.4f}")

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': test_preds
})

# submission.to_csv('optimized_submission.csv', index=False)
# print("Optimized submission saved as optimized_submission.csv")


submission.to_csv('submission.csv', index=False)



submission.head()




