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


# Import necessary libraries

import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import numpy as np
import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors




# Read all the datasets

train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


# Vectorize SMILES with TF-IDF

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=10_000,
    lowercase=False
)

X = vectorizer.fit_transform(train['SMILES'])
print("TF-IDF shape:", X.shape)

X_test_final = vectorizer.transform(test['SMILES'])


# Check if there are any null values

train.isna().sum()


# Impute values using KNN

train_imputed = train.copy()
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for col in targets:
    print(f"Imputing for {col}...")

    # Only use rows where the target is known
    known_idx = train_imputed[train_imputed[col].notna()].index
    X_known = X[known_idx]
    y_known = train_imputed.loc[known_idx, col].values

    # Rows needing imputation
    missing_idx = train_imputed[train_imputed[col].isna()].index
    X_missing = X[missing_idx]

    if len(missing_idx) == 0:
        continue

    # Fit KNN only on known targets
    knn = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
    knn.fit(X_known)

    # Get 3 nearest known neighbors for each missing sample
    distances, indices = knn.kneighbors(X_missing)

    # Impute as mean of neighbors' target values
    imputed_values = np.array([
        np.mean(y_known[neighbor_ids]) for neighbor_ids in indices
    ])

    # Fill in the missing values
    train_imputed.loc[missing_idx, col] = imputed_values

    # Final fallback in case anything is still NaN
    train_imputed[col] = train_imputed[col].fillna(train_imputed[col].median())



# Recheck for null values

train_imputed.isna().sum()


# Analyze dataset

print("Dataset shape: ",train_imputed.shape)
print("\nDataset: \n",train_imputed.head())
display(train.info())
display(train.describe())


# Prepare data

X_train, X_valid, y_train, y_valid = train_test_split(X, train_imputed[targets], test_size=0.25)


# Create a study using optuna for hyperparameter finetuning


def objective(trial):
    params = {
        'tree_method': 'hist',
        'device' : 'cuda',# Enables GPU training
        'predictor': 'gpu_predictor',
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'random_state': 42,
        'verbosity': 0, 
        'objective': 'reg:squarederror',
    }

    model = MultiOutputRegressor(xgb.XGBRegressor(**params))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    return rmse



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)



# Store the best parameters
best_params = study.best_params

# Add fixed GPU-related and other necessary parameters
best_params.update({
    'tree_method': 'hist',
    'device' : 'cuda',
    'predictor': 'gpu_predictor',
    'objective': 'reg:squarederror',
    'random_state': 42,
    'verbosity': 0,
})


# Use the best parameters to fit the model
final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params))
final_model.fit(X_train, y_train)


# Predict for validation data
y_pred = final_model.predict(X_valid)


# Using RMSE per feature predicted
rmse_per_output = np.sqrt(mean_squared_error(y_valid, y_pred, multioutput='raw_values'))
rmse_mean = np.mean(rmse_per_output)

# === MAE (per output and mean) ===
mae_per_output = mean_absolute_error(y_valid, y_pred, multioutput='raw_values')
mae_mean = np.mean(mae_per_output)

# === R² Score (per output and mean) ===
r2_per_output = r2_score(y_valid, y_pred, multioutput='raw_values')
r2_mean = np.mean(r2_per_output)

print("RMSE per output:", rmse_per_output)
print("Mean RMSE:", rmse_mean)
print("MAE per output:", mae_per_output)
print("Mean MAE:", mae_mean)
print("R² per output:", r2_per_output)
print("Mean R²:", r2_mean)


#Predict for test data
y_pred = final_model.predict(X_test_final)


# Prepare submission file
submission['id'] = test['id']
submission[targets] = y_pred
submission.to_csv('submission.csv', index=False)


submission.head()

