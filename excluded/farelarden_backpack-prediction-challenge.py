pip install pandas-profiling > /dev/null 2>&1


pip install pydantic==1.10.14 > /dev/null 2>&1


pip install pydantic-settings > /dev/null 2>&1


pip install ydata-profiling > /dev/null 2>&1


from ydata_profiling import ProfileReport


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from keras.layers import Activation, Dense
import gc
from sklearn.ensemble import RandomForestRegressor
from IPython.display import clear_output
from sklearn import model_selection
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import preprocessing
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
%matplotlib inline

import os, psutil

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


%%time
# taken from https://www.kaggle.com/ryanholbrook/getting-started-september-2021-tabular-playground

def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))

def score(X, y, model, cv):
    scoring = ["roc_auc"]
    scores = cross_validate(
        model, X_train, y_train, scoring=scoring, cv=cv, return_train_score=True
    )
    scores = pd.DataFrame(scores).T
    return scores.assign(
        mean = lambda x: x.mean(axis=1),
        std = lambda x: x.std(axis=1),
    )

## from: https://www.kaggle.com/bextuychiev/how-to-work-w-million-row-datasets-like-a-pro
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

print('Function built')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


df_train = reduce_memory_usage(df_train, verbose=True)
df_train_extra = reduce_memory_usage(df_train_extra, verbose=True)
df_test = reduce_memory_usage(df_test, verbose=True)
print(cpu_stats())
print('Memory reduced')


df_train = df_train.astype({col: 'float64' for col in df_train.select_dtypes(include='float16').columns})
df_train_extra = df_train_extra.astype({col: 'float64' for col in df_train_extra.select_dtypes(include='float16').columns})
df_test = df_test.astype({col: 'float64' for col in df_test.select_dtypes(include='float16').columns})


df_train_full = pd.concat([df_train, df_train_extra], axis=0)


df_train.head()


profile = ProfileReport(df_train, title="Pandas Profiling Report")
profile.to_widgets()


profile = ProfileReport(df_train_extra, title="Pandas Profiling Report")
profile.to_widgets()


profile = ProfileReport(df_test, title="Pandas Profiling Report")
profile.to_widgets()


df_train_full['Brand'].fillna('Unknown', inplace=True)
df_train_full['Material'].fillna('Unknown', inplace=True)
df_train_full['Size'].fillna('Unknown', inplace=True)
df_train_full['Compartments'].fillna('Unknown', inplace=True)
df_train_full['Laptop Compartment'].fillna(False, inplace=True)
df_train_full['Waterproof'].fillna(False, inplace=True)

df_train_full['Style'].fillna(df_train_full['Style'].mode()[0], inplace=True)
df_train_full['Color'].fillna(df_train_full['Color'].mode()[0], inplace=True)

# Replace NaN with the mean
df_train_full['Weight Capacity (kg)'].fillna(df_train_full['Weight Capacity (kg)'].mean(), inplace=True)


from sklearn.preprocessing import OrdinalEncoder

df_train_full = pd.get_dummies(df_train_full, columns=['Brand'], prefix='Brand', drop_first=True)
df_train_full = pd.get_dummies(df_train_full, columns=['Material'], prefix='Material', drop_first=True)
df_train_full = pd.get_dummies(df_train_full, columns=['Compartments'], prefix='Compartments', drop_first=True)
df_train_full = pd.get_dummies(df_train_full, columns=['Style'], prefix='Style', drop_first=True)
df_train_full = pd.get_dummies(df_train_full, columns=['Color'], prefix='Color', drop_first=True)

df_train_full = pd.get_dummies(df_train_full, columns=['Laptop Compartment'], prefix='Laptop Compartmen', drop_first=True)
df_train_full = pd.get_dummies(df_train_full, columns=['Waterproof'], prefix='Waterproof', drop_first=True)

ordinal_encoder = OrdinalEncoder()
df_train_full['Size'] = ordinal_encoder.fit_transform(df_train_full[['Size']])


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

X = df_train_full.drop(columns=['Price', 'id'])  # Features
y = df_train_full['Price']  # Target

scaler = StandardScaler()
X = scaler.fit_transform(X)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize Random Forest model
    modelRF = RandomForestRegressor(
        n_estimators=100,  # number of trees
        max_depth=10,      # maximum depth of trees
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1          # use all available cores
    )

    # Fit the model
    modelRF.fit(X_train, y_train)

    # Make predictions
    y_pred = modelRF.predict(X_val)

    # Calculate metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    val_mae = mean_absolute_error(y_val, y_pred)
    
    rmse_scores.append(val_rmse)
    mae_scores.append(val_mae)

    print(f"Fold: Validation RMSE = {val_rmse}, Validation MAE = {val_mae}")

# Calculate mean and standard deviation of scores
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
mean_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)

print(f"\nCross-Validation Results:")
print(f"Mean RMSE: {mean_rmse} (±{std_rmse})")
print(f"Mean MAE: {mean_mae} (±{std_mae})")


import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import optuna

# Assuming df_train_full is already loaded
X = df_train_full.drop(columns=['Price', 'id'])  # Features
y = df_train_full['Price']  # Target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# K-Fold Cross-Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []

# Define the objective function for Optuna
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-5, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False
    }

    # K-Fold Cross-Validation
    fold_rmse_scores = []
    fold_mae_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Create CatBoost Pool
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        # Initialize CatBoost model
        modelCat = CatBoostRegressor(**params)

        # Train the model
        modelCat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

        # Evaluate the model
        y_pred = modelCat.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        mae = np.mean(np.abs(y_val - y_pred))

        fold_rmse_scores.append(rmse)
        fold_mae_scores.append(mae)

    # Return the mean RMSE across folds
    return np.mean(fold_rmse_scores)

# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
final_rmse_scores = []
final_mae_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    modelCat = CatBoostRegressor(**best_params, verbose=False)
    modelCat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

    y_pred = modelCat.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
    mae = np.mean(np.abs(y_val - y_pred))

    final_rmse_scores.append(rmse)
    final_mae_scores.append(mae)

    print(f"Fold: Validation RMSE = {rmse}, Validation MAE = {mae}")

# Calculate mean and standard deviation of RMSE and MAE
mean_rmse = np.mean(final_rmse_scores)
std_rmse = np.std(final_rmse_scores)
mean_mae = np.mean(final_mae_scores)
std_mae = np.std(final_mae_scores)

print(f"\nCross-Validation Results:")
print(f"Mean RMSE: {mean_rmse} (±{std_rmse})")
print(f"Mean MAE: {mean_mae} (±{std_mae})")


X = df_train_full.drop(columns=['Price', 'id'])  # Features
y = df_train_full['Price']  # Target

scaler = StandardScaler()
X = scaler.fit_transform(X)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    modelDL = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])

    modelDL.compile(optimizer=Adam(learning_rate=0.0001, clipvalue=1.0), loss='mse', metrics=['mae', rmse])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = modelDL.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=1024,
        callbacks=[early_stopping],
        verbose=0
    )

    val_loss, val_mae, val_rmse = modelDL.evaluate(X_val, y_val, verbose=0)
    rmse_scores.append(val_rmse)
    mae_scores.append(val_mae)

    print(f"Fold: Validation RMSE = {val_rmse}, Validation MAE = {val_mae}")

mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
mean_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)

print(f"\nCross-Validation Results:")
print(f"Mean RMSE: {mean_rmse} (±{std_rmse})")
print(f"Mean MAE: {mean_mae} (±{std_mae})")


df_test.head()


df_test['Brand'].fillna('Unknown', inplace=True)
df_test['Material'].fillna('Unknown', inplace=True)
df_test['Size'].fillna('Unknown', inplace=True)
df_test['Compartments'].fillna('Unknown', inplace=True)
df_test['Laptop Compartment'].fillna(False, inplace=True)
df_test['Waterproof'].fillna(False, inplace=True)

df_test['Style'].fillna(df_test['Style'].mode()[0], inplace=True)
df_test['Color'].fillna(df_test['Color'].mode()[0], inplace=True)

# Replace NaN with the mean
df_test['Weight Capacity (kg)'].fillna(df_test['Weight Capacity (kg)'].mean(), inplace=True)
from sklearn.preprocessing import OrdinalEncoder

df_test = pd.get_dummies(df_test, columns=['Brand'], prefix='Brand', drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Material'], prefix='Material', drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Compartments'], prefix='Compartments', drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Style'], prefix='Style', drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Color'], prefix='Color', drop_first=True)

df_test = pd.get_dummies(df_test, columns=['Laptop Compartment'], prefix='Laptop Compartmen', drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Waterproof'], prefix='Waterproof', drop_first=True)

ordinal_encoder = OrdinalEncoder()
df_test['Size'] = ordinal_encoder.fit_transform(df_test[['Size']])


df_test.head()


X = df_test.drop(columns=['id'])
y = df_test['id']


X_new = df_test
predictions = modelCat.predict(X)
predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
result = pd.concat([y, predictions_df], axis=1)
result.to_csv('submission_catboost.csv', index=False)


X_new = df_test
predictions = modelRF.predict(X)
predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
result = pd.concat([y, predictions_df], axis=1)
result.to_csv('submission_RF.csv', index=False)


X_new = df_test
predictions = modelDL.predict(X)
predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
result = pd.concat([y, predictions_df], axis=1)
result.to_csv('submission_DL.csv', index=False)




