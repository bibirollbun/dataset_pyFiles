!pip install catboost
!pip install tensorflow keras
!pip install optuna
!ip install optuna-integration[tfkeras]


import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostRegressor
from catboost import Pool
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam, Adagrad
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import optuna
import optuna.visualization
from optuna.visualization import plot_terminator_improvement
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
from optuna.trial import TrialState
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned
import pickle


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


def load_csv_to_dataframe(file_path, ignore_fields=[]):
    """
    Load a CSV file into a pandas DataFrame, optionally ignoring specified fields.

    Parameters:
    file_path (str): The file path of the CSV file to be loaded.
    ignore_fields (list): A list of field names to be ignored when loading the CSV.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file, excluding the ignored fields.
    """
    # Read the CSV file from the given file path using pandas
    df = pd.read_csv(file_path)
    
    # Drop the fields that need to be ignored, if they exist in the DataFrame
    df = df.drop(columns=ignore_fields, errors='ignore')
    
    # Return the resulting DataFrame
    return df


train = "/kaggle/input/playground-series-s5e5/train.csv" 
train = load_csv_to_dataframe(train)


train.shape


print(train.isnull().sum())


train['Sex'] = train['Sex'].astype('category')


train.info()


train.describe()


test = "/kaggle/input/playground-series-s5e5/test.csv" 
test = load_csv_to_dataframe(test)


test.shape


print(test.isnull().sum())


test['Sex'] = test['Sex'].astype('category')


test.info()


test.describe()


sns.set_style("whitegrid")
sns.pairplot(train[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']],
             hue='Sex', palette='husl')
plt.show()


sns.set_style("whitegrid")
sns.pairplot(test[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']],
             hue='Sex', palette='husl')
plt.show()


train['Sex_encoded'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex_encoded'] = test['Sex'].map({'male': 1, 'female': 0})


train['BMI'] = train['Weight'] / (train['Height'] / 100) ** 2
test['BMI'] = test['Weight'] / (test['Height'] / 100) ** 2


train['VO2_Max'] = 15 * ((220 - train['Age']) / train['Heart_Rate'])
test['VO2_Max'] = 15 * ((220 - train['Age']) / train['Heart_Rate'])


# new feature: Heart Rate multiplied by Duration
train['Heart_Rate_Duration'] = train['Heart_Rate'] * train['Duration']
test['Heart_Rate_Duration'] = test['Heart_Rate'] * test['Duration']

# feature: Age multiplied by BMI
train['Age_BMI'] = train['Age'] * train['BMI']
test['Age_BMI'] = test['Age'] * test['BMI']

#Temperature Deviation from Normal (Assuming 37.0°C is baseline)
train['Temp_Deviation'] = train['Body_Temp'] - 37.0
test['Temp_Deviation'] = test['Body_Temp'] - 37.0


important_columns = ['id', 'Age', 'Sex_encoded', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Heart_Rate_Duration', 'Body_Temp', 'Temp_Deviation', 'BMI', 'Age_BMI', 'VO2_Max']


new_train = train[important_columns]
new_test = test[important_columns]


new_train = new_train.merge(train[['id', 'Calories']], on='id', how='left')


train_corr_matrix = new_train.corr()
print(train_corr_matrix)


test_corr_matrix = new_test.corr()
print(test_corr_matrix)


plt.figure(figsize=(10, 8))
sns.heatmap(train_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap on the Train Set')
plt.show()


important_columns_after_the_correlation_matrix = ['id', 'Duration', 'Heart_Rate', 'Heart_Rate_Duration', 'Body_Temp', 'Temp_Deviation', 'VO2_Max']


new_train = new_train[important_columns_after_the_correlation_matrix]
new_test = new_test[important_columns_after_the_correlation_matrix]


new_train = new_train.merge(train[['id', 'Calories']], on='id', how='left')


new_train, new_validation = train_test_split(new_train, test_size=0.2, random_state=42)


new_train.to_csv('new_train.csv', index=True)
new_test.to_csv('new_test.csv', index=True)
new_validation.to_csv('new_validation.csv', index=True)


train = "/kaggle/working/new_train.csv" 
train = load_csv_to_dataframe(train)

test = "/kaggle/working/new_test.csv" 
test = load_csv_to_dataframe(test)

validation = "/kaggle/working/new_validation.csv" 
validation = load_csv_to_dataframe(validation)


selected_features = ['Duration', 'Heart_Rate', 'Heart_Rate_Duration', 'Body_Temp', 'Temp_Deviation', 'VO2_Max']


#we will be scaling only the features (not the target) since we have to use RMSLE as the metric
scaler_for_the_features = StandardScaler()


train_scaled = train.copy()
test_scaled = test.copy()
validation_scaled = validation.copy()


train_scaled_features = scaler_for_the_features.fit_transform(train[selected_features])
test_scaled_features = scaler_for_the_features.fit_transform(test[selected_features])
validation_scaled_features = scaler_for_the_features.fit_transform(validation[selected_features])


train_scaled = pd.DataFrame(train_scaled_features, columns=selected_features, index=train.index)
train_scaled.insert(0, 'id', train['id'])
train_scaled.insert(1, 'Calories', train['Calories'])
train_scaled


test_scaled


validation_scaled = pd.DataFrame(validation_scaled_features, columns=selected_features, index=validation.index)
validation_scaled.insert(0, 'id', validation['id'])
validation_scaled.insert(1, 'Calories', validation['Calories'])
validation_scaled


train_scaled.to_csv('train_scaled.csv', index=True)
test_scaled.to_csv('test_scaled.csv', index=True)
validation_scaled.to_csv('validation_scaled.csv', index=True)


train_scaled = "/kaggle/working/train_scaled.csv" 
train_scaled = load_csv_to_dataframe(train_scaled)

test_scaled = "/kaggle/working/test_scaled.csv" 
test_scaled = load_csv_to_dataframe(test_scaled)

validation_scaled = "/kaggle/working/validation_scaled.csv" 
validation_scaled = load_csv_to_dataframe(validation_scaled)


features = ['Duration', 'Heart_Rate', 'Heart_Rate_Duration', 'Body_Temp', 'Temp_Deviation', 'VO2_Max']
target = ['Calories']


X_train = train_scaled[features]
y_train = train_scaled['Calories']

X_val = validation_scaled[features]
y_val = validation_scaled['Calories']

X_test = test_scaled[features]


X_train.shape[1]


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


print("Any NaNs in X_train?", X_train.isnull().values.any())
print("Any NaNs in y_train?", pd.isnull(y_train).any())


X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
y_train = y_train.astype('float32')
y_val = y_val.astype('float32')


print("Min y_train:", y_train.min())
print("Min y_val:", y_val.min())


def rmsle(y_true, y_pred):
    # clip to avoid log(0) or negative
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    return np.sqrt(np.mean((log_true - log_pred) ** 2))


model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',  
    eval_metric='MSLE',
    verbose=0
)


model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)


results = model.get_evals_result()

train_rmse = results['learn']['RMSE']
val_msle = results['validation']['MSLE']

plt.plot(train_rmse, label='Train RMSE (loss)')
plt.plot(val_msle, label='Validation MSLE (eval metric)')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Training vs Validation Metrics')
plt.legend()


preds = model.predict(X_val)
print("RMSLE:", rmsle(y_val, preds))


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'depth': trial.suggest_int('depth', 6, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'loss_function': 'RMSE',
        'eval_metric': 'MSLE',
        'verbose': 0
    }

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model = CatBoostRegressor(**params)

    model.fit(
        X_train, y_train_log,
        eval_set=(X_val, y_val_log),
        early_stopping_rounds=50,
        verbose=0
    )

    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    
    score = rmsle(y_val, preds)
    score = round(score, 5)

    if score < 0.05:
        print(f"Found promising trial: RMSLE = {score:.5f}")

    return score


study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=SuccessiveHalvingPruner(min_resource=5, reduction_factor=3),
)

study.optimize(objective, 
                n_trials = 100,
                timeout=1800, 
                catch=(Exception,)
)


# Get best parameters
best_params = study.best_params
print("Best parameters:", best_params)


best_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.035998772558212676,
    depth=12,
    l2_leaf_reg=5.568193163423996,
    border_count=104,
    bagging_temperature=0.845707457264183,
    loss_function='RMSE',
    eval_metric='MSLE',
    verbose=0
)


best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)


best_results = best_model.get_evals_result()

train_rmse = best_results['learn']['RMSE']
val_msle = best_results['validation']['MSLE']

plt.plot(train_rmse, label='Train RMSE (loss)')
plt.plot(val_msle, label='Validation MSLE (eval metric)')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Training vs Validation Metrics')
plt.legend()


best_preds = best_model.predict(X_val)
print("RMSLE:", rmsle(y_val, best_preds))


best_preds = best_model.predict(X_test)  

submission = pd.DataFrame({
    'id': test_scaled['id'], 
    'Calories': best_preds
})

submission.to_csv('submission.csv', index=False)

print("Submission file saved successfully!")

