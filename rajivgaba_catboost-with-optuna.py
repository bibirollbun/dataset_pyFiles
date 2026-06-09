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


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


train_file = "/kaggle/input/playground-series-s5e5/train.csv"
test_file = "/kaggle/input/playground-series-s5e5/test.csv"


train_data = pd.read_csv(train_file)
train_data.info()


train_data.head()


# Segregate columns

num_cols = train_data.select_dtypes(['int64','float64']).columns.to_list()

target = 'Calories'

cat_cols = train_data.select_dtypes('object').columns.to_list()

num_cols.remove(target)
num_cols.remove('id')


num_cols


# EDA

plt.figure()
sns.set_style('darkgrid')
sns.countplot(data=train_data, x='Sex', width = 0.4)
plt.tight_layout()
plt.show()


# EDA - Univariate Analysis 

plt.figure()
fig, axes = plt.subplots(3, 2, figsize=[15,10])
sns.set_style('darkgrid')
for i, col in enumerate(num_cols):
    sns.histplot(data=train_data, x=col, bins=30, kde=True, ax=axes[i//2,i%2], color='pink')
    axes[i//2,i%2].set_title(col)
    plt.tight_layout()
plt.show()

    


plt.figure()
fig, axes = plt.subplots(3, 2, figsize=[15,10])
sns.set_style('darkgrid')
# sns.color_palette("dark", 8)
for i, col in enumerate(num_cols):
    sns.boxplot(data=train_data, x='Sex', y=col, ax=axes[i//2,i%2])
    axes[i//2,i%2].set_title(col)
    plt.tight_layout()
plt.show()


plt.figure(figsize=[18,6])
sns.color_palette("tab10")
sns.scatterplot(data=train_data, y='Weight', x='Duration', hue='Calories')
plt.show()


train_data.head()


plt.figure(figsize=[8,8])
sns.heatmap(data=train_data[num_cols].corr(), cmap='coolwarm', annot=True, fmt='0.2f')
plt.show()


train_data.head()


train_data.describe()


plt.figure(figsize=[18,6])
sns.color_palette("tab10")
sns.scatterplot(data=train_data, y='Heart_Rate', x='Duration', hue='Calories')
plt.show()


plt.figure(figsize=[18,6])
sns.color_palette("tab10")
sns.scatterplot(data=train_data, y='Body_Temp', x='Duration', hue='Calories')
plt.show()


train_data.info()


import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



le = LabelEncoder()


train_data['Sex'] = le.fit_transform(train_data['Sex'])


train_data.info()


train_data.drop(columns=['id'], inplace=True)


train_data[num_cols].describe()


for col in num_cols:
    plt.figure(figsize=[7,4])
    sns.boxplot(data=train_data, y=col)
    plt.show()


train_data[num_cols].head()


df_train, df_test = train_test_split(train_data, test_size=0.20, random_state=101)


y_train = df_train.pop('Calories')
X_train = df_train


y_test = df_test.pop('Calories')
X_test = df_test


scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])


X_test[num_cols] = scaler.transform(X_test[num_cols])


df_train.head()


df_test.head()


import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from catboost import CatBoostRegressor


def objective_squaredlogerror(trial):
    # Suggest hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    subsample = trial.suggest_float("subsample", 0.7, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 0, 1)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True)

    # create the CatBoostRegressor model

    model = CatBoostRegressor(
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=max_depth,
        l2_leaf_reg=reg_alpha,
        random_seed=42,
        eval_metric='RMSE',
        verbose=0
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Ensure predictions are non-negative
    y_pred_clipped = np.maximum(y_pred, 0)

    # Calculate MSLE
    msle = mean_squared_log_error(y_test, y_pred_clipped)
    print(f"RMSLE={np.sqrt(msle)}")
    return msle


# create study
study_slerror = optuna.create_study(direction="minimize")

# optimize the study
study_slerror.optimize(objective_squaredlogerror, n_trials=30)

print("Best trial:")
trial_slerror = study_slerror.best_trial
print("  Value (MSLE): {}".format(trial_slerror.value))
print("  Params: {}".format(trial_slerror.params))


best_params_slerror = study_slerror.best_params
best_params_slerror



# final_model_slerror = xgb.XGBRegressor(**best_params_slerror, random_state=42, objective='reg:squaredlogerror')

final_model_slerror = CatBoostRegressor(
    iterations=best_params_slerror['n_estimators'],
        learning_rate=best_params_slerror['learning_rate'],
        depth=best_params_slerror['max_depth'],
        l2_leaf_reg=best_params_slerror['reg_alpha'],
        random_seed=42,
        eval_metric='RMSE',
        verbose=0
    )


final_model_slerror.fit(X_train, y_train)
final_predictions_slerror = final_model_slerror.predict(X_test)
final_predictions_slerror_clipped = np.maximum(final_predictions_slerror, 0)
final_msle_slerror = mean_squared_log_error(y_test, final_predictions_slerror_clipped)
print(f"\nFinal XGBoost RMSLE with optimized hyperparameters: {np.sqrt(final_msle_slerror)}")


# treatment of test data

test_data = pd.read_csv(test_file)
df = test_data.copy()
df['Sex'] = le.fit_transform(df['Sex'])
df.drop(columns=['id'], inplace=True)


df[num_cols] = scaler.transform(df[num_cols])


# Predictions on test data

predictions = final_model_slerror.predict(df)
predictions[0:10]


results_df = pd.DataFrame(
    {
        'id' : test_data['id'],
        'Calories' : predictions
    }
)


# create submission file

results_df.to_csv('submission.csv', index=False)

