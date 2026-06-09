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


# Importing libraries
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


# Let's load the data from the CSV files
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


# Function to check for missing values in the datasets
def missing_values(df):
    missing_values = df.isnull().sum()
    num_rows = df.shape[0]
    percent_missing_values = (missing_values/num_rows)*100
    return pd.DataFrame({'Num_missing_values': missing_values,
                         '%Missing_values': percent_missing_values})

missing_values(train_df)


missing_values(test_df)


#Handling Missing Values

train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median())
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median())

train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median())
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median())

train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0])


# List of categorical features
categorical_features = train_df.select_dtypes(include = ['object']).columns.difference(['Podcast_Name', 'Episode_Title']).tolist()

# List of numeric features
numerical_features = train_df.select_dtypes(include = ['float64', 'int64']).columns.difference(['id', 'Listening_Time_minutes']).tolist()


#Feature Selection and Target Variable Definition
x = train_df.drop(['Podcast_Name', 'Episode_Title', 'id', 'Listening_Time_minutes'], axis = 1)
test = test_df.drop(['Podcast_Name', 'Episode_Title', 'id'], axis = 1)

y = train_df['Listening_Time_minutes']


# Splitting the data into training and validation sets
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25, random_state = 24)


#categorical Encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown = 'ignore'), categorical_features)],
    remainder = 'passthrough'
)


x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_val)
test = transformer.transform(test)


# Define the objective function for Optuna optimization
def objective(trial):
    
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'n_estimators': trial.suggest_int('n_estimators', 3000, 7000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.2),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    model = lgb.LGBMRegressor(**param)

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        x_train, y_train,
        scoring='neg_root_mean_squared_error',
        cv=kf,
        n_jobs=-1
    )

    # Convert to positive RMSE and report to Optuna
    mean_rmse = -np.mean(scores)
    trial.report(mean_rmse, step=trial.number)

    # Early stopping if trial is not promising
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mean_rmse

# Create Optuna study
study = optuna.create_study(direction='minimize', study_name="LightGBM-RMSE-Optimization")
study.optimize(objective, n_trials=30)  # You can increase n_trials for better results

# Print best results
print("Best RMSE: ", study.best_value)
print("Best hyperparameters: ", study.best_params)



# Hyperparameters optimized using Optuna and validated with cross-validation from the previous cell
params = {'num_leaves': 60, 
          'n_estimators': 4084, 
          'learning_rate': 0.03944220931469, 
          'max_depth': 13, 
          'min_split_gain': 0.1619551546994377, 
          'lambda_l1': 1.318963219864037e-08, 
          'lambda_l2': 5.072684368048482e-07, 
          'feature_fraction': 0.8817028803723406, 
          'bagging_fraction': 0.9053207703444995, 
          'bagging_freq': 7, 
          'verbosity': -1}

lgt_model = lgb.LGBMRegressor(**params)


# Train the LightGBM model using the optimized hyperparameters on the training data
lgt_model.fit(x_train, y_train)


# Make predictions on the test set using the trained LightGBM model
y_pred = lgt_model.predict(x_test)

# Calculate the Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_pred, y_val, squared = False)


rmse





# Make final predictions on the test dataset using the trained LightGBM model
final_pred = lgt_model.predict(test)

submission = pd.DataFrame({
    'id' : test_df['id'],
    'Listening_Time_minutes' : final_pred
})


submission.head(5)


submission.to_csv('submission.csv', index = False)


# Thanks for viewing the notebook!
# If you found it helpful, kindly upvote and feel free to share any feedback for further improvement.
# I will be focusing on more feature engineering and exploring additional models to enhance performance.





