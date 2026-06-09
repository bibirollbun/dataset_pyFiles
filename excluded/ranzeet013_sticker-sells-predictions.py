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


train_dataset = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train_dataset = train_dataset.drop('id', axis=1)
test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
test_dataset = test_dataset.drop('id', axis=1)
train_dataset.head()


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# Preprocess the training dataset
def preprocess_data(train_dataset, test_dataset):
    # Remove duplicates and handle missing values
    train_dataset = train_dataset.drop_duplicates()
    train_dataset = train_dataset.dropna()

    train_dataset = train_dataset.set_index('date')
    test_dataset = test_dataset.set_index('date')

    train_dataset.plot(figsize=(15, 5))

    train_dataset.index = pd.to_datetime(train_dataset.index)
    test_dataset.index = pd.to_datetime(test_dataset.index)

    num_cols = list(train_dataset.select_dtypes(exclude=['object']).columns.difference(['num_sold']))
    cat_cols = list(train_dataset.select_dtypes(include=['object']).columns)

    num_cols_test = list(test_dataset.select_dtypes(exclude=['object']).columns.difference(['id']))
    cat_cols_test = list(test_dataset.select_dtypes(include=['object']).columns)

    print('Numeric Columns:', num_cols)
    print('Categorical Columns:', cat_cols)

    # Label encoding for categorical columns
    label_encoders = {col: LabelEncoder() for col in cat_cols}
    for col in cat_cols:
        combined_data = pd.concat([train_dataset[col], test_dataset[col]])
        le = label_encoders[col]
        le.fit(combined_data)
        train_dataset[col] = le.transform(train_dataset[col])
        test_dataset[col] = le.transform(test_dataset[col])

    return train_dataset, test_dataset, num_cols, cat_cols, num_cols_test, cat_cols_test


def split_data(train_dataset):
    X = train_dataset.iloc[:, :-1]
    y = train_dataset.iloc[:, -1]

    # Apply log transformation to target variable
    train_dataset['num_sold'] = np.log1p(train_dataset['num_sold'])

    return X, y


def train_lgbm_model(X_train, y_train):
    lgb_model = lgb.LGBMRegressor()

    param_grid = {
        'learning_rate': [0.08],
        'n_estimators': [1000],
        'max_depth': [12],
        'min_child_samples': [32],
        'subsample': [0.7],
        'colsample_bytree': [0.93],
    }

    grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Test MAPE:", mape)
    
    return y_pred


def predict_on_test_data(best_model, test_dataset):
    y_test_pred = best_model.predict(test_dataset)
    return y_test_pred


def create_submission(y_test_pred):
    submission_df = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
    submission_df['num_sold'] = y_test_pred
    submission_df.to_csv("submission.csv", index=False)
    print(submission_df.head())


def main(train_dataset, test_dataset):
    train_dataset, test_dataset, num_cols, cat_cols, num_cols_test, cat_cols_test = preprocess_data(train_dataset, test_dataset)

    # Split the train data into features and target
    X, y = split_data(train_dataset)

    # Train the LightGBM model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    best_model, best_params, best_score = train_lgbm_model(X_train, y_train)
    print("Best Parameters:", best_params)
    print("Best MAPE Score:", best_score)

    evaluate_model(best_model, X_test, y_test)

    y_test_pred = predict_on_test_data(best_model, test_dataset)

    create_submission(y_test_pred)

# Execute the main function
main(train_dataset, test_dataset)


import shutil

shutil.move("submission.csv", "/kaggle/working/submission.csv")



