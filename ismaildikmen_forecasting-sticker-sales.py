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


# Import required libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_percentage_error



# Load datasets
train_dataset = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

# Drop 'id' column and handle missing values
train_dataset = train_dataset.drop('id', axis=1).drop_duplicates().dropna()
test_dataset = test_dataset.drop('id', axis=1)

# Convert date to datetime and set as index
train_dataset['date'] = pd.to_datetime(train_dataset['date'])
test_dataset['date'] = pd.to_datetime(test_dataset['date'])
train_dataset.set_index('date', inplace=True)
test_dataset.set_index('date', inplace=True)



from sklearn.preprocessing import LabelEncoder

# Define categorical columns
cat_cols = ['country', 'store', 'product']

# Initialize LabelEncoder
label_encoders = {col: LabelEncoder() for col in cat_cols}

# Fit and transform both train and test datasets
for col in cat_cols:
    combined_data = pd.concat([train_dataset[col], test_dataset[col]])
    le = LabelEncoder()
    le.fit(combined_data)
    train_dataset[col] = le.transform(train_dataset[col])
    test_dataset[col] = le.transform(test_dataset[col])



# Target column is 'num_sold', features are all other columns
X = train_dataset.drop('num_sold', axis=1)
y = train_dataset['num_sold']

# Apply log transformation to the target
y = np.log1p(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Initialize LightGBM Regressor
lgb_model = lgb.LGBMRegressor()

# Define parameter grid (reduced for faster testing)
param_grid = {
    'learning_rate': [0.08],
    'n_estimators': [500],  # Reduced the number of estimators for quicker testing
    'max_depth': [10],
    'min_child_samples': [20],
    'subsample': [0.7],
    'colsample_bytree': [0.93]
}

# Setup GridSearchCV with MAPE as the scoring metric
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, 
                           scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False), 
                           cv=3, n_jobs=-1, verbose=1)

# Fit GridSearchCV on training data
grid_search.fit(X_train, y_train)

# Output best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best MAPE Score:", grid_search.best_score_)



# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate using MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Test MAPE:", mape)



# Predict on the test dataset
y_test_pred = best_model.predict(test_dataset)

# Prepare the submission dataframe
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission_df['num_sold'] = y_test_pred

# Save the submission file
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())





