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


df_sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

print(df_sample.head())
print(df_test.head())
print(df_train.head())


for column in df_train.columns:
    unique_values = df_train[column].unique()
    print(f"Column '{column}': {unique_values}")


for column in df_test.columns:
    unique_values = df_train[column].unique()
    print(f"Column '{column}': {unique_values}")


def preprocess(df):
    """
    Applies a log transformation to a predefined list of safe numerical columns.
    """
    df_copy = df.copy()
    
    # Only apply log transform to columns that are non-negative and not special cases.
    cols_to_log = ['age', 'day', 'duration', 'campaign', 'previous']
    
    for col in cols_to_log:
        df_copy[col] = np.log1p(df_copy[col])
            
    return df_copy


# Apply the new, safer preprocessing function
df_train_processed = preprocess(df_train)
df_test_processed = preprocess(df_test)

# Prepare training data, dropping 'id' and 'y'
X_train = df_train_processed.drop(columns=['id', 'y'])
y = df_train_processed['y']

# Prepare test data, dropping 'id'
X_test = df_test_processed.drop(columns=['id'])

# Ensure the column order is the same (good practice)
X_test = X_test[X_train.columns]

print("X_test Head:")
print(X_test.head())
print("\nX_train Head:")
print(X_train.head())


from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

model = CatBoostClassifier(
    iterations=50,
    learning_rate=0.2,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=10, # Print progress every 10 iterations
    cat_features=categorical_columns
)

model.fit(X_train, y)

preds = model.predict(X_test)   


predictions_for_submission = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame()
submission["id"] = df_test["id"]
submission["y"] = predictions_for_submission

# Save the final submission file
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission file with probabilities created successfully.")




