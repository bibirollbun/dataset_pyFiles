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


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


 #Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv").drop('id', axis=1)
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


# Display first few rows of the dataset
print("Training Data:")
print(train_df.head())
print("Test Data:")
print(test_df.head())


# Assuming 'Price' is the target variable and other columns are features
target = 'Price'
features = [col for col in train_df.columns if col != target]


# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df[target], test_size=0.2, random_state=42)


# Identify categorical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()


# Apply One-Hot Encoding
train_df = pd.get_dummies(train_df, columns=categorical_cols)
test_df = pd.get_dummies(test_df, columns=categorical_cols)


# Align test set columns with training set (ensure they have the same number of columns)
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)



# Separate features and target again
X = train_df.drop(columns=['Price'])  # Drop target variable from features
y = train_df['Price']



# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.impute import SimpleImputer
# Check for missing values before applying imputation
print("Missing values in X_train before imputation:")
print(X_train.isnull().sum())

# Apply imputation to the training data
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)

# Check if there are any missing values in the imputed X_train
print("\nMissing values in X_train after imputation:")
print(pd.DataFrame(X_train_imputed).isnull().sum())

# Now that imputation is complete, verify the shape and ensure data consistency
print("\nShape of X_train_imputed:", X_train_imputed.shape)

# Check for missing values in y_train (it should not have NaNs)
print("\nMissing values in y_train:")
print(y_train.isnull().sum())






# Train the model again on the imputed data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)



# Apply the same imputation to the validation and test data
X_val_imputed = imputer.transform(X_val) 
test_df_imputed = imputer.transform(test_df.drop(columns=['Price'], errors='ignore'))



# Predictions on the validation set
y_pred = model.predict(X_val_imputed)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse}")



# Predictions on the test data
test_predictions = model.predict(test_df_imputed)
print("Test Predictions:", test_predictions)


print(test_df.columns)



test_df['id'] = range(300000, 300000 + len(test_df))



# submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': test_predictions
})

# Ensure there are 200,000 rows (adjust size accordingly)
submission = submission.head(200000)
submission.to_csv("/content/submission.csv", index=False)
print("Submission file saved as submission.csv")






