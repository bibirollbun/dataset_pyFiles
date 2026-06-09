import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Display the first few rows of the dataset
print(train_data.head())


# Check for missing values
print(train_data.isnull().sum())

# Fill missing values if any
train_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='ffill', inplace=True)

# Separate features and target
X = train_data.drop('Price', axis=1)
y = train_data['Price']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_valid, preds))
print(f'RMSE: {rmse}')


# Example of using an ensemble method
from sklearn.ensemble import GradientBoostingRegressor

ensemble_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
ensemble_clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', ensemble_model)])
ensemble_clf.fit(X_train, y_train)
ensemble_preds = ensemble_clf.predict(X_valid)
ensemble_rmse = np.sqrt(mean_squared_error(y_valid, ensemble_preds))
print(f'Ensemble RMSE: {ensemble_rmse}')


# Preprocessing of test data, get predictions
test_preds = clf.predict(test_data)

# Save predictions to a CSV file
output = pd.DataFrame({'id': test_data.id, 'Price': test_preds})
output.to_csv('submission.csv', index=False)


import pickle

# Save the trained model to a file
with open('backpack_price_predictor.pkl', 'wb') as f:
    pickle.dump(clf, f)

