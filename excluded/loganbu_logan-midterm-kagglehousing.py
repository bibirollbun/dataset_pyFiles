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


import pandas as pd

# Assuming your data is in the 'playground-series-s3e1' folder
train_data = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv') 

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor  
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def regression_models(X_train, y_train, X_test):  # Removed y_test from parameters
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'Extra Tree': ExtraTreeRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'XGradientBoosting': XGBRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'SVR': SVR()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results.append([name]) # Add here any required data
                         
    
    return pd.DataFrame(results, columns=['Model']) # Add here any required columns


numerical_features = train_data.select_dtypes(include=['number']).columns.tolist()
numerical_features.remove('id')  # Remove 'id' if it's not a predictive feature
# remove the target variable from numerical_features list
if 'MedHouseVal' in numerical_features:
    numerical_features.remove('MedHouseVal')
try:
    numerical_features.remove('target') # Remove 'target' if present
except ValueError:
    pass  # If 'target' is not in the list, do nothing


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median' or other strategies
train_data[numerical_features] = imputer.fit_transform(train_data[numerical_features])
test_data[numerical_features] = imputer.transform(test_data[numerical_features])



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])


from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep components explaining 95% of variance
train_pca = pca.fit_transform(train_data[numerical_features])
test_pca = pca.transform(test_data[numerical_features])


results_df = regression_models(train_pca, train_data['MedHouseVal'], test_pca) # Pass 'target'  Removed y_test from parameters
print(results_df)

# Here you should choose your best model by your criteria.
# As an example:
best_model_name = 'Linear'  
print(f"Best performing model: {best_model_name}")


from sklearn.linear_model import LinearRegression

# Create and train the best model (Linear Regression in this case)
best_model = LinearRegression()
best_model.fit(train_pca, train_data['MedHouseVal'])  # Use train_pca and target

# Make predictions on the test data
predictions = best_model.predict(test_pca)  # Use test_pca

# Create a submission DataFrame
submission_df = pd.DataFrame({'id': test_data['id'], 'MedHouseVal': predictions})

# Save predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")

