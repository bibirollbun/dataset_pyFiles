# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from pprint import pprint
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


traindf = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
testdf = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

traindf = traindf.drop_duplicates()

traindf.head()


print("Unique Columns: ")
pprint(traindf.nunique())


pprint((traindf.isnull().sum()/len(traindf['id']))*100)


traindf.describe()


# Outliers
import pandas as pd

def detect_outliers(df, column, threshold=3):
    """
    Detects outliers in the specified column of a DataFrame using the standard deviation method.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name on which to detect outliers.
        threshold (float): The number of standard deviations from the mean to use as cutoff (default is 3).
    
    Returns:
        pd.DataFrame: A DataFrame containing only the outlier rows.
    """
    # Calculate mean and standard deviation of the column
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    # Set bounds for outliers
    lower_bound = mean_val - threshold * std_val
    upper_bound = mean_val + threshold * std_val
    
    # Filter and return outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

for col in ["Compartments",	"Weight Capacity (kg)", "Price"]:
    pprint(detect_outliers(traindf.copy(), col))


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor_pipeline(df):
    """
    Build a preprocessing pipeline that automatically identifies numeric and categorical columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        preprocessor (ColumnTransformer): The constructed preprocessor pipeline.
        numeric_features (list): List of numeric column names.
        categorical_features (list): List of categorical column names.
    """
    # Auto-detect column types
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if 'Price' in numeric_features:
        numeric_features.remove('Price')

    numeric_features.remove('id')
    
    # Numeric pipeline: Impute and scale
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute and one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor, numeric_features, categorical_features

# Build the preprocessor dynamically
preprocessor, numeric_features, categorical_features = build_preprocessor_pipeline(traindf.copy())

# Fit and transform the data
processed_data = preprocessor.fit_transform(traindf.copy())

cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
# Combine numeric and categorical feature names
feature_names = numeric_features + list(cat_feature_names)
processed_df = pd.DataFrame(processed_data, columns=feature_names)

pprint(processed_data.shape)
processed_df['Price'] = traindf['Price']
processed_df.head()


# Calculate the correlation matrix
correlation_matrix = processed_df.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


feature_names = [
'Compartments',
'Weight Capacity (kg)',
'Brand_Jansport',
'Brand_Nike',
'Brand_Puma',
'Brand_Under Armour',
'Material_Leather',
'Material_Nylon',
'Material_Polyester',
'Size_Medium',
'Size_Small',
'Laptop Compartment_Yes',
'Waterproof_Yes',
'Style_Messenger',
'Style_Tote',
'Color_Blue',
'Color_Gray',
'Color_Green',
'Color_Pink',
'Color_Red']
X = np.array(processed_df[feature_names])
pprint(X.shape)
y = np.array(processed_df['Price'])
pprint(y.shape)


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Add a bias term (intercept) by appending a column of 1s to X
def add_interept_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]  # X_b = [1, X]

def cost_function(theta, X, y):
    """
    Root Mean Squared Error (RMSE) cost function for polynomial regression.
    """
    predictions = X.dot(theta)  # Model predictions
    error = predictions - y     # Prediction error
    rmse = np.sqrt((1 / len(y)) * np.sum(error ** 2))  # RMSE formula
    return rmse

def train_regression(X, y):
    """
    Train linear regression using SciPy's minimize function.
    """
    initial_theta = np.zeros(X.shape[1])  # Initialize parameters [intercept, slope]
    
    # Minimize the cost function to find optimal theta
    result = minimize(cost_function, initial_theta, args=(X, y), method='BFGS')
    
    if result.success:
        print("Optimization successful. Found parameters:", result.x)
        return result.x
    else:
        print(result)
        raise ValueError("Optimization failed.")

def predict(X, theta):
    """
    Predict using the learned linear regression model.
    """
    return X.dot(theta)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
optimal_theta = train_regression(X_train, y_train)

# Predict on the test set
y_pred = predict(X_test, optimal_theta)



def find_error(y_test, y_pred):
    error = y_pred - y_test     # Prediction error
    rmse = np.sqrt((1 / len(y_test)) * np.sum(error ** 2))  # RMSE formula
    return rmse


find_error(y_test, y_pred)


# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual values (Test set)')
plt.scatter(X_test[:, 1], y_pred, color='red', label='Predicted values (Test set)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()


testdf


ids = testdf['id']

testdf = testdf.drop(columns=['id'])
processed_data = preprocessor.fit_transform(testdf.copy())


predictions = predict(processed_data, optimal_theta)


# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'id': ids,  # The id column from the test set
    'prediction': predictions  # The predicted values
})

# Save to CSV (ensure you don't have an index column in the CSV)
submission_df.to_csv('submission.csv', index=False)


submission_df.head()

