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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_data.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_data.head()


import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load training and test datasets
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Fill missing values in 'Brand' with a placeholder
train_data['Brand'] = train_data['Brand'].fillna('Unknown')
test_data['Brand'] = test_data['Brand'].fillna('Unknown')

# Feature engineering: creating new columns based on existing ones
def engineer_features(df):
    # Frequency encoding for brand
    brand_counts = train_data['Brand'].value_counts()
    df['Brand_Popularity'] = df['Brand'].map(brand_counts).fillna(1)
    
    # Binary flag for premium materials
    premium_materials = ['Leather', 'Canvas', 'Nylon']
    df['Premium_Material'] = df['Material'].isin(premium_materials).astype(int)
    
    # Map size categories to numeric values
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Numeric'] = df['Size'].map(size_mapping)
    
    # Create interaction features
    df['Size_Weight'] = df['Size_Numeric'] * df['Weight Capacity (kg)']
    df['Compartments_Weight'] = df['Compartments'] * df['Weight Capacity (kg)']
    
    return df

# Apply feature engineering to both datasets
train_data = engineer_features(train_data)
test_data = engineer_features(test_data)

# Define categorical and numerical feature lists
cat_cols = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']
num_cols = ['Compartments', 'Weight Capacity (kg)', 'Brand_Popularity', 'Premium_Material',
            'Size_Numeric', 'Size_Weight', 'Compartments_Weight']

# Define input features and target variable
X = train_data[cat_cols + num_cols]
y = train_data['Price']
X_test = test_data[cat_cols + num_cols]

# Preprocessing pipelines for numerical and categorical data
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy='median'))
])

# Combine transformers into a single column transformer
preprocessor = ColumnTransformer([
    ("cat", cat_transformer, cat_cols),
    ("num", num_transformer, num_cols)
])

# Build the full modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_iter=250,
        max_leaf_nodes=31,
        max_depth=6,
        random_state=42))
])

# Train the model
model.fit(X, y)

# Make predictions on the test set
predictions = model.predict(X_test)

# Save the predictions in the required format
submission = pd.DataFrame({'id': test_data['id'], 'Price': predictions})
submission.to_csv('submission_verbeserung.csv', index=False)

print("Improved model trained and submission file created successfully!")


