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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import display  
import warnings
warnings.filterwarnings("ignore")


# Load & Display shapes and columns of datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
original_data = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')
print(f"Training data shape: {train_data.shape}")
print(f"Original dataset shape: {original_data.shape}")
print("Columns in training data:", list(train_data.columns))
print("Columns in original data:", list(original_data.columns))


train_data.info()
original_data.info()


train_data.head()


#check duplicated
print("Number of duplicated rows:", train_data.duplicated().sum())


train_data.isna().sum()
plt.figure(figsize=(14, 8))
sns.heatmap(train_data.isnull(), cbar=False, cmap='plasma', yticklabels=False, alpha=0.8)
plt.title('Missing Data Visualization', fontsize=18, fontweight='bold')
plt.xticks(rotation=45, fontsize=10)
plt.show()


# One-Hot Encoding for categorical features
encoded_data = pd.get_dummies(train_data, drop_first=False)

# Calculate the correlation matrix
correlation_matrix = encoded_data.corr()

# Plot the correlation heatmap with a different colormap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='viridis',  # Changed colormap to 'viridis'
    fmt='.1f', 
    linewidths=0.2,
    cbar=True,
    cbar_kws={'label': 'Correlation Strength'}
)
plt.title('Correlation Heatmap of Encoded Features', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.show()


# Boxplot for Price by categorical variables
categorical_columns = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Compartments']  # Categorical columns to analyze

# Create subplots for each categorical variable
plt.figure(figsize=(14, 10))
for i, col in enumerate(categorical_columns):
    plt.subplot(2, 3, i+1)
    sns.boxplot(
        x=train_data[col], 
        y=train_data['Price'], 
        palette='pastel',  # Changed palette to 'pastel'
        showmeans=True,  # Show mean values
        meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'}  # Customize mean markers
    )
    plt.title(f'Price Distribution by {col}', fontsize=12, pad=10)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)

plt.suptitle('Price Distribution Across Categorical Features', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


import pandas as pd

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Simple null handling for categorical features - fill with mode
cols = [col for col in test.select_dtypes('object').columns]

for col in cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)

# Simple null handling for numerical features - fill with mean (since the columns aren't skewed)
train['Price'].fillna(train['Price'].mean(), inplace=True)
train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean(), inplace=True)

test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean(), inplace=True)



import pandas as pd

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Simple null handling for categorical features - fill with mode
cols = test.select_dtypes('object').columns

for col in cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

# Simple null handling for numerical features - fill with mean (since the columns aren't skewed)
train['Price'] = train['Price'].fillna(train['Price'].mean())
train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean())

test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean())




from sklearn.model_selection import train_test_split

# Load train and original data (Make sure these files/variables exist)
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')  # Update with actual path
original_data = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')  # Update with actual path

# Combine train and original data for more training samples
combined_data = pd.concat([train_data, original_data], axis=0).reset_index(drop=True)

# Ensure the target column exists
target_col = 'Price'  # Replace 'Price' if your target column has a different name

# Separate features and target
X = combined_data.drop(target_col, axis=1)
y = combined_data[target_col]

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from catboost import CatBoostRegressor
import numpy as np

# Identify categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Ensure categorical features have no NaN values and are strings
for col in categorical_features:
    X_train[col] = X_train[col].fillna('missing').astype(str)
    X_val[col] = X_val[col].fillna('missing').astype(str)

# Ensure target variable has no NaN values
y_train = y_train.fillna(y_train.mean())  # Replace NaNs with the mean (or use another strategy)
y_val = y_val.fillna(y_train.mean())  # Use `y_train.mean()` to avoid data leakage

# Initialize CatBoostRegressor
model = CatBoostRegressor(
    cat_features=categorical_features,  # Automatically handle categorical features
    random_seed=42,
    verbose=100  # Print training progress every 100 iterations
)

# Train the model
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)



X_test = test.copy()  # Replace 'test_data' with your actual test set variable

# Extract IDs from test data (adjust the column name if needed)
test_ids = X_test.index  # If test IDs are stored in index
# OR
# test_ids = X_test['Id']  # If IDs are in a specific column named 'Id'
# Extract IDs from test data
if 'Id' in X_test.columns:
    test_ids = X_test['Id']  # Use 'Id' if it exists in the dataset
elif 'id' in X_test.columns:
    test_ids = X_test['id']  # Sometimes datasets use lowercase 'id'
else:
    test_ids = pd.Series(range(1, len(X_test) + 1), name='Id')  # Generate dummy IDs if missing

# Ensure categorical features in test set have no NaN values and are strings
for col in categorical_features:
    X_test[col] = X_test[col].fillna('missing').astype(str)

# Make predictions
y_pred = model.predict(X_test)

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,  # Ensure column name matches Kaggle competition requirements
    'Price': y_pred  # Replace 'Price' with the actual target column name
})

# Save the predictions to CSV
submission.to_csv('submission.csv', index=False)
submission.to_csv('submissionV1.csv', index=False)

# Display the first few rows of the predictions
print(submission.head(10))

