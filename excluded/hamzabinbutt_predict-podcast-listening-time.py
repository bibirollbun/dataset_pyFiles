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


# Data Handling
import pandas as pd
import numpy as np
# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
# Model Evaluation
from sklearn.metrics import mean_squared_error


# Define the file path
train_file_path = "/kaggle/input/playground-series-s5e4/train.csv"
test_file_path = "/kaggle/input/playground-series-s5e4/train.csv"
# Read the CSV file
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
# Display the first few rows
train_df.head()



train_df = train_df.drop(columns=['id'])
train_df


# Identify numerical and categorical variables
numerical_vars = train_df.select_dtypes(include=['number']).columns.tolist()
categorical_vars = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Print the results
print("Numerical Variables:", numerical_vars)
print("Categorical Variables:", categorical_vars)



# Check for missing values
missing_values = train_df.isnull().sum()

# Separate missing values for numerical and categorical variables
missing_numerical = missing_values[train_df.select_dtypes(include=['number']).columns]
missing_categorical = missing_values[train_df.select_dtypes(include=['object', 'category']).columns]

# Display results
print("Missing Values in Numerical Variables:\n", missing_numerical)
print("\nMissing Values in Categorical Variables:\n", missing_categorical)

# Percentage of missing values
missing_percentage = (missing_values / len(train_df)) * 100
print("\nPercentage of Missing Values:\n", missing_percentage[missing_percentage > 0])



# Create imputers
median_imputer = SimpleImputer(strategy='median')
mode_imputer = SimpleImputer(strategy='most_frequent')

# Apply median imputation for numerical variables with significant missing values
train_df[['Episode_Length_minutes', 'Guest_Popularity_percentage']] = median_imputer.fit_transform(
    train_df[['Episode_Length_minutes', 'Guest_Popularity_percentage']]
)

# Apply mode imputation for 'Number_of_Ads'
train_df[['Number_of_Ads']] = mode_imputer.fit_transform(train_df[['Number_of_Ads']])

# Check if missing values are handled
print(train_df.isnull().sum())



# List of numerical variables
numerical_vars = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                  'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']

# Boxplot visualization for outliers
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_vars, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=train_df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Function to detect outliers using IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Detecting outliers for each numerical column
outlier_counts = {}
for col in numerical_vars:
    outliers = detect_outliers_iqr(train_df, col)
    outlier_counts[col] = len(outliers)

# Display outlier counts
print("Outlier Counts per Column:")
for col, count in outlier_counts.items():
    print(f"{col}: {count} outliers")



# Define categorical columns
categorical_vars = ['Podcast_Name', 'Episode_Title', 'Genre', 
                    'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Ensure categorical variables exist in the dataset
available_cats = [col for col in categorical_vars if col in train_df.columns]

# Apply Label Encoding
label_encoders = {}
for col in available_cats:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])  # Transform categorical column
    label_encoders[col] = le  # Store encoder for inverse transformation if needed

print(train_df.head())  # Check transformed dataset



# Define features (X) and target variable (y)
X = train_df.drop(columns=['Listening_Time_minutes'])  # Drop target variable
y = train_df['Listening_Time_minutes']  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")  # Check data split



# Initialize Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
# Train the model
rf_regressor.fit(X_train, y_train)
# Make predictions
y_pred = rf_regressor.predict(X_test)
# Evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")


test_df = test_df.drop(columns=['id'])
test_df


# Check for missing values
missing_values = test_df.isnull().sum()

# Separate missing values for numerical and categorical variables
missing_numerical = missing_values[test_df.select_dtypes(include=['number']).columns]
missing_categorical = missing_values[test_df.select_dtypes(include=['object', 'category']).columns]

# Display results
print("Missing Values in Numerical Variables:\n", missing_numerical)
print("\nMissing Values in Categorical Variables:\n", missing_categorical)

# Percentage of missing values
missing_percentage = (missing_values / len(test_df)) * 100
print("\nPercentage of Missing Values:\n", missing_percentage[missing_percentage > 0])



# Create imputers
median_imputer = SimpleImputer(strategy='median')
mode_imputer = SimpleImputer(strategy='most_frequent')

# Apply median imputation for numerical variables with significant missing values
test_df[['Episode_Length_minutes', 'Guest_Popularity_percentage']] = median_imputer.fit_transform(
    test_df[['Episode_Length_minutes', 'Guest_Popularity_percentage']]
)

# Apply mode imputation for 'Number_of_Ads'
test_df[['Number_of_Ads']] = mode_imputer.fit_transform(test_df[['Number_of_Ads']])

# Check if missing values are handled
test_df.isnull().sum()



# Define categorical columns
categorical_vars = ['Podcast_Name', 'Episode_Title', 'Genre', 
                    'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Ensure categorical variables exist in the dataset
available_cats = [col for col in categorical_vars if col in test_df.columns]

# Apply Label Encoding
label_encoders = {}
for col in available_cats:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])  # Transform categorical column
    label_encoders[col] = le  # Store encoder for inverse transformation if needed

test_df.head()  # Check transformed dataset



# Ensure test_df has the same preprocessing as X_train
X_test_final = test_df[X_train.columns]  # Select the same features as training
# Make predictions on the test set
test_df["Predicted_Listening_Time"] = rf_regressor.predict(X_test_final)
# Display the first few predictions
print(test_df[["Predicted_Listening_Time"]].head())

