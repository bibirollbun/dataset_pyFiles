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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")
X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]


df_train.head(3)


df_test.head(3)


df_train.shape


df_test.shape


df_train.isnull().sum()


df_test.isnull().sum()


numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)
print(f"\nTotal number of numerical columns: {len(numerical_columns)}")

print("\nCategorical Columns:")
print(categorical_columns)
print(f"\nTotal number of categorical columns: {len(categorical_columns)}")


df_train[numerical_columns] = df_train[numerical_columns].fillna(df_train[numerical_columns].mean())
df_test[numerical_columns] = df_test[numerical_columns].fillna(df_test[numerical_columns].mean())


df_train[categorical_columns] = df_train[categorical_columns].fillna("Missing")
df_test[categorical_columns] = df_test[categorical_columns].fillna("Missing")


df_train.isnull().sum()


df_test.isnull().sum()


df_train.describe()


df_train.columns


features=['Age','Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration']
plt.figure(figsize=(12, 6))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 4, i)  # Adjust grid size based on the number of features
    sns.boxplot(y=df_train[feature])  # Replace train_dataset with your actual dataset
    plt.title(feature)

plt.tight_layout()
plt.show()


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    
    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Remove outliers from 'Annual Income' and 'Previous Claims'
df_train = remove_outliers(df_train, 'Annual Income')
df_train = remove_outliers(df_train, 'Previous Claims')

# Check the dataset after removing outliers
print(df_train.shape)


features=['Age','Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration']
plt.figure(figsize=(12, 6))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 4, i)  # Adjust grid size based on the number of features
    sns.boxplot(y=df_train[feature])  # Replace train_dataset with your actual dataset
    plt.title(feature)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 4, i)  # Adjust grid size based on the number of features
    sns.histplot(df_train[feature], bins=30, kde=True)  # KDE=True adds a density curve
    plt.title(feature)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 10))

for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)  # Adjust grid size based on the number of features
    sns.scatterplot(x=df_train[feature], y=df_train['Premium Amount'])  # Replace 'Target' with actual target variable
    plt.title(f"{feature} vs Target")

plt.tight_layout()
plt.show()


sns.pairplot(df_train[features])
plt.show()




# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

corr_matrix = df_train[features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Display the heatmap
plt.title('Correlation Heatmap for Numerical Features')
plt.show()


featues_categorical = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 'Policy Type', 'Policy Start Date', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']


plt.figure(figsize=(15, 12))

for i, feature in enumerate(featues_categorical, 1):
    plt.subplot(4, 3, i)  # Adjust number of rows and columns based on the number of features
    sns.countplot(data=df_train, x=feature)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed

plt.tight_layout() 
plt.show()


for feature in featues_categorical:
    plt.figure(figsize=(7, 7))
    df_train[feature].value_counts().plot.pie(autopct='%1.1f%%', figsize=(7,7))
    plt.title(f'Proportions of {feature}')
    plt.ylabel('')
    plt.show()


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    
    ("scaler", StandardScaler())  # Standardize numerical features
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
   
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # One-hot encoding for categorical variables
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create a pipeline with XGBoost Regressor
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
score = pipeline.score(X_test, y_test)
print(f"Model R² Score: {score:.4f}")


X_test_dataset = preprocessor.transform(df_test)


y_pred = pipeline.predict(X_test)



y_test_pred = pipeline.predict(df_test)


submission_df = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')
submission_df['Premium Amount'] = y_test_pred
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

