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


df_train = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")



df_train.drop(['id', 'Surname'], axis=1, inplace=True)
df_test.drop(['id', 'Surname'], axis=1, inplace=True)


X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]


df_train.head(3)


df_train.info(),df_train.info()


df_train.shape,df_test.shape


df_train.describe()


df_train.isnull().sum(),df_test.isnull().sum()


numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Columns:")
print(numerical_columns)
print(f"\nTotal number of numerical columns: {len(numerical_columns)}")


import matplotlib.pyplot as plt
import seaborn as sns
features=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
plt.figure(figsize=(12, 6))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 4, i)  # Adjust grid size based on the number of features
    sns.boxplot(y=df_train[feature])  # Replace train_dataset with your actual dataset
    plt.title(feature)

plt.tight_layout()
plt.show()


for feature in features:
    Q1 = df_train[feature].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df_train[feature].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    
    # Define the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the dataset to remove outliers
    df_train = df_train[(df_train[feature] >= lower_bound) & (df_train[feature] <= upper_bound)]

# Display the dataset after outlier removal
df_train.shape


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


plt.figure(figsize=(12, 8))

corr_matrix = df_train[features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Display the heatmap
plt.title('Correlation Heatmap for Numerical Features')
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


# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    
    ("scaler", StandardScaler())  # Standardize numerical features
])



# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
   
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


y_pred = pipeline.predict(X_test)


y_test_pred = pipeline.predict(df_test)


submission_df = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')
submission_df['Exited'] = y_test_pred
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

