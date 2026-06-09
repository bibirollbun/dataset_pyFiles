# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as XGBRegressor
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train_df.head()


print("Training Data Info:")
train_df.info()


print("\nTraining Data Description:")
train_df.describe()


# Check for missing values
print("\nMissing Values in Training Data:")
train_df.isnull().sum()


# Distribution of numerical features in training set
plt.figure(figsize=(15, 10))
train_df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.title('Distribution of Numerical Features - Training Set')
plt.show()


# Box plots for numerical features by Sex
numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Sex', y=col, data=train_df)
    plt.title(f'{col} by Sex')
plt.tight_layout()
plt.show()


# Check the distribution of the target variable (Calories)
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Calories'], kde=True, bins=30, color='blue')
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.show()


# Analyze categorical variable 'Sex'
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Sex', palette='Set2')
plt.title('Count of Male and Female')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


# Analyze relationships between features and target variable
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Sex', y='Calories', palette='Set3')
plt.title('Calories Burned by Sex')
plt.xlabel('Sex')
plt.ylabel('Calories')
plt.show()


# Select only numeric columns
numeric_df = train_df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Numeric Features Only)')
plt.tight_layout()
plt.show()


# Pairplot for numerical features
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
sns.pairplot(train_df[numerical_features], diag_kind='kde', corner=True)
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()


# Analyze the relationship between Duration and Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='Duration', y='Calories', hue='Sex', palette='Set1')
plt.title('Duration vs Calories')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.show()


# Analyze the relationship between Heart Rate and Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='Heart_Rate', y='Calories', hue='Sex', palette='Set2')
plt.title('Heart Rate vs Calories')
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Calories')
plt.show()


# Analyze the relationship between Body Temperature and Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='Body_Temp', y='Calories', hue='Sex', palette='Set3')
plt.title('Body Temperature vs Calories')
plt.xlabel('Body Temperature (°C)')
plt.ylabel('Calories')
plt.show()


# Check for outliers using boxplots
plt.figure(figsize=(12, 8))
train_df[numerical_features].boxplot()
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=45)
plt.show()


# Compare distributions between train and test sets
common_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(common_features, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(data=train_df, x=feature, label='Train', alpha=0.5)
    sns.kdeplot(data=test_df, x=feature, label='Test', alpha=0.5)
    plt.title(f'Distribution of {feature}')
    plt.legend()
plt.tight_layout()
plt.show()


# Load datasets
original_train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Apply outlier removal to create cleaned training data
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
train_df = original_train_df.copy()

# Remove outliers using IQR method
for feature in numerical_features:
    Q1 = train_df[feature].quantile(0.25)
    Q3 = train_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_df = train_df[~((train_df[feature] < lower_bound) | (train_df[feature] > upper_bound))]

print(f"Original training data shape: {original_train_df.shape}")
print(f"Training data shape after outlier removal: {train_df.shape}")

