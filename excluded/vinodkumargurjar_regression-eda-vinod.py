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


df_train=pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s4e12/sample_submission.csv")


df_train.head(5)


df_train.drop(columns="id", axis=1, inplace=True)
df_test.drop(columns="id", axis=1, inplace=True)


# Convert 'date' to datetime format
df_train['Policy Start Date'] = pd.to_datetime(df_train['Policy Start Date'])
df_test['Policy Start Date'] = pd.to_datetime(df_test['Policy Start Date'])


import seaborn as sns
import matplotlib.pyplot as plt
target_variable="Premium Amount"
def eda_pipeline(df_train, df_test):
    target_variable="Premium Amount"
    # Display first few rows
    print("\n--- First few rows of train data ---")
    display(df_train.head())
    
    print("\n--- First few rows of test data ---")
    display(df_test.head())
    
    # Dataset info
    print("\n--- Train Data Info ---")
    print(df_train.info())
    
    print("\n--- Test Data Info ---")
    print(df_test.info())
    
    # Missing values
    print("\n--- Missing Values in Train Data ---")
    print(df_train.isnull().sum())
    
    print("\n--- Missing Values in Test Data ---")
    print(df_test.isnull().sum())
    
    print("\n--- Percentage of Missing Values in Train Data ---")
    print((df_train.isnull().sum() / len(df_train)) * 100)
    
    print("\n--- Percentage of Missing Values in Test Data ---")
    print((df_test.isnull().sum() / len(df_test)) * 100)
    
    # Summary statistics
    print("\n--- Train Data Summary Statistics ---")
    print(df_train.describe())
    
    print("\n--- Test Data Summary Statistics ---")
    print(df_test.describe())
    
    # Identify categorical columns
    train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
    test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']
    
    print("\n--- Categorical Columns in Train Data ---")
    print(train_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Train) ---")
    print(df_train[train_cat_columns].nunique())
    
    print("\n--- Categorical Columns in Test Data ---")
    print(test_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Test) ---")
    print(df_test[test_cat_columns].nunique())
    
    # Identify numerical columns
    train_num_columns = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64']]
    test_num_columns = [col for col in df_test.columns if df_test[col].dtype in ['int64', 'float64']]
    
    print("\n--- Numerical Columns in Train Data ---")
    print(train_num_columns)
    
    print("\n--- Numerical Columns in Test Data ---")
    print(test_num_columns)
    
    # Check for duplicate rows
    print("\n--- Duplicate Rows in Train Data ---")
    print(df_train.duplicated().sum())
    
    print("\n--- Duplicate Rows in Test Data ---")
    print(df_test.duplicated().sum())
    
    # Correlation matrix (excluding non-numeric columns)
    print("\n--- Correlation Matrix ---")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_train[train_num_columns].corr(), annot=True, cmap='coolwarm')
    plt.show()
    
    # Correlation with Target Variable
    print("\n--- Correlation with Target Variable ---")
    target_corr = df_train[train_num_columns].corr()[target_variable].sort_values(ascending=False)
    print(target_corr)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
    plt.xticks(rotation=90)
    plt.title(f'Feature Correlation with {target_variable}')
    plt.show()   
    
    # Distribution plots for numerical features
    print("\n--- Distribution of Numerical Features ---")
    df_train[train_num_columns].hist(figsize=(12, 10), bins=30)
    plt.show()
    
    # Box plots for outlier detection
    print("\n--- Box Plots for Outlier Detection ---")
    for col in train_num_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_train[col])
        plt.title(f'Box plot of {col}')
        plt.show()
    
    # Value counts for categorical features
    print("\n--- Value Counts for Categorical Columns ---")
    for col in train_cat_columns:
        print(f"\nValue counts for {col}:")
        print(df_train[col].value_counts())


eda_pipeline(df_train, df_test)


from sklearn.preprocessing import LabelEncoder

def data_preprocessing_pipeline(df_train, df_test):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.
    """
    # Fill missing values
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            mode_value = df_train[column].mode()[0]  # Fill categorical with mode
            df_train[column].fillna(mode_value, inplace=True)
        elif df_train[column].dtype in ['int64', 'float64']:
            mean_value = df_train[column].mean()  # Fill numerical with mean
            df_train[column].fillna(mean_value, inplace=True)
    
    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            mode_value = df_test[column].mode()[0]
            df_test[column].fillna(mode_value, inplace=True)
        elif df_test[column].dtype in ['int64', 'float64']:
            mean_value = df_test[column].mean()
            df_test[column].fillna(mean_value, inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            le = LabelEncoder()
            df_train[column] = le.fit_transform(df_train[column].astype(str))
            label_encoders[column] = le  # Store encoder for consistency
    
    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            if column in label_encoders:
                df_test[column] = label_encoders[column].transform(df_test[column].astype(str))
            else:
                le = LabelEncoder()
                df_test[column] = le.fit_transform(df_test[column].astype(str))
    
    return df_train, df_test




# df_train, df_test = data_preprocessing_pipeline(df_train, df_test)


# df_train.head(5)


# df_test.head(5)


# Create time-based features
df_train['year'] = df_train['Policy Start Date'].dt.year
df_train['month'] = df_train['Policy Start Date'].dt.month
df_train['day'] = df_train['Policy Start Date'].dt.day
df_train['dayofweek'] = df_train['Policy Start Date'].dt.dayofweek


# Preprocess test set
df_test['year'] = df_test['Policy Start Date'].dt.year
df_test['month'] = df_test['Policy Start Date'].dt.month
df_test['day'] = df_test['Policy Start Date'].dt.day
df_test['dayofweek'] = df_test['Policy Start Date'].dt.dayofweek


df_train.drop(columns="Policy Start Date", axis=1, inplace=True)
df_test.drop(columns="Policy Start Date", axis=1, inplace=True)


train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']


pd.set_option("display.max_columns",None)


df_train.head(3)


df_test.head(3)


df_train.shape,df_test.shape


from sklearn.preprocessing import StandardScaler

def standardize_data(df_train, df_test):
    """
    Standardize all numerical features using StandardScaler,
    ensuring both train and test have the same columns, while preserving the target variable.
    """
    # Separate target column from train data
    target_values = df_train[target_variable]
    df_train = df_train.drop(columns=[target_variable])
    
    # Ensure both datasets have the same feature columns
    common_columns = df_train.columns.intersection(df_test.columns)
    df_train = df_train[common_columns]
    df_test = df_test[common_columns]
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit on train data and transform both train and test data
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=common_columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=common_columns)
    
    # Reattach the target column to the scaled train data
    df_train_scaled[target_variable] = target_values.reset_index(drop=True)
    
    return df_train_scaled, df_test_scaled


# df_train_scaled, df_test_scaled = standardize_data(df_train, df_test)


# df_train_scaled.head(5)


# df_test_scaled.head(5)


X = df_train.drop(columns=[target_variable])
y = df_train[target_variable]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, r2_score,mean_squared_log_error
# # Initialize and train the XGBoost Regressor
# xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=10, random_state=42)
# xgb_model.fit(X_train, y_train)

# # Predictions
# y_pred = xgb_model.predict(X_test)

# # Model Evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# # Root Mean Squared Log Error (RMSLE)
# rmsle = np.sqrt(mean_squared_log_error(y_test, np.maximum(y_pred, 0))) 

# print(f"Mean Squared Error: {mse:.4f}")
# print(f"R² Score: {r2:.4f}")
# print(f"Root Mean Squared Log Error (RMSLE): {rmsle:.4f}")


train_cat_columns


for col in train_cat_columns:
    X_train[col] = X_train[col].astype(str).fillna("Unknown")
    X_test[col] = X_test[col].astype(str).fillna("Unknown")


from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error

# Initialize and train the CatBoost Regressor
catboost_model = CatBoostRegressor(iterations=500, 
                                   learning_rate=0.1, 
                                   depth=6, 
                                   random_seed=42, 
                                   verbose=100, cat_features=train_cat_columns)  # Shows progress every 100 iterations

catboost_model.fit(X_train, y_train)

# Predictions
y_pred = catboost_model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, np.maximum(y_pred, 0)))  # Ensure no negative values

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Log Error (RMSLE): {rmsle:.4f}")






for col in test_cat_columns:
    df_test[col] = df_test[col].astype(str).fillna("Unknown")
    


final_result=catboost_model.predict(df_test)


final_result


sample_submission.head(4)


sample_submission["Premium Amount"]=final_result
sample_submission.to_csv('submission.csv',index=False)


sample_submission.head(4)




