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


# Load data
df_train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
df_test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


df_train.head(5)


df_train.columns


target_variable="label"


import seaborn as sns
import matplotlib.pyplot as plt
  
def eda_pipeline(df_train, df_test):
    
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
    
    # # Summary statistics
    # print("\n--- Train Data Summary Statistics ---")
    # print(df_train.describe())
    
    # print("\n--- Test Data Summary Statistics ---")
    # print(df_test.describe())
    
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
    
    # # Identify numerical columns
    # train_num_columns = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64']]
    # test_num_columns = [col for col in df_test.columns if df_test[col].dtype in ['int64', 'float64']]
    
    # print("\n--- Numerical Columns in Train Data ---")
    # print(train_num_columns)
    
    # print("\n--- Numerical Columns in Test Data ---")
    # print(test_num_columns)
    
    # Check for duplicate rows
    print("\n--- Duplicate Rows in Train Data ---")
    print(df_train.duplicated().sum())
    
    print("\n--- Duplicate Rows in Test Data ---")
    print(df_test.duplicated().sum())
    
    # # Correlation matrix (excluding non-numeric columns)
    # print("\n--- Correlation Matrix ---")
    # plt.figure(figsize=(12, 6))
    # sns.heatmap(df_train[train_num_columns].corr(), annot=True, cmap='coolwarm')
    # plt.show()
       
    # # Correlation with Target Variable
    # print("\n--- Correlation with Target Variable ---")
    # target_corr = df_train[train_num_columns].corr()[target_variable].sort_values(ascending=False)
    # print(target_corr)
    
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
    # plt.xticks(rotation=90)
    # plt.title(f'Feature Correlation with {target_variable}')
    # plt.show()   
    
    # # Distribution plots for numerical features
    # print("\n--- Distribution of Numerical Features ---")
    # df_train[train_num_columns].hist(figsize=(12, 10), bins=30)
    # plt.show()
    
    # # Box plots for outlier detection
    # print("\n--- Box Plots for Outlier Detection ---")
    # for col in train_num_columns:
    #     plt.figure(figsize=(8, 4))
    #     sns.boxplot(x=df_train[col])
    #     plt.title(f'Box plot of {col}')
    #     plt.show()
    
    # # Value counts for categorical features
    # print("\n--- Value Counts for Categorical Columns ---")
    # for col in train_cat_columns:
    #     print(f"\nValue counts for {col}:")
    #     print(df_train[col].value_counts())


eda_pipeline(df_train, df_test)


# # Target Distribution Check
# print("\n--- Distribution of Target Variable for Class Balance Check ---\n")
# df_train[target_variable].value_counts(normalize=True).plot(kind='barh')


from sklearn.preprocessing import LabelEncoder

def data_preprocessing_pipeline(df_train, df_test, target_column='label'):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.
    Returns processed DataFrames and the label encoder for the target column.
    """
    # Fill missing values
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            mode_value = df_train[column].mode()[0]
            df_train[column].fillna(mode_value, inplace=True)
        elif df_train[column].dtype in ['int64', 'float64']:
            mean_value = df_train[column].mean()
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
    target_encoder = None  # separate encoder for target column

    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            le = LabelEncoder()
            df_train[column] = le.fit_transform(df_train[column].astype(str))
            label_encoders[column] = le

            if column == target_column:
                target_encoder = le  # store encoder for target

    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            if column in label_encoders:
                le = label_encoders[column]
                df_test[column] = df_test[column].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                df_test[column] = -1

    return df_train, df_test, target_encoder


# data_preprocessing_pipeline(df_train, df_test, target_column='label')


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


X = df_train.drop(columns=[target_variable])
y = df_train[target_variable]


X.head(2)


y.head(2)


df_test.head(2)


# Check if X's index (timestamp) is sorted
is_sorted = X.index.is_monotonic_increasing
print("Is X index sorted by time?\n", is_sorted)


is_y_sorted = y.index.is_monotonic_increasing
print("Is y index sorted by time?\n", is_y_sorted)


if not X.index.is_monotonic_increasing:
    print("Sorting X and y by timestamp...")
    X = X.sort_index()
    y = y.loc[X.index]
else:
    print("Timestamps already sorted.")


# Use the last 10% as validation
val_size = int(len(X) * 0.1)

X_train = X.iloc[val_size:]
y_train = y.iloc[val_size:]
X_val = X.iloc[:val_size]
y_val = y.iloc[:val_size]


X_train.shape,y_train.shape, X_val.shape, y_val.shape, X.shape,y.shape





from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


lgbm_model = LGBMRegressor(
    objective='regression',
    learning_rate=0.02,
    num_leaves=64,
    n_estimators=1000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    force_col_wise=True,
    verbosity=-1,
    random_state=42
)


lgbm_model.fit(X_train, y_train)


# Predict on validation set
val_preds = lgbm_model.predict(X_val)

# Pearson correlation (competition metric)
pearson = pearsonr(y_val, val_preds)[0]
print(f"ðŸ“Š Pearson Correlation on Validation: {pearson:.6f}")


df_sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


df_sub.head(5)


df_test.head(5)


df_test.drop(columns=["label"], axis=1, inplace=True)


# Predict on test set
test_preds = lgbm_model.predict(df_test)

# Load sample submission and replace predictions

df_sub['prediction'] = test_preds
df_sub.to_csv('submission.csv', index=False)

print("âœ… Submission saved!")



df_sub.head(5)


df_sub.shape,df_test.shape




