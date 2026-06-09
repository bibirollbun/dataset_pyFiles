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
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


# Set styles
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# --- Step 2: Load Dataset ---
# Adjust path if needed
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


# --- Step 3: Basic Information ---
print("Shape of dataset:", train.shape)
print("\nInfo about dataset:\n")
print(train.info())
print("\nFirst 5 rows:\n")
print(train.head())


# --- Step 4: Missing Values ---
print("\nMissing Values Summary:\n")
missing = train.isnull().sum()
missing_percent = (missing / train.shape[0]) * 100
missing_table = pd.DataFrame({
    'Missing Values': missing,
    'Percent Missing': missing_percent
})
print(missing_table[missing_table['Missing Values'] > 0])


# --- Step 5: Target Variable Analysis ---
target = 'Listening_Time_minutes'
print("\nTarget Statistics:\n")
print(train[target].describe())

# Target Distribution Plot
plt.figure(figsize=(10,6))
sns.histplot(train[target], kde=True, bins=50)
plt.title('Distribution of Listening Time (minutes)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Count')
plt.show()

# Check skewness
print("\nSkewness of Listening_Time_minutes:", train[target].skew())


# --- Step 6: Numerical Features ---
numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove('id')  # Remove id
numerical_cols.remove(target)  # Remove target

print("\nNumerical Features:", numerical_cols)

# Boxplots
for col in numerical_cols:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# --- Step 7: Categorical Features ---
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Features:", categorical_cols)

# Countplots
for col in categorical_cols:
    plt.figure(figsize=(10,6))
    sns.countplot(x=train[col], order=train[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()



# --- Step 8: Target vs Categorical Features ---
for col in categorical_cols:
    plt.figure(figsize=(12,6))
    sns.boxplot(x=col, y=target, data=train)
    plt.title(f'{target} vs {col}')
    plt.xticks(rotation=45)
    plt.show()


# --- Step 9: Correlation Analysis (Fixed) ---

# Select only numeric columns
numeric_cols = train.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix on numeric columns only
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Correlation with target
target_corr = corr_matrix['Listening_Time_minutes'].sort_values(ascending=False)
print("\nCorrelation of features with Listening_Time_minutes:\n")
print(target_corr)




# --- Function 1: Load Data ---
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


# --- Function 2: Summarize Data ---
def summarize_data(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nTarget Summary:\n", df['Listening_Time_minutes'].describe())


# --- Function 3: Clean Data (Handle Missing) ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaNs with mode
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# --- Function 4: Feature Engineering ---
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Example: convert day to ordered integer (if exists)
    if 'Day_of_Week' in df.columns:
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df['Day_of_Week'] = df['Day_of_Week'].map(day_map)

    # Encode sentiment as ordinal (if applicable)
    if 'Sentiment' in df.columns:
        sent_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        df['Sentiment'] = df['Sentiment'].map(sent_map)

    return df


# --- Function 5: Encode Categorical Features ---
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = df.select_dtypes(include='object').columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df


# --- Function 6: Scale Features ---
def scale_features(df: pd.DataFrame, target_col='Listening_Time_minutes'):
    X = df.drop(columns=['id', target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# --- Function 7: Full Preprocessing Pipeline ---
def preprocess_pipeline(file_path: str):
    df = load_data(file_path)
    summarize_data(df)

    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)

    X_scaled, y, scaler = scale_features(df)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("âœ… Preprocessing complete. Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)

    return X_train, X_val, y_train, y_val, scaler


# Run pipeline on your training Excel
file_path = '/kaggle/input/playground-series-s5e4/train.csv'
X_train, X_val, y_train, y_val, scaler = preprocess_pipeline(file_path)



# --- Function 1: Train All Models ---
def train_models(X_train, y_train, X_val, y_val):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = sqrt(mean_squared_error(y_val, preds))
        results[name] = {'model': model, 'rmse': rmse}
        print(f'{name} RMSE: {rmse:.4f}')

    return results


# --- Function 2: Choose Best Model ---
def get_best_model(results):
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    print(f"\nâœ… Best Model: {best_model_name} with RMSE = {results[best_model_name]['rmse']:.4f}")
    return results[best_model_name]['model']


# --- Function 3: Preprocess Test Data ---
def preprocess_test_data(test_file_path, scaler, train_columns):
    test_df = pd.read_csv(test_file_path)

    # Clean, engineer, encode similar to train
    test_df = clean_data(test_df)
    test_df = feature_engineering(test_df)
    test_df = encode_categoricals(test_df)

    # Align columns
    for col in train_columns:
        if col not in test_df.columns:
            test_df[col] = 0  # Add missing columns

    test_df = test_df[train_columns]

    X_test_scaled = scaler.transform(test_df.drop(columns=['id']))

    return test_df['id'], X_test_scaled



# Train models
results = train_models(X_train, y_train, X_val, y_val)

# Pick the best one
best_model = get_best_model(results)

# Preprocess test set using scaler and training columns
train_df = load_data(file_path)
train_df = clean_data(train_df)
train_df = feature_engineering(train_df)
train_df = encode_categoricals(train_df)
train_columns = train_df.drop(columns=['Listening_Time_minutes']).columns

test_file = '/kaggle/input/playground-series-s5e4/test.csv'
test_ids, X_test = preprocess_test_data(test_file, scaler, train_columns)



# --- Function 4: Generate Submission ---
def generate_submission(model, X_test, test_ids, filename="submission.csv"):
    predictions = model.predict(X_test)
    submission = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': predictions})
    submission.to_csv(filename, index=False)
    print(f"\nðŸ“„ Submission saved to: {filename}")



# Generate final submission
generate_submission(best_model, X_test, test_ids)

