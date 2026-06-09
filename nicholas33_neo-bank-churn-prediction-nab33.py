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
base_path = "/kaggle/input/neo-bank-non-sub-churn-prediction/"
training_files = [
    f"{base_path}train_2008.parquet",
    f"{base_path}train_2009.parquet",
    f"{base_path}train_2010.parquet",
    f"{base_path}train_2011.parquet",
    f"{base_path}train_2012.parquet",
    f"{base_path}train_2013.parquet",
    f"{base_path}train_2014.parquet",
    f"{base_path}train_2015.parquet",
    f"{base_path}train_2016.parquet",
    f"{base_path}train_2017.parquet",
    f"{base_path}train_2018.parquet",
    f"{base_path}train_2019.parquet",
    f"{base_path}train_2020.parquet",
    f"{base_path}train_2021.parquet",
    f"{base_path}train_2022.parquet",
    f"{base_path}train_2023.parquet",
]
train_data = pd.concat([pd.read_parquet(file) for file in training_files], ignore_index=True)

test_data = pd.read_parquet(f"{base_path}test.parquet")

# Display basic information about the dataset
print("Training Data Info:")
print(train_data.info())
print("\nTraining Data Head:")
print(train_data.head())

print("\nTest Data Info")
print(test_data.info())
print("\nTest Data Head")
print(test_data.head())


print(train_data.info())


train_data.head(5)


# Assuming your data is in a pandas DataFrame called 'df'
num_customers = train_data['customer_id'].nunique()
print(num_customers) 
num_customers = test_data['customer_id'].nunique()
print(num_customers) 


from datetime import datetime

# Ensure date_of_birth is in datetime format
train_data['date_of_birth'] = pd.to_datetime(train_data['date_of_birth'], errors='coerce')
test_data['date_of_birth'] = pd.to_datetime(test_data['date_of_birth'], errors='coerce')

# Calculate age using the last date in the dataset as a reference
reference_date = train_data['date'].max()  # Use the latest date from the 'date' column
train_data['Age'] = (reference_date - train_data['date_of_birth']).dt.days // 365

# Calculate age using the last date in the dataset as a reference
reference_date = test_data['date'].max()  # Use the latest date from the 'date' column
test_data['Age'] = (reference_date - test_data['date_of_birth']).dt.days // 365


# Check the new column
print(train_data[['date_of_birth', 'Age']].head())
print(test_data[['date_of_birth', 'Age']].head())


# Check data type
print(train_data['date'].dtypes)

# Count missing values
print(f"Number of missing dates: {train_data['date'].isnull().sum()}")

# Print min and max dates
print(f"Earliest date: {train_data['date'].min()}")
print(f"Latest date: {train_data['date'].max()}")


current_date = datetime.now()  # Confirm current date
print(f"Current date used for calculation: {current_date}")


current_date = datetime(2023, 12, 31)
print(f"Reference date for calculation: {current_date}")


train_data['last_tx_days'] = (current_date - train_data['date']).dt.days
test_data['last_tx_days'] = (current_date - train_data['date']).dt.days
# Verify the recalculated values
print(train_data['last_tx_days'].describe())
print(train_data['last_tx_days'].head(10))
print(test_data['last_tx_days'].describe())
print(test_data['last_tx_days'].head(10))


print(train_data['last_tx_days'].quantile([0.25, 0.5, 0.75, 0.95, 0.99]))
print(test_data['last_tx_days'].quantile([0.25, 0.5, 0.75, 0.95, 0.99]))


import matplotlib.pyplot as plt

# Get the value counts for complaints
complaints_counts = train_data['complaints'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(
    complaints_counts,
    labels=complaints_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Paired.colors
)

# Add a title
plt.title("Distribution of Complaints")
plt.axis('equal')  # Equal aspect ratio to ensure pie is a circle
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

financial_features = [
    'atm_transfer_in', 'atm_transfer_out', 'bank_transfer_in', 'bank_transfer_out',
    'crypto_in', 'crypto_out', 'bank_transfer_in_volume', 'bank_transfer_out_volume',
    'crypto_in_volume', 'crypto_out_volume'
]

for feature in financial_features:
    plt.figure(figsize=(4, 2))
    sns.histplot(train_data[feature], bins=30, kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()


train_data['net_balance'] = (
    train_data['bank_transfer_in_volume'] + train_data['crypto_in_volume']
    - train_data['bank_transfer_out_volume'] - train_data['crypto_out_volume']
)

test_data['net_balance'] = (
    test_data['bank_transfer_in_volume'] + test_data['crypto_in_volume']
    - test_data['bank_transfer_out_volume'] - test_data['crypto_out_volume']
)


train_data['total_deposits'] = (
    train_data['atm_transfer_in'] + train_data['bank_transfer_in'] +
    train_data['bank_transfer_in_volume'] + train_data['crypto_in_volume']
)

train_data['total_withdrawals'] = (
    train_data['atm_transfer_out'] + train_data['bank_transfer_out'] +
    train_data['bank_transfer_out_volume'] + train_data['crypto_out_volume']
)


train_data['higher_withdrawals'] = (train_data['total_withdrawals'] > train_data['total_deposits']).astype(int)


test_data['total_deposits'] = (
    test_data['atm_transfer_in'] + test_data['bank_transfer_in'] +
    test_data['bank_transfer_in_volume'] + test_data['crypto_in_volume']
)

test_data['total_withdrawals'] = (
    test_data['atm_transfer_out'] + test_data['bank_transfer_out'] +
    test_data['bank_transfer_out_volume'] + test_data['crypto_out_volume']
)


test_data['higher_withdrawals'] = (test_data['total_withdrawals'] > test_data['total_deposits']).astype(int)


import matplotlib.pyplot as plt

# Count the occurrences of each job
fc_counts = train_data['from_competitor'].value_counts()

# Plot the pie chart
plt.figure(figsize=(5, 4))
plt.pie(fc_counts, labels=fc_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("From competitor")
plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle
plt.show()

churn_due_to_fraud_counts = train_data['churn_due_to_fraud'].value_counts()

# Plot the pie chart
plt.figure(figsize=(5, 4))
plt.pie(churn_due_to_fraud_counts, labels=churn_due_to_fraud_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("churn_due_to_fraud_counts")
plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle
plt.show()

model_predicted_fraud_counts = train_data['model_predicted_fraud'].value_counts()

# Plot the pie chart
plt.figure(figsize=(5, 4))
plt.pie(model_predicted_fraud_counts, labels=model_predicted_fraud_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("model_predicted_fraud_counts")
plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle
plt.show()



# List of columns to convert
boolean_columns = ['from_competitor', 'churn_due_to_fraud', 'model_predicted_fraud']

# Convert True/False to 1/0
train_data[boolean_columns] = train_data[boolean_columns].astype(int)
test_data[boolean_columns] = test_data[boolean_columns].astype(int)


# Calculate total transaction volume
train_data['total_transaction_volume'] = (
    train_data['atm_transfer_in'] + train_data['atm_transfer_out'] +
    train_data['bank_transfer_in'] + train_data['bank_transfer_out'] +
    train_data['crypto_in'] + train_data['crypto_out']
)

train_data['tenure'] = train_data['tenure'].replace(0, 1)  # Replace 0 with 1 to avoid division by zero
# Calculate average transaction volume
train_data['average_transaction_volume'] = (
    train_data['total_transaction_volume'] / train_data['tenure']
)


# Compute recent transaction volume for the last 3 months
train_data['recent_transaction_volume'] = (
    train_data['bank_transfer_in_volume'] + train_data['bank_transfer_out_volume'] +
    train_data['crypto_in_volume'] + train_data['crypto_out_volume']
)


# Calculate total transaction volume for test 
test_data['total_transaction_volume'] = (
    test_data['atm_transfer_in'] + test_data['atm_transfer_out'] +
    test_data['bank_transfer_in'] + test_data['bank_transfer_out'] +
    test_data['crypto_in'] + test_data['crypto_out']
)

test_data['tenure'] = test_data['tenure'].replace(0, 1)  # Replace 0 with 1 to avoid division by zero
# Calculate average transaction volume
test_data['average_transaction_volume'] = (
    test_data['total_transaction_volume'] / test_data['tenure']
)


# Compute recent transaction volume for the last 3 months
test_data['recent_transaction_volume'] = (
    test_data['bank_transfer_in_volume'] + test_data['bank_transfer_out_volume'] +
    test_data['crypto_in_volume'] + test_data['crypto_out_volume']
)



print(train_data[['average_transaction_volume', 'recent_transaction_volume', 'last_tx_days']].head())



import numpy as np
# Select numeric columns
numeric_columns = train_data.select_dtypes(include=['float64', 'int64'])

# Check for infinite values using NumPy
inf_mask = np.isinf(numeric_columns.values)

# Count infinite values in each column
inf_counts = np.sum(inf_mask, axis=0)

# Map results back to column names
inf_values = dict(zip(numeric_columns.columns, inf_counts))

print("Number of infinite values in each column:")
print(inf_values)


import numpy as np
# Select numeric columns
numeric_columns = test_data.select_dtypes(include=['float64', 'int64'])

# Check for infinite values using NumPy
inf_mask = np.isinf(numeric_columns.values)

# Count infinite values in each column
inf_counts = np.sum(inf_mask, axis=0)

# Map results back to column names
inf_values = dict(zip(numeric_columns.columns, inf_counts))

print("Number of infinite values in each column:")
print(inf_values)


import seaborn as sns
import matplotlib.pyplot as plt

numeric_columns = train_data.select_dtypes(include=['float64', 'int64']).columns

for column in numeric_columns:
    plt.figure(figsize=(4, 2))
    sns.boxplot(x=train_data[column])
    plt.title(f'Boxplot for {column}')
    plt.show()


#drop columns
drop_columns = [
    #'Id',                # Unique identifier
    'name',              # Descriptive field
    'country',            # If not relevant to churn prediction (adjust based on analysis)
    'address',           # Descriptive field
    'date_of_birth',     # Raw date column (if age is derived)
    'date',              # Raw date column (if derived features exist)
    'job',
    'touchpoints',
    'csat_scores',
    'churn_due_to_fraud',
    'Age',
    'complaints',
]

train_data_cleaned = train_data.copy()
train_data_cleaned = train_data_cleaned.drop(columns=[col for col in drop_columns if col in train_data_cleaned.columns], errors='ignore')

test_data_cleaned = test_data.copy()
test_data_cleaned = test_data_cleaned.drop(columns=[col for col in drop_columns if col in test_data_cleaned.columns], errors='ignore')


print("Training Data Cleaned:")
print(train_data_cleaned.info())

print("\nTesting Data Cleaned:")
print(test_data_cleaned.info())


train_data.head(1)


transaction_columns = ['atm_transfer_in', 'atm_transfer_out', 
                       'bank_transfer_in', 'bank_transfer_out',
                       'crypto_in', 'crypto_out']

# Calculate total transactions directly
train_data_cleaned['total_transactions'] = train_data_cleaned[transaction_columns].sum(axis=1)
test_data_cleaned['total_transactions'] = test_data_cleaned[transaction_columns].sum(axis=1)


print(train_data_cleaned.shape)
train_data_cleaned.head(10)


# Select consistent features for both training and testing
relevant_columns = [col for col in train_data_cleaned.columns if col not in ['is_outlier', 'customer_id', 'Id']]

# Ensure these features exist in test data
test_relevant_columns = [col for col in relevant_columns if col in test_data_cleaned.columns]

print("Updated relevant columns:", relevant_columns)


print("Columns in train_data_cleaned:", train_data_cleaned.columns)
print("Relevant columns:", relevant_columns)


from sklearn.ensemble import IsolationForest
# Exclude non-informative features like 'customer_id' or 'Id'
numeric_columns = train_data_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
relevant_columns = [col for col in numeric_columns if col not in ['is_outlier', 'customer_id', 'Id']]


# Ensure no missing columns
relevant_columns = [col for col in relevant_columns if col in train_data_cleaned.columns]

# Initialize Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)

#fit
iso_forest.fit(train_data_cleaned[relevant_columns])

# predict outliers
outlier_labels = iso_forest.predict(train_data_cleaned[relevant_columns])

# Add is_outlier column with 1 for outliers and 0 for inliers
train_data_cleaned['is_outlier'] = (outlier_labels == -1).astype(int)

# Display the counts of inliers (0) and outliers (1)
print(train_data_cleaned['is_outlier'].value_counts())



train_data_cleaned.head(1)


# Regenerate relevant_columns to exclude 'is_outlier'
relevant_columns_test = [col for col in relevant_columns if col in test_data_cleaned.columns]
# Predict outliers in the test dataset using the same Isolation Forest model
outlier_labels_test = iso_forest.predict(test_data_cleaned[relevant_columns_test])

# Add is_outlier column to test data
test_data_cleaned['is_outlier'] = (outlier_labels_test == -1).astype(int)
# Check results
print("Outlier counts in test data:")
print(test_data_cleaned['is_outlier'].value_counts())


test_data_cleaned.head(1)


# Separate outliers and non-outliers
outliers = train_data_cleaned[train_data_cleaned['is_outlier'] == 1]
non_outliers = train_data_cleaned[train_data_cleaned['is_outlier'] == 0]

# Compare statistics for key features
features_to_check = ['total_withdrawals', 'total_deposits', 'net_balance']  # Add relevant features
for feature in features_to_check:
    print(f"Feature: {feature}")
    print("Outliers:")
    print(outliers[feature].describe())
    print("\nNon-Outliers:")
    print(non_outliers[feature].describe())
    print("-" * 50)


import matplotlib.pyplot as plt
import seaborn as sns

# Example: Boxplot for 'net_balance'
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_data_cleaned, x='is_outlier', y='net_balance')
plt.title("Boxplot of Net Balance (Outliers vs Non-Outliers)")
plt.xticks([0, 1], ['Non-Outliers', 'Outliers'])
plt.show()


# Example: Scatterplot for 'total_deposits' vs 'net_balance'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data_cleaned, x='total_deposits', y='net_balance', hue='is_outlier')
plt.title("Scatterplot of Total Deposits vs Net Balance (Outliers Highlighted)")
plt.show()


# Create the churn column based on the definition
train_data_cleaned['churn'] = (
    (train_data_cleaned['last_tx_days'] >= 365) &  # No transactions for 1 year
    ((train_data_cleaned['recent_transaction_volume'] < 0.5 * train_data_cleaned['average_transaction_volume']) &  # Recent volume < 50% of average volume
     (train_data_cleaned['average_transaction_volume'] > 0))  # Ensure average volume is non-zero
).astype(int)  # Convert True/False to 1/0

# Confirm the churn column creation
print(train_data_cleaned['churn'].value_counts())

# Display a sample of the churn column
print(train_data_cleaned[['last_tx_days', 'recent_transaction_volume', 'average_transaction_volume', 'churn']].head(10))

print(train_data_cleaned['churn'].value_counts())


train_data_cleaned.head(1)


# Count of customers with no transactions for 365+ days
no_transactions_count = (train_data_cleaned['last_tx_days'] >= 365).sum()
print(f"Transactions happened 365 days ago: {no_transactions_count}")

# Count of customers with significant transaction volume drop
volume_drop_count = (train_data_cleaned['average_transaction_volume'] > 50).sum()
print(f"transactions volume decrease > 50%: {volume_drop_count}")


#maintenance to keep working output folder clean 
import os
# List all files in the output directory
output_files = os.listdir("/kaggle/working")
print("Files in the Output Folder:", output_files)


#maintenance to keep working output folder clean 
files_to_delete = ['train_data_cleaned.csv']
for file in files_to_delete:
    file_path = os.path.join("/kaggle/working", file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file}")
    else:
        print(f"File not found: {file}")


# Save the DataFrame to a CSV file

#train_data_cleaned.to_csv("train_data_cleaned.csv", index=False)
#print("Dataset saved successfully to Kaggle working directory as train_data_cleaned.csv - push on bigquery")

#https://drive.google.com/open?id=1_gmbG6-QyLOcrNXOL3ntvGC3ixXXSBC6 


# Count total values in the 'churn' column
total_churn = train_data_cleaned['churn'].count()

# Count occurrences of 0 and 1 in the 'churn' column
churn_counts = train_data_cleaned['churn'].value_counts()

# Extract specific counts
churn_0 = churn_counts.get(0, 0)  # Count of churn == 0
churn_1 = churn_counts.get(1, 0)  # Count of churn == 1

# Display results
print(f"Total churn values: {total_churn}")
print(f"Churn = 0 (not churned): {churn_0}")
print(f"Churn = 1 (churned): {churn_1}")


"""
def summarize_dataframe(df):
    # Function to calculate null percentage
    def null_percentage(series):
        return series.isnull().mean() * 100

    # Create a summary DataFrame
    summary = pd.DataFrame({
        'column': df.columns,
        'approx_unique': [df[col].nunique() for col in df.columns],
        'avg': [df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns],
        'std': [df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns],
        'q25': [df[col].quantile(0.25) if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns],
        'q50': [df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns],
        'q75': [df[col].quantile(0.75) if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns],
        'count': [len(df[col]) for col in df.columns],
        'null_percentage': [null_percentage(df[col]) for col in df.columns]
    })
    return summary

def disp_summary(df_summary):
    # Numeric columns for formatting
    numeric_cols = ['approx_unique', 'avg', 'std', 'q25', 'q50', 'q75', 'count', 'null_percentage']
    result = df_summary.copy()
    
    # Convert numeric columns to double (if applicable) and round them
    for col in numeric_cols:
        result[col] = result[col].astype('double', errors='ignore')
    result = (
        result
         .drop(columns=['count'])  # Drop the 'count' column
         .applymap(lambda x: round(x, 1) if isinstance(x, (int, float)) else x)
    )
    return result

# Example Usage
df_summary = summarize_dataframe(train_data_cleaned)
displayed_summary = disp_summary(df_summary)
print(displayed_summary)
"""


#VISUAL BOXPLOT
import seaborn as sns
import matplotlib.pyplot as plt

# Example for a single column (e.g., 'tenure')
sns.boxplot(data=train_data_cleaned, x='churn')
plt.title("Boxplot of churn")
plt.show()


#VISUAL Scatterplot
sns.scatterplot(data=train_data_cleaned, x='crypto_out', y='crypto_in')
plt.title("Scatterplot of C out vs. Crypto In")
plt.show()


#VISUAL HISTPLOT Check summary statistics of net balance
print(train_data_cleaned['net_balance'].describe())

# Visualize distribution of net balances
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(train_data_cleaned['net_balance'], kde=True, bins=50)
plt.title("Net Balance Distribution")
plt.xlabel("Net Balance")
plt.ylabel("Frequency")
plt.show()


# VISUAL CORRELATION MATRIX Create heatmap

num_col = ['interest_rate', 'atm_transfer_in', 'atm_transfer_out', 'bank_transfer_in', 'bank_transfer_out', 'crypto_in', 'crypto_out', 'bank_transfer_in_volume', 'bank_transfer_out_volume', 'crypto_in_volume', 'crypto_out_volume', 'tenure']
# Calculate correlation matrix
corr_matrix = train_data_cleaned[num_col].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


categorical_features = ['from_competitor', 'churn_due_to_fraud', 'model_predicted_fraud', 'is_outlier', 'churn', 'higher_withdrawals' ]
#removed Id and customer_id from scaler 
numerical_features = [
    'interest_rate', 'atm_transfer_in', 'atm_transfer_out', 
    'bank_transfer_in', 'bank_transfer_out', 'crypto_in', 'crypto_out',
    'bank_transfer_in_volume', 'bank_transfer_out_volume', 'crypto_in_volume', 
    'crypto_out_volume', 'tenure', 'last_tx_days', 
    'net_balance', 'total_deposits', 'total_withdrawals', 'higher_withdrawals',
    'total_transaction_volume', 'average_transaction_volume', 
    'recent_transaction_volume', 'total_transactions'
]

print("Categorical Features:", categorical_features)
print("Numerical Features:", numerical_features)




#pre-process numerical columns 
from sklearn.preprocessing import StandardScaler
# Initialize the scaler
scaler = StandardScaler()
# Scale numerical features
train_data_cleaned[numerical_features] = scaler.fit_transform(train_data_cleaned[numerical_features])
# Verify scaling
print(train_data_cleaned[numerical_features].head())


"""
from sklearn.preprocessing import RobustScaler

# Initialize RobustScaler
scaler = RobustScaler()

# Select numeric columns
excluded_columns = ['is_outlier', 'churn']  # Add columns to exclude
numeric_columns = [
    col for col in train_data_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if col not in excluded_columns
]

# Scale numeric columns in the training dataset
train_data_cleaned[numeric_columns] = scaler.fit_transform(train_data_cleaned[numeric_columns])

# Scale numeric columns in the test dataset
test_data_cleaned[numeric_columns] = scaler.transform(test_data_cleaned[numeric_columns])

# Verify scaling
print(train_data_cleaned[numeric_columns].head())
print(test_data_cleaned[numeric_columns].head())
"""




# Transform the test data using the already-fitted scaler
test_data_cleaned[numerical_features] = scaler.transform(test_data_cleaned[numerical_features])

# Verify scaling on test data
print("Test data scaled:")
print(test_data_cleaned[numerical_features].head())


train_data_cleaned.head(1)


test_data_cleaned.head(1)


#train_data_cleaned[numerical_features].fillna(0, inplace=True)
#test_data_cleaned[numerical_features].fillna(0, inplace=True)


from sklearn.model_selection import train_test_split

# Preserve Id for submission
train_ids = train_data_cleaned['Id'] if 'Id' in train_data_cleaned.columns else None
test_ids = test_data_cleaned['Id'] if 'Id' in test_data_cleaned.columns else None

# Drop Id column if it exists
if 'Id' in train_data_cleaned.columns:
    train_data_cleaned = train_data_cleaned.drop(columns=['Id'])

if 'Id' in test_data_cleaned.columns:
    test_data_cleaned = test_data_cleaned.drop(columns=['Id'])

#Ensure Customer ID is not included in splitting or training 
if 'customer_id' in train_data_cleaned.columns: 
    train_data_cleaned = train_data_cleaned.drop(columns=['customer_id'])

if 'customer_id' in test_data_cleaned.columns: 
    test_data_cleaned = test_data_cleaned.drop(columns=['customer_id'])


# Define features (X) and target (y)
X = train_data_cleaned.drop(columns=['churn'])  # Exclude target column
y = train_data_cleaned['churn']  # Target column

if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("Warning: Missing values detected!")
    
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check sizes
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


#Start Logistic regression training 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
logreg = LogisticRegression(random_state=42, max_iter=1000) #to try class_weight='balanced'
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#[[780770    279]  # True Negatives (correctly identified non-churn) and False Positives
#[  1183   2980]] # False Negatives and True Positives (correctly identified churn)
"""Accuracy: 1.00
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    781049
           1       0.92      0.72      0.80      4163

    accuracy                           1.00    785212
   macro avg       0.96      0.86      0.90    785212
weighted avg       1.00      1.00      1.00    785212

Confusion Matrix:
 [[780777    272]
 [  1182   2981]]
 """
"""Accuracy: 1.00
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    781049
           1       0.92      0.72      0.80      4163

    accuracy                           1.00    785212
   macro avg       0.96      0.86      0.90    785212
weighted avg       1.00      1.00      1.00    785212

Confusion Matrix:
 [[780777    272]
 [  1182   2981]]
'Accuracy: 1.00\nClassification Report:\n               precision    recall  f1-score   support\n\n           0       1.00      1.00      1.00    781049\n           1       0.92      0.72      0.80      4163\n\n    accuracy                           1.00    785212\n   macro avg       0.96      0.86      0.90    785212\nweighted avg       1.00      1.00      1.00    785212\n\nConfusion Matrix:\n [[780777    272]\n [  1182   2981]]\n '"""


feature_importance = pd.DataFrame({
    'Feature': X_train.columns, 
    'Coefficient': logreg.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(feature_importance)


#Advanced Models:
#XGBoost: For handling tabular data with strong performance.
#CatBoost: Handles categorical data efficiently and is explainable.
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from catboost import CatBoostClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



#train xboost 
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("XGBoost results")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

"""XGBoost results
Accuracy: 1.00
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    781049
           1       0.99      0.99      0.99      4163

    accuracy                           1.00    785212
   macro avg       0.99      0.99      0.99    785212
weighted avg       1.00      1.00      1.00    785212

Confusion Matrix:
 [[780989     60]
 [    58   4105]]"""


#train lightgbm
# Initialize LightGBM classifier
lgbm_model = LGBMClassifier(random_state=42)

# Train the model
lgbm_model.fit(X_train, y_train)

# Predict and evaluate
lgbm_pred = lgbm_model.predict(X_test)
print("LightGBM Results")
print(f"Accuracy: {accuracy_score(y_test, lgbm_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, lgbm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lgbm_pred))

"""[LightGBM] [Info] Number of positive: 16553, number of negative: 3124293
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.211220 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 4037
[LightGBM] [Info] Number of data points in the train set: 3140846, number of used features: 24
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.005270 -> initscore=-5.240396
[LightGBM] [Info] Start training from score -5.240396
LightGBM Results
Accuracy: 1.00
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    781049
           1       0.94      0.96      0.95      4163

    accuracy                           1.00    785212
   macro avg       0.97      0.98      0.98    785212
weighted avg       1.00      1.00      1.00    785212

Confusion Matrix:
 [[780801    248]
 [   165   3998]]"""


#train Catboost 
catboost_model = CatBoostClassifier(random_state=42, verbose=0)

# Train the model
catboost_model.fit(X_train, y_train)

# Predict and evaluate
catboost_pred = catboost_model.predict(X_test)
print("CatBoost Results")
print(f"Accuracy: {accuracy_score(y_test, catboost_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, catboost_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, catboost_pred))

"""CatBoost Results
Accuracy: 1.00
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    781049
           1       0.99      0.99      0.99      4163

    accuracy                           1.00    785212
   macro avg       0.99      0.99      0.99    785212
weighted avg       1.00      1.00      1.00    785212

Confusion Matrix:
 [[780998     51]
 [    54   4109]]"""


# Compare results including Logistic Regression, XGBoost, LightGBM, and CatBoost
results = {
    "Model": ["Logistic Regression", "XGBoost", "LightGBM", "CatBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),          # Logistic Regression
        accuracy_score(y_test, xgb_pred),       # XGBoost
        accuracy_score(y_test, lgbm_pred),      # LightGBM
        accuracy_score(y_test, catboost_pred)   # CatBoost
    ],
    "Precision (Class 1)": [
        classification_report(y_test, y_pred, output_dict=True)['1']['precision'],          # Logistic Regression
        classification_report(y_test, xgb_pred, output_dict=True)['1']['precision'],       # XGBoost
        classification_report(y_test, lgbm_pred, output_dict=True)['1']['precision'],      # LightGBM
        classification_report(y_test, catboost_pred, output_dict=True)['1']['precision']   # CatBoost
    ],
    "Recall (Class 1)": [
        classification_report(y_test, y_pred, output_dict=True)['1']['recall'],          # Logistic Regression
        classification_report(y_test, xgb_pred, output_dict=True)['1']['recall'],       # XGBoost
        classification_report(y_test, lgbm_pred, output_dict=True)['1']['recall'],      # LightGBM
        classification_report(y_test, catboost_pred, output_dict=True)['1']['recall']   # CatBoost
    ],
    "F1-Score (Class 1)": [
        classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'],          # Logistic Regression
        classification_report(y_test, xgb_pred, output_dict=True)['1']['f1-score'],       # XGBoost
        classification_report(y_test, lgbm_pred, output_dict=True)['1']['f1-score'],      # LightGBM
        classification_report(y_test, catboost_pred, output_dict=True)['1']['f1-score']   # CatBoost
    ],
    "False Positives": [
        confusion_matrix(y_test, y_pred)[0, 1],          # Logistic Regression
        confusion_matrix(y_test, xgb_pred)[0, 1],       # XGBoost
        confusion_matrix(y_test, lgbm_pred)[0, 1],      # LightGBM
        confusion_matrix(y_test, catboost_pred)[0, 1]   # CatBoost
    ],
    "False Negatives": [
        confusion_matrix(y_test, y_pred)[1, 0],          # Logistic Regression
        confusion_matrix(y_test, xgb_pred)[1, 0],       # XGBoost
        confusion_matrix(y_test, lgbm_pred)[1, 0],      # LightGBM
        confusion_matrix(y_test, catboost_pred)[1, 0]   # CatBoost
    ]
}

# Create a DataFrame for results
import pandas as pd
results_df = pd.DataFrame(results)

print(results_df)

# Optionally, save the comparison table to a CSV for further analysis
#results_df.to_csv("model_comparison_results.csv", index=False)



"""
#finetuning catboost - ill finetune later after the first submission 


#Previous train Catboost 
catboost_model = CatBoostClassifier(random_state=42, verbose=0)

# Train the model
catboost_model.fit(X_train, y_train)

# Predict and evaluate
catboost_pred = catboost_model.predict(X_test)
print("CatBoost Results")
print(f"Accuracy: {accuracy_score(y_test, catboost_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, catboost_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, catboost_pred))
"""

"""
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=50
)
catboost_model.fit(X_train, y_train)

catboost_pred = catboost_model.predict(X_test)
print("CatBoost Results")
print(f"Accuracy: {accuracy_score(y_test, catboost_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, catboost_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, catboost_pred))
"""


#feature mismatch error 
# Features used during training
train_features = list(X.columns)

# Features in the test dataset
test_features = list(test_data_cleaned[numeric_columns].columns)

# Find missing and extra features
missing_in_test = [col for col in train_features if col not in test_features]
extra_in_test = [col for col in test_features if col not in train_features]

print(f"Features missing in test data: {missing_in_test}")
print(f"Extra features in test data: {extra_in_test}")


test_data_cleaned.head(1)



#Basic hyperparameter tuning 
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
}

# Initialize the XGBoost model
xgb_model = XGBClassifier(
    tree_method='hist',
    random_state=42, 
    eval_metric='logloss', 
    use_label_encoder=False
)

# Perform grid search
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    cv=3, 
    scoring='f1', 
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters
final_xgb_model = grid_search.best_estimator_


"""
#strategies to speed up hyperparameter tuning 
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Define the parameter grid
param_distributions = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
}

# Initialize the XGBoost model
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

# Perform randomized search
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions, 
                                   n_iter=20, scoring='f1', cv=5, verbose=1, random_state=42)

random_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", random_search.best_params_)

# Train the model with the best parameters
final_xgb_model = random_search.best_estimator_
"""


final_xgb_model = XGBClassifier(
    tree_method='hist',
    random_state=42, 
    eval_metric='logloss', 
    use_label_encoder=False
)

# Train the final model on the entire Training set 
final_xgb_model.fit(X_train, y_train)

# Evaluate on the validation set
y_pred = final_xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Add missing features to the test dataset with default values
for feature in ['atm_transfer_in', 'atm_transfer_out', 'bank_transfer_in', 'bank_transfer_out', 
                'crypto_in', 'crypto_out', 'tenure', 'from_competitor', 'model_predicted_fraud', 
                'last_tx_days', 'higher_withdrawals', 'total_transaction_volume', 'is_outlier']:
    if feature not in test_data_cleaned.columns:
        test_data_cleaned[feature] = 0  # Use a default value

test_data_cleaned = test_data_cleaned[X.columns]


# Predict churn for the test data
test_predictions = final_xgb_model.predict(test_data_cleaned)

# Optionally, get probabilities
test_probabilities = final_xgb_model.predict_proba(test_data_cleaned)[:, 1]

# Attach predictions to the test dataset
test_data_cleaned['churn'] = test_predictions


print("First 10 predictions:", test_predictions[:10])
print("First 10 probabilities:", test_probabilities[:10])


# Create the submission file
submission = pd.DataFrame({
    'Id': test_ids,  # Ensure you saved the Id column earlier
    'churn': test_probabilities  # Use probabilities if required, otherwise use `test_predictions`
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")


#Evaluate feature importance 
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Plot feature importance
plot_importance(final_xgb_model, max_num_features=10)
plt.show()


"""
#Deploy the model 
import joblib
joblib.dump(final_xgb_model, 'xgboost_model.pkl')
#load the model for predictions later
model = joblib.load('xgboost_model.pkl')
new_data = test_data_cleaned[X.columns]  # Align columns with training data
predictions = model.predict(new_data)

new_data['churn'] = predictions
new_data[['Id', 'churn']].to_csv('final_predictions.csv', index=False)
print("Predictions saved to final_predictions.csv")
"""


#selecting XGboost

from xgboost import XGBClassifier  # Import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

final_xgb_model = XGBClassifier(
    #objective='binary:logistic', 
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1
)

final_xgb_model.fit(X,y)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Step 1: Model loaded successfully.")
print("Step 2: Test dataset shape:", test_data_cleaned.shape)

print("Step 3: Predicting churn...")
# Predict churn for the test dataset
final_predictions = final_xgb_model.predict(test_data_cleaned)
print("Step 4: Predictions completed. First 10 predictions:", final_predictions[:10])

# Case 1: If true labels exist in test_data_cleaned, evaluate the model
if 'churn' in test_data_cleaned.columns:
    print("Step 5: Evaluating model...")
    print(f"Accuracy: {accuracy_score(test_data_cleaned['churn'], final_predictions):.2f}")
    print("Classification Report:\n", classification_report(test_data_cleaned['churn'], final_predictions))
    print("Confusion Matrix:\n", confusion_matrix(test_data_cleaned['churn'], final_predictions))
else:
    # Case 2: No churn column, save predictions for submission
    print("Step 5: 'churn' column not found in test_data_cleaned.")
    print("Saving predictions for submission...")
    submission = pd.DataFrame({
        'customer_id': test_customer_ids,  # Use the preserved customer IDs
        'churn': final_predictions  # Predicted churn
    })

    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")


#Understand which features contributed the most 

import matplotlib.pyplot as plt
from xgboost import plot_importance

# Plot feature importance
plt.figure(figsize=(10, 8))
plot_importance(final_xgb_model, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()



#this is for cross validation - Leave this for after the first submission 
from sklearn.model_selection import cross_val_score
catboost_cv_scores = cross_val_score(catboost_model, X, y, cv=5, scoring='f1')
print(f"CatBoost Cross-Validation F1-Score: {catboost_cv_scores.mean():.4f}")

