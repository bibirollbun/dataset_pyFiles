# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer


application_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
application_data.head()


application_data.info()


application_data.shape


application_data.columns.tolist()


# missing values
missing_values = application_data.isnull().sum().sort_values(ascending=False)
missing_values


# percentage of missing values
missing_percentage = (missing_values / len(application_data)) * 100
missing_percentage


missing_data =pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
print(missing_data[missing_data['Missing Values'] > 0])


columns_to_keep=missing_percentage[missing_percentage<=58].index
application_data=application_data[columns_to_keep]
application_data.shape


columns_to_impute_20 = missing_percentage[missing_percentage <= 20].index
for column in columns_to_impute_20:
    if application_data[column].dtype in ['int64', 'float64']:  
        median_value = application_data[column].median()
        application_data.loc[:, column] = application_data[column].fillna(median_value) 
    else:  
        mode_value = application_data[column].mode()[0] 
        application_data.loc[:, column] = application_data[column].fillna(mode_value)  
print(application_data[columns_to_impute_20].isnull().sum())



from sklearn.impute import IterativeImputer, SimpleImputer

# تحديد الأعمدة التي تحتوي على قيم مفقودة بين 20% و 58%
columns_to_impute_20_58 = missing_percentage[(missing_percentage > 20) & (missing_percentage <= 58)].index

# تحديد الأعمدة الرقمية
numerical_cols_20_58 = application_data[columns_to_impute_20_58].select_dtypes(include=['int64', 'float64']).columns

# تحديد الأعمدة الفئوية
categorical_cols_20_58 = application_data[columns_to_impute_20_58].select_dtypes(include=['object', 'category']).columns

# "Imputing numerical values using Iterative Imputer"
# زيادة عدد التكرارات إلى 50 أو أكثر

iterative_imputer = IterativeImputer(max_iter=100, random_state=0, n_nearest_features=10, imputation_order='ascending')
application_data.loc[:, numerical_cols_20_58] = iterative_imputer.fit_transform(application_data[numerical_cols_20_58])

# "Imputing categorical values using simple Imputer"
mode_imputer = SimpleImputer(strategy="most_frequent")
application_data.loc[:, categorical_cols_20_58] = mode_imputer.fit_transform(application_data[categorical_cols_20_58])

# التحقق من القيم المفقودة بعد المعالجة
print(application_data[columns_to_impute_20_58].isnull().sum())



print("Total missing values after imputing:", application_data.isnull().sum().sum())
application_data.shape


# Step 1: Drop the TARGET column
application_data = application_data.drop(columns=['TARGET'])

# Step 2: Function to check and drop columns with repeated values exceeding the threshold
def check_and_drop_repeated_columns(data, threshold=0.8):
    repeated_columns = []  # List to store columns to be dropped
    for column in data.columns:
        # Calculate the percentage of the most frequent value
        most_common_value_count = data[column].value_counts(normalize=True).max()
        if most_common_value_count > threshold:
            repeated_columns.append(column)  # Add column to drop list
    # Drop the repeated columns
    data = data.drop(columns=repeated_columns)
    return data

# Step 3: Apply the function on application_data
application_data = check_and_drop_repeated_columns(application_data)

# Step 4: Final shape of application_data
print(f"Shape of application_data: {application_data.shape}")



# Reload the original data to extract the TARGET column
original_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

# Add the TARGET column back to train_data
application_data['TARGET'] = original_data['TARGET']

# Verify the updated shape
print(f"Shape of train_data after adding TARGET: {application_data.shape}")



# عرض الأعمدة المتبقية
print(application_data.columns.tolist())  # يعرض قائمة الأعمدة المتبقية
print('-------------------------------------------')
print('shape = ',application_data.shape[1])  # يعرض عدد الأعمدة المتبقية



numeric_columns = application_data.select_dtypes(include=['float64', 'int64']).columns
for column in numeric_columns:
    Q1 = application_data[column].quantile(0.25)
    Q3 = application_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers_condition = (application_data[column] < lower_bound) | (application_data[column] > upper_bound)
    num_outliers = outliers_condition.sum()
    
    if num_outliers > 0:
        print(f"Column '{column}' has {num_outliers} outliers.")



# Create a new column AGE_YEARS by converting DAYS_BIRTH from negative days to positive age in years
application_data['AGE_YEARS'] = abs(application_data['DAYS_BIRTH']) / 365

# Delete the DAYS_BIRTH column
application_data = application_data.drop(columns=['DAYS_BIRTH'])


# Plot the distribution of AGE_YEARS
plt.figure(figsize=(10, 6))
sns.histplot(application_data['AGE_YEARS'], bins=30, kde=True, color='blue')
plt.title("Distribution of Age (Years)")
plt.xlabel("Age (Years)")
plt.ylabel("Frequency")
plt.show()


# Convert days to years
application_data['YEARS_EMPLOYED'] = abs(application_data['DAYS_EMPLOYED']) / 365.25

# Identify outliers using IQR
Q1 = application_data['YEARS_EMPLOYED'].quantile(0.25)
Q3 = application_data['YEARS_EMPLOYED'].quantile(0.75)
IQR = Q3 - Q1

# Calculate bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# ====== Before Handling Outliers ======
# Plot before handling outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=application_data['YEARS_EMPLOYED'], color='orange')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.title("Boxplot of YEARS_EMPLOYED (Before Handling Outliers)")
plt.xlabel("Employment (Years)")
plt.legend()
plt.show()

# Identify outliers
outliers_condition = (application_data['YEARS_EMPLOYED'] < lower_bound) | (application_data['YEARS_EMPLOYED'] > upper_bound)
application_data.loc[outliers_condition, 'YEARS_EMPLOYED'] = np.nan

# Handle missing values
median_years_employed = application_data['YEARS_EMPLOYED'].median()
application_data['YEARS_EMPLOYED'].fillna(median_years_employed, inplace=True)

# Check remaining outliers
remaining_outliers_condition = (application_data['YEARS_EMPLOYED'] < lower_bound) | (application_data['YEARS_EMPLOYED'] > upper_bound)
num_remaining_outliers = remaining_outliers_condition.sum()

print(f"Number of outliers after handling: {num_remaining_outliers}")

# ====== After Handling Outliers ======
# Plot after handling outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=application_data['YEARS_EMPLOYED'], color='green')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.title("Boxplot of YEARS_EMPLOYED (After Handling Outliers)")
plt.xlabel("Employment (Years)")
plt.legend()
plt.show()

# Plot final distribution after handling outliers
plt.figure(figsize=(10, 6))
sns.histplot(application_data['YEARS_EMPLOYED'], bins=30, kde=True, color='green')
plt.title("Histogram of YEARS_EMPLOYED (After Handling Outliers and Filling NaN)")
plt.xlabel("Employment (Years)")
plt.ylabel("Frequency")
plt.show()



# Delete the DAYS_EMPLOYED column
application_data = application_data.drop(columns=['DAYS_EMPLOYED'])



# ====== 1. Convert days to years ======
# Create a new column YEARS_REGISTRATION by converting DAYS_REGISTRATION to positive years
application_data['YEARS_REGISTRATION'] = abs(application_data['DAYS_REGISTRATION']) / 365.25

# ====== 2. Identify outliers using IQR ======
# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = application_data['YEARS_REGISTRATION'].quantile(0.25)
Q3 = application_data['YEARS_REGISTRATION'].quantile(0.75)
IQR = Q3 - Q1  # Interquartile range

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# ====== 3. Visualize outliers before handling ======
plt.figure(figsize=(8, 6))
sns.boxplot(x=application_data['YEARS_REGISTRATION'], color='orange')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.title("Boxplot of YEARS_REGISTRATION (Before Handling Outliers)")
plt.xlabel("Registration (Years)")
plt.legend()
plt.show()

# ====== 4. Identify outliers ======
# Create a condition to identify outlier values
outliers_condition = (application_data['YEARS_REGISTRATION'] < lower_bound) | (application_data['YEARS_REGISTRATION'] > upper_bound)

# Print the number of outliers detected
num_outliers = outliers_condition.sum()
print(f"Number of outliers before handling: {num_outliers}")

# ====== 5. Replace outliers with NaN ======
# Replace outliers with NaN
application_data.loc[outliers_condition, 'YEARS_REGISTRATION'] = np.nan

# Print the number of NaN values after replacing outliers
print(f"Number of NaN values after replacing outliers: {application_data['YEARS_REGISTRATION'].isnull().sum()}")

# ====== 6. Handle missing values ======
# Fill missing values (NaN) with the median
median_years_registration = application_data['YEARS_REGISTRATION'].median()
application_data['YEARS_REGISTRATION'].fillna(median_years_registration, inplace=True)

# ====== 7. Verify outliers after handling ======
# Recheck the presence of outliers after handling
remaining_outliers_condition = (application_data['YEARS_REGISTRATION'] < lower_bound) | (application_data['YEARS_REGISTRATION'] > upper_bound)
num_remaining_outliers = remaining_outliers_condition.sum()

print(f"Number of outliers after handling: {num_remaining_outliers}")

# ====== 8. Plot the final distribution ======
# Plot the distribution after handling outliers and missing values
plt.figure(figsize=(10, 6))
sns.histplot(application_data['YEARS_REGISTRATION'], bins=30, kde=True, color='purple')
plt.title("Histogram of YEARS_REGISTRATION (After Handling Outliers and Filling NaN)")
plt.xlabel("Registration (Years)")
plt.ylabel("Frequency")
plt.show()


# Delete the DAYS_REGISTRATION column
application_data.drop(columns=['DAYS_REGISTRATION'], inplace=True)


# ====== 1. Convert days to years ======
# Create a new column YEARS_ID_PUBLISH by converting DAYS_ID_PUBLISH to positive years
application_data['YEARS_ID_PUBLISH'] = abs(application_data['DAYS_ID_PUBLISH']) / 365.25

# ====== 2. Identify outliers using IQR ======
# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = application_data['YEARS_ID_PUBLISH'].quantile(0.25)
Q3 = application_data['YEARS_ID_PUBLISH'].quantile(0.75)
IQR = Q3 - Q1  # Interquartile range

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# ====== 3. Visualize outliers before handling ======
plt.figure(figsize=(8, 6))
sns.boxplot(x=application_data['YEARS_ID_PUBLISH'], color='orange')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.title("Boxplot of YEARS_ID_PUBLISH (Before Handling Outliers)")
plt.xlabel("ID Publish (Years)")
plt.legend()
plt.show()

application_data.drop(columns=['DAYS_ID_PUBLISH'], inplace=True)



# رسم boxplot لعمود EXT_SOURCE_1
plt.figure(figsize=(8, 6))
sns.boxplot(x=application_data['EXT_SOURCE_1'], color='purple')
plt.title("Boxplot of EXT_SOURCE_1")
plt.xlabel("EXT_SOURCE_1")
plt.show()



application_data.shape


bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
bureau_balance.head(50)


bureau_balance.info()


bureau_balance['STATUS'].unique()


grouped_status = bureau_balance.groupby(['SK_ID_BUREAU', 'STATUS']).size()
print(grouped_status)



pivoted_status = grouped_status.unstack(fill_value=0)
print(pivoted_status)


pivoted_status.columns = [f'STATUS_{col}' for col in pivoted_status.columns]
pivoted_status


# Aggregating the minimum months balance for each SK_ID_BUREAU
aggregated_months_balance = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].agg(['count']).reset_index()

# Renaming the column for better understanding
aggregated_months_balance.rename(columns={'count' :'TOTAL_MONTHS'}, inplace=True)

# Display the resulting DataFrame
aggregated_months_balance


# إضافة عمود جديد STATUS_DELAYED يحتوي على جمع القيم من الأعمدة STATUS_1 إلى STATUS_5
pivoted_status['STATUS_DELAYED'] = pivoted_status[['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5']].sum(axis=1)

# عرض النتائج
pivoted_status.head(20)



# إضافة العمود STATUS_DELAYED
pivoted_status['STATUS_DELAYED'] = pivoted_status[['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5']].sum(axis=1)

# دمج DataFrame pivoted_status مع aggregated_months_balance بناءً على SK_ID_BUREAU
pivoted_status = pivoted_status.reset_index().merge(aggregated_months_balance, on='SK_ID_BUREAU', how='left')

# عرض النتيجة النهائية
pivoted_status.head(20)



# دمج DataFrame pivoted_status مع aggregated_months_balance بناءً على SK_ID_BUREAU
pivoted_status = pivoted_status.reset_index().merge(aggregated_months_balance, on='SK_ID_BUREAU', how='left', suffixes=('', '_y'))

# إذا كان هناك عمود TOTAL_MONTHS_y، يمكنك حذف النسخة الزائدة
pivoted_status.drop(columns=['TOTAL_MONTHS_y'], inplace=True)

# إضافة عمود STATUS_DELAYED
pivoted_status['STATUS_DELAYED'] = pivoted_status[['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5']].sum(axis=1)

# إضافة عمود CREDIT_STATUS بناءً على المقارنة بين STATUS_C, STATUS_X, STATUS_DELAYED
pivoted_status['CREDIT_STATUS'] = pivoted_status.apply(
    lambda x: 'Completed' if x['STATUS_C'] > x['STATUS_DELAYED'] and x['STATUS_C'] > x['STATUS_X'] else
              'Delayed' if x['STATUS_DELAYED'] > x['STATUS_C'] and x['STATUS_DELAYED'] > x['STATUS_X'] else
              'X' if x['STATUS_X'] > x['STATUS_C'] and x['STATUS_X'] > x['STATUS_DELAYED'] else
              'No Data', axis=1
)

# عرض النتيجة النهائية
pivoted_status.head(30)



aggregated_bureau = pivoted_status
aggregated_bureau


aggregated_bureau.shape


aggregated_bureau


bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau.head(20)


bureau.shape


bureau['CREDIT_ACTIVE'].unique()


bureau['CREDIT_CURRENCY'].unique()


len(bureau['CREDIT_TYPE'].unique())


bureau.info()


# Perform a join between the aggregated bureau balance and bureau tables
left_tables = bureau.merge(
    aggregated_bureau,  # Aggregated bureau balance table
    on='SK_ID_BUREAU',               # Join on the SK_ID_BUREAU column
    how='left'                       # Left join to keep all rows from bureau
)
left_tables.fillna(0,inplace=True)
# Display the resulting merged table



left_tables


# List of categorical columns
categorical_columns = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE', 'CREDIT_STATUS']

# Initialize a dictionary to store results for each column
individual_aggregations = {}

# Iterate over categorical columns and apply aggregation
for column in categorical_columns:
    print(f"Processing column: {column}")
    
    # Group by SK_ID_CURR and the current categorical column, count occurrences
    grouped_categorical = left_tables.groupby(['SK_ID_CURR', column]).size().unstack(fill_value=0)
    
    # Rename columns for clarity
    grouped_categorical.columns = [f'{column}_{col}' for col in grouped_categorical.columns]
    
    # Reset index
    grouped_categorical.reset_index(inplace=True)
    
    # Save the result for this column
    individual_aggregations[column] = grouped_categorical
    
    # Display a preview of the processed column
    print(grouped_categorical.head())


 # Select numerical columns from bureau
 numerical_columns = left_tables.select_dtypes(include=['int64', 'float64']).columns.tolist()

 # Display numerical columns
 print("Numerical Columns in Final_data:")
 print(numerical_columns)



 aggregated_Days_Credit=left_tables.groupby('SK_ID_CURR')['DAYS_CREDIT'].agg(['max']).reset_index()
 aggregated_Days_Credit.rename(columns={'max' :'DAYS_CREDIT_MAX'}, inplace=True)
 aggregated_Days_Credit



 aggregated_CREDIT_DAY_OVERDUE=left_tables.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].agg(['max']).reset_index()
 aggregated_CREDIT_DAY_OVERDUE.rename(columns={'max' :'CREDIT_DAY_OVERDUE_MAX'}, inplace=True)
 aggregated_CREDIT_DAY_OVERDUE




aggregated_DAYS_CREDIT_ENDDATE=left_tables.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].agg(['min']).reset_index()
aggregated_DAYS_CREDIT_ENDDATE.rename(columns={'min' :'DAYS_CREDIT_ENDDATE_MIN'}, inplace=True)
aggregated_DAYS_CREDIT_ENDDATE



aggregated_DAYS_ENDDATE_FACT=left_tables.groupby('SK_ID_CURR')['DAYS_ENDDATE_FACT'].agg(['max']).reset_index()
aggregated_DAYS_ENDDATE_FACT.rename(columns={'max' :'DAYS_ENDDATE_FACT_MAX'}, inplace=True)
aggregated_DAYS_ENDDATE_FACT


aggregated_AMT_CREDIT_MAX_OVERDUE=left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].agg(['max']).reset_index()
aggregated_AMT_CREDIT_MAX_OVERDUE.rename(columns={'max' :'AMT_CREDIT_MAX_OVERDUE_MAX'}, inplace=True)
# # Fill NaN values in the aggregated DataFrame with 0
# aggregated_AMT_CREDIT_MAX_OVERDUE['AMT_CREDIT_MAX_OVERDUE_MAX'].fillna(0, inplace=True)
aggregated_AMT_CREDIT_MAX_OVERDUE


 aggregated_CNT_CREDIT_PROLONG=left_tables.groupby('SK_ID_CURR')['CNT_CREDIT_PROLONG'].agg(['max']).reset_index()
 aggregated_CNT_CREDIT_PROLONG.rename(columns={'max':'CNT_CREDIT_PROLONG_MAX'}, inplace=True)
 aggregated_CNT_CREDIT_PROLONG


aggregated_AMT_CREDIT_SUM=left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].agg(['sum']).reset_index()
aggregated_AMT_CREDIT_SUM.rename(columns={'sum' :'AMT_CREDIT_SUM_SUM','mean':'AMT_CREDIT_SUM_MEAN'}, inplace=True)
aggregated_AMT_CREDIT_SUM


aggregated_AMT_CREDIT_SUM_DEBT=left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].agg(['sum']).reset_index()
aggregated_AMT_CREDIT_SUM_DEBT.rename(columns={'sum' :'AMT_CREDIT_SUM_DEBT_SUM'}, inplace=True)
aggregated_AMT_CREDIT_SUM_DEBT


 aggregated_AMT_CREDIT_SUM_LIMIT=left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].agg(['mean']).reset_index()
 aggregated_AMT_CREDIT_SUM_LIMIT.rename(columns={'mean' :'AMT_CREDIT_SUM_LIMIT_MEAN'}, inplace=True)
 aggregated_AMT_CREDIT_SUM_LIMIT['AMT_CREDIT_SUM_LIMIT_MEAN'].fillna(0,inplace=True)
 aggregated_AMT_CREDIT_SUM_LIMIT


 aggregated_AMT_CREDIT_SUM_OVERDUE=left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].agg(['max']).reset_index()
 aggregated_AMT_CREDIT_SUM_OVERDUE.rename(columns={'max' :'AMT_CREDIT_SUM_OVERDUE_MAX'}, inplace=True)
 aggregated_AMT_CREDIT_SUM_OVERDUE


 aggregated_DAYS_CREDIT_UPDATE=left_tables.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].agg(['min']).reset_index()
 aggregated_DAYS_CREDIT_UPDATE.rename(columns={'min' :'DAYS_CREDIT_UPDATE_MIN'}, inplace=True)
 aggregated_DAYS_CREDIT_UPDATE


aggregated_AMT_ANNUITY=left_tables.groupby('SK_ID_CURR')['AMT_ANNUITY'].agg(['mean']).reset_index()
aggregated_AMT_ANNUITY.rename(columns={'mean' :'AMT_ANNUITY_MEAN'}, inplace=True)
aggregated_AMT_ANNUITY['AMT_ANNUITY_MEAN'].fillna(0,inplace=True)
aggregated_AMT_ANNUITY


aggregated_status_0=left_tables.groupby('SK_ID_CURR')['STATUS_0'].agg('sum').reset_index()
aggregated_status_0


aggregated_status_1=left_tables.groupby('SK_ID_CURR')['STATUS_1'].agg('sum').reset_index()
aggregated_status_1


aggregated_status_2=left_tables.groupby('SK_ID_CURR')['STATUS_2'].agg('count').reset_index()
aggregated_status_2


aggregated_status_3=left_tables.groupby('SK_ID_CURR')['STATUS_3'].agg('count').reset_index()
aggregated_status_3


aggregated_status_4=left_tables.groupby('SK_ID_CURR')['STATUS_4'].agg('count').reset_index()
aggregated_status_4


aggregated_status_5=left_tables.groupby('SK_ID_CURR')['STATUS_5'].agg('count').reset_index()
aggregated_status_5


aggregated_status_X=left_tables.groupby('SK_ID_CURR')['STATUS_X'].agg('sum').reset_index()
aggregated_status_X


aggregated_status_C=left_tables.groupby('SK_ID_CURR')['STATUS_C'].agg('sum').reset_index()
aggregated_status_C


aggregated_TOTAL_MONTHS=left_tables.groupby('SK_ID_CURR')['TOTAL_MONTHS'].agg('count').reset_index()
aggregated_TOTAL_MONTHS


aggregated_STATUS_DELAYED=left_tables.groupby('SK_ID_CURR')['STATUS_DELAYED'].agg('sum').reset_index()
aggregated_STATUS_DELAYED


# List of all aggregated DataFrames
aggregated_dataframes = [
    aggregated_Days_Credit,
    aggregated_CREDIT_DAY_OVERDUE,
    aggregated_DAYS_CREDIT_ENDDATE,
    aggregated_DAYS_ENDDATE_FACT,
    aggregated_AMT_CREDIT_MAX_OVERDUE,
    aggregated_CNT_CREDIT_PROLONG,
    aggregated_AMT_CREDIT_SUM,
    aggregated_AMT_CREDIT_SUM_DEBT,
    aggregated_AMT_CREDIT_SUM_LIMIT,
    aggregated_AMT_CREDIT_SUM_OVERDUE,
    aggregated_DAYS_CREDIT_UPDATE,
    aggregated_AMT_ANNUITY,
    aggregated_status_0,
    aggregated_status_1,
    aggregated_status_2,
    aggregated_status_3,
    aggregated_status_4,
    aggregated_status_5,
    aggregated_status_C,
    aggregated_status_X,aggregated_STATUS_DELAYED,
    aggregated_TOTAL_MONTHS
]

# Use pd.concat to merge all DataFrames based on the SK_ID_CURR column
aggregated_final_data = pd.concat(aggregated_dataframes, axis=1)

# Drop duplicate SK_ID_CURR columns if any
aggregated_final_data = aggregated_final_data.loc[:, ~aggregated_final_data.columns.duplicated()]

# Display the final aggregated DataFrame
aggregated_left_tables =aggregated_final_data


aggregated_left_tables


previous_application=pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
previous_application


previous_application.columns


# Drop the specified columns
columns_to_drop = [
    "SK_ID_PREV",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "FLAG_LAST_APPL_PER_CONTRACT",
    "NFLAG_LAST_APPL_IN_DAY",
    "NAME_CASH_LOAN_PURPOSE",
    "NAME_PAYMENT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_CLIENT_TYPE",
    "NAME_PRODUCT_TYPE",
    "CHANNEL_TYPE",
    "SELLERPLACE_AREA",
    "NAME_SELLER_INDUSTRY",
    "PRODUCT_COMBINATION",
    "NFLAG_INSURED_ON_APPROVAL"
]

# Drop the columns from df
previous_application = previous_application.drop(columns=[col for col in columns_to_drop if col in previous_application.columns])

# Display the filtered DataFrame
previous_application


aggregations = {
    'AMT_APPLICATION': 'mean',
    'AMT_ANNUITY': 'mean',
    'AMT_CREDIT': 'mean',
    'AMT_DOWN_PAYMENT': 'max',
    'AMT_GOODS_PRICE': 'mean',
    'RATE_DOWN_PAYMENT': 'mean',
    'RATE_INTEREST_PRIMARY': 'mean',
    'DAYS_DECISION': 'min',
    'CNT_PAYMENT': 'max',
    'DAYS_FIRST_DRAWING': 'min',
    'DAYS_FIRST_DUE': 'min',
    'DAYS_LAST_DUE': 'max',
    'DAYS_TERMINATION': 'max',
    'NAME_CONTRACT_TYPE': 'count',
    'NAME_CONTRACT_STATUS': 'count',
    'NAME_GOODS_CATEGORY': 'count',
    'NAME_PORTFOLIO': 'count',
    'NAME_YIELD_GROUP': 'count',
}

# Group by SK_ID_CURR
aggregated_previous_application = previous_application.groupby('SK_ID_CURR').agg(aggregations)

# Rename columns to reflect aggregation type
aggregated_previous_application.columns = [
    f"{col}_{aggregations[col].upper()}" for col in aggregated_previous_application.columns
]


# Reset index
aggregated_previous_application.reset_index(inplace=True)

# Display the aggregated data
aggregated_previous_application


POS_CACH_balance=pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
POS_CACH_balance


POS_CACH_balance.columns


# تعيين التجميعات المطلوبة
aggregations = {
    'MONTHS_BALANCE': 'count',  # Count months of balance
    'CNT_INSTALMENT': ['mean', 'max'],  # Average and maximum installments
    'CNT_INSTALMENT_FUTURE': ['mean', 'sum'],  # Average and total remaining installments
    'SK_DPD': ['mean', 'max'],  # Average and maximum days past due
    'SK_DPD_DEF': 'mean'  # Average days past due with tolerance
}

# إجراء التجميع
pos_agg = POS_CACH_balance.groupby('SK_ID_CURR').agg(aggregations)

# تسطيح أعمدة الـ MultiIndex (إذا كانت موجودة) بعد إجراء التجميعات
pos_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pos_agg.columns]

# إضافة أعمدة حالات العقود
contract_statuses = ['Active', 'Completed', 'Signed', 'Demand']

for status in contract_statuses:
    pos_agg[f'NAME_CONTRACT_STATUS_{status}'] = (
        (POS_CACH_balance['NAME_CONTRACT_STATUS'] == status)
        .groupby(POS_CACH_balance['SK_ID_CURR']).transform('sum')
    )

# إعادة تعيين الفهرس لتسهيل التعامل مع البيانات
pos_agg = pos_agg.reset_index()

aggregated_POS_CACH_balance = pos_agg



pos_agg


installments_payments=pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
installments_payments


installments_payments.columns


installments_agg = installments_payments.groupby('SK_ID_CURR').agg({
    'NUM_INSTALMENT_VERSION': ['count'],
    'NUM_INSTALMENT_NUMBER': ['max'],
    'DAYS_INSTALMENT': ['min'],
    'DAYS_ENTRY_PAYMENT': ['min'],
    'AMT_INSTALMENT': ['sum'],
    'AMT_PAYMENT': ['sum']    
}).reset_index()

installments_agg.columns = ['_'.join(col).strip('_') for col in installments_agg.columns]
aggregated_installments_payments=installments_agg
aggregated_installments_payments


credit_card_balance=pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')

credit_card_balance


credit_card_balance.columns


credit_card_balance['NAME_CONTRACT_STATUS'].unique()



credit_card_agg = credit_card_balance.groupby('SK_ID_CURR').agg({
    'MONTHS_BALANCE':['count'],
    'AMT_BALANCE': ['mean'],
    'AMT_CREDIT_LIMIT_ACTUAL':['max'],
    'AMT_DRAWINGS_ATM_CURRENT': ['sum'],
    'AMT_DRAWINGS_CURRENT': ['sum'],
    'AMT_DRAWINGS_OTHER_CURRENT': ['sum'],
    'AMT_DRAWINGS_POS_CURRENT': ['sum'],
    'AMT_INST_MIN_REGULARITY': ['mean'],
    'AMT_PAYMENT_CURRENT': ['sum'],
    'AMT_PAYMENT_TOTAL_CURRENT': ['sum'],
    'AMT_RECEIVABLE_PRINCIPAL': ['max'],
    'AMT_RECIVABLE': ['max'],
    'AMT_TOTAL_RECEIVABLE': ['max'],
    'CNT_DRAWINGS_ATM_CURRENT': ['sum'],
    'CNT_DRAWINGS_CURRENT': ['sum'],
    'CNT_DRAWINGS_OTHER_CURRENT': ['sum'],
    'CNT_DRAWINGS_POS_CURRENT': ['sum'],
    'CNT_INSTALMENT_MATURE_CUM': ['max'],
    'SK_DPD': ['max'],
    'SK_DPD_DEF': ['max'],
    'NAME_CONTRACT_STATUS': ['count'] 
}).reset_index()

credit_card_agg.columns = ['_'.join(col).strip('_') for col in credit_card_agg.columns]
aggregated_credit_card_balance=credit_card_agg
aggregated_credit_card_balance


# List of DataFrames to join with application_data
tables_to_merge = [
    aggregated_left_tables,
    pos_agg,
    aggregated_installments_payments,
    aggregated_credit_card_balance,
    aggregated_previous_application
]

# Ensure all tables in tables_to_merge have a flat index (single-level columns)
for i, table in enumerate(tables_to_merge):
    if isinstance(table.columns, pd.MultiIndex):  # If the table has a MultiIndex
        table.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in table.columns]  # Flatten the columns

# Perform sequential left joins on application_data
final_data = application_data.copy()  # Copy the original dataset to final_data
for table in tables_to_merge:
    final_data = final_data.merge(table, on='SK_ID_CURR', how='left')  # Merge each aggregated table

# Reset index after merging
final_data.reset_index(drop=True, inplace=True)

# Verify the shape of the final merged dataset
print(f"Shape of the final merged dataset: {final_data.shape}")





final_data


# حفظ البيانات في ملف CSV
final_data.to_csv('final_data.csv', index=False)
print("saved the final data")
print(f"Shape of the final merged dataset: {final_data.shape}")


print(f"Shape of the final merged dataset: {final_data.shape}")


final_data.dtypes.value_counts()


# عرض عدد ونسبة القيم المفقودة في كل عمود
missing_values = final_data.isnull().sum()
missing_percentage = (missing_values / len(final_data)) * 100
print(pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}).sort_values(by='Percentage', ascending=False))


from sklearn.impute import SimpleImputer

# تعويض القيم الرقمية بالوسيط
num_imputer = SimpleImputer(strategy='median')
final_data[final_data.select_dtypes(include=['float64', 'int64']).columns] = num_imputer.fit_transform(final_data.select_dtypes(include=['float64', 'int64']))

# تعويض القيم التصنيفية بالقيمة الأكثر تكرارًا
cat_imputer = SimpleImputer(strategy='most_frequent')
final_data[final_data.select_dtypes(include=['object']).columns] = cat_imputer.fit_transform(final_data.select_dtypes(include=['object']))



import seaborn as sns
import matplotlib.pyplot as plt

# استخدام Box Plot لرؤية القيم الشاذة
for column in final_data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=final_data[column])
    plt.title(f'Box Plot for {column}')
    plt.show()


# تعديل إزالة القيم الشاذة باستخدام عامل أكبر
final_data = final_data[~((numeric_data < (Q1 - 3 * IQR)) | (numeric_data > (Q3 + 3 * IQR))).any(axis=1)]
print("Final Data After Adjusted Outlier Removal:", final_data.shape)



final_data.head()


import seaborn as sns
import matplotlib.pyplot as plt

# توزيع أحد الأعمدة العددية
plt.figure(figsize=(10, 6))
sns.histplot(final_data['EXT_SOURCE_1'], bins=30, kde=True)
plt.title('Distribution of EXT_SOURCE_1')
plt.show()


# توزيع TARGET
plt.figure(figsize=(8, 5))
sns.countplot(x='TARGET', data=final_data)
plt.title('Distribution of TARGET')
plt.show()


# العلاقة بين EXT_SOURCE_1 و TARGET
plt.figure(figsize=(10, 6))
sns.boxplot(x='TARGET', y='EXT_SOURCE_1', data=final_data)
plt.title('EXT_SOURCE_1 vs TARGET')
plt.show()


# العلاقة بين EXT_SOURCE_1 و NONLIVINGAREA_MODE
plt.figure(figsize=(10, 6))
sns.scatterplot(x='EXT_SOURCE_1', y='NONLIVINGAREA_MODE', hue='TARGET', data=final_data)
plt.title('EXT_SOURCE_1 vs NONLIVINGAREA_MODE')
plt.show()


# مقارنة الأعمدة المهمة
columns_to_plot = ['EXT_SOURCE_1', 'NONLIVINGAREA_MODE', 'APARTMENTS_AVG']

plt.figure(figsize=(15, 8))
final_data[columns_to_plot].boxplot()
plt.title('Box Plot for Key Numeric Features')
plt.show()


# مصفوفة الارتباط لأهم 10 أعمدة عددية
important_columns = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'ELEVATORS_AVG', 'TARGET']
corr = final_data[important_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Key Features', fontsize=14)
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='EXT_SOURCE_1', y='EXT_SOURCE_2', hue='TARGET', data=final_data, palette='coolwarm')
plt.title('Relationship between EXT_SOURCE_1 and EXT_SOURCE_2 by TARGET', fontsize=14)
plt.xlabel('EXT_SOURCE_1', fontsize=12)
plt.ylabel('EXT_SOURCE_2', fontsize=12)
plt.legend(title='TARGET', fontsize=10)
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='TARGET', y='EXT_SOURCE_1', data=final_data, palette='coolwarm')
plt.title('EXT_SOURCE_1 by TARGET', fontsize=14)
plt.xlabel('TARGET', fontsize=12)
plt.ylabel('EXT_SOURCE_1', fontsize=12)
plt.show()


important_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']
sns.pairplot(final_data[important_features], hue='TARGET', palette='coolwarm', diag_kind='kde')
plt.suptitle('Pairplot of Key Features by TARGET', y=1.02, fontsize=16)
plt.show()


key_columns = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
plt.figure(figsize=(12, 8))
sns.boxplot(data=final_data[key_columns], palette='coolwarm')
plt.title('Box Plot of Key Features', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.show()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# اختيار السمات العددية وتوحيد المقياس
scaler = StandardScaler()
numeric_data = scaler.fit_transform(final_data[numeric_columns])

# تطبيق K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(numeric_data)

# إضافة نتائج التجميع إلى البيانات
final_data['Cluster'] = clusters

# رسم النتائج
plt.figure(figsize=(10, 6))
sns.scatterplot(x=final_data['EXT_SOURCE_1'], y=final_data['APARTMENTS_AVG'], hue=final_data['Cluster'], palette='coolwarm')
plt.title('Clustering of Clients')
plt.xlabel('EXT_SOURCE_1')
plt.ylabel('APARTMENTS_AVG')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# اختيار الأعمدة العددية فقط
numeric_columns = final_data.select_dtypes(include=['float64', 'int64']).columns

# حساب مصفوفة الارتباط
corr_matrix = final_data[numeric_columns].corr()

# رسم مصفوفة الارتباط
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


final_data


import seaborn as sns
import matplotlib.pyplot as plt

# رسم التوزيع لعمود معين
sns.histplot(final_data['EXT_SOURCE_1'], bins=30, kde=True)
plt.title('Distribution of EXT_SOURCE_1')
plt.show()


# رسم العلاقة بين عمودين
sns.scatterplot(x=final_data['EXT_SOURCE_1'], y=final_data['APARTMENTS_AVG'])
plt.title('Relationship between EXT_SOURCE_1 and APARTMENTS_AVG')
plt.show()


from sklearn.preprocessing import LabelEncoder

# تحديد الأعمدة التصنيفية
categorical_columns = X.select_dtypes(include=['object']).columns

# تطبيق Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le



# تطبيق One-Hot Encoding
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# توحيد القيم
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تطبيق KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# إضافة النتائج إلى البيانات
final_data['Cluster'] = clusters

# رسم النتائج
sns.scatterplot(x=final_data['EXT_SOURCE_1'], y=final_data['APARTMENTS_AVG'], hue=final_data['Cluster'])
plt.title('Clustering of Clients')
plt.show()



from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


from xgboost import XGBClassifier

# إنشاء النموذج مع تمكين القيم التصنيفية
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    enable_categorical=True,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# تدريب النموذج
xgb_model.fit(X_train, y_train)


print(y_test.value_counts())


from sklearn.model_selection import train_test_split

# إعادة تقسيم البيانات مع stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تحقق من توزيع الفئات بعد التقسيم
print("Training set distribution:\n", y_train.value_counts())
print("Testing set distribution:\n", y_test.value_counts())



print("Original TARGET distribution:\n", y.value_counts())


from sklearn.ensemble import IsolationForest

# تطبيق Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
anomalies = iso.fit_predict(X)

# إضافة النتائج إلى البيانات
final_data['Anomaly'] = anomalies
print(final_data['Anomaly'].value_counts())


# عرض القيم الشاذة
anomalies = final_data[final_data['Anomaly'] == -1]
print("Sample of anomalies:\n", anomalies.head())


# تحديد الأعمدة التصنيفية
categorical_columns = final_data.select_dtypes(include=['object']).columns
print("Categorical columns:\n", categorical_columns)


for col in ['OCCUPATION_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
            'CODE_GENDER', 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 
            'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE']:
    le = LabelEncoder()
    final_data.loc[:, col] = le.fit_transform(final_data[col].astype(str))


final_data = final_data.copy()


from sklearn.preprocessing import LabelEncoder

# عمل نسخة من البيانات لضمان تعديل النسخة الأصلية
final_data = final_data.copy()

# تحويل الأعمدة التصنيفية باستخدام Label Encoding
categorical_columns = ['OCCUPATION_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                       'CODE_GENDER', 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 
                       'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE']

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    # استخدام loc لضمان عدم ظهور التحذير
    final_data.loc[:, col] = le.fit_transform(final_data[col].astype(str))
    label_encoders[col] = le  # حفظ المحول للاستخدام لاحقًا إذا لزم الأمر

print("تم تحويل الأعمدة التصنيفية إلى قيم عددية بنجاح.")


print(final_data.dtypes.value_counts())


print(final_data[categorical_columns].head())


# تقسيم البيانات إلى شاذة وطبيعية
anomalies = final_data[final_data['Anomaly'] == -1]
normal_data = final_data[final_data['Anomaly'] == 1]

# التحليل الإحصائي
summary = pd.DataFrame({
    'Anomalies Mean': anomalies.mean(),
    'Normal Data Mean': normal_data.mean(),
    'Difference': anomalies.mean() - normal_data.mean()
})

# عرض أعلى الفروقات
print(summary.sort_values('Difference', ascending=False).head(10))


import matplotlib.pyplot as plt
import seaborn as sns

# الأعمدة الأعلى اختلافًا
top_diff_columns = summary.sort_values('Difference', ascending=False).head(10).index

# رسم Box Plot لهذه السمات
for col in top_diff_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Anomaly', y=col, data=final_data)
    plt.title(f'Comparison of {col} by Anomaly')
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

columns_to_plot = ['AMT_CREDIT_SUM_SUM', 'AMT_PAYMENT_sum', 'AMT_INSTALMENT_sum']

for col in columns_to_plot:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Anomaly', y=col, data=final_data)
    plt.title(f'{col} Comparison by Anomaly')
    plt.show()


summary.to_csv('anomalies_vs_normal_analysis.csv', index=True)
print("Analysis saved to 'anomalies_vs_normal_analysis.csv'.")


# تحديد الميزات
features = final_data.drop(columns=['Anomaly'])

# الهدف: استخدام العمود Anomaly كهدف
target = final_data['Anomaly']

# تقسيم البيانات إلى تدريب واختبار
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# تحويل القيم المستهدفة
y_train = y_train.replace({-1: 0})
y_test = y_test.replace({-1: 0})


print("Distribution in y_train:")
print(y_train.value_counts())
print("\nDistribution in y_test:")
print(y_test.value_counts())


from xgboost import XGBClassifier

# حساب نسبة التوازن
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# إنشاء النموذج مع ضبط scale_pos_weight
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# تدريب النموذج
xgb_model.fit(X_train, y_train)

# التنبؤ
y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

# تقييم الأداء
from sklearn.metrics import classification_report, roc_auc_score

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# إنشاء النموذج
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# تدريب النموذج
logistic_model.fit(X_train, y_train)

# التنبؤ
y_pred_logistic = logistic_model.predict(X_test)
y_pred_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]

# تقييم الأداء
print("Logistic Regression - Classification Report:")
print(classification_report(y_test, y_pred_logistic))

print("Logistic Regression - ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob_logistic))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# إنشاء النموذج
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# تدريب النموذج
rf_model.fit(X_train, y_train)

# التنبؤ
y_pred_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# تقييم الأداء
print("Random Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Random Forest - ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob_rf))



from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

# إنشاء النموذج
svm_model = SVC(probability=True, random_state=42)

# تدريب النموذج
svm_model.fit(X_train, y_train)

# التنبؤ
y_pred_svm = svm_model.predict(X_test)
y_pred_prob_svm = svm_model.predict_proba(X_test)[:, 1]

# تقييم الأداء
print("SVM - Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("SVM - ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob_svm))



import pandas as pd

# إنشاء قائمة لحفظ النتائج
results = []

# إضافة نتائج XGBoost
results.append({
    "Model": "XGBoost",
    "Precision (0)": classification_report(y_test, y_pred, output_dict=True)['0']['precision'],
    "Recall (0)": classification_report(y_test, y_pred, output_dict=True)['0']['recall'],
    "F1-Score (0)": classification_report(y_test, y_pred, output_dict=True)['0']['f1-score'],
    "ROC-AUC": roc_auc_score(y_test, y_pred_prob)
})

# إضافة نتائج Random Forest
results.append({
    "Model": "Random Forest",
    "Precision (0)": classification_report(y_test, y_pred_rf, output_dict=True)['0']['precision'],
    "Recall (0)": classification_report(y_test, y_pred_rf, output_dict=True)['0']['recall'],
    "F1-Score (0)": classification_report(y_test, y_pred_rf, output_dict=True)['0']['f1-score'],
    "ROC-AUC": roc_auc_score(y_test, y_pred_prob_rf)
})

# إضافة نتائج Logistic Regression
results.append({
    "Model": "Logistic Regression",
    "Precision (0)": classification_report(y_test, y_pred_logistic, output_dict=True)['0']['precision'],
    "Recall (0)": classification_report(y_test, y_pred_logistic, output_dict=True)['0']['recall'],
    "F1-Score (0)": classification_report(y_test, y_pred_logistic, output_dict=True)['0']['f1-score'],
    "ROC-AUC": roc_auc_score(y_test, y_pred_prob_logistic)
})

# إضافة نتائج SVM
results.append({
    "Model": "SVM",
    "Precision (0)": classification_report(y_test, y_pred_svm, output_dict=True)['0']['precision'],
    "Recall (0)": classification_report(y_test, y_pred_svm, output_dict=True)['0']['recall'],
    "F1-Score (0)": classification_report(y_test, y_pred_svm, output_dict=True)['0']['f1-score'],
    "ROC-AUC": roc_auc_score(y_test, y_pred_prob_svm)
})

# تحويل النتائج إلى DataFrame
results_df = pd.DataFrame(results)

# عرض النتائج
print(results_df)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['ROC-AUC'], color='skyblue')
plt.title('Model Comparison - ROC-AUC')
plt.ylabel('ROC-AUC Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Precision (0)'], color='lightgreen')
plt.title('Model Comparison - Precision (Class 0)')
plt.ylabel('Precision Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Recall (0)'], color='coral')
plt.title('Model Comparison - Recall (Class 0)')
plt.ylabel('Recall Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['F1-Score (0)'], color='violet')
plt.title('Model Comparison - F1-Score (Class 0)')
plt.ylabel('F1-Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()


from xgboost import plot_importance

# عرض الميزات الأكثر أهمية
plt.figure(figsize=(10, 8))
plot_importance(xgb_model, max_num_features=10, importance_type='weight')
plt.title('Most Significant Features (XGBoost)')
plt.show()


# استخراج الميزات الأكثر أهمية
xgb_feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 10 Features by XGBoost:")
print(xgb_feature_importance.head(10))


# استخراج الأهمية
rf_feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 10 Features by Random Forest:")
print(rf_feature_importance.head(10))

# رسم الميزات الأكثر أهمية
plt.figure(figsize=(10, 6))
rf_feature_importance.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.title('Most Significant Features (Random Forest)')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()



# استخراج الأهمية
lr_feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': logistic_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("Top 10 Features by Logistic Regression:")
print(lr_feature_importance.head(10))

# رسم الميزات الأكثر تأثيرًا
plt.figure(figsize=(10, 6))
lr_feature_importance.head(10).plot(kind='barh', x='Feature', y='Coefficient', legend=False)
plt.title('Most Significant Features (Logistic Regression)')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()




