import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer


train_data=pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
print(f'shape of the dataset :{train_data.shape}')


train_data.info()


train_data.describe



train_data.columns.tolist()


# missing values
missing_values = train_data.isnull().sum().sort_values(ascending=False)
missing_values


# percentage of missing values
missing_percentage = (missing_values / len(train_data)) * 100
missing_percentage


missing_data =pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
print(missing_data[missing_data['Missing Values'] > 0])


columns_to_keep=missing_percentage[missing_percentage<=58].index
train_data=train_data[columns_to_keep]
train_data.shape


columns_to_impute_20 = missing_percentage[missing_percentage <= 20].index
for column in columns_to_impute_20:
    if train_data[column].dtype in ['int64', 'float64']:  
        median_value = train_data[column].median()
        train_data.loc[:, column] = train_data[column].fillna(median_value) 
    else:  
        mode_value = train_data[column].mode()[0] 
        train_data.loc[:, column] = train_data[column].fillna(mode_value)  
print(train_data[columns_to_impute_20].isnull().sum())


# Get the columns with missing values between 20% and 57%
columns_to_impute_21_57 = missing_percentage[(missing_percentage > 20) & (missing_percentage <= 57)].index

# Separate numerical and categorical columns
numerical_cols_21_57 = train_data[columns_to_impute_21_57].select_dtypes(include=['int64', 'float64']).columns
categorical_cols_21_57 = train_data[columns_to_impute_21_57].select_dtypes(include=['object', 'category']).columns

# Impute numerical columns using the median
median_imputer = SimpleImputer(strategy="median")
train_data.loc[:, numerical_cols_21_57] = median_imputer.fit_transform(train_data[numerical_cols_21_57])

# Impute categorical columns using the most frequent value
mode_imputer = SimpleImputer(strategy="most_frequent")
train_data.loc[:, categorical_cols_21_57] = mode_imputer.fit_transform(train_data[categorical_cols_21_57])

# Check if there are any remaining missing values in the specified columns
print(train_data[columns_to_impute_21_57].isnull().sum())



print("Total missing values after imputing:", train_data.isnull().sum().sum())


print(train_data.columns)
train_data.shape


# Step 1: Drop the TARGET column
train_data = train_data.drop(columns=["TARGET","AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","NAME_CONTRACT_TYPE"
            ,"AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK"])

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

# Step 3: Apply the function on columns_to_check
train_data = check_and_drop_repeated_columns(train_data)


# Step 4: Update the original train_data if needed
# Add the TARGET column back to the cleaned dataset if necessary
#cleaned_train_data = columns_to_check.copy()
#cleaned_train_data['TARGET'] = train_data['TARGET']

# Final shape of cleaned_train_data
print(f"Shape of cleaned_train_data: {train_data.shape}")



train_data.columns


columns_to_add = ["TARGET","AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","NAME_CONTRACT_TYPE"
            ,"AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK"]

# Reload the original data to extract the TARGET column
original_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

# Add the TARGET column back to train_data
train_data[columns_to_add] = original_data[columns_to_add]

# Verify the updated shape
print(f"Shape of train_data after adding important columns: {train_data.shape}")



# Create a new column AGE_YEARS by converting DAYS_BIRTH from negative days to positive age in years
train_data['AGE_YEARS'] = abs(train_data['DAYS_BIRTH']) / 365

# Plot the distribution of AGE_YEARS
plt.figure(figsize=(10, 6))
sns.histplot(train_data['AGE_YEARS'], bins=30, kde=True, color='blue')
plt.title("Distribution of Age (Years)")
plt.xlabel("Age (Years)")
plt.ylabel("Frequency")
plt.show()




# Days_employed column
#======  Convert days to years ======
# Create a new column YEARS_EMPLOYED by converting DAYS_EMPLOYED from negative days to positive years
train_data['YEARS_EMPLOYED'] = abs(train_data['DAYS_EMPLOYED']) / 365.25

# ====== Identify outliers using IQR ======
# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = train_data['YEARS_EMPLOYED'].quantile(0.25)
Q3 = train_data['YEARS_EMPLOYED'].quantile(0.75)
IQR = Q3 - Q1  # Interquartile range

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# ======  Visualize outliers before handling ======
plt.figure(figsize=(8, 6))
sns.boxplot(x=train_data['YEARS_EMPLOYED'], color='skyblue')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.title("Boxplot of YEARS_EMPLOYED (Before Handling Outliers)")
plt.xlabel("Employment (Years)")
plt.legend()
plt.show()

# Create a condition to identify outlier values
outliers_condition = (train_data['YEARS_EMPLOYED'] < lower_bound) | (train_data['YEARS_EMPLOYED'] > upper_bound)

# Print the number of outliers detected
num_outliers = outliers_condition.sum()
print(f"Number of outliers before handling: {num_outliers}")

# Replace outliers with NaN
train_data.loc[outliers_condition, 'YEARS_EMPLOYED'] = np.nan

# Print the number of NaN values after replacing outliers
print(f"Number of NaN values after replacing outliers: {train_data['YEARS_EMPLOYED'].isnull().sum()}")

# Fill missing values (NaN) with the median
median_years_employed = train_data['YEARS_EMPLOYED'].median()
train_data['YEARS_EMPLOYED'].fillna(median_years_employed, inplace=True)

# Recheck the presence of outliers after handling
remaining_outliers_condition = (train_data['YEARS_EMPLOYED'] < lower_bound) | (train_data['YEARS_EMPLOYED'] > upper_bound)
num_remaining_outliers = remaining_outliers_condition.sum()

print(f"Number of outliers after handling: {num_remaining_outliers}")

# Plot the distribution after handling outliers and missing values
plt.figure(figsize=(10, 6))
sns.histplot(train_data['YEARS_EMPLOYED'], bins=30, kde=True, color='green')
plt.title("Histogram of YEARS_EMPLOYED (After Handling Outliers and Filling NaN)")
plt.xlabel("Employment (Years)")
plt.ylabel("Frequency")
plt.show()


if (train_data['YEARS_EMPLOYED'] <= 0).any():
    train_data['YEARS_EMPLOYED'] = train_data['YEARS_EMPLOYED'] + 1    

# ======  Box-Cox Transformation ======
YEARS_EMPLOYED, lambda_value = boxcox(train_data['YEARS_EMPLOYED'])

print(f"Optimal lambda for Box-Cox Transformation: {lambda_value}")

plt.figure(figsize=(10, 6))
sns.histplot(YEARS_EMPLOYED, bins=30, kde=True, color='blue')
plt.title("Histogram of YEARS_EMPLOYED (After Box-Cox Transformation)")
plt.xlabel("Transformed Employment (Years)")
plt.ylabel("Frequency")
plt.show()





from scipy.stats import skew

skewness = skew(YEARS_EMPLOYED)
print(f"Skewness after transformation: {skewness}")



train_data.shape


train_data.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED'], inplace=True)
train_data.shape



# ====== 1. Convert days to years ======
# Create a new column YEARS_REGISTRATION by converting DAYS_REGISTRATION to positive years
train_data['YEARS_REGISTRATION'] = abs(train_data['DAYS_REGISTRATION']) / 365.25

# ====== 2. Identify outliers using IQR ======
# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = train_data['YEARS_REGISTRATION'].quantile(0.25)
Q3 = train_data['YEARS_REGISTRATION'].quantile(0.75)
IQR = Q3 - Q1  # Interquartile range

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# ====== 3. Visualize outliers before handling ======
plt.figure(figsize=(8, 6))
sns.boxplot(x=train_data['YEARS_REGISTRATION'], color='orange')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.title("Boxplot of YEARS_REGISTRATION (Before Handling Outliers)")
plt.xlabel("Registration (Years)")
plt.legend()
plt.show()

# ====== 4. Identify outliers ======
# Create a condition to identify outlier values
outliers_condition = (train_data['YEARS_REGISTRATION'] < lower_bound) | (train_data['YEARS_REGISTRATION'] > upper_bound)

# Print the number of outliers detected
num_outliers = outliers_condition.sum()
print(f"Number of outliers before handling: {num_outliers}")

# ====== 5. Replace outliers with NaN ======
# Replace outliers with NaN
train_data.loc[outliers_condition, 'YEARS_REGISTRATION'] = np.nan

# Print the number of NaN values after replacing outliers
print(f"Number of NaN values after replacing outliers: {train_data['YEARS_REGISTRATION'].isnull().sum()}")

# ====== 6. Handle missing values ======
# Fill missing values (NaN) with the median
median_years_registration = train_data['YEARS_REGISTRATION'].median()
train_data['YEARS_REGISTRATION'].fillna(median_years_registration, inplace=True)

# ====== 7. Verify outliers after handling ======
# Recheck the presence of outliers after handling
remaining_outliers_condition = (train_data['YEARS_REGISTRATION'] < lower_bound) | (train_data['YEARS_REGISTRATION'] > upper_bound)
num_remaining_outliers = remaining_outliers_condition.sum()

print(f"Number of outliers after handling: {num_remaining_outliers}")

# ====== 8. Plot the final distribution ======
# Plot the distribution after handling outliers and missing values
plt.figure(figsize=(10, 6))
sns.histplot(train_data['YEARS_REGISTRATION'], bins=30, kde=True, color='purple')
plt.title("Histogram of YEARS_REGISTRATION (After Handling Outliers and Filling NaN)")
plt.xlabel("Registration (Years)")
plt.ylabel("Frequency")
plt.show()

# ====== 9. Drop the original column if necessary ======
# Drop the original DAYS_REGISTRATION column
train_data.drop(columns=['DAYS_REGISTRATION'], inplace=True)



# ====== 1. Convert days to years ======
# Create a new column YEARS_ID_PUBLISH by converting DAYS_ID_PUBLISH to positive years
train_data['YEARS_ID_PUBLISH'] = abs(train_data['DAYS_ID_PUBLISH']) / 365.25

# ====== 2. Identify outliers using IQR ======
# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = train_data['YEARS_ID_PUBLISH'].quantile(0.25)
Q3 = train_data['YEARS_ID_PUBLISH'].quantile(0.75)
IQR = Q3 - Q1  # Interquartile range

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# ====== 3. Visualize outliers before handling ======
plt.figure(figsize=(8, 6))
sns.boxplot(x=train_data['YEARS_ID_PUBLISH'], color='orange')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.title("Boxplot of YEARS_ID_PUBLISH (Before Handling Outliers)")
plt.xlabel("ID Publish (Years)")
plt.legend()
plt.show()

train_data.drop(columns=['DAYS_ID_PUBLISH'], inplace=True)





