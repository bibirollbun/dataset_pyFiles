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


!pip install scorecardbundle


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from scipy.stats import fisher_exact
from scipy.stats import skew, kurtosis
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from ydata_profiling import ProfileReport
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scorecardbundle.feature_discretization import ChiMerge as cm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



df=pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.info()



# Generate a profile report
profile = ProfileReport(df, title="DontGetKicked EDA", type_schema = {"Auction": "categorical", "IsBadBuy": "categorical"
                                                                               ,"Make": "categorical", "Model": "categorical"
                                                                               ,"Trim": "categorical","SubModel": "categorical"
                                                                              ,"Color": "categorical","Transmission": "categorical"
                                                                              ,"WheelType": "categorical","Nationality": "categorical"
                                                                              ,"Size": "categorical","TopThreeAmericanName": "categorical"
                                                                              ,"PRIMEUNIT": "categorical","AUCGUART": "categorical","VNST": "categorical"})
# Save the report to an HTML file
profile.to_file("DontGetKicked_profile_report.html")
profile


df_IsBadBuy_0 = df[df.IsBadBuy == 0]
df_IsBadBuy_1 = df[df.IsBadBuy == 1]

# Generate a profile report
profile_0 = ProfileReport(df_IsBadBuy_0, title="Don't Get Kicked EDA 0",minimal=True,type_schema = {"Auction": "categorical", "IsBadBuy": "categorical"
                                                                               ,"Make": "categorical", "Model": "categorical"
                                                                               ,"Trim": "categorical","SubModel": "categorical"
                                                                              ,"Color": "categorical","Transmission": "categorical"
                                                                              ,"WheelType": "categorical","Nationality": "categorical"
                                                                              ,"Size": "categorical","TopThreeAmericanName": "categorical"
                                                                              ,"PRIMEUNIT": "categorical","AUCGUART": "categorical","VNST": "categorical"})
profile_1 = ProfileReport(df_IsBadBuy_1, title="Don't Get Kicked EDA 1",minimal=True, type_schema = {"Auction": "categorical", "IsBadBuy": "categorical"
                                                                               ,"Make": "categorical", "Model": "categorical"
                                                                               ,"Trim": "categorical","SubModel": "categorical"
                                                                              ,"Color": "categorical","Transmission": "categorical"
                                                                              ,"WheelType": "categorical","Nationality": "categorical"
                                                                              ,"Size": "categorical","TopThreeAmericanName": "categorical"
                                                                              ,"PRIMEUNIT": "categorical","AUCGUART": "categorical","VNST": "categorical"})

comparison_report = profile_0.compare(profile_1)
comparison_report.to_file("comparison.html")
# comparison_report


# Filter only the numerical columns (int and float types)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Convert to a list and remove the specified columns
numeric_columns = [col for col in numeric_columns if col not in ['RefId','IsBadBuy', 'IsOnlineSale',  
    'VehYear',  
    'WheelTypeID',  
    'BYRNO', 
    'VNZIP1', 'VNST']]

# Calculate the correlation matrix
corr_matrix = df[numeric_columns].corr()

# Set heatmap plot settings
plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
sns.set(font_scale=1.2)

# Plot the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", annot_kws={"size": 8})  # Smaller annotation size

#Add the title to the heatmap
plt.title('Correlation Heatmap of Numerical Features', fontsize=18)

# Display the heatmap
plt.tight_layout()
plt.show()


# Set RefId as the index and drop the column
df.set_index('RefId', inplace=True, drop=True)
# Exclude inappropriate features
columns_to_drop = [
    'PurchDate',  # Dates need to be transformed for analysis
    'VehYear',  # "VehicleAge" is a better alternative
    'Model', 'Trim', 'SubModel',  # Too many classes, requires domain expertise for merging
    'WheelTypeID',  # "WheelType" is already present
    'BYRNO',  # Just an ID
    'VNZIP1', 'VNST'  # Location data may not contribute significantly to prediction
]

# Drop the selected columns
df.drop(columns=columns_to_drop, inplace=True)
df


# Define target variable (y) and input features (X)
Y = df['IsBadBuy']  # Target variable
X = df.drop(columns=['IsBadBuy'])  # Input features

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

inputs= X_train

# Display the shapes of the resulting sets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")


column_ranges = {
    'VehicleAge': (0,30),
    'VehOdo': (0,120000),
    'MMRAcquisitionAuctionAveragePrice': (800,46000),
    'MMRAcquisitionAuctionCleanPrice': (1000,46000),
    'MMRAcquisitionRetailAveragePrice': (1000,46000),
    'MMRAcquisitonRetailCleanPrice': (1000,46000),
    'MMRCurrentAuctionAveragePrice': (300,46000),
    'MMRCurrentAuctionCleanPrice': (400,46000),
    'MMRCurrentRetailAveragePrice': (800,46000),
    'MMRCurrentRetailCleanPrice': (1000,46000),
    'VehBCost': (1000,46000),
    'WarrantyCost': (400,8000)
}

# Store the initial count of missing values
initial_missing = inputs.isnull().sum()

# Iterate through each column and replace values outside the range with None
for column, (min_val, max_val) in column_ranges.items():
    inputs[column] = inputs[column].apply(lambda x: x if min_val <= x <= max_val else None)


# Calculate the new count of missing values after the range check
new_missing = inputs.isnull().sum()

# Calculate the difference in missing values for each column 
missing_diff = new_missing - initial_missing

# Report the number of new missing values introduced
print("New Missing Values Introduced:")
print(missing_diff[missing_diff > 0])  # Only show columns with increased missing values


# Report on count and percentage of missing values in each column
missing_values_report = pd.DataFrame({
    'Missing Values': inputs.isnull().sum(),
    'Percentage Missing': inputs.isnull().mean() * 100
})

# Set the column names as the index
missing_values_report.index.name = 'Column'

# Display the missing values report
print("Missing Values Report:")
print(missing_values_report)


continuous_fields = inputs.select_dtypes(include=['int64', 'float64']).columns
categorical_fields = inputs.select_dtypes(include=['object']).columns


def frequency_table(variable, column_name):
    # Remove missing values (NaN)
    variable = variable.dropna()
    
    # Get unique elements and their counts
    unique_elements, counts = np.unique(variable, return_counts=True)
    
    # Calculate percentages
    percentages = (counts / len(variable)) * 100
    
    # Create a DataFrame to store value counts and percentages
    inputs = pd.DataFrame({
        'Value': unique_elements,
        'Count': counts,
        'Percentage': percentages
    })
    
    # Sort the DataFrame by count in descending order
    inputs = inputs.sort_values(by='Count', ascending=False).reset_index(drop=True)
    # Print the column name and the frequency table
    print(f"\nFrequency Table for '{column_name}':")
    print(inputs.to_string(index=False, float_format="%.2f"))
    
    return inputs

# Iterate over all categorical columns
for column in categorical_fields:
    frequency_table(inputs[column], column)


inputs['Transmission'] = inputs['Transmission'].replace('Manual', 'MANUAL')


frequency_table(inputs['Transmission'],'Transmission')


frequency_table(inputs['Color'],'Color')


# Count of 'NOT AVAIL' before replacement
count_not_avail_before = inputs['Color'].value_counts().get('NOT AVAIL', 0)
print(f"'NOT AVAIL' count before replacement: {count_not_avail_before}")

# Replace 'NOT AVAIL' with pd.NA (pandas' missing value)
inputs['Color'] = inputs['Color'].replace('NOT AVAIL', np.nan)

# Count of missing (NaN) values after replacement using .isna() to check for missing values
count_nan_after = inputs['Color'].isna().sum()
print(f"NaN count after replacement: {count_nan_after}")

# Verify that there are no 'NOT AVAIL' values left after replacement
count_not_avail_after = (inputs['Color'] == 'NOT AVAIL').sum()
print(f"'NOT AVAIL' count after replacement: {count_not_avail_after}")

# To verify the replacement worked correctly, you can print out unique values
unique_values_after = inputs['Color'].unique()
print(f"Unique values in 'Color' after replacement: {unique_values_after}")

print(inputs)


# Check the frequency table for 'Make', 'Color'
for column in ['Make', 'Color']:
    frequency_table(inputs[column], column)

print('\n _______________  AFTER REPLACING  _______________')
# Define a function to replace rare classes with 'OTHER'
def replace_rare_classes(inputs, column, threshold=0.01):
    freq = inputs[column].value_counts(normalize=True)  # Calculate relative frequency of each value
    rare_classes = freq[freq < threshold].index  # Find values with less than the given threshold
    inputs[column] = inputs[column].apply(lambda x: 'OTHER' if x in rare_classes else x)
    return inputs

# Apply the function to both 'color' and 'make' columns
inputs = replace_rare_classes(inputs, 'Color')
inputs = replace_rare_classes(inputs, 'Make')
# Check the frequency table for 'Make', 'Color'
for column in ['Make', 'Color']:
    frequency_table(inputs[column], column)


# Define a minimum value for coefficient of variation
min_cv = 0.1

# Calculate the coefficient of variation for each column
cv_values = inputs[continuous_fields].std() / inputs[continuous_fields].mean()

# Display CV values
print("Coefficient of Variation for each column:")
print(cv_values)

# Filter out columns with CV less than 0.1
selected_columns = cv_values[cv_values < min_cv].index

# Count of columns with CV less than 0.1
count_selected_columns = len(selected_columns)
print(f"Number of columns with CV less than {min_cv}: {count_selected_columns}")

# Create a new DataFrame with only the selected columns
filtered_con = inputs[selected_columns]
# Create a DataFrame excluding selected columns
inputs_con = inputs[continuous_fields].drop(selected_columns, axis=1)


# Define a threshold for the dominant category percentage
threshold = 99

# Calculate the percentage of the mode category for each column
mode_category = (inputs[categorical_fields].apply(lambda x: x.value_counts().max() / len(x)) * 100)

# Display the mode category percentages
print("Percentage of the dominant category for each categorical column:")
print(mode_category)

# Select columns where the mode category percentage is greater than the threshold
selected_categorical_columns = mode_category[mode_category > threshold].index

# Count of columns with dominant category percentage greater than the threshold
count_selected_categorical_columns = len(selected_categorical_columns)
print(f"Number of categorical columns with dominant category percentage greater than {threshold}%: {count_selected_categorical_columns}")
# Create a new DataFrame with only the selected columns
mode_filtered_inputs = inputs[selected_categorical_columns]


inputs_cat = inputs[categorical_fields].drop(selected_categorical_columns, axis=1)


# Set a threshold for excluding columns 
threshold = 90

# Ensure categorical_fields only contains columns in inputs_cat
valid_categorical_fields = [col for col in categorical_fields if col in inputs_cat.columns]

# Calculate the percentage of distinct categories in categorical variables
distinct_percentage = (inputs_cat[valid_categorical_fields].apply(lambda x: x.dropna().nunique() / x.count()) * 100)

# Display distinct percentages for each column
print("Distinct percentage for each categorical column:")
print(distinct_percentage)

# Select categorical columns based on distinct percentage threshold
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index
# Count of selected categorical columns
count_selected_categorical_columns = len(selected_categorical_columns)
print(f"Number of categorical columns with distinct percentage greater than {threshold}%: {count_selected_categorical_columns}")

# Create a new DataFrame with only the selected columns
distinct_filtered_inputs = inputs_cat[selected_categorical_columns]

# Filter out selected columns and print the resulting DataFrame
inputs_cat = inputs_cat.drop(selected_categorical_columns, axis=1)


filtered_df = pd.concat([inputs_con, inputs_cat, y_train], axis=1)
filtered_df.shape


# Create a contingency table
contingency_table = pd.crosstab(filtered_df['PRIMEUNIT'], filtered_df['IsBadBuy'])

print("Contingency Table with Frequencies:")
print(contingency_table)
print("#"*60)

# Calculate row percentages
row_percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

print("\nRow Percentages:")
print(row_percentages)
print("#"*60) 

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)


print(f"\nChi-squared value: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print("#"*60)

# Calculate the percentage of cells with expected counts less than 5
percentage_low_expected = (expected < 5).sum().sum() / (expected.shape[0] * expected.shape[1]) * 100 

print(f"Percentage of cells with expected counts less than 5: {percentage_low_expected:.2f}%")
print("#"*60)

# Calculate residuals (observed minus expected values)
residuals = contingency_table - expected

print("\nResiduals (Observed - Expected):")
print(residuals)
print("#"*60)


# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(filtered_df['PRIMEUNIT'], filtered_df['IsBadBuy'])

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


# Create a contingency table
contingency_table = pd.crosstab(filtered_df['AUCGUART'], filtered_df['IsBadBuy'])

print("Contingency Table with Frequencies:")
print(contingency_table)
print("#"*60)
  
# Calculate row percentages
row_percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

print("\nRow Percentages:")
print(row_percentages)
print("#"*60) 

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-squared value: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print("#"*60)

# Calculate the percentage of cells with expected counts less than 5
percentage_low_expected = (expected < 5).sum().sum() / (expected.shape[0] * expected.shape[1]) * 100 

print(f"Percentage of cells with expected counts less than 5: {percentage_low_expected:.2f}%")
print("#"*60)

# Calculate residuals (observed minus expected values)
residuals = contingency_table - expected

print("\nResiduals (Observed - Expected):")
print(residuals)
print("#"*60)


# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(filtered_df['AUCGUART'], filtered_df['IsBadBuy'])

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


inputs.shape


# Remove the PRIMEUNIT column from the DataFrame
filtered_df = filtered_df.drop(['PRIMEUNIT', 'AUCGUART'], axis=1)

# Check the DataFrame after removal
print(filtered_df.columns)
print(filtered_df.shape)


target= filtered_df.iloc[:,-1]
inputs= filtered_df.iloc[:,0:-1]


# Separate numerical and categorical data
continuous_fields = inputs.select_dtypes(include=['int64', 'float64']).columns
categorical_fields = inputs.select_dtypes(include=['object']).columns


# Make a copy of the inputs data
inputs_iso = inputs.copy()



# Replace rows with NaN valuse with mean and mode
for col in inputs_iso.columns:
    if col in continuous_fields:
        inputs_iso[col] = inputs_iso[col].fillna(inputs_iso[col].mean())
    elif col in categorical_fields:
        mode_val = inputs_iso[col].mode().iloc[0]  # Extract mode value
        inputs_iso[col] = inputs_iso[col].fillna(mode_val)
        

one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(inputs_iso[categorical_fields])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out())
inputs_iso_encoded = pd.concat([one_hot_encoded_df, inputs_iso[continuous_fields].reset_index(drop=True)], axis=1) 
# Apply Z-score scaling to columns
scaler = StandardScaler()
inputs_iso_encoded_array = scaler.fit_transform(inputs_iso_encoded)

# Step 3: Fit Isolation Forest model
clf = IsolationForest(contamination=0.01, random_state=42)

# Here, use the actual values (NumPy array) for fitting
clf.fit(inputs_iso_encoded_array)

# Predict outliers
outliers = clf.predict(inputs_iso_encoded_array)

# Step 4: Add the outlier predictions to your DataFrame
inputs_iso['outlier'] = outliers

# Step 5: Display the DataFrame with outlier information
print(inputs_iso)

# Step 6: Calculate the percentage of outliers
percentage_outliers = (outliers[outliers == -1].shape[0] / len(outliers)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")

# Step 7: Count and display the number of rows that are outliers
num_outliers = (outliers == -1).sum()
print(f"Number of outlier rows: {num_outliers}")


outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index
inputs_outprep = inputs.drop(outlier_index)
y_train_outprep = y_train.drop(outlier_index)

train_outprep = pd.concat([inputs_outprep, y_train_outprep], axis=1)


print(train_outprep.columns)
print(train_outprep.shape)


# List of relevant price-related columns
price_columns = [
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice'
]


# Count missing values in the price-related columns for each row and store it in a new column
train_outprep['price_missing_counts'] = train_outprep[price_columns].isnull().sum(axis=1)


null_counts_sorted =train_outprep.sort_values(by='price_missing_counts', ascending=False)
print(null_counts_sorted)


# Define the threshold for maximum allowable missing values in the price-related columns
max_missing_values_threshold = 4

# Filter rows where the number of missing values in price-related columns is less than or equal to the threshold
train_outprep = train_outprep[train_outprep['price_missing_counts'] < max_missing_values_threshold]


# Display the resulting DataFrame
print(train_outprep.sort_values(by='price_missing_counts', ascending=False))

# Drop the 'price_missing_counts' column (optional) if no longer needed
train_outprep = train_outprep.drop(columns=['price_missing_counts'])


# Create a new column with the number of missing values in each row
train_outprep['Num_Missing_Values'] = train_outprep.isnull().sum(axis=1)

# Count and percentage of rows with missing values
rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] > 0]

total_rows = len(train_outprep)
rows_with_missing_count = len(rows_with_missing_values)
percentage_rows_with_missing = (rows_with_missing_count / total_rows) * 100

# Display the report
print("Report on Rows with Missing Values:")
print(f"Total Rows: {total_rows}")
print(f"Rows with Missing Values: {rows_with_missing_count} ({percentage_rows_with_missing:.2f}%)")


# Display the DataFrame with the new column
print("\nDataFrame with Num_Missing_Values column:")
print(train_outprep.sort_values(by='Num_Missing_Values', ascending = False))
train_outprep = train_outprep.drop(columns=['Num_Missing_Values'])


print(train_outprep.columns)
print(len(train_outprep.columns))


# Create a new column with the number of missing values in each row
train_outprep['Num_Missing_Values'] = train_outprep.isnull().sum(axis=1)

# Count and percentage of rows with missing values
print('Count and percentage of rows with 50% or more null values across all fields: ')
rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] > 12]
print(f"Number of rows with more than 12 missing values: {len(rows_with_missing_values)}")
train_outprep = train_outprep.drop(columns=['Num_Missing_Values'])


# Report on count and percentage of missing values in each column
missing_values_report = pd.DataFrame({
    'Missing Values': train_outprep.isnull().sum(),
    'Percentage Missing': train_outprep.isnull().mean() * 100
})

# Set the column names as the index
missing_values_report.index.name = 'Column'

# Reset the index to have 'Column' as a separate field if needed
# missing_values_report.reset_index(inplace=True)

# Display the missing values report
print("Missing Values Report:")
print(missing_values_report)


# Make a copy of the original DataFrame
train_outprep_no_missing_fix = train_outprep.copy()

# Dynamically identify numeric and categorical columns
numeric_columns = train_outprep_no_missing_fix.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = train_outprep_no_missing_fix.select_dtypes(include=['object']).columns

# Create an imputer for numeric columns with 'median' strategy (or 'mean' if preferred)
numeric_imputer = SimpleImputer(strategy='median')

# Create an imputer for categorical columns with 'most_frequent' strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputer for numeric columns
train_outprep_no_missing_fix[numeric_columns] = numeric_imputer.fit_transform(train_outprep_no_missing_fix[numeric_columns])
# Apply imputer for categorical columns
train_outprep_no_missing_fix[categorical_columns] = categorical_imputer.fit_transform(train_outprep_no_missing_fix[categorical_columns])

# Display DataFrame information after imputation
train_outprep_no_missing_fix.info()


# Calculate and print the total number of missing values in train_outprep_no_missing_fix
total_missing_values = train_outprep_no_missing_fix.isnull().sum().sum()
print(f"Total number of missing values: {total_missing_values}")


# Report on count and percentage of missing values in each column
missing_values_report = pd.DataFrame({
    'Missing Values': train_outprep_no_missing_fix.isnull().sum(),
    'Percentage Missing': train_outprep_no_missing_fix.isnull().mean() * 100
})

# Set the column names as the index
missing_values_report.index.name = 'Column'

# Reset the index to have 'Column' as a separate field if needed
# missing_values_report.reset_index(inplace=True)

# Display the missing values report
print("Missing Values Report:")
print(missing_values_report)


train_FS= train_outprep_no_missing_fix.copy()


train_FS['IsOnlineSale'] = train_FS[['IsOnlineSale']].astype('int8').astype('str')

train_FS.info()  


# List of variables you want to convert to string (categorical variables)
variables_to_convert = train_FS.select_dtypes(exclude=[np.number])

# Loop through the columns and convert them to string
for column in variables_to_convert:
    train_FS[column] = train_FS[column].astype(str).replace('nan', np.nan)


# Select numerical columns (both integers and floats)
numerical_columns = train_FS.select_dtypes(include=[np.int64, np.float64])

# Select object columns as categorical
categorical_columns = train_FS.select_dtypes(include=['object'])


desc_numerical=numerical_columns.describe()
# Calculate skewness and kurtosis for numerical columns
desc_numerical.loc['skewness'] = numerical_columns.apply(lambda x: skew(x))
desc_numerical.loc['kurtosis'] = numerical_columns.apply(lambda x: kurtosis(x))
desc_numerical


def frequency_table(variable):
    
    # Get unique elements and their counts
    unique_elements, counts = np.unique(variable, return_counts=True)

    # Calculate percentages
    percentages = (counts / len(variable)) * 100

    # Create a dictionary to store the value counts and percentages
    value_counts_and_percentages = zip(unique_elements, counts, percentages)

    # Print the value counts and percentages
    for i, j, k in value_counts_and_percentages:
        print(f"{i}: Count: {j}, Percentage: {k:.2f}%")


chi_merge_list = ['VehBCost' , 'WarrantyCost']

trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=5, decimal=3,output_dataframe=True)
result_cm = trans_cm.fit_transform(train_FS[chi_merge_list], train_FS['IsBadBuy'])
trans_cm.boundaries_


# Add -inf to the beginning of each array
boundaries_dict = {key: np.insert(boundaries, 0, -np.inf) for key, boundaries in trans_cm.boundaries_.items()}

# Iterate through the dictionary and add new columns to cleaned_df
for key, boundaries in boundaries_dict.items():
    column_name = f"{key}_cat_cm"
    train_FS[column_name] = pd.cut(train_FS[key], bins=boundaries, labels=[1,2,3,4,5], right=False)
    
    # Print the variable name and its bin edges
    print(f'{column_name} bin edges:', boundaries)
    # Print the frequency table
    frequency_table(train_FS[column_name])
    print("\n")

# Display the updated DataFrame
print(train_FS.columns)
train_FS = train_FS.drop(columns=['VehBCost', 'WarrantyCost'])
numerical_columns = train_FS.select_dtypes(include=[np.int64, np.float64])
train_FS.info()
desc_numerical=numerical_columns.describe()
# Calculate skewness and kurtosis for numerical columns
desc_numerical.loc['skewness'] = numerical_columns.apply(lambda x: skew(x))
desc_numerical.loc['kurtosis'] = numerical_columns.apply(lambda x: kurtosis(x))
desc_numerical
train_FS.to_csv('train_FS.csv', index=False)


train_FS= pd.read_csv('/kaggle/working/train_FS.csv')


print(f'AS YOU WISHED: {train_FS.shape[1]} fields,  MOVE FORWARD ')


# Apply One-Hot Encoding

one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

one_hot_encoded = one_hot_encoder.fit_transform(train_FS[["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"]])

# Add results into dataframe
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out())


# Add results into dataframe
encoded_train_FS = pd.concat([train_FS.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)
encoded_train_FS = encoded_train_FS.drop(columns=["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"])

encoded_train_FS.info()


print(f'Now, the dataset contains: {encoded_train_FS.shape[1]} fields!')


from sklearn.preprocessing import MinMaxScaler

# Make a copy of the original DataFrame
train_FS_scaled = encoded_train_FS.copy()

# Separate the target column 'IsBadBuy'
y_train = train_FS_scaled['IsBadBuy']

# Select features (exclude 'IsBadBuy') to apply scaling
selected_features = train_FS_scaled.columns.drop('IsBadBuy')

# Apply Min-Max Scaling to selected features
min_max_scaler = MinMaxScaler()
train_FS_scaled[selected_features] = min_max_scaler.fit_transform(train_FS_scaled[selected_features])

# The target column 'IsBadBuy' remains unchanged in the DataFrame

# Save the scaled DataFrame with the target column included to a CSV file
train_FS_scaled.to_csv('/kaggle/working/train_FS_scaled.csv', index=False)
print(train_FS_scaled)


train_FE= train_outprep_no_missing_fix.copy()


# List of features to transform
selected_features = ['VehBCost', 'WarrantyCost']

# Iterate through selected features
for feature in selected_features:
    # Check if the feature contains negative values
    has_negative_values = (train_FE[feature] <= 0).any()

    # Choose the appropriate transformation method
    if has_negative_values:
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    else:
        transformer = PowerTransformer(method='box-cox', standardize=False)

    # Fit and transform the feature, and store the result in the new DataFrame
    train_FE[f"{feature}_transformed"] = transformer.fit_transform(train_FE[[feature]])
# Get the lambda parameter used for transformation
    lambda_value = transformer.lambdas_[0]
    print(f"Lambda for {feature}: {lambda_value}")
    
    # Plot histograms for original and transformed features
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.hist(train_FE[feature], bins=30, color='blue', alpha=0.7)
    plt.title(f'Original {feature} Histogram')

    plt.subplot(1, 2, 2)
    plt.hist(train_FE[f"{feature}_transformed"], bins=30, color='green', alpha=0.7)
    plt.title(f'Transformed {feature} Histogram')

    plt.tight_layout()
    plt.show()

    
# Display the transformed DataFrame
print('\n')
print(train_FE)


train_FE = train_FE.drop(columns=['VehBCost', 'WarrantyCost'])


print(f'HERE YOU ARE: {train_FE.shape[1]} fields! ')



# Apply One-Hot Encoding


one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
nominal_fields= ["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"]

one_hot_encoded = one_hot_encoder.fit_transform(train_FE[["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"]])

# Add results into dataframe
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out())

encoded_train_FE = pd.concat([train_FE.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)


encoded_train_FE = encoded_train_FE.drop(columns=["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"])
encoded_train_FE.info()


print(f'ALRIGHT : {encoded_train_FE.shape[1]} fields!')


from sklearn.preprocessing import StandardScaler

# Make a copy of the original DataFrame
train_FE_scaled = encoded_train_FE.copy()

# Separate the target column 'IsBadBuy'
y_train = train_FE_scaled['IsBadBuy']

# Select features (exclude 'IsBadBuy') to apply scaling
selected_features = train_FE_scaled.columns.drop('IsBadBuy')

# Apply Z-Score Scaling to selected features
z_score_scaler = StandardScaler()
train_FE_scaled[selected_features] = z_score_scaler.fit_transform(train_FE_scaled[selected_features])

# The target column 'IsBadBuy' remains unchanged in the DataFrame

# Save the scaled DataFrame with the target column included to a CSV file
train_FE_scaled.to_csv('/kaggle/working/train_FE_scaled.csv', index=False)

# Display the scaled DataFrame
print(train_FE_scaled)


train_FS_scaled


X_train_scaled_FS = train_FS_scaled.drop('IsBadBuy', axis=1)
X_train_scaled_FS.info()


from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor

# configure to select all features
selector = RFECV(estimator=DecisionTreeRegressor(random_state=29), step=1, min_features_to_select=10, cv=5, n_jobs=-1)


# learn relationship from training data
selector.fit(X_train_scaled_FS, y_train)

selector.get_support()

print(f"Optimal number of features: {selector.n_features_}")

print("="*50)

wrapper_fs = selector.get_feature_names_out()
print("Wrapper Optimal Feature List:")
print(wrapper_fs)

X_train_scaled_wrapper_fs = X_train_scaled_FS[wrapper_fs]
X_train_scaled_wrapper_fs= X_train_scaled_wrapper_fs.to_csv('X_train_scaled_wrapper_fs.csv', index=False)


from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor

# configure to select all features
selector = RFECV(estimator=DecisionTreeRegressor(random_state=29), step=1, min_features_to_select=10, cv=5, n_jobs=-1)


# learn relationship from training data
selector.fit(X_train_scaled_FS, y_train)

selector.get_support()

print(f"Optimal number of features: {selector.n_features_}")

print("="*50)
wrapper_fs = selector.get_feature_names_out()
print("Wrapper Optimal Feature List:")
print(wrapper_fs)

X_train_scaled_wrapper_fs = X_train_scaled_FS[wrapper_fs]
X_train_scaled_wrapper_fs.to_csv('X_train_scaled_wrapper_fs.csv', index=False)


train_FE_scaled.drop('IsBadBuy', axis=1, inplace=True)
X_train_scaled_FE = train_FE_scaled
continuous_fields=['VehicleAge', 'VehOdo',
                      'MMRAcquisitionAuctionAveragePrice',
                      'MMRAcquisitionAuctionCleanPrice',
                      'MMRAcquisitionRetailAveragePrice',
                      'MMRAcquisitonRetailCleanPrice',
                      'MMRCurrentAuctionAveragePrice',
                      'MMRCurrentAuctionCleanPrice',
                      'MMRCurrentRetailAveragePrice',
                     'MMRCurrentRetailCleanPrice'
                   , 'VehBCost_transformed',   'WarrantyCost_transformed']



# Create the DataFrame with only continuous features
train_FE_scaled_continuous = X_train_scaled_FE[continuous_fields]


# Display the resulting DataFrame
print(train_FE_scaled_continuous)



# Compute correlation matrix
correlation_matrix = train_FE_scaled_continuous.corr()

# Step 2: Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 7})
plt.title('Pearson Correlation Heatmap')
plt.show()


from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components= None  , random_state=717)
pca.fit(train_FE_scaled_continuous)

pc_name = pd.DataFrame([f'pc_{i+1}' for i in range(pca.n_components_)], columns=['name'])
variance = pd.DataFrame(pca.explained_variance_, columns=['variance'])
variance_ratio = pd.DataFrame(pca.explained_variance_ratio_, columns=['variance_ratio'])
total_explained_variance = variance_ratio.sum()
component_weights = pd.DataFrame(pca.components_, columns=train_FE_scaled_continuous.columns)

pca_report = pd.concat((pc_name, variance, variance_ratio, component_weights), axis=1).set_index('name')



# Plotting the explained variance (Eigenvalue) and variance ratio
plt.figure(figsize=(12, 6))

# Plot the eigenvalues (variance)
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(variance) + 1), variance['variance'], marker='o', label='Eigenvalue (Variance)')
plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue = 1')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (Variance)')
plt.title('Eigenvalues of Principal Components')
plt.legend()


plt.tight_layout()
plt.show()

# Find the number of components with eigenvalue > 1
n_components_selected = sum(variance['variance'] > 1)
print(f"\nNumber of components with eigenvalue > 1: {n_components_selected}")


from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components= n_components_selected  , random_state=717)
pca.fit(train_FE_scaled_continuous)

pc_name = pd.DataFrame([f'pc_{i+1}' for i in range(pca.n_components_)], columns=['name'])
variance = pd.DataFrame(pca.explained_variance_, columns=['variance'])
variance_ratio = pd.DataFrame(pca.explained_variance_ratio_, columns=['variance_ratio'])
component_weights = pd.DataFrame(pca.components_, columns=train_FE_scaled_continuous.columns)

pca_report = pd.concat((pc_name, variance, variance_ratio, component_weights), axis=1).set_index('name')
pca_report


# Extract component weights (loadings)
component_weights = pd.DataFrame(pca.components_, columns=train_FE_scaled_continuous.columns)

# Display the weights of each feature in each component
component_weights


total_explained_variance = variance_ratio.sum()
print(f' Total {total_explained_variance}')


continuous_fields2=['VehicleAge', 'VehOdo',
                      'MMRAcquisitionAuctionAveragePrice',
                      'MMRAcquisitionAuctionCleanPrice',
                      'MMRAcquisitionRetailAveragePrice',
                      'MMRAcquisitonRetailCleanPrice',
                      'MMRCurrentAuctionAveragePrice',
                      'MMRCurrentAuctionCleanPrice',
                      'MMRCurrentRetailAveragePrice',
                     'MMRCurrentRetailCleanPrice']

# Create the DataFrame with only continuous features
train_FE_scaled_continuous2 = X_train_scaled_FE[continuous_fields2]


# Display the resulting DataFrame
print(train_FE_scaled_continuous2)


from sklearn.decomposition import PCA

# Perform PCA
pca2 = PCA(n_components= None  , random_state=717)
pca2.fit(train_FE_scaled_continuous2)

pc_name2 = pd.DataFrame([f'pc_{i+1}' for i in range(pca2.n_components_)], columns=['name'])
variance2 = pd.DataFrame(pca2.explained_variance_, columns=['variance'])
variance_ratio2 = pd.DataFrame(pca2.explained_variance_ratio_, columns=['variance_ratio'])
total_explained_variance2 = variance_ratio2.sum()
component_weights2 = pd.DataFrame(pca2.components_, columns=train_FE_scaled_continuous2.columns)

pca_report2 = pd.concat((pc_name2, variance2, variance_ratio2, component_weights2), axis=1).set_index('name')


# Plotting the explained variance (Eigenvalue) and variance ratio
plt.figure(figsize=(12, 6))

# Plot the eigenvalues (variance)
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(variance2) + 1), variance2['variance'], marker='o', label='Eigenvalue (Variance)')
plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue = 1')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (Variance)')
plt.title('Eigenvalues of Principal Components')
plt.legend()


plt.tight_layout()
plt.show()

# Find the number of components with eigenvalue > 1
n_components_selected2 = sum(variance2['variance'] > 1)
print(f"\nNumber of components with eigenvalue > 1: {n_components_selected2}")


from sklearn.decomposition import PCA

# Perform PCA
pca2 = PCA(n_components= n_components_selected2  , random_state=717)
pca2.fit(train_FE_scaled_continuous2)

pc_name2 = pd.DataFrame([f'pc_{i+1}' for i in range(pca2.n_components_)], columns=['name'])
variance2 = pd.DataFrame(pca2.explained_variance_, columns=['variance'])
variance_ratio2 = pd.DataFrame(pca2.explained_variance_ratio_, columns=['variance_ratio'])
component_weights2 = pd.DataFrame(pca2.components_, columns=train_FE_scaled_continuous2.columns)

pca_report2 = pd.concat((pc_name2, variance2, variance_ratio2, component_weights2), axis=1).set_index('name')
pca_report2


# Extract component weights (loadings)
component_weights2 = pd.DataFrame(pca2.components_, columns=train_FE_scaled_continuous2.columns)

# Display the weights of each feature in each component
component_weights2


total_explained_variance2 = variance_ratio2.sum()
print(f' Total {total_explained_variance2}')


# Assume that total_explained_variance and total_explained_variance2 
# were computed from two PCA runs (with and without discrete quantitative features)

# Sum the explained variance for both versions
total_explained_variance_sum = total_explained_variance.sum()
total_explained_variance2_sum = total_explained_variance2.sum()

# Compare the total explained variance between the two versions
if total_explained_variance2_sum > total_explained_variance_sum:
    print(f"Version 2 explains more variance with {total_explained_variance2_sum:.2f} compared to {total_explained_variance_sum:.2f}.")
    print("Removing discrete quantitative features seems to improve the total explained variance.")
else:
    print(f"Version 1 explains more variance with {total_explained_variance_sum:.2f} compared to {total_explained_variance2_sum:.2f}.")
    print("Including discrete quantitative features seems to capture more variance.")


pca_X_train = pca.transform(train_FE_scaled_continuous)
pca_X_train = pd.DataFrame(pca_X_train, columns = pc_name['name'].tolist())

k = 3
pca_train = pd.concat((y_train.reset_index(drop=True), pca_X_train.iloc[:, 0:k]), axis=1)


import seaborn as sns
# Set up pairplot with hue
sns.pairplot(pca_train, hue='IsBadBuy')
# Show the pairplot
plt.show()


#  Select non-continuous features from train_FE_scaled
non_continuous_features=train_FE_scaled.drop(columns=continuous_fields)

#  Combine PCA components with non-continuous features
train_pca_fe = pd.concat([pca_train, non_continuous_features], axis=1)

train_pca_fe.to_csv('train_pca_fe.csv', index=False)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Step 1: Fit the LDA model for full n_components
lda = LinearDiscriminantAnalysis(n_components= None)


lda.fit(train_FE_scaled_continuous, y_train)


plt.plot(lda.explained_variance_ratio_, marker='o')

plt.title("LDA component and their variance ratio")
plt.xlabel("nth LDA component")
plt.ylabel("variance ratio")
plt.show()


# Step 2: Fit and Transform the original features into the reduced-dimensional space
lda = LinearDiscriminantAnalysis(n_components=1)
lda_X_train = lda.fit_transform(train_FE_scaled_continuous, y_train)

columns_name = [f'lda_{i+1}' for i in range(lda_X_train.shape[1])]
lda_X_train = pd.DataFrame(lda_X_train, columns = columns_name)


lda_train = pd.concat((y_train.reset_index(drop=True), lda_X_train), axis=1)

import seaborn as sns

# Set up pairplot with hue
sns.pairplot(lda_train, hue='IsBadBuy')
# Show the pairplot
plt.show()


import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Step 1: Fit the LDA model
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(train_FE_scaled_continuous, y_train)

# Step 2: Plot LDA component for each class
plt.figure(figsize=(8, 6))

# Filter data points for each class
plt.scatter(X_train_lda[y_train == 0], [0] * sum(y_train == 0), 
            color='blue', label='Class 0', alpha=0.5)
plt.scatter(X_train_lda[y_train == 1], [0] * sum(y_train == 1), 
            color='red', label='Class 1', alpha=0.5)

plt.title("LDA Component Distribution by Class")
plt.xlabel("LDA Component 1")
plt.legend()
plt.show()


#  Combine PCA components with non-continuous features
train_lda_fe = pd.concat([lda_train, non_continuous_features], axis=1)

train_lda_fe.to_csv('train_lda_fe.csv', index=False)

