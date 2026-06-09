import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
print(df)


# Drop specific columns from the DataFrame
df_excluded = df.drop(['PurchDate', 'VehYear', 'Model', 'Trim', 'SubModel', 'WheelTypeID', 'BYRNO', 'VNZIP1', 'VNST'], axis=1)
df_excluded = df_excluded.set_index('RefId')



y = df_excluded.iloc[:,0]
x = df_excluded.iloc[:,1:]

# --- 3) ØªÙ‚Ø³ÛŒÙ… Ø¯Ø§Ø¯Ù‡ Ø¨Ù‡ Ø¢Ù…ÙˆØ²Ø´ Ùˆ ØªØ³Øª ---
from sklearn.model_selection import train_test_split

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

inputs = x_train



columns = inputs.columns

# Choose categorical elements 
categorical_indices = [0, 2, 3, 4, 5, 7, 8, 9, 18, 19, 21]

# Use a list comprehension to select the elements at the specified indices
categorical_fields = [columns[i] for i in categorical_indices]

# Create a new list of columns excluding categorical_fields (continuous)
continuous_fields = [j for j in columns if j not in categorical_fields]


import pandas as pd

# Define ranges for each column
column_ranges = {
    'VehicleAge': (0, 30),
    'VehOdo': (0, 120000),
    'MMRAcquisitionAuctionAveragePrice': (800, 46000),
    'MMRAcquisitionAuctionCleanPrice': (1000, 46000),
    'MMRAcquisitionRetailAveragePrice': (1000, 46000),
    'MMRAcquisitonRetailCleanPrice': (1000, 46000),
    'MMRCurrentAuctionAveragePrice': (300, 46000),
    'MMRCurrentAuctionCleanPrice': (400,46000),
    'MMRCurrentRetailAveragePrice': (800,46000),
    'MMRCurrentRetailCleanPrice': (1000,46000),
    'VehBCost': (1000,46000),
    'WarrantyCost': (400,8000)
}

# Iterate through each column and fill NaN values outside the defined range
for column, (min_val, max_val) in column_ranges.items():
    inputs[column] = inputs[column].apply(lambda x: x if min_val <= x <= max_val else None)

# Display the updated DataFrame
print(inputs)
inputs.describe()
inputs.info()


import numpy as np

def frequency_table(variable):
    
    # Get unique elements and their counts
    unique_elements, counts = np.unique(variable.dropna(), return_counts=True)

    # Calculate percentages
    percentages = (counts / len(variable)) * 100

    # Create a dictionary to store the value counts and percentages
    value_counts_and_percentages = zip(unique_elements, counts, percentages)

    # Print the value counts and percentages
    for i, j, k in value_counts_and_percentages:
        print(f"{i}: Count: {j}, Percentage: {k:.2f}%")
    return


frequency_table(inputs['Transmission'])


import numpy as np
inputs['Transmission'] = inputs['Transmission'].replace("Manual", "MANUAL")
frequency_table(inputs['Transmission'])



import numpy as np
inputs['Color'] = inputs['Color'].replace('NOT AVAIL', np.nan)
frequency_table(inputs['Color'])


# Function to group low-frequency classes
def group_rare_categories(df, column, threshold = 0.01):
    freq = df[column].value_counts(normalize=True)  # get frequency %
    rare = freq[freq < threshold].index              # find categories < 1%
    df[column] = df[column].apply(lambda x: 'OTHER' if x in rare else x)
    return df

# Apply it to 'color' and 'make'
inputs = group_rare_categories(inputs, 'Color')
inputs = group_rare_categories(inputs, 'Make')
frequency_table(inputs['Color'])
print('-' * 60)
frequency_table(inputs['Make'])


# Define a minimum value for coefficient of variation
min_cv = 0.1

# Calculate the coefficient of variation for each column
cv_values = inputs[continuous_fields].std() / inputs[continuous_fields].mean()

# Filter out columns with CV less than 0.1
selected_columns =  cv_values[cv_values < 0.1].index

# Create a new DataFrame with only the selected columns
filtered_con = inputs[selected_columns]

# Print the resulting DataFrame
inputs_con = inputs[continuous_fields].drop(selected_columns, axis=1)
print(inputs_con)


import pandas as pd

# Define a threshold for the dominant category percentage
threshold = 99

# Calculate the percentage of the mode category for each column
mode_category = (inputs[categorical_fields].apply(lambda x: x.value_counts().max() / len(x)) * 100)

# Select columns where the mode category percentage is greater than the threshold
selected_categorical_columns = mode_category[mode_category > threshold].index

# Create a new DataFrame with only the selected columns
mode_filtered_inputs = inputs[selected_categorical_columns]

# Filter out selected columns and print the resulting DataFrame
inputs_cat = inputs[categorical_fields].drop(selected_categorical_columns, axis=1)
print(inputs_cat)



# Set a threshold for excluding columns 
threshold = 90

# Calculate the percentage of distinct categories in categorical variables
distinct_percentage = (inputs_cat[categorical_fields].apply(lambda x: x.dropna().nunique() / x.count()) * 100)

# Select categorical columns based on distinct percentage threshold
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index

# Create a new DataFrame with only the selected columns
distinct_filtered_inputs = inputs_cat[selected_categorical_columns]

# Filter out selected columns and print the resulting DataFrame
inputs_cat = inputs_cat.drop(selected_categorical_columns, axis=1)
print(inputs_cat)


inputs = pd.concat([inputs_con, inputs_cat], axis=1)


import pandas as pd
from scipy.stats import chi2_contingency

#Delete Null Values
subset = df[['PRIMEUNIT', 'PRIMEUNIT']].dropna()

# Create a contingency table
contingency_table = pd.crosstab(inputs['PRIMEUNIT'], y_train)

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


percentage_low_expected = (expected < 5).sum().sum() / (expected.shape[0] * expected.shape[1]) * 100

print(f"Percentage of cells with expected counts less than 5: {percentage_low_expected:.2f}%")


import pandas as pd
from scipy.stats import fisher_exact

# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(inputs['PRIMEUNIT'], y_train)

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


import pandas as pd
from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table = pd.crosstab(inputs['AUCGUART'], y_train)

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


percentage_low_expected = (expected < 5).sum().sum() / (expected.shape[0] * expected.shape[1]) * 100

print(f"Percentage of cells with expected counts less than 5: {percentage_low_expected:.2f}%")


import pandas as pd
from scipy.stats import fisher_exact

# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(inputs['AUCGUART'], y_train)

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


inputs = inputs.drop(['PRIMEUNIT', 'AUCGUART'], axis=1)


values_to_remove = ["PRIMEUNIT", "AUCGUART"]

# Keep only values that are NOT in values_to_remove
categorical_fields = [x for x in categorical_fields if x not in values_to_remove]
print(categorical_fields)  


cols = ['PRIMEUNIT', 'AUCGUART']

# Replace missing values with 'unknown'
df[cols] = df[cols].fillna('unknown')

# Display the value counts of both columns to see the result
for col in cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())



import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

inputs_iso = inputs.copy()

# Discard rows with NaN valuse
inputs_iso = inputs_iso.dropna()

# Apply Z-score scaling to numerical columns
scaler = StandardScaler()
inputs_iso[continuous_fields] = scaler.fit_transform(inputs_iso[continuous_fields])

# Apply label encoding to categorical columns
label_encoder = LabelEncoder()
inputs_iso[categorical_fields] = inputs_iso[categorical_fields].apply(label_encoder.fit_transform)

# Fit Isolation Forest model
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(inputs_iso)

# Predict outliers
outliers = clf.predict(inputs_iso)

# Add the outlier predictions to your DataFrame
inputs_iso['outlier'] = outliers

# Display the DataFrame with outlier information
print(inputs_iso)

# Calculate the percentage of outliers
percentage_outliers = (outliers[outliers == -1].shape[0] / len(outliers)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")


outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index
inputs_outprep = inputs.drop(outlier_index)
y_train_outprep = y_train.drop(outlier_index)

train_outprep = pd.concat([inputs_outprep, y_train_outprep], axis=1)


print("Removed outliers:", len(outlier_index))



train_outprep.head(10)  # Û±Û° Ø±Ø¯ÛŒÙ� Ø§ÙˆÙ„ Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ù†Ù‡Ø§ÛŒÛŒ



import pandas as pd

# Specify the columns you want to check for missing values
cols_to_check = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']  # replace with your actual column names

# Create a new column counting missing values **only in those columns**
train_outprep['Num_Missing_Values'] = train_outprep[cols_to_check].isnull().sum(axis=1)

# Filter rows with 4 or more missing values in the selected columns
rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] > 0]

# Count and percentage
total_rows = len(train_outprep)
rows_with_missing_count = len(rows_with_missing_values)
percentage_rows_with_missing = (rows_with_missing_count / total_rows) * 100

# Display the report
print("Report on Rows with Missing Values in Specific Columns:")
print(f"Total Rows: {total_rows}")
print(f"Rows with â‰¥4 Missing Values: {rows_with_missing_count} ({percentage_rows_with_missing:.2f}%)")

# Show sorted DataFrame
print("\nRows Sorted by Number of Missing Values (in selected columns):")
print(train_outprep.sort_values(by='Num_Missing_Values', ascending=False))


# Discard rows with missing values
train_outprp_no_missing = train_outprep.dropna()

# Define the threshold for maximum allowable missing values per row
max_missing_values_threshold = 3

# Filter rows based on the 'Num_Missing_Values' column
train_outprep = train_outprep[train_outprep['Num_Missing_Values'] <= max_missing_values_threshold].iloc[:, :-1]


print("Before filtering:")
print("Total rows initially:", total_rows)

print("\nAfter filtering:")
print("Total rows remaining:", len(train_outprep))

print("\nNumber of rows removed:", total_rows - len(train_outprep))



import pandas as pd

# ØªØ¹Ø¯Ø§Ø¯ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§
num_columns = train_outprep.shape[1]

# Ø¢Ø³ØªØ§Ù†Ù‡ 50% Ø¯Ø§Ø¯Ù‡ Ú¯Ù…Ø´Ø¯Ù‡ (Ù†ØµÙ� Ø³ØªÙˆÙ†â€ŒÙ‡Ø§)
threshold = num_columns * 0.5

# Ø³ØªÙˆÙ† ØªØ¹Ø¯Ø§Ø¯ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ÛŒ Ú¯Ù…Ø´Ø¯Ù‡ Ø¨Ø±Ø§ÛŒ Ù‡Ø± Ø³Ø·Ø±
train_outprep['Num_Missing_Values'] = train_outprep.isnull().sum(axis=1)

# Ø§Ù†ØªØ®Ø§Ø¨ Ø³Ø·Ø±Ù‡Ø§ÛŒÛŒ Ú©Ù‡ Ø­Ø¯Ø§Ù‚Ù„ 50% Ø¯Ø§Ø¯Ù‡ Ú¯Ù…Ø´Ø¯Ù‡ Ø¯Ø§Ø±Ù†Ø¯
rows_50_missing = train_outprep[train_outprep['Num_Missing_Values'] >= threshold]

# Ú¯Ø²Ø§Ø±Ø´
total_rows = len(train_outprep)
rows_50_count = len(rows_50_missing)
percentage_50 = (rows_50_count / total_rows) * 100

print("Report on Rows with >=50% Missing Values:")
print(f"Total Rows: {total_rows}")
print(f"Rows with >=50% Missing Values: {rows_50_count} ({percentage_50:.2f}%)")

# Ù†Ù…Ø§ÛŒØ´ 10 Ø³Ø·Ø± Ø§ÙˆÙ„ Ø¨Ø§ Ø¨ÛŒØ´ØªØ±ÛŒÙ† Ù…Ù‚Ø¯Ø§Ø± Ú¯Ù…Ø´Ø¯Ù‡
print("\nTop Rows with Most Missing Values:")
print(train_outprep.sort_values(by='Num_Missing_Values', ascending=False).head(10))



import pandas as pd

# Report on count and percentage of missing values in each column
missing_values_report = pd.DataFrame({
    'Column': train_outprep.columns,
    'Missing Values': train_outprep.isnull().sum(),
    'Percentage Missing': train_outprep.isnull().mean() * 100
})

# Display the missing values report
print("Missing Values Report:")
print(missing_values_report)


import pandas as pd
from sklearn.impute import SimpleImputer

train_outprep_no_missing_fix = train_outprep.copy()

# Create SimpleImputer instances for 'age' and 'ed' columns
median_imputer = SimpleImputer(strategy='median') # you can use 'mean' 
mode_imputer = SimpleImputer(strategy='most_frequent')  # When strategy = â€œconstantâ€�, fill_value is used to 
                                                      #replace all occurrences of missing_values

# Impute missing values in continues_fields
train_outprep_no_missing_fix[continuous_fields] = median_imputer.fit_transform(train_outprep_no_missing_fix[continuous_fields])

# Impute missing values in categorical_fields
train_outprep_no_missing_fix[categorical_fields] = mode_imputer.fit_transform(train_outprep_no_missing_fix[categorical_fields])

# Display the DataFrame after imputation
print("DataFrame after Imputation:")
print(train_outprep_no_missing_fix)

train_outprep_no_missing_fix.info()


import pandas as pd

# Report on count and percentage of missing values in each column
missing_values_report = pd.DataFrame({
    'Column': train_outprep_no_missing_fix.columns,
    'Missing Values': train_outprep_no_missing_fix.isnull().sum(),
    'Percentage Missing': train_outprep_no_missing_fix.isnull().mean() * 100
})

# Display the missing values report
print("Missing Values Report:")
print(missing_values_report)


train_outprep_no_missing_fix.to_csv(
    '/kaggle/working/train_outprep_no_missing_fix.csv',
    index=False
)


