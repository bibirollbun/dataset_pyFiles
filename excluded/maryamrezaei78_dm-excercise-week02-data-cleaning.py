import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


df.info()


# List of features to drop
features_to_drop = [
    'PurchDate',    # Raw date
    'VehYear',      # Duplicate info (VehicleAge is better)
    'Model',        # High cardinality
    'Trim',         # High cardinality
    'SubModel',     # High cardinality
    'WheelTypeID',  # Redundant with WheelType
    'BYRNO',        # Just an ID
    'VNZIP1',       # Zip code, likely not helpful
    'VNST'          # State, likely not helpful
]

# Drop the features from the DataFrame
df = df.drop(columns=features_to_drop)


# Set 'RefId' as the index of the DataFrame
df = df.set_index("RefId")


from sklearn.model_selection import train_test_split

# Define the target variable (y)
y = df['IsBadBuy']

# Define the input features (X) by dropping the target column
X = df.drop(columns=['IsBadBuy'])

# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

inputs = X_train


columns = inputs.columns

# Choose categorical elements 
categorical_indices = [0, 2, 3, 4, 5, 7, 8, 9, 18, 19, 21]

# Use a list comprehension to select the elements at the specified indices
categorical_fields = [columns[i] for i in categorical_indices]

# Create a new list of columns excluding categorical_fields (continuous)
continuous_fields = [j for j in columns if j not in categorical_fields]


import numpy as np

# Define the logical ranges for each continuous feature
logical_ranges = {
    'VehicleAge': (0, 30),
    'VehOdo': (0, 120000),
    'MMRAcquisitionAuctionAveragePrice': (800, 46000),
    'MMRAcquisitionAuctionCleanPrice': (1000, 46000),
    'MMRAcquisitionRetailAveragePrice': (1000, 46000),
    'MMRAcquisitonRetailCleanPrice': (1000, 46000),
    'MMRCurrentAuctionAveragePrice': (300, 46000),
    'MMRCurrentAuctionCleanPrice': (400, 46000),
    'MMRCurrentRetailAveragePrice': (800, 46000),
    'MMRCurrentRetailCleanPrice': (1000, 46000),
    'VehBCost': (1000, 46000),
    'WarrantyCost': (400, 8000)
}

# Apply the logical ranges: out-of-range values are set to NaN
for col, (low, high) in logical_ranges.items():
    if col in X_train.columns:
        X_train.loc[~X_train[col].between(low, high), col] = np.nan


import numpy as np
# Replace 'Manual' with 'MANUAL in the 'Transmission' column
inputs['Transmission'] = inputs['Transmission'].replace("Manual", "MANUAL")

# Replace 'NOT AVAIL' with NaN in the 'Color' column
inputs['Color'] = inputs['Color'].replace('NOT AVAIL', np.nan)



print("Color value counts:")
print(inputs['Color'].value_counts(normalize=True).mul(100).round(2))

print("\nMake value counts:")
print(inputs['Make'].value_counts(normalize=True).mul(100).round(2))


# group classes with less than 1% frequency as 'OTHER' in 'color' and 'make' variables
for col in ['Color', 'Make']:
    freq = inputs[col].value_counts(normalize=True)
    rare_labels = freq[freq < 0.01].index
    inputs[col] = inputs[col].replace(rare_labels, 'OTHER')


print("Color value counts:")
print(inputs['Color'].value_counts(normalize=True).mul(100).round(2))

print("\nMake value counts:")
print(inputs['Make'].value_counts(normalize=True).mul(100).round(2))


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


import matplotlib.pyplot as plt

# Visualize coefficient of variation for continuous features
cv_values.sort_values().plot(kind='barh', figsize=(8, 6), title='Coefficient of Variation by Feature')
plt.axvline(0.1, color='red', linestyle='--', label='Threshold = 0.1')
plt.xlabel('CV')
plt.legend()
plt.tight_layout()
plt.show()


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


# Visualize the dominance of the most frequent category in each categorical feature
plt.figure(figsize=(8, 6))
mode_category.plot(kind='barh', title='Mode Category Percentage by Feature')
plt.axvline(99, color='red', linestyle='--', label='Threshold = 99%')
plt.xlabel('Percentage of Mode Category')
plt.legend()
plt.tight_layout()
plt.show()


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


from scipy.stats import chi2_contingency

#Delete Null Values
subset_primeunit = df[['PRIMEUNIT', 'IsBadBuy']].dropna()
subset_aucguart = df[['AUCGUART', 'IsBadBuy']].dropna()

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


from scipy.stats import fisher_exact

# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(inputs['PRIMEUNIT'], y_train)

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


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


# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(inputs['AUCGUART'], y_train)

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


# Fill missing values in 'AUCGUART' with 'unknown'
inputs['AUCGUART'].fillna('unknown', inplace=True)


# discard 'PRIMEUNIT'
inputs = inputs.drop(['PRIMEUNIT'], axis=1)


values_to_remove = ["PRIMEUNIT"]

# Keep only values that are NOT in values_to_remove
categorical_fields = [x for x in categorical_fields if x not in values_to_remove]
print(categorical_fields) 


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


# Find outliers by their index
outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index

# Drop those outliers from inputs and target
inputs_outprep = inputs.drop(outlier_index)
y_train_outprep = y_train.drop(outlier_index)

# Recombine inputs and target (cleaned)
train_outprep = pd.concat([inputs_outprep, y_train_outprep], axis=1)


price_columns = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 
                 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 
                 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 
                 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']  

# Create a new column counting missing values in price_columns
train_outprep['Num_Missing_Values'] = train_outprep[price_columns].isnull().sum(axis=1)

# Filter rows with 4 or more missing values in the selected columns
rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] > 0]

# Count and percentage
total_rows = len(train_outprep)
rows_with_missing_count = len(rows_with_missing_values)
percentage_rows_with_missing = (rows_with_missing_count / total_rows) * 100

# Display the report
print("Report on Rows with Missing Values in Specific Columns:")
print(f"Total Rows: {total_rows}")
print(f"Rows with ≥4 Missing Values: {rows_with_missing_count} ({percentage_rows_with_missing:.2f}%)")

# Show sorted DataFrame
print("\nRows Sorted by Number of Missing Values (in selected columns):")
print(train_outprep.sort_values(by='Num_Missing_Values', ascending=False))


# Discard rows with missing values
train_outprp_no_missing = train_outprep.dropna()

# Define the threshold for maximum allowable missing values per row
max_missing_values_threshold = 3

# Filter rows based on the 'Num_Missing_Values' column
train_outprep = train_outprep[train_outprep['Num_Missing_Values'] <= max_missing_values_threshold].iloc[:, :-1]


import pandas as pd

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


# Discard rows with missing values
train_outprp_no_missing = train_outprep.dropna()

# Define the threshold for maximum allowable missing values per row
max_missing_values_threshold = 11

# Filter rows based on the 'Num_Missing_Values' column
train_outprep = train_outprep[train_outprep['Num_Missing_Values'] <= max_missing_values_threshold].iloc[:, :-1]


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
mode_imputer = SimpleImputer(strategy='most_frequent')  # When strategy = “constant”, fill_value is used to 
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


train_outprep_no_missing_fix.info()
train_outprep_no_missing_fix.to_csv('/kaggle/working/Carvana_Cleaned.csv')

