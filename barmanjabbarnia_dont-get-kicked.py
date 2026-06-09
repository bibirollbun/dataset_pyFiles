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
training_file_path = "/kaggle/input/DontGetKicked/training.csv"
df = pd.read_csv(training_file_path)
df.info()


!pip install ydata_profiling


# Remove unnecessary or low-value columns before modeling

drop_cols = [
    "PurchDate",         # date column not used directly
    "VehYear",           # redundant with VehicleAge
    "Model",             # too many classes, needs expert consolidation
    "Trim",              # same as above
    "SubModel",          # same as above
    "WheelTypeID",       # redundant with WheelType
    "BYRNO",             # ID column
    "VNZIP1",            # location ZIP code
    "VNST"               # location state
]

# Drop columns if they exist in df
df = df.drop(columns=drop_cols, errors="ignore")

# Display remaining columns for verification
print("Remaining columns after drop:", df.columns.tolist())



# Separate the target variable
target = df["IsBadBuy"]

# Separate the input variables
inputs = df.drop(columns=["IsBadBuy"])

# Display shapes for validation
print("Inputs shape:", inputs.shape)
print("Target shape:", target.shape)



columns = inputs.columns

# Choose categorical elements 
# Update the indices based on your dataset: exclude the target and include actual categorical columns
categorical_indices = [
    columns.get_loc(col) for col in [
         "Auction", "Make",  
        "Color", "Transmission", "WheelType", "Nationality", "Size",
        "TopThreeAmericanName", "PRIMEUNIT", "AUCGUART", 
    ]
]

# Use a list comprehension to select the elements at the specified indices
categorical_fields = [columns[i] for i in categorical_indices]

# Create a new list of columns excluding categorical_fields (continuous)
continuous_fields = [j for j in columns if j not in categorical_fields]




# Define a minimum value for coefficient of variation
min_cv = 0.1

# Calculate the coefficient of variation for each continuous column
cv_values = inputs[continuous_fields].std() / inputs[continuous_fields].mean()
print(cv_values)

# Filter out columns with CV less than the minimum threshold
selected_columns = cv_values[cv_values < min_cv].index
print(selected_columns)

# Create a new DataFrame with only the filtered (low-variation) columns
filtered_con = inputs[selected_columns]

# Drop low-variation columns from the continuous fields to keep high-variation columns
inputs_con = inputs[continuous_fields].drop(selected_columns, axis=1)

# Print the resulting DataFrame
print(inputs_con)





import pandas as pd

# Define a threshold for the dominant category percentage
threshold = 99

# Calculate the percentage of the mode category for each categorical column
mode_category = (inputs[categorical_fields].apply(lambda x: x.value_counts().max() / len(x)) * 100)
print(mode_category)

# Select columns where the mode category percentage is greater than the threshold
selected_categorical_columns = mode_category[mode_category > threshold].index
print(selected_categorical_columns)

# Create a new DataFrame with only the selected high-dominance columns
mode_filtered_inputs = inputs[selected_categorical_columns]

# Filter out selected columns to keep more balanced categorical columns
inputs_cat = inputs[categorical_fields].drop(selected_categorical_columns, axis=1)

# Print resulting DataFrame for verification
print(inputs_cat)



import pandas as pd

# Set a threshold for excluding high-cardinality categorical columns
threshold = 90

# Use only the current columns in inputs_cat
current_categorical_cols = inputs_cat.columns

# Calculate the percentage of distinct categories in each categorical variable
distinct_percentage = (inputs_cat[current_categorical_cols]
                       .apply(lambda x: x.dropna().nunique() / x.count()) * 100)
print(distinct_percentage)
# Select categorical columns exceeding the distinct percentage threshold
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index
print(selected_categorical_columns)

# Create a new DataFrame with only the high-cardinality columns
distinct_filtered_inputs = inputs_cat[selected_categorical_columns]

# Filter out selected columns to keep more manageable categorical columns
inputs_cat = inputs_cat.drop(selected_categorical_columns, axis=1)

# Print resulting DataFrame for verification
print(inputs_cat)




import pandas as pd

# Recombine the filtered continuous features, filtered categorical features, and target variable
filtered_df = pd.concat([inputs_con, inputs_cat, target], axis=1)

# Display the shape to verify
print("Filtered DataFrame shape:", filtered_df.shape)

# Optional: display first few rows
print(filtered_df.head())



import pandas as pd

# Define logical ranges for relevant continuous columns
# Adjust the ranges based on realistic car dataset values
column_ranges = {
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
# Iterate through each column and set values outside the defined range to NaN
for column, (min_val, max_val) in column_ranges.items():
    filtered_df[column] = filtered_df[column].apply(lambda x: x if min_val <= x <= max_val else None)

# Display the updated DataFrame
print(filtered_df)
filtered_df.describe()
filtered_df.info()






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

# Call the function on the target variable 'IsBadBuy'
frequency_table(filtered_df["Transmission"])
# Call the frequency_table function on the 'Transmission' column
#frequency_table(filtered_df['Transmission'])




filtered_df["Color"] = filtered_df["Color"].replace("NOT AVAIL" , np.nan)


import pandas as pd

# Function to group rare categories
def group_rare_categories(df, column, threshold=0.01):
    """
    Groups categories in a column with frequency below 'threshold'
    into a new category called 'OTHER'.
    """
    freq = df[column].value_counts(normalize=True)   # relative frequency
    rare_classes = freq[freq < threshold].index       # categories <1%
    
    df[column] = df[column].replace(rare_classes, 'OTHER')
    return df

# Apply to Color and Make in filtered_df
filtered_df = group_rare_categories(filtered_df, 'Color', threshold=0.01)
filtered_df = group_rare_categories(filtered_df, 'Make', threshold=0.01)



# Standardize Transmission values
filtered_df['Transmission'] = filtered_df['Transmission'].replace({
    'Manual': 'MANUAL'      # removes trailing space if any
})
print(filtered_df['Transmission'].unique())


# Fill missing values with 'unknown' in-place
filtered_df['AUCGUART'].fillna('unknown', inplace=True)
filtered_df['PRIMEUNIT'].fillna('unknown', inplace=True)



# Save the cleaned DataFrame to a CSV file
filtered_df.to_csv("cleaned_car_data.csv", index=False)

# Optional: confirm save
print("Cleaned dataset saved successfully!")




from scipy.stats import chi2_contingency
import pandas as pd

# Variable to test
var = "PRIMEUNIT"  # change to "AUCGUART" for the second test

# Create contingency table (non-null only)
contingency_table = pd.crosstab(
    filtered_df[var].dropna(),
    filtered_df['IsBadBuy'].dropna()
)

print("Contingency Table:\n", contingency_table)

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p:.4f}")

# Decision
alpha = 0.05
if p < alpha:
    print(f"Reject Hâ‚€ â†’ There is a significant relationship between {var} and IsBadBuy.")
else:
    print(f"Fail to reject Hâ‚€ â†’ No significant relationship between {var} and IsBadBuy.")

# Percentage of low expected frequencies
percentage_low_expected = (expected < 5).sum().sum() / (expected.shape[0] * expected.shape[1]) * 100

print(f"Percentage of cells with expected counts less than 5: {percentage_low_expected:.2f}%")
print("#"*60)

# Calculate residuals (Observed - Expected)
residuals = contingency_table - expected

print("\nResiduals (Observed - Expected):")
print(residuals)
print("#"*60)



import pandas as pd
from scipy.stats import fisher_exact

# Variable to test (change to "AUCGUART" if needed)
var = "PRIMEUNIT"

# Drop missing values for Fisher test
df_test = filtered_df[[var, 'IsBadBuy']].dropna()

# Keep only the top 2 most frequent classes (Fisher requires 2Ã—2)
top2 = df_test[var].value_counts().index[:2]

df_top2 = df_test[df_test[var].isin(top2)]

# Create the 2Ã—2 contingency table
contingency_table = pd.crosstab(df_top2[var], df_top2['IsBadBuy'])

print("2Ã—2 Contingency Table for Fisher Test:\n", contingency_table)

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"\nOdds ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.6f}")

# Decision rule
alpha = 0.05
if p_value < alpha:
    print(f"Reject Hâ‚€ â†’ Significant relationship exists between {var} and IsBadBuy (Fisher test).")
else:
    print(f"Fail to reject Hâ‚€ â†’ No significant relationship between {var} and IsBadBuy.")



# Separate target variable
y = filtered_df["IsBadBuy"]

# Separate input variables
X = filtered_df.drop(columns=["IsBadBuy"])

# Display shapes for verification
print("Inputs (X) shape:", X.shape)
print("Target (y) shape:", y.shape)



from sklearn.model_selection import train_test_split

# Split dataset into training and testing sets
# 30% of the data will be used for testing, random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Assign training inputs for further preprocessing
inputs = X_train

# Optional: print shapes to confirm split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



columns = inputs.columns

# Choose categorical elements by index
# Update the indices based on your dataset (exclude the target, include actual categorical columns)
categorical_indices = [
    columns.get_loc(col) for col in [
        "Auction", "Make", 
        "Color",  "WheelType", "Nationality", "Size",
        "TopThreeAmericanName", "PRIMEUNIT", "AUCGUART", 'Transmission'
    ]
]

# Use a list comprehension to select the elements at the specified indices
categorical_fields = [columns[i] for i in categorical_indices]

# Create a new list of columns excluding categorical_fields (continuous)
continuous_fields = [j for j in columns if j not in categorical_fields]

# Optional: print for verification
print("Categorical fields:", categorical_fields)
print("Continuous fields:", continuous_fields)



import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Copy dataset
inputs_iso = inputs.copy()

# Remove rows with NaN values
inputs_iso = inputs_iso.dropna()

# Apply Z-score scaling to continuous columns
scaler = StandardScaler()
inputs_iso[continuous_fields] = scaler.fit_transform(inputs_iso[continuous_fields])

# Apply Label Encoding to categorical columns (column-wise)
for col in categorical_fields:
    le = LabelEncoder()
    inputs_iso[col] = le.fit_transform(inputs_iso[col])

# Fit Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(inputs_iso)

# Predict outliers (-1: outlier, 1: inlier)
outliers = clf.predict(inputs_iso)

# Add outlier predictions to DataFrame
inputs_iso['outlier'] = outliers

# Display the DataFrame
print(inputs_iso.head())

# Calculate percentage of outliers
percentage_outliers = (outliers[outliers == -1].shape[0] / len(outliers)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")



# Identify the indices of outliers
outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index

# Remove outlier rows from training inputs and target
inputs_outprep = inputs.drop(outlier_index)
y_train_outprep = y_train.drop(outlier_index)

# Recombine the cleaned inputs and target into a single DataFrame
train_outprep = pd.concat([inputs_outprep, y_train_outprep], axis=1)

# Optional: display the shape and first few rows
print("Cleaned training set shape:", train_outprep.shape)
print(train_outprep.head())



import pandas as pd

# Create a new column with the number of missing values in each row
train_outprep['Num_Missing_Values'] = train_outprep.isnull().sum(axis=1)

# Filter rows that have at least one missing value
rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] > 0]

# Calculate total rows, number of rows with missing values, and percentage
total_rows = len(train_outprep)
rows_with_missing_count = len(rows_with_missing_values)
percentage_rows_with_missing = (rows_with_missing_count / total_rows) * 100

# Display the summary report
print("Report on Rows with Missing Values:")
print(f"Total Rows: {total_rows}")
print(f"Rows with Missing Values: {rows_with_missing_count} ({percentage_rows_with_missing:.2f}%)")

# Display the DataFrame with the Num_Missing_Values column sorted descending
print("\nDataFrame with Num_Missing_Values column:")
print(train_outprep.sort_values(by='Num_Missing_Values', ascending=False).head(20))  # show top 20 rows with most missing



import pandas as pd

# Create a report of missing values for each column
missing_values_report = pd.DataFrame({
    'Column': train_outprep.columns,
    'Missing Values': train_outprep.isnull().sum(),
    'Percentage Missing': train_outprep.isnull().mean() * 100
})

# Sort the report by the percentage of missing values in descending order
missing_values_report = missing_values_report.sort_values(by='Percentage Missing', ascending=False).reset_index(drop=True)

# Display the missing values report
print("Missing Values Report:")
print(missing_values_report)



# List of price-related columns
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

# Count missing values across price-related columns
train_outprep['num_price_missing'] = train_outprep[price_columns].isnull().sum(axis=1)

# Discard rows with 4 or more missing values
train_outprep = train_outprep[train_outprep['num_price_missing'] < 4]

# Remove helper column
train_outprep = train_outprep.drop(columns=['num_price_missing'])

# Output results
print("Remaining rows:", train_outprep.shape[0])
print(train_outprep[price_columns].isnull().sum())



# Calculate the threshold: 50% of total number of columns
threshold = train_outprep.shape[1] * 0.5

# Count missing values per row
train_outprep['total_missing'] = train_outprep.isnull().sum(axis=1)

# Filter out rows with 50% or more missing
train_outprep = train_outprep[train_outprep['total_missing'] < threshold].copy()

# Drop the helper column
train_outprep.drop(columns=['total_missing'], inplace=True)

print("Remaining rows after removing those with â‰¥ 50% missing values:")
print(train_outprep.shape)



import pandas as pd

# Identify continuous and categorical fields
continuous_fields = [
    'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost'
]

categorical_fields = [
    'Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color',
    'Transmission', 'WheelType', 'Nationality', 'Size',
    'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART', 'VNST', 'IsOnlineSale'
]

# Impute continuous fields with median
for col in continuous_fields:
    if col in train_outprep.columns:
        median_value = train_outprep[col].median()
        train_outprep[col].fillna(median_value, inplace=True)

# Impute categorical fields with mode
for col in categorical_fields:
    if col in train_outprep.columns:
        mode_value = train_outprep[col].mode()[0]
        train_outprep[col].fillna(mode_value, inplace=True)

# Verify that no missing values remain
missing_values_report = pd.DataFrame({
    'Column': train_outprep.columns,
    'Missing Values': train_outprep.isnull().sum(),
    'Percentage Missing': train_outprep.isnull().mean() * 100
}).sort_values(by='Percentage Missing', ascending=False).reset_index(drop=True)

print("Missing Values Report After Imputation:")
print(missing_values_report)





train_outprep.info()



# Create a copy of the cleaned dataset for feature selection steps
train_FS = train_outprep.copy()

print("train_FS created successfully.")
print(train_FS.shape)
train_FS.head()



# 1) Descriptive statistics including skewness and kurtosis
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

# Replace this if your dataframe has different name
df = train_FS

# Define continuous fields (adjust if needed)
continuous_fields = [
    'VehicleAge', 'VehOdo',
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice',
    'VehBCost',
    'WarrantyCost'
]

# Keep only those that actually exist in dataframe (robust)
continuous_fields = [c for c in continuous_fields if c in df.columns]

# Build descriptive table
desc = df[continuous_fields].describe().T
desc['skew'] = df[continuous_fields].skew().values
desc['kurtosis'] = df[continuous_fields].apply(lambda x: kurtosis(x.dropna(), fisher=True)).values

# Show results sorted by absolute skew
desc = desc.sort_values(by='skew', key=lambda s: s.abs(), ascending=False)
print("Descriptive statistics (including skew & kurtosis):")
display(desc)



import matplotlib.pyplot as plt
import numpy as np

cols_to_plot = ["VehBCost", "WarrantyCost"]

for col in cols_to_plot:
    data = train_FS[col].dropna()
    
    plt.figure(figsize=(8,4))
    plt.hist(data, bins=40, alpha=0.6, density=True)
    
    # KDE-like smooth curve via numpy
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 300)
    plt.plot(x_vals, kde(x_vals), linewidth=2)
    
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(alpha=0.2)
    plt.show()





!pip install scorecardbundle




from scorecardbundle.feature_discretization import ChiMerge as cm
import numpy as np
import pandas as pd

# -------------------------------
# Variables to discretize
# -------------------------------
chi_merge_list = ['VehBCost', 'WarrantyCost']

# -------------------------------
# Apply Chi-Merge
# -------------------------------
trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=1, decimal=3, output_dataframe=True)

# Fit-transform on selected variables
result_cm = trans_cm.fit_transform(train_FS[chi_merge_list], train_FS['IsBadBuy'].astype(int))

# Extract boundaries
boundaries_dict = {key: np.insert(boundaries, 0, -np.inf)
                   for key, boundaries in trans_cm.boundaries_.items()}

# -------------------------------
# Add discretized variables to train_FS
# -------------------------------
for key, boundaries in boundaries_dict.items():
    column_name = f"{key}_cat_cm"
    
    # Create discrete categories using cut
    train_FS[column_name] = pd.cut(
        train_FS[key],
        bins=boundaries,
        labels=False,
        right=False
    )
    
    # Print bin edges
    print(f'{column_name} bin edges:', boundaries)
    
    # Print frequency table
    print(train_FS[column_name].value_counts().sort_index())
    print("\n")

# Display the updated DataFrame
print(train_FS.head())

train_FS.describe()



# Remove the original continuous variables after Chi-Merge
train_FS.drop(['VehBCost', 'WarrantyCost'], axis=1, inplace=True)

print("Remaining columns:", len(train_FS.columns))
train_FS.head()




train_FS.columns



columns_to_drop = [
    "VehBCost", "WarrantyCost",
    "Num_Missing_Values",
    "PRIMEUNIT", "AUCGUART"
]

# Drop only columns that actually exist
cols_present = [c for c in columns_to_drop if c in train_FS.columns]

train_FS = train_FS.drop(columns=cols_present)

print(train_FS.columns.tolist())




# Apply One-Hot Encoding for nominal fields
from sklearn.preprocessing import OneHotEncoder

# List of nominal fields to encode
nominal_fields = ["Auction", "Make", "Color", "Transmission", 
                  "WheelType", "Nationality", "Size", "TopThreeAmericanName"]

# Initialize the OneHotEncoder
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Fit and transform the nominal fields
one_hot_encoded = one_hot_encoder.fit_transform(train_FS[nominal_fields])

# Create a DataFrame from the encoded results
one_hot_encoded_df = pd.DataFrame(
    one_hot_encoded, 
    columns=one_hot_encoder.get_feature_names_out(nominal_fields)
)

# Concatenate the encoded columns with the original dataframe
train_FS_encoded = pd.concat(
    [train_FS.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], 
    axis=1
)

# Drop the original nominal fields from the dataframe
train_FS_encoded = train_FS_encoded.drop(columns=nominal_fields)

# Display the updated dataframe shape
print("Shape after One-Hot Encoding:", train_FS_encoded.shape)
train_FS_encoded.head()



from sklearn.preprocessing import MinMaxScaler

# Separate the target
target = 'IsBadBuy'
features = train_FS_encoded.drop(columns=[target])
y = train_FS_encoded[target]

# Initialize Min-Max Scaler
scaler = MinMaxScaler()

# Fit and transform the features
scaled_features = scaler.fit_transform(features)

# Convert back to DataFrame
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

# Concatenate target back
train_FS_scaled = pd.concat([scaled_features_df, y.reset_index(drop=True)], axis=1)

# Display shape and first few rows
print("Shape after Min-Max Scaling:", train_FS_scaled.shape)
train_FS_scaled.head()



# Create a copy of the cleaned dataset for feature selection steps
train_FE = train_outprep.copy()

print("train_FE created successfully.")
print(train_FE.shape)
train_FE.head()



# Create a copy of the cleaned dataset for feature engineering
train_FE = train_outprep.copy()

print("train_FE created successfully.")
print(train_FE.shape)
train_FE.head() 

import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

# Features to transform
selected_features = ['VehBCost', 'WarrantyCost']

# Iterate through selected features
for feature in selected_features:
    # Check if the feature contains negative or zero values
    has_negative_values = (train_FE[feature] <= 0).any()

    # Choose transformation method
    if has_negative_values:
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    else:
        transformer = PowerTransformer(method='box-cox', standardize=False)

    # Fit and transform, store in new column
    train_FE[f"{feature}_transformed"] = transformer.fit_transform(train_FE[[feature]])

    # Get lambda
    lambda_value = transformer.lambdas_[0]
    print(f"Lambda for {feature}: {lambda_value:.4f}")
    
    # Plot histograms for original and transformed
    plt.figure(figsize=(8, 3))
    
    plt.subplot(1, 2, 1)
    plt.hist(train_FE[feature], bins=30, color='blue', alpha=0.7)
    plt.title(f'Original {feature} Histogram')

    plt.subplot(1, 2, 2)
    plt.hist(train_FE[f"{feature}_transformed"], bins=30, color='green', alpha=0.7)
    plt.title(f'Transformed {feature} Histogram')

    plt.tight_layout()
    plt.show()



columns_to_drop = [
  "RefId" , "VehBCost", "WarrantyCost",
    "Num_Missing_Values",
    "PRIMEUNIT", "AUCGUART"
]

# Drop only columns that actually exist
cols_present = [c for c in columns_to_drop if c in train_FE.columns]

train_FE = train_FE.drop(columns=cols_present)

print(train_FE.columns.tolist())
train_FE.shape


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Nominal fields to encode
nominal_fields = ["Auction", "Make", "Color", "Transmission", "WheelType", 
                  "Nationality", "Size", "TopThreeAmericanName"]

# Initialize One-Hot Encoder
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Fit and transform the nominal columns
one_hot_encoded = one_hot_encoder.fit_transform(train_FE[nominal_fields])

# Convert to DataFrame with proper column names
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, 
                                  columns=one_hot_encoder.get_feature_names_out(nominal_fields))

# Concatenate with the original DataFrame
train_FE_encoded = pd.concat([train_FE.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)

# Drop original nominal columns
train_FE_encoded.drop(columns=nominal_fields, inplace=True)

# Display shape to verify 62 fields
print("Shape after One-Hot Encoding:", train_FE_encoded.shape)
train_FE_encoded.head()



from sklearn.preprocessing import StandardScaler
import pandas as pd

# Separate the target
target = 'IsBadBuy'
features = train_FE_encoded.drop(columns=[target])
y = train_FE_encoded[target]

# Initialize StandardScaler (z-score)
scaler = StandardScaler()

# Fit and transform the features
scaled_features = scaler.fit_transform(features)

# Convert back to DataFrame
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

# Concatenate target back
train_FE_scaled = pd.concat([scaled_features_df, y.reset_index(drop=True)], axis=1)

# Display shape and first few rows
print("Shape after Z-Score Scaling:", train_FE_scaled.shape)
train_FS_scaled.head()



import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------
# Separate inputs and target
# -------------------------------
X_train = train_FS.drop(columns=['IsBadBuy'])
y_train = train_FS['IsBadBuy']

# -------------------------------
# Define nominal and other fields
# -------------------------------
nominal = [
    "Auction", "Make", "Color", "Transmission",
    "WheelType", "Nationality", "Size", "TopThreeAmericanName"
]

other = [col for col in X_train.columns if col not in nominal]

# -------------------------------
# One-Hot Encoding for nominal fields
# -------------------------------
one_hot_encoder = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False
)

one_hot_encoded = one_hot_encoder.fit_transform(X_train[nominal])

one_hot_encoded_df = pd.DataFrame(
    one_hot_encoded,
    columns=one_hot_encoder.get_feature_names_out(nominal),
    index=X_train.index
)

# -------------------------------
# Combine encoded + numeric data
# -------------------------------
X_train_encoded = pd.concat(
    [X_train[other], one_hot_encoded_df],
    axis=1
)

# -------------------------------
# Z-score scaling (ALL except target)
# -------------------------------
scaler = StandardScaler()
X_train_encoded[X_train_encoded.columns] = scaler.fit_transform(X_train_encoded)

# Final dataset for wrapper FS
X_train_wrapper_fs = X_train_encoded.copy()

print("Shape after encoding & scaling:", X_train_wrapper_fs.shape)





 !pip install --upgrade scikit-learn




from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------
# Configure RFECV with Classifier
# ----------------------------------
selector_f1 = RFECV(
    estimator=DecisionTreeClassifier(random_state=29),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1,
    scoring='f1'   # F-measure for positive class (IsBadBuy = 1)
)

# ----------------------------------
# Fit selector
# ----------------------------------
selector_f1.fit(X_train_encoded, y_train)

# ----------------------------------
# Results
# ----------------------------------
print(f"Optimal number of features (F1-score): {selector_f1.n_features_}")
print("=" * 50)

wrapper_fs_f1 = selector_f1.get_feature_names_out()
print("Wrapper Optimal Feature List (F1-score, Class = 1):")
print(wrapper_fs_f1)

# ----------------------------------
# Reduced feature set
# ----------------------------------
X_train_wrapper_fs = X_train_encoded[wrapper_fs_f1]



X_train_wrapper_fs.to_csv("X_train_wrapper_fs_classifer_f1.csv")
print("X_train_wrapper_fs_classifer_f1.csv")


train_FE_scaled = train_FE_scaled.rename(columns={'Unnamed: 0': 'Id'})
train_FE_scaled.info()


# continuous features = true numeric (non one-hot) variables
continuous_cols = [
    'VehicleAge',
    'VehOdo',
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice',
    'IsOnlineSale',
    'VehBCost_transformed',
    'WarrantyCost_transformed'
]

train_FE_scaled_continuous = train_FE_scaled[continuous_cols].copy()



import matplotlib.pyplot as plt
import seaborn as sns

# compute Pearson correlation matrix
corr_matrix = train_FE_scaled_continuous.corr(method="pearson")

# plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Pearson Correlation Heatmap (Continuous Features)")
plt.tight_layout()
plt.show()



from sklearn.decomposition import PCA
import pandas as pd

# Perform PCA on continuous scaled features
pca = PCA(n_components=2, random_state=717)
pca.fit(train_FE_scaled_continuous)

pc_name = pd.DataFrame(
    [f'pc_{i+1}' for i in range(pca.n_components_)],
    columns=['name']
)

variance = pd.DataFrame(
    pca.explained_variance_,
    columns=['variance']
)

variance_ratio = pd.DataFrame(
    pca.explained_variance_ratio_,
    columns=['variance_ratio']
)

print(variance_ratio)

total_explained_variance = variance_ratio.sum()

component_weights = pd.DataFrame(
    pca.components_,
    columns=train_FE_scaled_continuous.columns
)

print(component_weights)

pca_report = pd.concat(
    (pc_name, variance, variance_ratio, component_weights),
    axis=1
).set_index('name')

print(pca_report)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Make sure Y_train exists
Y_train = train_FE_scaled['IsBadBuy']

# PCA transform (use the same data PCA was fitted on)
pca_X_train = pca.transform(train_FE_scaled_continuous)
pca_X_train = pd.DataFrame(
    pca_X_train,
    columns=pc_name['name'].tolist(),
    index=train_FE_scaled_continuous.index
)

# number of selected components
k = 2
pca_train = pd.concat(
    [Y_train.reset_index(drop=True), pca_X_train.iloc[:, :k]],
    axis=1
)

# color palette for two classes
palette_colors = ["#87CEEB", "#FF00FF"]

# pairplot of first k PCA components vs target
sns.pairplot(pca_train, hue='IsBadBuy', palette=palette_colors)
plt.show()



import pandas as pd

# ===============================
# 1. Apply PCA on continuous features
# ===============================
pca_X_train = pca.transform(train_FE_scaled_continuous)
pca_X_train = pd.DataFrame(
    pca_X_train,
    columns=pc_name['name'].tolist(),
    index=train_FE_scaled_continuous.index
)

# ===============================
# 2. Drop original continuous features from the dataset
# ===============================
train_FE_non_continuous = train_FE_scaled.drop(columns=train_FE_scaled_continuous.columns)

# ===============================
# 3. Concatenate non-continuous features with PCA components
# ===============================
train_pca_fe = pd.concat([train_FE_non_continuous, pca_X_train], axis=1)

# ===============================
# 4. Check final shape and preview
# ===============================
print("Final shape of train_pca_fe:", train_pca_fe.shape)
train_pca_fe.head()



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# ===============================
# 1. Fit the LDA model
# ===============================
lda = LinearDiscriminantAnalysis(n_components=1)  # for 2 classes, max n_components = 1
lda.fit(train_FE_scaled_continuous, Y_train)      # use continuous features

# ===============================
# 2. Plot the explained variance ratio
# ===============================
plt.figure(figsize=(6,4))
plt.plot(lda.explained_variance_ratio_, marker='o')
plt.title("LDA Component and Explained Variance Ratio")
plt.xlabel("LDA component")
plt.ylabel("Variance ratio")
plt.grid(True)
plt.show()

# ===============================
# 3. Transform the data to LDA components
# ===============================
lda_X_train = lda.transform(train_FE_scaled_continuous)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Make sure Y_train has a name
Y_train = Y_train.rename('IsBadBuy')

# Fit LDA on continuous features
lda = LinearDiscriminantAnalysis(n_components=1)
lda_X_train = lda.fit_transform(train_FE_scaled_continuous, Y_train)

# Convert LDA output to DataFrame
lda_X_train = pd.DataFrame(lda_X_train, columns=['lda_1'], index=train_FE_scaled_continuous.index)

# Concatenate label with LDA component
lda_train = pd.concat([Y_train.reset_index(drop=True), lda_X_train], axis=1)

# Plot pairplot
sns.pairplot(lda_train, hue='IsBadBuy', palette=["#87CEEB", "#FF00FF"])
plt.show()





