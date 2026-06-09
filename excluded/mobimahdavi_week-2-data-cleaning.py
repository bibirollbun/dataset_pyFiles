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
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


df_new=df.drop(columns=['PurchDate','VehYear','PurchDate','Model', 'Trim' , 'SubModel','WheelTypeID','BYRNO', 'VNZIP1','VNST'])


df_new.set_index('RefId', inplace=True)


y = df_new.iloc[:,0]
X = df_new.iloc[:,1::]


from sklearn.model_selection import train_test_split

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

inputs = X_train


columns = inputs.columns

# Choose categorical elements 
categorical_indices = [0,2,3,4,5,7,8,9,18,19,21]

# Use a list comprehension to select the elements at the specified indices
categorical_fields = [columns[i] for i in categorical_indices]

# Create a new list of columns excluding categorical_fields (continuous)
continuous_fields = [j for j in columns if j not in categorical_fields]


import pandas as pd

# Define ranges for each column
column_ranges ={
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


inputs['Transmission'] = inputs['Transmission'].replace(['Manual'], 'MANUAL')


frequency_table(inputs['Transmission'])


frequency_table(inputs['Color'])


inputs['Color'].replace('NOT AVAIL', np.nan, inplace=True)


frequency_table(inputs['Color'])


def group_infrequent(series, threshold=0.01):
    freq = series.value_counts(normalize=True)  # Get frequency of each class
    mask = series.isin(freq[freq >= threshold].index)
    return series.where(mask, 'OTHER')  # Replace infrequent classes with 'OTHER'

# Grouping 'Color' and 'Make'
inputs['Color'] = group_infrequent(inputs['Color'],threshold=0.01)
inputs['Make'] = group_infrequent(inputs['Make'],threshold=0.01)


frequency_table(inputs['Color'])


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


import pandas as pd

# Set a threshold for excluding columns 
threshold = 90

# Calculate the percentage of distinct categories in categorical variables
distinct_percentage = (inputs_cat.apply(lambda x: x.dropna().nunique() / x.count()) * 100)

# Select categorical columns based on distinct percentage threshold
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index

# Create a new DataFrame with only the selected columns
distinct_filtered_inputs = inputs_cat[selected_categorical_columns]

# Filter out selected columns and print the resulting DataFrame
inputs_cat = inputs_cat.drop(selected_categorical_columns, axis=1)
print(inputs_cat)


filtered_df = pd.concat([inputs_con, inputs_cat, y_train], axis=1)



import pandas as pd
from scipy.stats import chi2_contingency


# Step 1: Filter non-null values
filter_df = filtered_df.dropna(subset=['PRIMEUNIT', 'IsBadBuy'])

# Step 2: Create a contingency table
contingency_table = pd.crosstab(
    filtered_df['PRIMEUNIT'],
    filtered_df['IsBadBuy']
)

# Step 3: Conduct the Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Step 4: Display the results
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: There is a significant association between 'PRIMEUNIT' and 'IsBadBuy'.")
else:
    print("Fail to reject the null hypothesis: No significant association between 'PRIMEUNIT' and 'IsBadBuy'.")



import pandas as pd
from scipy.stats import chi2_contingency


# Step 1: Filter non-null values
filter_df = filtered_df.dropna(subset=['AUCGUART', 'IsBadBuy'])

# Step 2: Create a contingency table
contingency_table = pd.crosstab(
    filtered_df['AUCGUART'],
    filtered_df['IsBadBuy']
)

# Step 3: Conduct the Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Step 4: Display the results
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: There is a significant association between 'AUCGUART' and 'IsBadBuy'.")
else:
    print("Fail to reject the null hypothesis: No significant association between 'AUCGUART' and 'IsBadBuy'.")



filter2_df=filtered_df.drop(columns=['AUCGUART','PRIMEUNIT'])




columns = filter2_df.columns
categorical_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


categorical_fielded = columns[categorical_indices].tolist()


categorical_set = set(categorical_fielded)


continuous_fielded = [j for j in columns if j not in categorical_set]



import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

inputs_iso = filter2_df.copy()

# Discard rows with NaN valuse
inputs_iso = inputs_iso.dropna()

# Apply Z-score scaling to numerical columns
scaler = StandardScaler()
inputs_iso[continuous_fielded] = scaler.fit_transform(inputs_iso[continuous_fielded])

# Apply label encoding to categorical columns
label_encoder = LabelEncoder()
inputs_iso[categorical_fielded] = inputs_iso[categorical_fielded].apply(label_encoder.fit_transform)

# Fit Isolation Forest model
clf = IsolationForest(contamination='auto', random_state=42)
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
inputs_outprep = filter2_df.drop(outlier_index)



import pandas as pd

# Copy the dataset
df = inputs_outprep.copy()

# ---------------------------
# 1. Discard rows with 4 or more nulls in price-related columns
# ---------------------------
price_cols = [
    'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice'
]

df = df[df[price_cols].isnull().sum(axis=1) < 4]

# ---------------------------
# 2. Discard rows with 50% or more nulls across all columns
# ---------------------------
df = df[df.isnull().mean(axis=1) < 0.5]

# ---------------------------
# 3. Impute remaining missing values
# Continuous fields: median
# Categorical fields: mode
# ---------------------------
# Continuous fields
continuous_fields_final = [
    'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost'
]

for col in continuous_fields_final:
    df[col] = df[col].fillna(df[col].median())

# Categorical fields
categorical_fields_final = [
    'Auction', 'Make', 'Color', 'Transmission', 'WheelType',
    'Nationality', 'Size', 'TopThreeAmericanName', 'IsOnlineSale'
]

for col in categorical_fields_final:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---------------------------
# Summary
# ---------------------------
print("Shape after handling missing values:", df.shape)
print("Remaining missing values:\n", df.isnull().sum())


df.to_excel("Carvana_train_cleaned.xlsx", index=True)
print("Final cleaned dataset saved to Carvana_train_cleaned.xlsx")

