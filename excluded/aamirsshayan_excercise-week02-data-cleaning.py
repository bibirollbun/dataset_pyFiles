import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.info()


df = df.drop(columns=['PurchDate', 'VehYear', 'Model', 'Trim', 'SubModel', 'WheelTypeID', 'BYRNO', 'VNZIP1', 'VNST'])
df.set_index('RefId', inplace=True)
df.info()



from sklearn.model_selection import train_test_split

# Define target and features
y = df['IsBadBuy']
X = df.drop(columns='IsBadBuy')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# Optional: continue working with X_train under another name
inputs = X_train
inputs.info()


import pandas as pd
import numpy as np


# Define ranges for each column
column_ranges = {
'VehicleAge': (0,30) ,
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

# Iterate through each column and fill NaN values outside the defined range
for column, (min_val, max_val) in column_ranges.items():
   inputs[column] = inputs[column].apply(lambda x: x if min_val <= x <= max_val else np.nan)

# Display the updated DataFrame
print(inputs)
inputs.describe()
inputs.info()


(inputs['Make'].value_counts(normalize=True) * 100).round(2).astype(str) + '%'



for col in ['Color', 'Make']:
    freq = inputs[col].value_counts(normalize=True)     
    rare_classes = freq[freq < 0.01].index         
    inputs[col] = inputs[col].apply(lambda x: 'OTHER' if x in rare_classes else x)

print(inputs['Color'].value_counts())
print(inputs['Make'].value_counts())
inputs.info()


continuous_cols = [
    'VehicleAge', 'VehOdo',
    'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice',
    'VehBCost', 'WarrantyCost'
]



# Use a list comprehension to select the elements at the specified indices
categorical_cols = [col for col in inputs.columns if col not in continuous_cols]

print("Continuous columns:", continuous_cols)
print("Categorical columns:", categorical_cols)




# Define a minimum value for coefficient of variation
min_cv = 0.1

# Calculate the coefficient of variation for each column
cv_values = inputs[continuous_cols].std() / inputs[continuous_cols].mean()

# Filter out columns with CV less than 0.1
selected_columns =  cv_values[cv_values < min_cv].index



# Print the resulting DataFrame
inputs_filtered_continuous = inputs.drop(columns=selected_columns)
inputs_filtered_continuous.info()  



 import pandas as pd

# Define a threshold for the dominant category percentage
threshold = 99

# Calculate the percentage of the mode category for each column
mode_category = (inputs_filtered_continuous[categorical_cols].apply(lambda x: x.value_counts().max() / len(x)) * 100)

# Select columns where the mode category percentage is greater than the threshold
selected_categorical_columns = mode_category[mode_category > threshold].index

# Filter out selected columns and print the resulting DataFrame
inputs_cat_mode = inputs_filtered_continuous.drop(columns=selected_categorical_columns)
inputs_cat_mode.info()


# Set a threshold for excluding columns 
threshold = 90

# Calculate the percentage of distinct categories in categorical variables
distinct_percentage = (inputs_cat_mode.apply(lambda x: x.dropna().nunique() / x.count()) * 100)

# Select categorical columns based on distinct percentage threshold
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index


# Filter out selected columns and print the resulting DataFrame
inputs_filtered = inputs_cat_mode.drop(columns=selected_categorical_columns)

inputs_filtered.info()


filtered_df = pd.concat([inputs_filtered, y_train], axis=1)



import pandas as pd
from scipy.stats import chi2_contingency

# Delete Null Values for PRIMEUNIT and AUCGUART
subset_primeunit = filtered_df[['PRIMEUNIT', 'IsBadBuy']].dropna()
subset_aucguart = filtered_df[['AUCGUART', 'IsBadBuy']].dropna()

# Create contingency tables
contingency_table_primeunit = pd.crosstab(subset_primeunit['PRIMEUNIT'], subset_primeunit['IsBadBuy'])
contingency_table_aucguart = pd.crosstab(subset_aucguart['AUCGUART'], subset_aucguart['IsBadBuy'])

# Perform chi-square tests for both PRIMEUNIT and AUCGUART
def perform_chi_square(contingency_table):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof, expected

# Test for PRIMEUNIT
chi2_primeunit, p_primeunit, dof_primeunit, expected_primeunit = perform_chi_square(contingency_table_primeunit)

# Test for AUCGUART
chi2_aucguart, p_aucguart, dof_aucguart, expected_aucguart = perform_chi_square(contingency_table_aucguart)

# Output Results
print("Chi-square for PRIMEUNIT:")
print(f"Chi-squared value: {chi2_primeunit}")
print(f"P-value: {p_primeunit}")
print(f"Degrees of freedom: {dof_primeunit}")
print("#" * 60)

print("Chi-square for AUCGUART:")
print(f"Chi-squared value: {chi2_aucguart}")
print(f"P-value: {p_aucguart}")
print(f"Degrees of freedom: {dof_aucguart}")
print("#" * 60)

# Check if p-value < 0.05 for significance
if p_primeunit < 0.05:
    filtered_df['PRIMEUNIT'].fillna('unknown', inplace=True)
else:
    filtered_df.drop(columns=['PRIMEUNIT'], inplace=True)

if p_aucguart < 0.05:
    filtered_df['AUCGUART'].fillna('unknown', inplace=True)
else:
    filtered_df.drop(columns=['AUCGUART'], inplace=True)



filtered_df.info()


continuous_cols = [
    'VehicleAge', 'VehOdo',
    'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice',
    'VehBCost', 'WarrantyCost'
]



# Use a list comprehension to select the elements at the specified indices
categorical_cols = [col for col in filtered_df.columns if col not in continuous_cols]
print(categorical_cols)
print(continuous_cols)



from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Drop NaN rows for isolation process
inputs_iso = filtered_df.dropna().copy()

# Scale continuous features
scaler = StandardScaler()
inputs_iso[continuous_cols] = scaler.fit_transform(inputs_iso[continuous_cols])
for col in categorical_cols:
    inputs_iso[col] = LabelEncoder().fit_transform(inputs_iso[col])

# Apply Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
inputs_iso['outlier'] = clf.fit_predict(inputs_iso)

# Print percentage of outliers
outlier_ratio = (inputs_iso['outlier'] == -1).mean() * 100
print(f"Percentage of outliers: {outlier_ratio:.2f}%")

# Remove outliers
outlier_indices = inputs_iso[inputs_iso['outlier'] == -1].index

# Update filtered_df
filtered_df_cleaned = filtered_df.drop(index=outlier_indices)

filtered_df_cleaned.info()



import pandas as pd

# لیست فیلدهای مورد نظر
mmr_fields = [
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice'
]


filtered_df_cleaned['MMR_Missing_Count'] = filtered_df_cleaned[mmr_fields].isnull().sum(axis=1)


total_rows = len(filtered_df_cleaned)
rows_with_mmr_missing = filtered_df_cleaned[filtered_df_cleaned['MMR_Missing_Count'] >= 4]
rows_with_mmr_missing_count = len(rows_with_mmr_missing)
percentage_mmr_missing = (rows_with_mmr_missing_count / total_rows) * 100

print("Report on Rows with 4+ Nulls in MMR Fields:")
print(f"Total Rows: {total_rows}")
print(f"Rows with 4 or more MMR nulls: {rows_with_mmr_missing_count} ({percentage_mmr_missing:.2f}%)")


filtered_df_mmr= filtered_df_cleaned[filtered_df_cleaned['MMR_Missing_Count'] < 4].iloc[:, :-1]


import pandas as pd

# محاسبه تعداد ستون‌ها
total_columns = filtered_df_mmr.shape[1]

# ساخت ستون کمکی برای شمارش تعداد null در هر ردیف
filtered_df_mmr['Num_Missing_Values'] = filtered_df_mmr.isnull().sum(axis=1)

# تعیین آستانه: بیش از 50٪ از کل ستون‌ها
missing_threshold = total_columns * 0.5

# گزارش آماری
rows_over_threshold = filtered_df_mmr[filtered_df_mmr['Num_Missing_Values'] > missing_threshold]
num_rows_over_threshold = len(rows_over_threshold)
percentage_over_threshold = (num_rows_over_threshold / len(filtered_df_mmr)) * 100

print("Report on Rows with >=50% Missing Values:")
print(f"Total Rows: {len(filtered_df)}")
print(f"Rows with >50% Missing Values: {num_rows_over_threshold} ({percentage_over_threshold:.2f}%)")

# حذف ردیف‌هایی که بیشتر از 50٪ ویژگی‌هاشون null هست
filtered_df_mmr1 = filtered_df_mmr[filtered_df_mmr['Num_Missing_Values'] < missing_threshold].iloc[:, :-1]

filtered_df_mmr1.info()


# Continuous fields (اعداد اعشاری که میانه‌گیری می‌شن)
continuous_cols = [
    'VehOdo',
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

# Categorical fields (متغیرهای اسمی که مد روی آن‌ها اعمال می‌شود)
categorical_cols = [
    'Auction', 'Make', 'Color', 'Transmission',
    'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName'
]

# اعمال median برای متغیرهای عددی
for col in continuous_cols:
    median_val = filtered_df_mmr1[col].median()
    filtered_df_mmr1[col] = filtered_df_mmr1[col].fillna(median_val)

# اعمال mode برای متغیرهای اسمی
for col in categorical_cols:
    mode_val = filtered_df_mmr1[col].mode()[0]
    filtered_df_mmr1[col] = filtered_df_mmr1[col].fillna(mode_val)



print("Remaining missing values per column:")
print(filtered_df_mmr1.isnull().sum())



train_outprep_no_missing_fix = filtered_df_mmr1.copy()
train_outprep_no_missing_fix.to_csv("Train_Clean.csv", index=True)


