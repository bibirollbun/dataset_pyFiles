import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df


# Change some features type to object

df['RefId'] = df['RefId'].astype('object')
df['IsBadBuy'] = df['IsBadBuy'].astype('object')
df['WheelTypeID'] = df['WheelTypeID'].astype('object')
df['IsOnlineSale'] = df['IsOnlineSale'].astype('object')
df['BYRNO'] = df['BYRNO'].astype('object')
df['VNZIP1'] = df['VNZIP1'].astype('object')

df.info()


column_to_move = df.pop('IsBadBuy')
df.insert(len(df.columns), 'IsBadBuy', column_to_move)
df_logical_incons = df.copy()
df_logical_incons.describe()


# Define the valid (min, max) range for each column of interest 
column_ranges = {
    'VehYear': (2001, 2010),
    'VehicleAge': (0, 9),
    'VehOdo': (4825, 115717),
    'MMRAcquisitionAuctionAveragePrice': (884, 35722),
    'MMRAcquisitionAuctionCleanPrice': (1076, 36859),
    'MMRAcquisitionRetailAveragePrice': (1455, 39080),
    'MMRAcquisitonRetailCleanPrice': (1662, 41482),
    'MMRCurrentAuctionAveragePrice': (369, 35722),
    'MMRCurrentAuctionCleanPrice': (494, 36859),
    'MMRCurrentRetailAveragePrice': (899, 39080),
    'MMRCurrentRetailCleanPrice': (1034, 41062),
    'VehBCost': (225, 45469),
    'WarrantyCost': (462, 7498),
}

# For every column in column_ranges, the .apply() function is used to check each value
for column, (min_val, max_val) in column_ranges.items():
    df_logical_incons[column] = df_logical_incons[column].apply(lambda x: x if min_val <= x <= max_val else None)


#print the total number of rows in the original DataFrame.
print(len(df))
df_logical_incons.describe()



object_columns = df.select_dtypes(include=['object']).columns.tolist()
df_logical_incons[object_columns].head()


import numpy as np

def frequency_table(variable):
    
    # Get unique elements and their counts
    unique_elements, counts = np.unique(variable.dropna(), return_counts=True)

    if len(unique_elements) <= 16:
        # Calculate percentages
        percentages = (counts / len(variable)) * 100

        # Create a dictionary to store the value counts and percentages
        value_counts_and_percentages = zip(unique_elements, counts, percentages)

        # Print the value counts and percentages
        for i, j, k in value_counts_and_percentages:
            print(f"\t{i}: Count: {j}, Percentage: {k:.2f}%")
            
    else:
        print(f'\tnumber of classes :{len(unique_elements)}')
        print('\tLarge number of classes.')
    return


for cat in object_columns:
    print(f"\nFrequency table of {cat} :\n")
    frequency_table(df_logical_incons[cat])
    


#Cleaning the Transmission column,checking how often each type of transmission appears using a frequency table

df_logical_incons['Transmission'] = df_logical_incons['Transmission'].replace(['Manual'], 'MANUAL')
frequency_table(df_logical_incons['Transmission'])


#spliting the DataFrame df_logical_incons into two parts: features (inputs) and the target (label),
target = df_logical_incons.iloc[:,-1]
inputs = df_logical_incons.iloc[:,0:-1]


# organizing our features into categorical and continuous (numerical) types
#a super important step before preprocessing or feeding data into many machine learning models.



columns = inputs.columns

 
cat_features = ['RefId', 'PurchDate', 'Auction', 'Make', 'Model', 'Trim', 'SubModel','Color','Transmission', 'WheelTypeID', 'WheelType', 
                'Nationality', 'Size', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'VNZIP1', 'VNST', 'IsOnlineSale']

#Their positions in the DataFrame
categorical_indices = [inputs.columns.get_loc(feature) for feature in cat_features]

#List of all categorical column names

categorical_fields = [columns[i] for i in categorical_indices]

#All remaining columns assumed to be numerical

continuous_fields = [j for j in columns if j not in categorical_fields]


#min_cv stands for minimum coefficient of variation.
min_cv = 0.1

#Calculate the Coefficient of Variation
cv_values = inputs[continuous_fields].std() / inputs[continuous_fields].mean()

#Select columns with CV < 0.1
selected_columns =  cv_values[cv_values < 0.1].index

#Store dropped columns
filtered_con = inputs[selected_columns]

#From our original continuous features, remove the ones listed in selected_columns.
inputs_con = inputs[continuous_fields].drop(selected_columns, axis=1)

print(inputs_con.columns)
print("\nRemoved features:\n\t", selected_columns)


import pandas as pd

# Define a threshold for the dominant category percentage
threshold = 95

# Calculate the percentage of the mode category for each column
mode_category = (inputs[categorical_fields].apply(lambda x: x.value_counts().max() / len(x)) * 100)

# Select columns where the mode category percentage is greater than the threshold
selected_categorical_columns = mode_category[mode_category > threshold].index

# Create a new DataFrame with only the selected columns
mode_filtered_inputs = inputs[selected_categorical_columns]

# Filter out selected columns and print the resulting DataFrame
inputs_cat1 = inputs[categorical_fields].drop(selected_categorical_columns, axis=1)

print(inputs_cat1.columns)
print("\nRemoved features:\n\t", selected_categorical_columns)

for item in selected_categorical_columns:
    categorical_fields.remove(item)


import pandas as pd

# Set a threshold for excluding columns 
threshold = 90

# Calculate the percentage of distinct categories in categorical variables
distinct_percentage = (inputs[categorical_fields].apply(lambda x: x.dropna().nunique() / x.count()) * 100)

# Select categorical columns based on distinct percentage threshold
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index

# Create a new DataFrame with only the selected columns
distinct_filtered_inputs = inputs[selected_categorical_columns]

# Filter out selected columns and print the resulting DataFrame
inputs_cat2 = inputs_cat1.drop(selected_categorical_columns, axis=1)
print(inputs_cat2.columns)
print("\nRemoved features:\n\t", selected_categorical_columns)

for item in selected_categorical_columns:
    categorical_fields.remove(item)
    


#exploring categorical feature distributions
for cat in categorical_fields:
    print(f"\nFrequency table of {cat} :\n")
    frequency_table(inputs_cat2[cat])


inputs_cat2


# drop features whith lot of class
inputs_cat2 = inputs_cat2.drop(['PurchDate', 'Make', 'Model', 'Trim', 'SubModel', 'WheelTypeID'], axis=1)

# drop Irrelevant features
inputs_cat2 = inputs_cat2.drop(['BYRNO', 'VNZIP1', 'VNST'], axis=1)

inputs_cat2


#performing a data cleaning step to standardize certain categorical values in the Color column of inputs_cat2 DataFrame.
inputs_cat2['Color'] = inputs_cat2['Color'].replace(['BEIGE','BROWN', 'GREEN', 'MAROON', 'NOT AVAIL',
                                                                 'ORANGE', 'PURPLE', 'YELLOW'], 'OTHER')
frequency_table(inputs_cat2['Color'])


#performing a similar data cleaning operation on the Nationality column of the inputs_cat2 DataFrame.
inputs_cat2['Nationality'] = inputs_cat2['Nationality'].replace(['OTHER ASIAN', 'TOP LINE ASIAN'], 'OTHER')
frequency_table(inputs_cat2['Nationality'])


#This block of code is designed to simplify and standardize the values in the Size column of the inputs_cat2 DataFrame. The goal is to group similar categories into broader categories to make the dataset cleaner and more manageable
inputs_cat2['Size'] = inputs_cat2['Size'].replace(['LARGE SUV', 'MEDIUM SUV', 'SMALL SUV'], 'SUV')
inputs_cat2['Size'] = inputs_cat2['Size'].replace(['SPORTS', 'SPECIALTY'], 'SPORTS&SPECIALTY')
inputs_cat2['Size'] = inputs_cat2['Size'].replace(['LARGE TRUCK', 'SMALL TRUCK'], 'TRUCK')

frequency_table(inputs_cat2['Size'])


#combining different DataFrames (or parts of DataFrames) into a single DataFrame called df_logical_incons_filtered
df_logical_incons_filtered = pd.concat([inputs_con, inputs_cat2, target], axis=1)
df_logical_incons_filtered


df_logical_incons_filtered.to_csv('/kaggle/working/dontgetkicked_v1.csv')


import pandas as pd
df_logical_incons_filtered = pd.read_csv('/kaggle/working/dontgetkicked_v1.csv')
df_logical_incons_filtered = df_logical_incons_filtered.drop(['Unnamed: 0'], axis=1)


y = df_logical_incons_filtered.iloc[:,-1]
X = df_logical_incons_filtered.iloc[:,0:-1]


from sklearn.model_selection import train_test_split

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# combning the training features and target into a single DataFrame which is called train
train = pd.concat([X_train, y_train], axis=1)


#generating a report on missing values in the training dataset (train),
#and also adds a new column to show how many missing values are in each row.

import pandas as pd

#Creates a DataFrame of the same shape as train, with True for missing (NaN) values and False otherwise
train['Num_Missing_Values'] = train.isnull().sum(axis=1)

#Filtering and selects only the rows where at least one missing value exists.
rows_with_missing_values = train[train['Num_Missing_Values'] > 0]

total_rows = len(train)
rows_with_missing_count = len(rows_with_missing_values)
percentage_rows_with_missing = (rows_with_missing_count / total_rows) * 100

print("Report on Rows with Missing Values:")
print(f"Total Rows: {total_rows}")
print(f"Rows with Missing Values: {rows_with_missing_count} ({percentage_rows_with_missing:.2f}%)")


print("\nDataFrame with Num_Missing_Values column:")
print(train.sort_values(by='Num_Missing_Values', ascending = False))


#  a clean version of train with zero NaNs.
train_no_missing = train.dropna()

# Define the threshold for maximum allowable missing values per row
max_missing_values_threshold = 15

train = train[train['Num_Missing_Values'] <= max_missing_values_threshold].iloc[:, :-1]

print(f"length train_no_missing = {len(train_no_missing)}")
print(f"length train = {len(train)}")



#creating a summary report of missing values for each column in our train DataFrame.

import pandas as pd
missing_values_report = pd.DataFrame({
    'Missing Values': train.isnull().sum(),
    'Percentage Missing': train.isnull().mean() * 100
})

print("Missing Values Report:")
missing_values_report


#performing a Chi-squared test for independence between two categorical variables(PRIMEUNIT and IsBadBuy)

from scipy.stats import chi2_contingency

test_df = df_logical_incons_filtered[['PRIMEUNIT', 'IsBadBuy']].dropna()


#Creates a contingency table (cross-tabulation), rows are categories of PRIMEUNIT and values are of IsBadBuy
contingency_table = pd.crosstab(test_df['PRIMEUNIT'], test_df['IsBadBuy'])


row_percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

print("\nRow Percentages:")
print(row_percentages)
print("#"*60) 

chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-squared value: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")


#This function calculates the Odds Ratio (OR) and the 95% Confidence Interval (CI) for the Odds Ratio

import numpy as np
import pandas as pd
from scipy.stats import norm

#a function named OR_CIs that takes a 2x2 contingency table as input.
def OR_CIs(contingency_table):
    
    odds_ratio = (contingency_table.iloc[0, 0] / contingency_table.iloc[0, 1]) / (contingency_table.iloc[1, 0] / contingency_table.iloc[1, 1])

#This calculates the standard error (SE) of the log of the odds ratio.
    log_odds_std_error = np.sqrt(contingency_table.applymap(lambda x: 1/x).sum().sum())

    confidence_level = 0.95

    z_score = norm.ppf(1-(1 - confidence_level) / 2)
    
#the z-score multiplied by the standard error is subtracted and added (for the upper bound) to the log-odds ratio.
#the np.exp() is used to exponentiate the result, bringing it back to the original scale of the odds ratio.
    ci_low = np.exp(np.log(odds_ratio) - z_score * log_odds_std_error)
    ci_high = np.exp(np.log(odds_ratio) + z_score * log_odds_std_error)

    print(f"Odds Ratio: {odds_ratio:.2f}")
    print(f"95% Confidence Interval: {ci_low:.2f}, {ci_high:.2f}")
    
    return


from scipy.stats import fisher_exact

# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(test_df['PRIMEUNIT'], test_df['IsBadBuy'])

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


from scipy.stats import chi2_contingency

# Data
test_df2 = df_logical_incons_filtered[['AUCGUART', 'IsBadBuy']].dropna()

# Create a contingency table
contingency_table = pd.crosstab(test_df2['AUCGUART'], test_df2['IsBadBuy'])


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



from scipy.stats import fisher_exact

# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(test_df2['AUCGUART'], test_df2['IsBadBuy'])

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


from scipy.stats import chi2_contingency

# Data
test_df2 = df_logical_incons_filtered[['AUCGUART', 'IsBadBuy']].dropna()

# Create a contingency table
contingency_table = pd.crosstab(test_df2['AUCGUART'], test_df2['IsBadBuy'])


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


from scipy.stats import fisher_exact

# Create a 2x2 contingency table without margins
contingency_table = pd.crosstab(test_df2['AUCGUART'], test_df2['IsBadBuy'])

# Perform Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Print the result
print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")


import pandas as pd
from sklearn.impute import SimpleImputer

train_no_missing_over20p = train.copy()

# Create SimpleImputer instances for 'PRIMEUNIT' and 'AUCGUART'

PRIMEUNIT_imputer = SimpleImputer(strategy='constant', fill_value='Unknown') 
AUCGUART_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

# Impute missing values in 'PRIMEUNIT' and 'AUCGUART' columns
train_no_missing_over20p[['PRIMEUNIT']] = PRIMEUNIT_imputer.fit_transform(train_no_missing_over20p[['PRIMEUNIT']])
train_no_missing_over20p[['AUCGUART']] = AUCGUART_imputer.fit_transform(train_no_missing_over20p[['AUCGUART']])

# Display the DataFrame after imputation
print("DataFrame after Imputation:")
print(train_no_missing_over20p.info())
train_no_missing_over20p.head()


y_train = train_no_missing_over20p.iloc[:,-1]
X_train = train_no_missing_over20p.iloc[:,0:-1]

inputs = X_train
inputs.info()


columns = inputs.columns

# Choose categorical elements
cat_features = inputs.select_dtypes(include=['object']).columns.tolist()

categorical_indices = [inputs.columns.get_loc(feature) for feature in cat_features]

# Use a list comprehension to select the elements at the specified indices
categorical_fields = [columns[i] for i in categorical_indices]

# Create a new list of columns excluding categorical_fields (continuous)
continuous_fields = [j for j in columns if j not in categorical_fields]


import pandas as pd

# Iterate over each column in the DataFrame
for column in inputs[continuous_fields].columns:
    # Extract the column data
    data = inputs[column]

    # Calculate summary statistics
    data_mean, data_std = data.mean(), data.std()

    # Define outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    
    globals()[f'lower_{column}_sigma'] = lower
    globals()[f'upper_{column}_sigma'] = upper

    # Identify outliers
    globals()[f'outliers_{column}_sigma'] = [x for x in data if x < lower or x > upper]

    # Print the results for each column
    print(f"Column: {column}")
    print(f"Identified outliers: {len(globals()[f'outliers_{column}_sigma'])}")
    globals()[f'outliers_{column}_sigma'].sort()
    print('outlier values: ' ,globals()[f'outliers_{column}_sigma'])
    print("\n")


inputs_sigma = inputs.copy()

# # Remove rows containing outliers
# remove_list = ['VehicleAge', 'VehOdo','VehBCost']

# # Iterate over each column in remove list:
# for column in remove_list:
#     column_out = inputs_sigma[column].isin(globals()[f'outliers_{column}_sigma'])
#     inputs_sigma = inputs_sigma[~column_out]


    
# Coerce outliers to lower or upper bound
coerce_list = ['VehicleAge', 'VehOdo','VehBCost',
               'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
               'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'WarrantyCost']

# Iterate over each column in coerce list:
for column in coerce_list:
    inputs_sigma[column] = inputs_sigma[column].apply(lambda x: globals()[f'lower_{column}_sigma'] if x < globals()[f'lower_{column}_sigma'] 
                                                          else (globals()[f'upper_{column}_sigma'] if x > globals()[f'upper_{column}_sigma'] 
                                                          else x))

inputs_sigma.shape


import pandas as pd

# Iterate over each column in the DataFrame
for column in inputs[continuous_fields].columns:
   
    # Extract the column data
    data = inputs[column]
    
    # calculate interquartile range
    q25, q75 = data.quantile(0.25), data.quantile(0.75)
    iqr = q75 - q25
  
    # calculate the outlier cutoff
    cut_off = iqr * 2
    lower, upper = q25 - cut_off, q75 + cut_off
    
    globals()[f'lower_{column}_iqr'] = lower
    globals()[f'upper_{column}_iqr'] = upper
    
    # identify outliers
    globals()[f'outliers_{column}_iqr'] = [x for x in data if x < lower or x > upper]
    
    print(f"Column: {column}")
    print(f"Identified outliers: {len(globals()[f'outliers_{column}_iqr'])}")
    globals()[f'outliers_{column}_iqr'].sort()
    print('outlier values: ' ,globals()[f'outliers_{column}_iqr'])
    print("\n")
    


inputs_iqr = inputs.copy()

# # Remove rows containing outliers
# remove_list = ['VehicleAge', 'VehOdo','VehBCost']

# # Iterate over each column in remove list:
# for column in remove_list:
#     column_out = inputs_iqr[column].isin(globals()[f'outliers_{column}_iqr'])
#     inputs_iqr = inputs_iqr[~column_out]


    
# Coerce outliers to lower or upper bound
coerce_list = ['VehicleAge', 'VehOdo','VehBCost',
               'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
               'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'WarrantyCost']

# Iterate over each column in coerce list:
for column in coerce_list:
    inputs_iqr[column] = inputs_iqr[column].apply(lambda x: globals()[f'lower_{column}_iqr'] if x < globals()[f'lower_{column}_iqr'] 
                                                      else (globals()[f'upper_{column}_iqr'] if x > globals()[f'upper_{column}_iqr'] 
                                                      else x))
    
inputs_iqr.shape


inputs_mixed = inputs.copy()
    
# Coerce outliers to lower or upper bound
coerce_list_sigma = ['VehicleAge', 'VehOdo',
                     'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 
                     'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice']
coerce_list_iqr = ['VehBCost',
                   'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 
                   'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'WarrantyCost']

# Iterate over each column in coerce list:
for column in (coerce_list_sigma + coerce_list_iqr):
    if column in coerce_list_sigma:
        inputs_mixed[column] = inputs_mixed[column].apply(lambda x: globals()[f'lower_{column}_sigma'] if x < globals()[f'lower_{column}_sigma'] 
                                                      else (globals()[f'upper_{column}_sigma'] if x > globals()[f'upper_{column}_sigma'] 
                                                      else x))
    else:
        inputs_mixed[column] = inputs_mixed[column].apply(lambda x: globals()[f'lower_{column}_iqr'] if x < globals()[f'lower_{column}_iqr'] 
                                                      else (globals()[f'upper_{column}_iqr'] if x > globals()[f'upper_{column}_iqr'] 
                                                      else x))
        
inputs_mixed.shape


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import warnings
# to ignore warnings
warnings.filterwarnings("ignore")

inputs_iso = inputs.copy()

# Replace rows with NaN valuse with mean and mode
categorical = inputs_iso.select_dtypes(include=['object','category']).columns.tolist()
continuous = inputs_iso.select_dtypes(exclude=['object','category']).columns.tolist()

for col in inputs_iso.columns:
    if col in continuous:
        inputs_iso[col] = inputs_iso[col].fillna(inputs_iso[col].mean())
    elif col in categorical:
        mode_val = inputs_iso[col].mode().iloc[0]  # Extract mode value
        inputs_iso[col] = inputs_iso[col].fillna(mode_val)


ordinal = []

nominal = ['Auction', 'Color', 'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART']

# Apply encoding to categorical columns

ordinal_encoder = OrdinalEncoder()
inputs_iso[ordinal] = ordinal_encoder.fit_transform(inputs_iso[ordinal])

one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(inputs_iso[nominal])

one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out())

inputs_iso_encoded = pd.concat([inputs_iso[ordinal].reset_index(), one_hot_encoded_df, inputs_iso[continuous].reset_index(drop=True)], axis=1) 



# Apply Z-score scaling to columns
scaler = StandardScaler()
inputs_iso_encoded_array = scaler.fit_transform(inputs_iso_encoded)


# Fit Isolation Forest model
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(inputs_iso_encoded_array)

# Predict outliers
outliers = clf.predict(inputs_iso_encoded_array)

# Add the outlier predictions to your DataFrame
inputs_iso_encoded['outlier'] = outliers

# Display the DataFrame with outlier information
print(inputs_iso_encoded)

# Calculate the percentage of outliers
percentage_outliers = (outliers[outliers == -1].shape[0] / len(outliers)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")


outlier_index = inputs_iso_encoded[inputs_iso_encoded['outlier'] == -1].index

inputs_outprep = inputs.reset_index(drop=True).drop(outlier_index)
y_train_outprep = y_train.reset_index(drop=True).drop(outlier_index)

train_outprep = pd.concat([inputs_outprep, y_train_outprep], axis=1)
train_outprep.info()
train_outprep.to_csv ('Train-outprep.csv')


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


import pandas as pd

# Report on count and percentage of missing values in each column
missing_values_report = pd.DataFrame({
    #'Column': train.columns,
    'Missing Values': train_outprep.isnull().sum(),
    'Percentage Missing': train_outprep.isnull().mean() * 100
})

# Display the missing values report
print("Missing Values Report:")
missing_values_report


import pandas as pd
from sklearn.impute import SimpleImputer

train_outprep_no_missing_fix = train_outprep.copy()

# Create SimpleImputer instances for features with less than 3% missing values

MMRAcquisitionAuctionAveragePrice_imputer = SimpleImputer(strategy='median') 
MMRAcquisitionAuctionCleanPrice_imputer = SimpleImputer(strategy='median')
MMRAcquisitionRetailAveragePrice_imputer = SimpleImputer(strategy='mean')
MMRAcquisitonRetailCleanPrice_imputer = SimpleImputer(strategy='mean')

MMRCurrentAuctionAveragePrice_imputer = SimpleImputer(strategy='median') 
MMRCurrentAuctionCleanPrice_imputer = SimpleImputer(strategy='median')
MMRCurrentRetailAveragePrice_imputer = SimpleImputer(strategy='mean')
MMRCurrentRetailCleanPrice_imputer = SimpleImputer(strategy='mean')
VehBCost_imputer = SimpleImputer(strategy='mean')

Nationality_imputer = SimpleImputer(strategy='most_frequent')
Size_imputer = SimpleImputer(strategy='most_frequent')
Color_imputer = SimpleImputer(strategy='most_frequent')
WheelType_imputer = SimpleImputer(strategy='most_frequent')
TopThreeAmericanName_imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values
train_outprep_no_missing_fix[['MMRAcquisitionAuctionAveragePrice']] = MMRAcquisitionAuctionAveragePrice_imputer.fit_transform(train_outprep_no_missing_fix[['MMRAcquisitionAuctionAveragePrice']])
train_outprep_no_missing_fix[['MMRAcquisitionAuctionCleanPrice']] = MMRAcquisitionAuctionCleanPrice_imputer.fit_transform(train_outprep_no_missing_fix[['MMRAcquisitionAuctionCleanPrice']])
train_outprep_no_missing_fix[['MMRAcquisitionRetailAveragePrice']] = MMRAcquisitionRetailAveragePrice_imputer.fit_transform(train_outprep_no_missing_fix[['MMRAcquisitionRetailAveragePrice']])                                                          
train_outprep_no_missing_fix[['MMRAcquisitonRetailCleanPrice']] = MMRAcquisitonRetailCleanPrice_imputer.fit_transform(train_outprep_no_missing_fix[['MMRAcquisitonRetailCleanPrice']])                                                             

train_outprep_no_missing_fix[['MMRCurrentAuctionAveragePrice']] = MMRCurrentAuctionAveragePrice_imputer.fit_transform(train_outprep_no_missing_fix[['MMRCurrentAuctionAveragePrice']])
train_outprep_no_missing_fix[['MMRCurrentAuctionCleanPrice']] = MMRCurrentAuctionCleanPrice_imputer.fit_transform(train_outprep_no_missing_fix[['MMRCurrentAuctionCleanPrice']])
train_outprep_no_missing_fix[['MMRCurrentRetailAveragePrice']] = MMRCurrentRetailAveragePrice_imputer.fit_transform(train_outprep_no_missing_fix[['MMRCurrentRetailAveragePrice']])                                                          
train_outprep_no_missing_fix[['MMRCurrentRetailCleanPrice']] = MMRCurrentRetailCleanPrice_imputer.fit_transform(train_outprep_no_missing_fix[['MMRCurrentRetailCleanPrice']])                                                             
train_outprep_no_missing_fix[['VehBCost']] = VehBCost_imputer.fit_transform(train_outprep_no_missing_fix[['VehBCost']])                                                             

train_outprep_no_missing_fix[['Nationality']] = Nationality_imputer.fit_transform(train_outprep_no_missing_fix[['Nationality']])
train_outprep_no_missing_fix[['Size']] = Size_imputer.fit_transform(train_outprep_no_missing_fix[['Size']])
train_outprep_no_missing_fix[['Color']] = Color_imputer.fit_transform(train_outprep_no_missing_fix[['Color']])
train_outprep_no_missing_fix[['WheelType']] = WheelType_imputer.fit_transform(train_outprep_no_missing_fix[['WheelType']])
train_outprep_no_missing_fix[['TopThreeAmericanName']] = TopThreeAmericanName_imputer.fit_transform(train_outprep_no_missing_fix[['TopThreeAmericanName']])

# Display the DataFrame after imputation
print("\nDataFrame after Imputation:\n")
missing_values_report = pd.DataFrame({
    #'Column': train.columns,
    'Missing Values': train_outprep_no_missing_fix.isnull().sum(),
    'Percentage Missing': train_outprep_no_missing_fix.isnull().mean() * 100
})

# Display the missing values report
missing_values_report


import pandas as pd
import numpy as np

# Create a copy of the DataFrame to store imputed values
train_outprep_no_missing_rand = train_outprep.copy()

# Identify columns to impute
columns_to_impute = ['Nationality', 'Size', 'Color', 'WheelType', 'TopThreeAmericanName', 
                     'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
                    'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'VehBCost']

# Impute missing values 
for column in columns_to_impute:
    missing_values = train_outprep_no_missing_rand[column].isnull()
    num_missing = missing_values.sum()
    
    if num_missing > 0:
        # Generate random values based on the distribution of existing values
        np.random.seed(2027)
        random_values = np.random.choice(train_outprep_no_missing_rand[column].dropna(), size=num_missing)
        
        # Assign the random values to missing values
        train_outprep_no_missing_rand.loc[missing_values, column] = random_values

# Display the DataFrame after imputation
print("\nDataFrame after Imputation:\n")
missing_values_report = pd.DataFrame({
    #'Column': train.columns,
    'Missing Values': train_outprep_no_missing_rand.isnull().sum(),
    'Percentage Missing': train_outprep_no_missing_rand.isnull().mean() * 100
})

# Display the missing values report
missing_values_report


import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Create a copy of the DataFrame to store imputed values
train_outprep_no_missing_iter = train_outprep.copy()

# Separate categorical and continuous features
categorical_columns = train_outprep_no_missing_iter.select_dtypes(include=['object']).columns.tolist()
continuous_columns = train_outprep_no_missing_iter.select_dtypes(exclude=['object']).columns.tolist()

# Encode categorical columns
encoder = OrdinalEncoder()
encoded_categorical_data = encoder.fit_transform(train_outprep_no_missing_iter[categorical_columns])
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# Combine encoded categorical data with continuous data
combined_data = pd.concat([encoded_categorical_df, train_outprep_no_missing_iter[continuous_columns].reset_index(drop=True)], axis=1)

# Create IterativeImputer instance
imputer = IterativeImputer()

# Impute missing values
imputed_data = imputer.fit_transform(combined_data)

# Convert the imputed data back to DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=encoded_categorical_df.columns.tolist() + continuous_columns)

# Decode the categorical columns back to their original form
decoded_categorical_data = encoder.inverse_transform(imputed_df[encoded_categorical_df.columns])
decoded_categorical_df = pd.DataFrame(decoded_categorical_data, columns=categorical_columns)

# Combine the imputed continuous features with the decoded categorical features
train_outprep_no_missing_iter = pd.concat([decoded_categorical_df, imputed_df[continuous_columns].reset_index(drop=True)], axis=1)

# Display the DataFrame after imputation
print("\nDataFrame after Imputation:\n", train_outprep_no_missing_iter)

# Display the missing values report
missing_values_report = pd.DataFrame({
    'Missing Values': train_outprep_no_missing_iter.isnull().sum(),
    'Percentage Missing': train_outprep_no_missing_iter.isnull().mean() * 100
})

print(missing_values_report)


import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Create a copy of the DataFrame to store imputed values
train_outprep_no_missing_knn = train_outprep.copy()

# Separate categorical and continuous features
categorical_columns = train_outprep_no_missing_knn.select_dtypes(include=['object']).columns.tolist()
continuous_columns = train_outprep_no_missing_knn.select_dtypes(exclude=['object']).columns.tolist()

# Encode categorical columns
encoder = OrdinalEncoder()
encoded_categorical_data = encoder.fit_transform(train_outprep_no_missing_knn[categorical_columns])
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# Combine encoded categorical data with continuous data
combined_data = pd.concat([encoded_categorical_df, train_outprep_no_missing_knn[continuous_columns].reset_index(drop=True)], axis=1)

# Create IterativeImputer instance
imputer = KNNImputer()

# Impute missing values
imputed_data = imputer.fit_transform(combined_data)

# Convert the imputed data back to DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=encoded_categorical_df.columns.tolist() + continuous_columns)

# Decode the categorical columns back to their original form
decoded_categorical_data = encoder.inverse_transform(imputed_df[encoded_categorical_df.columns])
decoded_categorical_df = pd.DataFrame(decoded_categorical_data, columns=categorical_columns)

# Combine the imputed continuous features with the decoded categorical features
train_outprep_no_missing_knn = pd.concat([decoded_categorical_df, imputed_df[continuous_columns].reset_index(drop=True)], axis=1)

# Display the DataFrame after imputation
print("\nDataFrame after Imputation:\n", train_outprep_no_missing_iter)

# Display the missing values report
missing_values_report = pd.DataFrame({
    'Missing Values': train_outprep_no_missing_knn.isnull().sum(),
    'Percentage Missing': train_outprep_no_missing_knn.isnull().mean() * 100
})

print(missing_values_report)


train_outprep_no_missing_knn.iloc[:,0:-1].to_csv('/kaggle/working/dontgetkicked_v2.csv')


