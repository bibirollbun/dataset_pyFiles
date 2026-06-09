import numpy as np
import pandas as pd


!pip install ydata_profiling


!pip install --upgrade numba pandas visions ydata_profiling


df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


df.info()


df['IsOnlineSale'] = df['IsOnlineSale'].astype('object').replace('nan',np.nan)
df.info()


from ydata_profiling import ProfileReport
categorical_fields = ["IsBadBuy", "Auction", "VehYear", "Make", "Model", "Trim", "SubModel", "Color",
                      "Transmission", "WheelTypeID", "WheelType", "Nationality", "Size", "TopThreeAmericanName",
                      "PRIMEUNIT", "AUCGUART", "BYRNO", "VNZIP1", "VNST", "IsOnlineSale"]
type_schema= {i:"categorical" for i in categorical_fields}

profile = ProfileReport(df,title='Dont_get_kicked_2025_04_30',type_schema = type_schema,explorative=True)
profile.to_file("Dont_get_kicked_2025_04_30.html")


corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

print(corr_matrix)


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 12))
sns.heatmap(
    corr_matrix,
    annot=True,      # نمایش مقادیر در خانه‌ها
    fmt=".2f",       # نمایش اعداد با ۲ رقم اعشار
    cmap="coolwarm",  # انتخاب رنگ‌ها (coolwarm, RdYlBu, etc.)
    vmin=-1,         # حداقل مقدار همبستگی
    vmax=1,          # حداکثر مقدار همبستگی
    linewidths=0.5,  # عرض خطوط بین خانه‌ها
)

plt.title("Correlation Heatmap of Continuous Features")
plt.show()


from ydata_profiling import ProfileReport
IsBadBuy0 = df[df.IsBadBuy == 0]
IsBadBuy1 = df[df.IsBadBuy == 1]

profile0 = ProfileReport(IsBadBuy0,title='Good Buy', minimal = True, type_schema=type_schema)
profile1 = ProfileReport(IsBadBuy1,title='Bad Buy', minimal = True, type_schema=type_schema)

comparison_report = profile0.compare(profile1)
comparison_report.to_file("Buy comparison")


df = df.drop(['PurchDate','VehYear','Model','Trim','SubModel','WheelTypeID','BYRNO','VNZIP1','VNST'],axis=1)


df.info()


df= df.set_index('RefId')
df.head()


target = df['IsBadBuy']
features = df.drop(columns=['IsBadBuy'])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=7)

inputs = x_train


columns = inputs.columns
columns


for index, value in enumerate(columns):
    print(f"Index: {index}, Value: '{value}'")


categorical_indices = [0,2,3,4,5,7,8,9,18,19,21]
categorical_fields = [columns[i] for i in categorical_indices]
continuous_fields = [j for j in columns if j not in categorical_fields]


inputs.describe()


column_range ={'VehicleAge': (0,30),'VehOdo': (0,120000),
'MMRAcquisitionAuctionAveragePrice': (800,46000),
'MMRAcquisitionAuctionCleanPrice': (1000,46000),
'MMRAcquisitionRetailAveragePrice': (1000,46000),
'MMRAcquisitonRetailCleanPrice': (1000,46000),
'MMRCurrentAuctionAveragePrice': (300,46000),
'MMRCurrentAuctionCleanPrice': (400,46000),
'MMRCurrentRetailAveragePrice': (800,46000),
'MMRCurrentRetailCleanPrice': (1000,46000),
'VehBCost': (1000,46000),
'WarrantyCost': (400,8000)}



for column, (min_val,max_val) in column_range.items():
    inputs[column] = inputs[column].apply (lambda x: x if min_val <= x <= max_val else None) 


inputs.describe()


inputs.info()
print('=' * 10)
print(inputs)


def frequency_table(variable):
    
    # unique elements and counts
    unique_elements, counts = np.unique(variable.dropna(), return_counts=True)

    # percentages
    percentages = (counts / len(variable.dropna())) * 100

    # Create a dictionary to store the value counts and percentages
    value_counts_and_percentages = zip(unique_elements, counts, percentages)

    for i, j, k in value_counts_and_percentages:
        print(f"{i:>14}:  Count: {j:>7}, Percentage: {k:>5.3f}%")
    return


for i in categorical_fields:
    print(f"Frequency Table for {i}:")
    frequency_table(inputs[i])
    print('='*50)


inputs['Transmission'] = inputs['Transmission'].replace("Manual", "MANUAL")
# Replace 'NOT AVAIL' with NaN in the 'color' column
inputs['Color'] = inputs['Color'].replace('NOT AVAIL', np.nan)


for i in categorical_fields:
    print(f"Frequency Table for {i}:")
    frequency_table(inputs[i])
    print('='*50)


def group_rare_categories(df, column, threshold = 0.01):
    freq = df[column].value_counts(normalize=True)  
    rare = freq[freq < threshold].index              
    df[column] = df[column].apply(lambda x: 'OTHER' if x in rare else x)
    return df


# Apply to 'color' and 'make'
inputs = group_rare_categories(inputs, 'Color')
inputs = group_rare_categories(inputs, 'Make')
print('=' * 40)
frequency_table(inputs['Color'])
print('=' * 40)
frequency_table(inputs['Make'])



min_cv = 0.1
cv_values = inputs[continuous_fields].std() / inputs [continuous_fields].mean()
selected_columns = cv_values[cv_values < 0.1].index
filtered_con = inputs[selected_columns]
inputs_con = inputs[continuous_fields].drop(selected_columns,axis = 1)
print(inputs_con)


threshold = 99
mode_category = (inputs[categorical_fields].apply(lambda x: x.value_counts().max() / len(x))* 100)
selected_categorical_columns = mode_category[mode_category> threshold].index
mode_filtered_inputs = inputs[selected_categorical_columns]
inputs_cat = inputs[categorical_fields].drop(selected_categorical_columns,axis =1)
print(inputs_cat)


threshold= 90
distinct_percentage = (inputs_cat[categorical_fields].apply(lambda x: x.dropna().nunique() / x.count() ) * 100)
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index
distinct_filtered_inputs = inputs_cat[selected_categorical_columns]
inputs_cat = inputs_cat.drop(selected_categorical_columns,axis=1)
print(inputs_cat)


inputs = pd.concat([inputs_con, inputs_cat], axis=1)


from scipy.stats import chi2_contingency
contingency_table= pd.crosstab(inputs['PRIMEUNIT'].dropna(),y_train.dropna(),normalize=False)
print(contingency_table)


from scipy.stats import chi2_contingency

chi2,p,dof,expected = chi2_contingency(contingency_table)
print(f"\nChi-squared value: {chi2:,}")
print('=' * 30 )
print(f"P-Value: {p:,}")
print('=' * 30 )
print(f"Degree Of Freedom:{dof:,}")
print('=' * 30 )
print(f"Expected Value:\n{expected:}")
print('=' * 60 )

residual = contingency_table - expected
print("Residuals = Observed - Expected")
print(f"Residuals:{residual}")
print('=' * 60 )

expected_less_than_one = (expected<1)
print(expected_less_than_one)
expected_less_than_one = (expected<1).sum().sum()
print(expected_less_than_one)
print('=' * 60 )

percentage_low_expected = (expected<5).sum().sum()/(expected.shape[0] * expected.shape[1])*100
a=expected.shape[0] * expected.shape[1]
print('"Check if shape is 2 * 2 =4"',a)
print(f" Percentage of Cells with expected counts less than 5: {percentage_low_expected:.2f}%")
print('=' * 60 )



from scipy.stats import fisher_exact
odds_ratio,p_value = fisher_exact(contingency_table)
print(f"Fisher odds_ratio = {odds_ratio}")
print(f"Fisher P-Value = {p_value}")
print('='*50)


from scipy.stats import norm
def OR_CIS(contingency_table):
    odds_ratio = ((contingency_table.iloc[0,0] /contingency_table.iloc[0,1])/
                  (contingency_table.iloc[1,0]/contingency_table.iloc[1,1]))
    log_odds_std_error = np.sqrt(contingency_table.map(lambda x: 1/x).sum().sum())
    confidence_level = 0.95
    z_score = norm.ppf(1-(1-confidence_level)/2)
    ci_low = np.exp(np.log(odds_ratio)-z_score * log_odds_std_error)
    ci_high = np.exp(np.log(odds_ratio)+z_score * log_odds_std_error)
    print(f"Odds ratio: {odds_ratio:.2f}")
    print(f"95% confidence Interval: {ci_low:.2f},{ci_high:.2f}")
    return


OR_CIS(contingency_table)


contingency_table= pd.crosstab(inputs['AUCGUART'].dropna(),y_train.dropna(),normalize=False)
print(contingency_table)
chi2,p,dof,expected = chi2_contingency(contingency_table)
print(f"\nChi-squared value: {chi2:,}")
print('=' * 30 )
print(f"P-Value: {p:,}")
print('=' * 30 )
print(f"Degree Of Freedom:{dof:,}")
print('=' * 30 )
print(f"Expected Value:\n{expected:}")
print('=' * 60 )

residual = contingency_table - expected
print("Residuals = Observed - Expected")
print(f"Residuals:{residual}")
print('=' * 60 )

expected_less_than_one = (expected<1)
print(expected_less_than_one)
expected_less_than_one = (expected<1).sum().sum()
print(expected_less_than_one)
print('=' * 60 )

percentage_low_expected = (expected<5).sum().sum()/(expected.shape[0] * expected.shape[1])*100
a=expected.shape[0] * expected.shape[1]
print('"Check if shape is 2 * 2 =4"',a)
print(f" Percentage of Cells with expected counts less than 5: {percentage_low_expected:.2f}%")
print('=' * 60 )



from scipy.stats import fisher_exact
odds_ratio,p_value = fisher_exact(contingency_table)
print(f"Fisher odds_ratio = {odds_ratio}")
print(f"Fisher P-Value = {p_value}")
print('='*50)


OR_CIS(contingency_table)


inputs['PRIMEUNIT'] = df['PRIMEUNIT'].fillna('unknown')
inputs['AUCGUART'] = df['AUCGUART'].fillna('unknown')
frequency_table(inputs['PRIMEUNIT'])
print('='*50)
frequency_table(inputs['AUCGUART'])


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
clf = IsolationForest(contamination=0.01, random_state=7)
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
pd.set_option('Display.max_columns',None)
pd.set_option('Display.width',None)
train_outprep = pd.concat([inputs_outprep, y_train_outprep], axis=1)


train_outprep['Num_Missing_Values'] = train_outprep.isnull().sum(axis=1)
frequency_table(train_outprep['Num_Missing_Values'])
print('1'*20)
rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] > 0]
frequency_table(rows_with_missing_values['Num_Missing_Values'])
print('2'*20)
rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] >= 4]
frequency_table(rows_with_missing_values['Num_Missing_Values'])
print('3'*20)
price_fields_null_check = ['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
                           'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                           'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',
                           'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice']
train_outprep['Num_Missing_Values'] = train_outprep[price_fields_null_check].isnull().sum(axis=1)

frequency_table(train_outprep['Num_Missing_Values']) #double check


# Discard rows with 4 or more null values in the variables related to prices (from 'MMRAcquisitionAuctionAveragePrice' to 'MMRCurrentRetailCleanPrice').
price_fields_null_check = ['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
                           'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                           'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',
                           'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice']
train_outprep['Num_Missing_Values'] = train_outprep[price_fields_null_check].isnull().sum(axis=1)

frequency_table(train_outprep['Num_Missing_Values']) #double check
rows_with_missing_values_more_than_four = train_outprep[train_outprep['Num_Missing_Values'] >= 4]
print('*' * 10)
frequency_table(rows_with_missing_values_more_than_four['Num_Missing_Values']) #double check

total_rows = len(train_outprep)
rows_with_missing_count_more_than_four_in_selected_fields = len(rows_with_missing_values_more_than_four)
percentage_rows_with_missing_count_more_than_four_in_selected_fields = (rows_with_missing_count_more_than_four_in_selected_fields / total_rows) * 100
print("Report on rows with missing values more than 4 in selected columns:")
print(f"Total Rows: {total_rows}")
print(f"Rows with > 3 Missing Values in selected columns: {rows_with_missing_count_more_than_four_in_selected_fields} ({percentage_rows_with_missing_count_more_than_four_in_selected_fields:.2f}%)")
# pd.set_option('Display.max_columns',None)
# pd.set_option('Display.width',None)
# pd.set_option('Display.max_row',1000)

# Show sorted DataFrame
print("\nRows Sorted by Number of Missing Values (in selected columns):")
print(train_outprep.sort_values(by='Num_Missing_Values', ascending=False))
print(rows_with_missing_values_more_than_four.sort_values(by='Num_Missing_Values', ascending=False))


train_outprep = train_outprep[train_outprep['Num_Missing_Values'] < 4].iloc[:, :-1]


train_outprep['Num_Missing_Values'] = train_outprep.isnull().sum(axis=1)
rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] > 0]
frequency_table(rows_with_missing_values['Num_Missing_Values']) #double check

total_rows = len(train_outprep)
rows_with_missing_count = len(rows_with_missing_values)
percentage_rows_with_missing = (rows_with_missing_count / total_rows) * 100

# Display the report
print("Report on Rows with Missing Values:")
print(f"Total Rows: {total_rows}")
print(f"Rows with Missing Values: {rows_with_missing_count} ({percentage_rows_with_missing:.2f}%)")


# Display the DataFrame with new column
print("\nDataFrame with Num_Missing_Values column:")
print(train_outprep.sort_values(by='Num_Missing_Values', ascending = False))


train_outprep = train_outprep[train_outprep['Num_Missing_Values'] <= 11].iloc[:, :-1]


missing_values_report = pd.DataFrame({
    'Column': train_outprep.columns,
    'Missing Values': train_outprep.isnull().sum(),
    'Percentage Missing': train_outprep.isnull().mean() * 100})


print("Missing Values Report:")
print(missing_values_report)


from sklearn.impute import SimpleImputer

train_outprep_no_missing_fix = train_outprep.copy()


median_imputer_for_continuous_fields = SimpleImputer(strategy='median') 
mode_imputer_for_categorical_fields = SimpleImputer(strategy='most_frequent')  
                                                     

# Impute missing values in continuos columns
train_outprep_no_missing_fix[continuous_fields] = median_imputer_for_continuous_fields.fit_transform(train_outprep_no_missing_fix[continuous_fields])

# Impute missing values in categorical column
train_outprep_no_missing_fix[categorical_fields] = mode_imputer_for_categorical_fields.fit_transform(train_outprep_no_missing_fix[categorical_fields])


print("DataFrame after Imputation:")
print(train_outprep_no_missing_fix)

train_outprep_no_missing_fix['IsOnlineSale'].isnull().sum()

train_outprep_no_missing_fix.to_csv('/kaggle/working/Carvana_Cleaned2.csv')


train_FS = pd.read_csv('/kaggle/working/Carvana_Cleaned2.csv')
# train_FS['IsOnlineSale'] = df['IsOnlineSale'].astype('object').replace('nan',np.nan)
# mode_imputer_for_categorical_fields = SimpleImputer(strategy='most_frequent') 
# train_FS['IsOnlineSale'] = mode_imputer_for_categorical_fields.fit_transform(train_FS['IsOnlineSale'])
train_FS = train_FS.set_index('RefId')
train_FS.info()


continuous_var = train_FS.select_dtypes(include=['float64']).columns.tolist()

concatenated_series = pd.concat([
    train_FS[continuous_var].describe().T,
    train_FS[continuous_var].skew().rename('skewness'),
    train_FS[continuous_var].kurtosis().rename('kurtosis')], axis=1)

print(concatenated_series)


!pip install scorecardbundle


from scorecardbundle.feature_discretization import ChiMerge as cm

chi_merge_method_list = ['VehBCost', 'WarrantyCost']
trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=1, decimal=2,output_dataframe=True)
result_cm = trans_cm.fit_transform(train_FS[chi_merge_method_list], train_FS['IsBadBuy']) 
trans_cm.boundaries_



result_cm


boundaries_dict = {key: np.insert(boundaries, 0, -np.inf) for key, boundaries in trans_cm.boundaries_.items()}
boundaries_dict


for key, boundaries in boundaries_dict.items():
    column_name = f"{key}_cat_cm"
    train_FS[column_name] = pd.cut(train_FS[key], bins=boundaries, labels=False, right=False)
    
    print(f'{column_name} bin edges:', boundaries)
    
    frequency_table(train_FS[column_name])
    print("\n")
train_FS


train_FS = train_FS.drop(['VehBCost', 'WarrantyCost'], axis=1)
train_FS.info()


# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder


one_hot_encoder = OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False)

one_hot_encoded = one_hot_encoder.fit_transform(train_FS[['Auction', 'Make', 'Color', 
                                                          'Transmission', 'WheelType', 'Nationality',
                                                          'Size', 'TopThreeAmericanName','PRIMEUNIT','AUCGUART']])
# Add results
one_hot_encoded_train_FS = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out())
encoded_train_FS = pd.concat([train_FS.reset_index(drop=True), one_hot_encoded_train_FS.reset_index(drop=True)], axis=1)



encoded_train_FS = encoded_train_FS.drop(['Auction', 'Make', 'Color','Transmission', 
                                           'WheelType', 'Nationality','Size', 'TopThreeAmericanName',
                                           'PRIMEUNIT','AUCGUART'], axis=1)
encoded_train_FS.info()


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

selected_columns = [feature for feature in encoded_train_FS.columns if feature != "IsBadBuy"]

for feature in selected_columns: 
    min_max_scaler = MinMaxScaler()
    encoded_train_FS[f"{feature}_min_max"] = min_max_scaler.fit_transform(encoded_train_FS[[feature]])
print(encoded_train_FS)


pd.set_option('display.max_columns', None)  # Show all columns
encoded_train_FS.info(verbose=True)
encoded_train_FS.info()


scaled_train_FS = encoded_train_FS.drop(['VehicleAge','VehOdo',
                                         'MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
                                         'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
                                         'MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice',
                                         'IsOnlineSale','PRIMEUNIT_YES','PRIMEUNIT_unknown','AUCGUART_RED','AUCGUART_unknown',
                                         'VehBCost_cat_cm',
                                         'WarrantyCost_cat_cm',
                                         'Auction_MANHEIM','Auction_OTHER',
                                         'Make_CHRYSLER','Make_DODGE','Make_FORD','Make_HYUNDAI','Make_JEEP','Make_KIA', 
                                         'Make_MAZDA','Make_MERCURY','Make_MITSUBISHI','Make_NISSAN','Make_OTHER','Make_PONTIAC', 
                                         'Make_SATURN','Make_SUZUKI','Make_TOYOTA',
                                         'Color_BLACK','Color_BLUE','Color_GOLD','Color_GREEN','Color_GREY','Color_MAROON','Color_OTHER',
                                         'Color_RED','Color_SILVER','Color_WHITE',
                                         'Transmission_MANUAL',
                                         'WheelType_Covers','WheelType_Special',
                                         'Nationality_OTHER','Nationality_OTHER ASIAN','Nationality_TOP LINE ASIAN',
                                         'Size_CROSSOVER','Size_LARGE','Size_LARGE SUV','Size_LARGE TRUCK','Size_MEDIUM','Size_MEDIUM SUV','Size_SMALL SUV',
                                         'Size_SMALL TRUCK','Size_SPECIALTY','Size_SPORTS','Size_VAN',
                                         'TopThreeAmericanName_FORD','TopThreeAmericanName_GM','TopThreeAmericanName_OTHER'], axis=1)
scaled_train_FS.info()


scaled_train_FS.to_csv('/kaggle/working/Carvana_train_FS.csv')


train_FE = pd.read_csv('/kaggle/working/Carvana_Cleaned2.csv')


train_FE = train_FE.set_index('RefId')


from sklearn.preprocessing import PowerTransformer
selected_features = ['VehBCost', 'WarrantyCost']
for feature in selected_features:
    # Check negative values
    has_negative_values = (train_FE[feature] <= 0).any()
    
    
    if has_negative_values:
        print(has_negative_values)
        break
    else:
        transformer = PowerTransformer(method='box-cox', standardize=False)

    
    train_FE[f"{feature}_transformed"] = transformer.fit_transform(train_FE[[feature]])
    lambda_value = transformer.lambdas_[0]
    print(f"Lambda for {feature}: {lambda_value}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))

    plt.subplot(1, 2, 1)
    plt.hist(train_FE[feature], bins=30, color='purple', alpha=0.7)
    plt.title(f'Original {feature} Histogram')

    plt.subplot(1, 2, 2)
    plt.hist(train_FE[f"{feature}_transformed"], bins=30, color='green', alpha=0.7)
    plt.title(f'Transformed {feature} Histogram')

    plt.tight_layout()
    plt.show()

print(train_FE)


train_FE = train_FE.drop(['VehBCost', 'WarrantyCost'], axis=1)
train_FE.info()


# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder


one_hot_encoder = OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False)

one_hot_encoded = one_hot_encoder.fit_transform(train_FE[['Auction', 'Make', 'Color', 
                                                          'Transmission', 'WheelType', 'Nationality',
                                                          'Size', 'TopThreeAmericanName','PRIMEUNIT','AUCGUART']])
# Add results
one_hot_encoded_train_FE = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out())
encoded_train_FE = pd.concat([train_FE.reset_index(drop=True), one_hot_encoded_train_FE.reset_index(drop=True)], axis=1)


encoded_train_FE = encoded_train_FE.drop(['Auction', 'Make', 'Color','Transmission', 
                                           'WheelType', 'Nationality','Size', 'TopThreeAmericanName',
                                           'PRIMEUNIT','AUCGUART'], axis=1)
encoded_train_FE.info()


import pandas as pd
from sklearn.preprocessing import StandardScaler

selected_columns = [feature for feature in encoded_train_FE.columns if feature != "IsBadBuy"]

for feature in selected_columns:
    z_score_scaler = StandardScaler()
    encoded_train_FE[f"{feature}_z_score"] = z_score_scaler.fit_transform(encoded_train_FE[[feature]])

print(encoded_train_FE)


encoded_train_FE.info(verbose=True)
encoded_train_FE.info()


scaled_train_FE = encoded_train_FE.drop(['VehicleAge','VehOdo',
                                         'MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
                                         'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
                                         'MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice',
                                         'IsOnlineSale','PRIMEUNIT_YES','PRIMEUNIT_unknown','AUCGUART_RED','AUCGUART_unknown',
                                         'VehBCost_transformed',
                                         'WarrantyCost_transformed',
                                         'Auction_MANHEIM','Auction_OTHER',
                                         'Make_CHRYSLER','Make_DODGE','Make_FORD','Make_HYUNDAI','Make_JEEP','Make_KIA', 
                                         'Make_MAZDA','Make_MERCURY','Make_MITSUBISHI','Make_NISSAN','Make_OTHER','Make_PONTIAC', 
                                         'Make_SATURN','Make_SUZUKI','Make_TOYOTA',
                                         'Color_BLACK','Color_BLUE','Color_GOLD','Color_GREEN','Color_GREY','Color_MAROON','Color_OTHER',
                                         'Color_RED','Color_SILVER','Color_WHITE',
                                         'Transmission_MANUAL',
                                         'WheelType_Covers','WheelType_Special',
                                         'Nationality_OTHER','Nationality_OTHER ASIAN','Nationality_TOP LINE ASIAN',
                                         'Size_CROSSOVER','Size_LARGE','Size_LARGE SUV','Size_LARGE TRUCK','Size_MEDIUM','Size_MEDIUM SUV','Size_SMALL SUV',
                                         'Size_SMALL TRUCK','Size_SPECIALTY','Size_SPORTS','Size_VAN',
                                         'TopThreeAmericanName_FORD','TopThreeAmericanName_GM','TopThreeAmericanName_OTHER'], axis=1)
scaled_train_FE.info()


scaled_train_FE.to_csv('/kaggle/working/Carvana_train_FE.csv')


train_FS = pd.read_csv('/kaggle/working/Carvana_train_FS.csv')

