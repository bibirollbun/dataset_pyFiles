pip install scorecardbundle


import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import kurtosis, skew
from scorecardbundle.feature_discretization import ChiMerge as cm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt


df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.head()


df.shape


df.info()


#!pip install --upgrade numba pandas visions ydata_profiling


#profile = ProfileReport(df, title="Car purchased at the Auction data EDA", type_schema = {"Auction": "categorical",
 #                                                                                         "Default": "categorical",
  #                                                                                        "Make": "categorical",
   #                                                                                       "Model": "categorical",
    #                                                                                      "Trim": "categorical",
     #                                                                                     "Submodel": "categorical",
      #                                                                                    "Color": "categorical",
       #                                                                                   "Transmission": "categorical",
        #                                                                                  "WheelType": "categorical",
         #                                                                                "Nationality": "categorical",
          #                                                                               "Size": "categorical",
           #                                                                              "TopThreeAmericanName": "categorical",
            #                                                                              "PRIMEUNIT": "categorical",
             #                                                                             "AUCGUART": "categorical",
              #                                                                            "VNST": "categorical"})
#
#profile.to_file("your_dataset_profile_report.html")


Filtered_df = df.copy()


Filtered_df.drop(columns=['PurchDate','VehYear','Model','Trim','SubModel','WheelTypeID','BYRNO','VNZIP1','VNST'], inplace=True)
Filtered_df.head()


Filtered_df .set_index('RefId', inplace=True)
Filtered_df.head()


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

for column, (min_val, max_val) in column_ranges.items():
    Filtered_df[column] = Filtered_df[column].apply(lambda x: x if min_val <= x <= max_val else None)

print(Filtered_df)
Filtered_df.describe()
Filtered_df.info()


def frequency_table(variable):
    
    unique_elements, counts = np.unique(variable.dropna(), return_counts=True)

    percentages = (counts / len(variable)) * 100

    value_counts_and_percentages = zip(unique_elements, counts, percentages)

    for i, j, k in value_counts_and_percentages:
        print(f"{i}: Count: {j}, Percentage: {k:.2f}%")
    return


frequency_table(Filtered_df['IsBadBuy'])


y = Filtered_df.iloc[:,0]
X = Filtered_df.drop(Filtered_df.columns[0],axis=1)
X_train,y_train, X_test, y_test = train_test_split(X,y, test_size=0.2, random_state=123)


Inputs = X_train

columns = Inputs.columns

categorical_indices = [i for i , col in enumerate(Inputs.columns) if Inputs[col].dtype=='object']

categorical_fields = [columns[i] for i in categorical_indices]

continuous_fields = [j for j in columns if j not in categorical_fields]


Filtered_df['Color'] = Filtered_df['Color'].replace('NOT AVAIL', np.nan)


for col in ['Color', 'Make']:
    freq = Filtered_df[col].value_counts(normalize=True)  
    rare_classes = freq[freq < 0.01].index        
    Filtered_df[col] = Filtered_df[col].apply(lambda x: 'Other' if x in rare_classes else x)


min_cv = 0.1

cv_values = Inputs[continuous_fields].std() / Inputs[continuous_fields].mean()

selected_columns =  cv_values[cv_values < 0.1].index

filtered_con = Inputs[selected_columns]

Inputs_con = Inputs[continuous_fields].drop(selected_columns, axis=1)
Inputs_con.shape


threshold = 99

mode_category = (Inputs[categorical_fields].apply(lambda x: x.value_counts().max() / len(x)) * 100)

selected_categorical_columns = mode_category[mode_category > threshold].index

mode_filtered_inputs = Inputs[selected_categorical_columns]

Inputs_cat = Inputs[categorical_fields].drop(selected_categorical_columns, axis=1)
Inputs_cat.shape


threshold = 90

distinct_percentage = (Inputs_cat.apply(lambda x: x.dropna().nunique() / x.count()) * 100)

selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index

distinct_filtered_inputs = Inputs_cat[selected_categorical_columns]

Inputs_cat = Inputs_cat.drop(selected_categorical_columns, axis=1)
Inputs_cat.shape


primeunit_data = Filtered_df[['PRIMEUNIT', 'IsBadBuy']].dropna()

contingency_primeunit = pd.crosstab(primeunit_data['PRIMEUNIT'], primeunit_data['IsBadBuy'])

row_percentages_pu = contingency_primeunit.div(contingency_primeunit.sum(axis=1), axis=0) * 100


chi2, p, dof, expected_pu = chi2_contingency(contingency_primeunit)

percentage_low_expected_PRIMEUNIT = (expected_pu < 5).sum().sum() / (expected_pu.shape[0] * expected_pu.shape[1]) * 100

print(f"Percentage of cells with expected counts less than 5: {percentage_low_expected_PRIMEUNIT:.2f}%")


odds_ratio, p_value = fisher_exact(contingency_primeunit)

print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")

if p_value < 0.05:
   print("There is a significant difference in 'PRIMEUNIT' among the 'IsBadBuy' groups.")
else:
   print("There is no significant difference in 'PRIMEUNIT' among the 'IsBadBuy' groups.")


aucguart_data = Filtered_df[['AUCGUART', 'IsBadBuy']].dropna()

contingency_aucguart = pd.crosstab(aucguart_data['AUCGUART'], aucguart_data['IsBadBuy'])
row_percentages_au = contingency_aucguart.div(contingency_aucguart.sum(axis=1), axis=0) * 100

chi2_au, p_value, dof_au, expected_au = chi2_contingency(contingency_aucguart)

percentage_low_expected_AUCGUART = (expected_au < 5).sum().sum() / (expected_au.shape[0] * expected_au.shape[1]) * 100

print(f"Percentage of cells with expected counts less than 5: {percentage_low_expected_AUCGUART:.2f}%")

odds_ratio, p_value = fisher_exact(contingency_aucguart)

print(f"Odds ratio: {odds_ratio}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference in 'AUCGUART' among the 'IsBadBuy' groups.")
else:
    print("There is no significant difference in 'AUCGUART' among the 'IsBadBuy' groups.")


Filtered_df['PRIMEUNIT'] = Filtered_df['PRIMEUNIT'].fillna('unknown')
Filtered_df['AUCGUART'] = Filtered_df['AUCGUART'].fillna('unknown')


Filtered_df.to_csv('/kaggle/working/Auction_Car.csv')


Cleaning_df=Filtered_df.copy()


Target_Cleaning = Cleaning_df.iloc[:,0]
Inputs_Cleaning = Cleaning_df.drop(Cleaning_df.columns[0],axis=1)


columns_Cleaning = Inputs_Cleaning.columns

categorical_indices_Cleaning = [i for i , col in enumerate(Inputs_Cleaning.columns) if Inputs_Cleaning[col].dtype=='object']

categorical_fields_Cleaning = [columns_Cleaning[i] for i in categorical_indices_Cleaning]

continuous_fields_Cleaning = [j for j in columns_Cleaning if j not in categorical_fields_Cleaning]


categorical_fields_Cleaning


Inputs_iso = Inputs_Cleaning.copy()

Inputs_iso = Inputs_iso.dropna()

scaler = StandardScaler()
Inputs_iso[continuous_fields_Cleaning] = scaler.fit_transform(Inputs_iso[continuous_fields_Cleaning])

label_encoder = LabelEncoder()
Inputs_iso[categorical_fields_Cleaning] = Inputs_iso[categorical_fields_Cleaning].apply(label_encoder.fit_transform)

clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(Inputs_iso)

outliers = clf.predict(Inputs_iso)

Inputs_iso['outlier'] = outliers

percentage_outliers = (outliers[outliers == -1].shape[0] / len(outliers)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")


y_train_Cleaning = Target_Cleaning
outlier_index = Inputs_iso[Inputs_iso['outlier'] == -1].index
Inputs_outprep = Inputs_Cleaning.drop(outlier_index)
y_train_outprep = y_train_Cleaning.drop(outlier_index)

train_outprep = pd.concat([Inputs_outprep, y_train_outprep], axis=1)


train_outprep['Num_Missing_Values'] = train_outprep.isnull().sum(axis=1)

rows_with_missing_values = train_outprep[train_outprep['Num_Missing_Values'] > 0]

total_rows = len(train_outprep)
rows_with_missing_count = len(rows_with_missing_values)
percentage_rows_with_missing = (rows_with_missing_count / total_rows) * 100

print("Report on Rows with Missing Values:")
print(f"Total Rows: {total_rows}")
print(f"Rows with Missing Values: {rows_with_missing_count} ({percentage_rows_with_missing:.2f}%)")


print("\nDataFrame with Num_Missing_Values column:")
print(train_outprep.sort_values(by='Num_Missing_Values', ascending = False))


train_outprep.columns


columns_price = [
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice' ,
    'MMRCurrentAuctionCleanPrice' ,
    'MMRCurrentRetailAveragePrice' , 
    'MMRCurrentRetailCleanPrice'
]

train_outprep_price_clean = train_outprep.dropna(subset=columns_price)

max_missing_values_threshold = 4

missing_counts = train_outprep_price_clean[columns_price].isna().sum(axis=1)

train_outprep =  train_outprep_price_clean[missing_counts <= max_missing_values_threshold]


missing_values_report = pd.DataFrame({
    'Column': train_outprep.columns,
    'Missing Values': train_outprep.isnull().sum().values,
    'Percentage Missing': train_outprep.isnull().mean().values * 100
})

print("Missing Values Report:")
print(missing_values_report)


train_outprep_no_missing_fix = train_outprep.copy()

Con_Vars = ['VehBCost']
Cat_Vars = ['Color','Size','Transmission','WheelType','Nationality','TopThreeAmericanName']

Con_imputer = SimpleImputer(strategy='median') 
Cat_imputer = SimpleImputer(strategy='most_frequent')   
                                                     

train_outprep_no_missing_fix[Con_Vars] = Con_imputer.fit_transform(train_outprep_no_missing_fix[Con_Vars])

train_outprep_no_missing_fix[Cat_Vars] = Cat_imputer.fit_transform(train_outprep_no_missing_fix[Cat_Vars])

print("DataFrame after Imputation:")
print(train_outprep_no_missing_fix)

train_outprep_no_missing_fix.info()


missing_values_report_ = pd.DataFrame({
    'Column': train_outprep_no_missing_fix.columns,
    'Missing Values': train_outprep_no_missing_fix.isnull().sum().values,
    'Percentage Missing': train_outprep_no_missing_fix.isnull().mean().values * 100
})

print("Missing Values Report:")
print(missing_values_report_)


train_outprep_no_missing_fix = train_outprep_no_missing_fix.drop(columns=['Num_Missing_Values'])


train_outprep_no_missing_fix.to_csv('/kaggle/working/df_train_car.csv')


train_FS = pd.read_csv('/kaggle/working/df_train_car.csv')
train_FS.head()


train_FS.set_index('RefId', inplace=True)
train_FS.shape


Desc_Stats = train_FS.describe()

continuous_columns_ = train_FS.select_dtypes(include=['float64', 'int64']).columns

skewness_values = {}
kurtosis_values = {}

for column in continuous_columns_:
    skewness_values[column] = skew(train_FS[column])
    kurtosis_values[column] = kurtosis(train_FS[column])
    
skew_kurt_stats = pd.DataFrame({
    'Skewness': skewness_values,
    'Kurtosis': kurtosis_values
})


print(Desc_Stats)  
print(skew_kurt_stats) 


chi_merge_list = ['VehBCost', 'WarrantyCost']

trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=1, decimal=3,output_dataframe=True)
result_cm = trans_cm.fit_transform(train_FS[chi_merge_list], train_FS['IsBadBuy']) 
trans_cm.boundaries_

boundaries_dict = {key: np.insert(boundaries, 0, -np.inf) for key, boundaries in trans_cm.boundaries_.items()}

for key, boundaries in boundaries_dict.items():
    column_name = f"{key}_cat_cm"
    train_FS[column_name] = pd.cut(train_FS[key], bins=boundaries, labels=False, right=False)
    
    print(f'{column_name} bin edges:', boundaries)
    
    frequency_table(train_FS[column_name])
    print("\n")

train_FS = train_FS.drop(columns=['VehBCost', 'WarrantyCost'])

train_FS.describe()


train_FS.shape


one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

nominal_fields = ["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size","TopThreeAmericanName", "PRIMEUNIT", "AUCGUART"]

one_hot_encoded_FS = one_hot_encoder.fit_transform(train_FS[nominal_fields])

one_hot_encoded_FS = pd.DataFrame(one_hot_encoded_FS, columns=one_hot_encoder.get_feature_names_out(nominal_fields))

train_FS_ = train_FS.drop(columns=nominal_fields)

encoded_FS = pd.concat([train_FS_.reset_index(drop=True), one_hot_encoded_FS.reset_index(drop=True)], axis=1)
encoded_FS.shape


selected_features_FS = [col for col in encoded_FS.columns if col not in ['IsBadBuy']]


scaling_methods = ['min-max'] 


for feature in selected_features_FS:

        min_max_scaler = MinMaxScaler()
        encoded_FS[feature] = min_max_scaler.fit_transform(encoded_FS[[feature]])


encoded_FS.shape


train_FE = pd.read_csv('/kaggle/working/df_train_car.csv')
train_FE.head()


train_FE.set_index('RefId', inplace=True)
train_FE.shape


selected_features_FE = ['VehBCost', 'WarrantyCost']

for feature in selected_features_FE:

    transformer = PowerTransformer(method='box-cox', standardize=False)

    train_FE[f"{feature}_transformed"] = transformer.fit_transform(train_FE[[feature]])

    lambda_value = transformer.lambdas_[0]
    print(f"Lambda for {feature}: {lambda_value}")
    
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.hist(train_FE[feature], bins=30, color='blue', alpha=0.7)
    plt.title(f'Original {feature} Histogram')

    plt.subplot(1, 2, 2)
    plt.hist(train_FE[f"{feature}_transformed"], bins=30, color='green', alpha=0.7)
    plt.title(f'Transformed {feature} Histogram')

    plt.tight_layout()
    plt.show()

train_FE = train_FE.drop(columns=['VehBCost', 'WarrantyCost'])

print('\n')
print(train_FE)


one_hot_encoded_FE = one_hot_encoder.fit_transform(train_FE[nominal_fields])

one_hot_encoded_FE = pd.DataFrame(one_hot_encoded_FE, columns=one_hot_encoder.get_feature_names_out(nominal_fields))

train_FE_ = train_FE.drop(columns=nominal_fields)

encoded_FE = pd.concat([train_FE_.reset_index(drop=True), one_hot_encoded_FE.reset_index(drop=True)], axis=1)
encoded_FE.shape


selected_features_FE = [col for col in encoded_FE.columns if col not in ['IsBadBuy']]


scaling_methods = ['z-score'] 


for feature in selected_features_FE:

        z_score_scaler = StandardScaler()
        encoded_FE[feature] = z_score_scaler.fit_transform(encoded_FE[[feature]])

encoded_FE.shape


encoded_FE.to_csv('/kaggle/working/encoded_FE.csv')




