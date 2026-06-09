import pandas as pd
df=pd.read_csv("/kaggle/input/DontGetKicked/training.csv")
df.info()
df.head()


df=df.drop(["PurchDate","VehYear","Model","Trim","SubModel","WheelTypeID","BYRNO","VNZIP1","VNST"],axis=1)
df=df.set_index("RefId")
df.info()
df.head()


target=df.iloc[: ,0:1]
inputs=df.iloc[: , 1:]


import pandas as pd
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(inputs, target, test_size=0.2, random_state=1)
inputs=x_train



column_ranges={
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
for column, (min_val, max_val) in column_ranges.items():
    out_of_range_count = ((inputs[column] < min_val) | (inputs[column] > max_val)).sum()
    print(f"The number of out-of-range cells in '{column}' variable is: {out_of_range_count}")

for column , (min_val,max_val) in column_ranges.items():
    inputs[column]=inputs[column].apply(lambda x:x if min_val<=x<=max_val else None)
#print(inputs)
inputs.describe()


inputs.info()


import numpy as np
def frequency_table(variable):
    unique_elements,counts=np.unique(variable.dropna(),return_counts=True)
    percentage=(counts/len(variable)*100)
    for i , j , k in zip(unique_elements,counts,percentage):
        print(f"{i} :count{j} , percentage: {k:.2f}")
    return
categorical_fields = [
    "Auction",
    "Make",
    "Color",
    "Transmission",
    "WheelType",
    "Nationality",
    "Size",
    "TopThreeAmericanName",
    "PRIMEUNIT",
    "AUCGUART",
    "IsOnlineSale"
]
continuous_fields = [col for col in inputs.columns if col not in categorical_fields]
for col in categorical_fields:
    print(f"Frequency Table for {col}:")
    frequency_table(inputs[col])
    print("-" * 40)


inputs["Transmission"]=inputs["Transmission"].replace("Manual",'MANUAL')
frequency_table(inputs["Transmission"])
print("-"*40)
inputs["Color"].info()
print("-"*40)
inputs["Color"]=inputs["Color"].replace("NOT AVAIL",np.nan)
inputs["Color"].info()


def replace_rare_classes(df, column, threshold=1):
    unique_elements, counts = np.unique(df[column].dropna(), return_counts=True)
    percentage = (counts / len(df[column])) * 100
    rare_classes = [elem for elem, pct in zip(unique_elements, percentage) if pct < threshold]
    return df[column].replace(rare_classes, 'OTHER')

inputs['Make'] = replace_rare_classes(inputs, 'Make', threshold=1)
print("\nUpdated Frequency Table for 'Make':")
frequency_table(inputs['Make'])


inputs['Color'] = replace_rare_classes(inputs, 'Color', threshold=1)
print("\nUpdated Frequency Table for 'Color':")
frequency_table(inputs['Color'])


print(f"Number of continuous fields before feature screening: {len(continuous_fields)}")
print(f"Number of categorical fields before feature screening: {len(categorical_fields)}")

min_cv=0.1
cv_values=inputs[continuous_fields].std()/inputs[continuous_fields].mean()
selected_columns=cv_values[cv_values<0.1].index
print(f"\nNumber of Features with a coefficient of variation less than **0.1**: {len(selected_columns)}")
filtered_con=inputs[selected_columns]
inputs_con=inputs[continuous_fields].drop(selected_columns,axis=1)

threshold=99
mode_category=(inputs[categorical_fields].apply(lambda x:x.value_counts().max()/len(x)*100))
selected_categorical_columns=mode_category[mode_category>threshold].index
print(f"Number of features where the mode category percentage is greater than **99%**: {len(selected_columns)}")
mode_filtered_inputs=inputs[selected_categorical_columns]
inputs_cat=inputs[categorical_fields].drop(selected_categorical_columns,axis=1)

threshold=90
distinct_percentage=inputs_cat[categorical_fields].apply(lambda x:x.dropna().nunique()/x.count()*100)
selected_categorical_columns=distinct_percentage[distinct_percentage>threshold].index
print(f"Number of Features with a percentage of unique categories exceeding **90%**: {len(selected_columns)}")
distinct_filtered_inputs=inputs_cat[selected_categorical_columns]
inputs_cat=inputs[categorical_fields].drop(selected_categorical_columns,axis=1)

filtered_df=pd.concat([inputs_con,inputs_cat,target],axis=1)

print(f"\nNumber of continuous fields AFTER feature screening: {len(continuous_fields)}")
print(f"Number of categorical fields AFTER feature screening: {len(categorical_fields)}")


#Test Of Indepence
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact

contingency_table=pd.crosstab(filtered_df["PRIMEUNIT"].dropna(),filtered_df["IsBadBuy"])
print("contingency table with the frequencies for PRIMEUNIT variable")
print(contingency_table)
print("=="*50)

#CALCULATE ROW PERCENTAGE
row_pecentage=contingency_table.div(contingency_table.sum(axis=1),axis=0)*100
print("\nRow percentages:")
print(row_pecentage)

#TEST OF INDEPENCE
chi2,p,dof,expected=chi2_contingency(contingency_table)
print(f"\nchi square value: {chi2}")
print(f"p-value:{p}")
print(f"degree of freedom: {dof}")
print(f"expected frequencies:\n {expected}")
print("=="*50)

#CHECK RESIDUALS
percentage_low_expected=(expected<5).sum().sum()/(expected.shape[0]*expected.shape[1])*100
print(f"the percentage of cells with expected values less than 5 is: {percentage_low_expected}")
residuals=contingency_table-expected
print("=="*50)
print(f"residuals(observed -expected):\n{residuals}")

#CALCULATE ODDS RATIO
odds_ratio=(contingency_table.iloc[0,0]/contingency_table.iloc[0,1])/(contingency_table.iloc[1,0]/contingency_table.iloc[1,1])
print("=="*50)
print(f'odds ratio: {odds_ratio:.2f}')


odds_ratio_fisher,p_value_fisher=fisher_exact(contingency_table)
print(f"\nFisher's odds ratio:{odds_ratio_fisher:.2f}")
print("Fisher p-value:",p_value_fisher)



#Test Of Indepence
import pandas as pd
from scipy.stats import chi2_contingency

contingency_table=pd.crosstab(filtered_df["AUCGUART"].dropna(),filtered_df["IsBadBuy"])
print("contingency table with the frequencies for PRIMEUNIT variable")
print(contingency_table)
print("=="*50)

#CALCULATE ROW PERCENTAGE
row_pecentage=contingency_table.div(contingency_table.sum(axis=1),axis=0)*100
print("\nRow percentages:")
print(row_pecentage)

#TEST OF INDEPENCE
chi2,p,dof,expected=chi2_contingency(contingency_table)
print(f"\nchi square value: {chi2}")
print(f"p-value:{p}")
print(f"degree of freedom: {dof}")
print(f"expected frequencies:\n {expected}")
print("=="*50)

#CHECK RESIDUALS
percentage_low_expected=(expected<5).sum().sum()/(expected.shape[0]*expected.shape[1])*100
print(f"the percentage of cells with expected values less than 5 is: {percentage_low_expected}")
residuals=contingency_table-expected
print("=="*50)
print(f"residuals(observed -expected):\n{residuals}")

#CALCULATE ODDS RATIO
odds_ratio=(contingency_table.iloc[0,0]/contingency_table.iloc[0,1])/(contingency_table.iloc[1,0]/contingency_table.iloc[1,1])
print("=="*50)
print(f'odds ratio: {odds_ratio:.2f}')

odds_ratio_fisher,p_value_fisher=fisher_exact(contingency_table)
print(f"\nFisher's odds ratio:{odds_ratio_fisher:.2f}")
print("Fisher p-value:",p_value_fisher)


inputs = inputs.drop(columns=["AUCGUART", "PRIMEUNIT"])
inputs.info()
categorical_fields = [
    "Auction",
    "Make",
    "Color",
    "Transmission",
    "WheelType",
    "Nationality",
    "Size",
    "TopThreeAmericanName",
    "IsOnlineSale"
]


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

inputs_iso=inputs.copy()
inputs_iso=inputs_iso.dropna()

scaler=StandardScaler()
inputs_iso[continuous_fields]=scaler.fit_transform(inputs_iso[continuous_fields])

label_encoder=LabelEncoder()
inputs_iso[categorical_fields]=inputs_iso[categorical_fields].apply(label_encoder.fit_transform)

clf=IsolationForest(contamination=0.01, random_state=42)
clf.fit(inputs_iso)
outliers=clf.predict(inputs_iso)
inputs_iso["outlier"]=outliers
#print(inputs_iso)
percentage_outliers=(outliers[outliers == -1].shape[0]/len(outliers))*100
print(f"Percentage of outliers: {percentage_outliers: .2f}%")

outlier_index=inputs_iso[inputs_iso["outlier"]== -1].index
inputs_outprep=inputs.drop(outlier_index)
y_train_outprep=y_train.drop(outlier_index)
train_outprep=pd.concat([inputs_outprep,y_train_outprep],axis=1)
train_outprep.info()


columns_to_check=["MMRAcquisitionAuctionAveragePrice",
                  "MMRAcquisitionAuctionCleanPrice",
                  "MMRAcquisitionRetailAveragePrice",
                  "MMRAcquisitonRetailCleanPrice",
                  "MMRCurrentAuctionAveragePrice",
                  "MMRCurrentAuctionCleanPrice",
                  "MMRCurrentRetailAveragePrice",
                  "MMRCurrentRetailCleanPrice"]
train_outprep["num_missing_values"]=train_outprep[columns_to_check].isnull().sum(axis=1)
train_outprep=train_outprep[train_outprep["num_missing_values"]<4]
train_outprep.info()


train_outprep["num_missing_values"]=train_outprep.isnull().sum(axis=1)
row_with_missing_values=train_outprep[train_outprep["num_missing_values"]>0]
total_row=len(train_outprep)
row_with_missing_ccount=len(row_with_missing_values)
percentage_row_with_missing=(row_with_missing_ccount/total_row)*100
print(train_outprep.sort_values(by="num_missing_values",ascending=False))
max_missing_values_threshold=11
train_outprep=train_outprep[train_outprep["num_missing_values"]<=max_missing_values_threshold].iloc[:,:-1]


missing_values_report=pd.DataFrame({
    "column":train_outprep.columns,
    "missing_values":train_outprep.isnull().sum(),
    "pecentage":train_outprep.isnull().mean()*100
})
missing_values_report


from sklearn.impute import SimpleImputer

train_outprep_no_missing_fix = train_outprep.copy()

continuous_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")

train_outprep_no_missing_fix[continuous_fields] = continuous_imputer.fit_transform(train_outprep_no_missing_fix[continuous_fields])
train_outprep_no_missing_fix[categorical_fields] = categorical_imputer.fit_transform(train_outprep_no_missing_fix[categorical_fields])
train_outprep_no_missing_fix.info()
train_outprep_no_missing_fix.to_csv('train_outprep.csv')


train_outprep_no_missing_fix.describe()

