import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df = df.set_index('RefId')
df.info()
df.head()


import pandas as pd
from sklearn.model_selection import train_test_split

target = df.iloc[:,0]
inputs = df.iloc[:,1:] 

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size= 0.20, random_state =1 )

inputs = x_train

x_train.shape,x_test.shape


import numpy as np

def initial_preproc(data):
    
    processed_data = data.copy()
    processed_data = processed_data.drop(['PurchDate','VehYear','Model','Trim','SubModel','WheelTypeID','BYRNO','VNZIP1','VNST' ,'PRIMEUNIT',
    'AUCGUART'], axis = 1)

    processed_data['Transmission'] = processed_data['Transmission'].replace('Manual', 'MANUAL')
    processed_data['Color'] = processed_data['Color'].replace(['NOT AVAIL'], None)
    
    return processed_data


x_train = initial_preproc(x_train)
x_test = initial_preproc(x_test)

x_train.shape, x_test.shape


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
continuous_fields = [col for col in x_train.columns if col not in categorical_fields]


def feature_screening(data, min_cv=0.1, mode_threshold=99, distinct_threshold=90):
    
    min_cv = 0.1

    cv_values = data[continuous_fields].std() / data[continuous_fields].mean()
    screen_cv = cv_values[cv_values < 0.1].index.tolist()
    
# *****************************************

    threshold = 99

    mode_category = data[categorical_fields].apply(lambda x: x.value_counts().max() / len(x) * 100)
    screen_mode = mode_category[mode_category > threshold].index.tolist()
    
# *****************************************

    threshold = 90

    distinct_percentage = (data[categorical_fields]. apply(lambda x: x.dropna().nunique()/x.count()) * 100)
    screen_distinct = distinct_percentage[distinct_percentage > threshold].index.tolist()

# *****************************************
   
    screened_features  = list(set(screen_cv + screen_mode + screen_distinct))
    return screened_features


drop_list = feature_screening(x_train, min_cv=0.1, mode_threshold=99, distinct_threshold=90)

x_train = x_train.drop(drop_list, axis=1)
x_test = x_test.drop(drop_list, axis=1)

x_train.shape, x_test.shape


import pandas as pd

def range_consistency(data, target):
    

     column_ranges = {'VehicleAge': (0,30), 'VehOdo': (0,120000), 'MMRAcquisitionAuctionAveragePrice': (800,46000),
                'MMRAcquisitionAuctionCleanPrice': (1000,46000),'MMRAcquisitionRetailAveragePrice': (1000,46000),
                'MMRAcquisitonRetailCleanPrice': (1000,46000),'MMRCurrentAuctionAveragePrice': (300,46000),
                'MMRCurrentAuctionCleanPrice': (400,46000),'MMRCurrentRetailAveragePrice': (800,46000),
                'MMRCurrentRetailCleanPrice': (1000,46000),'VehBCost': (1000,46000),'WarrantyCost': (400,8000)}

     for column, (min_val,max_val) in column_ranges.items():
         df[column] = df[column].apply(lambda x: x if min_val <= x <= max_val else None)
        
     target = target.replace([':0', "'0'"], '0')
    
     return data, target


x_train = range_consistency(x_train, y_train)[0]
x_test = range_consistency(x_test, y_test)[0]

y_train = range_consistency(x_train, y_train)[1]
y_test = range_consistency(x_test, y_test)[1]

x_train.shape, x_test.shape


def replace_rare_classes(df, column, threshold=1):
    unique_elements, counts = np.unique(df[column].dropna(), return_counts=True)
    percentage = (counts / len(df[column])) * 100
    rare_classes = [elem for elem, pct in zip(unique_elements, percentage) if pct < threshold]
    return df[column].replace(rare_classes, 'OTHER')


def replace_rare(data):

     data['Make'] = replace_rare_classes(data, 'Make', threshold=1)
     data['Color'] = replace_rare_classes(data, 'Color', threshold=1)
     return data


x_train = replace_rare(x_train)
x_test = replace_rare(x_test)

x_train.shape, x_test.shape


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

def outlier_handling(data, contamination=0.01):

    inputs_iso = data.copy()

    inputs_iso = inputs_iso.dropna()

    scaler = StandardScaler()
    inputs_iso[continuous_fields] = scaler.fit_transform(inputs_iso[continuous_fields])


    lable_encoder =LabelEncoder()
    inputs_iso[categorical_fields] = inputs_iso[categorical_fields].apply(lable_encoder.fit_transform)

    clf = IsolationForest(contamination = 0.01 , random_state = 42)
    clf.fit(inputs_iso)
    
    outliers = clf.predict(inputs_iso)
    inputs_iso['outliers'] = outliers

    outlier_index = inputs_iso[inputs_iso['outliers'] == -1].index
    
    return outlier_index


outlier_index = outlier_handling(x_train, contamination=0.01)

x_train = x_train.drop(outlier_index.tolist())

y_train = y_train.drop(outlier_index.tolist())

x_train.shape


columns_to_check=["MMRAcquisitionAuctionAveragePrice",
                  "MMRAcquisitionAuctionCleanPrice",
                  "MMRAcquisitionRetailAveragePrice",
                  "MMRAcquisitonRetailCleanPrice",
                  "MMRCurrentAuctionAveragePrice",
                  "MMRCurrentAuctionCleanPrice",
                  "MMRCurrentRetailAveragePrice",
                  "MMRCurrentRetailCleanPrice"]
x_train["num_missing_values"]=x_train[columns_to_check].isnull().sum(axis=1)
valid_indices = x_train[x_train["num_missing_values"] < 4].index
x_train = x_train.loc[valid_indices].drop(columns=["num_missing_values"])
y_train = y_train.loc[valid_indices]
x_train.shape, y_train.shape


def missing_row_report(data, missrow=11):
    processed_data = data.copy()

    # Create a new column with the number of missing values in each row
    processed_data['Num_Missing_Values'] = processed_data.isnull().sum(axis=1)

    discard_missing_row = processed_data[processed_data['Num_Missing_Values'] > missrow].index.tolist()

    return discard_missing_row


discard_missing_row = missing_row_report(x_train, missrow=5)

x_train = x_train.drop(discard_missing_row)
y_train = y_train.drop(discard_missing_row)

x_train.shape, y_train.shape


def missing_col_report(data, misscol=50):
    processed_data = data.copy()

    # Report on count and percentage of missing values in each column
    missing_values_report = pd.DataFrame({
        'Column': processed_data.columns,
        'Missing Values': processed_data.isnull().sum(),
        'Percentage Missing': processed_data.isnull().mean() * 100
    })
    discard_missing_col = missing_values_report[missing_values_report['Percentage Missing'] > misscol].index.tolist()
    
    return discard_missing_col


discard_missing_col = missing_col_report(x_train, misscol=50)

x_train = x_train.drop(discard_missing_col, axis=1)
x_test = x_test.drop(discard_missing_col, axis=1)

x_train.shape, x_test.shape


from sklearn.impute import SimpleImputer


# Define imputation strategies for each subset of columns

cat_simple_list = [i for i in categorical_fields] 
con_simple_list = [i for i in continuous_fields]

def missing_imputer(train, test, cat_list, con_list):
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    con_imputer = SimpleImputer(strategy='median')

    # Impute missing values
    train[cat_list] = cat_imputer.fit_transform(train[cat_list])
    train[con_list] = con_imputer.fit_transform(train[con_list])

    test[cat_list] = cat_imputer.transform(test[cat_list])
    test[con_list] = con_imputer.transform(test[con_list])
    
    return train, test


x_train, x_test = missing_imputer(x_train, x_test, cat_simple_list, con_simple_list)

x_train.info()
x_test.info()


pip install scorecardbundle


from scorecardbundle.feature_discretization import ChiMerge as cm

chi_merge_list =['VehBCost', 'WarrantyCost']

def discretizer(train, test, y, chi_list):  
    
    trans_cm =cm.ChiMerge(max_intervals = 5 , min_intervals = 2 , decimal = 3, output_dataframe = True)
    trans_cm.fit_transform(train[chi_list], y.astype('int'))


    
    boundaries_dict = {key : np.insert(boundaries , 0 , -np.inf) for key , boundaries in trans_cm.boundaries_.items()}

    for key , boundaries in boundaries_dict.items() :
        column_name = f"{key}_cat_cm"
        train[column_name] = pd.cut(train[key], bins = boundaries, labels = False , right = False )
        test[column_name] = pd.cut(test[key], bins = boundaries, labels = False , right = False )
        

    #train = train.drop(['VehBCost','WarrantyCost'], axis = 1)
    #test = test.drop(['VehBCost','WarrantyCost'], axis = 1)
    return train, test


x_train, x_test = discretizer(x_train, x_test, y_train, chi_merge_list)

x_train.shape, x_test.shape


x_train.info()


from sklearn.preprocessing import PowerTransformer

selected_features =['VehBCost', 'WarrantyCost']

def transform(train, test, trans_list):
    
    for feature in trans_list:
       
         transformer = PowerTransformer(method = 'box-cox', standardize = False)
         train[f'{feature}-transformed'] = transformer.fit_transform(train[[feature]])
         test[f'{feature}-transformed'] = transformer.transform(test[[feature]])

    return train, test


x_train, x_test = transform(x_train, x_test, selected_features)

x_train.shape, x_test.shape


x_train.info()


label_encoded_list = ["Auction", "Make", "Color", "Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"]


le = LabelEncoder()
for col in label_encoded_list:
    x_train[col] = le.fit_transform(x_train[col])
    x_test[col] = le.transform(x_test[col])


x_train.info()


import pandas as pd
from sklearn.preprocessing import StandardScaler
    
# Apply Z-Score Scaling
z_score_scaler = StandardScaler()
x_train[x_train.columns.tolist()] = z_score_scaler.fit_transform(x_train)
x_test[x_test.columns.tolist()] = z_score_scaler.transform(x_test)


x_train.info()



original_features = [ 'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 
                         'MMRAcquisitionAuctionCleanPrice' , 'MMRAcquisitionRetailAveragePrice',  
                         'MMRAcquisitonRetailCleanPrice' , 'MMRCurrentAuctionAveragePrice' ,     
                         'MMRCurrentAuctionCleanPrice' , 'MMRCurrentRetailAveragePrice' ,      
                         'MMRCurrentRetailCleanPrice', 'IsOnlineSale' ,                      
                         'VehBCost', 'WarrantyCost', 'Auction', 'Make', 'Color', 'Transmission',
                         'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName']

discretized_features = [ 'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 
                         'MMRAcquisitionAuctionCleanPrice' , 'MMRAcquisitionRetailAveragePrice',  
                         'MMRAcquisitonRetailCleanPrice' , 'MMRCurrentAuctionAveragePrice' ,     
                         'MMRCurrentAuctionCleanPrice' , 'MMRCurrentRetailAveragePrice' ,      
                         'MMRCurrentRetailCleanPrice', 'IsOnlineSale' ,                      
                         'VehBCost_cat_cm', 'WarrantyCost_cat_cm',  
                         'Auction', 'Make', 'Color', 'Transmission',
                         'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName']

transformed_features = [ 'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 
                         'MMRAcquisitionAuctionCleanPrice' , 'MMRAcquisitionRetailAveragePrice',  
                         'MMRAcquisitonRetailCleanPrice' , 'MMRCurrentAuctionAveragePrice' ,     
                         'MMRCurrentAuctionCleanPrice' , 'MMRCurrentRetailAveragePrice' ,      
                         'MMRCurrentRetailCleanPrice', 'IsOnlineSale' ,                      
                         'VehBCost_cat_cm', 'WarrantyCost_cat_cm' , 'VehBCost-transformed' ,              
                         'WarrantyCost-transformed',                 
                         'Auction', 'Make', 'Color', 'Transmission',
                         'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName']

def scenario(data, scen_list):
    return data[scen_list]


x_train_original = scenario(x_train, original_features)
x_test_original = scenario(x_test, original_features)

x_train_discretized = scenario(x_train, discretized_features)
x_test_discretized = scenario(x_test, discretized_features)

x_train_transformed = scenario(x_train, transformed_features)
x_test_transformed = scenario(x_test, transformed_features)


x_train_original.to_csv('X_train_original.csv', index=True)
x_train_discretized.to_csv('X_train_discretized.csv', index=True)
x_train_transformed.to_csv('X_train_transformed.csv', index=True)
x_test_original.to_csv('X_test_original.csv', index=True)
x_test_discretized.to_csv('X_test_discretized.csv', index=True)
x_test_transformed.to_csv('X_test_transformed.csv', index=True)
y_train.to_csv('y_train.csv', index=True)
y_test.to_csv('y_test.csv', index=True)

