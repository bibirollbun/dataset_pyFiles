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


df_new=df.drop(columns=['PurchDate','VehYear','Model', 'Trim' , 'SubModel','WheelTypeID','BYRNO', 'VNZIP1','VNST'])


df_new.set_index('RefId', inplace=True)


y = df_new.iloc[:,0]
X = df_new.iloc[:,1::]

from sklearn.model_selection import train_test_split

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
X_train.shape,X_test.shape


import numpy as np

def initial_preproc(data):
    processed_data = data.copy()
    
    # Convert NaN values to "non-existent" in dataset
    columns_list = ['Transmission', 'Color', 'Make']

    # Replacing "Na" with "non-existent" in specified columns
    processed_data['Color'] = processed_data['Color'].replace(np.nan, 'NOT AVAIL')
    
    
    processed_data['Transmission'] = processed_data['Transmission'].replace(['Manual'],'MANUAL')
    
    for column in ['Color', 'Make']:
        freq = processed_data[column].value_counts(normalize=True)  # Get frequency of each class
        mask = freq[freq >= 0.01].index  # Get categories with frequency >= 1%
        processed_data[column] = processed_data[column].apply(lambda x: x if x in mask else 'OTHER')
   
    
    return processed_data


X_train = initial_preproc(X_train)
X_test = initial_preproc(X_test)

X_train.shape, X_test.shape


columns = X_train.columns

# Choose categorical elements 
categorical_indices = [0,2,3,4,5,7,8,9,18,19,21]

# Use a list comprehension to select the elements at the specified indices
categorical_fields = [columns[i] for i in categorical_indices]

# Create a new list of columns excluding categorical_fields (continuous)
continuous_fields = [j for j in columns if j not in categorical_fields]


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


frequency_table(X_train['Color'])


def feature_screening(data, min_cv=0.1, mode_threshold=99, distinct_threshold=90):
    processed_data = data.copy()
    
    # Define a minimum value for coefficient of variation
    min_cv = min_cv

    # Calculate the coefficient of variation for each column
    cv_values = processed_data[continuous_fields].std() / processed_data[continuous_fields].mean()

    # Filter out columns with CV less than 0.1
    screen_cv =  cv_values[cv_values < 0.1].index.tolist()



    # Define a threshold for the dominant category percentage
    threshold = mode_threshold

    # Calculate the percentage of the mode category for each column
    mode_category = (processed_data[categorical_fields].apply(lambda x: x.value_counts().max() / len(x)) * 100)

    # Select columns where the mode category percentage is greater than the threshold
    screen_mode = mode_category[mode_category > threshold].index.tolist()



    # Set a threshold for excluding columns 
    threshold = distinct_threshold

    # Calculate the percentage of distinct categories in categorical variables
    distinct_percentage = (processed_data[categorical_fields].apply(lambda x: x.dropna().nunique() / x.count()) * 100)

    # Select categorical columns based on distinct percentage threshold
    screen_distinct = distinct_percentage[distinct_percentage > threshold].index.tolist()

    screened_features  = list(set(screen_cv + screen_mode + screen_distinct))
    
    return screened_features


drop_list = feature_screening(X_train, min_cv=0.1, mode_threshold=95, distinct_threshold=90)

X_train = X_train.drop(drop_list, axis=1)
X_test = X_test.drop(drop_list, axis=1)

X_train.shape, X_test.shape


import pandas as pd

def range_consistency(data, target):
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
        data[column] = data[column].apply(lambda x: x if min_val <= x <= max_val else None)


  
    
    return data, target


X_train = range_consistency(X_train, y_train)[0]
X_test = range_consistency(X_test, y_test)[0]

y_train = range_consistency(X_train, y_train)[1]
y_test = range_consistency(X_test, y_test)[1]

X_train.info()


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

def outlier_handling(data, contamination=0.01):
    inputs_iso = data.copy()
    
    categorical = inputs_iso.select_dtypes(include=['object']).columns.tolist()
    continuous = inputs_iso.select_dtypes(exclude=['object']).columns.tolist()

    # Replace rows with NaN valuse with mean and mode
    for col in inputs_iso.columns:
        if col in continuous:
            inputs_iso[col] = inputs_iso[col].fillna(inputs_iso[col].mean())
        elif col in categorical:
            mode_val = inputs_iso[col].mode().iloc[0]  # Extract mode value
            inputs_iso[col] = inputs_iso[col].fillna(mode_val)


    ordinal = []

    nominal = ['Auction','Make','Size','Color','WheelType','Nationality','TopThreeAmericanName','PRIMEUNIT','AUCGUART']

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
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(inputs_iso_encoded_array)

    # Predict outliers
    outliers = clf.predict(inputs_iso_encoded_array)

    # Add the outlier predictions to your DataFrame
    inputs_iso_encoded['outlier'] = outliers

    outlier_index = inputs_iso_encoded[inputs_iso_encoded['outlier'] == -1]['RefId']
    return outlier_index


outlier_index = outlier_handling(X_train, contamination=0.01)

X_train = X_train.drop(outlier_index.tolist())

y_train = y_train.drop(outlier_index.tolist())

X_train.shape, y_train.shape


def missing_row(data,price_cols, missrow=4):
    processed_data = data.copy()

   
    # Create a new column with the number of missing values in each row
    processed_data['Num_Missing_Values'] = processed_data[price_cols].isnull().sum(axis=1)
    
    # Filter out rows with missing values greater than or equal to missrow
    processed_data = processed_data[processed_data['Num_Missing_Values'] < missrow]
    
    # Optionally, drop the 'Num_Missing_Values' column if you no longer need it
    processed_data = processed_data.drop(columns=['Num_Missing_Values'])
                                                                       
                                                                       
 

    return processed_data


price_cols = [
    'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']

discard_missing_row = missing_row(X_train,price_cols, missrow=4)


indices_to_discard = X_train.index[~X_train.index.isin(discard_missing_row.index)]


X_train = X_train.drop(indices_to_discard)
y_train = y_train.drop(indices_to_discard)

X_train.shape, y_train.shape


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


discard_missing_col = missing_col_report(X_train, misscol=50)

X_train = X_train.drop(discard_missing_col, axis=1)
X_test = X_test.drop(discard_missing_col, axis=1)

X_train.shape, X_test.shape


from sklearn.impute import KNNImputer, SimpleImputer

def missing_imputer(train, test):
    
    continuous = train.select_dtypes(exclude=['object','category']).columns.tolist()
    categorical = train.select_dtypes(include=['object','category']).columns.tolist()

    # Define imputation strategies for each subset of columns
    knn_imputer = KNNImputer()
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Impute missing values
    train[continuous] = knn_imputer.fit_transform(train[continuous])
    train[categorical] = cat_imputer.fit_transform(train[categorical])

    test[continuous] = knn_imputer.transform(test[continuous])
    test[categorical] = cat_imputer.transform(test[categorical])
    
    return train, test


X_train, X_test = missing_imputer(X_train, X_test )

X_train.shape, X_test.shape


from sklearn.preprocessing import PowerTransformer

# List of features to transform
selected_features = ['VehBCost', 'WarrantyCost']

def transform(train, test, trans_list):
    # Iterate through selected features
    for feature in trans_list:
        # Check if the feature contains negative values
        has_negative_values = (train[feature] <= 0).any()

        # Choose the appropriate transformation method
        if has_negative_values:
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        else:
            transformer = PowerTransformer(method='box-cox', standardize=False)

        # Fit and transform the feature, and store the result in the new DataFrame
        train[f"{feature}_transformed"] = transformer.fit_transform(train[[feature]])
        test[f"{feature}_transformed"] = transformer.transform(test[[feature]])
        
    return train, test


X_train, X_test = transform(X_train, X_test, selected_features)

X_train.shape, X_test.shape


continuous = X_train.select_dtypes(exclude=['object','category']).columns.tolist()
categorical = X_train.select_dtypes(include=['object','category']).columns.tolist()

nominal = ["Auction", "Make", "Color", "WheelType", "Nationality", "Size", "TopThreeAmericanName"]

ordinal = [i for i in categorical if i not in nominal]


len(continuous), len(categorical), len(ordinal), len(nominal)


from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# فقط روی ستون‌های عددی اعمال می‌کنیم
X_train_scaled[continuous] = scaler.fit_transform(X_train[continuous])
X_test_scaled[continuous] = scaler.transform(X_test_scaled[continuous])


from sklearn.decomposition import PCA
import pandas as pd

# PCA روی داده‌های عددی
pca = PCA(n_components=2, random_state=717)
pca.fit(X_train_scaled[continuous])

# خروجی PCA
X_train_pca = pd.DataFrame(pca.transform(X_train_scaled[continuous]),
                           columns=[f'pc_{i+1}' for i in range(pca.n_components_)],
                           index=X_train.index)
X_test_pca = pd.DataFrame(pca.transform(X_test_scaled[continuous]),
                          columns=[f'pc_{i+1}' for i in range(pca.n_components_)],
                          index=X_test.index)


from sklearn.preprocessing import OneHotEncoder



one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

X_train_nom = pd.DataFrame(one_hot_encoder.fit_transform(X_train_scaled[nominal]),
                           columns=one_hot_encoder.get_feature_names_out(nominal),
                           index=X_train.index)
X_test_nom = pd.DataFrame(one_hot_encoder.transform(X_test_scaled[nominal]),
                          columns=one_hot_encoder.get_feature_names_out(nominal),
                          index=X_test.index)


X_train_final = pd.concat([X_train_pca, X_train_nom], axis=1)
X_test_final = pd.concat([X_test_pca, X_test_nom], axis=1)

print(X_train_final.shape, X_test_final.shape)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

print(X_train_scaled.shape, X_test_scaled.shape)


import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier

# اگر خروجی اسکیل numpy هست، تبدیل به DataFrame برای نگه داشتن اسم فیچرها
X_train_df = pd.DataFrame(X_train_scaled, columns=X_train_final.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_test_final.columns)

# مدل پایه (چون هدف دوحالته است)
estimator = DecisionTreeClassifier(random_state=29)

# تعریف RFECV (روش Wrapper)
selector = RFECV(
    estimator=estimator,
    step=1,
    min_features_to_select=10,
    cv=5,
    scoring='accuracy',   # چون classification است
    n_jobs=-1
)

# آموزش روی train
selector.fit(X_train_df, y_train)

print("="*50)
print(f"Optimal number of features: {selector.n_features_}")
print("="*50)

# لیست فیچرهای منتخب
wrapper_fs = selector.get_feature_names_out()

print("Wrapper Optimal Feature List:")
print(wrapper_fs)

# ساخت دیتاست جدید با فیچرهای انتخاب‌شده
X_train_wrapper = X_train_df[wrapper_fs]
X_test_wrapper = X_test_df[wrapper_fs]

print("="*50)
print("New Shapes After Wrapper FS:")
print(X_train_wrapper.shape, X_test_wrapper.shape)



X_train_wrapper.to_csv('X_train_wrapper.csv', index=True)
X_test_wrapper.to_csv('X_test_wrapper.csv', index=True)

y_train.to_csv('y_train.csv', index=True)

# اگر y_test هم داری:
y_test.to_csv('y_test.csv', index=True)

