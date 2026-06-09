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


# Check if scikit-learn is installed
try:
    import sklearn
except ImportError:
    !pip install --upgrade scikit-learn

# Check if scorecardbundle is installed
try:
    import scorecardbundle
except ImportError:
    !pip install scorecardbundle




import pandas as pd

df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.set_index('RefId', inplace=True, drop=True)


def drop_columns(df):

    # Define the list of columns to drop
    columns_to_drop = [
        'PurchDate',  # Dates need to be transformed for analysis
        'VehYear',  # "VehicleAge" is a better alternative
        'Model', 'Trim', 'SubModel',  # Too many classes, requires domain expertise for merging
        'WheelTypeID',  # "WheelType" is already present
        'BYRNO',  # Just an ID
        'VNZIP1', 'VNST'  # Location data may not contribute significantly to prediction
    ]
    
    try:
        # Check which columns exist in the DataFrame
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if not existing_columns:
            return df  # Return the original DataFrame if no columns exist to drop
        
        # Drop the existing columns
        return df.drop(columns=existing_columns, inplace=False)
    except Exception as e:
        return df  # Return the original DataFrame in case of an error


df = drop_columns(df)


import numpy as np

y = df.IsBadBuy
X = df.drop('IsBadBuy', axis=1)

from sklearn.model_selection import train_test_split


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_train.shape,X_test.shape




# X_train.info()



import numpy as np

def initial_preproc(data):
    processed_data = data.copy()

    

    
    # Define ranges for each column
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

    # Iterate through each column and replace values outside the range with None
    for column, (min_val, max_val) in column_ranges.items():
        processed_data[column] = processed_data[column].apply(lambda x: x if min_val <= x <= max_val else None)
    

    # Additional transformations
    processed_data['Transmission'] = processed_data['Transmission'].replace('Manual', 'MANUAL')
        
    # Replace 'NOT AVAIL' with pd.NA (pandas' missing value)
    processed_data['Color'] = processed_data['Color'].replace('NOT AVAIL', np.nan)

    threshold = 0.01
    for column in ['Color', 'Make']:
        freq = processed_data[column].value_counts(normalize=True)  # Calculate relative frequency of each value
        rare_classes = freq[freq < threshold].index  # Find values with less than the given threshold
        processed_data[column] = processed_data[column].apply(lambda x: 'OTHER' if x in rare_classes else x)
    
    return processed_data


X_train  = initial_preproc(X_train)
X_test = initial_preproc(X_test)

X_train.shape, X_test.shape




def feature_screening(data, min_cv=0.1, mode_threshold=99, distinct_threshold=90):
    processed_data = data.copy()
    
    categorical_fields = processed_data.select_dtypes(include=['object','category']).columns.tolist()
    continuous_fields = processed_data.select_dtypes(exclude=['object','category']).columns.tolist()
    
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


drop_list_2 = feature_screening(X_train, min_cv=0.1, mode_threshold=99, distinct_threshold=90)

X_train = X_train.drop(drop_list_2, axis=1)
X_test = X_test.drop(drop_list_2, axis=1)

X_train.shape, X_test.shape




drop_list_3= ['PRIMEUNIT', 'AUCGUART']
X_train= X_train.drop(columns=drop_list_3, axis=1)
X_test= X_test.drop(columns=drop_list_3, axis=1)




import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

def outlier_handling(data, contamination=0.01):
    #Make a copy of the inputs data
    inputs_iso = data.copy()

    # Separate numerical and categorical data
    continuous_fields = inputs_iso.select_dtypes(include=['int64', 'float64']).columns
    categorical_fields = inputs_iso.select_dtypes(include=['object']).columns
    # Replace rows with NaN valuse with mean and mode
    for col in inputs_iso.columns:
         if col in continuous_fields:
             inputs_iso[col] = inputs_iso[col].fillna(inputs_iso[col].mean())
         elif col in categorical_fields:
             mode_val = inputs_iso[col].mode().iloc[0]  # Extract mode value
             inputs_iso[col] = inputs_iso[col].fillna(mode_val)
                


    one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(inputs_iso[categorical_fields])

    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out())

    inputs_iso_encoded = pd.concat([one_hot_encoded_df, inputs_iso[continuous_fields].reset_index(drop=True)], axis=1) 



    # Apply Z-score scaling to columns
    scaler = StandardScaler()
    inputs_iso_encoded_array = scaler.fit_transform(inputs_iso_encoded)
    # Step 3: Fit Isolation Forest model
    clf = IsolationForest(contamination=0.01, random_state=42)

    # Here, use the actual values (NumPy array) for fitting
    clf.fit(inputs_iso_encoded_array)
    
    # Predict outliers
    outliers = clf.predict(inputs_iso_encoded_array)
    
    # Step 4: Add the outlier predictions to your DataFrame
    inputs_iso['outlier'] = outliers
    
    outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index
    
    return outlier_index


outlier_index = outlier_handling(X_train, contamination=0.01)

X_train = X_train.drop(outlier_index.tolist())

y_train = y_train.drop(outlier_index.tolist())

X_train.shape, y_train.shape




def missing_prices_row_report(data, max_missing_values_threshold = 4 ):
    processed_data = data.copy()

    # List of relevant price-related columns
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

   
    # Count missing values in the price-related columns for each row and store it in a new column
    processed_data['price_missing_counts'] = processed_data[price_columns].isnull().sum(axis=1)

    # Filter rows where the number of missing values in price-related columns is less than or equal to the threshold
    processed_data = processed_data[processed_data['price_missing_counts'] < max_missing_values_threshold]

    # Drop the 'price_missing_counts' column (optional) if no longer needed
    processed_data = processed_data.drop(columns=['price_missing_counts'])
    return processed_data




X_train = missing_prices_row_report(X_train, max_missing_values_threshold = 4 )
y_train = y_train.loc[X_train.index]  # Use the same indices to keep them aligned


X_train.shape, y_train.shape


from sklearn.impute import KNNImputer, SimpleImputer

def missing_imputer(train, test):
    

    numeric_columns = train.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = train.select_dtypes(include=['object']).columns
    
    # Create an imputer for numeric columns with 'median' strategy (or 'mean' if preferred)
    numeric_imputer = SimpleImputer(strategy='median')
    
    # Create an imputer for categorical columns with 'most_frequent' strategy
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Apply imputer for numeric columns
    train[numeric_columns] = numeric_imputer.fit_transform(train[numeric_columns])
    
    # Apply imputer for categorical columns
    train[categorical_columns] = categorical_imputer.fit_transform(train[categorical_columns])

    # Impute missing values
    train[numeric_columns] = numeric_imputer.fit_transform(train[numeric_columns])
    train[categorical_columns] = categorical_imputer.fit_transform(train[categorical_columns])

    test[numeric_columns] = numeric_imputer.transform(test[numeric_columns])
    test[categorical_columns] = categorical_imputer.transform(test[categorical_columns])
    
    return train, test




X_train, X_test = missing_imputer(X_train, X_test)

X_train.isnull().sum().sum()


X_train.shape, y_train.shape




# X_train.head()


X_train.to_csv('X_train_orginal', index=True )
X_test.to_csv('X_test_orginal', index=True )
y_train.to_csv('y_train_orginal', index=True )
y_test.to_csv('y_test_orginal', index=True )




X_train_trans= X_train.copy()
X_test_trans= X_test.copy()
y_train_trans= y_train.copy()


from sklearn.preprocessing import PowerTransformer


def transform(train, test):

    transform_list= ['VehBCost', 'WarrantyCost']

    # Iterate through selected features
    for feature in transform_list:
        # Check if the feature contains negative values
        has_negative_values = (train[feature] <= 0).any()

        # Choose the appropriate transformation method
        if has_negative_values:
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        else:
            transformer = PowerTransformer(method='box-cox', standardize=False)

        # Fit and transform the features
        train[feature] = transformer.fit_transform(train[[feature]])
        test[feature] = transformer.transform(test[[feature]])
    
    return train, test


X_train_trans, X_test_trans = transform(X_train_trans, X_test_trans)

X_train_trans.shape, X_test_trans.shape




continuous = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist() 

categorical= X_train.select_dtypes(include=['object','category']).columns.tolist()


len(continuous), len(categorical)




# X_train.head()


# X_train_kmerge= X_train.copy()
# X_test_kmerge= X_test.copy()
# y_train_kmerge= y_train.copy()






# import pandas as pd
# import numpy as np
# from scorecardbundle.feature_discretization import ChiMerge as cm
# from sklearn.model_selection import train_test_split

# def chi_merge_discretization(train, test, target_col, max_intervals=5, min_intervals=5, decimal=3):
#     decimal = int(decimal)

#     # Define the list of features to discretize
#     chi_merge_list = ['VehBCost', 'WarrantyCost']
    
#     # Initialize ChiMerge
#     trans_cm = cm.ChiMerge(max_intervals=max_intervals, min_intervals=min_intervals, decimal=decimal, output_dataframe=True)

#     # Fit ChiMerge on training data with the target column as y_train
#     result_cm = trans_cm.fit_transform(train[chi_merge_list], target_col)

#     # Extract bin boundaries from the fitted result
#     boundaries_dict = {key: np.insert(boundaries, 0, -np.inf) for key, boundaries in trans_cm.boundaries_.items()}

#     # Apply discretization on training and test data
#     for key, boundaries in boundaries_dict.items():
#         # Discretize and replace the original column with discretized values
#         train[key] = pd.cut(train[key], bins=boundaries, labels=range(1, len(boundaries)), right=False)
#         test[key] = pd.cut(test[key], bins=boundaries, labels=range(1, len(boundaries)), right=False)

#     return train, test






# # Apply the function to training and test datasets, passing y_train as the target_col
# X_train_kmerge, X_test_kmerge = chi_merge_discretization(X_train_kmerge, X_test_kmerge, y_train_kmerge)






# X_train.head()






# X_train_kmerge.head()






# X_train_kmerge.shape, y_train_kmerge.shape, y_train.shape




# continuous_kmerge = X_train_kmerge.select_dtypes(exclude=['object','category']).columns.tolist()
# categorical_kmerge = X_train_kmerge.select_dtypes(include=['object','category']).columns.tolist()

# ordinal_kmerge = ['VehBCost', 'WarrantyCost']

# nominal_kmerge = [i for i in categorical_kmerge if i not in ordinal_kmerge]


# len(continuous_kmerge), len(categorical_kmerge), len(ordinal_kmerge), len(nominal_kmerge)




# nominal_kmerge




from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer


from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, f_classif, mutual_info_classif, RFECV
from sklearn.decomposition import PCA, KernelPCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import xgboost as xgb




# import random
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from imblearn.over_sampling import RandomOverSampler 

# # Function to create the model
# def create_model():


#     seed_value = 117
#     random.seed(seed_value)

#     # Perform random oversampling
#     ros = RandomOverSampler(random_state=0)
#     X_train_ros, y_train_ros = ros.fit_resample(X_train_trans, y_train_trans)
    
#     model = Sequential()
#     model.add(Dense(units=120, activation='sigmoid', input_shape=(X_train_ros.shape[1],)))
#     model.add(Dense(units=72, activation='sigmoid'))
#     model.add(Dense(units=56, activation='sigmoid'))
#     model.add(Dense(units=16, activation='sigmoid'))
#     model.add(Dense(units=1, activation='sigmoid'))  # Output layer
    
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['f1_score'])
#     return model
# Wrap the model using KerasClassifier
# keras_model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=100, verbose=1)


wrapper = RFECV(estimator=LinearRegression(), step=1, min_features_to_select=32, cv=5, n_jobs=-1)



one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

z_score = StandardScaler()
min_max = MinMaxScaler()
# wrapper = RFECV(estimator=DecisionTreeRegressor(random_state=29), step=1, min_features_to_select=5, scoring='r2', cv=5, n_jobs=-1)

pca = PCA(n_components=2, random_state=717)
kpca = KernelPCA(n_components=3, kernel='rbf', random_state=717)
lda= lda = LinearDiscriminantAnalysis(n_components=1)


# # Define the preprocessing steps for numerical and categorical features separately

# numerical_preprocessing_11 = Pipeline(steps=[
#     ('scaler', min_max)  # Scale numerical features
#     ])  

# nominal_preprocessing_11 = Pipeline(steps=[
#     ('encoder', one_hot_encoder),  # One-hot encode nominal features
#     ('scaler', min_max)])  # One-hot encode ordinal features

# ordinal_preprocessing_11 = Pipeline(steps=[
#     ('scaler', min_max)])  # Scale numerical features



# # Define the ColumnTransformer for numerical and categorical features
# preprocessor_11 = ColumnTransformer(transformers=[
#     ('num', numerical_preprocessing_11, continuous_kmerge),
#     ('nom', nominal_preprocessing_11, nominal_kmerge),
#     ('ord', ordinal_preprocessing_11, ordinal_kmerge)
# ], remainder='passthrough')  # Passthrough any columns not specified


# pipeline_11 = Pipeline(steps=[
#     ('preprocessor', preprocessor_11),
#     ('wrapper', wrapper),
#     ('model', DecisionTreeClassifier(random_state=1))  
# ])


# # Train the pipeline
# pipe_11 = pipeline_11.fit(X_train_kmerge, y_train_kmerge)
# pipe_11[:-1].get_feature_names_out().tolist()


# from sklearn.metrics import accuracy_score

# predictions_11 = pipe_11.predict(X_test_kmerge)
# accuracy11 = accuracy_score(y_test, predictions_11)
# print("Accuracy:", accuracy11)




# train_scen_11 = pipe_11[:-1].fit_transform(X_train_kmerge, y_train_kmerge)
# train_scen_11 = pd.DataFrame(train_scen_11, columns=pipe_11[:-1].get_feature_names_out(), index=X_train_kmerge.index)

# test_scen_11 = pipe_11[:-1].transform(X_test_kmerge)
# test_scen_11 = pd.DataFrame(test_scen_11, columns=pipe_11[:-1].get_feature_names_out(), index=X_test_kmerge.index)

# X_hp_data = pd.concat((train_scen_11, test_scen_11), keys=['train', 'test'])
# y_hp_data = pd.concat((y_train_kmerge, y_test), keys=['train', 'test'])

# X_hp_data.to_csv('X_kmerge_minmax_wrapper', index=True)
# y_hp_data.to_csv('y_kmerge_minmax_wrapper', index=True)






# numerical_preprocessing_22 = Pipeline(steps=[
#     ('scaler', min_max),
#     ('pca', pca)
# ])


# categorical_preprocessing_22 = Pipeline(steps=[
#     ('encoder', one_hot_encoder),
#     ('scaler', min_max)
# ])

# preprocessor_22 = ColumnTransformer(
#     transformers=[
#         ('num', numerical_preprocessing_22, continuous),
#         ('cat', categorical_preprocessing_22, categorical)],
#     remainder='passthrough'
# )

# pipeline_22 = Pipeline(steps=[
#     ('preprocessor', preprocessor_22),
#     ('model', DecisionTreeClassifier(random_state=17))
# ])


# # Train the pipeline
# pipe_22 = pipeline_22.fit(X_train_trans, y_train_trans)
# pipe_22[:-1].get_feature_names_out().tolist()



# from sklearn.metrics import accuracy_score


# predictions_22 = pipe_22.predict(X_test_trans)
# accuracy_22 = accuracy_score(y_test, predictions_22)
# print(f"Accuracy: {accuracy_22}")






# train_scen_22 = pipe_22[:-1].fit_transform(X_train_trans, y_train_trans)
# train_scen_22 = pd.DataFrame(train_scen_22, columns=pipe_22[:-1].get_feature_names_out(), index=X_train_trans.index)

# test_scen_22 = pipe_22[:-1].transform(X_test_trans)
# test_scen_22 = pd.DataFrame(test_scen_22, columns=pipe_22[:-1].get_feature_names_out(), index=X_test_trans.index)

# X_hp_data = pd.concat((train_scen_22, test_scen_22), keys=['train', 'test'])
# y_hp_data = pd.concat((y_train_trans, y_test), keys=['train', 'test'])

# X_hp_data.to_csv('X_transformed_minmax_PCA', index=True)
# y_hp_data.to_csv('y_transformed_minmax_PCA', index=True)




# numerical_preprocessing_33 = Pipeline(steps=[
#     ('scaler', min_max),
#     ('lda', lda)
# ])


# categorical_preprocessing_33 = Pipeline(steps=[
#     ('encoder', one_hot_encoder),
#     ('scaler', min_max)
# ])

# preprocessor_33 = ColumnTransformer(
#     transformers=[
#         ('num', numerical_preprocessing_33, continuous),
#         ('cat', categorical_preprocessing_33, categorical)],
#     remainder='passthrough'
# )

# pipeline_33 = Pipeline(steps=[
#     ('preprocessor', preprocessor_33),
#     ('model', DecisionTreeClassifier(random_state=17))
# ])


# # Train the pipeline
# pipe_33 = pipeline_33.fit(X_train_trans, y_train_trans)
# pipe_33[:-1].get_feature_names_out().tolist()



# from sklearn.metrics import accuracy_score


# predictions_33 = pipe_33.predict(X_test_trans)
# accuracy_33 = accuracy_score(y_test, predictions_33)
# print(f"Accuracy: {accuracy_33}")




# train_scen_33 = pipe_33[:-1].fit_transform(X_train_trans, y_train_trans)
# train_scen_33 = pd.DataFrame(train_scen_33, columns=pipe_33[:-1].get_feature_names_out(), index=X_train_trans.index)

# test_scen_33 = pipe_33[:-1].transform(X_test_trans)
# test_scen_33 = pd.DataFrame(test_scen_33, columns=pipe_33[:-1].get_feature_names_out(), index=X_test_trans.index)

# X_hp_data = pd.concat((train_scen_33, test_scen_33), keys=['train', 'test'])
# y_hp_data = pd.concat((y_train_trans, y_test), keys=['train', 'test'])

# X_hp_data.to_csv('X_transformed_minmax_LDA', index=True)
# y_hp_data.to_csv('y_transformed_minmax_LDA', index=True)




# numerical_preprocessing_44 = Pipeline(steps=[
#     ('scaler', min_max),
#     ('lda', lda)
# ])

# categorical_preprocessing_44 = Pipeline(steps=[
#     ('encoder', one_hot_encoder),
#     ('scaler', min_max)
# ])

# preprocessor_44 = ColumnTransformer(
#     transformers=[
#         ('num', numerical_preprocessing_44, continuous),
#         ('cat', categorical_preprocessing_44, categorical)],
#     remainder='passthrough'
# )

# pipeline_44 = Pipeline(steps=[
#     ('preprocessor', preprocessor_44),
#     ('wrapper', wrapper),
#     ('model', DecisionTreeClassifier(random_state=17))
# ])

# # Train the pipeline
# pipe_44 = pipeline_44.fit(X_train_trans, y_train_trans)
# pipe_44[:-1].get_feature_names_out().tolist()

# from sklearn.metrics import accuracy_score

# predictions_44 = pipe_44.predict(X_test_trans)
# accuracy_44 = accuracy_score(y_test, predictions_44)
# print(f"Accuracy: {accuracy_44}")


# train_scen_44 = pipe_44[:-1].fit_transform(X_train_trans, y_train_trans)
# train_scen_44 = pd.DataFrame(train_scen_44, columns=pipe_44[:-1].get_feature_names_out(), index=X_train_trans.index)

# test_scen_44 = pipe_44[:-1].transform(X_test_trans)
# test_scen_44 = pd.DataFrame(test_scen_44, columns=pipe_44[:-1].get_feature_names_out(), index=X_test_trans.index)

# X_hp_data = pd.concat((train_scen_44, test_scen_44), keys=['train', 'test'])
# y_hp_data = pd.concat((y_train_trans, y_test), keys=['train', 'test'])

# X_hp_data.to_csv('X_transformed_LDA_wrapper', index=True)
# y_hp_data.to_csv('y_transformed_LDA_wrapper', index=True)




numerical_preprocessing_1 = Pipeline(steps=[
    ('scaler', min_max),
    ('lda', lda)
])


categorical_preprocessing_1 = Pipeline(steps=[
    ('encoder', one_hot_encoder),
    ('scaler', min_max)
])

preprocessor_1 = ColumnTransformer(
    transformers=[
        ('num', numerical_preprocessing_1, continuous),
        ('cat', categorical_preprocessing_1, categorical)],
    remainder='passthrough'
)

pipeline_1 = Pipeline(steps=[
    ('preprocessor', preprocessor_1),
    ('model', DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=20,min_samples_leaf=100,random_state=17,
                             class_weight='balanced'))
])


# Train the pipeline
pipe_1 = pipeline_1.fit(X_train_trans, y_train_trans)
pipe_1[:-1].get_feature_names_out().tolist()


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


                             
# Train the pipeline
pipe_1 = pipeline_1.fit(X_train_trans, y_train_trans)
pipe_1[:-1].get_feature_names_out().tolist()



from sklearn.metrics import accuracy_score


predictions_1 = pipe_1.predict(X_test_trans)
accuracy_1 = accuracy_score(y_test, predictions_1)
print(f"Accuracy: {accuracy_1}")




numerical_preprocessing_2 = Pipeline(steps=[
    ('scaler', min_max),
    ('lda', lda)
])


categorical_preprocessing_2 = Pipeline(steps=[
    ('encoder', one_hot_encoder),
    ('scaler', min_max)
])

preprocessor_2 = ColumnTransformer(
    transformers=[
        ('num', numerical_preprocessing_2, continuous),
        ('cat', categorical_preprocessing_2, categorical)],
    remainder='passthrough'
)

pipeline_2 = Pipeline(steps=[
    ('preprocessor', preprocessor_2),
    ('wrapper', wrapper),
    ('model', LogisticRegression(penalty=None, C=1.0, fit_intercept=True, class_weight='balanced', l1_ratio=None))
])


# Train the pipeline
pipe_2 = pipeline_2.fit(X_train_trans, y_train_trans)
pipe_2[:-1].get_feature_names_out().tolist()


from sklearn.metrics import accuracy_score


predictions_2 = pipe_2.predict(X_test_trans)
accuracy_2 = accuracy_score(y_test, predictions_2)
print(f"Accuracy: {accuracy_2}")




# numerical_preprocessing_2 = Pipeline(steps=[
#     ('scaler', min_max),
#     ('lda', lda)
# ])


# categorical_preprocessing_2 = Pipeline(steps=[
#     ('encoder', one_hot_encoder),
#     ('scaler', min_max)
# ])

# preprocessor_2 = ColumnTransformer(
#     transformers=[
#         ('num', numerical_preprocessing_2, continuous),
#         ('cat', categorical_preprocessing_2, categorical)],
#     remainder='passthrough'
# )

# pipeline_2 = Pipeline(steps=[
#     ('preprocessor', preprocessor_2),
#     ('wrapper', wrapper),
#     ('model',SVC( probability=True, class_weight='balanced', max_iter=1000, random_state=111, C= 0.1, gamma= 'scale', kernel= 'rbf'))
# ])


# # Train the pipeline
# pipe_2 = pipeline_2.fit(X_train_trans, y_train_trans)
# pipe_2[:-1].get_feature_names_out().tolist()


# from sklearn.metrics import accuracy_score


# predictions_2 = pipe_2.predict(X_test_trans)
# accuracy_2 = accuracy_score(y_test, predictions_2)
# print(f"Accuracy: {accuracy_2}")




# numerical_preprocessing_final = Pipeline(steps=[
#     ('scaler', min_max),
#     ('lda', lda)
# ])


# categorical_preprocessing_final = Pipeline(steps=[
#     ('encoder', one_hot_encoder),
#     ('scaler', min_max)
# ])

# preprocessor_final = ColumnTransformer(
#     transformers=[
#         ('num', numerical_preprocessing_final, continuous),
#         ('cat', categorical_preprocessing_final, categorical)],
#     remainder='passthrough'
# )

# pipeline_final = Pipeline(steps=[
#     ('preprocessor', preprocessor_final),
#     ('wrapper', wrapper),
#     ('model',xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, scale_pos_weight=5, random_state=111))
# ])


# # Train the pipeline
# pipe_final = pipeline_final.fit(X_train_trans, y_train_trans)
# pipe_final[:-1].get_feature_names_out().tolist()


# from sklearn.metrics import accuracy_score


# predictions_final = pipe_final.predict(X_test_trans)
# accuracy_final = accuracy_score(y_test, predictions_final)
# print(f"Accuracy: {accuracy_final}")


# numerical_preprocessing_2 = Pipeline(steps=[
#     ('scaler', min_max),
#     ('lda', lda)
# ])


# categorical_preprocessing_2 = Pipeline(steps=[
#     ('encoder', one_hot_encoder),
#     ('scaler', min_max)
# ])

# preprocessor_2 = ColumnTransformer(
#     transformers=[
#         ('num', numerical_preprocessing_2, continuous),
#         ('cat', categorical_preprocessing_2, categorical)],
#     remainder='passthrough'
# )

# pipeline_2 = Pipeline(steps=[
#     ('preprocessor', preprocessor_2),
#     ('wrapper', wrapper),
#     ('model', KerasClassifier(build_fn=create_model, epochs=50, batch_size=100, verbose=1))
# ])


# # Train the pipeline
# pipe_2 = pipeline_2.fit(X_train_trans, y_train_trans)
# pipe_2[:-1].get_feature_names_out().tolist()


# from sklearn.metrics import accuracy_score


# predictions_2 = pipe_2.predict(X_test_trans)
# accuracy_2 = accuracy_score(y_test, predictions_2)
# print(f"Accuracy: {accuracy_2}")


test = pd.read_csv('/kaggle/input/DontGetKicked/test.csv')
test.set_index('RefId', inplace=True, drop=True)

test.shape




# test.columns




test=drop_columns(test)






test = initial_preproc(test)






test = test.drop(drop_list_2, axis=1)






test.info()




test = test.drop(drop_list_3, axis=1)


test.info()




test = missing_imputer(X_train, test)[1]




test.info()


test.shape




predictions_2 = pipe_2.predict(test)






# Create the submission DataFrame
submission_df = pd.DataFrame(data={'RefId': test.index, 'IsBadBuy': predictions_2})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")






import pandas as pd

df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.set_index('RefId', inplace=True, drop=True)


y = df.IsBadBuy
X = df.drop('IsBadBuy', axis=1)




test = pd.read_csv('/kaggle/input/DontGetKicked/test.csv')
test.set_index('RefId', inplace=True)

test.shape




X = drop_columns(X)
test=drop_columns(test)






X = initial_preproc(X)
test = initial_preproc(test)






drop_list_2 = feature_screening(X, min_cv=0.1, mode_threshold=99, distinct_threshold=90)
X = X.drop(drop_list_2, axis=1)
test = test.drop(drop_list_2, axis=1)




drop_list_3= ['PRIMEUNIT', 'AUCGUART']
X = X.drop(drop_list_3, axis=1)
test= test.drop(drop_list_3, axis=1)




outlier_index = outlier_handling(X, contamination=0.01)
X = X.drop(outlier_index.tolist())
y = y.drop(outlier_index.tolist())






X = missing_prices_row_report(X, max_missing_values_threshold = 4 )
y = y.loc[X.index]  # Use the same indices to keep them aligned




X, test = missing_imputer(X, test)




pipe_2 = pipeline_2.fit(X, y)
predictions_2 = pipe_2.predict(test)




# Create the submission DataFrame
submission_df = pd.DataFrame(data={'RefId': test.index, 'IsBadBuy': predictions_2})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")




import gc
gc.collect()






