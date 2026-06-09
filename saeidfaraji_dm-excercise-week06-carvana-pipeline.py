import pandas as pd
df=pd.read_csv("/kaggle/input/DontGetKicked/training.csv")


# Set 'RefId' as the index in df first
df = df.set_index('RefId')

# Then define target and inputs
target = df['IsBadBuy']
inputs = df.drop(columns=['IsBadBuy'])


from sklearn.model_selection import train_test_split


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.20, random_state=1)

X_train.shape,X_test.shape


# import numpy as np
# def frequency_table(variable):
#     unique_elements,counts=np.unique(variable.dropna(),return_counts=True)
#     percentage=(counts/len(variable)*100)
#     for i , j , k in zip(unique_elements,counts,percentage):
#         print(f"{i} :count{j} , percentage: {k:.2f}")
#     return

# for col in categorical_fields:
#     print(f"Frequency Table for {col}:")
#     frequency_table(inputs[col])
#     print("-" * 40)


import numpy as np

def initial_preproc(data):
    processed_data = data.copy()
    # List of columns to drop
    columns_to_drop = ["PurchDate","VehYear","Model","Trim","SubModel","WheelTypeID","BYRNO","VNZIP1","VNST", "PRIMEUNIT", "AUCGUART"]
    
    # Drop the columns
    processed_data = processed_data.drop(columns=columns_to_drop)
    # Replace 'NOT AVAIL' in the 'Color' column with NaN
    processed_data['Color'] = processed_data['Color'].replace('NOT AVAIL', np.nan)
    processed_data['Transmission'] = processed_data['Transmission'].replace(['Manual'], 'MANUAL')
    
    # Define a threshold for frequency (1% of total data)
    threshold=1
    for x in ('Make','Color'):
        
        unique_elements, counts = np.unique(processed_data[x].dropna(), return_counts=True)
        percentage = (counts / len(processed_data[x])) * 100
        rare_classes = [elem for elem, pct in zip(unique_elements, percentage) if pct < threshold]
        processed_data[x] = processed_data[x].replace(rare_classes, 'OTHER')
        # print(f"\nUpdated Frequency Table for {x}:")
        # frequency_table(processed_data[x])
        
    return processed_data


X_train = initial_preproc(X_train)
X_test = initial_preproc(X_test)
X_train.shape, X_test.shape


columns = X_train.columns
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
continuous_fields = [col for col in X_train.columns if col not in categorical_fields]


import pandas as pd

def range_consistency(data, target):
    # Define ranges for each column
    column_ranges = {
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

    # Iterate through each column and fill NaN values outside the defined range
    for column, (min_val, max_val) in column_ranges.items():
        data[column] = data[column].apply(lambda x: x if min_val <= x <= max_val else None)

    target = target.replace([':0', "'0'"], '0')
        
    return data, target


X_train = range_consistency(X_train, y_train)[0]
X_test = range_consistency(X_test, y_test)[0]

y_train = range_consistency(X_train, y_train)[1]
y_test = range_consistency(X_test, y_test)[1]
X_train.shape, X_test.shape


def feature_screening(data, min_cv=0.1, mode_threshold=99, distinct_threshold=90):
    processed_data = data.copy()
   # Define a minimum value for coefficient of variation
    min_cv = min_cv

    # Calculate the coefficient of variation for each column
    cv_values = processed_data[continuous_fields].std() / processed_data[continuous_fields].mean()

    # Filter out columns with CV less than 0.1
    screen_cv =  cv_values[cv_values < min_cv].index.tolist()


    # Define a threshold for the dominant category percentage
    mode_threshold = mode_threshold

    # Calculate the percentage of the mode category for each column
    mode_category = (processed_data[categorical_fields].apply(lambda x: x.value_counts().max() / len(x)) * 100)

    # Select columns where the mode category percentage is greater than the threshold
    screen_mode = mode_category[mode_category > mode_threshold].index.tolist()


    # Set a threshold for excluding columns 
    distinct_threshold = distinct_threshold

    # Calculate the percentage of distinct categories in categorical variables
    distinct_percentage = (processed_data[categorical_fields].apply(lambda x: x.dropna().nunique() / x.count()) * 100)

    # Select categorical columns based on distinct percentage threshold
    screen_distinct = distinct_percentage[distinct_percentage > distinct_threshold].index.tolist()

    screened_features  = list(set(screen_cv + screen_mode + screen_distinct))
     
    return screened_features 


drop_list = feature_screening(X_train, min_cv=0.1, mode_threshold=99, distinct_threshold=90)

X_train = X_train.drop(drop_list, axis=1)
X_test = X_test.drop(drop_list, axis=1)

X_train.shape, X_test.shape


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

def outlier_handling_replace(data, contamination=0.01):
    inputs_iso = data.copy()
    # Discard rows with NaN values
    inputs_iso = inputs_iso.dropna()
    
    # Apply Z-score scaling to numerical columns
    scaler = StandardScaler()
    inputs_iso[continuous_fields] = scaler.fit_transform(inputs_iso[continuous_fields])
    
    # Apply label encoding to categorical columns
    label_encoder = LabelEncoder()
    inputs_iso[categorical_fields] = inputs_iso[categorical_fields].apply(label_encoder.fit_transform)
    
    # Fit Isolation Forest model
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(inputs_iso)
    
    # Predict outliers
    outliers = clf.predict(inputs_iso)
    
    # Add the outlier predictions to your DataFrame
    inputs_iso['outlier'] = outliers
    
    # Identify outliers
    outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index   
    return outlier_index


outlier_index = outlier_handling_replace(X_train, contamination=0.01)

X_train = X_train.drop(outlier_index.tolist())

y_train = y_train.drop(outlier_index.tolist())
X_train.shape, y_train.shape


columns_to_check=["MMRAcquisitionAuctionAveragePrice",
                  "MMRAcquisitionAuctionCleanPrice",
                  "MMRAcquisitionRetailAveragePrice",
                  "MMRAcquisitonRetailCleanPrice",
                  "MMRCurrentAuctionAveragePrice",
                  "MMRCurrentAuctionCleanPrice",
                  "MMRCurrentRetailAveragePrice",
                  "MMRCurrentRetailCleanPrice"]
X_train["num_missing_values"]=X_train[columns_to_check].isnull().sum(axis=1)
valid_indices = X_train[X_train["num_missing_values"] < 4].index
X_train = X_train.loc[valid_indices].drop(columns=["num_missing_values"])
y_train = y_train.loc[valid_indices]
X_train.shape, y_train.shape


def missing_row_report(data, missrow=12):
    processed_data = data.copy()
    

    # Create a new column with the number of missing values in each row
    processed_data['Num_Missing_Values'] = processed_data.isnull().sum(axis=1)

    discard_missing_row = processed_data[processed_data['Num_Missing_Values'] > missrow].index.tolist()

    return discard_missing_row


discard_missing_row = missing_row_report(X_train, missrow=35)

X_train = X_train.drop(discard_missing_row)
y_train = y_train.drop(discard_missing_row)

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


from sklearn.impute import  SimpleImputer

def missing_imputer(train, test):
    
    continuous = train.select_dtypes(exclude=['object','category']).columns.tolist()
    categorical = train.select_dtypes(include=['object','category']).columns.tolist()

    # Define imputation strategies for each subset of columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cont_imputer = SimpleImputer(strategy='median')
    
    try:

    # Impute missing values
        train[continuous] = cont_imputer.fit_transform(train[continuous])
        train[categorical] = cat_imputer.fit_transform(train[categorical])
    
        test[continuous] = cont_imputer.transform(test[continuous])
        test[categorical] = cat_imputer.transform(test[categorical])

    except:
        test[continuous] = cont_imputer.transform(test[continuous])
        test[categorical] = cat_imputer.transform(test[categorical])
        
    return train, test


X_train, X_test = missing_imputer(X_train, X_test)

X_train.shape, X_test.shape


pip install scorecardbundle


import numpy as np
from scorecardbundle.feature_discretization import ChiMerge as cm

chi_merge_list = ['VehBCost', 'WarrantyCost']

def discretizer(train, test, y, chi_list):

    trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=1, decimal=3,output_dataframe=True)
    trans_cm.fit(train[chi_list], y.astype('int').squeeze()) 

    # Add -inf to the beginning of each array
    boundaries_dict = {key: np.insert(boundaries, 0, -np.inf) for key, boundaries in trans_cm.boundaries_.items()}

    # Iterate through the dictionary and add new columns to data
    for key, boundaries in boundaries_dict.items():
        column_name = f"{key}_cat_cm"
        train[column_name] = pd.cut(train[key], bins=boundaries, labels=False, right=False)
        test[column_name] = pd.cut(test[key], bins=boundaries, labels=False, right=False)
        
    return train, test


X_train, X_test = discretizer(X_train, X_test, y_train, chi_merge_list)

X_train.shape, X_test.shape


from sklearn.preprocessing import PowerTransformer
# List of features to transform
selected_features =  ['VehBCost', 'WarrantyCost']

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


skewed_list=['VehBCost','WarrantyCost']
transformed_list=['VehBCost_transformed','WarrantyCost_transformed']
discretized_list=['VehBCost_cat_cm','WarrantyCost_cat_cm']
X_train['IsOnlineSale']=X_train['IsOnlineSale'].astype('int')
categorical = X_train.select_dtypes(include=['object','category']).columns.tolist()
continuous = [i for i in X_train.columns if i not in skewed_list + transformed_list + discretized_list + categorical]


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, f_classif, mutual_info_classif, RFECV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

z_score = StandardScaler()
min_max = MinMaxScaler()
wrapper = RFECV(estimator=DecisionTreeClassifier(random_state=29), step=1, min_features_to_select=10, cv=5, n_jobs=-1)
pca = PCA(n_components=2, random_state=717)
lda = LinearDiscriminantAnalysis(n_components=1)
kpca = KernelPCA(n_components=3, kernel='rbf', random_state=717)


# Define the preprocessing steps for numerical and categorical features separately

numerical_preprocessing_1 = Pipeline(steps=[
    ('scaler', min_max)])  
    
nominal_preprocessing_1 = Pipeline(steps=[
    ('nominal', one_hot_encoder),  
    ('scaler', min_max)]) 


# Define the ColumnTransformer for numerical and categorical features
preprocessor_1 = ColumnTransformer(transformers=[
    ('num', numerical_preprocessing_1, discretized_list+continuous),
    ('nom', nominal_preprocessing_1, categorical),
]) 


pipeline_1 = Pipeline(steps=[
    ('preprocessor', preprocessor_1),
    ('wrapper', wrapper),
    ('model',DecisionTreeClassifier(random_state=17))])



# Train the pipeline
pipe_1 = pipeline_1.fit(X_train, y_train)
print(f"Optimal number of features: {wrapper.n_features_}")
pipe_1[:-1].get_feature_names_out().tolist()

# Use the pipeline for prediction or other tasks
predictions_1 = pipe_1.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy_discrete_wrapper = accuracy_score(y_test, predictions_1)
print("Accuracy:", accuracy_discrete_wrapper)

# نمایش گزارش طبقه‌بندی (Precision, Recall, F1-Score)
print("\nClassification Report:\n", classification_report(y_test, predictions_1))


# Define the preprocessing steps for numerical and categorical features separately

numerical_preprocessing_2 = Pipeline(steps=[
    ('scaler', z_score),
    ('pca',pca)])  
    
nominal_preprocessing_2 = Pipeline(steps=[
    ('nominal', one_hot_encoder),  
    ('scaler', z_score)]) 


# Define the ColumnTransformer for numerical and categorical features
preprocessor_2 = ColumnTransformer(transformers=[
    ('num', numerical_preprocessing_2, transformed_list+continuous),
    ('nom', nominal_preprocessing_2, categorical),
]) 


pipeline_2 = Pipeline(steps=[
    ('preprocessor', preprocessor_2),
    ('wrapper', wrapper),
    ('model',DecisionTreeClassifier(random_state=17))])



# Train the pipeline
pipe_2 = pipeline_2.fit(X_train, y_train)
print(f"Optimal number of features: {wrapper.n_features_}")
pipe_2[:-1].get_feature_names_out().tolist()

# Use the pipeline for prediction or other tasks
predictions_2 = pipe_2.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy_transformed_pca_wrapper = accuracy_score(y_test, predictions_2)
print("Accuracy:", accuracy_transformed_pca_wrapper)

# نمایش گزارش طبقه‌بندی (Precision, Recall, F1-Score)
print("\nClassification Report:\n", classification_report(y_test, predictions_2))


# Define the preprocessing steps for numerical and categorical features separately

numerical_preprocessing_3 = Pipeline(steps=[
    ('scaler', z_score),
    ('lda',lda)])  
    
nominal_preprocessing_3 = Pipeline(steps=[
    ('nominal', one_hot_encoder),  
    ('scaler', z_score)]) 


# Define the ColumnTransformer for numerical and categorical features
preprocessor_3 = ColumnTransformer(transformers=[
    ('num', numerical_preprocessing_3, transformed_list+continuous),
    ('nom', nominal_preprocessing_3, categorical),
]) 


pipeline_3 = Pipeline(steps=[
    ('preprocessor', preprocessor_3),
    ('wrapper', wrapper),
    ('model',DecisionTreeClassifier(random_state=17))])



# Train the pipeline
pipe_3 = pipeline_3.fit(X_train, y_train)
print(f"Optimal number of features: {wrapper.n_features_}")
pipe_3[:-1].get_feature_names_out().tolist()

# Use the pipeline for prediction or other tasks
predictions_3 = pipe_3.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy_transformed_lda_wrapper = accuracy_score(y_test, predictions_3)
print("Accuracy:", accuracy_transformed_lda_wrapper)

# نمایش گزارش طبقه‌بندی (Precision, Recall, F1-Score)
print("\nClassification Report:\n", classification_report(y_test, predictions_3))

