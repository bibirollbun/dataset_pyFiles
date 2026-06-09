pip install scorecardbundle


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.impute import SimpleImputer
from scorecardbundle.feature_discretization import ChiMerge as cm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, f_classif, mutual_info_classif, RFECV
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


Carvana = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
Carvana.set_index('RefId', inplace=True)
Carvana.head()


Carvana.shape


Carvana.info()


X = Carvana.iloc[:,1:]
y = Carvana.iloc[:,0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=880)

X_train.shape,X_test.shape


original_x_data = pd.concat([X_train,X_test], keys=["train", "test"])
original_y_data = pd.concat([y_train,y_test], keys=["train", "test"])
original_x_data.to_csv('/kaggle/working/appended_x_data_original.csv')
original_y_data.to_csv('/kaggle/working/appended_y_data_original.csv')
original_x_data.info()


def initial_preproc(data):
    processed_data = data.copy()
    
    columns_list = columns=['PurchDate','VehYear','Model','Trim','SubModel','WheelTypeID','BYRNO','VNZIP1','VNST']

    processed_data = processed_data.drop(columns=columns_list,axis=1)

    Carvana['Transmission'] = Carvana['Transmission'].replace('MANUAL','Manual')
    
    processed_data['Color'] = processed_data['Color'].replace('NOT AVAIL', np.nan)

    for col in ['Color', 'Make']:
        freq = processed_data[col].value_counts(normalize=True)  
        rare_classes = freq[freq < 0.01].index        
        processed_data[col] = processed_data[col].apply(lambda x: 'Other' if x in rare_classes else x)
    
    
    return processed_data


X_train = initial_preproc(X_train)
X_test = initial_preproc(X_test)

X_train.shape, X_test.shape


X_train.info()


def feature_screaning (data, min_cv, mode_treshold, distinct_treshold):

   processed_data = data.copy()

   cat_vars = processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
   con_vars = processed_data.select_dtypes(include=['number']).columns.tolist()

   min_cv = min_cv
   cv_values = processed_data[con_vars].std()/processed_data[con_vars].mean()
   screening_cv = cv_values[cv_values<0.1].index.tolist()

   treshold = mode_treshold
   mode_category = processed_data[cat_vars].apply(lambda x:x.value_counts().max()/len(x))*100
   screening_mode = mode_category[mode_category>treshold].index.tolist()

   treshold = distinct_treshold
   distinct_percentage = (processed_data[cat_vars].apply(lambda x: len(x.dropna().unique())/x.count())*100)
   screening_distinct = distinct_percentage[distinct_percentage>treshold].index.tolist()

   screened_features = list(set(screening_cv + screening_mode + screening_distinct))

   return screened_features 


drop_list = feature_screaning(X_train, min_cv=0.1, mode_treshold=99, distinct_treshold=90)

X_train = X_train.drop(drop_list, axis=1)
X_test = X_test.drop(drop_list, axis=1)

X_train.shape, X_test.shape


def range_consistency(data):
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
        data[column] = data[column].apply(lambda x: x if min_val <= x <= max_val else None)
    return(data)


range_consistency(X_train)

X_train.head()


X_train['PRIMEUNIT'] = X_train['PRIMEUNIT'].fillna('unknown')
X_train['AUCGUART'] = X_train['AUCGUART'].fillna('unknown')

X_test['PRIMEUNIT'] = X_test['PRIMEUNIT'].fillna('unknown')
X_test['AUCGUART'] = X_test['AUCGUART'].fillna('unknown')


def outlier_handling(data, contamination):
    
    inputs_iso = data.copy()

    cat_vars = inputs_iso.select_dtypes(include=['object', 'category']).columns.tolist()
    con_vars = inputs_iso.select_dtypes(include=['number']).columns.tolist()

    inputs_iso = inputs_iso.dropna()

    scaler = StandardScaler()
    inputs_iso[con_vars] = scaler.fit_transform(inputs_iso[con_vars])

    label_encoder = LabelEncoder()
    inputs_iso[con_vars] = inputs_iso[con_vars].apply(label_encoder.fit_transform)

    clf = IsolationForest(contamination = contamination, random_state=880)
    clf.fit(inputs_iso[con_vars])

    outliers = clf.predict(inputs_iso[con_vars])

    inputs_iso['outlier'] = outliers
    
    outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index
    
    return outlier_index


outlier_index = outlier_handling(X_train, contamination=0.01)

X_train = X_train.drop(outlier_index.tolist())
y_train = y_train.drop(outlier_index.tolist())
#X_test = X_test.drop(outlier_index.tolist())

X_train.shape , y_train.shape


def missing_row_report(data, missrow_price, missrow_other):
    
    processed_data = data.copy()

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
    columns_rest = [col for col in processed_data.columns if col not in columns_price]
    num_cols_rest = len(columns_rest)
    
    processed_data['Num_Missing_Price'] = processed_data[columns_price].isnull().sum(axis=1)
    
    processed_data['Num_Missing_Rest'] = processed_data[columns_rest].isnull().sum(axis=1)
    
    drop_missing = (processed_data['Num_Missing_Price'] > missrow_price) | \
                   (processed_data['Num_Missing_Rest'] > missrow_other * num_cols_rest)
    
    discard_missing_row = processed_data.loc[drop_missing].index.tolist()
    
    return discard_missing_row


discard_missing_row = missing_row_report(X_train, missrow_price=4, missrow_other=0.5)

X_train = X_train.drop(discard_missing_row)
y_train = y_train.drop(discard_missing_row)
#X_test = X_train.drop(outlier_index.tolist())

X_train.shape, y_train.shape


def impute_missing_values(train, test, strategy_con, strategy_cat):

    cat_vars = train.select_dtypes(include=['object', 'category']).columns.tolist()
    con_vars = train.select_dtypes(include=['number']).columns.tolist()
    
    con_imputer = SimpleImputer(strategy=strategy_con)
    cat_imputer = SimpleImputer(strategy=strategy_cat)
    
    train[con_vars] = con_imputer.fit_transform(train[con_vars])
    train[cat_vars] = cat_imputer.fit_transform(train[cat_vars])

    test[con_vars] = con_imputer.transform(test[con_vars])
    test[cat_vars] = cat_imputer.transform(test[cat_vars])
    
    return train, test


X_train, X_test = impute_missing_values(X_train, X_test, strategy_con='median', strategy_cat='most_frequent')

X_train.shape, X_test.shape


X_train.shape, y_train.shape,


y_train.shape


y_data_original.shape


x_data_original = pd.concat([X_train,X_test], keys=["train", "test"])
#x_data_original = x_data.drop(['VehBCost', 'WarrantyCost'], axis=1)
y_data_original = pd.concat([y_train,y_test], keys=["train", "test"])
x_data_original.to_csv('/kaggle/working/appended_x_data_original.csv')
y_data_original.to_csv('/kaggle/working/appended_y_data_original.csv')
x_data_original.info()


def discretizer_train(data, target, chi_merge_list):

    processed_data = data.copy()

    trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=2, decimal=3,output_dataframe=True)
    result_cm = trans_cm.fit_transform(processed_data[chi_merge_list], target.astype('int')) 

    boundaries_dict = {key: np.insert(boundaries, 0, -np.inf) for key, boundaries in trans_cm.boundaries_.items()}

    for key, boundaries in boundaries_dict.items():
        column_name = f"{key}_cat_cm"
        processed_data[column_name] = pd.cut(processed_data[key], bins=boundaries, labels=False, right=False)

    processed_data.drop(columns=chi_merge_list, inplace=True)
        
        
    return processed_data, boundaries_dict


def discretizer_test(data, boundaries_dict):
    processed_data = data.copy()

    for key, boundaries in boundaries_dict.items():
        column_name = f"{key}_cat_cm"
        processed_data[column_name] = pd.cut(processed_data[key], bins=boundaries, labels=False, right=False)

    return processed_data


chi_merge_list = ['VehBCost', 'WarrantyCost']

X_train_FS, chi_boundaries = discretizer_train(X_train, y_train, chi_merge_list)
X_test_FS = discretizer_test(X_test, chi_boundaries)

X_train_FS.shape, X_test_FS.shape


x_data = pd.concat([X_train_FS,X_test_FS], keys=["train", "test"])
x_data = x_data.drop(['VehBCost', 'WarrantyCost'], axis=1)
y_data = pd.concat([y_train,y_test], keys=["train", "test"])
x_data.to_csv('/kaggle/working/appended_x_data_discretized_no_scaling.csv')
y_data.to_csv('/kaggle/working/appended_y_data_discretized_no_scaling.csv')
x_data.info()


continuous_FS = X_train_FS.select_dtypes(exclude=['object','category']).columns.tolist()
categorical_FS = X_train_FS.select_dtypes(include=['object','category']).columns.tolist()
ordinal_FS = ['Size','VehBCost_cat_cm','WarrantyCost_cat_cm']
nominal_FS = [i for i in categorical_FS if i not in ordinal_FS]

len(continuous_FS), len(categorical_FS), len(ordinal_FS), len(nominal_FS)


one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
min_max_scaler = MinMaxScaler()
z_score_scaler = StandardScaler()
Box_Cox = PowerTransformer(method='box-cox', standardize=False)


def transform(train, test):

    transform_list = ['VehBCost', 'WarrantyCost'] 

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
        train[f"{feature}_transformed"] = transformer.fit_transform(train[[feature]])
        test[f"{feature}_transformed"] = transformer.transform(test[[feature]])

    train = train.drop(transform_list, axis=1)
    test = test.drop(transform_list, axis=1)
    
    return train, test


X_train_transformed, X_test_transformed = transform(X_train, X_test)

X_train_transformed.shape, X_test_transformed.shape


transformed_x_data = pd.concat([X_train_transformed,X_test_transformed], keys=["train", "test"])
#transformed_x_data = transformed_x_data.drop(['VehBCost_cat_cm', 'WarrantyCost_cat_cm'], axis=1)
#transformed_y_data = pd.concat([y_train,y_test], keys=["train", "test"])
transformed_x_data.to_csv('/kaggle/working/appended_x_data_transformed.csv')
#transformed_y_data.to_csv('/kaggle/working/appended_y_data_transformed.csv')
#transformed_x_data.info()


pca_pipeline = Pipeline([
    ('scaler', min_max_scaler),
    ('pca', PCA(n_components=2, random_state=880))
])

preprocessor_1 = ColumnTransformer(
    transformers=[
        ('nom', one_hot_encoder, nominal_FS),
        ('ord', ordinal_encoder, ordinal_FS),
        ('pca', pca_pipeline, continuous_FS)
    ],
    remainder='passthrough'  
)

wrapper = RFECV(
    estimator=DecisionTreeClassifier(random_state=880),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1
)

pipeline_1 = Pipeline(steps=[
    ('preprocessor', preprocessor_1),
    ('feature_selection', wrapper),
    ('model', DecisionTreeClassifier(random_state=880))
])

pipe_1 = pipeline_1.fit(X_train_FS, y_train)
pipe_1[:-1].get_feature_names_out().tolist()


prediction_1 = pipe_1.predict(X_test_FS)

print(classification_report(y_test, prediction_1))


lda_pipeline = Pipeline([
    ('scaler', min_max_scaler),
    ('lda', LinearDiscriminantAnalysis(n_components=1))
])

preprocessor_2 = ColumnTransformer(
    transformers=[
        ('nom', one_hot_encoder, nominal_FS),
        ('ord', ordinal_encoder, ordinal_FS),
        ('lda', lda_pipeline, continuous_FS)
    ],
    remainder='passthrough'  
)

wrapper = RFECV(
    estimator=DecisionTreeClassifier(random_state=880),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1
)

pipeline_2 = Pipeline(steps=[
    ('preprocessor', preprocessor_2),
    ('feature_selection', wrapper),
    ('model', DecisionTreeClassifier(random_state=880))
])

pipe_2 = pipeline_2.fit(X_train_FS, y_train)
pipe_2[:-1].get_feature_names_out().tolist()


prediction_2 = pipe_2.predict(X_test_FS)

print(classification_report(y_test, prediction_2))


continuous_FE = X_train.select_dtypes(exclude=['object','category']).columns.tolist()
categorical_FE = X_train.select_dtypes(include=['object','category']).columns.tolist()
ordinal_FE = ['Size']
nominal_FE = [i for i in categorical_FE if i not in ordinal_FE]

len(continuous_FE), len(categorical_FE), len(ordinal_FE), len(nominal_FE) 


selected_features = ['VehBCost', 'WarrantyCost']

preprocessor_3 = ColumnTransformer(transformers=[
    ('nom', one_hot_encoder, nominal_FE),  
    ('ord', ordinal_encoder, ordinal_FE),
    ('boxcox', Box_Cox, selected_features),
    ('con', z_score_scaler, continuous_FE)])


wrapper = RFECV(
    estimator=DecisionTreeClassifier(random_state=880),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1
)


pipeline_3 = Pipeline(steps=[
    ('preprocessing', preprocessor_3),
    ('feature_selection', wrapper),
    ('model', DecisionTreeClassifier(random_state=880))])



pipe_3 = pipeline_3.fit(X_train, y_train)
pipe_3[:-1].get_feature_names_out().tolist()


prediction_3 = pipe_3.predict(X_test)

print(classification_report(y_test, prediction_3))


Price = ['MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice' ,
    'MMRCurrentAuctionCleanPrice' ,
    'MMRCurrentRetailAveragePrice' , 
    'MMRCurrentRetailCleanPrice']


pcap_pipeline = Pipeline([
    ('scaler', min_max_scaler),
    ('pca', PCA(n_components=3, random_state=880))
])

preprocessor_4 = ColumnTransformer(
    transformers=[
        ('nom', one_hot_encoder, nominal_FS),
        ('ord', ordinal_encoder, ordinal_FS),
        ('pca', pca_pipeline, Price)
    ],
    remainder='passthrough'  
)

wrapper = RFECV(
    estimator=DecisionTreeClassifier(random_state=880),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1
)

pipeline_4 = Pipeline(steps=[
    ('preprocessor', preprocessor_4),
    ('feature_selection', wrapper),
    ('model', DecisionTreeClassifier(random_state=880))
])


pipe_4 = pipeline_4.fit(X_train_FS, y_train)
pipe_4[:-1].get_feature_names_out().tolist()


prediction_4 = pipe_4.predict(X_test_FS)

print(classification_report(y_test, prediction_4))


fs_num = SelectKBest(score_func=mutual_info_regression, k=7)
fs_cat = SelectKBest(score_func=mutual_info_classif, k=5)


numerical_preprocessing = Pipeline(steps=[
    ('fs', fs_num),  
    ('scaler', min_max_scaler)])  
    
nominal_preprocessing = Pipeline(steps=[  
    ('fs', fs_cat),  
    ('encoder', one_hot_encoder)])  

ordinal_preprocessing = Pipeline(steps=[
    ('fs', fs_cat),
    ('ordinal', ordinal_encoder),
    ('scaler', min_max_scaler)])  

preprocessor_5 = ColumnTransformer(transformers=[
    ('nom', one_hot_encoder, nominal_FE),  
    ('ord', ordinal_encoder, ordinal_FE),
    ('boxcox', Box_Cox, selected_features),
    ('con', z_score_scaler, continuous_FE)])




pipeline_5 = Pipeline(steps=[
    ('preprocessing', preprocessor_5),
    ('model', DecisionTreeClassifier(random_state=880))])



pipe_5 = pipeline_5.fit(X_train, y_train)
pipe_5[:-1].get_feature_names_out().tolist()


prediction_5 = pipe_5.predict(X_test)

print(classification_report(y_test, prediction_5))


preprocessor = ColumnTransformer(
    transformers=[
        ('nom', one_hot_encoder, nominal_FS),
        ('ord', ordinal_encoder, ordinal_FS),
    ],
    remainder='passthrough'  
)

wrapper = RFECV(
    estimator=DecisionTreeClassifier(random_state=880),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', wrapper),
])

pipe = pipeline.fit(X_train_FS, y_train)
pipe[:-1].get_feature_names_out().tolist()


X_train_prep = pipe.fit_transform(X_train_FS, y_train)
X_test_prep = pipe.transform(X_test_FS)


feature_names = pipeline.named_steps['feature_selection'].get_feature_names_out()


X_train_df = pd.DataFrame(X_train_prep, columns=feature_names)
X_test_df = pd.DataFrame(X_test_prep, columns=feature_names)

X_data = pd.concat([X_train_prep, X_test_prep])
X_data.to_csv('X_data.csv', index=False)

y_data = pd.concat([y_train, y_test])
y_data.to_csv('y_data.csv', index=False)




