# Load Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Ignore all pandas warnings
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from scipy.stats import mode




train_path = '/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data'
test_path = '/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data'


columns_dict_train = {'drugsexposure' : ['person_id', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id'],
                'observation' : ['person_id', 'observation_concept_name', 'valuefilled'],
                'devices' : ['person_id', 'device_datetime_hourly', 'device'],
                'proceduresoccurrences' : ['person_id', 'procedure_datetime_hourly', 'procedure'],
                'person_demographics_episode':['person_id', 'age_in_months', 'gender'],
                'SepsisLabel' : ['person_id', 'measurement_datetime', 'SepsisLabel' ]}

columns_dict_test = {'drugsexposure' : ['person_id', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id'],
                'observation' : ['person_id', 'observation_concept_name', 'valuefilled'],
                'devices' : ['person_id', 'device_datetime_hourly', 'device'],
                'proceduresoccurrences' : ['person_id', 'procedure_datetime_hourly', 'procedure'],
                'person_demographics_episode':['person_id', 'age_in_months', 'gender'],
                'SepsisLabel' : ['person_id', 'measurement_datetime']}


d1_cols = ['person_id', 'device_datetime_hourly', 'device', 'measurement_datetime',
       'age_in_months', 'gender', 'observation_concept_name', 'valuefilled']
d2_cols = ['person_id', 'drug_datetime_hourly', 'drug_concept_id',
       'route_concept_id', 'measurement_datetime', 'age_in_months', 'gender',
       'observation_concept_name', 'valuefilled']
d3_cols = ['person_id', 'procedure_datetime_hourly', 'procedure', 'measurement_datetime',
       'age_in_months', 'gender', 'observation_concept_name', 'valuefilled']





def load_data_from_folder(folder_path):
    """
    Load data files from a given folder into a dictionary.
    Returns:
        dict: A dictionary where keys are table names (derived from filenames) and values are Dask DataFrames.
    """
    file_dict = {}
    for file_name in os.listdir(folder_path):
        
        if file_name.endswith(".csv"):
            # remove the type of file 
            table_name = file_name.replace("_train.csv", "").replace("_test.csv", "").replace(".csv", "")
            file_path = os.path.join(folder_path, file_name)
            file_dict[table_name] = pd.read_csv(file_path)
            
    print('.... Data Loaded')
    
    return file_dict


def merge_grouped_datasets(d1, d2, d3, d4, d5, d6):
    d2_grouped = d2.groupby('person_id').first().reset_index()
    d3_grouped = d3.groupby('person_id').first().reset_index()
    d4_grouped = d4.groupby('person_id').first().reset_index()
    d5_grouped = d5.groupby('person_id').first().reset_index()
    d6_grouped = d6.groupby('person_id').first().reset_index()

    
    merged_df = d1.copy()
    merged_df = merged_df.merge(d2_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d3_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d4_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d5_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d6_grouped, on='person_id', how='left')

    return merged_df



def extract_datetime_features(df, column):
    df[column] = pd.to_datetime(df[column], errors='coerce')
    df[column + '_year'] = df[column].dt.year
    df[column + '_month'] = df[column].dt.month
    df[column + '_day'] = df[column].dt.day
    df[column + '_hour'] = df[column].dt.hour
    df[column + '_weekday'] = df[column].dt.weekday
    print('.... Extract Date Time')
    return df.drop(columns=[column])



def preprocess_data(df, datetime_col, preprocessor=None, is_train=True ):
    
    for col in datetime_col:
        df = extract_datetime_features(df=df, column=col)

    cat_features = df.select_dtypes(include=['category', 'object']).columns.to_list()
    fl_features = df.select_dtypes(include=['float']).columns.to_list()

    if is_train:
        # Create and fit the preprocessor for training data
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, fl_features),
            ('cat', categorical_pipeline, cat_features)
        ], remainder='passthrough')

        X_processed = preprocessor.fit_transform(df)
        print(f'... Cols Transform: {X_processed.shape}')
        return X_processed, preprocessor  # Return fitted preprocessor for later use
    else:
        # Use the same preprocessor fitted on training data
        X_processed = preprocessor.transform(df)
        return X_processed




def hard_voting(data1, data2, data3 , y): 
    # Split each dataset into train and test
    X1_train, X1_test, y1_train, y1_test = train_test_split(data1, y, test_size=0.01, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(data2, y, test_size=0.01, random_state=42)
    X3_train, X3_test, y3_train, y3_test = train_test_split(data3, y, test_size=0.01, random_state=42)

    dt1 = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=10, random_state=42)
    dt1.fit(X1_train, y1_train)
    print('.... Train DT1')
    
    dt2 = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=10, random_state=42)
    dt2.fit(X2_train, y2_train)
    print('.... Train DT2')

    dt3 = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=10, random_state=42)
    dt3.fit(X3_train, y3_train)
    print('.... Train DT3')

    pred_dt1, pred_dt2, pred_dt3 = dt1.predict(X1_test), dt2.predict(X2_test), dt3.predict(X3_test)
    print('.... Predictions')

    #majority_vote = [1 if sum(preds) >= 2 else 0 for preds in zip(pred_dt1, pred_dt2, pred_dt3)]
    # Hard Voting
    predictions = np.array([pred_dt1, pred_dt2, pred_dt3])
    majority_vote = mode(predictions, axis=0)[0].flatten()


    accuracy1 = accuracy_score(y1_test, majority_vote)
    accuracy2 = accuracy_score(y2_test, majority_vote)
    accuracy3 = accuracy_score(y3_test, majority_vote)
    
    conf_matrix1 = confusion_matrix(y1_test, majority_vote)
    conf_matrix2 = confusion_matrix(y2_test, majority_vote)
    conf_matrix3 = confusion_matrix(y3_test, majority_vote)


    class_report1 = classification_report(y1_test, majority_vote)
    class_report2 = classification_report(y2_test, majority_vote)
    class_report3 = classification_report(y3_test, majority_vote)

    print("Accuracy:", accuracy1)
    print("Accuracy:", accuracy2)
    print("Accuracy:", accuracy3)

    
    print("Confusion Matrix:\n", conf_matrix1)
    print("Confusion Matrix:\n", conf_matrix2)
    print("Confusion Matrix:\n", conf_matrix3)

    
    print("Classification Report 1:\n", class_report1)
    print("Classification Report 2 :\n", class_report2)
    print("Classification Report 3 :\n", class_report3)



    return dt1, dt2, dt3


def data_pipeline(data_dict, selected_cols, if_train): 
    # Select usefull datasets & cols
    datasets = {dataset_name: data_dict[dataset_name][cols] for dataset_name, cols in selected_cols.items()}

    # Seperate each dataset
    drugsexposure_data = datasets['drugsexposure'] 
    observation_data = datasets['observation'] 
    devices_data = datasets['devices']
    proceduresoccurrences_data = datasets['proceduresoccurrences'] 
    demographics_episode_data = datasets['person_demographics_episode'] 
    SepsisLabel_data = datasets['SepsisLabel']
    print('.... Split Datasets ')
    
    if if_train: 
        # find target col
        y = SepsisLabel_data['SepsisLabel'] 
        SepsisLabel_data = SepsisLabel_data.drop(columns=['SepsisLabel']) 
        print('.... Target Col')
        
    merged_df=merge_grouped_datasets(d1 = SepsisLabel_data, 
                                 d2 = drugsexposure_data,
                                 d3 = observation_data,
                                 d4 = devices_data, 
                                 d5 = proceduresoccurrences_data, 
                                 d6 = demographics_episode_data )
    
    merged_df = merged_df.sort_values('person_id').reset_index(drop=True)
    print(f'... Merge Data:{merged_df.shape} ')
    
    d1, d2, d3 = merged_df[d1_cols], merged_df[d2_cols], merged_df[d3_cols]
    print(f'... Main Datasets: {d1.shape}, {d2.shape}, {d3.shape}')

    if if_train:return d1, d2, d3, y
    else: return d1, d2,d3 


train_data_dict = load_data_from_folder(train_path)
test_data_dict = load_data_from_folder(test_path)


d1_train, d2_train, d3_train, y = data_pipeline(data_dict=train_data_dict, 
           selected_cols=columns_dict_train, 
           if_train=True)


d1_train, d1_preprocessor = preprocess_data(df = d1_train, 
                datetime_col = ['device_datetime_hourly', 'measurement_datetime'], 
                preprocessor=None, 
                is_train=True )

d2_train, d2_preprocessor = preprocess_data(df = d2_train, 
                datetime_col = ['drug_datetime_hourly', 'measurement_datetime'], 
                preprocessor=None, 
                is_train=True )

d3_train, d3_preprocessor = preprocess_data(df = d3_train, 
                datetime_col = ['procedure_datetime_hourly','measurement_datetime' ], 
                preprocessor=None, 
                is_train=True )


d1_test, d2_test, d3_test = data_pipeline(data_dict=test_data_dict, 
           selected_cols=columns_dict_test, 
           if_train=False)


prediction_df = test_data_dict['SepsisLabel']
prediction_df['person_id_datetime'] = prediction_df['person_id'].astype(str) + '_' + prediction_df['measurement_datetime']
prediction_df = prediction_df.drop(columns=['person_id', 'measurement_datetime'] )


d1_test = preprocess_data(df = d1_test, 
                datetime_col = ['device_datetime_hourly', 'measurement_datetime'], 
                preprocessor=d1_preprocessor, 
                is_train=False )

d2_test = preprocess_data(df = d2_test, 
                datetime_col = ['drug_datetime_hourly', 'measurement_datetime'], 
                preprocessor= d2_preprocessor, 
                is_train=False )

d3_test = preprocess_data(df = d3_test, 
                datetime_col = ['procedure_datetime_hourly','measurement_datetime' ], 
                preprocessor=d3_preprocessor, 
                is_train=False )


dt1, dt2, dt3 = hard_voting(data1 = d1_train , 
                               data2 = d2_train, 
                               data3 = d3_train, 
                               y = y)



dt1_pred = dt1.predict(d1_test)
dt2_pred = dt2.predict(d2_test)
dt3_pred = dt3.predict(d3_test)


y_test_pred = [1 if sum(preds) >= 2 else 0 for preds in zip(dt1_pred, dt2_pred, dt3_pred)]


prediction_df['SepsisLabel'] = y_test_pred


prediction_df.to_csv("submission.csv", index=False)







