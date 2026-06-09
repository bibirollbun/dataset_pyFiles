# Load Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
# Ignore all pandas warnings
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)



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
    return file_dict


class Data: 

    def __init__(self, data, train_bool): 
        self.data = data
        self.train_bool = train_bool
        if self.train_bool:
            self.columns_dict = {
                'drugsexposure' : ['person_id', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id'],
                'observation' : ['person_id', 'observation_concept_name', 'valuefilled'],
                'devices' : ['person_id', 'device_datetime_hourly', 'device'],
                'proceduresoccurrences' : ['person_id', 'procedure_datetime_hourly', 'procedure'],
                'person_demographics_episode':['person_id', 'age_in_months', 'gender'],
                'SepsisLabel' : ['person_id', 'SepsisLabel']}
            
            self.mapping_lst = {'SepsisLabel':['SepsisLabel'], 
                            'person_demographics_episode':['age_in_months', 'gender'],
                            'observation':['observation_concept_name', 'valuefilled']}
        else: 
            self.columns_dict = {
                'drugsexposure' : ['person_id', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id'],
                'observation' : ['person_id', 'observation_concept_name', 'valuefilled'],
                'devices' : ['person_id', 'device_datetime_hourly', 'device'],
                'proceduresoccurrences' : ['person_id', 'procedure_datetime_hourly', 'procedure'],
                'person_demographics_episode':['person_id', 'age_in_months', 'gender'], 
                'SepsisLabel' : ['person_id']}
            
            self.mapping_lst = { 
                            'person_demographics_episode':['age_in_months', 'gender'],
                            'observation':['observation_concept_name', 'valuefilled']}         
        
        self.datasets = {}
        self._fix_data()

    def data_shape(self, df_dict): 
        for key, df in df_dict.items():
            print(f'{key}:{df.shape}')
        
    def _create_map(self, df, mapping_col_base, mapping_col):
        map_dict = {}
        for col_base in df[mapping_col_base].unique(): 
            map_dict[col_base] = df.loc[df[mapping_col_base] == col_base, mapping_col].values[0]
        return map_dict
    
    def _mapping_features(self): 
        mapping = {}
        for mapping_df_name, mapping_cols in  self.mapping_lst.items(): 
            mapping_df = self.datasets[mapping_df_name]
            for mapping_col in mapping_cols: 
                mapping[mapping_col] = self._create_map(df =mapping_df,
                                                     mapping_col_base='person_id',
                                                     mapping_col= mapping_col)
        return mapping
            
    def _map_columns(self): 
        for key, df in self.datasets.items():
            for col in self.mapping:
                df.loc[:, col] = df['person_id'].map(self.mapping[col])

    
    def _fix_data(self): 
        # Extract columns from each dataset
        self.datasets = {dataset_name: self.data[dataset_name][cols] for dataset_name, cols in self.columns_dict.items()}
        print('.... Colums Extracted')
        self.data_shape(self.datasets)
        
        if self.train_bool: 
            # Identify the unique person IDs 
            unique_ids = set.union(*(set(df['person_id'].unique()) for df in self.datasets.values()))
            self.datasets = {key: df[df['person_id'].isin(unique_ids)] for key, df in self.datasets.items()}
            print('.... Unique Person Ids')
            self.data_shape(self.datasets)

        # Mapp the columns 
        self.mapping  = self._mapping_features()
        self._map_columns()
        self.data_shape(self.datasets)
        print('..... Mapping ')

    def get_data(self):
        return self.datasets


class Pre_processing: 

    def __init__(self, df, date_time_col, encoding_cols, standarize_cols): 
        self.df  = df
        self.date_time_col = date_time_col
        self.encoding_cols = encoding_cols
        self.standarize_cols = standarize_cols
        self._data_process()

    def _data_process(self): 
        self._data_cleaning()
        self._data_pre_processing()
        self._date_encoding_scaling()

    def _data_cleaning(self): 
        print(f'Before Preprocesing:{self.df.shape}')
        self.df = self.df.drop_duplicates().dropna()
        print(f'After Preprocesing:{self.df.shape}')

    def _data_pre_processing(self):
        # Convert timestamp to datetime and extract features
        self.df[str(self.date_time_col)] = pd.to_datetime(self.df[str(self.date_time_col)])
        self.df['year'] = self.df[str(self.date_time_col)].dt.year
        self.df['month'] = self.df[str(self.date_time_col)].dt.month
        self.df['day'] = self.df[str(self.date_time_col)].dt.day
        self.df['hour'] = self.df[str(self.date_time_col)].dt.hour
        self.df.drop(str(self.date_time_col), axis=1, inplace=True)  

    def _date_encoding_scaling(self): 
        # Encoding
        label_encoder = LabelEncoder()
        for col_encoding in self.encoding_cols: 
            self.df[str(col_encoding)] = label_encoder.fit_transform(self.df[str(col_encoding)])
        print('.... Encoding Data')
        # Normalize numerical features (age)
        scaler = StandardScaler()
        for col_standar in self.standarize_cols: 
            self.df[[str(col_standar)]] = scaler.fit_transform(self.df[[str(col_standar)]])
            
        print('....Standarize Data')
    def get_df(self): return self.df 


def summary_table(df): 
    summary_table = pd.DataFrame({
    'Null Values': [df.isnull().sum().sum() ],
    'Duplicate Rows': [df.duplicated().sum() ],
    'Shape': [df.shape]})
    print(summary_table)


# Define file paths
test_folder_path = "/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data"
train_folder_path = "/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data"


datasets_names = ['proceduresoccurrences', 'measurement_lab', 'observation', 'drugsexposure', 
                  'measurement_observation', 'person_demographics_episode', 'devices', 'measurement_meds',
                  'SepsisLabel']


drugsexposure_cols = ['person_id', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id']
observation_cols = ['person_id', 'observation_concept_name', 'valuefilled']
devices_cols = ['person_id', 'device_datetime_hourly', 'device']
proceduresoccurrences_cols = ['person_id', 'procedure_datetime_hourly', 'procedure']
person_demographics_episode_cols  = ['person_id', 'age_in_months' ,	'gender']
SepsisLabel_cols = ['person_id', 'SepsisLabel']

cols = [drugsexposure_cols, observation_cols, devices_cols, proceduresoccurrences_cols, person_demographics_episode_cols, SepsisLabel_cols]


train_data = load_data_from_folder(train_folder_path)
test_data = load_data_from_folder(test_folder_path)


create_train_data = Data(train_data, train_bool = True)
train_datasets = create_train_data.get_data()
(drugsexposure_train_data, _, devices_train_data, proceduresoccurrences_train_data, _,  _) = train_datasets.values()


summary_table(devices_train_data)
summary_table(drugsexposure_train_data)
summary_table(proceduresoccurrences_train_data)





create_test_data = Data(test_data, train_bool = False)
test_datasets = create_test_data.get_data()
(drugsexposure_test_data, _, devices_test_data, proceduresoccurrences_test_data, _, sepsis_data) = test_datasets.values()


summary_table(drugsexposure_test_data)
summary_table(devices_test_data)
summary_table(proceduresoccurrences_test_data)





pp_drug_train = Pre_processing(df= drugsexposure_train_data, 
                date_time_col = 'drug_datetime_hourly',
                encoding_cols = ['drug_concept_id', 'route_concept_id', 'observation_concept_name', 'valuefilled', 'gender'],    
                standarize_cols = ['drug_concept_id', 'route_concept_id', 'age_in_months', 'year',	'month', 'day', 'hour'], 
                 )
drugsexposure_train_data = pp_drug_train.get_df()



pp_device_train = Pre_processing(df= devices_train_data, 
                date_time_col = 'device_datetime_hourly',
                encoding_cols = ['device', 'observation_concept_name', 'valuefilled', 'valuefilled', 'gender'],    
                standarize_cols = ['age_in_months', 'year',	'month', 'day', 'hour'], 
                 )
devices_train_data = pp_device_train.get_df()


pp_procedure_train = Pre_processing(df= proceduresoccurrences_train_data, 
                date_time_col = 'procedure_datetime_hourly',
                encoding_cols = ['procedure', 'observation_concept_name', 'valuefilled', 'gender'],    
                standarize_cols = ['age_in_months', 'year',	'month', 'day', 'hour'], 
                 )
proceduresoccurrences_train_data = pp_procedure_train.get_df()


pp_drug_test = Pre_processing(df= drugsexposure_test_data, 
                date_time_col = 'drug_datetime_hourly',
                encoding_cols = ['drug_concept_id', 'route_concept_id', 'observation_concept_name', 'valuefilled', 'gender'],    
                standarize_cols = ['drug_concept_id', 'route_concept_id', 'age_in_months', 'year',	'month', 'day', 'hour'], 
                 )
drugsexposure_test_data = pp_drug_test.get_df()


pp_device_test = Pre_processing(df= devices_test_data, 
                date_time_col = 'device_datetime_hourly',
                encoding_cols = ['device', 'observation_concept_name', 'valuefilled', 'valuefilled', 'gender'],    
                standarize_cols = ['age_in_months', 'year',	'month', 'day', 'hour'], 
                 )
devices_test_data = pp_device_test.get_df()


pp_procedure_test = Pre_processing(df= proceduresoccurrences_test_data, 
                date_time_col = 'procedure_datetime_hourly',
                encoding_cols = ['procedure', 'observation_concept_name', 'valuefilled', 'gender'],    
                standarize_cols = ['age_in_months', 'year',	'month', 'day', 'hour'], 
                 )
proceduresoccurrences_test_data = pp_procedure_test.get_df()


def  train_ML_model(df, drop_col, taget_col): 
    #Drop column
    df = df.drop(columns=[drop_col])
    # Split Data
    X = df.drop(columns=[taget_col])  
    y = df[taget_col]  
    
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X, y)

    return model


dt_model_drugsexposure = train_ML_model(df=drugsexposure_train_data, 
               drop_col='person_id', 
               taget_col='SepsisLabel')

dt_model_devices = train_ML_model(df=devices_train_data, 
               drop_col='person_id', 
               taget_col='SepsisLabel')

dt_model_proceduresoccurrences = train_ML_model(df=proceduresoccurrences_train_data, 
               drop_col='person_id', 
               taget_col='SepsisLabel')




#id procedures occurrences
proceduresoccurrences_person_id = proceduresoccurrences_test_data['person_id']
proceduresoccurrences_test_data = proceduresoccurrences_test_data.drop(columns=['person_id'])

#id drug exposure
drugsexposure_test_data_person_id = drugsexposure_test_data['person_id']
drugsexposure_test_data = drugsexposure_test_data.drop(columns=['person_id'])

#id device
devices_test_data_person_id = devices_test_data['person_id']
devices_test_data = devices_test_data.drop(columns=['person_id'])


predict_drugexp = dt_model_drugsexposure.predict(drugsexposure_test_data)
predict_device = dt_model_devices.predict(devices_test_data)
predict_procedures = dt_model_proceduresoccurrences.predict(proceduresoccurrences_test_data)


pred_map_drug_exp = {person_id:pred for person_id, pred in zip(drugsexposure_test_data_person_id, predict_drugexp)}
pred_map_device = {person_id:pred for person_id, pred in zip(devices_test_data_person_id, predict_device)}
pred_map_procedures = {person_id:pred for person_id, pred in zip(proceduresoccurrences_person_id, predict_procedures)}


mapping_dict = pred_map_drug_exp.copy()
mapping_dict.update(pred_map_device)
mapping_dict.update(pred_map_procedures)


len(mapping_dict)


#merged_map = {**pred_map_drug_exp, **pred_map_device, **pred_map_procedures}


prediction_df = test_data['SepsisLabel']
prediction_df['person_id_datetime'] = prediction_df['person_id'].astype(str) + '_' + prediction_df['measurement_datetime']


prediction_df.loc[:, 'SepsisLabel'] = prediction_df['person_id'].map(mapping_dict)
prediction_df = prediction_df.drop(columns=['person_id', 'measurement_datetime'] )


prediction_df = prediction_df.fillna(0)
prediction_df


prediction_df.to_csv("submission.csv", index=False)































