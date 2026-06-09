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
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


train_path = "/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/"

proceduresoccurrences_train = pd.read_csv(train_path + 'proceduresoccurrences_train.csv')
devices_train = pd.read_csv(train_path + "devices_train.csv")
drugsexpesure_train = pd.read_csv(train_path + "drugsexposure_train.csv")
measurement_train = pd.read_csv(train_path + "measurement_lab_train.csv")
measurement_meds_train= pd.read_csv(train_path + "measurement_meds_train.csv")
measurement_observation_train = pd.read_csv(train_path + "measurement_observation_train.csv")
observation_train = pd.read_csv(train_path + "observation_train.csv")
person_demographics_episode_train = pd.read_csv(train_path + "person_demographics_episode_train.csv")
sepsis_label_train = pd.read_csv(train_path + "SepsisLabel_train.csv")


test_path = "/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/"

test_proceduresoccurrences = pd.read_csv(test_path + 'proceduresoccurrences_test.csv')
test_devices = pd.read_csv(test_path + "devices_test.csv")
test_drugsexpesure = pd.read_csv(test_path + "drugsexposure_test.csv")
test_measurement = pd.read_csv(test_path + "measurement_lab_test.csv")
test_measurement_meds= pd.read_csv(test_path + "measurement_meds_test.csv")
test_measurement_observation = pd.read_csv(test_path + "measurement_observation_test.csv")
test_observation = pd.read_csv(test_path + "observation_test.csv")
test_person_demographics_episode = pd.read_csv(test_path + "person_demographics_episode_test.csv")
test_sepsis_label = pd.read_csv(test_path + "SepsisLabel_test.csv")


import pandas as pd

def basic_eda(df, name):
    """Perform basic EDA for a given DataFrame."""
    print(f"\nğŸ”¹ Dataset: {name}")
    print("-" * 50)
    print(f" Shape: {df.shape}")
    print("\n Missing Values:")
    print(df.isnull().sum())
    print("\n Data Types:")
    print(df.dtypes)
    print("\n First 5 Rows:")
    print(df.head())
    print("\n" + "="*80)

# List of datasets with their names
datasets = {
    "proceduresoccurrences_train": proceduresoccurrences_train,
    "devices_train": devices_train,
    "drugsexpesure_train": drugsexpesure_train,
    "measurement_train": measurement_train,
    "measurement_meds_train": measurement_meds_train,
    "measurement_observation_train": measurement_observation_train,
    "observation_train": observation_train,
    "person_demographics_episode_train": person_demographics_episode_train,
    "sepsis_label_train": sepsis_label_train
}

# Run EDA on all datasets
for name, df in datasets.items():
    basic_eda(df, name)






test_datasets = {
    "test_proceduresoccurrences": test_proceduresoccurrences,
    "test_devices": test_devices,
    "test_drugsexpesure": test_drugsexpesure,
    "test_measurement": test_measurement,
    "test_measurement_meds": test_measurement_meds,
    "test_measurement_observation": test_measurement_observation,
    "test_observation": test_observation,
    "test_person_demographics_episode": test_person_demographics_episode,
    "test_sepsis_label": test_sepsis_label
}

# Run EDA on all test datasets
for name, df in test_datasets.items():
    basic_eda(df, name)


d1=sepsis_label_train[[ 'person_id','measurement_datetime' ]]
d2 = devices_train[['person_id', 'device']]
d3 = drugsexpesure_train[['person_id', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id']]
d4=measurement_meds_train[[ 'person_id','Systolic blood pressure','Diastolic blood pressure','Body temperature','Respiratory rate', 'Heart rate','Measurement of oxygen saturation at periphery','Oxygen/Gas total [Pure volume fraction] Inhaled gas']]
d5=observation_train[[ 'person_id','observation_concept_name','valuefilled' ]]
d6=person_demographics_episode_train[[ 'person_id','age_in_months','gender' ]]
d7=proceduresoccurrences_train[['person_id','procedure']]


import pandas as pd

def merge_grouped_datasets(d1, d2, d3, d4, d5, d6,d7):
    d2_grouped = d2.groupby('person_id').first().reset_index()
    d3_grouped = d3.groupby('person_id').first().reset_index()
    d4_grouped = d4.groupby('person_id').mean().reset_index()
    d5_grouped = d5.groupby('person_id').first().reset_index()
    d6_grouped = d6.groupby('person_id').first().reset_index()
    d7_grouped = d7.groupby('person_id').first().reset_index()

    
    merged_df = d1.copy()
    merged_df = merged_df.merge(d2_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d3_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d4_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d5_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d6_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d7_grouped, on='person_id', how='left')

    
    return merged_df
merged_df=merge_grouped_datasets(d1, d2, d3, d4, d5, d6,d7)
merged_df.shape


d8=test_sepsis_label
d9 = test_devices[['person_id', 'device']]
d10 = test_drugsexpesure[['person_id', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id']]
d11=test_measurement_meds[[ 'person_id','Systolic blood pressure','Diastolic blood pressure','Body temperature','Respiratory rate', 'Heart rate','Measurement of oxygen saturation at periphery','Oxygen/Gas total [Pure volume fraction] Inhaled gas']]
d12=test_observation[[ 'person_id','observation_concept_name','valuefilled' ]]
d13=test_person_demographics_episode[[ 'person_id','age_in_months','gender' ]]
d14=test_proceduresoccurrences[['person_id','procedure']]



test_merge=merge_grouped_datasets(d8, d9, d10, d11, d12,d13,d14)



test_merge.shape


train_merge=merged_df.copy()


train_merge.isnull().sum()


train_merge.info()


from sklearn.impute import SimpleImputer

numerical_columns = ['Systolic blood pressure', 'Diastolic blood pressure', 'Respiratory rate', 
                     'Heart rate', 'Measurement of oxygen saturation at periphery','Body temperature','Oxygen/Gas total [Pure volume fraction] Inhaled gas']

numerical_imputer = SimpleImputer(strategy='mean')

# Apply imputation on numerical columns
train_merge[numerical_columns] = numerical_imputer.fit_transform(train_merge[numerical_columns])

categorical_columns = ['device', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id', 
                       'observation_concept_name', 'valuefilled', 'procedure']

categorical_imputer = SimpleImputer(strategy='most_frequent')

train_merge[categorical_columns] = categorical_imputer.fit_transform(train_merge[categorical_columns])

print(train_merge.isnull().sum())



train=train_merge.dropna()


train.duplicated().sum()


train = train.drop_duplicates()



train['measurement_datetime'] = pd.to_datetime(train['measurement_datetime'], errors='coerce')
train['drug_datetime_hourly'] = pd.to_datetime(train['drug_datetime_hourly'], errors='coerce')

# Check data types and handle NaT values if necessary
print(train.dtypes)



# Extract features from datetime columns
train['measurement_hour'] = train['measurement_datetime'].dt.hour
train['measurement_day'] = train['measurement_datetime'].dt.day
train['measurement_month'] = train['measurement_datetime'].dt.month
train['measurement_year'] = train['measurement_datetime'].dt.year
train['measurement_weekday'] = train['measurement_datetime'].dt.weekday

# Ensure these are integer types and not float
train['measurement_hour'] = train['measurement_hour'].astype('int')
train['measurement_day'] = train['measurement_day'].astype('int')
train['measurement_month'] = train['measurement_month'].astype('int')
train['measurement_year'] = train['measurement_year'].astype('int')
train['measurement_weekday'] = train['measurement_weekday'].astype('int')




train.isnull().sum()


# Extract features from drug_datetime_hourly column
train['drug_measurement_hour'] = train['drug_datetime_hourly'].dt.hour
train['drug_measurement_day'] = train['drug_datetime_hourly'].dt.day
train['drug_measurement_month'] = train['drug_datetime_hourly'].dt.month
train['drug_measurement_year'] = train['drug_datetime_hourly'].dt.year
train['drug_measurement_weekday'] = train['drug_datetime_hourly'].dt.weekday

# Ensure these are integer types and not float
train['drug_measurement_hour'] = train['drug_measurement_hour'].astype('int')
train['drug_measurement_day'] = train['drug_measurement_day'].astype('int')
train['drug_measurement_month'] = train['drug_measurement_month'].astype('int')
train['drug_measurement_year'] = train['drug_measurement_year'].astype('int')
train['drug_measurement_weekday'] = train['drug_measurement_weekday'].astype('int')




train_data=train.drop(columns=['measurement_datetime','drug_datetime_hourly'])


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# Define categorical features
categorical_features = [
    'device', 'drug_concept_id', 'route_concept_id', 'observation_concept_name', 
    'valuefilled', 'gender', 'procedure'
]

# Apply OneHotEncoder to all categorical columns except 'gender'
onehot_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Apply LabelEncoder for 'gender' since it's likely a binary feature
le = LabelEncoder()
train_data['gender'] = le.fit_transform(train_data['gender'])  

# ColumnTransformer to apply the encoding
preprocessor = ColumnTransformer([
    ('cat', onehot_pipeline, [col for col in categorical_features if col != 'gender'])
], remainder='passthrough')  # Keep non-categorical features unchanged

# Fit and transform the train_data
train_encoded = preprocessor.fit_transform(train_data)

# Get OneHotEncoded column names
encoded_columns = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
    [col for col in categorical_features if col != 'gender']
)

# Get the remaining feature names (excluding categorical ones we transformed)
remaining_columns = list(train_data.drop(columns=[col for col in categorical_features if col != 'gender']).columns)

# Ensure column shape consistency
train_encoded_df = pd.DataFrame(train_encoded, columns=list(encoded_columns) + remaining_columns)

# Reset index
train_encoded_df = train_encoded_df.reset_index(drop=True)

# Display encoded train_data
print(train_encoded_df.head())



y=sepsis_label_train['SepsisLabel']



from sklearn.impute import SimpleImputer

numerical_columns = ['Systolic blood pressure', 'Diastolic blood pressure', 'Respiratory rate', 
                     'Heart rate', 'Measurement of oxygen saturation at periphery', 
                     'Body temperature', 'Oxygen/Gas total [Pure volume fraction] Inhaled gas']

categorical_columns = ['device', 'drug_datetime_hourly', 'drug_concept_id', 'route_concept_id', 
                       'observation_concept_name', 'valuefilled', 'procedure']

# Numerical imputation (Mean)
numerical_imputer = SimpleImputer(strategy='mean')
test_merge[numerical_columns] = numerical_imputer.fit_transform(test_merge[numerical_columns])

# Categorical imputation (Most Frequent)
categorical_imputer = SimpleImputer(strategy='most_frequent')
test_merge[categorical_columns] = categorical_imputer.fit_transform(test_merge[categorical_columns])

# Check for remaining missing values
print(test_merge.isnull().sum())



# Convert datetime columns
test_merge['measurement_datetime'] = pd.to_datetime(test_merge['measurement_datetime'], errors='coerce')
test_merge['drug_datetime_hourly'] = pd.to_datetime(test_merge['drug_datetime_hourly'], errors='coerce')

# Extract features from datetime columns
test_merge['measurement_hour'] = test_merge['measurement_datetime'].dt.hour.astype('Int64')
test_merge['measurement_day'] = test_merge['measurement_datetime'].dt.day.astype('Int64')
test_merge['measurement_month'] = test_merge['measurement_datetime'].dt.month.astype('Int64')
test_merge['measurement_year'] = test_merge['measurement_datetime'].dt.year.astype('Int64')
test_merge['measurement_weekday'] = test_merge['measurement_datetime'].dt.weekday.astype('Int64')

test_merge['drug_measurement_hour'] = test_merge['drug_datetime_hourly'].dt.hour.astype('Int64')
test_merge['drug_measurement_day'] = test_merge['drug_datetime_hourly'].dt.day.astype('Int64')
test_merge['drug_measurement_month'] = test_merge['drug_datetime_hourly'].dt.month.astype('Int64')
test_merge['drug_measurement_year'] = test_merge['drug_datetime_hourly'].dt.year.astype('Int64')
test_merge['drug_measurement_weekday'] = test_merge['drug_datetime_hourly'].dt.weekday.astype('Int64')

# Drop original datetime columns

print(test_merge.head())



test_merge.drop(columns=['measurement_datetime', 'drug_datetime_hourly'], inplace=True)



test_merge.duplicated().sum()


test_merge.info()





from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_features = [
    'device', 'drug_concept_id', 'route_concept_id', 'observation_concept_name', 
    'valuefilled', 'gender', 'procedure'
]

# OneHotEncoder for all categorical columns except 'gender'
onehot_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Apply LabelEncoder for 'gender' (binary feature)
test_merge['gender'] = le.transform(test_merge['gender'])  # Use same encoder from train

# Apply ColumnTransformer
test_encoded = preprocessor.transform(test_merge)

# Get OneHotEncoded column names
encoded_columns = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
    [col for col in categorical_features if col != 'gender']
)

# Get remaining feature names
remaining_columns = list(test_merge.drop(columns=[col for col in categorical_features if col != 'gender']).columns)

# Create DataFrame for encoded test data
test_encoded_df = pd.DataFrame(test_encoded, columns=list(encoded_columns) + remaining_columns)

# Reset index
test_encoded_df.reset_index(drop=True, inplace=True)

# Display transformed test data
print("Encoded Test Data:")
print(test_encoded_df.head())



from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Identify columns with non-numeric data types
categorical_cols = test_encoded_df.select_dtypes(include=['object']).columns

# Initialize a LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns
for col in categorical_cols:
    test_encoded_df[col] = label_encoder.fit_transform(test_encoded_df[col].astype(str))

# Now try the prediction


print(test_encoded_df.shape)
print(train_encoded_df.shape)


from sklearn.model_selection import train_test_split

X = train_encoded_df  





print(f"Shape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")



print("X index range:", X.index.min(), "to", X.index.max())
print("y index range:", y.index.min(), "to", y.index.max())

# Check if all y indices exist in X
missing_indices = set(y.index) - set(X.index)
print(f"Missing indices in X: {len(missing_indices)}")


y = y.loc[X.index]  # Select only the indices present in X



X = X.reset_index(drop=True)
y = y.reset_index(drop=True)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize the model (you can use any classifier of your choice)
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data (X_test)
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))





Y_predicted_for_test_data = model.predict(test_encoded_df)




Y_predicted_for_test_data


sub=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/SepsisLabel_sample_submission.csv')
sub['SepsisLabel']=Y_predicted_for_test_data 


sub.to_csv('submission.csv', index=False)





