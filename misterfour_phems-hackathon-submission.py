import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf


# Load test data
test_sepsislabel = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/SepsisLabel_test.csv')
test_sepsislabel['measurement_datetime'] = pd.to_datetime(test_sepsislabel['measurement_datetime'])
test_submission = test_sepsislabel.copy()

# Convert 'measurement_datetime' to datetime
test_sepsislabel['measurement_datetime'] = pd.to_datetime(test_sepsislabel['measurement_datetime'])

# Calculate duration for each person_id
test_sepsislabel['duration'] = test_sepsislabel.groupby('person_id')['measurement_datetime'].transform(lambda x: (x.max() - x.min()).total_seconds()/3600)
test_devices = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/devices_test.csv')
test_devices_grouped = test_devices.groupby('person_id')['device'].agg(lambda x: ', '.join(x.unique())).reset_index()
test_devices_grouped = pd.DataFrame(test_devices_grouped)

# Drop specified columns
test_devices_hr = test_devices.copy()
test_devices_hr = test_devices_hr.drop(columns=['visit_occurrence_id', 'device'])

# Convert 'device_datetime_hourly' to datetime objects
test_devices_hr['device_datetime_hourly'] = pd.to_datetime(test_devices_hr['device_datetime_hourly'])

# Extract the hour from the datetime column
test_devices_hr['device_hour'] = test_devices_hr['device_datetime_hourly'].dt.hour

# Group by 'person_id' and calculate the mean hour
mean_hours = test_devices_hr.groupby('person_id')['device_hour'].mean().reset_index()
mean_hours['device_mean_hour'] = mean_hours['device_hour']

# Merge the mean hour back into the original DataFrame
test_devices_hr = pd.merge(test_devices_hr, mean_hours[['person_id', 'device_mean_hour']], on='person_id', how='left')

# Remove duplicate 'person_id' rows (keeping the first occurrence)
test_devices_hr = test_devices_hr.drop_duplicates(subset='person_id')
test_devices_hr = test_devices_hr.drop(columns='device_datetime_hourly')
test_devices_merge = pd.merge(test_devices_grouped, test_devices_hr, on='person_id', how='left')
merged_test = pd.merge(test_sepsislabel, test_devices_merge, on='person_id', how='left')


test_lab=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/measurement_lab_test.csv')
test_lab = test_lab.groupby(['person_id','measurement_datetime'])[['Base excess in Venous blood by calculation',
       'Base excess in Arterial blood by calculation',
       'Phosphate [Moles/volume] in Serum or Plasma',
       'Potassium [Moles/volume] in Blood',
       'Bilirubin.total [Moles/volume] in Serum or Plasma',
       'Neutrophil Ab [Units/volume] in Serum',
       'Bicarbonate [Moles/volume] in Arterial blood',
       'Hematocrit [Volume Fraction] of Blood',
       'Glucose [Moles/volume] in Serum or Plasma',
       'Calcium [Moles/volume] in Serum or Plasma',
       'Chloride [Moles/volume] in Blood',
       'Sodium [Moles/volume] in Serum or Plasma',
       'C reactive protein [Mass/volume] in Serum or Plasma',
       'Carbon dioxide [Partial pressure] in Venous blood',
       'Oxygen [Partial pressure] in Venous blood',
       'Albumin [Mass/volume] in Serum or Plasma',
       'Bicarbonate [Moles/volume] in Venous blood',
       'Oxygen [Partial pressure] in Arterial blood',
       'Carbon dioxide [Partial pressure] in Arterial blood',
       'Interleukin 6 [Mass/volume] in Body fluid',
       'Magnesium [Moles/volume] in Blood', 'Prothrombin time (PT)',
       'Procalcitonin [Mass/volume] in Serum or Plasma',
       'Lactate [Moles/volume] in Blood', 'Creatinine [Mass/volume] in Blood',
       'Fibrinogen measurement', 'Bilirubin measurement',
       'Partial thromboplastin time', ' activated', 'Total white blood count',
       'Platelet count', 'White blood cell count', 'Blood venous pH',
       'D-dimer level', 'Blood arterial pH',
       'Hemoglobin [Moles/volume] in Blood', 'Ionised calcium measurement']].sum().reset_index()

# Ensure 'measurement_datetime' is converted to datetime64[ns] in all DataFrames
def ensure_datetime(df, datetime_column):
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    return df

# Convert 'measurement_datetime' to datetime in all relevant DataFrames
test_sepsislabel = ensure_datetime(test_sepsislabel, 'measurement_datetime')
test_lab = ensure_datetime(test_lab, 'measurement_datetime')

# Merge the DataFrames
merged_test = pd.merge(merged_test, test_lab, on=['person_id', 'measurement_datetime'], how='left')


test_meds=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/measurement_meds_test.csv')
test_meds = test_meds.groupby(['person_id','measurement_datetime'])[['Systolic blood pressure', 'Diastolic blood pressure',
       'Body temperature', 'Respiratory rate', 'Heart rate',
       'Measurement of oxygen saturation at periphery',
       'Oxygen/Gas total [Pure volume fraction] Inhaled gas']].sum().reset_index()


# Ensure 'measurement_datetime' is converted to datetime64[ns] in all DataFrames
def ensure_datetime(df, datetime_column):
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    return df

# Convert 'measurement_datetime' to datetime in all relevant DataFrames

merged_test = ensure_datetime(merged_test, 'measurement_datetime')
test_meds = ensure_datetime(test_meds, 'measurement_datetime')

# Merge the DataFrames

merged_test = pd.merge(merged_test, test_meds, on=['person_id', 'measurement_datetime'], how='left')


test_obs=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/measurement_observation_test.csv')

test_obs = test_obs.groupby(['person_id','measurement_datetime'])[['Left pupil Diameter Auto', 'Right pupil Diameter Auto',
       'Glasgow coma scale', 'Capillary refill [Time]', 'Pulse',
       'Arterial pulse pressure', 'Right pupil Pupillary response',
       'Left pupil Pupillary response']].sum().reset_index()


# Ensure 'measurement_datetime' is converted to datetime64[ns] in all DataFrames
def ensure_datetime(df, datetime_column):
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    return df

merged_test = ensure_datetime(merged_test, 'measurement_datetime')
test_obs = ensure_datetime(test_obs, 'measurement_datetime')
merged_test = pd.merge(merged_test, test_obs, on=['person_id', 'measurement_datetime'], how='left')


test_procedure = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/proceduresoccurrences_test.csv')
test_procedure['procedure_datetime_hourly'] = pd.to_datetime(test_procedure['procedure_datetime_hourly'])
test_procedure['procedure_duration'] = test_procedure.groupby('person_id')['procedure_datetime_hourly'].transform(lambda x: (x.max() - x.min()).total_seconds() / 3600)

test_procedure_A = test_procedure[['person_id','procedure_duration']]
test_procedure_A = test_procedure_A.drop_duplicates(subset=['person_id'])

test_procedure_grouped = test_procedure.groupby('person_id')[['procedure']].agg(lambda x: ', '.join(x.unique())).reset_index()
merged_test = pd.merge(merged_test, test_procedure_grouped, on='person_id', how='left')
merged_test = pd.merge(merged_test, test_procedure_A, on='person_id', how='left')
merged_test = merged_test.sort_values(by='measurement_datetime') 


test_drug=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/drugsexposure_test.csv')
test_drug=test_drug[['person_id','drug_datetime_hourly','drug_concept_id','route_concept_id']]
test_drug['route_concept_id'] = test_drug['route_concept_id'].astype(str)
test_drug['drug_datetime_hourly'] = pd.to_datetime(test_drug['drug_datetime_hourly'])
test_drug['drug_duration'] = test_drug.groupby('person_id')['drug_datetime_hourly'].transform(lambda x: (x.max() - x.min()).total_seconds()/3600)
test_drug_A = test_drug[['person_id','drug_duration']]
test_drug_A = test_drug_A.drop_duplicates(subset=['person_id'])
test_drug_grouped = test_drug.groupby(['person_id'])[['drug_concept_id','route_concept_id']].agg(lambda x: ', '.join(x.unique())).reset_index()
test_drug_grouped = pd.DataFrame(test_drug_grouped)
merged_test = pd.merge(merged_test, test_drug_grouped, on='person_id', how='left')
merged_test = pd.merge(merged_test, test_drug_A, on='person_id', how='left')



test_observation=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/observation_test.csv')
# Convert 'drug_datetime_hourly' to datetime
test_observation['observation_datetime'] = pd.to_datetime(test_observation['observation_datetime'])
test_observation['observation_duration'] = test_observation.groupby('person_id')['observation_datetime'].transform(lambda x: (x.max() - x.min()).total_seconds()/3600)
test_observation_A = test_observation[['person_id','observation_duration']]
test_observation_A = test_observation_A.drop_duplicates(subset=['person_id'])
test_observation = test_observation.groupby('person_id')[['observation_concept_name','valuefilled']].agg(lambda x: ', '.join(x.unique())).reset_index()
merged_test = pd.merge(merged_test, test_observation_A, on='person_id', how='left')
merged_test = pd.merge(merged_test, test_observation, on='person_id', how='left')


# Convert 'measurement_datetime' to datetime (if not already done)
merged_test['measurement_datetime'] = pd.to_datetime(merged_test['measurement_datetime'], errors='coerce')

# Sort the DataFrame by 'measurement_datetime' from earliest to latest (ascending order)
merged_test = merged_test.sort_values(by='measurement_datetime')

# Display the cleaned DataFrame
x_test = merged_test


date_time = ['measurement_datetime']

numerical_cols = ['person_id',
 'duration',
 'device_hour',
 'device_mean_hour',
 'Base excess in Venous blood by calculation',
 'Base excess in Arterial blood by calculation',
 'Phosphate [Moles/volume] in Serum or Plasma',
 'Potassium [Moles/volume] in Blood',
 'Bilirubin.total [Moles/volume] in Serum or Plasma',
 'Neutrophil Ab [Units/volume] in Serum',
 'Bicarbonate [Moles/volume] in Arterial blood',
 'Hematocrit [Volume Fraction] of Blood',
 'Glucose [Moles/volume] in Serum or Plasma',
 'Calcium [Moles/volume] in Serum or Plasma',
 'Chloride [Moles/volume] in Blood',
 'Sodium [Moles/volume] in Serum or Plasma',
 'C reactive protein [Mass/volume] in Serum or Plasma',
 'Carbon dioxide [Partial pressure] in Venous blood',
 'Oxygen [Partial pressure] in Venous blood',
 'Albumin [Mass/volume] in Serum or Plasma',
 'Bicarbonate [Moles/volume] in Venous blood',
 'Oxygen [Partial pressure] in Arterial blood',
 'Carbon dioxide [Partial pressure] in Arterial blood',
 'Interleukin 6 [Mass/volume] in Body fluid',
 'Magnesium [Moles/volume] in Blood',
 'Prothrombin time (PT)',
 'Procalcitonin [Mass/volume] in Serum or Plasma',
 'Lactate [Moles/volume] in Blood',
 'Creatinine [Mass/volume] in Blood',
 'Fibrinogen measurement',
 'Bilirubin measurement',
 'Partial thromboplastin time',
 ' activated',
 'Total white blood count',
 'Platelet count',
 'White blood cell count',
 'Blood venous pH',
 'D-dimer level',
 'Blood arterial pH',
 'Hemoglobin [Moles/volume] in Blood',
 'Ionised calcium measurement',
 'Systolic blood pressure',
 'Diastolic blood pressure',
 'Body temperature',
 'Respiratory rate',
 'Heart rate',
 'Measurement of oxygen saturation at periphery',
 'Oxygen/Gas total [Pure volume fraction] Inhaled gas',
 'Left pupil Diameter Auto',
 'Right pupil Diameter Auto',
 'Glasgow coma scale',
 'procedure_duration',
 'drug_duration',
 'observation_duration']

categorical_cols = ['device',
 'Capillary refill [Time]',
 'Pulse',
 'Arterial pulse pressure',
 'Right pupil Pupillary response',
 'Left pupil Pupillary response',
 'procedure',
 'drug_concept_id',
 'route_concept_id',
 'observation_concept_name',
 'valuefilled']

extra_cols = [
 'Right pupil Pupillary response',
 'Left pupil Pupillary response']

x_test.loc[:, extra_cols] = x_test[extra_cols].apply(lambda col: col.fillna(col.mode()[0]), axis=0)
x_test.loc[:, categorical_cols] = x_test[categorical_cols].ffill().bfill()
x_test.loc[:, numerical_cols] = x_test[numerical_cols].fillna(x_test[numerical_cols].mean())
x_test.loc[:, date_time] = x_test[date_time].fillna(x_test[date_time].mean())


x_test.drop('measurement_datetime', axis=1, inplace=True)


columns_to_drop = [
    'person_id',
    "Calcium [Moles/volume] in Serum or Plasma",
    "Left pupil Pupillary response",
    "Capillary refill [Time]",
    "Arterial pulse pressure",
    "Left pupil Diameter Auto",
    "Right pupil Pupillary response",
    "observation_concept_name",
    "Neutrophil Ab [Units/volume] in Serum",
    "Heart rate",
    "Interleukin 6 [Mass/volume] in Body fluid",
    "Glucose [Moles/volume] in Serum or Plasma",
    "C reactive protein [Mass/volume] in Serum or Plasma",
    "Oxygen [Partial pressure] in Venous blood",
    "Albumin [Mass/volume] in Serum or Plasma",
    "Oxygen [Partial pressure] in Arterial blood",
    "Carbon dioxide [Partial pressure] in Arterial blood",
    "Lactate [Moles/volume] in Blood",
    "D-dimer level",
    "Bilirubin measurement",
    " activated",  # extra space at the beginning
    "Total white blood count",
    "Platelet count",
    "White blood cell count",
    "Bicarbonate [Moles/volume] in Arterial blood",
    "Blood venous pH"
]

# Drop columns from x_test
x_test = x_test.drop(columns=columns_to_drop)

print("x_test shape after dropping columns:", x_test.shape)


# Load the saved transformer object
text_to_embedding_tuple = joblib.load("/kaggle/input/text_to_embedding_v0/scikitlearn/default/1/text_to_embedding_transformer.pkl")

# Debugging: Check what's inside
print("Loaded object:", text_to_embedding_tuple)
print("Type of loaded object:", type(text_to_embedding_tuple))
print("Length of loaded object:", len(text_to_embedding_tuple))

# Extract text_columns correctly
text_columns = ['device', 'Pulse', 'procedure', 'drug_concept_id', 'route_concept_id', 'valuefilled']
# Load Word2Vec model separately
word2vec_model = Word2Vec.load("/kaggle/input/word2vec_v2/scikitlearn/default/1/word2vec.model")

# Define tokenizer
def tokenize_text(text):
    if pd.isna(text):
        return []
    return str(text).split()

# Define transformer class
class TextToEmbeddingTransformer:
    def __init__(self, word2vec_model, text_columns):
        self.word2vec_model = word2vec_model
        self.text_columns = text_columns

    def transform_text_to_embedding(self, text):
        tokens = tokenize_text(text)
        if not tokens:
            return np.zeros(self.word2vec_model.vector_size)
        vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
        if not vectors:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(vectors, axis=0)

    def transform(self, X):
        X = X.copy()
        for col in self.text_columns:
            if col in X.columns:  # Ensure column exists before transformation
                X[col] = X[col].apply(self.transform_text_to_embedding)
            else:
                print(f"Warning: Column '{col}' not found in input DataFrame.")
        return X

# Recreate the TextToEmbeddingTransformer object
text_to_embedding = TextToEmbeddingTransformer(word2vec_model, text_columns)

# Apply text-to-embedding transformation
x_test_transformed = text_to_embedding.transform(x_test)

# Verify the transformation
#print("Sample transformed row:")
#print(x_test_transformed.iloc[0])

# Convert transformed embeddings to a structured NumPy array
embedding_dim = word2vec_model.vector_size
x_test_embedded = np.array([np.hstack(row[text_columns].values) for _, row in x_test_transformed.iterrows()])

# Load the saved scaler
scaler = joblib.load("/kaggle/input/standard_scaler_v3/scikitlearn/default/1/standard_scaler.joblib")

# Standardize the test data
x_test_scaled = scaler.transform(x_test_embedded)

# Load the saved PCA model
pca = joblib.load("/kaggle/input/pca_model/scikitlearn/default/1/pca_model.joblib")

# Apply PCA transformation
x_test_pca = pca.transform(x_test_scaled)

# Output the transformed test data
print("Transformed x_test shape:", x_test_pca.shape)


def create_time_series_dataset(data, time_steps):
    """
    Create time series dataset while retaining the same shape as input data.
    
    Args:
        data (np.ndarray): Input data of shape (n_samples, n_features).
        time_steps (int): Number of time steps to use for creating sequences.
    
    Returns:
        np.ndarray: Time series data of shape (n_samples, time_steps, n_features).
    """
    n_samples, n_features = data.shape
    Xs = np.zeros((n_samples, time_steps, n_features))  # Initialize output array
    
    for i in range(n_samples):
        if i < time_steps:
            # Pad the beginning with zeros if there aren't enough previous time steps
            Xs[i, :i+1, :] = data[:i+1, :]
        else:
            # Slice time series data
            Xs[i, :, :] = data[i-time_steps+1:i+1, :]
    
    return Xs

time_steps = 100  # Number of time steps

# Create time series dataset
x_test_time_series = create_time_series_dataset(x_test_pca, time_steps)

print("Shape of time series X_test:", x_test_time_series.shape)


# Load the TensorFlow model
model = tf.keras.models.load_model("/kaggle/input/mild_model_v6/tensorflow2/default/1/optimized_time_series_modelV1.h5")

# Make predictions
predictions = model.predict(x_test_time_series)


# Ensure 'measurement_datetime' is a string for concatenation
test_submission['measurement_datetime'] = test_submission['measurement_datetime'].astype(str)

# Add predictions to the merged_test dataset
merged_test['SepsisLabel'] = predictions

# Merge observations data into merged_test using a memory-efficient approach
# Instead of loading everything into memory, process chunks if the dataset is large
test_submission = pd.merge(
    test_submission,
    merged_test[['person_id', 'SepsisLabel']],  # Only select necessary columns
    on='person_id',
    how='left'
)

# Concatenate 'person_id' and 'measurement_datetime' with '_' as separator
test_submission['person_id_datetime'] = (
    test_submission['person_id'].astype(str) + '_' + test_submission['measurement_datetime']
)

# Select only the required columns
test_submission = test_submission[['person_id_datetime', 'SepsisLabel']]

# Remove duplicates based on 'person_id_datetime'
test_submission = test_submission.drop_duplicates(subset=['person_id_datetime'])

# Save the submission file
test_submission.to_csv('submission.csv', index=False)

print("Submission successfully created.")

