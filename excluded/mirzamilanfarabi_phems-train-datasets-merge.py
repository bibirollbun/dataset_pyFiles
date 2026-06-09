import numpy as np
import pandas as pd
import os


# Load the dataset
SepsisLabel = pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/SepsisLabel_train.csv")

person_demographics_episode_train = pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/person_demographics_episode_train.csv")

# Merge datasets on 'person_id'
person_age_merged_data = SepsisLabel.merge(person_demographics_episode_train, on='person_id', how='outer')

# Save the merged dataset to a .csv file
person_age_merged_data.to_csv('/kaggle/working/person_age_merged_dataset.csv', index=False)

print("Merged dataset saved to /kaggle/working/person_age_merged_dataset.csv")

person_age_merged_data


# Combine 'person_id' and 'measurement_datetime' into 'person_id_datetime'
person_age_merged_data['person_id_datetime'] = person_age_merged_data['person_id'].astype(str) + '_' + person_age_merged_data['measurement_datetime']

# Select only the required columns
SepsisLabel_train_final_dataset = person_age_merged_data[['person_id_datetime', 'SepsisLabel', 'birth_datetime', 'age_in_months', 'gender']]

# Save the final dataset to the specified location
SepsisLabel_train_final_dataset.to_csv("/kaggle/working/SepsisLabel_train.csv", index=False)

# Display the first few rows of the final dataset
SepsisLabel_train_final_dataset


# Load the dataset
devices_train = pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/devices_train.csv")

# Combine 'person_id' and 'measurement_datetime' into 'person_id_datetime'
devices_train['person_id_datetime'] = devices_train['person_id'].astype(str) + '_' + devices_train['device_datetime_hourly']

# Select only the required columns
devices_train_final_dataset = devices_train[['person_id_datetime', 'device']]

# Save the final dataset to the specified location
devices_train_final_dataset.to_csv("/kaggle/working/devices_train.csv", index=False)

# Display the first few rows of the final dataset
devices_train_final_dataset


# Load the dataset
drugsexposure_train = pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/drugsexposure_train.csv")

# Combine 'person_id' and 'measurement_datetime' into 'person_id_datetime'
drugsexposure_train['person_id_datetime'] = drugsexposure_train['person_id'].astype(str) + '_' + drugsexposure_train['drug_datetime_hourly']

# Select only the required columns
drugsexposure_train_final_dataset = drugsexposure_train[['person_id_datetime', 'drug_concept_id']]

# Save the final dataset to the specified location
drugsexposure_train_final_dataset.to_csv("/kaggle/working/drugsexposure_train.csv", index=False)

# Display the first few rows of the final dataset
drugsexposure_train_final_dataset


# Load the dataset
measurement_lab_train = pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_lab_train.csv")

# Combine 'person_id' and 'measurement_datetime' into 'person_id_datetime'
measurement_lab_train['person_id_datetime'] = measurement_lab_train['person_id'].astype(str) + '_' + measurement_lab_train['measurement_datetime']

# Select only the required columns
measurement_lab_train_final_dataset = measurement_lab_train[['person_id_datetime', 'Base excess in Venous blood by calculation', 'Base excess in Arterial blood by calculation', 'Phosphate [Moles/volume] in Serum or Plasma', 'Potassium [Moles/volume] in Blood', 'Bilirubin.total [Moles/volume] in Serum or Plasma', 'Neutrophil Ab [Units/volume] in Serum', 'Bicarbonate [Moles/volume] in Arterial blood', 'Hematocrit [Volume Fraction] of Blood', 'Glucose [Moles/volume] in Serum or Plasma', 'Calcium [Moles/volume] in Serum or Plasma', 'Chloride [Moles/volume] in Blood', 'Sodium [Moles/volume] in Serum or Plasma', 'C reactive protein [Mass/volume] in Serum or Plasma', 'Carbon dioxide [Partial pressure] in Venous blood', 'Oxygen [Partial pressure] in Venous blood', 'Albumin [Mass/volume] in Serum or Plasma', 'Bicarbonate [Moles/volume] in Venous blood', 'Oxygen [Partial pressure] in Arterial blood', 'Carbon dioxide [Partial pressure] in Arterial blood', 'Interleukin 6 [Mass/volume] in Body fluid', 'Magnesium [Moles/volume] in Blood', 'Prothrombin time (PT)', 'Procalcitonin [Mass/volume] in Serum or Plasma', 'Lactate [Moles/volume] in Blood', 'Creatinine [Mass/volume] in Blood', 'Fibrinogen measurement', 'Bilirubin measurement', 'Partial thromboplastin time', ' activated', 'Total white blood count', 'Platelet count', 'White blood cell count', 'Blood venous pH', 'D-dimer level', 'Blood arterial pH', 'Hemoglobin [Moles/volume] in Blood', 'Ionised calcium measurement']]

# Save the final dataset to the specified location
measurement_lab_train_final_dataset.to_csv("/kaggle/working/measurement_lab_train.csv", index=False)

# Display the first few rows of the final dataset
measurement_lab_train_final_dataset


# Load the dataset
measurement_meds_train = pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_meds_train.csv")

# Combine 'person_id' and 'measurement_datetime' into 'person_id_datetime'
measurement_meds_train['person_id_datetime'] = measurement_meds_train['person_id'].astype(str) + '_' + measurement_meds_train['measurement_datetime']

# Select only the required columns
measurement_meds_train_final_dataset = measurement_meds_train[['person_id_datetime', 'Systolic blood pressure', 'Diastolic blood pressure', 'Body temperature', 'Respiratory rate', 'Heart rate', 'Measurement of oxygen saturation at periphery', 'Oxygen/Gas total [Pure volume fraction] Inhaled gas']]

# Save the final dataset to the specified location
measurement_meds_train_final_dataset.to_csv("/kaggle/working/measurement_meds_train.csv", index=False)

# Display the first few rows of the final dataset
measurement_meds_train_final_dataset


# Load the dataset
measurement_observation_train = pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_observation_train.csv")

# Combine 'person_id' and 'measurement_datetime' into 'person_id_datetime'
measurement_observation_train['person_id_datetime'] = measurement_observation_train['person_id'].astype(str) + '_' + measurement_observation_train['measurement_datetime']

# Select only the required columns
measurement_observation_train_final_dataset = measurement_observation_train[['person_id_datetime', 'Left pupil Diameter Auto', 'Right pupil Diameter Auto', 'Glasgow coma scale', 'Capillary refill [Time]', 'Pulse', 'Arterial pulse pressure', 'Right pupil Pupillary response', 'Left pupil Pupillary response']]

# Save the final dataset to the specified location
measurement_observation_train_final_dataset.to_csv("/kaggle/working/measurement_observation_train.csv", index=False)

# Display the first few rows of the final dataset
measurement_observation_train_final_dataset


# Merge datasets on 'person_id'
train_merged_data = SepsisLabel_train_final_dataset.merge(devices_train_final_dataset, on='person_id_datetime', how='outer') \
                               .merge(drugsexposure_train_final_dataset, on='person_id_datetime', how='outer') \
                               .merge(measurement_lab_train_final_dataset, on='person_id_datetime', how='outer') \
                               .merge(measurement_meds_train_final_dataset, on='person_id_datetime', how='outer') \
                               .merge(measurement_observation_train_final_dataset, on='person_id_datetime', how='outer')

# Fill all NaN values with 0
train_merged_data_filled = train_merged_data.fillna(0)

train_merged_data_filled


train_merged_data_filled_column_list = train_merged_data_filled.columns.tolist()

# Print the list of column names
print("train_merged_data_filled Columns: ")
print(train_merged_data_filled_column_list)


# Define the new column order
new_column_order = [
    'person_id_datetime', 'birth_datetime', 'age_in_months', 'gender', 'device', 'drug_concept_id',
    'Base excess in Venous blood by calculation', 'Base excess in Arterial blood by calculation',
    'Phosphate [Moles/volume] in Serum or Plasma', 'Potassium [Moles/volume] in Blood',
    'Bilirubin.total [Moles/volume] in Serum or Plasma', 'Neutrophil Ab [Units/volume] in Serum',
    'Bicarbonate [Moles/volume] in Arterial blood', 'Hematocrit [Volume Fraction] of Blood',
    'Glucose [Moles/volume] in Serum or Plasma', 'Calcium [Moles/volume] in Serum or Plasma',
    'Chloride [Moles/volume] in Blood', 'Sodium [Moles/volume] in Serum or Plasma',
    'C reactive protein [Mass/volume] in Serum or Plasma', 'Carbon dioxide [Partial pressure] in Venous blood',
    'Oxygen [Partial pressure] in Venous blood', 'Albumin [Mass/volume] in Serum or Plasma',
    'Bicarbonate [Moles/volume] in Venous blood', 'Oxygen [Partial pressure] in Arterial blood',
    'Carbon dioxide [Partial pressure] in Arterial blood', 'Interleukin 6 [Mass/volume] in Body fluid',
    'Magnesium [Moles/volume] in Blood', 'Prothrombin time (PT)', 'Procalcitonin [Mass/volume] in Serum or Plasma',
    'Lactate [Moles/volume] in Blood', 'Creatinine [Mass/volume] in Blood', 'Fibrinogen measurement',
    'Bilirubin measurement', 'Partial thromboplastin time', ' activated', 'Total white blood count',
    'Platelet count', 'White blood cell count', 'Blood venous pH', 'D-dimer level', 'Blood arterial pH',
    'Hemoglobin [Moles/volume] in Blood', 'Ionised calcium measurement', 'Systolic blood pressure',
    'Diastolic blood pressure', 'Body temperature', 'Respiratory rate', 'Heart rate',
    'Measurement of oxygen saturation at periphery', 'Oxygen/Gas total [Pure volume fraction] Inhaled gas',
    'Left pupil Diameter Auto', 'Right pupil Diameter Auto', 'Glasgow coma scale', 'Capillary refill [Time]',
    'Pulse', 'Arterial pulse pressure', 'Right pupil Pupillary response', 'Left pupil Pupillary response',
    'SepsisLabel'
]

# Reorder the columns
train_merged_data_final = train_merged_data_filled[new_column_order]

# Save the reordered dataset to the specified directory
train_merged_data_final.to_csv('/kaggle/working/final_train_merged_dataset.csv', index=False)


train_merged_data_final

