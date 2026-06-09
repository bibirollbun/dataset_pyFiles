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


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the dataset
# Assuming the dataset is loaded from a CSV file
data = """
id,Age,Gender,Comorbidities,Current Medications,Symptoms
001,55,Male,"Hypertension, Hyperlipidemia, Type 2 Diabetes, CAD","Metformin, Lisinopril, Atorvastatin","Fatigue, Blurred vision, Polydipsia"
002,67,Female,"Osteoarthritis, COPD, Hypertension, Asthma","Metformin, Albuterol, Hydrochlorothiazide","Shortness of breath, Joint pain"
003,44,Male,"Obesity, Hypertension, Sleep Apnea, Hyperlipidemia","Simvastatin, CPAP (for sleep apnea)","Excessive daytime sleepiness, Headaches"
004,72,Female,"Hypertension, Osteoporosis, Chronic kidney disease, Hyperlipidemia","Lisinopril, Atorvastatin, Calcium supplements","Bone pain, Swelling in legs"
005,63,Male,"Prostate Enlargement, Hypertension, Stroke, Diabetes","Metformin, Losartan, Aspirin","Weakness, Nausea, Blurred vision"
006,49,Female,"Hypertension, Asthma, Fibromyalgia, Anxiety","Albuterol, Hydrochlorothiazide, Zoloft","Chest tightness, Chronic cough"
007,58,Male,"Obesity, Type 2 Diabetes, Hypertension, Angina, Hyperlipidemia","Metformin, Nitroglycerin, Amlodipine","Chest pain, Shortness of breath"
008,70,Female,"Hypertension, Hypothyroidism, Depression, Chronic Back Pain","Levothyroxine, Lisinopril, Sertraline","Fatigue, Depressed mood, Constipation"
009,45,Male,"Obesity, Hypertension, Depression, Hyperlipidemia, Anxiety","Zoloft, Hydrochlorothiazide, Lisinopril","Depression, Insomnia, Weight gain"
010,38,Female,"Asthma, Chronic Fatigue Syndrome, Irritable Bowel Syndrome","Albuterol, Fluticasone, Magnesium supplements","Abdominal pain, Shortness of breath"
"""

from io import StringIO
patient_data = pd.read_csv(StringIO(data))

# Step 3: Inspect data and clean
# Handle missing values (using ffill() directly)
patient_data.ffill(inplace=True)

# Inspect the column names to ensure correct references
print(patient_data.columns)

# Step 4: Encode categorical columns (Gender)
le_gender = LabelEncoder()
patient_data['Gender'] = le_gender.fit_transform(patient_data['Gender'])

# Step 5: Handle Comorbidities and Symptoms using One-Hot Encoding
comorbidity_column = 'Comorbidities'
symptom_column = 'Symptoms'

# One-hot encoding for 'Comorbidities' and 'Symptoms' (splitting by commas)
comorbidities = patient_data[comorbidity_column].str.get_dummies(sep=', ')
symptoms = patient_data[symptom_column].str.get_dummies(sep=', ')

# Concatenate the encoded columns with the original dataframe
patient_data = pd.concat([patient_data, comorbidities, symptoms], axis=1)

# Drop the original columns for Comorbidities and Symptoms
patient_data.drop(columns=[comorbidity_column, symptom_column], inplace=True)

# Step 6: Treatment Recommendation Logic (Augmented Modal Reasoning)
def recommend_treatment(patient_row, comorbidity_columns, symptom_columns):
    """
    Generate treatment recommendations based on patient features.
    
    Args:
        patient_row (pd.Series): A row of the patient data.
        comorbidity_columns (pd.DataFrame): The columns related to comorbidities.
        symptom_columns (pd.DataFrame): The columns related to symptoms.
        
    Returns:
        str: Recommended treatment.
    """
    # Extract relevant information from the row
    comorbidities = patient_row[comorbidity_columns]
    symptoms = patient_row[symptom_columns]
    
    recommendations = []
    
    # Apply treatment rules based on conditions
    if comorbidities['Hypertension'] == 1:
        recommendations.append("Consider prescribing antihypertensive medications.")
    if comorbidities['Diabetes'] == 1:
        recommendations.append("Monitor blood sugar and consider insulin or oral hypoglycemics.")
    if comorbidities['Obesity'] == 1:
        recommendations.append("Recommend weight management programs and lifestyle changes.")
    if comorbidities['Asthma'] == 1:
        recommendations.append("Prescribe inhalers or bronchodilators.")
    if 'Fatigue' in symptoms:
        recommendations.append("Evaluate for sleep disorders or anemia. Consider iron supplements or sleep apnea treatment.")
    if 'Shortness of breath' in symptoms:
        recommendations.append("Evaluate respiratory function and consider medications for COPD or heart failure.")
    
    if not recommendations:
        recommendations.append("General health check-up and specialist consultation.")
    
    return " ".join(recommendations)

# Apply the treatment recommendation logic to each patient in the dataset
patient_data['Treatment_Recommendation'] = patient_data.apply(
    lambda row: recommend_treatment(row, comorbidities.columns, symptoms.columns), axis=1
)

# Step 7: Display updated dataset
print(patient_data[['id', 'Treatment_Recommendation']])

# Save the final dataframe with treatment recommendations to CSV
patient_data.to_csv('patient_data_with_recommendations.csv', index=False)
print("Treatment recommendations have been added and saved to CSV successfully!")


