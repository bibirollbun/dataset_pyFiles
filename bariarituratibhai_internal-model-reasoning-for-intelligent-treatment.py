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

# Load dataset from Kaggle input folder
data = pd.read_csv("/kaggle/input/patient-data-csv/patient_data.csv")

# Internal reasoning model function
def recommend_treatment(row):
    if row["Condition"] == "Hypertension" and "Diabetes" in str(row["Comorbidities"]):
        return "ACE Inhibitor"
    elif row["Condition"] == "Asthma":
        return "Inhaler"
    elif row["Condition"] == "Heart Disease":
        return "Statins and Beta-blocker"
    elif row["Condition"] == "Diabetes":
        return "Metformin"
    else:
        return "Consult Specialist"

# Apply model to dataset
data["Treatment_Recommended"] = data.apply(recommend_treatment, axis=1)

# Create submission file with the necessary columns
submission = data[["id", "Treatment_Recommended"]]

# Save to CSV for submission
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("Submission file created successfully!")

# Display first few rows of the submission file
submission.head()



# One-hot encoding for categorical features (Condition, Comorbidities)
df_encoded = pd.get_dummies(df, columns=['Condition', 'Comorbidities'], drop_first=True)

# Mapping Treatment_Recommended to numerical labels
treatment_mapping = {
    'ACE Inhibitor': 0,
    'Inhaler': 1,
    'Statins and Beta-blocker': 2,
    'Metformin': 3,
    'Inhaler and Steroid': 4
}

df_encoded['Treatment_Recommended'] = df_encoded['Treatment_Recommended'].map(treatment_mapping)

# Display the preprocessed data
df_encoded.head()



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data into features (X) and target (y)
X = df_encoded.drop(columns=['id', 'Treatment_Recommended'])
y = df_encoded['Treatment_Recommended']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:\n', classification_report(y_test, y_pred))



import pandas as pd

# Make sure your treatment_mapping dictionary is defined
treatment_mapping = {
    'ACE Inhibitor': 0,
    'Inhaler': 1,
    'Statins and Beta-blocker': 2,
    'Metformin': 3,
    'Inhaler and Steroid': 4
}

def recommend_treatment(age, condition, comorbidities, model, feature_columns):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Condition_Heart Disease': [1 if condition == 'Heart Disease' else 0],
        'Condition_Hypertension': [1 if condition == 'Hypertension' else 0],
        'Condition_Asthma': [1 if condition == 'Asthma' else 0],
        'Condition_Diabetes': [1 if condition == 'Diabetes' else 0],
        'Comorbidities_Obesity': [1 if comorbidities == 'Obesity' else 0],
        'Comorbidities_Diabetes': [1 if comorbidities == 'Diabetes' else 0],
        'Comorbidities_Hypertension': [1 if comorbidities == 'Hypertension' else 0],
        'Comorbidities_Chronic Kidney Disease': [1 if comorbidities == 'Chronic Kidney Disease' else 0],
        'Comorbidities_Allergy': [1 if comorbidities == 'Allergy' else 0],
        'Comorbidities_Sinusitis': [1 if comorbidities == 'Sinusitis' else 0],
        'Comorbidities_Arthritis': [1 if comorbidities == 'Arthritis' else 0],
    })

    # Ensure input_data has all feature columns (add missing ones with 0)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the trained model's feature order
    input_data = input_data[feature_columns]

    # Predict treatment using the trained model
    treatment_index = model.predict(input_data)[0]

    # Reverse map from numeric index to treatment
    reverse_mapping = {v: k for k, v in treatment_mapping.items()}
    treatment = reverse_mapping[treatment_index]

    return treatment


# Example usage:
age = 65
condition = 'Hypertension'
comorbidities = 'Diabetes'

# List of feature columns used during training (must match model's feature columns)
feature_columns = X.columns.tolist()  # This assumes X is the feature set used during model training

recommended_treatment = recommend_treatment(age, condition, comorbidities, model, feature_columns)
print(f'Recommended Treatment: {recommended_treatment}')


