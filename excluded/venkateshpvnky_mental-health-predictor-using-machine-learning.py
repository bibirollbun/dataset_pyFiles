# prompt: import required libraries for data analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
df.head()


df.info()



# Create a mapping dictionary
mapping = {'Yes': 1, 'No': 0}

# Apply the mapping to the specified columns
for col in ['Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']:
    df[col] = df[col].map(mapping)

df.head()


df["Sleep Duration"].value_counts()


# Define a mapping to group similar durations
duration_mapping = {
    "Less than 5 hours": "Less than 5 hours",
    "5-6 hours": "5-6 hours",
    "6-7 hours": "6-7 hours",
    "7-8 hours": "7-8 hours",
    "8-9 hours": "8-9 hours",
    "9-11 hours": "9-11 hours",
    "More than 8 hours": "More than 8 hours",
    # Group other ranges into approximate categories
    "4-5 hours": "4-6 hours",
    "4-6 hours": "4-6 hours",
    "6-8 hours": "6-8 hours",
    "3-4 hours": "Less than 5 hours",
    "3-6 hours": "4-6 hours",
    "1-6 hours": "Less than 5 hours",
    "2-3 hours": "Less than 5 hours",
    # Handle invalid data
    "No": "Invalid",
    "Sleep_Duration": "Invalid",
    "Unhealthy": "Invalid",
    "Moderate": "Invalid",
    "Pune": "Invalid",
    "Indore": "Invalid",
    "Work_Study_Hours": "Invalid",
    "9-5": "Invalid",
    "10-6 hours": "Invalid",
    "than 5 hours": "Invalid",
    "49 hours": "Invalid",
    "45": "Invalid",
    "45-48 hours": "Invalid",
    "35-36 hours": "Invalid",
    "55-66 hours": "Invalid",
}

# Map the sleep durations
df['Sleep Duration clean'] = df['Sleep Duration'].map(duration_mapping)



df["Sleep Duration clean"].value_counts()


df["Sleep Duration clean"].unique()


map = {
    'More than 8 hours' : 9.0,
     'Less than 5 hours' : 4.0,
     '5-6 hours' : 5.5,
     '7-8 hours' : 7.5,
     'Invalid' : 0 ,
     '6-8 hours' : 7,
     '4-6 hours' : 5,
     '6-7 hours' : 6.5,
     '8-9 hours' : 8.5,
     '9-11 hours' : 10.0
}

df["sleep"] = df["Sleep Duration clean"].map(map)
df["sleep"].head()


df["sleep"].value_counts()


df["Depression"] = df["Depression"].astype(int)
df["Depression"].head()


df["Age"] = df["Age"].astype(int)
df["Age"].head()


df["Dietary Habits"].value_counts()


valid_categories = ["Moderate", "Unhealthy", "Healthy"]

# Step 2: Clean the data
# Map synonymous or invalid values to valid categories
synonym_mapping = {
    "More Healthy": "Healthy",
    "Less Healthy": "Unhealthy",
    "Less than Healthy": "Unhealthy",
    "No Healthy": "Unhealthy"
}

# Replace synonyms with valid categories
df["Dietary Habits"] = df["Dietary Habits"].replace(synonym_mapping)

df = df[df["Dietary Habits"].isin(valid_categories)]



df["Dietary Habits"].value_counts()


df["Work/Study Hours"].unique()


df["Work/Study Hours"] = df["Work/Study Hours"].astype(int)
df["Work/Study Hours"].head()


df["Financial Stress"].value_counts()
df["Financial Stress"].unique()


df["Financial Stress"] = df["Financial Stress"].fillna(0).astype(int)
df["Financial Stress"].head()


df["Study Satisfaction"].fillna(0, inplace=True)
df["Study Satisfaction"] = df["Study Satisfaction"].astype(int)


df["Job Satisfaction"].fillna(0, inplace=True)
df["Job Satisfaction"] = df["Job Satisfaction"].astype(int)


df["Study Satisfaction"].value_counts()


df["Job Satisfaction"].value_counts()


df["Satisfaction"] = df["Job Satisfaction"] + df["Study Satisfaction"]
df["Satisfaction"].value_counts()


df["Academic Pressure"].fillna(0, inplace=True)
df["Academic Pressure"] = df["Academic Pressure"].astype(int)


df["Work Pressure"].fillna(0, inplace=True)
df["Work Pressure"] = df["Work Pressure"].astype(int)


df["Pressure"] = df["Academic Pressure"] + df["Work Pressure"]
df["Pressure"].value_counts()


df["CGPA"].fillna(0, inplace=True)


map = {
   "Male" : 1,
   "Female" : 0
}

df["Gender"] = df["Gender"].map(map)


df["Gender"].head()


df["Profession"].unique()


df.info()


X = df[['Gender','Age','sleep','Work/Study Hours','Pressure','Financial Stress','Satisfaction','Family History of Mental Illness','Have you ever had suicidal thoughts ?']].fillna(0)
y = df["Depression"]


X.head()


# Create a DataFrame for rows containing "working profession"
work = df[df["Working Professional or Student"].str.contains("Working Professional", case=False)].reset_index(drop=True)

# Create a DataFrame for rows containing "student"
stud = df[df["Working Professional or Student"].str.contains("Student", case=False)].reset_index(drop=True)


# prompt: download newely created work , stud as csv

work.to_csv('work.csv', index=False)
stud.to_csv('stud.csv', index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.metrics import accuracy_score

# Initialize SVM models with different hyperparameters
svm_models = [
    SVC(kernel='linear', C=1),
    SVC(kernel='rbf', C=10, gamma=0.1),
    SVC(kernel='poly', degree=3, C=1)
]

# Train and evaluate each model
accuracies = []
for model in svm_models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for {model}: {accuracy}")

print(f"All accuracies: {accuracies}")


# Initialize and train a RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions using the RandomForestClassifier
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")


# Initialize and train an MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Make predictions using the MLPClassifier
mlp_predictions = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print(f"MLP Accuracy: {mlp_accuracy}")


# Create a voting classifier
estimators = [
    ('svm_linear', svm_models[0]),
    ('svm_rbf', svm_models[1]),
    ('svm_poly', svm_models[2]),
    ('random_forest', rf_model),
    ('mlp', mlp_model)
]

ensemble_model = VotingClassifier(estimators=estimators, voting='hard')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions using the ensemble model
ensemble_predictions = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Ensemble Accuracy: {ensemble_accuracy}")


#table the accuricies
data_acc = {
    'Model': ['SVM (linear)', 'SVM (rbf)', 'SVM (poly)', 'Random Forest', 'MLP', 'Ensemble'],
    'Accuracy': accuracies + [rf_accuracy, mlp_accuracy, ensemble_accuracy]
}

accuracy_df = pd.DataFrame(data_acc)
accuracy_df


import joblib

# Assuming 'ensemble_model' is your trained ensemble model
# Save the trained model to a file
joblib.dump(ensemble_model, '/kaggle/working/ensemble_model.pkl')

# Save the scaler to a file
joblib.dump(scaler, '/kaggle/working/scaler.pkl')


# Load the test dataset
df_test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')

# Preprocessing steps (mirroring the training data preprocessing)
mapping = {'Yes': 1, 'No': 0}
for col in ['Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']:
    df_test[col] = df_test[col].map(mapping)

duration_mapping = {
    "Less than 5 hours": "Less than 5 hours",
    "5-6 hours": "5-6 hours",
    "6-7 hours": "6-7 hours",
    "7-8 hours": "7-8 hours",
    "8-9 hours": "8-9 hours",
    "9-11 hours": "9-11 hours",
    "More than 8 hours": "More than 8 hours",
    # Group other ranges into approximate categories
    "4-5 hours": "4-6 hours",
    "4-6 hours": "4-6 hours",
    "6-8 hours": "6-8 hours",
    "3-4 hours": "Less than 5 hours",
    "3-6 hours": "4-6 hours",
    "1-6 hours": "Less than 5 hours",
    "2-3 hours": "Less than 5 hours",
    # Handle invalid data
    "No": "Invalid",
    "Sleep_Duration": "Invalid",
    "Unhealthy": "Invalid",
    "Moderate": "Invalid",
    "Pune": "Invalid",
    "Indore": "Invalid",
    "Work_Study_Hours": "Invalid",
    "9-5": "Invalid",
    "10-6 hours": "Invalid",
    "than 5 hours": "Invalid",
    "49 hours": "Invalid",
    "45": "Invalid",
    "45-48 hours": "Invalid",
    "35-36 hours": "Invalid",
    "55-66 hours": "Invalid",
}
df_test['Sleep Duration clean'] = df_test['Sleep Duration'].map(duration_mapping)

map_sleep = {
    'More than 8 hours' : 9.0,
     'Less than 5 hours' : 4.0,
     '5-6 hours' : 5.5,
     '7-8 hours' : 7.5,
     'Invalid' : 0 ,
     '6-8 hours' : 7,
     '4-6 hours' : 5,
     '6-7 hours' : 6.5,
     '8-9 hours' : 8.5,
     '9-11 hours' : 10.0
}
df_test["sleep"] = df_test["Sleep Duration clean"].map(map_sleep)


valid_categories = ["Moderate", "Unhealthy", "Healthy"]
synonym_mapping = {
    "More Healthy": "Healthy",
    "Less Healthy": "Unhealthy",
    "Less than Healthy": "Unhealthy",
    "No Healthy": "Unhealthy"
}
df_test["Dietary Habits"] = df_test["Dietary Habits"].replace(synonym_mapping)
df_test = df_test[df_test["Dietary Habits"].isin(valid_categories)]


df_test["Work/Study Hours"] = df_test["Work/Study Hours"].astype(int)
df_test["Financial Stress"] = df_test["Financial Stress"].fillna(0).astype(int)
df_test["Study Satisfaction"].fillna(0, inplace=True)
df_test["Study Satisfaction"] = df_test["Study Satisfaction"].astype(int)
df_test["Job Satisfaction"].fillna(0, inplace=True)
df_test["Job Satisfaction"] = df_test["Job Satisfaction"].astype(int)
df_test["Satisfaction"] = df_test["Job Satisfaction"] + df_test["Study Satisfaction"]
df_test["Academic Pressure"].fillna(0, inplace=True)
df_test["Academic Pressure"] = df_test["Academic Pressure"].astype(int)
df_test["Work Pressure"].fillna(0, inplace=True)
df_test["Work Pressure"] = df_test["Work Pressure"].astype(int)
df_test["Pressure"] = df_test["Academic Pressure"] + df_test["Work Pressure"]
df_test["CGPA"].fillna(0, inplace=True)

map_gender = {
   "Male" : 1,
   "Female" : 0
}
df_test["Gender"] = df_test["Gender"].map(map_gender)


 #Feature selection (same as training data)
X_test_final = df_test[['Gender','Age','sleep','Work/Study Hours','Pressure','Financial Stress','Satisfaction','Family History of Mental Illness','Have you ever had suicidal thoughts ?']].fillna(0)



# Scale the test data using the same scaler used for training

X_test_final = scaler.fit_transform(X_test_final)

# Make predictions on the test set
y_pred = ensemble_model.predict(X_test_final)


# Create a new DataFrame with 'id' and 'predicted' columns
new_df = pd.DataFrame({'id': df_test['id'], 'predicted': y_pred})

# Assuming 'new_df' is already defined as in the provided code.
new_df.to_csv('submission.csv', index=False)

