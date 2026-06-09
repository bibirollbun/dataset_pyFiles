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

# Load the dataset
file_path = "/kaggle/input/playground-series-s4e2/train.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Display dataset attributes
print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nFirst 5 Rows:\n", df.head())
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())



# Check data types
print("\nData Types:\n", df.dtypes)

# Check unique values for categorical features
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
for col in categorical_columns:
    print(f"\nUnique values in '{col}':\n", df[col].unique())



from sklearn.preprocessing import LabelEncoder, StandardScaler

# Drop the 'id' column as it's not useful for prediction
df = df.drop(columns=['id'])

# Encode categorical columns
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoder for future use

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Normalize numerical features
scaler = StandardScaler()
numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display processed dataset
print("\nProcessed Data Sample:\n", df.head())



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define features and target
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define features and target
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())  # Get sorted class labels

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



import joblib  # To save and load models

# Save the trained Random Forest model
joblib.dump(model, "obesity_risk_model.pkl")

# Save the scaler (if used for numerical features)
joblib.dump(scaler, "scaler.pkl")  # Ensure `scaler` exists

# Save label encoders for categorical features
joblib.dump(label_encoders, "label_encoders.pkl")  # Ensure `label_encoders` exists

print("\nâœ… Model, scaler, and label encoders saved successfully!")



import pandas as pd
import matplotlib.pyplot as plt

# Get feature importance scores
feature_importance = model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sort by importance (highest to lowest)
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance values
print("\nFeature Importance:\n", importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Obesity Prediction")
plt.gca().invert_yaxis()  # Invert to show highest at the top
plt.show()



import numpy as np
import pandas as pd

# Function to preprocess input data and make predictions
def predict_obesity_risk(model, input_data, scaler, label_encoders):
    """
    Predicts obesity risk based on input data.

    Parameters:
    - model: Trained machine learning model.
    - input_data: Dictionary containing feature values.
    - scaler: StandardScaler used for training.
    - label_encoders: Dictionary of LabelEncoders for categorical features.

    Returns:
    - Predicted class label
    """

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 
                           'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])  # Encode single value

    # Scale numerical features
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Decode the prediction back to the original label
    class_labels = label_encoders['NObeyesdad'].classes_
    predicted_label = class_labels[prediction]

    return predicted_label


# âœ… Example input data for Underweight (Insufficient Weight)
underweight_sample = {
    "Gender": "Male",
    "Age": 25,
    "Height": 1.75,
    "Weight": 50,  # Very low weight (BMI below 18.5)
    "family_history_with_overweight": "no",
    "FAVC": "no",  # Doesn't frequently consume high-calorie foods
    "FCVC": 4,  # High vegetable consumption
    "NCP": 1,  # Very few meals per day
    "CAEC": "no",  # No snacking between meals
    "SMOKE": "no",
    "CH2O": 3,  # High water intake
    "SCC": "no",
    "FAF": 3,  # High physical activity
    "TUE": 0.5,  # Low screen time
    "CALC": "no",  # No alcohol consumption
    "MTRANS": "Walking"  # Active transport
}

# âœ… Make prediction
predicted_class = predict_obesity_risk(model, underweight_sample, scaler, label_encoders)

# âœ… Print the result
print(f"Predicted Obesity Risk Category: {predicted_class}")



import joblib  # To load saved models
import pandas as pd
import numpy as np

# Load the trained model (Update the path if needed)
model = joblib.load("/kaggle/working/obesity_risk_model.pkl")  # Change filename if different

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")  # Make sure to save and load the same scaler

# Load label encoders for categorical features
label_encoders = joblib.load("label_encoders.pkl")  # Dictionary of encoders

# Load dataset again if needed (Ensure X_test and y_test are available)
# X_test and y_test should be extracted from your preprocessing pipeline

# Function to get max values by category
def get_max_values_by_category(model, X, y, scaler, label_encoders):
    import pandas as pd
    import numpy as np

    # Make predictions
    y_pred = model.predict(X)
    
    # Convert X back to original scale (Only apply inverse_transform to numerical features)
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    X_numerical = pd.DataFrame(scaler.inverse_transform(X[numerical_columns]), columns=numerical_columns)
    
    # Add categorical columns back
    X_original = X.copy()
    X_original[numerical_columns] = X_numerical

    # Add predicted labels
    X_original['Predicted_Label'] = y_pred

    # Get max values for each obesity category
    max_values_by_category = X_original.groupby('Predicted_Label').max()

    return max_values_by_category


# Ensure you have X_test and y_test before running
max_values_df = get_max_values_by_category(model, X_test, y_test, scaler, label_encoders)

# Print max values for each obesity category
print("\nðŸ”¹ Maximum values for each category based on model predictions:\n")
print(max_values_df)



underweight_sample = {
    "Gender": "Male",
    "Age": 18,  # Lower age
    "Height": 1.80,  # Taller height
    "Weight": 45,  # Very low weight (BMI below 18.5)
    "family_history_with_overweight": "no",
    "FAVC": "no",  # Doesn't frequently consume high-calorie foods
    "FCVC": 5,  # Very high vegetable consumption
    "NCP": 1,  # Very few meals per day
    "CAEC": "no",  # No snacking between meals
    "SMOKE": "no",
    "CH2O": 4,  # High water intake
    "SCC": "no",
    "FAF": 4,  # High physical activity
    "TUE": 0.3,  # Very low screen time
    "CALC": "no",  # No alcohol consumption
    "MTRANS": "Walking"  # Always walking
}

# âœ… Make prediction
predicted_class = predict_obesity_risk(model, underweight_sample, scaler, label_encoders)

# âœ… Print the result
print(f"Predicted Obesity Risk Category: {predicted_class}")



import numpy as np

# Function to preprocess input data and make predictions
def predict_obesity_risk(model, input_data, scaler, label_encoders):
    """
    Predicts obesity risk based on input data.

    Parameters:
    - model: Trained machine learning model.
    - input_data: Dictionary containing feature values.
    - scaler: StandardScaler used for training.
    - label_encoders: Dictionary of LabelEncoders for categorical features.

    Returns:
    - Predicted class label
    """

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])  # Encode single value

    # Scale numerical features
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Decode the prediction back to the original label
    class_labels = label_encoders['NObeyesdad'].classes_
    predicted_label = class_labels[prediction]

    return predicted_label


# Example input data (Replace with actual values)

new_sample = {
    "Gender": "Male",
    "Age": 18,  # Lower age
    "Height": 1.80,  # Taller height
    "Weight": 45,  # Very low weight (BMI below 18.5)
    "family_history_with_overweight": "no",
    "FAVC": "no",  # Doesn't frequently consume high-calorie foods
    "FCVC": 5,  # Very high vegetable consumption
    "NCP": 1,  # Very few meals per day
    "CAEC": "no",  # No snacking between meals
    "SMOKE": "no",
    "CH2O": 4,  # High water intake
    "SCC": "no",
    "FAF": 4,  # High physical activity
    "TUE": 0.3,  # Very low screen time
    "CALC": "no",  # No alcohol consumption
    "MTRANS": "Walking"  # Always walking
}



# Make prediction
predicted_class = predict_obesity_risk(model, new_sample, scaler, label_encoders)
print(f"Predicted Obesity Risk Category: {predicted_class}")



import numpy as np

# Function to preprocess input data and make predictions
def predict_obesity_risk(model, input_data, scaler, label_encoders):
    """
    Predicts obesity risk based on input data.

    Parameters:
    - model: Trained machine learning model.
    - input_data: Dictionary containing feature values.
    - scaler: StandardScaler used for training.
    - label_encoders: Dictionary of LabelEncoders for categorical features.

    Returns:
    - Predicted class label
    """

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])  # Encode single value

    # Scale numerical features
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Decode the prediction back to the original label
    class_labels = label_encoders['NObeyesdad'].classes_
    predicted_label = class_labels[prediction]

    return predicted_label


# Example input data (Replace with actual values)

new_sample =  {
    "Gender": "Male",
    "Age": 28,
    "Height": 1.70,  # in meters
    "Weight": 90,  # BMI â‰ˆ 31.1 (Obesity Class I)
    "family_history_with_overweight": "yes",
    "FAVC": "yes",  # Frequently eats high-calorie food
    "FCVC": 1.5,  # Low vegetable consumption
    "NCP": 4,  # Higher number of meals per day
    "CAEC": "Frequently",  # Frequent snacking
    "SMOKE": "no",
    "CH2O": 1.0,  # Low water intake
    "SCC": "no",
    "FAF": 0.5,  # Very low physical activity
    "TUE": 3.0,  # High screen time
    "CALC": "Frequently",  # Regular alcohol consumption
    "MTRANS": "Automobile"  # Low activity transport
}


# Make prediction
predicted_class = predict_obesity_risk(model, new_sample, scaler, label_encoders)
print(f"Predicted Obesity Risk Category: {predicted_class}")




import numpy as np

# Function to preprocess input data and make predictions
def predict_obesity_risk(model, input_data, scaler, label_encoders):
    """
    Predicts obesity risk based on input data.

    Parameters:
    - model: Trained machine learning model.
    - input_data: Dictionary containing feature values.
    - scaler: StandardScaler used for training.
    - label_encoders: Dictionary of LabelEncoders for categorical features.

    Returns:
    - Predicted class label
    """

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])  # Encode single value

    # Scale numerical features
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Decode the prediction back to the original label
    class_labels = label_encoders['NObeyesdad'].classes_
    predicted_label = class_labels[prediction]

    return predicted_label


# Example input data (Replace with actual values)

new_sample =  {
    "Gender": "Male",
    "Age": 21,
    "Height": 1.78,
    "Weight": 50,
    "family_history_with_overweight": "no",
    "FAVC": "no",
    "FCVC": 4,
    "NCP": 2,
    "CAEC": "no",
    "SMOKE": "no",
    "CH2O": 3,
    "SCC": "no",
    "FAF": 3,
    "TUE": 1.0,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation"
}


# Make prediction
predicted_class = predict_obesity_risk(model, new_sample, scaler, label_encoders)
print(f"Predicted Obesity Risk Category: {predicted_class}")





import numpy as np

# Function to preprocess input data and make predictions
def predict_obesity_risk(model, input_data, scaler, label_encoders):
    """
    Predicts obesity risk based on input data.

    Parameters:
    - model: Trained machine learning model.
    - input_data: Dictionary containing feature values.
    - scaler: StandardScaler used for training.
    - label_encoders: Dictionary of LabelEncoders for categorical features.

    Returns:
    - Predicted class label
    """

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])  # Encode single value

    # Scale numerical features
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Decode the prediction back to the original label
    class_labels = label_encoders['NObeyesdad'].classes_
    predicted_label = class_labels[prediction]

    return predicted_label


# Example input data (Replace with actual values)

new_sample =  {
    "Gender": "Female",
    "Age": 25,
    "Height": 1.65,
    "Weight": 58,
    "family_history_with_overweight": "no",
    "FAVC": "no",
    "FCVC": 4,
    "NCP": 3,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2.5,
    "SCC": "no",
    "FAF": 2,
    "TUE": 1.5,
    "CALC": "Frequently",
    "MTRANS": "Walking"
}


# Make prediction
predicted_class = predict_obesity_risk(model, new_sample, scaler, label_encoders)
print(f"Predicted Obesity Risk Category: {predicted_class}")



import numpy as np

# Function to preprocess input data and make predictions
def predict_obesity_risk(model, input_data, scaler, label_encoders):
    """
    Predicts obesity risk based on input data.

    Parameters:
    - model: Trained machine learning model.
    - input_data: Dictionary containing feature values.
    - scaler: StandardScaler used for training.
    - label_encoders: Dictionary of LabelEncoders for categorical features.

    Returns:
    - Predicted class label
    """

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])  # Encode single value

    # Scale numerical features
    numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Decode the prediction back to the original label
    class_labels = label_encoders['NObeyesdad'].classes_
    predicted_label = class_labels[prediction]

    return predicted_label


# Example input data (Replace with actual values)

new_sample =  {
    "Gender": "Male",
    "Age": 30,
    "Height": 1.72,
    "Weight": 80,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 3,
    "NCP": 3,
    "CAEC": "Frequently",
    "SMOKE": "no",
    "CH2O": 2,
    "SCC": "yes",
    "FAF": 1.5,
    "TUE": 1,
    "CALC": "Sometimes",
    "MTRANS": "Automobile"
}



# Make prediction
predicted_class = predict_obesity_risk(model, new_sample, scaler, label_encoders)
print(f"Predicted Obesity Risk Category: {predicted_class}")

