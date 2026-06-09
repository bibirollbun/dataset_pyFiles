# Rainfall Prediction using Random Forest Classifier

## Step 1: Import Libraries

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## Step 2: Load the Data

# Load the CSV files
def load_data():
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
    submission_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
    return train_df, test_df, submission_df

train_df, test_df, submission_df = load_data()

## Step 3: Data Preprocessing

# Handle missing values
test_df['winddirection'].fillna(test_df['winddirection'].mean(), inplace=True)

# Features and target
features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
X_train = train_df[features]
y_train = train_df['rainfall']
X_test = test_df[features]

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Step 4: Model Training - Random Forest Classifier

# Split the training data into train and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Initialize and train the model
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train_split, y_train_split)

# Validate the model
y_val_pred = rf_classifier.predict(X_val_split)
accuracy = accuracy_score(y_val_split, y_val_pred)
conf_matrix = confusion_matrix(y_val_split, y_val_pred)
class_report = classification_report(y_val_split, y_val_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

## Step 5: Predict on the Test Dataset 

test_predictions = rf_classifier.predict(X_test_scaled)
submission_df['rainfall'] = test_predictions



