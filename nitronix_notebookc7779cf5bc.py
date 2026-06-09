# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load data from Kaggle input directory
train_data = pd.read_csv('/kaggle/input/alzheimers-disease-risk-prediction-eu-business/train.csv')
test_data = pd.read_csv('/kaggle/input/alzheimers-disease-risk-prediction-eu-business/test.csv')

# Display the first few rows of the training data
print("Training Data:")
print(train_data.head())

# Separate features and target
X = train_data.drop(columns=['Diagnosis'])
y = train_data['Diagnosis']

# Preprocessing
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = RandomForestClassifier(random_state=42, n_estimators=100)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_val)
f1 = f1_score(y_val, y_pred)
print(f'Validation F1 Score: {f1}')

# Predict on the test set
X_test = test_data
test_predictions = pipeline.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({
    'PatientID': test_data['PatientID'],
    'Diagnosis': test_predictions
})

# Save the submission file to Kaggle output directory
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved as 'submission.csv' in the Kaggle working directory.")

