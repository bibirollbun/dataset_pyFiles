# Step 1: Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load


df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')
data_description = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


# Step 2: Load the Data
# Assuming the DataFrame is named 'df'
# df = pd.read_csv('your_data.csv')  # Uncomment if loading from CSV

# Step 3: Handle Missing Values
# Identify columns with missing values
missing_counts = df.isnull().sum()
print(missing_counts)

# Decide to drop columns with too many missing values or impute them
# For simplicity, drop columns with more than 50% missing values
threshold = len(df) * 0.5
cols_to_drop = missing_counts[missing_counts > threshold].index
df = df.drop(columns=cols_to_drop)

# Impute remaining missing values
# For numerical columns, fill with mean
# For categorical columns, fill with most frequent value

# Step 4: Select Features and Target Variable
# Choose 'efs' as the target variable
y = df['efs']
X = df.drop(columns=['efs', 'efs_time'])  # Drop target and any irrelevant columns

# Step 5: Encode Categorical Variables
# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Step 6: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Scale and Preprocess the Features
X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
X_test_preprocessed = preprocessor.transform(X_test)
# Save the preprocessing pipeline
dump(preprocessor, 'preprocessor.pkl')


# Step 8: Build a Simple Neural Network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Step 9: Compile the Model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

# Step 10: Train the Model
history = model.fit(X_train_preprocessed, y_train, epochs=3, batch_size=32, validation_split=0.1)
# Save the model
model.save('model.h5')

# Step 11: Evaluate the Model
loss, mae = model.evaluate(X_test_preprocessed, y_test)
print(f'Test MAE: {mae:.4f}')


# Load the preprocessing pipeline
preprocessor = load('/kaggle/working/preprocessor.pkl')

# Load the model
model = tf.keras.models.load_model('/kaggle/working/model.h5')


# Preprocess submission data
X_submission_preprocessed = preprocessor.transform(test)


# Make predictions
predictions = model.predict(X_submission_preprocessed).flatten()
sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv', index_col='ID')
sub.prediction = predictions
# Save submission
sub.to_csv('submission.csv')


sub

