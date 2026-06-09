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
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')

# Display the first 5 rows
print("First 5 rows of the training data:")
display(train_df.head())

# Display basic information about the DataFrame
print("\nInfo about the training data:")
train_df.info()

# Display descriptive statistics
print("\nDescriptive statistics of the training data:")
display(train_df.describe())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the DataFrame from the provided data
df = pd.DataFrame(train_df)

# Set the style for the plots
sns.set(style="whitegrid")

# Plot 1: Accident Risk by Road Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='road_type', y='accident_risk', data=df)
plt.title('Accident Risk by Road Type')
plt.show()

# Plot 2: Accident Risk by Weather
plt.figure(figsize=(10, 6))
sns.boxplot(x='weather', y='accident_risk', data=df)
plt.title('Accident Risk by Weather')
plt.show()

# Plot 3: Accident Risk by Time of Day
plt.figure(figsize=(10, 6))
sns.boxplot(x='time_of_day', y='accident_risk', data=df)
plt.title('Accident Risk by Time of Day')
plt.show()

# Plot 4: Number of Reported Accidents by Road Type
plt.figure(figsize=(10, 6))
sns.countplot(x='road_type', hue='num_reported_accidents', data=df)
plt.title('Number of Reported Accidents by Road Type')
plt.show()


#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder

# List of categorical columns
categorical_columns = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

# Initialize label encoders
label_encoders = {}

# Encode categorical variables
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le



#Scaling Numerical Features
from sklearn.preprocessing import StandardScaler

# List of numerical columns
numerical_columns = ['num_lanes', 'curvature', 'speed_limit', 'accident_risk']

# Initialize the scaler
scaler = StandardScaler()

# Scale numerical features
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# Check for missing values
print(df.isnull().sum())

# If missing values are found, you can use the following methods to handle them:

# Fill missing values with the mean (for numerical columns)
# df.fillna(df.mean(), inplace=True)

# Fill missing values with the mode (for categorical columns)
# df.fillna(df.mode().iloc[0], inplace=True)

# Or drop rows with missing values
# df.dropna(inplace=True)


# Plan
# 1. Correct the call to get_feature_names_out by removing the argument.
# 2. Ensure the indices of the original DataFrame and the new polynomial DataFrame align before concatenating.
# 3. Add a check for the first interaction term to ensure columns are numeric.

# Feature Engineering
# Interaction terms
# Note: This line assumes 'road_type' and 'weather' are numerically encoded.
# If they are strings (e.g., 'highway', 'rainy'), you must encode them first.
df['road_weather_interaction'] = df['road_type'] * df['weather']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
numerical_columns = ['num_lanes', 'curvature', 'speed_limit', 'accident_risk']
poly = PolynomialFeatures(degree=2, include_bias=False)

# Fit and transform the data
poly_features = poly.fit_transform(df[numerical_columns])

# Fix: Call get_feature_names_out() without arguments.
# The transformer already knows the names from the .fit_transform() step.
poly_df = pd.DataFrame(
    poly_features,
    columns=poly.get_feature_names_out(), # Corrected line
    index=df.index # Important: Keep original index to prevent misalignment
)

# Drop original columns to avoid duplication before concatenating
df_poly_ready = df.drop(columns=numerical_columns)
df = pd.concat([df_poly_ready, poly_df], axis=1)

# Binning numerical features
# Note: Binning should now use one of the new polynomial features or an original one
# if you hadn't dropped it. Let's assume you want to bin the original speed limit.
# We'll re-add it for this purpose.
df['speed_limit'] = poly_df['speed_limit'] # Re-add the original column from poly_df
df['speed_binned'] = pd.cut(df['speed_limit'], bins=3, labels=['low', 'medium', 'high'])

# Display the new features
print(df.head())


# Plan
# 1. Use pd.get_dummies() to perform one-hot encoding on the DataFrame.
# 2. Use this new, fully-numeric DataFrame to calculate the correlation matrix.
# 3. Add a best-practice tip: use drop_first=True to avoid multicollinearity in VIF.
# 4. Recalculate VIF on the final encoded data.

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np # Make sure numpy is imported

# --- Step 1: Perform One-Hot Encoding ---
# This will automatically convert 'speed_binned' and any other non-numeric columns.
# We use drop_first=True to avoid perfect multicollinearity (dummy variable trap).
df_encoded = pd.get_dummies(df, drop_first=True)

print("--- DataFrame after One-Hot Encoding ---")
print(df_encoded.head())


# --- Step 2: Calculate Correlation on the Encoded Data ---
# Now, .corr() will work because all columns are numeric.
correlation_matrix = df_encoded.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10)) # Increased size for better readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


# Plan
# 1. Perform one-hot encoding as before.
# 2. Add a new step to convert any remaining boolean columns to integers (0/1).
# 3. Proceed with the VIF calculation on the fully numeric DataFrame.

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

# --- Step 1: Perform One-Hot Encoding ---
df_encoded = pd.get_dummies(df, drop_first=True)


# --- Step 2: Convert Boolean Columns to Integers ---
# Find columns that are of boolean type and convert them to integers
for col in df_encoded.select_dtypes(include=['bool']).columns:
    df_encoded[col] = df_encoded[col].astype(int)

# You can check the data types to confirm the change
# print(df_encoded.dtypes)


# --- Step 3: Clean and Calculate VIF ---
# VIF requires a clean, numeric, and non-infinite dataset
df_vif = df_encoded.dropna()
df_vif = df_vif.loc[~df_vif.isin([np.inf, -np.inf]).any(axis=1)]

VIF = pd.DataFrame()
if not df_vif.empty:
    # This line should now work correctly
    VIF["VIF Factor"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
    VIF["features"] = df_vif.columns
    print("\n--- Variance Inflation Factor (VIF) ---")
    print(VIF.sort_values('VIF Factor', ascending=False))
else:
    print("\nDataFrame is empty after dropping NaN/inf values. Cannot calculate VIF.")


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Plan
# 1. Import all necessary libraries.
# 2. Load the training data.
# 3. Define features (X) and target (y).
# 4. Create a preprocessor, ensuring the OneHotEncoder can handle unknown categories.
# 5. Split data into training and validation sets.
# 6. Fit the preprocessor on the training data and transform both sets.
# 7. Build, compile, and train the neural network.
# 8. Evaluate the model and visualize its performance.
# 9. Save the final fitted preprocessor and trained model to files.

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 2. Load Data ---
# Replace with the actual path to your training data
df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')

# --- 3. Define Target and Features ---
target = 'accident_risk'
features = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
    'weather', 'road_signs_present', 'public_road', 'time_of_day',
    'holiday', 'school_season'
]
X = df[features]
y = df[target]

# Define categorical and numerical features
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'holiday', 'school_season']
numerical_features = ['num_lanes', 'curvature', 'speed_limit']

# --- 4. Create Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        # MODIFIED: handle_unknown='ignore' makes the pipeline robust
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 5. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Preprocess the Data ---
# Fit the preprocessor on the training data and transform both training and test sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- 7. Define and Train the Neural Network ---
# Define the architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_processed.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(1)
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
history = model.fit(
    X_train_processed, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# --- 8. Evaluate and Visualize ---
# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Evaluate the model on the held-out test data
y_pred = model.predict(X_test_processed).flatten()
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
print(f'\nTest Mean Absolute Error: {test_mae}')
print(f'Test Mean Squared Error: {test_mse}')

# Plot predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Display model summary
model.summary()

# --- 9. Save the Model and Preprocessor ---
# This is the crucial final step
joblib.dump(preprocessor, 'preprocessor.joblib')
model.save('model.keras')

print("\n✅ Preprocessor saved to 'preprocessor.joblib'")
print("✅ Model saved to 'model.keras'")


# Plan
# 1. Import all necessary libraries.
# 2. Define the filenames for the saved model and preprocessor.
# 3. Load the fitted preprocessor and the trained model.
# 4. Load the new test data from 'test.csv'.
# 5. Define the list of feature columns (must be identical to training).
# 6. Select features, transform them using the loaded preprocessor, and make predictions.
# 7. Format the predictions into a submission file.

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras # Or from keras.models import load_model

print("Libraries imported successfully.")

# --- 2. Define Filenames ---
PREPROCESSOR_FILE = '/kaggle/working/preprocessor.joblib'
MODEL_FILE = '/kaggle/working/model.keras' # Or 'model.h5' if you saved in that format

# --- 3. Load Model and Preprocessor ---
try:
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    model = keras.models.load_model(MODEL_FILE)
    print("Model and preprocessor loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure 'preprocessor.joblib' and 'model.keras' are in the same directory.")
    exit() # Exit the script if files are not found

# --- 4. Load Test Data ---
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
print(f"Test data loaded successfully with {test_df.shape[0]} rows.")

# --- 5. Define Feature Columns ---
# This list must be exactly the same as the one used for training
features = [
    'road_type',
    'num_lanes',
    'curvature',
    'speed_limit',
    'lighting',
    'weather',
    'road_signs_present',
    'public_road',
    'time_of_day',
    'holiday',
    'school_season'
]

# --- 6. Preprocess Data and Make Predictions ---
# Select the feature columns from the test data
X_new_test = test_df[features]

# Transform the new data using the loaded preprocessor
print("Transforming test data...")
X_new_test_processed = preprocessor.transform(X_new_test)

# Predict using the loaded model
print("Making predictions...")
predictions = model.predict(X_new_test_processed).flatten()

# --- 7. Format and Save Submission File ---
# Create a DataFrame for the submission file
# Assumes 'test.csv' has an 'id' column
results_df = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': predictions # Use the original target column name
})

# Save the results to a CSV file
results_df.to_csv('submission.csv', index=False)

print("\n--- Predictions complete! ---")
print("Submission file 'submission.csv' created successfully.")
print(results_df.head())

