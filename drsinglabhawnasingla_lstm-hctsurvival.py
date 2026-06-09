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


df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")


df.info()



df.sample(5)


df.describe()




def describe_categorical_columns(df):
    """
    Function to describe categorical columns of a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        dict: A dictionary with categorical column names as keys 
              and their description as values.
    """
    # Select categorical columns (object or category types)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Dictionary to hold descriptions
    description = {}
    
    for col in categorical_cols:
        unique_values = df[col].unique()
        value_counts = df[col].value_counts()
        description[col] = {
            'Number of Unique Categories': len(unique_values),
            'Unique Values': list(unique_values),
            'Value Counts': value_counts.to_dict()
        }
    
    return description

categorical_description = describe_categorical_columns(df)
for col, stats in categorical_description.items():
    print(f"Column: {col}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_kde_plots(df):
    """
    Draw KDE plots for all numerical columns in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    # Select numerical columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # Create subplots
    num_cols = len(numeric_columns)
    plt.figure(figsize=(10, 5 * num_cols))
    
    for i, col in enumerate(numeric_columns, start=1):
        plt.subplot(num_cols, 1, i)
        sns.kdeplot(data=df, x=col, fill=True)
        plt.title(f'KDE Plot for {col}', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()

# Draw KDE plots
draw_kde_plots(df)



import pandas as pd
from scipy.stats.mstats import winsorize

def remove_outliers_winsorization(df, lower_quantile=0.05, upper_quantile=0.95):
    """
    Apply Winsorization to numerical columns in a DataFrame to handle outliers.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        lower_quantile (float): The lower quantile limit for Winsorization (default 0.05).
        upper_quantile (float): The upper quantile limit for Winsorization (default 0.95).

    Returns:
        pd.DataFrame: A DataFrame with Winsorized numerical columns.
    """
    df_cleaned = df.copy()
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    for col in numeric_columns:
        # Determine the Winsorization limits
        lower_limit, upper_limit = df[col].quantile([lower_quantile, upper_quantile])
        print(f"\nColumn: {col}")
        print(f"Lower limit: {lower_limit}, Upper limit: {upper_limit}")
        
        # Apply Winsorization
        winsorized_data = winsorize(df[col], limits=(lower_quantile, 1 - upper_quantile))
        df_cleaned[col] = winsorized_data
    
    return df_cleaned

# Example DataFrame


# Remove outliers using Winsorization
df_cleaned = remove_outliers_winsorization(df)

print("\nData after Winsorization:")
print(df_cleaned)



import pandas as pd
from sklearn.impute import KNNImputer

def knn_imputation(df, n_neighbors=5):
    """
    Perform KNN imputation on numerical columns of a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame (can include missing values).
        n_neighbors (int): Number of neighbors to consider for imputing missing values (default 5).
    
    Returns:
        pd.DataFrame: A DataFrame with missing values imputed using KNN.
    """
    # Copy the original DataFrame
    df_imputed = df.copy()
    
    # Select numerical columns for imputation
    numeric_columns = df_imputed.select_dtypes(include=['number']).columns
    
    # Initialize KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Impute the numeric columns
    df_imputed[numeric_columns] = imputer.fit_transform(df_imputed[numeric_columns])
    
    return df_imputed



print("Original DataFrame with missing values:")
print(df_cleaned)

# Apply KNN imputation
df_imputed = knn_imputation(df_cleaned)

print("\nDataFrame after KNN Imputation:")
X_test_df.shape



categorical_description = describe_categorical_columns(df_imputed)
for col, stats in categorical_description.items():
    print(f"Column: {col}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


# Select all categorical columns in df_imputed
categorical_columns = df_imputed.select_dtypes(include=['object', 'category']).columns

# Drop categorical columns from df_imputed
df_imputed_without_categoricals = df_imputed.drop(columns=categorical_columns)

# Display the resulting DataFrame
df_imputed_without_categoricals.shape



# Select all categorical columns from df_imputed
categorical_columns_df_imputed = df_imputed.select_dtypes(include=['object', 'category'])

# Display the resulting DataFrame with categorical columns
categorical_columns_df_imputed.shape




import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder



# Step 1: Encode the categorical data into numerical values
label_encoders = {}
for column in categorical_columns_df_imputed.columns:
    le = LabelEncoder()
    categorical_columns_df_imputed[column] = categorical_columns_df_imputed[column].astype(str)
    categorical_columns_df_imputed[column] = le.fit_transform(categorical_columns_df_imputed[column])
    label_encoders[column] = le  # Save the encoder for future use if needed

# Step 2: Apply KNN Imputation
imputer = KNNImputer(n_neighbors=5)
categorical_columns_imputed = imputer.fit_transform(categorical_columns_df_imputed)

# Step 3: Convert the imputed result back to a DataFrame
categorical_columns_imputed_df = pd.DataFrame(
    categorical_columns_imputed, 
    columns=categorical_columns_df_imputed.columns
)
categorical_columns_imputed_df.shape



# Combine df_imputed_without_categoricals and categorical_columns_df_imputed_encoded
df_combined = pd.concat([df_imputed_without_categoricals, categorical_columns_imputed_df], axis=1)

# Display the resulting DataFrame
df_combined.shape



categorical_columns_imputed_df.shape


# Drop 'efs' and 'efs_time' columns from df_combined
X = df_combined.drop(columns=['efs', 'efs_time'])

# Select only the 'efs' and 'efs_time' columns from df_combined
y = df_combined[['efs', 'efs_time']]

import numpy as np

# Define time_steps (window size) for sequences
time_steps = 10

# Extract features (X) and labels (y) from df_combined
X_raw = df_combined.drop(columns=['efs', 'efs_time']).values  # Convert to NumPy array
y = df_combined[['efs', 'efs_time']].values  # Convert labels to NumPy array

# Function to create sequences
def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:i + time_steps])
        y.append(labels[i + time_steps - 1])  # Predict the last label in the sequence
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(X_raw, y, time_steps)

# X.shape -> (num_samples, time_steps, num_features)
# y.shape -> (num_samples, output_features)

print(f"X shape: {X.shape}, y shape: {y.shape}")






from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the split data
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Build the LSTM model
model = Sequential()

# Add LSTM layers (including input_shape)
model.add(LSTM(64, return_sequences=True, input_shape=(10, 58)))  # time_steps=10, num_features=58
model.add(LSTM(64))  # Add a second LSTM layer for better feature learning
model.add(Dense(2))  # Output layer with 2 units (one for 'efs', one for 'efs_time')

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using Mean Squared Error as the loss function for regression

# Train the model on X_train and y_train
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Print model summary
model.summary()



import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Build the LSTM model
model = Sequential()

# Add LSTM layers (including input_shape)
model.add(LSTM(64, return_sequences=True, input_shape=(10, 58)))  # time_steps=10, num_features=58
model.add(LSTM(64))  # Add a second LSTM layer for better feature learning
model.add(Dense(2))  # Output layer with 2 units (one for 'efs', one for 'efs_time')

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using Mean Squared Error as the loss function for regression

# Train the model on X_train and y_train
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Print model summary
model.summary()


import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# Predict on the test data
y_pred = model.predict(X_val)

# Print the predictions
print(f"Predictions on test data (y_pred):\n{y_pred}")

# Optional: If you want to separate the two outputs ('efs' and 'efs_time'):
predicted_efs = y_pred[:, 0]  # Predicted 'efs' values
predicted_efs_time = y_pred[:, 1]  # Predicted 'efs_time' values

# You can now compare `predicted_efs` and `predicted_efs_time` with the actual values.



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(10, 58)),  # Single output for the entire sequence
    Dense(2)  # Output layer: 2 neurons (efs, efs_time)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Summary
model.summary()

# Train the model
batch_size = 32
epochs = 50
history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
# Assuming `X_test` is already in shape (num_samples, time_steps, num_features)



import numpy as np
from lifelines.utils import concordance_index

# Assuming model has already been trained
# X_test and y_test are the test dataset features and labels
import pandas as pd
import numpy as np



X_test = X_val

# 2. Extract Ground Truth
# Assuming `y_test` contains the ground truth labels
true_times = y_pred[:, 0]  # First column in y_test is survival time
event_observed = y_pred[:, 1]  # Second column indicates event occurred (1 if occurred, 0 if censored)

# 3. Use First Output Feature for Risk Assessment
# (e.g., based on predicted `efs` scores)
predicted_risks = y_pred[:, 0]

# 4. Compute Concordance Index (C-index)
c_index = concordance_index(true_times, predicted_risks, event_observed)
print(f"Concordance Index (C-index): {c_index}")



!pip install lifelines



from lifelines.utils import concordance_index

# Example data
true_times = np.array([4, 3, 2, 1])  # Survival times
predicted_risks = np.array([0.1, 0.4, 0.35, 0.8])  # Predicted risks

c_index = concordance_index(true_times, predicted_risks)
print(f"Concordance Index: {c_index}")



X_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
#X_test.shape
X_test = X_test.values  # Convert to a NumPy array
# Step 1: Convert X_test (which is a NumPy array) to a pandas DataFrame
# Assuming X_test is of shape (num_samples, time_steps, num_features)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)  # Flatten each time step

# Convert to pandas DataFrame to add columns later
X_test_df = pd.DataFrame(X_test_reshaped)
X_test_df.info()




# Perform one-hot encoding on categorical columns
X_test_encoded_df = pd.get_dummies(X_test_df, drop_first=False)

# Display the result
X_test_encoded_df.info()()




# Count the number of null entries in each column of the DataFrame
null_counts = X_test_encoded_df.isnull().sum()

# Display the null counts for each column
null_counts



# Assuming predicted_efs_test and predicted_efs_time_test contain the predicted values for 'efs' and 'efs_time'
# Let's add the two new columns, 'efs' and 'efs_time', to the X_test_encoded_df.
predicted_efs_test = 0
predicted_efs_time_test = 0
# Add the predicted 'efs' and 'efs_time' as new columns to X_test_encoded_df
X_test_encoded_df['efs'] = predicted_efs_test
X_test_encoded_df['efs_time'] = predicted_efs_time_test

# Now, print the updated DataFrame to verify the columns have been added
print(X_test_encoded_df.head())



import numpy as np

# Step 1: Convert X_test_encoded_df (pandas DataFrame) to a NumPy array
X_test_array = X_test_encoded_df.values
print(X_test_encoded_df.shape)
# Step 2: Reshape X_test_array to match the input shape expected by the LSTM (3 samples, 10 time steps, 58 features)
X_test_reshaped = X_test_array.reshape((3, 1, 79))  # 3 samples, 10 time steps, 7 features per time step


# Step 3: Make predictions with the trained model
y_pred_test = model.predict(X_test_final)

# Step 4: Extract the predicted values for 'efs' and 'efs_time'
predicted_efs_test = y_pred_test[:, 0]  # First column corresponds to 'efs'
predicted_efs_time_test = y_pred_test[:, 1]  # Second column corresponds to 'efs_time'

# Step 5: Display the results
print("Predicted 'efs' values for X_test:", predicted_efs_test)
print("Predicted 'efs_time' values for X_test:", predicted_efs_time_test)





