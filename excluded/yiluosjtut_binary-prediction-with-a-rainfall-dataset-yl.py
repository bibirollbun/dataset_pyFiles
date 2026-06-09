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


# SECTION 1: Setup & Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Although SMOTE is imported in your original code,
# we are using a simple synthetic data augmentation (Gaussian noise)
# from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)




# SECTION 2: Data Loading & Initial Exploration

# Adjust file paths as needed
train_path = "/kaggle/input/playground-series-s5e3/train.csv"  # e.g., "./data/train.csv"
test_path  = "/kaggle/input/playground-series-s5e3/test.csv"   # e.g., "./data/test.csv"

# Load the datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# (Optional) Store the test IDs for submission if available
if 'id' in test.columns:
    test_ids = test['id']

# Display basic info and first few rows
print("Train Data Sample:")
print(train.head())
print("\nTrain Data Info:")
print(train.info())

# Plot a correlation heatmap for numeric features
corr = train.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='viridis', interpolation='none')
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()


# SECTION 3: Data Preprocessing & Feature Engineering

def process_data(df, is_train=True):
    """
    Preprocess and engineer features for the dataset.
    
    Steps:
      - Fill missing values with the median of each column.
      - Create new features:
          * 'temp_range' = maxtemp - mintemp
          * 'hci' = humidity * cloud
          * 'hsi' = humidity * sunshine
          * 'csr' = cloud / (sunshine + 1e-5)  [avoid division by zero]
          * 'rd' = 100 - humidity
          * 'sp' = sunshine / (sunshine + cloud + 1e-5)
          * 'wi' = 0.4*humidity + 0.3*cloud - 0.3*sunshine
      - Encode any categorical (object type) columns using LabelEncoder.
      - For training data, move the target column 'rainfall' to the end.
    
    Parameters:
      df: pandas DataFrame (train or test)
      is_train: Boolean indicating if df is training data
      
    Returns:
      Processed DataFrame.
    """
    # Fill missing values with median
    df.fillna(df.median(), inplace=True)
    
    # Create new features
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['hci'] = df['humidity'] * df['cloud']
    df['hsi'] = df['humidity'] * df['sunshine']
    df['csr'] = df['cloud'] / (df['sunshine'] + 1e-5)  # avoid division by zero
    df['rd'] = 100 - df['humidity']
    df['sp'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['wi'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    
    # Encode categorical variables if any
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = encoder.fit_transform(df[col])
    
    # For training data, ensure the target column is at the end
    if is_train and 'rainfall' in df.columns:
        cols = [col for col in df.columns if col != 'rainfall']
        df = df[cols + ['rainfall']]
    
    return df

# Apply the preprocessing function to both datasets
train = process_data(train, is_train=True)
test = process_data(test, is_train=False)


# SECTION 4: Feature Selection & Scaling

# Drop columns that are not helpful (only drop if they exist)
drop_cols = ['id', 'day', 'winddirection']
train.drop(columns=set(drop_cols).intersection(train.columns), errors='ignore', inplace=True)
test.drop(columns=set(drop_cols).intersection(test.columns), errors='ignore', inplace=True)

print("Remaining features in train:", train.columns.tolist())
print("Remaining features in test:", test.columns.tolist())

# Separate features (X) and target (y) in training data
X = train.drop(columns=['rainfall'])
y = train['rainfall']

# Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


# SECTION 5: Data Augmentation & Train/Validation Split

# Add small Gaussian noise to generate synthetic data
noise_factor = 0.01  # adjust noise level as needed
X_synthetic = X_scaled + noise_factor * np.random.normal(size=X_scaled.shape)
y_synthetic = y.copy()  # Labels remain the same

# Combine original and synthetic data (doubling the dataset)
X_augmented = np.vstack((X_scaled, X_synthetic))
y_augmented = np.hstack((y, y_synthetic))

# Split the augmented data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)


# SECTION 6: Model Building

# Get the number of input features
input_dim = X_train.shape[1]

# Define the model architecture
inputs = Input(shape=(input_dim,))

x = Dense(256)(inputs)
x = BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = Dropout(0.3)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = Dropout(0.3)(x)

x = Dense(64)(x)
x = BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = Dropout(0.3)(x)

x = Dense(32)(x)
x = BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = Dense(16)(x)
x = BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = Dense(8)(x)
x = BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

# Output layer with sigmoid activation (binary classification)
outputs = Dense(1, activation='sigmoid')(x)

# Build and compile the model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

# Display model summary
model.summary()


# SECTION 7: Model Training

# Early stopping callback to restore the best weights based on validation loss
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,       # You can adjust the number of epochs based on performance
    batch_size=32,
    callbacks=[early_stopping]
)


# SECTION 8: Model Evaluation & Visualization

plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot AUC
plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('AUC over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()


# SECTION 9: Predictions & Submission File

# Generate predictions (probabilities)
test_preds = model.predict(test_scaled)

# Create a submission DataFrame.
# Make sure 'test_ids' is available (stored before dropping the 'id' column)
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': test_preds.flatten()
})

# Save submission file
submission_file = "/kaggle/working/submission_dnn.csv"
submission.to_csv(submission_file, index=False)
print("Submission file saved as:", submission_file)
print(submission.head())

