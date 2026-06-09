%%time
# Cell 1: Import Libraries
import os  # For operating system functionalities
import pandas as pd  # For data manipulation and analysis with DataFrames
import numpy as np  # For numerical operations on arrays and matrices
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
from sklearn.experimental import enable_iterative_imputer  # Needed to use IterativeImputer
from sklearn.impute import IterativeImputer  # For iterative imputation of missing values
from sklearn.preprocessing import MinMaxScaler, RobustScaler  # For scaling feature values
from sklearn.decomposition import TruncatedSVD  # For dimensionality reduction
from sklearn.utils import resample  # For resampling methods
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # For data visualization
import tensorflow as tf  # For building and training the model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# Define constants for means
MEAN_TIME_SPENT_ALONE = 5
MEAN_SOCIAL_ATTENDANCE = 5
MEAN_GOING_OUTSIDE = 4 # Was 3 
MEAN_FRIENDS_CIRCLE_SIZE = 8 # Was 7
MEAN_POST_FREQUENCY = 5

def fill_nan(df):
    """
    Fills missing values in the DataFrame for specified columns.
    """
    df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(-1)
    df['Stage_fear'] = df['Stage_fear'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Social_event_attendance'] = df['Social_event_attendance'].fillna(-1)
    df['Going_outside'] = df['Going_outside'].fillna(-1)
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Friends_circle_size'] = df['Friends_circle_size'].fillna(-1)
    df['Post_frequency'] = df['Post_frequency'].fillna(-1)
    return df

def fill_nans_with_means(df):
    """
    Preprocess the DataFrame to fill missing values with predefined means.
    """
    df['Time_spent_Alone'] = df['Time_spent_Alone'].replace(-1, MEAN_TIME_SPENT_ALONE)
    df['Social_event_attendance'] = df['Social_event_attendance'].replace(-1, MEAN_SOCIAL_ATTENDANCE)
    df['Going_outside'] = df['Going_outside'].replace(-1, MEAN_GOING_OUTSIDE)
    df['Friends_circle_size'] = df['Friends_circle_size'].replace(-1, MEAN_FRIENDS_CIRCLE_SIZE)
    df['Post_frequency'] = df['Post_frequency'].replace(-1, MEAN_POST_FREQUENCY)
    return df

def party(df):
    """
    Preprocess the DataFrame to fill missing values for 'Stage_fear'.
    """
    df = fill_nans_with_means(df)
    mask = df['Stage_fear'] == -1

    df.loc[mask & (df['Social_event_attendance'] >= MEAN_SOCIAL_ATTENDANCE) & (df['Going_outside'] >= MEAN_GOING_OUTSIDE), 'Stage_fear'] = 0
    df.loc[mask & (df['Social_event_attendance'] < MEAN_SOCIAL_ATTENDANCE) & (df['Going_outside'] < MEAN_GOING_OUTSIDE), 'Stage_fear'] = 1
    return df

def the_hole(df):
    """
    Preprocess the DataFrame to fill missing values for 'Drained_after_socializing'.
    """
    mask = df['Drained_after_socializing'] == -1

    df.loc[mask & (df['Social_event_attendance'] >= MEAN_SOCIAL_ATTENDANCE) & (df['Friends_circle_size'] >= MEAN_FRIENDS_CIRCLE_SIZE), 'Drained_after_socializing'] = 0
    df.loc[mask & (df['Social_event_attendance'] < MEAN_SOCIAL_ATTENDANCE) & (df['Friends_circle_size'] < MEAN_FRIENDS_CIRCLE_SIZE), 'Drained_after_socializing'] = 1
    return df

def preprocess(df):
    """
    Main function to preprocess the DataFrame by filling missing values.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    df = fill_nan(df)
    df = party(df)
    df = the_hole(df)
    return df

def validate_no_nans(df):
    """
    Validate that there are no NaNs in the DataFrame.
    """
    assert df.isnull().sum().sum() == 0, "There are still NaNs in the DataFrame!"


# Cell 5: Load the Datasets

# Load the dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Cell 6: Preprocess Data

# Preprocess training data
# Store the target variable mapping 'Introvert' to 0 and 'Extrovert' to 1
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Target variable

# Drop 'id' and 'Personality' columns to focus on feature set
train_features = train.drop(['id', 'Personality'], axis=1)

# Preprocess the training features to handle missing values
X = preprocess(train_features)  

# Preprocess test data, dropping 'id' before processing
X_test = preprocess(test.drop('id', axis=1))  

# Check for NaN values in the training DataFrame
assert not X.isnull().values.any(), "NaN values found in the training DataFrame"

# Check for NaN values in the test DataFrame
assert not X_test.isnull().values.any(), "NaN values found in the test DataFrame"

# Display sample of processed features for better readability
from IPython.display import display

print("First 5 rows of the processed training features:")
display(X.head())

print("First 5 rows of the processed test features:")
display(X_test.head())


VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
# Scale the training data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Fit to training data (includes new features)
X_test_scaled = scaler.transform(X_test)  # Transform the test data with the same scaler

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y.values, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE )



# Adjustable parameters
BATCH_SIZE = 32  
EPOCHS = 10  
DROPOUT_RATE = 0.4  
HIDDEN_UNITS_1 = 64  
HIDDEN_UNITS_2 = 128  
HIDDEN_UNITS_3 = 256
LEARNING_RATE = 0.02

# Create TensorFlow Dataset for the training and validation sets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

# Model definition with Dropout for regularization
input_shape = X_scaled.shape[1]  # Now set to match the reduced dimensionality
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(HIDDEN_UNITS_1, activation='relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),  
    # tf.keras.layers.Dense(HIDDEN_UNITS_2, activation='relu'),
    # tf.keras.layers.Dropout(DROPOUT_RATE),
    # tf.keras.layers.Dense(HIDDEN_UNITS_3, activation='relu'),
    # tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define the learning rate reduction callback
lr_reduction = ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss
                                  factor=LEARNING_RATE,         # Reduce learning rate by a factor of 0.2
                                  patience=5,         # Wait 5 epochs before reducing the learning rate
                                  min_lr=0.001)       # Set minimum learning rate

# Train the model with validation and include the lr_reduction callback
history = model.fit(train_dataset, 
                    epochs=EPOCHS, 
                    validation_data=val_dataset, 
                    callbacks=[lr_reduction])

# Make predictions on the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(X_test_scaled).batch(BATCH_SIZE)  # No labels for test set
predictions = model.predict(test_dataset)
predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to class labels


# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='upper left')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Make predictions on the test dataset continues as is...


# Cell 9: Visualize Predictions
# Predict the probabilities for the test data
predictions = model.predict(test_dataset)  # This should give you the prediction probabilities
predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to binary class labels

# Prepare to visualize the first two features from the scaled test dataset
# Adding noise to the features to disperse the points
noise_scale = 0.1  # Adjust this for better spacing in the visualization

plt.figure(figsize=(10, 10))
plt.scatter(X_test_scaled[:, 0] + np.random.normal(0, noise_scale, size=len(X_test_scaled)), 
            X_test_scaled[:, 1] + np.random.normal(0, noise_scale, size=len(X_test_scaled)),
            c=predicted_classes.flatten(),  # Color based on predictions
            cmap='coolwarm', alpha=0.8, s=50, marker='o')

# Title and labels
plt.title("Personality Predictions: Introverts vs. Extroverts", fontsize=16)
plt.xlabel("Scaled Feature 1 + Noise", fontsize=14)  # Adjust the feature label as needed
plt.ylabel("Scaled Feature 2 + Noise", fontsize=14)  # Adjust the feature label as needed

# Custom legend for Introverts and Extroverts
introvert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Introvert',
                              markerfacecolor='blue', markersize=10)  
extrovert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Extrovert',
                              markerfacecolor='red', markersize=10)  

# Adding the custom legend
plt.legend(handles=[introvert_patch, extrovert_patch], title="Personality Classes")
plt.tight_layout()
plt.show()


# Convert probabilities to class labels
predicted_classes = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary class labels

# Display the first few predicted classes
print("Sample Predictions from Test Data (first 5):", predicted_classes[:5])

# Optional: Save Predictions to a CSV File
output_df = pd.DataFrame({'id': test['id'], 'Personality': predicted_classes})
output_df['Personality'] = output_df['Personality'].map({0: 'Introvert', 1: 'Extrovert'})  # Map back to original labels

# Save to CSV file
output_df.to_csv('submission.csv', index=False)  # Save the predictions for submission or review

