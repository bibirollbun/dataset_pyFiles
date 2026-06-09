!pip install tensorflow


import numpy as np
import pandas as pd
import os
import pydicom
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


train  = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')
label = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')
train_desc = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')
test_desc = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_series_descriptions.csv')


# Preview the data
train.head()


train_desc.head()


def reshape_dataframe(df):
    # Create a list of columns to exclude
    exclude_columns = ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']
    
    # Filter the columns to process
    columns_to_process = [col for col in df.columns if col not in exclude_columns]
    
    # Split the columns into condition and level, extract severity, and concatenate to form the new DataFrame
    reshaped_df = pd.DataFrame([
        {
            'study_id': row['study_id'],
            'condition': ' '.join([word.capitalize() for word in col.split('_')[:-2]]),
            'level': col.split('_')[-2].capitalize() + '/' + col.split('_')[-1].capitalize(),
            'severity': row[col]
        }
        for _, row in df.iterrows()
        for col in columns_to_process
    ])
    
    return reshaped_df

# Reshape the DataFrame
new_train_df = reshape_dataframe(train)

# Display the first few rows of the reshaped DataFrame
new_train_df.head()


# Print columns in a neat way
print("\nColumns in new_train_df:")
print(",".join(new_train_df.columns))

print("\nColumns in label:")
print(",".join(label.columns))

print("\nColumns in test_desc:")
print(",".join(test_desc.columns))


# Merge reshaped labels with coordinate labels
merged_df = pd.merge(
    new_train_df,
    label,
    on=['study_id', 'condition', 'level'],
    how='inner'
)

# Quick sanity check
print("Merged rows:", len(merged_df))
print(merged_df.head())


# Merge the dataframes on the common column 'series_id'
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id','study_id'], how='inner')
# Display the first few rows of the final merged dataframe
final_merged_df.head()


# Create the row_id column
final_merged_df['row_id'] = (
    final_merged_df['study_id'].astype(str) + '_' +
    final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
    final_merged_df['level'].str.lower().str.replace('/', '_')
)

# Create the image_path column
final_merged_df['image_path'] = (
    '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/' + 
    final_merged_df['study_id'].astype(str) + '/' +
    final_merged_df['series_id'].astype(str) + '/' +
    final_merged_df['instance_number'].astype(str) + '.dcm'
)

# Note: Check image path, since there's 1 instance id, for 1 image, but there's many more images other than the ones labelled in the instance ID. 

# Display the updated dataframe
final_merged_df.head()


# Define the base path for test images
base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images'

# Function to get image paths for a series
def get_image_paths(row):
    series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
    if os.path.exists(series_path):
        return [os.path.join(series_path, f) for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))]
    return []

# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# Create a list to store the expanded rows
expanded_rows = []

# Expand the dataframe by adding new rows for each file path
for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row)
    conditions = condition_mapping.get(row['series_description'], {})
    if isinstance(conditions, str):  # Single condition
        conditions = {'left': conditions, 'right': conditions}
    for side, condition in conditions.items():
        for image_path in image_paths:
            expanded_rows.append({
                'study_id': row['study_id'],
                'series_id': row['series_id'],
                'series_description': row['series_description'],
                'image_path': image_path,
                'condition': condition,
                'row_id': f"{row['study_id']}_{condition}"
            })

# Create a new dataframe from the expanded rows
expanded_test_desc = pd.DataFrame(expanded_rows)

# Display the resulting dataframe
expanded_test_desc.head(5)


final_merged_df['severity'] = final_merged_df['severity'].fillna('Normal/Mild')


test_data = expanded_test_desc
train_data = final_merged_df


train_data.isnull().sum()


# Display basic statistics for 'x' and 'y' columns
x_stats = train_data['x'].describe()
y_stats = train_data['y'].describe()

print("X Coordinate Statistics:")
print(x_stats)

print("\nY Coordinate Statistics:")
print(y_stats)


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a histogram for 'x' values
x_hist = go.Histogram(
    x=train_data['x'],
    nbinsx=30,
    name='X Coordinates',
    marker_color='blue',
    opacity=0.7
)

# Create a histogram for 'y' values
y_hist = go.Histogram(
    x=train_data['y'],
    nbinsx=30,
    name='Y Coordinates',
    marker_color='green',
    opacity=0.7
)

# Create a figure with subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Distribution of X Coordinates', 'Distribution of Y Coordinates'))

# Add the histograms to the figure
fig.add_trace(x_hist, row=1, col=1)
fig.add_trace(y_hist, row=1, col=2)

# Update layout for a cleaner look
fig.update_layout(
    title_text="Distribution of X and Y Coordinates",
    showlegend=False,
    xaxis_title="X Values",
    yaxis_title="Frequency",
    xaxis2_title="Y Values",
    yaxis2_title="Frequency",
    bargap=0.2,  # Gap between bars
)

# Show the plot
fig.show()


import plotly.express as px

# Count the occurrences of each severity within each condition
severity_condition_counts = train_data.groupby(['condition', 'severity']).size().reset_index(name='count')

# Create a grouped bar chart
fig = px.bar(
    severity_condition_counts,
    x='condition',
    y='count',
    color='severity',
    barmode='group',
    title='Distribution of Severities for Each Condition',
    labels={'condition': 'Condition', 'count': 'Number of Cases', 'severity': 'Severity'},
    color_discrete_sequence=px.colors.qualitative.Set1  # Custom color sequence
)

# Update the layout for better presentation
fig.update_layout(
    xaxis_title='Condition',
    yaxis_title='Number of Cases',
    legend_title='Severity',
    bargap=0.15,
    bargroupgap=0.1
)

fig.show()


# Count the occurrences of each severity within each condition
severity_condition_counts = train_data.groupby(['condition', 'series_description']).size().reset_index(name='count')

# Create a grouped bar chart
fig = px.bar(
    severity_condition_counts,
    x='condition',
    y='count',
    color='series_description',
    barmode='group',
    title='Distribution of Condition for Respective Angle',
    labels={'condition': 'Condition', 'count': 'Number of Cases', 'series_description': 'Angle of MR Image'},
    color_discrete_sequence=px.colors.qualitative.Set1  # Custom color sequence
)

# Update the layout for better presentation
fig.update_layout(
    xaxis_title='Condition',
    yaxis_title='Number of Cases',
    legend_title='Angle',
    bargap=0.15,
    bargroupgap=0.1
)

fig.show()


# Group by 'level' and 'condition' and count the occurrences
level_condition_counts = train_data.groupby(['condition', 'level']).size().reset_index(name='count')

# Create a grouped bar chart
fig = px.bar(
    level_condition_counts,
    x='condition',
    y='count',
    color='level',
    barmode='group',
    title='Distribution of Levels for Each Condition',
    labels={'condition': 'Condition', 'count': 'Number of Cases', 'level': 'Level'},
    color_discrete_sequence=px.colors.qualitative.Set1  # Custom color sequence
)

# Update the layout for better presentation
fig.update_layout(
    xaxis_title='Condition',
    yaxis_title='Number of Cases',
    legend_title='Level',
    bargap=0.15,
    bargroupgap=0.1
)

fig.show()


# Count the occurrences of each level within each condition
level_condition_counts = train_data.groupby(['level', 'condition']).size().reset_index(name='count')

# Create a pivot table to structure the data for the heatmap
heatmap_data = level_condition_counts.pivot(index='level', columns='condition', values='count')

# Create the heatmap
fig = px.imshow(
    heatmap_data,
    labels={'x': 'Condition', 'y': 'Level', 'color': 'Count'},
    title='Heatmap of Levels by Condition',
    color_continuous_scale='Viridis'
)

fig.show()


# Export the DataFrame to a CSV file
final_merged_df.to_csv('train_processed.csv', index=False)
test_data.to_csv('test_processed.csv', index=False)


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Mapping for the 'level' column
level_mapping = {'L1/L2': 0, 'L2/L3': 1, 'L3/L4': 2, 'L4/L5': 3, 'L5/S1': 4}
final_merged_df['level_encoded'] = final_merged_df['level'].map(level_mapping)

# Mapping for the 'series_description' column
series_description_mapping = {'Sagittal T1': 0, 'Sagittal T2/STIR': 1, 'Axial T2': 2}
final_merged_df['series_description_encoded'] = final_merged_df['series_description'].map(series_description_mapping)

# Convert categorical features to numerical using LabelEncoder for 'condition' and 'severity'
le_condition = LabelEncoder()
le_severity = LabelEncoder()

final_merged_df['condition_encoded'] = le_condition.fit_transform(final_merged_df['condition'])
final_merged_df['severity_encoded'] = le_severity.fit_transform(final_merged_df['severity'])

# Concatenate condition and severity labels
final_merged_df['combined_label'] = final_merged_df['condition_encoded'].astype(str) + "_" + final_merged_df['severity_encoded'].astype(str)

# Remove non-numeric columns and use only relevant features
X = final_merged_df.drop(['condition', 'severity', 'row_id', 'image_path', 'level', 'series_description', 'condition_encoded', 'severity_encoded', 'combined_label'], axis=1)
X['level'] = final_merged_df['level_encoded']  # Add the newly encoded 'level' column
X['series_description'] = final_merged_df['series_description_encoded']  # Add the newly encoded 'series_description' column

# SMOTE: Apply SMOTE to the concatenated labels (condition + severity)
y_combined = final_merged_df['combined_label']

smote = SMOTE(random_state=42)
X_smote, y_combined_smote = smote.fit_resample(X, y_combined)

# Split back the combined label into separate condition and severity labels
y_condition_smote = y_combined_smote.str.split("_", expand=True)[0].astype(int)  # Condition part
y_severity_smote = y_combined_smote.str.split("_", expand=True)[1].astype(int)  # Severity part

# One-hot encode the target labels for multi-class classification
onehot = OneHotEncoder()

y_condition_smote_oh = onehot.fit_transform(y_condition_smote.values.reshape(-1, 1)).toarray()
y_severity_smote_oh = onehot.fit_transform(y_severity_smote.values.reshape(-1, 1)).toarray()

# Print the shapes to verify correctness
print(f"Shape of X_smote: {X_smote.shape}")
print(f"Shape of y_condition_smote_oh: {y_condition_smote_oh.shape}")
print(f"Shape of y_severity_smote_oh: {y_severity_smote_oh.shape}")


import pandas as pd

# Step 1: Retain 'study_id', 'series_id', 'instance_number', 'x', 'y', and 'level' from the original dataset
X_combined = final_merged_df[['study_id', 'series_id', 'instance_number', 'x', 'y', 'level']].copy()

# Mapping for the 'level' column as specified
level_mapping = {'L1/L2': 1, 'L2/L3': 2, 'L3/L4': 3, 'L4/L5': 4, 'L5/S1': 5}
X_combined['level'] = X_combined['level'].map(level_mapping)

# Step 2: Initialize y_condition_smote_df and apply condition mapping as specified
condition_mapping = {
    'Spinal Canal Stenosis': 1,
    'Left Neural Foraminal Narrowing': 2,
    'Right Neural Foraminal Narrowing': 3,
    'Left Subarticular Stenosis': 4,
    'Right Subarticular Stenosis': 5
}

# Assuming y_condition_smote_oh is available from SMOTE
y_condition_smote_df = pd.DataFrame(y_condition_smote_oh, columns=[f"condition_{i}" for i in range(y_condition_smote_oh.shape[1])])

# Apply the condition mapping
y_condition_smote_df = y_condition_smote_df.replace(condition_mapping)

# Step 3: Initialize y_severity_smote_df from SMOTE results
# Assuming y_severity_smote_oh is available from SMOTE
y_severity_smote_df = pd.DataFrame(y_severity_smote_oh, columns=[f"severity_{i}" for i in range(y_severity_smote_oh.shape[1])])

# Step 4: Convert the integer-related columns in X_combined to actual integers
X_combined['study_id'] = X_combined['study_id'].astype(int)
X_combined['series_id'] = X_combined['series_id'].astype(int)
X_combined['instance_number'] = X_combined['instance_number'].astype(int)
X_combined['level'] = X_combined['level'].astype(int)

# Keep 'x' and 'y' as float
X_combined['x'] = X_combined['x'].astype(float)
X_combined['y'] = X_combined['y'].astype(float)

# Step 5: Initialize X_smote_df from SMOTE result
X_smote_df = pd.DataFrame(X_smote, columns=[f"feature_{i}" for i in range(X_smote.shape[1])])

# Step 6: Drop columns 'feature_0' to 'feature_8'
X_smote_df.drop(columns=[f"feature_{i}" for i in range(9)], inplace=True)

# Step 7: Combine SMOTE features with 'study_id', 'series_id', 'instance_number', 'x', 'y', and 'level'
X_smote_combined_df = pd.concat([X_combined, X_smote_df], axis=1)

# Step 8: Drop rows where all values are NaN/null
X_smote_combined_df.dropna(how='all', inplace=True)

# Step 9: Prevent scientific notation for integer columns and save as CSV
pd.set_option('display.float_format', '{:.0f}'.format)  # Disable scientific notation for integers



# Check the number of samples
print(f"Filtered X_smote_combined_df size: {len(X_smote_combined_df)}")  # Should be 48,692

# Ensure the labels are of the same size as X_smote_combined_df by filtering the corresponding labels
# Make sure this aligns with how rows were dropped earlier in X_smote_combined_df
y_condition_smote_df_filtered = y_condition_smote_df.iloc[:len(X_smote_combined_df)]
y_severity_smote_df_filtered = y_severity_smote_df.iloc[:len(X_smote_combined_df)]

# Now check the sizes
print(f"y_condition_smote_df size after filtering: {len(y_condition_smote_df_filtered)}")  # Should match 48,692
print(f"y_severity_smote_df size after filtering: {len(y_severity_smote_df_filtered)}")    # Should match 48,692


# Save the combined DataFrame and labels to CSV files
X_smote_combined_df.to_csv('/kaggle/working/X_smote_combined.csv', index=False)
y_condition_smote_df.to_csv('/kaggle/working/y_condition_smote_oh.csv', index=False)
y_severity_smote_df.to_csv('/kaggle/working/y_severity_smote_oh.csv', index=False)

# Confirm the export
print("/kaggle/working/X_smote_combined.csv, /kaggle/working/y_condition_smote_oh.csv, /kaggle/working/y_severity_smote_oh.csv exported successfully")


import os
import pydicom
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

# Data generator class
class DataGenerator(Sequence):
    def __init__(self, df, image_folder, y_condition, y_severity, batch_size=32, img_size=(224, 224)):
        self.df = df
        self.image_folder = image_folder
        self.y_condition = y_condition
        self.y_severity = y_severity
        self.batch_size = batch_size
        self.img_size = img_size
        self.indices = np.arange(len(df))
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get the batch data
        batch_df = self.df.iloc[batch_indices]
        batch_y_condition = self.y_condition[batch_indices]
        batch_y_severity = self.y_severity[batch_indices]
        
        # Load the images for the batch
        images = []
        for _, row in batch_df.iterrows():
            study_id = int(row['study_id'])  # Ensure integer format
            series_id = int(row['series_id'])  # Ensure integer format
            instance_number = int(row['instance_number'])  # Ensure integer format
            
            # Construct image path using integer values (convert to string)
            img_path = os.path.join(self.image_folder, str(study_id), str(series_id), f"{instance_number}.dcm")
            
            # Load and preprocess the DICOM image
            img = self.load_dicom_image(img_path)
            images.append(img)
        
        return np.array(images), {'condition_output': batch_y_condition, 'severity_output': batch_y_severity}
    
    def load_dicom_image(self, image_path):
        try:
            dicom = pydicom.dcmread(image_path)
            if hasattr(dicom, 'pixel_array'):
                img = dicom.pixel_array
                img = cv2.resize(img, self.img_size)  # Resize to 224x224
                img = np.stack((img,) * 3, axis=-1)  # Convert to RGB (3 channels)
                img = img / 255.0  # Normalize
                return img
            else:
                print(f"Warning: No pixel data in DICOM file: {image_path}")
                return np.zeros((*self.img_size, 3))  # Return a blank image if no pixel data
        except Exception as e:
            print(f"Error reading DICOM file: {image_path}. Error: {e}")
            return np.zeros((*self.img_size, 3))  # Return a blank image if any error occurs


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

# Function to build the EfficientNetB0 model
def build_efficientnet_model(num_classes_condition, num_classes_severity, input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Add global pooling and dense layers for both condition and severity
    x = Flatten()(base_model.output)
    
    # Output for condition classification
    condition_output = Dense(num_classes_condition, activation='softmax', name='condition_output')(x)
    
    # Output for severity classification
    severity_output = Dense(num_classes_severity, activation='softmax', name='severity_output')(x)
    
    # Define the model
    model = Model(inputs, outputs=[condition_output, severity_output])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss={'condition_output': 'categorical_crossentropy', 'severity_output': 'categorical_crossentropy'},
                  metrics={'condition_output': 'accuracy', 'severity_output': 'accuracy'})
    
    return model





from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, BatchNormalization, LSTM, Bidirectional, Dropout, TimeDistributed
from tensorflow.keras.models import Model
def build_rnn_model(num_classes_condition, num_classes_severity, input_shape=(224, 224, 3), timesteps=5):
    """
    IMPROVED RNN/LSTM Model
    - Deeper CNN (3 blocks)
    - Global Average Pooling (drastically reduces parameters/memory)
    - Bidirectional LSTM (better context)
    """
    inputs = Input(shape=(timesteps, *input_shape))
    
    # --- 1. Feature Extraction (Improved) ---
    # Block 1
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    # Block 2
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    # Block 3 (New separate block for deeper features)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    # Global Pooling instead of Flatten (Crucial Modification)
    # Reduces feature map to vector without massive Dense layers
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    
    # --- 2. Sequence Analysis ---
    # Bidirectional LSTM to capture patterns in both directions (up/down spine)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3))(x)
    
    # --- 3. Classification Heads ---
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    condition_output = Dense(num_classes_condition, activation='softmax', name='condition_output')(x)
    severity_output = Dense(num_classes_severity, activation='softmax', name='severity_output')(x)
    
    model = Model(inputs=inputs, outputs=[condition_output, severity_output])
    
    model.compile(
        optimizer='adam',
        loss={'condition_output': 'categorical_crossentropy', 'severity_output': 'categorical_crossentropy'},
        metrics={'condition_output': 'accuracy', 'severity_output': 'accuracy'}
    )
    
    return model
print("Improved build_rnn_model defined successfully")


# ALTERNATIVE: MEMORY-EFFICIENT GRU model (Use this if LSTM still causes OOM)
from tensorflow.keras.layers import GRU, BatchNormalization

def build_gru_model_lightweight(num_classes_condition, num_classes_severity, input_shape=(224, 224, 3), timesteps=3):
    """
    MINIMAL GRU-based model - ULTRA LIGHTWEIGHT for severe GPU memory constraints
    Uses single Conv2D + GRU instead of TimeDistributed
    """
    inputs = Input(shape=(timesteps, *input_shape))
    
    # Single lightweight Conv layer
    x = TimeDistributed(Conv2D(8, (3, 3), activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    
    # Single GRU layer (minimal units)
    x = GRU(16, return_sequences=False, dropout=0.2)(x)
    
    # Minimal Dense layer
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layers
    condition_output = Dense(num_classes_condition, activation='softmax', name='condition_output')(x)
    severity_output = Dense(num_classes_severity, activation='softmax', name='severity_output')(x)
    
    model = Model(inputs=inputs, outputs=[condition_output, severity_output])
    
    model.compile(
        optimizer='adam',
        loss={'condition_output': 'categorical_crossentropy', 'severity_output': 'categorical_crossentropy'},
        metrics={'condition_output': 'accuracy', 'severity_output': 'accuracy'}
    )
    
    return model

print("MEMORY-EFFICIENT GRU Model available as backup")
print("Use this if LSTM still causes ResourceExhaustedError:")
print("  rnn_model = build_gru_model_lightweight(5, 3, timesteps=3)")



class SequentialDataGenerator(Sequence):
    """
    Data generator for loading sequential DICOM images for RNN/LSTM models
    Loads multiple consecutive slices from the same series
    """
    def __init__(self, df, image_folder, y_condition, y_severity, batch_size=16, img_size=(224, 224), sequence_length=10):
        # RESET INDEX to ensure it aligns with the 0-based numpy arrays (y_condition, y_severity)
        self.df = df.reset_index(drop=True)
        
        self.image_folder = image_folder
        self.y_condition = y_condition
        self.y_severity = y_severity
        self.batch_size = batch_size
        self.img_size = img_size
        self.sequence_length = sequence_length
        
        # Group by study_id and series_id to get sequences
        self.grouped = self.df.groupby(['study_id', 'series_id'])
        self.group_keys = list(self.grouped.groups.keys())
        self.indices = np.arange(len(self.group_keys))
    
    def __len__(self):
        return int(np.ceil(len(self.group_keys) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_keys = [self.group_keys[i] for i in batch_indices]
        
        sequences = []
        batch_y_condition = []
        batch_y_severity = []
        
        for study_id, series_id in batch_keys:
            group_df = self.grouped.get_group((study_id, series_id))
            
            # Sort by instance number to maintain sequence
            group_df = group_df.sort_values('instance_number')
            
            # Load sequence of images
            sequence = []
            for _, row in group_df.head(self.sequence_length).iterrows():
                img_path = os.path.join(
                    self.image_folder, 
                    str(int(row['study_id'])), 
                    str(int(row['series_id'])), 
                    f"{int(row['instance_number'])}.dcm"
                )
                img = self.load_dicom_image(img_path)
                sequence.append(img)
            
            # Pad sequence if shorter than sequence_length
            while len(sequence) < self.sequence_length:
                sequence.append(np.zeros((*self.img_size, 3)))
            
            sequences.append(sequence)
            
            # Get labels (use first row of the group)
            # Since we reset index, first_idx will now be a valid index for y_condition/y_severity
            first_idx = group_df.index[0]
            batch_y_condition.append(self.y_condition[first_idx])
            batch_y_severity.append(self.y_severity[first_idx])
        
        return (
            np.array(sequences), 
            {
                'condition_output': np.array(batch_y_condition), 
                'severity_output': np.array(batch_y_severity)
            }
        )
    
    def load_dicom_image(self, image_path):
        try:
            dicom = pydicom.dcmread(image_path)
            if hasattr(dicom, 'pixel_array'):
                img = dicom.pixel_array
                img = cv2.resize(img, self.img_size)
                img = np.stack((img,) * 3, axis=-1)
                img = img / 255.0
                return img
            else:
                return np.zeros((*self.img_size, 3))
        except Exception as e:
            # print(f"Error reading DICOM file: {image_path}. Error: {e}")
            return np.zeros((*self.img_size, 3))

print("Sequential Data Generator updated with index fix.")


# Define image folder for training/validation generators
# Update this path if your data lives elsewhere
image_folder = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images'
print('Using image folder:', image_folder)


# Prepare data for RNN training
# Split data maintaining sequential structure
X_train_rnn, X_val_rnn, y_train_cond_rnn, y_val_cond_rnn, y_train_sev_rnn, y_val_sev_rnn = train_test_split(
    X_smote_combined_df, 
    y_condition_smote_df_filtered.values, 
    y_severity_smote_df_filtered.values, 
    test_size=0.2, 
    random_state=42
)

print(f"RNN Training set: {len(X_train_rnn)} samples")
print(f"RNN Validation set: {len(X_val_rnn)} samples")

# Create sequential data generators with ULTRA-REDUCED batch size and sequence length
train_seq_generator = SequentialDataGenerator(
    X_train_rnn, 
    image_folder, 
    y_train_cond_rnn, 
    y_train_sev_rnn, 
    batch_size=4,  # ULTRA-REDUCED from 8 to 4
    sequence_length=3  # REDUCED from 5 to 3
)

val_seq_generator = SequentialDataGenerator(
    X_val_rnn, 
    image_folder, 
    y_val_cond_rnn, 
    y_val_sev_rnn, 
    batch_size=4,  # ULTRA-REDUCED from 8 to 4
    sequence_length=3  # REDUCED from 5 to 3
)

# Build ULTRA-LIGHTWEIGHT RNN model with reduced complexity
rnn_model = build_rnn_model(num_classes_condition=5, num_classes_severity=3, timesteps=3)

# Display model summary
rnn_model.summary()



import tensorflow as tf
import gc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. Clear session
tf.keras.backend.clear_session()
gc.collect()

# 2. INCREASE BATCH SIZE for multi-GPU efficiency
GLOBAL_BATCH_SIZE = 16 

print(f"Re-creating generators with optimized batch size: {GLOBAL_BATCH_SIZE}")

# Re-initialize generators with larger batch size
train_seq_generator = SequentialDataGenerator(
    X_train_rnn, 
    image_folder, 
    y_train_cond_rnn, 
    y_train_sev_rnn, 
    batch_size=GLOBAL_BATCH_SIZE,
    sequence_length=3
)

val_seq_generator = SequentialDataGenerator(
    X_val_rnn, 
    image_folder, 
    y_val_cond_rnn, 
    y_val_sev_rnn, 
    batch_size=GLOBAL_BATCH_SIZE,
    sequence_length=3
)

# 3. Define Strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# 4. Build & Compile INSIDE scope
with strategy.scope():
    rnn_model = build_rnn_model(num_classes_condition=5, num_classes_severity=3, timesteps=3)
    
    # Optional: Slightly higher learning rate for larger batches
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
    
    rnn_model.compile(
        optimizer=opt,
        loss={'condition_output': 'categorical_crossentropy', 'severity_output': 'categorical_crossentropy'},
        metrics={'condition_output': 'accuracy', 'severity_output': 'accuracy'}
    )

# 5. Define Callbacks
rnn_earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
rnn_checkpoint = ModelCheckpoint('rnn_lstm_model.keras', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)

# 6. Train with workers REMOVED to fix TypeError
print("Starting Optimized Distributed RNN Training...")
rnn_history = rnn_model.fit(
    train_seq_generator,
    validation_data=val_seq_generator,
    epochs=10,
    callbacks=[rnn_earlystop, rnn_checkpoint, reduce_lr],
    verbose=1
)


# Plot RNN training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Condition accuracy
axes[0, 0].plot(rnn_history.history['condition_output_accuracy'], label='Train')
axes[0, 0].plot(rnn_history.history['val_condition_output_accuracy'], label='Validation')
axes[0, 0].set_title('RNN Condition Classification Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Condition loss
axes[0, 1].plot(rnn_history.history['condition_output_loss'], label='Train')
axes[0, 1].plot(rnn_history.history['val_condition_output_loss'], label='Validation')
axes[0, 1].set_title('RNN Condition Classification Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Severity accuracy
axes[1, 0].plot(rnn_history.history['severity_output_accuracy'], label='Train')
axes[1, 0].plot(rnn_history.history['val_severity_output_accuracy'], label='Validation')
axes[1, 0].set_title('RNN Severity Classification Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Severity loss
axes[1, 1].plot(rnn_history.history['severity_output_loss'], label='Train')
axes[1, 1].plot(rnn_history.history['val_severity_output_loss'], label='Validation')
axes[1, 1].set_title('RNN Severity Classification Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

print("RNN training visualization complete")


import numpy as np
from sklearn.metrics import classification_report

# 1. EVALUATE & PREDICT (Crucial Step: Generate predictions first)
print("Evaluating RNN model on validation set...")
rnn_results = rnn_model.evaluate(val_seq_generator, verbose=1)

print("\nGenerating predictions...")
rnn_pred_condition, rnn_pred_severity = rnn_model.predict(val_seq_generator)

# Convert predictions to class labels
rnn_pred_condition_classes = rnn_pred_condition.argmax(axis=1)
rnn_pred_severity_classes = rnn_pred_severity.argmax(axis=1)

# 2. Extract the TRUE labels from the generator (in the correct order)
print("Extracting true labels from generator...")
y_val_cond_true_seq = []
y_val_sev_true_seq = []

# Iterate through the generator to get the batch labels
for i in range(len(val_seq_generator)):
    # Generator returns (inputs, targets)
    _, y_batch = val_seq_generator[i]
    
    # targets is a dict: {'condition_output': ..., 'severity_output': ...}
    y_val_cond_true_seq.extend(y_batch['condition_output'])
    y_val_sev_true_seq.extend(y_batch['severity_output'])

# Convert to numpy arrays
y_val_cond_true_seq = np.array(y_val_cond_true_seq)
y_val_sev_true_seq = np.array(y_val_sev_true_seq)

# 3. Convert One-Hot Encoded labels to Class Integers (0, 1, 2...)
y_val_cond_classes_seq = y_val_cond_true_seq.argmax(axis=1)
y_val_sev_classes_seq = y_val_sev_true_seq.argmax(axis=1)

# 4. Generate the report using the SEQUENCE-aligned labels
print("\n=== RNN Classification Reports (Corrected) ===")

print("\n--- Condition Classification ---")
print(classification_report(y_val_cond_classes_seq, rnn_pred_condition_classes))

print("\n--- Severity Classification ---")
print(classification_report(y_val_sev_classes_seq, rnn_pred_severity_classes))


# Plot confusion matrices for RNN model
from sklearn.metrics import confusion_matrix
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Condition confusion matrix
# Use y_val_cond_classes_seq (4979 samples) instead of y_val_cond_classes (9739 samples)
condition_cm_rnn = confusion_matrix(y_val_cond_classes_seq, rnn_pred_condition_classes)
sns.heatmap(condition_cm_rnn, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('RNN Condition Classification Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['SCS', 'LNFN', 'RNFN', 'LSS', 'RSS'], rotation=45)
axes[0].set_yticklabels(['SCS', 'LNFN', 'RNFN', 'LSS', 'RSS'], rotation=0)

# Severity confusion matrix
# Use y_val_sev_classes_seq instead of y_val_sev_classes
severity_cm_rnn = confusion_matrix(y_val_sev_classes_seq, rnn_pred_severity_classes)
sns.heatmap(severity_cm_rnn, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('RNN Severity Classification Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['Normal/Mild', 'Moderate', 'Severe'], rotation=45)
axes[1].set_yticklabels(['Normal/Mild', 'Moderate', 'Severe'], rotation=0)

plt.tight_layout()
plt.show()

print("\nConfusion matrices generated for RNN model")


# Save the RNN model
rnn_model_save_path = os.path.join('/kaggle/working/', 'rnn_lstm_model.h5')
rnn_model.save(rnn_model_save_path)
print(f"RNN/LSTM model saved at {rnn_model_save_path}")

# Model comparison summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print("\n1. EfficientNetB0 Model:")
print("   - Architecture: Pre-trained CNN with transfer learning")
print("   - Input: Single image analysis")
print("   - Strengths: Strong feature extraction, good baseline performance")
print("\n2. RNN/LSTM Model:")
print("   - Architecture: Sequential model with Bidirectional LSTM")
print("   - Input: Sequence of images (temporal analysis)")
print("   - Strengths: Captures patterns across multiple slices")
print("   - Use case: Better for analyzing complete MRI series")
print("\n3. Recommendation:")
print("   - Use EfficientNetB0 for: Single slice classification, faster inference")
print("   - Use RNN/LSTM for: Complete series analysis, temporal patterns")
print("   - Consider ensemble: Combine both models for best results")
print("="*60)


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Prepare Data Split (Restored from previous step)
X_train_df, X_val_df, y_train_condition, y_val_condition, y_train_severity, y_val_severity = train_test_split(
    X_smote_combined_df, 
    y_condition_smote_df_filtered.values, 
    y_severity_smote_df_filtered.values, 
    test_size=0.2, 
    random_state=42
)

# Confirm sizes
print(f"X_train_df size: {len(X_train_df)}, X_val_df size: {len(X_val_df)}")

# 2. Define Image Folder and Generators (Restored)
image_folder = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'

train_generator = DataGenerator(X_train_df, image_folder, y_train_condition, y_train_severity, batch_size=32)
val_generator = DataGenerator(X_val_df, image_folder, y_val_condition, y_val_severity, batch_size=32)

# 3. Setup Callbacks (Restored)
earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('efficientnetb0_model.keras', save_best_only=True)

# 4. Distributed Training Setup (The new optimization)
tf.keras.backend.clear_session()
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

with strategy.scope():
    # Build model inside scope
    model = build_efficientnet_model(num_classes_condition=5, num_classes_severity=3)

print("Starting Distributed Training for EfficientNet...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[earlystop, checkpoint],
    verbose=1
)


# Save the trained model to the output directory
output_directory = '/kaggle/working/'  # Output folder in Kaggle
model_save_path = os.path.join(output_directory, 'efficientnetb0_model.h5')

# Save the model
model.save(model_save_path)

print(f"Model saved at {model_save_path}")


# Define batch size
batch_size = 32  # Set your desired batch size here

# Evaluate the model on validation data
val_generator = DataGenerator(X_val_df, image_folder, y_val_condition, y_val_severity, batch_size=batch_size)

# Perform evaluation, expecting 3 values: overall loss, condition accuracy, severity accuracy
results = model.evaluate(val_generator)
print(results)



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred, n_classes, title):
    # Binarize the true labels for ROC
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# Get the model predictions using the validation data generator
y_pred_condition, y_pred_severity = model.predict(val_generator)

# Plot ROC curves for condition classification
plot_roc_curve(y_val_condition.argmax(axis=1), y_pred_condition, 5, "ROC Curve for Condition Classification")

# Plot ROC curves for severity classification
plot_roc_curve(y_val_severity.argmax(axis=1), y_pred_severity, 3, "ROC Curve for Severity Classification")


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Convert predictions to class labels
y_pred_condition_classes = y_pred_condition.argmax(axis=1)
y_pred_severity_classes = y_pred_severity.argmax(axis=1)

y_val_condition_classes = y_val_condition.argmax(axis=1)
y_val_severity_classes = y_val_severity.argmax(axis=1)

# Confusion Matrix for Condition Classification
condition_cm = confusion_matrix(y_val_condition_classes, y_pred_condition_classes)
print("Confusion Matrix for Condition Classification:")
print(condition_cm)

# Confusion Matrix for Severity Classification
severity_cm = confusion_matrix(y_val_severity_classes, y_pred_severity_classes)
print("Confusion Matrix for Severity Classification:")
print(severity_cm)

# Classification Report for Condition Classification
condition_report = classification_report(y_val_condition_classes, y_pred_condition_classes)
print("Classification Report for Condition Classification:")
print(condition_report)

# Classification Report for Severity Classification
severity_report = classification_report(y_val_severity_classes, y_pred_severity_classes)
print("Classification Report for Severity Classification:")
print(severity_report)

# Overall Accuracy for both tasks
condition_accuracy = accuracy_score(y_val_condition_classes, y_pred_condition_classes)
severity_accuracy = accuracy_score(y_val_severity_classes, y_pred_severity_classes)

print(f"Condition Classification Accuracy: {condition_accuracy:.2f}")
print(f"Severity Classification Accuracy: {severity_accuracy:.2f}")


# Modified Data Generator class for test data using study_id and series_id from test_desc
class TestDataGenerator(Sequence):
    def __init__(self, df, image_folder, batch_size=32, img_size=(224, 224)):
        self.df = df  # Use the test_desc DataFrame
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.indices = np.arange(len(df))
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get the batch data
        batch_df = self.df.iloc[batch_indices]
        
        # Load the images for the batch
        images = []
        for _, row in batch_df.iterrows():
            study_id = int(row['study_id'])  # Ensure integer format
            series_id = int(row['series_id'])  # Ensure integer format
            
            # Construct image path using integer values
            series_path = os.path.join(self.image_folder, str(study_id), str(series_id))
            dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            
            # Load the first image in the series (adjust if needed)
            if dicom_files:
                img_path = os.path.join(series_path, dicom_files[0])
                img = self.load_dicom_image(img_path)
                images.append(img)
            else:
                print(f"Warning: No DICOM files found for series {series_id} in study {study_id}")
                images.append(np.zeros((*self.img_size, 3)))  # Blank image if no DICOM
            
        return np.array(images)
    
    def load_dicom_image(self, image_path):
        try:
            dicom = pydicom.dcmread(image_path)
            if hasattr(dicom, 'pixel_array'):
                img = dicom.pixel_array
                img = cv2.resize(img, self.img_size)  # Resize to 224x224
                img = np.stack((img,) * 3, axis=-1)  # Convert to RGB (3 channels)
                img = img / 255.0  # Normalize
                return img
            else:
                print(f"Warning: No pixel data in DICOM file: {image_path}")
                return np.zeros((*self.img_size, 3))  # Return a blank image if no pixel data
        except Exception as e:
            print(f"Error reading DICOM file: {image_path}. Error: {e}")
            return np.zeros((*self.img_size, 3))  # Return a blank image if any error occurs


# Define the test image folder
test_image_folder = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

# Create the test data generator using the test_desc DataFrame
test_generator = TestDataGenerator(test_desc, test_image_folder, batch_size=32)

# Get model predictions for the test set
y_pred_condition, y_pred_severity = model.predict(test_generator)

# Since there are no true labels, just output predictions
print(f"Predicted Conditions: {y_pred_condition}")
print(f"Predicted Severity: {y_pred_severity}")


ls

