# Install essential DICOM handlers
!pip install pylibjpeg pylibjpeg-libjpeg gdcm --quiet

# Core imports
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

# TF/Keras imports
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Verify TF version and GPU
print("TF Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# First, fix numpy version compatibility
!pip install --force-reinstall numpy==1.24.3
%reset -f  # Restart Python session

# Reinstall core packages with version locking
!pip install --no-cache-dir \
    pydicom==2.4.3 \
    tensorflow==2.17.1 \
    matplotlib==3.7.5 \
    pandas==2.0.3

# Verify installation
import numpy as np
print("NumPy Version:", np.__version__)  # Should be 1.24.3


# Corrected analyze_study function
def analyze_study(study_uid):
    path = f"/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/{study_uid}"
    if not os.path.exists(path):
        return None
    
    try:
        slices = sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]))
        sample_slice = pydicom.dcmread(f"{path}/{slices[0]}")  # Fixed missing }
        
        return {
            'num_slices': len(slices),
            'slice_thickness': sample_slice.get('SliceThickness', 'Missing'),
            'pixel_spacing': sample_slice.get('PixelSpacing', [1.0, 1.0]),
            'modality': sample_slice.get('Modality', 'CT')
        }
    except Exception as e:
        print(f"Error analyzing {study_uid}: {str(e)}")
        return None


# Completely clean the environment
!pip install --force-reinstall --ignore-installed \
    numpy==1.23.5 \
    tensorflow==2.12.0 \
    pandas==1.5.3 \
    matplotlib==3.7.1 \
    pydicom==2.3.1


import numpy as np
import tensorflow as tf

print("NumPy:", np.__version__)
print("TF:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# First, clean existing installations
!pip uninstall -y tensorflow numpy

# Install compatible versions
!pip install --no-cache-dir \
    tensorflow==2.13.0 \
    numpy==1.24.3 \
    pydicom==2.3.1

# Restart kernel after this


import tensorflow as tf
import numpy as np

print("TF Version:", tf.__version__)
print("NumPy Version:", np.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# Clean existing installations
!pip uninstall -y tensorflow numpy keras

# Install specific compatible versions
!pip install --user --no-cache-dir \
    tensorflow==2.12.0 \
    numpy==1.24.3 \
    protobuf==3.20.3 \
    absl-py==1.4.0 \
    pydicom==2.3.1

# Restart kernel after installation


import tensorflow as tf
print("TF Version:", tf.__version__)
print("Initialization successful:", tf.constant(1) + tf.constant(1))


import pydicom
import numpy as np
import matplotlib.pyplot as plt

def load_dicom_v2(path):
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None

# Test loading
sample_path = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/1.2.826.0.1.3680043.10001/1.dcm"
test_img = load_dicom_v2(sample_path)

if test_img is not None:
    plt.figure(figsize=(6,6))
    plt.imshow(test_img, cmap='bone')
    plt.title("Sample Cervical Spine Image")
    plt.axis('off')
    plt.show()
    print(f"Image shape: {test_img.shape}, Value range: [{test_img.min():.2f}, {test_img.max():.2f}]")
else:
    print("Failed to load sample image")


# Step 1: Load Data
import pandas as pd

try:
    train_df = pd.read_csv('/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv')
    print("Training data loaded successfully!")
    print("Columns:", train_df.columns.tolist())
    print("\nFirst 3 rows:")
    display(train_df.head(3))
except FileNotFoundError:
    print("Error: Training CSV file not found. Check dataset path!")
except Exception as e:
    print(f"Unexpected error: {str(e)}")


import os

csv_path = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv'
print(f"File exists: {os.path.exists(csv_path)}")  # Should output "True"


with open(csv_path, 'r') as f:
    lines = [next(f) for _ in range(5)]
print("Raw CSV lines:")
for line in lines:
    print(repr(line))  # Show hidden characters like quotes or commas


import pandas as pd

try:
    train_df = pd.read_csv(
        csv_path,
        dtype=str,  # Load everything as strings
        engine='python',  # Use Python parser for better error handling
        quotechar='"',  # Handle quoted fields properly
        on_bad_lines='warn'  # Flag problematic rows
    )
    
    print("Temporary columns:", train_df.columns.tolist())
    
    # Clean all columns of unexpected characters
    for col in train_df.columns:
        train_df[col] = train_df[col].str.replace(r'[^a-zA-Z0-9\.]', '', regex=True)
    
    # Convert numeric columns
    numeric_cols = ['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    train_df[numeric_cols] = train_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    print("\nData types after cleanup:")
    print(train_df.dtypes)
    print("\nSample data:")
    print(train_df.head(3).to_string())
    
except pd.errors.ParserError as e:
    print(f"CSV Parsing Error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")


# Check available columns
print("Training Data Columns:")
print(train_df.columns.tolist())

# Check study-slice relationship
print("\nUnique Studies:", train_df['StudyInstanceUID'].nunique())
print("Total Rows:", len(train_df))

# Verify one study
sample_study = train_df.iloc[0]['StudyInstanceUID']# Check available columns
print("Training Data Columns:")
print(train_df.columns.tolist())

# Check study-slice relationship
print("\nUnique Studies:", train_df['StudyInstanceUID'].nunique())
print("Total Rows:", len(train_df))

# Verify one study
sample_study = train_df.iloc[0]['StudyInstanceUID']
study_slices = train_df[train_df['StudyInstanceUID'] == sample_study]
print(f"\nSample Study ({sample_study}) has {len(study_slices)} slices")
study_slices = train_df[train_df['StudyInstanceUID'] == sample_study]
print(f"\nSample Study ({sample_study}) has {len(study_slices)} slices")


import seaborn as sns

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x='patient_overall', data=train_df)
plt.title('Overall Fracture Distribution')

plt.subplot(1,2,2)
vertebrae_counts = train_df[['C1','C2','C3','C4','C5','C6','C7']].sum()
sns.barplot(x=vertebrae_counts.index, y=vertebrae_counts.values)
plt.title('Vertebrae-wise Fracture Counts')
plt.tight_layout()
plt.show()


# Create study-level dataframe
study_df = train_df.groupby('StudyInstanceUID').agg({
    'patient_overall': 'max',
    'C1': 'max',
    'C2': 'max',
    'C3': 'max',
    'C4': 'max',
    'C5': 'max',
    'C6': 'max',
    'C7': 'max'
}).reset_index()

print("\nStudy-level Statistics:")
print(f"Total studies: {len(study_df)}")
print(f"Fracture prevalence: {study_df['patient_overall'].mean():.2%}")


import cv2
from sklearn.model_selection import train_test_split

class SpineDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, study_list, batch_size=8, dim=(512,512), n_channels=1):
        self.study_list = study_list
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.study_list) / self.batch_size))
    
    def __getitem__(self, index):
        batch_studies = self.study_list[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.empty((len(batch_studies), *self.dim, self.n_channels))
        y_patient = np.empty(len(batch_studies), dtype=np.float32)
        y_vertebrae = np.empty((len(batch_studies), 7), dtype=np.float32)
        
        for i, study_id in enumerate(batch_studies):
            study_path = f"/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/{study_id}"
            slices = sorted(os.listdir(study_path), key=lambda x: int(x.split('.')[0]))
            
            # Load middle slice
            mid_idx = len(slices) // 2
            img = load_dicom_v2(f"{study_path}/{slices[mid_idx]}")
            
            # Preprocessing
            img = cv2.resize(img, self.dim)
            if img.ndim == 2: 
                img = np.expand_dims(img, axis=-1)
                
            X[i,] = img
            study_data = study_df[study_df['StudyInstanceUID'] == study_id].iloc[0]
            y_patient[i] = study_data['patient_overall']
            y_vertebrae[i] = study_data[['C1','C2','C3','C4','C5','C6','C7']].values
            
        return X, {'patient_output': y_patient, 'vertebrae_output': y_vertebrae}
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.study_list))
        np.random.shuffle(self.indices)


def build_multi_task_model(input_shape=(512,512,1)):
    # Shared backbone
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Patient-level prediction
    patient_out = tf.keras.layers.Dense(1, activation='sigmoid', name='patient_output')(x)
    
    # Vertebrae-level predictions
    vertebrae_out = tf.keras.layers.Dense(7, activation='sigmoid', name='vertebrae_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[patient_out, vertebrae_out])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            'patient_output': 'binary_crossentropy',
            'vertebrae_output': 'binary_crossentropy'
        },
        metrics={
            'patient_output': ['accuracy', tf.keras.metrics.AUC(name='auc')],
            'vertebrae_output': tf.keras.metrics.AUC(name='vertebrae_auc', multi_label=True)
        },
        loss_weights=[0.7, 0.3]  # Weight patient-level prediction higher
    )
    return model

model = build_multi_task_model()
model.summary()


# Split studies into train/validation
train_studies, val_studies = train_test_split(
    study_df['StudyInstanceUID'].values,
    test_size=0.2,
    stratify=study_df['patient_overall'],
    random_state=42
)

print(f"Training studies: {len(train_studies)}")
print(f"Validation studies: {len(val_studies)}")


# Update ModelCheckpoint callback to use .keras extension
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model.keras',  # Changed to .keras format
                                      save_best_only=True,
                                      monitor='val_patient_output_auc',
                                      mode='max'),
    tf.keras.callbacks.EarlyStopping(patience=5,
                                   restore_best_weights=True,
                                   monitor='val_patient_output_auc',
                                   mode='max')
]

# Add class weights for imbalance handling (from Phase 5.2)
patient_weights = {0: 1, 1: study_df['patient_overall'].value_counts()[0]/study_df['patient_overall'].value_counts()[1]}

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    class_weight={'patient_output': patient_weights},
    verbose=1
)


# Add missing import at the top of your notebook
import os

# Revised SpineDataGenerator with error handling
class SpineDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, study_list, batch_size=8, dim=(512,512), n_channels=1):
        self.study_list = study_list
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.study_list) / self.batch_size))
    
    def __getitem__(self, index):
        batch_studies = self.study_list[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.empty((len(batch_studies), *self.dim, self.n_channels))
        y_patient = np.empty(len(batch_studies), dtype=np.float32)
        y_vertebrae = np.empty((len(batch_studies), 7), dtype=np.float32)
        
        for i, study_id in enumerate(batch_studies):
            try:
                study_path = f"/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/{study_id}"
                if not os.path.exists(study_path):
                    raise FileNotFoundError(f"Study directory not found: {study_path}")
                    
                slices = sorted(os.listdir(study_path), key=lambda x: int(x.split('.')[0]))
                mid_idx = len(slices) // 2
                
                img = load_dicom_v2(f"{study_path}/{slices[mid_idx]}")
                img = cv2.resize(img, self.dim)
                if img.ndim == 2: 
                    img = np.expand_dims(img, axis=-1)
                    
                X[i,] = img
                study_data = study_df[study_df['StudyInstanceUID'] == study_id].iloc[0]
                y_patient[i] = study_data['patient_overall']
                y_vertebrae[i] = study_data[['C1','C2','C3','C4','C5','C6','C7']].values
                
            except Exception as e:
                print(f"Error processing {study_id}: {str(e)}")
                # Handle failed samples by returning zeros
                X[i,] = np.zeros((*self.dim, self.n_channels))
                y_patient[i] = 0
                y_vertebrae[i] = np.zeros(7)
                
        return X, {'patient_output': y_patient, 'vertebrae_output': y_vertebrae}
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.study_list))
        np.random.shuffle(self.indices)


# Test the generator with first 5 studies
test_gen = SpineDataGenerator(study_df['StudyInstanceUID'].values[:5], batch_size=2)
X_batch, y_batch = test_gen[0]

print("Batch shape:", X_batch.shape)
print("Patient labels:", y_batch['patient_output'])
print("Vertebrae labels:", y_batch['vertebrae_output'][0])


# Add these at the top
import numpy as np
import cv2  # For image resizing


# Ensure study_df is properly created from train_df
study_df = train_df.groupby('StudyInstanceUID').agg({
    'patient_overall': 'max',
    'C1': 'max',
    'C2': 'max',
    'C3': 'max',
    'C4': 'max',
    'C5': 'max',
    'C6': 'max',
    'C7': 'max'
}).reset_index()


# Modify load_dicom_v2 to ensure proper normalization
def load_dicom_v2(path):
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None


import os
import numpy as np
import cv2
import pydicom
import tensorflow as tf

# --- Data Loading Functions ---
def load_dicom_v2(path):
    """Improved DICOM loader with error handling"""
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None

# --- Data Generator Class ---
class SpineDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, study_list, batch_size=8, dim=(512,512), n_channels=1):
        self.study_list = study_list
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.study_list) / self.batch_size))
    
    def __getitem__(self, index):
        batch_studies = self.study_list[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.empty((len(batch_studies), *self.dim, self.n_channels))
        y_patient = np.empty(len(batch_studies), dtype=np.float32)
        y_vertebrae = np.empty((len(batch_studies), 7), dtype=np.float32)
        
        for i, study_id in enumerate(batch_studies):
            try:
                study_path = f"/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/{study_id}"
                
                if not os.path.exists(study_path):
                    raise FileNotFoundError(f"Missing study: {study_id}")
                    
                slices = sorted(os.listdir(study_path), 
                              key=lambda x: int(x.split('.')[0]))
                mid_idx = len(slices) // 2
                
                img = load_dicom_v2(f"{study_path}/{slices[mid_idx]}")
                if img is None:
                    raise ValueError("Empty image")
                
                # Resize and normalize
                img = cv2.resize(img, self.dim)
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                
                X[i,] = img
                
                # Get labels
                study_data = study_df[study_df['StudyInstanceUID'] == study_id].iloc[0]
                y_patient[i] = study_data['patient_overall']
                y_vertebrae[i] = study_data[['C1','C2','C3','C4','C5','C6','C7']].values
                
            except Exception as e:
                print(f"Error processing {study_id}: {str(e)}")
                # Fallback to empty data
                X[i,] = np.zeros((*self.dim, self.n_channels))
                y_patient[i] = 0
                y_vertebrae[i] = np.zeros(7)
                
        return X, {'patient_output': y_patient, 'vertebrae_output': y_vertebrae}
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.study_list))
        np.random.shuffle(self.indices)


# Test with known valid study
test_study = study_df.iloc[0]['StudyInstanceUID']
test_gen = SpineDataGenerator([test_study], batch_size=1)
X, y = test_gen[0]

print("Image shape:", X.shape)
print("Patient label:", y['patient_output'][0])
print("Vertebrae labels:", y['vertebrae_output'][0])


# Add to model architecture
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.1)
])


# Calculate weights for imbalance
class_counts = study_df['patient_overall'].value_counts()
patient_weights = {0: 1/class_counts[0], 1: 1/class_counts[1]}


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    class_weight={'patient_output': patient_weights},
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_patient_output_auc',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_patient_output_auc',
            patience=5,
            mode='max'
        )
    ]
)


# Install required DICOM handlers
!pip install pylibjpeg pylibjpeg-libjpeg gdcm --force-reinstall

# Restart kernel after installation (Session → Restart Session)


import os
import numpy as np
import cv2
import pydicom
import tensorflow as tf

# ----- Data Loading Function -----
def load_dicom_safe(path):
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None

# ----- Fixed Data Generator -----
class SpineDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, study_list, batch_size=8, dim=(512,512)):
        super().__init__()  # Important fix!
        self.study_list = study_list
        self.batch_size = batch_size
        self.dim = dim
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.study_list) / self.batch_size))
    
    def __getitem__(self, index):
        batch_studies = self.study_list[index*self.batch_size:(index+1)*self.batch_size]
        
        X = []
        y_patient = []
        y_vertebrae = []
        
        for study_id in batch_studies:
            try:
                # 1. Check study folder exists
                study_path = f"/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/{study_id}"
                if not os.path.exists(study_path):
                    continue
                    
                # 2. Get middle slice
                slices = sorted(os.listdir(study_path), key=lambda x: int(x.split('.')[0]))
                mid_idx = len(slices) // 2
                img_path = f"{study_path}/{slices[mid_idx]}"
                
                # 3. Load and validate image
                img = load_dicom_safe(img_path)
                if img is None or img.size == 0:
                    continue
                
                # 4. Resize and format
                img = cv2.resize(img, self.dim)
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                    
                X.append(img)
                
                # 5. Get labels
                study_data = train_df[train_df['StudyInstanceUID'] == study_id].iloc[0]
                y_patient.append(study_data['patient_overall'])
                y_vertebrae.append(study_data[['C1','C2','C3','C4','C5','C6','C7']].values)
                
            except Exception as e:
                print(f"Skipping {study_id}: {str(e)}")
                continue
                
        # Handle empty batches
        if len(X) == 0:
            return np.zeros((self.batch_size, *self.dim, 1)), \
                   {'patient_output': np.zeros(self.batch_size), 
                    'vertebrae_output': np.zeros((self.batch_size, 7))}
        
        return np.array(X), {
            'patient_output': np.array(y_patient),
            'vertebrae_output': np.array(y_vertebrae)
        }
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.study_list))
        np.random.shuffle(self.indices)


# Load the training data
train_df = pd.read_csv('/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv')

# Create test generator
test_gen = SpineDataGenerator(
    study_list=train_df['StudyInstanceUID'].values[:5],  # First 5 studies
    batch_size=2,
    dim=(512,512)
)

# Get one batch
X_batch, y_batch = test_gen[0]
print("Batch shape:", X_batch.shape)
print("Patient labels:", y_batch['patient_output'])


# Run this FIRST in a new cell
!pip uninstall -y pydicom pylibjpeg pylibjpeg-libjpeg gdcm
!pip install --user pydicom==2.4.3 pylibjpeg pylibjpeg-libjpeg gdcm


import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def load_dicom_final(path):
    try:
        # Force GDCM decompression
        dicom = pydicom.dcmread(path, force=True)
        dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        
        if 'PixelData' not in dicom:
            return None
            
        img = apply_voi_lut(dicom.pixel_array, dicom)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)
    except Exception as e:
        print(f"Skipped {path.split('/')[-2]}: {str(e)}")
        return None


class FinalSpineGenerator(tf.keras.utils.Sequence):
    def __init__(self, study_list, batch_size=8, dim=(512,512)):
        super().__init__()
        self.study_list = study_list
        self.batch_size = batch_size
        self.dim = dim
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.study_list) / self.batch_size))
    
    def __getitem__(self, index):
        batch_studies = self.study_list[index*self.batch_size:(index+1)*self.batch_size]
        
        X = []
        y_patient = []
        y_vertebrae = []
        
        for study_id in batch_studies:
            try:
                study_path = f"/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/{study_id}"
                if not os.path.exists(study_path):
                    continue
                    
                # Load ALL slices and pick clearest middle one
                slices = sorted([f for f in os.listdir(study_path) if f.endswith('.dcm')], 
                              key=lambda x: int(x.split('.')[0]))
                if not slices:
                    continue
                    
                # Try multiple slices for valid image
                for slice_idx in [len(slices)//2, 0, -1]:  # Middle, first, last
                    img = load_dicom_final(f"{study_path}/{slices[slice_idx]}")
                    if img is not None:
                        break
                
                if img is None:
                    continue
                
                # Resize and format
                img = cv2.resize(img, self.dim)
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                
                X.append(img)
                study_data = train_df[train_df['StudyInstanceUID'] == study_id].iloc[0]
                y_patient.append(study_data['patient_overall'])
                y_vertebrae.append(study_data[['C1','C2','C3','C4','C5','C6','C7']].values)
                
            except Exception as e:
                continue
                
        # Fallback for empty batches
        if len(X) == 0:
            return np.zeros((self.batch_size, *self.dim, 1)), \
                   {'patient_output': np.zeros(self.batch_size), 
                    'vertebrae_output': np.zeros((self.batch_size, 7))}
        
        return np.array(X), {
            'patient_output': np.array(y_patient),
            'vertebrae_output': np.array(y_vertebrae)
        }
    
    def on_epoch_end(self):
        np.random.shuffle(self.study_list)


test_gen = FinalSpineGenerator(
    study_list=['1.2.826.0.1.3680043.9443'],  # Previously failing study
    batch_size=1
)

X, y = test_gen[0]
print("Final test - Batch shape:", X.shape)
print("Sample values:", X[0, 250:255, 250:255, 0])


# 1. Split data
train_studies, val_studies = train_test_split(
    train_df['StudyInstanceUID'].unique(),
    test_size=0.2,
    stratify=train_df.groupby('StudyInstanceUID')['patient_overall'].first(),
    random_state=42
)

# 2. Create generators
train_gen = FinalSpineGenerator(train_studies, batch_size=16)
val_gen = FinalSpineGenerator(val_studies, batch_size=16)

# 3. Train model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_patient_output_auc',
            save_best_only=True,
            mode='max'
        )
    ],
    verbose=1
)

