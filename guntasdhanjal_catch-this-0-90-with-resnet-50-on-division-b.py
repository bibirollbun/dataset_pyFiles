import numpy as np
import pandas as pd
import cv2
import os
import sys
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import tensorflow.keras.applications.resnet50 as resnet

# Check TensorFlow and GPU
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


# Load training data
try:
    df_train = pd.read_csv('/kaggle/input/grand-xray-slam-division-b/train2.csv')
    print(f"Loaded train2.csv with {len(df_train)} rows")
except FileNotFoundError:
    print("Error: train2.csv not found. Attach the dataset.")
    raise

# Define labels
conditions = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# Check for missing columns
missing = [col for col in conditions if col not in df_train.columns]
if missing:
    print(f"Error: Missing columns: {missing}")
    raise KeyError(f"Missing columns: {missing}")

# Split data
train_set, val_set = train_test_split(
    df_train, test_size=0.2, random_state=42, stratify=df_train['No Finding']
)
print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")


class XRayDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, img_size=(224, 224), is_test=False, **kwargs):
        super().__init__(**kwargs)
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_test = is_test
        self.image_dir = '/kaggle/input/grand-xray-slam-division-b/train2/' if not is_test else '/kaggle/input/grand-xray-slam-division-b/test2/'
        self.conditions = conditions
        
        if not os.path.exists(self.image_dir):
            print(f"Error: Directory {self.image_dir} not found.")
            raise FileNotFoundError(f"Directory {self.image_dir} missing.")
    
    def __len__(self):
        return (len(self.dataframe) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.dataframe))
        batch_data = self.dataframe.iloc[start:end]
        
        images, labels = [], []
        
        for _, row in batch_data.iterrows():
            img_path = os.path.join(self.image_dir, row['Image_name'])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                img = cv2.resize(img, self.img_size)
                img = resnet.preprocess_input(img)
                images.append(img)
                
                if not self.is_test:
                    labels.append(row[self.conditions].values.astype(np.float32))
        
        if not images:
            dummy_img = np.zeros((*self.img_size, 3), dtype=np.float32)
            images.append(dummy_img)
            if not self.is_test:
                labels.append(np.zeros(len(self.conditions), dtype=np.float32))
        
        if not self.is_test:
            return np.array(images), np.array(labels)
        else:
            return np.array(images)

# Create generators
batch_size = 32
train_generator = XRayDataGenerator(train_set, batch_size=batch_size)
val_generator = XRayDataGenerator(val_set, batch_size=batch_size)
print("Data generators created.")


def build_resnet_model(num_classes=14):
    base_model = resnet.ResNet50(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

model = build_resnet_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['AUC']
)

sys.stdout = old_stdout

print("Model Architecture: ResNet-50 + Custom Head")
print(f"Total parameters: {model.count_params():,}")
trainable_params = sum([tf.size(v) for v in model.trainable_variables])
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {model.count_params() - trainable_params:,}")
print("Model compiled successfully!")


# Train for 3 epochs
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3,
    verbose=1
)

# Display final validation AUC
val_auc = history.history['val_AUC'][-1] if 'val_AUC' in history.history else 0.0
print(f"Final Validation AUC-ROC: {val_auc:.4f}")


# Load sample submission
try:
    submission_df = pd.read_csv('/kaggle/input/grand-xray-slam-division-b/sample_submission_2.csv')
    print(f"Loaded sample_submission_2.csv with {len(submission_df)} rows")
except FileNotFoundError:
    print("Error: sample_submission_2.csv not found.")
    raise

# Create test generator
test_generator = XRayDataGenerator(submission_df, batch_size=batch_size, is_test=True)

# Predict on test set
predictions = []
for i in range(len(test_generator)):
    batch_images = test_generator[i]
    if isinstance(batch_images, tuple):
        batch_images = batch_images[0]
    
    batch_preds = model.predict(batch_images, verbose=0)
    predictions.append(batch_preds)

# Combine and save predictions
predictions = np.vstack(predictions)
predictions = predictions[:len(submission_df)]

submission_df[conditions] = predictions
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
print(f"Prediction shape: {predictions.shape}")
print(f"Submission shape: {submission_df.shape}")

