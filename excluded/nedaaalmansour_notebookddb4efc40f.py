import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Let's first explore what we actually have
base_path = "/kaggle/input/deepfake-detection-challenge"
train_videos_path = f"{base_path}/train_sample_videos"

print("=== Exploring the dataset ===")
print("Files in main directory:")
for item in os.listdir(base_path):
    print(f"  {item}")

print(f"\nFiles in train_sample_videos (first 10):")
if os.path.exists(train_videos_path):
    train_files = os.listdir(train_videos_path)
    for f in train_files[:10]:
        print(f"  {f}")
    print(f"Total files in train_sample_videos: {len(train_files)}")
else:
    print("train_sample_videos directory not found!")

# Check if there's a metadata.json in train_sample_videos
metadata_path = f"{train_videos_path}/metadata.json"
print(f"\nChecking for metadata.json: {os.path.exists(metadata_path)}")

# If no metadata.json, let's check what's in sample_submission.csv
sample_sub_path = f"{base_path}/sample_submission.csv"
if os.path.exists(sample_sub_path):
    print(f"\nReading sample_submission.csv:")
    df = pd.read_csv(sample_sub_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First 5 rows:")
    print(df.head())
    
    # Check if these files are in train_sample_videos
    if os.path.exists(train_videos_path):
        sample_files = df['filename'].tolist()[:5]
        print(f"\nChecking if sample_submission files exist in train_sample_videos:")
        for f in sample_files:
            exists = os.path.exists(f"{train_videos_path}/{f}")
            print(f"  {f}: {'âœ“' if exists else 'âœ—'}")

# Try to find metadata.json in different locations
possible_metadata_paths = [
    f"{train_videos_path}/metadata.json",
    f"{base_path}/metadata.json",
    f"{base_path}/train_metadata.json"
]

metadata_found = None
for path in possible_metadata_paths:
    if os.path.exists(path):
        metadata_found = path
        print(f"\nFound metadata at: {path}")
        break

if metadata_found:
    print("Trying to read metadata...")
    try:
        import json
        with open(metadata_found, 'r') as f:
            # Read first few characters to check format
            content = f.read(200)
            f.seek(0)  # Reset file pointer
            print(f"First 200 chars: {content}")
            
            # Try to load JSON
            metadata = json.load(f)
            print(f"Successfully loaded metadata with {len(metadata)} entries")
            
            # Show sample
            sample_key = next(iter(metadata))
            print(f"Sample entry: {sample_key} -> {metadata[sample_key]}")
            
        # Now do the split
        print("\n=== Starting data split ===")
        
        filenames = list(metadata.keys())
        labels = [metadata[f]['label'] for f in filenames]
        
        print(f"Total files: {len(filenames)}")
        print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        # Split: 80-10-10
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            filenames, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        val_files, test_files = train_test_split(
            temp_files, test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Create output directories
        output_dir = "/kaggle/working/split_data"
        for split in ['train', 'validation', 'test']:
            os.makedirs(f"{output_dir}/{split}", exist_ok=True)
        
        # Copy files
        splits = {'train': train_files, 'validation': val_files, 'test': test_files}
        
        for split_name, file_list in splits.items():
            print(f"\nCopying {split_name} files...")
            copied = 0
            for filename in file_list[:5]:  # Copy only first 5 for testing
                src = f"{train_videos_path}/{filename}"
                dst = f"{output_dir}/{split_name}/{filename}"
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    copied += 1
            print(f"Copied {copied} files to {split_name}")
        
        print("SUCCESS! Data split completed.")
        
    except Exception as e:
        print(f"Error processing metadata: {e}")
        print("Let's try a different approach...")
        
else:
    print("\nNo metadata.json found. Let's work with what we have...")
    
    # List all video files and create basic split
    if os.path.exists(train_videos_path):
        video_files = [f for f in os.listdir(train_videos_path) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Found {len(video_files)} video files")
        
        if len(video_files) > 0:
            # Simple random split without labels
            from sklearn.model_selection import train_test_split
            
            train_files, temp_files = train_test_split(video_files, test_size=0.2, random_state=42)
            val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
            
            print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
            
            # Create directories and copy files
            output_dir = "/kaggle/working/split_data"
            splits = {'train': train_files, 'validation': val_files, 'test': test_files}
            
            for split_name, file_list in splits.items():
                os.makedirs(f"{output_dir}/{split_name}", exist_ok=True)
                print(f"Created {output_dir}/{split_name}")
            
            print("Basic split completed without metadata!")
        else:
            print("No video files found!")
    else:
        print("train_sample_videos directory not found!")


import json
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

# Paths
base_path = "/kaggle/input/deepfake-detection-challenge"
train_videos_path = f"{base_path}/train_sample_videos"
metadata_path = f"{train_videos_path}/metadata.json"
output_dir = "/kaggle/working/split_data"

print("=== Loading metadata ===")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

filenames = list(metadata.keys())
labels = [metadata[f]['label'] for f in filenames]

print(f"Total videos: {len(filenames)}")
print(f"REAL: {labels.count('REAL')}")
print(f"FAKE: {labels.count('FAKE')}")

print("\n=== Splitting data (80-10-10) ===")
# Split: 80% train, 10% validation, 10% test
train_files, temp_files, train_labels, temp_labels = train_test_split(
    filenames, labels, test_size=0.2, stratify=labels, random_state=42
)

val_files, test_files = train_test_split(
    temp_files, test_size=0.5, stratify=temp_labels, random_state=42
)

print(f"Train: {len(train_files)} videos")
print(f"Validation: {len(val_files)} videos") 
print(f"Test: {len(test_files)} videos")

# Verify label distribution in each split
def count_labels(file_list):
    labels = [metadata[f]['label'] for f in file_list]
    return {'REAL': labels.count('REAL'), 'FAKE': labels.count('FAKE')}

print(f"\nLabel distribution:")
print(f"Train: {count_labels(train_files)}")
print(f"Val: {count_labels(val_files)}")
print(f"Test: {count_labels(test_files)}")

print("\n=== Creating directories ===")
for split in ['train', 'validation', 'test']:
    os.makedirs(f"{output_dir}/{split}", exist_ok=True)
    print(f"Created {output_dir}/{split}/")

print("\n=== Copying all files ===")
splits = {
    'train': train_files,
    'validation': val_files, 
    'test': test_files
}

total_copied = 0
for split_name, file_list in splits.items():
    print(f"\nCopying {split_name} files...")
    copied = 0
    missing = 0
    
    for i, filename in enumerate(file_list):
        src = f"{train_videos_path}/{filename}"
        dst = f"{output_dir}/{split_name}/{filename}"
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
        
        # Progress indicator
        if (i + 1) % 50 == 0 or i == len(file_list) - 1:
            print(f"  Progress: {i + 1}/{len(file_list)} - Copied: {copied}, Missing: {missing}")
    
    total_copied += copied
    print(f"  âœ“ {split_name}: {copied} files copied")

print(f"\n=== Creating metadata files ===")
# Create metadata for each split
train_metadata = {f: metadata[f] for f in train_files}
val_metadata = {f: metadata[f] for f in val_files}
test_metadata = {f: metadata[f] for f in test_files}

# Save metadata files
with open(f"{output_dir}/train_metadata.json", 'w') as f:
    json.dump(train_metadata, f, indent=2)

with open(f"{output_dir}/val_metadata.json", 'w') as f:
    json.dump(val_metadata, f, indent=2)

with open(f"{output_dir}/test_metadata.json", 'w') as f:
    json.dump(test_metadata, f, indent=2)

print(f"âœ“ Saved train_metadata.json ({len(train_metadata)} entries)")
print(f"âœ“ Saved val_metadata.json ({len(val_metadata)} entries)")
print(f"âœ“ Saved test_metadata.json ({len(test_metadata)} entries)")

print("\n=== Final Summary ===")
print(f"Total files copied: {total_copied}")
print(f"Data saved to: {output_dir}")
print(f"\nDirectory structure:")
print(f"{output_dir}/")
print(f"â”œâ”€â”€ train/ ({len(train_files)} videos)")
print(f"â”œâ”€â”€ validation/ ({len(val_files)} videos)")
print(f"â”œâ”€â”€ test/ ({len(test_files)} videos)")
print(f"â”œâ”€â”€ train_metadata.json")
print(f"â”œâ”€â”€ val_metadata.json")
print(f"â””â”€â”€ test_metadata.json")

print(f"\nðŸŽ‰ Split completed successfully!")
print(f"You can now use these paths for training:")
print(f"TRAIN_DIR = '{output_dir}/train'")
print(f"VAL_DIR = '{output_dir}/validation'")
print(f"TEST_DIR = '{output_dir}/test'")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception, InceptionV3, MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import cv2
from sklearn.utils.class_weight import compute_class_weight

# UPDATED DATA PATHS - Using split data
train_data_dir = '/kaggle/working/split_data/train'
valid_data_dir = '/kaggle/working/split_data/validation'  
test_data_dir = '/kaggle/working/split_data/test'

# Load metadata for class weights calculation
with open('/kaggle/working/split_data/train_metadata.json', 'r') as f:
    train_metadata = json.load(f)
with open('/kaggle/working/split_data/val_metadata.json', 'r') as f:
    val_metadata = json.load(f)
with open('/kaggle/working/split_data/test_metadata.json', 'r') as f:
    test_metadata = json.load(f)

print(f"Train samples: {len(train_metadata)}")
print(f"Validation samples: {len(val_metadata)}")
print(f"Test samples: {len(test_metadata)}")

# Image preprocessing settings
img_width, img_height = 224, 224
batch_size = 16
frames_per_video = 3

def extract_frames_from_video(video_path, num_frames=3):
    """Extract frames from video efficiently"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
    # Select frame indices evenly distributed
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Resize and normalize
            frame = cv2.resize(frame, (img_width, img_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
    
    cap.release()
    
    # If we couldn't extract enough frames, pad with copies of the last frame
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
            return None
    
    return np.array(frames)

class VideoDataGenerator:
    """Custom data generator for videos"""
    
    def __init__(self, metadata, video_dir, batch_size=16, frames_per_video=3):
        self.metadata = metadata
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.frames_per_video = frames_per_video
        self.video_files = list(metadata.keys())
        self.indices = np.arange(len(self.video_files))
        self.shuffle()
    
    def shuffle(self):
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.video_files) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X = []
        y = []
        
        for i in batch_indices:
            video_file = self.video_files[i]
            video_path = os.path.join(self.video_dir, video_file)
            
            # Extract frames
            frames = extract_frames_from_video(video_path, self.frames_per_video)
            
            if frames is not None:
                # Average frames to get single image
                avg_frame = np.mean(frames, axis=0)
                X.append(avg_frame)
                
                # Get label (FAKE=0, REAL=1)
                label = 1 if self.metadata[video_file]['label'] == 'REAL' else 0
                y.append(label)
        
        if len(X) == 0:
            # Return dummy data if no valid frames
            X = [np.zeros((img_height, img_width, 3))]
            y = [0]
        
        return np.array(X), np.array(y)

# Create data generators
print("Creating data generators...")
train_generator = VideoDataGenerator(train_metadata, train_data_dir, batch_size=batch_size)
validation_generator = VideoDataGenerator(val_metadata, valid_data_dir, batch_size=batch_size)
test_generator = VideoDataGenerator(test_metadata, test_data_dir, batch_size=batch_size)

print(f"Train batches: {len(train_generator)}")
print(f"Validation batches: {len(validation_generator)}")
print(f"Test batches: {len(test_generator)}")

# Calculate class weights for imbalanced dataset
def calculate_class_weights():
    """Calculate class weights for imbalanced dataset"""
    # Count samples in each class
    train_labels = [info['label'] for info in train_metadata.values()]
    real_count = train_labels.count('REAL')
    fake_count = train_labels.count('FAKE')
    
    total = real_count + fake_count
    
    # Calculate weights (inverse frequency)
    weight_real = total / (2 * real_count)
    weight_fake = total / (2 * fake_count)
    
    class_weights = {0: weight_fake, 1: weight_real}  # 0=FAKE, 1=REAL
    
    print(f"Class distribution - REAL: {real_count}, FAKE: {fake_count}")
    print(f"Class weights - FAKE: {weight_fake:.3f}, REAL: {weight_real:.3f}")
    
    return class_weights

class_weights = calculate_class_weights()

# Test data generator and visualize
print("\nTesting data generator...")
try:
    X_batch, y_batch = train_generator[0]
    print(f"Batch shape: {X_batch.shape}")
    print(f"Labels shape: {y_batch.shape}")
    print(f"Sample labels: {y_batch[:5]}")
    
    # Visualize some samples
    plt.figure(figsize=(15, 3))
    for i in range(min(5, len(X_batch))):
        plt.subplot(1, 5, i+1)
        plt.imshow(X_batch[i])
        label_name = 'REAL' if y_batch[i] == 1 else 'FAKE'
        plt.title(f'{label_name}')
        plt.axis('off')
    plt.suptitle("Sample Training Images (Extracted from Videos)")
    plt.show()
    
except Exception as e:
    print(f"Error in data generator: {e}")

# Calculate steps per epoch
train_steps_per_epoch = max(1, len(train_generator))
valid_steps_per_epoch = max(1, len(validation_generator))

print(f"Training steps per epoch: {train_steps_per_epoch}")
print(f"Validation steps per epoch: {valid_steps_per_epoch}")

# Training function
def train_model_with_generator(model, model_name, train_gen, val_gen, epochs=5):
    """Train model with custom generator"""
    print(f"\n{'='*50}")
    print(f"Training {model_name} Model")
    print(f"{'='*50}")
    
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss = 0
        train_acc = 0
        train_batches = 0
        
        train_gen.shuffle()
        
        for batch_idx in range(len(train_gen)):
            X_batch, y_batch = train_gen[batch_idx]
            
            if len(X_batch) == 0:
                continue
                
            # Train on batch
            batch_history = model.train_on_batch(X_batch, y_batch, class_weight=class_weights)
            train_loss += batch_history[0]
            train_acc += batch_history[1]
            train_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_gen)} - Loss: {batch_history[0]:.4f}, Acc: {batch_history[1]:.4f}")
        
        # Validation
        val_loss = 0
        val_acc = 0
        val_batches = 0
        
        for batch_idx in range(len(val_gen)):
            X_batch, y_batch = val_gen[batch_idx]
            
            if len(X_batch) == 0:
                continue
                
            batch_history = model.test_on_batch(X_batch, y_batch)
            val_loss += batch_history[0]
            val_acc += batch_history[1]
            val_batches += 1
        
        # Calculate averages
        if train_batches > 0:
            train_loss /= train_batches
            train_acc /= train_batches
        
        if val_batches > 0:
            val_loss /= val_batches
            val_acc /= val_batches
        
        # Store history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    return history

# 1. CNN Model
def create_cnn_model():
    """Create a simple CNN model"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# 2. Xception Model
def create_xception_model():
    """Create Xception-based model"""
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    
    # Freeze base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# 3. InceptionV3 Model
def create_inception_model():
    """Create InceptionV3-based model"""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    
    # Freeze base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# 4. MobileNet Model
def create_mobilenet_model():
    """Create MobileNet-based model"""
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    
    # Freeze base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Create and compile all models
print("\nCreating and compiling models...")

model_cnn = create_cnn_model()
model_cnn.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

xception_model = create_xception_model()
xception_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

inception_model = create_inception_model()
inception_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

mobilenet_model = create_mobilenet_model()
mobilenet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

print("All models created and compiled successfully!")

# Train all models
models = [
    (model_cnn, "CNN", 5),
    (xception_model, "Xception", 3),
    (inception_model, "InceptionV3", 3),  
    (mobilenet_model, "MobileNet", 3)
]

histories = {}
trained_models = {}

for model, name, epochs in models:
    print(f"\n{name} Model Summary:")
    if name == "CNN":
        model.summary()
    
    history = train_model_with_generator(model, name, train_generator, validation_generator, epochs=epochs)
    histories[name] = history
    trained_models[name] = model
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], 'b-', label='Train Accuracy')
    plt.plot(history['val_accuracy'], 'r-', label='Val Accuracy')
    plt.title(f'{name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], 'b-', label='Train Loss')
    plt.plot(history['val_loss'], 'r-', label='Val Loss')
    plt.title(f'{name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Evaluation Function
def evaluate_model(model, model_name, test_gen):
    """Evaluate a single model and display results"""
    print(f"\nEvaluating {model_name} Model...")
    
    # Get predictions
    predictions = []
    labels = []
    
    for batch_idx in range(len(test_gen)):
        X_batch, y_batch = test_gen[batch_idx]
        
        if len(X_batch) == 0:
            continue
        
        batch_pred = model.predict(X_batch, verbose=0)
        predictions.extend(batch_pred.flatten())
        labels.extend(y_batch)
    
    # Convert to binary predictions
    y_pred_binary = (np.array(predictions) > 0.5).astype(int)
    y_true = np.array(labels)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    print(f"{model_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=['FAKE', 'REAL']))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', 
                xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    
    return y_pred_binary, accuracy

# Evaluate all models
print("="*50)
print("MODEL EVALUATION")
print("="*50)

results = {}
for name, model in trained_models.items():
    pred, acc = evaluate_model(model, name, test_generator)
    results[name] = {'predictions': pred, 'accuracy': acc}

# Ensemble Method
def ensemble_predict(models_dict, test_gen):
    """Create ensemble predictions using majority voting"""
    print(f"\nCreating Ensemble Prediction...")
    
    all_predictions = []
    
    # Get predictions from each model (skip CNN for ensemble)
    for name, model in models_dict.items():
        if name == "CNN":  # Skip CNN for ensemble
            continue
            
        predictions = []
        for batch_idx in range(len(test_gen)):
            X_batch, y_batch = test_gen[batch_idx]
            if len(X_batch) == 0:
                continue
            batch_pred = model.predict(X_batch, verbose=0)
            predictions.extend(batch_pred.flatten())
        
        pred_binary = (np.array(predictions) > 0.5).astype(int)
        all_predictions.append(pred_binary)
        print(f"  {name} predictions added")
    
    # Majority voting
    all_predictions = np.array(all_predictions)
    ensemble_pred = np.apply_along_axis(
        lambda x: Counter(x).most_common(1)[0][0], 
        axis=0, 
        arr=all_predictions
    )
    
    return ensemble_pred

# Create ensemble
print("\n" + "="*50)
print("ENSEMBLE MODEL EVALUATION")
print("="*50)

ensemble_pred = ensemble_predict(trained_models, test_generator)

# Get test labels
test_labels = []
for batch_idx in range(len(test_generator)):
    X_batch, y_batch = test_generator[batch_idx]
    if len(X_batch) == 0:
        continue
    test_labels.extend(y_batch)

test_labels = np.array(test_labels)

# Ensemble evaluation
ensemble_acc = accuracy_score(test_labels, ensemble_pred)
ensemble_precision = precision_score(test_labels, ensemble_pred)
ensemble_recall = recall_score(test_labels, ensemble_pred)
ensemble_f1 = f1_score(test_labels, ensemble_pred)

print(f"Ensemble Results:")
print(f"  Accuracy:  {ensemble_acc:.4f}")
print(f"  Precision: {ensemble_precision:.4f}")  
print(f"  Recall:    {ensemble_recall:.4f}")
print(f"  F1-Score:  {ensemble_f1:.4f}")

print("\nEnsemble Classification Report:")
print(classification_report(test_labels, ensemble_pred, target_names=['FAKE', 'REAL']))

# Ensemble confusion matrix
conf_matrix_ensemble = confusion_matrix(test_labels, ensemble_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix_ensemble, annot=True, cmap='Blues', fmt='g', 
            xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.title('Ensemble Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Summary of all model performances
print("\n" + "="*50)
print("SUMMARY OF MODEL PERFORMANCES")
print("="*50)

all_accuracies = [(name, results[name]['accuracy']) for name in results.keys()]
all_accuracies.append(("Ensemble", ensemble_acc))

for name, acc in sorted(all_accuracies, key=lambda x: x[1], reverse=True):
    print(f"{name:12} Accuracy: {acc:.4f}")

# Find best performing model
best_model_name = max(all_accuracies, key=lambda x: x[1])[0]
print(f"\nBest performing model: {best_model_name}")

# Save all models
print(f"\nSaving models...")
model_cnn.save('/kaggle/working/cnn_model.h5')
xception_model.save('/kaggle/working/xception_model.h5')
inception_model.save('/kaggle/working/inception_model.h5')
mobilenet_model.save('/kaggle/working/mobilenet_model.h5')

print(f"All models saved:")
print(f"  CNN: /kaggle/working/cnn_model.h5")
print(f"  Xception: /kaggle/working/xception_model.h5")
print(f"  InceptionV3: /kaggle/working/inception_model.h5")
print(f"  MobileNet: /kaggle/working/mobilenet_model.h5")

# Save best individual model
if best_model_name != "Ensemble":
    best_model = trained_models[best_model_name]
    best_model.save('/kaggle/working/best_deepfake_model.h5')
    print(f"Best individual model saved as: /kaggle/working/best_deepfake_model.h5")

print(f"\nTraining completed successfully!")
print(f"All 4 models + ensemble trained and evaluated!")

