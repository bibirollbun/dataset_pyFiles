import zarr
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle
import gc

print("All imports successful!")


home = "/kaggle/input/czii-cryo-et-object-identification"

# Example: Explore one tomogram
current_file = "/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/TS_86_3/VoxelSpacing10.000/denoised.zarr"

z = zarr.open(current_file, mode='r')

# See what's inside
print("Contents:", list(z.keys()))
print("Tree structure:")
print(z.tree())

# Examine the zarr data at level 2 resolution
arr = z['2']

print("Shape:", arr.shape)
print("Data type:", arr.dtype)
print("Data sample:")
print(arr[:2])


train_overlay = "/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns"

# Get all experiments
experiments = os.listdir(train_overlay)

all_particles = []
particle_id = 0

for experiment in experiments:
    picks_path = os.path.join(train_overlay, experiment, "Picks")
    
    if not os.path.exists(picks_path):
        continue
    
    # Get all particle type JSON files
    json_files = [f for f in os.listdir(picks_path) if f.endswith('.json')]
    
    for json_file in json_files:
        # Particle type is the filename without .json
        particle_type = json_file.replace('.json', '')
        
        json_path = os.path.join(picks_path, json_file)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract coordinates from each point
        for point in data['points']:
            loc = point['location']
            all_particles.append({
                'id': particle_id,
                'experiment': experiment,
                'particle_type': particle_type,
                'x': loc['x'],
                'y': loc['y'],
                'z': loc['z']
            })
            particle_id += 1

# Create DataFrame
train_labels = pd.DataFrame(all_particles)

print(train_labels.head(20))
print(f"\nTotal particles: {len(train_labels)}")
print(f"\nParticle type counts:")
print(train_labels['particle_type'].value_counts())
print(f"\nExperiments:")
print(train_labels['experiment'].unique())

# Save to working directory
train_labels.to_csv('/kaggle/working/train_labels.csv', index=False)


print("\nExtracting training patches from ALL experiments...")

patch_size = 32
X_train_all = []
y_train_all = []
voxel_spacing = 10.0

train_static = "/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns"
all_train_experiments = os.listdir(train_static)

for experiment in tqdm(all_train_experiments, desc="Processing experiments"):
    zarr_path = os.path.join(train_static, experiment, "VoxelSpacing10.000", "denoised.zarr")
    
    if not os.path.exists(zarr_path):
        print(f"Skipping {experiment} - zarr not found")
        continue
    
    # Load tomogram
    z_exp = zarr.open(zarr_path, mode='r')
    arr_exp = z_exp['2']  # Use level 2 for memory efficiency
    scale_factor = 4  # Level 2 is 4x downsampled
    
    # Load particles for this experiment
    particles = train_labels[train_labels['experiment'] == experiment].reset_index(drop=True)
    
    if len(particles) == 0:
        continue
    
    # Convert Angstrom to pixel coordinates (accounting for scale)
    particles['x_pixel'] = (particles['x'] / voxel_spacing / scale_factor).astype(int)
    particles['y_pixel'] = (particles['y'] / voxel_spacing / scale_factor).astype(int)
    particles['z_pixel'] = (particles['z'] / voxel_spacing / scale_factor).astype(int)
    
    # Extract positive examples
    for idx, particle in particles.iterrows():
        x, y, z_coord = particle['x_pixel'], particle['y_pixel'], particle['z_pixel']
        particle_type = particle['particle_type']
        
        # Check bounds
        if (z_coord - patch_size//2 >= 0 and z_coord + patch_size//2 <= arr_exp.shape[0] and
            y - patch_size//2 >= 0 and y + patch_size//2 <= arr_exp.shape[1] and
            x - patch_size//2 >= 0 and x + patch_size//2 <= arr_exp.shape[2]):
            
            patch = arr_exp[
                z_coord - patch_size//2 : z_coord + patch_size//2,
                y - patch_size//2 : y + patch_size//2,
                x - patch_size//2 : x + patch_size//2
            ]
            
            if patch.shape == (patch_size, patch_size, patch_size):
                X_train_all.append(patch)
                y_train_all.append(particle_type)
    
    # Extract negative examples (limit to avoid too many)
    num_negatives = min(len(particles), 30)
    negative_count = 0
    attempts = 0
    max_attempts = num_negatives * 10
    
    while negative_count < num_negatives and attempts < max_attempts:
        attempts += 1
        z_rand = np.random.randint(patch_size//2, arr_exp.shape[0] - patch_size//2)
        y_rand = np.random.randint(patch_size//2, arr_exp.shape[1] - patch_size//2)
        x_rand = np.random.randint(patch_size//2, arr_exp.shape[2] - patch_size//2)
        
        # Check if far from particles
        min_dist = 20
        is_far = True
        for _, p in particles.iterrows():
            dist = np.sqrt((x_rand - p['x_pixel'])**2 + (y_rand - p['y_pixel'])**2 + (z_rand - p['z_pixel'])**2)
            if dist < min_dist:
                is_far = False
                break
        
        if is_far:
            patch = arr_exp[
                z_rand - patch_size//2 : z_rand + patch_size//2,
                y_rand - patch_size//2 : y_rand + patch_size//2,
                x_rand - patch_size//2 : x_rand + patch_size//2
            ]
            if patch.shape == (patch_size, patch_size, patch_size):
                X_train_all.append(patch)
                y_train_all.append('background')
                negative_count += 1
    
    # Clean up
    del z_exp, arr_exp
    gc.collect()

print(f"\nTotal patches collected: {len(X_train_all)}")

X_train_all = np.array(X_train_all)
y_train_all = np.array(y_train_all)

print(f"Training data shape: {X_train_all.shape}")
print(f"Unique labels: {np.unique(y_train_all)}")


# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_all)
y_train_categorical = to_categorical(y_train_encoded)

num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

# Normalize
X_train_all = X_train_all.astype('float32')
mean = X_train_all.mean()
std = X_train_all.std()
X_train_all = (X_train_all - mean) / std

# Add channel dimension
X_train_all = X_train_all[..., np.newaxis]

print(f"Final X_train shape: {X_train_all.shape}")
print(f"Final y_train shape: {y_train_categorical.shape}")


print("\nBuilding model...")
model = models.Sequential([
    layers.Input(shape=(32, 32, 32, 1)),
    
    layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D((2, 2, 2)),
    
    layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D((2, 2, 2)),
    
    layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D((2, 2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\nTraining model...")
history = model.fit(
    X_train_all, 
    y_train_categorical, 
    epochs=10,  # Reduced from 20 for faster submission
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Save model and parameters
model.save('/kaggle/working/particle_detector_3d.h5')
print("\nModel saved!")

# Save label encoder and normalization params
with open('/kaggle/working/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('/kaggle/working/normalization_params.pkl', 'wb') as f:
    pickle.dump({'mean': mean, 'std': std}, f)

print("Label encoder and normalization params saved!")

# Clean up training data to free memory
del X_train_all, y_train_all, y_train_categorical
gc.collect()


print("\n\n=== MAKING PREDICTIONS ===\n")

class_names = label_encoder.classes_
print(f"Class names: {class_names}")

# Path to test data
test_path = "/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns"

# Get all test experiments
test_experiments = os.listdir(test_path)
print(f"Found {len(test_experiments)} test experiments")

all_predictions = []
prediction_id = 0

# Parameters
patch_size = 32
stride = 48
voxel_spacing = 10.0
confidence_threshold = 0.6
batch_size = 32

# Process each test experiment with error handling
for experiment in test_experiments:
    print(f"\nProcessing experiment: {experiment}")
    
    try:
        test_zarr_path = os.path.join(test_path, experiment, "VoxelSpacing10.000", "denoised.zarr")
        
        if not os.path.exists(test_zarr_path):
            print(f"  Skipping {experiment} - zarr file not found")
            continue
        
        z_test = zarr.open(test_zarr_path, mode='r')
        arr_test = z_test['2']
        scale_factor = 4
        
        print(f"  Tomogram shape (level 2): {arr_test.shape}")
        
        patches = []
        patch_coords = []
        
        z_range = range(0, max(1, arr_test.shape[0] - patch_size), stride)
        y_range = range(0, max(1, arr_test.shape[1] - patch_size), stride)
        x_range = range(0, max(1, arr_test.shape[2] - patch_size), stride)
        
        total_windows = len(z_range) * len(y_range) * len(x_range)
        print(f"  Total windows: {total_windows}")
        
        processed = 0
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    z_end = min(z + patch_size, arr_test.shape[0])
                    y_end = min(y + patch_size, arr_test.shape[1])
                    x_end = min(x + patch_size, arr_test.shape[2])
                    
                    patch = arr_test[z:z_end, y:y_end, x:x_end]
                    
                    if patch.shape != (patch_size, patch_size, patch_size):
                        continue
                    
                    patch_norm = (patch - mean) / std
                    patches.append(patch_norm)
                    patch_coords.append((z, y, x))
                    
                    if len(patches) >= batch_size:
                        patches_array = np.array(patches)[..., np.newaxis]
                        preds = model.predict(patches_array, verbose=0)
                        
                        for i, pred in enumerate(preds):
                            if pred.max() > confidence_threshold:
                                particle_type = class_names[pred.argmax()]
                                
                                if particle_type != 'background':
                                    z_p, y_p, x_p = patch_coords[i]
                                    
                                    x_center = (x_p + patch_size // 2) * scale_factor
                                    y_center = (y_p + patch_size // 2) * scale_factor
                                    z_center = (z_p + patch_size // 2) * scale_factor
                                    
                                    x_angstrom = x_center * voxel_spacing
                                    y_angstrom = y_center * voxel_spacing
                                    z_angstrom = z_center * voxel_spacing
                                    
                                    all_predictions.append({
                                        'id': prediction_id,
                                        'experiment': experiment,
                                        'particle_type': particle_type,
                                        'x': x_angstrom,
                                        'y': y_angstrom,
                                        'z': z_angstrom
                                    })
                                    prediction_id += 1
                        
                        patches = []
                        patch_coords = []
                        processed += batch_size
                        
                        if processed % 500 == 0:
                            print(f"  Processed {processed}/{total_windows} windows")
        
        # Process remaining patches
        if len(patches) > 0:
            patches_array = np.array(patches)[..., np.newaxis]
            preds = model.predict(patches_array, verbose=0)
            
            for i, pred in enumerate(preds):
                if pred.max() > confidence_threshold:
                    particle_type = class_names[pred.argmax()]
                    
                    if particle_type != 'background':
                        z_p, y_p, x_p = patch_coords[i]
                        
                        x_center = (x_p + patch_size // 2) * scale_factor
                        y_center = (y_p + patch_size // 2) * scale_factor
                        z_center = (z_p + patch_size // 2) * scale_factor
                        
                        x_angstrom = x_center * voxel_spacing
                        y_angstrom = y_center * voxel_spacing
                        z_angstrom = z_center * voxel_spacing
                        
                        all_predictions.append({
                            'id': prediction_id,
                            'experiment': experiment,
                            'particle_type': particle_type,
                            'x': x_angstrom,
                            'y': y_angstrom,
                            'z': z_angstrom
                        })
                        prediction_id += 1
        
        print(f"  Completed {experiment}")
        
        del z_test, arr_test, patches, patch_coords
        gc.collect()
        
    except Exception as e:
        print(f"  ERROR processing {experiment}: {str(e)}")
        continue

print(f"\n=== PREDICTION SUMMARY ===")
print(f"Total predictions: {len(all_predictions)}")


submission = pd.DataFrame(all_predictions)

if len(submission) > 0:
    print(f"\nPredictions by type:")
    print(submission['particle_type'].value_counts())
    
    print(f"\nPredictions by experiment:")
    print(submission['experiment'].value_counts())
else:
    print("\nWARNING: No predictions made!")
    submission = pd.DataFrame(columns=['id', 'experiment', 'particle_type', 'x', 'y', 'z'])

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission saved to /kaggle/working/submission.csv")
print("Ready to submit!")

