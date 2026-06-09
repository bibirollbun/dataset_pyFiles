import pandas as pd 
import numpy as np


train_data = pd.read_csv('../input/rsna-intracranial-aneurysm-detection/train.csv')
train_data.head()


train_data.shape


localizers = pd.read_csv('../input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
localizers.head(2)


localizers_count = localizers['location'].value_counts()
localizers_count


import matplotlib.pyplot as plt 

plt.figure(figsize = (8,8))
plt.pie(localizers_count, labels=localizers_count.index, autopct='%1.1f%%')
plt.show()


scan_types = train_data['Modality'].unique()
print("\nDifferent Scan types: ", scan_types)
modality_count = train_data['Modality'].value_counts()
print("\n", modality_count)


plt.figure(figsize =(6,4))
modality_count.plot(kind='bar', color='pink')
plt.title("Modality Distribution")
plt.xlabel("Modality")
plt.ylabel("Number of Scans")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle=' ', alpha=0.7)
plt.tight_layout()

for i, v in enumerate(modality_count):
    plt.text(i, v , str(v), ha='center', va='bottom', fontsize=9, rotation=0)

plt.show()


import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
from scipy.ndimage import zoom


def resize_volume(img, target_shape=(64, 128, 128)):
    current_shape = img.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    return zoom(img, zoom_factors, order=1)  # order=1 = linear interpolation
    
    
def load_dicom_series(series_path, target_shape=(128, 128)):
    files = glob(os.path.join(series_path, "*.dcm"))
    print("Found DICOM files:", len(files))  

    files = [pydicom.dcmread(f) for f in files]
    files = [f for f in files if hasattr(f, "ImagePositionPatient")]
    print("Files with ImagePositionPatient:", len(files)) 

    files = sorted(files, key=lambda s: float(s.ImagePositionPatient[2]))

    resized_slices = []
    for s in files:
        img = s.pixel_array
        resized_img = resize_volume(img, target_shape)
        resized_slices.append(resized_img)

    if not resized_slices:
        raise ValueError(f"No valid slices in {series_path}")  

    volume = np.stack(resized_slices)
    return volume


def normalize(volume):
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume

def show_middle_slices(volume, n=9):
    depth = volume.shape[0]
    idxs = np.linspace(0.25*depth, 0.75*depth, n).astype(int)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(idxs):
        plt.subplot(1, n, i+1)
        plt.imshow(volume[idx], cmap='gray')
        plt.title(f"Slice {idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Picking a random DICOM series to visualise the brain 
series_root = '../input/rsna-intracranial-aneurysm-detection/series'
all_series = [d for d in os.listdir(series_root) if os.path.isdir(os.path.join(series_root, d))]
random_series = random.choice(all_series)
random_series_path = os.path.join(series_root, random_series)

print("Visualizing:", random_series)

volume = load_dicom_series(random_series_path)
volume = normalize(volume)
show_middle_slices(volume)



import ipywidgets as widgets
from IPython.display import display

def interactive_view(volume):
    def view_slice(slice_idx):
        plt.imshow(volume[slice_idx], cmap='gray')
        plt.title(f"Slice {slice_idx}")
        plt.axis('off')
        plt.show()

    slider = widgets.IntSlider(min=0, max=volume.shape[0]-1, step=1, value=volume.shape[0]//2)
    widgets.interact(view_slice, slice_idx=slider)

interactive_view(volume)


from skimage import measure
import plotly.graph_objects as go

v = volume  

# Pick a good isosurface level after normalization (0â€“1)
iso_level = 0.1  

# Extract mesh surface
verts, faces, _, _ = measure.marching_cubes(v, level=iso_level)

x, y, z = verts.T
i, j, k = faces.T

fig = go.Figure(data=[go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    color='lightgray',
    opacity=1.0,
    lighting=dict(ambient=0.3, diffuse=1),
    lightposition=dict(x=100, y=200, z=0)
)])

fig.update_layout(
    scene=dict(aspectmode='data'),
    title="Full-Resolution Brain Surface"
)
fig.show()


print(f"Total slices in this scan: {volume.shape[0]}")


shapes = []

for folder in random.sample(os.listdir(series_root), 5):  # More samples = better view
    path = os.path.join(series_root, folder)
    try:
        slices = [pydicom.dcmread(os.path.join(path, f)).pixel_array for f in sorted(os.listdir(path))]
        vol = np.stack(slices)
        shapes.append(vol.shape)  # (depth, height, width)
    except:
        continue

# Separate dims
depths = [s[0] for s in shapes]
heights = [s[1] for s in shapes]
widths = [s[2] for s in shapes]

# Plot distributions
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(depths, bins=10)
plt.title("Depth Distribution")

plt.subplot(1, 3, 2)
plt.hist(heights, bins=10)
plt.title("Height Distribution")

plt.subplot(1, 3, 3)
plt.hist(widths, bins=10)
plt.title("Width Distribution")

plt.tight_layout()
plt.show()



binary_cols = [col for col in train_data.columns if set(train_data[col].unique()) <= {0, 1, np.nan}]
binary_cols.remove('Aneurysm Present')
print("Coutn of binary col", len(binary_cols))
binary_cols


artery_cols = [col for col in train_data.columns if col != 'SeriesInstanceUID']
label_dict = train_data.set_index('SeriesInstanceUID')[artery_cols].to_dict(orient='index')


from tqdm import tqdm

X_data = []
y_data = []

valid_folders = [f for f in os.listdir(series_root) if f in label_dict]

# Taking a small random sample for now to move ahead quickly.
valid_folders = random.sample(valid_folders, 50)


processed = 0  # ğŸ‘ˆ manual counter

for folder in tqdm(valid_folders):
    try:
        vol = load_dicom_series(os.path.join(series_root, folder), target_shape=(128, 128))
        vol = normalize(vol)
        vol = resize_volume(vol, target_shape=(64, 128, 128))
        vol = vol.astype(np.float16)

        label = label_dict[folder]
        X_data.append(vol)
        y_data.append(label)

        processed += 1
        if processed % 10 == 0:
            tqdm.write(f"âœ… Processed: {processed} folders")
    except Exception as e:
        tqdm.write(f"âš ï¸� Skipped {folder} due to error: {e}")



# Save it and load later because it pre-processing again and again takes a lot of time. 

# np.save('/kaggle/working/X_data_100.npy', X_data)
# np.save('/kaggle/working/y_data_100.npy', y_data)


X_data = np.stack(X_data)[..., np.newaxis]  # shape: (N, 64, 128, 128, 1)
y_data = [label['Aneurysm Present'] for label in y_data] ## TODO. do this while loading the DCOM 
y_data = np.array(y_data) 


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_data, 
    test_size=0.2, 
    stratify=y_data,  # Keeps class balance
    random_state=42
)


print(X_data.shape)
print(X_train.shape)
print(X_train[0].shape)


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(64, 128, 128, 1)),
    layers.Conv3D(16, 3, activation='relu', padding='same'),
    layers.MaxPool3D(2),
    layers.Conv3D(32, 3, activation='relu', padding='same'),
    layers.MaxPool3D(2),
    layers.Conv3D(64, 3, activation='relu', padding='same'),
    layers.GlobalAveragePooling3D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)


model.evaluate(X_val, y_val)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


model.save('aneurysm_model.h5')


from sklearn.metrics import accuracy_score, classification_report

# Convert predicted probabilities to binary labels (0 or 1)
pred_labels = (preds > 0.5).astype(int).flatten()
print("Accuracy:", accuracy_score(y_val, pred_labels))

# Full classification report
print(classification_report(y_val, pred_labels))


# --- Submission maker (drop this after training) ---
import os, numpy as np, pandas as pd
from tqdm import tqdm

# Paths
DATA_ROOT = "../input/rsna-intracranial-aneurysm-detection"
TEST_ROOT = os.path.join(DATA_ROOT, "test")
SUB_PATH  = os.path.join(DATA_ROOT, "sample_submission.csv")

# 1) Read sample to get the exact required columns
sub = pd.read_csv(SUB_PATH)
label_cols = [c for c in sub.columns if c != "ID"]  # 13 location columns (or 1 "Label" if that's the format)

# 2) Collect test IDs (SeriesInstanceUID folders)
test_ids = [d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))]
test_ids = sorted(test_ids)

def predict_series(series_id: str) -> float:
    """Load â†’ preprocess â†’ predict single series â†’ return one probability."""
    series_path = os.path.join(TEST_ROOT, series_id)
    vol = load_dicom_series(series_path, target_shape=(128, 128))  # uses your existing helpers
    vol = normalize(vol)
    vol = resize_volume(vol, target_shape=(64, 128, 128)).astype(np.float32)
    vol = np.expand_dims(vol, axis=-1)  # (D,H,W,1)
    vol = np.expand_dims(vol, axis=0)   # (1,D,H,W,1) batch
    prob = float(model.predict(vol, verbose=0)[0, 0])  # sigmoid output
    return prob

# 3) Run inference and fill submission
rows = []
for sid in tqdm(test_ids, desc="Predicting"):
    p = predict_series(sid)
    row = {"ID": sid}
    # If competition needs 13 columns, we broadcast the same scalar for now.
    # If it's a single 'Label', this still works because label_cols == ['Label'].
    for c in label_cols:
        row[c] = p
    rows.append(row)

submission = pd.DataFrame(rows, columns=["ID"] + label_cols)
submission.to_csv("submission.csv", index=False)
print( Saved submission.csv with", len(submission), "rows and", len(label_cols), "label column(s).")





