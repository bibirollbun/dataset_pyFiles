import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings


train = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')


print("Total Cases: ", len(train))


train.columns


figure, axis = plt.subplots(1,3, figsize=(20,5)) 
for idx, d in enumerate(['foraminal', 'subarticular', 'canal']):
    diagnosis = list(filter(lambda x: x.find(d) > -1, train.columns))
    dff = train[diagnosis]
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        value_counts = dff.apply(pd.value_counts).fillna(0).T
    value_counts.plot(kind='bar', stacked=True, ax=axis[idx])
    axis[idx].set_title(f'{d} distribution')


# List out all of the Studies we have on patients.
part_1 = os.listdir('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images')
part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))


df_meta_f = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')


p1 = [(x, f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{x}") for x in part_1]
meta_obj = { p[0]: { 'folder_path': p[1], 
                    'SeriesInstanceUIDs': [] 
                   } 
            for p in p1 }


for m in meta_obj:
    meta_obj[m]['SeriesInstanceUIDs'] = list(
        filter(lambda x: x.find('.DS') == -1, 
               os.listdir(meta_obj[m]['folder_path'])
              )
    )


# grabs the correspoding series descriptions
for k in tqdm(meta_obj):
    for s in meta_obj[k]['SeriesInstanceUIDs']:
        if 'SeriesDescriptions' not in meta_obj[k]:
            meta_obj[k]['SeriesDescriptions'] = []
        try:
            meta_obj[k]['SeriesDescriptions'].append(
                df_meta_f[(df_meta_f['study_id'] == int(k)) & 
                (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
        except:
            print("Failed on", s, k)


meta_obj[list(meta_obj.keys())[1]]


patient = train.iloc[1]


ptobj = meta_obj[str(patient['study_id'])]


print(ptobj)


# Get data into the format
"""
im_list_dcm = {
    '{SeriesInstanceUID}': {
        'images': [
            {'SOPInstanceUID': ...,
             'dicom': PyDicom object
            },
            ...,
        ],
        'description': # SeriesDescription
    },
    ...
}
"""
im_list_dcm = {}
for idx, i in enumerate(ptobj['SeriesInstanceUIDs']):
    im_list_dcm[i] = {'images': [], 'description': ptobj['SeriesDescriptions'][idx]}
    images = glob.glob(f"{ptobj['folder_path']}/{ptobj['SeriesInstanceUIDs'][idx]}/*.dcm")
    for j in sorted(images, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):
        im_list_dcm[i]['images'].append({
            'SOPInstanceUID': j.split('/')[-1].replace('.dcm', ''), 
            'dicom': pydicom.dcmread(j) })


# Function to display images
def display_images(images, title, max_images_per_row=4):
    # Calculate the number of rows needed
    num_images = len(images)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row  # Ceiling division

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(5, 1.5 * num_rows))
    
    # Flatten axes array for easier looping if there are multiple rows
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable for consistency

    # Plot each image
    for idx, image in enumerate(images):
        ax = axes[idx]
        ax.imshow(image, cmap='gray')  # Assuming grayscale for simplicity, change cmap as needed
        ax.axis('off')  # Hide axes

    # Turn off unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()


for i in im_list_dcm:
    display_images([x['dicom'].pixel_array for x in im_list_dcm[i]['images']], 
                   im_list_dcm[i]['description'])


df_coor = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')


df_coor.head()


def display_coor_on_img(c, i, title):
    center_coordinates = (int(c['x']), int(c['y']))
    radius = 10
    color = (255, 0, 0)  # Red color in BGR
    thickness = 2
    IMG = i['dicom'].pixel_array
    IMG_normalized = cv2.normalize(IMG, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    IMG_with_circle = cv2.circle(IMG_normalized.copy(), center_coordinates, radius, color, thickness)
    
    # Convert the image from BGR to RGB for correct color display in matplotlib
    IMG_with_circle = cv2.cvtColor(IMG_with_circle, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.imshow(IMG_with_circle)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.title(title)
    plt.show()



coor_entries = df_coor[df_coor['study_id'] == int(patient['study_id'])]


print("Only showing severe cases for this patient")
for idc, c in coor_entries.iterrows():
    for i in im_list_dcm[str(c['series_id'])]['images']:
        if int(i['SOPInstanceUID']) == int(c['instance_number']):
            try:
                patient_severity = patient[
                    f"{c['condition'].lower().replace(' ', '_')}_{c['level'].lower().replace('/', '_')}"
                ]
            except Exception as e:
                patient_severity = "unknown severity"
            title = f"{i['SOPInstanceUID']} \n{c['level']}, {c['condition']}: {patient_severity} \n{c['x']}, {c['y']}"
            if patient_severity == 'Severe':
                display_coor_on_img(c, i, title)


import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.patches as patches
import glob

train_label_coordinates = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')
train_series_descriptions = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')
train = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')  # already loaded earlier

def load_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    # Apply VOI LUT / windowing if present
    if "VOILUTSequence" in dicom or ("WindowCenter" in dicom and "WindowWidth" in dicom):
        img = apply_voi_lut(img, dicom)
    # Normalize to 0–1
    img = (img - img.min()) / (img.ptp() + 1e-6)
    return img

def plot_study_with_annotations(study_id=4003253):  # great example with many findings
    base_path = f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}"
    coord_df = train_label_coordinates[train_label_coordinates.study_id == study_id]
    desc = train_series_descriptions[train_series_descriptions.study_id == study_id]

    # Get one series of each type
    sag_t1 = desc[desc.series_description == "Sagittal T1"].iloc[0]
    sag_t2 = desc[desc.series_description.str.contains("Sagittal T2")].iloc[0]  # T2 or T2/STIR
    ax_t2  = desc[desc.series_description == "Axial T2"].iloc[0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # ------------------ Sagittal T1 ------------------
    files = sorted(glob.glob(f"{base_path}/{sag_t1.series_id}/*.dcm"))
    mid_slice = files[len(files)//2]
    img_t1 = load_dicom(mid_slice)
    axes[0].imshow(img_t1, cmap='gray')
    axes[0].set_title(f"Sagittal T1 – middle slice\nStudy {study_id}")
    axes[0].axis('off')

    # ------------------ Sagittal T2 (best for foraminal narrowing) ------------------
    files = sorted(glob.glob(f"{base_path}/{sag_t2.series_id}/*.dcm"))
    mid_slice = files[len(files)//2]
    img_t2 = load_dicom(mid_slice)
    axes[1].imshow(img_t2, cmap='gray')
    axes[1].set_title("Sagittal T2/STIR – middle slice\n+ Neural Foraminal annotations")
    axes[1].axis('off')

    # Draw bounding boxes for LEFT / RIGHT neural foraminal narrowing
    for _, row in coord_df.iterrows():
        if 'foraminal' in row.condition.lower():
            x, y = row.x, row.y
            color = 'red' if 'left' in row.condition.lower() else 'lime'
            rect = patches.Rectangle((x-40, y-40), 80, 80,
                                     linewidth=3, edgecolor=color, facecolor='none')
            axes[1].add_patch(rect)
            axes[1].text(x, y-50, row.level, color='yellow', fontsize=11, weight='bold')

    # ------------------ Axial T2 (best for canal & subarticular stenosis) ------------------
    files = sorted(glob.glob(f"{base_path}/{ax_t2.series_id}/*.dcm"),
                   key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber)
    # Pick a slice around L4/L5 (usually ~30–40% through the stack)
    axial_img = load_dicom(files[len(files)//3])
    axes[2].imshow(axial_img, cmap='gray')
    axes[2].set_title("Axial T2 – example slice (L4/L5 level)")
    axes[2].axis('off')


    # Empty placeholders
    for i in [3,4,5]:
        axes[i].axis('off')

    plt.suptitle(f"RSNA 2024 Lumbar Spine – Study {study_id} – Example with annotations", fontsize=18)
    plt.tight_layout()
    plt.show()

good_examples = [4003253, 464674857, 24063143, 24414502]  # hand-picked rich cases
plot_study_with_annotations(4003253)


import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import os

train = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')
coords = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')
desc = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')

def load_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    if "VOILUTSequence" in dicom or ("WindowCenter" in dicom and "WindowWidth" in dicom):
        img = apply_voi_lut(img, dicom)
    img = (img - img.min()) / (img.ptp() + 1e-6)
    return img

def plot_study_rich(study_id):
    print(f"Visualizing study {study_id} ...")
    base_path = f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}"
    if not os.path.exists(base_path):
        print("Study folder not found!")
        return
    
    study_coords = coords[coords.study_id == study_id]
    study_desc = desc[desc.study_id == study_id]
    
    try:
        sag_t1_series = study_desc[study_desc.series_description == "Sagittal T1"].iloc[0].series_id.values[0]
        sag_t2_series = study_desc[study_desc.series_description.str.contains("Sagittal T2")].iloc[0].series_id
        ax_t2_series  = study_desc[study_desc.series_description == "Axial T2"].iloc[0].series_id
    except:
        print("Missing one of the three series in this study")
        return
    
    fig, ax = plt.subplots(2, 3, figsize=(20, 14))
    ax = ax.ravel()

    # 1. Sagittal T1
    files = sorted(glob.glob(f"{base_path}/{sag_t1_series}/*.dcm"))
    img = load_dicom(files[len(files)//2])
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Sagittal T1", fontsize=15)
    ax[0].axis('off')

    # 2. Sagittal T2 + FORAMINAL boxes
    files = sorted(glob.glob(f"{base_path}/{sag_t2_series}/*.dcm"))
    img = load_dicom(files[len(files)//2])
    ax[1].imshow(img, cmap='gray')
    ax[1].set_title("Sagittal T2/STIR → Neural Foraminal Narrowing", fontsize=15)
    for _, row in study_coords.iterrows():
        if 'foraminal' in row.condition.lower():
            color = 'red' if 'left' in row.condition.lower() else 'lime'
            rect = patches.Rectangle((row.x-40, row.y-40), 80, 80,
                                     linewidth=4, edgecolor=color, facecolor='none')
            ax[1].add_patch(rect)
            ax[1].text(row.x, row.y-55, row.level, color='yellow', fontsize=12, weight='bold')
    ax[1].axis('off')

    # 3. Axial T2 + CANAL & SUBARTICULAR boxes
    files = sorted(glob.glob(f"{base_path}/{ax_t2_series}/*.dcm"),
                   key=lambda p: pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber)
    img = load_dicom(files[len(files)//3])
    ax[2].imshow(img, cmap='gray')
    ax[2].set_title("Axial T2 → Canal & Subarticular Stenosis", fontsize=15)
    for _, row in study_coords.iterrows():
        if 'canal' in row.condition.lower() or 'subarticular' in row.condition.lower():
            if 'canal' in row.condition.lower():
                color = 'yellow'
                label = 'Canal'
            else:
                color = 'magenta' if 'left' in row.condition.lower() else 'cyan'
                label = 'Subart'
            rect = patches.Rectangle((row.x-50, row.y-50), 100, 100,
                                     linewidth=3, edgecolor=color, facecolor='none')
            ax[2].add_patch(rect)
            ax[2].text(row.x, row.y-70, f"{row.level}\n{label}", color=color, fontsize=11, weight='bold')
    ax[2].axis('off')

    # Hide unused subplots
    for i in [3,4,5]:
        ax[i].axis('off')

    # Count real severity
    labels = train[train.study_id == study_id].iloc[0, 1:]  # all 25 columns
    n_severe = (labels == 'Severe').sum()
    n_moderate = (labels == 'Moderate').sum()
    plt.suptitle(f"Study {study_id}  —  Severe: {n_severe}  |  Moderate: {n_moderate}  |  Total positive: {n_severe + n_moderate}",
                 fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============== NOW FIND REALLY SEVERE CASES (the correct way) =============
severity_count = train.set_index('study_id') \
    .isin(['Severe', 'Moderate']) \
    .sum(axis=1) \
    .sort_values(ascending=False)

print("Top 20 genuinely severe studies:")
print(severity_count.head(20))

# Visualize the top 3 richest cases automatically
for rank, (study_id, count) in enumerate(severity_count.head(3).items(), 1):
    print(f"\n=== #{rank} most severe study ===")
    plot_study_rich(study_id)


import os

base_img_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"

existing_studies = set(int(x) for x in os.listdir(base_img_path) if x.isdigit())

severity_count_existing = severity_count[severity_count.index.isin(existing_studies)]

print("Top 15 most severe studies THAT HAVE IMAGES:")
print(severity_count_existing.head(15))

print("\nVisualizing the top 3 real monsters now...\n")

top3_real = severity_count_existing.head(3).index.tolist()

for i, study_id in enumerate(top3_real, 1):
    print(f"=== #{i} REAL severe case: {study_id} ({severity_count_existing.loc[study_id]} positive labels) ===")
    plot_study_rich(study_id)


import os
import cv2
import glob
import pandas as pd
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
CONFIG = {
    "img_size": 256,
    "backbone": "tf_efficientnet_b0_ns",
    "batch_size": 16,
    "lr": 1e-4,
    "epochs": 10,          # UPDATED: 10 Epochs
    "num_classes": 3,      # Normal/Mild, Moderate, Severe
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "n_slices": 3,         # 2.5D Stacking
}

# --- UTILS ---
def load_dicom(path):
    """Loads a DICOM file, normalizes, and resizes to fixed size."""
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        
        # FIX: Resize immediately to ensure all images in the stack match
        if data.shape != (CONFIG["img_size"], CONFIG["img_size"]):
            data = cv2.resize(data, (CONFIG["img_size"], CONFIG["img_size"]), interpolation=cv2.INTER_AREA)
        return data
    except Exception:
        return np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8)

def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])

# --- DATASET ---
class RSYNASpineDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row['study_id'])
        study_path = os.path.join(self.images_dir, study_id)
        series_dirs = os.listdir(study_path) if os.path.exists(study_path) else []
        
        if not series_dirs:
            image = torch.zeros((3, CONFIG["img_size"], CONFIG["img_size"]))
            label = torch.tensor(0, dtype=torch.long)
            return image, label

        series_id = series_dirs[0]
        series_path = os.path.join(study_path, series_id)
        
        files = sorted(glob.glob(f"{series_path}/*.dcm"), 
                       key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        # Select middle slices
        mid = len(files) // 2
        start = max(0, mid - CONFIG["n_slices"] // 2)
        end = min(len(files), start + CONFIG["n_slices"])
        selected_files = files[start:end]

        img_list = []
        for f in selected_files:
            img = load_dicom(f)
            img_list.append(img)
        
        # Padding
        while len(img_list) < CONFIG["n_slices"]:
            img_list.append(np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8))
            
        image = np.stack(img_list, axis=-1) 
        if self.transform:
            image = self.transform(image=image)['image']
            
        target_str = row.get('spinal_canal_stenosis_l4_l5', 'Normal/Mild')
        if pd.isna(target_str): target_str = 'Normal/Mild'
        label = torch.tensor(self.label_map.get(target_str, 0), dtype=torch.long)
        
        return image, label

# --- MODEL ---
class SpineModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(SpineModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=CONFIG["n_slices"])
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# --- TRAINING FUNCTIONS ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"Loss": loss.item()})
        
    return running_loss / len(loader), correct / total

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
    df = pd.read_csv(f'{base_path}/train.csv')

    # 2. Split Data
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    # 3. Loaders
    train_dataset = RSYNASpineDataset(train_df, f'{base_path}/train_images', transform=get_transforms("train"))
    val_dataset = RSYNASpineDataset(val_df, f'{base_path}/train_images', transform=get_transforms("valid"))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # 4. Calculate Class Weights (Fix for Imbalance)
    target_col = 'spinal_canal_stenosis_l4_l5'
    class_counts = train_df[target_col].value_counts().sort_index()
    
    # Ensure order: Normal/Mild, Moderate, Severe
    counts = np.array([class_counts.get('Normal/Mild', 0), 
                       class_counts.get('Moderate', 0), 
                       class_counts.get('Severe', 0)])
    
    counts = np.maximum(counts, 1) # Avoid div by zero
    total_samples = sum(counts)
    weights = total_samples / (len(counts) * counts)
    class_weights = torch.FloatTensor(weights).to(CONFIG["device"])
    
    print(f"Using Class Weights: {class_weights.cpu().numpy()}")

    # 5. Initialize Model
    model = SpineModel(CONFIG["backbone"], CONFIG["num_classes"]).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Apply weights
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # 6. Training Loop
    history = {'train_loss': [], 'train_acc': []}
    print(f"Starting training on {CONFIG['device']} for {CONFIG['epochs']} epochs...")

    for epoch in range(CONFIG["epochs"]):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {loss:.4f} - Acc: {acc:.4f}")

    # 7. Plotting
    print("Generating plots...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', marker='o', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save plot to avoid javascript display errors
    plt.savefig('training_history.png')
    plt.show()

    # 8. Final Evaluation
    print("Running final evaluation...")
    true_labels, pred_labels = get_predictions(model, val_loader, CONFIG["device"])
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))


import os
import cv2
import glob
import pandas as pd
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
CONFIG = {
    "img_size": 256,
    "backbone": "tf_efficientnet_b0_ns",
    "batch_size": 16,
    "lr": 3e-4,            # Slightly higher initial LR for scheduler
    "weight_decay": 1e-4,  # NEW: Regularization parameter
    "epochs": 10,
    "num_classes": 3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "n_slices": 3,
}

# --- UTILS ---
def load_dicom(path):
    """Loads a DICOM file, normalizes, and resizes to fixed size."""
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        
        if data.shape != (CONFIG["img_size"], CONFIG["img_size"]):
            data = cv2.resize(data, (CONFIG["img_size"], CONFIG["img_size"]), interpolation=cv2.INTER_AREA)
        return data
    except Exception:
        return np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8)

# --- NEW: STRONGER AUGMENTATIONS ---
def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            # Geometry
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            # Pixel Intensity
            A.RandomBrightnessContrast(p=0.5),
            # Regularization: Cut holes in image to force model to look at context
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])

# --- DATASET ---
class RSYNASpineDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row['study_id'])
        study_path = os.path.join(self.images_dir, study_id)
        series_dirs = os.listdir(study_path) if os.path.exists(study_path) else []
        
        if not series_dirs:
            image = torch.zeros((3, CONFIG["img_size"], CONFIG["img_size"]))
            label = torch.tensor(0, dtype=torch.long)
            return image, label

        series_id = series_dirs[0]
        series_path = os.path.join(study_path, series_id)
        files = sorted(glob.glob(f"{series_path}/*.dcm"), 
                       key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        mid = len(files) // 2
        start = max(0, mid - CONFIG["n_slices"] // 2)
        end = min(len(files), start + CONFIG["n_slices"])
        selected_files = files[start:end]

        img_list = []
        for f in selected_files:
            img = load_dicom(f)
            img_list.append(img)
        
        while len(img_list) < CONFIG["n_slices"]:
            img_list.append(np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8))
            
        image = np.stack(img_list, axis=-1) 
        if self.transform:
            image = self.transform(image=image)['image']
            
        target_str = row.get('spinal_canal_stenosis_l4_l5', 'Normal/Mild')
        if pd.isna(target_str): target_str = 'Normal/Mild'
        label = torch.tensor(self.label_map.get(target_str, 0), dtype=torch.long)
        
        return image, label

# --- MODEL WITH DROPOUT ---
class SpineModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(SpineModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=CONFIG["n_slices"])
        in_features = self.backbone.classifier.in_features
        
        # NEW: Add Dropout before the final layer
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# --- TRAINING FUNCTIONS ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"Loss": loss.item()})
        
    return running_loss / len(loader), correct / total

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

# --- MAIN ---
if __name__ == "__main__":
    # 1. Load Data
    base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
    df = pd.read_csv(f'{base_path}/train.csv')

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    # 2. Loaders
    train_dataset = RSYNASpineDataset(train_df, f'{base_path}/train_images', transform=get_transforms("train"))
    val_dataset = RSYNASpineDataset(val_df, f'{base_path}/train_images', transform=get_transforms("valid"))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # 3. Class Weights
    target_col = 'spinal_canal_stenosis_l4_l5'
    class_counts = train_df[target_col].value_counts().sort_index()
    counts = np.array([class_counts.get('Normal/Mild', 0), 
                       class_counts.get('Moderate', 0), 
                       class_counts.get('Severe', 0)])
    counts = np.maximum(counts, 1)
    total_samples = sum(counts)
    weights = total_samples / (len(counts) * counts)
    class_weights = torch.FloatTensor(weights).to(CONFIG["device"])
    print(f"Using Class Weights: {class_weights.cpu().numpy()}")

    # 4. Initialize Model & Optimizer
    model = SpineModel(CONFIG["backbone"], CONFIG["num_classes"]).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # NEW: Add weight_decay
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
    # NEW: Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

    # 5. Training Loop
    history = {'train_loss': [], 'train_acc': []}
    print(f"Starting training on {CONFIG['device']} for {CONFIG['epochs']} epochs...")

    for epoch in range(CONFIG["epochs"]):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        
        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {loss:.4f} - Acc: {acc:.4f} - LR: {current_lr:.6f}")

    # 6. Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', marker='o', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_history_path_a.png')
    plt.show()

    # 7. Evaluation
    print("Running final evaluation...")
    true_labels, pred_labels = get_predictions(model, val_loader, CONFIG["device"])
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_path_a.png')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))


import os
import cv2
import glob
import pandas as pd
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
CONFIG = {
    "img_size": 256,
    "backbone": "tf_efficientnet_b0_ns",
    "batch_size": 16,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 10,
    "num_classes": 3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "n_slices": 3,
}

# --- UTILS ---
def load_dicom(path):
    """Loads a DICOM file, normalizes, and resizes to fixed size."""
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        
        if data.shape != (CONFIG["img_size"], CONFIG["img_size"]):
            data = cv2.resize(data, (CONFIG["img_size"], CONFIG["img_size"]), interpolation=cv2.INTER_AREA)
        return data
    except Exception:
        return np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8)

def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])

# --- DATASET & MODEL (Unchanged) ---
class RSYNASpineDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row['study_id'])
        study_path = os.path.join(self.images_dir, study_id)
        series_dirs = os.listdir(study_path) if os.path.exists(study_path) else []
        
        if not series_dirs:
            image = torch.zeros((3, CONFIG["img_size"], CONFIG["img_size"]))
            label = torch.tensor(0, dtype=torch.long)
            return image, label

        series_id = series_dirs[0]
        series_path = os.path.join(study_path, series_id)
        files = sorted(glob.glob(f"{series_path}/*.dcm"), 
                       key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        mid = len(files) // 2
        start = max(0, mid - CONFIG["n_slices"] // 2)
        end = min(len(files), start + CONFIG["n_slices"])
        selected_files = files[start:end]

        img_list = []
        for f in selected_files:
            img = load_dicom(f)
            img_list.append(img)
        
        while len(img_list) < CONFIG["n_slices"]:
            img_list.append(np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8))
            
        image = np.stack(img_list, axis=-1) 
        if self.transform:
            image = self.transform(image=image)['image']
            
        target_str = row.get('spinal_canal_stenosis_l4_l5', 'Normal/Mild')
        if pd.isna(target_str): target_str = 'Normal/Mild'
        label = torch.tensor(self.label_map.get(target_str, 0), dtype=torch.long)
        
        return image, label

class SpineModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(SpineModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=CONFIG["n_slices"])
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"Loss": loss.item()})
        
    return running_loss / len(loader), correct / total

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
    df = pd.read_csv(f'{base_path}/train.csv')

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    # 2. Setup Label Mapping
    target_col = 'spinal_canal_stenosis_l4_l5'
    label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    
    # Map targets safely, filling NaNs (missing labels) with 'Normal/Mild' (0) 
    temp_targets = train_df[target_col].fillna('Normal/Mild').map(label_map)
    targets = temp_targets.values.astype(int) # Ensure targets is a clean integer array
    
    # 3. Calculate Class Weights (for Loss)
    class_counts = train_df[target_col].value_counts().sort_index()
    counts = np.array([class_counts.get('Normal/Mild', 0), 
                       class_counts.get('Moderate', 0), 
                       class_counts.get('Severe', 0)])
    counts = np.maximum(counts, 1)
    total_samples = sum(counts)
    weights = total_samples / (len(counts) * counts)
    class_weights = torch.FloatTensor(weights).to(CONFIG["device"])
    print(f"Loss Class Weights: {class_weights.cpu().numpy()}")

    # 4. Setup Sampler (FIXED: Uses clean integer targets for indexing)
    # Calculate inverse weights for ALL classes (0, 1, 2)
    label_indices = [0, 1, 2]
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in label_indices])
    
    # Calculate sample weight: inverse frequency
    sample_weight_map = 1. / class_sample_count
    sample_weight_map[class_sample_count == 0] = 0 # Set weight to 0 if class is missing
    
    # Map the clean integer targets array to the calculated weights
    samples_weight = sample_weight_map[targets]
    samples_weight = torch.from_numpy(samples_weight).double()

    # Create the sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight, 
        len(samples_weight), 
        replacement=True
    )

    # 5. Loaders (Use Sampler for Training Loader)
    train_dataset = RSYNASpineDataset(train_df, f'{base_path}/train_images', transform=get_transforms("train"))
    val_dataset = RSYNASpineDataset(val_df, f'{base_path}/train_images', transform=get_transforms("valid"))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        sampler=sampler, 
        num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # 6. Initialize Model
    model = SpineModel(CONFIG["backbone"], CONFIG["num_classes"]).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

    # 7. Training Loop
    history = {'train_loss': [], 'train_acc': []}
    print(f"Starting training with Weighted Sampler and Regularization on {CONFIG['device']} for {CONFIG['epochs']} epochs...")

    for epoch in range(CONFIG["epochs"]):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {loss:.4f} - Acc: {acc:.4f} - LR: {current_lr:.6f}")

    # 8. Plotting & Evaluation
    print("\nRunning final evaluation...")
    true_labels, pred_labels = get_predictions(model, val_loader, CONFIG["device"])
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_resample_fixed.png')
    plt.show()

    # Report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))


import os
import cv2
import glob
import pandas as pd
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- FULL REQUIRED TARGET DEFINITION (Used for filtering) ---
# Spinal Canal Stenosis is assessed at all 5 levels
SPINAL_CANAL_TARGETS = [f'spinal_canal_stenosis_{level}' for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']]

# Neural/Subarticular Stenosis are assessed only at 4 levels (L2-L3 to L5-S1)
FOUR_LEVEL_CONDITIONS = [
    'left_neural_foraminal_stenosis', 
    'right_neural_foraminal_stenosis', 
    'left_subarticular_stenosis', 
    'right_subarticular_stenosis'
]
FOUR_LEVEL_TARGETS = [f'{cond}_{level}' 
                      for cond in FOUR_LEVEL_CONDITIONS 
                      for level in ['l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']]

# Theoretical total of 21 columns
FULL_TARGET_COLUMNS = SPINAL_CANAL_TARGETS + FOUR_LEVEL_TARGETS

# --- CONFIGURATION (Initial values, updated later) ---
CONFIG = {
    "img_size": 256,
    "backbone": "tf_efficientnet_b0_ns",
    "batch_size": 16,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 10,
    "num_classes_per_task": 3, 
    "output_size": 0, # Placeholder, will be updated dynamically
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "n_slices": 3,
}
# Global placeholder for the *actual* target columns
TARGET_COLUMNS = []


# --- UTILS (Unchanged) ---
def load_dicom(path):
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        
        if data.shape != (CONFIG["img_size"], CONFIG["img_size"]):
            data = cv2.resize(data, (CONFIG["img_size"], CONFIG["img_size"]), interpolation=cv2.INTER_AREA)
        return data
    except Exception:
        return np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8)

def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])

# --- DATASET ---
class MultiTaskSpineDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        self.num_targets = len(TARGET_COLUMNS) 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row['study_id'])
        study_path = os.path.join(self.images_dir, study_id)
        series_dirs = os.listdir(study_path) if os.path.exists(study_path) else []
        
        # --- Image Loading ---
        if not series_dirs:
            image = torch.zeros((CONFIG["n_slices"], CONFIG["img_size"], CONFIG["img_size"]))
        else:
            series_id = series_dirs[0]
            series_path = os.path.join(study_path, series_id)
            files = sorted(glob.glob(f"{series_path}/*.dcm"), key=lambda x: int(x.split('/')[-1].split('.')[0]))
            
            mid = len(files) // 2
            start = max(0, mid - CONFIG["n_slices"] // 2)
            end = min(len(files), start + CONFIG["n_slices"])
            selected_files = files[start:end]

            img_list = []
            for f in selected_files:
                img = load_dicom(f)
                img_list.append(img)
            
            while len(img_list) < CONFIG["n_slices"]:
                img_list.append(np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8))
                
            image = np.stack(img_list, axis=-1) 
            if self.transform:
                image = self.transform(image=image)['image']
        
        # --- MULTI-TASK TARGET LOADING ---
        # **Note**: Must use .replace() in main block; using .map() in getitem for safety on a series row
        labels_series = row[TARGET_COLUMNS].fillna('Normal/Mild').map(self.label_map)
        labels_array = labels_series.values.astype(int)
        
        target = torch.tensor(labels_array, dtype=torch.long)
        
        return image, target

# --- MODEL (Uses corrected output_size) ---
class MultiTaskSpineModel(nn.Module):
    def __init__(self, model_name, output_size, pretrained=True):
        super(MultiTaskSpineModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=CONFIG["n_slices"])
        in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, output_size)
        )

    def forward(self, x):
        return self.backbone(x)

# --- TRAINING FUNCTIONS ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    num_tasks = len(TARGET_COLUMNS) 
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        
        # Flatten targets for loss function: [N, num_tasks] -> [N*num_tasks]
        labels_flat = labels.view(-1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images) 
        
        # Reshape outputs for loss: [N, num_tasks*3] -> [N*num_tasks, 3]
        outputs_reshaped = outputs.view(-1, CONFIG["num_classes_per_task"])
        
        loss = criterion(outputs_reshaped, labels_flat)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy based on flattened predictions
        _, predicted_flat = torch.max(outputs_reshaped.data, 1)
        correct_predictions += (predicted_flat == labels_flat).sum().item()
        total_predictions += labels_flat.size(0)
        
        acc_batch = correct_predictions / total_predictions
        pbar.set_postfix({"Loss": loss.item(), "Acc_Flat": acc_batch})
        
    return running_loss / len(loader), correct_predictions / total_predictions

def get_predictions(model, loader, device):
    model.eval()
    all_preds_flat = []
    all_labels_flat = []
    num_tasks = len(TARGET_COLUMNS) 
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images) 
            outputs_reshaped = outputs.view(-1, CONFIG["num_classes_per_task"]) 
            _, predicted_flat = torch.max(outputs_reshaped, 1)

            all_preds_flat.extend(predicted_flat.cpu().numpy())
            all_labels_flat.extend(labels.view(-1).cpu().numpy())
            
    return np.array(all_labels_flat), np.array(all_preds_flat)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
    df = pd.read_csv(f'{base_path}/train.csv')

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # 2. Dynamic Target Column Filtering and CONFIG Update
    global TARGET_COLUMNS
    actual_df_columns = train_df.columns.tolist()
    TARGET_COLUMNS = [col for col in FULL_TARGET_COLUMNS if col in actual_df_columns]
    
    NUM_TASKS = len(TARGET_COLUMNS)
    CONFIG["output_size"] = NUM_TASKS * CONFIG["num_classes_per_task"]

    print(f"Detected {NUM_TASKS} target columns in the CSV. Output layer size set to {CONFIG['output_size']}.")
    
    label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

    # 3. Prepare Multi-Task Targets for Weights Calculation (FIXED: Using .replace())
    # Fill NaNs first, then replace string labels with integers across the DataFrame
    # This avoids the TypeError encountered with .map() on a DataFrame
    all_targets_train = train_df[TARGET_COLUMNS].fillna('Normal/Mild').replace(label_map).values.flatten().astype(int)
    
    # 4. Calculate Class Weights (for Loss)
    unique_labels, counts = np.unique(all_targets_train, return_counts=True)
    full_counts = np.zeros(3)
    full_counts[unique_labels] = counts
    
    counts = np.maximum(full_counts, 1)
    total_samples = np.sum(counts)
    weights = total_samples / (len(counts) * counts)
    class_weights = torch.FloatTensor(weights).to(CONFIG["device"])
    print(f"Loss Class Weights (Across all {NUM_TASKS} tasks): {class_weights.cpu().numpy()}")

    # 5. Loaders 
    train_loader = DataLoader(
        MultiTaskSpineDataset(train_df, f'{base_path}/train_images', transform=get_transforms("train")), 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        MultiTaskSpineDataset(val_df, f'{base_path}/train_images', transform=get_transforms("valid")), 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=2
    )

    # 6. Initialize Model
    model = MultiTaskSpineModel(CONFIG["backbone"], CONFIG["output_size"]).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

    # 7. Training Loop
    history = {'train_loss': [], 'train_acc': []}
    print(f"Starting FINAL Multi-Task Training on {CONFIG['device']} for {CONFIG['epochs']} epochs...")

    for epoch in range(CONFIG["epochs"]):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {loss:.4f} - Flat Acc: {acc:.4f} - LR: {current_lr:.6f}")

    # 8. Plotting & Evaluation
    print("\nRunning final Multi-Task evaluation...")
    true_labels_flat, pred_labels_flat = get_predictions(model, val_loader, CONFIG["device"])
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    # Classification Report
    print("\nClassification Report (Flattened over all tasks):")
    print(classification_report(true_labels_flat, pred_labels_flat, target_names=class_names))

    # Confusion Matrix (Flattened)
    cm = confusion_matrix(true_labels_flat, pred_labels_flat)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Flattened over {NUM_TASKS} tasks)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_multi_task_dynamic.png')
    plt.show()


import os
import cv2
import glob
import pandas as pd
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

SPINAL_CANAL_TARGETS = [f'spinal_canal_stenosis_{level}' for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']]
FOUR_LEVEL_CONDITIONS = [
    'left_neural_foraminal_stenosis', 
    'right_neural_foraminal_stenosis', 
    'left_subarticular_stenosis', 
    'right_subarticular_stenosis'
]
FOUR_LEVEL_TARGETS = [f'{cond}_{level}' 
                      for cond in FOUR_LEVEL_CONDITIONS 
                      for level in ['l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']]
FULL_TARGET_COLUMNS = SPINAL_CANAL_TARGETS + FOUR_LEVEL_TARGETS

CONFIG = {
    "img_size": 224,
    "backbone_eff": "efficientnet_b0",
    "backbone_vgg": "vgg19",
    "batch_size": 16,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 10,
    "num_classes_per_task": 3, 
    "output_size": 0, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "n_slices": 3,
}
TARGET_COLUMNS = []

def load_dicom(path):
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        
        if data.shape != (CONFIG["img_size"], CONFIG["img_size"]):
            data = cv2.resize(data, (CONFIG["img_size"], CONFIG["img_size"]), interpolation=cv2.INTER_AREA)
        return data
    except Exception:
        return np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8)

def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])

class MultiTaskSpineDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        self.num_targets = len(TARGET_COLUMNS) 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row['study_id'])
        study_path = os.path.join(self.images_dir, study_id)
        series_dirs = os.listdir(study_path) if os.path.exists(study_path) else []
        
        if not series_dirs:
            image = torch.zeros((CONFIG["n_slices"], CONFIG["img_size"], CONFIG["img_size"]))
        else:
            series_id = series_dirs[0]
            series_path = os.path.join(study_path, series_id)
            files = sorted(glob.glob(f"{series_path}/*.dcm"), key=lambda x: int(x.split('/')[-1].split('.')[0]))
            
            mid = len(files) // 2
            start = max(0, mid - CONFIG["n_slices"] // 2)
            end = min(len(files), start + CONFIG["n_slices"])
            selected_files = files[start:end]

            img_list = []
            for f in selected_files:
                img = load_dicom(f)
                img_list.append(img)
            
            while len(img_list) < CONFIG["n_slices"]:
                img_list.append(np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8))
                
            image = np.stack(img_list, axis=-1) 
            if self.transform:
                image = self.transform(image=image)['image']
        
        labels_series = row[TARGET_COLUMNS].fillna('Normal/Mild').map(self.label_map)
        labels_array = labels_series.values.astype(int)
        
        target = torch.tensor(labels_array, dtype=torch.long)
        
        return image, target

class PseudoNewtonBlock(nn.Module):
    def __init__(self, in_features, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HybridSpineModel(nn.Module):
    def __init__(self, output_size, pretrained=True):
        super().__init__()
        
        # Load backbones to return feature lists
        self.eff_net = timm.create_model(CONFIG["backbone_eff"], pretrained=pretrained, in_chans=CONFIG["n_slices"], features_only=True)
        self.vgg = timm.create_model(CONFIG["backbone_vgg"], pretrained=pretrained, in_chans=CONFIG["n_slices"], features_only=True)
        
        # Get feature dimensions from the last stage of each backbone
        eff_features = self.eff_net.feature_info[-1]['num_chs']
        
        # VGG19 last feature stage is 512 channels
        vgg_features = 512 

        # Global average pooling layers for the last feature map of each backbone
        self.eff_pool = nn.AdaptiveAvgPool2d(1)
        self.vgg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fusion_features = eff_features + vgg_features
        
        self.pseudo_newton = PseudoNewtonBlock(self.fusion_features)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.fusion_features, self.fusion_features // 2),
            nn.ReLU(),
            nn.Linear(self.fusion_features // 2, output_size)
        )

    def forward(self, x):
        # EfficientNet features (returns list, take last stage)
        eff_feats = self.eff_net(x)[-1]
        
        # VGG19 features (returns list, take last stage)
        # FIX: Call the model instance itself, not forward_features
        vgg_feats = self.vgg(x)[-1]
        
        # Global pooling and flattening
        eff_pooled = self.eff_pool(eff_feats).flatten(1)
        vgg_pooled = self.vgg_pool(vgg_feats).flatten(1)
        
        # Feature Fusion (Concatenation)
        combined_features = torch.cat((eff_pooled, vgg_pooled), dim=1)
        
        # Pseudo-Newton Block: requires input to be [B, C, 1, 1] for the 2D pooling/conv structure
        reshaped_features = combined_features.unsqueeze(-1).unsqueeze(-1)
        refined_features = self.pseudo_newton(reshaped_features).flatten(1)
        
        output = self.classifier(refined_features)
        return output

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    num_tasks = len(TARGET_COLUMNS) 
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        
        labels_flat = labels.view(-1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images) 
        
        outputs_reshaped = outputs.view(-1, CONFIG["num_classes_per_task"])
        
        loss = criterion(outputs_reshaped, labels_flat)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted_flat = torch.max(outputs_reshaped.data, 1)
        correct_predictions += (predicted_flat == labels_flat).sum().item()
        total_predictions += labels_flat.size(0)
        
        acc_batch = correct_predictions / total_predictions
        pbar.set_postfix({"Loss": loss.item(), "Flat Acc": acc_batch})
        
    return running_loss / len(loader), correct_predictions / total_predictions

def get_predictions(model, loader, device):
    model.eval()
    all_preds_flat = []
    all_labels_flat = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images) 
            outputs_reshaped = outputs.view(-1, CONFIG["num_classes_per_task"]) 
            _, predicted_flat = torch.max(outputs_reshaped, 1)

            all_preds_flat.extend(predicted_flat.cpu().numpy())
            all_labels_flat.extend(labels.view(-1).cpu().numpy())
            
    return np.array(all_labels_flat), np.array(all_preds_flat)

if __name__ == "__main__":
    base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
    df = pd.read_csv(f'{base_path}/train.csv')

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    global TARGET_COLUMNS
    actual_df_columns = train_df.columns.tolist()
    TARGET_COLUMNS = [col for col in FULL_TARGET_COLUMNS if col in actual_df_columns]
    
    NUM_TASKS = len(TARGET_COLUMNS)
    CONFIG["output_size"] = NUM_TASKS * CONFIG["num_classes_per_task"]

    print(f"Detected {NUM_TASKS} target columns. Output layer size: {CONFIG['output_size']}.")
    
    label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

    all_targets_train = train_df[TARGET_COLUMNS].fillna('Normal/Mild').replace(label_map).values.flatten().astype(int)
    
    unique_labels, counts = np.unique(all_targets_train, return_counts=True)
    full_counts = np.zeros(3)
    full_counts[unique_labels] = counts
    
    counts = np.maximum(full_counts, 1)
    total_samples = np.sum(counts)
    weights = total_samples / (len(counts) * counts)
    class_weights = torch.FloatTensor(weights).to(CONFIG["device"])
    print(f"Loss Class Weights: {class_weights.cpu().numpy()}")

    train_loader = DataLoader(
        MultiTaskSpineDataset(train_df, f'{base_path}/train_images', transform=get_transforms("train")), 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        MultiTaskSpineDataset(val_df, f'{base_path}/train_images', transform=get_transforms("valid")), 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=2
    )

    model = HybridSpineModel(CONFIG["output_size"]).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

    history = {'train_loss': [], 'train_acc': []}
    print(f"Starting Hybrid CNN (VGG19+EfficientNet) Training on {CONFIG['device']} for {CONFIG['epochs']} epochs...")

    for epoch in range(CONFIG["epochs"]):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {loss:.4f} - Flat Acc: {acc:.4f} - LR: {current_lr:.6f}")

    print("\nRunning final evaluation...")
    true_labels_flat, pred_labels_flat = get_predictions(model, val_loader, CONFIG["device"])
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    print("\nClassification Report (Flattened over all tasks):")
    print(classification_report(true_labels_flat, pred_labels_flat, target_names=class_names))

    cm = confusion_matrix(true_labels_flat, pred_labels_flat)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Flattened over {NUM_TASKS} tasks)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_hybrid_cnn.png')
    plt.show()

