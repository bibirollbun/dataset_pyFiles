!pip install dicom2nifti


import os

# List all files in the input directory and subdirectories
input_dir = '/kaggle/input'
file_paths = []

for dirname, _, filenames in os.walk(input_dir):
    for filename in filenames:
        file_paths.append(os.path.join(dirname, filename))

# Print confirmation message
if file_paths:
    print("Input added successfully.")
else:
    print("No input files found.")


import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as models
import dicom2nifti
import tempfile
import shutil
from pathlib import Path


MRI_dcm_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
nii_out_path = "/kaggle/working/nifti_data"


def dcm2nii_all_sequences(patient_folder, nii_out_path):
    """
    Converts DICOM images from FLAIR, T1wCE, and T2w sequences to NIfTI format.

    - patient_folder: Path to the directory containing patient subfolders (FLAIR, T1wCE, T2w)
    - nii_out_path: Path where NIfTI files should be saved
    """
    sequences = ["T1w", "FLAIR", "T1wCE", "T2w"]  # Define the required sequences

    for seq in sequences:
        mri_dcm_path = os.path.join(patient_folder, seq)  # Path to the DICOM files of a sequence
        seq_out_path = os.path.join(nii_out_path, f"{seq}.nii.gz")  # Output file path

        if not os.path.exists(mri_dcm_path):  # Check if the sequence folder exists
            print(f"Skipping {seq}: Folder not found in {patient_folder}")
            continue

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(str(tmp))

                # Convert DICOM directory to NIfTI
                dicom2nifti.convert_directory(mri_dcm_path, str(tmp),
                                              compression=True, reorient=True)

                # Find the generated NIfTI file
                nii_file = next(tmp.glob('*nii.gz'))

                # Ensure the output directory exists
                os.makedirs(nii_out_path, exist_ok=True)

                # Copy and rename the NIfTI file
                shutil.copy(nii_file, seq_out_path)
                print(f"Saved {seq_out_path}")
        except Exception as e:
            print(f"Error converting {seq} for patient {os.path.basename(patient_folder)}: {e}")


def dcm2nii_all_patients(mri_data_folder, nii_output_folder):
    """
    Converts all patient DICOM images from FLAIR, T1wCE, and T2w sequences to NIfTI format.

    - mri_data_folder: Path to the main directory containing patient subfolders
    - nii_output_folder: Path where NIfTI files should be saved
    """
    if not os.path.exists(mri_data_folder):
        print(f"Error: MRI data folder '{mri_data_folder}' not found.")
        return
    
    os.makedirs(nii_output_folder, exist_ok=True)

    for patient_id in os.listdir(mri_data_folder):
        patient_folder = os.path.join(mri_data_folder, patient_id)
        patient_out_path = os.path.join(nii_output_folder, patient_id)
        
        if not os.path.isdir(patient_folder):
            continue  # Skip if not a directory

        os.makedirs(patient_out_path, exist_ok=True)
        print(f"Processing patient: {patient_id}...")
        dcm2nii_all_sequences(patient_folder, patient_out_path)


dcm2nii_all_patients(MRI_dcm_path, nii_out_path)


import os

# List all files and folders in /kaggle/working
print(os.listdir("/kaggle/working/nifti_data/00688"))



import os
import shutil

# Define paths
source_root = "/kaggle/working/nifti_data"  # The main folder containing patient subfolders
destination_root = "/kaggle/working/FLAIR_data"  # New folder for storing only FLAIR images

# Ensure the destination directory exists
os.makedirs(destination_root, exist_ok=True)

# Iterate through patient subfolders
for patient_folder in sorted(os.listdir(source_root)):  # Sorting for consistency
    patient_path = os.path.join(source_root, patient_folder)
    
    # Ensure it's a directory
    if os.path.isdir(patient_path):
        flair_path = os.path.join(patient_path, "FLAIR.nii.gz")
        
        # Debug: Print expected file path
        print(f"Checking: {flair_path}")
        
        # Check if FLAIR image exists before copying
        if os.path.exists(flair_path):
            destination_path = os.path.join(destination_root, f"{patient_folder}_T2w.nii.gz")
            shutil.copy(flair_path, destination_path)
            print(f"Copied: {flair_path} -> {destination_path}")
        else:
            print(f"Missing T2w image in: {patient_folder}")

print("FLair image extraction completed.")



import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# ---------------------------
# ğŸ“Œ Custom Dataset Class
# ---------------------------
class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, csv_file=None, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        if not is_test:
            self.data = pd.read_csv(csv_file)
        else:
            self.data = pd.DataFrame({'BraTS21ID': sorted([int(f[:5]) for f in os.listdir(image_dir) if f.endswith('_FLAIR.nii.gz')])})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        brats_id = self.data.iloc[idx, 0]  # BraTS21ID
        img_path = os.path.join(self.image_dir, f"{brats_id:05d}_FLAIR.nii.gz")

        try:
            image = nib.load(img_path).get_fdata()
            image = np.nan_to_num(image, nan=0.0)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

            if self.transform:
                image = self.transform(image)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros((1, 64, 128, 128), dtype=torch.float32)

        if self.is_test:
            return image, brats_id
        else:
            label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)  # MGMT_value (0 or 1)
            return image, label

# ---------------------------
# ğŸ“Œ Data Transformations
# ---------------------------
fixed_size = (64, 128, 128)
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.unsqueeze(0) if img.ndimension() == 3 else img),
    transforms.Lambda(lambda img: F.interpolate(img.unsqueeze(0), size=fixed_size, mode="trilinear", align_corners=False).squeeze(0)),
    transforms.Lambda(lambda img: torch.nan_to_num(img, nan=0.0)),
    transforms.Lambda(lambda img: (img - img.min()) / (img.max() - img.min() + 1e-8)),  # Min-Max Normalize
    transforms.Lambda(lambda img: (img - 0.5) / 0.5),  # Scale to [-1,1]
    transforms.Lambda(lambda img: torch.clamp(img, min=-1.0, max=1.0))  # Prevent extreme values
])

# ---------------------------
# ğŸ“Œ Define 3D ResNet Model
# ---------------------------
class BrainTumorResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super(BrainTumorResNet3D, self).__init__()
        self.model = models.video.r3d_18(pretrained=True)
        self.model.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=True
        )
        self.bn = nn.BatchNorm3d(64)
        self.dropout = nn.Dropout3d(0.3)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model.stem[0](x)
        x = self.bn(x)
        x = self.model.stem[1](x)
        x = self.model.stem[2](x)
        x = self.model.layer1(x)
        x = self.dropout(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

# ---------------------------
# ğŸ“Œ Training Setup
# ---------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BrainTumorResNet3D(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ---------------------------
# ğŸ“Œ Data Loaders (Fixing 'data_loader' error)
# ---------------------------
train_image_dir = '/kaggle/input/train/images'
csv_file = '/kaggle/input/train_labels.csv'

train_dataset = BrainTumorDataset(train_image_dir, csv_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

# ---------------------------
# ğŸ“Œ Train the Model
# ---------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            print("NaN loss detected. Skipping batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(100.0 * correct / total):.2f}%")

    return running_loss / len(loader), 100.0 * correct / total

num_epochs = 30  
best_loss = float('inf')

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step(train_loss)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("âœ… Model Saved!")

print("ğŸ�‰ Training Complete!")

# ---------------------------
# ğŸ“Œ Generate Submission File
# ---------------------------
test_image_dir = '/kaggle/input/test/images'
test_dataset = BrainTumorDataset(test_image_dir, transform=transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

submission = []

with torch.no_grad():
    for images, brats_id in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        for id, prob in zip(brats_id, probabilities):
            submission.append({'BraTS21ID': int(id), 'MGMT_value': prob})

submission_df = pd.DataFrame(submission)
submission_df.to_csv('submission.csv', index=False)

print("ğŸ“� Submission file 'submission.csv' generated successfully!")





