import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import timm



# --- 1. Configuration ---
# Paths for the test data and saved model
TEST_DIR = '/kaggle/input/grand-xray-slam-division-a/test1'
MODEL_PATH = '//kaggle/input/grand-x-ray-slam-division-a/best_model.pth' 
SUBMISSION_PATH = '/kaggle/working/submission.csv'

# Model settings 
MODEL_NAME = 'efficientnet_b0'
IMAGE_SIZE = 256
BATCH_SIZE = 64 # Can be larger for inference
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]


# --- 2. Load the Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-create the model architecture
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(LABELS))

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval() # Set model to evaluation mode


# --- 3. Create Test Dataset and DataLoader ---
# Simpler dataset class for test images (no labels)
class TestXRayDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)

# Use the same transforms as validation set
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = TestXRayDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



# --- 4. Generate Predictions ---
all_preds = []
all_img_names = []

with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        # Use sigmoid to get probabilities between 0 and 1
        preds = torch.sigmoid(outputs)
        
        all_preds.append(preds.cpu().numpy())
        all_img_names.extend(img_names)

all_preds = np.vstack(all_preds)


# --- 5. Create submission.csv File ---
submission_df = pd.DataFrame(all_preds, columns=LABELS)
submission_df.insert(0, 'Image_name', all_img_names) # Use the correct image ID column name

# Save the submission file
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission file created at: {SUBMISSION_PATH}")
print(submission_df.head())

