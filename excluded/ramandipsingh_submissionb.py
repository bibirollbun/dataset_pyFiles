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
# IMPORTANT: Update this path to saved model file
MODEL_PATH = '/kaggle/input/grand-x-ray-slam-division-b/best_model.pth' 

# Paths for the competition's test data
TEST_DIR = '/kaggle/input/grand-xray-slam-division-b/test2'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/grand-xray-slam-division-b/sample_submission_2.csv'

# Model settings (
MODEL_NAME = 'efficientnet_b0'
IMAGE_SIZE = 256
BATCH_SIZE = 64 # Use a larger batch size for faster inference
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]



# --- 2. Load the Trained Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Re-create the model architecture
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(LABELS))

# Load your trained weights, mapping them to the correct device
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode (very important!)



# --- 3. Create Test Dataset and DataLoader ---
# This is a simplified dataset for test images, as they have no labels
class TestXRayDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)

# Use the same normalization as your validation set
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get list of test image paths
test_image_paths = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)]

test_dataset = TestXRayDataset(test_image_paths, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)



# --- 4. Generate Predictions ---
all_preds = []
all_img_names = []

print("Starting inference...")
with torch.no_grad(): # Disable gradient calculation for speed
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        # Use sigmoid to convert model outputs to probabilities (0 to 1)
        preds = torch.sigmoid(outputs)
        
        all_preds.append(preds.cpu().numpy())
        all_img_names.extend(img_names)

all_preds = np.vstack(all_preds)
print("Inference complete.")


# --- 5. Create submission.csv File ---
submission_df = pd.DataFrame(all_preds, columns=LABELS)
submission_df.insert(0, 'Image_name', all_img_names)

# Ensure the order of rows matches the sample submission file
sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
submission_df = submission_df.set_index('Image_name').loc[sample_df['Image_name']].reset_index()

# Save the final submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(submission_df.head())

