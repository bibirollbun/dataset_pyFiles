# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# --- 1. Configuration for this Independent Test Script ---
class Config:
    # --- Dataset Paths (pointing to your Kaggle data) ---
    DATA_ROOT = '/kaggle/input/cassava-leaf-disease-classification' 
    TEST_CSV = os.path.join(DATA_ROOT, 'sample_submission.csv') # Path to sample_submission.csv for test image IDs
    TEST_IMAGES_DIR = os.path.join(DATA_ROOT, 'test_images') # Path to test_images folder

    # --- Model Loading Path ---
    # This is the path to your pre-trained ASDA model weights
    # IMPORTANT: If you uploaded your model as a separate Kaggle dataset,
    # adjust this path accordingly, e.g., '/kaggle/input/your-model-dataset-name/resnet50_asda_best_model.pth'
    BEST_MODEL_ASDA_PATH = os.path.join("/kaggle/input/resnetasda2/pytorch/default/1", "resnet50_asda_best_model.pth") 

    # --- Inference Parameters ---
    IMAGE_SIZE = 384
    BATCH_SIZE_INFERENCE = 64 # Larger batch size for efficient inference
    NUM_CLASSES = 5 # Number of disease classes (0, 1, 2, 3, 4)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Feature Map Sizes for ASDA (must match what model was trained with) ---
    # These sizes are crucial for correctly rebuilding the ASDA module in your classifier
    RESNET_FM_SIZES = {
        'layer1': (96, 96), 
        'layer2': (48, 48), 
        'layer3': (24, 24), 
        'layer4': (12, 12)
    }

print(f"Using device: {Config.DEVICE}")
print(f"Loading model weights from: {Config.BEST_MODEL_ASDA_PATH}")
print(f"Loading test images from: {Config.TEST_IMAGES_DIR}")

# --- 2. Custom Dataset for Test (Reads from TEST_IMAGES_DIR) ---
class TestDataset(Dataset): # Renamed for clarity in this independent script
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = image_ids # image_ids is expected to be a list here
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_name # Return image tensor and its ID for submission

# --- 3. Data Transforms for Inference ---
# Use transforms consistent with validation/test phase during training
inference_transforms = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet means and stds
])

# --- 4. ASDA Module Definition (MUST EXACTLY MATCH TRAINED MODEL'S DEFINITION) ---
class ASDA(nn.Module):
    def __init__(self, channel, input_H, input_W, reduction_ratio=4):
        super(ASDA, self).__init__()
        self.input_H = input_H
        self.input_W = input_W
        self.conv_3x3 = nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1, bias=False)
        self.conv_5x5 = nn.Conv2d(channel, channel // 2, kernel_size=5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1_reduce = nn.Conv2d(channel, 1, kernel_size=1, bias=False) 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        self.fc_spatial1 = nn.Linear(4 * 4, (4 * 4) // reduction_ratio, bias=False)
        self.fc_spatial2 = nn.Linear((4 * 4) // reduction_ratio, input_H * input_W, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        f_3x3 = self.relu(self.conv_3x3(x))
        f_5x5 = self.relu(self.conv_5x5(x))
        f_local = torch.cat([f_3x3, f_5x5], dim=1) 
        f_spatial_pre = self.conv_1x1_reduce(f_local) 
        f_pooled = self.adaptive_pool(f_spatial_pre) 
        f_pooled = f_pooled.view(b, -1) 
        f_linear = self.relu(self.fc_spatial1(f_pooled))
        spatial_weights = self.fc_spatial2(f_linear).view(b, 1, self.input_H, self.input_W) 
        spatial_weights = self.sigmoid(spatial_weights)
        return x * spatial_weights.expand_as(x)

# --- 5. CCIA Module Definition (MUST EXACTLY MATCH TRAINED MODEL'S DEFINITION if used) ---
# This class needs to be defined even if the loaded model only uses ASDA, 
# because ResNet50_Classifier might reference it in its __init__ (even if use_ccia is False).
class CCIA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCIA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.channel_interaction_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y) 
        y = y.unsqueeze(1) 
        y = self.channel_interaction_conv(y) 
        y = y.squeeze(1) 
        y = self.sigmoid(y).view(b, c, 1, 1) 
        return x * y.expand_as(x) 

# --- 6. ResNet50_Classifier Model Definition (MUST EXACTLY MATCH TRAINED MODEL'S DEFINITION) ---
class ResNet50_Classifier(nn.Module):
    # This definition must match the exact __init__ and forward of the model you *trained*
    # Even the use_ccia/use_asda flags passed here must reflect how it was trained.
    def __init__(self, num_classes=Config.NUM_CLASSES, use_ccia=False, use_asda=False, weights_init_type='random'):
        super(ResNet50_Classifier, self).__init__()
        
        # In this inference script, we primarily create the model structure to load weights.
        # So, weights=None is typically used here, as we load state_dict later.
        self.resnet = models.resnet50(weights=None) 
        
        self.use_ccia = use_ccia
        self.use_asda = use_asda

        self.resnet.fc = nn.Identity() 

        if self.use_ccia:
            self.ccia_layer1 = CCIA(channel=256) 
            self.ccia_layer2 = CCIA(channel=512) 
            self.ccia_layer3 = CCIA(channel=1024) 
            self.ccia_layer4 = CCIA(channel=2048) 
        
        if self.use_asda:
            self.asda_layer1 = ASDA(channel=256, input_H=Config.RESNET_FM_SIZES['layer1'][0], input_W=Config.RESNET_FM_SIZES['layer1'][1])
            self.asda_layer2 = ASDA(channel=512, input_H=Config.RESNET_FM_SIZES['layer2'][0], input_W=Config.RESNET_FM_SIZES['layer2'][1])
            self.asda_layer3 = ASDA(channel=1024, input_H=Config.RESNET_FM_SIZES['layer3'][0], input_W=Config.RESNET_FM_SIZES['layer3'][1])
            self.asda_layer4 = ASDA(channel=2048, input_H=Config.RESNET_FM_SIZES['layer4'][0], input_W=Config.RESNET_FM_SIZES['layer4'][1])
        
        self.fc = nn.Linear(2048, num_classes) 

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        if self.use_ccia: x = self.ccia_layer1(x)
        if self.use_asda: x = self.asda_layer1(x) 

        x = self.resnet.layer2(x)
        if self.use_ccia: x = self.ccia_layer2(x)
        if self.use_asda: x = self.asda_layer2(x)
        
        x = self.resnet.layer3(x)
        if self.use_ccia: x = self.ccia_layer3(x)
        if self.use_asda: x = self.asda_layer3(x)

        x = self.resnet.layer4(x)
        if self.use_ccia: x = self.ccia_layer4(x)
        if self.use_asda: x = self.asda_layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Main Inference Logic ---
if __name__ == "__main__":
    print("\n--- Starting Inference on Test Set ---")

    # Load test image IDs from sample_submission.csv
    submission_df_template = pd.read_csv(Config.TEST_CSV)
    test_image_ids = submission_df_template['image_id'].tolist()
    
    # Create the dataset for inference on the TEST set
    test_dataset = TestDataset(
        image_ids=test_image_ids,
        img_dir=Config.TEST_IMAGES_DIR, # CRUCIAL: Pointing to the TEST image directory
        transform=inference_transforms # Use inference transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE_INFERENCE,
        shuffle=False, # No shuffling for consistent order
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    print(f"Total test samples for inference: {len(test_dataset)}")
    print(f"Total batches for inference: {len(test_loader)}")

    # Instantiate the model (must match your trained model's architecture)
    # If your best model used both CCIA and ASDA, set both to True here.
    # weights_init_type='random' is used because we're loading specific weights, not ImageNet.
    model_for_inference = ResNet50_Classifier(
        num_classes=Config.NUM_CLASSES, 
        use_ccia=False,  # Set this based on how your best model was trained (True if it included CCIA)
        use_asda=True,   # Set this based on how your best model was trained (True if it included ASDA)
        weights_init_type='random' # Match what you chose for training (random vs imagenet)
    )

    # Load the trained model weights
    if not os.path.exists(Config.BEST_MODEL_ASDA_PATH):
        print(f"Error: Best model weights not found at {Config.BEST_MODEL_ASDA_PATH}.")
        print("Please ensure your trained model is correctly mounted as a Kaggle dataset or saved in /kaggle/working/.")
        exit() # Exit if model not found
    
    model_for_inference.load_state_dict(torch.load(Config.BEST_MODEL_ASDA_PATH, map_location=Config.DEVICE))
    model_for_inference.to(Config.DEVICE)
    model_for_inference.eval() # Set model to evaluation mode for inference

    all_predictions = []
    all_image_ids_from_loader = [] # Collect image IDs from the loader to ensure correct order

    # Perform inference
    with torch.no_grad(): # Disable gradient calculation for inference
        for inputs, img_ids in tqdm(test_loader, desc="Predicting on test set"):
            inputs = inputs.to(Config.DEVICE)
            outputs = model_for_inference(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_image_ids_from_loader.extend(img_ids) # Collect image IDs from the DataLoader

    # Create the final submission DataFrame
    # IMPORTANT: Ensure the order of image_ids matches the original submission_df_template.
    # The DataLoader provides images in a consistent order based on the initial list,
    # so matching `test_image_ids` with `all_predictions` directly should be correct.
    submission_df = pd.DataFrame({
        'image_id': test_image_ids, # Use the original order from the template CSV
        'label': all_predictions
    })

    # Save the submission file to /kaggle/working/
    # This path is where Kaggle expects your submission.csv
    submission_file_path = os.path.join('/kaggle/working', 'submission.csv') 
    submission_df.to_csv(submission_file_path, index=False)

    print(f"\nSubmission file saved to: {submission_file_path}")
    print("First 5 rows of generated submission.csv:")
    print(submission_df.head())

    print("\nInference on test set complete.")




