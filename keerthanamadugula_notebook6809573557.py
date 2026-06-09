import os
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import time

# --- Configuration ---
# IMPORTANT: REPLACE 'your-microplastics-dataset-name' 
# with the actual folder name of your dataset linked to the Kaggle notebook.
DATA_ROOT = '/kaggle/input/your-microplastics-dataset-name/' 
NUM_CLASSES = 4  # 1 (Background) + 3 (Fiber, Fragment, Bead)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.005

# --- 1. Dataset Class (Mock Implementation) ---
class MicroplasticsDataset(Dataset):
    """
    Custom Dataset for Microplastics Detection.
    
    NOTE: This implementation uses mock data (random tensors) for images and targets
    to make the training loop runnable without actual files.
    
    You MUST replace the __getitem__ logic with code to load your real images and 
    parse your annotation files (e.g., XML/JSON) to get accurate bounding boxes.
    """
    def __init__(self, data_root, subset='train', transforms=None):
        self.data_root = data_root
        self.transforms = transforms
        self.image_dir = os.path.join(data_root, 'images', subset)

        # For the sake of making the code runnable on Kaggle without a dataset:
        # We simulate having 100 image file names.
        # In a real scenario, you would list your actual files here.
        self.image_ids = [f'{i:04d}.jpg' for i in range(100)]
        
        print(f"Initialized MicroplasticsDataset for {subset} with {len(self.image_ids)} mock samples.")
        
        # In a real implementation, you'd check if the folder exists:
        # if not os.path.isdir(self.image_dir):
        #     raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # --- 1. Load Image ---
        # In a real scenario:
        # img_path = os.path.join(self.image_dir, self.image_ids[idx])
        # img = Image.open(img_path).convert("RGB")
        
        # Mock Image (3 channels, 512x512 pixels)
        img = torch.rand(3, 512, 512).float() 
        
        # --- 2. Load Targets (Bounding Boxes and Labels) ---
        # In a real scenario, you would parse your XML/JSON annotation file here
        # to get real bounding box coordinates (x_min, y_min, x_max, y_max)
        
        # Mock Targets: Create 2-5 random bounding boxes for demonstration
        num_objs = random.randint(2, 5)
        boxes = []
        labels = []
        
        for _ in range(num_objs):
            # Generate random normalized coordinates (0-512)
            x_min = random.randint(0, 400)
            y_min = random.randint(0, 400)
            x_max = random.randint(x_min + 50, 512)
            y_max = random.randint(y_min + 50, 512)
            
            boxes.append([x_min, y_min, x_max, y_max])
            # Assign random label: 1=Fiber, 2=Fragment, 3=Bead
            labels.append(random.randint(1, NUM_CLASSES - 1))

        # Convert to Tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Suppose all instances are not crowded
        iscrowd = torch.zeros((num_objs,), dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # Note: For simplicity, transforms are omitted, 
            # but they would typically include ToTensor and normalization.
            pass

        return img, target

# --- 2. Model Definition (Faster R-CNN) ---
def get_model_instance_segmentation(num_classes):
    # Load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained box predictor with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# --- 3. Utility Function (Collate for DataLoader) ---
def collate_fn(batch):
    """
    Custom collate function to handle targets in object detection.
    """
    return tuple(zip(*batch))

# --- 4. Training Loop Function ---
def train_model():
    print(f"Using device: {DEVICE}")

    # Initialize Dataset and DataLoader
    try:
        dataset = MicroplasticsDataset(DATA_ROOT, subset='train')
    except FileNotFoundError as e:
        print("\n" * 5)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: DATASET NOT FOUND (FileNotFoundError)         !!!")
        print(f"!!! Please update DATA_ROOT in the code to your dataset path: {DATA_ROOT} !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("\n" * 5)
        return # Stop execution if the path is invalid

    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )

    # Initialize Model, Optimizer, and Learning Rate Scheduler
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)

    # Use a standard SGD optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=LEARNING_RATE, 
        momentum=0.9, 
        weight_decay=0.0005
    )
    # A simple learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # --- Start Training ---
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        start_time = time.time()
        epoch_loss = 0

        for batch_idx, (images, targets) in enumerate(data_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Forward pass: model returns a dictionary of loss tensors
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(data_loader)} | Loss: {losses.item():.4f}")

        # Update the learning rate
        lr_scheduler.step()
        
        avg_loss = epoch_loss / len(data_loader)
        end_time = time.time()
        print(f"\n--- Epoch {epoch+1} finished ---")
        print(f"Average Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f}s\n")
        
        # Optional: Save a checkpoint (replace 'best_model.pth' with a proper naming convention)
        # torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    print("Training Complete!")
    
    # Save final model weights
    torch.save(model.state_dict(), 'final_microplastics_detector.pth')
    print("Model saved as final_microplastics_detector.pth")

if __name__ == '__main__':
    train_model()

    


