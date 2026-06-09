import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from PIL import Image


# Load Data
image_dir = '/kaggle/input/isic-2024-challenge/train-image/image'
mask_dir = '/kaggle/working/masks_isic'

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Define the encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        enc = self.encoder(x)
        middle = self.middle(enc)
        dec = self.decoder(middle)
        return torch.sigmoid(dec)

# Function to create masks
def create_mask(image_path, output_mask_path):
    # Ensure the mask directory exists
    mask_dir = os.path.dirname(output_mask_path)
    os.makedirs(mask_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to create a binary mask
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Save the mask with a "_mask.png" suffix
    output_mask_path = output_mask_path.replace(".jpg", "_mask.png")
    cv2.imwrite(output_mask_path, mask)

# Create masks for all images
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)
    create_mask(image_path, mask_path)

# Dataset class for skin lesion images and masks
class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(128, 128)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.image_files[idx].replace(".jpg", "_mask.png"))

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")  # Load mask as grayscale

        # Resize both image and mask
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)  # Convert mask to tensor

        return image, mask


# Define the transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


from torch.utils.data import Subset, DataLoader
# Load the full dataset
full_dataset = SkinLesionDataset(image_dir, mask_dir, transform=transform, image_size=(128, 128))

# Get 15,000 random indices
num_samples = 15000

# Create a subset with 15,000 randomly selected images
subset_dataset = Subset(full_dataset, range(num_samples))

# Split into train (80%) and test (20%)
train_size = int(0.8 * len(subset_dataset))
test_size = len(subset_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(subset_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_size)
print(test_size)


import os

mask_dir = "/kaggle/working/masks_isic"
if not os.path.exists(mask_dir):
    print("Mask directory does not exist!")
else:
    mask_files = os.listdir(mask_dir)
    if not mask_files:
        print("Mask directory is empty!")
    else:
        print("Sample masks:", mask_files[:5])  # Print a few filenames


import torch
import torch.nn as nn
import torch.nn.functional as F

class ROILoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(ROILoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, rois):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        rois = rois.view(-1)
        
        intersection = (inputs * rois).sum()
        union = (inputs + rois).sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        roi_loss = 1 - iou
        
        return roi_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class BoundaryAwareLoss(nn.Module):
    def __init__(self, weight_dice=0.4, weight_boundary=0.4, weight_roi=0.2):
        super(BoundaryAwareLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_boundary = weight_boundary
        self.weight_roi = weight_roi
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.roi_loss = ROILoss()  # Ensure ROILoss is defined as needed
    
    def forward(self, inputs, targets, boundaries, rois):
        dice = self.dice_loss(inputs, targets)
        boundary_loss = self.bce_loss(inputs, boundaries)
        roi = self.roi_loss(inputs, rois)
        total_loss = (self.weight_dice * dice + 
                      self.weight_boundary * boundary_loss + 
                      self.weight_roi * roi)
        return total_loss



import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def generate_roi(masks, dilation_size=5):
    # Convert masks to binary if not already
    masks = (masks > 0.5).float()

    # Create a dilation kernel
    dilation_kernel = torch.ones((1, 1, dilation_size, dilation_size)).to(masks.device)

    # Dilate the mask to create ROI regions
    rois = F.conv2d(masks, dilation_kernel, padding=dilation_size // 2)
    
    # Binarize the ROIs to ensure they are either 0 or 1
    rois = (rois > 0).float()

    return rois
def generate_boundaries(masks):
    # Convert masks to binary if not already
    masks = (masks > 0.5).float()

    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(masks.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(masks.device)

    # Convolve masks with Sobel filters
    edges_x = F.conv2d(masks, sobel_x, padding=1)
    edges_y = F.conv2d(masks, sobel_y, padding=1)

    # Calculate edge magnitude
    edges = torch.sqrt(edges_x*2 + edges_y*2)

    # Binarize the edges
    edges = (edges > 0.5).float()

    return edges



import torch.optim as optim
import torch.nn.functional as F

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model and move it to the device
model = UNet().to(device)

# Define your criterion with BoundaryAwareLoss
criterion = BoundaryAwareLoss(weight_dice=0.4, weight_boundary=0.4, weight_roi=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0  # Initialize running loss for each epoch
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Resize masks to match output size
        masks_resized = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)

        # Generate boundary maps from masks
        boundaries = generate_boundaries(masks_resized)

        # Generate ROI regions from masks
        rois = generate_roi(masks_resized)
        
        # Calculate loss with the generated boundaries and ROIs
        loss = criterion(outputs, masks_resized, boundaries, rois)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate running loss
        running_loss += loss.item()
        
        # Print the current batch number and loss
        print(f'\rEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Images Processed: {(batch_idx+1) * images.size(0)}, Loss: {loss.item():.4f}', end='')

    # Calculate and print the average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f'\nEpoch [{epoch+1}/{num_epochs}] Completed, Average Loss: {avg_loss:.4f}')



import torch
import torch.nn.functional as F

# Set the model to evaluation mode
model.eval()
test_loss = 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Resize masks to match the output size
        masks_resized = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
        
        # Generate boundary maps from resized masks
        boundaries = generate_boundaries(masks_resized)
        
        # Generate ROI regions from resized masks
        rois = generate_roi(masks_resized)
        
        # Calculate loss with the generated boundaries and ROIs
        loss = criterion(outputs, masks_resized, boundaries, rois)
        
        test_loss += loss.item()
    
    # Calculate and print the average test loss
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')



import torch
import torch.nn.functional as F
import numpy as np

def generate_boundaries(masks):
    # Convert masks to binary if not already
    masks = (masks > 0.5).float()

    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(masks.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(masks.device)

    # Convolve masks with Sobel filters
    edges_x = F.conv2d(masks, sobel_x, padding=1)
    edges_y = F.conv2d(masks, sobel_y, padding=1)

    # Calculate edge magnitude
    edges = torch.sqrt(edges_x**2 + edges_y**2)

    # Binarize the edges
    edges = (edges > 0.5).float()

    return edges

def evaluate_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_images = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)  # Move to the correct device
            outputs = model(images)
            
            # Resize the outputs to match the target size
            output_size = masks.size()[2:]
            outputs_resized = F.interpolate(outputs, size=output_size, mode='bilinear', align_corners=False)
            
            # Generate boundary maps for masks
            boundaries = generate_boundaries(masks)  # Generate boundaries
            
            # Generate ROIs
            rois = generate_roi(masks)  # Generate or load your ROIs

            # Calculate loss
            loss = criterion(outputs_resized, masks, boundaries, rois)  # Pass the ROIs to the criterion
            total_loss += loss.item() * images.size(0)

            # Calculate metrics
            accuracy = calculate_accuracy(outputs_resized, masks)
            iou, dice, precision, recall = calculate_metrics(outputs_resized, masks)
            
            total_accuracy += accuracy * images.size(0)
            total_iou += iou * images.size(0)
            total_dice += dice * images.size(0)
            total_precision += precision * images.size(0)
            total_recall += recall * images.size(0)
            
            total_images += images.size(0)
    
    avg_loss = total_loss / total_images
    avg_accuracy = total_accuracy / total_images
    avg_iou = total_iou / total_images
    avg_dice = total_dice / total_images
    avg_precision = total_precision / total_images
    avg_recall = total_recall / total_images

    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Accuracy: {avg_accuracy:.4f}')
    print(f'Mean IoU: {avg_iou:.4f}')
    print(f'Mean Dice Coefficient: {avg_dice:.4f}')
    print(f'Mean Precision: {avg_precision:.4f}')
    print(f'Mean Recall: {avg_recall:.4f}')


def calculate_accuracy(outputs, targets):
    preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid to get probabilities and threshold at 0.5
    preds = preds.view(-1)
    targets = targets.view(-1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

def calculate_metrics(outputs, targets):
    preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid to get probabilities and threshold at 0.5
    preds = preds.view(-1)
    targets = targets.view(-1)

    # True Positives, False Positives, False Negatives
    TP = (preds * targets).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()

    # Intersection over Union (IoU)
    intersection = TP
    union = TP + FP + FN
    iou = intersection / union if union != 0 else 0

    # Dice Coefficient
    dice = (2. * TP) / (2. * TP + FP + FN) if (2. * TP + FP + FN) > 0 else 0

    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return iou, dice, precision, recall

# Define the loss function
criterion = BoundaryAwareLoss()  # Ensure BoundaryAwareLoss is defined or imported

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate the model
evaluate_model(model, test_loader, criterion, device)


torch.save(model.state_dict(), 'segmentation_part_15k_line.pth')

