# -*- coding: utf-8 -*-
"""
Underwater Image Enhancement using WaterNet on EUVP Dataset

This notebook demonstrates training the WaterNet model for underwater image
enhancement using the EUVP dataset, specifically the 'paired/underwater_dark' subset.
"""

# %% [markdown]
# ## 1. Setup and Imports

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 # Using OpenCV for initial image processing techniques
from tqdm import tqdm # For training progress bar

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# %% [markdown]
# ## 2. WaterNet Model Architecture

# %%
# Copy the provided WaterNet model definition

class ConfidenceMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Confidence maps
        # Accepts input of size (N, 3*4, H, W)
        self.conv1 = nn.Conv2d(
            in_channels=12, out_channels=128, kernel_size=7, dilation=1, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=5, dilation=1, padding="same"
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, dilation=1, padding="same"
        )
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=1, dilation=1, padding="same"
        )
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=7, dilation=1, padding="same"
        )
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, dilation=1, padding="same"
        )
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, dilation=1, padding="same"
        )
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, wb, ce, gc):
        out = torch.cat([x, wb, ce, gc], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = self.relu7(self.conv7(out))
        out = self.sigmoid(self.conv8(out))
        # The output channels of the last conv are 3, interpreted as 3 single-channel maps
        out1, out2, out3 = torch.split(out, [1, 1, 1], dim=1)
        return out1, out2, out3


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=7, dilation=1, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, dilation=1, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, xbar):
        out = torch.cat([x, xbar], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        return out


class WaterNet(nn.Module):
    """
    WaterNet model for underwater image enhancement.
    Takes input image and three initial estimations (WB, CE, GC)
    and outputs the enhanced image.
    """
    def __init__(self):
        super().__init__()
        self.cmg = ConfidenceMapGenerator()
        self.wb_refiner = Refiner()
        self.ce_refiner = Refiner()
        self.gc_refiner = Refiner()

    def forward(self, x, wb, ce, gc):
        wb_cm, ce_cm, gc_cm = self.cmg(x, wb, ce, gc) # Confidence maps (1 channel each)
        refined_wb = self.wb_refiner(x, wb)       # Refined WB output (3 channels)
        refined_ce = self.ce_refiner(x, ce)       # Refined CE output (3 channels)
        refined_gc = self.gc_refiner(x, gc)       # Refined GC output (3 channels)

        # Apply confidence maps - need to ensure broadcasting or repeat channels
        # Confidence maps are 1 channel, refined outputs are 3 channels.
        # Repeat confidence maps across the channel dimension.
        wb_cm_3ch = wb_cm.repeat(1, 3, 1, 1)
        ce_cm_3ch = ce_cm.repeat(1, 3, 1, 1)
        gc_cm_3ch = gc_cm.repeat(1, 3, 1, 1)


        return (
            torch.mul(refined_wb, wb_cm_3ch)
            + torch.mul(refined_ce, ce_cm_3ch)
            + torch.mul(refined_gc, gc_cm_3ch)
        )


# %% [markdown]
# ## 3. Initial Image Processing Estimations (WB, CE, GC)
# These functions take a PIL Image and return a PIL Image or NumPy array representing the initial estimation. These will be used to generate the additional inputs for WaterNet.

# %%
def apply_white_balance(img_pil):
    """Applies a simple Grey World White Balance."""
    img_np = np.array(img_pil) # Convert PIL to numpy (H, W, C), RGB
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV

    # Simple Grey World
    img_float = img_np.astype(np.float32) / 255.0
    avg_b, avg_g, avg_r = np.mean(img_float[:,:,0]), np.mean(img_float[:,:,1]), np.mean(img_float[:,:,2])
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    # Avoid division by zero
    b_scale = avg_all / avg_b if avg_b > 1e-5 else 1.0
    g_scale = avg_all / avg_g if avg_g > 1e-5 else 1.0
    r_scale = avg_all / avg_r if avg_r > 1e-5 else 1.0


    img_float[:,:,0] = img_float[:,:,0] * b_scale
    img_float[:,:,1] = img_float[:,:,1] * g_scale
    img_float[:,:,2] = img_float[:,:,2] * r_scale


    # Clip values to [0, 1] and convert back to uint8
    img_wb_np = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    img_wb_np = cv2.cvtColor(img_wb_np, cv2.COLOR_BGR2RGB) # Convert BGR back to RGB
    return Image.fromarray(img_wb_np)

def apply_contrast_enhancement(img_pil):
    """Applies Contrast Enhancement (Histogram Equalization on V channel in HSV)."""
    img_np = np.array(img_pil) # RGB
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV) # Convert to HSV

    # Apply histogram equalization to the V channel
    img_np[:,:,2] = cv2.equalizeHist(img_np[:,:,2])

    img_ce_np = cv2.cvtColor(img_np, cv2.COLOR_HSV2RGB) # Convert back to RGB
    return Image.fromarray(img_ce_np)


def apply_gamma_correction(img_pil, gamma=0.45):
    """Applies Gamma Correction."""
    img_np = np.array(img_pil) # RGB
    img_float = img_np.astype(np.float32) / 255.0

    # Apply gamma correction
    img_gc_float = np.power(img_float, gamma)

    # Clip values to [0, 1] and convert back to uint8
    img_gc_np = np.clip(img_gc_float * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_gc_np)


# %% [markdown]
# ## 4. EUVP Dataset Class

# %%
class EUVPDataset(Dataset):
    def __init__(self, root_dir, subset='paired/underwater_dark', phase='train', transform=None, fixed_size=(256, 256)):
        """
        Args:
            root_dir (string): Directory with the EUVP dataset (e.g., 'EUVP').
            subset (string): The specific subset to use (e.g., 'paired/underwater_dark').
            phase (string): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            fixed_size (tuple, optional): Resize images to this size (H, W). Set to None to avoid resizing.
        """
        self.root_dir = root_dir
        self.subset = subset
        self.phase = phase # For paired data, phase is typically part of the subset path (e.g., trainA/trainB)

        # Construct paths based on the provided structure
        # For paired data, trainA is low-light, trainB is reference
        self.low_light_dir = os.path.join(root_dir, subset, 'trainA')
        self.reference_dir = os.path.join(root_dir, subset, 'trainB')

        if not os.path.exists(self.low_light_dir):
             raise FileNotFoundError(f"Low light directory not found: {self.low_light_dir}")
        if not os.path.exists(self.reference_dir):
             raise FileNotFoundError(f"Reference directory not found: {self.reference_dir}")


        self.low_light_files = sorted([f for f in os.listdir(self.low_light_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.reference_files = sorted([f for f in os.listdir(self.reference_dir) if f.endswith('.png') or f.endswith('.jpg')])

        # Ensure file lists match (assuming paired data by filename)
        # Create a mapping from low_light filename to reference filename
        # This handles cases where filenames might be slightly different but match logically
        self.reference_map = {os.path.splitext(f)[0]: f for f in self.reference_files}
        self.paired_files = [(f, self.reference_map.get(os.path.splitext(f)[0]))
                             for f in self.low_light_files if os.path.splitext(f)[0] in self.reference_map]

        if len(self.paired_files) == 0:
             raise RuntimeError(f"No paired images found in {self.low_light_dir} and {self.reference_dir}. Check filenames.")

        print(f"Found {len(self.paired_files)} paired images in {self.subset}/{self.phase}.")


        self.transform = transform
        self.fixed_size = fixed_size

        # Define transforms to apply after initial processing and before feeding to model
        self.to_tensor = transforms.ToTensor()
        if fixed_size:
            self.resize_transform = transforms.Resize(fixed_size)
        else:
            self.resize_transform = None


    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        low_light_name, reference_name = self.paired_files[idx]

        low_light_path = os.path.join(self.low_light_dir, low_light_name)
        reference_path = os.path.join(self.reference_dir, reference_name)

        # Load images
        low_light_img = Image.open(low_light_path).convert('RGB')
        reference_img = Image.open(reference_path).convert('RGB')

        # Apply initial processing techniques to the low-light image
        wb_img = apply_white_balance(low_light_img)
        ce_img = apply_contrast_enhancement(low_light_img)
        gc_img = apply_gamma_correction(low_light_img)

        # Apply resizing if defined
        if self.resize_transform:
            low_light_img = self.resize_transform(low_light_img)
            reference_img = self.resize_transform(reference_img)
            wb_img = self.resize_transform(wb_img)
            ce_img = self.resize_transform(ce_img)
            gc_img = self.resize_transform(gc_img)

        # Convert all images to tensors [0, 1]
        low_light_tensor = self.to_tensor(low_light_img)
        reference_tensor = self.to_tensor(reference_img)
        wb_tensor = self.to_tensor(wb_img)
        ce_tensor = self.to_tensor(ce_img)
        gc_tensor = self.to_tensor(gc_img)

        # Ensure tensors have the same dimensions after processing and conversion
        # (This is handled by resize_transform if fixed_size is used)
        # If not using fixed_size, batching might require custom collate_fn or padding.
        # Fixed size is simpler for initial implementation.


        return low_light_tensor, wb_tensor, ce_tensor, gc_tensor, reference_tensor


# %% [markdown]
# ## 5. Data Loading

# %%
# Define dataset path and hyperparameters
DATASET_ROOT = '/kaggle/input/euvp-dataset/EUVP' # Change this to the actual root path of your EUVP dataset folder
BATCH_SIZE = 8
NUM_EPOCHS = 25 # Reduce for quicker test, increase for better results
LEARNING_RATE = 0.0001
FIXED_IMAGE_SIZE = (256, 256) # Or adjust as needed

train_dataset = EUVPDataset(root_dir=DATASET_ROOT,
                            subset='Paired/underwater_dark', # Use the dark paired subset
                            phase='trainA', # This is part of the path, but kept for clarity
                            fixed_size=FIXED_IMAGE_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # num_workers can speed up data loading

# Optional: Create a test dataloader from a test subset if available and paired
# EUVP dataset structure shows test sets under 'unpaired/test/low' and 'unpaired/test/high', which are unpaired.
# For paired testing, you would need a paired test set if available, or manually select pairs from train and split.
# If you want to use the 'unpaired/test' for visualization, you'd need a different dataset class or handling.
# For now, we'll just use the train_dataset for visualization examples after training.

print(f"Number of training images: {len(train_dataset)}")


# %% [markdown]
# ## 6. Model, Loss Function, and Optimizer

# %%
model = WaterNet().to(device)

# Loss function: L1 Loss (Mean Absolute Error) is common for image tasks
criterion = nn.L1Loss()

# Optimizer: Adam is a good default choice
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %% [markdown]
# ## 7. Training Loop

# %%
train_losses = []

print("Starting training...")

for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    # Wrap dataloader with tqdm for a progress bar
    for i, (low_light, wb_est, ce_est, gc_est, reference) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        # Move data to device
        low_light = low_light.to(device)
        wb_est = wb_est.to(device)
        ce_est = ce_est.to(device)
        gc_est = gc_est.to(device)
        reference = reference.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(low_light, wb_est, ce_est, gc_est)

        # Calculate loss
        loss = criterion(outputs, reference)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item() * low_light.size(0) # Multiply by batch size

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] finished. Average Loss: {epoch_loss:.4f}')

print("Training finished.")

# Optional: Save the trained model
# torch.save(model.state_dict(), 'waternet_euvp_dark.pth')
# print("Model saved to waternet_euvp_dark.pth")

# %% [markdown]
# ## 8. Visualize Training Loss

# %%
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o', linestyle='-')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss (L1)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 9. Visualize Example Results

# %%
# Visualize some examples from the training dataset
# You could create a separate small validation set or use the unpaired test set with different visualization
# For simplicity, we'll visualize random examples from the training dataset.

def visualize_results(model, dataset, device, num_examples=5):
    model.eval()
    indices = np.random.choice(len(dataset), num_examples, replace=False)

    plt.figure(figsize=(15, 5 * num_examples)) # Adjust figure size

    for i, idx in enumerate(indices):
        # Get data using the dataset's __getitem__
        low_light, wb_est, ce_est, gc_est, reference = dataset[idx]

        # Add batch dimension and move to device
        low_light_b = low_light.unsqueeze(0).to(device)
        wb_est_b = wb_est.unsqueeze(0).to(device)
        ce_est_b = ce_est.unsqueeze(0).to(device)
        gc_est_b = gc_est.unsqueeze(0).to(device)


        # Get model output
        with torch.no_grad(): # Disable gradient calculation for inference
            output = model(low_light_b, wb_est_b, ce_est_b, gc_est_b)

        # Move tensors back to CPU and convert to numpy arrays
        low_light_np = low_light.cpu().numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
        reference_np = reference.cpu().numpy().transpose(1, 2, 0)
        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0) # Remove batch dim

        # Clip values to [0, 1] just in case network output goes slightly outside
        output_np = np.clip(output_np, 0, 1)

        # Display images
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(low_light_np)
        plt.title('Input (Low Light)')
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(output_np)
        plt.title('WaterNet Output')
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(reference_np)
        plt.title('Reference (Target)')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Visualize some examples
visualize_results(model, train_dataset, device, num_examples=5) # Use train_dataset for visualization

