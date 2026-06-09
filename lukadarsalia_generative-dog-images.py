import zipfile
from IPython.display import clear_output
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torchvision.io as io
from torch.utils.data import Dataset
import random
from tqdm.notebook import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from IPython.display import display


def zip2dir(input_zip, output_dir):
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the zip file
    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        # Extract all files
        zip_ref.extractall(output_dir)


annot_dir = "Annotation/"
all_dogs = "all-dogs/"

zip2dir("data/generative-dog-images/Annotation.zip", "./")
zip2dir("data/generative-dog-images/all-dogs.zip", "./")


all_dogs_files = [Path(all_dogs+i) for i in os.listdir(all_dogs)]
annot_files = [file for file in Path(annot_dir).rglob("*") if file.is_file()]

print(f"all_dogs_files: {all_dogs_files[:5]}")
print(f"annot_files: {annot_files[:5]}")


tmp_map = {i.name: str(i) for i in annot_files}
input_file_dict = {str(i): tmp_map[i.stem] for i in all_dogs_files}


data = []

for k,v in input_file_dict.items():
    # Load and parse the XML file
    tree = ET.parse(v)  # Replace 'file.xml' with the path to your file
    root = tree.getroot()
    
    # Extract metadata
    folder = root.find('folder').text
    filename = root.find('filename').text
    database = root.find('source/database').text
    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text
    depth = size.find('depth').text
    
    # Extract object details and add metadata for each object
    for obj in root.findall('object'):
        name = obj.find('name').text
        pose = obj.find('pose').text
        truncated = obj.find('truncated').text
        difficult = obj.find('difficult').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
    
        # Add a row for each object
        data.append({
            "ImageDirectory": k,
            "AnnotationDirectory": v,
            'Folder': folder,
            'Filename': filename,
            'Database': database,
            'Width': width,
            'Height': height,
            'Depth': depth,
            'ObjectName': name,
            'Pose': pose,
            'Truncated': truncated,
            'Difficult': difficult,
            'XMin': xmin,
            'YMin': ymin,
            'XMax': xmax,
            'YMax': ymax
        })

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
df.head()


# General info about the DataFrame
print(df.info())

# Summary statistics of numeric columns
print(df.describe())


# Check for missing values
print(df.isnull().sum())

# Find unique values in categorical columns
for col in ['Folder', 'Database', 'ObjectName', 'Pose']:
    print(f"Unique values in {col}: {df[col].unique()}")


# Count occurrences of each object name
object_counts = df['ObjectName'].value_counts()
print(object_counts)

# Plot the distribution
object_counts.plot(kind='bar', title='Object Name Distribution', figsize=(16, 6))


df['XCenter'] = (df['XMin'].astype(int) + df['XMax'].astype(int)) / 2
df['YCenter'] = (df['YMin'].astype(int) + df['YMax'].astype(int)) / 2

# Plot centers
plt.scatter(df['XCenter'], df['YCenter'], alpha=0.5)
plt.title('Bounding Box Centers')
plt.xlabel('XCenter')
plt.ylabel('YCenter')
plt.show()


def denormalize(tensor, mean, std):
    """
    Denormalize a normalized tensor image back to the [0, 1] range for visualization.
    Args:
        tensor (torch.Tensor): The normalized image tensor.
        mean (list): Mean values used for normalization (per channel).
        std (list): Standard deviation values used for normalization (per channel).
    Returns:
        torch.Tensor: Denormalized image tensor in range [0, 1].
    """
    # Input validation
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if tensor.dim() != 3:
        raise ValueError("Input must be a 3D tensor (C,H,W)")
        
    # Convert to float if needed
    tensor = tensor.float()
    
    # Reshape mean and std for broadcasting
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    
    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor

def show_image(image_path, denorm=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Display a single image given its path.
    Args:
        image_path (str): Path to the image file.
        denorm (bool): Whether to denormalize the image.
        mean (list): Mean values used for normalization.
        std (list): Standard deviation values used for normalization.
    """
    try:
        # Read image and convert to float
        img = io.read_image(image_path).float()
        
        # Normalize to [0, 1] range
        if img.max() > 1.0:
            img = img / 255.0
            
        # Apply denormalization if requested
        if denorm:
            img = denormalize(img, mean, std)
        
        # Convert to numpy and display
        plt.figure(figsize=(8, 8))
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(image_path.split('/')[-1])
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

# Example usage
random_image_path = df.sample().iloc[0]['ImageDirectory']
show_image(random_image_path)


# Assuming `df` is your DataFrame
df['XMin'] = pd.to_numeric(df['XMin'], errors='coerce')
df['XMax'] = pd.to_numeric(df['XMax'], errors='coerce')
df['YMin'] = pd.to_numeric(df['YMin'], errors='coerce')
df['YMax'] = pd.to_numeric(df['YMax'], errors='coerce')
df['BoundingBoxArea'] = (df['XMax'] - df['XMin']) * (df['YMax'] - df['YMin'])


df.to_csv("data.csv", index=False)


# Convert relevant columns to numeric types
columns_to_convert = ['XMax', 'XMin', 'YMax', 'YMin', 'BoundingBoxArea', 'Width', 'Height',]
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with invalid numeric data
bbox_data = df.dropna(subset=columns_to_convert)

# Extract bounding box dimensions
bbox_widths = bbox_data['XMax'] - bbox_data['XMin']
bbox_heights = bbox_data['YMax'] - bbox_data['YMin']
bbox_areas = bbox_data['BoundingBoxArea']
image_widths = bbox_data['Width']
image_heights = bbox_data['Height']

# Summarize statistics
summary = {
    "Bounding Box Widths": {
        "Min": bbox_widths.min(),
        "Max": bbox_widths.max(),
        "Mean": bbox_widths.mean(),
        "Median": bbox_widths.median(),
        "StdDev": bbox_widths.std()
    },
    "Bounding Box Heights": {
        "Min": bbox_heights.min(),
        "Max": bbox_heights.max(),
        "Mean": bbox_heights.mean(),
        "Median": bbox_heights.median(),
        "StdDev": bbox_heights.std()
    },
    "Image Widths": {
        "Min": image_widths.min(),
        "Max": image_widths.max(),
        "Mean": image_widths.mean(),
        "Median": image_widths.median(),
        "StdDev": image_widths.std()
    },
    "Image Heights": {
        "Min": image_heights.min(),
        "Max": image_heights.max(),
        "Mean": image_heights.mean(),
        "Median": image_heights.median(),
        "StdDev": image_heights.std()
    }
}

# Convert the summary to a DataFrame
summary_df = pd.DataFrame(summary)

# Display statistics
print("Bounding Box and Image Statistics:")
print(summary_df)

# Plot histograms for bounding boxes and image dimensions
plt.figure(figsize=(18, 10))

# Bounding box widths
plt.subplot(2, 2, 1)
plt.hist(bbox_widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Bounding Box Widths')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')

# Bounding box heights
plt.subplot(2, 2, 2)
plt.hist(bbox_heights, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Bounding Box Heights')
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency')

# Image widths
plt.subplot(2, 2, 3)
plt.hist(image_widths, bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.title('Image Widths')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')

# Image heights
plt.subplot(2, 2, 4)
plt.hist(image_heights, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.title('Image Heights')
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Directories for saving images
os.makedirs("original_images", exist_ok=True)
os.makedirs("augmented_images", exist_ok=True)

# Define normalization parameters
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Transforms
crop_transform = transforms.Compose([
    transforms.Resize((256, 256))  # Crop and resize centered at BBox
])

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),  # Added blur
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)  # Apply normalization
])

def denormalize(tensor):
    """
    Denormalize a tensor image with fixed mean and std
    """
    tensor = tensor.clone()  # Avoid modifying the original tensor
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)  # Clamp to [0,1] range

# DataFrame to store results
result = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row['ImageDirectory']
    object_name = row['ObjectName']
    x_center, y_center = row['XCenter'], row['YCenter']
    width, height = row['Width'], row['Height']
    bbox_width, bbox_height = row['XMax'] - row['XMin'], row['YMax'] - row['YMin']
    
    try:
        # Open and crop image
        img = Image.open(img_path).convert("RGB")
        left = max(0, x_center - bbox_width // 2)
        upper = max(0, y_center - bbox_height // 2)
        right = min(width, x_center + bbox_width // 2)
        lower = min(height, y_center + bbox_height // 2)
        cropped_img = img.crop((left, upper, right, lower)).resize((256, 256))
        
        # Save original cropped image
        original_path = f"original_images/{os.path.basename(img_path)}"
        cropped_img.save(original_path)
        
        # Apply augmentations
        aug_img = augment_transform(cropped_img)  # Returns normalized tensor
        
        # Denormalize before saving
        aug_img_denorm = denormalize(aug_img)
        
        # Convert to PIL and save
        aug_img_pil = transforms.ToPILImage()(aug_img_denorm)
        augmented_path = f"augmented_images/{os.path.basename(img_path)}"
        aug_img_pil.save(augmented_path)
        
        # Append to result DataFrame
        result.append({
            "OriginalImagePath": original_path,
            "AugmentedImagePath": augmented_path,
            "ObjectName": object_name
        })
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

# Create DataFrame
final_df = pd.DataFrame(result)
final_df.to_csv("image_paths.csv", index=False)
print("Processing complete! Data saved in 'image_paths.csv'.")

# Function to view results
def view_pair(row_idx, final_df):
    """
    Display original and augmented image pair
    """
    import matplotlib.pyplot as plt
    
    row = final_df.iloc[row_idx]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display original
    img1 = Image.open(row['OriginalImagePath'])
    ax1.imshow(img1)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Display augmented
    img2 = Image.open(row['AugmentedImagePath'])
    ax2.imshow(img2)
    ax2.set_title('Augmented')
    ax2.axis('off')
    
    plt.show()

# View a random pair
random_idx = np.random.randint(len(final_df))
view_pair(random_idx, final_df)


final_df


final_df.to_csv('final_df.csv', index=False)


# View a random pair
random_idx = np.random.randint(len(final_df))
view_pair(random_idx, final_df)


class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim


        # Initial convolution and pooling
        self.down_scale = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        self.conv_final = nn.Conv2d(512, latent_dim * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_final = nn.BatchNorm2d(latent_dim * 2)
        self.fact=nn.SiLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x3 = self.layer_3(x)
        x = self.layer_4(x3)
        x = self.fact(self.layer_5(x)+self.down_scale(x3))
        x = self.down_scale(x)
        x = self.conv_final(x)
        x = self.bn_final(x)
        x = self.fact(x)
        return x[:, :self.latent_dim, :, :], x[:, self.latent_dim:, :, :]

class Decoder(nn.Module):
    def __init__(self, in_channels, feature_dim, latent_dim):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.latent_dim  = latent_dim

        # Upscaling layers with residual blocks
        self.up_scale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.layer_1 = nn.Sequential(
                    nn.Conv2d(latent_dim, 512, kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm2d(512),
                    nn.SiLU(),
                )

        self.layer_3 = nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm2d(256),
                    nn.SiLU(),
                )
        self.layer_4 = nn.Sequential(
                    nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(64),
                    nn.SiLU(),
                )
                
        self.layer_5 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.SiLU(),
                )
        # Final convolution layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, self.in_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.in_channels),
            nn.Tanh()  # Outputs in range [-1, 1]
        )
        self.fact=nn.SiLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.up_scale(x)
        x = self.up_scale(x)
        x = self.layer_3(x)
        x = self.up_scale(x)
        x = self.layer_4(x)
        x = self.up_scale(x)
        x = self.fact(self.layer_5(x)+x)
        x = self.up_scale(x)
        x = self.final_layer(x) 
        return x

class VAE(nn.Module):
    def __init__(self, in_channels, feature_dim, latent_dim, cond_dim, n_conditions):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.latent_dim  = latent_dim
        self.encoder = Encoder(in_channels, feature_dim, latent_dim)
        self.decoder = Decoder(in_channels, feature_dim, latent_dim + cond_dim)
        self.n_conditions = n_conditions

        # Learnable embeddings for conditions
        self.condition_embeddings = nn.Embedding(n_conditions, cond_dim)
        self.condition_ffn = nn.Linear(cond_dim, cond_dim)
        self.tanh = nn.Tanh()

    def add_cond_emb(self, z, condition):
        # Add condition embedding
        cond_embedding = self.condition_ffn(self.condition_embeddings(condition).squeeze())
        cond_embedding = self.tanh(cond_embedding)
        cond_embedding = cond_embedding.unsqueeze(-1).unsqueeze(-1) # (batch_size, 1, cond_dim, 1, 1)
        if len(cond_embedding.shape) == 3: cond_embedding = cond_embedding.unsqueeze(0)
        cond_embedding = cond_embedding.expand(-1, -1, 8, 8)  # Expand for concatenation

        z_cond = torch.cat([z, cond_embedding], dim=1)
        return z_cond

    def reparameterize(self, mu, logvar, condition):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return self.add_cond_emb(z, condition)

    def forward(self, x, condition):
        mu, logvar = self.encoder(x)
        # Reparametrizing trick
        z = self.reparameterize(mu, logvar, condition)

        # Decode
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        mse_loss = 0

        mse_loss += F.mse_loss(recon_x.view(-1, self.in_channels * self.feature_dim * self.feature_dim),
                               x.view(-1, self.in_channels * self.feature_dim * self.feature_dim))
        mse_loss /= len(recon_x)

        kl_loss = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())

        return mse_loss, kl_loss

    def generate(self, batch_size, condition=None, device="cuda"):
        """
        Generates images from random latents (and optional condition).
        """
        # 1) Sample random latent
        z = torch.randn(batch_size, self.latent_dim, 8, 8, device=device)

        # 2) Add condition embedding if you want conditional generation
        if condition is None:
            condition = torch.randint(0, self.n_conditions, (batch_size,), device=device)

        z = self.add_cond_emb(z, condition)

        # 3) Decode
        fake = self.decoder(z)
        return fake


model = VAE(3, 256, 256, 16, 120)
print(f"VAE has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} Trainable Parameters")
print(f"VAE Encoder has {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,} Trainable Parameters")
print(f"VAE Decoder has {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad):,} Trainable Parameters")
print(model.forward(torch.rand(2, 3, 256, 256), torch.tensor([[1], [5]]))[0].shape)
print(model.generate(5,device='cpu').shape)


class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None, include_augmented=False):
        """
        Args:
            df: DataFrame with image paths and conditions (ObjectName)
            transform: Optional transforms to apply
            include_augmented: Whether to include augmented images
        """
        # Handle original and augmented data
        self.image_paths = df['OriginalImagePath'].tolist()
        self.aug_image_paths = df['AugmentedImagePath'].tolist()
        self.conditions = df['ObjectName'].tolist()
        self.include_augmented = include_augmented
        self.transform = transform

        # Load all images into memory
        self.orig_images = [self.transform(Image.open(p).convert('RGB')) if self.transform else Image.open(p).convert('RGB') for p in tqdm(self.image_paths)]
        self.aug_images = [self.transform(Image.open(p).convert('RGB')) if self.transform else Image.open(p).convert('RGB') for p in tqdm(self.aug_image_paths)]

        # Create condition encoder
        unique_conditions = sorted(set(self.conditions))
        self.condition_to_idx = {cond: idx for idx, cond in enumerate(unique_conditions)}
        self.num_conditions = len(unique_conditions)
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Select image based on augmentation inclusion
        if self.include_augmented:
            aug = random.random() < 0.2
            image = self.aug_images[idx] if aug else self.orig_images[idx]
        else:
            image = self.orig_images[idx]

        # Target image is the original image
        target_image = self.orig_images[idx]

        # Get condition
        condition = self.conditions[idx]
        condition_idx = self.condition_to_idx[condition]
        

        return image, torch.tensor(condition_idx, dtype=torch.long), target_image


def prepare_data(csv_path, batch_size=32):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Create train/val split (80/20)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # transforms.Resize((128, 128))
    ])
    
    # Create datasets
    train_dataset = CustomImageDataset(train_df, transform=transform, include_augmented=True)
    val_dataset = CustomImageDataset(val_df, transform=transform, include_augmented=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.num_conditions


def plot_losses(train_losses, val_losses):
    """Plot training progress"""
    clear_output(wait=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Total loss
    ax1.plot([x['total'] for x in train_losses], label='Train')
    ax1.plot([x['total'] for x in val_losses], label='Val')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Reconstruction loss
    ax2.plot([x['recon'] for x in train_losses], label='Train')
    ax2.plot([x['recon'] for x in val_losses], label='Val')
    ax2.set_title('Reconstruction Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    # KL loss
    ax3.plot([x['kl'] for x in train_losses], label='Train')
    ax3.plot([x['kl'] for x in val_losses], label='Val')
    ax3.set_title('KL Loss')
    ax3.set_xlabel('Epoch')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def plot_gradients(model):
    encoder_grads = []
    decoder_grads = []

    # Collect gradients for encoder and decoder separately
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name.startswith("encoder"):
                encoder_grads.append(param.grad.abs().mean().item())
            elif name.startswith("decoder"):
                decoder_grads.append(param.grad.abs().mean().item())

    # Combine x-axis positions for encoder and decoder
    total_layers = list(range(len(encoder_grads) + len(decoder_grads)))
    encoder_layers = total_layers[:len(encoder_grads)]
    decoder_layers = total_layers[len(encoder_grads):]

    # Plot gradients with a continuous x-axis
    plt.plot(encoder_layers, encoder_grads, label='Encoder Gradients', color='blue')
    plt.plot(decoder_layers, decoder_grads, label='Decoder Gradients', color='orange')
    plt.title('Gradient Magnitude (Encoder to Decoder Continuity)')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Magnitude')
    plt.legend()
    plt.show()


def get_cyclic_beta(epoch, T_0, T_mult=2, beta_min=0.1, beta_max=1.0):
    """
    Compute cyclic beta based on epoch, similar to a cosine annealing schedule.
    
    Args:
        epoch (int): Current epoch.
        T_0 (int): Number of epochs for the first cycle.
        T_mult (int): Cycle multiplier for successive cycles.
        beta_min (float): Minimum value of beta.
        beta_max (float): Maximum value of beta.

    Returns:
        beta (float): Beta value for the current epoch.
    """
    cycle_length = T_0
    while epoch >= cycle_length:
        epoch -= cycle_length
        cycle_length *= T_mult
    beta = beta_min + 0.5 * (beta_max - beta_min) * (1 + torch.cos(torch.tensor(epoch / cycle_length * 3.1415926535)))
    return beta.item()

# Scheduler parameters
T_0 = 10   # Number of epochs in the first cycle
T_mult = 2 # Cycle length multiplier
beta_min = 0.1
beta_max = 1.0
epochs = 100  # Total number of epochs to plot

# Compute beta values for all epochs
beta_values = [get_cyclic_beta(epoch, T_0, T_mult, beta_min, beta_max) for epoch in range(epochs)]

# Plot the cyclic beta values
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), beta_values, label="Cyclic Beta", color="blue")
plt.axhline(beta_min, color="red", linestyle="--", label="Beta Min")
plt.axhline(beta_max, color="green", linestyle="--", label="Beta Max")
plt.xlabel("Epoch")
plt.ylabel("Beta Value")
plt.title("Cyclic Beta Schedule")
plt.legend()
plt.grid()
plt.show()


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
def denormalize(tensor):
    """
    Denormalize a tensor image with fixed mean and std
    """
    tensor = tensor.clone()  # Avoid modifying the original tensor
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)  # Clamp to [0,1] range

    
def train_vae(model, train_loader, val_loader, epochs=100, device='cuda', n_conditions=120):
    # Scheduler parameters
    T_0 = epochs   # Number of epochs in the first cycle
    T_mult = 2 # Cycle length multiplier
    beta_min = 0.01
    beta_max = 0.05
    
    # Loss function
    reconstruction_loss = nn.MSELoss()
    avg_val_loss = 0
    avg_val_recon = 0
    avg_val_kl = 0
    # Optimizer with adjusted learning rate
    optimizer = optim.AdamW(model.parameters(), lr=1e-4,)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    # Move model to device
    model = model.to(device)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    fixed_z = torch.randn(1, model.latent_dim, 4, 4).to(device)
    fixed_cond = torch.randint(0, n_conditions, (1,1,)).to(device)
    val_iter = iter(val_loader)
    train_iter = iter(train_loader)
    val_samples, val_conditions, val_targets = next(val_iter)
    train_samples, train_conditions, train_targets = next(train_iter)
    val_samples = val_samples[4:6].to(device)  # Pick first two images
    val_conditions = val_conditions[4:6].to(device)
    val_targets = val_targets[4:6].to(device)
    
    train_samples = train_samples[4:6].to(device)  # Pick first two images
    train_conditions = train_conditions[4:6].to(device)
    train_targets = train_targets[4:6].to(device)
    fixed_val_samples = (val_samples, val_conditions, val_targets)
    fixed_train_samples = (train_samples, train_conditions, train_targets)
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Training phase
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        beta = get_cyclic_beta(epoch, T_0, T_mult, beta_min, beta_max)
        print("Current beta is:", beta)
        train_pbar = tqdm(train_loader, leave=False, desc=f'Training Epoch {epoch+1}')
        for batch_idx, (data, condition, target) in enumerate(train_pbar):
            data, condition, target = data.to(device), condition.to(device), target.to(device)
            batch_size = data.size(0)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(data, condition)
            
            recon_loss, kl_loss = model.loss_function(recon_batch, target, mu, logvar)
            
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item() * batch_size
            train_recon_loss += recon_loss.item() * batch_size
            train_kl_loss += kl_loss.item() * batch_size
            
            train_pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item()
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, leave=False, desc=f'Validation Epoch {epoch+1}')
            for data, condition, target in val_pbar:
                data, condition, target = data.to(device), condition.to(device), target.to(device)
                batch_size = data.size(0)
                
                recon_batch, mu, logvar = model.forward(data, condition)

                recon_loss, kl_loss = model.loss_function(recon_batch, target, mu, logvar)
                loss = recon_loss + beta * kl_loss
                
                val_loss += loss.item() * batch_size
                val_recon_loss += recon_loss.item() * batch_size
                val_kl_loss += kl_loss.item() * batch_size
                
                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'recon': recon_loss.item(),
                    'kl': kl_loss.item()
                })
        
        scheduler.step()
        
        # Calculate average losses
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        
        avg_train_loss = train_loss / train_size
        avg_train_recon = train_recon_loss / train_size
        avg_train_kl = train_kl_loss / train_size
        
        avg_val_loss = val_loss / val_size
        avg_val_recon = val_recon_loss / val_size
        avg_val_kl = val_kl_loss / val_size
        
        # Store losses for plotting
        train_losses.append({
            'total': avg_train_loss,
            'recon': avg_train_recon,
            'kl': avg_train_kl
        })
        val_losses.append({
            'total': avg_val_loss,
            'recon': avg_val_recon,
            'kl': avg_val_kl
        })
        
        if (epoch + 1) % 5 == 0:
            plot_losses(train_losses, val_losses)
            
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'condition_to_idx': train_loader.dataset.condition_to_idx  # Save mapping
            }, 'best_model.pth')
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f})')
        print(f'Valid Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f})')
        print('-' * 50)
        with torch.no_grad():
            plt.figure(figsize=(20, 4))
            fake_images = model.generate(batch_size=2, device='cuda')
            fake_images = denormalize(fake_images)
            fake_images = fake_images.clamp(0, 1).cpu()
            for i in range(fake_images.size(0)):
                plt.subplot(2, fake_images.size(0), i + 1)
                plt.imshow(fake_images[i].permute(1, 2, 0))
                plt.axis("off")
                plt.title("Generated")
            plt.tight_layout()
            plt.show()

            # Unpack fixed validation samples
            val_samples, val_conditions, val_targets = fixed_val_samples
            train_samples, train_conditions, train_targets = fixed_train_samples
            
            # Forward pass through the model
            recon_batch, mu, logvar = model.forward(val_samples, val_conditions)
            recon_batch_t, mu_t, logvar_t = model.forward(train_samples, train_conditions)

            # Compute reconstruction loss
            recon_loss, kl_loss = model.loss_function(recon_batch, val_targets, mu, logvar)
            recon_loss_t, kl_loss_t = model.loss_function(recon_batch_t, train_targets, mu_t, logvar_t)

            # Total loss
            total_loss = recon_loss + kl_loss
            print("Reconstruction Loss:", recon_loss.item())
            print("KL Divergence Loss:", kl_loss.item())
            print("Total Loss:", total_loss.item())

            total_loss_t = recon_loss_t + kl_loss_t
            print("Train Reconstruction Loss:", recon_loss_t.item())
            print("Train KL Divergence Loss:", kl_loss_t.item())
            print("Train Total Loss:", total_loss_t.item())

        
            # Convert original and reconstructed images to numpy
            recon_batch = denormalize(recon_batch)
            val_targets = denormalize(val_targets)
            original_imgs = val_targets.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            reconstructed_imgs = recon_batch.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)

            recon_batch_t = denormalize(recon_batch_t)
            train_targets = denormalize(train_targets)
            original_imgs_t = train_targets.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            reconstructed_imgs_t = recon_batch_t.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            
            # Plot original and reconstructed images
            for i in range(2):  # Iterate over the two images
                plt.subplot(1, 3, i + 2)
                plt.imshow(reconstructed_imgs[i])
                plt.axis('off')
                plt.title(f'Reconstructed Image {i+1} (Epoch {epoch+1})')

            # Show all images
            plt.tight_layout()
            plt.show()

            # Optional: Display original images separately
            plt.figure(figsize=(10, 5))
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                plt.imshow(original_imgs[i])
                plt.axis('off')
                plt.title(f'Original Image {i+1}')
            plt.tight_layout()
            plt.show()



            # Plot original and reconstructed images
            for i in range(2):  # Iterate over the two images
                plt.subplot(1, 3, i + 2)
                plt.imshow(reconstructed_imgs_t[i])
                plt.axis('off')
                plt.title(f'Reconstructed Train Image {i+1} (Epoch {epoch+1})')

            # Show all images
            plt.tight_layout()
            plt.show()

            # Optional: Display original images separately
            plt.figure(figsize=(10, 5))
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                plt.imshow(original_imgs_t[i])
                plt.axis('off')
                plt.title(f'Original Train Image {i+1}')
            plt.tight_layout()
            plt.show()
            plot_gradients(model)
        
        
    print(f'Valid Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f})')
    print('-' * 50)
    plot_losses(train_losses, val_losses)
    return train_losses, val_losses


train_loader, val_loader, n_conditions = prepare_data("final_df.csv", batch_size=128)


# Create a tiny dataset with 2 examples
tiny_loader = DataLoader(
    torch.utils.data.Subset(train_loader.dataset, [i+torch.randint(0, 10000, (1,)).item() for i in range(128)]),  # Use two examples
    batch_size=128,
    shuffle=True
)
# Adjust model capacity (optional, depends on results)
model = VAE(in_channels=3, feature_dim=256, latent_dim=128, cond_dim=16, n_conditions=n_conditions)
print(f"VAE has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} Trainable Parameters")
print(f"VAE Encoder has {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,} Trainable Parameters")
print(f"VAE Decoder has {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad):,} Trainable Parameters")
model = model.to('cuda')
feature_maps = {}
to_img = torchvision.transforms.ToPILImage()

def hook_fn(module, input, output):
    feature_maps[module] = output

# Register hooks for layers
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(hook_fn)
# Train on a single batch
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
epochs = 100
T_0 = 20   # Number of epochs in the first cycle
T_mult = 2 # Cycle length multiplier
beta_min = 0.001
beta_max = 1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for epoch in tqdm(range(epochs)):
    beta = 0.05#get_cyclic_beta(epoch, T_0, T_mult, beta_min, beta_max)
    
    model.train()
    for data, condition, target in tiny_loader:
        data, condition, target = data.to('cuda'), condition.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, condition)
        recon_loss, kl_loss = model.loss_function(recon_batch, target, mu, logvar)

        loss = recon_loss + beta*kl_loss
        loss.backward()
        optimizer.step()

    # Visualize every 100 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Recon Loss: {recon_loss.item()}, KL Loss: {kl_loss.item()}, Total Loss: {loss.item()}")

        # Plot labels and predictions
        with torch.no_grad():
            
            for data, condition, target in tiny_loader:
                data, condition, target = data.to('cuda'), condition.to('cuda'), target.to('cuda')

            data = data[:2]
            condition = condition[:2]
            target = target[:2]

            recon_batch, mu, logvar = model.forward(data, condition)

            for i in range(data.size(0)):
                recon_batch_i = to_img(denormalize(recon_batch[i]))
                target_i = to_img(denormalize(target[i]))
                print("Target Image:")
                display(target_i)
                print("Reconstruction Image:")
                display(recon_batch_i)

        plot_gradients(model)



with torch.no_grad():
    fake_images = model.generate(batch_size=2, device='cuda')
    for i in range(fake_images.size(0)):
        display(to_img(denormalize(fake_images[i])))


# Run a batch through the model
data, condition = [(data.to('cpu'), condition.to('cpu')) for data, condition, target in tiny_loader][0]
model = model.to('cpu')
model.eval()
with torch.no_grad():
    _ = model(data, condition)



def visualize_feature_maps(feature_maps, num_channels=8):
    for layer_name in feature_maps:
        fmap = feature_maps[layer_name][0].detach().cpu().numpy()  # Batch index 0
        num_channels = min(num_channels, fmap.shape[0])
        fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
        for i in range(num_channels):
            axes[i].imshow(fmap[i], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Channel {i}')
        plt.show()
visualize_feature_maps(feature_maps)


batch_img=[data for data, condition, target in tiny_loader][0].to('cpu')

print(batch_img.shape)
mu, logvar = model.encoder(batch_img)

# Plot histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(mu.detach().cpu().numpy().flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Distribution of Mu")
plt.subplot(1, 2, 2)
plt.hist(logvar.detach().cpu().numpy().flatten(), bins=50, color='red', alpha=0.7)
plt.title("Distribution of Logvar")
plt.show()


# Dictionary to store activations
activations = {}

# Hook function to capture pre- and post-activation values
def hook_fn(module, input, output):
    activations[module] = {'input': input[0], 'output': output}

# Register hooks for layers of interest
hooks = []
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        hooks.append(layer.register_forward_hook(hook_fn))


# Run a batch through the model
data, condition = [(data.to('cpu'), condition.to('cpu')) for data, condition, target in tiny_loader][0]

model.eval()
with torch.no_grad():
    _ = model(data, condition)

# Print statistics for each layer
for module, act in activations.items():
    inp = act['input']
    out = act['output']
    print(f"Layer: {module}")
    print(f"Input - Mean: {inp.mean().item():.4f}, Std: {inp.std().item():.4f}, Min: {inp.min().item():.4f}, Max: {inp.max().item():.4f}")
    print(f"Output - Mean: {out.mean().item():.4f}, Std: {out.std().item():.4f}, Min: {out.min().item():.4f}, Max: {out.max().item():.4f}")
    print("-" * 50)


plot_gradients(model)


# Initialize data and model
model = VAE(in_channels=3, feature_dim=256, latent_dim=512, cond_dim=32, n_conditions=n_conditions)
print(f"VAE has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} Trainable Parameters")
# Train model and get loss history
train_losses, val_losses = train_vae(model, train_loader, val_loader, epochs=50, device='cuda', n_conditions=n_conditions)


test_model = VAE(in_channels=3, feature_dim=256, latent_dim=512, cond_dim=32, n_conditions=n_conditions)
checkpoint = torch.load('best_model.pth')
test_model.load_state_dict(checkpoint['model_state_dict'])


# Enter name of dog breeds (you can find them below)
conditions = ['Eskimo_dog']

conditions = torch.tensor([train_loader.dataset.condition_to_idx[i] for i in conditions])
to_img = torchvision.transforms.ToPILImage()
with torch.no_grad():
    fake_images = test_model.generate(batch_size=1, condition=conditions, device='cpu')
    for i in range(fake_images.size(0)):
        display(to_img(denormalize(fake_images[i])))


train_loader.dataset.conditions


# Random Generator

to_img = torchvision.transforms.ToPILImage()
with torch.no_grad():
    fake_images = test_model.generate(batch_size=1,  device='cpu')
    for i in range(fake_images.size(0)):
        display(to_img(denormalize(fake_images[i])))




