import glob
from PIL import Image
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch
import torch.nn.functional as F
from typing import Tuple, List
from diffusers import UNet2DModel
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision
from typing import List


class ImageDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images from a directory pattern.

    Args:
        img_pattern (str): A glob pattern to match image file paths (e.g., '/path/*.jpg').
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, img_pattern: str, transform=None):
        self.image_list = glob.glob(img_pattern)
        if not self.image_list:
            raise ValueError(f"No images found with pattern: {img_pattern}")
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.image_list)

    def __getitem__(self, idx: int):
        """
        Loads an image and applies optional transformations.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (transformed image tensor, dummy label)
        """
        img_path = self.image_list[idx]

        # Open the image and ensure it has 3 channels (RGB)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Dummy label (can be replaced with actual label logic)
        label = 1
        return image, label


# === Configuration ===
IMG_SIZE = 64               # Resize all images to 64x64
BATCH_SIZE = 64            # Number of images per batch (adjust based on GPU memory)
SHUFFLE_DATA = True         # Shuffle data at each epoch
DROP_LAST = False           # Whether to drop the last incomplete batch

# === Dataset Preparation ===
def get_dataset(path_pattern: str) -> ImageDataset:
    """
    Creates a dataset with basic image augmentation and normalization.

    Args:
        path_pattern (str): Glob pattern to match image file paths.

    Returns:
        ImageDataset: The dataset with applied transforms.
    """
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),        # Resize image to fixed size
        transforms.RandomHorizontalFlip(),              # Randomly flip image horizontally
        transforms.ToTensor(),                          # Convert to tensor [0, 1]
        transforms.Normalize([0.5]*3, [0.5]*3)           # Normalize to [-1, 1]
    ])
    return ImageDataset(path_pattern, transform=data_transform)

# === Path to Dataset ===
DATA_PATH_PATTERN = os.path.join(
    r"/kaggle/input/stanford-cars-dataset/cars_train/cars_train",
    "*"
)

# === Create Dataset & Dataloader ===
dataset = get_dataset(DATA_PATH_PATTERN)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_DATA,
    drop_last=DROP_LAST,
)


def display_image_grid(
    imgs: torch.Tensor, 
    nrow: int = 8, 
    dpi: int = 150, 
    normalize: bool = True,
    title: str = None
) -> plt.Figure:
    """
    Displays a grid of images using matplotlib.

    Args:
        imgs (Tensor): A batch of image tensors (B, C, H, W).
        nrow (int): Number of images per row.
        dpi (int): Figure resolution.
        normalize (bool): Whether to normalize pixel values to [0, 1].
        title (str, optional): Title to display above the image grid.

    Returns:
        matplotlib.figure.Figure: The displayed figure.
    """
    # Create the image grid tensor (C, H, W) -> (H, W, C)
    grid_img = make_grid(imgs, nrow=nrow, padding=2, normalize=normalize).cpu()
    np_img = np.transpose(grid_img.numpy(), (1, 2, 0))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(nrow * 2, 2), dpi=dpi)
    ax.imshow(np_img)
    ax.axis("off")

    # Optional title
    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    return fig

# Load a batch from the DataLoader
batch, _ = next(iter(dataloader))
print("Loaded batch shape:", batch.shape)

# Display the first 8 images from the batch
_ = display_image_grid(batch[:8], nrow=8, dpi=150, title="Sample Car Images")


# Define beta schedule
T = 512  # number of diffusion steps
# YOUR CODE HERE
betas = torch.linspace(start=0.0001, end=0.02, steps=T)  # linear schedule

plt.plot(range(T), betas.numpy(), label='Beta Values')
plt.xlabel('Diffusion Step')
plt.ylabel('Beta Value')
_ = plt.title('Beta Schedule over Diffusion Steps')


# Pre-calculate different terms for closed form
alphas = 1. - betas
# alpha bar
alphas_cumprod = torch.cumprod(alphas, axis=0)
# alpha bar at t-1
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt of alpha bar
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

# Inference:
# 1 / sqrt(alpha)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# sqrt of one minus alpha bar
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# sigma_t
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)



def display_sequence(imgs: List[torch.Tensor], nrow: int = 8, dpi: int = 100) -> plt.Figure:
    """
    Display a sequence of images in a grid.

    Args:
        imgs (List[torch.Tensor]): List of image tensors to be displayed.
        nrow (int): Number of images per row.
        dpi (int): Dots per inch (quality) of the resulting image.
    
    Returns:
        fig (plt.Figure): The matplotlib figure object containing the image grid.
    """
    # Convert list of tensors to a single tensor
    imgs_tensor = torch.cat([img.unsqueeze(0) for img in imgs], dim=0)  # Add batch dimension
    grid = torchvision.utils.make_grid(imgs_tensor, nrow=nrow, padding=2, normalize=True)

    # Plot the images
    fig = plt.figure(figsize=(nrow * 2, len(imgs) // nrow * 2))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # Convert to HWC format for plotting
    plt.axis('off')  # Hide axes for a cleaner look
    return fig
    
@torch.no_grad()
def forward_diffusion_viz(
    image: torch.Tensor, 
    device: str = "cpu", 
    num_images: int = 16, 
    dpi: int = 75, 
    interleave: bool = False
) -> Tuple[plt.Figure, torch.Tensor]:
    """
    Simulates the forward diffusion process (image → noise) for visualization.

    Args:
        image (Tensor): Input image tensor, shape (1, C, H, W).
        T (int): Total number of diffusion steps.
        sqrt_alphas_cumprod (Tensor): Precomputed sqrt(α̅ₜ) schedule.
        sqrt_one_minus_alphas_cumprod (Tensor): Precomputed sqrt(1 - α̅ₜ).
        device (str): Torch device.
        num_images (int): Number of intermediate visual steps to display.
        dpi (int): DPI for visualization.
        interleave (bool): Whether to interleave original + noise images.

    Returns:
        fig (Figure): Matplotlib figure.
        final_image (Tensor): Final noisy image.
    """
    step_size = max(1, T // num_images)
    imgs, noises = [], []

    for t_idx in range(0, T, step_size):
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)
        noise = torch.randn_like(image, device=device)

        noised_img = (
            sqrt_alphas_cumprod[t].view(1, 1, 1, 1) * image +
            sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1) * noise
        )

        imgs.append(torch.clamp(noised_img.squeeze(0), -1, 1))
        noises.append(torch.clamp(noise.squeeze(0), -1, 1))

    if interleave:
        imgs = [x for pair in zip(imgs, noises) for x in pair]

    fig = display_image_grid(imgs, dpi=dpi, nrow=8, normalize=True)
    return fig, imgs[-1]


@torch.no_grad()
def make_inference(
    input_noise: torch.Tensor,
    device: str = "cuda:0",
    diff_steps = 512,
    return_all: bool = False
) -> List[torch.Tensor] or torch.Tensor:
    """
    Implements reverse denoising sampling (DDPM inference loop).

    Args:
        input_noise (Tensor): Initial noise input tensor, shape (B, C, H, W).
        T (int): Total number of timesteps.
        model (nn.Module): Trained denoising model (e.g. UNet).
        betas, sqrt_recip_alphas, etc.: Precomputed diffusion schedule constants.
        device (str): Torch device.
        return_all (bool): If True, returns list of intermediate images.

    Returns:
        If return_all is False: final denoised image (Tensor).
        If True: list of all intermediate denoised images (List[Tensor]).
    """
    x = input_noise
    batch_size = x.size(0)
    imgs = []

    for t_idx in reversed(range(diff_steps)):
        t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
        noise = torch.randn_like(x) if t_idx > 0 else 0

        predicted_noise = model(x, t).sample

        x = (
            sqrt_recip_alphas[t].view(batch_size, 1, 1, 1) * 
            (x - betas[t].view(batch_size, 1, 1, 1) * predicted_noise / 
            sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1))
        ) + torch.sqrt(posterior_variance[t].view(batch_size, 1, 1, 1)) * noise

        imgs.append(torch.clamp(x, -1, 1))

    return imgs if return_all else imgs[-1]


for image in batch[:5]:
    _ = forward_diffusion_viz(image.unsqueeze(dim=0))


model = UNet2DModel(
    sample_size=32,           # image size
    in_channels=3,            # RGB
    out_channels=3,
    block_out_channels=(64, 128, 256, 512))
n_params = sum(p.numel() for p in model.parameters())
print(
    f"Number of parameters: {n_params:,}"
)


device = "cuda" if torch.cuda.is_available() else "cpu"
# Move everything to GPU
model.to(device)

sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
alphas_cumprod_prev = alphas_cumprod_prev.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
posterior_variance = posterior_variance.to(device)
betas = betas.to(device)
criterion = torch.nn.MSELoss()
base_lr = 0.0006 # Maximum learning rate we will use
epochs = 2 # Total number of epochs
T_max = epochs  # Number of epochs for Cosine Annealing. We do only one cycle
warmup_epochs = 1  # Number of warm-up epochs

# Uncomment the following lines
# if you want to do the _VERY_ long training,
# base_lr = 0.0001 # Maximum learning rate we will use
# epochs = 300 # Total number of epochs
# T_max = epochs  # Number of epochs for Cosine Annealing. We do only one cycle
# warmup_epochs = 10  # Number of warm-up epochs


optimizer = Adam(model.parameters(), lr=base_lr)
scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=T_max - warmup_epochs,
    eta_min=base_lr / 10  # starting value for the LR
)


# We will use this noise to generate some images during training to check
# where we stand


alpha = 0.1  # Smoothing factor
ema_loss = None  # Initialize EMA loss

for epoch in range(epochs):
    
    if epoch < warmup_epochs:
        # Linear warm-up
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        # Cosine Annealing after warm-up
        scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
        
    for batch, _ in tqdm(dataloader):
        
        batch = batch.to(device)
        bs = batch.shape[0]
        
        optimizer.zero_grad()
        
        # Generate random time values between 0 and 512 : a vector with the size of batch size
        t = torch.randint(0, T, (batch.shape[0],), device=device).long()
        
        # Create random noise and apply it to the images from the batch (each image with a corresponding time generated 
        # previosly)
        noise = torch.randn_like(batch, device=device)
        x_noisy = (
            sqrt_alphas_cumprod[t].view(bs, 1, 1, 1) * batch + 
            sqrt_one_minus_alphas_cumprod[t].view(bs, 1, 1, 1) * noise
        )
        # Feed the noisy images and the corresponding timesteps to the UNet
        noise_pred = model(x_noisy, t)
        noise_pred = noise_pred.sample
        # loss the mse between the real generated noise and the predicted noise by the UNet
        loss = criterion(noise, noise_pred)
        
        loss.backward()
        optimizer.step()
        
        if ema_loss is None:
            # First batch
            ema_loss = loss.item()
        else:
            # Exponential moving average of the loss
            ema_loss = alpha * loss.item() + (1 - alpha) * ema_loss
    
    if epoch == epochs-1:
        with torch.no_grad():
    #         fig, _ = sample_image(fixed_noise, forward=False, device=device)
            for i in range(8):
                fixed_noise = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
                imgs = make_inference(fixed_noise, return_all=True)
                fig = display_sequence([imgs[0].squeeze(dim=0)] + [x.squeeze(dim=0) for x in imgs[63::64]], nrow=9, dpi=150)

            plt.show(fig)
        os.makedirs("diffusion_output_long", exist_ok=True)
        fig.savefig(f"diffusion_output_long/frame_{epoch:05d}.png")
    #plt.close(fig)
    
    print(f"epoch {epoch+1}: loss: {ema_loss:.3f}, lr: {current_lr:.6f}")


input_noise = torch.randn((32, 3, IMG_SIZE, IMG_SIZE), device=device)
imgs = make_inference(input_noise)
_ = display_sequence(imgs, dpi=75, nrow=4)


for i in range(8):
    fixed_noise = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    imgs = make_inference(fixed_noise, return_all=True, diff_steps=512)
    fig = display_sequence([imgs[0].squeeze(dim=0)] + [x.squeeze(dim=0) for x in imgs[63::64]], nrow=9, dpi=150)

plt.show(fig)




