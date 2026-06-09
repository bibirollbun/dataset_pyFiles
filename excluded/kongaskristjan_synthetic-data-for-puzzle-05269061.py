from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


class DatasetType(Enum):
    TRAIN = "training"
    VALIDATION = "validation"


class SyntheticDataset(Dataset):
    def __init__(
        self,
        epoch_size: int,
        n_samples: int,
        size: int,
        n_stripes: int,
        dataset_type: DatasetType,
    ):
        super().__init__()

        assert 2 * size - 1 >= n_stripes
        assert n_samples <= epoch_size

        self.epoch_size = epoch_size
        self.n_samples = n_samples
        self.size = size
        self.n_stripes = n_stripes
        self.dataset_type = dataset_type

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, i: int) -> tuple[int, Tensor, Tensor]:
        rng = torch.Generator()
        sample_idx = i % self.n_samples

        if self.dataset_type == DatasetType.TRAIN:
            rng.manual_seed(sample_idx + 1)
        elif self.dataset_type == DatasetType.VALIDATION:
            rng.manual_seed(-sample_idx - 1)
        input, target = _generate_pair(rng, self.size, self.size, self.n_stripes)
        return sample_idx, input, target


def _generate_pair(rng: torch.Generator, height: int, width: int, n_stripes: int) -> tuple[Tensor, Tensor]:
    """
    Generates a puzzle pair with a repetitive diagonal color pattern.

    The target image displays a full diagonal pattern, while the input image
    shows only one stripe of each color from the pattern against a
    background.

    Args:
        height: The height of the input and output grids.
        width: The width of the input and output grids.
        n_stripes: The number of unique colors in the repeating diagonal pattern.

    Returns:
        A tuple containing the (input, target) puzzle pair as torch.int64 tensors.
    """
    # Validate input arguments
    assert 1 <= n_stripes <= 9, "n_stripes must be between 1 and 9."
    max_diag_index = height + width - 2
    assert n_stripes <= max_diag_index + 1, "n_stripes cannot be larger than the number of possible diagonals."

    # 1. Select n_stripes unique colors from [1, 9] for the pattern
    all_colors = torch.arange(1, 10, dtype=torch.int64)
    shuffled_indices = torch.randperm(9, generator=rng)
    stripe_colors = all_colors[shuffled_indices[:n_stripes]]

    # 2. Create a grid where each cell's value is its diagonal index (r + c)
    rows = torch.arange(height, dtype=torch.int64).view(height, 1)
    cols = torch.arange(width, dtype=torch.int64).view(1, width)
    diagonal_indices = rows + cols

    # 3. Generate the full target tensor
    # The color of each cell is determined by its diagonal index modulo n_stripes
    color_map_indices = diagonal_indices % n_stripes
    target = stripe_colors[color_map_indices]

    # 4. Generate the sparse input tensor
    # A 1D boolean mask to mark which diagonals to keep
    kept_diagonal_mask = torch.zeros(max_diag_index + 1, dtype=torch.bool)

    # For each color in the pattern, randomly select one diagonal to display
    for i in range(n_stripes):
        # Find all diagonals that correspond to the i-th color in the pattern
        possible_diags = torch.arange(i, max_diag_index + 1, n_stripes)

        # Randomly choose one of these diagonals
        if len(possible_diags) > 0:
            rand_idx = torch.randint(len(possible_diags), size=(1,), generator=rng)
            chosen_diag = possible_diags[rand_idx]
            kept_diagonal_mask[chosen_diag] = True

    # Create the 2D grid mask from the 1D diagonal mask
    grid_mask = kept_diagonal_mask[diagonal_indices]

    # Create the input tensor by applying the mask to the target.
    # Pixels not in the mask are set to the background color 0.
    input_tensor = torch.zeros_like(target)
    input_tensor[grid_mask] = target[grid_mask]

    # To channels last one hot
    input_tensor = F.one_hot(input_tensor, num_classes=10)
    target = F.one_hot(target, num_classes=10)

    return input_tensor, target




def visualize_synthetic_dataset(dataset: SyntheticDataset, num_samples_to_show: int = 5):
    """Visualizes samples from the provided SyntheticDataset."""
    print("\n" + "="*80)
    print("VISUALIZING SYNTHETIC DATASET SAMPLES")
    print("="*80)
    
    # ARC color map - colors for values 0-9
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    # Helper function to visualize a single sample pair
    def visualize_sample(sample_idx, input_tensor, target_tensor):
        # The dataset returns one-hot encoded tensors of shape (H, W, 10).
        # We need to convert them back to 2D integer arrays (H, W) for visualization.
        input_grid = torch.argmax(input_tensor, dim=-1).numpy()
        target_grid = torch.argmax(target_tensor, dim=-1).numpy()
        
        # Create visualization
        fig = plt.figure(figsize=(10, 5))
        grid_spec = plt.GridSpec(1, 2, width_ratios=[1, 1])
        
        # Input
        ax1 = fig.add_subplot(grid_spec[0, 0])
        ax1.imshow(input_grid, cmap=cmap, norm=norm)
        ax1.grid(True, which='both', color='lightgrey', linewidth=0.5)
        ax1.set_title("Input")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Target
        ax2 = fig.add_subplot(grid_spec[0, 1])
        ax2.imshow(target_grid, cmap=cmap, norm=norm)
        ax2.grid(True, which='both', color='lightgrey', linewidth=0.5)
        ax2.set_title("Target")
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        plt.suptitle(f"Synthetic Sample #{sample_idx}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    # Iterate over the dataset and visualize the requested number of samples
    count = 0
    # We iterate up to the number of unique samples defined in the dataset
    # to avoid showing duplicates if num_samples_to_show > dataset.n_samples
    limit = min(num_samples_to_show, dataset.n_samples)
    
    print(f"Showing {limit} unique samples from the dataset.")
    
    for i in range(limit):
        sample_idx, input_tensor, target_tensor = dataset[i]
        print(f"\nVisualizing Sample #{sample_idx}")
        visualize_sample(sample_idx, input_tensor, target_tensor)
        count += 1
        
    print(f"\nVisualized {count} samples.")


# Parameters for the synthetic dataset
EPOCH_SIZE = 10
N_SAMPLES = 10   # Number of unique samples to generate
SIZE = 7         # Grid size
N_STRIPES = 3    # Number of colors in the pattern

# Create the training dataset instance
train_dataset = SyntheticDataset(
    epoch_size=EPOCH_SIZE,
    n_samples=N_SAMPLES,
    size=SIZE,
    n_stripes=N_STRIPES,
    dataset_type=DatasetType.TRAIN,
)

# Visualize
visualize_synthetic_dataset(train_dataset, num_samples_to_show=N_SAMPLES)

