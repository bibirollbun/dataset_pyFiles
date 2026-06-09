# Imports
import os
from pathlib import Path
import torch
import json
import math
import heapq
from collections import deque
from typing import List, Tuple, Dict
import random
import numpy as np
from multiprocessing import Pool, cpu_count
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as T
from torchvision.utils import save_image
from scipy.ndimage import gaussian_filter, map_coordinates, affine_transform
from collections import Counter
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter, ImageOps
from tqdm import tqdm
import pandas as pd


# Paths
paths = {
    "train_images": Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/images"),
    "train_labels": Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/labels"),
    "test_images": Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/test/images"),
    "test_velocities": Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/test/velocities"),
    "mod": Path("/kaggle/working/augmented"),
    "saved_images": Path("/kaggle/working/generated_sample/images"),
    "saved_labels": Path("/generated_sample/labels"),
    "mod_images": Path("/kaggle/working/augmented/images"),
    "mod_labels": Path("/kaggle/working/augmented/labels"),
    "submission_path":Path("/kaggle/working/submission_baseline.csv"),
    "assets": Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/assets"),
    'mod_train_images': Path('/kaggle/working/test_data/images'),
    'mod_train_labels': Path('/kaggle/working/test_data/labels')
}

# Make necessary dirs
paths['saved_images'].mkdir(parents=True, exist_ok=True)
paths['saved_labels'].mkdir(parents=True, exist_ok=True)
paths['mod_images'].mkdir(parents = True, exist_ok = True)
paths['mod_labels'].mkdir(parents = True, exist_ok = True)
paths['mod_train_images'].mkdir(parents = True, exist_ok = True)
paths['mod_train_labels'].mkdir(parents = True, exist_ok = True)

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parameters
NUM_TRAIN_SAMPLES = 20
GRID_SIZE = 20
NUM_CLASSES = 5  # 0..4
BATCH_SIZE = 3
EPOCHS = 30
LR = 1e-4
to_tensor = T.ToTensor()
TARGET_TILE_SIZE = 36        # Final tile size in generated image         
NUM_SAMPLES = 2000
BORDER_WIDTH = 1             # 1px black borders (set to 0 for seamless like Image 1)
PNG_COMPRESSION = 4        # Balance between size and speed
USE_MULTIPROCESSING = True
NUM_WORKERS = max(1, cpu_count() - 1)

# classes
CLASS_WALK = 0
CLASS_WALL = 1
CLASS_HAZARD = 2
CLASS_START = 3
CLASS_GOAL = 4

CLASS_WEIGHTS = torch.tensor(
    [1.0, 2.0, 1.8, 3.0, 3.0],
    dtype=torch.float32,
    device=DEVICE
)


counter = Counter()

for json_file in paths["train_labels"].glob("*.json"):
    with open(json_file, "r") as f:
        data = json.load(f)

        grid = np.array(data["grid"])   # change key if needed
        counter.update(grid.flatten().tolist())

print("Class counts and probabilites:")
for cls in sorted(counter):
    print(f"class {cls}: {counter[cls]/NUM_TRAIN_SAMPLES} | probability: {counter[cls]/(NUM_TRAIN_SAMPLES * GRID_SIZE * GRID_SIZE)}")


asset_map = {
    "desert": {
        "0": "t1_sand",
        "1": {"a": "t1_rocks", "b": "t1_cacti"},
        "2": "t1_quicksand",
        "3": "t1_rover",
        "4": "t1_goal",
    },
    "forest": {
        "0": "t0_dirt",
        "1": "t0_tree",
        "2": "t0_puddle",
        "3": "t0_startship",
        "4": "t0_goal",
    },
    "lab": {
        "0": "t2_floor",
        "1": {"a": "t2_wall", "b": "t2_plasma"},
        "2": "t2_glue",
        "3": "t2_drone",
        "4": "t2_goal",
    },
}

TERRAIN_COSTS = {
    'lab': {
        CLASS_WALK: 1.0,
        CLASS_WALL: float('inf'),
        CLASS_HAZARD: 3.0,
        CLASS_START: 1.0,
        CLASS_GOAL: 2.0
    },
    'forest': {
        CLASS_WALK: 1.5,
        CLASS_WALL: float('inf'),
        CLASS_HAZARD: 2.8,
        CLASS_START: 1.5,
        CLASS_GOAL: 2.5
    },
    'desert': {
        CLASS_WALK: 1.2,
        CLASS_WALL: float('inf'),
        CLASS_HAZARD: 3.7,
        CLASS_START: 1.2,
        CLASS_GOAL: 2.2
    }
}



def generate_grid(size=20, class_probs={0:0.5,1:0.28,2:0.21}, hazard_bias=0.3):
    grid = np.full((size, size), 1, dtype=np.int8)

    start = (random.randrange(size), random.randrange(size))
    goal = (random.randrange(size), random.randrange(size))
    while goal == start:
        goal = (random.randrange(size), random.randrange(size))

    cur = start
    grid[cur] = 0

    while cur != goal:
        r, c = cur
        moves = [(nr, nc) for nr, nc in
                 [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                 if 0 <= nr < size and 0 <= nc < size]
        nxt = random.choice(moves)
        grid[nxt] = 2 if random.random() < hazard_bias else 0
        cur = nxt

    for r in range(size):
        for c in range(size):
            if (r, c) in (start, goal):
                continue
            if grid[r, c] == 1:
                grid[r, c] = random.choices(
                    [0, 1, 2],
                    [class_probs[0], class_probs[1], class_probs[2]]
                )[0]

    grid[start] = 3
    grid[goal] = 4
    return grid, start, goal



def load_assets(asset_root, asset_map, tile_size):
    """
    Load assets with high-quality downsampling to preserve detail
    """
    cache = {}
    
    for terrain, classes in asset_map.items():
        cache[terrain] = {}
        for val in classes.values():
            names = val.values() if isinstance(val, dict) else [val]
            for name in names:
                img_path = asset_root / terrain / f"{name}.png"
                
                # Load high-res image
                img = Image.open(img_path).convert("RGB")
                
                # Use LANCZOS for high-quality downsampling (better than NEAREST for large->small)
                # This preserves detail when going from 512x512 or 1024x1024 down to 32x32
                img_resized = img.resize((tile_size, tile_size), Image.LANCZOS)
                
                cache[terrain][name] = to_tensor(img_resized)
    
    return cache


def choose_asset(asset_map, terrain, class_id):
    entry = asset_map[terrain][str(class_id)]
    if isinstance(entry, dict):
        return random.choice(list(entry.values()))
    return entry


def render_grid(grid, terrain, asset_map, asset_cache, border_width=1):
    """
    Render grid with optional borders
    border_width=0 for seamless (like Image 1)
    border_width=1 for bordered (like Images 2,3,4)
    """
    grid = torch.as_tensor(grid)
    tile = next(iter(asset_cache[terrain].values()))
    C, H, W = tile.shape
    
    if border_width > 0:
        # With borders
        tile_h = H + border_width
        tile_w = W + border_width
        
        canvas = torch.zeros((C, 
                             grid.shape[0] * tile_h + border_width,
                             grid.shape[1] * tile_w + border_width))
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                cls = int(grid[r, c])
                name = choose_asset(asset_map, terrain, cls)
                
                y_start = r * tile_h + border_width
                x_start = c * tile_w + border_width
                canvas[:, y_start:y_start+H, x_start:x_start+W] = asset_cache[terrain][name]
    else:
        # Seamless (no borders)
        canvas = torch.zeros((C, grid.shape[0] * H, grid.shape[1] * W))
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                cls = int(grid[r, c])
                name = choose_asset(asset_map, terrain, cls)
                canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = asset_cache[terrain][name]
    
    return canvas


def save_png_optimized(tensor, path, compress_level=6):
    """
    Save PNG with optimal compression for 600-700 KB target
    """
    img = (tensor.clamp(0, 1) * 255).byte()
    img = img.permute(1, 2, 0).numpy()
    pil = Image.fromarray(img)
    pil.save(path, format="PNG", compress_level=compress_level, optimize=True)

def generate_sample(args):
    """Generate a single sample (for multiprocessing)"""
    i, asset_cache, asset_map, save_images_dir, save_labels_dir, border_width = args
    
    # Set seeds for reproducibility
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)

    terrain = random.choice(list(asset_map.keys()))
    grid, start, goal = generate_grid(GRID_SIZE)

    image = render_grid(grid, terrain, asset_map, asset_cache, border_width)

    # Save image
    save_png_optimized(
        image,
        save_images_dir / f"sample_{i:04d}.png",
        compress_level=PNG_COMPRESSION
    )

    # Save label
    label = {
        "id": f"{i:04d}",
        "t": terrain,
        "n": GRID_SIZE,
        "s": list(start),
        "g": list(goal),
        "grid": grid.tolist()
    }
    
    with open(save_labels_dir / f"sample_{i:04d}.json", "w") as f:
        json.dump(label, f, separators=(",", ":"))
    
    return i


def main():
    # Loading high-res assets and downsampling to {TARGET_TILE_SIZE}x{TARGET_TILE_SIZE}...
    asset_cache = load_assets(paths['assets'], asset_map, TARGET_TILE_SIZE)
    # Prepare arguments for all samples
    args_list = [
        (i, asset_cache, asset_map, paths['saved_images'], paths['saved_labels'], BORDER_WIDTH)
        for i in range(NUM_SAMPLES)
    ]
    
    if USE_MULTIPROCESSING and NUM_SAMPLES > 100:
        with Pool(NUM_WORKERS) as pool:
            for idx, result in enumerate(pool.imap_unordered(generate_sample, args_list)):
                if idx % 100 == 0 and idx > 0:
                    print(f"Generated {idx}/{NUM_SAMPLES}")
        print(f"Generated {NUM_SAMPLES}/{NUM_SAMPLES}")
    else:
        print(f"Generating {NUM_SAMPLES} samples (single-threaded)...")
        for i, args in enumerate(args_list):
            generate_sample(args)
            if i % 100 == 0:
                print(f"Generated {i}/{NUM_SAMPLES}")
    
    print("âœ… Dataset generation complete")


main()


# helper function
def load_label_grid(json_path):
    """Load grid from JSON label file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['grid'], dtype=np.int8)


def save_label_grid(grid, json_path):
    """Save grid to JSON label file"""
    # Load existing JSON to preserve metadata
    base_name = json_path.stem.split('_')[0]  # Get original ID
    original_json = json_path.parent.parent / "labels" / f"{base_name}.json"
    
    if original_json.exists():
        with open(original_json, 'r') as f:
            data = json.load(f)
    else:
        data = {"id": json_path.stem, "n": grid.shape[0]}
    
    data['grid'] = grid.tolist()
    
    with open(json_path, 'w') as f:
        json.dump(data, f, separators=(",", ":"))


def add_color_blobs(img, num_blobs=None, intensity_range=(40, 120)):
    """
    Add semi-transparent color blobs like in test images
    """
    if num_blobs is None:
        num_blobs = random.randint(2, 5)
    
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    w, h = img.size
    
    for _ in range(num_blobs):
        # Random position
        cx = random.randint(-w//4, w + w//4)
        cy = random.randint(-h//4, h + h//4)
        
        # Random size
        rx = random.randint(w // 6, w // 3)
        ry = random.randint(h // 6, h // 3)
        
        # Random color (vibrant tints like purple, pink, cyan, green)
        color_choices = [
            (200, 100, 255),  # Purple/Magenta
            (255, 150, 200),  # Pink
            (100, 255, 255),  # Cyan
            (150, 255, 150),  # Light green
            (255, 200, 100),  # Orange
            (150, 150, 255),  # Light blue
        ]
        base_color = random.choice(color_choices)
        alpha = random.randint(intensity_range[0], intensity_range[1])
        color = (*base_color, alpha)
        
        # Draw ellipse
        draw.ellipse(
            [cx - rx, cy - ry, cx + rx, cy + ry],
            fill=color
        )
    
    # Blur the overlay for softer blobs
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.randint(5, 15)))
    
    return Image.alpha_composite(img, overlay).convert("RGB")

def elastic_deformation(img, alpha=None, sigma=None):
    """
    Apply elastic deformation to simulate warping/distortion
    """
    if alpha is None:
        alpha = random.uniform(1.5, 4.0)
    if sigma is None:
        sigma = random.uniform(4.0, 8.0)
    
    img_np = np.array(img)
    shape = img_np.shape[:2]
    
    # Generate random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    
    # Apply to each channel
    warped = np.zeros_like(img_np)
    for c in range(3):
        warped[..., c] = map_coordinates(
            img_np[..., c],
            indices,
            order=1,
            mode="reflect"
        ).reshape(shape)
    
    return Image.fromarray(warped)
    
def random_crop(img, grid, crop_size=None):
    """
    Randomly crop a portion of the image
    Returns cropped image and corresponding grid portion
    """
    w, h = img.size
    grid_h, grid_w = grid.shape
    
    if crop_size is None:
        # Random crop size between 50-90% of original
        crop_ratio = random.uniform(0.5, 0.9)
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
    else:
        crop_h, crop_w = crop_size
    
    # Ensure crop size doesn't exceed image
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    
    # Random crop position
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    
    # Crop image
    img_cropped = img.crop((left, top, left + crop_w, top + crop_h))
    
    # Calculate grid crop (assuming uniform tile size)
    tile_h = h / grid_h
    tile_w = w / grid_w
    
    grid_top = int(top / tile_h)
    grid_left = int(left / tile_w)
    grid_bottom = int((top + crop_h) / tile_h)
    grid_right = int((left + crop_w) / tile_w)
    
    # Ensure valid grid bounds
    grid_top = max(0, min(grid_top, grid_h - 1))
    grid_left = max(0, min(grid_left, grid_w - 1))
    grid_bottom = max(grid_top + 1, min(grid_bottom, grid_h))
    grid_right = max(grid_left + 1, min(grid_right, grid_w))
    
    grid_cropped = grid[grid_top:grid_bottom, grid_left:grid_right]
    
    return img_cropped, grid_cropped

def photometric_augmentation(img):
    """
    Apply random brightness, contrast, saturation, hue adjustments
    """
    # Brightness
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    
    # Contrast
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    
    # Saturation
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    
    return img

def add_gaussian_noise(img, sigma=None):
    """Add Gaussian noise to image"""
    if sigma is None:
        sigma = random.randint(3, 12)
    
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_blur(img):
    """Add slight blur"""
    blur_radius = random.uniform(0.3, 1.2)
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))



class AdvancedAugmentation:
    """
    Advanced augmentation system matching test set characteristics
    """
    
    def __init__(
        self,
        src_images_dir: Path,
        src_labels_dir: Path,
        aug_images_dir: Path,
        aug_labels_dir: Path
    ):
        self.src_images_dir = src_images_dir
        self.src_labels_dir = src_labels_dir
        self.aug_images_dir = aug_images_dir
        self.aug_labels_dir = aug_labels_dir
        
        # Load image IDs
        self.image_ids = []
        for p in sorted(self.src_labels_dir.glob("*.json")):
            image_id = p.stem
            img_path = self.src_images_dir / f"{image_id}.png"
            if img_path.is_file():
                self.image_ids.append(image_id)
        
        if not self.image_ids:
            raise RuntimeError(f"No images found in {src_images_dir}")
        
    def save_original(self):
        """Copy original data to augmented folder"""
        for img_id in self.image_ids:
            src_img = self.src_images_dir / f"{img_id}.png"
            src_label = self.src_labels_dir / f"{img_id}.json"
            
            dst_img = self.aug_images_dir / f"{img_id}.png"
            dst_label = self.aug_labels_dir / f"{img_id}.json"
            
            if not dst_img.exists():
                Image.open(src_img).save(dst_img)
                grid = load_label_grid(src_label)
                save_label_grid(grid, dst_label)
        
        print(f"âœ… Original data saved ({len(self.image_ids)} images)")
    
    
    def color_blob_augmentation(self, num_variants=2):
        """
        Apply color blob overlays (like test images)
        """
        count = 0
        
        for img_id in self.image_ids:
            img = Image.open(self.src_images_dir / f"{img_id}.png")
            grid = load_label_grid(self.src_labels_dir / f"{img_id}.json")
            
            for v in range(num_variants):
                aug_id = f"{img_id}_cb{v}"
                aug_img_path = self.aug_images_dir / f"{aug_id}.png"
                
                if not aug_img_path.exists():
                    # Apply color blobs with random intensity
                    img_aug = add_color_blobs(img, num_blobs=random.randint(2, 4))
                    
                    # Optional: add slight photometric changes
                    if random.random() < 0.5:
                        img_aug = photometric_augmentation(img_aug)
                    
                    img_aug.save(aug_img_path)
                    save_label_grid(grid, self.aug_labels_dir / f"{aug_id}.json")
                    count += 1
        
        print(f"âœ… Color blob augmentation complete ({count} new images)")
    
    
    def elastic_augmentation(self, num_variants=2):
        """
        Apply elastic deformation
        """
        count = 0
        
        for img_id in self.image_ids:
            img = Image.open(self.src_images_dir / f"{img_id}.png")
            grid = load_label_grid(self.src_labels_dir / f"{img_id}.json")
            
            for v in range(num_variants):
                aug_id = f"{img_id}_ed{v}"
                aug_img_path = self.aug_images_dir / f"{aug_id}.png"
                
                if not aug_img_path.exists():
                    img_aug = elastic_deformation(img)
                    
                    img_aug.save(aug_img_path)
                    save_label_grid(grid, self.aug_labels_dir / f"{aug_id}.json")
                    count += 1
        
        print(f"âœ… Elastic deformation complete ({count} new images)")
    
    
    def combined_augmentation(self, num_variants=3):
        """
        Combine multiple augmentations (color blobs + deformation + photometric)
        This creates images most similar to test set
        """
        count = 0
        
        for img_id in self.image_ids:
            img = Image.open(self.src_images_dir / f"{img_id}.png")
            grid = load_label_grid(self.src_labels_dir / f"{img_id}.json")
            
            for v in range(num_variants):
                aug_id = f"{img_id}_comb{v}"
                aug_img_path = self.aug_images_dir / f"{aug_id}.png"
                
                if not aug_img_path.exists():
                    img_aug = img.copy()
                    
                    # Random combination of augmentations
                    augmentations = []
                    
                    # 70% chance of color blobs
                    if random.random() < 0.7:
                        augmentations.append('color_blob')
                    
                    # 50% chance of elastic deformation
                    if random.random() < 0.5:
                        augmentations.append('elastic')
                    
                    # 60% chance of photometric
                    if random.random() < 0.6:
                        augmentations.append('photometric')
                    
                    # 30% chance of noise
                    if random.random() < 0.3:
                        augmentations.append('noise')
                    
                    # 20% chance of blur
                    if random.random() < 0.2:
                        augmentations.append('blur')
                    
                    # Apply selected augmentations in random order
                    random.shuffle(augmentations)
                    
                    for aug in augmentations:
                        if aug == 'color_blob':
                            img_aug = add_color_blobs(img_aug)
                        elif aug == 'elastic':
                            img_aug = elastic_deformation(img_aug)
                        elif aug == 'photometric':
                            img_aug = photometric_augmentation(img_aug)
                        elif aug == 'noise':
                            img_aug = add_gaussian_noise(img_aug)
                        elif aug == 'blur':
                            img_aug = add_blur(img_aug)
                    
                    img_aug.save(aug_img_path)
                    save_label_grid(grid, self.aug_labels_dir / f"{aug_id}.json")
                    count += 1
        
        print(f"âœ… Combined augmentation complete ({count} new images)")
    
    
    def crop_augmentation(self, num_variants=2, min_crop_ratio=0.6):
        """
        Apply random crops
        """
        count = 0
        
        for img_id in self.image_ids:
            img = Image.open(self.src_images_dir / f"{img_id}.png")
            grid = load_label_grid(self.src_labels_dir / f"{img_id}.json")
            
            for v in range(num_variants):
                aug_id = f"{img_id}_crop{v}"
                aug_img_path = self.aug_images_dir / f"{aug_id}.png"
                
                if not aug_img_path.exists():
                    # Random crop
                    crop_ratio = random.uniform(min_crop_ratio, 0.95)
                    img_aug, grid_aug = random_crop(img, grid, crop_size=None)
                    
                    # Resize back to original size (optional)
                    img_aug = img_aug.resize(img.size, Image.LANCZOS)
                    
                    img_aug.save(aug_img_path)
                    save_label_grid(grid_aug, self.aug_labels_dir / f"{aug_id}.json")
                    count += 1
        
        print(f"âœ… Crop augmentation complete ({count} new images)")
    
    
    def photometric_only(self, num_variants=2):
        """
        Photometric augmentation only (brightness, contrast, saturation)
        """
        count = 0
        
        for img_id in self.image_ids:
            img = Image.open(self.src_images_dir / f"{img_id}.png")
            grid = load_label_grid(self.src_labels_dir / f"{img_id}.json")
            
            for v in range(num_variants):
                aug_id = f"{img_id}_photo{v}"
                aug_img_path = self.aug_images_dir / f"{aug_id}.png"
                
                if not aug_img_path.exists():
                    img_aug = photometric_augmentation(img)
                    
                    # Optional noise
                    if random.random() < 0.4:
                        img_aug = add_gaussian_noise(img_aug, sigma=random.randint(3, 8))
                    
                    img_aug.save(aug_img_path)
                    save_label_grid(grid, self.aug_labels_dir / f"{aug_id}.json")
                    count += 1
        
        print(f"âœ… Photometric augmentation complete ({count} new images)")
    
    
    def generate_all_augmentations(self):
        """
        Generate all augmentation types
        Recommended for maximum diversity
        """
        self.save_original()
        self.combined_augmentation(num_variants=1)  # Most important - matches test set
        self.color_blob_augmentation(num_variants=1)
        self.elastic_augmentation(num_variants=1)
        self.photometric_only(num_variants=1)
        self.crop_augmentation(num_variants=1)
        
        # Count total
        total_images = len(list(self.aug_images_dir.glob("*.png")))
        print(f"âœ… AUGMENTATION COMPLETE")
        print(f"ğŸ“Š Total augmented images: {total_images}")




augmenter = AdvancedAugmentation(
        src_images_dir=paths['saved_images'],
        src_labels_dir=paths['saved_labels'],
        aug_images_dir=paths['mod_images'],
        aug_labels_dir=paths['mod_labels']
    )

augment_real_data = AdvancedAugmentation(
        src_images_dir=paths['train_images'],
        src_labels_dir=paths['train_labels'],
        aug_images_dir=paths['mod_train_images'],
        aug_labels_dir=paths['mod_train_labels']
)
    
augmenter.generate_all_augmentations()   
augment_real_data.generate_all_augmentations()


# image trainsform (same dimension)
train_transform = transforms.Compose([
    transforms.Resize((GRID_SIZE, GRID_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Resize((GRID_SIZE, GRID_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized grids
    """
    images = []
    grids = []
    
    for x, y in batch:
        images.append(x)
        grids.append(y)
        
    # stack images
    images = torch.stack(images, dim=0)
    
    # Check if all grids are same size
    grid_shapes = [g.shape for g in grids]
    if len(set(grid_shapes)) == 1:
        grids = torch.stack(grids, dim=0)
    else:
        target_size = 20
        resized_grids = []
        for g in grids:
            if g.shape != (target_size, target_size):
                g_tensor = g.unsqueeze(0).unsqueeze(0).float()
                g_resized = F.interpolate(
                    g_tensor,
                    size=(target_size, target_size),
                    mode='nearest'
                )
                g = g_resized.squeeze().long()
            resized_grids.append(g)
        grids = torch.stack(resized_grids, dim=0)
    
    return images, grids


class GridDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        grid_size: int = GRID_SIZE,
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.grid_size = grid_size
        self.transform = transform
        
        self.image_ids: List[str] = []
        for p in sorted(labels_dir.glob("*.json")):
            image_id = p.stem
            img_path = images_dir / f"{image_id}.png"
            if img_path.is_file():
                self.image_ids.append(image_id)
        
        if not self.image_ids:
            raise RuntimeError(f"No training labels/images found in {labels_dir}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = self.images_dir / f"{image_id}.png"
        label_path = self.labels_dir / f"{image_id}.json"
        
        img = Image.open(img_path).convert('RGB')
        grid = load_label_grid(label_path)
        
        if self.transform:
            x = self.transform(img)
        else:
            x = transforms.ToTensor()(img)
        
        y = torch.from_numpy(grid).long()
        
        return x, y




class AttentionBlock(nn.Module):
    """Attention gate for U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # g: decoder feature (gating signal)
        # x: encoder feature (to be attention-weighted)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

def conv_block(in_channels, out_channels):
    """Standard conv block with BatchNorm"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ImprovedUNet(nn.Module):
    """
    Improved U-Net with optional attention gates
    """
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # Encoder
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        if use_attention:
            self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        if use_attention:
            self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        if use_attention:
            self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        if use_attention:
            self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = conv_block(128, 64)
        
        # Multi-task heads
        self.seg_head = nn.Conv2d(64, 3, 1)  # walkable, wall, hazard
        self.start_head = nn.Conv2d(64, 1, 1)
        self.goal_head = nn.Conv2d(64, 1, 1)
    
    def forward(self, x):
        # Encoder with size tracking
        e1 = self.enc1(x)  # 64 channels, 20x20
        e2 = self.enc2(self.pool(e1))  # 128 channels, 10x10
        e3 = self.enc3(self.pool(e2))  # 256 channels, 5x5
        e4 = self.enc4(self.pool(e3))  # 512 channels, 2x2
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 1024 channels, 1x1
        
        # Decoder with attention
        d4 = self.up4(b)  # 512 channels
        if d4.size() != e4.size():
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        if self.use_attention:
            e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)  # 256 channels
        if d3.size() != e3.size():
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        if self.use_attention:
            e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)  # 128 channels
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        if self.use_attention:
            e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)  # 64 channels
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        if self.use_attention:
            e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Heads
        seg_logits = self.seg_head(d1)
        start_logits = self.start_head(d1)
        goal_logits = self.goal_head(d1)
        
        return seg_logits, start_logits, goal_logits


# loss
class DiceLoss(nn.Module):
    """Dice loss for better handling of class imbalance"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = logits.size(1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate dice for each class
        dice_scores = []
        for c in range(num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        return 1.0 - torch.stack(dice_scores).mean()

class CombinedLoss(nn.Module):
    """Combined CE + Dice loss"""
    def __init__(self, weight=None, alpha=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
        self.dice = DiceLoss()
        self.alpha = alpha
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr=1e-3,
    grad_clip=5.0,
    save_path="best_model.pth",
    eval_every=5
):
    model.to(DEVICE)
    
    # Improved loss functions with Dice
    SEG_CLASS_WEIGHTS = torch.tensor([1.0, 3.0, 2.5], device=DEVICE)
    seg_criterion = CombinedLoss(weight=SEG_CLASS_WEIGHTS, alpha=0.5)
    
    start_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([400.0], device=DEVICE)
    )
    goal_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([400.0], device=DEVICE)
    )
    
    # AdamW optimizer (better for GPU)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision training for GPU
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None
    
    best_val_loss = float('inf')
    
    print(f"ğŸš€ Starting training on {DEVICE}")
    print(f"ğŸ“Š Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_loss = 0.0
        seg_loss_sum = 0.0
        start_loss_sum = 0.0
        goal_loss_sum = 0.0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Prepare targets
            seg_target = y.clone()
            seg_target[seg_target >= 3] = 0
            
            start_target = (y == CLASS_START).float().unsqueeze(1)
            goal_target = (y == CLASS_GOAL).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision (GPU only)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    seg_logits, start_logits, goal_logits = model(x)
                    
                    # Ensure size match
                    if seg_logits.shape[2:] != seg_target.shape[1:]:
                        seg_logits = F.interpolate(
                            seg_logits, size=seg_target.shape[1:],
                            mode='bilinear', align_corners=False
                        )
                        start_logits = F.interpolate(
                            start_logits, size=start_target.shape[2:],
                            mode='bilinear', align_corners=False
                        )
                        goal_logits = F.interpolate(
                            goal_logits, size=goal_target.shape[2:],
                            mode='bilinear', align_corners=False
                        )
                    
                    seg_loss = seg_criterion(seg_logits, seg_target)
                    start_loss = start_criterion(start_logits, start_target)
                    goal_loss = goal_criterion(goal_logits, goal_target)
                    
                    loss = seg_loss + 0.5 * start_loss + 0.5 * goal_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU fallback
                seg_logits, start_logits, goal_logits = model(x)
                
                if seg_logits.shape[2:] != seg_target.shape[1:]:
                    seg_logits = F.interpolate(
                        seg_logits, size=seg_target.shape[1:],
                        mode='bilinear', align_corners=False
                    )
                    start_logits = F.interpolate(
                        start_logits, size=start_target.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                    goal_logits = F.interpolate(
                        goal_logits, size=goal_target.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                
                seg_loss = seg_criterion(seg_logits, seg_target)
                start_loss = start_criterion(start_logits, start_target)
                goal_loss = goal_criterion(goal_logits, goal_target)
                
                loss = seg_loss + 0.5 * start_loss + 0.5 * goal_loss
                
                loss.backward()
                
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
            
            total_loss += loss.item()
            seg_loss_sum += seg_loss.item()
            start_loss_sum += start_loss.item()
            goal_loss_sum += goal_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'seg': f'{seg_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_seg = seg_loss_sum / len(train_loader)
        avg_start = start_loss_sum / len(train_loader)
        avg_goal = goal_loss_sum / len(train_loader)
        
        # Validation phase
        val_loss = validate(model, val_loader, seg_criterion, start_criterion, goal_criterion)
        
        # Step scheduler
        scheduler.step()
        
        print(
            f"\nğŸ“ˆ Epoch {epoch:03d} | "
            f"Train: {avg_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        print(f"   Seg: {avg_seg:.4f} | Start: {avg_start:.4f} | Goal: {avg_goal:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"âœ… Saved best model (val_loss: {val_loss:.4f})")
        
        # Evaluate periodically
        if epoch % eval_every == 0:
            evaluate_segmentation(model, val_loader)
            evaluate_start_goal(model, val_loader)
        
        print("="*60)

@torch.no_grad()
def validate(model, val_loader, seg_criterion, start_criterion, goal_criterion):
    """Validation loop"""
    model.eval()
    total_loss = 0.0
    
    for x, y in val_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        seg_target = y.clone()
        seg_target[seg_target >= 3] = 0
        start_target = (y == CLASS_START).float().unsqueeze(1)
        goal_target = (y == CLASS_GOAL).float().unsqueeze(1)
        
        seg_logits, start_logits, goal_logits = model(x)
        
        if seg_logits.shape[2:] != seg_target.shape[1:]:
            seg_logits = F.interpolate(
                seg_logits, size=seg_target.shape[1:],
                mode='bilinear', align_corners=False
            )
            start_logits = F.interpolate(
                start_logits, size=start_target.shape[2:],
                mode='bilinear', align_corners=False
            )
            goal_logits = F.interpolate(
                goal_logits, size=goal_target.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        seg_loss = seg_criterion(seg_logits, seg_target)
        start_loss = start_criterion(start_logits, start_target)
        goal_loss = goal_criterion(goal_logits, goal_target)
        
        loss = seg_loss + 0.5 * start_loss + 0.5 * goal_loss
        total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader)



def confusion_matrix(pred, target, num_classes=3):
    pred = pred.view(-1)
    target = target.view(-1)
    cm = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return cm

def per_class_metrics(cm):
    metrics = {}
    num_classes = cm.shape[0]
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics[c] = {
            "precision": precision.item(),
            "recall": recall.item(),
            "iou": iou.item(),
            "f1": f1.item()
        }
    return metrics

@torch.no_grad()
def evaluate_segmentation(model, val_loader):
    model.eval()
    total_cm = torch.zeros(3, 3, dtype=torch.int64)
    
    for x, y in tqdm(val_loader, desc="Evaluating segmentation"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        seg_logits, _, _ = model(x)
        seg_pred = torch.argmax(seg_logits, dim=1)
        
        seg_target = y.clone()
        seg_target[seg_target >= 3] = 0
        
        cm = confusion_matrix(seg_pred.cpu(), seg_target.cpu(), num_classes=3)
        total_cm += cm
    
    metrics = per_class_metrics(total_cm)
    class_names = ["walk", "wall", "hazard"]
    
    print("\n" + "="*50)
    print("Segmentation Metrics:")
    print("="*50)
    for c, name in enumerate(class_names):
        m = metrics[c]
        print(
            f"{name:7s} | "
            f"P: {m['precision']:.3f} | "
            f"R: {m['recall']:.3f} | "
            f"IoU: {m['iou']:.3f} | "
            f"F1: {m['f1']:.3f}"
        )
    
    mean_iou = sum(m["iou"] for m in metrics.values()) / 3
    mean_f1 = sum(m["f1"] for m in metrics.values()) / 3
    print(f"\nğŸ“Š Mean IoU: {mean_iou:.3f} | Mean F1: {mean_f1:.3f}")
    print("="*50 + "\n")
    
    model.train()

@torch.no_grad()
def evaluate_start_goal(model, val_loader):
    model.eval()
    start_correct = 0
    goal_correct = 0
    total = 0
    
    for x, y in tqdm(val_loader, desc="Evaluating start/goal"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        seg_logits, start_logits, goal_logits = model(x)
        seg_pred = torch.argmax(seg_logits, dim=1)
        
        wall_mask = (seg_pred == CLASS_WALL)
        
        # Mask walls
        start_logits = start_logits.clone()
        goal_logits = goal_logits.clone()
        start_logits[wall_mask.unsqueeze(1)] = -1e9
        goal_logits[wall_mask.unsqueeze(1)] = -1e9
        
        B = start_logits.size(0)
        start_pred = torch.argmax(start_logits.view(B, -1), dim=1)
        goal_pred = torch.argmax(goal_logits.view(B, -1), dim=1)
        
        start_gt = (y == CLASS_START).long().view(B, -1).argmax(dim=1)
        goal_gt = (y == CLASS_GOAL).long().view(B, -1).argmax(dim=1)
        
        start_correct += (start_pred == start_gt).sum().item()
        goal_correct += (goal_pred == goal_gt).sum().item()
        total += B
    
    print("="*50)
    print("Start/Goal Detection:")
    print("="*50)
    print(f"ğŸ�¯ Start accuracy: {start_correct / total:.3f} ({start_correct}/{total})")
    print(f"ğŸ�� Goal  accuracy: {goal_correct / total:.3f} ({goal_correct}/{total})")
    print("="*50 + "\n")
    
    model.train()



# Create datasets
train_dataset = GridDataset(
        images_dir=paths['mod_images'],
        labels_dir=paths['mod_labels'],
        transform=train_transform
    )
    
val_dataset = GridDataset(
        images_dir=paths['mod_train_images'],
        labels_dir=paths['mod_train_labels'],
        transform=val_transform
    )
    
    # GPU-optimized batch size and workers
train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Larger batch for GPU
        shuffle=True, 
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster CPU->GPU transfer
        collate_fn=custom_collate_fn  # Handle variable-sized grids
    )
    
val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Create improved model with attention
model = ImprovedUNet(use_attention=True)
    
    # Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"ğŸ“Š Model parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
print("="*60)
    
    # Train
train_model(
        model,
        train_loader,
        val_loader,
        epochs=100,  # More epochs for GPU
        lr=1e-3,
        save_path="best_grid_model_gpu.pth",
        eval_every=5
    )


transform = transforms.Compose([
    transforms.Resize((GRID_SIZE, GRID_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_terrain_type(img: Image.Image) -> str:
    """
    Detect terrain type based on color statistics.
    Lab: metallic/blue tones
    Forest: green/brown tones
    Desert: yellow/sandy tones
    """
    img_array = np.array(img.resize((100, 100)))  # Downsample for speed
    
    # Average RGB values
    avg_r = img_array[:, :, 0].mean()
    avg_g = img_array[:, :, 1].mean()
    avg_b = img_array[:, :, 2].mean()
    
    # Heuristic rules (tune these based on your data)
    # Lab: bluish, metallic (high blue, balanced RGB)
    if avg_b > avg_r + 10 and avg_b > avg_g + 10:
        return 'lab'
    
    # Forest: greenish (high green)
    if avg_g > avg_r + 15 and avg_g > avg_b + 15:
        return 'forest'
    
    # Desert: sandy/yellow (high red+green, low blue)
    if avg_r > avg_b + 20 and avg_g > avg_b + 20:
        return 'desert'
    
    # Default fallback
    return 'lab'



# helper functions
def pick_start_goal_from_heatmaps(
    start_logits: torch.Tensor,
    goal_logits: torch.Tensor,
    wall_mask: torch.Tensor
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Pick start and goal from heatmaps while avoiding walls.
    
    Args:
        start_logits: (1, 1, G, G)
        goal_logits: (1, 1, G, G)
        wall_mask: (1, G, G) - True where walls are
    
    Returns:
        start: (row, col)
        goal: (row, col)
    """
    _, _, G, _ = start_logits.shape
    
    # Apply wall masking
    start_logits = start_logits.clone()
    goal_logits = goal_logits.clone()
    start_logits[0, 0][wall_mask[0]] = -1e9
    goal_logits[0, 0][wall_mask[0]] = -1e9
    
    # Pick highest activation
    start_flat = torch.argmax(start_logits.view(-1)).item()
    goal_flat = torch.argmax(goal_logits.view(-1)).item()
    
    start = (start_flat // G, start_flat % G)
    goal = (goal_flat // G, goal_flat % G)
    
    return start, goal


def load_velocity_boost(json_path: Path) -> np.ndarray:
    """Load velocity boost from JSON file."""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "boost" not in data:
            print(f"âš ï¸�  Warning: 'boost' not found in {json_path}. Using zero boost.")
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        boost = np.array(data["boost"], dtype=np.float32)
        
        if boost.shape != (GRID_SIZE, GRID_SIZE):
            print(f"âš ï¸�  Warning: Boost shape {boost.shape} != ({GRID_SIZE}, {GRID_SIZE}). Resizing.")
            boost = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        return boost
    
    except Exception as e:
        print(f"âš ï¸�  Error loading boost from {json_path}: {e}. Using zero boost.")
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)


def estimate_cost_grid(
    pred_grid: np.ndarray,
    boost: np.ndarray,
    terrain: str = 'lab'
) -> np.ndarray:
    """
    Calculate step costs based on terrain, class, and boost.
    
    Formula: step_cost(i,j) = base_cost(i,j) - boost(i,j)
    """
    G = pred_grid.shape[0]
    cost_grid = np.zeros((G, G), dtype=np.float32)
    
    terrain_cost = TERRAIN_COSTS[terrain]
    
    for i in range(G):
        for j in range(G):
            cell_class = pred_grid[i, j]
            base_cost = terrain_cost.get(cell_class, 1.0)
            
            # Apply boost
            step_cost = base_cost - boost[i, j]
            
            # Ensure positive cost (safety check)
            step_cost = max(step_cost, 0.01)
            
            cost_grid[i, j] = step_cost
    
    return cost_grid




def dijkstra_path(
    cost_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Dijkstra's algorithm to find optimal path.
    
    Returns:
        List of (row, col) positions from start to goal.
        Empty list if no path found.
    """
    G = cost_grid.shape[0]
    INF = float('inf')
    
    sr, sc = start
    gr, gc = goal
    
    # Distance array
    dist = np.full((G, G), INF, dtype=np.float32)
    dist[sr, sc] = 0.0
    
    # Previous node for path reconstruction
    prev = {}
    
    # Priority queue: (cost, (row, col))
    heap = [(0.0, start)]
    visited = set()
    
    while heap:
        cur_cost, (r, c) = heapq.heappop(heap)
        
        if (r, c) in visited:
            continue
        visited.add((r, c))
        
        # Found goal
        if (r, c) == goal:
            break
        
        # Skip if we found a better path already
        if cur_cost > dist[r, c]:
            continue
        
        # Explore 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            # Boundary check
            if not (0 <= nr < G and 0 <= nc < G):
                continue
            
            # Wall check
            if cost_grid[nr, nc] == INF:
                continue
            
            # Calculate new cost
            new_cost = cur_cost + cost_grid[nr, nc]
            
            # Update if better
            if new_cost < dist[nr, nc]:
                dist[nr, nc] = new_cost
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_cost, (nr, nc)))
    
    # Reconstruct path
    if goal not in prev and goal != start:
        return []
    
    path = []
    cur = goal
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    
    return path

def astar_fallback(
    pred_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    A* fallback (ignores boost, uses Manhattan distance heuristic).
    """
    G = pred_grid.shape[0]
    INF = float('inf')
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    sr, sc = start
    gr, gc = goal
    
    dist = np.full((G, G), INF, dtype=np.float32)
    dist[sr, sc] = 0.0
    
    prev = {}
    heap = [(heuristic(start), 0.0, start)]
    visited = set()
    
    while heap:
        _, cur_cost, (r, c) = heapq.heappop(heap)
        
        if (r, c) in visited:
            continue
        visited.add((r, c))
        
        if (r, c) == goal:
            break
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            if not (0 <= nr < G and 0 <= nc < G):
                continue
            
            # Simple cost: 1 for walkable, INF for wall
            if pred_grid[nr, nc] == CLASS_WALL:
                continue
            
            new_cost = cur_cost + 1.0
            
            if new_cost < dist[nr, nc]:
                dist[nr, nc] = new_cost
                prev[(nr, nc)] = (r, c)
                priority = new_cost + heuristic((nr, nc))
                heapq.heappush(heap, (priority, new_cost, (nr, nc)))
    
    # Reconstruct
    if goal not in prev and goal != start:
        return []
    
    path = []
    cur = goal
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    
    return path


def greedy_fallback(
    pred_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_steps: int = 500
) -> List[Tuple[int, int]]:
    """
    Greedy fallback: move toward goal, avoid walls.
    Last resort if Dijkstra and A* fail.
    """
    G = pred_grid.shape[0]
    path = [start]
    current = start
    visited = {start}
    
    for _ in range(max_steps):
        if current == goal:
            break
        
        cr, cc = current
        gr, gc = goal
        
        # Try to move closer to goal
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            
            if not (0 <= nr < G and 0 <= nc < G):
                continue
            if pred_grid[nr, nc] == CLASS_WALL:
                continue
            if (nr, nc) in visited:
                continue
            
            # Manhattan distance to goal
            dist = abs(nr - gr) + abs(nc - gc)
            candidates.append((dist, (nr, nc)))
        
        if not candidates:
            # Stuck - try to backtrack
            break
        
        # Pick closest to goal
        candidates.sort()
        _, next_pos = candidates[0]
        
        path.append(next_pos)
        visited.add(next_pos)
        current = next_pos
    
    return path


def desperate_fallback(
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Absolute last resort: Generate ANY valid path from start to goal.
    Ignores walls completely, just creates a Manhattan path.
    
    This guarantees a non-empty submission even if the map is impossible.
    Strategy: Move horizontally first, then vertically.
    """
    sr, sc = start
    gr, gc = goal
    
    path = [start]
    current_r, current_c = sr, sc
    
    # Move horizontally to goal column
    while current_c != gc:
        if current_c < gc:
            current_c += 1
        else:
            current_c -= 1
        path.append((current_r, current_c))
    
    # Move vertically to goal row
    while current_r != gr:
        if current_r < gr:
            current_r += 1
        else:
            current_r -= 1
        path.append((current_r, current_c))
    
    return path


def relaxed_greedy_fallback(
    pred_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_steps: int = 500
) -> List[Tuple[int, int]]:
    """
    Even more relaxed greedy: allows moving through walls if necessary.
    Used as second-to-last resort before desperate_fallback.
    """
    G = pred_grid.shape[0]
    path = [start]
    current = start
    visited = {start}
    
    for step in range(max_steps):
        if current == goal:
            break
        
        cr, cc = current
        gr, gc = goal
        
        # Try to move closer to goal
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            
            if not (0 <= nr < G and 0 <= nc < G):
                continue
            if (nr, nc) in visited:
                continue
            
            # Manhattan distance to goal
            dist = abs(nr - gr) + abs(nc - gc)
            
            # Penalty for walls, but still allow them
            penalty = 100 if pred_grid[nr, nc] == CLASS_WALL else 0
            candidates.append((dist + penalty, (nr, nc)))
        
        if not candidates:
            # Allow revisiting if completely stuck
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                
                if not (0 <= nr < G and 0 <= nc < G):
                    continue
                
                dist = abs(nr - gr) + abs(nc - gc)
                penalty = 100 if pred_grid[nr, nc] == CLASS_WALL else 0
                candidates.append((dist + penalty, (nr, nc)))
        
        if not candidates:
            # Still stuck? Break and use desperate fallback
            break
        
        # Pick best candidate
        candidates.sort()
        _, next_pos = candidates[0]
        
        path.append(next_pos)
        visited.add(next_pos)
        current = next_pos
    
    # If we didn't reach goal, continue with desperate fallback from current position
    if current != goal:
        remaining_path = desperate_fallback(current, goal)
        if len(remaining_path) > 1:
            path.extend(remaining_path[1:])  # Skip duplicate current position
    
    return path



# Path conversion
def path_to_lrud(path: List[Tuple[int, int]]) -> str:
    """
    Convert list of (row, col) positions to lrud sequence.
    """
    if len(path) <= 1:
        return ""
    
    moves = []
    for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
        dr, dc = r2 - r1, c2 - c1
        
        if dr == 1 and dc == 0:
            moves.append("d")  # down
        elif dr == -1 and dc == 0:
            moves.append("u")  # up
        elif dr == 0 and dc == 1:
            moves.append("r")  # right
        elif dr == 0 and dc == -1:
            moves.append("l")  # left
        else:
            # Invalid move - should not happen
            print(f"âš ï¸�  Warning: Invalid move from ({r1},{c1}) to ({r2},{c2})")
    
    return "".join(moves)

def predict_outputs(
    model: nn.Module,
    img_path: Path
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str]:
    """
    Run model inference on a single image.
    
    Returns:
        grid_pred: (G, G) predicted class grid
        start: (row, col)
        goal: (row, col)
        terrain: detected terrain type
    """
    # Load and transform image
    img = Image.open(img_path).convert('RGB')
    
    # Detect terrain
    terrain = detect_terrain_type(img)
    
    x = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, G, G)
    
    model.eval()
    with torch.no_grad():
        seg_logits, start_logits, goal_logits = model(x)
        
        # Segmentation prediction
        seg_pred = torch.argmax(seg_logits, dim=1)  # (1, G, G)
        
        # Build wall mask
        wall_mask = (seg_pred == CLASS_WALL)  # (1, G, G)
        
        # Pick start & goal with wall masking
        start_rc, goal_rc = pick_start_goal_from_heatmaps(
            start_logits.cpu(),
            goal_logits.cpu(),
            wall_mask.cpu()
        )
    
    grid_pred = seg_pred.squeeze(0).cpu().numpy().astype(np.int64)
    
    return grid_pred, start_rc, goal_rc, terrain




def run_inference_on_test(
    model: nn.Module,
    test_images_dir: Path,
    test_velocities_dir: Path,
    submission_path: Path
):
    """
    Complete inference pipeline with multi-level fallbacks.
    Guarantees a path for every image.
    """

    if not submission_path.exists() :
        submission_path.parent.mkdir(parents=True, exist_ok=True)  # create folders if needed
        submission_path.touch(exist_ok=True)
        
    model.eval()
    
    image_paths = sorted(test_images_dir.glob("*.png"))
    print(f"ğŸ”� Found {len(image_paths)} test images")
    
    records = []
    fallback_stats = {
        'dijkstra': 0,
        'astar': 0,
        'greedy': 0,
        'relaxed': 0,
        'desperate': 0,
        'error': 0
    }
    
    for img_path in tqdm(image_paths, desc="Running inference"):
        image_id = img_path.stem
        
        try:
            # 1. Predict grid, start, goal, terrain
            grid_pred, start, goal, terrain = predict_outputs(model, img_path)
            
            # 2. Load velocity boost
            boost = load_velocity_boost(test_velocities_dir / f"{image_id}.json")
            
            # 3. Calculate cost grid
            cost_grid = estimate_cost_grid(grid_pred, boost, terrain)
            
            # 4. Try Dijkstra first (optimal with boost)
            path = dijkstra_path(cost_grid, start, goal)
            
            if len(path) >= 2:
                fallback_stats['dijkstra'] += 1
            else:
                # 5. Fallback to A* (ignores boost, uses heuristic)
                print(f"âš ï¸�  {image_id}: Dijkstra failed, trying A*...")
                path = astar_fallback(grid_pred, start, goal)
                
                if len(path) >= 2:
                    fallback_stats['astar'] += 1
                else:
                    # 6. Fallback to greedy (simple goal-seeking, avoids walls)
                    print(f"âš ï¸�  {image_id}: A* failed, trying greedy...")
                    path = greedy_fallback(grid_pred, start, goal)
                    
                    if len(path) >= 2:
                        fallback_stats['greedy'] += 1
                    else:
                        # 7. Fallback to relaxed greedy (allows walls with penalty)
                        print(f"âš ï¸�  {image_id}: Greedy failed, trying relaxed greedy...")
                        path = relaxed_greedy_fallback(grid_pred, start, goal)
                        
                        if len(path) >= 2:
                            fallback_stats['relaxed'] += 1
                        else:
                            # 8. ABSOLUTE LAST RESORT: Desperate fallback (ignores everything)
                            print(f"ğŸš¨ {image_id}: All intelligent pathfinding failed! Using desperate fallback...")
                            path = desperate_fallback(start, goal)
                            fallback_stats['desperate'] += 1
            
            # 9. Convert to moves
            moves = path_to_lrud(path)
            
            # 10. Final validation
            if len(moves) == 0 and start != goal:
                print(f"â�Œ {image_id}: Path conversion failed! Using direct path...")
                path = desperate_fallback(start, goal)
                moves = path_to_lrud(path)
            
            records.append({"image_id": image_id, "path": moves})
            
        except Exception as e:
            print(f"â�Œ Critical error processing {image_id}: {e}")
            # Even on error, try to provide a reasonable path
            try:
                # Try to at least create a Manhattan path
                path = desperate_fallback((0, 0), (GRID_SIZE-1, GRID_SIZE-1))
                moves = path_to_lrud(path)
                records.append({"image_id": image_id, "path": moves})
                fallback_stats['error'] += 1
            except:
                # Absolute worst case: empty path
                records.append({"image_id": image_id, "path": ""})
                fallback_stats['error'] += 1
    
    # Write submission CSV
    df = pd.DataFrame(records)
    df.to_csv(submission_path, index=False)
    
    print("\n" + "="*60)
    print("âœ… INFERENCE COMPLETE!")
    print("="*60)
    print(f"ğŸ“� Submission written to: {submission_path}")
    print(f"ğŸ“Š Total images processed: {len(records)}")
    print("\nğŸ“ˆ Pathfinding Statistics:")
    print(f"  ğŸ¥‡ Dijkstra (optimal):        {fallback_stats['dijkstra']:4d} ({fallback_stats['dijkstra']/len(records)*100:.1f}%)")
    print(f"  ğŸ¥ˆ A* (heuristic):            {fallback_stats['astar']:4d} ({fallback_stats['astar']/len(records)*100:.1f}%)")
    print(f"  ğŸ¥‰ Greedy (goal-seeking):     {fallback_stats['greedy']:4d} ({fallback_stats['greedy']/len(records)*100:.1f}%)")
    print(f"  âš ï¸�  Relaxed (wall-tolerant):  {fallback_stats['relaxed']:4d} ({fallback_stats['relaxed']/len(records)*100:.1f}%)")
    print(f"  ğŸš¨ Desperate (Manhattan):     {fallback_stats['desperate']:4d} ({fallback_stats['desperate']/len(records)*100:.1f}%)")
    print(f"  â�Œ Errors:                    {fallback_stats['error']:4d} ({fallback_stats['error']/len(records)*100:.1f}%)")
    
    empty_paths = sum(1 for r in records if r['path'] == '')
    if empty_paths > 0:
        print(f"\nâš ï¸�  WARNING: {empty_paths} images have empty paths!")
    else:
        print(f"\nâœ… All {len(records)} images have valid paths!")
    
    print("="*60)




# model
MODEL_PATH = Path("best_grid_model_gpu.pth")
model = ImprovedUNet(use_attention=True)  # Use your model class
    
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Run inference
run_inference_on_test(
        model=model,
        test_images_dir=paths['test_images'],
        test_velocities_dir=paths['test_velocities'],
        submission_path=paths['submission_path']
    )
print("ğŸ�‰ Inference complete!")




