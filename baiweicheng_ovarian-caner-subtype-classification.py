# Installs the C library without showing output
!apt-get update &> /dev/null && apt-get install -y libvips &> /dev/null
# Installs the Python wrapper without showing output
!pip install -q --no-cache-dir pyvips &> /dev/null


import cv2
import numpy as np
# from openslide import OpenSlide
import openslide
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocessing import Pool
from tqdm import tqdm
import os
import pandas as pd
import random
from sklearn.neighbors import KernelDensity
from PIL import Image

def get_sampled_points_density_proportional_KDE(points, desired_sample_size):
    num_points = len(points)
    if num_points <= desired_sample_size:
        return points

    points_arr = np.array(points)
    
    # Fit KDE model to the points
    kde = KernelDensity(bandwidth=0.1)  # You can adjust the bandwidth
    kde.fit(points_arr)

    # Generate samples from the KDE
    samples = kde.sample(desired_sample_size)
    final_sample = samples.tolist()

    return final_sample


def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,3)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,:,0] / (D) - 1.0
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,3)
    cx = np.expand_dims(cx,3)
    cy = np.expand_dims(cy,3)
            
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD


def clean_thumbnail(thumbnail):
    # thumbnail_arr = np.asarray(thumbnail)
    
    # wthumbnail = np.zeros_like(thumbnail_arr)
    # wthumbnail[:, :, :] = thumbnail_arr[:, :, :]

    # thumbnail_std = np.std(wthumbnail, axis=2)
    # wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    # thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
    # kernel = np.ones((30,30),np.float32)/900
    # thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    # wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    # return wthumbnail
    
    # Change np.asarray to np.array to create a writable copy
    thumbnail_arr = np.array(thumbnail)

    # Add a grayscale threshold to remove non-white background
    gray_thumbnail = cv2.cvtColor(thumbnail_arr, cv2.COLOR_RGB2GRAY)
    _, grayscale_mask = cv2.threshold(gray_thumbnail, 220, 255, cv2.THRESH_BINARY)
    thumbnail_arr[grayscale_mask == 255] = [255, 255, 255]


    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]

    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std < 5] = (np.ones((1, 3), dtype="uint8") * 255)
    thumbnail_HSD = RGB2HSD(np.array([wthumbnail.astype('float32') / 255.]))[0]
    kernel = np.ones((30, 30), np.float32) / 900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:, :, 2], -1, kernel)
    wthumbnail[thumbnail_HSD_mean < 0.05] = (np.ones((1, 3), dtype="uint8") * 255)
    return wthumbnail

                
def is_far_enough(new_point, existing_points, min_distance):
    for point in existing_points:
        if np.sqrt((new_point[0] - point[0])**2 + (new_point[1] - point[1])**2) < min_distance:
            return False
    return True


def get_patch_locations(tissue_mask, cthumbnail,  mask_hratio, mask_wratio, tissue_threshold, stride):
    contours, mm = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cthumbnail.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Draw contours on the image
    
    image_with_rectangles = cthumbnail.copy()
    
    # Calculate the step size for the grid based on the stride
    step_w = int(mask_wratio * stride)
    step_h = int(mask_hratio * stride)
    
    patch_locations = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # plot the rectangles on the image_with_rectangles
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if w >= mask_wratio and h >= mask_hratio:
            for i in range(x, x + w - mask_wratio, step_w):
                for j in range(y, y + h - mask_hratio, step_h):
                    tissue_patch = tissue_mask[j:j + mask_hratio, i:i + mask_wratio]
                    # if np.sum(tissue_patch) / (mask_hratio ** 2) > tissue_threshold:
                    tissue_magnitude = np.count_nonzero(tissue_patch)/tissue_patch.size
                    if tissue_magnitude  >= tissue_threshold:
                        patch_locations.append(((i, j),tissue_magnitude))

    return patch_locations, image_with_contours, image_with_rectangles

def process_wsi(wsi_obj, wsi_path, thumbnail_path, is_tma, output_patch_size=1000, tissue_percent=0.9, returnSamples=30, stride=1):
    wsi_name = Path(wsi_path).stem + ".svs"

    if is_tma:
        thumbnail = Image.open(wsi_path)
        objective_power = 40
    else:
        thumbnail = Image.open(thumbnail_path)
        objective_power = 20
    
    cthumbnail = clean_thumbnail(thumbnail)
    tissue_mask = ((cthumbnail.mean(axis=2) != 255) * 255).astype(np.uint8)
    # Add morphological operations to clean the mask
    kernel = np.ones((5,5),np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations = 2)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    # try:
    #     objective_power = int(wsi_obj.properties['openslide.objective-power'])
    # except:
    #     objective_power = 20
         
    w, h = wsi_obj.dimensions
    mask_hratio = int((tissue_mask.shape[0] / h) * output_patch_size)
    mask_wratio = int((tissue_mask.shape[1] / w) * output_patch_size)
    # Ensure the step size is at least 1 pixel
    if mask_hratio == 0:
        mask_hratio = 1
    if mask_wratio == 0:
        mask_wratio = 1
    # print(f"mask_hratio is {mask_hratio} and mask_wratio is {mask_wratio}")
    # estimate the mask patch size given the size of the WSI, the size of the mask, and the output patch size
    mask_patch_size = int(output_patch_size / mask_wratio)
    
    Mask_to_WSI_ratioW = int(w / tissue_mask.shape[1])
    Mask_to_WSI_ratioH = int(h / tissue_mask.shape[0])
    
    patch_locations, image_with_contours, image_with_rectangles = get_patch_locations(tissue_mask, cthumbnail, mask_hratio, mask_wratio, tissue_percent, stride)
    # print(f"initially generated {len(patch_locations)} patch locations")
    min_distance = mask_hratio * 0.5  # Minimum distance between points

    filtered_patch_locations = []
    for (x, y), _ in patch_locations:
        if is_far_enough((x, y), filtered_patch_locations, min_distance):
            filtered_patch_locations.append((x, y))

    # print(f"after is_far_enough there are {len(filtered_patch_locations)} patch locations")
    filtered_patch_locations = get_sampled_points_density_proportional_KDE(filtered_patch_locations, returnSamples)

    scaled_patch_coordinates = []
    for (x, y) in filtered_patch_locations:
        scaled_patch_coordinates.append((int(x * Mask_to_WSI_ratioW), int(y * Mask_to_WSI_ratioH)))

    return scaled_patch_coordinates



import os
import torch
import torchvision.transforms as T
import openslide
import pyvips
import gc
from contextlib import contextmanager

class SlidePatchExtractor:
    def __init__(self, image_id, patch_size=224, mode='train', tissue_threshold=0.9, label='HGSC'):
        
        self.image_id = image_id
        self.patch_size = patch_size
        self.mode = mode
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.patch_size, self.patch_size), antialias=True),
            # T.Normalize(mean=[0.2585, 0.2556, 0.2506], std=[0.229, 0.224, 0.225])
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.tissue_threshold = tissue_threshold
        if label == 'HGSC':
            self.num_patches = 100
        elif label == 'EC':
            self.num_patches = 180
        elif label == 'CC':
            self.num_patches = 220
        elif label == 'LGSC':
            self.num_patches = 470
        elif label == 'MC':
            self.num_patches = 480
        else:
            self.num_patches = 256

        # self.train_transform = T.Compose([
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.RandomVerticalFlip(p=0.5),
        #     T.RandomRotation(degrees=45),
        #     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        #     T.ToTensor(),
        #     T.Resize((224, 224), antialias=True),
        #     T.Normalize(mean=[0.2585, 0.2556, 0.2506], std=[0.229, 0.224, 0.225])
        # ])

        # Define paths for the source WSI and its thumbnail
        self.source_path = os.path.join('/kaggle/input/UBC-OCEAN', f'{self.mode}_images', self.image_id + '.png')
        self.thumbnail_path = os.path.join('/kaggle/input/UBC-OCEAN', f'{self.mode}_thumbnails', self.image_id + '_thumbnail.png')
        
        try:
            temp_slide = openslide.open_slide(self.source_path)
            self.width, self.height = temp_slide.dimensions
            self.is_tma = self.width < 5000 and self.height < 5000
            standard_magnification = 20
            self.objective_power = 40 if self.is_tma else 20
            magnification_factor = self.objective_power / standard_magnification
            self.extraction_patch_size = int(self.patch_size * magnification_factor)
            
            self.stride = 1 if self.is_tma else 4
            
            self.patch_locations = process_wsi(
                wsi_obj=temp_slide,
                wsi_path=self.source_path,
                thumbnail_path=self.thumbnail_path,
                is_tma=self.is_tma,
                output_patch_size=patch_size,
                tissue_percent=tissue_threshold,
                returnSamples=self.num_patches,
                stride=self.stride
            )
        except openslide.OpenSlideError as e:
            print(f"Could not open slide {self.source_path}: {e}")
            self.patch_locations = []
            return
        finally:
            if 'temp_slide' in locals():
                temp_slide.close()
                del temp_slide
            # gc.collect()
    
    def __len__(self):
        """Returns the number of patches found for this slide."""
        return len(self.patch_locations)
    
    @contextmanager
    def _get_pyvips_slide(self):
        """Context manager for proper PyVIPS resource management"""
        pyvips_slide = None
        try:
            pyvips_slide = pyvips.Image.new_from_file(self.source_path)
            yield pyvips_slide
        finally:
            if pyvips_slide is not None:
                del pyvips_slide
                gc.collect()
    
    def get_all_patch_tensors(self):
        """
        Extracts all patches from the slide and returns them as a stacked tensor.
        """
        patch_tensors = []
        flat_feature_size = 3 * self.patch_size * self.patch_size
        if not self.patch_locations:
            # If no patches were found, return an empty tensor with the correct shape
            return torch.empty((0, flat_feature_size))

        with self._get_pyvips_slide() as pyvips_slide:
            for (x, y) in self.patch_locations:
                try:
                    # patch_image = self.slide.read_region(
                    #     (x, y), 0, (self.extraction_patch_size, self.extraction_patch_size)
                    # ).convert('RGB')
                    patch_image = pyvips_slide.crop(x, y, self.patch_size, self.patch_size).numpy()[..., :3]
                    patch_tensor = self.transform(patch_image)
                    patch_tensors.append(patch_tensor)
                except Exception as e:
                    print(f"Error reading patch at ({x},{y}) for slide {self.image_id}: {e}")
                    continue
        
        if not patch_tensors:
            return torch.empty((0, flat_feature_size))
            
        # Stack all patch tensors into a single 4D tensor (num_patches, 3, H, W)
        stacked_patches = torch.stack(patch_tensors)
        # Flatten the patch dimensions (3, H, W) into a single vector for each patch
        flattened_patches = torch.flatten(stacked_patches, start_dim=1)

        # Clean up intermediate tensors
        del patch_tensors, stacked_patches
        gc.collect()
        
        return flattened_patches
    
    def get_patch(self, idx):
        x, y = self.patch_locations[idx]
        with self._get_pyvips_slide() as pyvips_slide:
            patch_image = pyvips_slide.crop(x, y, self.patch_size, self.patch_size).numpy()[..., :3]
            # patch_image = self.slide.read_region((x, y), 0, (self.extraction_patch_size, self.extraction_patch_size)).convert('RGB')
            patch_tensor = self.transform(patch_image)
        return patch_tensor, patch_image


import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import openslide
from PIL import Image
import numpy as np
from tqdm import tqdm
import gc

class UBCDataset(Dataset):
    """
    The main Dataset class for loading slides and their labels.
    """
    def __init__(self, dataframe, label_map, mode='train', patch_size=224, tissue_threshold=0.9):
        self.df = dataframe
        self.label_map = label_map
        self.mode = mode
        self.patch_size = patch_size
        self.tissue_threshold = tissue_threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        
        string_label = row['label']
        # Use the label map to convert the string label to an integer
        int_label = self.label_map[string_label]
        # Create the tensor from the integer
        label = torch.tensor(int_label, dtype=torch.long)

        extractor = SlidePatchExtractor(
            image_id=image_id,
            mode=self.mode,
            patch_size=self.patch_size,
            tissue_threshold=self.tissue_threshold,
            label=string_label
        )
        
        patch_tensors = extractor.get_all_patch_tensors()

        # Clean up the extractor
        del extractor
        gc.collect()

        return {"patches": patch_tensors, "label": label, "image_id": image_id}


# --- Collate Function for the DataLoader ---
def collate_fn(batch):
    patches_list = [item['patches'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    image_ids = [item['image_id'] for item in batch]

    return {"patches": patches_list, "labels": labels, "image_ids": image_ids}


import psutil
import shutil
import pyvips
import gc
import os

def force_clear_vips_cache():
    """
    More aggressively clears pyvips caches and the dedicated temp subdirectory.
    """
    # 1. Tell pyvips to drop its operation cache
    pyvips.cache_set_max(0)
    
    # 2. Call Python's garbage collector
    gc.collect()

    # 3. Define the path to your dedicated temp directory
    vips_temp_dir = '/kaggle/temp/my_vips_temp/'
    
    # 4. Use shutil to safely remove and recreate only your subdirectory
    if os.path.exists(vips_temp_dir):
        shutil.rmtree(vips_temp_dir)
    os.makedirs(vips_temp_dir) # Recreate it for the next batch

    # 5. Reset the pyvips cache to a normal size
    pyvips.cache_set_max(100)

def print_memory_usage(prefix=""):
    """Prints current RAM usage of the system."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    ram_usage_gb = memory_info.rss / (1024 ** 3)  # Resident Set Size in GB
    
    # Get total system memory
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    # Get percentage usage
    memory_percent = process.memory_percent()
    
    print(
        f"{prefix} Memory Usage: "
        f"{ram_usage_gb:.2f} GB / {total_memory_gb:.2f} GB "
        f"({memory_percent:.2f}%)"
    )

def get_dir_size(path='.'):
    """
    Calculates the total size of all files in a directory and its subdirectories.
    """
    total = 0
    # Use a try-except block in case of permission errors
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
    except FileNotFoundError:
        return 0 # If directory doesn't exist, its size is 0
    return total

def print_disk_usage(path, prefix=""):
    """
    Prints the size of the specified directory's contents and the overall
    usage of the disk partition it resides on.
    """
    try:
        # --- Overall Partition Usage ---
        total_partition, used_partition, free_partition = shutil.disk_usage(path)
        
        # --- Specific Directory Size ---
        dir_contents_size = get_dir_size(path)

        # --- Convert to GB for printing ---
        total_partition_gb = total_partition / (1024**3)
        used_partition_gb = used_partition / (1024**3)
        dir_contents_gb = dir_contents_size / (1024**3)
        
        # --- THE FIX IS HERE ---
        # Strip trailing slashes from the path before getting the basename
        dir_name = os.path.basename(path.rstrip('/'))
        
        print(
            f"{prefix} Dir '{dir_name}' Size: {dir_contents_gb:.2f} GB. "
            f"Total Partition Usage: {used_partition_gb:.2f} GB / {total_partition_gb:.2f} GB"
        )
        
    except FileNotFoundError:
        # This can happen right after cleanup, so we handle it gracefully
        print(f"{prefix} Dir '{os.path.basename(path.rstrip('/'))}' Size: 0.00 GB. (Directory removed during cleanup)")
    except Exception as e:
        print(f"An error occurred while checking disk usage for {path}: {e}")


import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import gzip
import time

# Disable the Decompression Bomb check
Image.MAX_IMAGE_PIXELS = None

# Create a dedicated directory for pyvips temporary files
vips_temp_dir = '/kaggle/temp/my_vips_temp/'
os.makedirs(vips_temp_dir, exist_ok=True)

# Tell pyvips to use this directory
os.environ['VIPS_TMPDIR'] = vips_temp_dir

# Hyperparameters
batch_size = 1
patch_size = 224
num_patches = 256
tissue_threshold = 0.9

base_path = '/kaggle/input/UBC-OCEAN'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))

# Create a mapping from string labels to integers
unique_labels = sorted(train_df['label'].unique())
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_to_label = {i: label for label, i in label_to_int.items()}

# testing the code, only take the top 20 reocrds in training file
train_df = train_df.head(50)

# create the dataset using PyTorch Dataset
ubc_dataset = UBCDataset(
    dataframe=train_df,
    label_map=label_to_int,
    patch_size=patch_size,
    tissue_threshold=tissue_threshold,
    num_patches=num_patches
)

# create the training data loader, potentially change shuffle and num_workers
train_loader = DataLoader(
    ubc_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

# all_patch_tensors = {}

start_time = time.perf_counter()

for i, batch in enumerate(tqdm(train_loader, desc="Processing Batches")):
    print(f"\n--- Batch {i+1} ---")
    
    patches_list = batch['patches']
    labels = batch['labels']
    image_ids = batch['image_ids']
    
    print(f"Number of slides in this batch: {len(patches_list)}")
    print(f"Labels for this batch (as integers): {labels.numpy()}")
    print(f"Image IDs for this batch: {image_ids}")

    for slide_idx in range(len(image_ids)):
        slide_id = image_ids[slide_idx]
        slide_patches = patches_list[slide_idx]
        slide_label_int = labels[slide_idx].item()

        # file_name = slide_id + '_patches.pt.gz'
        # torch.save(slide_patches, file_name)
        # with gzip.open(file_name, 'wb') as f:
        #     torch.save(slide_patches, f)
        
        print(f"  - Slide ID: {slide_id}, Label: {int_to_label[slide_label_int]} ({slide_label_int}), Patches: {slide_patches.shape[0]}")
        
        # all_patch_tensors[slide_id] = slide_patches

    # Clean up batch variables at the end of each iteration
    del patches_list, labels, image_ids, batch
    gc.collect()
    print_memory_usage(f"End of Batch {i+1}:  ")
    print_disk_usage('/kaggle/working', prefix=f"End of Batch {i+1}:")
    print_disk_usage('/kaggle/temp/my_vips_temp/', prefix=f"End of Batch {i+1}:   ") # Check the pyvips temp dir
    print_disk_usage('/tmp', prefix=f"End of Batch {i+1}:")
    force_clear_vips_cache()
    print("Cleanup complete.")
    print_disk_usage('/kaggle/temp/my_vips_temp/', prefix="IMMEDIATELY AFTER CLEANUP:")
    
    # if (i + 1) % 3 == 0:
    #     print(f"\\nCleaning up temporary directory at batch {i+1}...")
    #     # The '!' runs a shell command to remove all files in the directory
    #     !rm -rf /kaggle/temp/*
    #     print("Cleanup complete.")

end_time = time.perf_counter()
duration = end_time - start_time
print(f"The code block took {duration:.4f} seconds to execute.")
# print(f"\n--- Finished processing. Total slides with stored tensors: {len(all_patch_tensors)} ---")


# !rm /kaggle/working/*.gz


# import gzip

# # torch.save(slide_patches, 'slide_patches.pt')
# with gzip.open('slide_patches.pt.gz', 'wb') as f:
#     torch.save(slide_patches, f)


# import gzip
# import torch

# with gzip.open('/kaggle/input/ovarian-caner-subtype-classification/1020_patches.pt.gz', 'rb') as f:
#     my_tensor = torch.load(f)

# print(my_tensor.shape)


# from joblib import dump

# dump(all_patch_tensors, 'all_patch_tensors.joblib')


# from joblib import load
# import os

# input_path = '/kaggle/input/ovarian-caner-subtype-classification'
# file_path = os.path.join(input_path, 'all_patch_tensors.joblib')
# new_dict = load(file_path)


import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Hyperparameters
# num_patches = 256
tissue_threshold = 0.9
patch_size = 224

# Disable the Decompression Bomb check
Image.MAX_IMAGE_PIXELS = None

base_path = '/kaggle/input/UBC-OCEAN'
train_labels_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
# image_id = str(train_labels_df.loc[2, 'image_id'])
image_id = str(38669)
wsi = SlidePatchExtractor(image_id=image_id, patch_size=patch_size, tissue_threshold=tissue_threshold, label='CC')
print(f'The image is {wsi.width} width and {wsi.height} height')
# HGSC 100, EC 180, CC 220, LGSC 470, MC 480
# MC: 5456, 21445
# LGSC: 31300, 57162
# CC: 38669
# EC: 39269
# HGSC: 39425


print(len(wsi.patch_locations))


patch_tensors = wsi.get_all_patch_tensors()


print(patch_tensors.shape)


import gc
del wsi
gc.collect()


import time
import matplotlib.pyplot as plt

start_time = time.perf_counter()

patch_tensor, patch_image = wsi.get_patch(50)

end_time = time.perf_counter()

duration = end_time - start_time
print(f"The code block took {duration:.4f} seconds to execute.")
# print(patch_tensor)

plt.imshow(patch_image)
plt.show()

# The code block took 154.5610 seconds to execute. openslide 66

# The code block took 105.2772 seconds to execute. pyvips 2666
# The code block took 140.6164 seconds to execute. openslide 2666


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- 1. Define Thumbnail and Get Image Dimensions ---
# Get the full resolution image dimensions (width, height)
original_width = wsi.width
original_height = wsi.height

patch_coordinates = wsi.patch_locations
patch_size = 224

# --- 2. Generate the Thumbnail ---
# The get_thumbnail function maintains the aspect ratio,
# creating an image that fits within the given size.
if wsi.is_tma:
    thumbnail = Image.open(wsi.source_path)
else:
    thumbnail = Image.open(wsi.thumbnail_path)
# Get the actual size of the generated thumbnail
thumb_width, thumb_height = thumbnail.size

# --- 3. Calculate Scaling Factors ---
# These factors will scale coordinates from the original image to the thumbnail
width_scale = thumb_width / original_width
height_scale = thumb_height / original_height

# --- 4. Visualize the Thumbnail and Patches ---
# Create a figure and axes for plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Display the thumbnail image
ax.imshow(thumbnail)

# Loop through each patch coordinate to draw it on the thumbnail
for x, y in patch_coordinates:
    # Scale the patch's top-left corner coordinates
    scaled_x = x * width_scale
    scaled_y = y * height_scale

    # Scale the patch's dimensions
    scaled_patch_width = patch_size * width_scale
    scaled_patch_height = patch_size * height_scale

    # Create a rectangle patch with a red edge and no fill
    rect = patches.Rectangle(
        (scaled_x, scaled_y),
        scaled_patch_width,
        scaled_patch_height,
        linewidth=1,
        edgecolor='r',  # Red color for the patch border
        facecolor='none'  # No fill
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)

# --- 5. Finalize and Show the Plot ---
ax.set_title("WSI Thumbnail with Selected Patches")
plt.axis('off')  # Hide the axes ticks and labels
plt.tight_layout()
plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt

base_path = '/kaggle/input/UBC-OCEAN'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))

# plot image dimensions
plt.figure(figsize=(10, 5)) 
plt.scatter(train_df['image_width'], train_df['image_height'], c=train_df['is_tma'], cmap='viridis')
plt.colorbar(label='is_tma')
plt.xlabel('Image Width')
plt.ylabel('Image Height')
plt.title('Image Dimensions')
plt.grid(True)
plt.show()


# plot label distribution
labels_count = train_df.label.value_counts().to_dict() 
categories = labels_count.keys()
values = labels_count.values() 
plt.bar(categories, values) 
plt.title('Label Distribution')
plt.xlabel('Labels')
plt.ylabel('Count') 
plt.show()


# plot label distribution
labels_count = train_df.label.value_counts().to_dict()
categories = labels_count.keys()
values = list(labels_count.values()) # Using list() to ensure it's a list

# Calculate the total number of labels for percentage calculation
total = sum(values)

# Create the bar plot and get the bar container object
bars = plt.bar(categories, values)

# Set the title and labels for the plot
plt.title('Label Distribution')
plt.xlabel('Labels')
plt.ylabel('Percentage')

# Iterate over each bar to add the percentage text
for bar in bars:
    # Get the height of the bar
    height = bar.get_height()
    # Add text on top of the bar
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,  # X position (center of the bar)
        height,                               # Y position (top of the bar)
        f'{height / total:.1%}',              # The text to display (formatted as percentage)
        ha='center',                          # Horizontal alignment
        va='bottom'                           # Vertical alignment
    )

# Display the plot
plt.show()




