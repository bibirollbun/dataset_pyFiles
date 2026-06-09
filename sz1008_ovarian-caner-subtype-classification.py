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
    thumbnail_arr = np.asarray(thumbnail)
    
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]

    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
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
    # print(f"the shape of tissue_mask is {tissue_mask.shape}")
    
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
    min_distance = mask_hratio * 2  # Minimum distance between points

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

class SlidePatchExtractor:
    def __init__(self, image_id, patch_size=224, mode='train', tissue_threshold=0.9, num_patches=100):
        
        self.image_id = image_id
        self.patch_size = patch_size
        self.mode = mode
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.patch_size, self.patch_size), antialias=True),
            T.Normalize(mean=[0.2585, 0.2556, 0.2506], std=[0.229, 0.224, 0.225])
        ])
        self.tissue_threshold = tissue_threshold
        self.num_patches = num_patches
        
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
            self.slide = openslide.open_slide(self.source_path)
        except openslide.OpenSlideError as e:
            print(f"Could not open slide {self.source_path}: {e}")
            self.patch_locations = []
            return

        self.width, self.height = self.slide.dimensions
        self.is_tma = self.width < 5000 and self.height < 5000
        standard_magnification = 20
        self.objective_power = 40 if self.is_tma else 20
        magnification_factor = self.objective_power / standard_magnification
        self.extraction_patch_size = int(self.patch_size * magnification_factor)
        
        self.stride = 1 if self.is_tma else 4
        
        self.patch_locations = process_wsi(
            wsi_obj=self.slide,
            wsi_path=self.source_path,
            thumbnail_path=self.thumbnail_path,
            is_tma=self.is_tma,
            output_patch_size=patch_size,
            tissue_percent=tissue_threshold,
            returnSamples=num_patches,
            stride=self.stride
        )
    
    def __len__(self):
        """Returns the number of patches found for this slide."""
        return len(self.patch_locations)
    
    def get_all_patch_tensors(self):
        """
        Extracts all patches from the slide and returns them as a stacked tensor.
        """
        patch_tensors = []
        if not self.patch_locations:
            # If no patches were found, return an empty tensor with the correct shape
            return torch.empty((0, 3, self.patch_size, self.patch_size))

        for (x, y) in self.patch_locations:
            try:
                patch_image = self.slide.read_region(
                    (x, y), 0, (self.extraction_patch_size, self.extraction_patch_size)
                ).convert('RGB')
                
                patch_tensor = self.transform(patch_image)
                patch_tensors.append(patch_tensor)
            except Exception as e:
                print(f"Error reading patch at ({x},{y}) for slide {self.image_id}: {e}")
                continue
        
        if not patch_tensors:
            return torch.empty((0, 3, self.patch_size, self.patch_size))
            
        return torch.stack(patch_tensors)
    
    def get_patch(self, idx):
        x, y = self.patch_locations[idx]
        patch_image = self.slide.read_region((x, y), 0, (self.extraction_patch_size, self.extraction_patch_size)).convert('RGB')
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

class UBCDataset(Dataset):
    """
    The main Dataset class for loading slides and their labels.
    """
    def __init__(self, dataframe, label_map, mode='train', patch_size=224, tissue_threshold=0.9, num_patches=100):
        self.df = dataframe
        self.label_map = label_map
        self.mode = mode
        self.patch_size = patch_size
        self.tissue_threshold = tissue_threshold
        self.num_patches = num_patches

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
            num_patches=self.num_patches
        )
        
        patch_tensors = extractor.get_all_patch_tensors()

        return {"patches": patch_tensors, "label": label, "image_id": image_id}


# --- Collate Function for the DataLoader ---
def collate_fn(batch):
    patches_list = [item['patches'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    image_ids = [item['image_id'] for item in batch]

    return {"patches": patches_list, "labels": labels, "image_ids": image_ids}


import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Disable the Decompression Bomb check
Image.MAX_IMAGE_PIXELS = None

# Hyperparameters
batch_size = 4
patch_size = 224
num_patches = 256
tissue_threshold = 0.9

base_path = '/kaggle/input/UBC-OCEAN'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))

# Create a mapping from string labels to integers
unique_labels = sorted(train_df['label'].unique())
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_to_label = {i: label for label, i in label_to_int.items()}

# testing the code, only take the top 4 reocrds in training file
train_df = train_df.head(20)

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

all_patch_tensors = {}

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
        
        print(f"  - Slide ID: {slide_id}, Label: {int_to_label[slide_label_int]} ({slide_label_int}), Patches: {slide_patches.shape[0]}")
        
        all_patch_tensors[slide_id] = slide_patches

print(f"\n--- Finished processing. Total slides with stored tensors: {len(all_patch_tensors)} ---")


import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Hyperparameters
num_patches = 512
tissue_threshold = 0.9

# Disable the Decompression Bomb check
Image.MAX_IMAGE_PIXELS = None

base_path = '/kaggle/input/UBC-OCEAN'
train_labels_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
# image_id = str(train_labels_df.loc[2, 'image_id'])
image_id = str(4)
wsi = SlidePatchExtractor(image_id=image_id, tissue_threshold=tissue_threshold, num_patches=num_patches)
print(f'The image is {wsi.width} width and {wsi.height} height')



print(len(wsi.patch_locations))


patch_tensors = wsi.get_all_patch_tensors()


print(patch_tensors.shape)


patch_tensor, patch_image = wsi.get_patch(7)
print(patch_tensor)


print(patch_tensor.shape)


import matplotlib.pyplot as plt
plt.imshow(patch_image)
plt.show()


print(wsi.patch_locations)


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
    thumbnail = wsi.slide.get_thumbnail((1024, 1024))
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


print("hello")

