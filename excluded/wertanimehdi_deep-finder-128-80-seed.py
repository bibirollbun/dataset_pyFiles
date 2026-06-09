import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import torch
import torch.nn as nn
import gc
import random
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from IPython.display import display, clear_output
from torch.optim.lr_scheduler import ExponentialLR
try :
    import zarr
    import monai
    import cc3d
    import torch_optimizer as optim
    from monai.networks.blocks import MaxAvgPool
except :
    !pip install zarr
    !pip install monai
    !pip install segmentation_models_pytorch
    !pip install --no-index --find-links=/kaggle/input/hengck-czii-cryo-et-01/wheel_file connected-components-3d
    !pip install torch-optimizer
    import zarr
    import monai
    import cc3d
    import torch_optimizer as optim
    from monai.networks.blocks import MaxAvgPool


DATA_KAGGLE_DIR = '/kaggle/input/czii-cryo-et-object-identification'
TRAIN_DIR = f'{DATA_KAGGLE_DIR}/train'
TEST_DIR = f'{DATA_KAGGLE_DIR}/test'

TRAIN_EXP = ["TS_5_4","TS_69_2","TS_6_4","TS_6_6","TS_73_6","TS_86_3","TS_99_9"]
TEST_EXP = ["TS_5_4","TS_69_2","TS_6_4"]

scale = 10.012444196428572

PATCH_SIZE = 64

OBJECT_DICT = {
    'apo-ferritin': {'label': 1, 'radius': 60/scale}, 
    'beta-galactosidase': {'label': 2, 'radius': 90/scale}, 
    'ribosome': {'label': 3, 'radius': 150/scale}, 
    'thyroglobulin': {'label': 4, 'radius': 130/scale}, 
    'virus-like-particle': {'label': 5, 'radius': 135/scale},
    #'beta-amylase' : {'label': 6, 'radius':65/scale},
}

def read_one_data(id, static_dir):
    zarr_dir = f'{static_dir}/{id}/VoxelSpacing10.000'
    zarr_file = f'{zarr_dir}/denoised.zarr'
    zarr_data = zarr.open(zarr_file, mode='r')
    volume = zarr_data[0][:]
    # mean = volume.mean()
    # std = volume.std()
    # volume = (volume - mean) / std
    return volume


def read_one_truth(id, overlay_dir):
    location={}
    json_dir = f'{overlay_dir}/{id}/Picks'
    for p in OBJECT_DICT.keys():
        json_file = f'{json_dir}/{p}.json'
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        num_point = len(json_data['points'])
        loc = [list(reversed(list(json_data['points'][i]['location'].values())))  for i in range(num_point)]
        location[p] = [[coo/scale for coo in coos] for coos in loc ]
    return location



def visualize(data):
    """
    Visualize multiple images or masks side by side for each slice in the dataset.
    
    Parameters:
    data (dict): Dictionary where keys are the names of the data items,
                 and values are either 3D images or masks.
    """
    # Ensure all data items have the same depth (number of slices)
    z_sizes = {name: item.shape[0] for name, item in data.items() if name!="center"}
    if len(set(z_sizes.values())) != 1:
        raise ValueError("All items must have the same number of slices along the Z-axis.")
    
    z_size = next(iter(z_sizes.values()))  # Depth of slices (Z-axis)
    keys = list(data.keys())  # Get all keys for consistent ordering
    # Loop over each slice (Z-index)
    for z in range(z_size):
        fig, axes = plt.subplots(1, len(keys), figsize=(3 * len(keys), 5))
        for idx, key in enumerate(keys):
            item = data[key]
            cmap = "gray" if "volume" in key.lower() else "jet"
            
            # Display the slice
            axes[idx].imshow(item[z, :, :], cmap=cmap)
            axes[idx].set_title(f"{key} - Slice (Z={z})")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()

def draw_cylinder_in_image_fast(image, center, radius, z_factor, yx_factor, value):
    new_radius = radius * yx_factor
    half_height = round(radius * z_factor)
    
    z_min = max(round(center[0] - half_height), 0)
    y_min = max(round(center[1] - new_radius), 0)
    x_min = max(round(center[2] - new_radius), 0)

    z_max = min(round(center[0] + half_height) + 1, image.shape[0])
    y_max = min(round(center[1] + new_radius) + 1, image.shape[1])
    x_max = min(round(center[2] + new_radius) + 1, image.shape[2])
    
    local_region = image[z_min:z_max, y_min:y_max, x_min:x_max]
    local_center = (half_height, new_radius, new_radius)
    
    local_region = draw_cylinder_in_local_image(local_region, local_center, new_radius, half_height, value)
    image[z_min:z_max, y_min:y_max, x_min:x_max] = np.bitwise_or(local_region,image[z_min:z_max, y_min:y_max, x_min:x_max])

    return image

def draw_cylinder_in_local_image(image, center, radius, half_height, value):
    shape = image.shape

    z, y, x = np.indices(shape)

    distance = (y - center[1])**2 + (x - center[2])**2

    z_bounds = (z >= center[0] - half_height) & (z <= center[0] + half_height)
    yx_bounds = distance <= radius**2
    
    cylinder = z_bounds & yx_bounds

    image[cylinder] = value

    return image

def draw_sphere_in_image_fast(image, center, radius, radius_factor, value):
    new_radius = radius * radius_factor
    z_min = max(round(center[0] - new_radius), 0)
    y_min = max(round(center[1] - new_radius), 0)
    x_min = max(round(center[2] - new_radius), 0)

    z_max = min(round(center[0] + new_radius) + 1, image.shape[0])
    y_max = min(round(center[1] + new_radius) + 1, image.shape[1])
    x_max = min(round(center[2] + new_radius) + 1, image.shape[2])
    
    local_region = image[z_min:z_max, y_min:y_max, x_min:x_max]
    local_center = (new_radius, new_radius, new_radius)
    
    local_region = draw_sphere_in_local_image(local_region, local_center, new_radius, value)
    image[z_min:z_max, y_min:y_max, x_min:x_max] = np.bitwise_or(local_region,image[z_min:z_max, y_min:y_max, x_min:x_max])

    return image

def draw_sphere_in_local_image(image, center, radius, value):
    shape = image.shape

    z, y, x = np.indices(shape)

    distance = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    
    cylinder = distance <= radius**2

    image[cylinder] = value

    return image

def draw_dentroid(image, center,value):

    z, y, x = map(round, center)
    
    z_range = [z-1, z + 1]
    y_range = [y-1, y + 1]
    x_range = [x-1, x + 1]
    
    for zi in z_range:
        for yi in y_range:
            for xi in x_range:
                # Ensure the indices are within the bounds of the image
                if 0 <= zi < image.shape[0] and 0 <= yi < image.shape[1] and 0 <= xi < image.shape[2]:
                    image[zi, yi, xi] = value

    return image

def generate_heatmap(shape, centers, sigma=2):
    heatmap = np.zeros(shape)  # Initialize the 3D heatmap

    for center in centers:
        z, y, x = center
        radius = 3 * sigma  # Define a cutoff radius (3 sigma covers ~99.7% of Gaussian)
        
        # Define bounds for the local region
        z_min = max(round(z - radius), 0)
        y_min = max(round(y - radius), 0)
        x_min = max(round(x - radius), 0)

        z_max = min(round(z + radius) + 1, shape[0])
        y_max = min(round(y + radius) + 1, shape[1])
        x_max = min(round(x + radius) + 1, shape[2])

        # Extract the local region
        local_region = heatmap[z_min:z_max, y_min:y_max, x_min:x_max]
        local_center = (z - z_min, y - y_min, x - x_min)  # Local center coordinates

        # Generate Gaussian blob in the local region
        heatmap[z_min:z_max, y_min:y_max, x_min:x_max] += generate_local_gaussian(
            local_region.shape, local_center, sigma
        )

    return heatmap


def generate_local_gaussian(shape, center, sigma):
    z, y, x = np.indices(shape)  # Create coordinate grids for the local region

    # Compute the squared distance from the center
    distance = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2

    # Compute the Gaussian values
    gaussian = np.exp(-distance / (2 * sigma**2))

    return gaussian

def generate_centers(image_shape, patch_size):
    
    centers = [
        [z, y, x]
        for z in range(patch_size[0]//2, image_shape[0], 3*patch_size[0] // 4)
        for y in range(patch_size[1]//2, image_shape[1], 3*patch_size[1] // 4)
        for x in range(patch_size[2]//2, image_shape[2], 3*patch_size[2] // 4) 
        if z < image_shape[0]-patch_size[0]//2 and y < image_shape[1]-patch_size[1]//2 and x < image_shape[2]-patch_size[2]//2
    ]
    centers.append([c-patch_size[i] for i,c in enumerate(image_shape)])
    return centers
    
def crop_with_center(image, patch_size, center):

    center = [round (c) for c in center]
    dims = image.shape
    if len(dims) == 3:
        start_indices = [max(0, c - p // 2) for c, p in zip(center, patch_size)]
        end_indices = [min(dim, c + (p + 1) // 2) for c, p, dim in zip(center, patch_size, dims)]
        
        cropped_patch = image[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ]
    else:
        start_indices = [max(0, c - p // 2) for c, p in zip(center, patch_size)]
        end_indices = [min(dim, c + (p + 1) // 2) for c, p, dim in zip(center, patch_size, dims[1:])]
        
        cropped_patch = image[
            :,
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ]
    return cropped_patch
    
def pad(result, patch_size):
    shape = np.array(result["volume"].shape)
    expected_shape = np.array(patch_size)
    diff = expected_shape - shape

    if np.all(diff == 0):
        return result
    else:
        # Compute mean and std of the original volume
        mean = np.mean(result["volume"])
        std = np.std(result["volume"])

        # Create a fill matrix with the same mean and std
        fill_matrix = np.random.normal(mean, std, size=tuple(expected_shape)).astype(result["volume"].dtype)

        # Determine random start points for padding
        zyx_start = [np.random.randint(0, diff[i] + 1) for i in range(3)]

        # Fill the padded matrix with the original volume
        fill_matrix[
            zyx_start[0]:zyx_start[0] + shape[0],
            zyx_start[1]:zyx_start[1] + shape[1],
            zyx_start[2]:zyx_start[2] + shape[2]
        ] = result["volume"]
        result["volume"] = fill_matrix

        # Pad other keys in the result
        for key in result.keys():
            if key == "volume":
                continue

            current_shape = result[key].shape
            if len(current_shape) > 3:
                mask_padder = np.zeros((current_shape[0], *expected_shape), dtype=bool)
                mask_padder[
                    :,
                    zyx_start[0]:zyx_start[0] + shape[0],
                    zyx_start[1]:zyx_start[1] + shape[1],
                    zyx_start[2]:zyx_start[2] + shape[2]
                ] = result[key] if key != "containers" else result[key] + 0.5
                result[key] = mask_padder

            else:
                mask_padder = np.zeros(expected_shape, dtype=bool)
                mask_padder[
                    zyx_start[0]:zyx_start[0] + shape[0],
                    zyx_start[1]:zyx_start[1] + shape[1],
                    zyx_start[2]:zyx_start[2] + shape[2]
                ] = result[key] if key != "containers" else result[key] + 0.5
                result[key] = mask_padder

    return result

def pad_tensor(result, patch_size):
    shape = torch.tensor(result["volume"].shape)
    expected_shape = torch.tensor(patch_size)
    diff = expected_shape - shape

    if (diff == 0).all():
        return result
    else:
        # Compute mean and std of the original volume
        mean = result["volume"].mean()
        std = result["volume"].std()

        # Create a fill matrix with the same mean and std
        fill_matrix = torch.normal(mean, std, size=tuple(expected_shape), dtype=result["volume"].dtype)

        # Determine random start points for padding
        zyx_start = [torch.randint(0, diff[i].item() + 1, (1,)).item() for i in range(3)]

        # Fill the padded matrix with the original volume
        fill_matrix[
            zyx_start[0]:zyx_start[0] + shape[0],
            zyx_start[1]:zyx_start[1] + shape[1],
            zyx_start[2]:zyx_start[2] + shape[2]
        ] = result["volume"]
        result["volume"] = fill_matrix

        # Pad other keys in the result
        for key in result.keys():
            if key == "volume":
                continue

            current_shape = result[key].shape
            if len(current_shape) > 3:
                mask_padder = torch.zeros(
                    (current_shape[0], *expected_shape), dtype=torch.bool
                )
                mask_padder[
                    :,
                    zyx_start[0]:zyx_start[0] + shape[0],
                    zyx_start[1]:zyx_start[1] + shape[1],
                    zyx_start[2]:zyx_start[2] + shape[2]
                ] = result[key] if key != "containers" else result[key] + 0.5
                result[key] = mask_padder

            else:
                mask_padder = torch.zeros(list(expected_shape), dtype=torch.bool)
                mask_padder[
                    zyx_start[0]:zyx_start[0] + shape[0],
                    zyx_start[1]:zyx_start[1] + shape[1],
                    zyx_start[2]:zyx_start[2] + shape[2]
                ] = result[key] if key != "containers" else result[key] + 0.5
                result[key] = mask_padder

    return result
def one_hot(label):
    return np.stack( [label==i for i in range(6)] ,  0)


train_data = {}
for exp_name in tqdm(TRAIN_EXP):
    train_data[exp_name] = {}
    train_data[exp_name]["volume"] = read_one_data(exp_name, static_dir=f'{TRAIN_DIR}/static/ExperimentRuns')
    train_data[exp_name]["truth"] = read_one_truth(exp_name, overlay_dir=f'{TRAIN_DIR}/overlay/ExperimentRuns')
    pmin,pmax = np.percentile(train_data[exp_name]["volume"],(5,99))
    train_data[exp_name]["min"] = pmin
    train_data[exp_name]["max"] = pmax
    
    train_data[exp_name]["label"] = np.zeros((184,630,630) , dtype = np.int8)
    train_data[exp_name]["centroid"] = np.zeros((184,630,630) , dtype = np.int8)
    train_data[exp_name]["containers"] = np.zeros((184,630,630) , dtype = np.bool_)
    #train_data[exp_name]["heat_map"] = np.zeros((184,630,630) , dtype = np.float16)
    
    for particle in train_data[exp_name]["truth"].keys():
        radius = OBJECT_DICT[particle]["radius"]
        radius_factor = np.log2(radius)/radius *.8
        
        label = OBJECT_DICT[particle]["label"]
        
        #train_data[exp_name]["heat_map"] += generate_heatmap((184,630,630) , train_data[exp_name]["truth"][particle])
        
        for point in train_data[exp_name]["truth"][particle]:
            train_data[exp_name]["containers"] = draw_cylinder_in_image_fast(train_data[exp_name]["containers"], point, radius, z_factor = 1.7, yx_factor = 1.2, value = True)
            train_data[exp_name]["label"] = draw_sphere_in_image_fast(train_data[exp_name]["label"], point, radius, radius_factor = radius_factor, value = label)
            train_data[exp_name]["centroid"] = draw_dentroid(train_data[exp_name]["centroid"], point , value = label)
    #train_data[exp_name]["heat_map"] = np.stack([train_data[exp_name]["heat_map"][particle] for particle in OBJECT_DICT.keys()], 0).astype(np.float16)


min_ = 0
max_ = 0

for k in train_data.keys():
    pmin,pmax = np.percentile(train_data[k]["volume"],(5,99))
    print(pmin,pmax)
    min_ += pmin/7
    max_ += pmax/7

print(min_,max_)


for k in train_data.keys():
    train_data[k]["volume"] = (train_data[k]["volume"]-min_)/(max_-min_)


def mean_std_shift (image,shift = 0.03):
    factor = 1/(shift*2)
    std = image.std()
    mean = image.mean()
    shift_mean = (torch.rand(1)/factor - shift).item()
    shift_std = (torch.rand(1)/factor - shift).item()
    new_mean = mean + mean * shift_mean
    new_std = std + std * shift_std

    new_image = (image-mean)/std*new_std+new_mean
    return new_image

def generate_random_mask(mask_ratio, image_size):
    """
    Generate a random numpy mask for a 3D image with a given mask ratio.

    Parameters:
        image_size (tuple): A tuple representing the dimensions of the 3D image (depth, height, width).
        mask_ratio (float): A float between 0 and 1 indicating the fraction of elements to be masked (1 = fully masked, 0 = no mask).

    Returns:
        np.ndarray: A 3D numpy array with the same shape as the image, containing 0s (masked) and 1s (unmasked).
    """
    if not (0 <= mask_ratio <= 1):
        raise ValueError("mask_ratio must be between 0 and 1.")

    # Total number of elements in the image
    total_elements = np.prod(image_size)

    # Number of elements to mask
    num_masked_elements = int(total_elements * mask_ratio)

    # Create a flattened array with the specified number of masked (0) and unmasked (1) elements
    mask_flat = np.ones(total_elements, dtype=np.uint8)
    mask_flat[:num_masked_elements] = 0

    # Shuffle the array to randomize the mask positions
    np.random.shuffle(mask_flat)

    # Reshape the flat mask array back to the original image size
    mask = mask_flat.reshape(image_size)

    return mask

# Example usage
image_size = (96, 96, 96 )  # Example 3D image dimensions (depth, height, width)
mask_ratio = 0.3           # 30% of the image will be masked
mask = generate_random_mask(mask_ratio, image_size)

print((mask == 1).sum())


import torch.nn.functional as F


class SegmentationDataset(Dataset):
    def __init__(self, patch_size,length ,experiments = ["TS_6_4"]):
        self.patch_size = patch_size
        self.experiments = experiments
        self.length = length
        self.label_counts = {i:0 for i in range(6)}
        for exp_name in experiments :
            unique_values, counts = np.unique(train_data[exp_name]["label"], return_counts=True)
            for i,unique in enumerate(unique_values):
                self.label_counts[unique] += counts[i]
        array = np.array([v for _,v in {0: 72996892, 1: 4176, 2: 1548, 3: 18574, 4: 6240, 5: 2170}.items()] )
        array = 1/np.log1p(array)
        self.weights = array/array.sum()
        print(self.weights)
    def __len__(self):
        return self.length

    def augment(self,result):
        
        do_flip_z = torch.rand(1)<.5
        do_flip_y = torch.rand(1)<.5
        do_flip_x = torch.rand(1)<.5
        rot_times = random.choice([0,1,2,3])

        for key in result.keys():
            if do_flip_z:
                result[key] = np.flip(result[key], axis=-3)
            if do_flip_y:
                result[key] = np.flip(result[key], axis=-2)
            if do_flip_x:
                result[key] = np.flip(result[key], axis=-1)

            if rot_times !=0:
                result[key] = np.rot90(result[key] , k = rot_times, axes=(-2,-1))
        return result
        
    def _to_tensor(self,result):
        for k in result.keys():
            if "label" in k or "heat_map" in k:
                #result[k] = numpy_one_hot(result[k])
                result[k] = torch.tensor(result[k].copy() , dtype = torch.long if "label" in k else torch.float32)
                #result[k] = one_hot_encode_3d(result[k] ,2)
            else : 
                result[k] = torch.tensor(result[k].copy() , dtype = torch.float32)

        #result["label"] = torch.stack([result[k] for k in result.keys() if "label" in k])
        return result
        
    def __getitem__(self,idx):
        zyx = [random.choice(range(self.patch_size[i]//2 ,dim-self.patch_size[i]//2)) for i,dim in enumerate((184,630,630))]
        exp_name = random.choice(self.experiments)
        
        result = {}
        for key in ["label", "containers", "volume"]:#,"heat_map"
            result [key] = crop_with_center(train_data[exp_name][key], self.patch_size, zyx)

        result = self.augment(result)

        result["volume"] = mean_std_shift(result["volume"])

        result = pad(result,self.patch_size)

        result["label_hot"] = one_hot(result["label"])

        #result["label"] = one_hot(result["label"])
        result = self._to_tensor(result)
        return result



import torch.nn as nn

class ConvBNReLU2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU2D, self).__init__()
        if kernel_size == 5:
            padding = 2

        if kernel_size == 7:
            padding = 3
            
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvBNReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class Input(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3 , num_conv2d = 2):
        super(Input, self).__init__()
        self.norm = nn.BatchNorm2d(1)
        self.conv2d_layers = nn.Sequential(
            *[ConvBNReLU2D(in_channels * 4 if i == 0 else out_channels, out_channels) 
              for i in range(num_conv2d)]
        )
        self.conv2d_7 = ConvBNReLU2D(1,in_channels,7)
        self.conv2d_5 = ConvBNReLU2D(1,in_channels,5)
        self.conv2d_3 = ConvBNReLU2D(1,in_channels,3)

    def forward(self, x):
        # Apply Conv2D layers
        b, c, d, h, w = x.shape  # Batch size, Channels, Depth, Height, Width
        x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)  # Reshape for Conv2D
        
        x = self.norm(x)
        x = torch.cat([x,self.conv2d_7(x), self.conv2d_5(x), self.conv2d_3(x)] , 1)
        x = self.conv2d_layers(x)
        # Reshape back to 5D for Conv3D layers
        bd, c_out, h, w = x.shape
        x = x.reshape(b, d, c_out, h, w).permute(0, 2, 1, 3, 4)
        return dotdict({"out":x})

import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels = 0, num_conv3d=1, 
                 do_up = True , do_down = True ,use_transpose = False):
        super(EncoderBlock, self).__init__()
        self.do_up = do_up
        
        if self.do_up:
            self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='trilinear')
            
        if use_transpose:
            self.upsample = nn.Sequential(
                nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        self.do_down = do_down 
        if self.do_down:
            self.downsample = lambda x: F.interpolate(x, scale_factor=.5, mode='trilinear')

        
        self.conv3d_layers = nn.Sequential(
            *[ConvBNReLU3D(out_channels if i!=0 else in_channels + skip_channels, out_channels, stride= (1, 1, 1)) 
              for i in range(num_conv3d)]
        )

    def forward(self, x , xskip = None):
        if xskip is not None:
            x = torch.cat([x, xskip], dim=1)

        out = self.conv3d_layers(x)
        output = {
            "out": out,
            "up":None,
            "down":None
        }

        if self.do_up:
            output["up"] = self.upsample(out)   
        if self.do_down:
            output["down"] = self.downsample(out)   
        return dotdict(output)


class Model(nn.Module):
    def __init__(self , channels = [28,32,36]):
        super(Model, self).__init__()
        self.register_buffer('D', torch.tensor(0))
        self.output_type = ['particle', 'loss']

        self.norm = nn.BatchNorm3d(1)
        
        self.encoder1 = EncoderBlock(in_channels = 1, out_channels = channels[0], num_conv3d=2 , do_up = False, do_down=True)
        self.encoder2 = EncoderBlock(in_channels = channels[0], out_channels = channels[1], num_conv3d=2 , do_up = True, do_down=True)
        
        self.decoder1 = EncoderBlock(in_channels = channels[1], out_channels = channels[2], num_conv3d=4 , do_up = True, do_down=False)
        self.decoder2 = EncoderBlock(in_channels = channels[2], out_channels = channels[1], num_conv3d=2 , skip_channels = channels[1], do_up = True, do_down=False , use_transpose = True)

        self.pre = EncoderBlock(in_channels = channels[1], out_channels = channels[0], num_conv3d=2 , do_up = False, do_down=False)

        self.mask = nn.Conv3d(channels[0], 6, 1, 1, bias=False)
        
        

    def forward(self,batch):
        device = self.D.device
        volume = batch["volume"].to(device).unsqueeze(1)

        input_ = self.norm(volume)
        
        encode1 = self.encoder1(input_)
        encode2 = self.encoder2(encode1.down)
        
        decode1 = self.decoder1(encode2.down)
        #print(encode2.out.shape , decode1.up.shape)
        decode2 = self.decoder2(encode2.out , decode1.up)

        pre = self.pre(decode2.up)

        logit = self.mask(pre.out)
        #print(mask.shape)

        output = {}
        
        if "loss" in self.output_type and "label" in batch.keys():
        
            # Apply weighted cross-entropy loss
            output["loss"] = F.cross_entropy(
                logit, 
                batch['label'].to(device), 
                label_smoothing=0.01,
            )

        if "particle" in self.output_type:
            output['particle'] = F.softmax(logit,1)
            
        return output



def set_seed(seed):

    random.seed(seed)  # Python's built-in random
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # Torch CPU random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Torch GPU random seed
        torch.cuda.manual_seed_all(seed)  # All GPUs

    # For deterministic behavior in CuDNN operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(80)


def calculate_patch_starts(dimension_size: int, patch_size: int):
    if dimension_size <= patch_size:
        return [0]
        
    # Calculate number of patches needed
    n_patches = np.ceil(dimension_size / patch_size) + 3
    
    if n_patches == 1:
        return [0]
    
    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)
    
    return positions
    
class PredDataset(Dataset):
    def __init__(self, patch_size, experiment = TRAIN_EXP[0] ,is_local = True):
        self.is_local = is_local
        self.experiment = experiment
        self.patch_size = patch_size
        #self.zyx = np.ones((3, 184+patch_size , 630+patch_size, 630+patch_size))*-1
        pad_size = [patch_size[i]//2 for i in range(3)]
        #self.zyx [:,pad_size:184+pad_size, pad_size:630+pad_size, pad_size:630+pad_size] = np.indices((184,630,630))
        
        self.volume = np.zeros((184+patch_size[0],630+patch_size[1],630+patch_size[2]))
        self.volume [pad_size[0]:184+pad_size[0], pad_size[1]:630+pad_size[1], pad_size[2]:630+pad_size[2]] = train_data[experiment]["volume"]
        self.label = np.zeros((184+patch_size[0],630+patch_size[1],630+patch_size[2]))
        self.label [pad_size[0]:184+pad_size[0], pad_size[1]:630+pad_size[1], pad_size[2]:630+pad_size[2]] = train_data[experiment]["label"]
        
        self.locations = read_one_truth(experiment, overlay_dir=f'{TRAIN_DIR}/overlay/ExperimentRuns') if is_local else None

        self.indexes = [[z,y,x] 
                       for z in calculate_patch_starts(184+patch_size[0],patch_size[0])
                       for y in calculate_patch_starts(630+patch_size[1],patch_size[1])
                       for x in calculate_patch_starts(630+patch_size[2],patch_size[2])]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self,idx):
        zyx = self.indexes [idx]
        patch = self.volume[zyx[0]:zyx[0]+self.patch_size[0],zyx[1]:zyx[1]+self.patch_size[1],zyx[2]:zyx[2]+self.patch_size[2]]
        label = self.label [zyx[0]:zyx[0]+self.patch_size[0],zyx[1]:zyx[1]+self.patch_size[1],zyx[2]:zyx[2]+self.patch_size[2]]
        shape = patch.shape
        if shape[0]*shape[1]*shape[2] != self.patch_size[0]*self.patch_size[1]*self.patch_size[2]:
            padder = np.zeros(self.patch_size)
            padder[:shape[0],:shape[1],:shape[2]] = patch
            patch = padder
            padder[:shape[0],:shape[1],:shape[2]] = label
            label = padder
            
        return {"volume":torch.tensor(patch,dtype = torch.float32),'zyx':  torch.tensor(zyx,dtype = torch.long),"label":torch.tensor(label,dtype = torch.long) , "label_hot":torch.tensor(one_hot(label),dtype = torch.long)}



def get_probs():
    val_loss = 0
    
    weight = torch.zeros((patch_size[0], patch_size[1], patch_size[2]) , dtype =torch.float16).to("cuda")
    weight[8:patch_size[0]-8, 8:patch_size[1]-8, 8:patch_size[2]-8] += 1
    
    # Initialize output tensors
    all_logits = torch.zeros((6, 184 + patch_size[0], 630 + patch_size[1], 630 + patch_size[2]) , dtype =torch.float16).to("cuda")
    count = torch.zeros((184 + patch_size[0], 630 + patch_size[1], 630 + patch_size[2]) , dtype =torch.float16).to("cuda")
    
    model.output_type = ["particle","loss"]
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            for batch in tqdm(pl):
                zyxs = batch["zyx"]
                outputs = model(batch)
                local_logits = outputs["particle"]
                
                val_loss += outputs["loss"].item()
                for i in range(len(zyxs)):
                    zyx = zyxs[i]
                    count[zyx[0]:zyx[0]+patch_size[0], zyx[1]:zyx[1]+patch_size[1], zyx[2]:zyx[2]+patch_size[2]] += weight
        
                    all_logits[:, zyx[0]:zyx[0]+patch_size[0], zyx[1]:zyx[1]+patch_size[1], zyx[2]:zyx[2]+patch_size[2]] += local_logits[i] * weight
                    
        # Print epoch loss
        print(f" Val Loss: {val_loss/len(pl):.8f}")
        # Crop to remove padding
        all_logits = all_logits[:, patch_size[0]//2:patch_size[0]//2+184, patch_size[1]//2:patch_size[1]//2+630, patch_size[2]//2:patch_size[2]//2+630]
        count = count[patch_size[0]//2:patch_size[0]//2+184, patch_size[1]//2:patch_size[1]//2+630, patch_size[2]//2:patch_size[2]//2+630]
        probs = nn.Softmax(dim=0)(torch.tensor(all_logits/count)).detach().cpu().numpy()
        # Compute probabilities
        probs = (all_logits/count).detach().cpu().numpy()
    
        # Convert tensors to numpy arrays
        all_logits = all_logits.detach().cpu().numpy()
        count = count.detach().cpu().numpy()
        weight = weight.detach().cpu().numpy()

    return probs

def evaluate_predictions(stats, pred_loader, distance_threshold=3, beta=4 , particle_name = None):
    best_f_beta = 0
    best_metric = None
    voxel_threshold = 5
    # Filter predictions based on voxel count
    pred = np.array([centroid for i, centroid in enumerate(stats["centroids"]) if i != 0 and stats["voxel_counts"][i] > voxel_threshold])
    truth_locations = np.array(pred_loader.dataset.locations[particle_name])
    # Perform evaluation
    hit, fp, miss, metric = do_one_eval(truth_locations, pred, distance_threshold)

    # Calculate precision, recall, and F-beta score
    precision = len(hit[0]) / (len(hit[0]) + len(fp)) if (len(hit[0]) + len(fp)) > 0 else 0
    recall = len(hit[0]) / (len(hit[0]) + len(miss)) if (len(hit[0]) + len(miss)) > 0 else 0

    beta_squared = beta ** 2
    f_beta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall) if (precision + recall) > 0 else 0
    if f_beta>= best_f_beta:
        best_f_beta = f_beta
        best_metric = {
            "truth": len(truth_locations),
            "predict": len(pred),
            "hit": len(hit[0]),
            "fp": len(fp),
            "miss": len(miss),
            "f_b": f_beta,
            "thresh": voxel_threshold
        }
    # Return results as JSON-like dictionary
    return best_metric
    
def do_one_eval(truth, predict, threshold = 3):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    #---
    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)] #for lb metric F-beta copmutation
    return hit, fp, miss, metric

def do_evaluate():
    probs = get_probs()
    evals = {}
    for particle in OBJECT_DICT.keys():
        label = OBJECT_DICT[particle]["label"]
        thresh = OBJECT_DICT[particle]["radius"]/2
        labels_out = cc3d.connected_components(np.array(probs[label, :, :, :] > 0.06), connectivity=18)
        stats = cc3d.statistics(labels_out)

        evals[particle] = evaluate_predictions(stats, pl, distance_threshold = thresh, particle_name = particle)
    return evals





import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna


def find_best_learning_rate(model, objective_loader , learning_rates):
    """Find the best learning rate using Optuna."""
    def objective(trial):
        lr = trial.suggest_loguniform("lr", learning_rates[1], learning_rates[0])  # Suggest a learning rate in a wide range
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        scaler = GradScaler()
        
        model.train()
        total_loss = 0.0
        for batch in tqdm(objective_loader):
            with autocast():
                outputs = model(batch)
                loss = outputs["loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        return total_loss / len(objective_loader)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=4)  # Adjust n_trials as needed for computational limits

    best_lr = study.best_params["lr"]
    print(f"Best Learning Rate: {best_lr}")
    return best_lr

learning_rates = {
    0 : [1e-3 , 1e-4],
    10 : [1e-4 , 1e-5],
    20 : [1e-5 , 1e-7],
    25 : [1e-5 , 1e-7],
    30 : [1e-6 , 1e-8],
    35 : [1e-7 , 1e-10],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 4
num_epochs = 60
patch_size = (128, 128, 128)

folds = []
train_experiments = [TRAIN_EXP[i] for i in range(7) if i not in folds]

# Functions
def get_model():
    """Initialize and return the model."""
    model = Model().to(device)
    return model

def get_loaders():
    """Prepare and return the dataloaders."""
    train_dataset = SegmentationDataset(patch_size, 1024, experiments=train_experiments)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    objective_dataset = SegmentationDataset(patch_size, 1024, experiments=train_experiments)
    objective_loader = DataLoader(objective_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader, objective_loader

def train_one_epoch(model, train_loader, optimizer, scaler, max_norm=1.0):
    """Train the model for one epoch with gradient clipping and return the average loss."""
    model.train()
    model.output_type = ["loss"]
    train_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        with autocast():
            outputs = model(batch)
            loss = outputs["loss"]

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params and clips gradients
        scaler.unscale_(optimizer)  
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss

def train_model():
    """Train the model across all epochs."""
    model = get_model()
    train_loader , objective_loader = get_loaders()

    scaler = GradScaler()
    learning_rate = 1e-4
    for epoch in range(num_epochs):


        if epoch == 35 :        
            learning_rate = 1e-5

        if epoch == 45 :        
            learning_rate = 1e-7
            
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.999)
        )
        
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = f"model_all_{epoch}.bin"
        torch.save(model.state_dict(), checkpoint_path)

    # Free GPU memory
    torch.cuda.empty_cache()

# Call the training function
train_model()




