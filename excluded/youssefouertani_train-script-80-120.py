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
try :
    import zarr
    import cc3d
    import torch_optimizer as optim
except :
    !pip install zarr
    !pip install --no-index --find-links=/kaggle/input/hengck-czii-cryo-et-01/wheel_file connected-components-3d
    import zarr
    import cc3d


DATA_KAGGLE_DIR = '/kaggle/input/czii-cryo-et-object-identification'
TRAIN_DIR = f'{DATA_KAGGLE_DIR}/train'
TEST_DIR = f'{DATA_KAGGLE_DIR}/test'

TRAIN_EXP = ["TS_5_4","TS_69_2","TS_6_4","TS_6_6","TS_73_6","TS_86_3","TS_99_9"]
TEST_EXP = ["TS_5_4","TS_69_2","TS_6_4"]

scale = 10.012444196428572

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

def one_hot(label):
    return np.stack( [label==i for i in range(6)] ,  0)


train_data = {}
for exp_name in tqdm(TRAIN_EXP):
    train_data[exp_name] = {}
    train_data[exp_name]["volume"] = read_one_data(exp_name, static_dir=f'{TRAIN_DIR}/static/ExperimentRuns')
    train_data[exp_name]["truth"] = read_one_truth(exp_name, overlay_dir=f'{TRAIN_DIR}/overlay/ExperimentRuns')
    
    train_data[exp_name]["label"] = np.zeros((184,630,630) , dtype = np.int8)
    
    for particle in train_data[exp_name]["truth"].keys():
        radius = OBJECT_DICT[particle]["radius"]
        radius_factor = np.log2(radius)/radius *.8
        
        label = OBJECT_DICT[particle]["label"]
        
        for point in train_data[exp_name]["truth"][particle]:
            train_data[exp_name]["label"] = draw_sphere_in_image_fast(train_data[exp_name]["label"], point, radius, radius_factor = radius_factor, value = label)


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


def mean_std_shift (image,shift = 0.02):
    factor = 1/(shift*2)
    std = image.std()
    mean = image.mean()
    shift_mean = (torch.rand(1)/factor - shift).item()
    shift_std = (torch.rand(1)/factor - shift).item()
    new_mean = mean + mean * shift_mean
    new_std = std + std * shift_std

    new_image = (image-mean)/std*new_std+new_mean
    return new_image


import torch.nn.functional as F

class SegmentationDataset(Dataset):
    def __init__(self, patch_size,length ,experiments = ["TS_6_4"] , shift = 0.02):
        self.patch_size = patch_size
        self.experiments = experiments
        self.length = length
        self.shift = shift

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
        for key in ["label", "volume"]:#,"heat_map"
            result [key] = crop_with_center(train_data[exp_name][key], self.patch_size, zyx)

        result = self.augment(result)

        result["volume"] = mean_std_shift(result["volume"],self.shift)

        # i left this so i get the same result (i forgot to delete it in original notebook i thaught it might interfere with the seed result if i don't keep it)
        if torch.rand(1)> .8:
            1


        #result["label"] = one_hot(result["label"])
        result = self._to_tensor(result)
        return result



class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
            
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


import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
num_epochs = 36
patch_size = (128, 128, 128)

folds = []
train_experiments = [TRAIN_EXP[i] for i in range(7) if i not in folds]


def get_model():
    """Initialize and return the model."""
    model = Model().to(device)
    return model

def get_loader():
    """Prepare and return the dataloaders."""
    train_dataset = SegmentationDataset(patch_size, 1024, experiments=train_experiments)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader

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

def train_model(seed = 42):
    """Train the model across all epochs."""
    set_seed(seed)
    model = get_model()
    train_loader = get_loader()

    # there is no logic behind using different shift factor i just forgot to change in different notebooks i keep here for same result
    shift = 0.02 if seed == 42 else 0.03
    train_loader.dataset.shift = shift
    
    scaler = GradScaler()

    for epoch in range(num_epochs):


        learning_rate = 1e-4
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.999)
        )
        
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = f"model_all_{epoch}_{seed}.bin"
        torch.save(model.state_dict(), checkpoint_path)

    # Free GPU memory
    torch.cuda.empty_cache()


for seed in [80,120]:
    train_model(seed)

