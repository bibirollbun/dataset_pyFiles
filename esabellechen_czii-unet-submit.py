class CONFIG:
    DEPS_PATH = '/kaggle/input/cziidependencies'
    TRAIN_DATA_DIR="/kaggle/input/cziinumpy-dataset-exp"
    TEST_DATA_DIR="/kaggle/input/czii-cryo-et-object-identification/test/static"
    MODEL_DIR="/kaggle/input/cziiunet-light/model_weights1000_epoches.pth"


! cp -r /kaggle/input/cziidependencies/asciitree-0.3.3/ asciitree-0.3.3/
! pip wheel asciitree-0.3.3/asciitree-0.3.3/
!pip install asciitree-0.3.3-py3-none-any.whl
! pip install -q --no-index --find-links {CONFIG.DEPS_PATH} --requirement {CONFIG.DEPS_PATH}/requirements.txt


import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Union

import torch
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import(
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)


def calculate_patch_starts(dimension_size,patch_size):
    if dimension_size <= patch_size:
        return[0]

    # number og patch
    n_patches = np.ceil(dimension_size/patch_size)
    if n_patches ==1:
        return[0]

    total_overlap = (n_patches*patch_size - dimension_size)/(n_patches-1)
    positions = []
    for i in range(int(n_patches)):
        pos = int(i*(patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:
            positions.append(pos)
    return positions


def extract_3d_patches_minimal_overlap(arrays,patch_size):
    if not arrays or not isinstance(arrays,list):
        raise ValueError("Input must be a non-empty list of arrays")

    shape = arrays[0].shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have yhe same shape")

    if patch_size> min(shape):
        raise ValueError(f"patch_size({patch_size}) must be smaller than smallest dimension {min(shape)}")

    m,n,l = shape
    patches= []
    coordinates = []

    x_starts = calculate_patch_starts(m,patch_size)
    y_starts = calculate_patch_starts(n,patch_size)
    z_starts = calculate_patch_starts(l,patch_size)

    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch=arr[
                    x:x+patch_size,
                    y:y+patch_size,
                    z:z+patch_size
                    ]
                    patches.append(patch)
                    coordinates.append(((x,y,z)))
    return patches, coordinates

def reconstruct_array(patches, coordinates,original_shape):
    reconstructed= np.zeros(original_shape,dtype = np.int64)
    patch_size = patches[0].shape[0]
    for patch,(x,y,z) in zip(patches,coordinates):
        reconstructed[
            x:x+patch_size,
            y:y+patch_size,
            z:z+patch_size
        ]= patch
    return reconstructed

def dict_to_df(coor_dict, experiment_name):
    all_coords = []
    all_labels = []
    for label, coords in coor_dict.items():
        all_coords.append(coords)
        all_labels.extend([label]*len(coords))

    all_coords = np.vstack(all_coords)

    df = pd.DataFrame({
        'experiment': experiment_name,
        'particle_type': all_labels,
        'x':all_coords[:,0],
        'y':all_coords[:,1],
        'z':all_coords[:,2]
    })
    
    return df


copick_config_path = CONFIG.TRAIN_DATA_DIR + "/copick.config"

with open(copick_config_path,'r') as f:
    copick_config = json.load(f)

copick_config['static_root'] = CONFIG.TEST_DATA_DIR

copick_test_config_path = 'copick_test.config'

with open(copick_test_config_path,'w') as outfile:
    json.dump(copick_config,outfile)


import copick

root = copick.from_file(copick_test_config_path)
copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
tomo_type = "denoised"


# Non-random transform

inference_transforms = Compose([
    EnsureChannelFirstd(keys=["image"], channel_dim = "no_channel"),
    NormalizeIntensityd(keys = 'image'),
    Orientationd(keys = ["image"], axcodes="RAS")
])


import cc3d

id_to_name = {
    1: "apo-ferritin", 
    2: "beta-amylase",
    3: "beta-galactosidase", 
    4: "ribosome", 
    5: "thyroglobulin", 
    6: "virus-like-particle"
}


import pytorch_lightning as pl
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import Union, Tuple, List

class Model(pl.LightningModule):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 7,
        channels: Union[Tuple[int, ...], List[int]] = (48, 64, 80, 80),
        strides: Union[Tuple[int, ...], List[int]] = (2, 2, 1),
        num_res_units: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            strides=self.hparams.strides,
            num_res_units=self.hparams.num_res_units,
        )
        self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        self.train_loss = 0
        self.val_metric = 0
        self.num_train_batch = 0
        self.num_val_batch = 0
        
    def forward(self, x):
        return self.model(x)


channels = (48, 64, 80, 80)
strides_pattern = (2, 2, 1)       
num_res_units = 1
learning_rate = 1e-3
num_epochs = 1000

if str(CONFIG.MODEL_DIR).split(".")[1] =='ckpt':
    model = Model.load_from_checkpoint(CONFIG.MODEL_DIR,channels=channels, strides=strides_pattern, num_res_units=num_res_units, lr=learning_rate)
elif str(CONFIG.MODEL_DIR).split(".")[1] =='pth':
    model = Model(channels=channels, strides=strides_pattern, num_res_units=num_res_units, lr=learning_rate)
    model.load_state_dict(torch.load(CONFIG.MODEL_DIR))


model.eval()
model.to("cuda")


BLOB_THRESHOLD = 500
CERTAINTY_THRESHOLD=0.5

classes = [1,2,3,4,5,6]
with torch.no_grad():
    location_df = []
    for run in root.runs:
        print(run)

        tomo = run.get_voxel_spacing(10)
        tomo =tomo.get_tomogram(tomo_type).numpy()

        tomo_patches, coordinates = extract_3d_patches_minimal_overlap([tomo],96)
        tomo_patched_data = [{"image":img} for img in tomo_patches]

        tomo_ds = CacheDataset(data=tomo_patched_data,transform = inference_transforms, cache_rate=1.0)

        pred_masks=[]

        for i in range(len(tomo_ds)):
            input_tensor = tomo_ds[i]['image'].unsqueeze(0).to("cuda")
            model_output = model(input_tensor)
            probs = torch.softmax(model_output[0],dim=0)
            thresh_probs = probs > CERTAINTY_THRESHOLD
            _,max_classes = thresh_probs.max(dim=0)
            pred_masks.append(max_classes.cpu().numpy())
        reconstructed_mask = reconstruct_array(pred_masks, coordinates, tomo.shape)
        location ={}
        
        for c in classes:
            cc = cc3d.connected_components(reconstructed_mask == c)
            stats = cc3d.statistics(cc)
            zyx=stats['centroids'][1:]*10 #.012444 #https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895#3040071
            zyx_large = zyx[stats['voxel_counts'][1:] > BLOB_THRESHOLD]
            xyz =np.ascontiguousarray(zyx_large[:,::-1])

            location[id_to_name[c]] = xyz


        df = dict_to_df(location, run.name)
        location_df.append(df)
    
    location_df = pd.concat(location_df)



location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))
location_df.to_csv("submission.csv", index=False)


location_df




