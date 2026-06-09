!pip install git+https://github.com/copick/copick-utils.git matplotlib tqdm copick 
!pip install -q "monai-weekly[mlflow]"


!pip install zarr


import os
import shutil
config_blob = """{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 128],
            "radius": 60,
            "map_threshold": 0.0418
        },
        {
            "name": "beta-amylase",
            "is_particle": true,
            "pdb_id": "1FA2",
            "label": 2,
            "color": [153,  63,   0, 128],
            "radius": 65,
            "map_threshold": 0.035
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 128],
            "radius": 90,
            "map_threshold": 0.0578
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 128],
            "radius": 150,
            "map_threshold": 0.0374
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 128],
            "radius": 130,
            "map_threshold": 0.0278
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "pdb_id": "6N4V",            
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 135,
            "map_threshold": 0.201
        }
    ],

    "overlay_root": "/kaggle/working/overlay",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "/kaggle/input/czii-cryo-et-object-identification/train/static"
}"""



copick_config_path = "/kaggle/working/copick.config"
ouput_overlay = "/kaggle/working/overlay"

with open(copick_config_path,"w") as f:
    f.write(config_blob)

source_dir =  '/kaggle/input/czii-cryo-et-object-identification/train/overlay'
destination_dir = '/kaggle/working/overlay'

for root, dirs,files in os.walk(source_dir):
    relative_path = os.path.relpath(root,source_dir)
    target_dir= os.path.join(destination_dir,relative_path)
    os.makedirs(target_dir,exist_ok = True)
    #print(relative_path)
    #print(root)
    #print(dirs)
    #print(files)
    #print("-----")

    # copy and rename each file
    for file in files:
        if file.startswith("curation_0_"):
            new_filename = file
        else:
            new_filename = f"curation_0_{file}"

        #Define full paths for the source and destination files
        source_file = os.path.join(root,file)
        destination_file = os.path.join(target_dir,new_filename)

        # copy the file with the new name
        shutil.copy2(source_file,destination_file)
        print(f"Copied {source_file} to {destination_file}")


import os
import numpy as np
from pathlib import Path

import torch
import torchinfo
import zarr,copick
from tqdm import tqdm
from monai.data import DataLoader, Dataset,CacheDataset,decollate_batch
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
from monai.networks.nets import UNet
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
import mlflow
import mlflow.pytorch


root = copick.from_file(copick_config_path)
copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size =10
tomo_type="denoised"


#print(root.version)             # "1.0.0"
#print(root.static_root)         # "/kaggle/input/czii-cryo-et-object-identification/train/static"
print(root.pickable_objects)    # List of pickable objects
print(root.runs)


from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
from collections import defaultdict

#Just do this once
generate_masks = True

if generate_masks:
    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]['label'] = object.label
            target_objects[object.name]['radius'] = object.radius

    for run in tqdm(root.runs):
        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram(tomo_type).numpy()
        target = np.zeros(tomo.shape,dtype=np.uint8)
        for pickable_object in root.pickable_objects:
            pick = run.get_picks(object_name = pickable_object.name, user_id = "curation")
            if len(pick):
                target = segmentation_from_picks.from_picks(pick[0],
                                                            target,
                                                            target_objects[pickable_object.name]['radius']*0.8,
                                                            target_objects[pickable_object.name]['label']
                                                           )
                write.segmentation(run, target, copick_user_name, name=copick_segmentation_name)
        


data_dicts = []
for run in tqdm(root.runs):
    tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type).numpy()
    segmentation = run.get_segmentations(name=copick_segmentation_name, user_id=copick_user_name, voxel_size=voxel_size, is_multilabel=True)[0].numpy()
    data_dicts.append({"name": run.name, "image": tomogram, "label": segmentation})
    
print(np.unique(data_dicts[0]['label']))


data_dicts[0]['label'].shape


data_dicts[0]['image'].shape # (184, 630, 630)


for i in range(7):
    with open(f"train_image_{data_dicts[i]['name']}.npy", 'wb') as f:
        np.save(f, data_dicts[i]['image'])
        
    with open(f"train_label_{data_dicts[i]['name']}.npy", 'wb') as f:
        np.save(f, data_dicts[i]['label'])


!ls -lh




