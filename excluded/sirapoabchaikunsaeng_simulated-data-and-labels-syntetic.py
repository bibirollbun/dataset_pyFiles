!pip install git+https://github.com/copick/copick-utils.git matplotlib tqdm copick


import copick
import fileinput
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import zarr

from glob import glob
from tqdm import tqdm


def ndjson_to_pick(run, particle, src_path, dest_path):
    pick = {}
    pick['pickable_object_name'] = particle
    pick['user_id'] = 'curation'
    pick['session_id'] = '0'
    pick['run_name'] = run
    pick['voxel_spacing'] = None
    pick['unit'] = 'angstrom'
    pick['points'] = []

    lines = fileinput.input(files=[src_path])
    for line in lines:
        nd_point = json.loads(line)
        point = {}
        point['location'] = {}
        point['location']['x'] = 10.012*nd_point['location']['x']
        point['location']['y'] = 10.012*nd_point['location']['y']
        point['location']['z'] = 10.012*nd_point['location']['z']
        point['transformation_'] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        point['instance_id'] = 0
        pick['points'].append(point)
        
    lines.close()
    
    with open(dest_path, 'w') as f:
        f.write(json.dumps(pick))

from scipy.ndimage import gaussian_filter, median_filter

def denoise_tomogram(tomogram, method='gaussian', **kwargs):
    """
    Apply denoising to a tomogram.

    Parameters:
        tomogram (np.ndarray): The input tomogram to denoise.
        method (str): The denoising method ('gaussian' or 'median').
        kwargs: Parameters for the respective method.
    
    Returns:
        np.ndarray: The denoised tomogram.
    """
    if method == 'gaussian':
        return gaussian_filter(tomogram, sigma=kwargs.get('sigma', 1))
    elif method == 'median':
        return median_filter(tomogram, size=kwargs.get('size', 3))
    else:
        raise ValueError(f"Unsupported denoising method: {method}")


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
    }

}"""
# "static_root": "/kaggle/input/czii-cryo-et-object-identification/train/static"

copick_config_path = "/kaggle/working/copick.config"
output_overlay = "/kaggle/working/overlay"

with open(copick_config_path, "w") as f:
    f.write(config_blob)


pick_map = {
    'ferritin_complex': 'apo-ferritin',
    'beta_amylase': 'beta-amylase',
    'beta_galactosidase': 'beta-galactosidase',
    'cytosolic_ribosome': 'ribosome',
    'thyroglobulin': 'thyroglobulin',
    'pp7_vlp': 'virus-like-particle',
}

#/kaggle/input/czii10441/10441/**/*.ndjson
#ndjson_files = glob('/kaggle/input/czii-cryoet-simulated-training-data-ts*/**/*.ndjson',

ndjson_files = glob('/kaggle/input/czii10441/10441/**/*.ndjson',
                    recursive=True)

print(f"found {len(ndjson_files)} files")

for file in ndjson_files:
    print(file)
    run = file.split('/')[5]
    print(run)
    if "_" not in run:
        continue
    particle = pick_map[file.split('/')[10].split('-')[0]]
    dest_dir = f'/kaggle/working/overlay/ExperimentRuns/{run}/Picks'
    dest_path = f'{dest_dir}/curation_0_{particle}.json'
    os.makedirs(dest_dir, exist_ok=True)
    ndjson_to_pick(run, particle, file, dest_path)


root = copick.from_file(copick_config_path)

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
tomo_type = "denoised"


from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
from collections import defaultdict

# Just do this once
generate_masks = True

if generate_masks:
    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]['label'] = object.label
            target_objects[object.name]['radius'] = object.radius


    for run in tqdm(root.runs):
        # tomo = run.get_voxel_spacing(10)
        # tomo = tomo.get_tomogram(tomo_type).numpy()
        # target = np.zeros(tomo.shape, dtype=np.uint8)
        target = np.zeros((200, 630, 630), dtype=np.uint8)
        for pickable_object in root.pickable_objects:
            pick = run.get_picks(object_name=pickable_object.name, user_id="curation")
            if len(pick):  
                target = segmentation_from_picks.from_picks(pick[0], 
                                                            target, 
                                                            target_objects[pickable_object.name]['radius'] * 0.8,
                                                            target_objects[pickable_object.name]['label']
                                                            )
        write.segmentation(run, target, copick_user_name, name=copick_segmentation_name)


#tomograms = glob('/kaggle/input/czii-cryoet-simulated-training-data-ts*/**/Tomograms/**/*.zarr',

#/kaggle/input/czii10441/10441/**/*.ndjson

tomograms = glob('/kaggle/input/czii10441/10441/**/Tomograms/**/*.zarr',
                 recursive=True)
tomogram_map = {}
for t in tomograms:
    print(t.split('/')[5])
    tomogram_map[t.split('/')[5]] = t


print(tomogram_map)


num_tomograms = len(root.runs)
first_third = int(num_tomograms) / 3
second_third = 2 * int(num_tomograms) / 3
chunk_boundaries = [
    (0, first_third),
    (first_third, second_third),
    (second_third, num_tomograms),
]

chunk_index = 0
start_idx, end_idx = chunk_boundaries[chunk_index]


data_dicts = []
print(root.runs)
count = 0
for i, run in tqdm(enumerate(root.runs), total=num_tomograms):
    if not (start_idx <= i < end_idx):
        continue
    # if count == 7:
    #     break
    print(run.name)
    if run.name not in tomogram_map:
        print(f"Run '{run.name}' does not have a corresponding tomogram. Skipping.")
        continue
    tomogram_path = tomogram_map[run.name]
    tomogram = np.array(zarr.open(tomogram_path, mode='r')[0])[:184]
    tomogram_denoised = denoise_tomogram(tomogram, method='gaussian', sigma=1)  # Apply denoising

    segmentation = run.get_segmentations(name=copick_segmentation_name, user_id=copick_user_name, voxel_size=voxel_size, is_multilabel=True)[0].numpy()[:184]
    data_dicts.append({
        "name": run.name, 
        "original_image": tomogram,          # Storing original image
        "denoised_image": tomogram_denoised, # Storing denoised image
        "label": segmentation
    })
    count +=1
    
print(np.unique(data_dicts[0]['label']))


data_dicts[0]['label'].shape


data_dicts[0]['original_image'].shape





for i in range(len(data_dicts)):
    with open(f"train_original_image_{data_dicts[i]['name']}.npy", 'wb') as f:
        np.save(f, data_dicts[i]['original_image'])
        
    with open(f"train_image_{data_dicts[i]['name']}.npy", 'wb') as f:
        np.save(f, data_dicts[i]['denoised_image'])
        
    with open(f"train_label_{data_dicts[i]['name']}.npy", 'wb') as f:
        np.save(f, data_dicts[i]['label'])


!ls -lh











plt.figure(figsize=(15,5))

# Plot Original Image
plt.subplot(1,3,1)
plt.title("Original Image")
plt.xticks([])
plt.yticks([])
plt.imshow(data_dicts[0]['original_image'][92], cmap='gray', vmin=-2, vmax=2)

# Plot Denoised Image
plt.subplot(1,3,2)
plt.title("Denoised Image")
plt.xticks([])
plt.yticks([])
plt.imshow(data_dicts[0]['denoised_image'][92], cmap='gray', vmin=-2, vmax=2)

# Plot Label
plt.subplot(1,3,3)
plt.title("Label")
plt.xticks([])
plt.yticks([])
_ = plt.imshow(data_dicts[0]['label'][92])

plt.tight_layout()
plt.show()








# !ls -lah


# !mkdir -p /tmp/outputs
# !mv train* /tmp/outputs
# !mv overlay /tmp/outputs
# !mv copick.config /tmp/outputs
# !ls -lah /tmp/outputs


# !zip -r data.zip /tmp/outputs


print("done")


!ls -lah

