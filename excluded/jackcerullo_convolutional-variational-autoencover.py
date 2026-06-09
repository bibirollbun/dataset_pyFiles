!pip install tensorflow-probability
!pip install imageio
!pip install git+https://github.com/tensorflow/docs
!pip install zarr
!pip install fsspec
!pip install pydantic
!pip install trimesh
!pip install "copick[all]"
!pip install copick git+https://github.com/copick/copick-utils.git git+https://github.com/copick/DeepFindET.git


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
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 135,
            "map_threshold": 0.201
        },
        {
            "name": "membrane",
            "is_particle": false,
            "label": 8,
            "color": [100, 100, 100, 128]
        },
        {
            "name": "background",
            "is_particle": false,
            "label": 9,
            "color": [10, 150, 200, 128]
        }
    ],

    "overlay_root": "/kaggle/working/overlay",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "/kaggle/input/czii-cryo-et-object-identification/train/static"
}"""

copick_config_path = "/kaggle/working/copick.config"
output_overlay = "/kaggle/working/overlay"

with open(copick_config_path, "w") as f:
    f.write(config_blob)


# Setup new overlay directory
import os
import shutil

# Define source and destination directories
source_dir = '/kaggle/input/czii-cryo-et-object-identification/train/overlay'
destination_dir = '/kaggle/working/overlay'

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    # Create corresponding subdirectories in the destination
    relative_path = os.path.relpath(root, source_dir)
    target_dir = os.path.join(destination_dir, relative_path)
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy and rename each file
    for file in files:
        if file.startswith("curation_0_"):
            new_filename = file
        else:
            new_filename = f"curation_0_{file}"
            
        
        # Define full paths for the source and destination files
        source_file = os.path.join(root, file)
        destination_file = os.path.join(target_dir, new_filename)
        
        # Copy the file with the new name
        shutil.copy2(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")


from deepfindET.entry_points import step1
from deepfindET.utils import copick_tools
import matplotlib.pyplot as plt
import copick

%matplotlib inline

################## Input Parameters #################

# Config File
config = '/kaggle/working/copick.config'

# Query Tomogram
voxel_size = 10 
tomogram_algorithm = 'denoised'

# Output Name for the Segmentation Targets
out_name = 'remotetargets'
out_user_id = 'deepfindET'
out_session_id = '0'

# Read Copick Directory
copickRoot = copick.from_file(config)




[(obj.name, None, None, (obj.radius / voxel_size)) for obj in copickRoot.pickable_objects if obj.is_particle]


# Query Train Protein Coordiantes and any Associated Segmentations
train_targets = {}

# Define protein targets with their respective radii
# We can Provide two forms of inputs, either 
# ('protein-name',radius) or ('protein-name', 'user-id', 'session-id', 'radius')
targets = [(obj.name, None, None, (obj.radius / voxel_size)) for obj in copickRoot.pickable_objects if obj.is_particle]

# Set run_ids to None, indicating that targets will be generated for the entire CoPick project by default.
# If specific Run-IDs were provided, this variable would contain a list of those IDs.
run_ids = None


# Generate train target information
for t in targets:
    obj_name, user_id, session_id, radius = t
    info = {
        "label": copickRoot.get_object(obj_name).label,
        "user_id": user_id,
        "session_id": session_id,
        "radius": radius,
        "is_particle_target": True,
    }
    train_targets[obj_name] = info


# Define segmentation target (e.g., membrane)
seg_targets = [('membrane', None, None)]

# Generate segmentation target information
for s in seg_targets:
    obj_name, user_id, session_id = s
    info = {
        "label": copickRoot.get_object(obj_name).label,
        "user_id": user_id,
        "session_id": session_id,
        "radius": None,       
        "is_particle_target": False,                 
    }
    train_targets[obj_name] = info

# Call the create_train_targets function from step1 to generate the training targets for the 3D U-Net model.
# The function will use the parameters defined in the previous cells and the following inputs:
step1.create_train_targets(
    config,              # The configuration file path specifying various settings and parameters for the project.
    train_targets,       # A dictionary containing the target information for each protein or object to be segmented.
    run_ids,             # The list of Run-IDs for which to generate targets. None means targets for the entire project.
    voxel_size,          # The voxel size to be used in the tomogram data.
    tomogram_algorithm,  # The reconstruction algorithm used for the tomograms, e.g., 'wbp' (weighted back projection).
    out_name,            # The output name for the generated segmentation targets.
    out_user_id,         # The user ID under which the output targets will be saved.
    out_session_id,      # The session ID associated with the output, typically used for tracking purposes.
)


# Option 1: Query All RunIDs
# Retrieve all available Run-IDs from the CoPick project. This generates a list of Run-IDs by iterating over all runs in copickRoot.
run_ids = [run.name for run in copickRoot.runs]

# Option 2: Manually Specify Specific Run
# Define a specific Run-ID manually. This is useful for extracting volumes for a specific run.
runID = 'TS_6_4'

# Retrieve the specific run object from CoPick using the manually specified Run-ID.
copick_run = copickRoot.get_run(runID)

# Extract the segmentation target associated with the specified run.
# The function get_copick_segmentation retrieves the segmentation data (e.g., target volume) based on the run object,
# segmentation name, user ID, and session ID.
train_target = copick_tools.get_copick_segmentation(
    copick_run,                 # The run object obtained from CoPick for the specific Run-ID.
    segmentationName='remotetargets',  # The name of the segmentation target to retrieve.
    userID='deepfindET',        # The user ID under which the segmentation data is saved.
    sessionID='0'               # The session ID associated with the segmentation data.
)

# Retrieve the tomogram associated with the specified Run-ID from the CoPick project.
# The function get_copick_tomogram extracts the tomogram data, using the voxel size, algorithm, and Run-ID.
train_tomogram = copick_tools.get_copick_tomogram(
    copickRoot,                 # The root object for the CoPick project, containing all runs and associated data.
    voxelSize=voxel_size,       # The voxel size to be used for retrieving the tomogram.
    tomoAlgorithm='wbp',        # The reconstruction algorithm used for the tomogram, e.g., 'wbp' (weighted back projection).
    tomoID=runID                # The specific Run-ID for which the tomogram is being retrieved.
)


from deepfindET.entry_points import step1
from deepfindET.utils import copick_tools
import matplotlib.pyplot as plt
import copick

%matplotlib inline

################## Input Parameters #################

# Config File
config = '/kaggle/working/copick.config'

# Query Tomogram
voxel_size = 10 
tomogram_algorithm = 'denoised'

# Output Name for the Segmentation Targets
out_name = 'remotetargets'
out_user_id = 'deepfindET'
out_session_id = '0'

# Read Copick Directory
copickRoot = copick.from_file(config)


# Plot the images
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Tomogram')
plt.imshow(train_tomogram[90,],cmap='gray')
print(train_tomogram[90,])
plt.axis('off')

# Original Image
plt.subplot(1, 2, 2)
plt.title('Train Target')
plt.imshow(train_target[90,])
plt.axis('off')

plt.tight_layout()
plt.show()


from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time


import tensorflow as tf
import numpy as np

train_runs = []
for runID in run_ids:
    copick_run = copickRoot.get_run(runID)
    train_target = copick_tools.get_copick_segmentation(
        copick_run,                 # The run object obtained from CoPick for the specific Run-ID.
        segmentationName='remotetargets',  # The name of the segmentation target to retrieve.
        userID='deepfindET',        # The user ID under which the segmentation data is saved.
        sessionID='0'               # The session ID associated with the segmentation data.
    )
    train_tomogram = copick_tools.get_copick_tomogram(
        copickRoot,                 # The root object for the CoPick project, containing all runs and associated data.
        voxelSize=voxel_size,       # The voxel size to be used for retrieving the tomogram.
        tomoAlgorithm='wbp',        # The reconstruction algorithm used for the tomogram, e.g., 'wbp' (weighted back projection).
        tomoID=runID                # The specific Run-ID for which the tomogram is being retrieved.
    )
    tf_tomogram = np.array(train_tomogram[:])
    train_runs.append(tf_tomogram)
    print(tf_tomogram.shape)


new_train_runs = []
for run in train_runs:
    for image in run:
        new_train_runs.append(image)


print(np.array(new_train_runs).shape)


train_size = 1000
batch_size = 16
test_size = 288


train_dataset = new_train_runs[0:train_size]
test_dataset = new_train_runs[train_size:]

import tensorflow as tf
import cv2
def normalize_image(image):
    """Ensure image is in the range [0, 1] regardless of its original scale."""
    # If the image data is already in the expected range [0, 1], no changes are needed.
    image = (image+1)/2
#     image_min = np.min(image)
#     image_max = np.max(image)

# # Normalize to [0, 1]
#     image_normalized = (image - image_min) / (image_max - image_min)
    return image

train_dataset_normalized = [normalize_image(img[:256, :256]) for img in train_dataset]
test_dataset_normalized = [normalize_image(img[:256, :256]) for img in test_dataset]
train_dataset_normalized = np.expand_dims(train_dataset_normalized, axis=1)
test_dataset_normalized = np.expand_dims(test_dataset_normalized, axis=1)


# # Fill the new dimension with 0.5
# train_dataset_normalized[0] = 0.5
# test_dataset_normalized[0] = 0.5


print(train_dataset_normalized.shape)


print(train_dataset_normalized[0])


plt.imshow(train_dataset_normalized[0][0], cmap='gray')  # 'gray' colormap for grayscale
plt.axis('off')
plt.show()


pip install torchgan==0.6.0 torchvision==0.14.1


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F


class NumpyDataset(Dataset):
    """Custom Dataset for loading normalized NumPy arrays."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset.shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Return the image at index idx
        image = self.dataset[idx]
        # Convert to torch tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        return image_tensor


# Convert your normalized train and test datasets to Dataset objects
train_dataset_pytorch = NumpyDataset(train_dataset_normalized)
test_dataset_pytorch = NumpyDataset(test_dataset_normalized)
print(train_dataset_pytorch.shape)
# Create DataLoader for train and test datasets
train_loader = DataLoader(train_dataset_pytorch, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset_pytorch, batch_size=64, shuffle=False)



import torch
import torch.nn as nn
import torch.optim as optim
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss
from torchgan.trainer import Trainer

latent_dim = 128
image_size = 256
image_channels = 1

generator = DCGANGenerator(
    encoding_dims=latent_dim,
    out_size=image_size,
    out_channels=image_channels,
    last_nonlinearity=torch.nn.Sigmoid()
)

discriminator = DCGANDiscriminator(
    in_size=image_size,
    in_channels=image_channels,
    last_nonlinearity=torch.nn.Sigmoid()
)

# Define the optimizers for the generator and discriminator
optimizer_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define the loss functions
losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]

# Define the model dictionary
models = {
    "generator": {
        "name": DCGANGenerator,
        "args": {"out_channels": image_channels, "out_size": image_size, "step_channels": 64},
        "optimizer": {"name": optim.Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}
    },
    "discriminator": {
        "name": DCGANDiscriminator,
        "args": {"in_channels": image_channels, "in_size": image_size, "step_channels": 64},
        "optimizer": {"name": optim.Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}
    }
}

# Define the devices (for multi-GPU use, list all available GPUs, e.g., [0, 1])
devices = [0]  # Change to your device IDs (use [0] for CPU or a single GPU)

# Define the Trainer with other necessary parameters
trainer = Trainer(
    models,
    losses,
    sample_size=64,
    epochs=1000,
    ncritic=1,  # Train the generator and discriminator equally
    retain_checkpoints=3,  # Retain the last 3 checkpoints
    nrow=8,  # Arrange the generated images in 8 rows
    
)

# Assuming you have a DataLoader `train_loader` for your dataset
trainer.train(train_loader)

