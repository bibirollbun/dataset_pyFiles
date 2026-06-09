deps_path = '/kaggle/input/cziidependencies'


! cp -r /kaggle/input/cziidependencies/asciitree-0.3.3/ asciitree-0.3.3/


! pip wheel asciitree-0.3.3/asciitree-0.3.3/


!pip install asciitree-0.3.3-py3-none-any.whl


! pip install -q --no-index --find-links {deps_path} --requirement {deps_path}/requirements.txt


!pip show pytorch-lightning
!pip install --upgrade pytorch-lightning



from typing import List, Tuple, Union
import gc
import numpy as np
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


def calculate_patch_starts(dimension_size: int,patch_size:int)->List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.
    
    Parameters:
    -----------
    dimension_size : int
        Size of the dimension
    patch_size : int
        Size of the patch in this dimension
        
    Returns:
    --------
    List[int]
        List of starting positions for patches
    """
    if dimension_size <= patch_size:
        return [0]

    n_patches = np.ceil(dimension_size/patch_size)

    if n_patches ==1:
        return [0]

    total_overlap = (n_patches * patch_size-dimension_size)/(n_patches-1)
    positions = []
    for i in range(int(n_patches)):
        pos = int(i*(patch_size-total_overlap))
        if pos + patch_size> dimension_size:
            pos= dimension_size - patch_size
        if pos not in positions:
            positions.append(pos)
    return positions

def extract_3d_patches_minimal_overlap(arrays:List[np.ndarray],
                                       patch_size: int) -> Tuple[List[np.ndarray],List[Tuple[int,int,int]]]:
    """
    Extract 3D patches from multiple arrays with minimal overlap to cover the entire array.
    
    Parameters:
    -----------
    arrays : List[np.ndarray]
        List of input arrays, each with shape (m, n, l)
    patch_size : int
        Size of cubic patches (a x a x a)
        
    Returns:
    --------
    patches : List[np.ndarray]
        List of all patches from all input arrays
    coordinates : List[Tuple[int, int, int]]
        List of starting coordinates (x, y, z) for each patch
    """

    if not arrays or not isinstance(arrays,list):
        raise ValueError("Input must be a non-empty list of arrays")

    shape = arrays[0].shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")

    if patch_size > min(shape):
        raise ValueError(f"patch_size({patch_size}) must be smaller than smallest dimension {min(shape)}")
    m,n,l = shape
    patches =[]
    coordinates =[]

    x_starts = calculate_patch_starts(m,patch_size)
    y_starts = calculate_patch_starts(n, patch_size)
    z_starts = calculate_patch_starts(l, patch_size)

    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = arr[
                    x:x + patch_size,
                    y:y + patch_size,
                    z:z + patch_size
                    ]
                    patches.append(patch)
                    coordinates.append((x,y,z))
    return patches, coordinates

def reconstruct_array(patches: List[np.ndarray],
                      coordinates: List[Tuple[int,int,int]],
                      original_shape: Tuple[int,int,int])->np.ndarray:
    """
    Reconstruct array from patches.
    
    Parameters:
    -----------
    patches : List[np.ndarray]
        List of patches to reconstruct from
    coordinates : List[Tuple[int, int, int]]
        Starting coordinates for each patch
    original_shape : Tuple[int, int, int]
        Shape of the original array
        
    Returns:
    --------
    np.ndarray
        Reconstructed array
    """
    reconstructed = np.zeros(original_shape,dtype=np.int64)
    patch_size = patches[0].shape[0]

    for patch,(x,y,z) in zip(patches,coordinates):
        reconstructed[
            x:x + patch_size,
            y:y + patch_size,
            z:z + patch_size
        ] = patch
        
    return reconstructed


import pandas as pd

def dict_to_df(coord_dict,experiment_name):
    """
    Convert dictionary of coordinates to pandas DataFrame.
    
    Parameters:
    -----------
    coord_dict : dict
        Dictionary where keys are labels and values are Nx3 coordinate arrays
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['x', 'y', 'z', 'label']
    """
    all_coords = []
    all_labels = []
    for label, coords in coord_dict.items():
        all_coords.append(coords)
        all_labels.extend([label] * len(coords))

    all_coords = np.vstack(all_coords)

    df = pd.DataFrame({
        'experiment': experiment_name,
        'particle_type': all_labels,
        'x': all_coords[:,0],
        'y': all_coords[:,1],
        'z': all_coords[:,2]
    })

    return df


TRAIN_DATA_DIR="/kaggle/input/cziinumpy-dataset-exp"
TEST_DATA_DIR="/kaggle/input/czii-cryo-et-object-identification"


train_names = ['TS_5_4','TS_69_2','TS_6_6','TS_73_6','TS_86_3','TS_99_9']
valid_names = ['TS_6_4']

train_files = []
valid_files = []

for name in train_names:
    image = np.load(f"{TRAIN_DATA_DIR}/train_image_{name}.npy")
    label = np.load(f"{TRAIN_DATA_DIR}/train_label_{name}.npy")

    train_files.append({"image": image, "label":label})

for name in valid_names:
    image = np.load(f"{TRAIN_DATA_DIR}/train_image_{name}.npy")
    label = np.load(f"{TRAIN_DATA_DIR}/train_label_{name}.npy")

    valid_files.append({"image": image, "label":label})



print(train_files[0]['label'].shape)
print(train_files[0]['image'].shape)
print(valid_files[0]['label'].shape)
print(valid_files[0]['image'].shape)


non_random_transforms = Compose([
    EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS")
])

#train_images,train_labels = [dcts['image'] for dcts in train_files],[dcts['label'] for dcts in train_files]
#train_image_patches, _ = extract_3d_patches_minimal_overlap(train_images,96)
#train_label_patches, _ = extract_3d_patches_minimal_overlap(train_labels,96)
#train_files = [{"image":img, "label":lbl} for img,lbl in zip(train_image_patches,train_label_patches)]
#train_ds = CacheDataset(data= train_files, transform = non_random_transforms, cache_rate=1.0)
raw_train_ds  = CacheDataset(data= train_files, transform = non_random_transforms, cache_rate=1.0)

my_num_samples= 32
train_batch_size=16

random_transforms = Compose([
    RandCropByLabelClassesd(
        keys = ["image","label"],
        label_key = "label",
        spatial_size=[96,96,96],
        num_classes = 7,
        num_samples = my_num_samples
    ),
    RandRotate90d(keys=["image","label"],prob=0.5,spatial_axes=[0,2]),
    RandFlipd(keys=["image","label"],prob=0.5, spatial_axis = 0),
])
#train_ds = Dataset(data = raw_train_ds, transform = random_transforms)
train_ds = CacheDataset(data = raw_train_ds, transform = random_transforms, cache_rate=1.0)
train_files = [
    {"image": sub_file["image"], "label": sub_file["label"]}
    for file in train_ds
    for sub_file in file
]

train_loader = DataLoader(
    train_files,
    batch_size = train_batch_size,
    shuffle = True,
    num_workers = 4,
    pin_memory = torch.cuda.is_available()
)

del train_ds
gc.collect


print(len(train_loader))
for i in train_loader:
    print(i['label'].shape)
    break


val_images,val_labels = [dcts['image'] for dcts in valid_files],[dcts['label'] for dcts in valid_files]

val_image_patches, _ = extract_3d_patches_minimal_overlap(val_images,96) #98
val_label_patches, _ = extract_3d_patches_minimal_overlap(val_labels,96)

val_patched_data = [{"image":img, "label":lbl} for img,lbl in zip(val_image_patches,val_label_patches)]

valid_ds = CacheDataset(data=val_patched_data, transform = non_random_transforms,cache_rate=1.0)

valid_batch_size=16
val_loader = DataLoader(
    valid_ds,
    batch_size = valid_batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)


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

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_loss += loss
        self.num_train_batch += 1
        return loss
        
    def on_train_epoch_end(self):
        loss_per_epoch = self.train_loss / self.num_train_batch
        print(f"Epoch {self.current_epoch} - Average Train Loss: {loss_per_epoch:.4f}")
        self.log('train_loss', loss_per_epoch, prog_bar=True)
        self.train_loss = 0
        self.num_train_batch = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch['image'], batch['label']
            y_hat = self(x)
            metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels)(i) for i in decollate_batch(y_hat)]
            metric_val_labels = [AsDiscrete(to_onehot=self.hparams.out_channels)(i) for i in decollate_batch(y)]

            self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_metric = torch.mean(metrics)
            self.val_metric += val_metric 
            self.num_val_batch += 1
        return {'val_metric': val_metric}

    def on_validation_epoch_end(self):
        metric_per_epoch = self.val_metric / self.num_val_batch
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']  # Get the current learning rate
        print(f"------Epoch {self.current_epoch} - Learning Rate: {current_lr:.6f} - Average Val Metric: {metric_per_epoch:.4f}")
        self.log('val_metric', metric_per_epoch, prog_bar=True)
        self.val_metric = 0
        self.num_val_batch = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True),
            "monitor": "val_metric",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]



channels = (48, 64, 80, 80)
strides_pattern = (2, 2, 1)       
num_res_units = 1
learning_rate = 1e-3
num_epochs = 500

model = Model(channels=channels, strides=strides_pattern, num_res_units=num_res_units, lr=learning_rate)


torch.set_float32_matmul_precision('medium')

# Check if CUDA is available and then count the GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
else:
    print("No GPU available. Running on CPU.")
devices = list(range(num_gpus))
print(devices)


import matplotlib.pyplot as plt

# Create a custom callback to track metrics during training
class MetricPlotCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_metrics = []

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        # Save training loss at the end of each epoch
        self.train_losses.append(trainer.callback_metrics['train_loss'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Save validation loss at the end of each validation epoch
        self.val_metrics.append(trainer.callback_metrics['val_metric'].item())

    def plot_metrics(self):
        # Plot training and validation losses
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 6))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        # Plot validation metric
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_metrics, label='Validation Metric', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.title('Validation Metric Over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig("taining_epoches.png")
        print(f"Metrics plot saved.")
        plt.show()
        
checkpoint_callback = ModelCheckpoint(
    monitor="val_metric",            # Metric to monitor
    mode="max",                      # Maximize the monitored metric (use "min" for loss)
    save_top_k=1,                    # Save only the best model
    filename="best-checkpoint",  # Save format
    verbose=True                     # Print save messages
)

# Instantiate the callback
metric_plot_callback = MetricPlotCallback()

# Add this callback to your trainer
trainer = pl.Trainer(
    callbacks=[checkpoint_callback],
    max_epochs=num_epochs,
    accelerator="gpu",
    devices=[0],
    num_nodes=1,
    log_every_n_steps=10,
    enable_progress_bar=True,
)




trainer.fit(model, train_loader, val_loader)


# Call the plot method after training finishes
#metric_plot_callback.plot_metrics()



# Save the model manually
torch.save(model.state_dict(), "model_weights1000_epoches.pth")

# To load the model later:
#model.load_state_dict(torch.load("model_weights.pth"))





