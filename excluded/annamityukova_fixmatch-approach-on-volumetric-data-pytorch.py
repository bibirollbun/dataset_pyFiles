!pip install -q pylibraft-cu12==24.12.0 rmm-cu12==24.12.0 pylibcugraph-cu12==24.12.0


!pip install -q torchio


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from PIL import Image
import math
import numpy as np
import random
from skimage import io
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torch.nn.functional as F
import torchio as tio
import torch.optim as optim
from torch.cuda.amp import autocast
import time
import warnings
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" #optimise GPU usage
warnings.filterwarnings("ignore") #ignore warnings


input_path = "/kaggle/input/forams-classification-2025"

# Load the labelled data
df_labelled = pd.read_csv(f"{input_path}/labelled.csv")
df_unlabelled = pd.read_csv(f"{input_path}/unlabelled.csv")

print("Number of labelled samples: ", len(df_labelled))
print("Number of unlabelled samples: ", len(df_unlabelled)) 

# Some of the first labelled rows
print("First few rows of labelled dataset: ")
df_labelled.head()



# Labels distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df_labelled, x='label', order=df_labelled['label'].value_counts().index)
plt.title("Distribution of Labels")
plt.show()


# Defining path for labelled volumes
labelled_volumes = "/kaggle/input/forams-classification-2025/volumes/volumes/labelled"

# Merging all images in one folder
labelled_images = f"{input_path}/visualizations/visualizations/labelled"
unlabelled_images = f"{input_path}/visualizations/visualizations/unlabelled"

image_paths = []

# Merge images in one folder
for image_id in df_labelled['id']:
    image_filename = image_id.replace("labelled_", "labelled_foram_") + ".jpg"
    full_path = os.path.join(labelled_images, image_filename)
    image_paths.append(full_path)

for image_id in df_unlabelled['id']:
    image_filename = f"foram_{int(image_id):05}.jpg"
    full_path = os.path.join(unlabelled_images, image_filename)
    image_paths.append(full_path)


# Pick up 1 random labelles sample to visualise it
idx = random.choice(range(len(df_labelled))) # random index from labelled data
row = df_labelled.iloc[idx]
image_id = row['id']
label = row['label']

# Image filename and image path
labelled_filename_base = f"labelled_foram_{image_id.split('_')[1]}" # base name for the labelled images (e.g labelled_foram_00129)
image_path = f"{labelled_images}/{labelled_filename_base}.jpg"

# labelled_filename base is included in the name of volume, however scaling factor is also included in the name.
# Hence, we need to match volume name using regular expressions. 

matched_volume = next(
    (f for f in os.listdir(labelled_volumes)
     if f.startswith(labelled_filename_base) and f.endswith('.tif')),
    None
)
volume_path = f"{labelled_volumes}/{matched_volume}"

# Read the volume using io (to get )
volume = io.imread(volume_path)
print(f"Shape of the volume array: {volume.shape}") # (Depth, Height, Width) = (128, 128, 128)

# Get the scaling factor using regular expressions
match = re.search(r'_sc_(\d+_\d+)', matched_volume)
scaling_factor = float(match.group(1).replace('_', '.')) if match else 1.0
print(f"Scaling Factor: {scaling_factor}")

# Open and plot visualisation
img = Image.open(image_path)
plt.imshow(img) # (-0.5, 299.5, 599.5, -0.5) --> width: 300 pixels, height: 600 pixels
plt.title(f" Label is {label}", fontsize=12)
plt.axis('off')

# Plot all 128 slices on 16x8 grid
num_slices = volume.shape[0]  # number of slices (128)
rows, cols = 16, 8 

fig, axes = plt.subplots(rows, cols, figsize=(20, 40))  # adjust figsize for readability
fig.suptitle("Plot of the slices", fontsize=18)

for i in range(num_slices):
    r, c = divmod(i, cols)
    axes[r, c].imshow(volume[i, :, :], cmap='gray')  # Show slice i
    axes[r, c].set_title(f"slice={i+1}", fontsize=8)
    axes[r, c].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # adjust to fit suptitle

plt.show()


# Helper function to extract file IDs and labels from the CSV.
def get_ids_and_labels(labelled, df):
    file_ids = df["id"].values
    if not labelled:
        # Unlabeled samples get a dummy label (-1)
        labels = [-1] * len(file_ids)
    else:
        labels = df["label"].values
    return file_ids, labels
    
# Transforms
# Weak augmentations 
weak_transform = tio.Compose([
    tio.RandomFlip(axes=('LR', 'AP', 'IS')),
    tio.RandomAffine(scales=(0.95, 1.05), degrees=15),
    tio.RandomBlur(std=(0.3, 1.5)),  # Random Gaussian blur
    tio.RandomGamma(log_gamma=(-0.8, 0.8)), 
    tio.ZNormalization(),
])

# Strong augmentations 
strong_transform = tio.Compose([
    tio.RandomFlip(axes=('LR', 'AP', 'IS')),  # Flip along any of the 3 axes
    tio.RandomAffine(scales=(0.9, 1.1), degrees=30),  # Afine variations
    tio.RandomBlur(std=(0.5, 1.7)),  # Random Gaussian blur
    tio.RandomGamma(log_gamma=(-0.8, 0.8)),  # Adjust contrast via gamma correction
    tio.RandomNoise(std=(0.02, 0.05)),  # Apply Gaussian noise to simulate acquisition artifacts
    tio.ZNormalization(),  # Normalize to zero mean and unit variance
])

# Dataset
class ForamDataset3D(Dataset):
    def __init__(self, volume_dir, csv_path, labelled=False, weak_transform=None, strong_transform=None):
        self.volume_dir = volume_dir
        self.labelled = labelled
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.df = pd.read_csv(csv_path)
        self.file_ids, self.labels = get_ids_and_labels(labelled, self.df)

    def __len__(self):
        return len(self.file_ids)  # Return the total number of samples

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]
        volume_dir = self.volume_dir
        
        if self.labelled:
            # file_id might be 'labelled_00000', extract numeric part after 'labelled_'
            numeric_id = file_id.split('_')[1]  # e.g. '00000'
            prefix = "labelled_foram_"
            # filenames have a suffix like _sc_0_752.tif - we need to find matching file
            search_prefix = prefix + numeric_id
        else:
            # unlabelled IDs are like '1', so zero-pad to 5 digits (assuming 5-digit IDs)
            numeric_id = str(file_id).zfill(5)  # '1' -> '00001'
            prefix = "foram_"
            search_prefix = prefix + numeric_id
        
        # Now find the file matching search_prefix (e.g. "labelled_foram_00000")
        matched_volume = next(
            (f for f in os.listdir(volume_dir) if f.startswith(search_prefix) and f.endswith('.tif')),
            None
        )
        if matched_volume is None:
            raise FileNotFoundError(f"No file matching prefix {search_prefix} found in {volume_dir}")
        
        filepath = os.path.join(volume_dir, matched_volume)
        volume = io.imread(filepath)[25:-25]
        volume = np.expand_dims(volume, axis=0)
        subject = tio.Subject(volume=tio.ScalarImage(tensor=volume))
        
        if not self.labelled:
            subject_weak = self.weak_transform(subject) if self.weak_transform else subject
            subject_strong = self.strong_transform(subject) if self.strong_transform else subject
            return subject_weak['volume'].data.float(), subject_strong['volume'].data.float(), label
        else:
            subject_weak = self.weak_transform(subject) if self.weak_transform else subject
            return subject_weak['volume'].data.float(), label




# Load your CSV with columns 'id' and 'label' (assuming 'label' is the class column)
df = pd.read_csv("/kaggle/input/forams-classification-2025/labelled.csv")

# Stratified split — keeps class proportions
train_df, val_df = train_test_split(
    df,
    test_size=0.2,           # 20% validation, 80% training
    stratify=df['label'],    # stratify on the label column
    random_state=42          # for reproducibility
)

print("Train class distribution:\n", train_df['label'].value_counts())
print("Test class distribution:\n", val_df['label'].value_counts())

# Optional: save splits to CSV
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("test_split.csv", index=False)



# Check if everything works as expected

# directories
labelled_volumes_dir = '/kaggle/input/forams-classification-2025/volumes/volumes/labelled'
labelled_train_csv_path = "/kaggle/working/train_split.csv"
labelled_test_csv_path = "/kaggle/working/test_split.csv"

unlabelled_volumes_dir = '/kaggle/input/forams-classification-2025/volumes/volumes/unlabelled'
unlabelled_csv_path = "/kaggle/input/forams-classification-2025/unlabelled.csv"

labelled_train_dataset = ForamDataset3D(
    labelled_volumes_dir, 
    labelled_train_csv_path, 
    labelled=True,
    weak_transform=weak_transform  # Pass the weak transformation
)

labelled_test_dataset = ForamDataset3D(
    labelled_volumes_dir, 
    labelled_test_csv_path, 
    labelled=True,
    weak_transform=weak_transform  # Pass the weak transformation
)

unlabelled_dataset = ForamDataset3D(
    unlabelled_volumes_dir,
    unlabelled_csv_path,
    labelled=False,
    weak_transform=weak_transform,   # Pass weak augmentation
    strong_transform=strong_transform  # Pass strong augmentation
)

labelled_volume, labelled_label = labelled_train_dataset[120]
unlabelled_volume1, unlabelled_volume2, unlabelled_label = unlabelled_dataset[6072]

print(f"unabelled dataset volume without augmentation: {unlabelled_volume1.shape}, volume for the input for augmentation: {unlabelled_volume2}, label: {unlabelled_label}")


# Plotting the strongly augmented data to see how the transforms look like

# Ensure the volume has the correct shape
unlabelled_volume = unlabelled_volume2.squeeze()
print(f"Volume shape after squeeze: {unlabelled_volume.shape}")

num_slices = unlabelled_volume.shape[0]  # Number of slices
print(f"Number of slices: {num_slices}")

rows, cols = 13, 6  # Adjust grid size for readability
fig, axes = plt.subplots(rows, cols, figsize=(20, 40))
fig.suptitle("Unlabelled Volume Slices after srong augmnetation", fontsize=18)

# Loop through slices and display them
for i in range(min(num_slices, rows * cols)):  # Ensure we don't exceed available slices
    r, c = divmod(i, cols)
    axes[r, c].imshow(unlabelled_volume[i, :, :], cmap='gray')  # Show slice i
    axes[r, c].set_title(f"Slice {i+1}", fontsize=8)
    axes[r, c].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust to fit suptitle
plt.show()



# Sourse: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/densenet.py
class _DenseLayer(nn.Sequential):
    """ Internal building block - Dense layer 
    Args: 
    n_layers (int) - number of layers
    n_input_features (int) - number of input features
    growth_rate (int) - growth rate (how many new feature maps are added after each layer)
    drop_rate (float) - drop_rate (probability with which features will be dropped. This helps overcome overfitting)
    bn_size - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):
    """ Transition Layer
    Used to downsample the feature maps calculated by Dense Block and 
    to reduce computational load
    """

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """DenseNet model class
    Args:
        n_input_channels (int) - how many input channels, 3 if RGB img
        conv_1_t_size (int) - size of the first convolution along the depth/time dimension.
        conv_1_t_stride (int) - stride for the first convolution along the depth/time dimension.
        no_max_pool (boolean) - whether to skip the initial MaxPooling layer after the first conv
            (False -> (default) maxpool after first conv layer -> downsampling of the input,
             True -> no maxpool)
        growth_rate (int) - how many new features are added after each layer
        block_config (list of 4 ints) - how many layers in each pooling block
        n_init_features (int) - number of features for first layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
           (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer (should be between 0.2-0.5)
        n_classes (int) - number of classification classes, in our case, 14   
    """

    def __init__(self,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=14):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 7, 7),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out,
                                    output_size=(1, 1,
                                                 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def generate_DenseNet121(model_depth=121, **kwargs):
    assert model_depth == 121 

    # Call the DenseNet model constructor
    model = DenseNet(num_init_features=64,
                     growth_rate=32,
                     block_config=(6, 12, 24, 16),  
                     **kwargs)  # Pass additional kwargs (like num_classes)

    return model


# Source: https://github.com/kekmodel/FixMatch-pytorch/tree/master


# Utility functions

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEMA(object):
    def __init__(self, model, decay):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])



class Args:
    # hardware
    world_size    = 1
    local_rank    = 0
    num_workers   = 4
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer        = None
    out           = 'result'

    # data / dataset
    dataset       = 'Foram3D'     
    num_labeled   = 210
    expand_labels = False

    # model / arch
    use_ema       = True
    ema_decay     = 0.999
    num_classes   = 14

    # training schedule
    total_steps           = 2**20
    eval_step             = 1024
    start_epoch           = 0
    batch_size            = 7
    mu                    = 3 # unlabeled to labeled batch ratio
    lambda_u              = 1.0
    threshold             = 0.95
    T                     = 1.0 # pseudo-label temperature
    epochs                = 10

    # optimizer / lr
    lr              = 0.001
    wdecay          = 5e-4
    nesterov        = True
    warmup          = 0            # warmup steps
    label_smoothing = 0.1

    # reproducibility & logging
    seed          = 1234
    out           = 'result'
    amp           = False
    opt_level     = 'O1'
    no_progress   = False
    resume        = ''           # e.g. 'result/checkpoint.pth.tar

args = Args()


best_acc = 0.0
if args.seed is not None:
    set_seed(args)

args.out = "/kaggle/working/runs"
os.makedirs(args.out, exist_ok=True)
args.writer = SummaryWriter(log_dir=args.out)

criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

# DataLoaders
labelled_trainloader = DataLoader(
    labelled_train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers= args.num_workers,  
    drop_last=False,
)

test_loader = DataLoader(
    labelled_test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers= args.num_workers,  
    drop_last=False,
)


num_unlabelled =  18216 # Total samples
subset_size = 5000  # Desired subset size

# Split dataset into a subset (5000 samples) and the rest
subset_unlabelled, _ = random_split(unlabelled_dataset, [subset_size, num_unlabelled - subset_size])

# Create DataLoader for the subset
unlabelled_trainloader = DataLoader(
    subset_unlabelled,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
)

# Create the primary model and an EMA model.
model = generate_DenseNet121(121, num_classes=args.num_classes)
model = nn.DataParallel(model)
model.to(args.device)
ema_model = ModelEMA(model, args.ema_decay)

no_decay = ['bias', 'bn']
grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
    {'params': [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                      momentum=0.9, nesterov=args.nesterov)

#args.epochs = math.ceil(args.total_steps / args.eval_step)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, args.warmup, args.total_steps)

args.start_epoch = 0

if args.resume:
    logger.info("==> Resuming from checkpoint..")
    assert os.path.isfile(
        args.resume), "Error: no checkpoint directory found!"
    args.out = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if args.use_ema:
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, criterion):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            #print("inputs_x shape:", inputs_x.shape)
            #print("inputs_u_w shape:", inputs_u_w.shape)
            #print("inputs_u_s shape:", inputs_u_s.shape)
            #print("args.mu:", args.mu)
            #print("Concatenated shape:", torch.cat((inputs_x, inputs_u_w, inputs_u_s)).shape)
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            #Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            Lx = criterion(logits_x, targets_x)

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            print('Best top-1 acc: {:.2f}'.format(best_acc))
            print('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()



def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
    
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    
    print("top-1 acc: {:.2f}".format(top1.avg))
    print("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg



model.zero_grad()
train(args, labelled_trainloader, unlabelled_trainloader, test_loader,
      model, optimizer, ema_model, scheduler, criterion)


torch.save(model.state_dict(), "/kaggle/working/model.pth")


torch.save(ema_model.ema.state_dict(), "/kaggle/working/ema_model.pth")


model = generate_DenseNet121(121, num_classes=args.num_classes)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("/kaggle/working/model.pth"))
model.to(args.device)


ema_model = ModelEMA(model, args.ema_decay)
ema_model.ema.load_state_dict(torch.load("/kaggle/working/ema_model.pth"))


# make a class for the inference dataset
def get_ids( df):
    file_ids = df["id"].values
    return file_ids


class InferenceDataset(Dataset):
    def __init__(self, volume_dir, csv_path, transform=None):
        self.volume_dir = volume_dir
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.file_ids = get_ids(self.df)

    def __len__(self):
        return len(self.file_ids)  # Return the total number of samples

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        volume_dir = self.volume_dir
        numeric_id = str(file_id).zfill(5)  # '1' -> '00001'
        prefix = "foram_"
        search_prefix = prefix + numeric_id

        matched_volume = next(
            (f for f in os.listdir(volume_dir) if f.startswith(search_prefix) and f.endswith('.tif')),
            None
        )
        if matched_volume is None:
            raise FileNotFoundError(f"No file matching prefix {search_prefix} found in {volume_dir}")
  
        filepath = os.path.join(volume_dir, matched_volume)
        volume = io.imread(filepath)[25:-25]
        volume = np.expand_dims(volume, axis=0)
        subject = tio.Subject(volume=tio.ScalarImage(tensor=volume))
        return subject['volume'].data.float(), file_id



unlabelled_dataset_inference = InferenceDataset(
    unlabelled_volumes_dir,
    unlabelled_csv_path,
    transform=weak_transform
)


inference_volume, inference_id = unlabelled_dataset_inference[1008]

print(f"inference dataset volume: {unlabelled_volume.shape}, id: {inference_id}")


# Create DataLoader for the subset
inference_loader = DataLoader(
    unlabelled_dataset_inference,
    batch_size=8,
    shuffle=False,
    num_workers=8,
)


def evaluate_and_save_predictions(model, inference_loader, device=args.device, output_csv="submission.csv", temperature=1.0, suppress_class=12, suppress_strength=150):
    model.eval()
    predictions = []
    ids = []
    
    print(f"Starting evaluation... Number of batches: {len(inference_loader)}")
    
    with torch.no_grad():
        for batch_idx, (volumes, file_ids) in enumerate(inference_loader):
            print(f"Batch {batch_idx} - volumes shape: {volumes.shape}, file_ids: {file_ids}")
            volumes = volumes.to(device)
            
            outputs = model(volumes)  # raw logits: shape (batch_size, num_classes)
            print(f"Outputs shape: {outputs.shape}")
            
            # Apply temperature scaling
            scaled_outputs = outputs / temperature

            # ↓↓↓ Suppress logits for class 12 ↓↓↓
            scaled_outputs[:, suppress_class] -= suppress_strength

            # Print logits of first sample in batch
            print("Scaled & suppressed logits for first sample:", scaled_outputs[0])
            
            # Compute softmax
            probs = torch.softmax(scaled_outputs, dim=1)
            print("Softmax probabilities for first sample:", probs[0])
            
            max_prob, pred_label = torch.max(probs, dim=1)
            print("Max probabilities:", max_prob)
            print("Predicted labels:", pred_label)
            
            predictions.extend(pred_label.cpu().numpy())
            ids.extend(file_ids.cpu().numpy())
    
    # Save to CSV
    submission_df = pd.DataFrame({
        "id": ids,
        "label": predictions
    })
    submission_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")



evaluate_and_save_predictions(model, inference_loader, device=args.device, temperature=2.0)


x = torch.randn(1, 1, 78, 128, 128).to(args.device)
with torch.no_grad():
    logits = model(x)
    print(logits)

