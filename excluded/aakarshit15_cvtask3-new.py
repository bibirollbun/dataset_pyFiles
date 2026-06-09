import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import cv2 as cv
import gc
from IPython.display import clear_output


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from sklearn.model_selection import train_test_split
from torchvision import models, transforms


!pip install pydicom -q


import pydicom
import numpy as np
import cv2

def read_xray(file_path, img_size=None):
    """
    Read the DICOM data and get the image.
    Args:
        file_path: The path of the DICOM file.
        img_size: Size of the output image as a tuple (width, height).
    Returns:
        Preprocessed image as a NumPy array.
    """

    # Read the DICOM file
    dicom = pydicom.dcmread(file_path)

    # Extract the pixel array from the DICOM file
    img = dicom.pixel_array

    # Check if the image is monochrome and invert if necessary
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = np.max(img) - img  # Invert pixel values
        # Inverting is necessary because MONOCHROME1 represents higher values as darker,
        # while MONOCHROME2 represents higher values as brighter.

    # Resize the image if a target size is specified
    if img_size:
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    # Add a channel dimension at the first axis (required for many ML models)
    img = img[np.newaxis, ...]

    # Normalize the image to the range [0, 1]
    img = img / np.max(img)

    # Convert to float32
    img = img.astype(np.float32)

    return img


def patchify(batch, patch_size):
    b, h, w, c = batch.shape
    ph, pw = patch_size

    # Calculate required padding for height and width
    pad_h = (ph - h % ph) % ph
    pad_w = (pw - w % pw) % pw

    # Pad the batch along height and width dimensions
    batch = nn.functional.pad(batch, (0, 0, 0, pad_w, 0, pad_h, 0, 0))  # Padding format: (dim3-right, dim3-left, dim2-right, dim2-left, ...)
    
    nh, nw = (h + pad_h) // ph, (w + pad_w) // pw

    # Reshape and permute to get patches
    batch_patches = batch.reshape(b, nh, ph, nw, pw, c)
    batch_patches = batch_patches.permute(0, 1, 3, 2, 4, 5)

    return batch_patches


FILE_PATH = ('/kaggle/input/rsna-breast-cancer-detection/'
             'train_images/10006/1459541791.dcm')

img = read_xray(FILE_PATH, img_size=(512, 512))

batch = torch.tensor(img[None])
patch_size = (16, 16)
batch_patches = patchify(batch, patch_size)

patches = batch_patches[0]
c, nh, nw, ph, pw = patches.shape

plt.figure(figsize=(5, 5))
plt.imshow(img[0], cmap="gray")
plt.axis("off")

plt.figure(figsize=(5, 5))
for i in range(nh):
    for j in range(nw):
        plt.subplot(nh, nw, i * nw + j + 1)
        plt.imshow(patches[0, i, j], cmap="gray")
        plt.axis("off")


def get_mlp(in_features, hidden_units, out_features):
    """
    Returns a Multi-Layer Perceptron (MLP) head.

    Args:
        in_features: Number of input features.
        hidden_units: List of integers, representing the number of units in hidden layers.
        out_features: Number of output features.

    Returns:
        A Sequential model with Linear and ReLU layers.
    """
    dims = [in_features] + hidden_units + [out_features]
    layers = []

    # Add Linear + ReLU for hidden layers
    for dim1, dim2 in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(dim1, dim2))
        layers.append(nn.ReLU())

    # Add the final Linear layer
    layers.append(nn.Linear(dims[-2], dims[-1]))

    # Return the Sequential model
    return nn.Sequential(*layers)


class Img2Seq(nn.Module):
    """
    This layer takes a batch of images as input and
    returns a batch of sequences.

    Shape:
        input: (b, h, w, c)
        output: (b, s, d)
    """
    def __init__(self, img_size, patch_size, n_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        nh, nw = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        n_tokens = nh * nw

        # Dimension of each patch flattened
        token_dim = patch_size[0] * patch_size[1] * n_channels
        self.linear = nn.Linear(token_dim, d_model)

        # Learnable parameters: CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model))

    def __call__(self, batch):
        # Patchify the input batch
        batch = patchify(batch, self.patch_size)

        # Get the dimensions of the patched batch
        b, nh, nw, ph, pw, c = batch.shape

        # Flatten the patches and permute the dimensions
        batch = batch.permute(0, 1, 2, 5, 3, 4).reshape(b, nh * nw, -1)

        # Apply the linear transformation to project into d_model dimensions
        batch = self.linear(batch)

        # Expand cls_token to match the batch size
        cls = self.cls_token.expand(b, -1, -1)

        # Add positional embeddings
        emb = batch + self.pos_emb

        # Concatenate the cls_token and the embeddings
        return torch.cat([cls, emb], axis=1)



class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        n_channels,
        d_model,
        nhead,
        dim_feedforward,
        blocks,
        mlp_head_units,
        n_classes,
    ):
        super().__init__()
        """
        Args:
            img_size: Size of the image
            patch_size: Size of the patch
            n_channels: Number of image channels
            d_model: The number of features in the transformer encoder
            nhead: The number of heads in the multiheadattention models
            dim_feedforward: The dimension of the feedforward network model in the encoder
            blocks: The number of sub-encoder-layers in the encoder
            mlp_head_units: The hidden units of mlp_head
            n_classes: The number of output classes
        """
        self.img2seq = Img2Seq(img_size, patch_size, n_channels, d_model)
        
        # Create an encoder layer for the transformer using GELU activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
        )
        
        # Create an encoder block with the specified number of layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=blocks)
        
        # Define the MLP head for classification
        self.mlp = get_mlp(d_model, mlp_head_units, n_classes)
        
        # Define the output activation function (sigmoid for binary classification, softmax for multi-class)
        self.output = nn.Sigmoid() if n_classes == 1 else nn.Softmax(dim=1)

    def forward(self, batch):
        # Convert the input images into sequences
        batch = self.img2seq(batch)
        
        # Pass the sequences through the transformer encoder
        batch = self.transformer_encoder(batch)
        
        # Use only the [CLS] token's output for classification
        batch = batch[:, 0, :]
        
        # Pass through the MLP head
        batch = self.mlp(batch)
        
        # Apply the output activation function
        output = self.output(batch)
        return output



class ViT(nn.Module):
    def __init__(self, img_size, patch_size, n_channels, d_model, nhead, dim_feedforward, blocks, mlp_head_units, n_classes):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.blocks = blocks
        self.mlp_head_units = mlp_head_units
        self.n_classes = n_classes

        # Define patch embedding
        self.patch_embedding = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Define transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=blocks)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, mlp_head_units[0]),
            nn.ReLU(),
            nn.Linear(mlp_head_units[0], mlp_head_units[1]),
            nn.ReLU(),
            nn.Linear(mlp_head_units[1], n_classes)
        )

    def img2seq(self, batch):
        """
        Converts the image batch into a sequence of patches.
        """
        batch_size, _, height, width = batch.shape

        # Apply patch embedding
        patches = self.patch_embedding(batch)
        patches = patches.flatten(2).transpose(1, 2)  # Flatten patches and transpose for sequence input
        
        return patches

    def forward(self, batch):
        """
        Forward pass through the model.
        """
        # Convert the input images into sequences (patches)
        batch = self.img2seq(batch)

        # Check the shape of batch after img2seq for debugging
        # print(f"Shape of batch after img2seq: {batch.shape}")  # Debug print

        # Pass the sequence through the transformer encoder
        batch = self.transformer_encoder(batch)

        # Reshape the output before passing through the final MLP head
        # batch has shape (batch_size, seq_length, d_model)
        batch = batch.mean(dim=1)  # Global average pooling (or any other pooling)

        # Pass the output through the MLP head to get predictions
        output = self.mlp_head(batch)
        return output


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


device


class RSNADataset(Dataset):

    def __init__(self, df, img_path, device):
        """
        Args:
            df: DataFrame containing patient_id, image_id, and cancer labels.
            img_path: Path to the directory containing image files.
            device: Torch device (e.g., 'cuda' or 'cpu').
        """
        self.df = df
        self.img_path = img_path
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract patient_id, image_id, and cancer label from the DataFrame
        patient_id, image_id, cancer = self.df.iloc[idx][['patient_id', 'image_id', 'cancer']]

        # Construct the full file path for the image
        file_path = os.path.join(self.img_path, f'{patient_id}_{image_id}.png')

        # Read the image file
        file = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if file is None:
            raise FileNotFoundError(f"Image file not found at path: {file_path}")

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        file = clahe.apply(file)

        # Normalize the image to the range [0, 1]
        file = file / 255.0

        # Convert the image to a torch tensor with an additional channel dimension
        X = torch.tensor(file[np.newaxis, :, :].astype('float32')).to(self.device)

        # Convert the label to a torch tensor
        y = torch.tensor([cancer]).float().to(self.device)

        return X, y



# Read the dataframe from CSV
df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')

# Apply value_counts to get class distribution in the 'cancer' column
counts = df['cancer'].value_counts()

# Apply weights based on class distribution
df['weights'] = df['cancer'].apply(lambda x: 1 / counts[x])

# Split the data into train and validation sets, stratifying by the 'cancer' column
train_df, val_df = train_test_split(df, test_size=0.25, stratify=df['cancer'])


img_path = '/kaggle/input/rsna-breast-cancer-512-pngs'

train_samples = 1000  # Number of samples for the training set
val_samples = 500     # Number of samples for the validation set

# Create the datasets with the device argument
train_ds = RSNADataset(train_df, img_path, device)
val_ds = RSNADataset(val_df, img_path, device)

# WeightedRandomSampler for training set
train_sampler = WeightedRandomSampler(
    train_df['weights'].values, 
    train_samples
)

# DataLoader for training set
train_loader = DataLoader(
    train_ds, 
    batch_size=8, 
    sampler=train_sampler
)

# WeightedRandomSampler for validation set
val_sampler = WeightedRandomSampler(
    val_df['weights'].values, 
    val_samples
)

# DataLoader for validation set
val_loader = DataLoader(
    val_ds, 
    batch_size=8, 
    sampler=val_sampler
)


# Initialize model
model = ViT(
    img_size = (512, 512),
    patch_size = (16, 16),
    n_channels = 1,
    d_model = 1024,
    nhead = 4,
    dim_feedforward = 1024,
    blocks = 8,
    mlp_head_units = [512, 512],
    n_classes = 1,
).to(device)

# Set up the optimizer (Adam optimizer for the model)
optimizer = Adam(model.parameters(), lr=1e-4)

# Set the loss function (BCE Loss for binary classification)
criterion = nn.BCEWithLogitsLoss()

# Create the trainer
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

# Define evaluation metrics
val_metrics = {
    "bce": Loss(criterion)
}

# Create the evaluator
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


log_interval = 10
max_epochs = 5
best_loss = float('inf')

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

pbar = ProgressBar()
pbar.attach(trainer, ['loss'])

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    global best_loss
    evaluator.run(val_loader)
    loss = evaluator.state.metrics['bce']
    
    # Save the best model if the validation loss improves
    if loss < best_loss:
        best_loss = loss  # Update the best_loss with the new lower loss
        torch.save(model.state_dict(), 'best_model_vit.pt')  # Save the model's state_dict

    print(f"Validation Results - Epoch: {trainer.state.epoch} Avg loss: {loss:.2f}")

output_state = trainer.run(train_loader, max_epochs=max_epochs)

