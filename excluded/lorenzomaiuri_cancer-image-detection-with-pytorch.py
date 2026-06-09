import os
import json
from pathlib import Path
from tqdm.notebook import trange, tqdm
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings('ignore')

random.seed(2025)
torch.manual_seed(2025)

sns.set_context('notebook')
sns.set_style('white')

%matplotlib inline


if torch.cuda.is_available():
    print(f"Compatible GPU ({torch.cuda.get_device_name()}) found")
else:
    print(f"No compatible GPU found.")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CancerDataset(Dataset):
    """
    Custom Dataset for loading histopathologic cancer detection images.

    Args:
        data_dir (str or Path): Root directory containing the image data and labels.
        transform (callable, optional): Transformation function to apply to the images.
        data_type (str): Directory selection - "train", "test", or "val".
        num_samples (int, optional): Number of samples to randomly select from the dataset.
    """
    
    def __init__(self, data_dir, transform=None, data_type="train"):
        self.data_dir = Path(data_dir)
        self.data_type = data_type
        self.transform = transform

        # Define the image directory based on the data_type (e.g., train, test)
        image_dir = self.data_dir / data_type
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory '{image_dir}' not found.")        
        
        # Load all valid image files (.tif) in the directory
        all_files = list(image_dir.glob("*.tif"))

        # No sample
        num_samples = len(all_files)
        
        if num_samples > len(all_files):
            raise ValueError(f"num_samples ({num_samples}) exceeds available images ({len(all_files)}).")
        
        # Randomly select a subset of image files
        self.full_filenames = np.random.choice(all_files, num_samples, replace=False).tolist()

        # Load labels from a CSV file (expects columns 'id' and 'label')
        labels_file = self.data_dir / "train_labels.csv"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file '{labels_file}' not found.")
        
        labels_df = pd.read_csv(labels_file).set_index("id")
        self.labels = [labels_df.loc[img.stem].values[0] for img in self.full_filenames]

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # Retrieve the image path and corresponding label using the index
        img_path = self.full_filenames[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format
    
        if self.transform:
            image = self.transform(image)
    
        label = self.labels[idx]
        img_id = img_path.stem  # Extract the image ID (filename without extension)
        return image, label, img_id


# Define transformations for training and validation datasets
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.ToTensor()
])


# Set the path to the dataset directory
data_dir = '/kaggle/input/histopathologic-cancer-detection/'

# Initialize the CancerDataset for training data
dataset = CancerDataset(data_dir, transform=train_transforms, data_type="train")


# Create an array of indices for the entire dataset
indices = np.arange(len(dataset))

# First split: Separate 20% (temporary set) from the 80% training set
train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=2025)

# Second split: Divide the temporary set into two halves for validation and test (10% each)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=2025)

# Create subsets for training, validation, and testing using the indices
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)


# Apply appropriate transformations
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms
test_dataset.dataset.transform = val_transforms


# Print dataset sizes for verification
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# Define DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


def plot_sample_images(dataset, data_dir, data_type="train", num_images=12, title="Sample Images"):
    """
    Plots a grid of sample images from the dataset to visualize their appearance.

    Args:
        dataset (Dataset or Subset): The dataset object (train, validation, or test).
        data_dir (str or Path): Root directory containing image data and labels.
        data_type (str): Directory selection - "train", "test", or "val".
        num_images (int): Number of images to display in the grid.
        title (str): Title of the plot.
    
    Displays:
        A matplotlib figure with sample images and their corresponding labels.
    """

    # Define the grid layout (3 rows, num_images/3 columns)
    fig, axes = plt.subplots(nrows=3, ncols=num_images // 3, figsize=(15, 7))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for ax in axes.flat:
        # Select a random image index
        idx = random.randint(0, len(dataset) - 1)

        # Retrieve image and label (handling test set separately)
        if data_type == "test":
            filename, label = dataset[idx]
        else:
            img, label, filename = dataset[idx]        

        # Load and display the image
        img_path = os.path.join(data_dir, data_type, filename + '.tif')
        ax.imshow(Image.open(img_path))
        ax.set_title(f"{'Cancer' if label == 1 else 'Normal'}", fontsize=10)
        ax.axis("off")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# Plot sample images from training and validation sets
plot_sample_images(train_dataset, data_dir, data_type="train", title="Training Set Samples")
plot_sample_images(val_dataset, data_dir, data_type="train", title="Validation Set Samples")


def findConv2dOutShape(hin, win, conv, pool=2):
    """
    Computes the output shape of a convolutional layer.

    Args:
        hin (int): Input height.
        win (int): Input width.
        conv (torch.nn.Conv2d): Convolutional layer instance.
        pool (int, optional): Pooling factor (default: 2).

    Returns:
        tuple: Output height and width after convolution and pooling.
    """
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    if pool:
        hout /= pool
        wout /= pool

    return int(hout), int(wout)


# Define the Convolutional Neural Network
class Network(nn.Module):
    """
    Convolutional Neural Network for histopathological cancer detection.

    Args:
        params (dict): Dictionary containing model hyperparameters.

    Model Architecture:
        - 4 Convolutional layers with ReLU activation and MaxPooling
        - 2 Fully Connected (FC) layers with ReLU activation and Dropout
        - LogSoftmax activation in the output layer
    """

    def __init__(self, params):
        super(Network, self).__init__()

        # Extract parameters
        Cin, Hin, Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        # Convolutional Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h, w = findConv2dOutShape(Hin, Win, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv4)

        # Fully Connected Layers
        self.fc1 = nn.Linear(1024, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        """
        Forward pass of the CNN.

        Args:
            X (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Log probabilities for each class.
        """
        # Convolution + Activation + Pooling
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        # Flatten feature maps
        X = X.view(X.shape[0], -1)  

        # Fully Connected Layers
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)

        return F.log_softmax(X, dim=1)  # Output log-probabilities


# Define model hyperparameters
params_model = {
    "shape_in": (3, 46, 46),  # Input shape: (Channels, Height, Width)
    "initial_filters": 8,      # Number of filters in the first Conv layer
    "num_fc1": 100,            # Number of neurons in the first fully connected layer
    "dropout_rate": 0.25,      # Dropout rate
    "num_classes": 2           # Number of output classes (Cancer, Normal)
}


# Instantiate the CNN model
cnn_model = Network(params_model)

# Move model to the specified device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cnn_model.to(device)


def get_lr(optimizer):
    """
    Retrieve the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer used for training.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']



def loss_batch(loss_func, output, target, optimizer=None):
    """
    Computes loss and accuracy for a batch of data.

    Args:
        loss_func (torch.nn.Module): The loss function.
        output (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.
        optimizer (torch.optim.Optimizer, optional): The optimizer for updating model weights. Defaults to None.

    Returns:
        tuple: (loss value, number of correct predictions)
    """
    loss = loss_func(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    batch_correct = pred.eq(target.view_as(pred)).sum().item()

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), batch_correct



def loss_epoch(model, loss_func, dataloader, optimizer=None, device='cpu'):
    """
    Computes the average loss and accuracy for an entire epoch.

    Args:
        model: The neural network model.
        loss_func: Loss function.
        dataloader: DataLoader providing the dataset.
        optimizer: Optimizer instance for training; if None, evaluation is assumed.
        device: Device to run computation on.

    Returns:
        Tuple of (average loss, average accuracy).
    """
    model = model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = len(dataloader.dataset)

    for xb, yb, _ in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        loss_value, batch_correct = loss_batch(loss_func, outputs, yb, optimizer)
        total_loss += loss_value
        total_correct += batch_correct

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy


def train_validate(model, params, device='cpu', verbose=True):
    """
    Trains and validates the model over a specified number of epochs.

    Args:
        model (torch.nn.Module): The neural network model.
        params (dict): Dictionary of training parameters.
        device (str): Device to run computation ('cpu' or 'cuda').
        verbose (bool): Whether to print training progress.

    Returns:
        tuple: (trained model, loss history, accuracy history)
    """
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    optimizer = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        current_lr = get_lr(optimizer)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Current LR: {current_lr:.6f}")

        # Training Phase
        model.train()
        train_loss, train_accuracy = loss_epoch(model, loss_func, train_dl, optimizer, device)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_accuracy)

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = loss_epoch(model, loss_func, val_dl, None, device)
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_accuracy)

        # Save best model weights
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
            if verbose:
                print("Saved best model weights.")

        # Adjust learning rate
        lr_scheduler.step(val_loss)
        new_lr = get_lr(optimizer)
        if new_lr != current_lr:
            if verbose:
                print("LR reduced; reloading best model weights.")
            model.load_state_dict(best_model_wts)

        if verbose:
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Accuracy: {100*val_accuracy:.2f}%")
            print("-" * 30)

    # Load best model weights after training
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


# Define optimizer and learning rate scheduler
optimizer = Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)



params_train = {
    "train": train_loader,
    "val": val_loader,
    "epochs": 50,
    "optimiser": optimizer,
    "lr_change": lr_scheduler,
    "f_loss": nn.NLLLoss(reduction="sum"),
    "weight_path": "cnn_weights.pt",
}


trained_model, loss_hist, metric_hist = train_validate(cnn_model, params_train, device=device, verbose=True)


epochs_range = range(1, params_train["epochs"] + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss Convergence Plot
sns.lineplot(x=list(epochs_range), y=loss_hist["train"], ax=axes[0], label="Train Loss")
sns.lineplot(x=list(epochs_range), y=loss_hist["val"], ax=axes[0], label="Validation Loss")
axes[0].set_title("Loss Convergence History")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# Accuracy Convergence Plot
sns.lineplot(x=list(epochs_range), y=metric_hist["train"], ax=axes[1], label="Train Accuracy")
sns.lineplot(x=list(epochs_range), y=metric_hist["val"], ax=axes[1], label="Validation Accuracy")
axes[1].set_title("Accuracy Convergence History")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()


def generate_predictions(model, dataset, device, batch_size=32):
    """
    Performs inference on the dataset and returns a dictionary of predictions.

    Args:
        model (torch.nn.Module): The trained CNN model.
        dataset (torch.utils.data.Dataset): The dataset for inference.
        device (str): Device for computation ('cpu' or 'cuda').
        batch_size (int, optional): Batch size for inference. Defaults to 32.

    Returns:
        dict: A dictionary where keys are filenames and values are predicted classes.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = {}

    with torch.no_grad():
        for images, _, filenames in tqdm(dataloader, desc="Generating Predictions"):
            images = images.to(device)
            outputs = model(images)

            # Get the index of the highest probability class
            _, preds = torch.max(outputs, 1)

            # Store predictions with corresponding filenames
            for filename, pred in zip(filenames, preds.cpu().numpy()):
                predictions[filename] = pred

    return predictions


# load any model weights for the model
cnn_model.load_state_dict(torch.load('cnn_weights.pt'))


def evaluate_model(model, dataset, labels_csv, device, batch_size=32):
    """
    Evaluates the CNN model by comparing predictions with actual labels.

    Args:
        model (torch.nn.Module): The trained CNN model.
        dataset (torch.utils.data.Dataset): The dataset for inference.
        labels_csv (str): Path to CSV file containing ground truth labels. 
                          The CSV must have 'id' and 'label' columns.
        device (str): Device for inference ('cpu' or 'cuda').
        batch_size (int, optional): Batch size for inference. Defaults to 32.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    
    # Generate model predictions
    predictions = generate_predictions(model, dataset, device, batch_size=batch_size)
    
    # Load ground truth labels from CSV
    labels_df = pd.read_csv(labels_csv)
    
    # Create a dictionary mapping file names to their true labels
    labels_dict = dict(zip(labels_df['id'], labels_df['label']))
    
    y_true = []
    y_pred = [] 
    
    # Match predictions with true labels
    for filename, pred in predictions.items():
        if filename in labels_dict:
            y_true.append(labels_dict[filename])
            y_pred.append(pred)
    
    # Convert lists to NumPy arrays for metric calculations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute key evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)    
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["No Tumor", "Tumor"], yticklabels=["No Tumor", "Tumor"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Print classification report
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
    }


labels_csv = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"


metrics = evaluate_model(cnn_model, test_dataset, labels_csv, device)


with open("metrics.json", "w") as f:
    json.dump(metrics, f)


class PredictionsDataset(Dataset):
    """
    Custom dataset class for loading test images for inference.

    Args:
        data_dir (str): Path to the directory containing test images.
        transform (torchvision.transforms.Compose): Image transformations.

    Returns:
        image (Tensor): Transformed image.
        filename (str): Name of the image file.
    """
    def __init__(self, data_dir, transform):
        self.path2data = data_dir
        self.filenames = os.listdir(self.path2data)  # List all image files
        self.full_filenames = [os.path.join(self.path2data, f) for f in self.filenames]        
        self.transform = transform       

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])  # Open image with PIL
        image = self.transform(image)  # Apply transformations
        filename = self.filenames[idx]  # Get filename
        return image, _, filename  # Return image, placeholder for label and filename


data_dir = '/kaggle/input/histopathologic-cancer-detection/test'

# Since the model expects input images in tensor format, we use torchvision.transforms to convert images to tensors
data_transformer = transforms.Compose([
    transforms.ToTensor()
])

dataset_predictions = PredictionsDataset(data_dir=data_dir, transform=data_transformer)


predictions = generate_predictions(cnn_model, dataset_predictions, device)


df = pd.DataFrame(list(predictions.items()), columns=["id", "label"])
df["id"] = df["id"].apply(lambda x: x.replace(".tif", ""))
df.to_csv("prediction.csv", index=False)


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("huggingface")


from huggingface_hub import login

login(token=secret_value_0)


from huggingface_hub import HfApi

username = "maiurilorenzo"
repo_name = "histoplastic-cancer-CNN-classifier"

api = HfApi()
api.create_repo(f"{username}/{repo_name}", exist_ok=True)  # Create repo if it doesn't exist

# Upload the model file
api.upload_file(
    path_or_fileobj="cnn_weights.pt",
    path_in_repo="cnn_weights.pt",
    repo_id=f"{username}/{repo_name}"
)

api.upload_file(
    path_or_fileobj="metrics.json",
    path_in_repo="metrics.json",
    repo_id=f"{username}/{repo_name}"
)


from huggingface_hub import hf_hub_download
import tensorflow as tf
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Load model
repo_id = f"{username}/{repo_name}"

model_path = hf_hub_download(repo_id=repo_id, filename="cnn_weights.pt")

# Neural Network Predefined Parameters
params_model={
        "shape_in": (3,46,46), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2}

# Create instantiation of Network class
cnn_model = Network(params_model)

cnn_model.load_state_dict(torch.load('cnn_weights.pt'))

predictions = generate_predictions(model, dataset_predictions, device)


data_dir = '/kaggle/input/histopathologic-cancer-detection'


plot_sample_images(
    [(filename.replace(".tif", ""), label) for filename, label in predictions.items()], 
    data_dir, 
    data_type="test", 
    title="Test Set Samples"
)

