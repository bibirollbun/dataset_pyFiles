import os
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
import torchvision
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import pandas as pd

plt.style.use("ggplot")


try:
    import torchinfo
except:
    print("Installing `torchinfo`...")
    !pip install torchinfo
    import torchinfo

from torchinfo import summary


device = "cuda" if torch.cuda.is_available() else "cpu"
# we might like some indication of what's going on
#print(f"We'll be using {device} for this finetuning")   


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


os.getcwd()


# We're dealing with a zipped dataset on kaggle. 
# Otherwise we might've loaded it from huggingface using Datasets directly

import zipfile

zip_file_path = "/kaggle/input/oxford-102-flower-pytorch/flower_data.zip"

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall("kaggle/input/oxford-102-flower-pytorch/")
    print("Zip file extraction complete!")


data_path = "kaggle/input/oxford-102-flower-pytorch/flower_data"

def validate_images(directory):
    for dirname, _, filenames in os.walk(directory):
        if dirname != data_path:
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except (UnidentifiedImageError, IOError):
                    print(f"Invalid image found: {file_path}")

validate_images(data_path)


train_data_path = os.path.join(data_path, "train")
val_data_path = os.path.join(data_path, "valid")
test_data_path = os.path.join(data_path, "test")


import json

cat_file_path = os.path.join(data_path, "cat_to_name.json")

with open(cat_file_path, "r") as json_file:
    cat_names_dict = json.load(json_file)

cat_names_dict.keys(), len(cat_names_dict)


# some more output to inspect if we want
# print(f"Our categories are {cat_names_dict}")


cat_names_list = []

for i in range(1, len(cat_names_dict) + 1):
    cat_names_list.append(cat_names_dict[str(i)])

print(cat_names_list)


# Defining a function to quickly check the amount of images in each category directory
def check_cat_dirs_lengths(directory):
    for category_dir in os.listdir(directory):
        category_path = os.path.join(directory, category_dir)

        if os.path.isdir(category_path):
            img_count = len(os.listdir(category_path))
            print(f"Category {category_dir}: {img_count} images.")


check_cat_dirs_lengths(train_data_path)


from torchvision.datasets import ImageFolder

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)), # as I want to use pretrained ResNet50 model, the expected input size is 224x224
    transforms.TrivialAugmentWide(num_magnitude_bins=6),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CustomTestDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        Parameters:
        ----------
        root_dir: str
                Directory with all the test images.
        transform: callable, optional
                Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, filename) 
                            for filename in os.listdir(root_dir)
                            if filename.endswith((".jpg"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert("RGB") 

        if self.transform:
            image = self.transform(image)

        return image


# Creating datasets
train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
val_dataset = ImageFolder(root=val_data_path, transform=val_test_transform)
test_dataset = CustomTestDataset(root_dir=test_data_path, transform=val_test_transform)

# Creating dataloaders
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() if device == "cpu" else torch.cuda.device_count()

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

#train_dataloader, val_dataloader, test_dataloader


len(train_dataloader), len(val_dataloader), len(test_dataloader)


def display_images_from_dataloader(dataloader, cat_names_list, num_images=8):
    """
    Display a grid of images from a DataLoader with category names as titles.
    
    Args:
        dataloader (DataLoader): PyTorch DataLoader containing the images.
        cat_names_list (list): List containing the category names (indexed from 1).
        num_images (int): Number of images to display (default is 4).
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    images, labels = next(iter(dataloader))
        
    num_images = min(num_images, images.size(0))
    
    plt.figure(figsize=(4 * num_images, 10))
    
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        
        img = images[i] * std.view(3, 1, 1) + mean.view(3, 1, 1)
        img = img.numpy().transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
        
        label_idx = labels[i].item()  
        category_name = cat_names_list[label_idx]
        
        plt.imshow(img)
        plt.title(category_name)  
        plt.axis('off')  
        plt.grid(False)
        plt.tight_layout()

    plt.show()


# let's see some of the images our model would be trained on
torch.manual_seed(42)
display_images_from_dataloader(train_dataloader, cat_names_list)


from torch.optim import Adam
from torch.nn import CrossEntropyLoss


# okay, let's instantiate a pretrained resnet
model = resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

num_classes = len(cat_names_list)

model.fc = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(model.fc.in_features, num_classes)
)


# Displaying model's summary
# We don't have to, but it's there for completeness' sake
summary(model, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20, row_settings=["var_names"])


# Let's define an accuracy function first:
def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
  """
  Computes the accuracy of multi-class predictions compared to true labels.

  Parameters:
  ----------
  y_pred : torch.Tensor
      A tensor containing the predicted class labels as integers in the range [1, n_classes] for each sample.
      The shape should be (batch_size,).
      Apply `torch.argmax(y_pred, dim=1)` to get the predicted class indices.
  y_true : torch.Tensor
      A tensor containing the true class labels (ground truth) as integers in the range [1, n_classes].
      The shape should be (batch_size,).
  """
  if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
    raise TypeError("Both y_pred and y_true must be torch.Tensor.")

  if y_pred.size(0) != y_true.size(0):
    raise ValueError("y_pred and y_true must have the same number of elements.")

  correct = torch.eq(y_pred, y_true).sum().item()
  acc = (correct / len(y_true)) * 100

  return acc


criterion = CrossEntropyLoss()
optimiser = Adam(model.fc.parameters())


# We could've defined and used a LoRA layer for our resnet, but it's already tiny by modern standards, so let's not

from typing import Callable

def train_step(model: nn.Module,
               train_dataloader: DataLoader,
               criterion: nn.Module,
               acc_fn: Callable[[torch.Tensor, torch.Tensor], float],
               optimiser: torch.optim.Optimizer,
               device: str = device):
  train_loss, train_acc = 0, 0

  model.train()

  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)

    y_pred_logits = model(X)
    y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

    loss = criterion(y_pred_logits, y)
    train_loss += loss.item()

    acc = acc_fn(y_preds, y)
    train_acc += acc

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

  train_loss = train_loss / len(train_dataloader)
  train_acc = train_acc / len(train_dataloader)

  return train_loss, train_acc


def val_step(model: nn.Module,
              val_dataloader: DataLoader,
              criterion: nn.Module,
              acc_fn: Callable[[torch.Tensor, torch.Tensor], float],
              device: str = device):
  val_loss, val_acc = 0, 0

  model.eval()

  with torch.inference_mode():
    for batch, (X, y) in enumerate(val_dataloader):
      X, y = X.to(device), y.to(device)

      y_pred_logits = model(X)
      y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

      loss = criterion(y_pred_logits, y)
      val_loss += loss.item()

      acc = acc_fn(y_preds, y)
      val_acc += acc

    val_loss = val_loss / len(val_dataloader)
    val_acc = val_acc / len(val_dataloader)

  return val_loss, val_acc


from tqdm.auto import tqdm

# Defining a model training function
def train_model(model: nn.Module,
               train_dataloader: DataLoader,
               val_dataloader: DataLoader,
               criterion: nn.Module,
               acc_fn: Callable[[torch.Tensor, torch.Tensor], float],
               optimiser: torch.optim.Optimizer,
               num_epochs=3):
    results = {"train_loss": [],
              "train_acc": [],
              "val_loss": [],
              "val_acc": []}

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, criterion, acc_fn, optimiser)
        val_loss, val_acc = val_step(model, val_dataloader, criterion, acc_fn)
        
        print(f"Epoch {epoch + 1}: | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
    
    return results


torch.manual_seed(42)
torch.cuda.manual_seed(42)

from timeit import default_timer as timer
start_time = timer()


# let's try 10 epochs first, and if it's not enough (by the metrics defined later) we can increase that number
# okay, the train accuracy was around 83%, which is good but not great. 
# let's use 30 epochs
model_results = train_model(model, train_dataloader, val_dataloader, criterion, accuracy_fn, optimiser, num_epochs=30)

end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")


from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]]):
  """
  Plots training curves of a results dictionary.
  """
  # Get the loss values of the results dictionary (training and test)
  loss = results["train_loss"]
  val_loss = results["val_loss"]

  # Get the accuray values of the results dictionary (training and test)
  acc = results["train_acc"]
  val_acc = results["val_acc"]

  # Figure out how many epochs there were
  epochs = range(len(results["train_loss"]))

  # Setup a plot
  plt.figure(figsize=(15, 7))

  # Plot the loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  # Plot the accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, acc, label="train_accuracy")
  plt.plot(epochs, val_acc, label="val_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()


plot_loss_curves(model_results)


def visualize_predictions(model, val_dataloader, device, cat_names_list, n_images=8):
    """
    Visualizes predictions for a batch of images from the validation set.

    Parameters:
    ----------
    model : torch.nn.Module
        The PyTorch model to make predictions.
    val_dataloader : torch.utils.data.DataLoader
        The dataloader for the validation dataset.
    device : torch.device
        The device to run the model on (CPU or GPU).
    cat_names_list : list
        List of class names corresponding to class indices.
    n_images : int, optional
        Number of images to visualize (default is 8).
    """
    X_batch, y_batch = next(iter(val_dataloader))

    model.eval()
    
    with torch.inference_mode():
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred_logits = model(X_batch)
        y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

    X_batch, y_batch, y_pred_logits = X_batch.cpu(), y_batch.cpu(), y_pred_logits.cpu()
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    X_batch = X_batch * std.view(3, 1, 1) + mean.view(3, 1, 1)

    plt.figure(figsize=(16, 8))

    for i in range(n_images):
        plt.subplot(2, 4, i + 1)

        # Convert the image to (H, W, C) for visualization
        image = X_batch[i].permute(1, 2, 0)
        plt.imshow(image)

        prob = torch.max(torch.softmax(y_pred_logits[i].unsqueeze(dim=0), dim=1))
        class_label = cat_names_list[y_preds[i]]
        is_correct = y_preds[i] == y_batch[i]

        plt.title(
            f"{class_label} (p={prob:.2f})",
            color="green" if is_correct else "red"
        )
        plt.axis('off')

    plt.tight_layout()
    plt.show()


visualize_predictions(model, val_dataloader, device, cat_names_list)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model, dataloader, device, class_names):
    """
    Visualizes the confusion matrix for a PyTorch model on a dataset.

    Parameters:
    ----------
    model : torch.nn.Module
        The PyTorch model for inference.
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset (validation or test set).
    device : torch.device
        The device to run the model on (CPU or GPU).
    class_names : list
        List of category names corresponding to class indices.
    """
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred_logits = model(X_batch)
            y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
            
            all_preds.append(y_preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(30, 30))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='d')

    ax.set_title("Confusion Matrix", fontsize=20)
    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("True Label", fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    plt.grid(False)
    plt.show()


plot_confusion_matrix(model, val_dataloader, device, cat_names_list)


example = pd.read_csv("/kaggle/input/oxford-102-flower-pytorch/sample_submission.csv")
example.head()


model.eval()

results = []

with torch.inference_mode():
    for file_name in os.listdir(test_data_path):
        file_path = os.path.join(test_data_path, file_name)
        
        image = Image.open(file_path).convert("RGB")
        input_tensor = val_test_transform(image).unsqueeze(0).to(device) 
        
        y_pred_logits = model(input_tensor)
        y_pred_prob = torch.softmax(y_pred_logits, dim=1)
        y_pred_class = torch.argmax(y_pred_prob, dim=1).item()
        
        results.append({
            "file_name": file_name,
            "id": y_pred_class
        })

res_df = pd.DataFrame(results)
res_df.head()


res_df.to_csv('submission.csv', index=False)


!mkdir models
!mkdir models/pytorch


torch.save(model.state_dict(), 'models/pytorch/resnet50_flowers_classifier_weights.pth')




