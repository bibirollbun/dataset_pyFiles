# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torch
import csv

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision.transforms import v2



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


emotion_map = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, training=True):
      self.data_path = data_path
      self.transform = transform
      self.training = training

      try:
        print(f"Loading dataset from {data_path} into memory...")
        self.data = pd.read_csv(data_path)
        print("Dataset loaded!")

      except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
      except Exception as e:
          print(f"An error occurred: {e}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        pixels = np.fromstring(row[' pixels'], sep=' ', dtype=np.uint8)  # uint8 expceted for images
        image = Image.fromarray(pixels.reshape(48,48))

        # Get label
        if self.training:
          label = int(row['emotion'])
        else:
          label = -1

        if self.transform:
            image = self.transform(image)

        return image, label




file_path = '/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv'


dataset = CustomDataset(file_path)


def vis_dataset(dataset):
  # Create the 2x2 subplot layout
  fig, axs = plt.subplots(2, 3, figsize=(6,6))

  # Flatten the axes array for easier iteration
  axs = axs.flatten()

  sample_ints = [random.randint(0, len(dataset)) for _ in range(6)]

  # Plot random image
  for ax, i in zip(axs, sample_ints):
      img, label = dataset[i]
      # Use 'gray' colormap for grayscale images
      ax.imshow(img, cmap='gray')
      ax.set_title(emotion_map[label])
      # Turn off axis ticks and labels for cleaner image visualization
      ax.axis('off')

  # Adjust spacing
  plt.tight_layout(rect=[0, 0, 1, 1])


vis_dataset(dataset)


counts = dataset.data['emotion'].value_counts().sort_index()
counts.index = counts.index.map(lambda x: emotion_map[x])
counts.plot(kind='bar')


counts_percent = dataset.data['emotion'].value_counts().sort_index().apply(lambda x : x/len(dataset))*100
counts_percent.index = counts_percent.index.map(lambda x: emotion_map[x])
counts_percent


# class weights used in loss
weights = (dataset.data['emotion'].value_counts().sort_index().apply(lambda x : 1/(x/len(dataset)))).to_list()
weights


tmp_transform = v2.Compose([
    v2.ToImage(), # converts to CHW image
    v2.ToDtype(torch.float32, scale=True),
])

tmp_dataset = CustomDataset(file_path, transform=tmp_transform)
loader = DataLoader(tmp_dataset, batch_size=256)


mean = 0.0
sq_mean = 0.0
total_pixels = 0

for img_batch, _ in loader:
  b,c,h,w = img_batch.shape
  total_pixels += b*c*h*w

  mean += img_batch.sum()  # sum of all pixels
  sq_mean += (img_batch**2).sum()  # sum of all squared pixels

mean /= total_pixels
sq_mean /= total_pixels

std = torch.sqrt(sq_mean - mean**2)



mean, std = mean.item(), std.item()
mean,std


dataset_length = len(dataset)
train_size = int(0.70 * dataset_length)
val_size = dataset_length - train_size

g = torch.Generator().manual_seed(42)

train_raw, val_raw = random_split(dataset, [train_size, val_size], generator=g)


# Wrapper class for transforms
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# Training set transformation pipeline
train_transform = v2.Compose([
    v2.RandomHorizontalFlip(0.25),
    v2.RandomRotation(10),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[mean], std=[std]),
])

# Validation set transformation pipeline
val_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[mean], std=[std]),
])


batch_size = 100

train_dataset = TransformSubset(train_raw, train_transform)
val_dataset = TransformSubset(val_raw, val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=g,  num_workers=2, pin_memory=True)


class ConvBlock(nn.Module):

  def __init__(self, n_channels, n_filters, kernel_size=3, padding=1, max_pool_kernel_size=2):
    super().__init__()
    self.conv_block = nn.Sequential(
            nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(max_pool_kernel_size)
    )


  def forward(self, x):
      return self.conv_block(x)


class LinearBlock(nn.Module):

  def __init__(self, input_size, output_size, dropout=0.2):
      super().__init__()

      self.linear_block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )


  def forward(self, x):
      return self.linear_block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # 48x48 → 24x24 → 12x12 → 6x6
        self.conv_block0 = ConvBlock(1, 32)
        self.conv_block1 = ConvBlock(32, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)

        self.flatten = nn.Flatten()

        self.fc_block0 = LinearBlock(256*3*3, 512, dropout=0.40)
        self.fc_block1 = LinearBlock(512, 256, dropout=0.20)
        self.fc_block2 = LinearBlock(256, 128, dropout=0.10)
        self.fc_block3 = LinearBlock(128, 64, dropout=0.5)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_block0(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc_block0(x)
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)
        return self.out(x)


# Init
model = SimpleCNN(num_classes=7).to(device)


# define loss function
loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,        # 30% warm-up
    anneal_strategy="cos",
    div_factor=25,        # initial lr = max_lr / 25 ≈ 1.2e-4
    final_div_factor=1e4  # final lr ≈ 3e-7
)



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def update(self, val_loss):
        # check if improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True


def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device, scheduler, checkpoint=None, save_path='best_model.pth'):

    if checkpoint:
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint['val_loss']
    else:
        start_epoch = 0
        best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    val_accuracies = []
    batch_val_losses = []
    predicted_pts = []

    early_stopping = EarlyStopping(patience=7, min_delta=0.001)

    print("--- Training Started ---")

    # Loop over the specified number of epochs
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        predicted_pt = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                val_loss = loss_function(outputs, labels)
                running_val_loss += val_loss.item() * images.size(0)

                batch_val_losses.append(val_loss.item())

                pts, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                probs = torch.softmax(outputs, dim=1)
                pts, predicted = torch.max(probs, 1)
                predicted_pt += pts.sum()


        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        predicted_pts.append(predicted_pt / len(val_loader.dataset))

        epoch_accuracy = 100.0 * correct / total
        val_accuracies.append(epoch_accuracy)

        early_stopping.update(epoch_val_loss)

        print(f"Epoch [{epoch-start_epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")

        if early_stopping.stop:
            print("Validation loss did not improve. Stopping training.")
            break

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "mean_std" : (mean, std)
            }, save_path)
            print(f"Saved best model at epoch {epoch}")

    print("--- Finished Training ---")

    metrics = [train_losses, val_losses, val_accuracies, batch_val_losses, predicted_pts]

    return model, metrics


checkpoint_path = "best_model.pth"

num_epochs = 100

# --- Load checkpoint if exists ---
import os
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    
    print(f"Resuming training from epoch {start_epoch}")
else:
    checkpoint=None
    print("No checkpoint found. Starting fresh.")
# ---------------------------------

# Start the training process by calling the training loop function
trained_proto_model, training_metrics_proto = training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    scheduler=scheduler,
    checkpoint=checkpoint,
    save_path=checkpoint_path
)


train_loss = training_metrics_proto[0]
val_loss = training_metrics_proto[1]
val_accuracy = training_metrics_proto[2]


plt.plot(train_loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.legend()
plt.show()


plt.plot(val_accuracy, label='validation accuracy')
plt.legend()
plt.show()


model.eval()
incorrect = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)                     # [B, num_classes]
        preds = outputs.argmax(dim=1)               # [B]

        incorrect_indices = preds != labels         # boolean mask

        wrong_samples = images[incorrect_indices]  
        wrong_labels = labels[incorrect_indices]
        wrong_preds = preds[incorrect_indices]

        incorrect.append((wrong_samples, wrong_labels, wrong_preds))

        break


incorrect_samples = incorrect[0][0]
incorrect_samples.shape


correct_labels = incorrect[0][1]
wrong_preds = incorrect[0][2]
wrong_preds.cpu().numpy()


fig, axs = plt.subplots(5, 6)
axs = axs.flatten()

for i in range(30):
  axs[i].imshow(incorrect_samples[i].cpu().reshape(48, 48))
  axs[i].axis('off')
  axs[i].set_title(f'{emotion_map[wrong_preds.cpu().numpy()[i]]}, {emotion_map[correct_labels.cpu().numpy()[i]]}', fontsize=9)

plt.tight_layout()
plt.show()


def confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


import torch.functional as F
from collections import Counter

model.eval()
counter = Counter()

all_preds = []
all_labels = []

with torch.no_grad():

  for images, labels in val_loader:

      images, labels = images.to(device), labels.to(device)
      logits = model(images)
      preds = torch.argmax(logits, dim=1)
      correct_labels = labels[labels==preds]

      counter.update(label.item() for label in correct_labels)

      all_preds.append(preds.cpu())
      all_labels.append(labels.cpu())

print(counter)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Concatenate batches
y_true = torch.cat(all_labels).numpy()
y_pred = torch.cat(all_preds).numpy()

labels = np.unique(y_pred)

cm = confusion_matrix(y_true, y_pred, labels=labels)

display_labels = [emotion_map[int(x)] for x in labels]

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(cmap="Blues", values_format="d")
plt.show()




