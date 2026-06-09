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


!pip install -U imbalanced-learn scikit-learn -q


street_view_getting_started_with_julia_path = '/kaggle/input/street-view-getting-started-with-julia'


file_path = os.path.join(street_view_getting_started_with_julia_path, 'resizeData.py')

try:
    with open(file_path, 'r') as f:
        python_code = f.read()
        print(python_code)
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


import os
import zipfile

file_list = os.listdir(street_view_getting_started_with_julia_path)

for file_name in file_list:
    file_path = os.path.join(street_view_getting_started_with_julia_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith('.zip'):
        print(f"--- Contents of {file_name} ---")
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                print(zip_ref.namelist())
        except Exception as e:
            print(f"Could not read zip file {file_name}: {e}")
        print("-" * (len(f"--- Contents of {file_name} ---")))


sample_sub = pd.read_csv(f'{street_view_getting_started_with_julia_path}/sampleSubmission.csv')
print(sample_sub.shape)
sample_sub.head()


# Get the .zip files

import os

zip_files = ['trainResized.zip', 'testResized.zip', 'test.zip', 'train.zip']
zip_file_paths = [os.path.join(street_view_getting_started_with_julia_path, f) for f in zip_files]

print("Found zip files:")
for zip_path in zip_file_paths:
    print(zip_path)


# Extract the .zip files in new directory

import zipfile

main_extracted_path = os.path.join('/kaggle/working/', 'extracted_data')
os.makedirs(main_extracted_path, exist_ok=True)

for zip_path in zip_file_paths:
    zip_file_name = os.path.basename(zip_path).replace('.zip', '')
    # extracted_folder_path = os.path.join(main_extracted_path, zip_file_name)
    # os.makedirs(extracted_folder_path, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(main_extracted_path)
        # print(f"Extracted {os.path.basename(zip_path)} to {main_extracted_path}")
        # print(f"Contents of {zip_file_name} folder:")
        # for root, _, filenames in os.walk(main_extracted_path):
        #     for filename in filenames:
                # print(os.path.join(root, filename))
        # print("-" * 90) # Separator
    except Exception as e:
        print(f"Could not extract {os.path.basename(zip_path)}: {e}")


# Read the training labels
all_labels = pd.read_csv(f'{street_view_getting_started_with_julia_path}/trainLabels.csv')
print(all_labels.shape)
all_labels.head()


num_classes = len(all_labels['Class'].unique())
print(num_classes)

all_labels['Class'].unique()


all_labels


import matplotlib.pyplot as plt
import seaborn as sns

class_distribution = all_labels['Class'].value_counts()

plt.figure(figsize=(18, 8))

sns.barplot(x=class_distribution.index, y=class_distribution.values, palette='viridis')

plt.title('Image Count Per Class', fontsize=16)
plt.xlabel('Class Label', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()

plt.show()


folders = ['test','train']

folder_paths = [
    f'{main_extracted_path}/{folder}'
    for folder in folders
    ]
folder_paths


# Let's load our images and plot them

from PIL import Image
import glob

all_images_paths_by_folder = {}

for folder, folder_path in zip(folders, folder_paths):
    all_images_paths_by_folder[folder] = []

    image_paths_in_folder = glob.glob(os.path.join(folder_path, '**', '*.Bmp'), recursive=True)

    all_images_paths_by_folder[folder] = image_paths_in_folder

    print(f"Found {len(image_paths_in_folder)} images in {folder} folder.")


train_images_paths = all_images_paths_by_folder['train']
test_images_paths = all_images_paths_by_folder['test']
len(train_images_paths), len(test_images_paths)


train_images_paths[0]


train_label_ids = [
    int(os.path.splitext(os.path.basename(path))[0])
    for path in train_images_paths
]

train_labels=[
    all_labels[all_labels['ID'] == label_idx]['Class'].values[0]
    for label_idx in train_label_ids
]
train_labels


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
train_labels_int = encoder.fit_transform(train_labels)
train_labels_int


from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_images_paths, train_labels_int, test_size=0.2, random_state=42, stratify=train_labels_int
)


import torchvision.transforms as transforms

TARGET_SIZE = (32 , 32)

train_base_transforms = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.Grayscale(),
    transforms.ToTensor()
])


test_transforms = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),

    transforms.Grayscale(),

    transforms.ToTensor()
])


from torch.utils.data import Dataset, DataLoader, TensorDataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        if self.labels is None:
          return image, image_path

        label = self.labels[idx]
        return image, label



train_dataset = ImageDataset(train_paths, train_labels, transform=train_base_transforms)
val_dataset = ImageDataset(val_paths, val_labels, transform=test_transforms)
test_dataset = ImageDataset(test_images_paths, transform=test_transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)


images, labels = next(iter(train_loader))

print(f"\n--- Verifying one batch ---")
print(f"Batch of images shape: {images.shape}")
print(f"Batch of labels shape: {labels.shape}")


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 1, figsize=(15, 5))


axes.imshow(images[31][0], cmap='gray', vmin=0, vmax=1)


encoder.inverse_transform(labels)[31]


X_train_original_np, y_train_original_np = images.numpy(), labels.numpy()
X_train_original_np.shape, y_train_original_np.shape


val_images, val_labels = next(iter(val_loader))
X_val_original_np, y_val_original_np = val_images.numpy(), val_labels.numpy()
X_val_original_np.shape, y_val_original_np.shape


test_images, test_paths = next(iter(test_loader))
X_test_original_np =  test_images.numpy()
X_test_original_np.shape


test_ids = [
    int(os.path.splitext(os.path.basename(path))[0])
    for path in test_paths
]
test_ids


df = pd.DataFrame({'label': y_train_original_np})
label_counts = df['label'].value_counts()
target_count = label_counts.max()
target_count


from imblearn.over_sampling import SMOTE

X_flattened = X_train_original_np.reshape(X_train_original_np.shape[0], -1)

smote = SMOTE(sampling_strategy='auto', random_state=42)

X_resampled, y_train = smote.fit_resample(X_flattened, y_train_original_np)

X_train = X_resampled.reshape(X_resampled.shape[0], 1, 32, 32)
X_train


X_train.shape, y_train.shape


X_train_numpy_flat = X_train.reshape(X_train.shape[0], -1)
X_train_numpy_flat.shape


X_val_numpy_flat = X_val_original_np.reshape(X_val_original_np.shape[0], -1)
X_val_numpy_flat.shape


X_test_numpy_flat = X_test_original_np.reshape(X_test_original_np.shape[0], -1)
X_test_numpy_flat.shape


from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train_numpy_flat)
X_train_reduced.shape


X_val_reduced = pca.transform(X_val_numpy_flat)
X_val_reduced.shape


X_test_reduced = pca.transform(X_test_numpy_flat)
X_test_reduced.shape


import torch
# Let's prepare our dataset for training

X_train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_reduced, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_original_np, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32)


BATCH_SIZE = 64

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ANN_Arch(nn.Module):

  def __init__(self, input_size, num_classes=num_classes):
    super(ANN_Arch, self).__init__()

    self.flatten = nn.Flatten()

    self.fc1 = nn.Linear(input_size, 2048)
    self.bn1 = nn.BatchNorm1d(2048)

    self.fc2 = nn.Linear(2048, 1024)
    self.bn2 = nn.BatchNorm1d(1024)

    self.fc3 = nn.Linear(1024, 512)
    self.bn3 = nn.BatchNorm1d(512)

    self.fc4 = nn.Linear(512, 256)
    self.bn4 = nn.BatchNorm1d(256)

    self.fc5 = nn.Linear(256, 128)
    self.bn5 = nn.BatchNorm1d(128)

    self.output_fc = nn.Linear(128, num_classes)

    self.dropout = nn.Dropout(0.5)

  def forward(self, X):

    # Flatten the input image
    X = self.flatten(X)

    # Block 1
    X = self.fc1(X)
    X = self.bn1(X)
    X = F.relu(X)
    X = self.dropout(X)

    # Block 2
    X = self.fc2(X)
    X = self.bn2(X)
    X = F.relu(X)
    X = self.dropout(X)

    # Block 3
    X = self.fc3(X)
    X = self.bn3(X)
    X = F.relu(X)
    X = self.dropout(X)

    # Block 4
    X = self.fc4(X)
    X = self.bn4(X)
    X = F.relu(X)
    X = self.dropout(X)

    # Block 5
    X = self.fc5(X)
    X = self.bn5(X)
    X = F.relu(X)
    X = self.dropout(X)

    # Output Layer
    X = self.output_fc(X)

    return X


from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ANN_Arch(
    input_size = X_train_tensor.shape[1]
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


import torch

def model_training(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs=10,
    device='cpu'
    ):

  train_losses, val_losses = [], []
  train_accuracies, val_accuracies = [], []

  model.to(device)

  for epoch in range(1, num_epochs + 1):

      model.train()
      running_train_loss = 0.0
      train_correct = 0

      for images, labels in train_loader:

          images = images.to(device)
          labels = labels.to(device)

          outputs = model(images)
          loss = criterion(outputs, labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          running_train_loss += loss.item() * images.size(0)

          _, predicted = torch.max(outputs.data, 1)
          train_correct += (predicted == labels).sum().item()

      epoch_train_loss = running_train_loss / len(train_loader.dataset)
      train_losses.append(epoch_train_loss)

      epoch_train_acc = 100 * train_correct / len(train_loader.dataset)
      train_accuracies.append(epoch_train_acc)

      model.eval()
      running_val_loss = 0.0
      val_correct = 0

      with torch.no_grad():
          for images, labels in val_loader:
              images = images.to(device)
              labels = labels.to(device)

              outputs = model(images)
              loss = criterion(outputs, labels)
              running_val_loss += loss.item() * images.size(0)

              _, predicted = torch.max(outputs.data, 1)
              val_correct += (predicted == labels).sum().item()

      epoch_val_loss = running_val_loss / len(val_loader.dataset)
      val_losses.append(epoch_val_loss)

      epoch_val_acc = 100 * val_correct / len(val_loader.dataset)
      val_accuracies.append(epoch_val_acc)

      print(f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

  return model, train_losses, val_losses, train_accuracies, val_accuracies


trained_model, train_losses, val_losses, train_accuracies, val_accuracies = model_training(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs=50,
    device=device
)


# Plotting the losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting the accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()


import torch
import pandas as pd

def get_predictions(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for (images,) in data_loader:
            images = images.to(device)

            outputs = model(images)

            _, predicted_classes = torch.max(outputs.data, 1)

            predictions.extend(predicted_classes.cpu().numpy())

    return predictions


predictions = get_predictions(
    model=trained_model,
    data_loader=test_loader,
    device=device
    )

predictions


prediction_labels = [
    str(encoder.inverse_transform([prediction])[0])
    for prediction in predictions
]
prediction_labels



#Prepare the submission
submission_df = pd.DataFrame({'ID': test_ids, 'Class': prediction_labels})
submission_df.head()


submission_df.to_csv('submission_go.csv', index=False)

