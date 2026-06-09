# Loading and Inspecting the Dataset

import os

# Path to the dataset folder
dataset_path = "/kaggle/input/isic-2024-challenge/"

# List all files and directories in the dataset folder
dataset_files = os.listdir(dataset_path)

# Print the structure to understand how files are organized
print("Dataset Structure:")
for file in dataset_files:
    print(file)


# Data Manipulation Modules

import pandas as pd
import numpy as np
import random
from scipy.stats import pointbiserialr
from sklearn.preprocessing import OrdinalEncoder
import h5py
import os

# Data Viz

import seaborn as sns
import matplotlib.pyplot as plt

# Image Processing

import cv2
from PIL import Image
from io import BytesIO
import os

# Deep Learning Modules

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# Loading Metadata


df = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv")


# Grouping and Aggregating Data
sex_target_count = df.groupby(["sex", "target"]).size()
sex_target_count = pd.DataFrame(sex_target_count).reset_index()
sex_target_count.columns = ["Sex", "Status", "Count"]



anatom = pd.DataFrame(df.groupby(["anatom_site_general", "sex", "target"]).size()).reset_index()
anatom.columns = ["Anatom", "Sex", "Target", "Count"]




# Handling Missing Values
numeric_columns = df.select_dtypes(include=['number'])


numeric_columns = numeric_columns.drop("mel_thick_mm", axis = 1)


def set_seed(seed=1234):
    '''Sets the seed of the entire notebook for reproducibility.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed for reproducibility across runs
set_seed()

# Initialize an ordinal encoder
encoder = OrdinalEncoder()

# Select numeric columns and include 'isic_id' for identification
numeric_columns = df.select_dtypes(include=['number'])
numeric_columns["isic_id"] = df["isic_id"]

# Drop columns 'mel_thick_mm' and 'tbp_lv_dnn_lesion_confidence', drop rows with missing values, and reset index
numeric_columns = numeric_columns.drop(["mel_thick_mm", "tbp_lv_dnn_lesion_confidence"], axis=1).dropna().reset_index(drop=True)

# Select object (categorical) columns
object_columns = df.select_dtypes(include=['object'])

# Merge numeric and selected object columns on 'isic_id'
dataframe = numeric_columns.merge(object_columns[["isic_id", "tbp_lv_location", "tbp_lv_location_simple"]], how="inner", on="isic_id")

# Encode categorical columns 'tbp_lv_location' and 'tbp_lv_location_simple' using OrdinalEncoder
dataframe[["tbp_lv_location", "tbp_lv_location_simple"]] = encoder.fit_transform(dataframe[["tbp_lv_location", "tbp_lv_location_simple"]])

# Open HDF5 file for reading
fp_hdf = h5py.File("/kaggle/input/isic-2024-challenge/train-image.hdf5", mode="r")

# Create list of paths to image files based on 'isic_id'
path_list = [f"/kaggle/input/isic-2024-challenge/train-image/image/{idx}.jpg" for idx in dataframe["isic_id"]]

# Drop 'isic_id' from dataframe
dataframe.drop("isic_id", axis=1, inplace=True)


def apply_transformations(image_paths):
    selected_images = np.random.choice(image_paths, 3, replace=False)

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(16, 20))
    methods = ["Without Gaussian Blur", "With Gaussian Blur", "Hue, Saturation, Brightness", "LUV Color Space", "Greyscale + Gaussian Blur"]
    
    for index, method in enumerate(methods):
        for i, path in enumerate(selected_images):
            image = cv2.imread(path)

            if method == "Without Gaussian Blur":
                transformed_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                transformed_image = cv2.resize(transformed_image, (200, 200))

            elif method == "With Gaussian Blur":
                transformed_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                transformed_image = cv2.resize(transformed_image, (200, 200))
                transformed_image = cv2.addWeighted(transformed_image, 4, cv2.GaussianBlur(transformed_image, (0, 0), 256/10), -4, 128)

            elif method == "Hue, Saturation, Brightness":
                transformed_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                transformed_image = cv2.resize(transformed_image, (200, 200))

            elif method == "LUV Color Space":
                transformed_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                transformed_image = cv2.resize(transformed_image, (200, 200))

            elif method == "Greyscale + Gaussian Blur":
                transformed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                transformed_image = cv2.resize(transformed_image, (200, 200))
                transformed_image = cv2.GaussianBlur(transformed_image, (5, 5), 0)

            axes[index, i].imshow(transformed_image, cmap=plt.cm.bone if len(transformed_image.shape) == 2 else None)
            axes[index, i].axis('off')
            axes[index, i].set_title(method, fontsize=10)

    plt.tight_layout()
    plt.show()

apply_transformations(path_list)



import matplotlib.image as mpimg

image_hair = np.array(path_list)[[1, 2, 8, 11, 13, 17]]

def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))

    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)

    return final_image




# Definir la función para calcular pauc_80
def pauc_80(y_true, y_scores, min_tpr=0.8):
    sorted_indices = torch.argsort(y_scores, descending=True)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    pos_count = torch.sum(y_true_sorted == 1).item()
    neg_count = torch.sum(y_true_sorted == 0).item()
    total_pos = pos_count
    
    tp = 0
    fp = 0
    tpr = 0.0
    fpr = 0.0
    pauc = 0.0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
            tpr = tp / total_pos
        else:
            fp += 1
            fpr = fp / neg_count
            
            if tpr >= min_tpr:
                pauc += (tpr - min_tpr) * fpr
                min_tpr = tpr
                if tpr == 1.0:
                    break
    return pauc


def set_seed(seed=1234):
    '''Sets the seed of the entire notebook for reproducibility.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed for reproducibility across runs
set_seed()

# Initialize an ordinal encoder
encoder = OrdinalEncoder()

# Select numeric columns and include 'isic_id' for identification
numeric_columns = df.select_dtypes(include=['number'])
numeric_columns["isic_id"] = df["isic_id"]

# Drop columns 'mel_thick_mm' and 'tbp_lv_dnn_lesion_confidence', drop rows with missing values, and reset index
numeric_columns = numeric_columns.drop(["mel_thick_mm", "tbp_lv_dnn_lesion_confidence"], axis=1).dropna().reset_index(drop=True)

# Select object (categorical) columns
object_columns = df.select_dtypes(include=['object'])

# Merge numeric and selected object columns on 'isic_id'
dataframe = numeric_columns.merge(object_columns[["isic_id", "tbp_lv_location", "tbp_lv_location_simple"]], how="inner", on="isic_id")

# Encode categorical columns 'tbp_lv_location' and 'tbp_lv_location_simple' using OrdinalEncoder
dataframe[["tbp_lv_location", "tbp_lv_location_simple"]] = encoder.fit_transform(dataframe[["tbp_lv_location", "tbp_lv_location_simple"]])

# Open HDF5 file for reading
fp_hdf = h5py.File("/kaggle/input/isic-2024-challenge/train-image.hdf5", mode="r")

# Create list of paths to image files based on 'isic_id'
path_list = [f"/kaggle/input/isic-2024-challenge/train-image/image/{idx}.jpg" for idx in dataframe["isic_id"]]


numeric_columns.iloc[:, 0] = pd.cut(numeric_columns.iloc[:, 0], bins=[-np.inf, 0.1, 0.5, np.inf], labels=['Low', 'Medium', 'High'])

'''
# Histograms of the first two numerical columns
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  # First subplot
plt.hist(numeric_columns.iloc[:, 0], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Feature 1')
plt.xlabel('Feature 1')
plt.ylabel('Frequency')
'''
# TODO: balnce the data is dataframe where isic_id == isic_id of numeric_columns

# Drop 'isic_id' from dataframe
dataframe.drop("isic_id", axis=1, inplace=True)


import numpy as np
import random
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import h5py

def set_seed(seed=1234):
    '''Sets the seed of the entire notebook for reproducibility.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed for reproducibility across runs
set_seed()

# Initialize an ordinal encoder
encoder = OrdinalEncoder()

# Select numeric columns and include 'isic_id' for identification
numeric_columns = df.select_dtypes(include=['number'])
numeric_columns["isic_id"] = df["isic_id"]

# Drop columns 'mel_thick_mm' and 'tbp_lv_dnn_lesion_confidence', drop rows with missing values, and reset index
numeric_columns = numeric_columns.drop(["mel_thick_mm", "tbp_lv_dnn_lesion_confidence"], axis=1).dropna().reset_index(drop=True)

# Select object (categorical) columns
object_columns = df.select_dtypes(include=['object'])

# Merge numeric and selected object columns on 'isic_id'
dataframe = numeric_columns.merge(object_columns[["isic_id", "tbp_lv_location", "tbp_lv_location_simple"]], how="inner", on="isic_id")

# Encode categorical columns 'tbp_lv_location' and 'tbp_lv_location_simple' using OrdinalEncoder
dataframe[["tbp_lv_location", "tbp_lv_location_simple"]] = encoder.fit_transform(dataframe[["tbp_lv_location", "tbp_lv_location_simple"]])

# Open HDF5 file for reading
fp_hdf = h5py.File("/kaggle/input/isic-2024-challenge/train-image.hdf5", mode="r")

# Create list of paths to image files based on 'isic_id'
path_list = [f"/kaggle/input/isic-2024-challenge/train-image/image/{idx}.jpg" for idx in dataframe["isic_id"]]

# Bin Feature 1
numeric_columns.iloc[:, 0] = pd.cut(numeric_columns.iloc[:, 0], bins=[-np.inf, 0.1, 0.5, np.inf], labels=['Low', 'Medium', 'High'])
'''
# Plot histograms of the first two numerical columns
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  # First subplot
plt.hist(numeric_columns.iloc[:, 0], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Feature 1')
plt.xlabel('Feature 1')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)  # Second subplot
plt.hist(numeric_columns.iloc[:, 1], bins=30, color='green', alpha=0.7)
plt.title('Histogram of Feature 2')
plt.xlabel('Feature 2')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
'''
# Balance the data by subsampling
# Separate the samples
feature_1_zeros = dataframe[dataframe.iloc[:, 0] == 0]
feature_1_ones = dataframe[dataframe.iloc[:, 0] == 1]

# Determine the smaller group size
sample_size = min(len(feature_1_zeros), len(feature_1_ones))

# Randomly sample from both groups
feature_1_zeros_sample = feature_1_zeros.sample(sample_size, random_state=1234)
feature_1_ones_sample = feature_1_ones.sample(sample_size, random_state=1234)

# Combine the samples into a balanced dataframe
balanced_dataframe = pd.concat([feature_1_zeros_sample, feature_1_ones_sample]).reset_index(drop=True)

# Plot the distribution of the balanced Feature 1
'''
plt.figure(figsize=(8, 6))
balanced_dataframe.iloc[:, 0].value_counts().plot(kind='bar', color='purple', alpha=0.7)
plt.title('Distribution of Balanced Feature 1')
plt.xlabel('Feature 1')
plt.ylabel('Frequency')
plt.show()
path_list = [f"/kaggle/input/isic-2024-challenge/train-image/image/{idx}.jpg" for idx in balanced_dataframe["isic_id"]]
'''
# Drop 'isic_id' from dataframe
dataframe.drop("isic_id", axis=1, inplace=True)
balanced_dataframe.drop("isic_id", axis=1, inplace=True)



class MelanomaDataset(Dataset):
    
    def __init__(self, path_list, dataframe, vertical_flip, horizontal_flip, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.path_list = path_list
        
        # Define torchvision transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(p=horizontal_flip),
                transforms.RandomVerticalFlip(p=vertical_flip),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Removed hue
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        image_path = self.path_list[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if necessary
    
        # Convert to PIL Image and print shape and dtype
        image = Image.fromarray(image)
    
        # Apply transforms
        try:
            image = self.transform(image)
        except OverflowError as e:
            print(f"Error with image at index {index}: {e}")
        # Apply some default transformation or handle the error specifically

    
        csv_data = torch.tensor(self.dataframe.iloc[index, :].values, dtype=torch.float32)
        
        if self.is_train:
            target = torch.tensor(self.dataframe['target'][index], dtype=torch.long)
            return (image, csv_data), target
        else:
            return (image, csv_data)


# Data object and Loader
train_ds = MelanomaDataset(path_list, balanced_dataframe, vertical_flip=0.5, horizontal_flip=0.5, is_train=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

# Obtener un batch de datos
for batch_idx, ((images, csv_data), labels) in enumerate(train_dl):
    if batch_idx >= 1:  # Muestra solo el primer batch
        break
    
    # Convertir imágenes de Tensor a numpy para visualización
    images_np = images.numpy()
    csv_data_np = csv_data.numpy()
    
    # Mostrar las primeras 24 imágenes
    num_images_to_show = 24
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for i in range(num_images_to_show):
        row = i // 6
        col = i % 6
        ax = axes[row, col]
        ax.imshow(np.transpose(images_np[i], (1, 2, 0)))  # Transponer para convertir de (3, 224, 224) a (224, 224, 3)
        ax.axis('off')
        #print(np.shape(images_np[i]))
    
    plt.tight_layout()
    plt.show()



import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, csv_data, transform=None):
        self.image_paths = image_paths
        self.csv_data = csv_data
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        csv_data = torch.tensor(self.csv_data.iloc[idx].values, dtype=torch.float32)
        return image, csv_data

class SimpleCNN(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns = no_columns
        self.output_size = output_size

        # Define simple CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Define fully connected network for CSV data
        self.csv = nn.Sequential(
            nn.Linear(self.no_columns, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        # Define classification network
        self.classification = nn.Sequential(
            nn.Linear(128 * 28 * 28 + 128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_size)
        )

    def forward(self, image, csv_data):
        # Image CNN
        image = self.cnn(image)
        image = image.view(image.size(0), -1)

        # CSV FNN
        csv_data = self.csv(csv_data)
        
        # Concatenate image and CSV data
        combined = torch.cat((image, csv_data), dim=1)
        
        # Classification
        out = self.classification(combined)
        
        return out

# Data augmentation and normalization for training
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assuming image_paths and csv_data are defined elsewhere
# image_paths = [...]  # List of image paths
# csv_data = pd.read_csv('path/to/your/csvfile.csv')

# Create Dataset and DataLoader
# dataset = CustomDataset(image_paths, csv_data, transform=data_transforms)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create an instance of the Simple CNN model
output_size = 1  # Adjust as needed
no_columns = csv_data.shape[1]  # Adjust this according to your data
model = SimpleCNN(output_size=output_size, no_columns=no_columns)

# Print the model architecture
print(model)

# Calculate and print the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters.')






from torch.utils.data import random_split

# Total dataset length
total_size = len(train_ds)
val_size = int(0.2 * total_size)  # 20% validation
train_size = total_size - val_size

# Split dataset
train_ds, val_ds = random_split(train_ds, [train_size, val_size])

# Dataloaders
from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)



import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training lop
epochs = 10
losses = []

try:
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        for batch_idx, ((images, csv_data), labels) in enumerate(train_dl):
            try:
                # Move data to device
                images, csv_data, labels = images.to(device), csv_data.to(device), labels.to(device)

                # Forward pass
                outputs = model(images, csv_data)
                loss = criterion(outputs.squeeze(), labels.float())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Store loss value
                losses.append(loss.item())

                if (batch_idx + 1) % 7 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_dl)}], Loss: {loss.item():.4f}')

            except Exception as e:
                print(f"An error occurred in batch {batch_idx+1}: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Plot the loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.legend()
plt.show()
model_path = 'model_50.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


# Function to calculate accuracy
def calculate_accuracy(model, data_loader, device):
    model.eval()  # Evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation for evaluation
        for (images, csv_data), labels in data_loader:
            images, csv_data, labels = images.to(device), csv_data.to(device), labels.to(device)

            outputs = model(images, csv_data)
            predictions = torch.sigmoid(outputs.squeeze()) > 0.5  # Convert logits to binary predictions

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

# Compute accuracy
train_accuracy = calculate_accuracy(model, train_dl, device)
print(f"Training Accuracy: {train_accuracy:.2f}%")

# If you have a validation/test dataset:
# test_accuracy = calculate_accuracy(model, test_dl, device)
# print(f"Test Accuracy: {test_accuracy:.2f}%")




import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import h5py
from PIL import Image
import io
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, hdf5_file, metadata_file, transform=None, target_size=(224, 224)):
        self.hdf5_file = hdf5_file
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.target_size = target_size
        self.h5_file = h5py.File(self.hdf5_file, 'r')
        self.file_names = list(self.h5_file.keys())

        # Ensure all columns except 'isic_id' are numeric
        for col in self.metadata.columns:
            if col != 'isic_id':
                self.metadata[col] = pd.to_numeric(self.metadata[col], errors='coerce')
        self.metadata = self.metadata.fillna(0)
        self.numeric_metadata = self.metadata.select_dtypes(include=[np.number])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        isic_id = self.file_names[idx]
        jpeg_string = self.h5_file[isic_id][()]

        image = Image.open(io.BytesIO(jpeg_string)).convert('RGB').resize(self.target_size)

        # Convert PIL image to OpenCV format
        image_cv = np.array(image)
        image_cv = image_cv[:, :, ::-1].copy()  # Convert RGB to BGR

        # Apply hair removal
        processed_image_cv = hair_remove(image_cv)

        # Convert processed OpenCV image back to PIL format
        processed_image = Image.fromarray(processed_image_cv[:, :, ::-1]) 
        if self.transform:
                    image = self.transform(image)
        
        # Match the metadata row by 'isic_id'
        row = self.metadata[self.metadata['isic_id'] == isic_id].iloc[0]
        csv_data = torch.tensor(row.drop('isic_id').values.astype(np.float32))
        
        return image, csv_data, isic_id

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()



import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import h5py
from tqdm import tqdm
import gc
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

class CustomDataset(Dataset):
    def __init__(self, hdf5_file, metadata_file, transform=None):
        self.hdf5_file = hdf5_file
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

        with h5py.File(hdf5_file, 'r') as f:
            self.image_ids = list(f.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        with h5py.File(self.hdf5_file, 'r') as f:
            image = Image.open(io.BytesIO(np.array(f[image_id])))
        if self.transform:
            image = self.transform(image)
        csv_data = torch.tensor(self.metadata.loc[self.metadata['isic_id'] == image_id].drop('isic_id', axis=1).values[0], dtype=torch.float32)
        return image, csv_data, image_id

class SimpleCNN(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns = no_columns
        self.output_size = output_size

        # Define simple CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Define fully connected network for CSV data
        self.csv = nn.Sequential(
            nn.Linear(self.no_columns, 128),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        # Define classification network
        self.classification = nn.Sequential(
            nn.Linear(128 * 28 * 28 + 128, 128),
            #nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_size)
        )

    def forward(self, image, csv_data):
        # Image CNN
        image = self.cnn(image)
        image = image.view(image.size(0), -1)

        # CSV FNN
        csv_data = self.csv(csv_data)
        
        # Concatenate image and CSV data
        combined = torch.cat((image, csv_data), dim=1)
        
        # Classification
        out = self.classification(combined)
        
        return out

# Define the transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to the HDF5 and metadata files
hdf5_file = "/kaggle/input/isic-2024-challenge/test-image.hdf5"
metadata_file = "/kaggle/input/isic-2024-challenge/sample_submission.csv"

# Create the test dataset and dataloader
test_ds = CustomDataset(hdf5_file, metadata_file, transform=data_transforms)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

# Create an instance of the Simple CNN model
output_size = 1
no_columns = test_ds.metadata.shape[1] - 1  # Adjust this according to your data

model = SimpleCNN(output_size=output_size, no_columns=no_columns)

# Load the model state
model_path = "/kaggle/working/model_50.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Partial state dict loading
pretrained_dict = torch.load(model_path)
model_dict = model.state_dict()

# Filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

# Overwrite entries in the existing state dict
model_dict.update(pretrained_dict)

# Load the new state dict
model.load_state_dict(model_dict)

model.to(device)
model.eval()

# Make predictions
print("Making predictions...")
all_predictions = []
all_ids = []

with torch.no_grad():
    for images, csv_data, ids in tqdm(test_dl):
        images, csv_data = images.to(device), csv_data.to(device)
        outputs = model(images, csv_data)
        all_predictions.extend(outputs.squeeze().cpu().numpy())
        all_ids.extend(ids)

# Ensure predictions are within the range [0, 1]
all_predictions = np.clip(all_predictions, 0, 1)

# Prepare submission
print("Preparing submission...")
submission = pd.DataFrame({
    'isic_id': all_ids,
    'target': all_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Predictions completed and saved to submission.csv")

# Print some predictions
print("\nSample Predictions:")
print(submission.head(10))

# Final cleanup
del model, all_predictions, all_ids, submission
gc.collect()

end = time.time()
print(end - start, "seconds")





