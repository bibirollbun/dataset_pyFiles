import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns

import matplotlib.pyplot as plt
import os
import time
import numpy as np
import glob
import json
import collections
import torch
import torch.nn as nn

import pydicom as dicom
import matplotlib.patches as patches

from matplotlib import animation, rc
import pandas as pd

import pydicom as dicom # dicom
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# read data
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

train  = pd.read_csv(train_path + 'train.csv')
label = pd.read_csv(train_path + 'train_label_coordinates.csv')
train_desc  = pd.read_csv(train_path + 'train_series_descriptions.csv')
test_desc   = pd.read_csv(train_path + 'test_series_descriptions.csv')
sub         = pd.read_csv(train_path + 'sample_submission.csv')
len(test_desc) #number of test_description.csv rows 


test_desc.head(5)


# Function to generate image paths based on directory structure
def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        images = os.listdir(series_dir)
        image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths

test_image_paths = generate_image_paths(test_desc, f'{train_path}/test_images')
from concurrent.futures import ThreadPoolExecutor
def load_dicom(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data
from concurrent.futures import ThreadPoolExecutor
def load_all_dicom(paths):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_dicom, paths))
    return images


# Define the base path for test images
base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

# Function to get image paths for a series
def get_image_paths(row):
    series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
    if os.path.exists(series_path):
        return [os.path.join(series_path, f) for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))]
    return []

# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# Create a list to store the expanded rows
expanded_rows = []
"""
# Expand the dataframe by adding new rows for each file path
for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row)
    conditions = condition_mapping.get(row['series_description'], {})
    if isinstance(conditions, str):  # Single condition
        conditions = {'left': conditions, 'right': conditions}
    for side, condition in conditions.items():
        for image_path in image_paths:
            expanded_rows.append({
                'study_id': row['study_id'],
                'series_id': row['series_id'],
                'series_description': row['series_description'],
                'image_path': image_path,
                'condition': condition,
                'row_id': f"{row['study_id']}_{condition}"
            })

# Create a new dataframe from the expanded rows
expanded_test_desc = pd.DataFrame(expanded_rows)
"""
for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row)
    loaded_images = load_all_dicom(image_paths)  # Paralel DICOM yükleme
    conditions = condition_mapping.get(row['series_description'], {})
    if isinstance(conditions, str):  # Single condition
        conditions = {'left': conditions, 'right': conditions}
    for side, condition in conditions.items():
        for image_path, dicom_image in zip(image_paths, loaded_images):
            expanded_rows.append({
                'study_id': row['study_id'],
                'series_id': row['series_id'],
                'series_description': row['series_description'],
                'image_path': image_path,
                'dicom_image': dicom_image,  # Yüklenmiş görüntü
                'condition': condition,
                'row_id': f"{row['study_id']}_{condition}"
            })

# DataFrame oluşturma
expanded_test_desc = pd.DataFrame(expanded_rows)
# Display the resulting dataframe
expanded_test_desc.head(5)


test_data = expanded_test_desc


import os

# Define a function to check if a path exists
def check_exists(path):
    return os.path.exists(path)

# Define a function to check if a study ID directory exists
def check_study_id(row):
    study_id = row['study_id']
    path = f'{train_path}/train_images/{study_id}'
    return check_exists(path)

# Define a function to check if a series ID directory exists
def check_series_id(row):
    study_id = row['study_id']
    series_id = row['series_id']
    path = f'{train_path}/train_images/{study_id}/{series_id}'
    return check_exists(path)

# Define a function to check if an image file exists
def check_image_exists(row):
    image_path = row['image_path']
    return check_exists(image_path)


def load_dicom(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

from concurrent.futures import ThreadPoolExecutor

def load_all_dicom(paths):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_dicom, paths))
    return images



import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


expanded_test_desc


levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

# Function to update row_id with levels
def update_row_id(row, levels):
    level = levels[row.name % len(levels)]
    return f"{row['study_id']}_{row['condition']}_{level}"

# Update row_id in expanded_test_desc to include levels
expanded_test_desc['row_id'] = expanded_test_desc.apply(lambda row: update_row_id(row, levels), axis=1)


expanded_test_desc.head(2)



import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
# Define a custom test dataset class
class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        if self.transform:
            image = self.transform(image)
        return image

# Define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Create a test dataset and dataloader
test_dataset = TestDataset(expanded_test_desc, transform)
testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


for image in testloader:
    print(image.shape)
    break


import torch
from torchvision import models
import torch.nn as nn

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained_weights=None):
        super(CustomResNet50, self).__init__()
        
        # Yeni weights parametresini kullanarak ResNet50'yi başlatıyoruz
        self.model = models.resnet50(weights=None).to(device)
        
        # Eğer özel bir ağırlık dosyası varsa yükle
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights, map_location=device), strict=False)
                

        
        # Son katmanı sınıf sayısına göre değiştir
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

"""!!!!!!!!!!!!!!!!!!!!!!!!BURADA YOLLARI OGRENMEN GEREK SABAH DUZELTCEKSIN!!!!!!!!!!!!!!!!!!!!!!!!"""
import torch

# Function to get the model based on series_description and load pretrained weights
def get_model(series_description, weights_paths):
    model_name = series_description
    if model_name in weights_paths:
        model = CustomResNet50(num_classes=3, pretrained_weights=weights_paths[model_name]).to(device)
        model.eval()  # Modeli değerlendirme moduna al
        return model
    return None


"""
# Function to make predictions on the test data
def predict_test_data(testloader, expanded_test_desc, weights_paths):
    predictions = []
    normal_mild_probs = []
    moderate_probs = []
    severe_probs = []
    
    for idx, images in enumerate(tqdm(testloader)):
        images = images.to(device)
        series_description = expanded_test_desc.iloc[idx]['series_description']
        model = get_model(series_description, weights_paths)
        
        if model:
            with torch.no_grad():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).squeeze(0)
                normal_mild_probs.append(probs[0].item())
                moderate_probs.append(probs[1].item())
                severe_probs.append(probs[2].item())
                predictions.append(probs)
        else:
            normal_mild_probs.append(None)
            moderate_probs.append(None)
            severe_probs.append(None)
            predictions.append(None)
    
    return normal_mild_probs, moderate_probs, severe_probs, predictions
    """
def predict_test_data(testloader, expanded_test_desc, weights_paths):
    predictions = []
    normal_mild_probs = []
    moderate_probs = []
    severe_probs = []
    test_predictions = []  # Eklenen liste
    
    for idx, images in enumerate(tqdm(testloader)):
        images = images.to(device)
        series_description = expanded_test_desc.iloc[idx]['series_description']
        model = get_model(series_description, weights_paths)
        
        if model:
            with torch.no_grad():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                for prob in probs:  # Her batch içindeki her görüntü için ayrı ayrı ele al
                    normal_mild_probs.append(prob[0].item())
                    moderate_probs.append(prob[1].item())
                    severe_probs.append(prob[2].item())
                    test_predictions.append(prob.cpu().numpy())  # NumPy formatında sakla
        else:
            for _ in range(len(images)):  # Batch boyutuna göre None ekle
                normal_mild_probs.append(None)
                moderate_probs.append(None)
                severe_probs.append(None)
                test_predictions.append(None)
    
    return normal_mild_probs, moderate_probs, severe_probs, test_predictions




weights_paths = {
    'Sagittal T1': '/kaggle/input/unpretrained-plz-dont-give-any-error-messages/best_model_1.pth',
    'Axial T2': '/kaggle/input/unpretrained-plz-dont-give-any-error-messages/best_model_2.pth',
    'Sagittal T2/STIR': '/kaggle/input/unpretrained-plz-dont-give-any-error-messages/best_model_3.pth'
}


# Make predictions on the test data
normal_mild_probs, moderate_probs, severe_probs, test_predictions = predict_test_data(testloader, expanded_test_desc, weights_paths)


test_predictions[0]


# Add predictions and probabilities to the test DataFrame
expanded_test_desc['normal_mild'] = normal_mild_probs
expanded_test_desc['moderate'] = moderate_probs
expanded_test_desc['severe'] = severe_probs


submission = expanded_test_desc[["row_id","normal_mild","moderate","severe"]]


# Group by 'row_id' and sum the values
grouped_submission = submission.groupby('row_id').max().reset_index()

# Normalize the columns
grouped_submission[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']].div(grouped_submission[['normal_mild', 'moderate', 'severe']].sum(axis=1), axis=0)

# Check the first 3 rows
grouped_submission


len(grouped_submission)


sub[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']]


import os

# Save the DataFrame to "submission.csv" in the desired directory
sub.to_csv("/kaggle/working/submission.csv", index=False)


sub.head(5)

