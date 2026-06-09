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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import albumentations as A

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import zipfile
import os

from typing import List

import warnings
warnings.filterwarnings(action='ignore')


facial_keypoints_detection_path = '/kaggle/input/facial-keypoints-detection'

facial_keypoints_detection_path_out = '/kaggle/working/'


test_zip_path = os.path.join(facial_keypoints_detection_path, 'test.zip')
train_zip_path = os.path.join(facial_keypoints_detection_path, 'training.zip')

# Create directories to extract the files
test_extracted_path = os.path.join(facial_keypoints_detection_path_out, 'test')
train_extracted_path = os.path.join(facial_keypoints_detection_path_out, 'training')

os.makedirs(test_extracted_path, exist_ok=True)
os.makedirs(train_extracted_path, exist_ok=True)

# Unzip the files
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test_extracted_path)

with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(train_extracted_path)

print("Extraction complete.")


train_df = pd.read_csv(os.path.join(train_extracted_path, 'training.csv'))
train_df.head()


def convert_image_to_np(images):

  images_np = images['Image'].apply(lambda x: np.array(x.split(' ')).astype(np.uint8).reshape(96, 96))
  return images_np

def handle_training_data(training_df):

  dropped_nan_training_df = training_df.dropna()
  dropped_nan_images = convert_image_to_np(dropped_nan_training_df)
  dropped_nan_images_np_stacked = np.stack(dropped_nan_images.values)
  dropped_nan_labels = dropped_nan_training_df.iloc[:, :-1]

  images_np = convert_image_to_np(training_df)
  images_np_stacked = np.stack(images_np.values)

  labels = training_df.iloc[:, :-1]
  new_labels = labels.fillna(labels.mean())
  return images_np_stacked, new_labels, dropped_nan_images_np_stacked, dropped_nan_labels


original_images, original_labels, cleaned_images, cleaned_labels = handle_training_data(train_df)


id_lookup_df = pd.read_csv(os.path.join(facial_keypoints_detection_path, 'IdLookupTable.csv'))
id_lookup_df.head()


sub_df = pd.read_csv(os.path.join(facial_keypoints_detection_path, 'SampleSubmission.csv'))
sub_df.head()


test_df = pd.read_csv(os.path.join(test_extracted_path, 'test.csv'))
test_df.head()


def transform_labels_to_list(labels_df):

    targets_list = []
    for index, row in labels_df.iterrows():
        keypoints = []
        for i in range(0, len(row), 2):
            x_val = row.iloc[i]
            y_val = row.iloc[i+1]
            # if pd.notna(x_val) and pd.notna(y_val):
            keypoints.append([float(x_val), float(y_val)])
            # else:
                # keypoints.append([0, 0]) # Handle NaN values if needed
        targets_list.append(keypoints)

    labels_list = []

    for i in range(0, len(labels_df.columns), 2):
        labels_list.append(labels_df.columns[i][:-2])

    return labels_list, targets_list

original_labels_list, original_targets_list = transform_labels_to_list(original_labels)
print(original_labels_list)
len(original_targets_list), len(original_targets_list[0]), len(original_targets_list[0][0])


_ , cleaned_targets_list = transform_labels_to_list(cleaned_labels)

len(cleaned_targets_list), len(cleaned_targets_list[0]), len(cleaned_targets_list[0][0])


def plot_image_with_keypoints(image, targets, title="Image with Keypoints"):

    x_coords = [coor[0] for coor in targets]
    y_coords = [coor[1] for coor in targets]

    plt.imshow(image, cmap='gray')
    plt.scatter(x_coords, y_coords, color='red', marker='o', s=10)
    plt.title(title)
    plt.axis('off')
    plt.show()


plot_image_with_keypoints(original_images[2] ,original_targets_list[2])


train_transform = A.Compose([

    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), rotate=(-20, 20), translate_percent=(0.05, 0.05), p=0.8),

    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.GaussNoise(p=0.2,std_range=(0,0.05)),

    A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(5, 10), hole_width_range=(5, 10),
                      fill=0, p=0.2)

], keypoint_params=A.KeypointParams(format='xy', label_fields=[]))



def plot_transformed_images(image, keypoints, keypoint_labels, transform, num_plots=10):

    fig, axes = plt.subplots(2, int((num_plots+1)/2), figsize=(20, 8))

    for i, ax in enumerate(axes.flatten()):
        try:

            augmented = transform(image=image, keypoints=keypoints, keypoint_labels=keypoint_labels)
            transformed_image = augmented['image']
            transformed_keypoints = augmented['keypoints']

            x_coords = [coor[0] for coor in transformed_keypoints]
            y_coords = [coor[1] for coor in transformed_keypoints]

            ax.imshow(transformed_image, cmap='gray')
            ax.scatter(x_coords, y_coords, color='red', marker='o', s=10)
            ax.set_title(f"Transformed {i+1}")
            ax.axis('off')

        except Exception as e:
            print(f"Error applying transform for plot {i+1}: {e}")



plot_transformed_images(
    image=original_images[2],
    keypoints=original_targets_list[2],
    keypoint_labels=original_labels_list,
    transform=train_transform
)


class FacialKeypointsDataset(Dataset):

    def __init__(self, images, keypoints=None, transform=None):
        self.images = images
        self.keypoints = keypoints
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]

        new_keypoints, keypoints = None, None
        if self.keypoints is not None:
          keypoints = self.keypoints[idx].copy()

        image = image.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, keypoints=keypoints)
            image = augmented['image']
            if self.keypoints is not None:
                new_keypoints = augmented['keypoints']

        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float()

        if new_keypoints is not None and len(new_keypoints)==keypoints:
            keypoints = new_keypoints

        if self.keypoints is not None:
            keypoints_tensor = torch.tensor(keypoints).view(-1).float()
            return image_tensor, keypoints_tensor
        else:
            return image_tensor



import torch
import torch.nn as nn
from torchvision import models

class KeypointRegressor(nn.Module):
    def __init__(self):
        super(KeypointRegressor, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.encoder.parameters():
            param.requires_grad = False

        num_features = resnet.fc.in_features

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 30)
        )

    def forward(self, x):

        features = self.encoder(x)

        keypoints = self.head(features)

        return keypoints


X_train, X_test, y_train, y_test = train_test_split(
    original_images,
    original_targets_list,
    test_size=0.2,
    random_state=42
)

print("Shape of training images:", X_train.shape)
print("Shape of testing images:", X_test.shape)
print("Number of training labels:", len(y_train))
print("Number of testing labels:", len(y_test))


supervised_training_dataset = FacialKeypointsDataset(
    images=X_train,
    keypoints=y_train,
    transform=train_transform
)

supervised_testing_dataset = FacialKeypointsDataset(
    images=X_test,
    keypoints=y_test,
)

supervised_training_dl = DataLoader(
    supervised_training_dataset,
    batch_size=32,
    shuffle=True
)

supervised_testing_dl = DataLoader(
    supervised_testing_dataset,
    batch_size=32,
    shuffle=False
)


reg_model = KeypointRegressor()

criterion = nn.MSELoss()

optimizer = optim.Adam(reg_model.head.parameters(),lr=0.001)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reg_model.to(device);


NUM_EPOCHS = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reg_model.to(device);

print("Starting Supervised Training...")
supervised_training_loss , supervised_testing_loss = [], []
for epoch in range(NUM_EPOCHS):
    reg_model.train()
    running_train_loss = 0.0

    # Training loop
    progress_bar = tqdm(supervised_training_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Training)")
    for images, keypoints in progress_bar:
        images, keypoints = images.to(device), keypoints.to(device)

        optimizer.zero_grad()

        predicted_keypoints = reg_model(images)
        loss = criterion(predicted_keypoints, keypoints)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

    epoch_train_loss = running_train_loss / len(supervised_training_dl)
    supervised_training_loss.append(epoch_train_loss)

    # Validation loop
    reg_model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        progress_bar_val = tqdm(supervised_testing_dl, desc=f"Validation : ")
        for images, keypoints in progress_bar_val:
            images, keypoints = images.to(device), keypoints.to(device)

            predicted_keypoints = reg_model(images)
            loss = criterion(predicted_keypoints, keypoints)

            running_val_loss += loss.item()

            progress_bar_val.set_postfix(loss=f"{loss.item():.6f}")

    epoch_val_loss = running_val_loss / len(supervised_testing_dl)
    supervised_testing_loss.append(epoch_val_loss)

    print(f"Training Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}")

print("Finished Training!")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(supervised_training_loss, label='Training Loss')
plt.plot(supervised_testing_loss, label='Validation Loss')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


test_images_numpy = np.stack(convert_image_to_np(test_df).values)
test_images_numpy.shape


testing_dataset = FacialKeypointsDataset(
    images=test_images_numpy,
)

testing_dl = DataLoader(
    testing_dataset,
    batch_size=32,
    shuffle=False
)


reg_model.eval()
test_predictions = []

with torch.no_grad():
    for images in tqdm(testing_dl, desc="Predicting on test data"):
        images = images.to(device)
        outputs = reg_model(images)
        test_predictions.extend(outputs.cpu().numpy())


test_predictions_df = pd.DataFrame(test_predictions, columns=original_labels.columns)

test_predictions_df.insert(0, 'ImageId', test_df['ImageId'].values)

display(test_predictions_df)


# Prepare submission file
submission_df = pd.DataFrame(columns=['RowId', 'Location'])
row_id_counter = 1

for index, row in id_lookup_df.iterrows():
    image_id = row['ImageId']
    feature_name = row['FeatureName']
    row_id = row['RowId']

    predicted_location = test_predictions_df.loc[test_predictions_df['ImageId'] == image_id, feature_name].values[0]

    submission_df = pd.concat([submission_df, pd.DataFrame({'RowId': [row_id], 'Location': [predicted_location]})], ignore_index=True)


submission_df['Location'] = submission_df['Location'].apply(lambda x: max(0,min(x,96))) # to handle if the keypoint coordinate not in the 96*96 pixels
submission_df


# Save the submission file
submission_df.to_csv('facial_keypoints_submission2.csv', index=False)




