import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import os
import time
import glob
import zipfile
import requests
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.metrics import accuracy_score, mean_squared_error

%matplotlib inline


input_dir = "/kaggle/input/galaxy-zoo-the-galaxy-challenge"
output_dir = "/kaggle/working/data"

zip_files = glob.glob(input_dir + "/*.zip")

for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(output_dir)



# create new subdirectories for images (if they don't exist -- not sure why sometimes Kaggle removes them and sometimes not)
train_dir = os.path.join(output_dir, 'images_training_rev1')
test_dir = os.path.join(output_dir, 'images_test_rev1')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Dfine image file extensions to check against (will be only jpg here but hey never hurts to have some redundancy)
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

# Iterate over files in the base directory
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    # skip directories
    if os.path.isfile(file_path):
        # Check if its a CSV file. If so, leave it 
        if filename.lower().endswith('.csv'):
            continue

        # get file extension
        ext = os.path.splitext(filename)[1].lower()

        # if the file is an image, decide where to move it based on filename
        if ext in image_extensions:
            if 'train' in filename.lower():
                destination = os.path.join(train_dir, filename)
            elif 'test' in filename.lower():
                destination = os.path.join(test_dir, filename)
            else:
                # If no clear indicator print a message to skip the file
                print(f"Skipping {filename}: no train/test indicator found.")
                continue
            # Move the file
            shutil.move(file_path, destination)
            print(f"Moved {filename} to {destination}")



output_dir = "/kaggle/working/data/"
train_solutions = pd.read_csv(output_dir+'training_solutions_rev1.csv')
train_solutions.head()


convnext = models.convnext_tiny(pretrained=True)
num_classes = len(train_solutions.columns)-1


convnext.classifier


# freeze all parameters
for param in convnext.parameters():
    param.requires_grad = False

# modify final layer
in_features = convnext.classifier[2].in_features
convnext.classifier[2] = nn.Linear(in_features, num_classes)

# unfreeze only the final layer parameters
for param in convnext.classifier.parameters():
    param.requires_grad = True

# uncomment to confirm only final layer is trainable
# for name, param in convnext.named_parameters():
#     print(name, param.requires_grad)


convnext = models.convnext_small(pretrained=True)
num_classes = len(train_solutions.columns)-1

# Freeze all parameters
for param in convnext.parameters():
    param.requires_grad = False

# Replace the final layer
in_features = convnext.classifier[2].in_features
convnext.classifier[2] = nn.Linear(in_features, num_classes)

# Unfreeze only the classifier's parameters
for param in convnext.classifier.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
convnext = convnext.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # means and stds of pretrained ImageNet model
])

class GalaxyZooTrainDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        #self.length = len(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # first column is 'GalaxyId'
        galaxy_id = int(row.iloc[0])
        image_path = os.path.join(self.image_dir, f"{galaxy_id}.jpg")
        image = Image.open(image_path).convert('RGB')# /255
        if self.transform:
            image = self.transform(image)
        # get labels from the remaining columns
        labels = row.iloc[1:].values.astype(np.float32)
        return image, labels

train_dataset_full = GalaxyZooTrainDataset(
    csv_file=output_dir+'training_solutions_rev1.csv',
    image_dir=output_dir+'images_training_rev1',
    transform=transform
) # Create the full training dataset

# make a validation set to.. well, validate
total = len(train_dataset_full)
train_size = int(0.8 * total)
val_size = int(0.2 * total)
unused_size = total - (train_size + val_size)

# random_split to partition the dataset into three disjoint subsets
train_subset, val_subset, _ = random_split(train_dataset_full, [train_size, val_size, unused_size])

# creates DataLoaders
train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

criterion = nn.BCEWithLogitsLoss() # tried KLDivLoss first, realized the classes dont all sum to 1.0. fixed
optimizer = torch.optim.Adam(convnext.parameters(), lr=1e-4)

num_epochs = 30

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # training loop
    convnext.train()
    running_loss = 0.0
    last_print_time = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = convnext(images)
        #log_probs = F.log_softmax(outputs, dim=1) # this was needed for KLDivLoss, not anymore. if needed, change loss to criterion(log_probs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        if batch_idx % 50 == 0:
            current_time = time.time()
            elapsed = current_time - last_print_time
            print(f"\n[{elapsed:.1f}s elapsed] Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}\n")
            last_print_time = current_time
    
    avg_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")
    
    # validation loop
    convnext.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = convnext(images)
            #log_probs = F.log_softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
    



fig = figure(figsize=(8,8), facecolor='w')
frame = fig.add_subplot(1,1,1)
frame.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color="b", lw=3)
frame.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color="orange", lw=3)
frame.set_xlabel(r"Epoch", fontsize = 15)
frame.set_ylabel(r"Loss", fontsize = 15)
#frame.set_yscale('log')
frame.set_title('Training vs. Validation Loss', fontsize = 18)
frame.legend(prop={'size': 18})
frame.grid()
show()
fig.savefig('loss_curve.png')


class GalaxyZooTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        galaxy_id = os.path.splitext(filename)[0]
        return image, galaxy_id

test_dataset = GalaxyZooTestDataset(image_dir=output_dir+'images_test_rev1', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

convnext.eval()
all_predictions = []
all_galaxy_ids = []

with torch.no_grad():
    for images, galaxy_ids in test_loader:
        images = images.to(device)
        outputs = convnext(images) # raw logits
        probs = torch.sigmoid(outputs) # convert logits -> [0, 1] for each output
        all_predictions.append(probs.cpu().numpy())
        all_galaxy_ids.extend(galaxy_ids)

predictions = np.concatenate(all_predictions, axis=0)

columns = ['GalaxyId']
questions = {1: 3, 2: 2, 3: 2, 4: 2, 5: 4, 6: 2, 7: 3, 8: 7, 9: 3, 10: 3, 11: 6}
for q, count in questions.items():
    for i in range(1, count + 1):
        columns.append(f'Class{q}.{i}')

num_pred_columns = sum(questions.values())
assert predictions.shape[1] == num_pred_columns, "Mismatch between predictions and expected class columns."

submission_df = pd.DataFrame(predictions, columns=columns[1:])
submission_df.insert(0, 'GalaxyId', all_galaxy_ids)
print(submission_df.head())
submission_df.to_csv('/kaggle/working/submission.csv', index=False)




