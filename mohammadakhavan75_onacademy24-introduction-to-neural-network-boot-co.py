import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms.v2 as v2
import pydicom as dicom
import matplotlib.pylab as plt

from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, random_split




# specify your image path
image_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/100206310/1012284084/10.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array, cmap='gray')


# specify your image path
# image_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/100206310/1012284084/8.dcm'
# image_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/100206310/1792451510/8.dcm'
# image_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/100206310/2092806862/8.dcm'
image_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/100206310/1792451510/1.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array, cmap='gray')


df1 = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')


df1.head()


df2 = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')


df2.head()


patients = {}

study_ids = df1['study_id'].unique()
for study_id in study_ids:
    patients[study_id] = {}
    for row_id in range(len(df2[df2['study_id']==study_id]['condition'])):
        condition = df2[df2['study_id']==study_id]['condition'].iloc[row_id].lower().replace(' ', '_')
        level = df2[df2['study_id']==study_id]['level'].iloc[row_id].lower().replace('/','_')
        col_id = condition + "_" + level
        status = df1[df1['study_id']==study_id][col_id].values[0]

        if type(status) is not str and math.isnan(status):
            continue
        
        if status.lower() == 'Normal/Mild'.lower():
            label = 0
        elif status.lower() == 'Moderate'.lower():
            label = 1
        if status.lower() == 'Severe'.lower():
            label = 2

        serie_id = df2[df2['study_id']==study_id]['series_id'].iloc[row_id]
        instance_number = df2[df2['study_id']==study_id]['instance_number'].iloc[row_id]

        if serie_id in patients[study_id].keys():
            if instance_number in patients[study_id][serie_id].keys():
                if patients[study_id][serie_id][instance_number] == 0 and label != 0:
                    patients[study_id][serie_id][instance_number] = label
                elif patients[study_id][serie_id][instance_number] == 1 and label == 2:
                    patients[study_id][serie_id][instance_number] = label
            else:
                patients[study_id][serie_id][instance_number] = label
        else:
            patients[study_id][serie_id] = {}
            patients[study_id][serie_id][instance_number] = label





root_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/"
for patient_id in patients.keys():
    for serie_id in patients[patient_id].keys():
        for img_id in patients[patient_id][serie_id].keys():
            print("img_path is: ", os.path.join(root_path, str(patient_id), str(serie_id), str(img_id) + ".dcm"), "Label is: ", patients[patient_id][serie_id][img_id])
    break


class lumbar_dataset(Dataset):
    def __init__(self, root_path, transform):
        patients = {}
        study_ids = df1['study_id'].unique()
        for study_id in study_ids:
            patients[study_id] = {}
            for row_id in range(len(df2[df2['study_id']==study_id]['condition'])):
                condition = df2[df2['study_id']==study_id]['condition'].iloc[row_id].lower().replace(' ', '_')
                level = df2[df2['study_id']==study_id]['level'].iloc[row_id].lower().replace('/','_')
                col_id = condition + "_" + level
                status = df1[df1['study_id']==study_id][col_id].values[0]
        
                if type(status) is not str and math.isnan(status):
                    continue
                
                if status.lower() == 'Normal/Mild'.lower():
                    label = 0
                elif status.lower() == 'Moderate'.lower():
                    label = 1
                if status.lower() == 'Severe'.lower():
                    label = 2
        
                serie_id = df2[df2['study_id']==study_id]['series_id'].iloc[row_id]
                instance_number = df2[df2['study_id']==study_id]['instance_number'].iloc[row_id]
        
                if serie_id in patients[study_id].keys():
                    if instance_number in patients[study_id][serie_id].keys():
                        if patients[study_id][serie_id][instance_number] == 0 and label != 0:
                            patients[study_id][serie_id][instance_number] = label
                        elif patients[study_id][serie_id][instance_number] == 1 and label == 2:
                            patients[study_id][serie_id][instance_number] = label
                    else:
                        patients[study_id][serie_id][instance_number] = label
                else:
                    patients[study_id][serie_id] = {}
                    patients[study_id][serie_id][instance_number] = label
        
        
        self.root_path = root_path
        self.imgs_path = []
        self.labels = []
        self.transform = transform
        
        for patient_id in patients.keys():
            for serie_id in patients[patient_id].keys():
                for img_id in patients[patient_id][serie_id].keys():
                    self.imgs_path.append(os.path.join(root_path, str(patient_id), str(serie_id), str(img_id) + ".dcm"))
                    self.labels.append(patients[patient_id][serie_id][img_id])

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        label = self.labels[idx]
        dcm = dicom.dcmread(img_path)
        img = dcm.pixel_array
        # img.expand_dims(axis=2)
        img = self.transform(np.array(img).astype(np.float32))

        return img, label



train_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/"

transform = v2.Compose([
    v2.ToPILImage(),
    v2.Resize(230),
    v2.CenterCrop(224),
    v2.RandomPerspective(distortion_scale=0.05, p=0.3),
    v2.RandomChoice([
        v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.05, 1.)),
        v2.RandomRotation(degrees=(-10, 10)),
    ], p=[0.3, 0.3]),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])

full_dataset = lumbar_dataset(train_path, transform)

# Define split sizes
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)


from sklearn.metrics import accuracy_score

def train(loader, model, criterion, optimizer, device):
    model.train()
    loss_per_epoch = []
    acc_per_epoch = []
    for data, label in tqdm(loader):
        optimizer.zero_grad()

        data = data.to(device)
        label = label.to(device)

        pred = model(data)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        acc = accuracy_score(label.detach().cpu(), pred.argmax(dim=1).detach().cpu())
        acc_per_epoch.append(acc)
        loss_per_epoch.append(loss.item())

    return torch.mean(torch.tensor(loss_per_epoch)), torch.mean(torch.tensor(acc_per_epoch))

def validation(loader, model, criterion, device):
    model.eval()
    loss_per_epoch = []
    acc_per_epoch = []
    with torch.no_grad():
        for data, label in tqdm(loader):

            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)
            acc = accuracy_score(label.detach().cpu(), pred.argmax(dim=1).detach().cpu())
            acc_per_epoch.append(acc)
            loss_per_epoch.append(loss.item())

    return torch.mean(torch.tensor(loss_per_epoch)), torch.mean(torch.tensor(acc_per_epoch))



####################################################################
######## This cell is for multiple node (server) execution #########
####################################################################
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

# def ddp_setup(rank: int, world_size: int):
#    """
#    Args:
#        rank: Unique identifier of each process
#       world_size: Total number of processes
#    """
#    os.environ["MASTER_ADDR"] = "localhost"
#    os.environ["MASTER_PORT"] = "12355"
#    torch.cuda.set_device(rank)
#    init_process_group(backend="nccl", rank=rank, world_size=world_size)

# model = DDP(model, device_ids=["0", "1"])


device = "cuda"
# device = "cpu"

# model = resnet18(weights="IMAGENET1K_V1")
model = resnet18(weights=None)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)
# model.requires_grad = False
# model.fc.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

model = nn.DataParallel(model)

train_loss_per_epoch = []
train_acc_per_epoch = []
val_loss_per_epoch = []
val_acc_per_epoch = []
for epoch in range(50):
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
    val_loss, val_acc = validation(val_loader, model, criterion, device)
    train_loss_per_epoch.append(train_loss)
    train_acc_per_epoch.append(train_acc)
    val_loss_per_epoch.append(val_loss)
    val_acc_per_epoch.append(val_acc)
    print(f"Epoch {epoch+1}/10: Train loss: {train_loss}, Train acc: {train_acc}, Val loss: {val_loss}, Val acc: {val_acc}")


torch.save(model, "./model_epoch_50.pt")
model = torch.load("./model_epoch_50.pt")
##################################################################
##################################################################
torch.save(model.state_dict(), "./model_epoch_50.pt") # Recomended!
model.load_state_dict(torch.load("./model_epoch_50.pt"))


model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)

# model.load_state_dict(torch.load("./model_epoch_50.pt"))


model.layer1.required_grad = False
model.layer2.required_grad = False
model.layer3.required_grad = True
model.layer4.required_grad = True


for i, layer in enumerate(model.children()):
    print(layer)
    layer.required_grad = False
    if i > 10:
        break


model.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

model = nn.DataParallel(model)

train_loss_per_epoch = []
train_acc_per_epoch = []
val_loss_per_epoch = []
val_acc_per_epoch = []
for epoch in range(50):
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
    val_loss, val_acc = validation(val_loader, model, criterion, device)
    train_loss_per_epoch.append(train_loss)
    train_acc_per_epoch.append(train_acc)
    val_loss_per_epoch.append(val_loss)
    val_acc_per_epoch.append(val_acc)
    print(f"Epoch {epoch+1}/10: Train loss: {train_loss}, Train acc: {train_acc}, Val loss: {val_loss}, Val acc: {val_acc}")

