import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
import math
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import zipfile

# set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


kaggle_base_path = "/kaggle/input/denoising-dirty-documents"
save_base_path = "/kaggle/working/image"

file_to_folder = {
    "train.zip": "train",
    "train_cleaned.zip": "train_cleaned",
    "test.zip": "test"
}

for zip_file, folder_name in file_to_folder.items():
    zip_path = os.path.join(kaggle_base_path, zip_file)
    extract_path = os.path.join(save_base_path, folder_name)

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

train_path = os.path.join(save_base_path, "train/train")
cleaned_path = os.path.join(save_base_path, "train_cleaned/train_cleaned")
test_path = os.path.join(save_base_path, "test/test")


train_lst = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
train_lst.sort(key=lambda x: int(x[:-4]))

cleaned_lst = [f for f in os.listdir(cleaned_path) if os.path.isfile(os.path.join(cleaned_path, f))]
cleaned_lst.sort(key=lambda x: int(x[:-4]))

test_lst = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
test_lst.sort(key=lambda x: int(x[:-4]))

def call_data(head, path):
    lst = []
    for filename in path:
        p = head + '/' + filename
        lst.append(np.array(Image.open(p)))
    return lst

train_data = call_data(train_path, train_lst)
cleaned_data = call_data(cleaned_path, cleaned_lst)
test_data = call_data(test_path, test_lst)


def resizing(data):
    empty = np.zeros((81, 540))
    for i in range(len(data)):
        if data[i].shape[0] == 258:
            data[i] = np.concatenate([empty, data[i], empty], axis = 0)
        data[i] = data[i] / 255.0
        data[i] = data[i]
    return torch.from_numpy(np.array(data, dtype = np.float32))

train_tensor = resizing(train_data).unsqueeze(1)
cleaned_tensor = resizing(cleaned_data).unsqueeze(1)
test_tensor = resizing(test_data).unsqueeze(1)


class make_Dataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(make_Dataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

train_dataset = make_Dataset(train_tensor, cleaned_tensor)

train_loader = DataLoader(dataset = train_dataset, batch_size= 16, shuffle= True, num_workers = 2)
test_loader = DataLoader(dataset = test_tensor, batch_size= 16, num_workers = 2)


class CNN_AUTO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        # ë””ì½”ë�”
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


# ref : https://discuss.pytorch.org/t/rmse-loss-function/16540/2

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


from torchsummary import summary
model = CNN_AUTO().to(device)
summary(model, (1, 420, 540))


## train_Loop

epochs = 50
model = CNN_AUTO().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)

        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred = model(test)


def visualize_images_and_outputs(images, outputs):
    num_images = 5
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))

    for i in range(num_images):
        axes[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Original {i + 1}", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Output {i + 1}", fontsize=10)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

visualize_images_and_outputs(test, pred)


class CreateDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.x_data = []

        for path in self.x:
            img = np.array(Image.open(path), dtype=np.float32)
            if transform is not None:
                img = transform(img)
            img = torch.tensor(img).unsqueeze(0)
            self.x_data.append(img)

        self.y = y
        if self.y is not None:
            self.y_data = []
            for path in self.y:
                img = np.array(Image.open(path), dtype=np.float32)
                if transform is not None:
                    img = transform(img)
                img = torch.tensor(img).unsqueeze(0)
                self.y_data.append(img)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_item = self.x_data[idx]

        if self.y is None:
            return x_item
        else:
            y_item = self.y_data[idx]
            return x_item, y_item



def resizing(data):
    height = data.shape[0]

    if height < 420:
        pad = 420 - height
        top = np.zeros((pad // 2, 540))
        bottom = np.zeros((pad - (pad // 2), 540))

        data = np.concatenate([top, data, bottom], axis = 0, dtype = np.float32)
    data = data / 255.0
    return data


transform = resizing
train_dataset = CreateDataset([train_path + '/' + t for t in train_lst],[ cleaned_path + '/' + c for c in cleaned_lst], transform)
test_dataset = CreateDataset([test_path + '/' + te for te in test_lst], transform = transform)

train_loader = DataLoader(train_dataset, batch_size= 16, shuffle= True, drop_last= True)
test_loader = DataLoader(test_dataset, shuffle= False, batch_size= 16, drop_last= True)


class CNN_AUTO_256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_AUTO_256().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
losses_256 = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses_256.append(loss.item())
    if epoch % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_256 = model(test)


class CNN_AUTO_512(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_AUTO_512().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
losses_512 = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses_512.append(loss.item())
    if epoch % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_512 = model(test)


class CNN_AUTO_1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_AUTO_1024().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
losses_1024 = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses_1024.append(loss.item())
    if epoch % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_1024 = model(test)


def visualize_images_and_outputs(ori, a, b, c):
    num_images = 5
    fig, axes = plt.subplots(num_images, 4, figsize=(10, num_images * 3))

    for i in range(num_images):
        axes[i, 0].imshow(ori[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 0].set_title(f"Origianl", fontsize=10)

        axes[i, 0].axis('off')

        axes[i, 1].imshow(a[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 1].set_title(f"256", fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(b[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 2].set_title(f"512", fontsize=10)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(c[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 3].set_title(f"1024", fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

visualize_images_and_outputs(test, pred_256, pred_512, pred_1024)


plt.plot(losses_256, label = '256')
plt.plot(losses_512, label = '512')
plt.plot(losses_1024, label = '1024')

plt.xlabel('step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.show()


train_loader = DataLoader(train_dataset, batch_size= 8, shuffle= True, drop_last= True)
test_loader = DataLoader(test_dataset, shuffle= False, batch_size= 8, drop_last= True)


class CNN_AUTO_B8(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_AUTO_B8().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses_B8 = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    losses_B8.append(loss.item())


model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_B8 = model(test)


train_loader = DataLoader(train_dataset, batch_size= 16, shuffle= True, drop_last= True)
test_loader = DataLoader(test_dataset, shuffle= False, batch_size= 8, drop_last= True)


class CNN_AUTO_B16(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_AUTO_B16().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses_B16 = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    losses_B16.append(loss.item())



model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_B16 = model(test)


train_loader = DataLoader(train_dataset, batch_size= 32, shuffle= True, drop_last= True)
test_loader = DataLoader(test_dataset, shuffle= False, batch_size= 8, drop_last= True)


class CNN_AUTO_B32(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_AUTO_B32().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses_B32 = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 5 == 0:
        losses_B32.append(loss.item())
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_B32 = model(test)


def visualize_images_and_outputs(ori, a, b, c):
    num_images = 5
    fig, axes = plt.subplots(num_images, 4, figsize=(10, num_images * 3))

    for i in range(num_images):
        axes[i, 0].imshow(ori[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 0].set_title(f"Origianl", fontsize=10)

        axes[i, 0].axis('off')

        axes[i, 1].imshow(a[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 1].set_title(f"8", fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(b[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 2].set_title(f"16", fontsize=10)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(c[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 3].set_title(f"32", fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

visualize_images_and_outputs(test, pred_B8, pred_B16, pred_B32)


plt.plot(losses_B8, label = 'Batch_size = 8')
plt.plot(losses_B16, label = 'Batch_size = 16')
plt.plot(losses_B32, label = 'Batch_size = 32')

plt.xlabel('step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.show()


train_loader = DataLoader(train_dataset, batch_size= 16, shuffle= True, drop_last= True)
test_loader = DataLoader(test_dataset, shuffle= False, batch_size= 8, drop_last= True)


class CNN_AUTO_NAdam(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_AUTO_NAdam().to(device)
criterion = RMSELoss()
optimizer = optim.NAdam(model.parameters(), lr=0.001)
losses_NAdam = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    losses_NAdam.append(loss.item())



model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_NAdam = model(test)


plt.plot(losses_NAdam, label = 'NAdam')
plt.plot(losses_B16, label = 'Adam')

plt.xlabel('step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.show()


def visualize_images_and_outputs(ori, a):
    num_images = 5
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))

    for i in range(num_images):
        axes[i, 0].imshow(ori[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 0].set_title(f"NAdam", fontsize=10)

        axes[i, 0].axis('off')

        axes[i, 1].imshow(a[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 1].set_title(f"Adam", fontsize=10)
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

visualize_images_and_outputs(pred_NAdam, pred_B16)

plt.plot(losses_NAdam, label = 'NAdam')
plt.plot(losses_B16, label = 'Adam')

plt.xlabel('step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.show()


class CNN_AUTO_L1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output


def l1_penalty(model):
    l1_norm = 0.0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_norm += param.abs().sum()
    return l1_norm


epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_AUTO_L1().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses_L1 = []
lambda_l1 = 0.0001

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss = loss + lambda_l1 * l1_penalty(model)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    losses_L1.append(loss.item())



model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_L1 = model(test)


def visualize_images_and_outputs(ori, a, b):
    num_images = 5
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))

    for i in range(num_images):
        axes[i, 0].imshow(ori[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 0].set_title(f"test", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(a[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 1].set_title(f"L1", fontsize=10)

        axes[i, 1].axis('off')

        axes[i, 2].imshow(b[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 2].set_title(f"Adam", fontsize=10)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

visualize_images_and_outputs(test, pred_L1, pred_B16)

plt.plot(losses_L1, label = 'L1')
plt.plot(losses_B16, label = 'None')

plt.xlabel('step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.show()


epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_AUTO_B16().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0002)

losses_BCNN_AUTO_scd = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for train, clean in train_loader:
        optimizer.zero_grad()
        train, clean = train.to(device), clean.to(device)
        pred = model(train)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    losses_BCNN_AUTO_scd.append(loss.item())

    scheduler.step()

model.eval()
for test in test_loader:
    optimizer.zero_grad()
    test = test.to(device)
    pred_scd = model(test)


losses_Base = losses_B16
pred_Base = losses_B16


def visualize_images_and_outputs(ori, a, b):
    num_images = 5
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))

    for i in range(num_images):
        axes[i, 0].imshow(ori[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 0].set_title(f"test", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(a[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 1].set_title(f"scheduled", fontsize=10)

        axes[i, 1].axis('off')

        axes[i, 2].imshow(b[i].detach().cpu().numpy().squeeze(), cmap='gray')
        if i == 0:
            axes[i, 2].set_title(f"Adam", fontsize=10)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

visualize_images_and_outputs(test, pred_scd, pred_B16)

plt.plot(losses_BCNN_AUTO_scd[20:90], label = 'scheduled')
plt.plot(losses_B16[20:90], label = 'None')

plt.xlabel('step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.show()

