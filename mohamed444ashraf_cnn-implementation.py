

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



!apt-get install -y p7zip-full


!7z x  /kaggle/input/cifar-10/train.7z -o/kaggle/working/train


import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from PIL import Image
import os 


CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

class CIFAR10CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = LABEL_TO_INDEX

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx, 0]
        label_str = self.labels_df.iloc[idx, 1]
        label_index = self.label_map[label_str]
        label_tensor = torch.tensor(label_index , dtype = torch.long)

        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_tensor


transform = transforms.Compose([transforms.ToTensor()])


dataset = CIFAR10CustomDataset(
    csv_file="/kaggle/input/cifar-10/trainLabels.csv",
    img_dir="/kaggle/working/train/train",
    transform=transform
)

loader = DataLoader(dataset,batch_size=8,shuffle=True)


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

images, labels = next(iter(loader))

plt.figure(figsize=(10,4))
for i in range(8):
    img = np.transpose(images[i].numpy(), (1,2,0))
    plt.subplot(2,4,i+1)
    plt.imshow(img)
    plt.title(labels[i])
    plt.axis("off")

plt.show()


from torchvision.transforms import functional as F


img_pil = F.to_pil_image(images[0])


ops = {
    'original' : img_pil ,
    'Resized' : transforms.Resize((16 , 16))(img_pil) , 
    'Center Crop' : transforms.CenterCrop(16)(img_pil) ,
    'Rotated' : transforms.RandomRotation(45)(img_pil) ,
    'Grayscale' : transforms.Grayscale(num_output_channels = 1)(img_pil)
}


plt.figure(figsize=(10,4))
 
for i, (name, im) in enumerate(ops.items()):
    plt.subplot(2, 3, i+1)
 
    arr = np.array(im)
 
    if name == "Grayscale":
        plt.imshow(arr, cmap="gray")
    else:
        plt.imshow(arr)
 
    plt.title(name)
    plt.axis("off")
 
plt.show()


from skimage.feature import hog
from skimage.color import rgb2gray


sample = images[0].numpy().transpose(1 ,2, 0)

gray = rgb2gray(sample)

hog_features , hog_image = hog(gray , visualize = True)

plt.subplot(1 ,2,1)
plt.imshow(gray , cmap = 'gray')
plt.title("Original Iamge")
plt.subplot(1 ,2, 2)
plt.imshow(hog_image , cmap = 'gray' )
plt.title("HOG Iamge")
plt.show()


import torch.nn as nn
import torch


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16x16
 
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 8x8
 
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
       
    def forward(self, x):
        return self.net(x)
 
model = SimpleCNN()


from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_cnn(epochs=3):
    for epoch in range(epochs):
        total, correct, total_loss = 0, 0, 0
        
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)

            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}: Loss={total_loss:.3f}, Acc={correct/total:.3f}")

train_cnn()

