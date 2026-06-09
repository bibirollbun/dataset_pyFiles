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


import cv2
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import albumentations as A
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms


!unzip -q /kaggle/input/tgs-salt-identification-challenge/train.zip -d train_data


image_dir = '/kaggle/working/train_data/images'
mask_dir = '/kaggle/working/train_data/masks'


len(os.listdir(mask_dir))


#Don't need any manipulation as image and mask names are exactly same
filenames = os.listdir(image_dir)
# for k in os.listdir(image_dir):
#     print(k)
#     break,

train_files,valid_files = train_test_split(filenames,test_size=0.2,random_state=42)


image = Image.open(os.path.join(mask_dir,filenames[10])).convert('L')
image_np = np.array(image)
image_np.shape


#Create a dataset 
class Saltdataset(Dataset):
    def __init__(self,image_dir,mask_dir,filenames,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = filenames
        self.transform = transform 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
        image_name = self.filenames[idx]
        image_path = os.path.join(self.image_dir,image_name)
        mask_path = os.path.join(self.mask_dir,image_name)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        #Need to convert to Numpy arrays as Albumentations library needs it
        image = np.array(image)
        mask = np.array(mask)/255.0

        
        
        if self.transform:
            augmented = self.transform(image = image,mask = mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0)

        # Dont need this as you're using Albumentations
        # image = torch.tensor(image,dtype=torch.float32).permute(2,0,1)
        # mask = torch.tensor(mask,dtype=torch.float32).unsqueeze(0)

        return image,mask


# transform_func = transforms.Compose([transforms.ToTensor()])
#Main difference between albumentations and torchvision transforms is use of ToTensorV2
train_transform = A.Compose([
    A.Resize(128,128),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    A.ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(128,128),
    # A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    A.ToTensorV2()
])


train_dataset = Saltdataset(image_dir,mask_dir,train_files,transform=train_transform)
valid_dataset = Saltdataset(image_dir,mask_dir,valid_files,transform=valid_transform)


train_dataset[3][1].shape


train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=16,shuffle=False)


class Doubleconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
        )

    def forward(self,x):
        return self.doubleconv(x)




class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        #encoder
        self.enc1 = Doubleconv(3,64)
        self.max1 = nn.MaxPool2d(2)
        self.enc2 = Doubleconv(64,128)
        self.max2 = nn.MaxPool2d(2)
        self.enc3 = Doubleconv(128,256)
        self.max3 = nn.MaxPool2d(2)
        self.enc4 = Doubleconv(256,512)
        self.max4 = nn.MaxPool2d(2)

        #bottleneck
        self.bottleneck = Doubleconv(512,1024)

        #decoder
        self.up1 = nn.ConvTranspose2d(1024,512,2,stride=2) #upsample
        self.dec1 = Doubleconv(1024,512)
        self.up2 = nn.ConvTranspose2d(512,256,2,stride=2) #upsample
        self.dec2 = Doubleconv(512,256)
        self.up3 = nn.ConvTranspose2d(256,128,2,stride=2) #upsample
        self.dec3 = Doubleconv(256,128)
        self.up4 = nn.ConvTranspose2d(128,64,2,stride=2) #upsample
        self.dec4 = Doubleconv(128,64)

        self.final = nn.Conv2d(64,1,1)

    def forward(self,x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.max1(x1))
        x3 = self.enc3(self.max2(x2))
        x4 = self.enc4(self.max3(x3))

        #bottleneck
        x5 = self.bottleneck(self.max4(x4))

        #decoder
        d1 = self.up1(x5)
        d1 = torch.cat([d1,x4],dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2,x3],dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3,x2],dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4,x1],dim=1)
        d4 = self.dec4(d4)

        output = self.final(d4)

        return output
        


# !pip install segmentation-models-pytorch


#Using pretrained unet
import segmentation_models_pytorch as smp
# Define model
# model = smp.Unet(
#     encoder_name="resnet34",        # choose encoder
#     encoder_weights="imagenet",     # use ImageNet pre-trained weights
#     in_channels=3,                  # RGB input
#     classes=1,                      # Binary segmentation
# )
# Using efficientnet
# model = smp.Unet(
#     encoder_name="efficientnet-b4",    # EfficientNet-B4 backbone
#     encoder_weights="imagenet",        # Use ImageNet pre-trained weights
#     in_channels=3,                     # RGB input
#     classes=1,                         # Binary segmentation
# )
model = smp.Unet(
    encoder_name="se_resnext50_32x4d", # SE-ResNeXt50_32x4d backbone
    encoder_weights="imagenet",        # Use ImageNet pre-trained weights
    in_channels=3,                     # RGB input
    classes=1,                         # Binary segmentation
)
# efficientnet-b0
# model = smp.Unet(
#     encoder_name="efficientnet-b4", # SE-ResNeXt50_32x4d backbone
#     encoder_weights="imagenet",        # Use ImageNet pre-trained weights
#     in_channels=3,                     # RGB input
#     classes=1,                         # Binary segmentation
# )
# model = smp.UnetPlusPlus(
#     encoder_name="resnet34",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
# )


device = 'cuda'


# model = Unet().to(device)
model = model.to(device)
# model = ResNetUNet(n_classes=1).to(device)


#Hyperparams
from torch.optim import Adam
epochs = 100
loss_fn = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(),lr = 1e-4)



train_losses = []
val_losses = []
val_dice_scores = []

for epoch in range(epochs):
    model.train()
    train_loss = 0 
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        pred = model(images)
        loss = loss_fn(pred, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Train loss for epoch {epoch}: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            pred = model(images)
            loss = loss_fn(pred, masks)
            val_loss += loss.item()
        
            preds = torch.sigmoid(pred)
            preds = (preds > 0.5).float()
            intersection = (preds * masks).sum(dim=(1,2,3))
            union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
            dice = (2. * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())
        
    avg_val_loss = val_loss / len(valid_loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_dice)

    print(f"Validation Loss: {avg_val_loss:.4f} | Dice Score: {avg_dice:.4f}")



import matplotlib.pyplot as plt

epochs_range = range(epochs)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_dice_scores, label='Val Dice Score')
plt.legend()
plt.title('Dice Score over Epochs')

plt.show()



# #trainingloop

# for epoch in range(epochs):
#     model.train()
#     train_loss = 0 
#     for images,masks in train_loader:
#         images,masks = images.to(device),masks.to(device)
#         pred = model(images)
#         loss = loss_fn(pred,masks)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         train_loss += loss.item()
#     print(f"Train loss for epoch{epoch}:{train_loss/len(train_loader)}")

#     model.eval()
#     val_loss = 0.0
#     dice_scores = []
#     with torch.no_grad():
#         for images,masks in valid_loader:
#             images,masks = images.to(device),masks.to(device)
#             pred = model(images)
#             loss = loss_fn(pred,masks)
#             val_loss += loss.item()
    
#             preds = torch.sigmoid(pred)
#             preds = (preds>0.5).float()
#             # Dice score (simple implementation)
#             intersection = (preds * masks).sum(dim=(1,2,3))
#             union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
#             dice = (2. * intersection + 1e-7) / (union + 1e-7)
#             dice_scores.extend(dice.cpu().numpy())
#         avg_loss = val_loss/len(valid_loader)      
#         avg_dice = sum(dice_scores) / len(dice_scores)
    
#         print(f"Validation Loss: {avg_loss:.4f} | Dice Score: {avg_dice:.4f}")
            


# After training is done or when you get the best model
torch.save(model.state_dict(), "unet_salt_resnet.pth")



!unzip -q /kaggle/input/tgs-salt-identification-challenge/test.zip -d test_data1


len(os.listdir('/kaggle/working/test_data1/images'))


# model = Unet().to(device)
# model.load_state_dict(torch.load('/kaggle/input/unet-resnet/pytorch/default/1/unet_salt.pth', map_location='cpu'))
model.eval()


import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T

def rle_encode(mask):
    '''
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    '''
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

test_dir = '/kaggle/working/test_data1/images'
test_files = sorted(os.listdir(test_dir))
results = []

# Use the same transforms as used for validation (resize, tensor, normalize)
val_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.0, 0.0, 0.0], [1.0,1.0,1.0])
     # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
])

model.eval()
for fname in tqdm(test_files):
    img_path = os.path.join(test_dir, fname)
    image = Image.open(img_path).convert('RGB')
    orig_size = image.size  # (width, height)
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.3).float()
        mask = pred.squeeze().cpu().numpy().astype(np.uint8)
    # Resize mask back to original size
    mask = Image.fromarray(mask)
    mask = mask.resize(orig_size, resample=Image.NEAREST)
    mask = np.array(mask)
    rle = rle_encode(mask)
    img_id = os.path.splitext(fname)[0]
    results.append({'id': img_id, 'rle_mask': rle})

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('submission.csv', index=False)


df.info()




