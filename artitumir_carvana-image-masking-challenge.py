# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm.notebook import tqdm
import os
import numpy as np
import torch

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def get_config():
    return {'Width' : 1918,
            'Height': 1280,
            'W_pad_size': 2,
            'Resize_W': 1920 // 4,
            'Resize_H': 1280 // 4,
            'Batch_size': 8,
            'Eval_batch_size': 16,
            'Num_workers': 2,
            'Device': 'cuda' if torch.cuda.is_available() else 'cpu', 
            'Num_epochs': 20,
           }
config = get_config()
config


import zipfile
import os

extract_path = '/kaggle/working'

# Extract the zip file
def unzip(file):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)



train_mask_dir = '/kaggle/input/carvana-image-masking-challenge/train_masks.zip'
unzip(train_mask_dir)

train_dir = '/kaggle/input/carvana-image-masking-challenge/train.zip'
unzip(train_dir)


# unzip('/kaggle/input/carvana-image-masking-challenge/train_hq.zip')
# unzip('/kaggle/input/carvana-image-masking-challenge/metadata.csv.zip')
unzip('/kaggle/input/carvana-image-masking-challenge/sample_submission.csv.zip')
unzip('/kaggle/input/carvana-image-masking-challenge/train_masks.csv.zip')


pd.read_csv('/kaggle/working/train_masks.csv')



def read_file(dir_):
    files = os.listdir(dir_)
    files.sort()
    images, images_resized = [], []
    for fname in tqdm(files, desc="Loading..."):
        path = os.path.join(dir_, fname)
        with Image.open(path) as img:  # Auto-close image after use
            # plt.imshow(img, cmap='gray')
            # plt.show()
            W, H = img.size
            if W != config['Width'] or H != config['Height']:
                print(f"Size mismatch! Expected: {config['Width'], config['Height']}. Got : {W, H}.")
                return
            img = np.array(img)
            pad_width = ((0, 0), (0, 2)) if img.ndim == 2 else ((0, 0), (0, 2), (0, 0))  
            img = np.pad(img, pad_width, mode='edge')           
            # images.append(Image.fromarray(img))

            # plt.imshow(img, cmap='gray')
            # plt.show()
            
            img = cv2.resize(img, (config['Resize_W'], config['Resize_H']), interpolation=cv2.INTER_AREA)
            # plt.imshow(img, cmap='gray')
            # plt.show()
            images_resized.append(Image.fromarray(img))
            # break

    return files, images, images_resized



train_files, train_images, train_images_resized  = read_file('/kaggle/working/train')


# len(train_files), len(train_images), len(train_images_resized), train_images[0].size, train_images_resized[0].size
len(train_files), len(train_images_resized), train_images_resized[0].size


plt.imshow(train_images_resized[0])


train_mask_files, train_masks, train_masks_resized  = read_file('/kaggle/working/train_masks')


len(train_mask_files), len(train_masks_resized), train_masks_resized[0].size


plt.imshow(train_masks_resized[0], cmap='gray')


train_masks_resized[0].size


train_files[22], train_mask_files[22]


import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, files, transform=None):
        self.images = images
        self.masks = masks
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask) * 255

        return img, mask, self.files[idx]


transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [C, H, W] with values in [0, 1]
])

full_dataset = SegmentationDataset(train_images_resized, train_masks_resized, train_files, transform=transform)


val_ratio = 0.1
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=config['Batch_size'], shuffle=True, num_workers=config['Num_workers'], drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['Batch_size'], shuffle=False, num_workers=config['Num_workers'])


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import torch.optim as optim
import copy



class Double_Conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        resnet = models.resnet50(pretrained=True)        
        # Freeze ResNet encoder
        # for param in resnet.parameters():
        #     param.requires_grad = False

        # Encoder
        self.base_layers = list(resnet.children())
        self.input_layer = nn.Sequential(*self.base_layers[:3])       # conv1 + bn1 + relu
        self.maxpool = self.base_layers[3]                            # maxpool
        self.encoder1 = self.base_layers[4]                           # layer1
        self.encoder2 = self.base_layers[5]                           # layer2
        self.encoder3 = self.base_layers[6]                           # layer3
        self.encoder4 = self.base_layers[7]                           # layer4

        # Decoder
        self.decoder5 = DecoderBlock(2048, 1024, 1024)
        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(1024, 512, 256)
        self.decoder2 = DecoderBlock(512, 256, 64)
        self.decoder1 = DecoderBlock(128, 64, 64)
        
        self.decoder_x0 = DecoderBlock(64, 64, 64)

        self.double_conv = Double_Conv(128, 64, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.input_layer(x)          # (B, 64, H/2, W/2)
        x1 = self.maxpool(x0)             # (B, 64, H/4, W/4)
        x2 = self.encoder1(x1)            # (B, 256, H/4, W/4)
        x3 = self.encoder2(x2)            # (B, 512, H/8, W/8)
        x4 = self.encoder3(x3)            # (B, 1024, H/16, W/16)
        x5 = self.encoder4(x4)            # (B, 2048, H/32, W/32)

        d4 = self.decoder5(x5)            # (B, 256, H/16, W/16)
        d3 = self.decoder4(torch.cat([d4, x4], dim=1))  # + x4
        d2 = self.decoder3(torch.cat([d3, x3], dim=1))  # + x3
        d1 = self.decoder2(torch.cat([d2, x2], dim=1))  # + x2
        d0 = self.decoder1(torch.cat([d1, x0], dim=1))  # + x0

        out = self.double_conv(torch.cat([self.decoder_x0(x0), d0], dim=1))

        out = self.final_conv(out)
        return torch.sigmoid(out)


model = ResNetUNet(n_classes=1)
input_tensor = torch.randn(1, 3, 320, 480)  # B, C, H, W
output = model(input_tensor)
print(output.shape)  # [1, 1, 480, 320]



device = config['Device']
model = model.to(device)
# criterion = nn.BCEWithLogitsLoss()  # Use sigmoid inside loss
criterion = nn.BCELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)



def train_model(model,train_L, val_L, criterion, optimizer, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    train_loss_history = []
    val_loss_history = []

    for epoch in tqdm(range(num_epochs), desc='Training '):
        print(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_L
            else:
                model.eval()
                dataloader = val_L

            running_loss = 0.0

            for inputs, masks, _ in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # outputs = outputs.squeeze(1)  # [B, H, W]
                    # masks = masks.squeeze(1)
                    # print(outputs.size(), masks.shape)
                    loss = criterion(outputs, masks)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, "best_model.pth")

    print("Training complete. Best val loss: {:.4f}".format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history



trained_model, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=config['Num_epochs']
)


plt.figure(figsize=(8,5))
plt.plot(range(1,len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1,len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.show()



for im, m, _ in train_loader:
    outputs=model(im.to(device))
    out = outputs.detach().to('cpu')

    out[out > .5] = 1
    out[out < .6] = 0
    
    fig, axes = plt.subplots(8, 3, figsize=(12, 20))

    for i in range(8):
        # Original image
        axes[i, 0].imshow(im[i].permute(1,2,0))# cmap='gray' if images[i].ndim == 2 else None)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')

        # True mask
        axes[i, 1].imshow(m[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("True Mask")
        axes[i, 1].axis('off')

        # Predicted mask
        axes[i, 2].imshow(out[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("Pred Mask")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
    
    break



def dice_coefficient(pred_mask, true_mask, epsilon=1e-8):
    pred_mask = (pred_mask > 0.4).float()
    true_mask = true_mask.float()

    intersection = (pred_mask * true_mask).sum(dim=(-2, -1))
    union = pred_mask.sum(dim=(-2, -1)) + true_mask.sum(dim=(-2, -1))

    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice.numpy()  # return average over batch if batched


def evaluate(mdl, loader):
    mdl.eval()
    dice_coeff = []
    for im, m, _ in tqdm(loader, desc="Evaluating..."):
        out = mdl(im.to(device))
        out = out.detach().to('cpu')

        dice_coeff.extend(list(dice_coefficient(out, m) ))

    return dice_coeff


import torch
import gc

def free_gpu():
    # Clear cache
    torch.cuda.empty_cache()
    
    # Collect garbage
    gc.collect()
free_gpu()

def free_ram():
    # Collect garbage
    gc.collect()
free_ram()


coeff = evaluate(model, train_loader)

sum(coeff) / len(coeff)


# best_model_2 is best

best_model_wts = torch.load('/kaggle/input/model-carvana/best_model_2.pth')
model.load_state_dict(best_model_wts)


class Double_Conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.double_conv = Double_Conv(in_ch, mid_ch, mid_ch)
        self.upsample    = nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upsample(self.double_conv(x))

class ResNetUNet_v2(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        resnet = models.resnet50(pretrained=True)        
        # Freeze ResNet encoder
        # for param in resnet.parameters():
        #     param.requires_grad = False

        # Encoder
        self.base_layers = list(resnet.children())
        self.input_layer = nn.Sequential(*self.base_layers[:3])       # conv1 + bn1 + relu
        self.maxpool = self.base_layers[3]                            # maxpool
        self.encoder1 = self.base_layers[4]                           # layer1
        self.encoder2 = self.base_layers[5]                           # layer2
        self.encoder3 = self.base_layers[6]                           # layer3
        self.encoder4 = self.base_layers[7]                           # layer4

        # Decoder
        self.decoder5 = DecoderBlock(2048, 1024, 1024)
        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(1024, 512, 256)
        self.decoder2 = DecoderBlock(512, 256, 64)
        self.decoder1 = DecoderBlock(128, 64, 64)
        
        self.decoder_x0 = DecoderBlock(64, 64, 64)

        self.double_conv = Double_Conv(128, 64, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.input_layer(x)          # (B, 64, H/2, W/2)
        x1 = self.maxpool(x0)             # (B, 64, H/4, W/4)
        x2 = self.encoder1(x1)            # (B, 256, H/4, W/4)
        x3 = self.encoder2(x2)            # (B, 512, H/8, W/8)
        x4 = self.encoder3(x3)            # (B, 1024, H/16, W/16)
        x5 = self.encoder4(x4)            # (B, 2048, H/32, W/32)

        d4 = self.decoder5(x5)            # (B, 256, H/16, W/16)
        d3 = self.decoder4(torch.cat([d4, x4], dim=1))  # + x4
        d2 = self.decoder3(torch.cat([d3, x3], dim=1))  # + x3
        d1 = self.decoder2(torch.cat([d2, x2], dim=1))  # + x2
        d0 = self.decoder1(torch.cat([d1, x0], dim=1))  # + x0

        out = self.double_conv(torch.cat([self.decoder_x0(x0), d0], dim=1))

        out = self.final_conv(out)
        return torch.sigmoid(out)


model_v2 = ResNetUNet_v2().to(device)


models_files = os.listdir('/kaggle/input/model-carvana')
models_files.sort()

model_name, val_disc, train_disc = [], [], []
for file in tqdm(models_files):
    if 'v2' in file:
        mdl = model_v2
    else:
        mdl = model
    best_model_wts = torch.load(os.path.join('/kaggle/input/model-carvana', file))
    mdl.load_state_dict(best_model_wts)
    
    coeff_vals = evaluate(mdl, val_loader)

    val_disc.append(sum(coeff_vals) / len(coeff_vals))
    
    coeff_trains = evaluate(mdl, train_loader)

    train_disc.append(sum(coeff_trains) / len(coeff_trains))
    model_name.append(file)



# Create DataFrame
df = pd.DataFrame({
    'Model': model_name,
    'Train_Dice': train_disc,
    'Val_Dice': val_disc
})
df = df.sort_values(by='Model').reset_index(drop=True)

# Save as CSV
df.to_csv('/kaggle/working/model_dice_scores.csv', index=False)

df


unzip('/kaggle/input/carvana-image-masking-challenge/sample_submission.csv.zip')
unzip('/kaggle/input/carvana-image-masking-challenge/test.zip')
unzip('/kaggle/input/carvana-image-masking-challenge/train_masks.csv.zip')



import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class Test_dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = os.listdir(self.root)
        self.files.sort()
        print(len(self.files), " files found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        with Image.open(os.path.join(self.root, self.files[idx])) as img:  # Auto-close image after use

            W, H = img.size
            if W != config['Width'] or H != config['Height']:
                print(f"Size mismatch! Expected: {config['Width'], config['Height']}. Got : {W, H}.")
                return
            img = np.array(img)
            pad_width = ((0, 0), (0, 2)) if img.ndim == 2 else ((0, 0), (0, 2), (0, 0))  
            img = np.pad(img, pad_width, mode='edge')           

            img = cv2.resize(img, (config['Resize_W'], config['Resize_H']), interpolation=cv2.INTER_AREA)

            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, self.files[idx]


transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [C, H, W] with values in [0, 1]
])

test_dataset = Test_dataset('/kaggle/working/test', transform=transform)

test_loader = DataLoader(test_dataset, batch_size=config['Eval_batch_size'], shuffle=False, num_workers=config['Num_workers'])


best_model_wts = torch.load('/kaggle/input/model-carvana/best_model_2.pth')
model.load_state_dict(best_model_wts)
model = model.to(config['Device'])


free_gpu()
free_ram()


df = pd.DataFrame({'img': [], 'rle_mask': []})

model.eval()

for im, f in tqdm(test_loader, total=len(test_loader), desc='Evaluating...'):
    out=model(im.to(device))
    out = out.detach().to('cpu')

    
    # fig, axes = plt.subplots(out.size()[0], 2, figsize=(10, 20))

    for i in range(out.size()[0]):
        # Original image
        # axes[i, 0].imshow(im[i].permute(1,2,0))# cmap='gray' if images[i].ndim == 2 else None)
        # axes[i, 0].set_title(f[i])
        # axes[i, 0].axis('off')

        # Predicted mask
        mask = out[i].squeeze()
        mask = cv2.resize(mask.numpy(), (mask.size()[1] * 4, mask.size()[0] * 4), interpolation=cv2.INTER_CUBIC)
       
        mask[mask > .4] = 1
        mask[mask < .5] = 0
        mask = mask[:,:-2]

        # axes[i, 1].imshow(mask, cmap='gray')
        # axes[i, 1].set_title("Pred Mask")
        # axes[i, 1].axis('off')

        # print(mask.shape)
        mask = mask.flatten()
        mask = np.concatenate([[0], mask, [0]])
        
        changes = np.where(mask[1:] != mask[:-1])[0] + 1
        
        starts = changes[::2]
        lengths = changes[1::2] - starts

        rle = ' '.join(f'{s} {l}' for s, l in zip(starts, lengths))

        df.loc[len(df)] = [f[i], rle]

    # plt.tight_layout()
    # plt.show()

    # break


df


df.to_csv('/kaggle/working/carvana_submission_ar_titumir.csv', index=False)



df = pd.read_csv('/kaggle/working/train_masks.csv')
df



import numpy as np

def rle_encode(mask):
    """
    Encode a binary mask using run-length encoding, and count mask pixels.
    
    Parameters:
        mask (np.ndarray): 2D numpy array of binary mask (0 for background, 1 for mask)
    
    Returns:
        tuple:
            rle (str): RLE encoded string
            count (int): total number of mask pixels (1s)
    """
    # Count the number of 1’s in the mask
    count = int(mask.sum())
    
    # Flatten the mask column-wise (Fortran order)
    pixels = mask.flatten()
    # Pad with zeros at start/end to catch runs at edges
    pixels = np.concatenate([[0], pixels, [0]])
    
    # Find where pixel value changes
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    starts = changes[::2]           # run starts
    lengths = changes[1::2] - starts  # run lengths
    
    # Build RLE string
    rle = ' '.join(f'{s} {l}' for s, l in zip(starts, lengths))
    
    return rle, count



x = np.array([
    [0, 1, 7],
    [3, 4, 6],
    [0, 2, 0]
])

rle_encode(x)











import shutil
import os

# Check if the file exists and delete it
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print("File does not exist.")
        
# Check if directory exists and delete it
def delete_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Deleted directory: {dir_path}")
    else:
        print("Directory does not exist.")


# List all files and folders
def empty_dir(dir_path):
    items = os.listdir(dir_path)
    print("Contents to delete:")
    for item in items:
        print(item)
    
    # Delete everything
    for item in items:
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            delete_file(item_path)
        elif os.path.isdir(item_path):
            delete_dir(item_path)

    print(f"\nDirectory '{dir_path}' is now empty.")



file_path = '/kaggle/working'
empty_dir(file_path)


best_model_wts = copy.deepcopy(trained_model.state_dict())
torch.save(best_model_wts, "best_model_1_resnet_tuning.pth")

