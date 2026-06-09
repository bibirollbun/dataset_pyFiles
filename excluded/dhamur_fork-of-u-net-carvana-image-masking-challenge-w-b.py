import numpy as np 
import pandas as pd 

import os
import zipfile
from tqdm import tqdm
import random

import torch as t 
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, draw_segmentation_masks
from torchvision import transforms

import matplotlib.pyplot as plt

import requests

from PIL import Image


DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Current available device = {DEVICE}")


gpu_counts = t.cuda.device_count()
for i in range(gpu_counts):
    print(f"Device name = {t.cuda.get_device_name(i)}")
    print(f"Device properties = {t.cuda.get_device_properties(i)}")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


if not os.path.exists('/kaggle/working/dataset/train_masks'):
    print("Train mask does not exists... Extracting the data")
    with zipfile.ZipFile('/kaggle/input/carvana-image-masking-challenge/train_masks.zip') as z:
        z.extractall('/kaggle/working/dataset/')

if not os.path.exists('/kaggle/working/dataset/train'):
    print("Train does not exists... Extracting the data")
    with zipfile.ZipFile('/kaggle/input/carvana-image-masking-challenge/train.zip') as z:
        z.extractall('/kaggle/working/dataset/')


unet_parts_git = 'https://raw.githubusercontent.com/Dhamu785/U-Nets/refs/heads/main/01_U-net/PyTorch/01_Uygar%20Kurt/unet_parts.py'
unet_git = 'https://raw.githubusercontent.com/Dhamu785/U-Nets/refs/heads/main/01_U-net/PyTorch/01_Uygar%20Kurt/unet.py'
dataset_git = 'https://raw.githubusercontent.com/Dhamu785/U-Nets/refs/heads/main/01_U-net/PyTorch/01_Uygar%20Kurt/dataset.py'
unet_a = 'https://raw.githubusercontent.com/Dhamu785/U-Nets/refs/heads/main/01_U-net/PyTorch/01_Uygar%20Kurt/unet_a.py'

if os.path.exists('unet_parts.py'):
    print('Removing the file...')
    os.remove('unet_parts.py')
    print("Downloading unet_parts.py....")
    r = requests.get(unet_parts_git).content
    with open("unet_parts.py", 'wb') as f:
        f.write(r)
else:
    print("Downloading unet_parts.py....")
    r = requests.get(unet_parts_git).content
    with open("unet_parts.py", 'wb') as f:
        f.write(r)

if os.path.exists('unet_a.py'):
    print('Removing the file...')
    os.remove('unet_a.py')
    print("Downloading unet_a.py....")
    r = requests.get(unet_a).content
    with open("unet_a.py", 'wb') as f:
        f.write(r)
else:
    print("Downloading unet_a.py....")
    r = requests.get(unet_a).content
    with open("unet_a.py", 'wb') as f:
        f.write(r)



if os.path.exists('unet.py'):
    print('Removing the file...')
    os.remove('unet.py')
    print("Downloading unet.py....")
    r = requests.get(unet_git).content
    with open('unet.py', 'wb') as f:
        f.write(r)
else:
    print("Downloading unet.py....")
    r = requests.get(unet_git).content
    with open('unet.py', 'wb') as f:
        f.write(r)

if os.path.exists('dataset.py'):
    print('Removing the file...')
    os.remove('dataset.py')
    print("Downloading dataset.py....")
    r = requests.get(dataset_git).content
    with open('dataset.py', 'wb') as f:
        f.write(r)
else:
    print("Downloading dataset.py....")
    r = requests.get(dataset_git).content
    with open('dataset.py', 'wb') as f:
        f.write(r)


from dataset import seg_dataset
from unet_a import unet


# import torch.nn.functional as F


def loss_iou(y_pred, y_true, inf):
    if not inf:
        if not y_pred.requires_grad:
            raise ValueError("y_pred should have gradient tracking")
    
    device = y_pred.device
    # binary_pred = t.where(y_pred <= 0, t.zeros_like(y_pred, device=device, requires_grad=True), t.ones_like(y_pred, device=device, requires_grad=True))
    y_true = t.where(y_true <= 0, t.zeros_like(y_pred, device=device), t.ones_like(y_pred, device=device))
    
    # y_pred = F.sigmoid(y_pred)
    
    intersection = t.abs((y_pred.view((-1)) * y_true.view((-1))).sum().float())
    union = t.abs((y_pred.sum() + y_true.sum()).float())
    # print(f"intersection = {t.abs(intersection)}, union={union}")

    iou = (t.abs(intersection) + 1e-5) / ((union + 1e-5) - t.abs(intersection))
    iou_loss = 1 - iou
    return iou_loss


def acc_iou(y_pred, y_true, inf):
    ls = loss_iou(y_pred, y_true, inf)
    return 1-ls


img_path = '/kaggle/working/dataset/train'
img_msk_path = '/kaggle/working/dataset/train_masks'

img_name = os.listdir(img_path)[100]
msk_list = os.listdir(img_msk_path)
msk_idx = msk_list.index(img_name.split('.')[0]+'_mask.gif')
msk_name = msk_list[msk_idx]
# print(img_name, msk_name)
single_img_path = os.path.join(img_path, img_name)
single_msk_path = os.path.join(img_msk_path, msk_name)

transforms_pipe = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
img = Image.open(single_img_path).convert('RGB')
msk = Image.open(single_msk_path).convert('L')

img_transformed = transforms_pipe(img)
msk_transformed = transforms_pipe(msk)
# print(img_transformed.shape, msk_transformed.shape)
msk_transformed = t.where(msk_transformed <= 0, t.zeros_like(msk_transformed, device='cpu'), t.ones_like(msk_transformed, device='cpu'))
# print(t.unique(msk_transformed))
plt.subplot(1,2,1)
plt.imshow(img_transformed.permute(1, 2, 0).to('cpu'))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(msk_transformed[0].to('cpu'), cmap='gray')
plt.axis('off')
plt.show()


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 20
DATA_PATH = "/kaggle/working/dataset/"


def plot_img(model):
    img = img_transformed.unsqueeze(0).to(DEVICE)
    # print(img.shape)
    pred = model(img)
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(img_transformed.permute(1, 2, 0).to('cpu'))
    plt.axis('off')
    plt.title('Original img')
    plt.subplot(1,3,2)
    plt.imshow(msk_transformed[0].to('cpu'), cmap='gray')
    plt.title('Original mask')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(pred[0][0].detach().to('cpu').numpy(), cmap='gray')
    plt.title('Predicted mask')
    plt.axis('off')
    plt.show()


data = seg_dataset(DATA_PATH)
generator = t.Generator().manual_seed(42)
# train_dataset, val_dataset, test_dataset = random_split(dataset=data, lengths=(0.05, 0.02, 0.93), generator=generator)
train_dataset, val_dataset, test_dataset = random_split(dataset=data, lengths=(0.8, 0.2, 0.0), generator=generator)

print(f"{len(train_dataset) = } || {len(val_dataset) = } || {len(test_dataset) = }")


train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True)
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, True)

print(f"{len(train_dataloader) = } || {len(val_dataloader) = }")


import wandb
from kaggle_secrets import UserSecretsClient


user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb_key")


wandb.login(key=wandb_key, relogin=True)


wandb.init(
    project = "Autoencoders for car",
    config = {"learning_rate":0.0001, "architecture":"U-Net", "dataset": "cars-data", "epochs":20},
    name = '01_lr1e-5 all data-2'
)


model = unet(in_channel=3, num_classes=1).to(DEVICE)

wandb.watch(model, log="all", log_freq=5)

optimizer = optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
# loss = nn.BCEWithLogitsLoss()

history = {'loss_per_epoch_train':list(), 'loss_per_epoch_test':list(), 
           'acc_per_epoch_train':list(), 'acc_per_epoch_test':list()}

for epoch in range(EPOCHS):
    model.train()
    train_loss_per_batch = 0
    train_acc_per_batch = 0
    epoch_pbar = tqdm(range(len(train_dataloader)), desc="Batch processing",unit="batchs")
    for idx,batch in enumerate(train_dataloader):
        img = batch[0].float().to(DEVICE)
        mask = batch[1].float().to(DEVICE)

        # 1. Forward pass
        y_pred = model(img)
        # print(y_pred.min(), y_pred.max())
        # 2. Calculate the loss
        ls = loss_iou(y_pred, mask, False)
        # print("Loss = ", ls)

        acc = acc_iou(y_pred, mask, False)
        train_loss_per_batch += ls.item()
        train_acc_per_batch += acc.item()

        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        epoch_pbar.update(1)
    epoch_pbar.close()
    train_loss_per_batch /= idx+1
    train_acc_per_batch /= idx+1
    history['loss_per_epoch_train'].append(train_loss_per_batch)
    history['acc_per_epoch_train'].append(train_acc_per_batch)

    wandb.log({'train_loss':train_loss_per_batch})
    wandb.log({'train_acc':train_acc_per_batch})
    
    model.eval()
    test_loss_per_batch = 0
    test_acc_per_batch = 0
    with t.inference_mode():
        for idx, batch in enumerate(val_dataloader):
            img = batch[0].float().to(DEVICE)
            mask = batch[1].float().to(DEVICE)

            y_pred_test = model(img)
            test_ls = loss_iou(y_pred_test, mask, True)
            test_acc = acc_iou(y_pred_test, mask, True)

            test_loss_per_batch += test_ls.item()
            test_acc_per_batch += test_acc.item()
        
        test_loss_per_batch /= idx+1
        test_acc_per_batch /= idx+1
        plot_img(model)

    history['loss_per_epoch_test'].append(test_loss_per_batch)
    history['acc_per_epoch_test'].append(test_acc_per_batch)

    wandb.log({'test_loss':test_loss_per_batch})
    wandb.log({'test_acc':test_acc_per_batch})

    print(f"{epoch+1} / {EPOCHS} | train_loss = {train_loss_per_batch:.4f} | train_acc = {train_acc_per_batch:.4f} | test_loss = {test_loss_per_batch:.4f} | test_acc = {test_acc_per_batch:.4f}")

    wandb.log({'Epochs':epoch})
# wandb.finish()


t.save(model.state_dict(), 'model_wt-iou.pt')
# t.save(model, 'entire-model-iou.pt')


artifact = wandb.Artifact('Test-2', type='model')
artifact.add_file('/kaggle/working/model_wt-iou.pt')
wandb.log_artifact(artifact)


wandb.log_model('/kaggle/working/model_wt-iou.pt', 'model_wt-iou.pt')


wandb.finish()


plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.plot(history['acc_per_epoch_train'], color = 'orange')
plt.plot(history['acc_per_epoch_test'], color = 'green')
plt.legend(['train', 'test'])
plt.title("Accuracy")
plt.subplot(1,2,2)
plt.plot(history['loss_per_epoch_train'], color = 'orange')
plt.plot(history['loss_per_epoch_test'], color = 'green')
plt.legend(['train', 'test'])
plt.title("Loss")
plt.show()


# Model loading

## Type-1
## model_from_saved_1 = t.load('/kaggle/input/instance-segmentation-carvana-image-challenge/pytorch/default/1/Carvana Image_mdl.pt')
## Type-2
model_from_saved_2 = unet(in_channel=3, num_classes=1).to(DEVICE)
model_from_saved_2.load_state_dict(t.load('model_wt-iou.pt', weights_only=True, map_location=t.device(DEVICE)))


# Load the random test data

class get_data:
    def __init__(self, no_of_sample):
        self.train_img_path = '/kaggle/working/dataset/train'
        self.train_msk_path = '/kaggle/working/dataset/train_masks'
        test_files = os.listdir(self.train_img_path)
        random_100 = random.choices(test_files, k=no_of_sample)
        
        transformed_img = []
        transformed_msk = []
        
        for i in random_100:
            transformed_img.append(transforms_pipe(Image.open(os.path.join(self.train_img_path, i)).convert('RGB')))
            transformed_msk.append(transforms_pipe(Image.open(os.path.join(self.train_msk_path, i.split('.')[0]+"_mask.gif")).convert('L')))
            
        transformed_img = np.array(transformed_img)
        transformed_msk = np.array(transformed_msk)
        self.img = t.from_numpy(transformed_img)
        self.msk = t.from_numpy(transformed_msk)

    def mask_overlay(self, preds):
        preds = preds >= 0.5
        overlays = []
        for i in range(len(self.img)):
            overlays.append(draw_segmentation_masks(self.img[i], preds[i], colors='blue'))
        overlays = t.from_numpy(np.array(overlays))
        return overlays

    def make_grids(self, preds, overlay):
        predictions = make_grid(preds, 10, 1, pad_value = 2).moveaxis(0, 2)
        seg_overlay = make_grid(overlay, 10, 1, pad_value = 2).moveaxis(0, 2)
        images = make_grid(self.img, 10, 1, pad_value = 2).moveaxis(0, 2)
        masks = make_grid(self.msk, 10, 1, pad_value = 2).moveaxis(0, 2)

        return images, masks, predictions, seg_overlay


data = get_data(8)

with t.inference_mode():
    preds = model_from_saved_2(data.img.to(DEVICE))

seg_overlay = data.mask_overlay(preds)
images, masks, predictions, seg_overlay = data.make_grids(preds, seg_overlay)


plot_data = [images, masks, predictions, seg_overlay]
titles = ['Original images', 'Labels', 'Predictions', 'Prediction mask applied']
for i in range(len(plot_data)):
    plt.figure(figsize=(20, 30))
    plt.subplot(4,1,i+1)
    plt.imshow(plot_data[i].cpu().numpy())
    plt.title(label = titles[i], fontweight=10, fontstyle='normal')
    plt.axis('off')

plt.show()

