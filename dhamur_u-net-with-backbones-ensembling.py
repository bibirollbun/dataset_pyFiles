!pip install segmentation-models-pytorch -q


import segmentation_models_pytorch as smp
import torch as t
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor
from torch import optim

import os
import shutil
import zipfile
import gc
import random

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm


print("Available device = ", t.cuda.is_available())
print("CUDA version = ", t.version.cuda)
print("Cudnn version = ", t.backends.cudnn.version())
print("Cudnn is enabled? = ", t.backends.cudnn.enabled)


if not os.path.exists('/kaggle/working/dataset/train_masks'):
    print("Train mask does not exists... Extracting the data")
    with zipfile.ZipFile('/kaggle/input/carvana-image-masking-challenge/train_masks.zip') as z:
        z.extractall('/kaggle/working/dataset/')

if not os.path.exists('/kaggle/working/dataset/train'):
    print("Train does not exists... Extracting the data")
    with zipfile.ZipFile('/kaggle/input/carvana-image-masking-challenge/train.zip') as z:
        z.extractall('/kaggle/working/dataset/')

# if not os.path.exists('/kaggle/working/dataset/test'):
#     print("Test does not exists... Extracting the data")
#     with zipfile.ZipFile('/kaggle/input/carvana-image-masking-challenge/test.zip') as z:
#         z.extractall('/kaggle/working/dataset/')


img_path = '/kaggle/working/dataset/train'
msk_path = '/kaggle/working/dataset/train_masks'

img_lst = os.listdir(img_path)
msk_lst = os.listdir(msk_path)

print(f"Total images = {len(img_lst)}, Total masks = {len(msk_lst)}")


class load_process(Dataset):
    def __init__(self, img_path, msk_path, bb_name):
        self.img_path = img_path
        self.msk_path = msk_path
        self.process = smp.encoders.get_preprocessing_fn(bb_name)
        self.lst_img = os.listdir(img_path)

    def __len__(self):
        return len(self.lst_img)

    def __getitem__(self, index):
        s_img = Image.open(os.path.join(self.img_path, self.lst_img[index])).convert('RGB').resize((512,512), Image.LANCZOS)
        s_msk = Image.open(os.path.join(self.msk_path, self.lst_img[index].replace('.jpg', '_mask.gif'))).convert('L').resize((512,512), Image.LANCZOS)

        img_arr = np.array(s_img)
        msk_arr = np.array(s_msk)

        return t.tensor(self.process(img_arr), dtype=t.float32).permute(2,0,1), t.tensor(msk_arr, dtype=t.int64).unsqueeze(-1).permute(2,0,1)


def plot_predict(bb_name, model, idx=1001):
    processor = smp.encoders.get_preprocessing_fn(bb_name)
    img_data = Image.open(os.path.join(img_path, img_lst[idx])).convert('RGB').resize((512,512), Image.LANCZOS)
    arr_img = np.array(img_data)
    prcsd = t.tensor(processor(arr_img), dtype=t.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
    predicted = t.sigmoid(model(prcsd)).squeeze(0)
    msk_data = Image.open(os.path.join(msk_path, img_lst[idx].replace('.jpg', '_mask.gif'))).convert('L').resize((512,512), Image.LANCZOS)
    imgs = [prcsd.squeeze(0).permute(1,2,0).to('cpu').numpy(), np.array(msk_data), 
           (predicted.squeeze(0).to('cpu').detach().numpy() > 0.5).astype(int)]
    headings = ['Original', 'Mask', 'Prediction']
    plt.figure(figsize=(15,5))
    for i in range(len(imgs)):
        plt.subplot(1,3,i+1)
        if i >= 1:
            plt.imshow(imgs[i], cmap='gray')
        plt.imshow(imgs[i], cmap='gray')
        plt.title(headings[i])
        plt.axis('off')
    plt.show()
        


def train(model, data_loader, optimizer, ls_fn):
    model.to(DEVICE)
    model.train()
    loss_per_batch = 0
    progress_bar = tqdm(range(len(data_loader)), desc="Batch processing", unit="bath")
    for idx, batch in enumerate(data_loader):
        img = batch[0].to(device = DEVICE, dtype = t.float32)
        msk = batch[1].to(device = DEVICE, dtype = t.int64)
        # 1.Forward pass
        x_preds = model(img)
        # 2. Calculate the loss
        loss = ls_fn(x_preds, msk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)
        loss_per_batch += loss.item()
        
    progress_bar.close()
    loss_per_batch /= idx+1
    return loss_per_batch


def loss_iou(y_pred, y_true, inf=False):
    if not inf:
        if not y_pred.requires_grad:
            raise ValueError("y_pred should have gradient tracking")
    
    device = y_pred.device
    y_pred = t.sigmoid(y_pred)
    y_true = t.where(y_true <= 0, t.zeros_like(y_pred, device=device), t.ones_like(y_pred, device=device))
    intersection = t.abs((y_pred.view((-1)) * y_true.view((-1))).sum().float())
    union = t.abs((y_pred.sum() + y_true.sum()).float())

    iou = (t.abs(intersection) + 1e-5) / ((union + 1e-5) - t.abs(intersection))
    iou_loss = 1 - iou
    return iou_loss


DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Available device = {DEVICE}")
NUM_CLASS = 1
EPOCHS = 10
LEARNING_RATE = 1e-2
ENCODER_NAME = 'resnet101'
ENCODER_WEIGHT = 'imagenet'
BATCH_SIZE = 8

loss_dict = dict()


model1 = smp.Unet(encoder_name = ENCODER_NAME, encoder_weights=ENCODER_WEIGHT, in_channel=3, classes=NUM_CLASS)


# ls_fn = smp.losses.DiceLoss('binary')


optimizer = optim.Adam(params = model1.parameters(), lr=LEARNING_RATE)
resnet_data = load_process(img_path, msk_path, ENCODER_NAME)
subset = Subset(resnet_data, range(1000))
train_data = DataLoader(subset, BATCH_SIZE, True)

loss_now = []
for epoch in range(1, EPOCHS+1):
    ls = train(model1, train_data, optimizer, loss_iou)
    print(f"{epoch} / {EPOCHS} | train_loss = {ls:.2f}")
    plot_predict(ENCODER_NAME, model1)
    loss_now.append(ls)
loss_dict[ENCODER_NAME] = loss_now


t.save(model1.state_dict(), "U-Net_resnet101.pt")


del model1, train_data
gc.collect()
t.cuda.empty_cache()
t.cuda.reset_peak_memory_stats()


ENCODER_NAME = 'efficientnet-b3'


model2 = smp.Unet(encoder_name = ENCODER_NAME, encoder_weights=ENCODER_WEIGHT, in_channel=3, classes=NUM_CLASS)


optimizer = optim.Adam(params = model2.parameters(), lr=LEARNING_RATE)
efficientnet_data = load_process(img_path, msk_path, ENCODER_NAME)
subset = Subset(efficientnet_data, range(1000))
train_data = DataLoader(subset, BATCH_SIZE, True)
loss_now = []

for epoch in range(1, EPOCHS+1):
    ls = train(model2, train_data, optimizer, loss_iou)
    print(f"{epoch} / {EPOCHS} | train_loss = {ls:.2f}")
    plot_predict(ENCODER_NAME, model2)
    loss_now.append(ls)
    
loss_dict[ENCODER_NAME] = loss_now


t.save(model2.state_dict(), 'U-net_efficientnet.pt')


del model2, train_data
gc.collect()
t.cuda.empty_cache()
t.cuda.reset_peak_memory_stats()


ENCODER_NAME = 'inceptionv4'


model3 =smp.Unet(encoder_name = ENCODER_NAME, encoder_weights=ENCODER_WEIGHT, in_channel=3, classes=NUM_CLASS)


optimizer = optim.Adam(params = model3.parameters(), lr=LEARNING_RATE)
inceptionv4_data = load_process(img_path, msk_path, ENCODER_NAME)
subset = Subset(inceptionv4_data, range(1000))
train_data = DataLoader(subset, BATCH_SIZE, True)
loss_now = []

for epoch in range(1, EPOCHS+1):
    ls = train(model3, train_data, optimizer, loss_iou)
    print(f"{epoch} / {EPOCHS} | train_loss = {ls:.2f}")
    plot_predict(ENCODER_NAME, model3)
    loss_now.append(ls)
    
loss_dict[ENCODER_NAME] = loss_now


t.save(model3.state_dict(), 'U-Net_inception.pt')


del model3, train_data
gc.collect()
t.cuda.empty_cache()
t.cuda.reset_peak_memory_stats()


for i,j in loss_dict.items():
    plt.plot(j, label=i)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


if os.path.exists('/kaggle/working/U-Net_inception.pt'):
    inception_path = '/kaggle/working/U-Net_inception.pt'
else:
    inception_path = '/kaggle/input/instance-segmentation-carvana-image-challenge/pytorch/default/2/U-Net_inception.pt'
    
if os.path.exists('/kaggle/working/U-Net_resnet101.pt'):
    resnet_path = '/kaggle/working/U-Net_resnet101.pt'
else:
    resnet_path = '/kaggle/input/instance-segmentation-carvana-image-challenge/pytorch/default/2/U-Net_resnet101.pt'
    
if os.path.exists('/kaggle/working/U-net_efficientnet.pt'):
    efficientnet_path = '/kaggle/working/U-net_efficientnet.pt'
else:
    efficientnet_path = '/kaggle/input/instance-segmentation-carvana-image-challenge/pytorch/default/2/U-net_efficientnet.pt'

inception_mdl = smp.Unet(encoder_name = 'inceptionv4', encoder_weights=ENCODER_WEIGHT, in_channel=3, classes=NUM_CLASS).to(DEVICE)
resnet_mdl = smp.Unet(encoder_name = 'resnet101', encoder_weights=ENCODER_WEIGHT, in_channel=3, classes=NUM_CLASS).to(DEVICE)
efficientnet_mdl = smp.Unet(encoder_name = 'efficientnet-b3', encoder_weights=ENCODER_WEIGHT, in_channel=3, classes=NUM_CLASS).to(DEVICE)


inception_mdl.load_state_dict(t.load(inception_path, weights_only=True, map_location=t.device(DEVICE)))
resnet_mdl.load_state_dict(t.load(resnet_path, weights_only=True, map_location=t.device(DEVICE)))
efficientnet_mdl.load_state_dict(t.load(efficientnet_path, weights_only=True, map_location=t.device(DEVICE)))


models = [inception_mdl, resnet_mdl, efficientnet_mdl]
backbones = ['inceptionv4', 'resnet101', 'efficientnet-b3']
data_set = [inceptionv4_data, resnet_data, efficientnet_data]

rand_idx = random.randint(1000, 5000)
model_loss = dict()
predictions = []
masks = []
for i in range(len(models)):
    models[i].eval()
    img = data_set[i][rand_idx][0].to(device=DEVICE, dtype=t.float32).unsqueeze(0)
    msk = data_set[i][rand_idx][1].to(device=DEVICE, dtype=t.int64)
    masks.append(msk)
    with t.inference_mode():
        pred = models[i](img)
        predictions.append(pred.cpu().numpy())
        loss = loss_iou(pred, msk, True)
    plot_predict(backbones[i], models[i], rand_idx)
    model_loss[backbones[i]] = [loss.item()]
    print(f"Backbone = {backbones[i]}, Loss = {loss.item()}")


pd.DataFrame(model_loss)


weighted_summary = dict()
for i in range(4):
    for j in range(4):
        for k in range(4):
            weights = [i/10, j/10, k/10]
            weighted_preds = np.tensordot(weights, predictions, axes=(0,0))
            L = loss_iou(t.from_numpy(weighted_preds), masks[0].to('cpu'), True).item()
            weighted_summary[str(i/10)+"_"+str(j/10)+"_"+str(k/10)] = round(L, 6)


summary_wgt = np.array([])
summary_ls = np.array([])
for i,j in weighted_summary.items():
    summary_wgt = np.append(summary_wgt, [i])
    summary_ls = np.append(summary_ls, [j*100])


df_wgt = pd.DataFrame(data=summary_wgt.reshape((8,8)))
df_wgt


summary_ls.min(), summary_ls.max()


df_ls = pd.DataFrame(data=summary_ls.reshape((8,8)))
df_ls.style.background_gradient(cmap='Greens', low=0, high=0)


df_wgt[(df_ls <= 2.5) & (df_ls >= 2)]

