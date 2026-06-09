import os
import time
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Dice
import seaborn as sns


workdir = '/kaggle/input/severstal-steel-defect-detection/'
train_lbls = pd.read_csv(f'{workdir}train.csv')
val_lbls = pd.read_csv(f'{workdir}sample_submission.csv')

image_train_dir = f'{workdir}/train_images/'
image_val_dir = f'{workdir}/test_images/'


train_lbls.head()


train_lbls.info()


sns.histplot(train_lbls, x='ClassId')


weights = torch.zeros(5)
for cls in train_lbls['ClassId'].unique():
    _cls = train_lbls['ClassId'] == cls
    weights[cls] = _cls.sum() 
weights =   F.sigmoid(weights.max() / weights)
weights[0] = weights.min()
weights


def get_sample(index, workdir, labels):
    sample = labels.iloc[index]
    sample_name = sample['ImageId']
    sample_class = sample['ClassId']
    sample_coords = np.array(sample['EncodedPixels'].split(), dtype=int)
    sample_img = cv.imread(f'{workdir}{sample_name}')
    
    return sample_img, sample_coords, sample_class
    
def resize(image, shape, interpolation=cv.INTER_AREA):
    resized_image = cv.resize(image, shape, interpolation=interpolation)
    return resized_image
    

def make_mask(index, workdir, labels, n_classes=4):
    sample_img, sample_coords, sample_class = get_sample(index, workdir, labels)
    sample_mask = np.zeros_like(sample_img[:, :, 0].flatten(), dtype=np.int8)
    coord = sample_coords[::2]
    run_len = sample_coords[1::2]
    
    for _coord, _len in zip(coord, run_len):
        sample_mask[_coord: _coord+_len] = 1
    
    _mask = sample_mask.reshape(sample_img.shape[:2], order='F')
    sample_mask = np.zeros((n_classes+1, sample_img.shape[0],sample_img.shape[1]))
    sample_mask[sample_class] = _mask
    sample_mask[0] = 1 - _mask
    
    return sample_img, sample_mask, sample_class


sample_img, sample_mask, sample_class = make_mask(4, image_train_dir, train_lbls)

fig, axs = plt.subplots(2)

axs[0].imshow(sample_img)
axs[1].imshow(sample_mask[sample_class])
print(f"Defect class: {sample_class}")


class SteelDefectDataset(Dataset):
    def __init__(self, datalist, image_dir, it_has_mask=True, shape=(512, 128)):
        
        self.image_dir = image_dir
        self.datalist = datalist
        self.mask = it_has_mask
        classes = datalist['ClassId'].unique()
        self.n_class = len(classes)
        if shape:
            self.shape = shape
        else:
            self.shape = False
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if self.mask:
            image, mask, cls = make_mask(idx, self.image_dir, self.datalist)
            if self.shape:
                image = resize(image, self.shape)
                _m = np.zeros((5, self.shape[1], self.shape[0]))
                _m[cls] = resize(mask[cls].astype(float), self.shape)
                mask = _m
                mask[0] = 1 - _m[cls]
            return image, mask, cls
            
        else:
            image, _, cls = get_sample(idx, self.image_dir, self.datalist)
            if self.shape:
                image = resize(image, self.shape)
            
            return image, self.datalist.iloc[idx]['ImageId']

    def show(self, idx):
        if self.mask:
            sample_img, sample_mask, cls = self[idx]
            fig, axs = plt.subplots(2)
            axs[0].imshow(sample_img)
            axs[1].imshow(sample_mask[cls])
        else:
            sample_img, cls = self[idx]
            plt.imshow(sample_img)
        print(f"Defect class: {cls}")


torch.manual_seed(42)

train_ds = SteelDefectDataset(train_lbls, image_train_dir)
validate_ds = SteelDefectDataset(val_lbls, image_val_dir, it_has_mask=False)

train_size = int(len(train_ds)*0.9)
test_size = len(train_ds) - train_size

train_set, test_set = random_split(train_ds, [train_size, test_size])

data_loader_train = DataLoader(train_set, batch_size=8, shuffle=False)
data_loader_test = DataLoader(test_set, batch_size=1, shuffle=False)
data_loader_validate = DataLoader(validate_ds, batch_size=1, shuffle=False)


train_ds.show(1)


validate_ds.show(0)


class Conv2DLayer(nn.Module):
    def __init__(self, in_layers, out_layers, pooling_stride=2):
        super().__init__()
        
        self.stack = nn.Sequential(
                nn.Conv2d(in_layers, out_layers, kernel_size=3, padding=1, stride=1), 
                nn.ReLU(inplace=True),
                nn.Conv2d(out_layers, out_layers, kernel_size=3, padding=1, stride=1), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pooling_stride, stride=pooling_stride)
            )

    def forward(self, x):
        return self.stack(x)


class UpScaleX2(nn.Module):
    def __init__(self, in_layers, out_layers, skip_con_layers=0, padding=None):
        super().__init__()
        self.padding = padding
        self.transpose = nn.ConvTranspose2d(in_layers, in_layers, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upscale_stack = nn.Sequential(
            nn.Conv2d(in_layers + skip_con_layers, out_layers, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_layers, out_layers, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2=None):

        x1 = self.transpose(x1)

        if self.padding is not None:
            x1 = F.pad(x1, self.padding)

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        y = self.upscale_stack(x)

        return y



class DefectDetectionNet(nn.Module):
    def __init__(self, init_count=64, color_channels=3, n_classes=2):
        """
        UNet-like CNN model
        
        :param init_count: Initial number of filters. Ordinary UNet has 64 filters for the first layer, (int)
        :param color_channels:  Number of color channels. 3 for RGB, 1 for grayscale, (int)
        :param n_classes:  Count of classes, (int)
        :param dropout_rate:  Dropout rate, (float)
        """
        super().__init__()
        
        self.ch_init = init_count

        # Encoder:
        self.l_in = Conv2DLayer(color_channels, self.ch_init)
        self.l_1 = Conv2DLayer(self.ch_init, 2 * self.ch_init)
        self.l_2 = Conv2DLayer(2 * self.ch_init, 4 * self.ch_init)
        self.l_3 = Conv2DLayer(4 * self.ch_init, 8 * self.ch_init)
        self.l_4 = Conv2DLayer(8 * self.ch_init, 16 * self.ch_init)

        # Decoder:
        self.l_trans_5 = UpScaleX2(16 * self.ch_init, 8 * self.ch_init, skip_con_layers=8 * self.ch_init)
        self.l_trans_6 = UpScaleX2(8 * self.ch_init, 4 * self.ch_init, skip_con_layers=4 * self.ch_init)
        self.l_trans_7 = UpScaleX2(4 * self.ch_init, 2 * self.ch_init, skip_con_layers=2 * self.ch_init)
        self.l_trans_8 = UpScaleX2(2 * self.ch_init, self.ch_init, skip_con_layers=1 * self.ch_init)
        self.l_trans_9 = UpScaleX2(self.ch_init, self.ch_init//2, skip_con_layers=3)

        self.final_conv = nn.Conv2d(self.ch_init//2, n_classes, kernel_size=1, stride=1)


    def forward(self, x, training=False):
        
        y1 = self.l_in(x)
        y2 = self.l_1(y1)
        y3 = self.l_2(y2)
        y4 = self.l_3(y3)
        y = self.l_4(y4)
        
        y = self.l_trans_5(y, y4)
        y = self.l_trans_6(y, y3)
        y = self.l_trans_7(y, y2)
        y = self.l_trans_8(y, y1)
        y = self.l_trans_9(y, x)
        y = self.final_conv(y)
        return F.relu(y)

    def predict(self, x, threshold=0.5):
        
        logits = self(x)
        prediction = F.softmax(logits, dim=1)
        return (prediction > threshold).int()


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

model = DefectDetectionNet(64, 3, train_ds.n_class+1).to(device)
metric = Dice(num_classes=5, ignore_index=0).to(device)
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
epochs = 5


def train(dataset, model, loss_fn, optimizer,  device, b_size=10):
    size = len(dataset.dataset)
    model.train()
    losses = []
    for batch, (X, y, c) in enumerate(dataset):
        torch.cuda.empty_cache()
        print(batch, end='\r')
        X, y = X.to(device), y.to(device)

        logits = model(X.float().permute(0,3,1,2))
        loss = loss_fn(logits, y)
        losses.append(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % b_size == 0:
            loss, current = loss.item(), (batch) * len(X)
            print(f"{time.ctime()} - loss: {torch.mean(torch.tensor(losses))}  [{current:>5d}/{size:>5d}]")
            losses = []

def test(dataset, model, metric, device):
    size = len(dataset.dataset)
    model.eval()
    metric_val, correct = 0, 0
    with torch.no_grad():
        for X, y, c in dataset:
            X, y = X.to(device), y.to(device)
            pred = model.predict(X.float().permute(0,3,1,2))
            m_el = metric(pred[0], y[0].int())
            metric_val += m_el.item()
    metric_val /= size
    print(f"Test Error: Avg Metric value: {metric_val} \n")
    torch.cuda.empty_cache()
    return metric_val


metrics = []

for epoch in range(epochs):
    print(f'Epoch: {epoch+1} of {epochs}')
    train(data_loader_train, model, loss_fn, optimizer, device=device, b_size=150)
    metrics.append(test(data_loader_test, model, metric, device=device))
plt.plot(metrics)
plt.grid()


fig, axs = plt.subplots(5,3)
axs[0, 0].set_title("Image")
axs[0, 1].set_title("Target")
axs[0, 2].set_title("Prediction")

for i, batch in enumerate(data_loader_test):
    X, y, c = batch
    c = c[0]
    axs[i, 0].imshow(X[0])
    X = X.to(device)
    with torch.no_grad():
        y_hat = model.predict(X.permute(0, 3, 1,2).float()).cpu()
    axs[i, 1].imshow(y[0][c])
    axs[i, 2].imshow(y_hat[0][c])
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')
    axs[i, 2].axis('off')
    if i == 4:
        break


def draw_contures(img, mask, cls):
    colors = ((255,0,0), (0,255,0),(0,255,0),(255,255,0))

    contours, hierarchy = cv.findContours(mask[cls].astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img, contours, -1, colors[c-1], 2)
    return img

fig, axs = plt.subplots(5)
axs[0].set_title("Test set")

for i, batch in enumerate(data_loader_test):
    X, y, c = batch
    c = c[0]
    X = X.to(device)
    with torch.no_grad():
        y_hat = model.predict(X.permute(0, 3, 1,2).float()).cpu()[0]
    classes = np.where(np.array([Y.max() for Y in y_hat.numpy()])>0)[0]
    img = X.cpu().numpy()[0]
    for cls in classes[1:]:
        img = draw_contures(img, y_hat.numpy(), c)
        
    axs[i].imshow(img)
    axs[i].axis('off')
    
    if i == 4:
        break


from statistics import mode
fig, axs = plt.subplots(5,2)
axs[0,0].set_title("Validation set")

for i, batch in enumerate(data_loader_validate):
    X, c = batch
    
    X = X.to(device)
    with torch.no_grad():
        y_hat = model.predict(X.permute(0, 3, 1,2).float()).cpu()[0]
    if y_hat[1:].max()>0:
        c = mode(np.where(y_hat[1:]>0)[0]) + 1
        axs[i%5, i//5].imshow(draw_contures(X.cpu().numpy()[0], y_hat.numpy(), c))
        
    else:
        axs[i%5, i//5].imshow(X.cpu().numpy()[0])
        
    axs[i%5, i//5].axis('off')
    
    if i == 9:
        break


def encode_pixels(mask, cls):
    idxs = np.sort(np.where(mask.reshape(-1, order='F')==cls))[0]
    if len(idxs) > 0:
        n = 1
        value = idxs[0]
        encoded_pixels = str()
        for i in range(1, len(idxs)):
            if (idxs[i] - idxs[i-1]) == 1:
                n += 1
            else:
                encoded_pixels += f'{value} {n} '
                
                value = idxs[i]
                n = 1
        return encoded_pixels
    else:
        return ''


ImageId_ClassId = []
EncodedPixels = []
with torch.no_grad():
    for X, c in data_loader_validate:
        X= X.to(device)
        pred = model.predict(X.float().permute(0,3,1,2))
        y_hat = torch.argmax(pred[:, :-1], dim=1)
        y_hat = y_hat.cpu().numpy()
        y_hat = resize(y_hat[0].astype(float), (256, 1600), interpolation=cv.INTER_NEAREST)
        name = c[0]
        
        classes = (1,2,3,4)
        for cls in classes:
            ImageId_ClassId.append(f'{name}_{int(cls)}')
            EncodedPixels.append(encode_pixels(y_hat, int(cls)))


out = pd.DataFrame({'ImageId_ClassId':ImageId_ClassId, 'EncodedPixels':EncodedPixels})


out.to_csv("submission.csv", index=False)
out.head()


out

