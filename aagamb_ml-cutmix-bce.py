import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as tfms
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import cv2 
import torch
from PIL import Image
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm
import sys; 
sys.path.insert(0,'../input/timm-nfnet')
import timm
from sklearn.metrics import f1_score, average_precision_score
import random


import warnings
warnings.filterwarnings("ignore")

base = "/kaggle/input/hpa-single-cell-image-classification"
# base = "/kaggle/input/hpa-single-small-dataset/"

TRAIN_DF_PATH = base + "train.csv"
TRAIN_IMG_PATH = base + "train"
TEST_IMG_PATH = base + "test"
SAMPLE_SUB = base + "sample_submission.csv"
CELL_LABEL = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles",
5:  "Nuclear bodies",
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus",
8:  "Intermediate filaments",
9:  "Actin filaments", 
10: "Microtubules",
11:  "Mitotic spindle",
12:  "Centrosome",   
13:  "Plasma membrane",
14:  "Mitochondria",   
15:  "Aggresome",
16:  "Cytosol",   
17:  "Vesicles and punctate cytosolic patterns",   
18:  "Negative"
}

train_df = pd.read_csv(TRAIN_DF_PATH)
train_df['label_count'] = train_df['Label'].apply(lambda x: len(x.split("|")))
# train_df.head()

#hyperparameters
CLASS = 19
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
RESIZE = 256
DEVICE = torch.device('cuda') if torch.cuda.is_available() \
         else torch.device('cpu')
PATH = base
TRAIN_DIR = PATH + 'train/'
TEST_DIR = PATH + 'test/'

#imagenet transform
img_tfms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.RandomHorizontalFlip(),
    tfms.RandomVerticalFlip(),
    tfms.RandomRotation(20),
    tfms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    tfms.ToTensor(),
    tfms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

DEVICE

class HPADataset(Dataset):
    def __init__(self, csv_path, ids, labels=None, resize=None, transforms=None):
        self.csv_path = csv_path
        self.ids = ids
        self.labels = labels  # Can be None for test set
        self.resize = resize
        self.transforms = transforms
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.csv_path, img_id + '_green.png')
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        if self.resize:
            image = cv2.resize(image, (self.resize, self.resize))

        # Ensure image is uint8 (for transforms compatibility)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        if self.transforms:
            image = self.transforms(image)

        if self.labels is not None:  # Training/Validation mode
            label = self.labels[index]
            return image, torch.tensor(label, dtype=torch.float32)
        else:  # Test mode
            return image, img_id
#model
class NFNet(nn.Module):
    def __init__(self,output_features, model_name = 'nfnet_f1', pertrained=True):
        super(NFNet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pertrained)
        self.model.head.fc = nn.Sequential(nn.Linear(self.model.head.fc.in_features, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, output_features))
        
    def forward(self, x):
        x = self.model(x)
        return x

class CNNet(nn.Module):
    def __init__(self, input_features, output_features):
        super(CNNet, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Sequential(nn.Linear(input_features, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, output_features))

    def forward(self, x):
        out = self.model(x)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_term = self.alpha * (1 - pt) ** self.gamma
        loss = focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

import random

class CutMixDataset(Dataset):
    def __init__(self, dataset, num_classes=19, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_classes = num_classes
        self.beta = beta
        self.prob = prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image1, label1 = self.dataset[idx]

        # Apply CutMix with some probability
        if random.random() < self.prob:
            # Select random second sample
            idx2 = random.randint(0, len(self.dataset) - 1)
            image2, label2 = self.dataset[idx2]

            lam = np.random.beta(self.beta, self.beta)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image1.size()[1:], lam)
            image1[:, bby1:bby2, bbx1:bbx2] = image2[:, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image1.size()[-1] * image1.size()[-2]))
            label = label1 * lam + label2 * (1. - lam)
            return image1, label

        else:
            return image1, label1

    def rand_bbox(self, size, lam):
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


print("cell run")




model = NFNet(CLASS)
model = model.to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_df = pd.read_csv(PATH + "train.csv")
train_df = train_df.sample(frac=1).reset_index(drop=True)  # Shuffle
from sklearn.preprocessing import MultiLabelBinarizer

# Split string labels into list of ints
train_df['Label'] = train_df['Label'].apply(lambda x: list(map(int, x.split("|"))))

# Fit binarizer
mlb = MultiLabelBinarizer(classes=range(CLASS))  # Ensure consistent order
y_bin = mlb.fit_transform(train_df['Label'])

# Then pass y_bin to your dataset
X_train = train_df['ID'].values
X_ds = HPADataset(TRAIN_DIR, X_train, y_bin, RESIZE, img_tfms)

# 80-20 split
total_size = len(X_ds)
val_size = int(0.2 * total_size)
train_size = total_size - val_size
train_ds, valid_ds = random_split(X_ds, [train_size, val_size])


cutmix_train_ds = CutMixDataset(train_ds, num_classes=CLASS, beta=1.0, prob=0.5)
train_dl = DataLoader(cutmix_train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,shuffle=True)

from sklearn.metrics import f1_score, average_precision_score, multilabel_confusion_matrix
import seaborn as sns

def evaluate_model(model, valid_dl, threshold=0.5, device=DEVICE):
    model.eval()
    all_preds, all_trues = [], []
    images_sample, preds_sample, labels_sample = [], [], []

    with torch.no_grad():
        for img, lbl in valid_dl:
            img = img.to(device)
            lbl = lbl.to(device)
            out = torch.sigmoid(model(img.float()))

            all_preds.append(out.cpu().numpy())
            all_trues.append(lbl.cpu().numpy())

            if len(images_sample) < 8:  # Save samples for plotting
                images_sample.extend(img.cpu().numpy())
                preds_sample.extend(out.cpu().numpy())
                labels_sample.extend(lbl.cpu().numpy())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)

    preds_bin = (preds > threshold).astype(int)

    f1 = f1_score(trues, preds_bin, average='macro')
    map_score = average_precision_score(trues, preds, average='macro')
    conf_matrix = multilabel_confusion_matrix(trues, preds_bin)

    print(f"F1 Score: {f1:.4f}", end = "\t")
    print(f"mAP: {map_score:.4f}")

print("cell done")


#train loop
loss_hist = []
for epoch in tqdm(range(EPOCHS)):
    losses = []
    model = model.train()
    for batch_idx, (image, label) in enumerate(train_dl):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        output = model(image.float())
        loss = loss_fn(output, label)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_hist.append(sum(losses)/len(losses))
    
    print(f"epoch: {epoch} loss:{sum(losses)/len(losses):.4f}", end="\t")

    evaluate_model(model, valid_dl)


plt.figure(figsize=(15, 8))
plt.title('Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(loss_hist)
plt.show()

