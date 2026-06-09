%%capture
!pip install torchshow


from typing import List
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchshow as ts
import timm
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

df = pd.read_csv("/kaggle/input/animal-clef-2025/metadata.csv")

# select only salamanders
df = df.loc[df["dataset"] == "SalamanderID2025"]

# append dataset root dir to paths
df["path"] = "/kaggle/input/animal-clef-2025/"+ df["path"]


# Select 20 paths to create a simple dataset
PATHS = df["path"].sample(100, random_state=45).values

fig, ax = plt.subplots(10, 10, figsize=(10,10))
k = 0
for i in range(10):
    for j in range(10):
        img = Image.open(PATHS[k])
        ax[i,j].imshow(img)
        ax[i,j].set_xticklabels([])
        ax[i,j].set_yticklabels([])
        k += 1
plt.show()


# Label 100 images according to the above figure
# 0 : 0°
# 1 : 90°
# 2 : 180°
# 3 : 270°

LABELS = [
    1,1,1,3,1,1,0,1,3,3,
    2,1,1,1,3,3,1,1,2,1,
    0,0,1,3,1,0,1,1,2,1,
    2,1,0,3,0,1,1,3,1,1,
    1,1,3,1,1,3,3,3,1,0,
    2,1,1,1,3,3,1,1,2,1,
    2,1,1,1,1,3,1,1,1,1,
    3,3,1,2,3,0,0,1,3,0,
    0,3,1,3,1,1,1,2,1,1,
    1,1,1,3,2,3,0,1,3,1
]


import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

BATCH_SIZE = 16
LR = 4e-4
N_EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset Class
class RotClassifierDataset:
    def __init__(
        self,
        paths: List[str],
        labels: List[List[int]],
        transform=None,
        is_test=False
    ):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        label = self.labels[idx]

        if self.transform is not None:
            # Augment also labels
            if not self.is_test:
                r = np.random.randint(4)
                angle = 90*r  # random rotation angle in degrees
                image = image.rotate(angle, expand=True)
                label = (label + r) % 4
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.paths)


# Create Transforms [0.4, 0.36, 0.28] and [0.1, 0.08, 0.08] has computed offline
train_tr = T.Compose([
    T.ColorJitter(0.1,0.1,0.1),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.4, 0.36, 0.28], std=[0.1, 0.08, 0.08]),
])

val_tr = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.4, 0.36, 0.28], std=[0.1, 0.08, 0.08]),
])

# Create datasets and loaders
TRAIN_PATHS, VAL_PATHS, TRAIN_LABELS, VAL_LABELS = train_test_split(PATHS, LABELS, shuffle=True, stratify=LABELS, random_state=45)

ds_train = RotClassifierDataset(TRAIN_PATHS, TRAIN_LABELS, transform=train_tr)
ds_val = RotClassifierDataset(VAL_PATHS, VAL_LABELS, transform=val_tr, is_test=True)

loader_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
loader_val = torch.utils.data.DataLoader(ds_val, batch_size=2*BATCH_SIZE)


# Test dataset augmentations
data = [ds_train[i] for i in range(10)]
plt.figure()
ts.show([data[i][0] for i in range(10)], figsize=(10,2))
print([data[i][1] for i in range(10)])


# Create the model
model = timm.create_model("resnet18", pretrained=True, num_classes=0)
# The following layers helps rotation discovery
model.global_pool = torch.nn.Flatten(1)
model.fc = torch.nn.Linear(512*7*7,4)
model.eval()
model.to(DEVICE);


# Define optimizers and scheduler
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=N_EPOCHS, eta_min=1e-5
)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(N_EPOCHS):
    model.train()
    pbar = tqdm(total=len(loader_train))
    for iter, (images, labels) in enumerate(loader_train):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optim.step()
        optim.zero_grad()
        
        pbar.set_description(f"Loss {loss.item():.4f}")
        pbar.update(1)

    lr_sched.step()
    model.eval()
    
    y_true = []
    y_pred = []
    for iter, (images, labels) in enumerate(loader_val):
        images = images.to(DEVICE)
        y_true += list(labels)
        with torch.no_grad():
            logits = model(images)
    
        y_pred += list(torch.argmax(logits,1).cpu().numpy())
    
    print(f"epoch {epoch} Val Macro F1: {f1_score(y_true, y_pred, average='macro')}")
    




