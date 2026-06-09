import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models import resnet18,ResNet18_Weights,resnet50,ResNet50_Weights
from torch.utils.data import Dataset,DataLoader
import random

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


train_path = r'/kaggle/input/hackathon-de-ia-unsch/train_set.npz'
test_path = r'/kaggle/input/hackathon-de-ia-unsch/test_set.npz'
val_path = r'/kaggle/input/hackathon-de-ia-unsch/val_set.npz'

##returns dictionary like objects with keys 'images','labels'
train_file = np.load(train_path)
val_file = np.load(val_path)
test_file = np.load(test_path)

keys = list(train_file.keys())

print(
    f'Train Images: {train_file[keys[0]].shape}'
    '\n'
    f'Val Images: {val_file[keys[0]].shape}'
    '\n'
    f'Test Images: {test_file[keys[0]].shape}'
)


fig,ax = plt.subplots(1,2)
train_labels,val_labels = train_file['labels'],val_file['labels']
ax[0].hist(train_labels,bins=np.unique(train_labels).shape[0])
ax[0].set_title('Training Target Distribution')

ax[1].hist(val_labels,bins=np.unique(val_labels).shape[0])
ax[1].set_title('Validation Target Distribution')
plt.tight_layout()


examples = 15
select_idx = np.random.randint(0,train_file['images'].shape[0],size=examples)
selected_images = train_file['images'][select_idx,:,:]
selected_labels = train_file['labels'][select_idx]

fig,ax = plt.subplots(3,5)
ax = ax.flatten()

for idx in range(examples):
    ax[idx].imshow(selected_images[idx])
    ax[idx].set_title(f'Label: {selected_labels[idx].item()}')
plt.tight_layout()


class MNIST(Dataset):
    def __init__(self,file,transform,keys=keys):
        super().__init__()
        self.images = file[keys[0]]
        self.labels = file[keys[1]].squeeze() ##convert to one dimensional list
        self.transform = transform
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self,idx):
        img_array = torch.tensor(self.images[idx],dtype=torch.float).div(255.0).unsqueeze(0)
        label = self.labels[idx] ## 1x1

        if self.transform:
            img = self.transform(img_array)

        return img,torch.tensor(label,dtype=torch.long) ##shape (1,28,28)
        
class MNISTTest(Dataset):
    def __init__(self,file,transform):
        super().__init__()
        self.images = file['images']
        self.ids = file['ids'].squeeze() ##convert to one dimensional list
        self.transform = transform
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self,idx):
        img_array = torch.tensor(self.images[idx],dtype=torch.float).div(255.0).unsqueeze(0)
        ids = self.ids[idx] ## 1x1

        if self.transform:
            img = self.transform(img_array)

        return img,torch.tensor(ids,dtype=torch.long) ##shape (1,28,28)


def get_loaders(data, transforms, batches, shuffle, is_test=False):
    if not is_test:
        loaders = []
        for idx in range(len(data)):
            dataset = MNIST(data[idx], transforms[idx])
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=batches[idx],
                    shuffle=shuffle[idx],
                    num_workers=0,
                    pin_memory=True
                )
            )
        return loaders
    else:
        dataset = MNISTTest(data, transforms)
        return DataLoader(
            dataset,
            batch_size=batches,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )



train_tfms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.Normalize( mean=[0.485],std=[0.229])
])

val_tfms = transforms.Compose([
    transforms.Normalize(    mean=[0.485],
    std=[0.229]) #imagenet transforms
])

train_loader,val_loader = get_loaders([train_file,val_file],[train_tfms,val_tfms],[32,64],[True,False])
test_loader = get_loaders(test_file,val_tfms,64,None,is_test=True)


def get_model(num_classes=11):
    model = resnet18(weights=ResNet18_Weights.DEFAULT,progress=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features,512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256,num_classes)
    )
    return model

def get_model50(num_classes=11):
    model = resnet50(weights=ResNet50_Weights.DEFAULT,progress=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features,512),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(512,256),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(256,num_classes)
    # )
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


import sklearn
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_cosine_schedule_with_warmup
epochs = 20
lr_base=2e-4 ##chnage from 1e-4
lr_head = 3e-4
patience=5
trigger=0

model = get_model50().to(DEVICE)
weights = compute_class_weight('balanced',classes=(np.unique(train_file['labels'])),y=list(train_file['labels'].squeeze()))
weights = torch.tensor(weights,device=DEVICE,dtype=torch.float)

base_params = [p for n,p in model.named_parameters() if n!= 'fc']
head_params = [p for n,p in model.named_parameters() if n=='fc']
optimizer = torch.optim.AdamW([
    {'params':base_params,'lr':lr_base},
    {'params':head_params,'lr':lr_head}
],weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(weight=weights)
schedule = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps = int(0.04*epochs*len(train_loader)),
                                          num_training_steps=int(epochs*len(train_loader)))
training_loss,validation_loss,accuracy = [],[],[]


import numpy as np
import torch
from tqdm.auto import tqdm, trange
from sklearn.metrics import accuracy_score
from torch.amp import autocast,GradScaler
scaler = GradScaler(DEVICE)

best_val_loss = float("inf")
epochs_no_improve = 0
save_path = f'/kaggle/working/resnet50_4.pth'
for epoch in trange(epochs, desc="EPOCHS"):
    model.train()
    train_losses = []

    train_bar = tqdm(train_loader, desc="TRAIN", leave=False)
    for img, label in train_bar:
        img = img.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(img)
        loss = criterion(outputs, label)

        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        schedule.step()
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = np.mean(train_losses)

    # ------------------ VALIDATION ------------------
    model.eval()
    val_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="VAL", leave=False)
        for img, label in val_bar:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            outputs = model(img)
            loss = criterion(outputs, label)

            val_losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    avg_val_loss = np.mean(val_losses)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    val_acc = accuracy_score(all_labels, all_preds)

    # ------------------ LOGGING ------------------
    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # ------------------ CHECKPOINT + EARLY STOP ------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
            },
            save_path
        )
        print("✓ Best model saved")

    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break


# state_dict = torch.load('/kaggle/working/resnet18.pth',map_location=DEVICE,weights_only=False)
# model = get_model()
# model.load_state_dict(state_dict['model_state_dict'])

state_dict = torch.load('/kaggle/working/resnet50_4.pth',map_location=DEVICE,weights_only=False)
model = get_model50()
model.load_state_dict(state_dict['model_state_dict'])


model.to(DEVICE)
model.eval()

num_samples = test_file['images'].shape[0]
preds = np.zeros((num_samples, 2), dtype=np.int64)

ptr = 0  # running pointer (important)

with torch.no_grad():
    for images, idx in tqdm(test_loader, desc="INFERENCE"):
        images = images.to(DEVICE)

        outputs = model(images)
        pred_labels = torch.argmax(outputs, dim=1)

        batch_size = images.size(0)

        preds[ptr:ptr + batch_size, 0] = idx.cpu().numpy()
        preds[ptr:ptr + batch_size, 1] = pred_labels.cpu().numpy()

        ptr += batch_size



pd.DataFrame(preds,columns=['Id','label']).to_csv('submission50_04.csv',index=False)




