import numpy as np 
import pandas as pd 
import torch
import matplotlib.pyplot as plt
import os
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Literal, Callable
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from collections import Counter
from tqdm.notebook import tqdm
from torchinfo import summary
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)


dirname = '/kaggle/input/histopathologic-cancer-detection'

train_df = pd.read_csv(f'{dirname}/train_labels.csv')

train_df.head()


print(f"No Cancer: {len(train_df[train_df['label'] == 0])}, Cancer: {len(train_df[train_df['label'] == 1])}")


plt.figure(figsize=(8,5))
train_df['label'].map({1: 'Cancer', 0: 'No Cancer'}).value_counts().plot(kind='bar', color=['gray', 'black'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


train_imgs = os.listdir(f'{dirname}/train')
test_imgs = os.listdir(f'{dirname}/test')
print(f'Example Files: {train_imgs[:5]}')
print(f"Shape: {Image.open(os.path.join(dirname, 'train', train_imgs[0])).convert('RGB').size}")


class CancerDS(Dataset):
    '''
        Custom dataset for histopathologic cancer detection images

        Args:
            data_dir: root directory of image data
            transform: transformation function to apply to the images
            imgs: list of the image filenames 
            labels: list of the labels (0 (no cancer) or 1 (cancer))
    '''
    def __init__(self, data_dir: str = "", transform: Callable = None, d_type: Literal['train', 'test'] = 'train'):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # find images
        image_dir = self.data_dir / d_type
        if not image_dir.exists():
            raise FileNotFoundError(f'Directory {image_dir} not found')
    
        self.imgs = list(image_dir.glob('*.tif'))

        # find labels
        labels_dir = self.data_dir / 'train_labels.csv'
        if not labels_dir.exists():
            raise FileNotFoundError(f'Directory {labels_dir} not found')

        df = pd.read_csv(labels_dir).set_index('id')
        self.labels = [df.loc[img.stem].values[0] for img in self.imgs]
    
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        # open image
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # apply transform if it exists
        if self.transform:
            img = self.transform(img)

        # return image, label, and the image name (the ID)
        img_id = img_path.stem
        return img, label, img_id


# Transform for training
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Transform for testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# create dataset and split into train, testing
ds = CancerDS(dirname, transform_train, 'train')

indices = np.arange(len(ds))
train_indices, temp_indices = train_test_split(indices, test_size=0.3)
test_indices, val_indices = train_test_split(temp_indices, test_size=0.5)

train_ds = Subset(ds, train_indices)
test_ds = Subset(ds, test_indices)
val_ds = Subset(ds, val_indices)
test_ds.dataset.transform = transform_test
val_ds.dataset.transform = transform_test


print(f'Training: {len(train_ds)}, Testing: {len(test_ds)}, Validation: {len(val_ds)}')


train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4, persistent_workers=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=4, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=4, persistent_workers=True)

for indices in train_dl:
    print(indices[0].shape)
    break


# showcasing a couple of the images from the dataset, without the normalization
example = next(iter(train_dl))

images = example[0]
labels = example[1]

fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

random_indices = torch.randint(low=0, high=32, size=(5,))
for i, num in enumerate(random_indices):
    # take out the normalization and reorganize so matplotlib can show the images
    img = images[num]
    img = img * std + mean
    img = img.permute(1, 2, 0)
    label = 'Cancer' if labels[num] == 1 else 'No Cancer'
    
    axes[i].set_title(label)
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


class CancerModel(nn.Module):
    def __init__(self):
        super(CancerModel, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32, 48 x 48

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64, 24 x 24

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 128, 12 x 12
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)
        return x


model = CancerModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

params = {
    'epochs': 25,
    'optimizer': optimizer,
    'lr_scheduler': lr,
    'weight_path': 'cnn1_weights.pt',
    'loss_fn': nn.BCELoss(),
    'patience': 7,
}

summary(model, input_size=(1, 3, 96, 96))


def train(model, dl, loss_fn, opt, device):
    model.train()
    for batch, (X, y, _) in enumerate(dl):
        X, y = X.to(device), y.to(device).float().unsqueeze(1)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()

def val(model, dl, loss_fn, device):
    model.eval()
    total_loss, corr = 0, 0
    with torch.no_grad():
        for X, y, _ in dl:
            X, y = X.to(device), y.to(device).float().unsqueeze(1)
            pred = model(X)
            total_loss += loss_fn(pred, y).item() * X.size(0)
            pred = (pred > 0.5).float() # binary output
            corr += (pred == y).sum().item()
    avg_loss = total_loss / len(dl.dataset)
    accuracy = corr / len(dl.dataset)
    return avg_loss, accuracy

def train_model(model, train_dl, val_dl, params, device):
    epochs = params['epochs']
    optimizer = params['optimizer']
    scheduler = params['lr_scheduler']
    weight_path = params['weight_path']
    loss_func = params['loss_fn']
    patience = params['patience']
    
    best_loss = float('inf')
    no_improve = 0
    total_train_loss = []
    total_val_loss = []
    
    for epoch in tqdm(range(epochs), desc='Training'):
        # training 
        print(f'Epoch {epoch+1}/{epochs}')
        train_loss = train(model, train_dl, loss_func, optimizer, device)
        val_loss, val_acc = val(model, val_dl, loss_func, device)
        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), weight_path)
            print('New best model.')
        else:
            no_improve += 1
            if no_improve >= patience:
                print('Model might be overfitting. Stopping early...')
                break
    return total_train_loss, total_val_loss


train_losses, val_losses = train_model(model, train_dl, val_dl, params, device) 


best_model = CancerModel().to(device)
# best_model.load_state_dict(torch.load(params['weight_path'])) # -> if just trained data
best_model.load_state_dict(torch.load('/kaggle/input/cnn-cancerdetection-weights/cnn1_weights.pt', weights_only=True), strict=True) # when uploading weights previously trained
best_model.eval()
best_loss, best_acc = val(best_model, val_dl, params['loss_fn'], device)
print(f'Loss: {best_loss:.4}, Accuracy: {best_acc:.4}')


all_predictions = []
all_labels = []

with torch.no_grad():
    for X, y, _ in val_dl:
        X = X.to(device)
        y = y.to(device)
        preds = (best_model(X) > 0.5).float()

        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

prec = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)

print(f'Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')


plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Val Loss')
plt.legend()


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = Path(test_dir)
        self.image_ids = sorted(os.listdir(self.test_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.test_dir / image_id
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, image_id


test_dir = f"{dirname}/test"
batch_size = 64

test_dataset = TestDataset(test_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_model.eval()
final_preds = []
final_ids = []

with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = best_model(images)
        preds = (outputs > 0.5).float().squeeze().cpu().numpy()

        if preds.ndim == 0:
            preds = [preds.item()]
        else:
            preds = preds.tolist()

        final_preds.extend(preds)
        final_ids.extend([i.split('.')[0] for i in ids])

results_df = pd.DataFrame({
    'id': final_ids,
    'label': final_preds
})

results_df.to_csv('submission.csv', index=False)
results_df.head()

