%%capture
!unzip '/kaggle/input/the-nature-conservancy-fisheries-monitoring/train.zip'
!unzip '/kaggle/input/the-nature-conservancy-fisheries-monitoring/test_stg1.zip'
# !py7zr '/kaggle/input/the-nature-conservancy-fisheries-monitoring/test_stg2.7z'
# !unzip '/kaggle/input/the-nature-conservancy-fisheries-monitoring/sample_submission_stg1.csv.zip'
!unzip '/kaggle/input/the-nature-conservancy-fisheries-monitoring/sample_submission_stg2.csv.zip'


%%capture
import os

# пути
path_in = "/kaggle/input/the-nature-conservancy-fisheries-monitoring/test_stg2.7z"
path_out = "/kaggle/working/test_stg2"

# создаём папку для распаковки
os.makedirs(path_out, exist_ok=True)

# проверяем, есть ли 7zip
if os.system("which 7za > /dev/null") != 0:
    !apt-get install -y p7zip-full

# распаковка
!7za x {path_in} -o{path_out}

print("✅ Распаковка завершена!")



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from tqdm import tqdm
import pandas as pd


from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root="/kaggle/working/train", transform=transform)

# классы будут автоматически определены по именам папок
print(dataset.classes)   # ['class1', 'class2', 'class3']



# Разделение train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)


class CatDogClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = 8

        # VGG16 с предобученными весами
        self.model = models.vgg16(pretrained=True)

        # Замораживаем сверточные слои
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Новый классификатор
        self.model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)

    def predict_step(self, batch, batch_idx):
        imgs, names = batch  # если твой TestDataset возвращает (image, path)
        outputs = self(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        return [(name, prob) for (name, prob) in zip(names, probs)]


model = CatDogClassifier(learning_rate=1e-4)
trainer = Trainer(max_epochs=1, accelerator='auto', log_every_n_steps=10,
                 # fast_dev_run=3
                 )
trainer.fit(model, train_loader, val_loader)


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir) #sorted(os.listdir(image_dir), key=lambda x: int(x[6:11]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_name

test_dataset_1 = TestDataset('/kaggle/working/test_stg1', transform=transform)
test_loader_1 = DataLoader(test_dataset_1, batch_size=16, shuffle=False)

predictions_1 = trainer.predict(model, test_loader_1)


test_dataset_2 = TestDataset('/kaggle/working/test_stg2/test_stg2', transform=transform)
test_loader_2 = DataLoader(test_dataset_2, batch_size=16, shuffle=False)

predictions_2 = trainer.predict(model, test_loader_2)


import pandas as pd

# разворачиваем
flat = []
for predictions in [predictions_1, predictions_2]:
    for group in predictions:                # внешний список
        for fname, vec in group:      # внутри — кортеж
            if predictions == predictions_2:
                flat.append(['test_stg2/' + fname] + vec.tolist())  # tensor -> list
            else:
                flat.append([fname] + vec.tolist())  # tensor -> list
    
# создаём DataFrame
df = pd.DataFrame(flat, columns=["image"] + [cls for cls in dataset.classes])

print(df.head())
print(f'{df.shape = }')



sample_submission = pd.read_csv('/kaggle/working/sample_submission_stg2.csv')
sample_submission


merged = sample_submission[['image']].merge(df, on='image', how='left')
merged.to_csv('submission.csv', index=False)
merged

