#basic imports 
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import warnings
import os
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)


#metadata
train_df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')
train_df.head()


#image
path = '/kaggle/input/rsna-breast-cancer-detection/train_images/10006/1459541791.dcm'
dicom = pydicom.dcmread(path)

plt.imshow(dicom.pixel_array, cmap='gray')
plt.title("example image")
plt.axis('off')
plt.show()


#checks % of positive vs negative in train
print("counts: ")
cnts = train_df['cancer'].value_counts()
print(f"positive: {cnts[1]}, negative: {cnts[0]}")

print("") #newline

print("%'s': ")
pres = train_df['cancer'].value_counts(normalize=True)
print(f"positive: {pres[1]*100:.2f}%, negative: {pres[0]*100:.2f}%")




#random downsample
positive_count = cnts[1]
train_df = pd.concat([train_df[train_df['cancer'] == 1], train_df[train_df['cancer'] == 0].sample(n=positive_count)]).sample(frac=1)  

print("updated counts: ")
cnts = train_df['cancer'].value_counts()
print(f"positive: {cnts[1]}, negative: {cnts[0]}")

print("") #newline

print("updated %'s': ")
pres = train_df['cancer'].value_counts(normalize=True)
print(f"positive: {pres[1]*100:.2f}%, negative: {pres[0]*100:.2f}%")


class RSNADataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient_id']
        image_id = row['image_id']
        label = row['cancer']

        image_path = os.path.join(self.image_dir, str(patient_id), f"{image_id}.dcm")
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array.astype(np.float32)

        image -= image.min()
        image /= image.max()

        image = torch.tensor(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


image_dir = "/kaggle/input/rsna-breast-cancer-detection/train_images"

dataset = RSNADataset(train_df, image_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)


for images, labels in dataloader:
    print(images.shape)  # [batch_size, 1, H, W]
    print(labels)
    break

