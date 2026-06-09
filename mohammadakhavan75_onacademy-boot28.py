# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import os


root="/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/"


os.path.join(root, "train.csv")


df_train = pd.read_csv(os.path.join(root, "train.csv"))


df_train.head()


df_train_label_coordinates = pd.read_csv(os.path.join(root, "train_label_coordinates.csv"))


df_train_label_coordinates.head()


sample_img_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/100206310/1012284084/1.dcm"


from pydicom import dcmread


dcm_output = dcmread(sample_img_path)


dcm_output


dcm_img = dcm_output.pixel_array


dcm_img.shape


import matplotlib.pyplot as plt


plt.imshow(dcm_img, cmap='gray')


idx=10
sample_img_path = f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/100206310/2092806862/{idx}.dcm"
dcm_img = dcmread(sample_img_path).pixel_array
plt.imshow(dcm_img, cmap='gray')


# df_train 
df_labels = pd.read_csv(os.path.join(root, "train.csv"))
# df_train_label_coordinates
df_img = pd.read_csv(os.path.join(root, "train_label_coordinates.csv"))


patients_id = df_labels['study_id'].unique()


df_labels[df_labels['study_id']==4003253]['spinal_canal_stenosis_l1_l2']


patient_table = df_img[df_img['study_id']==4003253]
series_id = patient_table['series_id'].unique()
patient_position = patient_table[patient_table['series_id']==702807833]
patient_img = patient_position[patient_position['instance_number']==8]
condition = patient_img.iloc[0]['condition']
level = patient_img.iloc[0]['level']

col_name = condition.lower().replace(' ','_') + '_' + level.lower().replace('/', '_')


status = df_labels[df_labels['study_id']==4003253][col_name].values[0]


for row in patient_img:
    print(row)


imgs_path = []
labels = []
root_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'
for patient_id in patients_id:
    patient_table = df_img[df_img['study_id']==patient_id]
    series_id = patient_table['series_id'].unique()
    for serie_id in series_id:
        patient_position = patient_table[patient_table['series_id']==serie_id]
        instance_numbers = patient_position['instance_number'].unique()
        for instance_number in instance_numbers:
            patient_img = patient_position[patient_position['instance_number']==instance_number]
            for row in range(len(patient_img)):
                condition = patient_img.iloc[row]['condition']
                level = patient_img.iloc[row]['level']
                col_name = condition.lower().replace(' ','_') + '_' + level.lower().replace('/', '_')

                status = df_labels[df_labels['study_id']==patient_id][col_name].values[0]

                if status.lower() == 'Normal/Mild'.lower()
                    label = 0
                
                if status.lower() == 'Moderate'.lower()
                    label = 1
                
                if status.lower() == 'Severe'.lower()
                    label = 2

                img_path = os.path.join(root_path, str(patient_id), str(serie_id), str(instance_number))

                imgs_path.append(img_path)
                labels.append(label)


os.path.join("1",'2','3')


root_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'


from torch.utils.data import Dataset, DataLoader, random_split
import math

class lumbar_dataset(Dataset):
    def __init__(self, root_path, df_train_path, df_coord_path, transform):
        self.imgs_path = []
        self.labels = []

        self.transform = transform
        
        df_img = pd.read_csv(df_coord_path)
        df_labels = pd.read_csv(df_train_path)
        
        for patient_id in patients_id[:10]:
            patient_table = df_img[df_img['study_id']==patient_id]
            series_id = patient_table['series_id'].unique()
            for serie_id in series_id:
                patient_position = patient_table[patient_table['series_id']==serie_id]
                instance_numbers = patient_position['instance_number'].unique()
                for instance_number in instance_numbers:
                    patient_img = patient_position[patient_position['instance_number']==instance_number]
                    for row in range(len(patient_img)):
                        condition = patient_img.iloc[row]['condition']
                        level = patient_img.iloc[row]['level']
                        col_name = condition.lower().replace(' ','_') + '_' + level.lower().replace('/', '_')
        
                        status = df_labels[df_labels['study_id']==patient_id][col_name].values[0]
                        if type(status) is not str: #or math.isnan(status):
                            # print(f"patient_id: {patient_id}, col_name: {col_name}, status: {status},\
                            # serie_id: {serie_id}, instance_number: {instance_number}, row: {row},\
                            # condition: {condition}, level: {level}")
                            print(f"In patient_id: {patient_id} Nan occured!")
                            continue
                            
                        try:
                            if status.lower() == 'Normal/Mild'.lower():
                                label = 0
                            
                            elif status.lower() == 'Moderate'.lower():
                                label = 1
                            
                            elif status.lower() == 'Severe'.lower():
                                label = 2
    
                            else:
                                assert False, f"The status didn't match any of predefined labels: {status}"

                        except:
                            assert False, f"patient_id: {patient_id}, col_name: {col_name}, status: {status},\
                            serie_id: {serie_id}, instance_number: {instance_number}, row: {row},\
                            condition: {condition}, level: {level}"
                        
                        img_path = os.path.join(root_path, str(patient_id), str(serie_id), str(instance_number) + '.dcm')

                        if os.path.exists(img_path):
                            self.imgs_path.append(img_path)
                            self.labels.append(label)
                        else:
                            print(f"Image path does not exists! {img_path}")
                            continue

        # assert False==True, "Error"
        assert len(self.imgs_path) == len(self.labels), "ERROR: images and labels mismatch lentgh"
        
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_array = dcmread(self.imgs_path[idx]).pixel_array
        label = self.labels[idx]

        img = self.transform(img_array.astype(np.uint8))
        
        return img, label


from torchvision.transforms import v2


torch.__version__


import torchvision
torchvision.__version__


transform = v2.Compose([
    v2.ToPILImage(),
    v2.Grayscale(num_output_channels=3),
    v2.Resize(256), # v2.Resize(224, 224)
    v2.CenterCrop(224),
    v2.RandomRotation(30),
    v2.RandomApply([v2.GaussianBlur(3, 1)],p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.RandomApply([v2.GaussianNoise(0,0.01),],p=0.5),
])


# TODO: check dataset for len images !!!!
full_dataset = lumbar_dataset(root_path='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/',
                            df_train_path='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv',
                            df_coord_path='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv',
                            transform = transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"len(train_dataset): {len(train_dataset)}, len(val_dataset): {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


from tqdm import tqdm
def train(loader, optimizer, criterion, model, device, epoch):
    total_loss = []
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, labels)
        pbar.set_postfix(loss=loss.item())
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(total_loss)


def evaluation(loader, criterion, model, device, epoch):
    total_loss = []
    model.eval()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels)
            pbar.set_postfix(loss=loss.item())
            total_loss.append(loss.item())
            
    return np.mean(total_loss)


model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)

for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss().to(device)
model = model.to(device)

epochs=50


for epoch in range(epochs):
    train_loss = train(train_loader, optimizer, criterion, model, device)
    eval_loss = evaluation(eval_loader, criterion, model, device)
    print(f"Epoch {epoch}/{epochs}: train loss: {train_loss}, validation loss: {eval_loss}")


!wget https://www.mount-it.com/cdn/shop/articles/Ultrawide_Monitor_a8d20e1e-12b9-4e0c-b6cf-678ca76b3ccf.webp?v=1747237890 -O ./im.jpg


t


import cv2


img = cv2.imread('./im.jpg')


t1 = v2.Compose([
    v2.ToPILImage(),
    v2.Resize(256), # v2.Resize(224, 224)
    v2.CenterCrop(224),
    v2.ToTensor()
])
t2 = v2.Compose([
    v2.ToPILImage(),
    v2.Resize((224, 224)),
    v2.ToTensor()
])


im1 = t1(img)
im2 = t2(img)


import matplotlib.pyplot as plt


plt.imshow(im1.permute(1,2,0))


plt.imshow(im2.permute(1,2,0))




