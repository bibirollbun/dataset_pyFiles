# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


work_dir = "../input/mnist-rotation"
os.listdir(work_dir)


train=pd.read_pickle(os.path.join(work_dir,"train.pkl"))
train.image





import torch
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset,random_split
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import rotate as scipy_rotate


def show_image(image, title=None):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    if title is not None:
        plt.title(title)
    plt.show()

def rotate(img: np.ndarray, angle: int) -> np.ndarray:
    if not (-120 <= angle <= 120):
        raise ValueError("Angle must be between -120 and 120 degrees.")
    
    rotated_img = scipy_rotate(
        img,
        angle=angle,
        reshape=False,      # сохраняем размер 28x28
        order=1,            # билинейная интерполяция для аккуратного поворота
        mode='constant',    # пиксели вне исходного изображения заполняются константой
        cval=0.0            # заполняем черным (0)
    )
    return rotated_img


class NoizeGenerator:
    def __init__(
        self,
        discrete_noise_proba=0.02,
        beta_alpha=0.3,
        beta_beta=0.3,
        gaussian_sigma=0.0,
        shift_prob=1.0,
        seed=None,
    ):
        """
        discrete_noise_proba: вероятность заменить пиксель значением из бета-распределения
        beta_alpha, beta_beta: параметры бета-распределения (бимодальность при < 1)
        gaussian_sigma: стандартное отклонение для нормального шума
        shift_prob: вероятность случайного сдвига изображения на 1 пиксель
        """
        self.discrete_noise_proba = discrete_noise_proba
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.gaussian_sigma = gaussian_sigma
        self.shift_prob = shift_prob
        self.rng = np.random.default_rng(seed)

    def apply_beta_noise(self, img):
        mask = self.rng.random(img.shape) < self.discrete_noise_proba
        beta_noise = (
            self.rng.beta(self.beta_alpha, self.beta_beta, size=img.shape) * 255
        )
        noisy_img = img.copy()
        noisy_img[mask] = beta_noise[mask]
        return noisy_img

    def apply_gaussian_noise(self, img):
        if self.gaussian_sigma > 0:
            noise = self.rng.normal(loc=0.0, scale=self.gaussian_sigma, size=img.shape)
            img = img + noise
            img = np.clip(img, 0, 255)
        return img

    def apply_random_shift(self, img):
        direction = self.rng.choice(["up", "down", "left", "right"])
        shifted = np.zeros_like(img)

        if direction == "up":
            shifted[:-1, :] = img[1:, :]
        elif direction == "down":
            shifted[1:, :] = img[:-1, :]
        elif direction == "left":
            shifted[:, :-1] = img[:, 1:]
        elif direction == "right":
            shifted[:, 1:] = img[:, :-1]

        return shifted

    def transform_image(self, image):
        tmp_image = image.astype(np.float32)
        tmp_image = self.apply_beta_noise(tmp_image)
        tmp_image = self.apply_gaussian_noise(tmp_image)

        if self.rng.random() < self.shift_prob:
            tmp_image = self.apply_random_shift(tmp_image)
        return tmp_image.astype(np.uint8)

    def transform_dataset(self, X):
        n_samples = X.shape[0]
        X_aug = np.zeros_like(X)

        for i in range(n_samples):
            img = self.transform_image(X[i].reshape(28, 28))
            X_aug[i] = img.flatten()
        return X_aug
    

noize_gen = NoizeGenerator(
    discrete_noise_proba=0.2,
    beta_alpha=0.3,
    beta_beta=0.3,
    gaussian_sigma=40,
    shift_prob=1.0,
    seed=42,
)


test_dataset=pd.read_pickle(os.path.join(work_dir,"test.pkl"))
X_val=np.stack([np.asarray(img).reshape(-1) for img in test_dataset.image])


test_dataset.label.iloc[1]


show_image(test_dataset.image.iloc[1])


show_image(noize_gen.transform_image(train.image.iloc[0]))


class PickleDataset(Dataset):
    def __init__(self,path,angles):
        train_dataset=pd.read_pickle(os.path.join(work_dir,path))
        self.image=train_dataset["image"]
        self.label=train_dataset["label"]
        self.angles=angles
        self.lengthOf=len(self.label)
        columns_names=['image','label','angle']
        new_arr=[]
        for img,lbl in zip(self.image,self.label):
            for angle in self.angles:
                new_info={}
                rotated_img=scipy_rotate(
                    img,
                    angle=angle,
                    reshape=False,
                    order=1,
                    mode='constant',
                    cval=0.0
                )
                blurimg=noize_gen.transform_image(rotated_img)
                new_info={'image':blurimg,'label':lbl,'angle':angle}
                new_arr.append(new_info)

        self.new_dataset=pd.DataFrame(new_arr,columns=columns_names)

        self.image=self.new_dataset["image"]
        self.label=self.new_dataset["label"]
        self.angles=self.new_dataset["angle"]
                

    def __len__(self):
        return len(self.label)

    # def __getitem__(self,index):
    #     return self.image[index],self.angles[index]
        
    


from sklearn.ensemble import RandomForestClassifier
angles=[-120,-90,-60,-30,0,30,60,90,120]
new_dataset=PickleDataset("train.pkl",angles)





show_image(new_dataset.image.iloc[1])


dataset=new_dataset.new_dataset
X=np.stack([np.asarray(img).reshape(-1) for img in dataset['image']])
y=dataset['angle']


show_image(dataset.image.iloc[1])


model=RandomForestClassifier(n_estimators=600,
    max_depth=None,          
    min_samples_leaf=3,      
    max_features="log2",
    n_jobs=-1,
    oob_score=True,
    bootstrap=True,
    random_state=0)
model.fit(X,y)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.fc1=nn.Linear(64,128)
        self.dropout=nn.Dropout(p=0.3)
        self.fc2=nn.Linear(128,9)

        self.bn1=nn.BatchNorm2d(16)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.bn_fc1=nn.BatchNorm1d(128)

    def forward(self,x,show_embedding=False):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.pool(x)
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.pool(x)
        x=self.relu(self.bn3(self.conv3(x)))
        x=F.adaptive_avg_pool2d(x,(1,1))
        x=x.view(x.size(0),-1)
        embeddings=self.bn_fc1(self.fc1(x))
        x=self.relu(embeddings)
        x=self.dropout(x)
        logits=self.fc2(x)
        if show_embedding==False:
            return logits
        return logits,embeddings
        

    


def evaluate_test_loss(model,test_loader,loss_fn):
    total_loss=0
    with torch.no_grad():
        for image,label in test_loader:
            image,label=image.to(device),label.to(device)
            pred=model(image)
            loss=loss_fn(pred,label)
            total_loss+=loss.item()

    avg=total_loss/len(test_loader)
    print(f"Validation_loss: {avg:.4f}")
    return avg
        
        


from torch.utils.data import Dataset
from PIL import Image
angles=[-120,-90,-60,-30,0,30,60,90,120]
ANGLE2IDX={angle:index for index,angle in enumerate(angles)}
import torchvision.transforms as transforms
class NormalizeDataset(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset
        self.image=dataset.image.tolist()
        self.label=[ANGLE2IDX[angle] for angle in dataset.angle.tolist()]
        self.transform=transforms.ToTensor()

    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):
        img=Image.fromarray(self.image[index])
        img=self.transform(img)
        lbl=self.label[index]
        return img,lbl
        


class EarlyStopping():
    def __init__(self,min_delta,patience=3):
        self.min_delta=min_delta
        self.patience=patience
        self.counter=0
        self.best_loss=None
        self.early_stop=False

    def __call__(self,value_loss):
        if self.best_loss is None:
            self.best_loss=values_loss
        elif value_loss<self.best_loss-self.min_delta:
            self.best_loss=self.value_loss
            self.counter=0
        else:
            self.counter+=1
            if self.counter==self.patience:
                self.early_stop=True


print(dataset.image.iloc[0])


print(dataset.image.iloc[0].reshape(1,1,-1).shape)

img=transforms.ToTensor()(dataset.image.iloc[0])
print(img.shape)
print(img.view(1,1,28,28).shape)


img=transforms.ToTensor()(dataset.image.iloc[0].reshape(1,1,-1))
print(img.reshape(1,1,28,28).shape)


from PIL import Image
import torchvision.transforms as transforms
device=torch.device("cpu")
def train(model,dataset,optimizer,scheduler,loss_fn,early_stop):
    total_batches=len(y)
    best_loss=float("inf")
    for epoch in range(50):
        model.train()
        epoch_loss=0
        for i in range(len(y)):
            
            # img=transforms.ToTensor()(dataset.image.iloc[i])
            optimizer.zero_grad()
            outputs,embeddings=model(img.view(1,1,28,28))
            loss=loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss+=loss.item()
        avg_train_loss=epoch_loss/total_batches
        print(f"epoch {epoch+1}, train_loss: {avg_train_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    # val_loss=evaluate_test_loss(model,test_dataset,loss_fn)
    # if val_loss<best_loss:
    #     best_loss=val_loss
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state':   model.state_dict(),
    #         'opt_state':     optimizer.state_dict(),
    #         'sched_state':   scheduler.state_dict()
    #     }, "checkpoint_CNN1.pth")
            
            


from PIL import Image
import torchvision.transforms as transforms
device=torch.device("cpu")
def train1(model,train_loader,test_loader,optimizer,scheduler,loss_fn,early_stop):
    total_batches=len(dataset)
    best_loss=float("inf")
    for epoch in range(50):
        model.train()
        epoch_loss=0
        for image,label in train_loader:
            image,label=image.to(device),label.to(device)
            optimizer.zero_grad()
            output=model(image,False)
            loss=loss_fn(output,label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss+=loss.item()
        avg_train_loss=epoch_loss/total_batches
        
        print(f"Epoch {epoch+1} train_loss: {avg_train_loss}")
        val_loss=evaluate_test_loss(model,test_loader,loss_fn)
        if val_loss<best_loss:
            best_loss=val_loss
            torch.save({
                'epoch': epoch,
                'model_state':   model.state_dict(),
                'opt_state':     optimizer.state_dict(),
                'sched_state':   scheduler.state_dict()
            }, "checkpoint_CNN1.pth")


import torch.optim as optim
import torch
from torch.utils.data import random_split, DataLoader
model=CNN()
generator=torch.Generator().manual_seed(42)
normal_dataset=NormalizeDataset(dataset)
test_size=int(0.2*len(normal_dataset))
train_dataset,test_dataset=random_split(normal_dataset,[len(normal_dataset)-test_size,test_size],generator=generator)
optimizer=optim.Adam(model.parameters(),lr=0.0015, weight_decay=1e-5)
total_steps=len(y)*50
lr_scheduler=torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy='cos',
    max_lr=0.01,
)
loss_fn=nn.CrossEntropyLoss(label_smoothing=0.02)
early_stop=EarlyStopping(min_delta=0.01,patience=5)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=6,pin_memory=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=6,pin_memory=True)
train1(model,train_loader,test_loader,optimizer,lr_scheduler,loss_fn,early_stop)


val_dataset=pd.read_pickle(os.path.join(work_dir,"test.pkl"))
image_val=val_dataset.image.tolist()
transform=transforms.ToTensor()
img=Image.fromarray(image_val[i])
img=transform(img)
print(img.reshape(1,1,28,28).shape)


from PIL import Image
import torch
import pandas as pd
model.eval()
val_dataset=pd.read_pickle(os.path.join(work_dir,"test.pkl"))
image_val=val_dataset.image.tolist()
transform=transforms.ToTensor()
y_pred=[]
with torch.no_grad():
    for i in range(len(val_dataset)):
        img=Image.fromarray(image_val[i])
        img=transform(img).unsqueeze(0).to(device)
        logits=model(img,False)
        pred=torch.argmax(logits,dim=1).item()
        y_pred.append(pred)


angles=[-120,-90,-60,-30,0,30,60,90,120]
IDX2ANGLE={index:angle for index,angle in enumerate(angles)}
y_pred_angles=[IDX2ANGLE[i] for i in y_pred]


test_dataset=pd.read_pickle(os.path.join(work_dir,"test.pkl"))
X_val=np.stack([np.asarray(img).reshape(-1) for img in test_dataset.image])


y_pred=model.predict(X_val)


results=pd.DataFrame({
    "ID":range(150000),
    "angle":y_pred_angles
})
results.to_csv("results.csv",index=False)

