import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn
import pandas as pd


import os
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=True):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform=transform
        self.target_transform=target_transform
        
    
    def __len__(self):
        return len(self.img_labels)

    def _fill(self, num):
        n=str(num)
        return '0'*(6-len(n))+n+'.png'
            
        
    def __getitem__(self, idx):
        path=self._fill(self.img_labels.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, path)
        image = read_image(img_path)[0].float()
        label = torch.tensor([self.img_labels.iloc[idx, 1]], dtype=torch.float32)

        # normalize
        image = image/255.0
        if self.target_transform:
            label = label/100.0
      
        return image, label


age_ds=ImageDataset("/kaggle/input/spr-x-ray-age/train_age.csv", 
               "/kaggle/input/spr-x-ray-age/kaggle/kaggle/train")

gender_ds=ImageDataset("/kaggle/input/x-ray-train-gender/train_gender.csv", 
               "/kaggle/input/spr-x-ray-age/kaggle/kaggle/train", target_transform=False)




import numpy as np
def plot(img, label):
    img*=255.0
    img = img.numpy().astype(np.uint8)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"{label}")
    plt.show()

plot(*age_ds[0])
plot(*age_ds[341])
img, label=age_ds[341]
print(img)
print(label)


# Check the Alpha channel, if all 255, we can safely delete alpha channel
""" 
l=len(ds)
for i in range(l):
    image,label=ds[i]
    alpha = image[3].numpy() #tested when there are 4 channels
    if i%1000==0:
        print(f"Processing the {i}th image")
    if not np.all(alpha == 255):
        print(f"{image} has non-255 alpha values!")
        break
"""


from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset

def generate_dataloader(ds, workers=4, bs=64,val_bs=64): 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    train_idx, val_idx=next(iter(kf.split(ds)))
    
    train_subset = Subset(ds, train_idx)
    val_subset = Subset(ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=val_bs, shuffle=False,num_workers=workers, pin_memory=True)
    return train_loader, val_loader

age_loader, age_val_loader=generate_dataloader(age_ds)
gender_loader, gender_val_loader=generate_dataloader(gender_ds)


train_feature, label=next(iter(age_loader))
print(train_feature.size())
print(label.size())
print(train_feature)


net_1_age = nn.Sequential(nn.Flatten(), nn.Linear(1024*1024, 1)) 


x=nn.Flatten()(train_feature)
x.size()


from torch import optim

# def init_weights(l):
#     if type(l) == nn.Linear:
#         nn.init.normal_(l.weight, std=0.01)

def train(ep, model, optimizer, loss_func, data_loader, device, val_loader):
    
    model=model.to(device)
    model.train()
    # model.apply(init_weights)
    total=len(data_loader)
    val_loss=[]
    train_loss=[]
    for e in range(ep):
        for i, batch in enumerate(data_loader):
            train, label = batch
            train, label = train.to(device), label.to(device)
            
            X=model(train)
            loss=loss_func(X, label)
            with torch.no_grad():
                if i%10==0:
                    print(f"Epoch {e+1}/{ep}, Process {i}/{total}th batch, loss: {loss.item()}")
                if e>0 and i%50==0:
                    train_loss.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss=0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets=inputs.to(device), targets.to(device)
                outputs = model(inputs)
    
                loss = loss_func(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
            val_loss.append(total_loss/len(val_loader.dataset))
        
        # scheduler.step()
        # print(f"Epoch {e}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    fig, ax = plt.subplots()
    ax.plot(train_loss, 'o-', markeredgewidth=2)
    ax.plot(val_loss, linewidth=2.0)
    plt.show()
    
def validate_gender(model, loss_func, data_loader, device):
    model=model.to(device)
    model.eval()
    correct=0
    total_loss=0
    total=0
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets=inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = loss_func(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # change outputs to [0,1]
            probs = torch.sigmoid(outputs)
            
            # preds=0 or 1 
            preds = (probs > 0.5).float()  
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Accuracy: {accuracy}, Avg Loss:{avg_loss}")
            
  

def validate_age(model, loss_func, data_loader, device):
    model=model.to(device)
    model.eval()
    losses=0
    val_loss=[]
    preds_list=[]
    labels_list=[]
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            
            inputs, targets = batch
            inputs, targets=inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss=loss_func(outputs, targets)
            if i%5==0:
                val_loss.append(loss.item())
            losses+=loss

            preds_list.append(outputs * 100)  
            labels_list.append(targets * 100)

    preds_all = torch.cat(preds_list)
    labels_all = torch.cat(labels_list)
    mae = torch.mean(torch.abs(preds_all - labels_all))
    return mae.item()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torch.optim import lr_scheduler
train(10, net_1_age, optim.Adam(net_1_age.parameters(), lr=0.001), nn.MSELoss(),age_loader,device, age_val_loader)


feature, label=next(iter(age_val_loader))
feature, label=feature.to(device), label.to(device)
print(feature.size())
pred=net_1_age.to(device)(feature)
print(pred.view(1,-1))
print(label.view(1,-1))


validate_age(net_1_age, nn.MSELoss(), age_val_loader,device)


del net_1_age
torch.cuda.empty_cache()


net_1_gender = nn.Sequential(nn.Flatten(), nn.Linear(1024*1024, 1)) 
# train gender
train(5, net_1_gender, optim.Adam(net_1_gender.parameters(), lr=0.0001), nn.BCEWithLogitsLoss(), gender_loader, device, gender_val_loader)

# validate_gender(net_1_gender, nn.BCELoss(), gender_val_loader)


validate_gender(net_1_gender, nn.BCEWithLogitsLoss(), gender_val_loader, device)


del net_1_gender
torch.cuda.empty_cache()


class SingleLayerNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*1024, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


net_2=SingleLayerNet()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(5, net_2, optim.Adam(net_2.parameters(), lr=0.001), nn.MSELoss(), age_loader, device, age_val_loader) 


validate_age(net_2, nn.MSELoss(), age_val_loader,device)


feature, label=next(iter(age_val_loader))
feature, label=feature.to(device), label.to(device)
print(feature.size())
pred=net_2.to(device)(feature)
print(pred.view(1,-1))
print(label.view(1,-1))


del net_2
torch.cuda.empty_cache()


net_2_g=SingleLayerNet()
train(5, net_2_g, optim.Adam(net_2_g.parameters(), lr=0.0001), nn.BCEWithLogitsLoss(), gender_loader, device, gender_val_loader)


validate_gender(net_2_g, nn.BCEWithLogitsLoss(), gender_val_loader,device)



del net_2_g
torch.cuda.empty_cache()


# test multiple layer model, separated part
class MultipleLayerNet(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*1024, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


MLP=MultipleLayerNet()
train(10, MLP, optim.Adam(MLP.parameters(), lr=0.001), nn.MSELoss(), age_loader, device, age_val_loader) 


validate_age(MLP, nn.MSELoss(), age_val_loader,device)


del MLP
torch.cuda.empty_cache()


MLP_weight_decay=MultipleLayerNet()
train(5, MLP_weight_decay, optim.Adam(MLP_weight_decay.parameters(), lr=0.001, weight_decay=1e-5), nn.MSELoss(), age_loader, device, age_val_loader) 


validate_age(MLP_weight_decay, nn.MSELoss(), age_val_loader,device)


del MLP_weight_decay
torch.cuda.empty_cache()


class ImageDatasetJoint(Dataset):
    def __init__(self, annotations_file, gender_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.gender_labels = pd.read_csv(gender_file)
        self.img_dir = img_dir
        
        # self.max_label=torch.tensor(self.img_labels['age'].max())
        # self.min_label=torch.tensor(self.img_labels['age'].min())
    
    def __len__(self):
        return len(self.img_labels)

    def _fill(self, num):
        n=str(num)
        return '0'*(6-len(n))+n+'.png'
            
        
    def __getitem__(self, idx):
        path=self._fill(self.img_labels.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, path)
        image = read_image(img_path)[0].float().unsqueeze(0)
        age = torch.tensor([self.img_labels.iloc[idx, 1]], dtype=torch.float32)
        gender=torch.tensor([self.gender_labels.iloc[idx, 1]], dtype=torch.float32)
        # normalize
        image = image/255.0
        age = age/100.0
      
        return image, (age, gender)



ds=ImageDatasetJoint("/kaggle/input/spr-x-ray-age/train_age.csv", 
                    "/kaggle/input/x-ray-train-gender/train_gender.csv",
               "/kaggle/input/spr-x-ray-age/kaggle/kaggle/train")



tot, val=generate_dataloader(ds,4)


feature, label=next(iter(tot))
age, gender= label
print(feature.size())
print(age.size())
print(gender.size())


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), #16*512*512
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) , #16*256*256
            
            nn.Conv2d(16, 32, kernel_size=5, padding=2), #32*256*256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #32*128*128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),#64*128*128
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #64*64*64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),#128*64*64
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #128*32*32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),#256*32*32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4) #256*8*8
        )
        self.flat=nn.Flatten()
        # Regression head for age
        self.reg_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Classification head for gender
        self.cls_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x=self.conv(x)
        x=self.flat(x)
        age = self.reg_head(x)
        gender_logits = self.cls_head(x)
        return age, gender_logits


model=CNNet()


def train_3(epoch, model, optimizer, ds, device, val_ds):
    model.to(device)
    model.train()
    age_loss=nn.MSELoss()
    gender_loss=nn.BCEWithLogitsLoss()
    length=len(ds)
    train_loss=[]
    val_loss=[]
    for ep in range(epoch):
        for i, batch in enumerate(tot):
            train, (age, gender)=batch
            train, age, gender=train.to(device), age.to(device), gender.to(device)
            age_o, gender_o=model(train)
            loss=age_loss(age_o, age)+gender_loss(gender_o, gender)
            with torch.no_grad():
                if i%10==0:
                    print(f"Epoch {ep+1}/{epoch}, processing {i}/{length}, loss: {loss}")
                if i%50==0:
                    train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss=0
        with torch.no_grad():
            for batch in val_ds:
                inputs, targets = batch
                age_label, gender_label=targets
                inputs, age_label, gender_label=inputs.to(device), age_label.to(device), gender_label.to(device)
                age_o, gender_o = model(inputs)
    
                loss = age_loss(age_o, age_label)+gender_loss(gender_o, gender_label)
                total_loss += loss.item() * inputs.size(0)
            val_loss.append(total_loss/len(val_ds.dataset))
        
    fig, ax = plt.subplots()
    ax.plot(train_loss, 'o-', markeredgewidth=2)
    ax.plot(val_loss, linewidth=2.0)
    plt.show()

def valuate(model, data_loader, device):
    model=model.to(device)
    model.eval()
    correct=0
    total_loss=0
    total=0
    losses=0
    preds_list=[]
    labels_list=[]
    age_loss=nn.MSELoss()
    gender_loss=nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs, label = batch
            age, gender=label
            inputs, age, gender=inputs.to(device), age.to(device), gender.to(device)
            age_o, gender_o = model(inputs)

            loss = gender_loss(gender_o, gender)
            total_loss += loss.item() * inputs.size(0)
        
            probs = torch.sigmoid(gender_o)
            
            preds = (probs > 0.5).float()  
            
            correct += (preds == gender).sum().item()
            total += gender.size(0)
       
            # age
            loss_a=age_loss(age_o, age)
            losses+=loss_a.item() * inputs.size(0)

            preds_list.append(age_o * 100)  
            labels_list.append(age * 100)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Accuracy of gender: {accuracy}, Avg Loss:{avg_loss}")
    preds_all = torch.cat(preds_list)
    labels_all = torch.cat(labels_list)
    mae = torch.mean(torch.abs(preds_all - labels_all))
    print(f"MAE of age: {mae}")
    return mae.item()
    


train_3(10, model, torch.optim.Adam(model.parameters(), lr=0.001), tot, device, val)


valuate(model, val, device)


feature, label=next(iter(gender_val_loader))
feature=feature.unsqueeze(1)
feature, label=feature.to(device), label.to(device)
print(feature.size())
pred=model.to(device)(feature)
age_pred, gender_pred=pred
probs = torch.sigmoid(gender_pred)
preds = (probs > 0.5).float()
print(preds.view(1,-1))
print(label.view(1,-1))


# this is a separate model, just to check overfitting 
model1=CNNet()
train_3(20, model1, torch.optim.Adam(model1.parameters(), lr=0.001), tot, device, val)


valuate(model1, val, device)


# Example: Visualize first conv layer's feature map

model.eval()
feature, label=next(iter(val))
age, gender= label
image = feature[1].unsqueeze(0).to(device)  # shape: [1, 1, 1024, 1024]
print(image.size())
feature_map=image
with torch.no_grad():
    for i in range(0, 13, 1):
        feature_map = model.conv[i](feature_map)  # Conv1 output
        print(feature_map.size())
       
        for i in range(min(8, feature_map.shape[1])):
            plt.subplot(2, 4, i+1)
            plt.imshow(feature_map[0, i].cpu(), cmap='gray')
            plt.title(f'Channel {i}')
            plt.axis('off')
        plt.show()


del model
torch.cuda.empty_cache()




