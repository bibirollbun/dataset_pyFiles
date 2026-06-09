import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import zipfile
import os


import torchvision.transforms.functional as F
class InterConv(nn.Module):
    def __init__(self,in_channels,out_channels,dropout_probab = 0.0):
        super(InterConv,self).__init__()
        layers=[
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False), ##here 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_probab > 0:
            layers.append(nn.Dropout(p=dropout_probab))
        self.conv = nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv(x)
        return x


class UNET(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,p=0.0,features=[64,128,256,512]):
        super(UNET,self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #downsampling - Part-01 of UNET
        for feature in features:
            self.downs.append(InterConv(in_channels,feature))
            in_channels = feature
            
        # in between these two processes, their exists another process of "Bottle-Necking" used to match the dimension requirements 
        
        #upsampling -  Part-02 of UNET
        for feats in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feats*2,
                    feats,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(InterConv(feats*2,feats))

        self.BottleNeck = InterConv(features[-1],features[-1]*2,dropout_probab=p)
        self.FinalLayer = nn.Conv2d(features[0],out_channels,1)

    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) #these will be used in the decoder (upsampling)
            x = self.pool(x)
        x = self.BottleNeck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip_connect = skip_connections[idx//2]
            if x.shape != skip_connect.shape: #to manage shape mismatch while forward pass, even sized dimension 
                x = F.resize(x, size=skip_connect.shape[2:])
            concat_skip = torch.concat((skip_connect,x),dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.FinalLayer(x)


# we need train_imgs, train_mask, val_imgs, val_mask and finallly some test cases to test our models
for things in os.listdir('/kaggle/input/carvana-image-masking-challenge'):
    print(things)


folders = ['train_masks.zip','train.zip','test.zip']
destination = ['train_mask','train','test']
def UnZipFile(path,dest):
    with zipfile.ZipFile(path,'r') as ref:
        ref.extractall(dest)

root = r'/kaggle/input/carvana-image-masking-challenge'

for path,dest in zip(folders,destination):
    print(f'=> Extracting {path}....')
    path = os.path.join(root,path)
    os.makedirs(dest,exist_ok=True)
    UnZipFile(path,dest)


%%time

#make validation set out of the training images, let's use randomized 90/10 split 
# which roughly means about 600 images for the 
import random
random.seed(42)
#make list of img_paths and corresponding mask_paths
train_dir = r'/kaggle/working/train/train'
train_mask_dir = r'/kaggle/working/train_mask/train_masks'
test_dir = r'/kaggle/working/test/test'

images = sorted([p for p in os.listdir(train_dir)])
masks = sorted([p for p in os.listdir(train_mask_dir)])
testing = [p for p in os.listdir(test_dir)][:100]
random.shuffle(images)

train_img = images[:int(0.9*len(images))]
val_img = images[int(0.9*len(images)):]
print(f'=> # of Training Images: {len(train_img)}\n=> # of Validation Images: {len(val_img)}')

train_masks = [m.replace(".jpg","_mask.gif") for m in train_img]
val_masks = [m.replace(".jpg","_mask.gif") for m in val_img]
print(f'=> # of Training Masks: {len(train_masks)}\n=> # of Validation Masks: {len(val_masks)}')

print(f'=> # of Testing Images: {len(testing)}')


#checking if all the masks are present or not 

for img,mask in zip(train_img+val_img,train_masks+val_masks):
    img = img.split(".")[0]
    mask = mask.split(".")[0].replace("_mask","")
    if img != mask:
        print(f'\n{mask} not found !')
        break
print(f'\n=> All masks matched!')


from PIL import Image
class CarvanaDataset(Dataset):
    
    def __init__(self,img_root:str,mask_root:str,data:tuple,transform):
        self.img_root = img_root
        self.mask_root = mask_root
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self,idx:int) -> torch.tensor:
        img_path = os.path.join(self.img_root,self.data[0][idx])
        mask_path = os.path.join(self.mask_root,self.data[1][idx])
        
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)
        mask[mask == 255.0] = 1.0
        
        if self.transform:
            augmentation = self.transform(image=img,mask=mask)
            img = augmentation['image']
            mask = augmentation['mask']
        return img,mask

def get_loaders(directories,train_data,val_data,transforms=None,workers=2,pin=True,batch_size=64):
    loaders = []
    data = [train_data,val_data]
    img_root,mask_root = directories[0],directories[1]
    for idx in range(2):
        dataset = CarvanaDataset(img_root,mask_root,data[idx],transform=transforms[idx])
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers,pin_memory=pin)
        loaders.append(loader)
    return loaders
def get_loaders_med(lookup_table,data,transforms=None,workers=2,pin=True,batch_size=64):
    loaders=[]
    for idx in range(len(data)):
        dataset = MedDataSet(data[idx],lookup_table,transform=(transforms[idx] if transforms else None))
        loader= DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers,pin_memory=pin)
        loaders.append(loader)
    return loaders
    


import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm,trange

#hyperparams 
lr = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size=8
num_epochs = 50

# CARVANA TASK
IMAGE_HEIGTH= 320
IMAGE_WIDTH = 480

# ULTRASOUND TASK
# IMAGE_HEIGTH= 420
# IMAGE_WIDTH = 580

PIN_MEMORY = True


#this function represents 1 Epoch of training 
def train_this_model(loader, model, optimizer, loss_fn,scaler):
    loop = tqdm(loader)
    epoch_everage_loss = 0.0
    numBatch=0
    for idx,(data,targets) in enumerate(loop):
        numBatch+=1
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        #FORWARD PASS
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions,targets)
            
        #BACKWARD PASS
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss = loss.item())
        epoch_everage_loss += loss.item()
    epoch_everage_loss /= numBatch
    return epoch_everage_loss


train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGTH, width = IMAGE_WIDTH),
        # A.ShiftScaleRotate(shift_limit=0.05,scale_limit=0.05,rotate_limit=15,p=0.5),
        A.Rotate(limit=40,p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        A.Normalize(
            mean = [0.0,0.0,0.0],
            std = [1.0,1.0,1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGTH, width = IMAGE_WIDTH),
        A.Normalize(
            mean = [0.0,0.0,0.0],
            std = [1.0,1.0,1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2(),
    ]
)



with open('transforms.py','w') as f:
    f.write('''
import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
IMAGE_HEIGTH=160
IMAGE_WIDTH = 240
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGTH, width = IMAGE_WIDTH),
        A.Rotate(limit=30,p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean = [0.0,0.0,0.0],
            std = [1.0,1.0,1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGTH, width = IMAGE_WIDTH),
        A.Normalize(
            mean = [0.0,0.0,0.0],
            std = [1.0,1.0,1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2(),
    ]
)

if __name__ == '__main__':
    x = np.random.randn(3,1980,2560)
    print(f'=> Shape of original tensor: {x.shape}')
    print(f'=> Train Transformed: {train_transform(image=x)["image"].shape}')
    print(f'=> Validation/Test Transformed: {val_transform(image=x)["image"].shape}')
    ''')

# %run transforms.py


def save_model_checkpoint(state,filename='checkpoint_model.pth.tar'):
    print(f'=> saving checkpoint')
    torch.save(state,filename)
    
def load_model_checkpoint(checkpoint,model):
    print(f'=> Loading checkpoint')
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy_binary(loader,model,device):
    numCorrect = 0
    numPixels = 0
    model.eval()
    dice_score = 0
    val_loss = 0.0
    numBatch=0
    with torch.no_grad():
        for x,y in loader:
            numBatch+=1
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            loss = loss_fn(preds,y)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            numCorrect += (preds == y).sum()
            numPixels += torch.numel(preds)
            intersection = (preds * y).sum()
            dice_score += (2. * intersection) / (preds.sum() + y.sum() + 1e-8)
            val_loss += loss.item()
    model.train()
    acc = (numCorrect/numPixels)
    val_loss /=numBatch
    print(
        f'Got {numCorrect}/{numPixels} with acc {acc*100:.2f}'
    )
    print(f'Dice Score: {dice_score/len(loader):.2f}')
    return acc,val_loss
    


batch_size


model = UNET(p=0.4).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
scaler = torch.amp.GradScaler('cuda')


train_loader,val_loader = get_loaders(
    (train_dir,train_mask_dir),
    (train_img,train_masks),
    (val_img,val_masks),
    (train_transform,val_transform),
    batch_size=batch_size,
    pin = True
)


best_acc = float('-inf')
patience = 5
trigger = 0
avg_training_loss = []
avg_val_acc=[]
avg_val_loss = []
for epochs in trange(num_epochs):
    torch.cuda.empty_cache()
    train_loss = train_this_model(train_loader,model,optimizer,loss_fn,scaler)
    print(f'Epoch: {epochs+1}, Avg Training Loss: {train_loss}')
    avg_training_loss.append(train_loss)
    acc,val_loss = check_accuracy_binary(val_loader,model,device=DEVICE)
    avg_val_acc.append(acc)
    avg_val_loss.append(val_loss)
    if acc > best_acc:
        best_acc = acc
        checkpoint = {
            "state_dict":model.state_dict(),
            "optimzer":optimizer.state_dict()
        }
        save_model_checkpoint(checkpoint)
        trigger = 0
    else:
        trigger +=1
    if trigger > patience:
        print(f'=> EARLY STOPPING')
        break 


val_acc = [p.item() for p in avg_val_acc]


#plotting different model configuration graphs 
import matplotlib.pyplot as plt

fig,ax = plt.subplots(3,1,figsize=(10,8))
ax = ax.flatten()
titles = ['Avg Training Loss','Avg Val Accuracy','Avg Val Loss']
y = [avg_training_loss,val_acc,avg_val_loss]
x = np.arange(1,26+2,1)
for idx,(title,data) in enumerate(zip(titles,y)):
    ax[idx].plot(x,data)
    ax[idx].set_title(f'{title} v/s Epoch')
    ax[idx].set_xlabel('Epochs')
    ax[idx].set_ylabel(title)
    ax[idx].grid()
plt.tight_layout()

#I forgot to run this cell !


my_model = UNET(p=0.4).to(DEVICE) #from the model definition 
checkpoint_path = r'/kaggle/working/checkpoint_model.pth.tar'
checkpoint_dict = torch.load(checkpoint_path,map_location=DEVICE)
load_model_checkpoint(checkpoint_dict,my_model)


#testing the model on test images 
test_root = r'/kaggle/working/test/test'
test_img = [test_root + "/"+p  for p in testing]

def img_to_tensor(img_path,transform=None):
    img = np.array(Image.open(img_path).convert('RGB'))
    if transform:
        img = transform(image=img)
        img = img['image'].unsqueeze(0)
    return img 

def show_image(img_path,axis=None,transform=None):
    img = np.array(Image.open(img_path).convert("RGB"))
    if transform:
        img = transform(image=img)
        img = img['image'].permute(1,2,0).numpy()
    if axis:
        axis.imshow(img)
        axis.axis('off')
    else:
        plt.imshow(img)
        plt.axis('off')
    return img.shape

def get_mask(img_path,model,transform=val_transform,device='cpu'):
    model.to(device)
    x = img_to_tensor(img_path,transform).to(device)
    y = torch.sigmoid(model(x).detach().cpu())
    y = (y>0.5).float()
    y = y.squeeze(0).permute(1,2,0)
    return y


import matplotlib.pyplot as plt
n = 2
fig,ax = plt.subplots(n,2,figsize=(12,12))
transformations = [None,val_transform]
title = ['Original','Transformed']
for i in range(n):
    idx = np.random.randint(0,100)
    for j in range(2):
        shape = show_image(test_img[idx],ax[i,j],transformations[j])
        ax[i,j].set_title(f'{title[j]}, Shape:{shape}')
plt.tight_layout()


nimages = 5
fig,ax = plt.subplots(nimages,3,figsize=(12,12))
for i in range(nimages):
    idx = np.random.randint(0,len(test_img))
    image = test_img[idx]
    show_image(image,ax[i,0],None)
    ax[i,0].set_title(f'Original Image')
    ax[i,0].axis('off')
    show_image(image,ax[i,1],val_transform) #shows the image
    ax[i,1].set_title('Transformed Image(downsampled to resize)')
    ax[i,1].axis('off')
    mask = get_mask(image,my_model,val_transform,DEVICE)
    ax[i,2].imshow(mask,cmap='gray')
    ax[i,2].set_title('Predicted Mask')
    ax[i,2].axis('off')
plt.tight_layout()


import os
SEED = 42
ultrasound_train_path = r'/kaggle/input/ultrasound-nerve-segmentation/train'
ultrasound_test_path = r'/kaggle/input/ultrasound-nerve-segmentation/test'


train_images_ultrasound = []
for idx,things in enumerate(os.listdir(ultrasound_train_path)):
    if things.endswith('_mask.tif'):
        continue
    else:
        train_images_ultrasound.append(things)
print(f'# of Train Images: {len(train_images_ultrasound)}')
print(f'First 5 examples: {train_images_ultrasound[:5]}')

lookup_ultrasound = {} ## this dictionary stores the mask path as the key 

for images in train_images_ultrasound:
    lookup_ultrasound[ultrasound_train_path+"/"+images] = ultrasound_train_path + "/"+images.replace(".tif","_mask.tif")
print(f'First Five examples from the lookup dictionary')
print(list(lookup_ultrasound.items())[:5])


#making splits on the data train,val and hold set 
# randomly holding out some images 
n_hold = int(0.01*len(train_images_ultrasound))
hold_idx = np.random.randint(0,len(train_images_ultrasound),n_hold) #indices of val set 
np.random.seed(SEED)
train_ultrasound = [ultrasound_train_path+"/"+img for idx,img in enumerate(train_images_ultrasound) if idx not in hold_idx]
val_ultrasound = [ultrasound_train_path+"/"+img for idx,img in enumerate(train_images_ultrasound) if idx in hold_idx]

print(f'=> # of Train Images: {len(train_ultrasound)}')
print(f'=> # of Validation Images: {len(val_ultrasound)}')

for t in train_ultrasound:
    if t in val_ultrasound:
        print("ERROR")
        break


class MedDataSet(Dataset):
    def __init__(self,list_of_images,lookup_table,transform=None):
        self.list_of_images = list_of_images
        self.lookup_table = lookup_table
        self.transform = transform
    def __len__(self):
        return len(self.list_of_images)
    def __getitem__(self,idx):
        img_path = self.list_of_images[idx]
        mask_path = self.lookup_table[img_path]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask[mask == 255.0] = 1.0
        if self.transform:
            transformation = self.transform(image = img,mask = mask)
            img = transformation['image']
            mask = transformation['mask']
        return img,mask

n_train_transform = A.Compose(
    [
        A.Resize(height=336, width = 464),
        # A.ShiftScaleRotate(shift_limit=0.05,scale_limit=0.05,rotate_limit=15,p=0.5),
        A.Rotate(limit=40,p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        A.Normalize(
            mean = [0.0,0.0,0.0],
            std = [1.0,1.0,1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2(),
    ],
)

n_val_transform = A.Compose(
    [
        A.Resize(height=336, width = 464),
        A.Normalize(
            mean = [0.0,0.0,0.0],
            std = [1.0,1.0,1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2(),
    ]
)


train_loader,val_loader = get_loaders_med(
    lookup_ultrasound,
    data=[train_ultrasound,val_ultrasound],
    transforms=[n_train_transform,n_val_transform],
    batch_size=8
)


n_rows = 3
fig,axis = plt.subplots(n_rows,2,figsize=(6,6)) 
for x,y in train_loader:
    idx = np.random.randint(0,len(x),n_rows)
    for i in range(n_rows):
        img = x[idx[i]]
        mask = y[idx[i]]
        axis[i,0].imshow(img.permute(1,2,0))
        axis[i,0].axis('off')
        axis[i,1].imshow(mask,cmap='gray')
        axis[i,1].axis('off')
        print(img.shape)
        print(mask.shape)
    break
plt.tight_layout()


n_model = UNET(p=0.4).to(DEVICE) #from the model definition 
checkpoint_path = r'/kaggle/working/checkpoint_model.pth.tar'
checkpoint_dict = torch.load(checkpoint_path,map_location=DEVICE)
load_model_checkpoint(checkpoint_dict,my_model)


n_LR = 1e-5
n_optimizer = torch.optim.AdamW(n_model.parameters(),lr=n_LR)
n_scaler = torch.amp.GradScaler('cuda')


best_acc = float('-inf')
patience = 5
trigger = 0
n_avg_training_loss = []
n_avg_val_acc=[]
n_avg_val_loss = []
fig,ax = plt.subplots(nimages,3,figsize=(12,12))
for epochs in trange(5):
    torch.cuda.empty_cache()
    train_loss = train_this_model(train_loader,n_model,n_optimizer,loss_fn,n_scaler)
    print(f'Epoch: {epochs+1}, Avg Training Loss: {train_loss}')
    n_avg_training_loss.append(train_loss)
    nimages = 2
    for i in range(nimages):
        idx = np.random.randint(0,len(train_ultrasound))
        image = train_ultrasound[idx]
        show_image(image,ax[i,0],None)
        ax[i,0].set_title(f'Original Image')
        ax[i,0].axis('off')
        show_image(image,ax[i,1],n_val_transform) #shows the image
        ax[i,1].set_title('Transformed Image(downsampled to resize)')
        ax[i,1].axis('off')
        mask = get_mask(image,n_model,n_val_transform,DEVICE)
        ax[i,2].imshow(mask,cmap='gray')
        ax[i,2].set_title('Predicted Mask')
        ax[i,2].axis('off')
        # break
    plt.tight_layout()
    plt.show()
    
    # acc,val_loss = check_accuracy_binary(val_loader,n_model,device=DEVICE)
    # n_avg_val_acc.append(acc)
    # n_avg_val_loss.append(val_loss)
    # if acc > best_acc:
    #     best_acc = acc
    #     checkpoint = {
    #         "state_dict":n_model.state_dict(),
    #         "optimzer":n_optimizer.state_dict()
    #     }
    #     save_model_checkpoint(checkpoint,filename="fine_tuned.pth.tar")
    #     trigger = 0
    # else:
    #     trigger +=1
    # if trigger > patience:
    #     print(f'=> EARLY STOPPING')
    #     break 




