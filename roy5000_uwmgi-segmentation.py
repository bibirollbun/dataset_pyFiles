!pip install -U segmentation-models-pytorch


import numpy as np
import pandas as pd
import cv2
from glob import glob
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/uwmgi-mask-dataset/train.csv")
df_copy = df.copy()

df_copy= df_copy.drop(['case','day','slice','image_path','height','width','mask_path'],axis=1)


df_copy.rename(columns= {'class':'classes'} , inplace=True)


df_copy['case'] = df_copy['id'].apply(lambda x : int(x.split('_')[0].replace("case" , "")))
df_copy['day'] = df_copy['id'].apply(lambda x : int(x.split('_')[1].replace('day',"")))
df_copy['slice'] = df_copy['id'].apply(lambda x : x.split('_')[3])


# train_images = glob(os.path.join("/kaggle/input/uwmgi-mask-dataset/png/uw-madison-gi-tract-image-segmentation/train","**","*.png"),recursive=True)


prefix = '/kaggle/input/uwmgi-mask-dataset/png/uw-madison-gi-tract-image-segmentation/train'


temporary_1 = []
for idx, row in df_copy.iterrows():

    temporary_1.append(os.path.join(prefix,"case"+str(row['case']),
                                   "case"+str(row['case'])+"_"+"day"+str(row['day']),
                                   'scans'))

df_copy['partial_path'] = temporary_1
df_copy


print(df_copy['id'].values[0])

print(df_copy['partial_path'].values[0])


temporary_1 = []

for idx,path in enumerate(df_copy['partial_path'].values):
    for file in glob(os.path.join(path,'slice_'+str(df_copy['slice'].values[idx])+"*.png")):
        temporary_1.append(file)

temporary_1[0]


df_copy['file_path'] = temporary_1
df_copy = df_copy.drop('partial_path',axis=1)
df_copy


df_copy['width'] = df_copy['file_path'].apply(lambda x : x.split('_')[3])
df_copy['height'] = df_copy['file_path'].apply(lambda x : x.split('_')[4])


# Here we take 3 step gap-gap row

train_df = pd.DataFrame({'id':df_copy['id'][::3]})
train_df['large_bowel'] = df_copy['segmentation'][::3].values
train_df['small_bowel'] = df_copy['segmentation'][1::3].values
train_df['stomach'] = df_copy['segmentation'][2::3].values
train_df['path'] = df_copy['file_path'][::3]
train_df['case'] = df_copy['case'][::3]
train_df['day'] = df_copy['day'][::3]
train_df['slice'] = df_copy['slice'][::3]
train_df['width'] = df_copy['width'][::3]
train_df['height'] = df_copy['height'][::3]


train_df.reset_index(inplace=True , drop=True)


train_df['counts'] = np.sum(train_df.loc[: , ['large_bowel','small_bowel','stomach']] != '',axis=1)
train_df['counts'].values


train_df['small_bowel'].unique()


null_cols = train_df.iloc[: , 1:4].isnull().sum().reset_index()
null_cols.rename(columns={'index':'column_type',0:'total_null'},inplace=True)

fig , ax = plt.subplots(1 ,1 , figsize=(15,5))
sns.barplot(null_cols , x='column_type',y='total_null',palette='coolwarm',edgecolor='black',ax=ax)
ax.set_title("Number of Null Bar")
for bar in ax.patches:
    ax.annotate(f"{bar.get_height():.2f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',
               va='bottom',fontsize=10,fontweight='bold')
plt.show()


fig , ax = plt.subplots(1,1,figsize=(15,5))
sns.countplot(x='counts',data=train_df,ax=ax)

for p in ax.patches:
    ax.annotate(f"{p.get_height()}",(p.get_x()+p.get_width()/2,
               p.get_height()),color='black',fontweight='bold',fontsize=10,
                 ha='center', va='bottom'
               )

ax.set_xlabel('count')
plt.show()


height_width_df = train_df.iloc[:,8:].value_counts().reset_index()

fig,ax=plt.subplots(1,1,figsize=(15,5))
sns.barplot(height_width_df,x=height_width_df.index.values,y='count',palette='inferno',edgecolor='black',ax=ax)

for bins,(index,row) in zip(ax.patches,height_width_df.iterrows()):
    ax.annotate(f"W:{row[0]} x H:{row[1]}",
                (bins.get_x()+bins.get_width()/2,bins.get_height()),
                ha='center',va='bottom',fontweight='bold'
               )
ax.set_title("Width x Height plot")
ax.set_xticklabels([])
plt.show()


width_count = train_df.iloc[: , 8:9].value_counts().reset_index()
height_count = train_df.iloc[:,9:10].value_counts().reset_index()

fig , ax = plt.subplots(1 , 2 , figsize=(10 , 6))

axes = ax.flatten()
sns.barplot(height_count,x='height',y='count',edgecolor='black',palette='coolwarm',ax=axes[0])
ax[0].set_title("Height Counts")
sns.barplot(width_count,x='width',y='count',edgecolor='black',palette='coolwarm',ax=axes[1])
ax[1].set_title("Width Counts")

fig.suptitle("Individual Heigth and Width Counts",color='Blue',fontsize=10)
plt.show()


train_df['organ_annotation_count'] = train_df[['large_bowel','small_bowel','stomach']].notnull().sum(axis=1)
train_df['organ_annotation_count'] = train_df['organ_annotation_count'].apply(lambda x : int(x))
sns.histplot(train_df , x='organ_annotation_count',kde=True)
plt.show()


case_slices = train_df.groupby(['case']).agg(slice_count = ('slice','count')).reset_index()
fig , ax = plt.subplots(1 , 1 , figsize=(15,10))
sns.barplot(case_slices , x='case',y='slice_count',ax=ax)
ax.set_title("Number of Slices per case",fontsize=10, color='blue')
ax.set_xlabel("Case")
ax.set_ylabel("Slice Count")
plt.show()



train_df.fillna('', inplace=True) # This is important if we use notna() technique we've to deal with NaN


train_df[train_df['large_bowel']!=''].index


train_mask = list(train_df[train_df['large_bowel']!=''].index)
train_mask += list(train_df[train_df['small_bowel']!=''].index)
train_mask += list(train_df[train_df['stomach']!=''].index)


mydf = train_df[train_df.index.isin(train_mask)] # Only train_df.index.isin(train_mask) gives True,False... boolean array
mydf.reset_index(inplace=True,drop=True)
mydf.shape


from sklearn.model_selection import StratifiedGroupKFold


skf = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)

for fold, (_, val_idx) in enumerate(skf.split(X=mydf, y=mydf['counts'],groups =mydf['case']), 1):
    # y: This specifies the target variable used for stratification.
    mydf.loc[val_idx, 'fold'] = fold


train_ids = mydf[mydf["fold"]!=1].index
validation_ids = mydf[mydf["fold"]==1].index


folds_information = mydf.groupby(['fold','counts'])['id'].count().reset_index()
folds_information['fold'] = folds_information['fold'].apply(lambda x : int(x))
folds_information.rename(columns={'id':'total'},inplace=True)

fig , ax = plt.subplots(1,1,figsize=(15,5))
sns.lineplot(folds_information,x='fold',y='total',hue='counts',palette='coolwarm',marker='o',ax=ax)
for index,row in folds_information.iterrows():
    ax.annotate(f"{row['total']}",(row['fold'],row['total']),fontweight='bold',fontsize=10)
plt.title("Total Count Per Fold", fontsize=14, fontweight='bold')
plt.xlabel("Fold", fontsize=12)
plt.ylabel("Total", fontsize=12)
plt.legend(title="Counts")
plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import models
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from tqdm import tqdm
import copy


mydf.head()


def rle2mask(mask_rle,shape:tuple):

    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    Source: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    '''
    s = mask_rle.split()
    starts,lengths = [np.asarray(x , dtype=int) for x in (s[0:][::2] , s[1:][::2])] # "10 5 25 10 50 5"
    starts -= 1 # zero based indexing
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1] , dtype=np.uint8)
    for lo,hi in zip(starts , ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# def mask2rle(img):
#     pixels = img.T.flatten()
#     pixels = np.pad(pixels,((1,1),))
#     runs = np.where(pixels[1:] != pixels[:-1])[0]+1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)

# rle = "10 5 25 10 50 5"
# mask = rle2mask(rle,(10,10))
# print(mask2rle(mask))


class MyDataset(Dataset):
    def __init__(self,df,whatset='train'):
        self.df = df.reset_index(drop=True)
        self.whatset = whatset
        self.img_shape = (224,224)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        img_path = self.df['path'][idx]
        img = self.load_img(img_path)
        if self.whatset == 'train':
            mask = self.load_mask(idx)
            return torch.tensor(img) , torch.tensor(mask)

        return torch.tensor(img)
        
    def rle2mask(self,mask_rle):
        s = mask_rle.split()
        start,length = [np.asarray(x , dtype=int) for x in (s[:][::2] , s[1:][::2])]
        start -= 1 # 0 based indexing
        end = start+length
        # Careful here i'm taking a linear zero array and later we rehape it
        img = np.zeros(self.img_shape[0]*self.img_shape[1],dtype=np.uint8) # Img not mask
        for lo,hi in (zip(start,end)):
            img[lo:hi] = 1
        return img.reshape(self.img_shape).T
        
    def load_img(self,img):
        img = cv2.imread(img , cv2.IMREAD_UNCHANGED)
        img = cv2.normalize(img,None,0,255,norm_type=cv2.NORM_MINMAX)
        img = cv2.resize(img,self.img_shape)
        if len(img.shape) == 2:
           img = np.repeat(img[...,None],3,axis=-1)
        return img.transpose(2,0,1).astype(np.float32)/255.0

    def load_mask(self,idx):
        """Loads and decodes RLE masks for each class."""
        masks = np.zeros((*self.img_shape,3),dtype=np.float32) # masks pixels always float
        for channel,label in enumerate(['large_bowel','small_bowel','stomach']):
            rle = self.df[label].iloc[idx]
            if not isinstance(rle,str):
                print(f"Invalid RLE at index {idx}, label {label}: {rle}")
                raise ValueError("NOT GOOD")
                
            mask = self.rle2mask(self.df[label].iloc[idx])
            masks[:,:,channel] = mask.astype(np.float32)
        return masks.transpose(2,0,1)


train_transforms = transforms.Compose([
    transforms.Resize(224,interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(brightness=0.2)
    ],p=0.2)
])
validation_transforms = transforms.Compose([
    transforms.Resize(224 , interpolation=transforms.InterpolationMode.NEAREST)
])


cpuCount  = os.cpu_count()
torch.set_num_threads(cpuCount)
torch.set_num_interop_threads(cpuCount) 


train_dataset = MyDataset(train_df[train_df.index.isin(train_ids)],whatset="train")
val_dataset =  MyDataset(train_df[train_df.index.isin(validation_ids)],whatset="validation")

train_loader = DataLoader(train_dataset , batch_size=32,num_workers=cpuCount,shuffle=True,pin_memory=True,drop_last=False)
val_loader = DataLoader(val_dataset , batch_size=64,num_workers=cpuCount,shuffle=False,pin_memory=True)


import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def show_img(img,mask):
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img = clahe.apply(img)
    plt.imshow(img,cmap='bone')
    if mask is not None:
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        cmap = ListedColormap(["red","green","blue"])
        plt.imshow(mask,cmap=cmap,alpha=0.5)
        handles = [mpatches.Patch(color=c,label=l) for c,l in zip(["red","green","blue"],["Large Bowel", "Small Bowel", "Stomach"])]
        plt.legend(handles=handles)
 
    plt.axis('off')
    
def group_plot(imgs,masks,size):
    plt.figure(figsize=(size * 5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx + 1)
        # print(imgs[idx].shape) # shaop is 3, 224, 224
        img = imgs[idx].permute((1,2,0)).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        msk = masks[idx].permute((1,2,0)).cpu().numpy()
        msk = (msk * 255).astype(np.uint8)
        show_img(img,msk)
    plt.tight_layout()
    plt.show()

imgs, masks = next(iter(train_loader))
print("Image Shape:", imgs.shape, "Mask Shape:", masks.shape)
group_plot(imgs,masks,5)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.double_conv(X)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, X):
        return self.down(X)

class Up(nn.Module):

    """
    Bilinear Upsampling (nn.Upsample):
    Does not change channels → Needs extra processing (Conv2d) to reduce channels.
    We manually reduce channels and concatenate with skip connection, so we need +out_channels.
    
    Transposed Convolution (nn.ConvTranspose2d):
    Automatically reduces channels from in_channels to in_channels // 2	.
    It directly matches the expected input for DoubleConv, so no need for +out_channels.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # Reduce the number of channels using a 1x1 convolution
            self.reduce_channels = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, X1, X2):
        X1 = self.up(X1)
        X1 = self.reduce_channels(X1) if hasattr(self, 'reduce_channels') else X1  # Apply channel reduction if bilinear
        diffY = X2.size()[2] - X1.size()[2]
        diffX = X2.size()[3] - X1.size()[3]
        X1 = F.pad(X1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        out = torch.cat([X2, X1], dim=1)
        return self.conv(out)


class Out(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_classes, kernel_size=1),
            nn.Sigmoid()  # Use nn.Softmax(dim=1) for multi-class segmentation
        )

    def forward(self, X):
        return self.out(X)

class UNet(nn.Module):
    def __init__(self, in_channels, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.conv = DoubleConv(in_channels=in_channels, out_channels=n_channels)
        self.enc1 = Down(in_channels=n_channels, out_channels=2 * n_channels)
        self.enc2 = Down(in_channels=2 * n_channels, out_channels=4 * n_channels)
        self.enc3 = Down(in_channels=4 * n_channels, out_channels=8 * n_channels)
        self.enc4 = Down(in_channels=8 * n_channels, out_channels=16 * n_channels)

        self.dec1 = Up(in_channels=16 * n_channels, out_channels=8 * n_channels, bilinear=bilinear)
        self.dec2 = Up(in_channels=8 * n_channels, out_channels=4 * n_channels, bilinear=bilinear)
        self.dec3 = Up(in_channels=4 * n_channels, out_channels=2 * n_channels, bilinear=bilinear)
        self.dec4 = Up(in_channels=2 * n_channels, out_channels=n_channels, bilinear=bilinear)

        self.out = Out(in_channels=n_channels, n_classes=n_classes)

    def forward(self, X):
        X1 = self.conv(X)
        X2 = self.enc1(X1)
        X3 = self.enc2(X2)
        X4 = self.enc3(X3)
        X5 = self.enc4(X4)

        X = self.dec1(X5, X4)
        X = self.dec2(X, X3)
        X = self.dec3(X, X2)
        X = self.dec4(X, X1)

        return self.out(X)


x = torch.randn(1, 3, 224, 224)  # Example input
print(tuple(x.size()))
model = UNet(in_channels=3, n_classes=3, n_channels=48, bilinear=False)
summary(model, (3, 224, 224))


device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_model():
    model = UNet(in_channels=3, n_classes=3, n_channels=48)
#     model = DeepLabV3Plus(num_classes=3)
    model.to(device)
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


DiceLoss = smp.losses.DiceLoss(mode="multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()

def dice_coef(y_true,y_pred,epsilon=0.001,dim=(2,3),thr=0.5):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32) # (y_pred > thr) gives boolean array and .to(torch.float32) converts them to 1. and 0.
    intersec = (y_true * y_pred).sum(dim=dim)
    denominator = y_true.sum(dim=dim)+y_pred.sum(dim=dim)
    dice = ((2*(intersec+epsilon))/(denominator+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true,y_pred,epsilon=0.001,dim=(2,3),thr=0.5):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    intersec = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((intersec+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def mylossfn(y_pred,y_true):
    return 0.6 * BCELoss(y_pred, y_true) + 0.4 * DiceLoss(y_pred, y_true)

y_true = torch.randint(0,2,(4, 3, 224, 224)).float()
y_pred = torch.randn(4, 3, 224, 224)

# Compute Dice and IoU metrics
dice_score = dice_coef(y_true, torch.sigmoid(y_pred))  # Apply sigmoid for probability output
iou_score = iou_coef(y_true, torch.sigmoid(y_pred))

myloss_score = mylossfn(y_true,torch.sigmoid(y_pred))

print("My Loss Score :",myloss_score.item())
print("Dice Score :", dice_score.item())
print("IoU Score :", iou_score.item())


start = time.time()
history = {}

for epoch in range(1 , epochs+1):
    gc.collect()

    model.train()
    print(f"EPOCH:{epoch}/{epochs}")
    running_loss = 0.0
    dataset_size = 0
    scaler = amp.GradScaler()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader),desc="Training:")
    for step,(images,masks) in pbar:
        images = images.to(device,dtype=torch.float)
        masks = masks.to(device,dtype=torch.float)
        batch_size=images.size()[0]

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = mylossfn(y_pred,masks) / n_accumulate

        scaler.scale(loss).backward()
        if (step+1) % n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if schedular is not None:
                schedular.step()

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = runnung_loss / dataset_size
        mem = torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(Epoch_Loss=f"{epoch_loss:.4f}",lr=f"{lr:.4f}"Memory=f"{mem:.4f}")
        
    history["Train Loss"].append(epoch_loss)

    model.eval()
    running_loss = 0.0
    dataset_size=0
    val_scores = []
    pbar = tqdm(enumerate(val_loader),total=len(val_loader),desc="Validating:")
    for step , (images,masks) in pbar:
        images = images.to(device,torch.float)
        masks = masks.to(device,torch.float)
        batch_size = images.size()[0]

        with torch.no_grad():
            y_pred = model(images)
            loss = mylossfn(y_pred,masks)

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        y_pred = torch.sigmoid(y_pred)
        val_dice = dice_coef(masks,y_pred).cpu().numpy()
        val_jaccard = iou_coef(masks,y_pred).cpu().numpy()

        val_scores.append([val_dice , val_jaccard])
        lr = optimizer.param_groups[0]['lr']
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(Epoch_Loss=f"{epoch_loss:.4f}",lr=f"{lr:.4f}"Memory=f"{mem:.4f}")

    val_dice,val_jaccard = np.mean(val_scores,axis=0)
    history["Valid Loss"].append(epoch_loss)
    history["Valid Dice"].append(val_dice)
    history["val_jaccard"].append(val_jaccard)

    if val_dice > best_dice:
         best_dice = val_dice
         best_jaccard = val_jaccard
         best_epoch = epoch
         best_model_wts = copy.deepcopy(model.state_dict())
         torch.save(best_model_wts,f"best_epochdeeplab-{fold:02d}.bin")
         print("Best model saved.")
    torch_save(model.state_dict(),f"last_epochdeeplab-{fold:02d}.bin")

end = time.time()
elapsed_time = start-end
print("Training Complete in : {:.0f}h {:.0f}m {:.0f}s".format(elapsed_time//3600,(elapsed_time%3600)//60),(elapsed_time%3600)//60)
print("Best Dice Score: {:.4f} | Best Jaccard Score: {:.4f}".format(best_dice, best_jaccard))
model.load_state_dict(best_model_wts)
    


# LEARNING PURPOSE ------->

# from sklearn.model_selection import KFold,StratifiedKFold,StratifiedGroupKFold,GroupKFold,TimeSeriesSplit
# np.random.seed(42)
# sample_data = pd.DataFrame({
#     'id': range(1,21),
#     'feature1':np.random.rand(20),
#     'feature2':np.random.rand(20),
#     'target':np.random.choice([0,1],size=20,p=[0.7,0.3]),
#     'group': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=20)   
# })

# X = sample_data.iloc[:,1:3]
# y = sample_data.loc[:,'target']
# groups = sample_data['group']

# # Use case: General cross-validation, no need for stratification or grouping.
# kf = KFold(n_splits=5 , shuffle=True,random_state=42)
# for fold , (train_idx, test_idx) in enumerate(kf.split(X) , 1):
#     print(f"Fold {fold}:")
#     print("Train indices:", train_idx)
#     print("Test indices:", test_idx, "\n")

# print("-"*80)

# #  Use case: Ensures each fold has a similar class distribution.
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# for fold ,(train_idx , test_idx) in enumerate(skf.split(X , y) , 1):
#     print(f"Fold {fold}:")
#     print("Train target distribution:",y.iloc[train_idx].value_counts().to_dict())
#     print("Test target distribution:",y.iloc[test_idx].value_counts().to_dict(),"\n")

# print("-"*80)

# # Use case: Ensures all data from the same group stays together.
# gkf = GroupKFold(n_splits=3)

# for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
#     print(f"Fold {fold}:")
#     print("Train groups:", groups.iloc[train_idx].unique())
#     print("Test groups:", groups.iloc[test_idx].unique(), "\n")

# print("-"*80)

# # Use case: Maintains both class distribution and group integrity.
# sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

# for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
#     print(f"Fold {fold}:")
#     print("Train target distribution:", y.iloc[train_idx].value_counts().to_dict())
#     print("Test target distribution:", y.iloc[test_idx].value_counts().to_dict())
#     print("Train groups:", groups.iloc[train_idx].unique())
#     print("Test groups:", groups.iloc[test_idx].unique(), "\n")

# print("-"*80)

# #Use case: Ensures training data always precedes test data (for time-series forecasting).
# tscv = TimeSeriesSplit(n_splits=5)

# for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
#     print(f"Fold {fold}:")
#     print("Train indices:", train_idx)
#     print("Test indices:", test_idx, "\n")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Define a simple model
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a GradScaler for gradient scaling
scaler = GradScaler()

# Dummy data
inputs = torch.randn(32, 128).cuda()  # Batch size 32, input size 128
targets = torch.randint(0, 10, (32,)).cuda()  # 10 classes

# Training loop
for epoch in range(10):
    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast(enabled=True):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()

    # Update weights
    scaler.step(optimizer)

    # Update the scale for next iteration
    scaler.update()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


import torch

def auto_n_accumulate(model, batch_size_per_step, desired_batch_size):
    """Automatically find n_accumulate based on GPU memory."""
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    reserved_memory = torch.cuda.memory_reserved(0) / 1e9  # GB
    available_memory = total_memory - reserved_memory  # GB

    print(f"GPU Memory: {available_memory:.2f} GB available out of {total_memory:.2f} GB")

    # If memory is sufficient, reduce accumulation
    if available_memory > 5:  # Adjust threshold as needed
        return max(1, desired_batch_size // batch_size_per_step)
    else:
        return max(1, (desired_batch_size // batch_size_per_step) // 2)  # Use smaller accumulation if memory is tight

# Example usage
batch_size_per_step = 8
desired_batch_size = 64
n_accumulate = auto_n_accumulate(model, batch_size_per_step, desired_batch_size)
print("Dynamically chosen n_accumulate:", n_accumulate)



len(range(1,30000))



import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

data = {
    "feature1": [1.0, 4.0, 7.0, 10.0, 13.0, 16.0],
    "feature2": [2.0, 5.0, 8.0, 11.0, 14.0, 17.0],
    "feature3": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
    "label":    [0, 1, 0, 1, 0, 1]  # Classification labels (0 or 1)
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)



# Convert DataFrame to NumPy arrays
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

# Define a simple Dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define a simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, 2)  # 2 output classes
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc(self.relu(x))

# K-Fold Cross-Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training Fold {fold}")

    # Split data
    train_dataset = MyDataset(X[train_idx], y[train_idx])
    val_dataset = MyDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Model, Loss, Optimizer
    model = SimpleNN(input_size=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(5):  # 5 epochs for demonstration
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the model for this fold
    torch.save(model.state_dict(), f"model_fold-{fold}.pth")
    print(f"Saved model for fold {fold}\n")



X = torch.tensor([[1,2,3],
                  [4,5,6]],dtype=torch.float32)
X[0]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.random.rand(100, 3) * 10  # 100 samples, 3 features
y = 3 * X[:, 0] + 2 * X[:, 1] - 5 * X[:, 2] + np.random.randn(100) * 2  # Linear relation with noise

print(y.shape)
# Create DataFrame
df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
df["target"] = y

# Extract features and target
X = df.drop(columns=["target"]).values
y = df["target"].values.reshape(-1, 1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
 
# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create PyTorch Dataset
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = RegressionDataset(X_tensor, y_tensor)

# Define PyTorch Model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Simple Linear Regression

    def forward(self, x):
        return self.fc(x)

# Training Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 50
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perform K-Fold Cross-Validation
mae_scores = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor), 1):
    print(f"Fold {fold}/{kf.get_n_splits()}")

    # Create DataLoaders
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model, Loss, Optimizer
    model = SimpleNN(input_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    total_mae = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            total_mae += torch.abs(y_pred - y_batch).mean().item()

    mae = total_mae / len(test_loader)
    mae_scores.append(mae)
    print(f"Fold {fold} MAE: {mae:.4f}")

# Print average MAE
print(f"Average MAE: {sum(mae_scores) / len(mae_scores):.4f}")


