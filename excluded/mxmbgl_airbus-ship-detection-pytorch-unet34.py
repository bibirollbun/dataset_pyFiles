import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet34 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


PATH = './'
TRAIN = '../input/airbus-ship-detection/train_v2'
TEST = '../input/airbus-ship-detection/test_v2'
SEGMENTATION = '../input/airbus-ship-detection/train_ship_segmentations_v2.csv'
PRETREINED = '../input/fine-tuning-resnet34-on-ship-detection/models/Resnet34_lable_256_1.h5'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']


nw = 2
arch = resnet34


train_names = [i for i in os.listdir(TRAIN)]
test_names = [i for i in os.listdir(TEST)]

for i in exclude_list:
    if(i in train_names): train_names.remove(i)
    if(i in test_names): test_names.remove(i)

train_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)
segmentation_df = pd.read_csv(os.path.join(PATH, SEGMENTATION)).set_index('ImageId')


def cut_empty(names):
    return [name for name in names if(type(segmentation_df.loc[name]['EncodedPixels']) != float)]
    
tr_n_cut = cut_empty(train_n)
val_n_cut = cut_empty(val_n)
print(len(tr_n_cut),len(val_n_cut))


def get_mask(img_id, df):
    shape=(768,768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return img.reshape(shape) 
    if(type(masks) == str): masks = [masks]
    for mask in masks:      
        s = mask.split()
        for i in range(len(s)//2): 
            start = int(s[2*i]) - 1 
            length = int(s[2*i + 1]) 
            img[start:start+length] = 1
    return img.reshape(shape).T 


image_transform = transforms.Compose([
    transforms.Resize((768,768), interpolation=Image.BILINEAR),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((768,768), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x>0).float())
])

test_transform = transforms.Compose([
    transforms.Resize((768, 768)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

class ShipDataset(Dataset):
    def __init__(self, fnames, path, segmentation_df, transform=None, mask_transform=None,is_test=False):
        self.fnames = fnames 
        self.path = path 
        self.segmentation_df = segmentation_df
        self.transform = transform 
        self.mask_transform = mask_transform 
        self.is_test = is_test

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx): 
        img_name = self.fnames[idx]
        img_path = os.path.join(self.path, img_name)

        img = Image.open(img_path).convert('RGB')
   
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, img_name

        if self.segmentation_df is None:
            mask = np.zeros((768,768), dtype=np.uint8) 
        else:
            mask = get_mask(img_name, self.segmentation_df) 
            
        #print("Ğ”Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ğ¹:", np.unique(mask))
        mask = Image.fromarray(mask) 
        if self.mask_transform:
            mask = self.mask_transform(mask)
        #print("ĞŸĞ¾Ñ�Ğ»Ğµ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ğ¹:", mask.min(), mask.max())
            
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze()

        return img, mask


def get_data(sz, bs, image_transform=None, mask_transform=None):
    train_names = train_n if (len(tr_n_cut) % bs == 0) else train_n[:-(len(tr_n_cut) % bs)] # Ğ�Ğ±Ñ€ĞµĞ·Ğ°ĞµĞ¼ Ğ½ĞµĞ¿Ğ¾Ğ»Ğ½Ñ‹Ğ¹ Ğ±Ğ°Ñ‚Ñ‡
    val_names = val_n_cut

    train_dataset = ShipDataset(fnames=tr_n_cut, path=TRAIN, segmentation_df=segmentation_df, transform=image_transform, mask_transform=mask_transform)
    val_dataset = ShipDataset(fnames=val_n_cut, path=TRAIN, segmentation_df=segmentation_df, transform=image_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw)

    return train_loader, val_loader


def get_base(cut=8):
    base_model = models.resnet34(pretrained=False)    
    layers = list(base_model.children())[:cut]
    return nn.Sequential(*layers)

def load_pretrained(model, path):
    weights = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(weights, strict=False)
    return model


class UnetBlock(nn.Module):

    def __init__(self, up_in, x_in, n_out):
        super().__init__()

        up_out = x_out = n_out//2
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2) 
        self.x_conv = nn.Conv2d(x_in, x_out, 1) 
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p,x_p):
        up_p = self.tr_conv(up_p) 
        x_p = self.x_conv(x_p) 
        cat_p = torch.cat([up_p, x_p], dim=1) 
        return self.bn(F.relu(cat_p)) 


class SaveFeatures():
    features = None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn) 
    def hook_fn(self, module, input, output):
        self.features = output
    def remove(self):
        self.hook.remove 

class Unet34(nn.Module):
        def __init__(self, rn):
            super().__init__()
            self.rn = rn 
            self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]] 
    
            self.up1 = UnetBlock(512,256,256)
            self.up2 = UnetBlock(256,128,256)
            self.up3 = UnetBlock(256,64,256)
            self.up4 = UnetBlock(256,64,256)
            self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

        def forward(self, x):
            x = F.relu(self.rn(x)) 
            x = self.up1(x, self.sfs[3].features)
            x = self.up2(x, self.sfs[2].features)
            x = self.up3(x, self.sfs[1].features)
            x = self.up4(x, self.sfs[0].features)
            
            x = self.up5(x)
            return x[:,0]

        def close(self):
            for sf in self.sfs: sf.remove()       


def dice_loss(input, target):

    input = torch.sigmoid(input)
    smooth = 1.0 

    iflat = input.view(-1)
    tflat = target.view(-1)

    intersection = (iflat*tflat).sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma 
    def forward(self, input, target): 
        if not(target.size() == input.size()): 
            raise ValueError('Target size ({}) must be the same sa input size ({})'.format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input*target+max_val+((-max_val).exp() + (-input - max_val).exp()).log()
        invporbs = F.logsigmoid(-input*(target*2.0-1.0))
        loss = (invporbs*self.gamma).exp()*loss
        return loss.mean()


class MixedLoss(nn.Module):
        def __init__(self, alpha, gamma):
            super().__init__()
            self.alpha = alpha 
            self.focal = FocalLoss(gamma) 
        
        
        def forward(self, input, target):
            loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
            return loss.mean()


def dice(pred, targs):
    pred = (pred > 0).float()
    return 2.0*(pred*targs).sum()/((pred+targs).sum()+1.0)

def Iou(pred, targs):
    pred = (pred > 0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet34(get_base()).to(device)
model = load_pretrained(model, PRETREINED)
sz = 768
bs = 16
train_loader, val_loader = get_data(sz, bs, image_transform=image_transform, mask_transform=mask_transform)

for image, mask in train_loader:
    print(mask.max(), mask.min())
    break
for img, mask in train_loader:
    plt.imshow(mask[0].cpu().numpy(), cmap='gray')
    plt.show()
    img = img[0].cpu().numpy().transpose((1,2,0))
    #img = (img*255).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    break
    
criterion = MixedLoss(10.0, 2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)


weights = torch.load(PRETREINED, map_location='cpu')

renamed_weights = {f"rn.{k}": v for k, v in weights.items()}

missing, unexpected = model.load_state_dict(renamed_weights, strict=False)

print("ğŸ”� Ğ�Ğµ Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶ĞµĞ½Ñ‹ (missing):")
for key in missing:
    print("  -", key)

print("\nâ�— Ğ›Ğ¸ÑˆĞ½Ğ¸Ğµ (unexpected):")
for key in unexpected:
    print("  -", key)

model_dict = model.state_dict()
filtered_weights = {k: v for k, v in renamed_weights.items() if k in model_dict and v.size() == model_dict[k].size()}

model_dict.update(filtered_weights)
model.load_state_dict(model_dict)


print(list(renamed_weights.keys())[:10])  
print([k for k, _ in model.named_parameters()][:10])


def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)

        for images, masks in train_loader_tqdm:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

        validate(model, val_loader, criterion)

def validate(model, val_loader, criterion):
    model.eval()
    mix_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            mix_loss += loss.item()
            dice_score += dice(outputs, masks).item()
            iou_score += Iou(outputs, masks).item()

    avg_mix_loss = mix_loss / len(val_loader)  
    avg_dice = dice_score / len(val_loader)
    avg_iou = iou_score / len(val_loader)
    
    print(f'Validation Loss: {avg_mix_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}')
    
    return avg_mix_loss, avg_dice, avg_iou



# for param in model.encoder[:1].parameters():# Ğ—Ğ°Ğ¼Ğ¾Ñ€Ğ°Ğ¶Ğ¸Ğ²Ğ°ĞµĞ¼ Ñ�Ğ°Ğ¼Ñ‹Ğµ Ğ¿ĞµÑ€Ğ²Ñ‹Ğµ Ñ�Ğ»Ğ¾Ğ¸
#     param.requires_grad = False

train(model, train_loader, val_loader, criterion, optimizer, epochs=15)


torch.save(model, '/kaggle/working/trained_model.pth')


#model = torch.load('/kaggle/working/trained_model.pth')


!ls /kaggle/working/


def Show_images(x, yp, yt):
    columns = 3
    rows = min(bs,8)
    fig = plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        # ĞŸĞµÑ€ĞµÑ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ° Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹ Ğ´Ğ»Ñ� Ğ¾Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�: (C, H, W) -> (H, W, C)
        img_x = x[i].permute(1, 2, 0).cpu().numpy() if x[i].dim() == 3 else x[i].cpu().numpy()
        img_yp = yp[i].permute(1, 2, 0).cpu().numpy() if yp[i].dim() == 3 else yp[i].cpu().numpy()
        img_yt = yt[i].permute(1, 2, 0).cpu().numpy() if yt[i].dim() == 3 else yt[i].cpu().numpy()
        
        fig.add_subplot(rows, columns, 3 * i + 1)
        plt.axis('off')
        plt.imshow(img_x)
        plt.title('Input')

        fig.add_subplot(rows, columns, 3 * i + 2)
        plt.axis('off')
        plt.imshow(img_yp)
        plt.title('Prediction')

        fig.add_subplot(rows, columns, 3 * i + 3)
        plt.axis('off')
        plt.imshow(img_yt)
        plt.title('Ground Truth')

    plt.tight_layout()
    plt.show()
    


model.eval()
x, y = next(iter(val_loader))
x = x.to(device)
y = y.to(device)
with torch.no_grad():
    yp = torch.sigmoid(model(x))


def denormalize(tensor, mean, std):
    if tensor.ndimension() != 4:
        raise ValueError(f"Ğ�Ğ¶Ğ¸Ğ´Ğ°ĞµÑ‚Ñ�Ñ� Ñ‚ĞµĞ½Ğ·Ğ¾Ñ€ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ğ½Ğ¾Ñ�Ñ‚Ğ¸ (B, C, H, W), Ğ½Ğ¾ Ğ¿Ğ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ¾ {tensor.shape}")

    device = tensor.device

    if tensor.is_sparse:
        tensor = tensor.to_dense()

    mean = torch.as_tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, device=device).view(1, -1, 1, 1)

    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)

    return denorm


mean = [0.485, 0.456, 0.406]  
std = [0.229, 0.224, 0.225]

xt = denormalize(x, mean, std)
print(f"Denormalized tensor shape: {xt.shape}")

Show_images(xt, y, yp)


subset_fnames = random.sample(test_names, 100)
test_dataset = ShipDataset(fnames=subset_fnames, path=TEST, segmentation_df=None, transform=test_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw)


def encode_mask(mask, shape=(768, 768)):
    if mask.shape != shape:        
        mask = np.array(Image.fromarray(mask).resize(shape))  
    #print(f"Resizing mask from {mask.shape} to {shape}")
    pixels = mask.T.flatten() 
    
    if len(pixels) != shape[0] * shape[1]:
        print(f"Warning: Unexpected mask shape {mask.shape}. Expected shape: {shape}")
    
    pixels = np.concatenate([pixels, [0]])  # Ğ¢Ğ¾Ğ»ÑŒĞºĞ¾ Ğ¾Ğ´Ğ¸Ğ½ 0 Ğ² ĞºĞ¾Ğ½Ñ†Ğµ, Ğ±ĞµĞ· Ğ´Ğ¾Ğ±Ğ°Ğ²Ğ»ĞµĞ½Ğ¸Ñ� Ğ»Ğ¸ÑˆĞ½ĞµĞ³Ğ¾ 0 Ğ² Ğ½Ğ°Ñ‡Ğ°Ğ»Ğµ
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    if len(runs) % 2 != 0:
        runs = np.concatenate([runs, [runs[-1]]])
    
    runs[1::2] -= runs[::2]  
    
    return ' '.join(str(x) for x in runs)


model.eval()
ship_list_dict = []

for inputs, img_names in test_loader:
    inputs = inputs.to(device)  

    with torch.no_grad():  
        outputs = model(inputs)  
    
    pred_masks = torch.sigmoid(outputs).cpu().numpy()

    for idx, img_name in enumerate(img_names):
        mask = pred_masks[idx]  
        mask = (mask > 0.5).astype(np.uint8)
        encoded_pixels = encode_mask(mask)  
        ship_list_dict.append({'ImageId': img_name, 'EncodedPixels': encoded_pixels})

pred_df = pd.DataFrame(ship_list_dict)
pred_df.to_csv('submission.csv', index=False)


if 'model' in locals() and model is not None:
    model.close()
    print("Ğ¥ÑƒĞºĞ¸ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ ÑƒĞ´Ğ°Ğ»ĞµĞ½Ñ‹, Ñ€ĞµÑ�ÑƒÑ€Ñ�Ñ‹ Ğ¾Ñ�Ğ²Ğ¾Ğ±Ğ¾Ğ¶Ğ´ĞµĞ½Ñ‹.")
    del model


path_to_images = '/kaggle/working/'
files = os.listdir(path_to_images)
print("Ğ¡Ğ¾Ğ´ĞµÑ€Ğ¶Ğ¸Ğ¼Ğ¾Ğµ Ğ¿Ğ°Ğ¿ĞºĞ¸:", files)

