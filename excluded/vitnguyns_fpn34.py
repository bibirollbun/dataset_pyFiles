from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split


PATH = './'
TRAIN = '../input/airbus-ship-detection/train_v2'
TEST = '../input/airbus-ship-detection/test_v2'
SEGMENTATION = '../input/airbus-ship-detection/train_ship_segmentations_v2.csv'
PRETREINED = '../input/fine-tuning-resnet34-on-ship-detection/models/Resnet34_lable_256_1.h5'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']


nw = 2   #number of workers for data loader
arch = resnet34 #specify target architecture


train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
for el in exclude_list:
    if(el in train_names): train_names.remove(el)
    if(el in test_names): test_names.remove(el)
#5% of data in the validation set is sufficient for model evaluation
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)
segmentation_df = pd.read_csv(os.path.join(PATH, SEGMENTATION)).set_index('ImageId')


def cut_empty(names):
    return [name for name in names 
            if(type(segmentation_df.loc[name]['EncodedPixels']) != float)]

tr_n = cut_empty(tr_n)
val_n = cut_empty(val_n)


def get_mask(img_id, df):
    shape = (768,768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return img.reshape(shape)
    if(type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    return img.reshape(shape).T


class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        mask = np.zeros((768,768), dtype=np.uint8) if (self.path == TEST) \
            else get_mask(self.fnames[i], self.segmentation_df)
        img = Image.fromarray(mask).resize((self.sz, self.sz)).convert('RGB')
        return np.array(img).astype(np.float32)
    
    def get_c(self): return 0


class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  #add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x


def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.CLASS),
                RandomDihedral(tfm_y=TfmType.CLASS),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, 
                aug_tfms=aug_tfms)
    tr_names = tr_n if (len(tr_n)%bs == 0) else tr_n[:-(len(tr_n)%bs)] #cut incomplete batch
    ds = ImageData.get_ds(pdFilesDataset, (tr_names,TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    md.is_multi = False
    return md


cut,lr_cut = model_meta[arch]


def get_base():                   #load ResNet34 model
    layers = cut_model(arch(True), cut)
    return nn.Sequential(*layers)

def load_pretrained(model, path): #load a model pretrained on ship/no-ship classification
    weights = torch.load(PRETRAINED, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights, strict=False)
            
    return model


def dice_loss(input, target):
    input = torch.sigmoid(input)
    
    # Đảm bảo target có shape [B, 1, H, W]
    if target.data.dim() == 3:
        target = target.unsqueeze(1)
    
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)



class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
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
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


from sklearn.metrics import precision_score, recall_score, f1_score

def precision(input, target, thresh=0.5):
    input = input.sigmoid() > thresh  # Chuyển thành nhị phân
    target = target.byte()
    preds = input.cpu().numpy().flatten()
    targs = target.cpu().numpy().flatten()
    try:
        return precision_score(targs, preds, zero_division=1)
    except TypeError:
        return precision_score(targs, preds)

def recall(input, target, thresh=0.5):
    input = input.sigmoid() > thresh
    target = target.byte()
    preds = input.cpu().numpy().flatten()
    targs = target.cpu().numpy().flatten()
    try:
        return recall_score(targs, preds, zero_division=1)
    except TypeError:
        return recall_score(targs, preds)

def f1(input, target, thresh=0.5):
    input = input.sigmoid() > thresh
    target = target.byte()
    preds = input.cpu().numpy().flatten()
    targs = target.cpu().numpy().flatten()
    try:
        return f1_score(targs, preds, zero_division=1)
    except TypeError:
        return f1_score(targs, preds)


sz = 256
bs = 8
md = get_data(sz, bs)



import torchvision.models as models

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        return self.conv2(x)

class FPN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        self.enc0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.enc1 = self.backbone.layer1
        self.enc2 = self.backbone.layer2
        self.enc3 = self.backbone.layer3
        self.enc4 = self.backbone.layer4
        
        # Conv1x1 để chuẩn hóa channel
        self.lat4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lat2 = nn.Conv2d(128, 256, kernel_size=1)
        self.lat1 = nn.Conv2d(64,  256, kernel_size=1)
        
        self.fpn4 = FPNBlock(256, 256)
        self.fpn3 = FPNBlock(256, 256)
        self.fpn2 = FPNBlock(256, 256)
        self.fpn1 = FPNBlock(256, 256)
        
        self.upconv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.enc0(x)  # 64
        c2 = self.enc1(c1) # 64
        c3 = self.enc2(c2) # 128
        c4 = self.enc3(c3) # 256
        c5 = self.enc4(c4) # 512

        # chuẩn hóa channel
        c5 = self.lat4(c5)
        c4 = self.lat3(c4)
        c3 = self.lat2(c3)
        c2 = self.lat1(c2)

        p4 = self.fpn4(c5)
        p3 = self.fpn3(F.upsample(p4, scale_factor=2, mode='nearest') + c4)
        p2 = self.fpn2(F.upsample(p3, scale_factor=2, mode='nearest') + c3)
        p1 = self.fpn1(F.upsample(p2, scale_factor=2, mode='nearest') + c2)

        out = F.upsample(p1, scale_factor=4, mode='nearest')
        return self.upconv(out).squeeze(1)


        
resnet34 = models.resnet34(pretrained=True)
fpn_model = FPN(resnet34)

# Bọc lại như UnetModel để hỗ trợ FastAI
class FPNWrapper():
    def __init__(self, model, name='FPN'):
        self.model = model
        self.name = name

    def get_layer_groups(self, precompute):
        # Encoder (resnet layers)
        encoder = nn.Sequential(
            self.model.enc0,
            self.model.enc1,
            self.model.enc2,
            self.model.enc3,
            self.model.enc4
        )
        # Decoder (fpn blocks + upconv)
        decoder = nn.Sequential(
            self.model.fpn4,
            self.model.fpn3,
            self.model.fpn2,
            self.model.fpn1,
            self.model.upconv
        )
        return [encoder, decoder]

model = FPNWrapper(fpn_model)
learn = ConvLearner(md, model)
learn.model.cuda()



learn.opt_fn=optim.Adam
learn.crit = MixedLoss(10.0, 2.0)
learn.metrics=[accuracy_thresh(0.5),dice,IoU,precision,recall,f1]
wd=1e-5
lr = 1e-4


learn.fit(lr,5,wds=wd,cycle_len=1,use_clr=(5,8))


learn.fit(lr,1,wds=wd,cycle_len=1,use_clr=(5,8))


x, y = next(iter(learn.data.val_dl))

# Chuyển ảnh từ GPU về CPU và chuẩn hóa
img = x[0].cpu().permute(1, 2, 0).numpy()
img = (img - img.min()) / (img.max() - img.min())  # chuẩn hóa về [0, 1]

mask = y[0].cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title("Mask")

plt.show()



learn.fit(lr,1,wds=wd,cycle_len=1,use_clr=(5,8))


sz = 256 #image size
bs = 64  #batch size

dls = get_data(sz,bs)


learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit = MixedLoss(10.0, 2.0)
learn.metrics=[accuracy_thresh(0.5),dice,IoU]
wd=1e-7
lr = 1e-2


learn.freeze_to(1)


learn.fit(lr,1,wds=wd,cycle_len=1,use_clr=(5,8))


learn.save('Unet34_256_0')


def Show_images_with_titles(x, yp, yt):
    columns = 3
    rows = min(len(x), 8)
    fig = plt.figure(figsize=(columns * 5, rows * 5))
    
    for i in range(rows):
        # Ảnh gốc
        ax1 = fig.add_subplot(rows, columns, 3*i + 1)
        ax1.axis('off')
        ax1.set_title('Original Image')
        ax1.imshow(x[i])

        # Mask dự đoán
        ax2 = fig.add_subplot(rows, columns, 3*i + 2)
        ax2.axis('off')
        ax2.set_title('Predicted Mask')
        ax2.imshow(yp[i], cmap='gray')

        # Mask ground truth
        ax3 = fig.add_subplot(rows, columns, 3*i + 3)
        ax3.axis('off')
        ax3.set_title('Ground Truth Mask')
        ax3.imshow(yt[i], cmap='gray')

    plt.tight_layout()
    plt.show()



learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))


Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)


lrs = np.array([lr/100,lr/10,lr])
learn.unfreeze() #unfreeze the encoder
learn.bn_freeze(True)


learn.fit(lrs,2,wds=wd,cycle_len=1,use_clr=(20,8))


learn.fit(lrs/3,2,wds=wd,cycle_len=2,use_clr=(20,8))


learn.sched.plot_lr()


learn.save('Unet34_256_1')


def Show_images(x,yp,yt):
    columns = 3
    rows = min(bs,8)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        fig.add_subplot(rows, columns, 3*i+1)
        plt.axis('off')
        plt.imshow(x[i])
        fig.add_subplot(rows, columns, 3*i+2)
        plt.axis('off')
        plt.imshow(yp[i])
        fig.add_subplot(rows, columns, 3*i+3)
        plt.axis('off')
        plt.imshow(yt[i])
    plt.show()


learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))


Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)


sz = 384 #image size
bs = 32  #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)


learn.fit(lrs/5,1,wds=wd,cycle_len=2,use_clr=(10,8))


learn.save('Unet34_384_1')


learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))


Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)


sz = 768 #image size
bs = 6  #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)


learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))


learn.save('Unet34_768_1')

