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


import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator




traindf = pd.read_csv("/kaggle/input/landmark-recognition-2021/train.csv")



traindf.head()



landmark_unique = traindf['landmark_id'].unique()[0:60]
image_ids = []
labels = []
temp_labels = []

for i, id_ in enumerate(landmark_unique):
    for iid in traindf['id'][traindf['landmark_id'] == id_]:
        image_ids.append(iid)
        labels.append(id_)
        temp_labels.append(i)


mainpath = '../input/landmark-recognition-2021/train'
images_pixels = []

for iid in image_ids:
    first_dir = os.path.join(mainpath, iid[0])
    second_dir = os.path.join(first_dir, iid[1])
    third_dir = os.path.join(second_dir, iid[2])
    finalpath = os.path.join(third_dir, iid + '.jpg')
    
    img_pix = cv2.imread(finalpath, 1)
    images_pixels.append(cv2.resize(img_pix, (100, 100)))


from tensorflow.keras.utils import to_categorical
X_data = np.array(images_pixels) / 255.0
Y_data = to_categorical(temp_labels, num_classes=60)


X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.3, random_state=101)


print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_train[i])
    plt.title(f"Label: {np.argmax(Y_train[i])}")
    plt.axis('off')
    
plt.tight_layout()
plt.show()


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



pretrained_model = tf.keras.applications.DenseNet201(input_shape=(100,100,3),
                                                      include_top=False,
                                                      weights='imagenet',
                                                      pooling='avg')
pretrained_model.trainable = False


inputs = pretrained_model.input
drop_layer = tf.keras.layers.Dropout(0.25)(pretrained_model.output)
x_layer = tf.keras.layers.Dense(512, activation='relu')(drop_layer)
x_layer1 = tf.keras.layers.Dense(128, activation='relu')(x_layer)
drop_layer1 = tf.keras.layers.Dropout(0.20)(x_layer1)
outputs = tf.keras.layers.Dense(60, activation='softmax')(drop_layer1)


model2 = tf.keras.Model(inputs=inputs, outputs=outputs)


datagen = ImageDataGenerator(horizontal_flip=False,
                             vertical_flip=False,
                             rotation_range=0,
                             zoom_range=0.2,
                             width_shift_range=0,
                             height_shift_range=0,
                             shear_range=0,
                             fill_mode="nearest")


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model2.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])
history = model2.fit(datagen.flow(X_train,Y_train,batch_size=32),validation_data=(X_val,Y_val),epochs=30)


base_model = MobileNetV2(input_shape=(100,100,3), include_top=False, weights='imagenet', pooling='avg')
base_model.trainable = False


x = base_model.output
x = Dropout(0.25)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(60, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])


datagen = ImageDataGenerator(horizontal_flip=False,
                             vertical_flip=False,
                             rotation_range=0,
                             zoom_range=0.2,
                             width_shift_range=0,
                             height_shift_range=0,
                             shear_range=0,
                             fill_mode="nearest")


history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                    validation_data=(X_val, Y_val),
                    epochs=30)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


y_pred = model.predict(X_val)


Y_val.shape


import matplotlib.pyplot as plt

# Plotting training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')


pip install timm


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense

input_shape = (224, 224, 3)

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape
)

base_model.trainable = True

inputs = Input(shape=input_shape)
x = base_model(inputs, training=True)  # no preprocess_input here!
x = GlobalAveragePooling2D()(x)
outputs = Dense(60, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet import preprocess_input

# Resize to 224x224 and normalize using EfficientNet preprocessing
X_data_resized = np.array([cv2.resize(img, (224, 224)) for img in images_pixels])
X_data_resized = preprocess_input(X_data_resized)

# Labels to one-hot
Y_data = to_categorical(temp_labels, num_classes=60)

# Split
X_train, X_val, Y_train, Y_val = train_test_split(X_data_resized, Y_data, test_size=0.3, random_state=101)



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
# with tf.device('/GPU:0'):
#     model = Model(inputs, outputs)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     history = model.fit(
#         X_train, Y_train,
#         validation_data=(X_val, Y_val),
#         epochs=30,
#         batch_size=32,
#         callbacks=[early_stop, checkpoint],
#         verbose=1
#     )


from sklearn.metrics import average_precision_score
import numpy as np

# Predict
y_pred_probs = model.predict(X_val)
y_true = Y_val

# Compute GAP
gap_score = average_precision_score(y_true, y_pred_probs, average='macro')
print(f"GAP score: {gap_score:.4f}")



import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class NumpyImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Convert to uint8 if needed for PIL, else normalize here
        img = img.astype(np.uint8)
        if self.transform:
            img = self.transform(img)

        return img, label



from torchvision.transforms import ToTensor, ToPILImage, Resize, Normalize, Compose

transform = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.5] * 3, [0.5] * 3)
])

# One-hot to label index if needed
if Y_train.ndim == 2:  # (N, C)
    Y_train = np.argmax(Y_train, axis=1)
    Y_val = np.argmax(Y_val, axis=1)

train_dataset = NumpyImageDataset(X_train, Y_train, transform)
val_dataset = NumpyImageDataset(X_val, Y_val, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)



import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_dim, ls_=0.9):
        super().__init__()
        self.n_dim = n_dim
        self.ls_ = ls_

    def forward(self, x, target):
        target = F.one_hot(target, self.n_dim).float()
        target *= self.ls_
        target += (1 - self.ls_) / self.n_dim

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.out_dim =n_classes
            
    def forward(self, logits, labels):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", weight=None, reduction="mean",class_weights_norm=None ):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        
        if crit == "focal":
            self.crit = FocalLoss(gamma=args.focal_loss_gamma)
        elif crit == "bce":
            self.crit = nn.CrossEntropyLoss(reduction="none")   
        elif crit == "label_smoothing":
            self.crit = LabelSmoothingLoss(classes=args.n_classes)   

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        #print(self.weight[labels])
        #print(self.s)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

#         labels2 = torch.nn.functional.one_hot(labels, num_classes=args.n_classes+2)
#         labels2 = labels2[:,:args.n_classes+1]
        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss    

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

    
    
from timm.models.vision_transformer_hybrid import HybridEmbed    

class Net(nn.Module):
    def __init__(self, cfg, dataset):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = self.cfg.n_classes
        
        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=cfg.pretrained, 
                                          num_classes=0, 
                                          in_chans=self.cfg.in_channels)
        embedder = timm.create_model(cfg.embedder, 
                                          pretrained=cfg.pretrained, 
                                          in_chans=self.cfg.in_channels,features_only=True, out_indices=[1])

        
        self.backbone.patch_embed = HybridEmbed(embedder,img_size=cfg.img_size[0], 
                                              patch_size=1, 
                                              feature_size=self.backbone.patch_embed.grid_size, 
                                              in_chans=3, 
                                              embed_dim=self.backbone.embed_dim)
#         if 'efficientnet' in cfg.backbone:
#             backbone_out = self.backbone.num_features
#         else:
#             backbone_out = self.backbone.feature_info[-1]['num_chs']

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            
        if "xcit_small_24_p16" in cfg.backbone:
            backbone_out = 384
        elif "xcit_medium_24_p16" in cfg.backbone:
            backbone_out = 512
        elif "xcit_small_12_p16" in cfg.backbone:
            backbone_out = 384
        elif "xcit_medium_12_p16" in cfg.backbone:
            backbone_out = 512   
        elif "swin" in cfg.backbone:
            backbone_out = self.backbone.num_features
        elif "vit" in cfg.backbone:
            backbone_out = self.backbone.num_features
        elif "cait" in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = 2048 

        self.embedding_size = cfg.embedding_size

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if cfg.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif cfg.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(backbone_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif cfg.neck == "option-X":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )
            
        elif cfg.neck == "option-S":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size),
                Swish_module()
            )

        if not self.cfg.headless:    
            self.head_in_units = self.embedding_size
            self.head = ArcMarginProduct_subcenter(self.embedding_size, self.n_classes)
        if self.cfg.loss == 'adaptive_arcface':
            self.loss_fn = ArcFaceLossAdaptiveMargin(dataset.margins,self.n_classes,cfg.arcface_s)
        elif self.cfg.loss == 'arcface':
            self.loss_fn = ArcFaceLoss(cfg.arcface_s,cfg.arcface_m)
        else:
            pass
        
        if cfg.freeze_backbone_head:
            for name, param in self.named_parameters():
                if not 'patch_embed' in name:
                    param.requires_grad = False

    def forward(self, batch):

        x = batch['input']

        x = self.backbone(x)

        x_emb = self.neck(x)

        if self.cfg.headless:
            return {"target": batch['target'],'embeddings': x_emb}
        
        logits = self.head(x_emb)
#         loss = self.loss_fn(logits, batch['target'].long(), self.n_classes)
        preds = logits.softmax(1)
        preds_conf, preds_cls = preds.max(1)
        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())
            return {'loss': loss, "target": batch['target'], "preds_conf":preds_conf,'preds_cls':preds_cls}
        else:
            loss = torch.zeros((1),device=x.device)
            return {'loss': loss, "target": batch['target'],"preds_conf":preds_conf,'preds_cls':preds_cls,
                    'embeddings': x_emb
                   }


class CFG:
    n_classes = 60
    backbone = "swin_base_patch4_window7_224"
    embedder = "efficientnet_b0"
    pretrained = True
    in_channels = 3
    img_size = (100, 100)
    pool = "avg"
    embedding_size = 512
    neck = "option-D"
    headless = False
    loss = "arcface"  # or "adaptive_arcface" if you have per-class margins
    arcface_s = 45.0
    arcface_m = 0.3
    freeze_backbone_head = False
    gem_p_trainable = False



cfg = CFG()


# 1. Imports and Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import timm
import math

class CFG:
    n_classes = 60
    backbone = "swin_base_patch4_window7_224"
    embedder = "efficientnet_b0"
    pretrained = True
    in_channels = 3
    img_size = (100, 100)
    pool = "avg"
    embedding_size = 512
    neck = "option-D"
    headless = False
    loss = "arcface"
    arcface_s = 45.0
    arcface_m = 0.3
    freeze_backbone_head = False
    gem_p_trainable = False
    batch_size = 16
    num_workers = 2
    lr = 1e-4
    epochs = 3

cfg = CFG()

X_data_resized = np.array([cv2.resize(img, (224, 224)) for img in images_pixels]).astype(np.float32)
X_data_resized /= 255.0

mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
X_data_resized = (X_data_resized - mean) / std

X_train, X_val, Y_train, Y_val = train_test_split(X_data_resized, Y_data, test_size=0.3, random_state=101)



class LandmarkDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).permute(0, 3, 1, 2)
        self.labels = torch.tensor(labels).argmax(1).long()

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return {'input': self.images[idx], 'target': self.labels[idx]}

train_loader = DataLoader(LandmarkDataset(X_train, Y_train), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

# 3. ArcMargin and Loss
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x): return F.linear(F.normalize(x), F.normalize(self.weight))

class ArcFaceLoss(nn.Module):
    def __init__(self, s=45.0, m=0.1):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        cosine = logits
        sine = torch.sqrt(1.0 - cosine**2)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter(1, labels.unsqueeze(1), 1)
        output = (one_hot * phi + (1 - one_hot) * cosine) * self.s
        return self.crit(output, labels)

# 4. Full Model
class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0)
        out_dim = self.backbone.num_features
        self.neck = nn.Sequential(nn.Linear(out_dim, cfg.embedding_size), nn.BatchNorm1d(cfg.embedding_size), nn.ReLU())
        self.head = ArcMarginProduct(cfg.embedding_size, cfg.n_classes)
        self.loss_fn = ArcFaceLoss(cfg.arcface_s, cfg.arcface_m)

    def forward(self, batch):
        x, y = batch['input'].cuda(), batch['target'].cuda()
        feat = self.backbone(x)
        emb = self.neck(feat)
        logits = self.head(emb)
        loss = self.loss_fn(logits, y)
        preds = logits.softmax(1).argmax(1)
        return {"loss": loss, "preds_cls": preds, "target": y}

# 5. Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(cfg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

for epoch in range(cfg.epochs):
    model.train()
    total_loss, correct = 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        out['loss'].backward()
        optimizer.step()
        total_loss += out['loss'].item() * batch['input'].size(0)
        correct += (out['preds_cls'] == out['target']).sum().item()
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {correct/len(train_loader.dataset):.4f}")
    
   






import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

mainpath = '../input/landmark-recognition-2021/train'
images_pixels = []
valid_labels = []

for i, iid in enumerate(image_ids):
    iid = str(iid).strip()  # Ensure iid is a string and remove spaces

    # Skip invalid ids
    if len(iid) < 3:
        print(f"Skipping ID with <3 characters: {iid}")
        continue

    try:
        # Build path: /train/a/b/c/abc123.jpg
        finalpath = os.path.join(mainpath, iid[0], iid[1], iid[2], iid + '.jpg')

        if os.path.exists(finalpath):
            img_pix = cv2.imread(finalpath, cv2.IMREAD_COLOR)
            if img_pix is not None:
                resized = cv2.resize(img_pix, (100, 100))
                images_pixels.append(resized)
                valid_labels.append(temp_labels[i])
            else:
                print(f"Warning: Failed to load image at {finalpath}")
        else:
            print(f"Warning: File not found: {finalpath}")

    except Exception as e:
        print(f"Error with image ID {iid}: {e}")

# Final processing
X_dataa = np.array(images_pixels, dtype=np.float32) / 255.0
Y_dataa = to_categorical(valid_labels, num_classes=60)



import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ================================
# Dataset Wrapper
# ================================
class LandmarkDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).permute(0, 3, 1, 2).float()  # (N, 3, 100, 100)
        self.labels = torch.tensor(labels).argmax(dim=1).long()         # one-hot to class index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'input': self.images[idx],
            'target': self.labels[idx]
        }

# ================================
# Train/Test Split and Loaders
# ================================
x_train, x_val, y_train, y_val = train_test_split(X_dataa, Y_dataa, test_size=0.1, random_state=42)

train_ds = LandmarkDataset(x_train, y_train)
val_ds = LandmarkDataset(x_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# ================================
# Config (same as yours)
# ================================
class CFG:
    n_classes = 60
    backbone = "swin_base_patch4_window7_224"
    embedder = "efficientnet_b0"
    pretrained = True
    in_channels = 3
    img_size = (100, 100)
    pool = "avg"
    embedding_size = 512
    neck = "option-D"
    headless = False
    loss = "arcface"
    arcface_s = 45.0
    arcface_m = 0.3
    freeze_backbone_head = False
    gem_p_trainable = False
    lr = 1e-4
    epochs = 5

cfg = CFG()

# Dummy dataset object with margins for loss init
class Dummy:
    margins = np.ones(cfg.n_classes) * 0.3
dataset = Dummy()

# ================================
# Model and Optimizer
# ================================
model = Net(cfg, dataset).cuda()
optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

# ================================
# Training Loop
# ================================
for epoch in range(cfg.epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        optimizer.zero_grad()
        output = model(batch)
        output['loss'].backward()
        optimizer.step()
        total_loss += output['loss'].item()

    print(f"Epoch [{epoch+1}/{cfg.epochs}] - Train Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            output = model(batch)
            val_loss += output['loss'].item()
            correct += (output['preds_cls'] == batch['target']).sum().item()
            total += batch['target'].size(0)
    val_acc = correct / total
    print(f"          Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc*100:.2f}%")



import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_dim, ls_=0.9):
        super().__init__()
        self.n_dim = n_dim
        self.ls_ = ls_

    def forward(self, x, target):
        target = F.one_hot(target, self.n_dim).float()
        target *= self.ls_
        target += (1 - self.ls_) / self.n_dim

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.out_dim =n_classes
            
    def forward(self, logits, labels):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", weight=None, reduction="mean",class_weights_norm=None ):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        
        if crit == "focal":
            self.crit = FocalLoss(gamma=args.focal_loss_gamma)
        elif crit == "bce":
            self.crit = nn.CrossEntropyLoss(reduction="none")   
        elif crit == "label_smoothing":
            self.crit = LabelSmoothingLoss(classes=args.n_classes)   

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        #print(self.weight[labels])
        #print(self.s)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

#         labels2 = torch.nn.functional.one_hot(labels, num_classes=args.n_classes+2)
#         labels2 = labels2[:,:args.n_classes+1]
        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss    

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

    
    
from timm.models.vision_transformer_hybrid import HybridEmbed    

class Net(nn.Module):
    def __init__(self, cfg, dataset):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = self.cfg.n_classes
        
        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=cfg.pretrained, 
                                          num_classes=0, 
                                          in_chans=self.cfg.in_channels)
        embedder = timm.create_model(cfg.embedder, 
                                          pretrained=cfg.pretrained, 
                                          in_chans=self.cfg.in_channels,features_only=True, out_indices=[1])

        
        self.backbone.patch_embed = HybridEmbed(embedder,img_size=cfg.img_size[0], 
                                              patch_size=1, 
                                              feature_size=self.backbone.patch_embed.grid_size, 
                                              in_chans=3, 
                                              embed_dim=self.backbone.embed_dim)
#         if 'efficientnet' in cfg.backbone:
#             backbone_out = self.backbone.num_features
#         else:
#             backbone_out = self.backbone.feature_info[-1]['num_chs']

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            
        if "xcit_small_24_p16" in cfg.backbone:
            backbone_out = 384
        elif "xcit_medium_24_p16" in cfg.backbone:
            backbone_out = 512
        elif "xcit_small_12_p16" in cfg.backbone:
            backbone_out = 384
        elif "xcit_medium_12_p16" in cfg.backbone:
            backbone_out = 512   
        elif "swin" in cfg.backbone:
            backbone_out = self.backbone.num_features
        elif "vit" in cfg.backbone:
            backbone_out = self.backbone.num_features
        elif "cait" in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = 2048 

        self.embedding_size = cfg.embedding_size

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if cfg.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif cfg.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(backbone_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif cfg.neck == "option-X":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )
            
        elif cfg.neck == "option-S":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, self.embedding_size),
                Swish_module()
            )

        if not self.cfg.headless:    
            self.head_in_units = self.embedding_size
            self.head = ArcMarginProduct_subcenter(self.embedding_size, self.n_classes)
        if self.cfg.loss == 'adaptive_arcface':
            self.loss_fn = ArcFaceLossAdaptiveMargin(dataset.margins,self.n_classes,cfg.arcface_s)
        elif self.cfg.loss == 'arcface':
            self.loss_fn = ArcFaceLoss(cfg.arcface_s,cfg.arcface_m)
        else:
            pass
        
        if cfg.freeze_backbone_head:
            for name, param in self.named_parameters():
                if not 'patch_embed' in name:
                    param.requires_grad = False

    def forward(self, batch):

        x = batch['input']

        x = self.backbone(x)

        x_emb = self.neck(x)

        if self.cfg.headless:
            return {"target": batch['target'],'embeddings': x_emb}
        
        logits = self.head(x_emb)
#         loss = self.loss_fn(logits, batch['target'].long(), self.n_classes)
        preds = logits.softmax(1)
        preds_conf, preds_cls = preds.max(1)
        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())
            return {'loss': loss, "target": batch['target'], "preds_conf":preds_conf,'preds_cls':preds_cls}
        else:
            loss = torch.zeros((1),device=x.device)
            return {'loss': loss, "target": batch['target'],"preds_conf":preds_conf,'preds_cls':preds_cls,
                    'embeddings': x_emb
                   }




