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


disease = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json')
train = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')


disease.head()


train.head()


train.shape


train['image_id'][0]


import os
import imp
import torch
import random
import torchvision

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models

# ml lib
from sklearn.model_selection import train_test_split


SEED = 127
torch.manual_seed(SEED)


# img_path = '/kaggle/input/cassava-leaf-disease-classification/train_images/'
# img_path + train['image_id'][0]


# img_path


# len(os.listdir(img_path))


test_path = '/kaggle/input/cassava-leaf-disease-classification/test_images/2216849948.jpg'

with Image.open(test_path) as img:
        width, height = img.size
        plt.imshow(img)


train_img_path = '/kaggle/input/cassava-leaf-disease-classification/train_images/'

img_path =  os.path.join(train_img_path + train['image_id'][2])

with Image.open(img_path) as img:
        width, height = img.size
        plt.imshow(img)


len(os.listdir(train_img_path))


img.size


import cv2
import numpy as np

r_mean_arr = []
g_mean_arr = []
b_mean_arr = []

r_std_arr = []
g_std_arr = []
b_std_arr = []

for i in range(len(os.listdir(train_img_path))):
    img_path =  os.path.join(train_img_path + train['image_id'][i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img)

    r_mean, g_mean, b_mean = np.mean(img, axis =(0,1))
    r_std, g_std, b_std = np.std(img, axis=(0,1))

    r_mean_arr.append(r_mean)
    g_mean_arr.append(g_mean)
    b_mean_arr.append(b_mean)
    
    r_std_arr.append(r_std)
    g_std_arr.append(g_std)
    b_std_arr.append(b_std)


len(r_mean_arr) #, g_mean_arr, b_mean_arr


r_mean_arr[0]


R_MEAN = np.mean(r_mean_arr) / 255
G_MEAN = np.mean(g_mean_arr) / 255
B_MEAN = np.mean(b_mean_arr) / 255


print(f"Red ch mean   = {R_MEAN}\nGreen ch mean = {G_MEAN}\nBlue ch mean  = {B_MEAN}")


# ìœ„ì—�ì„œ í•œë²ˆì—�
# r_std_arr = []
# g_std_arr = []
# b_std_arr = []

# for i in range(len(os.listdir(train_img_path))):
#     img_path =  os.path.join(train_img_path + train['image_id'][i])
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # print(img)

#     # r_mean, g_mean, b_mean = np.mean(img, axis =(0,1))
#     r_std, g_std, b_std = np.std(img, axis=(0,1))

#     # r_mean_arr.append(r_mean)
#     # g_mean_arr.append(g_mean)
#     # b_mean_arr.append(b_mean)
    
#     r_std_arr.append(r_std)
#     g_std_arr.append(g_std)
#     b_std_arr.append(b_std)


R_STD = np.mean(r_std_arr) / 255
G_STD = np.mean(g_std_arr) / 255
B_STD = np.mean(b_std_arr) / 255


print(f"Red ch std   = {R_STD}\nGreen ch std = {G_STD}\nBlue ch std  = {B_STD}")


X_train, X_test, y_train, y_test = train_test_split(train['image_id'], train['label'], test_size = 0.2, random_state = SEED, stratify = train['label'])

print(f'Train size = {X_train.shape[0]} \n Test size = {X_test.shape[0]}')


type(X_train)


X_train.iloc[123]


import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# y_train, y_testê°€ ì •ìˆ˜ í�´ë�˜ìŠ¤(Label)ë�¼ë©´
y_train.value_counts().plot.pie(autopct='%1.1f%%', ax=axs[0], colors=['skyblue', 'orange', 'lightgreen', 'salmon', 'plum'])
axs[0].set_title("Train Label ë¶„í�¬")
axs[0].set_ylabel("")  # yì¶• ë�¼ë²¨ ì œê±°

y_test.value_counts().plot.pie(autopct='%1.1f%%', ax=axs[1], colors=['skyblue', 'orange', 'lightgreen', 'salmon', 'plum'])
axs[1].set_title("Test Label ë¶„í�¬")
axs[1].set_ylabel("")


# kFold ì„¤ì • ë°� ë�°ì�´í„° ë¶„í• 

from sklearn.model_selection import StratifiedKFold

train_df = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')

# 2. fold ì»¬ëŸ¼ ì´ˆê¸°í™”
train_df['fold'] = -1

# 3. StratifiedKFold ê°�ì²´ ìƒ�ì„±
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 4. fold ê°’ í• ë‹¹
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    train_df.loc[val_idx, 'fold'] = fold


train_df


sample_df = train_df.groupby('label').head(1).sort_values(by = 'label')
sample_df


train_img_path = '/kaggle/input/cassava-leaf-disease-classification/train_images/'

import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2í–‰ 3ì—´ ê·¸ë¦¼íŒ� ë§Œë“¤ê¸°
axes = axes.flatten()  # ì¶•ë“¤ì�„ 1ì°¨ì›� ë¦¬ìŠ¤íŠ¸ë¡œ ë°”ê¾¸ê¸°

for i in range(len(sample_df)):  # sample_df ê°œìˆ˜ë§Œí�¼ ë°˜ë³µ
    img_path = os.path.join(train_img_path, sample_df['image_id'].iloc[i])  # ì�´ë¯¸ì§€ ê²½ë¡œ
    img = Image.open(img_path)  # ì�´ë¯¸ì§€ ì—´ê¸°
    axes[i].imshow(img)  # ië²ˆì§¸ ì¹¸ì—� ì�´ë¯¸ì§€ ê·¸ë¦¬ê¸°
    axes[i].set_title(f"Label: {sample_df['label'].iloc[i]}")  # ì œëª© ë¶™ì�´ê¸°
    axes[i].axis('off')  # ì¶• ëˆˆê¸ˆ ì—†ì• ê¸°

# ë‚¨ëŠ” subplotì�´ ì�ˆì�„ ê²½ìš° ë¹ˆ ì¹¸ìœ¼ë¡œ ë‘ 
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()  # ê·¸ë¦¼ ê°„ê²© ë”± ë§�ì¶”ê¸°
plt.show()  # ê·¸ë¦¼ ë³´ì—¬ì£¼ê¸°



train_transforms = transforms.Compose([
transforms.Resize(256),

transforms.RandomCrop(224),

transforms.RandomHorizontalFlip(),

transforms.ColorJitter(0.1, 0.1, 0.1, 0.05), # ë„ˆë¬´ ê³¼í•˜ì§€ ì•Šê²Œ

transforms.ToTensor(),

transforms.Normalize(
    mean=[0.430, 0.496, 0.313],
    std=[0.219, 0.224, 0.201]
)
])

# ì±„ë„�ë³„ ì •ê·œí™”
    # ê°� ì±„ë„�(R, G, B)ì�˜ í�‰ê· ê³¼ í‘œì¤€í�¸ì°¨ë¥¼ ê¸°ì¤€ìœ¼ë¡œ
    # (ê°’ - í�‰ê· ) / í‘œì¤€í�¸ì°¨ â†’ í�‰ê·  0, ë¶„ì‚° 1ë¡œ ë§Œë“¤ì–´ì¤Œ



test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomHorizontalFlip(), í…ŒìŠ¤íŠ¸ì—�ëŠ” ì—†ì�Œ ->ì›�ë³¸ ë¬¸ì œ ê·¸ëŒ€ë¡œ ë‚˜ì˜µë‹ˆë‹¤. ê·¸ê±¸ë¡œ ì •í™•ë�„ë¥¼ ë´�ì•¼ í•˜ë‹ˆê¹Œìš”!
    
    transforms.ToTensor(),
    # 1.ì�´ë¯¸ì§€ í…�ì„œë¡œ ë³€í™˜, 2. (H, W, C) â†’ (C, H, W) êµ¬ì¡° ë³€ê²½ 3. ê°’ ë²”ìœ„: 0~255 â†’ 0.0~1.0 (ì •ê·œí™”)
    
    transforms.Normalize (mean =[0.430, 0.496, 0.313],
                           std = [0.219,0.224,0.201])
                                          # ì±„ë„�ë³„ ì •ê·œí™”
    # ê°� ì±„ë„�(R, G, B)ì�˜ í�‰ê· ê³¼ í‘œì¤€í�¸ì°¨ë¥¼ ê¸°ì¤€ìœ¼ë¡œ
    # (ê°’ - í�‰ê· ) / í‘œì¤€í�¸ì°¨ â†’ í�‰ê·  0, ë¶„ì‚° 1ë¡œ ë§Œë“¤ì–´ì¤Œ
])


from torch.utils.data import Dataset, DataLoader


train.iloc[9]['image_id']


class DiseaseData(Dataset):
    def __init__(self,x,y, transform = None, submission = False):
        self.x = x.reset_index()
        self.y = y.reset_index()
        self.submission = submission
        self.transform = transform


    def __len__(self):
        return self.x.shape[0]

    def load_image(self, path): # ì�´ë¯¸ì§€ í•œ ì�¥ ë¶ˆëŸ¬ì˜¤ê¸°
        prefix = train_img_path
        return Image.open(os.path.join(prefix , path['image_id']))

    def __getitem__(self, index): # ì�´ë¯¸ì§€ + ë�¼ë²¨ í•œ ìŒ� ë°˜í™˜
        image = self.load_image(self.x.iloc[index])
        label = self.y.iloc[index]['label']
        if self.transform:
            image = self.transform(image)
        if self.submission:
            image = np.array(image)
        sample = {'image' : image, 'label' : label}
        # return sample
        return image, label # kfold

        


# PyTorchìš© Dataset ê°�ì²´ë¥¼ ìƒ�ì„±

train_data = DiseaseData(X_train, y_train, transform = train_transforms)
test_data = DiseaseData(X_test, y_test, transform = test_transforms)


train_data[8]


#  transformê¹Œì§€ ëª¨ë‘� ì �ìš©ë�œ ì�´ë¯¸ì§€ë¥¼ ì‹œê°�í™” í•´ë³´ê¸° (transforms.Normalizeê°€ ì �ìš©ë�˜ì–´ì�ˆì–´ì„œ ìƒ‰ì�´ ë‹¤ë¥´ê²Œ ë‚˜ì˜¨ë‹¤.)
sample = train_data[8]
# sample
import matplotlib.pyplot as plt
plt.imshow(sample['image'].permute(1, 2, 0))  # CHW â†’ HWC
plt.title(f"Label: {sample['label'].item()}")
plt.axis("off")
plt.show()


# ê¸°ì¡´ ì�´ë¯¸ì§€ í™•ì�¸í•´ë³´ê¸°

sample = train_data[8]  # transform í�¬í•¨ë�œ ì�´ë¯¸ì§€
transformed_img = sample['image']      # shape: [C, H, W]
label = sample['label']

# Tensor â†’ numpyë¡œ ë°”ê¿”ì„œ ì‹œê°�í™”
import matplotlib.pyplot as plt
import torch

# Normalize ë�˜ì–´ ì�ˆë‹¤ë©´ ë‹¤ì‹œ ë�˜ë�Œë ¤ì•¼ ëˆˆìœ¼ë¡œ í™•ì�¸ ê°€ëŠ¥
def unnormalize(img_tensor):
    """Normalizeë¥¼ ë°˜ëŒ€ë¡œ ì �ìš©í•´ì„œ ì‹œê°�í™” ê°€ëŠ¥í•˜ê²Œ ë§Œë“¤ê¸°"""
    mean = torch.tensor([0.430, 0.496, 0.313]).view(3, 1, 1)
    std = torch.tensor([0.219, 0.224, 0.201]).view(3, 1, 1)
    return img_tensor * std + mean  # ì±„ë„�ë³„ë¡œ ê³±í•˜ê³  ë�”í•¨

img_to_show = unnormalize(transformed_img).permute(1, 2, 0)  # CHW â†’ HWC
plt.imshow(img_to_show.numpy())
plt.title(f"Transformë�œ ì�´ë¯¸ì§€ (label={label})")
plt.axis('off')
plt.show()


trainloader = torch.utils.data.DataLoader(train_data,batch_size=64, num_workers= 2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, num_workers= 2)


# trainloader êµ¬ì¡° í™•ì�¸í•´ë³´ê¸°

for index, sample_batch in enumerate(trainloader):
    print(index,
          sample_batch['image'].__len__(),
          sample_batch['label'].__len__(),
          sample_batch['image'].size(),
          sample_batch['label'].size()
          
          )
    # if batch > 4:
    break


# create model 
resnet18 = models.resnet18(pretrained = True)


# optimizer and criterion

optimizer = optim.Adam(resnet18.parameters(), lr =1e-4)
criterion = nn.CrossEntropyLoss()


# ResNet18ì�˜ ë§ˆì§€ë§‰ fully connected layerì�˜ ì�…ë ¥ feature ìˆ˜ë¥¼ ê°€ì ¸ì˜´
num_ftrs = resnet18.fc.in_features

# ResNet18ì�˜ ë§ˆì§€ë§‰ fully connected layer(fc)ë¥¼ ìƒˆë¡œ ì •ì�˜í•¨
# ì›�ë�˜ëŠ” 1000ê°œì�˜ ImageNet í�´ë�˜ìŠ¤ë¥¼ ì˜ˆì¸¡í•˜ë�˜ êµ¬ì¡° â†’ ì§€ê¸ˆì�€ 5ê°œ í�´ë�˜ìŠ¤ ì˜ˆì¸¡ìœ¼ë¡œ ë³€ê²½

resnet18.fc = nn.Sequential(
    nn.Linear(num_ftrs, 500),  # ë¨¼ì € 500ì°¨ì›�ìœ¼ë¡œ ì¤„ì�´ëŠ” hidden layer ì¶”ê°€
    nn.Linear(500, 5)          # ë§ˆì§€ë§‰ ì¶œë ¥ì¸µ: 5ê°œ í�´ë�˜ìŠ¤ ë¶„ë¥˜
)



# train_cycle í•¨ìˆ˜ëŠ” ëª¨ë�¸ì�„ ì£¼ì–´ì§„ ë�°ì�´í„°ì…‹(trainloader)ìœ¼ë¡œ í•™ìŠµí•˜ê³ , ì£¼ê¸°ì �ìœ¼ë¡œ lossì™€ accuracyë¥¼ ì¶œë ¥ ë°� ì‹œê°�í™” 
def train_cycle(model, optimizer, criterion, p_iter, n_epochs):
    model.train()              # ëª¨ë�¸ì�„ í•™ìŠµ ëª¨ë“œë¡œ ì„¤ì • (Dropout, BatchNorm ë“± í™œì„±í™”)
    model.to(DEVICE)           # ëª¨ë�¸ì�„ ì§€ì •í•œ ë””ë°”ì�´ìŠ¤(CPU ë˜�ëŠ” GPU)ë¡œ ì�´ë�™

    itr = 1                    # ë°˜ë³µ(iteration) íšŸìˆ˜ ì¹´ìš´í„°
    total_loss = 0            # p_iterë§Œí�¼ ëˆ„ì �í•œ ì†�ì‹¤ê°’ ì €ì�¥ìš©
    loss_list = []            # ì‹œê°�í™”ë¥¼ ìœ„í•œ ì†�ì‹¤ ê¸°ë¡� ë¦¬ìŠ¤íŠ¸
    acc_list = []             # ì‹œê°�í™”ë¥¼ ìœ„í•œ ì •í™•ë�„ ê¸°ë¡� ë¦¬ìŠ¤íŠ¸

    # ì—�í�­ ë‹¨ìœ„ ë£¨í”„ (ì „ì²´ í•™ìŠµ ë°˜ë³µ íšŸìˆ˜ë§Œí�¼)
    for epoch in range(n_epochs):

        # ë°°ì¹˜ ë‹¨ìœ„ ë£¨í”„
        for batch_no, data in enumerate(trainloader, 0): # trainloaderëŠ” ì—¬ëŸ¬ ì�¥ì�˜ ì�´ë¯¸ì§€ì™€ ë�¼ë²¨ì�„ ë¬¶ì�€ **ë°°ì¹˜(batch)**ë¥¼ ë°˜ë³µí•´ì„œ ì¤˜.batch_no = 0ë¶€í„° ì‹œì�‘í•˜ëŠ” ì�¸ë�±ìŠ¤

            # ë°°ì¹˜ì—�ì„œ ì�´ë¯¸ì§€ì™€ ë�¼ë²¨ ì¶”ì¶œ
            samples, labels = data['image'], data['label'] # samplesëŠ” [batch_size, ì±„ë„�, ë†’ì�´, ë„ˆë¹„] í˜•íƒœì�˜ ì�´ë¯¸ì§€ ë�°ì�´í„°
            samples = samples.to(DEVICE)  # ì�…ë ¥ ì�´ë¯¸ì§€ë¥¼ ë””ë°”ì�´ìŠ¤ë¡œ ì „ì†¡, 32ê°œ
            labels = labels.to(DEVICE)    # ì •ë‹µ ë ˆì�´ë¸”ë�„ ë””ë°”ì�´ìŠ¤ë¡œ ì „ì†¡, 32ê°œ
            
            optimizer.zero_grad()         # ì�´ì „ ë°°ì¹˜ì—�ì„œ ëˆ„ì �ë�œ gradient ì´ˆê¸°í™”

            output = model(samples)       # ëª¨ë�¸ì�˜ forward pass ì‹¤í–‰ â†’ ì˜ˆì¸¡ê°’ ì¶œë ¥,  ìˆœì „íŒŒ (forward pass) ë�¼ê³  ë¶€ë¥´ëŠ” ë‹¨ê³„
            loss = criterion(output, labels)  # ì†�ì‹¤ í•¨ìˆ˜ ê³„ì‚°
            loss.backward()              # ì†�ì‹¤ì—� ëŒ€í•œ gradient ê³„ì‚° (ì—­ì „íŒŒ)
            optimizer.step()             # ê³„ì‚°ë�œ gradientë¡œ ê°€ì¤‘ì¹˜ ì—…ë�°ì�´íŠ¸

            total_loss += loss.item()    # p_iterë§Œí�¼ ì†�ì‹¤ ëˆ„ì �

            # print(loss)        # tensor(0.3456, grad_fn=<NllLossBackward>)
            # print(loss.item()) # 0.3456 (float)

            # ì�¼ì • iterationë§ˆë‹¤ ë¡œê·¸ ì¶œë ¥ ë°� ì •í™•ë�„ ê³„ì‚°
            if itr % p_iter == 0:
                pred = torch.argmax(output, dim=1)   # ê°€ì�¥ ë†’ì�€ í™•ë¥ ê°’ì�˜ í�´ë�˜ìŠ¤ ì„ íƒ�
                correct = pred.eq(labels)            # ì˜ˆì¸¡ê³¼ ì •ë‹µ ë¹„êµ� (True/False)
                acc = torch.mean(correct.float())    # ì •í™•ë�„ ê³„ì‚° (ì •ë‹µ ë§�ì¶˜ ë¹„ìœ¨)
                acc = acc.to('cpu')                  # ì‹œê°�í™”ë¥¼ ìœ„í•´ CPUë¡œ ì�´ë�™

                # ë¡œê·¸ ì¶œë ¥
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'
                      .format(epoch+1, n_epochs, itr, total_loss/p_iter, acc))
                
                # ê·¸ë�˜í”„ìš© ê¸°ë¡� ì €ì�¥
                loss_list.append(total_loss / p_iter)
                acc_list.append(acc)

                total_loss = 0  # ì†�ì‹¤ ì´ˆê¸°í™” (ë‹¤ì�Œ p_iterì�„ ìœ„í•œ)

            itr += 1  # ë°˜ë³µ íšŸìˆ˜ ì¦�ê°€

    # í›ˆë ¨ ê²°ê³¼ ì‹œê°�í™”
    plt.plot(loss_list[1:], label='loss')       # ì†�ì‹¤ ê·¸ë�˜í”„
    plt.plot(acc_list[1:], label='accuracy')    # ì •í™•ë�„ ê·¸ë�˜í”„
    plt.legend()
    plt.title('training loss and accuracy')
    plt.show()

    print('Finished Training')  # í•™ìŠµ ì™„ë£Œ ë©”ì‹œì§€


# ëª¨ë�¸ í›ˆë ¨
EPOCHS_resnet18 = 7
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_cycle(resnet18, optimizer= optimizer, criterion= criterion, p_iter= 200, n_epochs= EPOCHS_resnet18)


def train_fn(model, train_loader, val_loader, optimizer, criterion, device, n_epochs=7):
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print(f"  ğŸŸ¦ Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  ğŸŸ¥ Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


            # ğŸŸ¢ ì„±ëŠ¥ì�´ ì¢‹ì•„ì¡Œì�„ ë•Œ , early stop ê¸°ëŠ¥ ë¶€ë¶„ 
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         best_model_wts = copy.deepcopy(model.state_dict())
    #         counter = 0
    #         print(f"âœ… Improved! Saving model at epoch {epoch+1}")
    #     else:
    #         counter += 1
    #         print(f"âš ï¸� No improvement ({counter}/{patience})")
    #         if counter >= patience:
    #             print("ğŸ›‘ Early stopping triggered!")
    #             break
    
    # # ğŸ”„ ê°€ì�¥ ì¢‹ì•˜ë�˜ ê°€ì¤‘ì¹˜ë¡œ ë³µì›�
    # model.load_state_dict(best_model_wts)



def test_model(model, test_data, criterion):
    model.eval()   # ëª¨ë�¸ì�„ í�‰ê°€(evaluation) ëª¨ë“œë¡œ ì„¤ì • (Dropout, BatchNorm ë“± ë¹„í™œì„±í™”)

    acc_list = []   # ë°°ì¹˜ë³„ ì •í™•ë�„ ì €ì�¥ ë¦¬ìŠ¤íŠ¸
    loss_list = []  # ë°°ì¹˜ë³„ ì†�ì‹¤ ì €ì�¥ ë¦¬ìŠ¤íŠ¸
    val_loss = 0    # ì „ì²´ ê²€ì¦� ì†�ì‹¤ ëˆ„ì � ë³€ìˆ˜

    with torch.no_grad():   # í�‰ê°€ ì‹œì—�ëŠ” ê¸°ìš¸ê¸° ê³„ì‚°í•˜ì§€ ì•Šì�Œ (ë©”ëª¨ë¦¬ ì ˆì•½ ë°� ì†�ë�„ í–¥ìƒ�) -> ëª¨ë�¸ì�´ ì�´ë¯¸ í•™ìŠµë�œ ê°€ì¤‘ì¹˜ë¥¼ ê·¸ëŒ€ë¡œ ì‚¬ìš©í•´ì„œ ê²°ê³¼ë§Œ ì˜ˆì¸¡í•˜ë©´ ë�¼.
        for batch_no, data in tqdm(enumerate(test_data, 0)):   # ê²€ì¦� ë�°ì�´í„°ì…‹ì�„ ë°°ì¹˜ ë‹¨ìœ„ë¡œ ë°˜ë³µ
            samples, labels = data['image'], data['label']     # ë°°ì¹˜ì—�ì„œ ì�´ë¯¸ì§€ì™€ ë�¼ë²¨ ì¶”ì¶œ
            samples = samples.to(DEVICE)   # GPU ë˜�ëŠ” CPU ë“± ì§€ì •í•œ ë””ë°”ì�´ìŠ¤ë¡œ ë�°ì�´í„° ì�´ë�™
            labels = labels.to(DEVICE)     # ë�¼ë²¨ë�„ ê°™ì�€ ë””ë°”ì�´ìŠ¤ë¡œ ì�´ë�™
            
            output = model(samples)         # ëª¨ë�¸ì—� ì�…ë ¥ ë„£ì–´ ì˜ˆì¸¡ê°’(output) ê³„ì‚°
            loss = criterion(output, labels) # ì†�ì‹¤ í•¨ìˆ˜ë¡œ ë°°ì¹˜ë³„ ì†�ì‹¤ ê³„ì‚°
            
            val_loss += loss.item()         # ì†�ì‹¤ ê°’ ëˆ„ì � (float ê°’ìœ¼ë¡œ ì €ì�¥) -> ì „ì²´ í�‰ê°€ ì†�ì‹¤ ëˆ„ì � ë°� í�‰ê·  ê³„ì‚° ëª©ì �

            pred = torch.argmax(output, dim=1)  # ì˜ˆì¸¡ê°’ ì¤‘ ê°€ì�¥ ë†’ì�€ í™•ë¥ ì�„ ê°€ì§„ í�´ë�˜ìŠ¤ ì„ íƒ�
            correct = pred.eq(labels)            # ì˜ˆì¸¡ê³¼ ì‹¤ì œ ë�¼ë²¨ ë¹„êµ� (True/False tensor)
            acc = torch.mean(correct.float())   # ë°°ì¹˜ ì •í™•ë�„ ê³„ì‚° (ë§�ì�€ ë¹„ìœ¨)
            acc = acc.to('cpu')                  # ì •í™•ë�„ë¥¼ CPUë¡œ ì˜®ê²¨ì„œ ë¦¬ìŠ¤íŠ¸ì—� ì €ì�¥
            acc_list.append(acc)                 # ì •í™•ë�„ ì €ì�¥
            loss = loss.to('cpu')                # ì†�ì‹¤ë�„ CPUë¡œ ì˜®ê¹€
            loss_list.append(loss)               # ì†�ì‹¤ ì €ì�¥

    # ë°°ì¹˜ë³„ ì •í™•ë�„ì™€ ì†�ì‹¤ì�˜ í�‰ê· ê°’ ì¶œë ¥
    print(f'Mean acc = {np.mean(acc_list): .3f}. Mean loss = {np.mean(loss_list): .3f}')



criterion = nn.CrossEntropyLoss()


import copy
import torch.optim as optim
import torch.nn as nn

num_folds = 3
num_epochs = 5

# ì˜ˆ: resnet18 ìƒ�ì„± í•¨ìˆ˜ (ê°€ì¤‘ì¹˜ ì´ˆê¸°í™” í�¬í•¨)
def create_model():
    model = torchvision.models.resnet18(pretrained=True)  # ë˜�ëŠ” True í•„ìš”ì—� ë”°ë�¼
    # model.fc = nn.Linear(model.fc.in_features, num_classes)  # ì¶œë ¥ ë ˆì�´ì–´ ìˆ˜ì •
    # return model

    # ResNet18ì�˜ ë§ˆì§€ë§‰ fully connected layerì�˜ ì�…ë ¥ feature ìˆ˜ë¥¼ ê°€ì ¸ì˜´
    num_ftrs = model.fc.in_features
    
    # ResNet18ì�˜ ë§ˆì§€ë§‰ fully connected layer(fc)ë¥¼ ìƒˆë¡œ ì •ì�˜í•¨
    # ì›�ë�˜ëŠ” 1000ê°œì�˜ ImageNet í�´ë�˜ìŠ¤ë¥¼ ì˜ˆì¸¡í•˜ë�˜ êµ¬ì¡° â†’ ì§€ê¸ˆì�€ 5ê°œ í�´ë�˜ìŠ¤ ì˜ˆì¸¡ìœ¼ë¡œ ë³€ê²½
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 500),  # ë¨¼ì € 500ì°¨ì›�ìœ¼ë¡œ ì¤„ì�´ëŠ” hidden layer ì¶”ê°€
        nn.Linear(500, 5)          # ë§ˆì§€ë§‰ ì¶œë ¥ì¸µ: 5ê°œ í�´ë�˜ìŠ¤ ë¶„ë¥˜
    )
    return model

# K-Fold ì‹œì�‘
for fold in range(num_folds):
    print(f'\n===== Fold {fold} =====')

    # ë�°ì�´í„° ë¶„í•  (ê¸°ì¡´ ì½”ë“œ ì°¸ê³ )
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data   = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    X_train = train_data.drop(columns=['label'])
    y_train = train_data[['label']]
    X_val = val_data.drop(columns=['label'])
    y_val = val_data[['label']]

    train_dataset = DiseaseData(X_train, y_train, transform=train_transforms)
    val_dataset   = DiseaseData(X_val, y_val, transform=test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # **ë§¤ í�´ë“œë§ˆë‹¤ ëª¨ë�¸ ìƒˆë¡œ ìƒ�ì„±, ì´ˆê¸°í™”**
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 3
    counter = 0

    for epoch in range(num_epochs):
        # train_fn í•¨ìˆ˜ ë‚´ë¶€ë¥¼ epoch ë‹¨ìœ„ë¡œ ë‚˜ëˆ ì„œ ì‚¬ìš©í•˜ê±°ë‚˜,
        # train_fnì—� epoch 1ì”© ë�Œë¦¬ë�„ë¡� ìˆ˜ì • ê°€ëŠ¥
        
    # í›ˆë ¨ í•¨ìˆ˜ ë¶€ë¶„
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)


        # í�‰ê°€ í•¨ìˆ˜ ë¶€ë¶„
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Early Stopping & best model ì €ì�¥
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else: # ì†�ì‹¤ì�´ ì»¤ì§€ëŠ” êµ¬ê°„ì—�ì„œ ë©ˆì¶”ê¸°
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break

    # ê°€ì�¥ ì¢‹ì•˜ë�˜ ëª¨ë�¸ ê°€ì¤‘ì¹˜ë¡œ ë³µì›�
    model.load_state_dict(best_model_wts)

    # ì—¬ê¸°ì„œ ëª¨ë�¸ ì €ì�¥í•˜ê±°ë‚˜, í�‰ê°€, ì˜ˆì¸¡ ë“±ì—� ì‚¬ìš© ê°€ëŠ¥
    torch.save(model.state_dict(), f'model_fold{fold}.pth')



for fold in range(3):
    print(f'\n==== Fold {fold} í…ŒìŠ¤íŠ¸ ì¤‘ ====')
    
    model = create_model().to(DEVICE)  # ìƒˆ ëª¨ë�¸ ê°�ì²´ ìƒ�ì„±
    model.load_state_dict(torch.load(f'model_fold{fold}.pth'))  # í•´ë‹¹ foldì—�ì„œ ì €ì�¥í•œ ëª¨ë�¸ ë¡œë“œ
    
    test_model(model, testloader, criterion)



%%time
print('ResNet18:')
test_model(resnet18, testloader, criterion)


READY_MODELS_PATH = 'saved_models/'

os.makedirs(os.path.join('../working/', READY_MODELS_PATH))
torch.save(resnet18.state_dict(), os.path.join(READY_MODELS_PATH, 'resnet18.pt'))


sub = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv')

sub


[x for x in os.listdir('/kaggle/input/cassava-leaf-disease-classification') if 'test' in x]





def predict(model, image_path, device='cuda'):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = test_transforms(image).unsqueeze(0).to(device)  # ë°°ì¹˜ ì°¨ì›� ì¶”ê°€ ë°� device ì�´ë�™
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred


predict(resnet18, '/kaggle/input/cassava-leaf-disease-classification/test_images/2216849948.jpg')


submission = pd.DataFrame([{
    'image_id': '2216849948.jpg',
    'label': predict(resnet18, '/kaggle/input/cassava-leaf-disease-classification/test_images/2216849948.jpg')
}])


submission


submission.to_csv('submission.csv', index=False)




