import os, zipfile, glob, random
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# --- User settings (keep your flags) ---
ComputeLB = False
DogsOnly = False

# --- base root depending on context (Kaggle vs local) ---
if ComputeLB:
    ROOT = '/kaggle/input/generative-dog-images'
else:
    ROOT = '/kaggle/input/generative-dog-images/'

print("Initial ROOT:", ROOT)

# --- helper: try to find an images folder ---
def find_images_dir(root):
    # Common possibilities
    candidates = [
        os.path.join(root, 'all-dogs', 'all-dogs'),
        os.path.join(root, 'all-dogs'),
        os.path.join(root, 'all_dogs'),
        root
    ]
    # Also search recursively for a folder with many .jpg files
    for c in candidates:
        if os.path.isdir(c):
            jpgs = glob.glob(os.path.join(c, '*.jpg'))
            if len(jpgs) > 50:
                return c
    # fallback: walk to find dir with many jpgs
    for dirpath, dirnames, filenames in os.walk(root):
        jpgs = [f for f in filenames if f.lower().endswith('.jpg')]
        if len(jpgs) > 50:
            return dirpath
    return None

# --- look for zipped dataset and extract if necessary ---
all_dogs_zip_candidates = [
    os.path.join(ROOT, 'all-dogs.zip'),
    os.path.join(ROOT, 'All-Dogs.zip'),
    os.path.join(ROOT, 'all-dogs_all-dogs.zip')
]
for zpath in all_dogs_zip_candidates:
    if os.path.isfile(zpath):
        print("Found zip at:", zpath, " -> extracting to /tmp/all-dogs_extracted")
        extract_to = '/tmp/all-dogs_extracted'
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zpath, 'r') as z:
            z.extractall(extract_to)
        # reset ROOT to extracted folder for scanning
        ROOT = extract_to + '/'
        break

IMG_DIR = find_images_dir(ROOT)
if IMG_DIR is None:
    raise FileNotFoundError(f"Could not find images folder under {ROOT}. Try checking file tree or adjust ROOT.")
print("Using image directory:", IMG_DIR)

# --- annotation directory (if DogsOnly) ---
ANN_DIR = None
if DogsOnly:
    # possible annotation path
    cand = os.path.join(ROOT, 'annotation', 'Annotation')
    if os.path.isdir(cand):
        ANN_DIR = cand
    else:
        # try to find folder named 'Annotation' recursively
        for dirpath, dirnames, filenames in os.walk(ROOT):
            if 'Annotation' in dirnames:
                ANN_DIR = os.path.join(dirpath, 'Annotation')
                break
    if ANN_DIR is None:
        raise FileNotFoundError(f"Could not find annotation folder under {ROOT}. Needed for DogsOnly=True.")

print("Annotation directory:", ANN_DIR)

# --- gather image file list ---
IMAGES = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
n_total = len(IMAGES)
print("Found", n_total, "jpg images.")

# --- parameters ---
MAX_IMAGES = 10000  # you used 10000; change as needed
sel_count = min(MAX_IMAGES, n_total)
print("Selecting", sel_count, "images.")

# choose a random subset like your original code
random.seed(810)
indices = random.sample(range(n_total), sel_count)

# --- processing ---
images_list = []
names_list = []

for idx, i in enumerate(indices):
    image_path = IMAGES[i]
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print("skip (open error):", image_path, e)
        continue

    if DogsOnly:
        # expect annotation file path based on filename without extension
        fname = os.path.splitext(os.path.basename(image_path))[0]
        # search for annotation file - could be in breed subfolders
        ann_file = None
        for root_dir, dirs, files in os.walk(ANN_DIR):
            # many annotation files end with the same base name (no extension)
            if fname in files:
                ann_file = os.path.join(root_dir, fname)
                break
        if ann_file is None:
            # skip images without annotation
            continue
        try:
            tree = ET.parse(ann_file)
            r = tree.getroot()
            objects = r.findall('object')
            for o in objects:
                bndbox = o.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                w = min((xmax - xmin, ymax - ymin))
                img2 = img.crop((xmin, ymin, xmin + w, ymin + w))
                img2 = img2.resize((64, 64), Image.LANCZOS)
                images_list.append(np.asarray(img2, dtype=np.uint8))
                # breed could be derived from ann file path (parent folder)
                breed = os.path.basename(os.path.dirname(ann_file))
                names_list.append(breed)
        except Exception as e:
            print("skip (annotation parse error):", ann_file, e)
            continue
    else:
        # random crop/resizing logic similar to your original
        w, h = img.size
        if (idx % 2 == 0) or (idx % 3 == 0):
            w2 = 100
            h2 = int(h / (w / 100)) if w != 0 else h
            a = 18; b = 0
        else:
            a = 0; b = 0
            if w < h:
                w2 = 64
                h2 = int((64 / w) * h) if w != 0 else h
                b = (h2 - 64) // 2
            else:
                h2 = 64
                w2 = int((64 / h) * w) if h != 0 else w
                a = (w2 - 64) // 2
        try:
            img2 = img.resize((w2, h2), Image.LANCZOS)
            img2 = img2.crop((0 + a, 0 + b, 64 + a, 64 + b))
            images_list.append(np.asarray(img2, dtype=np.uint8))
            names_list.append(os.path.basename(image_path))
        except Exception as e:
            print("skip (resize/crop error):", image_path, e)
            continue

print("Collected", len(images_list), "processed images.")

# convert to numpy array
if len(images_list) == 0:
    raise RuntimeError("No processed images collected. Check DogsOnly flag and annotation availability.")
imagesIn = np.stack(images_list, axis=0)  # shape (N,64,64,3)
namesIn = np.array(names_list)

# --- display random samples ---
import matplotlib.pyplot as plt
rnd = np.random.randint(0, len(imagesIn), size=min(25, len(imagesIn)))
for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        idx_plot = rnd[k*5 + j] if (k*5 + j) < len(rnd) else rnd[0]
        plt.subplot(1,5,j+1)
        plt.axis('off')
        if not DogsOnly:
            plt.title(namesIn[idx_plot], fontsize=9)
        else:
            # if breed names have format 'n02085620-Chihuahua' this keeps after '-'
            nm = namesIn[idx_plot]
            if '-' in nm:
                try:
                    plt.title(nm.split('-')[1], fontsize=9)
                except:
                    plt.title(nm, fontsize=9)
            else:
                plt.title(nm, fontsize=9)
        plt.imshow(imagesIn[idx_plot])
    plt.show()



import gzip, pickle
import os
import numpy as np
import pandas as pd
import random
import shutil
import numpy as np
import os

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader


from scipy import linalg
import pathlib
import urllib
import warnings
from PIL import Image
from tqdm import tqdm


class dog_dataset(Dataset):
    def __init__(self, train_y, train_X, zeros, device):
        self.train_y = torch.Tensor(train_y).to(device)
        self.train_X = torch.Tensor(train_X).to(device)
        self.zeros = torch.Tensor(zeros).to(device)
        
    def __len__(self):
        return len(train_y)

    def __getitem__(self, idx):
        return self.train_y[idx], self.train_X[idx], self.zeros[idx]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bs = 256

train_y = (imagesIn[:10000,:,:,:]/255.).reshape((-1,12288))
train_X = np.zeros((10000,10000))
for i in range(10000): train_X[i,i] = 1
zeros = np.zeros((10000,12288))

data_set = dog_dataset(train_y, train_X, zeros, device)    
data_loader = DataLoader(data_set, bs)


# Sanity check
print(train_y.shape, train_X.shape, zeros.shape)
print(train_y[0].shape, train_X[0].shape, zeros[0].shape)
print(len(train_y), len(train_X), len(zeros))


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(10000, 12288)
        self.conv1 = nn.Conv2d(1, 1, (2, 1), bias=False)
        
    def forward(self, imgs, imgnames):
        x = self.fc1(imgnames)
        x = torch.sigmoid(x)
        x = torch.cat((imgs, x), dim=1).view(-1, 1, 2, 12288)
        x = self.conv1(x)
        return x.view(-1, 12288)


lr = 0.001
epochs = 500

netD = Discriminator().to(device)
optimizerD = optim.Adam(netD.parameters(), lr=lr)
criteria = nn.BCELoss()
netD.conv1.weight = nn.Parameter(torch.Tensor([[[[ -1.0],
                                    [1.0]]]]).to(device))    
for param in netD.conv1.parameters():
    param.requires_grad = False 


# TRAIN DISCRIMINATOR NETWORK
for k in tqdm(range(epochs)):
    for i, (y, X, Zeros) in enumerate(data_loader):
        netD.zero_grad()
        y_pred = netD(Zeros, X)
        loss = criteria(y_pred, y)
        loss.backward()
        optimizerD.step()
    if (k + 1) % 5 == 0:    
        print(f"Epoch: {k+1}/{epochs} | Loss: {loss}")   


for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        xx = torch.Tensor(np.zeros((10000))).to(device)
        xx[np.random.randint(10000)] = 1
        plt.subplot(1,5,j+1)
#         img = netD([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))
        img = netD(torch.Tensor(zeros[0,:]).to(device).reshape((-1,12288)),xx.reshape((-1,10000))).reshape((-1,64,64,3))
        img = img.detach().cpu().numpy()
        img = Image.fromarray((255 * img).astype('uint8').reshape((64,64,3)))
        plt.imshow(img)
    plt.show()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(10000, 12288)
        
    def forward(self, imgnames):
        x = self.fc1(imgnames)
        return x, imgnames.view(-1, 10000)


def show_image():
    plt.figure(figsize=(15,3))
    for j in range(5):
        with torch.no_grad():
            xx = np.zeros((10000))
            xx[np.random.randint(10000)] = 1
            plt.subplot(1,5,j+1)
            inp = torch.Tensor(xx.reshape((-1,10000))).to(device)
            img = netG(inp)[0].reshape((-1,64,64,3)).to("cpu").clone().detach().numpy()
            img = Image.fromarray( (img).astype('uint8').reshape((64,64,3)))
            plt.axis('off')
            plt.imshow(img)
    plt.show()  
    
def show_d_images(imgs, title):
    with torch.no_grad():
        imgs = imgs.reshape((-1,64,64,3))
        plt.figure(figsize=(15,3))
        for j in range(5):
            plt.subplot(1,5,j+1)
            img = imgs[j].detach().cpu().numpy()
            img = Image.fromarray((255*img).astype('uint8').reshape((64,64,3)))
            plt.title(title)
            plt.imshow(img)
        plt.show()


lr = 0.01
beta1 = 0.5
netG = Generator().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=lr)
criterion = nn.MSELoss()

netD.conv1.weight = nn.Parameter(torch.Tensor([[[[ -1.],
                                    [1.]]]]).to(device))

# Discriminator is already trained
for param in netD.parameters():
    param.requires_grad = False


epochs = 70
for epoch in tqdm(range(epochs)):
    for i, (y, X, Zeros) in enumerate(data_loader):
        ############################|
        # (2) Train only the Generator
        ############################
        netG.zero_grad()
        fake, seed = netG(X)
        y_pred = netD(Zeros, seed)
        errG = criterion(fake, y_pred)
        errG.backward()
        optimizerG.step()
    if (epoch+1) % 5 == 0:    
        print(f"Epoch: {epoch+1}: G_Loss: {errG}")
        show_d_images(fake, "Generator output")         


class DogGenerator():
    index = 0   
    t = [
        transforms.RandomCrop((48, 48), padding=None, pad_if_needed=True, fill=0, padding_mode='symmetric'),
        transforms.Resize((64,64))
    ]
    tfms = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
#                                 transforms.RandomApply(t, p=0.5),
                                transforms.ColorJitter(brightness=(1,1.3), contrast=(1,1.3), saturation=0, hue=0)
                               ])
    def getDog(self,seed):
        xx = torch.Tensor(np.zeros((10000))).to(device)
        xx[self.index] = 0.999999
        xx[np.random.randint(10000)] = 0.000001
        img = netG(xx.reshape((-1,10000)))[0].reshape((64,64,3)).detach().cpu().numpy() * 255
        self.index = (self.index+1)%10000
        return self.tfms(Image.fromarray( img.astype('uint8')))


d = DogGenerator()
for k in range(3):
    plt.figure(figsize=(20,5))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = d.getDog(seed = np.random.normal(0,1,100))
        plt.axis('off')
        plt.imshow(img)
    plt.show()


z = zipfile.PyZipFile('images.zip', mode='w')
d = DogGenerator()
for k in range(10000):
    img = d.getDog(np.random.normal(0,1,100))
    f = str(k)+'.png'
    img.save(f,'PNG'); z.write(f); os.remove(f)
z.close()

