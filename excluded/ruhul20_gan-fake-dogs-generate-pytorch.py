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
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset


from scipy import linalg
import pathlib
import urllib
import warnings
from PIL import Image
from tqdm import tqdm


# --- Hyperparams ---
batch_size = 512
image_size = 64
nc = 3           # num channels
nz = 100         # latent dim
ngf = 256        # used in generator fc -> 256*4*4 (matches your Generator)
ndf = 64         # feature size for discriminator convs (you used 64 first)
lr = 2e-4
beta1 = 0.5
num_epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

try:
    #imagesIn  # if exists, use it
    #assert isinstance(imagesIn, np.ndarray)
    # normalize to [0,1] then to [-1,1]
    x = imagesIn.astype(np.float32) / 255.0
    # Permute channels to (N, C, H, W)
    x = np.transpose(x, (0, 3, 1, 2))
    # Convert to tensor and normalize
    tx = torch.from_numpy(x)
    # apply normalization to [-1,1]
    tx = (tx - 0.5) / 0.5
    dataset = TensorDataset(tx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("Using imagesIn TensorDataset with", len(dataset), "samples.")
except Exception:
    # -------------------------
    # OPTION B: Load from disk using ImageFolder
    # -------------------------
    data_root = '/kaggle/input/generative-dog-images'  # <-- update to folder that contains images (or attach dataset)
    tfms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)   # map to [-1,1]
    ])
    dataset = datasets.ImageFolder(data_root, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    print("Using ImageFolder at", data_root, "with", len(dataset), "samples.")



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bs = 256

# train_y = (imagesIn[:10000,:,:,:]/255.).reshape((-1,12288))
# train_X = np.zeros((10000,10000))
# for i in range(10000): train_X[i,i] = 1
# zeros = np.zeros((10000,12288))

# data_set = dog_dataset(train_y, train_X, zeros, device)    
# train_loader = DataLoader(data_set, bs)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.leaky_relu4 = nn.LeakyReLU(0.2)

        # force the spatial dims to (4,4) regardless of previous size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        x = self.leaky_relu4(self.conv4(x))
        x = self.pool(x)                # <-- ensures fixed (4,4)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 256 * 4 * 4)
        self.relu1 = nn.ReLU(True)
        self.unflatten = nn.Unflatten(1, (256, 4, 4))
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(True)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU(True)
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu1(x)
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.relu2(x)
        x = self.deconv2(x)
        x = self.relu3(x)
        x = self.deconv3(x)
        x = self.tanh(x)
        return x


# D = Discriminator()
# G = Generator()


# criterion = nn.BCELoss()
# d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


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
        for j in range(4):
            plt.subplot(1,5,j+1)
            img = imgs[j].detach().cpu().numpy()
            img = Image.fromarray((255*img).astype('uint8').reshape((64,64,3)))
            plt.title(title)
            plt.imshow(img)
        plt.show()


# --- Assumes previous code defined: G, D, dataloader, criterion, optimizerG, optimizerD, device, nz, fixed_noise, num_epochs, utils, plt ---
import torchvision.utils as vutils


G = Generator().to(device)
D = Discriminator().to(device)

# Initialize weights (recommended for GANs)
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G.apply(weights_init)
D.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# Fixed noise for visualizing progress
fixed_noise = torch.randn(64, nz, device=device)

# Labels
real_label = 1.0
fake_label = 0.0

os.makedirs('samples', exist_ok=True)

best_errG = float('inf')
best_epoch = -1
best_path = 'best_generator.pth'

for epoch in range(1, num_epochs + 1):
    running_errD = 0.0
    running_errG = 0.0
    for i, data in enumerate(dataloader, 0):
        real_images = data[0].to(device)
        b_size = real_images.size(0)

        # ---------------------------
        # (1) Update D
        # ---------------------------
        D.zero_grad()
        # real
        label = torch.full((b_size,), 1.0, device=device)
        output = D(real_images).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # fake
        noise = torch.randn(b_size, nz, device=device)
        fake = G(noise)
        label.fill_(0.0)
        output = D(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # ---------------------------
        # (2) Update G
        # ---------------------------
        G.zero_grad()
        label.fill_(1.0)
        output = D(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        running_errD += errD.item()
        running_errG += errG.item()

        if i % 20 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Step [{i}/{len(dataloader)}] "
                  f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} D(x): {D_x:.6f}")

    # epoch averages
    avg_errD = running_errD / len(dataloader)
    avg_errG = running_errG / len(dataloader)
    
    if epoch % 20 == 0: 
        print(f"--- Epoch {epoch} summary: avg Loss_D={avg_errD:.6f}, avg Loss_G={avg_errG:.6f}")
        show_d_images(fake, "Generator output") 
        
    # Save sample grid each epoch (optional, keeps visual progress)
    with torch.no_grad():
        fake_images = G(fixed_noise).detach().cpu()
    grid = vutils.make_grid((fake_images * 0.5 + 0.5), nrow=8, padding=2)
    
    vutils.save_image(grid, f"samples/epoch_{epoch:03d}.png")

    # --- Save only the best Generator (based on average Generator loss) ---
    # Lower errG is considered better here
    if avg_errG < best_errG:
        best_errG = avg_errG
        best_epoch = epoch
        # Save only G weights and metadata (small)
        torch.save({
            'epoch': epoch,
            'best_errG': best_errG,
            'G_state_dict': G.state_dict()
        }, best_path)
        print(f"Saved new best generator at epoch {epoch} with avg_errG={best_errG:.6f} -> {best_path}")

print("Training finished.")
print(f"Best Generator found at epoch {best_epoch} with avg_errG={best_errG:.6f}")



# --- Load best generator and display samples ---
ckpt = torch.load('best_generator.pth', map_location=device)
G_best = Generator().to(device)
G_best.load_state_dict(ckpt['G_state_dict'])
G_best.eval()

# Generate from fixed noise (the same you used during training)
with torch.no_grad():
    fake_images = G_best(fixed_noise.to(device)).cpu()

# Grid for display (map to [0,1])
grid = vutils.make_grid((fake_images * 0.5 + 0.5), nrow=8, padding=2)
np_grid = grid.permute(1, 2, 0).numpy()

plt.figure(figsize=(8,8))
plt.imshow(np_grid)
plt.axis('off')
plt.title(f'Best Generator: epoch {ckpt.get("epoch", "?")}, errG={ckpt.get("best_errG", "?"):.6f}')
plt.show()

# Save preview file
vutils.save_image(grid, 'best_generator_preview.png')
print("Saved preview: best_generator_preview.png")

# Also show one individual random sample (single image)
rand_z = torch.randn(1, nz, device=device)
with torch.no_grad():
    single_fake = G_best(rand_z).cpu()

single_img = single_fake.squeeze(0)           # (C,H,W)
single_img = (single_img * 0.5 + 0.5).permute(1,2,0).numpy().clip(0,1)

plt.figure(figsize=(4,4))
plt.imshow(single_img)
plt.axis('off')
plt.title('Single random sample from best generator')
plt.show()





