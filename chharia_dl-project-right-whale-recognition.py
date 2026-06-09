# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Fast unzip for Kaggle
import os
import subprocess
import time
from zipfile import ZipFile

SRC = "/kaggle/input/noaa-right-whale-recognition/imgs.zip"   # zip in read-only input
DST = "/kaggle/working/imgs"                                 # writable extraction target

# Safety: don't re-extract if already present
if os.path.exists(DST) and os.listdir(DST):
    print(f"Already extracted to {DST} (skipping).")
else:
    os.makedirs(DST, exist_ok=True)
    t0 = time.time()
    try:
        subprocess.run(['unzip', '-qq', SRC, '-d', DST], check=True)
        print(f"Unzipped with system 'unzip' into {DST} in {time.time() - t0:.1f}s")
    except FileNotFoundError:
        print("System 'unzip' not available — falling back to Python ZipFile.extractall()")


df = pd.read_csv('/kaggle/input/noaa-right-whale-recognition/train.csv')


df.head()


import random
from pathlib import Path
from PIL import Image


imgs_dir = Path("/kaggle/working/imgs")
img_paths = list(imgs_dir.rglob("*.jpg"))
chosen = random.choice(img_paths)
img = Image.open(chosen)

plt.figure(figsize=(10,6))
plt.imshow(img)
plt.axis("off")
plt.title(chosen.name)
plt.show()


img_paths[:5]


imgs_dir = Path("/kaggle/working/imgs/imgs")
filename = "w_7440" + ".jpg"

img_path = next(imgs_dir.rglob(filename))  # returns a single Path

img = Image.open(img_path)

plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis("off")
plt.title(img_path.name)
plt.show()



imgs_dir = Path("/kaggle/working/imgs/imgs")
filename = "w_7489" + ".jpg"

img_path = next(imgs_dir.rglob(filename))  # returns a single Path

img = Image.open(img_path)

plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis("off")
plt.title(img_path.name)
plt.show()


df = df[df['Image']!='w_7489.jpg']


df.info


print(len(df), len(df['whaleID'].unique()))


counts_per_id = df['whaleID'].value_counts()
counts_per_id


freq = counts_per_id.value_counts().sort_index()
freq


fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))

x = freq.index.values
y = freq.values

ax1.bar(x, y, edgecolor='black')
ax1.set_xlabel("Number of images per whale ID")
ax1.set_ylabel("Number of whale IDs")
ax1.set_title("Count of whale IDs by number of images")
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# reduce x-tick clutter: show at most ~20 ticks evenly spaced
max_ticks = 20
if len(x) > max_ticks:
    step = max(1, int((x.max() - x.min()) / (max_ticks - 1)))
    ticks = np.arange(x.min(), x.max() + 1, step)
else:
    ticks = x

ax1.set_xticks(ticks)
ax1.set_xticklabels([str(t) for t in ticks], rotation=45)

plt.tight_layout()
plt.show()


imgs_dir = Path("/kaggle/working/imgs")
img_paths = list(imgs_dir.rglob("*.jpg"))

widths = []
heights = []

for p in img_paths:
    with Image.open(p) as img:
        w, h = img.size
        widths.append(w)
        heights.append(h)


print(len(widths), len(heights))


plt.figure(figsize=(12, 5))

# WIDTH HISTOGRAM
plt.subplot(1, 2, 1)
plt.hist(widths, bins=30, edgecolor='black', alpha=0.8)
plt.title("Image Width Distribution", fontsize=14)
plt.xlabel("Width (pixels)", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# stats on width
w_mean = int(np.mean(widths))
w_median = int(np.median(widths))
w_min = min(widths)
w_max = max(widths)

plt.text(0.98, 0.95,
         f"Mean: {w_mean}\nMedian: {w_median}\nMin: {w_min}\nMax: {w_max}",
         ha='right', va='top', transform=plt.gca().transAxes,
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# HEIGHT HISTOGRAM
plt.subplot(1, 2, 2)
plt.hist(heights, bins=30, edgecolor='black', alpha=0.8)
plt.title("Image Height Distribution", fontsize=14)
plt.xlabel("Height (pixels)", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# stats on height
h_mean = int(np.mean(heights))
h_median = int(np.median(heights))
h_min = min(heights)
h_max = max(heights)

plt.text(0.98, 0.95,
         f"Mean: {h_mean}\nMedian: {h_median}\nMin: {h_min}\nMax: {h_max}",
         ha='right', va='top', transform=plt.gca().transAxes,
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))

# first histogram (widths)
plt.hist(widths, bins=30, alpha=0.6, edgecolor='black', label='Widths')

# create a second axis sharing x or y
ax2 = plt.gca().twinx()

# second histogram (heights)
ax2.hist(heights, bins=30, alpha=0.6, edgecolor='black', color='orange', label='Heights')

plt.title("Combined Histogram of Image Widths and Heights", fontsize=14)
plt.xlabel("Pixel Value (Width / Height)", fontsize=12)
plt.grid(alpha=0.3, linestyle='--')

# legends (from both axes)
plt.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.show()


import shutil
from pathlib import Path

out_dir = Path("/kaggle/working/imgs_resized")

# Delete the directory if it exists
if out_dir.exists():
    shutil.rmtree(out_dir)

print("Output directory deleted.")


from multiprocessing import Pool, cpu_count
from tqdm import tqdm

imgs_dir = Path("/kaggle/working/imgs")
out_dir = Path("/kaggle/working/imgs_resized")
out_dir.mkdir(exist_ok=True)

img_paths = list(imgs_dir.rglob("*.jpg"))
max_side = 480


def resize_one(path):
    try:
        with Image.open(path) as img:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            img.save(out_dir / path.name)
        return True
    except:
        return False

# multiprocessing with tqdm progress bar
with Pool(cpu_count()) as p:
    list(tqdm(p.imap_unordered(resize_one, img_paths, chunksize=20),
              total=len(img_paths),
              desc="Resizing images",
              ncols=90))

print("Resized images saved to:", out_dir)


imgs_dir_resized = Path("/kaggle/working/imgs_resized")
img_paths_resized = list(imgs_dir_resized.rglob("*.jpg"))

widths_resized = []
heights_resized = []

for p in img_paths_resized:
    with Image.open(p) as img:
        w, h = img.size
        widths_resized.append(w)
        heights_resized.append(h)


plt.figure(figsize=(12, 5))

# WIDTH HISTOGRAM
plt.subplot(1, 2, 1)
plt.hist(widths_resized, bins=30, edgecolor='black', alpha=0.8)
plt.title("Image Width Distribution", fontsize=14)
plt.xlabel("Width (pixels)", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# stats on width
w_mean = int(np.mean(widths_resized))
w_median = int(np.median(widths_resized))
w_min = min(widths_resized)
w_max = max(widths_resized)

plt.text(0.98, 0.95,
         f"Mean: {w_mean}\nMedian: {w_median}\nMin: {w_min}\nMax: {w_max}",
         ha='right', va='top', transform=plt.gca().transAxes,
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# HEIGHT HISTOGRAM
plt.subplot(1, 2, 2)
plt.hist(heights_resized, bins=30, edgecolor='black', alpha=0.8)
plt.title("Image Height Distribution", fontsize=14)
plt.xlabel("Height (pixels)", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# stats on height
h_mean = int(np.mean(heights_resized))
h_median = int(np.median(heights_resized))
h_min = min(heights_resized)
h_max = max(heights_resized)

plt.text(0.98, 0.95,
         f"Mean: {h_mean}\nMedian: {h_median}\nMin: {h_min}\nMax: {h_max}",
         ha='right', va='top', transform=plt.gca().transAxes,
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 5))

# first histogram (widths)
plt.hist(widths_resized, bins=30, alpha=0.6, edgecolor='black', label='Widths')

# create a second axis sharing x or y
ax2 = plt.gca().twinx()

# second histogram (heights)
ax2.hist(heights_resized, bins=30, alpha=0.6, edgecolor='black', color='orange', label='Heights')

plt.title("Combined Histogram of Image Widths and Heights", fontsize=14)
plt.xlabel("Pixel Value (Width / Height)", fontsize=12)
plt.grid(alpha=0.3, linestyle='--')

# legends (from both axes)
plt.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.show()


from fastai.vision.all import * 


import torch
import torch.nn.functional as F
from fastai.metrics import Metric

class MultiLogLoss(Metric):
    "Multiclass logarithmic loss (average cross-entropy). Handles logits or probabilities."
    def __init__(self, eps=1e-7):
        self.eps = eps
        self._name = 'log_loss'
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.count = 0

    def accumulate(self, learn):
        "Called for each batch: uses learn.pred and learn.y"
        preds = learn.pred
        targs = learn.y

        # ensure targs is (bs,) of indices
        if targs.ndim > 1:
            targs = targs.view(-1)

        # If preds look like probabilities (rows sum to ~1), use -log(p)
        if preds.ndim > 1 and torch.allclose(preds.sum(dim=1), torch.ones(preds.size(0)).to(preds.device), atol=1e-3):
            probs = preds.clamp(self.eps, 1.0 - self.eps)
            batch_loss = -torch.log(probs[range(probs.size(0)), targs]).sum()
        else:
            # treat preds as logits -> use numerically-stable cross_entropy
            batch_loss = F.cross_entropy(preds, targs, reduction='sum')

        # accumulate on CPU
        self.total_loss += batch_loss.detach().cpu().item()
        self.count += targs.size(0)

    @property
    def value(self):
        if self.count == 0: 
            return None
        return self.total_loss / self.count


import numpy as np
import matplotlib.pyplot as plt

def plot_multilogloss(learner):
    """
    Plots validation MultiLogLoss per epoch from a FastAI learner 
    that was trained using MultiLogLoss() as a metric.
    """
    # metric names corresponding to recorder.values columns
    mnames = learner.recorder.metric_names[1:-1]
    vals   = np.array(learner.recorder.values)

    idx = mnames.index('multi_log_loss')
    val_logloss = vals[:, idx]

    epochs = np.arange(1, len(val_logloss) + 1)

    plt.plot(epochs, val_logloss, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('validation log_loss')
    plt.title('Validation MultiLogLoss per epoch')
    plt.grid(True)
    plt.show()


items = df['Image']
item = items[0]
item


image_file = f'imgs_resized/{item}'
image = PILImage.create(image_file)
image


image2whaleID = {o.Image: o.whaleID for o in df.itertuples()}
label = image2whaleID[item]
label


def get_image_file(item):
    return f'imgs_resized/{item}'

def get_label(item):
    return image2whaleID[item]


x_pipe = [get_image_file, PILImage.create]
y_pipe = [get_label, Categorize()]


# make sure at least one of each whale is in training set, then randomly split
must_train_whales = df.groupby('whaleID').first()['Image']
must_train_ids = pd.Series(df['Image'].index, index=df['Image']).get(must_train_whales)
must_train_ids = set(must_train_ids)


train_ids, valid_ids = RandomSplitter(seed=42)(items)
print(f"Before: train_ids={len(train_ids)}, valid_ids={len(valid_ids)}")
train_ids = L(set(train_ids).union(must_train_ids))
valid_ids = L(set(valid_ids) - must_train_ids)
print(f"After: train_ids={len(train_ids)}, valid_ids={len(valid_ids)}")
splits = (train_ids, valid_ids)


dss = Datasets(items, [x_pipe, y_pipe], splits=splits)


dss.show(dss[0])


after_item = [ToTensor(), Resize((320, 480))]
after_batch = [IntToFloatTensor(), *aug_transforms(size=(224, 336))]


dls = dss.dataloaders(32, after_item=after_item, after_batch=after_batch)
dls.show_batch()


metrics = [error_rate, MultiLogLoss()]
learn_resnet26d_w = vision_learner(dls, 'resnet26d', metrics=metrics).to_fp16()


learn_resnet26d_w.lr_find(num_it=100)


learn_resnet26d_w.fine_tune(10, 0.009120108559727669)
learn_resnet26d_w.recorder.plot_loss()


plot_multilogloss(learn_resnet26d_w)


def submit(learn, path):
    test_df = pd.read_csv('/kaggle/input/noaa-right-whale-recognition/sample_submission.csv')
    test_dl = learn.dls.test_dl(test_df['Image'])
    
    preds, targs = learn.get_preds(dl=test_dl)
    
    df = pd.DataFrame(preds.numpy(), columns=learn.dls.vocab)
    df["Image"] = test_df['Image']

    preds_path = path
    df.to_csv(preds_path, index=False)


submit(learn_resnet26d_w, "submission_resnet26_w.csv")


metrics = [error_rate, MultiLogLoss()]
learn_resnet50d = vision_learner(dls, 'resnet50d', metrics=metrics).to_fp16()


learn_resnet50d.lr_find(num_it=100)


learn_resnet50d.fine_tune(10, 0.002511886414140463)
learn_resnet50d.recorder.plot_loss()


plot_multilogloss(learn_resnet50d)


submit(learn_resnet50d, "submission_resnet50.csv")


metrics = [error_rate, MultiLogLoss()]
learn_effb3 = vision_learner(dls, 'efficientnet_b3', metrics=metrics).to_fp16()


learn_effb3.lr_find(num_it=100)


learn_effb3.fine_tune(10, 0.0020892962347716093)
learn_effb3.recorder.plot_loss()


plot_multilogloss(learn_effb3)


submit(learn_effb3, "submission_efficientnet_b3.csv")


import pandas as pd
import numpy as np

a = pd.read_csv('/kaggle/working/submission_resnet50.csv')     # model A
b = pd.read_csv('/kaggle/working/submission_efficientnet_b3.csv')      # model B

# Ensure same order and identical Image column
assert (a['Image'].values == b['Image'].values).all(), "Image orders differ!"

# columns that are class probs
cols = [c for c in a.columns if c != 'Image']

# simple (unweighted) average
probs = (a[cols].values + b[cols].values) / 2.0

# re-normalize rows just in case
probs = probs / probs.sum(axis=1, keepdims=True)

out = pd.DataFrame(probs, columns=cols)
out['Image'] = a['Image']
out.to_csv('ensemble_avg.csv', index=False)


import torch
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            # --- Layer 1: Detect Edges ---
            # Input: 3 colors -> Output: 16 features
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 

            # --- Layer 2: Detect Shapes ---
            # Input: 16 features -> Output: 32 features
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),

            # --- Layer 3: Detect Objects ---
            # Input: 32 features -> Output: 64 features
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            
            # --- Classifier Head ---
            # Squashes everything to a single list of numbers and predicts
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# instantiate and train
n_classes = dls.c
model = BaseCNN(n_classes)
learn_cnn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=metrics).to_fp16()
learn_cnn.lr_find(num_it=100)


learn_cnn.fit_one_cycle(10, 0.005248074419796467)


learn_cnn.recorder.plot_loss()


plot_multilogloss(learn_cnn)


# predict on test and write submission
import pandas as pd
test_df = pd.read_csv('/kaggle/input/noaa-right-whale-recognition/sample_submission.csv')
test_dl = learn_cnn.dls.test_dl(test_df['Image'])
preds, _ = learn_cnn.get_preds(dl=test_dl)
probs = preds.cpu().numpy()
out = pd.DataFrame(probs, columns=learn_cnn.dls.vocab)
out['Image'] = test_df['Image']
out.to_csv('base_cnn_submission.csv', index=False)

