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


!pip install torch torchvision Pillow
!pip install fastai


import random
import shutil
import zipfile
import tempfile

from fastai.vision.all import *
import albumentations as Alb
import PIL
import torch 
from pathlib import Path
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from fastai.vision.all import vision_learner, error_rate, accuracy
import matplotlib.pyplot as plt


#è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�®è§£å‡�
zip_path_train = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
temp_dir_train = tempfile.TemporaryDirectory()
with zipfile.ZipFile(zip_path_train, 'r') as zip_ref:
    zip_ref.extractall(temp_dir_train.name)
train_path = Path(temp_dir_train.name) / 'train'
val_path = Path(temp_dir_train.name) / 'val'  # ä»Šã�¯ä½¿ã‚�ã�ªã��ã�¦ã‚‚OK

# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®è§£å‡�
zip_path_test = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
temp_dir_test = tempfile.TemporaryDirectory()
with zipfile.ZipFile(zip_path_test, 'r') as zip_ref:
    zip_ref.extractall(temp_dir_test.name)
test_path = Path(temp_dir_test.name) / 'test'

# train ãƒ•ã‚©ãƒ«ãƒ€å†…ãƒ•ã‚¡ã‚¤ãƒ«ä¸€è¦§
file_list_train = list(train_path.iterdir())

# çŒ«ã�¨çŠ¬ã�®ãƒ•ã‚¡ã‚¤ãƒ«å��ãƒ‘ã‚¿ãƒ¼ãƒ³ã�§ã‚«ã‚¦ãƒ³ãƒˆ
cat_img_train = [f for f in file_list_train if f.name.startswith("cat")]
dog_img_train = [f for f in file_list_train if f.name.startswith("dog")]

print(f"Cat_img_train: {len(cat_img_train)}")
print(f"Dog_img_train: {len(dog_img_train)}")
print(f"Total_img_train: {len(file_list_train)}")

# val ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�®ä¸­èº«ç¢ºèª�
cat_img_val = val_path / 'cat'
dog_img_val = val_path / 'dog'

if not cat_img_val.exists() or len(list(cat_img_val.iterdir())) == 0:
    print("val/cat ãƒ•ã‚©ãƒ«ãƒ€ã�«ã�¯ç”»åƒ�ã�Œå…¥ã�£ã�¦ã�„ã�¾ã�›ã‚“")
else:
    print(f"val/cat ãƒ•ã‚©ãƒ«ãƒ€ã�« {len(list(cat_img_val.iterdir()))} æ�šã�®ç”»åƒ�ã�Œã�‚ã‚Šã�¾ã�™")

if not dog_img_val.exists() or len(list(dog_img_val.iterdir())) == 0:
    print("val/dog ãƒ•ã‚©ãƒ«ãƒ€ã�«ã�¯ç”»åƒ�ã�Œå…¥ã�£ã�¦ã�„ã�¾ã�›ã‚“")
else:
    print(f"val/dog ãƒ•ã‚©ãƒ«ãƒ€ã�« {len(list(dog_img_val.iterdir()))} æ�šã�®ç”»åƒ�ã�Œã�‚ã‚Šã�¾ã�™")

# test ãƒ‡ãƒ¼ã‚¿ç¢ºèª�
test_files = list(test_path.iterdir())
print(f"Total_img_test: {len(test_files)}")



# ä¸�è‰¯ç”»åƒ�ãƒªã‚¹ãƒˆ
bad_images = [
    'dog.10797.jpg', 'dog.10747.jpg', 'dog.10237.jpg', 'dog.9517.jpg',
    'dog.8736.jpg', 'dog.5604.jpg', 'dog.1043.jpg', 'cat.4338.jpg',
    'dog.10161.jpg', 'dog.10190.jpg'
]

# train_path ã�¯å…ˆã�»ã�©ã�®ã‚³ãƒ¼ãƒ‰ã�§å®šç¾©ã�—ã�Ÿ Path ã‚ªãƒ–ã‚¸ã‚§ã‚¯ãƒˆ
removed_count = 0
for img in bad_images:
    img_path = train_path / img
    if img_path.exists():
        img_path.unlink()  # ãƒ•ã‚¡ã‚¤ãƒ«å‰Šé™¤
        removed_count += 1
        print(f"å‰Šé™¤ã�—ã�¾ã�—ã�Ÿ: {img}")
    else:
        print(f"è¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“ã�§ã�—ã�Ÿ: {img}")

print(f"\nå�ˆè¨ˆ {removed_count} ä»¶ã�®ä¸�è‰¯ç”»åƒ�ã‚’å‰Šé™¤ã�—ã�¾ã�—ã�Ÿ")


#ã‚¯ãƒ©ã‚¹åˆ¥ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�®ä½œæˆ�
cat_dir_train = os.path.join(train_path, 'cat')
dog_dir_train = os.path.join(train_path, 'dog')
os.makedirs(cat_dir_train, exist_ok=True)
os.makedirs(dog_dir_train, exist_ok=True)

for filename in os.listdir(train_path):
    filepath = os.path.join(train_path, filename)
    if os.path.isfile(filepath):
        if filename.startswith('cat'):
            shutil.move(filepath, os.path.join(cat_dir_train, filename))
        elif filename.startswith('dog'):
            shutil.move(filepath, os.path.join(dog_dir_train, filename))

print(f'åˆ†é¡�å®Œäº†: cat={len(os.listdir(cat_dir_train))}, dog={len(os.listdir(dog_dir_train))}')


# ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�®å®šç¾©
class AlbTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

# Albumentationsã�®ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µï¼ˆã�¼ã�‹ã�—å�«ã‚€ï¼‰
def get_augs():
    return Alb.Compose([
        Alb.ShiftScaleRotate(rotate_limit=20, border_mode=0),
        Alb.Transpose(),
        Alb.HorizontalFlip(),  # â†� ä¿®æ­£
        Alb.RandomRotate90(),
        Alb.RandomBrightnessContrast(),
        Alb.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5),
        Alb.CoarseDropout(),
        Alb.OneOf([
            Alb.Blur(blur_limit=3, p=0.5),              # å�˜ç´”ã�ªã�¼ã�‹ã�—
            Alb.GaussianBlur(blur_limit=(3, 5), p=0.5)  # ã‚¬ã‚¦ã‚¹ã�¼ã�‹ã�—
        ], p=0.5)  # 50%ã�®ç¢ºç�‡ã�§ã�©ã�¡ã‚‰ã�‹ã‚’é�©ç”¨
    ])

# ç”»åƒ�1æ�šã�”ã�¨ã�®æ‹¡å¼µå‡¦ç�†
item_tfms = [Resize(224), AlbTransform(get_augs())]

# ãƒ�ãƒƒãƒ�å�˜ä½�ã�®å‰�å‡¦ç�†
batch_tfms = [Normalize.from_stats(*imagenet_stats), *aug_transforms()]



# ç”»åƒ�ãƒ‘ã‚¹
path = Path(train_path)  # train ãƒ•ã‚©ãƒ«ãƒ€ã�¯ cat/ ã�¨ dog/ ã‚’å�«ã‚€

# ãƒ‡ãƒ¼ã‚¿ãƒ­ãƒ¼ãƒ€ãƒ¼ã�®ä½œæˆ�
dls = ImageDataLoaders.from_folder(
    path,
    train='.', 
    valid_pct=0.2,
    seed=42,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
    bs=64
)


# ğŸ”¹ Trainãƒ‡ãƒ¼ã‚¿ã�®è¡¨ç¤º
print("ğŸ”� Train ãƒ‡ãƒ¼ã‚¿ï¼ˆãƒ©ãƒ™ãƒ«ä»˜ã��ï¼‰")
dls.train.show_batch(max_n=9, figsize=(8,8), title='Train')

# ğŸ”¹ Validationãƒ‡ãƒ¼ã‚¿ã�®è¡¨ç¤º
print("ğŸ”� Validation ãƒ‡ãƒ¼ã‚¿ï¼ˆãƒ©ãƒ™ãƒ«ä»˜ã��ï¼‰")
dls.valid.show_batch(max_n=9, figsize=(8,8), title='Validation')


# æ�šæ•°ã�®ç¢ºèª�
print(f"ãƒ‡ãƒ¼ã‚¿æ•°ç¢ºèª�:")
print(f"Train ãƒ‡ãƒ¼ã‚¿æ•°: {len(dls.train_ds)} æ�š")
print(f"Valid ãƒ‡ãƒ¼ã‚¿æ•°: {len(dls.valid_ds)} æ�š")

# Train ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�‹ã‚‰ãƒ©ãƒ³ãƒ€ãƒ ã�«5æ�šè¡¨ç¤º
print("Train ãƒ‡ãƒ¼ã‚¿")
for img, lbl in random.sample(list(dls.train_ds), 5):
    print(f"ãƒ©ãƒ™ãƒ«ID: {lbl}, ã‚¯ãƒ©ã‚¹å��: {dls.vocab[lbl]}")
    display(img.to_thumb(128, 128))

# Valid ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�‹ã‚‰ãƒ©ãƒ³ãƒ€ãƒ ã�«5æ�šè¡¨ç¤º
print("\nValidation ãƒ‡ãƒ¼ã‚¿")
for img, lbl in random.sample(list(dls.valid_ds), 5):
    print(f"ãƒ©ãƒ™ãƒ«ID: {lbl}, ã‚¯ãƒ©ã‚¹å��: {dls.vocab[lbl]}")
    display(img.to_thumb(128, 128))


# ãƒ¢ãƒ‡ãƒ«ã�®ä½œæˆ�
learn = vision_learner(dls, resnet101, metrics=[error_rate, accuracy])

#æœ€é�©ã�ªå­¦ç¿’ç�‡ã�®æ�¨å®š
learn_RN101 = learn.lr_find()
print(learn_RN101) 


# å­¦ç¿’å®Ÿè¡Œ
learn.fit_one_cycle(5, lr_max=learn_RN101.valley)


# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®æº–å‚™
print('Testing', len(test_files), 'items')


# ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�ªã�—ã�® test_dl ã‚’æ˜�ç¤ºçš„ã�«ä½œã‚‹
tst_dl = dls.test_dl(test_files, with_labels=False, shuffle=False)
tst_dl.show_batch(max_n=12)


# æ�¨è«–å®Ÿè¡Œ
startTime = time.time()
probs, _ = learn.tta(dl=tst_dl, n=5, use_max=False)
print('TTA in:', time.time()-startTime, 'secs')

# ãƒ©ãƒ™ãƒ«å�–å¾—ï¼ˆæœ€å¤§ç¢ºç�‡ã�®ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ï¼‰
preds = probs.argmax(dim=1)
class_names = learn.dls.vocab


# ãƒ©ãƒ³ãƒ€ãƒ ã�«12æ�šè¡¨ç¤º
idxs = random.sample(range(len(preds)), 12)
fig, axs = plt.subplots(3, 4, figsize=(12, 9))

for ax, idx in zip(axs.flatten(), idxs):
    img = PILImage.create(tst_dl.items[idx])
    pred_label = preds[idx].item()
    pred_class = class_names[pred_label]
    prob = probs[idx][pred_label].item()
    ax.imshow(img)
    ax.set_title(f'{pred_class} ({pred_label})\nprob: {prob:.2f}')
    ax.axis('off')

plt.tight_layout()
plt.show()


# æ��å‡ºç”¨ DataFrame ã�®ä½œæˆ�
subm_df = pd.DataFrame()
subm_df['id'] = [item.stem for item in tst_dl.items]        # ãƒ•ã‚¡ã‚¤ãƒ«å��ï¼ˆæ‹¡å¼µå­�ã�ªã�—ï¼‰
subm_df['label'] = preds.numpy()                            # argmax ã�«ã‚ˆã‚‹ãƒ©ãƒ™ãƒ«ï¼ˆ0=cat, 1=dogï¼‰

# CSV ãƒ•ã‚¡ã‚¤ãƒ«ã�¨ã�—ã�¦ä¿�å­˜
subm_df.to_csv('submission.csv', index=False)
print("âœ… submission.csv ã‚’ä¿�å­˜ã�—ã�¾ã�—ã�Ÿã€‚")


# ã‚½ãƒ•ãƒˆãƒ�ãƒƒã‚¯ã‚¹ã�§ã‚¯ãƒ©ã‚¹1ï¼ˆdogï¼‰ã�®ç¢ºç�‡ã‚’å�–å¾—
probs_softmax = torch.softmax(probs, dim=1)  # shape: [N, 2]
dog_probs = probs_softmax[:, 1].clip(0.005, 0.995).numpy()  # é��å‰°ã�ª0,1ã‚’é˜²ã��

# DataFrame ã�®ä½œæˆ�
subm_softmax_df = pd.DataFrame({
    'id': [item.stem for item in tst_dl.items],
    'label': dog_probs
})

# CSVä¿�å­˜
subm_softmax_df.to_csv('submission_softmax.csv', index=False)
print("âœ… submission_softmax.csv ã‚’ä¿�å­˜ã�—ã�¾ã�—ã�Ÿã€‚")

