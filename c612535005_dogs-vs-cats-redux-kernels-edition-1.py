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


import numpy as np
import pandas as pd
import os
import time
from fastai.vision.all import *
import zipfile
import albumentations as Alb
import random
import shutil
import PIL
import torch 
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from fastai.vision.augment import RandTransform
from albumentations.pytorch import ToTensorV2


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


from sklearn.model_selection import StratifiedShuffleSplit

# 1. ãƒ‡ãƒ¼ã‚¿ä¸€è¦§ã�¨ãƒ©ãƒ™ãƒ«ã‚’æº–å‚™
files = get_image_files(train_path)
labels = [1 if f.name.startswith('dog') else 0 for f in files]  # 0=cat,1=dog

# 2. StratifiedSplitã�§ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã‚’åˆ†å‰²
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(splitter.split(files, labels))

# 3. DataBlockã‚’å®šç¾©
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=lambda _: files,
    splitter=IndexSplitter(valid_idx),
    get_y=lambda o: 'dog' if o.name.startswith('dog') else 'cat',
    item_tfms=Resize(256),
    batch_tfms=aug_transforms()
)

# 4. DataLoadersä½œæˆ�
dls = dblock.dataloaders(source=None, bs=64)

# 5. ç¢ºèª�ï¼ˆå­¦ç¿’ãƒ»æ¤œè¨¼ã�®æ�šæ•°ã�¨ã‚¯ãƒ©ã‚¹åˆ†å¸ƒï¼‰
print(f"Train count: {len(dls.train_ds)}, Valid count: {len(dls.valid_ds)}")

# 6. ãƒ©ãƒ™ãƒ«åˆ†å¸ƒã�®ç¢ºèª�
from collections import Counter
print(f"Train labels: {Counter([dls.train_ds.items[i].name.startswith('dog') for i in range(len(dls.train_ds))])}")
print(f"Valid labels: {Counter([dls.valid_ds.items[i].name.startswith('dog') for i in range(len(dls.valid_ds))])}")

# ğŸ”¹ Trainãƒ‡ãƒ¼ã‚¿ã�®è¡¨ç¤º
print("ğŸ”� Train ãƒ‡ãƒ¼ã‚¿ï¼ˆãƒ©ãƒ™ãƒ«ä»˜ã��ï¼‰")
dls.train.show_batch(max_n=9, figsize=(8,8), title='Train')

# ğŸ”¹ Validationãƒ‡ãƒ¼ã‚¿ã�®è¡¨ç¤º
print("ğŸ”� Validation ãƒ‡ãƒ¼ã‚¿ï¼ˆãƒ©ãƒ™ãƒ«ä»˜ã��ï¼‰")
dls.valid.show_batch(max_n=9, figsize=(8,8), title='Validation')


# ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�®å®šç¾©
class AlbTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
    
def get_augs():
    return Alb.Compose([
        Alb.Affine(
            rotate=(-20, 20),
            translate_percent=(0, 0.1),
            scale=(0.9, 1.1),
            shear=0,
            border_mode=0,   # å¢ƒç•Œã�®åŸ‹ã‚�æ–¹ï¼ˆ0ã�¯ä¸€å®šå€¤ã�§åŸ‹ã‚�ã‚‹ï¼‰
        ),
        Alb.Transpose(),
        Alb.HorizontalFlip(),
        Alb.RandomRotate90(),
        Alb.RandomBrightnessContrast(),
        Alb.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5),
        Alb.CoarseDropout()
    ])

item_tfms = [Resize(256), AlbTransform(get_augs())] 
batch_tfms = [Normalize.from_stats(*imagenet_stats), *aug_transforms()]


# ãƒ¢ãƒ‡ãƒ«ã�®ä½œæˆ�
learn = vision_learner(dls, resnet101, metrics=[error_rate, accuracy])

#æœ€é�©ã�ªå­¦ç¿’ç�‡ã�®æ�¨å®š
learn_RN101 = learn.lr_find()
print(learn_RN101) 


# å­¦ç¿’å®Ÿè¡Œ
learn.fit_one_cycle(5, lr_max=learn_RN101.valley)


# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®æº–å‚™
print('Testing', len(test_files), 'items')


tst_dl = dls.test_dl(test_files, with_labels=False, shuffle=False, batch_tfms=None)
batch_tfms=[Normalize.from_stats(*imagenet_stats)]
tst_dl.show_batch(max_n=12)


# ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�ªã�—ã�® test_dl ã‚’æ˜�ç¤ºçš„ã�«ä½œã‚‹
tst_dl = dls.test_dl(test_files, with_labels=False, shuffle=False)
tst_dl.show_batch(max_n=12)


# æ�¨è«–å®Ÿè¡Œ
startTime = time.time()
probs, _ = learn.tta(dl=tst_dl, n=5, use_max=False)
print('TTA in:', time.time()-startTime, 'secs')


print(learn.dls.vocab)       # ä¾‹: CategoryMap(['cat', 'dog'], sort=False)
print(learn.dls.vocab.o2i)   # ä¾‹: {'cat': 0, 'dog': 1}


# dogã‚¯ãƒ©ã‚¹ã�®ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹å�–å¾—
dog_idx = learn.dls.vocab.o2i['dog']
dog_probs = probs[:, dog_idx]

# ãƒ•ã‚¡ã‚¤ãƒ«å��ã�‹ã‚‰IDæŠ½å‡º
ids = [int(f.name.split('.')[0]) for f in tst_dl.items]

# æ��å‡ºç”¨CSVä½œæˆ�
import pandas as pd
submission = pd.DataFrame({'id': ids, 'label': dog_probs.numpy()})
submission.to_csv('submission.csv', index=False)


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

