# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import random
from os.path import expanduser, join, dirname, basename

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from fastai.vision.all import *
# Import VGG16 (using th batch-normalized version for better stability)
from torchvision.models import vgg16_bn


# let's create an easy config
class cfg(dict):
    # dot.notation access to dictionary attributes
    # Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    seed = 1
    
    # you have 4 sizes: 28, 64, 128, 224
    data_root = '/kaggle/input/cp-3501-retinamnist-v-2024/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b'
    data_path = join(data_root, 'retinamnist_28')
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')
    train_img_size = 32   # 32, 64, 128, 224, 256, or try matching to pretrained model's native image size.
    batch_size = 32 
    
    item_tfms = RandomResizedCrop(train_img_size, min_scale=0.5)
    batch_tfms = aug_transforms(mult=2, flip_vert=True, max_rotate=360.)
    
    sample_submission_path = '/kaggle/input/cp-3501-retinamnist-v-2024/sample_submission.csv'




# before we do anything, let's fix randomness!!
# Function to set the random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # this may crash in some models!
#     torch.backends.cudnn.deterministic = False  # change to False if crashing
    torch.backends.cudnn.benchmark = False
    
set_seed(cfg.seed) 


# let's follow lesson-2: https://www.kaggle.com/code/dmitrykonovalov/fastai-lesson2-part1b-02-production-v2024a
# bears = DataBlock(
#     blocks=(ImageBlock, CategoryBlock), 
#     get_items=get_image_files, 
#     splitter=RandomSplitter(valid_pct=0.2, seed=42),
#     get_y=parent_label,
#     item_tfms=Resize(128))


# 
# Define the path to your dataset
# path = Path('/path/to/your/data')

# Create a DataBlock
# data_block0 is our data block without any agmentations
data_block0 = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=cfg.seed),
    get_y=parent_label,
#     item_tfms=Resize(224),
    item_tfms=Resize(cfg.train_img_size),
#     batch_tfms=aug_transforms()  # let's connect augmentations later
)

# Create DataLoaders
# dls = data_block.dataloaders(data_path, bs=64)
dls0 = data_block0.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)

# Let's view a raining batch
dls0.train.show_batch(max_n=8, nrows=2)


dls0.valid.show_batch(max_n=4, nrows=1)  # validation samples are not randomized


# Here is fastai's lesson code:
# bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
# dls = bears.dataloaders(path)
# dls.train.show_batch(max_n=4, nrows=1, unique=True)

# bears is now data_block
# be careful here: 
# if you have data_block = data_block.new() and run this cell multiple times you will get multiples augs attached
data_block1 = data_block0.new(item_tfms=RandomResizedCrop(cfg.train_img_size, min_scale=0.3))
dls1 = data_block1.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls1.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


dls1.train.show_batch(max_n=4, nrows=1, unique=False)  # False= will show different images


# Here is fastai's lesson code:
# bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
# dls = bears.dataloaders(path)
# dls.train.show_batch(max_n=8, nrows=2, unique=True)

# bears is now data_block
# data_block2 = data_block0.new(batch_tfms=aug_transforms(mult=2))  # try different mult. from data_block0
data_block2 = data_block1.new(batch_tfms=aug_transforms(mult=2))  # try different mult. from data_block1 will be double trans

dls2 = data_block2.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls2.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


# let's verify that validation images are not augmented:
dls2.valid.show_batch(max_n=4, nrows=1)  # dls2 <--- validation samples are not randomized


# let's verify that validation images are not augmented:  
dls0.valid.show_batch(max_n=4, nrows=1)  # dls0 <---- validation samples are not randomized


# Data augmentation. Let's examine options
?aug_transforms


# NOTE! possible better augment options
# flip_vert=True
# max_rotate=360.  # default 10 degrees is too small for our images


# Now that we played with augmentaions, let's move them to cfg:

# Lesson's example uses both batch_tfms and item_tfms
data_block = data_block0.new(item_tfms=cfg.item_tfms, batch_tfms=cfg.batch_tfms)  

dls = data_block.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


# Create a learner
# learn = vision_learner(dls, resnet18, metrics=accuracy)
learn = cnn_learner(dls, vgg16_bn, pretrained=True, metrics=accuracy)

# Find an optimal learning rate
# learn.lr_find()
learn.lr_find(suggest_funcs=(valley, slide))


?learn.lr_find


?learn.fine_tune


# valley=0.0005754399462603033, slide=0.0030199517495930195)
# Let's try slide's
lr = 0.003  


learn.fine_tune(3, base_lr=lr)


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


import glob
test_files = glob.glob(join(cfg.test_images_dir, '*.png'))
print(test_files[:10])


?learn.predict


# Iterate over the list of image paths and make predictions
# let's try a few first files:
for img_path in test_files[:10]:
    # https://github.com/fastai/fastai/issues/3366
    with learn.no_bar(), learn.no_logging():
        res = learn.predict(img_path)
    print(res)
#     print(1)


from tqdm import tqdm
# from tqdm.notebook import tqdm
# Iterate over the list of image paths and make predictions
preds = []
pred_ids = []
for img_path in tqdm(test_files):  # tqdm not working?  learn was breaking it https://github.com/fastai/fastai/issues/3366
# for img_path in test_files:
    img_id = basename(img_path)
    pred_ids += [img_id]
    # https://github.com/fastai/fastai/issues/3366
    with learn.no_bar(), learn.no_logging():
        res = learn.predict(img_path)
    pred_label, val, probs = res
    preds += [pred_label]
#     print(res)
    
df = pd.DataFrame({'image_id':pred_ids})
df['label'] = preds
df


# let's sort it as per given submission sample
sub = pd.read_csv(cfg.sample_submission_path)
sub


# let's sort it as per given submission sample
sub = sub[['image_id']].copy()
# sub
sub = pd.merge(sub, df, how='left', on='image_id', validate='1:1')
sub


sub.to_csv('submission.csv', index=False)
!head submission.csv




