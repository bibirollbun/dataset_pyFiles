import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import random
from os.path import expanduser, join, dirname, basename


from fastai.vision.all import *


class cfg(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    seed = 42
    
    # you have 4 sizes: 28, 64, 128, 224
    data_root = '/kaggle/input/cp-3501-retinamnist-v-2024/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b'
    data_path = join(data_root, 'retinamnist_224')
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')
    train_img_size = 224  # 32, 64, 128, 224, 256, or try matching to pretrained model's native image size.
    batch_size = 16
    
    item_tfms = RandomResizedCrop(train_img_size, min_scale=0.5)
    batch_tfms = aug_transforms(mult=2, flip_vert=True, max_rotate=360.)
    
    sample_submission_path = '/kaggle/input/cp-3501-retinamnist-v202501/sample_submission.csv'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # this may crash in some models!
#     torch.backends.cudnn.deterministic = False  # change to False if crashing
    torch.backends.cudnn.benchmark = False
    
set_seed(cfg.seed) 


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
dls = data_block0.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)

# Let's view a raining batch
dls.train.show_batch(max_n=8, nrows=2)


# Create a vision learner using the 'levit_384' pre-trained model
learn = vision_learner(dls,'convnext_tiny', metrics=accuracy)

# Find the learning rate
learn.lr_find(suggest_funcs=(valley, slide))


# Fine tune model
learn.fine_tune(6, base_lr=0.00630)


# Interpreting the performance of a classification model
interp = ClassificationInterpretation.from_learner(learn)

# Plot the confusion matrix for the trained model
interp.plot_confusion_matrix()


import glob
test_files = glob.glob(join(cfg.test_images_dir, '*.png'))
print(test_files[:10])


for img_path in test_files[:10]:
    # https://github.com/fastai/fastai/issues/3366
    with learn.no_bar(), learn.no_logging():
        res = learn.predict(img_path)
    print(res)


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


sub = pd.read_csv(cfg.sample_submission_path)
sub


sub = sub[['image_id']].copy()
# sub
sub = pd.merge(sub, df, how='left', on='image_id', validate='1:1')
sub


sub.to_csv('submission.csv', index=False)
!head submission.csv




