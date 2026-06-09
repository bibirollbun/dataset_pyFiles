#Adding all libraries needed 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch 
import random
from os.path import expanduser, join, dirname, basename
from fastai.vision.all import * #Normalize, imagenet_stats also imported here 


# Create an easy config
class cfg(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    seed = 1
    
    # you have 4 sizes: 28, 64, 128, 224, different models might be more efficient with different sizes 
    data_root = '/kaggle/input/cp-3501-retinamnist-v-2024/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b'
    data_path = join(data_root, 'retinamnist_224') 
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')
    train_img_size = 224   # 32, 64, 128, 224, matched to pretrained model's native image size.
    batch_size = 32
    
    item_tfms = RandomResizedCrop(train_img_size, min_scale=0.5)
    batch_tfms = aug_transforms(mult=2, flip_vert=True, max_rotate=360.)
    
    sample_submission_path = '/kaggle/input/cp-3501-retinamnist-v-2024/sample_submission.csv'


# Fix randomness
# Function to set the random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
#     torch.backends.cudnn.deterministic = False  # change to False if crashing [only some models are affected]
    torch.backends.cudnn.benchmark = False
    
set_seed(cfg.seed) 


# Create a DataBlock
# data_block0 is our data block without any augmentations
data_block0 = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=cfg.seed),
    get_y=parent_label,
    item_tfms=Resize(cfg.train_img_size),
)

batch_tfms = [Resize(224), Normalize.from_stats(*imagenet_stats)]
# Create DataLoaders
dls0 = data_block0.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)

# View a training batch
dls0.train.show_batch(max_n=8, nrows=2)


dls0.valid.show_batch(max_n=4, nrows=1)  # validation samples are not randomized


#Augmentation with same images
#Data block 1 starting augmentation 
data_block1 = data_block0.new(item_tfms=RandomResizedCrop(cfg.train_img_size, min_scale=0.3)) 
dls1 = data_block1.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls1.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


#Augmentation with unqiue images
dls1.train.show_batch(max_n=4, nrows=1, unique=False) 


# Data augmentation 
?aug_transforms


#Datablock 2 Augmentation. More augmentation to be added


data_block2 = data_block1.new(batch_tfms=aug_transforms(mult=2)) 
dls2 = data_block2.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls2.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


#Validation images are not augmented 
dls2.valid.show_batch(max_n=4, nrows=1)  # dls2: validation samples are not randomized


dls0.valid.show_batch(max_n=4, nrows=1) # dls0: validation samples are not randomized


# To cfg

# Uses both batch_tfms and item_tfms
data_block = data_block0.new(item_tfms=cfg.item_tfms, batch_tfms=cfg.batch_tfms)  

dls = data_block.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls.train.show_batch(max_n=4, nrows=1, unique=True) 


# Create a learner

learn = vision_learner(dls, alexnet, metrics=accuracy)


# Find an optimal learning rate
# learn.lr_find()
learn.lr_find(suggest_funcs=(valley, slide))


?learn.lr_find


?learn.fine_tune


# Let's try slide's and valley

lr = 0.011


learn.fine_tune(10, base_lr=lr,wd = 0.01,freeze_epochs=5)


#Confusion_matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


#For improving the model
import glob
test_files = glob.glob(join(cfg.test_images_dir, '*.png'))
print(test_files[:10])


?learn.predict


# Iterate over the list of image paths and make predictions

for img_path in test_files[:10]:
    # https://github.com/fastai/fastai/issues/3366
    with learn.no_bar(), learn.no_logging():
        res = learn.predict(img_path)
    print(res)


from tqdm import tqdm

# Iterate over the list of image paths and make predictions
preds = []
pred_ids = []
for img_path in tqdm(test_files):  
    img_id = basename(img_path)
    pred_ids += [img_id]
    # https://github.com/fastai/fastai/issues/3366
    with learn.no_bar(), learn.no_logging():
        res = learn.predict(img_path)
    pred_label, val, probs = res
    preds += [pred_label]
    
df = pd.DataFrame({'image_id':pred_ids})
df['label'] = preds
df


# Sort it as per given submission sample
sub = pd.read_csv(cfg.sample_submission_path)
sub


sub = sub[['image_id']].copy()
# sub
sub = pd.merge(sub, df, how='left', on='image_id', validate='1:1')
sub


#Submission file
sub.to_csv('submission.csv', index=False)
!head submission.csv




