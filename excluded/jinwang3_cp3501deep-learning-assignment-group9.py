import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import random
from os.path import expanduser, join, dirname, basename


from fastai.vision.all import *


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
    data_path = join(data_root, 'retinamnist_224')
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')
    train_img_size = 224   # 32, 64, 128, 224, 256, or try matching to pretrained model's native image size.
    batch_size = 18 
    
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


# if you have data_block = data_block.new() and run this cell multiple times you will get multiples augs attached
data_block1 = data_block0.new(item_tfms=RandomResizedCrop(cfg.train_img_size, min_scale=0.3))
dls1 = data_block1.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls1.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


dls1.train.show_batch(max_n=4, nrows=1, unique=False)  # False= will show different images


data_block2 = data_block1.new(batch_tfms=aug_transforms(mult=2))  # try different mult. from data_block1 will be double trans

dls2 = data_block2.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls2.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


# let's verify that validation images are not augmented:
dls2.valid.show_batch(max_n=4, nrows=1)  # dls2 <--- validation samples are not randomized


# let's verify that validation images are not augmented:  
dls0.valid.show_batch(max_n=4, nrows=1)  # dls0 <---- validation samples are not randomized


# Data augmentation. Let's examine options
?aug_transforms


data_block = data_block0.new(item_tfms=cfg.item_tfms, batch_tfms=cfg.batch_tfms)  

dls = data_block.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


# Create a learner
learn = vision_learner(dls, densenet121, metrics=accuracy)

# Find an optimal learning rate
# learn.lr_find()
learn.lr_find(suggest_funcs=(valley, slide))


?learn.lr_find


?learn.fine_tune


# valley=0.0012022644514217973, slide=0.005248074419796467
# Let's try slide
lr = 0.005248074419796467


from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback

# learn.fine_tune(
#     70, 
#     base_lr=lr,
#     cbs=[
#         EarlyStoppingCallback(monitor='valid_loss', comp=np.less, patience=7),
#         SaveModelCallback(monitor='valid_loss', comp=np.less, fname='best-validloss')
#     ]
# )

learn.fine_tune(15, base_lr=lr)


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.print_classification_report()


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




