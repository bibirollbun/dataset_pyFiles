import numpy as np
import pandas as pd
import torch
import random
from os.path import expanduser, join, dirname, basename


from fastai.vision.all import *


class cfg(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    seed = 1
    
    data_root = '/kaggle/input/cp-3501-retinamnist-v-2024/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b'
    data_path = join(data_root, 'retinamnist_224') # 修改成最清晰的224数据集
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')
    train_img_size = 224 # 和数据集里的尺寸匹配
    batch_size = 16 # 提高精度
    
    item_tfms = RandomResizedCrop(train_img_size, min_scale=0.5) # mult=2可以看情况增加
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


data_block0 = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=cfg.seed),
    get_y=parent_label,
    item_tfms=Resize(cfg.train_img_size),
)

dls0 = data_block0.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)

dls0.train.show_batch(max_n=8, nrows=2)


dls0.valid.show_batch(max_n=4, nrows=1)


data_block1 = data_block0.new(item_tfms=RandomResizedCrop(cfg.train_img_size, min_scale=0.3))
dls1 = data_block1.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls1.train.show_batch(max_n=4, nrows=1, unique=True)


dls1.train.show_batch(max_n=4, nrows=1, unique=False)


data_block2 = data_block1.new(batch_tfms=aug_transforms(mult=2))

dls2 = data_block2.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls2.train.show_batch(max_n=4, nrows=1, unique=True)


dls2.valid.show_batch(max_n=4, nrows=1)



dls0.valid.show_batch(max_n=4, nrows=1)


data_block = data_block0.new(item_tfms=cfg.item_tfms, batch_tfms=cfg.batch_tfms)  

dls = data_block.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls.train.show_batch(max_n=4, nrows=1, unique=True)


learn = vision_learner(dls, densenet121, metrics=accuracy)

learn.lr_find(suggest_funcs=(valley, slide))


lr = 0.00019


learn.fine_tune(9, base_lr=lr)


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


import glob
test_files = glob.glob(join(cfg.test_images_dir, '*.png'))
print(test_files[:10])


for img_path in test_files[:10]:
    with learn.no_bar(), learn.no_logging():
        res = learn.predict(img_path)
    print(res)


from tqdm import tqdm

preds = []
pred_ids = []
for img_path in tqdm(test_files):
    img_id = basename(img_path)
    pred_ids += [img_id]
    with learn.no_bar(), learn.no_logging():
        res = learn.predict(img_path)
    pred_label, val, probs = res
    preds += [pred_label]
    
df = pd.DataFrame({'image_id':pred_ids})
df['label'] = preds
df


sub = pd.read_csv(cfg.sample_submission_path)
sub


sub = sub[['image_id']].copy()
sub = pd.merge(sub, df, how='left', on='image_id', validate='1:1')
sub


sub.to_csv('submission.csv', index=False)
!head submission.csv




