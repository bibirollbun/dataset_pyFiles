# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from fastai.vision.all import *
import numpy as np
import pandas as pd
import torch
import random
from os.path import join, basename

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Configuration
class cfg(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    seed = 1

    # Choose image size (28, 64, 128, 224)
    image_size = 128  # You can change this size

    # Path configuration
    data_root = '/kaggle/input/cp-3501-retinamnist-v-2024/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b'
    data_path = join(data_root, f'retinamnist_{image_size}')
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')

    # Batch size
    batch_size = 32

    # Data augmentation
    item_tfms = RandomResizedCrop(image_size, min_scale=0.3) # You can adjust min_scale
    batch_tfms = aug_transforms(mult=2, flip_vert=True, max_rotate=360) # Adjust as necessary

    # Submission file path
    sample_submission_path = '/kaggle/input/cp-3501-retinamnist-v-2024/sample_submission.csv'



# Seed setting function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


# let's follow lesson-2: https://www.kaggle.com/code/dmitrykonovalov/fastai-lesson2-part1b-02-production-v2024a
# bears = DataBlock(
#     blocks=(ImageBlock, CategoryBlock), 
#     get_items=get_image_files, 
#     splitter=RandomSplitter(valid_pct=0.2, seed=42),
#     get_y=parent_label,
#     item_tfms=Resize(128))


# Data loading and DataBlock
data_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=cfg.seed),
    get_y=parent_label,
    item_tfms=cfg.item_tfms,
    batch_tfms=cfg.batch_tfms
)

# Create DataLoaders
dls = data_block.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)

# --- EfficientNet-B0 Model ---
learn_efficientnet = vision_learner(dls, efficientnet_b0, metrics=accuracy).to_fp16()




# Find optimal learning rate
lr_suggestion_effnet = learn_efficientnet.lr_find(suggest_funcs=(valley, slide))
print("Suggested learning rates for EfficientNet-B0:", lr_suggestion_effnet)


# Choose your learning rate and epochs based on lr_find result
chosen_lr_effnet = 0.000575  # adjust according to lr_find results
num_epochs_effnet = 3        # adjust epochs as desired

# Fine-tuning EfficientNet-B0
learn_efficientnet.fine_tune(num_epochs_effnet, base_lr=chosen_lr_effnet)

# Evaluate and visualize results for EfficientNet-B0 
interp_effnet = ClassificationInterpretation.from_learner(learn_efficientnet)
interp_effnet.plot_confusion_matrix()


# Predictions on the test set
import glob
from tqdm import tqdm

test_files = glob.glob(join(cfg.test_images_dir, '*.png'))
preds, pred_ids = [], []

for img_path in tqdm(test_files):
    img_id = basename(img_path)
    pred_ids.append(img_id)
    with learn_efficientnet.no_bar(), learn_efficientnet.no_logging():
        res = learn_efficientnet.predict(img_path)
    pred_label, _, _ = res
    preds.append(pred_label)



# Create submission
submission_df = pd.read_csv(cfg.sample_submission_path)
submission_df = submission_df[['image_id']].copy()
submission_df = pd.merge(submission_df, pd.DataFrame({'image_id': pred_ids, 'label': preds}), on='image_id', how='left')
submission_df.to_csv('submission_efficientnet.csv', index=False)


# Display first few rows of submission
print(submission_df.head())

