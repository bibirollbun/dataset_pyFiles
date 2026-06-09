# Import libraries
from fastai.vision.all import *
import numpy as np
import pandas as pd
import torch
import random
import glob
from os.path import join, basename
from tqdm import tqdm


# Configuration class
class cfg:
    seed = 1
    image_size = 224
    batch_size = 32
    data_root = '/kaggle/input/cp-3501-retinamnist-v-2024/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b'
    data_path = join(data_root, f'retinamnist_{image_size}')
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')
    sample_submission_path = '/kaggle/input/cp-3501-retinamnist-v-2024/sample_submission.csv'


# Seed setup
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


# Transformations
item_tfms = Resize(cfg.image_size)
batch_tfms = [*aug_transforms(mult=2.0, max_rotate=20.0, flip_vert=True),
              Normalize.from_stats(*imagenet_stats)]  # Normalize to ImageNet

# DataBlock
data_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=cfg.seed),
    get_y=parent_label,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms
)

# DataLoaders
dls = data_block.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)

# Build learner with pretrained ResNet50
learn_resnet50 = vision_learner(dls, resnet50, metrics=accuracy, pretrained=True)



# Find learning rate
lr_min, lr_steep = learn_resnet50.lr_find(suggest_funcs=(valley, steep))
print(f"Suggested learning rates -> valley: {lr_min}, steep: {lr_steep}")
chosen_lr_resnet50 = lr_min



# Fine-tune the model
learn_resnet50.fine_tune(10, base_lr=chosen_lr_resnet50, freeze_epochs=1)

# Interpretation
interp_resnet50 = ClassificationInterpretation.from_learner(learn_resnet50)
interp_resnet50.plot_confusion_matrix(figsize=(8, 6))


# Predictions on the test set
import glob
from tqdm import tqdm

test_files = glob.glob(join(cfg.test_images_dir, '*.png'))
preds, pred_ids = [], []

for img_path in tqdm(test_files):
    img_id = basename(img_path)
    pred_ids.append(img_id)
    with learn_resnet50.no_bar(), learn_resnet50.no_logging():
        res = learn_resnet50.predict(img_path)
    pred_label, _, _ = res
    preds.append(pred_label)



# Create submission DataFrame
submission_df = pd.read_csv(cfg.sample_submission_path)
submission_df = submission_df[['image_id']].copy()
submission_df = pd.merge(submission_df, pd.DataFrame({'image_id': pred_ids, 'label': preds}), on='image_id', how='left')
submission_df.to_csv('submission_resnet50_optimized.csv', index=False)


# Display first few rows of submission
print(submission_df.head())

