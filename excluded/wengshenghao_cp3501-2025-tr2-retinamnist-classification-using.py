# Importing Core libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import random
import glob
from os.path import expanduser, join, dirname, basename
from pathlib import Path
import matplotlib.pyplot as plt


# FastAI for computer vision
from fastai.vision.all import *
from fastai.callback.all import *


# Configuration
class cfg(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # Reproducibility
    seed = 42

    # Dataset paths
    data_root = '/kaggle/input/cp-3501-retinamnist-v-2024/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b'
    data_path = join(data_root, 'retinamnist_28')
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')
    sample_submission_path = '/kaggle/input/cp-3501-retinamnist-v-2024/sample_submission.csv'

    # Experiment image sizes
    img_sizes = [28, 64, 128, 224]
    current_img_size = 128  # Starting with 128 for balanced performance

    # Training settings
    batch_size = 64
    epochs = 5

    # Augmentations
    item_tfms = RandomResizedCrop(current_img_size, min_scale=0.5)
    batch_tfms = aug_transforms(mult=2, flip_vert=True, max_rotate=360., max_lighting=0.2, max_zoom=1.1)



# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


# FastAI DataBlock pipeline
retina_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=RandomSplitter(valid_pct=0.2, seed=cfg.seed),
    item_tfms=cfg.item_tfms,
    batch_tfms=cfg.batch_tfms
)


# Create DataLoaders with custom augmentations
dls = retina_block.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)

# Preview a batch of training images
dls.train.show_batch(max_n=6, nrows=2)


# Initialize ResNet18 as baseline
learn_r18 = vision_learner(dls, resnet18, metrics=accuracy)


# Automatically find the best learning rate range
lr_valley_r18, lr_slide_r18 = learn_r18.lr_find(suggest_funcs=(valley, slide))
print(f"Suggested learning rates for ResNet18:\n- valley: {lr_valley_r18:.2e}\n- steepest: {lr_slide_r18:.2e}")



# Train the model using the suggested valley learning rate
learn_r18.fine_tune(cfg.epochs, base_lr=lr_valley_r18)


# Evaluation
interp_r18 = ClassificationInterpretation.from_learner(learn_r18)
interp_r18.plot_confusion_matrix(figsize=(6,6))


# Initialize ResNet50 with Label Smoothing and macro F1 for balanced evaluation
learn_r50 = vision_learner(
    dls,
    resnet50,
    metrics=[accuracy, F1Score(average='macro')],
    loss_func=LabelSmoothingCrossEntropy()
)


# Automatically find the best learning rate range
try:
    lr_valley_r50, lr_slide_r50 = learn_r50.lr_find(suggest_funcs=(valley, slide))
    print(f"Suggested learning rates for ResNet50:\n- valley: {lr_valley_r50:.2e}\n- steepest: {lr_slide_r50:.2e}")
except Exception as e:
    print("lr_find failed:", e)
    learn_r50.recorder.plot_lr_find()


# Train ResNet50 with MixUp regularization and model saving callback
learn_r50.fine_tune(cfg.epochs + 1, base_lr=lr_valley_r50, cbs=[MixUp(), SaveModelCallback(monitor='valid_loss')])


# Evaluation
interp_r50 = ClassificationInterpretation.from_learner(learn_r50)
interp_r50.plot_confusion_matrix(figsize=(6,6))


# Compare validation accuracy
acc_r18 = learn_r18.validate()[1]  
acc_r50 = learn_r50.validate()[1]  

print(f"ResNet18 Accuracy: {acc_r18:.4f}")
print(f"ResNet50 Accuracy: {acc_r50:.4f}")


# BEst model
best_model = learn_r50 if acc_r50 > acc_r18 else learn_r18


# Create test dataloader
test_dl = dls.test_dl(get_image_files(cfg.test_images_dir))

# Apply Test Time Augmentation
tta_preds, _ = learn_r50.tta(dl=test_dl)
tta_labels = tta_preds.argmax(dim=1).numpy()



# Save softmax probabilities
probs = tta_preds.numpy()
submission_proba = pd.DataFrame(probs, columns=[f'class_{i}' for i in range(probs.shape[1])])
submission_proba.insert(0, 'image_id', [f.name for f in get_image_files(cfg.test_images_dir)])
submission_proba.to_csv('submission_probabilities.csv', index=False)



# Predict with both models
tta_r18, _ = learn_r18.tta(dl=test_dl)
tta_r50, _ = learn_r50.tta(dl=test_dl)

# Average softmax outputs
avg_preds = (tta_r18 + tta_r50) / 2
final_labels = avg_preds.argmax(dim=1).numpy()



# Generate final submission
image_ids = [f.name for f in get_image_files(cfg.test_images_dir)]
submission_df = pd.DataFrame({'image_id': image_ids, 'label': final_labels})

# Merge with sample format
sample_sub = pd.read_csv(cfg.sample_submission_path)
final_sub = sample_sub[['image_id']].merge(submission_df, on='image_id', how='left')

# Save
final_sub.to_csv('submission.csv', index=False)
print("✅ Ensemble submission with TTA and raw probabilities complete!")
!head submission.csv

