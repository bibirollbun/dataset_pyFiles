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
batch_tfms = [*aug_transforms(mult=2.0, max_rotate=20.0, flip_vert=True, max_zoom=1.2, max_lighting=0.2, p_lighting=0.75),
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

# =========================
# ðŸ§  3. Build Learner with Ranger Optimizer
# =========================

# Build learner with pretrained ResNet50
# learn_resnet50 = vision_learner(dls, resnet50, metrics=accuracy, pretrained=True)
# learn_resnet50 = vision_learner(dls, resnet50, metrics=accuracy, pretrained=True, opt_func=ranger)
# ðŸŽ¯ Hyperparameter optimization loop
optimizers_to_try = [Adam, ranger]
lr_choices = [1e-3, slice(1e-4, 1e-3)]
results = []

for opt_func in optimizers_to_try:
    for lr in lr_choices:
        learn = vision_learner(dls, resnet50, metrics=accuracy, opt_func=opt_func, pretrained=True)
        learn.freeze()
        learn.fit_one_cycle(3, lr_max=1e-3)
        learn.unfreeze()
        learn.fit_one_cycle(5, lr_max=lr)
        acc = learn.validate()[1]
        results.append((opt_func.__name__, str(lr), round(acc, 4)))
        print(f"âœ… Optimizer: {opt_func.__name__}, LR: {lr}, Accuracy: {acc:.4f}")

# ðŸ“Š Show results
results_df = pd.DataFrame(results, columns=["Optimizer", "LR", "Accuracy"])
results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)


# Reinitialize learner with the best hyperparameters from tuning
learn_efficientnet = vision_learner(
    dls, efficientnet_b0, metrics=accuracy, opt_func=Adam, pretrained=True
)


# Find learning rate
# =========================
# 4. Learning Rate Finder + Discriminative LR
learn_efficientnet_b0 = vision_learner(
    dls, efficientnet_b0, metrics=accuracy, opt_func=Adam, pretrained=True
)

# =========================
lr_min, lr_steep = learn_efficientnet_b0.lr_find(suggest_funcs=(valley, steep))
print(f"Suggested learning rates -> valley: {lr_min}, steep: {lr_steep}")
chosen_lr_efficientnet_b0 = slice(1e-4, 1e-3)



# =========================
# 6. Model Interpretation
# =========================

# Generate interpretation from the learner
interp_efficientnet_b0 = ClassificationInterpretation.from_learner(learn_efficientnet_b0)

# Plot the confusion matrix
interp_efficientnet_b0.plot_confusion_matrix(figsize=(8, 6))

# Optionally: Show top losses (images that the model struggled with)
interp_efficientnet_b0.plot_top_losses(9, nrows=3, figsize=(12, 10))



# Predictions on the test set
import glob
from tqdm import tqdm

test_files = glob.glob(join(cfg.test_images_dir, '*.png'))
preds, pred_ids = [], []

for img_path in tqdm(test_files):
    img_id = basename(img_path)
    pred_ids.append(img_id)
    with learn_efficientnet_b0.no_bar(), learn_efficientnet_b0.no_logging():
        res = learn_efficientnet_b0.predict(img_path)
    pred_label, _, _ = res
    preds.append(pred_label)



# =========================
# 7. Make predictions on test set
# =========================

test_files = get_image_files(cfg.test_images_dir)
test_dl = learn_efficientnet_b0.dls.test_dl(test_files)

preds_raw, _ = learn_efficientnet_b0.get_preds(dl=test_dl)
preds = preds_raw.argmax(dim=1)  # convert to class labels

# Extract filenames for submission
pred_ids = [f.name for f in test_files]



# =========================
# 8. Create Submission File
# =========================
import os
print(os.listdir('/kaggle/working'))

submission_df = pd.read_csv(cfg.sample_submission_path)
submission_df = submission_df[['image_id']].copy()
submission_df = pd.merge(
    submission_df,
    pd.DataFrame({'image_id': pred_ids, 'label': preds}),
    on='image_id'
)
submission_df.to_csv('/kaggle/working/submission_efficientnet_b0_optimized.csv', index=False)
print(submission_df.head())



# Display first few rows of submission
print(submission_df.head())

