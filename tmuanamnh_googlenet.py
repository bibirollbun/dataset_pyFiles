# # Import necessary libraries
from fastai.vision.all import *
import torch
import random
import os
from torchvision.models import googlenet



# Define configuration using a simple class
class CFG:
    def __init__(self):
        self.seed = 1
        self.data_root = '/kaggle/input/cp-3501-retinamnist-v-202501/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b/'
        self.data_path = os.path.join(self.data_root, 'retinamnist_224')
        self.train_img_dir = os.path.join(self.data_path, 'train_images')
        self.test_img_dir = os.path.join(self.data_path, 'test_images')
        self.train_img_size = 128
        self.batch_size = 32
        
        # Augmentations
        self.item_tfms = RandomResizedCrop(224, min_scale=0.5)
        self.batch_tfms = aug_transforms(mult=2, flip_vert=True, max_rotate=360.)


        sample_submission_path = '/kaggle/input/cp-3501-retinamnist-v-2024/sample_submission.csv'

cfg = CFG()  # Instantiate the config object



# Set seed for reproducibility
def set_seed(seed=cfg.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


# Verify dataset structure
if not os.path.exists(cfg.train_img_dir):
    print("â�Œ ERROR: train_images directory not found! Verify dataset structure.")
else:

# Load data from folder structure
    dls = ImageDataLoaders.from_folder(
        path=cfg.train_img_dir, 
        valid_pct=0.2, 
        seed=cfg.seed,
        item_tfms=cfg.item_tfms,
        batch_tfms=cfg.batch_tfms,
        bs=cfg.batch_size
)
print("âœ… Data loaded successfully")

# âœ… Show a batch of training images
dls.show_batch(max_n=9, figsize=(6,6))


##Submission generation function

from datetime import datetime
import pandas as pd
from fastai.vision.all import get_image_files
import os

def generate_submission(learn, test_dir, model_name="googlenet"):
    """
    Runs inference on test images and saves a timestamped submission CSV and model file.
    """
    # Save trained model
    export_path = f"/kaggle/working/{model_name}.pkl"
    learn.export(export_path)

    # Run inference
    test_files = get_image_files(test_dir)
    preds, pred_ids = [], []

    for img_path in test_files:
        img_id = os.path.basename(img_path)
        pred_ids.append(img_id)
        with learn.no_bar(), learn.no_logging():
            pred_label, _, _ = learn.predict(img_path)
        preds.append(pred_label)

    # Save submission CSV (without index)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame({'image_id': pred_ids, 'label': preds})
    submission_path = f"/kaggle/working/submission_{model_name}_{timestamp}.csv"
    df.to_csv(submission_path, index=False)  # âœ… Index removed here

    print(f"âœ… Submission saved as: {submission_path}")
    return submission_path


# Create a baseline learner 
learn = vision_learner(
    dls,
    googlenet,
    metrics=accuracy,
    path=Path('/kaggle/working'),
)

# Evaluate model performance
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()

# Evaluate initial (untrained) model performance
loss, acc = learn.validate()
print(f"ğŸ”� Base Model Accuracy (untrained): {acc:.4f}")

generate_submission(learn, cfg.test_img_dir)


# Automatically find the best learning rate range
lr_min, lr_steep = learn.lr_find(suggest_funcs=(valley, slide))
print(f"Suggested learning rates:\n- valley: {lr_min:.2e}\n- steepest: {lr_steep:.2e}")


from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback

learn.fine_tune(
    epochs=10,                  # More training for deeper convergence
    base_lr=lr_min,            # Use valley LR for stability
    freeze_epochs=2,           # Train head first, then unfreeze
    cbs=[
        SaveModelCallback(monitor='accuracy', fname='best_model'),  # Save best checkpoint
        EarlyStoppingCallback(monitor='accuracy', patience=3)       # Stop early if no improvement
    ]
)

generate_submission(learn, cfg.test_img_dir)

