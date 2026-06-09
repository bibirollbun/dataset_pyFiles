from collections import defaultdict
import random

import pandas as pd
import numpy as np

import torch
import fastai
import torchvision
import timm

from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from torch.nn import Flatten

from fastai.vision.learner import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.metrics import ActivationType

from torchvision.models import ResNet50_Weights

from sklearn.model_selection import StratifiedKFold, train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
    
from pathlib import Path
import multiprocessing as mp
import cv2


def set_seed(use_cuda: bool, seed_value: int = 42) -> None:
    """
    Set random number generator seeds for CPU and GPU (CUDA) processing to ensure reproducibility.

    :param use_cuda: Boolean flag indicating if GPU (CUDA) processing is available.
    :param seed_value: Seed value to be used for random number generation (default: 1234).
    :return: None
    """
    np.random.seed(seed_value) # CPU Numpy variable (NumPy CPU results)
    torch.manual_seed(seed_value) # CPU PyTorch variable 
    random.seed(seed_value) # Python variable (random Python module)
    if use_cuda: # if GPU (CUDA) processing is available
        torch.cuda.manual_seed(seed_value) # GPU PyTorch variable
        torch.cuda.manual_seed_all(seed_value) # GPU PyTorch variable
        torch.backends.cudnn.deterministic = True # cuDNN (CUDA Deep Neural Network library) set as deterministic
        torch.backends.cudnn.benchmark = False
    print(f'Seed set to: {seed_value}, with use_cuda={use_cuda}')


SEED=42
NUM_EPOCHS=20
NUM_SPLITS=4
BATCH_SIZE=32
SIZE=512
DEV=True
EXTERNAL=False
OVERSAMPLING=True
UNDERSAMPLING=False

assert not (OVERSAMPLING and UNDERSAMPLING), "Either OVERSAMPLING or UNDERSAMPLING should be True, but not both."

model_weights_path = '/kaggle/working/rsna-trained-model-weights'

data_base_path = "/kaggle/input/rsna-breast-cancer-detection/"
images_base_path = "/kaggle/input/rsna-mammography-images-as-pngs/"
df_images_path = os.path.join(images_base_path, f"images_as_pngs_cv2_vl_{SIZE}/train_images_processed_cv2_vl_{SIZE}/")

df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
df["image_path"] = df_images_path + df["patient_id"].astype(str) + "/" + df["image_id"].astype(str) + ".png"

use_cuda = torch.cuda.is_available()
set_seed(use_cuda, SEED)
DEVICE = torch.device('cuda' if use_cuda else 'cpu')
print('Device available now:', DEVICE)


if DEV:
    print("DEV SPLIT")
    original_dist = df['cancer'].value_counts(normalize=True)
    original_size = len(df)
    sample_size = 0.2
    unique_patients = df['patient_id'].unique()
    external_patients, sampled_patients = train_test_split(unique_patients, test_size=sample_size, random_state=SEED, stratify=df.groupby('patient_id')['cancer'].max().values)
    df = df[df['patient_id'].isin(sampled_patients)]
    sampled_dist = df['cancer'].value_counts(normalize=True)
    sampled_size = len(df)
    print("Dev is true then the number of samples is reduced to: " + str(sample_size*100) + "%\n")
    print("Original Size:\n", original_size)
    print("Original Distribution:\n", original_dist)
    print("----------")
    print("Sampled Size:\n", sampled_size)
    print("Sampled Distribution:\n", sampled_dist)
    print()


df = df.reset_index(drop=True)


print("TRAIN TEST SPLIT")
print()
unique_patients = df['patient_id'].unique()
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=SEED, stratify=df.groupby('patient_id')['cancer'].max().values)

# Create train and test dataframes based on the patient_id split
train_df = df[df['patient_id'].isin(train_patients)]
test_df = df[df['patient_id'].isin(test_patients)]

train_patients_set = set(train_df['patient_id'])
test_patients_set = set(test_df['patient_id'])
assert train_patients_set.isdisjoint(test_patients_set), "Overlap found between train and test sets"

train_dist = train_df['cancer'].value_counts(normalize=False)
test_dist = test_df['cancer'].value_counts(normalize=False)
train_dist_normalized = train_df['cancer'].value_counts(normalize=True)
test_dist_normalized = test_df['cancer'].value_counts(normalize=True)

print("Train Distribution:\n", train_dist)
print()
print("Test Distribution:\n", test_dist)
print()
print("Train Distribution Normalized:\n", train_dist_normalized)
print()
print("Test Distribution Normalized:\n", test_dist_normalized)


train_image_id_to_cancer_label = train_df.set_index('image_id')['cancer'].to_dict()
test_image_id_to_cancer_label = test_df.set_index('image_id')['cancer'].to_dict()

def labeling_function(path, labels_dict):
    try:
        image_id = int(path.stem) 
        return labels_dict[image_id] 
    except KeyError:
        return -1


patient_labels = train_df.groupby('patient_id')['cancer'].max().reset_index()
skf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
splits = list(skf.split(patient_labels['patient_id'], patient_labels['cancer']))


def filter_training_items(items, split_index):
    """Filter out items that belong to the training set."""
    train_patient_ids = set(patient_labels['patient_id'][splits[split_index][0]].values)
    training_items = [item for item in items if int(Path(item).parent.name) in train_patient_ids]
    return training_items


import random

def get_undersampling_factor(train_df):
    class_counts = train_df['cancer'].value_counts()
    factor = class_counts[1] // class_counts[0]  # Ratio of minority to majority class
    return factor

def get_all_items_with_undersampling(image_dir_path, split_index):
    train_patient_ids = set(patient_labels['patient_id'][splits[split_index][0]].values)
    valid_patient_ids = set(patient_labels['patient_id'][splits[split_index][1]].values)

    cancer_items = []
    non_cancer_items = []
    items = []

    # Collect cancer and non-cancer items from training set
    for p in get_image_files(image_dir_path):
        patient_id = int(Path(p).parent.name)
        label = labeling_function(p, train_image_id_to_cancer_label)
        
        if patient_id in train_patient_ids:
            if label == 1:
                cancer_items.append(p)
            elif label == 0:
                non_cancer_items.append(p)

    # Handle undersampling if enabled
    if UNDERSAMPLING:
        undersampled_non_cancer_items = random.sample(non_cancer_items, len(cancer_items))
        items.extend(cancer_items + undersampled_non_cancer_items)
        random.shuffle(items)
    else:
        items.extend(cancer_items)
        items.extend(non_cancer_items)

    # Add valid items without any sampling
    for p in get_image_files(image_dir_path):
        patient_id = int(Path(p).parent.name)
        if patient_id in valid_patient_ids:
            items.append(p)

    return items


def get_oversampling_factor(train_df, desired_ratio=2):
    total_parts = desired_ratio + 1 
    majority_percentage = (desired_ratio / total_parts) * 100
    minority_percentage = (1 / total_parts) * 100
    majority_percentage, minority_percentage
    print(f"Percentage after oversampling\nMajority Percentage: {round(majority_percentage, 2)}\nMinority Percentage: {round(minority_percentage, 2)}")
    class_counts = train_df['cancer'].value_counts()
    factor = (class_counts[0] / desired_ratio) / class_counts[1]
    factor = math.floor(factor)
    return factor

if OVERSAMPLING:
    oversampling_factor = get_oversampling_factor(train_df, 2)
    print(f"Oversampling factor, based on desired_ratio: {oversampling_factor}" )

def get_all_items_with_oversampling(image_dir_path, split_index):
    train_patient_ids = set(patient_labels['patient_id'][splits[split_index][0]].values)
    valid_patient_ids = set(patient_labels['patient_id'][splits[split_index][1]].values)
    items = []
    for p in get_image_files(image_dir_path):
        patient_id = int(Path(p).parent.name)
        if patient_id in train_patient_ids or patient_id in valid_patient_ids:
            label = labeling_function(p, train_image_id_to_cancer_label)
            if OVERSAMPLING and patient_id in train_patient_ids:
                items.extend([p] * (oversampling_factor if label == 1 else 1))
            else:
                items.append(p)
    random.shuffle(items)
    return items


get_all_items = get_all_items_with_oversampling
if OVERSAMPLING:
    print("TRAIN OVERSAMPLING METHOD")
    get_all_items = get_all_items_with_oversampling
elif UNDERSAMPLING:
    print("TRAIN UNDERSAMPLING METHOD")
    get_all_items = get_all_items_with_undersampling


def check_class_distribution(items, label_dict):
    class_counts = {'cancer': 0, 'non-cancer': 0}
    for item in items:
        image_id = int(Path(item).stem)
        label = label_dict.get(image_id, -1)
        if label == 1:
            class_counts['cancer'] += 1
        elif label == 0:
            class_counts['non-cancer'] += 1
    return class_counts

def aggregate_class_distribution(image_dir_path, num_splits, label_dict):
    total_class_counts = {'non-cancer': 0, 'cancer': 0}
    for split_index in range(num_splits):
        items = get_all_items(image_dir_path, split_index)
        training_items = filter_training_items(items, split_index)
        class_counts = check_class_distribution(training_items, label_dict)
        total_class_counts['cancer'] += class_counts['cancer']
        total_class_counts['non-cancer'] += class_counts['non-cancer']
    return total_class_counts

def aggregate_and_normalize_class_distribution(image_dir_path, num_splits, label_dict):
    total_class_counts = aggregate_class_distribution(image_dir_path, num_splits, label_dict)
    total_samples = total_class_counts['cancer'] + total_class_counts['non-cancer']
    normalized_class_counts = {k: v / total_samples for k, v in total_class_counts.items()}
    return normalized_class_counts

print("TRAIN DISTRIBUTION:")
total_distribution = aggregate_class_distribution(df_images_path, NUM_SPLITS, train_image_id_to_cancer_label)
normalized_distribution = aggregate_and_normalize_class_distribution(df_images_path, NUM_SPLITS, train_image_id_to_cancer_label)
print(normalized_distribution)


def calculate_label_weights(train_df, class_counts, smoothing_factor=1.0):
    weights = (1 / class_counts) * (len(train_df) / len(class_counts))
    weights = weights * smoothing_factor
    return torch.tensor(weights.values).float()

class_counts = pd.Series({0: total_distribution['non-cancer'], 1: total_distribution['cancer']}, name='count')
# label_smoothing_weights = calculate_label_weights(train_df, class_counts)
label_smoothing_weights = torch.tensor([1,10]).float()
if torch.cuda.is_available():
  label_smoothing_weights = label_smoothing_weights.cuda()
print(label_smoothing_weights)


from fastai.vision.augment import aug_transforms, RandomResizedCrop, Brightness, Contrast, RandomErasing

def train_labeling_wrapper(path):
    return labeling_function(path, train_image_id_to_cancer_label)

def custom_splitter(items, split_index):
    train_patient_ids = set(patient_labels['patient_id'][splits[split_index][0]].values)
    valid_patient_ids = set(patient_labels['patient_id'][splits[split_index][1]].values)
    train_idxs, valid_idxs = [], []
    for idx, item in enumerate(items):
        patient_id = int(Path(item).parent.name)  
        if patient_id in train_patient_ids:
            train_idxs.append(idx)
        elif patient_id in valid_patient_ids:
            valid_idxs.append(idx)
    return train_idxs, valid_idxs


aug_tfms = aug_transforms(
    do_flip=True,
    flip_vert=False,
    max_rotate=10.0,
    min_zoom=1.0,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
    p_affine=0.75,
    p_lighting=0.75,
)


def get_train_dataloaders(image_dir_path, split_index, batch_size=BATCH_SIZE):
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=partial(get_all_items, split_index=split_index),
        splitter=partial(custom_splitter, split_index=split_index),
        get_y=train_labeling_wrapper,
        batch_tfms=aug_tfms
    )
    dls = dblock.dataloaders(df_images_path, batch_size=batch_size)
    return dls

def get_learner(split_index, arch=resnet18):
    learner = vision_learner(
        get_train_dataloaders(df_images_path, split_index),
        arch,
        custom_head=nn.Sequential(SelectAdaptivePool2d(pool_type='avg', flatten=Flatten()), nn.Linear(1280, 2)),
        metrics=[
            RocAucBinary(),
            BrierScore()
        ],
        loss_func=FocalLossFlat(weight=torch.tensor([1,10]).float().cuda(), gamma=2.0),
        pretrained=True,
        normalize=False
    ).to_fp16()
    return learner


def show_sample_images(dl, n=5):
    for i, (images, labels) in enumerate(dl):
        if i >= n: break
        dls.show_batch((images, labels), max_n=4, nrows=1)
        print("Labels:", labels)


from collections import Counter

dls = get_train_dataloaders(df_images_path, 0)

# Validation
print("\n-----VALIDATION------\n")
valid_dl = dls.valid
print(f"Validation number of batches: {len(valid_dl)}")

# Check the shape of input tensors in a five validation batches
for i, batch in enumerate(valid_dl):
    images, _ = batch
    print(f"Validation Batch {i} - Input shape:", images.shape)
    if i >= 5:  # Check first 5 batches 
        break
        
# Check class distribution by Counter
label_counts = Counter()
for _, labels in valid_dl:
    label_counts.update(labels.cpu().numpy())
print(f"Validation set class distribution: {label_counts}")

# Check labels corresponding to given bathces
for i, batch in enumerate(valid_dl):
    _, labels = batch
    print(f"Valid Batch {i} labels:", labels)
    if i >= 5:
        break

# Train
print("\n-----TRAIN------\n")
train_dl = dls.train
print(f"Train number of batches: {len(train_dl)}")

# Check the shape of input tensors in a five train batches
for i, batch in enumerate(train_dl):
    images, _ = batch
    print(f"Train Batch {i} - Input shape:", images.shape)
    if i >= 5:  # Check first 5 batches
        break
        
# Check class distribution by Counter
label_counts = Counter()
for _, labels in train_dl:
    label_counts.update(labels.cpu().numpy())
print(f"Validation set class distribution: {label_counts}")

# Check labels corresponding to given bathces
for i, batch in enumerate(train_dl):
    _, labels = batch
    print(f"Train Batch {i} labels:", labels)
    if i >= 5:
        break


#  show_sample_images(valid_dl, n=5)


#  show_sample_images(train_dl, n=5)


from fastai.callback.mixup import MixUp
import shutil
import gc

torch.cuda.empty_cache()
gc.collect()

# ACTUAL
arch="tf_efficientnetv2_s.in21k"
# arch="resnet50.a1_in1k"
# arch="mobilenetv2_100.ra_in1k"


# GOOD - TEST FURHTER
# arch="mobilenetv2_100.ra_in1k"
# arch="resnet34.a1_in1k" ~60 AUC

# OK - TO TEST FURTHER
# arch="resnext50_32x4d.fb_swsl_ig1b_ft_in1k"
# arch="regnety_002.pycls_in1k"

# DECENT - MAYBE TEST
# arch="resnext50_32x4d.fb_swsl_ig1b_ft_in1k" OK BUT LONG
# arch="convnextv2_nano.fcmae_ft_in22k_in1k"
# arch="convnextv2_atto.fcmae_ft_in1k" TOO LONG
# arch="mobilenetv3_large_100.ra_in1k"
preds, targets = [], []

arch_name = arch if isinstance(arch, str) else arch.__name__
trained_model_weights_path = Path(f"{model_weights_path}/{arch_name}")

if trained_model_weights_path.exists() and trained_model_weights_path.is_dir():
    shutil.rmtree(trained_model_weights_path)
trained_model_weights_path.mkdir(parents=True)

SPLIT=0
for SPLIT in range(NUM_SPLITS):
    torch.cuda.empty_cache()
    gc.collect()
    learn = get_learner(SPLIT, arch)
    learn.unfreeze()
    
    learn.add_cb(EarlyStoppingCallback(monitor='valid_loss', min_delta=0.05, patience=5))

    lr_valley = learn.lr_find(suggest_funcs=valley, num_it=200)
    print(f"Suggested Learning Rate for Split {SPLIT}: Valley point: {lr_valley}")
    chosen_lr = lr_valley 

    learn.fit_one_cycle(
    NUM_EPOCHS,
    chosen_lr,
    pct_start=0.3 
    )
    model_path = trained_model_weights_path/str(SPLIT)
    print(f"Saving as: {model_path}")
    learn.save(model_path)
    output = learn.get_preds()
    preds.append(output[0])  
    targets.append(output[1])
    torch.cuda.empty_cache()
    gc.collect()


test_image_id_to_cancer_label = test_df.set_index('image_id')['cancer'].to_dict()

def test_labeling_wrapper(path):
    return labeling_function(path, test_image_id_to_cancer_label)


def get_test_items(test_df, image_dir_path, _=None):
    return [Path(row["image_path"]) for _, row in test_df.iterrows()]

def get_test_dataloader(test_df, image_dir_path, batch_size=BATCH_SIZE):
    test_db = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=partial(get_test_items, test_df, image_dir_path),
        get_y=test_labeling_wrapper,
        batch_tfms=None
    )
    test_dls = test_db.dataloaders(Path(image_dir_path), batch_size=batch_size)
    return test_dls

test_dls = get_test_dataloader(test_df, df_images_path)


from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from scipy.stats import sem
import numpy as np
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def calculate_auc_confidence_interval(y_true, y_scores, alpha=0.95):
    auc_scores = []
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        auc_scores.append(score)
    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int((1.0 - alpha) / 2.0 * n_bootstraps)]
    confidence_upper = sorted_scores[int((alpha + (1.0 - alpha) / 2.0) * n_bootstraps)]
    return confidence_lower, confidence_upper



def plot_roc_auc_curve(true_labels, predictions):
    fpr, tpr, _ = roc_curve(true_labels, average_preds[:, 1])
    roc_auc = roc_auc_score(true_labels, average_preds[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

def calculate_precision_recall_gain(true_labels, predictions):
    pi = np.mean(true_labels)
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    precision_gain = (precision - pi) / ((1 - pi) * precision)
    recall_gain = (recall - pi) / ((1 - pi) * recall)
    precision_gain[precision_gain < 0] = 0
    recall_gain[recall_gain < 0] = 0
    return precision_gain, recall_gain

def plot_pr_gain_curve(true_labels, predictions):
    precision_gain, recall_gain = calculate_precision_recall_gain(true_labels, predictions)
    pr_gain_auc = auc(recall_gain, precision_gain)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_gain, precision_gain, color='blue', lw=2, label=f'PR-Gain curve (AUPRG = {pr_gain_auc:.2f})')
    plt.fill_between(recall_gain, precision_gain, alpha=0.2, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, max(precision_gain) + 0.05])
    plt.xlabel('Recall Gain')
    plt.ylabel('Precision Gain')
    plt.title('Precision-Recall-Gain Curve')
    plt.legend(loc="lower left")
    plt.show()
    return pr_gain_auc




def get_model(trained_model_dir_path, split):
    model_path = Path(f"{trained_model_dir_path}/{split}")
    print(f"Loading model {model_path}")
    return learn.load(model_path)

preds_all = []
targets_all = []

for SPLIT in range(NUM_SPLITS):
    learn = get_model(trained_model_weights_path, SPLIT)
    learn.dls = test_dls 
    preds, targets = learn.get_preds(dl=test_dls.train) 
    preds_all.append(preds)
    targets_all.append(targets)

average_preds = torch.mean(torch.stack(preds_all), dim=0)
true_labels = targets_all[0].numpy()


roc_auc = plot_roc_auc_curve(true_labels, average_preds[:, 1])
print("ROC AUC:", roc_auc)

auc_lower, auc_upper = calculate_auc_confidence_interval(true_labels, average_preds[:, 1])
print(f"AUC: {roc_auc} with 95% confidence interval [{auc_lower:.3f}, {auc_upper:.3f}]")


pr_gain_auc = plot_pr_gain_curve(true_labels, average_preds[:, 1])
print("PR-Gain AUC:", pr_gain_auc)


precision, recall, thresholds = precision_recall_curve(true_labels, preds[:, 1])
plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.title('Precision and Recall vs. Threshold')
plt.legend()
plt.show()


prob_true, prob_pred = calibration_curve(true_labels, average_preds[:, 1], n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, 's-', label='Calibration curve')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.legend(loc='lower right')
plt.title('Calibration Curve')
plt.show()


from sklearn.metrics import log_loss
logloss = log_loss(true_labels, average_preds)
print(f"Log Loss (Cross-Entropy Loss): {logloss}")


from sklearn.metrics import brier_score_loss
brier_score = brier_score_loss(true_labels, average_preds[:, 1])
print(f"Brier Score: {brier_score}")


preds_all = []
targets_all = []

SPLIT = 0 
learn = get_learner(SPLIT, "tf_efficientnetv2_s.in21k")

for SPLIT in range(NUM_SPLITS):
    learn.load(f"/kaggle/input/effnet-weights-oversampled/{SPLIT}")
    learn.dls = test_dls 
    preds, targets = learn.get_preds(dl=test_dls.train) 
    preds_all.append(preds)
    targets_all.append(targets)


df = pd.DataFrame({'Predicted Probability': preds[:, 1], 'True Label': true_labels})
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Predicted Probability', hue='True Label', common_norm=False)
plt.title('Gaussian Probability Distribution Plot for Both Classes')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.xlim(0, 1)  
plt.ylim(0, 7.1136639420207235)
plt.show()


df = pd.DataFrame({'Predicted Probability': preds[:, 1], 'True Label': true_labels})
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Predicted Probability', hue='True Label', common_norm=False)
plt.title('Gaussian Probability Distribution Plot for Both Classes')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.show()


!zip -r file.zip /kaggle/working
from IPython.display import FileLink
FileLink(r'file.zip')

