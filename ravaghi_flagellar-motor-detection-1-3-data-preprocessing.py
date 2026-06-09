from sklearn.model_selection import StratifiedGroupKFold
from tqdm.notebook import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import random
import torch
import os


class CFG:
    dataset_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
    train_image_path = os.path.join(dataset_path, "train")
    train_label_path = os.path.join(dataset_path, "train_labels.csv")

    cryoet_image_path_1 = "/kaggle/input/byu-2025-cryoet-dataset-part-1/dataset"
    cryoet_image_path_2 = "/kaggle/input/byu-2025-cryoet-dataset-part-2/dataset"
    cryoet_label_path = "/kaggle/input/byu-2025-cryoet-dataset-part-1/labels.csv"
   
    trust = 4
    seed = 42
    n_folds = 5


torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)


base_dir = "/kaggle/working/dataset"
for fold in range(CFG.n_folds):
    for dataset in ["images", "labels"]:
        for split in ["train", "val"]:
            dir_path = os.path.join(base_dir, f"fold_{fold}", dataset, split)
            os.makedirs(dir_path, exist_ok=True)


labels_df = pd.read_csv(CFG.train_label_path)
labels_df = labels_df[labels_df['Number of motors'] <= 1].reset_index(drop=True)

incorrectly_labeled = ["tomo_ca1d13", "tomo_1da097", "tomo_b33d4e", "tomo_6cf2df", "tomo_73173f", "tomo_319f79", "tomo_5764d6", "tomo_2b3cdf", "tomo_62eea8", "tomo_c84b8e", "tomo_e6f7f"]
labels_df = labels_df[~labels_df.tomo_id.isin(incorrectly_labeled)].reset_index(drop=True)


cryoet_tomo_ids = ['mba2011-02-16-23', 'mba2011-02-16-21', 'mba2011-02-16-121', 'mba2011-02-16-122', 'mba2011-02-16-123', 'mba2011-02-16-124', 'mba2011-02-16-127', 'mba2011-02-16-128', 'mba2011-02-16-129', 'mba2011-02-16-13', 'mba2011-02-16-130', 'mba2011-02-16-131', 'mba2011-02-16-132', 'mba2011-02-16-133', 'mba2011-02-16-134', 'mba2011-02-16-14', 'mba2011-02-16-140', 'mba2011-02-16-141', 'mba2011-02-16-142', 'mba2011-02-16-143', 'mba2011-02-16-144', 'mba2011-02-16-146', 'mba2011-02-16-147', 'mba2011-02-16-148', 'mba2011-02-16-149', 'mba2011-02-16-150', 'mba2011-02-16-151', 'mba2011-02-16-152', 'mba2011-02-16-154', 'mba2011-02-16-155', 'mba2011-02-16-156', 'mba2011-02-16-173', 'mba2011-02-16-175', 'mba2011-02-16-177', 'mba2011-02-16-178', 'mba2011-02-16-179', 'mba2011-02-16-18', 'mba2011-02-16-19', 'mba2011-02-16-20', 'mba2011-02-16-26', 'mba2011-02-16-27', 'mba2011-02-16-29', 'mba2011-02-16-3', 'mba2011-02-16-30', 'mba2011-02-16-31', 'mba2011-02-16-32', 'mba2011-02-16-33', 'mba2011-02-16-34', 'mba2011-02-16-4', 'mba2011-02-16-41', 'mba2011-02-16-43', 'mba2011-02-16-44', 'mba2011-02-16-45', 'mba2011-02-16-46', 'mba2011-02-16-47', 'mba2011-02-16-48', 'mba2011-02-16-50', 'mba2011-02-16-52', 'mba2011-02-16-53', 'mba2011-02-16-54', 'mba2011-02-16-55', 'mba2011-02-16-56', 'mba2011-02-16-57', 'mba2011-02-16-58', 'mba2011-02-16-6', 'mba2011-02-16-60', 'mba2011-02-16-61', 'mba2011-02-16-89', 'mba2011-02-16-9', 'mba2011-02-16-90', 'mba2011-02-16-91', 'mba2011-02-16-94', 'mba2011-02-16-95', 'mba2011-02-16-96', 'mba2011-02-16-99', 'mba2011-02-16-139', 'mba2011-02-16-104', 'mba2011-02-16-105', 'mba2011-02-16-106', 'mba2011-02-16-107', 'mba2011-02-16-176', 'mba2011-02-16-162', 'mba2011-02-16-164', 'mba2011-02-16-166', 'mba2011-02-16-167', 'mba2011-02-16-17', 'mba2011-02-16-37', 'mba2011-02-16-38', 'mba2011-02-16-62', 'mba2011-02-16-63', 'mba2011-02-16-64', 'mba2011-02-16-65', 'mba2011-02-16-66', 'mba2011-02-16-67', 'mba2011-02-16-68', 'mba2011-02-16-69', 'mba2011-02-16-7', 'mba2011-02-16-71', 'mba2011-02-16-72', 'mba2011-02-16-74', 'mba2011-02-16-75', 'mba2011-02-16-77', 'mba2011-02-16-8', 'mba2011-02-16-80', 'mba2011-02-16-81', 'mba2011-02-16-82', 'mba2011-02-16-83', 'mba2011-02-16-85', 'mba2011-02-16-86', 'mba2011-02-16-88', 'mba2011-02-16-170', 'mba2011-02-16-35', 'mba2011-02-16-51', 'mba2011-02-16-92', 'mba2011-02-16-93', 'mba2011-02-16-108', 'mba2011-02-16-110', 'mba2011-02-16-101', 'mba2011-02-16-102', 'mba2011-02-16-103', 'mba2011-02-16-136', 'mba2011-02-16-11', 'mba2011-02-16-138', 'mba2011-02-16-174', 'mba2011-02-16-25', 'mba2011-02-16-100', 'mba2011-02-16-111', 'mba2011-02-16-120', 'mba2011-02-16-157', 'mba2011-02-16-158', 'mba2011-02-16-113', 'mba2011-02-16-114', 'mba2011-02-16-115', 'mba2011-02-16-116', 'mba2011-02-16-117', 'mba2011-02-16-12', 'mba2011-02-16-159', 'mba2011-02-16-16', 'mba2011-02-16-161', 'mba2011-04-22-18', 'mba2011-04-22-20', 'mba2011-04-22-23', 'mba2010-08-26-2', 'mba2010-08-26-3', 'mba2010-08-26-4', 'mba2011-04-22-12', 'mba2011-04-22-15', 'mba2011-04-22-8', 'mba2011-04-22-9', 'mba2011-04-22-3', 'mba2011-04-22-4', 'mba2010-09-09-2', 'mba2012-03-09-1', 'mba2012-03-09-2', 'mba2012-03-09-4', 'mba2012-03-09-5', 'mba2012-03-09-6', 'mba2012-03-09-7', 'mba2012-03-09-8', 'mba2012-03-09-9', 'mba2011-12-30-19', 'mba2011-12-30-2', 'mba2011-12-30-21', 'mba2011-12-30-22', 'mba2011-12-30-1', 'mba2011-12-30-10', 'mba2011-12-30-12', 'mba2011-12-30-13', 'mba2011-12-30-14', 'mba2011-12-30-15', 'mba2011-12-30-17', 'mba2011-12-30-18', 'mba2011-12-30-9', 'mba2011-03-25-3', 'mba2011-03-25-5', 'mba2011-03-25-7', 'mba2012-09-28-22', 'mba2012-09-28-20', 'mba2012-09-28-23', 'mba2012-09-28-1', 'mba2012-09-28-14', 'mba2012-09-28-15', 'mba2012-09-28-17', 'mba2012-09-28-19', 'mba2012-09-28-24', 'mba2012-09-28-25', 'mba2012-09-28-26', 'mba2012-09-28-3', 'mba2012-09-28-4', 'mba2012-09-28-5', 'mba2012-09-28-6', 'mba2012-09-28-7', 'mba2012-09-28-8', 'mba2011-03-24-7', 'mba2011-03-24-11', 'mba2011-03-24-12', 'mba2011-03-24-13', 'mba2011-03-24-14', 'mba2011-03-02-12', 'mba2011-03-02-8', 'mba2012-01-13-17', 'mba2012-01-13-18', 'mba2012-01-13-19', 'mba2012-01-13-2', 'mba2012-01-13-20', 'mba2012-01-13-21', 'mba2012-01-13-22', 'mba2012-01-13-23', 'mba2012-01-13-1', 'mba2012-01-13-10', 'mba2012-01-13-14', 'mba2012-01-13-24', 'mba2012-01-13-25', 'mba2012-01-13-26', 'mba2012-01-13-27', 'mba2012-01-13-28', 'mba2012-01-13-30', 'mba2012-01-13-31', 'mba2012-01-13-32', 'mba2012-01-13-5', 'mba2012-01-13-6', 'mba2012-01-13-7', 'mba2012-01-13-8', 'mba2012-01-13-9', 'mba2011-12-17-20', 'mba2011-12-17-22', 'mba2011-12-17-28', 'mba2011-12-17-29', 'mba2012-01-12-20', 'mba2012-01-12-21', 'mba2012-01-12-22', 'mba2012-01-12-23', 'mba2012-01-12-1', 'mba2012-01-12-24', 'mba2012-01-12-26', 'mba2012-01-12-10', 'mba2012-01-12-11', 'mba2012-01-12-12', 'mba2012-01-12-14', 'mba2012-01-12-15', 'mba2012-01-12-18', 'mba2012-01-12-32', 'mba2012-01-12-33', 'mba2012-01-12-34', 'mba2012-01-12-4', 'mba2012-01-12-7', 'mba2012-01-12-9', 'mba2012-01-12-28', 'mba2012-01-12-29', 'mba2011-11-23-19', 'mba2011-11-23-20', 'mba2011-11-23-22', 'mba2011-11-23-24', 'mba2011-11-23-25', 'mba2011-11-23-26', 'mba2011-11-23-27', 'mba2011-11-23-9', 'mba2011-11-23-1', 'mba2011-11-23-11', 'mba2011-11-23-12', 'mba2011-11-23-18', 'mba2011-11-23-32', 'mba2011-11-23-33', 'mba2011-11-23-36', 'mba2011-11-23-5', 'mba2011-11-23-6', 'mba2011-11-23-29', 'mba2011-11-23-3', 'mba2011-11-23-30', 'mba2012-04-20-1', 'mba2012-04-20-2', 'mba2012-04-20-3', 'mba2012-04-20-5', 'mba2012-04-20-6', 'mba2012-04-20-7', 'mba2012-04-20-8', 'mba2012-04-20-9', 'mba2010-08-29-2', 'mba2010-08-29-3', 'mba2011-07-31-1', 'mba2012-01-02-10', 'mba2012-01-02-1', 'mba2012-01-02-5', 'mba2012-01-02-6', 'mba2012-01-02-8', 'mba2012-01-02-9', 'mba2012-01-02-11', 'mba2012-01-02-13', 'mba2012-01-02-15', 'mba2012-01-02-3', 'mba2011-07-18-2', 'mba2011-07-18-20', 'mba2011-07-18-21', 'mba2011-07-18-5', 'mba2011-07-18-10', 'mba2011-07-18-11', 'mba2011-08-10-16', 'mba2011-08-10-19', 'mba2011-08-10-2', 'mba2011-08-10-21', 'mba2011-08-10-22', 'mba2011-08-10-23', 'mba2011-08-10-24', 'mba2011-08-10-25', 'mba2011-08-10-26', 'mba2011-08-10-27', 'mba2011-08-10-28', 'mba2011-08-10-42', 'mba2011-08-10-5', 'mba2011-08-10-7', 'mba2011-08-10-9', 'mba2011-08-10-1', 'mba2011-08-10-11', 'mba2011-08-10-14', 'mba2011-08-10-35', 'mba2011-08-10-36', 'mba2011-08-10-37', 'mba2011-08-10-38', 'mba2011-08-10-39', 'mba2011-08-10-4', 'mba2011-08-10-41', 'mba2011-08-10-30', 'mba2011-08-10-31', 'mba2011-12-18-3', 'mba2011-12-18-5', 'mba2012-08-29-13', 'mba2012-08-29-15', 'mba2012-08-29-17', 'mba2012-08-29-18', 'mba2012-08-29-2', 'mba2012-08-29-20', 'mba2012-08-29-21', 'mba2012-08-29-4', 'mba2012-08-29-22', 'mba2012-08-29-23', 'mba2012-08-29-3', 'mba2012-08-29-1', 'mba2012-08-29-7', 'mba2012-08-29-8', 'mba2012-08-29-9', 'mba2012-03-09-20', 'mba2012-03-09-21', 'mba2012-03-09-24', 'mba2012-03-09-25', 'mba2012-03-20-1', 'mba2012-03-20-3', 'mba2012-03-20-7', 'mba2012-03-09-13', 'mba2012-03-09-18', 'mba2012-03-09-32', 'mba2012-03-09-34', 'mba2012-03-09-37', 'mba2012-03-09-38', 'mba2012-03-09-48', 'mba2012-03-09-40', 'mba2012-03-09-41', 'mba2012-03-09-42', 'mba2012-03-09-53', 'mba2012-03-09-54', 'mba2012-03-09-50', 'mba2012-03-09-51', 'mba2012-03-09-52', 'mba2012-04-24-20', 'mba2012-04-24-21', 'mba2012-04-24-22', 'mba2012-04-24-3', 'mba2012-04-24-4', 'mba2012-04-24-5', 'mba2012-04-24-6', 'mba2012-04-20-10', 'mba2012-04-20-11', 'mba2012-04-21-3', 'mba2012-04-21-4', 'mba2012-04-22-4', 'mba2012-04-22-6', 'mba2012-04-22-8', 'mba2012-04-23-1', 'mba2012-04-24-12', 'mba2012-04-24-9', 'mba2012-04-23-14', 'mba2012-04-23-15', 'mba2012-04-23-16', 'mba2012-04-23-4', 'mba2012-04-23-6', 'mba2012-04-23-7', 'mba2012-04-24-1', 'mba2012-04-24-11', 'mba2012-04-24-14', 'mba2012-04-24-16', 'mba2012-04-24-17', 'mba2012-04-24-18', 'mba2012-04-24-19', 'mba2012-04-24-2', 'mba2012-04-24-7', 'mba2012-04-24-8', 'mba2011-08-01-2', 'mba2011-08-01-21', 'mba2011-08-01-6', 'mba2012-02-02-10', 'mba2012-02-02-11', 'mba2012-02-02-12', 'mba2012-02-02-13', 'mba2012-02-02-15', 'mba2012-02-02-16', 'mba2012-02-02-17', 'mba2012-02-02-18', 'mba2012-02-02-20', 'mba2012-02-02-4', 'mba2012-02-02-7', 'mba2010-08-30-4', 'mba2010-08-30-5', 'mba2010-08-30-6', 'mba2010-08-26-19', 'mba2010-08-26-20', 'mba2010-08-26-11', 'mba2010-08-26-12', 'mba2010-08-26-13', 'mba2010-08-26-14', 'mba2010-08-26-15', 'mba2010-08-26-16', 'mba2010-08-26-18', 'mba2010-08-26-6', 'mba2010-08-26-7', 'mba2010-08-26-8', 'mba2010-08-26-9', 'mba2010-08-30-3', 'mba2010-08-30-9', 'mba2011-08-26-12', 'mba2011-08-26-14', 'mba2011-08-26-18', 'mba2011-08-26-19', 'mba2011-08-26-9', 'mba2011-04-12-29', 'mba2011-04-12-20', 'mba2011-04-12-22', 'mba2011-04-12-23', 'mba2011-04-12-25', 'mba2011-04-12-27', 'mba2011-04-12-28', 'mba2011-04-12-33', 'mba2011-04-12-35', 'mba2012-09-19-1', 'mba2012-09-19-10', 'mba2012-09-19-13', 'mba2012-09-19-15', 'mba2012-09-19-18', 'mba2012-09-19-21', 'mba2012-09-19-22', 'mba2012-09-19-23', 'mba2012-09-19-3', 'mba2012-09-19-4', 'mba2012-09-26-1', 'mba2012-09-26-2', 'mba2012-09-26-4', 'mba2012-09-26-6', 'mba2012-09-26-7', 'mba2012-09-26-9', 'mba2011-07-16-2', 'mba2011-07-16-3', 'mba2011-07-15-1', 'mba2011-07-15-10', 'mba2011-07-15-11', 'mba2011-07-15-12', 'mba2011-07-15-13', 'mba2011-07-15-2', 'mba2011-07-15-5', 'mba2011-07-15-6', 'mba2011-07-15-9', 'mba2010-08-27-9', 'mba2010-08-27-13', 'ycw2013-08-20-43', 'ycw2012-11-14-17', 'ycw2012-11-14-19', 'ycw2012-11-14-20', 'ycw2012-11-14-21', 'ycw2012-11-14-23', 'ycw2012-11-14-27', 'ycw2012-11-14-30', 'ycw2012-11-14-31', 'ycw2012-11-14-33', 'ycw2012-11-14-52', 'ycw2013-01-03-11', 'ycw2012-09-23-53', 'aba2014-02-21-17', 'aba2014-02-21-11', 'aba2014-02-21-13', 'aba2014-02-21-14', 'aba2014-02-21-15', 'aba2014-02-21-2', 'aba2014-02-21-6', 'aba2014-02-21-8', 'aba2014-10-15-22', 'aba2014-10-15-1', 'aba2014-10-15-8', 'aba2014-10-15-9', 'aba2014-10-29-10', 'aba2014-10-29-23', 'aba2013-04-06-16', 'aba2013-04-06-18', 'aba2013-04-06-19', 'aba2013-04-06-14', 'aba2015-02-23-59', 'aba2015-05-29-5', 'aba2015-07-15-6', 'aba2013-12-24-1', 'aba2014-03-04-13', 'aba2014-03-04-18', 'aba2014-03-04-22', 'aba2014-03-04-27', 'aba2014-03-04-29', 'aba2014-03-04-6', 'aba2014-03-05-5', 'aba2014-03-05-11', 'aba2014-04-10-23', 'aba2014-04-10-24', 'aba2014-04-10-28', 'aba2014-04-10-3', 'aba2015-06-03-24', 'aba2015-01-16-18', 'aba2015-01-16-19', 'aba2014-04-03-39', 'aba2014-04-03-2', 'aba2015-07-15-61', 'aba2015-07-15-56', 'aba2015-07-15-57', 'aba2015-07-15-71']


cryoet_labels_df = pd.read_csv(CFG.cryoet_label_path)
cryoet_labels_df = cryoet_labels_df[cryoet_labels_df["Number of motors"] <= 1]
cryoet_labels_df = cryoet_labels_df[cryoet_labels_df.tomo_id.isin(cryoet_tomo_ids)]

labels_df["source"] = "competition"
cryoet_labels_df["source"] = "cryoet"

labels_df = pd.concat([labels_df, cryoet_labels_df]).reset_index(drop=True)


labels_df["fold"] = -1
split = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed).split(labels_df, labels_df["Number of motors"], groups=labels_df["tomo_id"])
for fold_idx, (train_idx, val_idx) in enumerate(split):
    labels_df.loc[val_idx, "fold"] = fold_idx


def calculate_box_size(voxel_spacing, array_shape_y, array_shape_x):
    # Base size for the flagellar motor (typically 25-30nm in diameter)
    # Converting from angstroms to nm (1nm = 10 angstroms)
    base_motor_size_nm = 27.5  # average motor diameter in nm
    base_motor_size_angstroms = base_motor_size_nm * 10  # convert to angstroms
    
    # Convert physical size to pixels using voxel spacing
    base_box_size_pixels = base_motor_size_angstroms / voxel_spacing
    
    # Add context padding proportional to the tomogram dimensions
    # Using the geometric mean of dimensions to account for varying aspect ratios
    avg_dimension = (array_shape_y * array_shape_x) ** 0.5
    context_factor = 0.08  # 8% of the average dimension for context
    context_padding = avg_dimension * context_factor
    
    # Calculate final box size with minimum and maximum constraints
    box_size = base_box_size_pixels + context_padding
    
    # Ensure minimum box size is at least 16 pixels
    min_box_size = 16
    # Ensure box isn't too large (more than 20% of the smaller dimension)
    max_box_size = min(array_shape_y, array_shape_x) * 0.2
    
    # Apply constraints
    box_size = max(min_box_size, min(box_size, max_box_size))
    
    # Round to nearest even integer for better compatibility with YOLO
    box_size = round(box_size / 2) * 2
    
    return box_size


def prepare_yolo_dataset(fold_idx):    
    train_tomos = labels_df[labels_df['fold'] != fold_idx]['tomo_id'].unique()
    val_tomos = labels_df[labels_df['fold'] == fold_idx]['tomo_id'].unique()
    
    yolo_train_image_dir = os.path.join(base_dir, f"fold_{fold_idx}", "images", "train")
    yolo_train_label_dir = os.path.join(base_dir, f"fold_{fold_idx}", "labels", "train")
    yolo_val_image_dir = os.path.join(base_dir, f"fold_{fold_idx}", "images", "val")
    yolo_val_label_dir = os.path.join(base_dir, f"fold_{fold_idx}", "labels", "val")
    
    def normalize_slice(slice_data):
        p2 = np.percentile(slice_data, 2)
        p98 = np.percentile(slice_data, 98)
        clipped_data = np.clip(slice_data, p2, p98)
        normalized = 255 * (clipped_data - p2) / (p98 - p2)
        return np.uint8(normalized)
    
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     motor['Voxel spacing'],
                     int(motor['Array shape (axis 0)']),
                     int(motor['Array shape (axis 1)']),
                     int(motor['Array shape (axis 2)']),
                     motor["source"]
                    )
                )
        
        processed_slices = 0
        for tomo_id, z_center, y_center, x_center, v_space, z_max, y_max, x_max, source in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            if z_center == -1:
                random_slices = [random.randint(0, z_max-1) for _ in range(10)]
                for z in random_slices:
                    slice_filename = f"slice_{z:04d}.jpg"
                    
                    if source == "competition":
                        src_path = os.path.join(CFG.train_image_path, tomo_id, slice_filename)
                        img = Image.open(src_path)
                    else:
                        try:
                            src_path = os.path.join(CFG.cryoet_image_path_1, tomo_id, slice_filename)
                            img = Image.open(src_path)
                        except:
                            src_path = os.path.join(CFG.cryoet_image_path_2, tomo_id, slice_filename)
                            img = Image.open(src_path)
                    
                    img_array = np.array(img)
                    normalized_img = normalize_slice(img_array)
                    dest_filename = f"{tomo_id}_z{z:04d}_y0000_x0000.jpg"
                    dest_path = os.path.join(images_dir, dest_filename)
                    Image.fromarray(normalized_img).save(dest_path)
                    
                    img_width, img_height = img.size
                    
                    label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                    with open(label_path, 'w') as f:
                        f.write("")
                        
                    processed_slices += 1
                    
            else:                
                z_min = max(0, z_center - CFG.trust)
                z_max = min(z_max - 1, z_center + CFG.trust)
                
                for z in range(z_min, z_max + 1):
                    slice_filename = f"slice_{z:04d}.jpg"
                    if source == "competition":
                        src_path = os.path.join(CFG.train_image_path, tomo_id, slice_filename)
                        img = Image.open(src_path)
                    else:
                        try:
                            src_path = os.path.join(CFG.cryoet_image_path_1, tomo_id, slice_filename)
                            img = Image.open(src_path)
                        except:
                            src_path = os.path.join(CFG.cryoet_image_path_2, tomo_id, slice_filename)
                            img = Image.open(src_path)
                        
                    img_array = np.array(img)
                    normalized_img = normalize_slice(img_array)
                    dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                    dest_path = os.path.join(images_dir, dest_filename)
                    Image.fromarray(normalized_img).save(dest_path)

                    box_size = calculate_box_size(v_space, y_max, x_max)
                    
                    img_width, img_height = img.size
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    box_width_norm = box_size / img_width
                    box_height_norm = box_size / img_height
                    
                    label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                    with open(label_path, 'w') as f:
                        f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
                    processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    train_slices, train_motors = process_tomogram_set(train_tomos, yolo_train_image_dir, yolo_train_label_dir, "training")
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_val_image_dir, yolo_val_label_dir, "validation")
    
    return {
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices
    }


for fold_idx in range(1):
    print(f"Processing data for fold {fold_idx}")

    summary = prepare_yolo_dataset(fold_idx)
    print(f"--- Training data:    {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
    print(f"--- Validation data:  {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
    print(f"--- Total data:       {summary['train_tomograms'] + summary['val_tomograms']} tomograms, {summary['train_motors'] + summary['val_motors']} motors, {summary['train_slices'] + summary['val_slices']} slices\n")

