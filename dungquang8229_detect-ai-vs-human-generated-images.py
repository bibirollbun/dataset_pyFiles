import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import timm

from fastai.vision.all import *
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from fastai.metrics import F1Score
from sklearn.metrics import f1_score

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import shutil
from tqdm import tqdm


train_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/test.csv')

train_df.head()


data_dir = Path("/kaggle/input/ai-vs-human-generated-dataset/")


item_tfms = [
    RandomResizedCrop(384, min_scale=0.8),  # Cắt ngẫu nhiên và thay đổi kích thước
    FlipItem(p=0.5),                        # Lật ngang ảnh với xác suất 50%
    DihedralItem(p=0.3),                    # Phản chiếu ảnh với xác suất 30%
]

batch_tfms = [
    *aug_transforms(size=224,
                    min_scale=0.75,         # Cắt ngẫu nhiên 80%-100% hình ảnh
                    max_zoom=1.2,          # Zoom tối đa 1.2 lần
                    max_rotate=15,         # Xoay tối đa ±15 độ
                    max_lighting=0.2,      # Điều chỉnh độ sáng 0.2
                    max_warp=0.2,          # Biến dạng ảnh nhẹ 0.2
                    p_affine=0.75, 
                    p_lighting=0.3),
    Normalize.from_stats(*imagenet_stats)
]


model_nms = ['resnet34', 'resnet18']
bs = [32, 16]
test_dl = None

# train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
# test_df = test_df.sample(frac=0.1, random_state=42).reset_index(drop=True)


folds = 5
valid_ratio = 0.02

valid_sz = len(train_df) // folds
valid_actual_sz = int(len(train_df) * valid_ratio)

valid_splits = []

for i in range(folds):
    start = i * valid_sz
    end = i * valid_sz + valid_actual_sz

    valid_subset = train_df.iloc[start:end].index.tolist()

    valid_splits.append(valid_subset)

    print (f"Fold {i}: {start}, {end}")


def temperature_scaled_preds(preds, T=2.0):
    return F.softmax(preds / T, dim=1)


tta_preds = []

for fold in tqdm(np.arange(3), desc="Training and Inference Progress"):
    ai_human_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=lambda x: data_dir / x['file_name'],
        get_y=lambda x: str(x['label']),
        splitter=IndexSplitter(valid_splits[fold]),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    
    dls = ai_human_block.dataloaders(train_df, bs=bs[fold % len(bs)])

    if test_dl is None:
        test_dl = dls.test_dl(test_df['id'].apply(lambda x: data_dir / x))

    learn = vision_learner(
        dls, 
        model_nms[fold % len(model_nms)],
        loss_func=CrossEntropyLoss(label_smoothing=0.1), 
        metrics=(F1Score(), accuracy),
        wd=0.02,
        wd_bn_bias=True,
        ps=0.5
    )

    learn.fine_tune(4)
    # learn.freeze_to(-2)  # Unfreeze 1 vài layer cuối
    # learn.freeze()
    # learn.fit_one_cycle(3)  # Học nhanh ở layer cuối
    
    # learn.unfreeze()  # Unfreeze toàn bộ
    # learn.fit_one_cycle(2)  # Học sâu hơn, tránh overfit
    
    test_preds, _ = learn.tta(dl=test_dl)
    test_probs = temperature_scaled_preds(test_preds, T=2.0)
    test_probs = test_probs[:, 1]

    threshold = np.median(test_probs)
    print(f"Test preds median: {threshold:.4f}")
    
    threshold = np.quantile(test_probs, 0.25)
    print(f"25% quantile: {threshold:.4f}")

    tta_preds.append(test_probs)


tta_preds = np.array([pred.numpy() for pred in tta_preds])

final_probs = tta_preds.mean(axis=0)

threshold = 0.275
final_labels = (final_probs > threshold).astype(int)

print(pd.Series(final_labels).value_counts())


test_df['label'] = final_labels.tolist()
test_df[['id', 'label']].to_csv('submission.csv', index=False)


test_df.head(10)

