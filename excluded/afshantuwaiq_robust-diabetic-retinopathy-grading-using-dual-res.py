import os, cv2, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time, threading

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

from torch.cuda.amp import autocast, GradScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

# ğŸ”„ Keep Kaggle session alive
def keep_alive():
    while True:
        print("â�³ Notebook alive...", flush=True)
        time.sleep(120)

t = threading.Thread(target=keep_alive)
t.daemon = True
t.start()


