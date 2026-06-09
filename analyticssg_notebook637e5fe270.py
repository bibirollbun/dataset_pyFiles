# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


print("=" * 80)
print("CASSAVA LEAF DISEASE CLASSIFICATION")
print("=" * 80)
print("\n[1/13] Importing libraries...")

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import cv2
import os
import time
import random
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Install required packages (uncomment if needed)
# !pip install timm
# !pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

# Advanced libraries
import timm  # PyTorch Image Models
from timm.utils import AverageMeter

print("✓ All libraries imported successfully")

