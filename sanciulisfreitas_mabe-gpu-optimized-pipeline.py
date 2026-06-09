# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import gc
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Progress tracking
from tqdm.auto import tqdm
tqdm.pandas()

# Environment check
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')


# GPU-optimized setup
def setup_environment():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    return device, SEED

device, SEED = setup_environment()

# Configuration
CONFIG = {
    'BATCH_SIZE': 32 if torch.cuda.is_available() else 8,
    'NUM_WORKERS': 4 if torch.cuda.is_available() else 2,
    'SEQUENCE_LENGTH': 20,
    'IMAGE_SIZE': (224, 224),
    'NUM_CLASSES': 38,
    'EPOCHS': 25,
    'LEARNING_RATE': 2e-4,
    'WEIGHT_DECAY': 1e-4,
    'DROPOUT': 0.3
}

print('GPU-optimized configuration loaded!')
print(CONFIG)

