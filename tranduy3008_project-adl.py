!pip install -q efficientnet_pytorch > /dev/null
!pip install pyuploadcare
from pyuploadcare import Uploadcare
import os
from glob import glob
import time
from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# import warnings
# warnings.filterwarnings("ignore", category=ResourceWarning)
import shutil
import os
from catalyst.data.sampler import BalanceClassSampler


def upload_to_uploadcare(files, kaggle_output_dir="/kaggle/working", public_key="1b392feaa5532feaf3b7", secret_key="670a01f0a11dd7bbfb6f"):
    """
    Upload files tá»« Kaggle lÃªn Uploadcare thÃ´ng qua API
    
    Args:
        files: List cÃ¡c tÃªn file cáº§n upload hoáº·c pattern (vÃ­ dá»¥: ["model.pth", "*.json"])
        kaggle_output_dir: ThÆ° má»¥c chá»©a file output trÃªn Kaggle
        public_key: Public key cá»§a Uploadcare
        secret_key: Secret key cá»§a Uploadcare
        
    Returns:
        list: Danh sÃ¡ch thÃ´ng tin cÃ¡c file Ä‘Ã£ upload (URL, file ID)
    """
    try:
        uploadcare = Uploadcare(public_key=public_key, secret_key=secret_key)
        print(f"[INFO] Ä�Ã£ khá»Ÿi táº¡o client Uploadcare vá»›i public key: {public_key}")
    except Exception as e:
        print(f"[ERROR] KhÃ´ng thá»ƒ khá»Ÿi táº¡o client Uploadcare: {str(e)}")
        return []
    # TÃ¬m táº¥t cáº£ files cáº§n upload
    all_files = []
    
    if isinstance(files, str):
        files = [files]
    
    for file_pattern in files:
        if '*' in file_pattern:
            # Náº¿u lÃ  pattern, tÃ¬m táº¥t cáº£ file phÃ¹ há»£p
            matched_files = glob(os.path.join(kaggle_output_dir, file_pattern))
            all_files.extend(matched_files)
        else:
            # Náº¿u lÃ  tÃªn file cá»¥ thá»ƒ
            file_path = os.path.join(kaggle_output_dir, file_pattern)
            if os.path.exists(file_path):
                all_files.append(file_path)
            else:
                print(f"[WARNING] File khÃ´ng tá»“n táº¡i: {file_path}")
    
    if not all_files:
        print("[ERROR] KhÃ´ng tÃ¬m tháº¥y file nÃ o Ä‘á»ƒ upload!")
        return []
    
    # Upload tá»«ng file
    uploaded_files = []
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024 # MB
        
        print(f"[UPLOAD] Ä�ang upload {file_name} ({file_size:.2f} KB)...")
        start_time = time.time()
        
        try:
            with open(file_path, "rb") as file_object:
                ucare_file = uploadcare.upload(file_object)
            
            upload_time = time.time() - start_time
            upload_speed = file_size / upload_time if upload_time > 0 else 0
            
            file_info = {
                'name': file_name,
                'size_mb': file_size,
                'uuid': ucare_file.uuid,
                'cdn_url': f"https://ucarecdn.com/{ucare_file.uuid}/",
                'original_path': file_path,
                'upload_time_sec': upload_time
            }
            
            uploaded_files.append(file_info)
            
            print(f"[SUCCESS] Ä�Ã£ upload {file_name} ({file_size:.2f} KB) trong {upload_time:.2f}s ({upload_speed:.2f} KB/s)")
            print(f"[SUCCESS] URL: {file_info['cdn_url']}")
            
        except Exception as e:
            print(f"[ERROR] KhÃ´ng thá»ƒ upload {file_name}: {str(e)}")
    
    #return uploaded_files


SEED = 45

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

DATA_ROOT_PATH = '../input/alaska2-image-steganalysis'


def prepare_dataset_with_groupkfold(n_splits=5):
    """Chuáº©n bá»‹ dataset vá»›i GroupKFold, phÃ¢n chia theo image_name
    
    LÆ°u Ã½: CÃ¡ch phÃ¢n chia nÃ y Ä‘áº£m báº£o ráº±ng cÃ¡c biáº¿n thá»ƒ khÃ¡c nhau cá»§a cÃ¹ng má»™t áº£nh
    (vÃ­ dá»¥: Cover, JMiPOD, JUNIWARD, UERD cá»§a cÃ¹ng má»™t áº£nh gá»‘c) sáº½ Ä‘Æ°á»£c Ä‘áº·t trong
    cÃ¹ng má»™t fold Ä‘á»ƒ trÃ¡nh data leakage.
    """
    dataset = []

    # Táº¡o dataframe chá»©a thÃ´ng tin vá»� táº¥t cáº£ áº£nh
    for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
        for path in glob(f'{DATA_ROOT_PATH}/{kind}/*.jpg'):
            # Láº¥y tÃªn áº£nh tá»« Ä‘Æ°á»�ng dáº«n
            image_name = os.path.basename(path)
            dataset.append({
                'kind': kind,
                'image_name': image_name,
                'label': label
            })

    # XÃ¡o trá»™n dá»¯ liá»‡u
    random.shuffle(dataset)
    dataset = pd.DataFrame(dataset)
    
    # Khá»Ÿi táº¡o GroupKFold
    gkf = GroupKFold(n_splits=n_splits)

    # Ã�p dá»¥ng GroupKFold, nhÃ³m theo image_name
    dataset.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
        dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number
    
    # Hiá»ƒn thá»‹ phÃ¢n phá»‘i lá»›p theo fold
    plt.figure(figsize=(12, 6))
    for fold in range(n_splits):
        plt.subplot(1, n_splits, fold+1)
        fold_data = dataset[dataset['fold'] == fold]
        sns.countplot(x='label', data=fold_data, palette='viridis')
        plt.title(f'Fold {fold}')
        plt.xlabel('Lá»›p')
        if fold == 0:
            plt.ylabel('Sá»‘ lÆ°á»£ng áº£nh')
        else:
            plt.ylabel('')
        plt.xticks(range(4), ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD'], rotation=45)
    
    plt.tight_layout()
    plt.suptitle('PhÃ¢n phá»‘i lá»›p trong tá»«ng fold', fontsize=16, y=1.05)
    plt.show()
    
    return dataset

# Táº¡o dataset vá»›i GroupKFold
dataset = prepare_dataset_with_groupkfold()

# Hiá»ƒn thá»‹ thÃ´ng tin dataframe
print("ThÃ´ng tin dataframe:")
print(dataset.head())
print("\nPhÃ¢n phá»‘i fold:")
print(dataset['fold'].value_counts())
print("\nPhÃ¢n phá»‘i lá»›p:")
print(dataset['label'].value_counts())


def get_train_transforms():
    """Augmentations Ä‘Æ¡n giáº£n cho táº­p huáº¥n luyá»‡n tá»« Notebook-3"""
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    """Transforms cho validation tá»« Notebook-3"""
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def preprocess_image(image_path, transform=None, color_space='rgb'):
    """
    HÃ m tiá»�n xá»­ lÃ½ áº£nh cho cáº£ khÃ´ng gian mÃ u RGB vÃ  YCbCr
    
    Args:
        image_path: Ä�Æ°á»�ng dáº«n tá»›i file áº£nh
        transform: CÃ¡c biáº¿n Ä‘á»•i cáº§n Ã¡p dá»¥ng (albumentations)
        color_space: KhÃ´ng gian mÃ u ('rgb' hoáº·c 'ycbcr')
        
    Returns:
        áº¢nh Ä‘Ã£ Ä‘Æ°á»£c tiá»�n xá»­ lÃ½
    """
    if color_space.lower() == 'rgb':
        # Ä�á»�c áº£nh vá»›i OpenCV (RGB)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
    else:
        # Ä�á»�c áº£nh vá»›i PIL vÃ  chuyá»ƒn sang YCbCr
        image = Image.open(image_path).convert('YCbCr')
        image = np.array(image).astype(np.float32) / 255.0
    
    # Ã�p dá»¥ng cÃ¡c biáº¿n Ä‘á»•i náº¿u cÃ³
    if transform is not None:
        image = transform(image=image)['image']
    
    return image


def onehot(size, target):
    """
    Chuyá»ƒn Ä‘á»•i nhÃ£n thÃ nh vector one-hot
    
    Args:
        size: KÃ­ch thÆ°á»›c cá»§a vector one-hot (sá»‘ lÆ°á»£ng lá»›p)
        target: NhÃ£n cáº§n chuyá»ƒn Ä‘á»•i
        
    Returns:
        Vector one-hot
    """
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):
    """
    Lá»›p quáº£n lÃ½ dataset, xá»­ lÃ½ viá»‡c táº£i áº£nh vÃ  Ã¡p dá»¥ng transforms
    """
    def __init__(self, kinds, image_names, labels, transforms=None, color_space='rgb'):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.color_space = color_space

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        
        # Sá»­ dá»¥ng hÃ m preprocess_image Ä‘Ã£ Ä‘á»‹nh nghÄ©a trÆ°á»›c Ä‘Ã³
        image = preprocess_image(
            f'{DATA_ROOT_PATH}/{kind}/{image_name}', 
            transform=self.transforms, 
            color_space=self.color_space
        )
        
        # Chuyá»ƒn Ä‘á»•i nhÃ£n thÃ nh vector one-hot
        target = onehot(4, label)
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        """
        Tráº£ vá»� danh sÃ¡ch cÃ¡c nhÃ£n, há»¯u Ã­ch cho viá»‡c cÃ¢n báº±ng lá»›p
        """
        return list(self.labels)

def create_dataloaders(dataset, fold_number, batch_size=16, num_workers=4, color_space='rgb'):
    """
    Táº¡o DataLoader cho táº­p huáº¥n luyá»‡n vÃ  táº­p kiá»ƒm thá»­
    
    Args:
        dataset: DataFrame chá»©a thÃ´ng tin vá»� áº£nh
        fold_number: Fold Ä‘Æ°á»£c sá»­ dá»¥ng lÃ m táº­p kiá»ƒm thá»­
        batch_size: KÃ­ch thÆ°á»›c batch
        num_workers: Sá»‘ worker cho viá»‡c táº£i dá»¯ liá»‡u
        color_space: KhÃ´ng gian mÃ u ('rgb' hoáº·c 'ycbcr')
        
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Táº¡o dataset cho táº­p huáº¥n luyá»‡n
    train_dataset = DatasetRetriever(
        kinds=dataset[dataset['fold'] != fold_number].kind.values,
        image_names=dataset[dataset['fold'] != fold_number].image_name.values,
        labels=dataset[dataset['fold'] != fold_number].label.values,
        transforms=get_train_transforms(),
        color_space=color_space
    )
    
    # Táº¡o dataset cho táº­p kiá»ƒm thá»­
    val_dataset = DatasetRetriever(
        kinds=dataset[dataset['fold'] == fold_number].kind.values,
        image_names=dataset[dataset['fold'] == fold_number].image_name.values,
        labels=dataset[dataset['fold'] == fold_number].label.values,
        transforms=get_valid_transforms(),
        color_space=color_space
    )
    
    # Sá»­ dá»¥ng BalanceClassSampler Ä‘á»ƒ cÃ¢n báº±ng lá»›p trong quÃ¡ trÃ¬nh huáº¥n luyá»‡n
    train_loader = DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        batch_size=batch_size,
        pin_memory=True,  # Báº­t pin_memory
        drop_last=True,
        num_workers=num_workers,    # TÄƒng sá»‘ worker (nÃªn báº±ng sá»‘ lÃµi CPU)
    )
    
    # DataLoader cho táº­p kiá»ƒm thá»­
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=True,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

# VÃ­ dá»¥ sá»­ dá»¥ng
def show_dataset_example():
    """Hiá»ƒn thá»‹ má»™t sá»‘ vÃ­ dá»¥ tá»« dataset"""
    fold_number = 0
    #dataset = prepare_dataset_with_groupkfold()
    # Táº¡o train vÃ  validation dataset
    train_dataset = DatasetRetriever(
        kinds=dataset[dataset['fold'] != fold_number].kind.values,
        image_names=dataset[dataset['fold'] != fold_number].image_name.values,
        labels=dataset[dataset['fold'] != fold_number].label.values,
        transforms=get_valid_transforms()  # Sá»­ dá»¥ng valid transforms Ä‘á»ƒ khÃ´ng lÃ m biáº¿n Ä‘á»•i áº£nh
    )
    
    # Hiá»ƒn thá»‹ má»™t sá»‘ vÃ­ dá»¥
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(8):
        image, target = train_dataset[i]
        label = torch.argmax(target).item()
        label_names = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
        
        # Chuyá»ƒn tensor thÃ nh numpy Ä‘á»ƒ hiá»ƒn thá»‹
        numpy_image = image.permute(1, 2, 0).cpu().numpy()
        
        row, col = i // 4, i % 4
        axes[row, col].imshow(numpy_image)
        axes[row, col].set_title(f'Class: {label_names[label]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle('VÃ­ dá»¥ tá»« Dataset', fontsize=16, y=1.02)
    plt.show()
    
    return train_dataset

# Hiá»ƒn thá»‹ vÃ­ dá»¥ tá»« dataset
train_dataset = show_dataset_example()

# Kiá»ƒm tra vector one-hot
print("VÃ­ dá»¥ vá»� vector one-hot cho tá»«ng lá»›p:")
for i in range(4):
    print(f"Class {i}: {onehot(4, i)}")


class CheckpointCallback:
    """
    Callback Ä‘á»ƒ lÆ°u checkpoint trong quÃ¡ trÃ¬nh huáº¥n luyá»‡n
    """
    def __init__(self, checkpoint_interval=1, save_path='checkpoints', model_prefix='model'):
        self.checkpoint_interval = checkpoint_interval
        self.save_path = save_path
        self.model_prefix = model_prefix
        os.makedirs(save_path, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        """LÆ°u checkpoint sau má»—i sá»‘ epoch nháº¥t Ä‘á»‹nh"""
        logs = logs or {}
        
        if (epoch + 1) % self.checkpoint_interval == 0:
            # Láº¥y cÃ¡c giÃ¡ trá»‹ cáº§n thiáº¿t
            model = logs.get('model')
            optimizer = logs.get('optimizer')
            scheduler = logs.get('scheduler')
            scaler = logs.get('scaler')
            val_auc = logs.get('val_auc', 0)
            
            # Táº¡o dá»¯ liá»‡u Ä‘á»ƒ lÆ°u vÃ o file data
            data = {
                'epoch': epoch,
                'train_loss': logs.get('train_loss', 0),
                'val_loss': logs.get('val_loss', 0),
                'val_acc': logs.get('val_acc', 0),
                'val_auc': val_auc,
                'lr': logs.get('lr', 0),
                'class_acc': logs.get('class_acc', {})
            }
            
            # Táº¡o tÃªn file data vÃ  model
            data_path = os.path.join(
                self.save_path, 
                f"data_epoch{epoch+1}.json"
            )
            
            model_path = os.path.join(
                self.save_path, 
                f"{self.model_prefix}_epoch{epoch+1}_{val_auc:.4f}.pth"
            )
            
            # LÆ°u file data
            with open(data_path, 'w') as f:
                import json
                json.dump(data, f, indent=4)
            
            # LÆ°u file model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_auc': val_auc
            }, model_path)
            
            print(f"\n[CHECKPOINT] Ä�Ã£ lÆ°u data táº¡i: {data_path}")
            print(f"[CHECKPOINT] Ä�Ã£ lÆ°u model táº¡i: {model_path}")


from efficientnet_pytorch import EfficientNet
## # Import thÆ° viá»‡n
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SequentialSampler
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm.notebook import tqdm
from torch.cuda.amp import  GradScaler

# Compatibility layer for PyTorch versions before 1.6
try:
    from torch.cuda.amp import autocast
except ImportError:
    # Create a dummy autocast context manager for older PyTorch versions
    class autocast:
        def __init__(self, enabled=True, dtype=None):
            self.enabled = enabled
            self.dtype = dtype
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


# XÃ¡c Ä‘á»‹nh thiáº¿t bá»‹
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Ä�á»‹nh nghÄ©a lá»›p EfficientNetwork dá»±a trÃªn Notebook-1
class EfficientNetwork(nn.Module):
    """
    MÃ´ hÃ¬nh EfficientNet-B2 cho bÃ i toÃ¡n phÃ¢n loáº¡i áº£nh Alaska2
    
    Args:
        output_size: Sá»‘ lÆ°á»£ng lá»›p Ä‘áº§u ra (4 cho Alaska2)
    """
    def __init__(self, output_size=4):
        super().__init__()
        
        # Táº£i pretrained EfficientNet-B2
        self.features = EfficientNet.from_pretrained('efficientnet-b2')
        
        # Lá»›p fully connected cuá»‘i cÃ¹ng Ä‘á»ƒ phÃ¢n loáº¡i
        # EfficientNet-B2 cÃ³ Ä‘áº§u ra lÃ  1408 features
        self.classifier = nn.Linear(1408, output_size)
        
    def forward(self, x, feature_extract=False):
        """
        Forward pass
        
        Args:
            x: Tensor Ä‘áº§u vÃ o
            feature_extract: Náº¿u True, tráº£ vá»� features trÆ°á»›c lá»›p phÃ¢n loáº¡i
            
        Returns:
            Logits hoáº·c features tÃ¹y thuá»™c vÃ o feature_extract
        """
        # TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
        features = self.features.extract_features(x)
        
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(features, 1)
        pooled_features = pooled_features.flatten(start_dim=1)
        
        # Náº¿u chá»‰ trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
        if feature_extract:
            return pooled_features
        
        # PhÃ¢n loáº¡i
        logits = self.classifier(pooled_features)
        
        return logits

# Ä�á»‹nh nghÄ©a lá»›p EfficientNetwork b3
class EfficientNetB3Model(nn.Module):
    """
    MÃ´ hÃ¬nh EfficientNet-B3 cho bÃ i toÃ¡n phÃ¢n loáº¡i áº£nh Alaska2
    
    Args:
        output_size: Sá»‘ lÆ°á»£ng lá»›p Ä‘áº§u ra (4 cho Alaska2)
        pretrained: CÃ³ sá»­ dá»¥ng pretrained weights khÃ´ng
    """
    def __init__(self, output_size=4, pretrained=True):
        super().__init__()
        
        # Táº£i pretrained EfficientNet-B3
        if pretrained:
            self.features = EfficientNet.from_pretrained('efficientnet-b3')
        else:
            self.features = EfficientNet.from_name('efficientnet-b3')
        
        # Lá»›p fully connected cuá»‘i cÃ¹ng
        # EfficientNet-B3 cÃ³ Ä‘áº§u ra lÃ  1536 features
        self.classifier = nn.Linear(1536, output_size)
        
        # ThÃªm dropout Ä‘á»ƒ giáº£m overfitting
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, feature_extract=False):
        # TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
        features = self.features.extract_features(x)
        
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(features, 1)
        pooled_features = pooled_features.flatten(start_dim=1)
        
        # Náº¿u chá»‰ trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
        if feature_extract:
            return pooled_features
        
        # Ã�p dá»¥ng dropout
        x = self.dropout(pooled_features)
        
        # PhÃ¢n loáº¡i
        logits = self.classifier(x)
        
        return logits
# HÃ m tÃ­nh weighted AUC tá»« Notebook-1

def alaska_weighted_auc(y_true, y_valid):
    """
    HÃ m tÃ­nh weighted AUC theo Ä‘á»‹nh nghÄ©a cá»§a cuá»™c thi Alaska2
    
    Args:
        y_true: NhÃ£n thá»±c (binary: 0 = Cover, 1 = Steganography)
        y_valid: Dá»± Ä‘oÃ¡n xÃ¡c suáº¥t
        
    Returns:
        weighted_auc: Weighted AUC score
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    
    # Size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    
    # The total area is normalized by the sum of weights
    normalization = np.dot(areas, weights)
    
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        
        if mask.any():
            x_padding = np.linspace(fpr[mask][-1], 1, 100)
            
            x = np.concatenate([fpr[mask], x_padding])
            y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
            # Normalize such that curve starts at y = 0
            y = y - y_min 
            score = metrics.auc(x, y)
            submetric = score * weight
            competition_metric += submetric
        
    return competition_metric / normalization


def train_model(model, train_loader, val_loader, num_epochs=10, 
               learning_rate=1e-3, weight_decay=1e-5,
               factor=0.5, patience=1, device=device, save_path='models',
               use_amp=True, accumulation_steps=4, cp_path=None, callbacks=None, best_model = None, mode = 2):
    """
    Huáº¥n luyá»‡n mÃ´ hÃ¬nh vá»›i há»— trá»£ callbacks vÃ  checkpoint
    
    Args:
        model: MÃ´ hÃ¬nh cáº§n huáº¥n luyá»‡n
        train_loader, val_loader: DataLoader cho táº­p train vÃ  validation
        num_epochs: Sá»‘ epochs huáº¥n luyá»‡n
        learning_rate: Tá»‘c Ä‘á»™ há»�c
        weight_decay: Há»‡ sá»‘ weight decay cho regularization
        label_smoothing: Há»‡ sá»‘ label smoothing
        factor, patience: ThÃ´ng sá»‘ cho ReduceLROnPlateau
        device: Thiáº¿t bá»‹ huáº¥n luyá»‡n (CPU/GPU)
        save_path: ThÆ° má»¥c lÆ°u model
        use_amp: Báº­t/táº¯t Automatic Mixed Precision
        accumulation_steps: Sá»‘ bÆ°á»›c tÃ­ch lÅ©y gradient
        resume_from: Ä�Æ°á»�ng dáº«n Ä‘áº¿n checkpoint Ä‘á»ƒ tiáº¿p tá»¥c huáº¥n luyá»‡n
        callbacks: Danh sÃ¡ch callbacks sá»­ dá»¥ng trong quÃ¡ trÃ¬nh huáº¥n luyá»‡n
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Khá»Ÿi táº¡o callbacks náº¿u khÃ´ng cÃ³
    callbacks = callbacks or []
    
    # Khá»Ÿi táº¡o biáº¿n theo dÃµi vÃ  lá»‹ch sá»­
    start_epoch = 0
    best_auc = 0
    best_model_epoch = 0
    best_model_state = None
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'lr': []
    }
    
    # Ä�á»‹nh nghÄ©a loss function 
    criterion = nn.CrossEntropyLoss()
    
    # Khá»Ÿi táº¡o optimizer AdamW
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Khá»Ÿi táº¡o GradScaler cho Mixed Precision Training
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')
    
    # Náº¿u cÃ³ checkpoint, tiáº¿p tá»¥c tá»« checkpoint
    if cp_path and os.path.exists(cp_path) and best_model:
        
        model_path, best_auc = load_best_model(best_model)
        model.load_state_dict(torch.load(model_path))
        print(f"[RESUME] Loading checkpoint from {cp_path}")
        checkpoint = torch.load(cp_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, 
                            patience=1,  # Patience má»›i
                            threshold=0.001,  # NgÆ°á»¡ng cáº£i thiá»‡n
                            threshold_mode='abs',  # So sÃ¡nh tuyá»‡t Ä‘á»‘i
                            verbose=True)
    
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        
        # KhÃ´i phá»¥c GradScaler náº¿u cÃ³
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and use_amp and device.type == 'cuda':
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        
        print(f"[RESUME] Tiáº¿p tá»¥c huáº¥n luyá»‡n tá»« epoch {start_epoch}, Best AUC: {best_auc:.4f}")
    else:
            # Khá»Ÿi táº¡o scheduler ReduceLROnPlateau
         scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',               # DÃ¹ng AUC (max mode)
                factor=factor,               # Giáº£m LR 50%
                patience=patience,               # Chá»‰ Ä‘á»£i 1 epoch
                threshold=0.001,          # <-- Tham sá»‘ nÃ y
                threshold_mode='abs',     # So sÃ¡nh tuyá»‡t Ä‘á»‘i
                verbose=True
                )

    end_epoch = start_epoch + num_epochs
    print(f"[INFO] Báº¯t Ä‘áº§u quÃ¡ trÃ¬nh huáº¥n luyá»‡n tá»« epoch {start_epoch+1} Ä‘áº¿n {end_epoch}")
    print(f"[CONFIG] Learning rate: {optimizer.param_groups[0]['lr']}, Weight decay: {weight_decay}")
    print(f"[CONFIG] Scheduler: factor={scheduler.factor}, patience={scheduler.patience}")
    print(f"[CONFIG] Mixed Precision: {use_amp}, Gradient Accumulation: {accumulation_steps} steps")
    print(f"[CONFIG] Device: {device}")
    
    # In sá»‘ lÆ°á»£ng batch
    print(f"[INFO] Sá»‘ lÆ°á»£ng batch trong táº­p huáº¥n luyá»‡n: {len(train_loader)}")
    print(f"[INFO] Sá»‘ lÆ°á»£ng batch trong táº­p validation: {len(val_loader)}")
    
    # Thá»‘ng kÃª sá»‘ lÆ°á»£ng tham sá»‘ mÃ´ hÃ¬nh
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Tá»•ng sá»‘ tham sá»‘: {total_params:,}")
    print(f"[MODEL] Sá»‘ tham sá»‘ cÃ³ thá»ƒ huáº¥n luyá»‡n: {trainable_params:,}")
    
    # ThÃ´ng bÃ¡o báº¯t Ä‘áº§u huáº¥n luyá»‡n cho callbacks

    
    # VÃ²ng láº·p huáº¥n luyá»‡n
    for epoch in range(start_epoch, end_epoch):
        print(f"\n{'='*30} EPOCH {epoch+1}/{end_epoch} {'='*30}")
        start_time = time.time()
        
        # ThÃ´ng bÃ¡o báº¯t Ä‘áº§u epoch cho callbacks
        epoch_logs = {'epoch': epoch}
        # === Training ===
        model.train()
        train_loss = 0
        batch_times = []
        optimizer.zero_grad()  # Zero gradients trÆ°á»›c khi báº¯t Ä‘áº§u epoch
        
        print(f"[TRAIN] Báº¯t Ä‘áº§u huáº¥n luyá»‡n epoch {epoch+1}")
        batch_start = time.time()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # ThÃ´ng bÃ¡o báº¯t Ä‘áº§u batch cho callbacks
                
            # Ä�Æ°a dá»¯ liá»‡u lÃªn device
            images = images.to(device)
            targets = targets.to(device)
            
            # Convert targets to class indices if they are one-hot encoded
            if targets.dim() > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            
            # Ensure targets are Long type
            targets = targets.long()
            
            # Mixed Precision Training
            with autocast(enabled=use_amp and device.type == 'cuda'):
                # Forward pass
                outputs = model(images)
                
                # TÃ­nh loss vÃ  chia cho sá»‘ bÆ°á»›c tÃ­ch lÅ©y
                loss = criterion(outputs, targets) / accumulation_steps
            
            # Backward vá»›i Mixed Precision
            scaler.scale(loss).backward()
            
            # Cáº­p nháº­t weights sau khi tÃ­ch lÅ©y Ä‘á»§ gradient
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Cáº­p nháº­t weights vÃ  zero gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            
            # TÃ­nh thá»�i gian cho batch nÃ y
            if (batch_idx + 1) % 50 == 0:
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)
                
                log_msg = f"[TRAIN] Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item()*accumulation_steps:.4f}, Thá»�i gian: {batch_time:.2f}s"
                print(log_msg, end='\r')  
            # Báº¯t Ä‘áº§u tÃ­nh thá»�i gian cho batch tiáº¿p theo
                batch_start = time.time()
        
        # TÃ­nh loss trung bÃ¬nh trÃªn táº­p train
        train_loss = train_loss / len(train_loader)
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        
        print(f"[TRAIN] Káº¿t thÃºc epoch {epoch+1} - Loss: {train_loss:.4f}, Thá»�i gian trung bÃ¬nh/batch: {avg_batch_time:.4f}s")
        
        # === Validation ===
        print(f"[VALID] Báº¯t Ä‘áº§u Ä‘Ã¡nh giÃ¡ epoch {epoch+1}")
        model.eval()
        val_loss = 0
        all_targets = []
        all_predictions = []
        val_start_time = time.time()
        
        with torch.no_grad():
            for images, targets in val_loader:
                # Ä�Æ°a dá»¯ liá»‡u lÃªn device
                images = images.to(device)
                targets = targets.to(device)
                
                # Save original targets for metrics calculation
                original_targets = targets.clone()
                
                # Convert targets to class indices if they are one-hot encoded
                if targets.dim() > 1 and targets.size(1) > 1:
                    targets = torch.argmax(targets, dim=1)
                
                # Ensure targets are Long type for loss calculation
                targets = targets.long()
                
                # Forward pass - khÃ´ng cáº§n Mixed Precision trong validation
                outputs = model(images)
                
                # TÃ­nh loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # LÆ°u láº¡i targets vÃ  predictions
                all_targets.append(original_targets.cpu().numpy())
                all_predictions.append(F.softmax(outputs, dim=1).cpu().numpy())
        
        val_time = time.time() - val_start_time
        print(f"[VALID] Thá»�i gian Ä‘Ã¡nh giÃ¡: {val_time:.2f}s")
        
        # TÃ­nh loss trung bÃ¬nh trÃªn táº­p validation
        val_loss = val_loss / len(val_loader)
        
        # GhÃ©p all_targets vÃ  all_predictions thÃ nh má»™t máº£ng duy nháº¥t
        all_targets = np.vstack(all_targets)
        all_predictions = np.vstack(all_predictions)
        
        # TÃ­nh Ä‘á»™ chÃ­nh xÃ¡c
        predicted_classes = np.argmax(all_predictions, axis=1)
        true_classes = np.argmax(all_targets, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        
        # PhÃ¢n tÃ­ch sai sá»‘ theo lá»›p
        class_acc = {}
        for class_idx in range(4):
            class_mask = (true_classes == class_idx)
            if np.sum(class_mask) > 0:
                class_correct = np.sum((predicted_classes == true_classes) & class_mask)
                class_total = np.sum(class_mask)
                class_acc[class_idx] = class_correct / class_total
        
        # Chuyá»ƒn vá»� dáº¡ng binary classification cho AUC
        # NhÃ£n: 0 = Cover (class 0), 1 = Steganography (class 1, 2, 3)
        binary_targets = (true_classes != 0).astype(int)
        
        # Sá»­a láº¡i pháº§n tÃ­nh binary_predictions Ä‘á»ƒ rÃµ rÃ ng hÆ¡n
        # Dá»± Ä‘oÃ¡n: XÃ¡c suáº¥t steganography = 1 - P(Cover)
        binary_predictions = 1 - all_predictions[:, 0]
        
        # TÃ­nh AUC
        auc_score = alaska_weighted_auc(binary_targets, binary_predictions)
        
        # LÆ°u láº¡i lá»‹ch sá»­
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(accuracy)
        history['val_auc'].append(auc_score)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - start_time
        
        # In káº¿t quáº£ epoch vá»›i thÃªm thÃ´ng tin
        print(f"\n[RESULT] Epoch {epoch+1}/{end_epoch} - Tá»•ng thá»�i gian: {epoch_time:.2f}s")
        print(f"[RESULT] Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"[RESULT] Accuracy: {accuracy:.4f} - AUC: {auc_score:.4f}")
        print(f"[RESULT] Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # LÆ°u model tá»‘t nháº¥t dá»±a trÃªn AUC
        if auc_score > best_auc:
            improvement = auc_score - best_auc
            best_auc = auc_score
            best_model_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            print(f"\n[SAVE] ğŸŒŸ Cáº­p nháº­t model tá»‘t nháº¥t vá»›i cáº£i thiá»‡n AUC: +{improvement:.4f}")
        else:
            print(f"\n[INFO] KhÃ´ng cÃ³ cáº£i thiá»‡n AUC. Tá»‘t nháº¥t: {best_auc:.4f}, Hiá»‡n táº¡i: {auc_score:.4f}")
        
        # Cáº­p nháº­t learning rate vá»›i scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(auc_score)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"\n[LR] ğŸ“‰ Giáº£m learning rate: {old_lr:.6f} -> {new_lr:.6f}")
        
        # ThÃ´ng bÃ¡o káº¿t thÃºc epoch cho callbacks
        epoch_logs.update({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': accuracy,
            'val_auc': auc_score,
            'class_acc': class_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scaler': scaler,
        })
        for callback in callbacks:
            callback.on_epoch_end(epoch, epoch_logs)
    
    # LÆ°u model tá»‘t nháº¥t sau khi hoÃ n thÃ nh táº¥t cáº£ cÃ¡c epochs
    if best_model_state is not None:
        model_path = os.path.join(save_path, f"efficientnet_best_auc{best_auc:.4f}.pth")
        torch.save(best_model_state, model_path)
        print("\n" + "="*70)
        print(f"[SAVE] Ä�Ã£ lÆ°u model tá»‘t nháº¥t (epoch {best_model_epoch}) vá»›i AUC: {best_auc:.4f}")
        print(f"[SAVE] Path: {model_path}")
    else:
        model_path = None
        print("[WARNING] KhÃ´ng tÃ¬m tháº¥y model tá»‘t nháº¥t Ä‘á»ƒ lÆ°u")
    
    # ThÃ´ng bÃ¡o káº¿t thÃºc huáº¥n luyá»‡n cho callbacks

    
    print("\n" + "="*70)
    print(f"[DONE] HoÃ n thÃ nh quÃ¡ trÃ¬nh huáº¥n luyá»‡n sau {num_epochs} epochs")
    print(f"[BEST] AUC tá»‘t nháº¥t Ä‘áº¡t Ä‘Æ°á»£c: {best_auc:.4f} táº¡i epoch {best_model_epoch}")
    if model_path:
        print(f"[BEST] Model tá»‘t nháº¥t Ä‘Æ°á»£c lÆ°u táº¡i: {model_path}")
    print("="*70)
    
    return best_model_epoch, model_path


def plot_training_from_json(json_file_path, figsize=(18, 15), save_path=None):
    """
    Váº½ biá»ƒu Ä‘á»“ quÃ¡ trÃ¬nh huáº¥n luyá»‡n tá»« file JSON chá»©a dá»¯ liá»‡u training metrics
    
    Args:
        json_file_path: Ä�Æ°á»�ng dáº«n Ä‘áº¿n file JSON chá»©a dá»¯ liá»‡u
        figsize: KÃ­ch thÆ°á»›c cá»§a figure (width, height)
        save_path: Ä�Æ°á»�ng dáº«n Ä‘á»ƒ lÆ°u hÃ¬nh áº£nh (náº¿u None, chá»‰ hiá»ƒn thá»‹)
        
    Returns:
        fig: Matplotlib figure object
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker
    
    # Ä�á»�c dá»¯ liá»‡u tá»« file JSON
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] KhÃ´ng thá»ƒ Ä‘á»�c file JSON: {str(e)}")
        return None
    
    # Kiá»ƒm tra xem dá»¯ liá»‡u cÃ³ Ä‘Ãºng Ä‘á»‹nh dáº¡ng khÃ´ng
    required_fields = ['epochs', 'train_loss', 'val_loss', 'val_acc', 'val_auc', 'lr']
    for field in required_fields:
        if field not in data:
            print(f"[ERROR] Thiáº¿u trÆ°á»�ng dá»¯ liá»‡u '{field}' trong file JSON")
            return None
    
    # Táº¡o figure vÃ  axes
    fig = plt.figure(figsize=figsize)
    
    # Ä�áº£m báº£o epochs lÃ  cÃ¡c sá»‘ nguyÃªn
    epochs = data['epochs']
    
    # 1. Biá»ƒu Ä‘á»“ Loss
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=1)
    ax1.plot(epochs, data['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, data['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Loss qua cÃ¡c Epoch', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Ä�Ã¡nh dáº¥u loss tháº¥p nháº¥t
    min_val_loss_idx = np.argmin(data['val_loss'])
    min_val_loss = data['val_loss'][min_val_loss_idx]
    min_epoch = epochs[min_val_loss_idx]
    ax1.plot(min_epoch, min_val_loss, 'ro', markersize=8)
    ax1.annotate(f'Min: {min_val_loss:.4f}', 
                xy=(min_epoch, min_val_loss),
                xytext=(min_epoch + 0.5, min_val_loss),
                fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Chá»‰ Ä‘áº·t ticks lÃ  cÃ¡c sá»‘ nguyÃªn
    
    # 2. Biá»ƒu Ä‘á»“ Accuracy vÃ  AUC
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2, rowspan=1)
    ax2.plot(epochs, data['val_acc'], 'g-', label='Accuracy')
    ax2.plot(epochs, data['val_auc'], 'm-', label='AUC')
    ax2.set_title('Accuracy & AUC', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Ä�Ã¡nh dáº¥u AUC cao nháº¥t
    max_auc_idx = np.argmax(data['val_auc'])
    max_auc = data['val_auc'][max_auc_idx]
    max_auc_epoch = epochs[max_auc_idx]
    ax2.plot(max_auc_epoch, max_auc, 'mo', markersize=8)
    ax2.annotate(f'Max AUC: {max_auc:.4f}', 
                xy=(max_auc_epoch, max_auc),
                xytext=(max_auc_epoch, max_auc + 0.05),
                fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Chá»‰ Ä‘áº·t ticks lÃ  cÃ¡c sá»‘ nguyÃªn
    
    # 3. Biá»ƒu Ä‘á»“ Learning Rate
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=1)
    ax3.plot(epochs, data['lr'], 'c-', marker='o')
    ax3.set_title('Learning Rate', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')  # Log scale cho learning rate
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Chá»‰ Ä‘áº·t ticks lÃ  cÃ¡c sá»‘ nguyÃªn
    
    # 4. Biá»ƒu Ä‘á»“ Ä‘á»™ chÃ­nh xÃ¡c theo lá»›p (Class Accuracy)
    if 'class_acc' in data and data['class_acc']:
        ax4 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=1)
        
        # Láº¥y sá»‘ lÆ°á»£ng lá»›p tá»« dá»¯ liá»‡u
        class_keys = sorted([key for key in data['class_acc'][0].keys()])
        class_names = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']  # TÃªn lá»›p tÆ°Æ¡ng á»©ng
        
        # Táº¡o máº£ng class accuracy
        class_acc_data = {}
        for class_key in class_keys:
            class_acc_data[class_key] = [epoch_data.get(class_key, 0) for epoch_data in data['class_acc']]
        
        # Váº½ biá»ƒu Ä‘á»“ cho tá»«ng lá»›p
        for class_key in class_keys:
            class_idx = int(class_key)
            if class_idx < len(class_names):
                label = f"{class_names[class_idx]} (Class {class_key})"
            else:
                label = f"Class {class_key}"
            ax4.plot(epochs, class_acc_data[class_key], marker='o', label=label)
        
        ax4.set_title('Ä�á»™ ChÃ­nh XÃ¡c Theo Lá»›p', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)
        
        # Chá»‰ Ä‘áº·t ticks lÃ  cÃ¡c sá»‘ nguyÃªn
    

    
    # 6. Biá»ƒu Ä‘á»“ tÃ³m táº¯t
    ax6 = plt.subplot2grid((4, 4), (3, 0), colspan=2, rowspan=1)
    
    # Táº¡o barchart cho epoch cuá»‘i cÃ¹ng
    if 'class_acc' in data and data['class_acc']:
        last_epoch_class_acc = data['class_acc'][-1]
        class_idxs = [int(key) for key in class_keys]
        class_accs = [last_epoch_class_acc.get(key, 0) for key in class_keys]
        
        classes = [class_names[idx] if idx < len(class_names) else f"Class {idx}" for idx in class_idxs]
        bars = ax6.bar(classes, class_accs, color='skyblue')
        
        # ThÃªm giÃ¡ trá»‹ trÃªn má»—i bar
        for bar in bars:
            height = bar.get_height()
            ax6.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)
        
        ax6.set_title('Ä�á»™ ChÃ­nh XÃ¡c Theo Lá»›p (Epoch Cuá»‘i)', fontsize=14)
        ax6.set_ylim(0, 1.0)
        ax6.set_ylabel('Accuracy')
        ax6.tick_params(axis='x', rotation=45)
    
    # Thiáº¿t láº­p layout
    for ax in [ax1, ax2, ax3, ax4]:
        if hasattr(ax, 'xaxis'):
            ax.set_xticks(epochs)
    plt.tight_layout()
    plt.suptitle(f'QuÃ¡ TrÃ¬nh Huáº¥n Luyá»‡n (Epoch {min(epochs)}-{max(epochs)})', 
                fontsize=18, y=0.98)
    plt.subplots_adjust(top=0.93)
    
    # LÆ°u hÃ¬nh náº¿u Ä‘Æ°á»£c yÃªu cáº§u
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Ä�Ã£ lÆ°u biá»ƒu Ä‘á»“ táº¡i: {save_path}")
    
    plt.show()
    #return fig


def load_best_model(folder_path):
    """
    TÃ¬m file model cÃ³ AUC cao nháº¥t trong má»™t thÆ° má»¥c dá»±a vÃ o tÃªn file
    
    Args:
        folder_path: Ä�Æ°á»�ng dáº«n Ä‘áº¿n thÆ° má»¥c chá»©a cÃ¡c file model
        
    Returns:
        tuple: (model_path, auc) - Ä�Æ°á»�ng dáº«n Ä‘áº¿n file model cÃ³ AUC cao nháº¥t vÃ  giÃ¡ trá»‹ AUC
    """
    import os
    import re
    from glob import glob
    
    # Kiá»ƒm tra thÆ° má»¥c cÃ³ tá»“n táº¡i khÃ´ng
    if not os.path.exists(folder_path):
        print(f"[ERROR] ThÆ° má»¥c '{folder_path}' khÃ´ng tá»“n táº¡i")
        return None, 0.0
    
    # TÃ¬m táº¥t cáº£ file .pth trong thÆ° má»¥c
    model_files = glob(os.path.join(folder_path, "*.pth"))
    
    if not model_files:
        print(f"[WARNING] KhÃ´ng tÃ¬m tháº¥y file .pth nÃ o trong thÆ° má»¥c '{folder_path}'")
        return None, 0.0
    
    # TÃ¬m file cÃ³ AUC cao nháº¥t
    best_model = None
    best_auc = -1.0
    
    for model_file in model_files:
        # TÃ¬m giÃ¡ trá»‹ AUC tá»« tÃªn file
        # Pattern Ä‘á»ƒ tÃ¬m auc + sá»‘ tháº­p phÃ¢n, vÃ­ dá»¥: auc0.8241
        match = re.search(r'auc(\d+\.\d+)', os.path.basename(model_file))
        
        if match:
            try:
                auc = float(match.group(1))
                if auc > best_auc:
                    best_auc = auc
                    best_model = model_file
            except ValueError:
                # Bá»� qua náº¿u khÃ´ng thá»ƒ chuyá»ƒn Ä‘á»•i thÃ nh float
                continue
    
    if best_model is None:
        print(f"[WARNING] KhÃ´ng tÃ¬m tháº¥y file .pth nÃ o cÃ³ chá»©a 'auc' trong tÃªn file")
        return None, 0.0
    
    print(f"[INFO] TÃ¬m tháº¥y model cÃ³ AUC cao nháº¥t: {os.path.basename(best_model)} (AUC: {best_auc:.4f})")
    return best_model, best_auc

# HÃ m Ä‘á»ƒ load model tá»« file
def load_model(model_path, device=device, mode = 2):
    """
    Load model tá»« file
    
    Args:
        model_path: Ä�Æ°á»�ng dáº«n Ä‘áº¿n file model
        device: Thiáº¿t bá»‹ Ä‘á»ƒ cháº¡y mÃ´ hÃ¬nh
        
    Returns:
        model: MÃ´ hÃ¬nh Ä‘Ã£ load
    """
    if mode == 3:
        model = EfficientNetB3Model().to(device)
    else:
        model = EfficientNetwork().to(device)
    pretrained = torch.load(model_path)
    if 'model_state_dict' in pretrained:
        print('* msd detect \n')
        model.load_state_dict(pretrained['model_state_dict'])
    else:
        model.load_state_dict(pretrained)
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

def combine_epoch_data(directory, output_file=None):
    """
    Káº¿t há»£p cÃ¡c file data_epoch{number}.json trong má»™t thÆ° má»¥c
    
    Args:
        directory: ThÆ° má»¥c chá»©a cÃ¡c file data
        output_file: TÃªn file output. Náº¿u None, tá»± Ä‘á»™ng táº¡o tÃªn dá»±a trÃªn epochs
        
    Returns:
        combined_data: Dá»¯ liá»‡u Ä‘Ã£ káº¿t há»£p
        output_path: Ä�Æ°á»�ng dáº«n file output
    """
    import glob
    import json
    import re
    
    # TÃ¬m táº¥t cáº£ file data_epoch*.json
    data_files = glob.glob(os.path.join(directory, "data_epoch*.json"))
    if not data_files:
        print(f"[WARNING] KhÃ´ng tÃ¬m tháº¥y file data_epoch*.json trong {directory}")
        return None, None
    
    # Sáº¯p xáº¿p file theo sá»‘ epoch
    data_files.sort(key=lambda x: int(re.search(r'data_epoch(\d+)\.json', x).group(1)))
    
    # Khá»Ÿi táº¡o combined_data
    combined_data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'lr': [],
        'class_acc': []
    }
    
    # Ä�á»�c vÃ  káº¿t há»£p dá»¯ liá»‡u tá»« cÃ¡c file
    for file_path in data_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # ThÃªm dá»¯ liá»‡u vÃ o combined_data
            combined_data['epochs'].append(data['epoch'] + 1)  # +1 vÃ¬ epoch Ä‘áº¿m tá»« 0
            combined_data['train_loss'].append(data['train_loss'])
            combined_data['val_loss'].append(data['val_loss'])
            combined_data['val_acc'].append(data['val_acc'])
            combined_data['val_auc'].append(data['val_auc'])
            combined_data['lr'].append(data['lr'])
            combined_data['class_acc'].append(data['class_acc'])
            
            print(f"[INFO] Ä�Ã£ Ä‘á»�c file {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[ERROR] KhÃ´ng thá»ƒ Ä‘á»�c file {file_path}: {str(e)}")
    
    # TÃ¬m epoch Ä‘áº§u vÃ  cuá»‘i
    if combined_data['epochs']:
        start_epoch = min(combined_data['epochs'])
        end_epoch = max(combined_data['epochs'])
        
        # Táº¡o tÃªn file output náº¿u khÃ´ng Ä‘Æ°á»£c cung cáº¥p
        if output_file is None:
            output_file = f"data_epoch_{start_epoch}_{end_epoch}.json"
        
        output_path = os.path.join('/kaggle/working/', output_file)
        
        # LÆ°u combined_data ra file
        with open(output_path, 'w') as f:
            json.dump(combined_data, f, indent=4)
        
        print(f"[INFO] Ä�Ã£ lÆ°u dá»¯ liá»‡u káº¿t há»£p táº¡i: {output_path}")
        return combined_data, output_path
    else:
        print("[WARNING] KhÃ´ng cÃ³ dá»¯ liá»‡u epoch Ä‘á»ƒ káº¿t há»£p")
        return None, None
def combine_result_files(directory, output_file=None):
    """
    Káº¿t há»£p cÃ¡c file result (data_epoch_x_y.json) thÃ nh má»™t file duy nháº¥t
    
    Args:
        directory: ThÆ° má»¥c chá»©a cÃ¡c file result
        output_file: TÃªn file output. Náº¿u None, tá»± Ä‘á»™ng táº¡o tÃªn dá»±a trÃªn epochs
        
    Returns:
        combined_data: Dá»¯ liá»‡u Ä‘Ã£ káº¿t há»£p
        output_path: Ä�Æ°á»�ng dáº«n file output
    """
    import glob
    import json
    import re
    
    # TÃ¬m táº¥t cáº£ file data_epoch_x_y.json
    result_files = glob.glob(os.path.join(directory, "data_epoch_*_*.json"))
    if not result_files:
        print(f"[WARNING] KhÃ´ng tÃ¬m tháº¥y file data_epoch_*_*.json trong {directory}")
        return None, None
    
    # PhÃ¢n tÃ­ch start_epoch vÃ  end_epoch tá»« tÃªn file
    file_info = []
    for file_path in result_files:
        match = re.search(r'data_epoch_(\d+)_(\d+)\.json', file_path)
        if match:
            start_epoch = int(match.group(1))
            end_epoch = int(match.group(2))
            file_info.append((file_path, start_epoch, end_epoch))
    
    # Sáº¯p xáº¿p file theo start_epoch
    file_info.sort(key=lambda x: x[1])
    
    # Khá»Ÿi táº¡o combined_data
    combined_data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'lr': [],
        'class_acc': []
    }
    
    # Ä�á»�c vÃ  káº¿t há»£p dá»¯ liá»‡u tá»« cÃ¡c file
    for file_path, start_epoch, end_epoch in file_info:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # ThÃªm dá»¯ liá»‡u vÃ o combined_data
            combined_data['epochs'].extend(data['epochs'])
            combined_data['train_loss'].extend(data['train_loss'])
            combined_data['val_loss'].extend(data['val_loss'])
            combined_data['val_acc'].extend(data['val_acc'])
            combined_data['val_auc'].extend(data['val_auc'])
            combined_data['lr'].extend(data['lr'])
            combined_data['class_acc'].extend(data['class_acc'])
            
            print(f"[INFO] Ä�Ã£ Ä‘á»�c file {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[ERROR] KhÃ´ng thá»ƒ Ä‘á»�c file {file_path}: {str(e)}")
    
    # Loáº¡i bá»� cÃ¡c báº£n ghi trÃ¹ng láº·p dá»±a trÃªn epoch
    unique_epochs = {}
    for i, epoch in enumerate(combined_data['epochs']):
        if epoch not in unique_epochs:
            unique_epochs[epoch] = i
    
    # Lá»�c dá»¯ liá»‡u Ä‘á»ƒ loáº¡i bá»� trÃ¹ng láº·p
    for key in combined_data.keys():
        if key != 'class_acc':  # class_acc lÃ  má»™t list cá»§a dict, xá»­ lÃ½ riÃªng
            combined_data[key] = [combined_data[key][i] for i in unique_epochs.values()]
    
    # Xá»­ lÃ½ riÃªng cho class_acc
    combined_data['class_acc'] = [combined_data['class_acc'][i] for i in unique_epochs.values()]
    
    # TÃ¬m epoch Ä‘áº§u vÃ  cuá»‘i sau khi Ä‘Ã£ lá»�c
    if combined_data['epochs']:
        start_epoch = min(combined_data['epochs'])
        end_epoch = max(combined_data['epochs'])
        
        # Táº¡o tÃªn file output náº¿u khÃ´ng Ä‘Æ°á»£c cung cáº¥p
        if output_file is None:
            output_file = f"data_epoch_{start_epoch}_{end_epoch}.json"
        
        output_path = os.path.join('/kaggle/working/', output_file)
        
        # LÆ°u combined_data ra file
        with open(output_path, 'w') as f:
            json.dump(combined_data, f, indent=4)
        
        print(f"[INFO] Ä�Ã£ lÆ°u dá»¯ liá»‡u káº¿t há»£p táº¡i: {output_path}")
        return combined_data, output_path
    else:
        print("[WARNING] KhÃ´ng cÃ³ dá»¯ liá»‡u epoch Ä‘á»ƒ káº¿t há»£p")
        return None, None




def load_checkpoint(directory, epoch=None):
    """
    Táº£i checkpoint tá»« má»™t thÆ° má»¥c vá»›i epoch cá»¥ thá»ƒ
    hoáº·c tÃ¬m checkpoint cá»§a epoch cao nháº¥t náº¿u epoch=None
    
    Args:
        directory: ThÆ° má»¥c chá»©a cÃ¡c file checkpoint
        epoch: Epoch cá»¥ thá»ƒ cáº§n táº£i. Náº¿u None, táº£i epoch cao nháº¥t
        
    Returns:
        checkpoint: Dá»¯ liá»‡u checkpoint Ä‘Ã£ táº£i
        model_path: Ä�Æ°á»�ng dáº«n Ä‘áº¿n file model
    """
    import glob
    import re
    
    # TÃ¬m táº¥t cáº£ file model
    model_files = glob.glob(os.path.join(directory, "*_epoch*_*.pth"))
    
    if not model_files:
        print(f"[WARNING] KhÃ´ng tÃ¬m tháº¥y file model trong {directory}")
        return None, None
    
    # Náº¿u epoch Ä‘Æ°á»£c chá»‰ Ä‘á»‹nh, tÃ¬m file model tÆ°Æ¡ng á»©ng
    if epoch is not None:
        model_path = None
        for file_path in model_files:
            match = re.search(r'_epoch(\d+)_', file_path)
            if match and int(match.group(1)) == epoch:
                model_path = file_path
                break
        
        if model_path is None:
            print(f"[WARNING] KhÃ´ng tÃ¬m tháº¥y model cho epoch {epoch} trong {directory}")
            return None, None
    else:
        # Náº¿u epoch khÃ´ng Ä‘Æ°á»£c chá»‰ Ä‘á»‹nh, tÃ¬m file model cá»§a epoch cao nháº¥t
        models_with_epoch = []
        for file_path in model_files:
            match = re.search(r'_epoch(\d+)_', file_path)
            if match:
                epoch_num = int(match.group(1))
                models_with_epoch.append((file_path, epoch_num))
        
        if not models_with_epoch:
            print(f"[WARNING] KhÃ´ng thá»ƒ trÃ­ch xuáº¥t thÃ´ng tin epoch tá»« tÃªn file model trong {directory}")
            return None, None
        
        # Sáº¯p xáº¿p theo epoch vÃ  láº¥y epoch cao nháº¥t
        model_path, highest_epoch = max(models_with_epoch, key=lambda x: x[1])
        print(f"[INFO] TÃ¬m tháº¥y model cho epoch cao nháº¥t ({highest_epoch}): {model_path}")
    
    # Táº£i checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"[INFO] Ä�Ã£ táº£i checkpoint tá»« {model_path}")
        return checkpoint, model_path
    except Exception as e:
        print(f"[ERROR] KhÃ´ng thá»ƒ táº£i checkpoint tá»« {model_path}: {str(e)}")
        return None, None


def train_and_evaluate_auto_session(fold_number=0, batch_size=16, 
                              resume_from=None, 
                                   start_epoch=0,
                                 epochs_per_session=3,
                                   save_dir='training',mode=2):
    """
    Huáº¥n luyá»‡n mÃ´ hÃ¬nh vá»›i tÃ­nh toÃ¡n tá»± Ä‘á»™ng sá»‘ epoch cho má»—i session dá»±a trÃªn thá»�i gian cháº¡y thá»±c táº¿
    
    Args:
        fold_number: Fold Ä‘Æ°á»£c sá»­ dá»¥ng lÃ m táº­p kiá»ƒm thá»­
        batch_size: KÃ­ch thÆ°á»›c batch
        color_space: KhÃ´ng gian mÃ u ('rgb' hoáº·c 'ycbcr')
        resume_from: File checkpoint Ä‘á»ƒ tiáº¿p tá»¥c tá»« phiÃªn trÆ°á»›c
        session_number: Sá»‘ thá»© tá»± phiÃªn
        total_desired_epochs: Tá»•ng sá»‘ epoch mong muá»‘n cháº¡y cho toÃ n bá»™ quÃ¡ trÃ¬nh huáº¥n luyá»‡n
        runtime_limit_hours: Giá»›i háº¡n thá»�i gian cháº¡y (giá»�)
        epochs_per_session: Sá»‘ epoch cho má»—i session (náº¿u None, sáº½ tá»± Ä‘á»™ng tÃ­nh)
        save_dir: ThÆ° má»¥c lÆ°u dá»¯ liá»‡u
        
    Returns:
        session_info: ThÃ´ng tin vá»� phiÃªn huáº¥n luyá»‡n
    """
    
    if resume_from:
        _, cp_path = load_checkpoint(resume_from)
    else:
        cp_path = None
    # Táº¡o thÆ° má»¥c lÆ°u trá»¯
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Báº­t CUDA benchmarking Ä‘á»ƒ tÄƒng tá»‘c
    torch.backends.cudnn.benchmark = True
    
    # Táº¡o tÃªn phiÃªn dá»±a trÃªn thá»�i gian vÃ  sá»‘ thá»© tá»±
    # session_path = os.path.join(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Táº¡o dataloader
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        dataset, fold_number, batch_size, num_workers=4
    )
    
    # Khá»Ÿi táº¡o model
    if mode == 3:
        model = EfficientNetB3Model().to(device)
    else:
        model = EfficientNetwork().to(device)
    #pretrained = torch.load('/kaggle/input/b2/pytorch/default/1/efficientnet_b2_best_auc0.7873.pth')
    # if 'model_state_dict' in pretrained:
    #     print('* msd detect \n')
    #     #model.load_state_dict(pretrained['model_state_dict'])
    # else:
    #     model.load_state_dict(pretrained)
    # Thiáº¿t láº­p callbacks vá»›i checkpoint
    checkpoint_callback = CheckpointCallback(
        checkpoint_interval=1,
        save_path=save_dir,
    )
    if mode == 3:
        lr = 0.0005
        wd = 0.0001
        acc = 8
    else:
        lr = 0.001
        wd = 0.00001
        acc = 4
    
    history, best_model_path = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=epochs_per_session,
        accumulation_steps=acc,
        callbacks=[checkpoint_callback],
        device=device,
        learning_rate = lr,
        weight_decay = wd,
        save_path=save_dir,
        cp_path = cp_path,
        best_model =  resume_from,
        mode = mode
    )
    
    # LÆ°u lá»‹ch sá»­ huáº¥n luyá»‡n
 
    # Combine epoch data tá»« session nÃ y
    # TÃ­nh toÃ¡n sá»‘ epoch vÃ  session cÃ²n láº¡i
    
    # TÃ¬m best model cá»§a session nÃ y
    if resume_from:
        combined_data, combined_path = combine_result_files(resume_from)
    _, combined_path = combine_epoch_data('/kaggle/working/training')
    _, total_path = combine_result_files('/kaggle/working/')
    _,best_model_info  = load_checkpoint('/kaggle/working/training')
    best_model_path,_ = load_best_model('/kaggle/working/training')

    
    # Táº¡o file session_info.json tÃ³m táº¯t káº¿t quáº£ session
    uploaded_files = upload_to_uploadcare(files=[best_model_path,total_path,best_model_info])
    # In thÃ´ng bÃ¡o cÃ¡c file cáº§n táº£i xuá»‘ng cho láº§n cháº¡y tiáº¿p theo
    print("\n[IMPORTANT] HÃƒY Táº¢I XUá»�NG CÃ�C FILE SAU Ä�á»‚ TIáº¾P Tá»¤C HUáº¤N LUYá»†N:")
    print(f"1. Model / Session : {os.path.basename(best_model_path)}")
    print(f"2. Data / Session :{os.path.basename(total_path)}")
    print(f"3. Checkpoint / Epoch: {os.path.basename(best_model_info)}")



    
   


def modify_scheduler_patience(checkpoint_path, new_patience=1, save_path=None):
    """
    Sá»­a Ä‘á»•i giÃ¡ trá»‹ patience cá»§a scheduler trong file checkpoint
    
    Args:
        checkpoint_path: Ä�Æ°á»�ng dáº«n Ä‘áº¿n file checkpoint
        new_patience: GiÃ¡ trá»‹ patience má»›i (máº·c Ä‘á»‹nh lÃ  1)
        save_path: Ä�Æ°á»�ng dáº«n Ä‘á»ƒ lÆ°u checkpoint má»›i (náº¿u None, sáº½ ghi Ä‘Ã¨ lÃªn file cÅ©)
    """
    # Xá»­ lÃ½ Ä‘Æ°á»�ng dáº«n lÆ°u
    if save_path is None:
        save_path = checkpoint_path
    
    print(f"[INFO] Ä�ang sá»­a Ä‘á»•i scheduler.patience trong {checkpoint_path}")
    
    # Load checkpoint hiá»‡n táº¡i
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'scheduler_state_dict' not in checkpoint or checkpoint['scheduler_state_dict'] is None:
        print(f"[ERROR] KhÃ´ng tÃ¬m tháº¥y scheduler_state_dict trong checkpoint")
        return False
    
    # Táº¡o optimizer táº¡m Ä‘á»ƒ khá»Ÿi táº¡o scheduler
    dummy_model = torch.nn.Linear(1, 1)
    dummy_optimizer = torch.optim.AdamW(dummy_model.parameters())
    
    # LÆ°u trá»¯ state_dict cÅ©
    old_scheduler_state = checkpoint['scheduler_state_dict']
    
    # In thÃ´ng tin scheduler cÅ©
    print(f"[INFO] ThÃ´ng tin scheduler cÅ©:")
    for key, value in old_scheduler_state.items():
        print(f"  - {key}: {value}")
    
    # Táº¡o scheduler má»›i vá»›i patience má»›i
    # Láº¥y mode vÃ  factor tá»« scheduler cÅ© náº¿u cÃ³
    mode = 'max'  # Máº·c Ä‘á»‹nh cho AUC
    factor = 0.5  # Máº·c Ä‘á»‹nh
    threshold = 0.0001  # Máº·c Ä‘á»‹nh
    threshold_mode = 'rel'  # Máº·c Ä‘á»‹nh
    
    # Náº¿u cÃ³ thÃ´ng tin vá»� threshold trong scheduler cÅ©, giá»¯ nguyÃªn
    if hasattr(old_scheduler_state, 'threshold'):
        threshold = old_scheduler_state['threshold']
    if hasattr(old_scheduler_state, 'threshold_mode'):
        threshold_mode = old_scheduler_state['threshold_mode']
    
    # Táº¡o scheduler má»›i vá»›i patience má»›i
    new_scheduler = ReduceLROnPlateau(
        dummy_optimizer,
        mode=mode,
        factor=factor,
        patience=new_patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        verbose=True
    )
    
    # Load state_dict cÅ© Ä‘á»ƒ giá»¯ nguyÃªn tráº¡ng thÃ¡i nhÆ° best, num_bad_epochs
    new_scheduler.load_state_dict(old_scheduler_state)
    
    # GÃ¡n trá»±c tiáº¿p patience má»›i
    # Ä�Ã¢y lÃ  cÃ¡ch tiáº¿p cáº­n "Ã©p buá»™c" scheduler sá»­ dá»¥ng patience má»›i
    new_scheduler.patience = new_patience
    
    # Cáº­p nháº­t state_dict trong checkpoint
    checkpoint['scheduler_state_dict'] = new_scheduler.state_dict()
    
    # LÆ°u checkpoint Ä‘Ã£ sá»­a Ä‘á»•i
    torch.save(checkpoint, save_path)
    
    print(f"[SUCCESS] Ä�Ã£ sá»­a Ä‘á»•i scheduler.patience tá»« {old_scheduler_state.get('patience', 'unknown')} thÃ nh {new_patience}")
    print(f"[SUCCESS] Ä�Ã£ lÆ°u checkpoint táº¡i: {save_path}")
    
    return True


def get_max_number_folder(base_path):
    """
    Tráº£ vá»� Ä‘Æ°á»�ng dáº«n Ä‘áº¿n thÆ° má»¥c cÃ³ tÃªn lÃ  sá»‘ lá»›n nháº¥t trong path gá»‘c
    
    Args:
        base_path: Ä�Æ°á»�ng dáº«n gá»‘c cáº§n tÃ¬m kiáº¿m
        
    Returns:
        String: Ä�Æ°á»�ng dáº«n Ä‘áº§y Ä‘á»§ Ä‘áº¿n thÆ° má»¥c cÃ³ sá»‘ lá»›n nháº¥t
    """
    import os
    
    # Kiá»ƒm tra xem path gá»‘c cÃ³ tá»“n táº¡i khÃ´ng
    if not os.path.exists(base_path):
        print(f"[ERROR] Path gá»‘c khÃ´ng tá»“n táº¡i: {base_path}")
        return None
    
    # Láº¥y danh sÃ¡ch cÃ¡c thÆ° má»¥c trong path gá»‘c
    try:
        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    except Exception as e:
        print(f"[ERROR] KhÃ´ng thá»ƒ liá»‡t kÃª thÆ° má»¥c trong {base_path}: {str(e)}")
        return None
    
    # Lá»�c ra cÃ¡c thÆ° má»¥c cÃ³ tÃªn lÃ  sá»‘
    numeric_folders = []
    for folder in folders:
        try:
            numeric_folders.append(int(folder))
        except ValueError:
            # Bá»� qua cÃ¡c thÆ° má»¥c khÃ´ng pháº£i sá»‘
            continue
    
    # Náº¿u khÃ´ng cÃ³ thÆ° má»¥c nÃ o lÃ  sá»‘
    if not numeric_folders:
        print(f"[WARNING] KhÃ´ng tÃ¬m tháº¥y thÆ° má»¥c nÃ o cÃ³ tÃªn lÃ  sá»‘ trong {base_path}")
        return None
    
    # TÃ¬m sá»‘ lá»›n nháº¥t
    max_number = max(numeric_folders)
    
    # Tráº£ vá»� Ä‘Æ°á»�ng dáº«n Ä‘áº§y Ä‘á»§
    return os.path.join(base_path, str(max_number))


path = get_max_number_folder('/kaggle/input/model-b2/pytorch/model/')
path3 = None
path3 = get_max_number_folder('/kaggle/input/b3/pytorch/default/')



train_and_evaluate_auto_session(
    fold_number=0,
    batch_size=16,
    start_epoch=0,  # Báº¯t Ä‘áº§u tá»« epoch 0
    epochs_per_session=3,   # 3 epochs Ä‘áº§u tiÃªn
    resume_from= path,  # KhÃ´ng cÃ³ checkpoint trÆ°á»›c Ä‘Ã³
    mode = 2
)


# PhiÃªn huáº¥n luyá»‡n model 2
train_and_evaluate_auto_session(
    fold_number=0,
    batch_size=16,
    start_epoch=0,  # Báº¯t Ä‘áº§u tá»« epoch 0
    epochs_per_session=3,   # 3 epochs Ä‘áº§u tiÃªn
    resume_from= path,  # KhÃ´ng cÃ³ checkpoint trÆ°á»›c Ä‘Ã³
    mode = 2
)
# PhiÃªn huáº¥n luyá»‡n model 3
# train_and_evaluate_auto_session(
#     fold_number=0,
#     batch_size= 12,
#     start_epoch=0,  # Báº¯t Ä‘áº§u tá»« epoch 0
#     epochs_per_session=3,   # 3 epochs Ä‘áº§u tiÃªn
#     resume_from= path3,  # KhÃ´ng cÃ³ checkpoint trÆ°á»›c Ä‘Ã³
#     mode = 3
# )



plot_training_from_json('/kaggle/input/model-b2/pytorch/model/13/data_epoch_1_24.json')


plot_training_from_json('/kaggle/input/model-b2/pytorch/model/13/data_epoch_1_4.json')


plot_training_from_json('/kaggle/input/b3/pytorch/default/3/data_epoch_1_4.json')


class DatasetSubmissionRetriever(Dataset):
    """
    Dataset dÃ¹ng cho pháº§n dá»± Ä‘oÃ¡n táº­p Test
    """
    def __init__(self, image_names, transforms=None, color_space='rgb'):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms
        self.color_space = color_space

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        # Sá»­ dá»¥ng hÃ m tiá»�n xá»­ lÃ½ Ä‘Ã£ táº¡o trÆ°á»›c Ä‘Ã³
        image = preprocess_image(
            f'{DATA_ROOT_PATH}/Test/{image_name}',
            transform=self.transforms,
            color_space=self.color_space
        )
        return image_name, image

    def __len__(self) -> int:
        return len(self.image_names)


def predict_with_tta(model, images, device=device):
    """
    Dá»± Ä‘oÃ¡n vá»›i Test Time Augmentation (TTA)
    
    Args:
        model: MÃ´ hÃ¬nh Ä‘Ã£ huáº¥n luyá»‡n
        images: Tensor áº£nh Ä‘áº§u vÃ o
        device: Thiáº¿t bá»‹ Ä‘á»ƒ cháº¡y mÃ´ hÃ¬nh
        
    Returns:
        outputs: Dá»± Ä‘oÃ¡n vá»›i TTA
    """
    # Dá»± Ä‘oÃ¡n gá»‘c
    out0 = model(images)
    
    # Láº­t dá»�c
    images_vertical = images.flip(2)
    out1 = model(images_vertical)
    
    # Láº­t ngang
    images_horizontal = images.flip(3)
    out2 = model(images_horizontal)
    
    # Láº­t cáº£ dá»�c vÃ  ngang
    images_both = images_horizontal.flip(2)
    out3 = model(images_both)
    
    # Káº¿t há»£p káº¿t quáº£ (trá»�ng sá»‘ báº±ng nhau)
    outputs = (out0 + out1 + out2 + out3) / 4.0
    
    return outputs


def generate_predictions(model_path, batch_size=16, num_workers=4, use_tta=True, mode=2):
    """
    Táº¡o dá»± Ä‘oÃ¡n cho táº­p Test vÃ  táº¡o file submission
    
    Args:
        model_path: Ä�Æ°á»�ng dáº«n Ä‘áº¿n mÃ´ hÃ¬nh tá»‘t nháº¥t
        batch_size: KÃ­ch thÆ°á»›c batch
        num_workers: Sá»‘ lÆ°á»£ng worker
        use_tta: CÃ³ sá»­ dá»¥ng Test Time Augmentation hay khÃ´ng
        color_space: KhÃ´ng gian mÃ u sá»­ dá»¥ng ('rgb' hoáº·c 'ycbcr')
    
    Returns:
        submission_df: DataFrame chá»©a káº¿t quáº£ dá»± Ä‘oÃ¡n
    """
    # Táº£i mÃ´ hÃ¬nh tá»‘t nháº¥t
    print(f"[INFO] Ä�ang táº£i mÃ´ hÃ¬nh tá»« {model_path}")
    model = load_model(model_path)
    model.eval()
    
    # Láº¥y danh sÃ¡ch áº£nh Test
    test_images = np.array([os.path.basename(path) for path in glob(f'{DATA_ROOT_PATH}/Test/*.jpg')])
    print(f"[INFO] TÃ¬m tháº¥y {len(test_images)} áº£nh trong táº­p Test")
    
    # Táº¡o dataset vÃ  dataloader
    test_dataset = DatasetSubmissionRetriever(
        image_names=test_images,
        transforms=get_valid_transforms(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    
    # Biáº¿n lÆ°u káº¿t quáº£
    result = {'Id': [], 'Label': []}
    
    # Thá»±c hiá»‡n dá»± Ä‘oÃ¡n
    print(f"[INFO] Báº¯t Ä‘áº§u dá»± Ä‘oÃ¡n vá»›i {'TTA' if use_tta else 'khÃ´ng TTA'}")
    with torch.no_grad():
        for image_names, images in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            
            # Náº¿u dÃ¹ng TTA
            if use_tta:
                outputs = predict_with_tta(model, images)
            else:
                outputs = model(images)
            
            # TÃ­nh xÃ¡c suáº¥t
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            # TÃ­nh Ä‘iá»ƒm steganographic (CÃ¡ch tÃ­nh tá»« cuá»™c thi: 1 - P(Cover))
            stego_scores = 1 - probs[:, 0]
            
            # LÆ°u káº¿t quáº£
            result['Id'].extend(image_names)
            result['Label'].extend(stego_scores)
    
    # Táº¡o DataFrame káº¿t quáº£
    submission_df = pd.DataFrame(result)
    print(f"[INFO] HoÃ n thÃ nh dá»± Ä‘oÃ¡n. Sá»‘ lÆ°á»£ng dá»± Ä‘oÃ¡n: {len(submission_df)}")
    
    return submission_df


def save_submission(submission_df, filename='submission.csv'):
    """
    LÆ°u káº¿t quáº£ dá»± Ä‘oÃ¡n vÃ o file submission
    
    Args:
        submission_df: DataFrame chá»©a káº¿t quáº£ dá»± Ä‘oÃ¡n
        filename: TÃªn file lÆ°u káº¿t quáº£
    """
    submission_df.to_csv(filename, index=False)
    print(f"[INFO] Ä�Ã£ lÆ°u file submission táº¡i: {filename}")
    



def run_inference(model_path=None, batch_size=32, use_tta=True):
    """
    Cháº¡y toÃ n bá»™ quÃ¡ trÃ¬nh inference tá»« táº£i mÃ´ hÃ¬nh Ä‘áº¿n táº¡o file submission
    
    Args:
        model_path: Ä�Æ°á»�ng dáº«n Ä‘áº¿n mÃ´ hÃ¬nh tá»‘t nháº¥t. Náº¿u None, sáº½ tÃ¬m mÃ´ hÃ¬nh cÃ³ AUC cao nháº¥t
        batch_size: KÃ­ch thÆ°á»›c batch
        use_tta: CÃ³ sá»­ dá»¥ng Test Time Augmentation hay khÃ´ng
        color_space: KhÃ´ng gian mÃ u sá»­ dá»¥ng ('rgb' hoáº·c 'ycbcr')
        output_file: TÃªn file submission. Náº¿u None, sáº½ táº¡o tÃªn theo thá»�i gian
    
    Returns:
        submission_df: DataFrame chá»©a káº¿t quáº£ dá»± Ä‘oÃ¡n
    """

    
    # Náº¿u khÃ´ng cung cáº¥p tÃªn file output, táº¡o tÃªn theo thá»�i gian
    
    # Táº¡o dá»± Ä‘oÃ¡n
    submission_df = generate_predictions(
        model_path=model_path,
        batch_size=batch_size,
        use_tta=use_tta,
        mode = 2
    )
    
    # LÆ°u káº¿t quáº£
    save_submission(submission_df)
    
    return submission_df


best_model,_ = load_best_model(path)

submission_df = run_inference(
    model_path= best_model ,  
    batch_size=16,    # CÃ³ thá»ƒ tÄƒng batch size khi dá»± Ä‘oÃ¡n vÃ¬ khÃ´ng cáº§n lÆ°u gradient
    use_tta=True,     # Sá»­ dá»¥ng Test Time Augmentation 
)


def visualize_submission(submission_file='submission.csv', num_examples=5, test_dir=None):
    """
    Trá»±c quan hÃ³a dá»¯ liá»‡u tá»« file submission.csv
    
    Args:
        submission_file: Ä�Æ°á»�ng dáº«n Ä‘áº¿n file submission.csv
        num_examples: Sá»‘ lÆ°á»£ng vÃ­ dá»¥ áº£nh hiá»ƒn thá»‹
        test_dir: Ä�Æ°á»�ng dáº«n Ä‘áº¿n thÆ° má»¥c chá»©a áº£nh test (máº·c Ä‘á»‹nh lÃ  DATA_ROOT_PATH/Test)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import cv2
    from glob import glob
    import os
    
    # Ä�á»�c file submission
    df = pd.read_csv(submission_file)
    print(f"Ä�Ã£ Ä‘á»�c file {submission_file} vá»›i {len(df)} dá»± Ä‘oÃ¡n")
    
    # Táº¡o giao diá»‡n subplot
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Biá»ƒu Ä‘á»“ phÃ¢n phá»‘i xÃ¡c suáº¥t
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
    try:
        sns.histplot(df['Label'], bins=50, kde=True, ax=ax1)
    except:
        # Fallback náº¿u khÃ´ng cÃ³ histplot
        ax1.hist(df['Label'], bins=50)
        ax1.set_ylabel('Sá»‘ lÆ°á»£ng')
    
    ax1.set_title('PhÃ¢n phá»‘i xÃ¡c suáº¥t steganography', fontsize=15)
    ax1.set_xlabel('XÃ¡c suáº¥t chá»©a steganography')
    
    # 2. Biá»ƒu Ä‘á»“ táº§n suáº¥t tÃ­ch lÅ©y
    ax2 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
    sorted_probs = np.sort(df['Label'].values)
    cumulative = np.linspace(0, 1, len(sorted_probs))
    ax2.plot(sorted_probs, cumulative)
    ax2.set_title('Táº§n suáº¥t tÃ­ch lÅ©y', fontsize=15)
    ax2.set_xlabel('XÃ¡c suáº¥t')
    ax2.set_ylabel('Táº§n suáº¥t tÃ­ch lÅ©y')
    ax2.grid(True)
    
    # 3. Báº£ng thá»‘ng kÃª
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=1, rowspan=1)
    ax3.axis('off')
    
    stats = {
        'Min': df['Label'].min(),
        'Max': df['Label'].max(),
        'Mean': df['Label'].mean(),
        'Median': df['Label'].median(),
        'Std': df['Label'].std(),
        'Count': len(df),
        '% High Confidence\n(>0.8)': (df['Label'] > 0.8).mean() * 100,
        '% Low Confidence\n(<0.2)': (df['Label'] < 0.2).mean() * 100
    }
    
    stats_text = "THá»�NG KÃŠ SUBMISSION\n\n"
    for k, v in stats.items():
        if isinstance(v, float):
            stats_text += f"{k}: {v:.4f}\n"
        else:
            stats_text += f"{k}: {v}\n"
    
    ax3.text(0.1, 0.9, stats_text, fontsize=14, va='top')
    
    # 4. Dá»± Ä‘oÃ¡n theo phÃ¢n vá»‹
    ax4 = plt.subplot2grid((3, 3), (1, 1), colspan=2, rowspan=1)
    
    # Chia thÃ nh 5 nhÃ³m phÃ¢n vá»‹
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df['quantile'] = pd.qcut(df['Label'], q=quantiles, labels=labels)
    
    quantile_counts = df['quantile'].value_counts().sort_index()
    ax4.bar(quantile_counts.index, quantile_counts.values, color='skyblue')
    ax4.set_title('PhÃ¢n phá»‘i theo nhÃ³m xÃ¡c suáº¥t', fontsize=15)
    ax4.set_xlabel('NhÃ³m xÃ¡c suáº¥t')
    ax4.set_ylabel('Sá»‘ lÆ°á»£ng')
    
    for i, v in enumerate(quantile_counts.values):
        ax4.text(i, v + 5, str(v), ha='center')
    
    # 5. Hiá»ƒn thá»‹ má»™t sá»‘ áº£nh vÃ­ dá»¥
    if test_dir is None:
        test_dir = f'{DATA_ROOT_PATH}/Test/'
    
    # Chá»‰ hiá»ƒn thá»‹ áº£nh náº¿u thÆ° má»¥c test tá»“n táº¡i
    if os.path.exists(test_dir):
        # Chá»�n má»™t sá»‘ áº£nh Ä‘áº¡i diá»‡n tá»« cÃ¡c má»©c xÃ¡c suáº¥t khÃ¡c nhau
        sample_indices = []
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        for q in quantiles:
            idx = (np.abs(df['Label'].values - np.quantile(df['Label'].values, q))).argmin()
            sample_indices.append(idx)
            
        # Hiá»ƒn thá»‹ cÃ¡c áº£nh vÃ­ dá»¥
        for i, idx in enumerate(sample_indices[:num_examples]):
            img_name = df.iloc[idx]['Id']
            prob = df.iloc[idx]['Label']
            
            ax = plt.subplot2grid((3, 3), (2, i % 3), colspan=1, rowspan=1)
            
            try:
                img = cv2.imread(os.path.join(test_dir, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"P(Stego): {prob:.4f}", fontsize=14)
                ax.axis('off')
            except:
                ax.text(0.5, 0.5, f"KhÃ´ng thá»ƒ hiá»ƒn thá»‹ áº£nh\n{img_name}\nP(Stego): {prob:.4f}", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Trá»±c quan hÃ³a káº¿t quáº£ Submission', fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Má»™t sá»‘ phÃ¢n tÃ­ch thÃªm
    print("\nPhÃ¢n tÃ­ch dá»± Ä‘oÃ¡n:")
    print(f"Sá»‘ lÆ°á»£ng áº£nh cÃ³ kháº£ nÄƒng cao chá»©a steganography (>0.8): {(df['Label'] > 0.8).sum()} ({(df['Label'] > 0.8).mean()*100:.2f}%)")
    print(f"Sá»‘ lÆ°á»£ng áº£nh cÃ³ kháº£ nÄƒng tháº¥p chá»©a steganography (<0.2): {(df['Label'] < 0.2).sum()} ({(df['Label'] < 0.2).mean()*100:.2f}%)")
    print(f"Sá»‘ lÆ°á»£ng áº£nh cÃ³ dá»± Ä‘oÃ¡n khÃ´ng cháº¯c cháº¯n (0.4-0.6): {((df['Label'] >= 0.4) & (df['Label'] <= 0.6)).sum()} ({((df['Label'] >= 0.4) & (df['Label'] <= 0.6)).mean()*100:.2f}%)")
    
    return df


# VÃ­ dá»¥ sá»­ dá»¥ng
def analyze_submission_results(submission_file='/kaggle/working/submission.csv'):
    """
    PhÃ¢n tÃ­ch káº¿t quáº£ tá»« file submission vá»›i trá»±c quan hÃ³a
    
    Args:
        submission_file: Ä�Æ°á»�ng dáº«n Ä‘áº¿n file submission.csv
    """
    print(f"[INFO] PhÃ¢n tÃ­ch káº¿t quáº£ tá»« file {submission_file}")
    
    # Trá»±c quan hÃ³a káº¿t quáº£
    submission_df = visualize_submission(
        submission_file=submission_file,
        num_examples=3,
        test_dir=f'{DATA_ROOT_PATH}/Test/'
    )
    
    return submission_df

analyze_submission_results()

