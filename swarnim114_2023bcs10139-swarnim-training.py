import sys
sys.path.append("../input/tez-lib/")
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


import random
import numpy as np
import torch
import os
import torch.nn as nn
import tez
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
import albumentations
import pandas as pd
import cv2
import timm
from sklearn import metrics
from tqdm import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)





class args:
    batch_size = 16
    image_size = 384
    epochs = 10
    fold = 0


class PawpularDataset:
    def __init__(self, image_paths, dense_features, targets, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        features = self.dense_features[item, :]
        targets = self.targets[item]
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


class PawpularModel(tez.Model):
    def __init__(self, num_train_steps):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True, in_chans=3, num_classes=0)
        num_image_features = self.model.num_features
        num_dense_features = 12
        
        # Improved head with LayerNorm and better architecture
        self.head = nn.Sequential(
            nn.Linear(num_image_features + num_dense_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.step_scheduler_after = "batch"
        self.num_train_steps = num_train_steps

    def monitor_metrics(self, outputs, targets):
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        rmse = np.sqrt(metrics.mean_squared_error(targets, outputs))
        return {"rmse": rmse}

    def fetch_optimizer(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=3e-4,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

    def fetch_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=3e-4,
            total_steps=self.num_train_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )

    def forward(self, image, features, targets=None):
        image_features = self.model(image)
        combined_features = torch.cat([image_features, features], dim=1)
        x = self.head(combined_features)
        
        if targets is not None:
            loss = nn.SmoothL1Loss()(x, targets.view(-1, 1))
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}


# Custom callback for manual early stopping and model saving
class SimpleEarlyStopping:
    def __init__(self, patience=4, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0
            
    def save_checkpoint(self, model, path):
        print(f'Validation loss improved. Saving model to {path}')
        torch.save(model.state_dict(), path)



train_aug = albumentations.Compose(
    [
        albumentations.LongestMaxSize(args.image_size, p=1),
        albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=cv2.BORDER_CONSTANT),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.1),
        albumentations.Rotate(limit=180, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


valid_aug = albumentations.Compose(
    [
        albumentations.LongestMaxSize(args.image_size, p=1),
        albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=cv2.BORDER_CONSTANT),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


df = pd.read_csv("/kaggle/input/swarnim-23bcs10139-kfold-pt/train_5folds.csv")
IMAGE_DIR_PATH = "/kaggle/input/petfinder-pawpularity-score/train/"

df.head()


import numpy as np

start_fold = 0

for i in range(start_fold, 5):
    # This is the corrected line
    print(f'{"-"*20} Training Fold: {i} {"-"*20}')
    args.fold = i

    df_train = df[df.kfold != args.fold].reset_index(drop=True)
    df_valid = df[df.kfold == args.fold].reset_index(drop=True)

    dense_features = [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ]

    train_img_paths = [os.path.join(IMAGE_DIR_PATH, f"{x}.jpg") for x in df_train["Id"].values]
    valid_img_paths = [os.path.join(IMAGE_DIR_PATH, f"{x}.jpg") for x in df_valid["Id"].values]

    train_dataset = PawpularDataset(
        image_paths=train_img_paths,
        dense_features=df_train[dense_features].values,
        targets=df_train.Pawpularity.values,
        augmentations=train_aug,
    )

    valid_dataset = PawpularDataset(
        image_paths=valid_img_paths,
        dense_features=df_valid[dense_features].values,
        targets=df_valid.Pawpularity.values,
        augmentations=valid_aug,
    )

    num_train_steps = int(np.ceil(len(train_dataset) / args.batch_size)) * args.epochs

    model = PawpularModel(num_train_steps)

    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=args.batch_size,
        valid_bs=2 * args.batch_size,
        epochs=args.epochs,
        fp16=True,
        n_jobs=os.cpu_count()
    )

    model.save(f"model_f{args.fold}.bin")
    # And this line is also corrected for consistency
    print(f'{"-"*20} Fold {i} training complete {"-"*20}\\n')





































