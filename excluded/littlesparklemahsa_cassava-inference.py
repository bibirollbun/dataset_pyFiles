import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
import timm

class CFG:
    img_size = 384
    batch_size = 64
    num_workers = 4
    device = 'cuda'
    test_dir = '/kaggle/input/cassava-leaf-disease-classification/test_images'
    model_paths = [
        '/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold0.pth',
        '/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold1.pth',
        '/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold2.pth',
        '/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold3.pth',
        '/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold4.pth',
    ]

test_tfms = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class TestDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted([str(p) for p in Path(folder).glob("*.jpg")])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_tfms(image=img)['image']
        return img, os.path.basename(self.paths[idx])

class CassavaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=5)
    def forward(self, x):
        return self.backbone(x)

@torch.no_grad()
def inference():
    dataset = TestDataset(CFG.test_dir)
    loader = DataLoader(dataset, batch_size=CFG.batch_size,
                        shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

    ensemble_preds = None

    for fold, path in enumerate(CFG.model_paths):
        print(f"Loading fold {fold} → {os.path.basename(path)}")
        model = CassavaModel().to(CFG.device)
        
        state = torch.load(path, map_location=CFG.device)
        model.load_state_dict(state)        
        model.eval()

        fold_preds = []
        for imgs, _ in tqdm(loader, leave=False, desc=f"Fold {fold} TTA"):
            imgs = imgs.to(CFG.device)
            with autocast(device_type='cuda'):
                p1 = torch.softmax(model(imgs), dim=1)
                p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)  
            fold_preds.append(((p1 + p2) / 2).cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        ensemble_preds = fold_preds if ensemble_preds is None else ensemble_preds + fold_preds
        
        del model, state
        torch.cuda.empty_cache()

    final_labels = np.argmax(ensemble_preds / len(CFG.model_paths), axis=1)

    sub = pd.DataFrame({
        'image_id': [os.path.basename(p) for p in dataset.paths],
        'label': final_labels
    })
    sub = sub.sort_values('image_id').reset_index(drop=True)
    sub.to_csv('submission.csv', index=False)
    
    print(f"\n{len(sub)} predictions")
    print(sub.head())
    return sub

inference()

