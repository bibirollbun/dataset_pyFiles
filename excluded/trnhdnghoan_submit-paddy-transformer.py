import os
import timm
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from dataclasses import dataclass
from PIL import Image


class CFG:
    comp_root: str = "/kaggle/input/paddy-disease-classification"
    test_dir:  str = "/kaggle/input/paddy-disease-classification/test_images"
    sample_csv: str = "/kaggle/input/paddy-disease-classification/sample_submission.csv"
    ckpt_path: str = "/kaggle/input/transformer_our_data/pytorch/default/1/mobilevit_s_best_our_data.pth"  # đổi nếu khác

    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 lớp mô hình đã train (thứ tự PHẢI đúng lúc train)
    CLASSES_4 = ["Brow_Spot", "Leaf_Blast", "Leaf_Blight", "Normal"]
    id2label_4 = {i:c for i,c in enumerate(CLASSES_4)}

    # Map 4 → 10 lớp của challenge
    MAP_4_TO_10 = {
        "Brow_Spot": "brown_spot",
        "Leaf_Blast": "blast",
        "Leaf_Blight": "bacterial_leaf_blight",
        "Normal": "normal",
    }

    @staticmethod
    def infer_tfms(img_size):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])

cfg = CFG()


class TestDataset(Dataset):
    def __init__(self, folder, transform):
        self.ids = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.paths = [os.path.join(folder, f) for f in self.ids]
        self.tfm = transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.ids[i]


def create_model(backbone: str, num_classes: int, pretrained=True) -> nn.Module:
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, in_chans=3)
    return model


def load_model(backbone, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model = create_model(backbone, num_classes = ckpt["meta"]["num_classes"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(cfg.device).eval()
    return model


# BACKBONES = [
#     ('mobilevit_s','/kaggle/input/transformer_our_data/pytorch/default/1/mobilevit_s_best_our_data.pth'),
#     ('mobilevit_xs','/kaggle/input/transformer_our_data/pytorch/default/1/mobilevit_xs_best_our_data.pth'),
#     ('mobilevit_xxs','/kaggle/input/transformer_our_data/pytorch/default/1/mobilevit_xxs_best_our_data.pth'),
#     ('deit_tiny_patch16_224','/kaggle/input/transformer_our_data/pytorch/default/1/deit_tiny_patch16_224_best_our_data.pth'),
#     ('deit_small_patch16_224','/kaggle/input/transformer_our_data/pytorch/default/1/deit_small_patch16_224_best_our_data.pth'),
#     ('vit_tiny_patch16_224','/kaggle/input/transformer_our_data/pytorch/default/1/vit_tiny_patch16_224_best_our_data.pth'),
#     ('vit_small_patch16_224','/kaggle/input/transformer_our_data/pytorch/default/1/vit_small_patch16_224_best_our_data.pth')
# ]

BACKBONES = [
    ('vit_small_patch16_224','/kaggle/input/model-tk_deeplearning/pytorch/default/1/vit_small_patch16_224_best.pth')
]


ds = TestDataset(cfg.test_dir, CFG.infer_tfms(cfg.img_size))
dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)


for (backbone, ckpt_path) in BACKBONES:
    model = load_model(backbone, ckpt_path)
    preds_map = {}
    with torch.inference_mode():
        for x, ids in dl:
            x = x.to(cfg.device, non_blocking=True)
            pred_ids = model(x).argmax(1).cpu().tolist()
            for img_id, pid in zip(ids, pred_ids):
                four = cfg.id2label_4[pid]
                preds_map[img_id] = cfg.MAP_4_TO_10[four]
    
    sub = pd.read_csv(cfg.sample_csv)  # image_id, label
    sub["label"] = sub["image_id"].map(preds_map).fillna("normal")
    sub.to_csv(f"{backbone}_our_data.csv", index=False)
    print("Saved submission.csv")




