import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
import timm


import numpy as np
import pandas as pd
import random
import os
import math


from PIL import Image as pil_image
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader


SEED = 42
IMG_SIZE = 256

PROJECT_FOLDER = "../input/hotel-id-to-combat-human-trafficking-2022-fgvc9/"
TEST_DATA_FOLDER = PROJECT_FOLDER + "test_images/"


print(os.listdir(PROJECT_FOLDER))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


import albumentations as A
import albumentations.pytorch as APT
import cv2 

# used for training dataset - augmentations and occlusions
train_transform = A.Compose([
    A.HorizontalFlip(p=0.75),
    A.VerticalFlip(p=0.25),
    A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.OpticalDistortion(p=0.25),
    A.Perspective(p=0.25),
    A.CoarseDropout(p=0.5, min_holes=1, max_holes=6, 
                    min_height=IMG_SIZE//16, max_height=IMG_SIZE//4,
                    min_width=IMG_SIZE//16,  max_width=IMG_SIZE//4), # normal coarse dropout
    
    A.CoarseDropout(p=0.75, max_holes=1, 
                    min_height=IMG_SIZE//4, max_height=IMG_SIZE//2,
                    min_width=IMG_SIZE//4,  max_width=IMG_SIZE//2, 
                    fill_value=(255,0,0)),# simulating occlusions in test data

    A.RandomBrightnessContrast(p=0.75),
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])

# used for validation dataset - only occlusions
val_transform = A.Compose([
    A.CoarseDropout(p=0.75, max_holes=1, 
                    min_height=IMG_SIZE//4, max_height=IMG_SIZE//2,
                    min_width=IMG_SIZE//4,  max_width=IMG_SIZE//2, 
                    fill_value=(255,0,0)),# simulating occlusions
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])


def pad_image(img):
    w, h, c = np.shape(img)
    if w > h:
        pad = int((w - h) / 2)
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = int((h - w) / 2)
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        
    return img


def open_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pad_image(img)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


class HotelImageDataset:
    def __init__(self, data, transform=None, data_folder="train_images/"):
        self.data = data
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        image_path = self.data_folder + record["image_id"]
        
        image = np.array(open_and_preprocess_image(image_path)).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return {
            "image" : image,
        }


class HotelIdModel(nn.Module):
    def __init__(self, n_classes=100, backbone_name="resnet34"):
        super(HotelIdModel, self).__init__()
        
        self.backbone = timm.create_model(backbone_name, num_classes=n_classes, pretrained=False)

    def forward(self, x):
        return self.backbone(x)


def predict(loader, model, n_matches=5):
    preds = []
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample['image'].to(args.device)
            outputs = model(input)
            outputs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds.extend(outputs)
    
    # get 5 top predictions
    preds = np.argsort(-np.array(preds), axis=1)[:, :5]
    return preds


import albumentations as A
import albumentations.pytorch as APT
import cv2 

# used for training dataset - augmentations and occlusions
train_transform = A.Compose([
    A.RandomCrop(width=64, height=64),
    A.HorizontalFlip(p=0.75),
    #A.VerticalFlip(p=0.0),
    A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
    A.OpticalDistortion(p=0.25, distort_limit=0.05, shift_limit=0.01),
    A.Perspective(p=0.25, scale=(0.05, 0.1)),
    A.ColorJitter(p=0.75, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    A.CoarseDropout(p=0.5, min_holes=1, max_holes=5, 
                    min_height=IMG_SIZE//16, max_height=IMG_SIZE//8,
                    min_width=IMG_SIZE//16,  max_width=IMG_SIZE//8), # normal coarse dropout
    
    A.CoarseDropout(p=0.75, max_holes=1, 
                    min_height=IMG_SIZE//4, max_height=IMG_SIZE//2,
                    min_width=IMG_SIZE//4,  max_width=IMG_SIZE//2, 
                    fill_value=(255,0,0)),# simulating occlusions in test data
    #A.RandomBrightnessContrast(p=0.75),
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])

# used for validation dataset - only occlusions
val_transform = A.Compose([
    A.CoarseDropout(p=0.75, max_holes=1, 
                    min_height=IMG_SIZE//4, max_height=IMG_SIZE//2,
                    min_width=IMG_SIZE//4,  max_width=IMG_SIZE//2, 
                    fill_value=(255,0,0)),# simulating occlusions
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])

# no augmentations
base_transform = A.Compose([
    A.ToFloat(),
    APT.transforms.ToTensorV2(),
])


test_df = pd.DataFrame(data={"image_id": os.listdir(TEST_DATA_FOLDER), "hotel_id": ""}).sort_values(by="image_id")


# code hotel_id mapping created in training notebook by encoding hotel_ids
hotel_id_code_df = pd.read_csv('../input/resnet-training/hotel_id_code_mapping.csv')
hotel_id_code_map = hotel_id_code_df.set_index('hotel_id_code').to_dict()["hotel_id"]


def get_model(model_type, backbone_name, checkpoint_path, args):
    model = HotelIdModel(args.n_classes, backbone_name)
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model = model.to(args.device)
    
    return model


class args:
    batch_size = 64
    num_workers = 2
    n_classes = hotel_id_code_df["hotel_id"].nunique()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    
seed_everything(seed=SEED)

test_dataset = HotelImageDataset(test_df, base_transform, data_folder=TEST_DATA_FOLDER)
test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


model = get_model("classification", "resnet34",
                  "../input/resnet-training/checkpoint-classification-model-resnet34-256x256.pt", 
                  args)


%%time

preds = predict(test_loader, model)
# replace classes with hotel_id using mapping created in trainig notebook
preds = [[hotel_id_code_map[b] for b in a] for a in preds]
# transform array of hotel_ids into string
test_df["hotel_id"] = [str(list(l)).strip("[]").replace(",", "") for l in preds]

test_df.to_csv("submission.csv", index=False)
test_df.head()

