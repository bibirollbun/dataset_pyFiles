!git clone https://github.com/VuTrinhNguyenHoang/Paddy-Disease-Classification.git
%cd Paddy-Disease-Classification


from src.models.backbones import mobilenet, resnet
from src.utils.data import PaddyDataset
from src.utils.metrics import *
from src.training import train_model, get_param_groups

import os, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms, datasets


from sklearn.model_selection import train_test_split


# SRC PATH
DATA_DIR = "/kaggle/input/data-group21"
TRAIN_DIR = f"{DATA_DIR}/train_images"
TEST_DIR = f"/kaggle/input/paddy-disease-classification/test_images"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TEST_CSV_PATH = f"/kaggle/input/paddy-disease-classification/sample_submission.csv"
MODELS_DIR = "/kaggle/input/midterm-model/pytorch/default/2"

# TRAIN
NUM_CLASSES = 4
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
LR = 6e-5
WD = 0.01
NUM_WORKERS = min(os.cpu_count(), 8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

# SEED 
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED);
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

print(NUM_WORKERS, DEVICE)


train_tfm = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE*1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

import kornia.augmentation as K
gpu_aug = nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.3),
    K.RandomAffine(degrees=15),
    K.ColorJitter(0.2, 0.2, 0.2, 0.05),
).to(DEVICE)

IMNET_MEAN_T = torch.tensor(IMNET_MEAN, device=DEVICE).view(1,3,1,1)
IMNET_STD_T  = torch.tensor(IMNET_STD,  device=DEVICE).view(1,3,1,1)


# full_train_df = pd.read_csv(TRAIN_CSV_PATH)
# test_df = pd.read_csv(TEST_CSV_PATH)

# full_train_df["path"] = TRAIN_DIR + "/" + full_train_df["label"] + "/" + full_train_df["image_id"]
# test_df["path"]= TEST_DIR + "/" + test_df["image_id"]

# display(full_train_df)


# train_df, valid_df = train_test_split(full_train_df, test_size=0.2, random_state=SEED, stratify=full_train_df["label"])
# train_df.shape, valid_df.shape


# unique_labels = train_df["label"].unique()

# label2id = {label: idx for idx, label in enumerate(unique_labels)}
# id2label = {idx: label for label, idx in label2id.items()}

# print(label2id)


# train_dataset = PaddyDataset(train_df, train_tfm, label2id)
# valid_dataset = PaddyDataset(valid_df, test_tfm, label2id)
# test_dataset = PaddyDataset(test_df, test_tfm, None)
# len(train_dataset), len(valid_dataset), len(test_dataset)


# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
#                           num_workers=NUM_WORKERS, pin_memory=True,
#                           persistent_workers=True, prefetch_factor=4)

# valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
#                           num_workers=NUM_WORKERS, pin_memory=True,
#                           persistent_workers=True, prefetch_factor=4)

# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
#                          num_workers=NUM_WORKERS, pin_memory=True,
#                          persistent_workers=True, prefetch_factor=4)

# len(train_loader), len(valid_loader), len(test_loader)


train_dir = "/kaggle/input/privatedataset-group21/rice_disease_ds_v1/train"
test_dir = "/kaggle/input/privatedataset-group21/rice_disease_ds_v1/test"
train_dataset = datasets.ImageFolder(train_dir, transform=train_tfm)
valid_dataset = datasets.ImageFolder(test_dir, transform=test_tfm)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)

len(train_dataset), len(valid_dataset), len(train_loader), len(valid_loader)


# MODELS = {
#     "mobile_bot": mobilenet.MobileNetV3_Small_BoT(NUM_CLASSES, 2, dropout=0.3),
#     "mobile_hybrid": mobilenet.MobileNetV3_Small_Hybrid(NUM_CLASSES, 2, 16, dropout=0.3),
#     "mobile_ca": mobilenet.MobileNetV3_Small_CA(NUM_CLASSES, 32, dropout=0.3),
#     "mobile_eca": mobilenet.MobileNetV3_Small_ECA(NUM_CLASSES, dropout=0.3),
    
#     "mobilevit": mobilenet.MobileViT_XXS(NUM_CLASSES, IMAGE_SIZE, dropout=0.3),
    
#     "resnet_bot": resnet.ResNet18_BoT(NUM_CLASSES, 2, dropout=0.3),
#     "resnet_bot_linear": resnet.ResNet18_BoTLinear(NUM_CLASSES, 2, dropout=0.3),
# }


# label_counts = train_df["label"].value_counts()
# total_samples = len(train_df)
# class_weights = []

# for label in unique_labels:
#     n_i = label_counts[label]
#     weight = total_samples / (NUM_CLASSES * n_i)
#     class_weights.append(weight)

# class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
# class_weights = class_weights / class_weights.sum() * len(class_weights)

# class_weights


import torch
from collections import Counter
import matplotlib.pyplot as plt

# train_dataset = datasets.ImageFolder(...)

labels = train_dataset.targets
label_counts = Counter(labels)

NUM_CLASSES = len(train_dataset.classes)
total_samples = len(train_dataset)

class_weights = []
for i in range(NUM_CLASSES):
    n_i = label_counts[i]
    weight = total_samples / (NUM_CLASSES * n_i)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
class_weights = class_weights / class_weights.sum() * len(class_weights)
print(class_weights)


# Chuyển sang cùng thứ tự với train_dataset.classes
class_names = train_dataset.classes
counts = [label_counts[i] for i in range(len(class_names))]

plt.figure(figsize=(6, 6))
plt.pie(counts,
        labels=class_names,
        autopct="%.1f%%",
        startangle=90,
        textprops={'fontsize': 10})

plt.title("Tỷ lệ ảnh theo từng lớp (Train Set)")
plt.show()


# results = []
# histories = {}


# for model_name, model in MODELS.items():
#     model.to(DEVICE)
#     criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
#     param_groups = get_param_groups(model, base_lr=LR, head_lr=LR*10, weight_decay=WD)
#     optimizer = torch.optim.AdamW(param_groups)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
#     scaler = torch.amp.GradScaler(enabled=(DEVICE.type=="cuda"))

#     history, result = train_model(model_name, model, train_loader, valid_loader, criterion, optimizer, scaler, scheduler, 
#                 gpu_aug=gpu_aug, MEAN=IMNET_MEAN_T, STD=IMNET_STD_T, epochs=EPOCHS, patience=PATIENCE)

#     results.append(result)
#     histories[model_name] = history


# results_df = pd.DataFrame(results).sort_values(["valid_f1"], ascending=[False]).reset_index(drop=True)
# results_df.to_csv("benchmark.csv", index=False)
# results_df


# best_model_name = results_df.iloc[0].model_name
# best_model = MODELS[best_model_name]
# best_model_name


# plot_history(histories[best_model_name], title=f"Best History Training")


# y_true, y_pred = predict(best_model, valid_loader, IMNET_MEAN_T, IMNET_STD_T)
# acc, f1 = accuracy_fscore(y_true, y_pred)
# cm = confusion(y_true, y_pred)

# acc, f1


# plot_confusion_matrix(cm, unique_labels)


# plot_loss_comparision(
#     histories=list(histories.values()),
#     labels=list(histories.keys())
# )


# metrics = {
#     row["model_name"]: {
#         "acc": row["valid_acc"], 
#         "fps": row["fps"]
#     }
#     for _, row in results_df.iterrows()
# }

# plot_accuracy_vs_fps(
#     metrics,
#     title="Accuracy vs FPS",
#     range_acc=(0.8, 1.0),
#     range_fps=(1000, 7000)
# )


# metrics = {
#     row["model_name"]: {
#         "size_mb": row["size_mb"]
#     }
#     for _, row in results_df.iterrows()
# }

# plot_param(metrics)


DATA = "ours data"
benchmark_df = pd.read_csv(f"{MODELS_DIR}/{DATA}/benchmark.csv")
benchmark_df = benchmark_df.sort_values(["valid_f1"], ascending=[False]).reset_index(drop=True)
benchmark_df.head(10)


EVALUATE_MODELS = {
    # "mobile_bot": mobilenet.MobileNetV3_Small_BoT(NUM_CLASSES, 2, dropout=0.3),
    "mobile_eca": mobilenet.MobileNetV3_Small_ECA(NUM_CLASSES, dropout=0.3),
    "mobile_ca": mobilenet.MobileNetV3_Small_CA(NUM_CLASSES, 32, dropout=0.3),
    "mobile_hybrid": mobilenet.MobileNetV3_Small_Hybrid(NUM_CLASSES, 2, 16, dropout=0.3),
    
    # "mobilevit": mobilenet.MobileViT_XXS(NUM_CLASSES, IMAGE_SIZE, dropout=0.3),
    
    # "resnet_bot": resnet.ResNet18_BoT(NUM_CLASSES, 2, dropout=0.3)
}


EVALUATE_MODELS.keys()


for _, row in benchmark_df.iterrows():
    if row["model_name"] in EVALUATE_MODELS.keys():
        model = EVALUATE_MODELS[row["model_name"]].to(DEVICE)
        ckpt = torch.load(os.path.join(MODELS_DIR, DATA, row["ckpt_path"]), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])


results = []
histories = {}


for model_name, model in EVALUATE_MODELS.items():
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    param_groups = get_param_groups(model, base_lr=1e-5, head_lr=5e-4, weight_decay=0.01)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type=="cuda"))

    history, result = train_model(model_name, model, train_loader, valid_loader, criterion, optimizer, scaler, scheduler, 
                gpu_aug=gpu_aug, MEAN=IMNET_MEAN_T, STD=IMNET_STD_T, epochs=20, patience=PATIENCE)

    results.append(result)
    histories[model_name] = history


finetune_df = pd.DataFrame(results).sort_values(["valid_f1"], ascending=[False]).reset_index(drop=True)
finetune_df


best_model_name = finetune_df.iloc[0].model_name
best_model = EVALUATE_MODELS[best_model_name]
best_model_name


plot_history(histories[best_model_name], title=f"Best History Training")


y_true, y_pred = predict(best_model, valid_loader, IMNET_MEAN_T, IMNET_STD_T)
acc, f1 = accuracy_fscore(y_true, y_pred)
cm = confusion(y_true, y_pred)
fps_value = fps(model, BATCH_SIZE, IMAGE_SIZE)

print("=" * 20, best_model_name, "=" * 20)
print(f"Valid Accuracy: {acc:.4f}")
print(f"Valid F1-Score: {f1:.4f}")
print(f"FPS:            {fps_value:.4f}")


plot_confusion_matrix(cm, unique_labels)

