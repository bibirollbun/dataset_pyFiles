pip install imgaug


import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.patches import Rectangle
from matplotlib import cm
plt.style.use('ggplot')
sns.set_palette('viridis')



data_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"

with h5py.File(data_path, "r") as f:
    print("HDF5 Groups:", list(f.keys()))
    
    train_slides = list(f["images/Train"].keys())
    test_slides = list(f["images/Test"].keys())
    print("\nTraining Slides:", train_slides)
    print("Test Slide:", test_slides)

    train_images = {k: f["images/Train"][k][()] for k in f["images/Train"]}
    train_spots = {k: f["spots/Train"][k][()] for k in f["spots/Train"]}
    test_images = f["images/Test"]
    test_spots = f["spots/Test"]


    print("\n Train Slides Overview")
    print("------------------------")
    for name in train_images.keys():
        img_shape = train_images[name].shape
        spot_count = len(train_spots[name])
        spot_fields = train_spots[name].dtype.names
        print(f"Slide {name}:")
        print(f"   Image Shape : {img_shape}")
        print(f"   Spot Count  : {spot_count}")
        print(f"   Spot Fields : {spot_fields}")
        print("")

    print(" Test Slide Overview")
    print("------------------------")
    for name in test_images.keys():
        img_shape = test_images[name].shape
        spot_count = len(test_spots[name])
        spot_fields = test_spots[name].dtype.names
        print(f"Slide {name}:")
        print(f"   Image Shape : {img_shape}")
        print(f"   Spot Count  : {spot_count}")
        print(f"   Spot Fields : {spot_fields}")
        print("")




patch_w = patch_h = 75
fig, axes = plt.subplots(len(train_images), 3, figsize=(12, 4 * len(train_images)))

for i, slide in enumerate(train_images):
    img = train_images[slide]
    spots = train_spots[slide]

    idx = np.random.randint(len(spots))
    x0, y0 = spots["x"][idx], spots["y"][idx]
    vals = [spots[f"C{j}"][idx] for j in range(1, 36)]

    x1 = max(0, x0 - patch_w // 2)
    x2 = min(img.shape[1], x0 + patch_w // 2)
    y1 = max(0, y0 - patch_h // 2)
    y2 = min(img.shape[0], y0 + patch_h // 2)

    ax = axes[i, 0]
    ax.imshow(img)
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"{slide}: spot {idx}")
    ax.axis("off")

    ax = axes[i, 1]
    patch = img[y1:y2, x1:x2]
    ax.imshow(patch)
    rel_x = x0 - x1
    rel_y = y0 - y1
    ax.scatter(rel_x, rel_y, marker='x', color='red', s=100, lw=2)
    ax.set_title("Zoomed patch")
    ax.axis("off")

    ax = axes[i, 2]
    ax.bar(range(1, 36), vals, color=cm.viridis(np.linspace(0, 1, 35)))
    ax.set_xlabel("C1–C35")
    ax.set_ylabel("Abundance")
    ax.set_title("Spot composition")

fig.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

slide = "S_2"  
img = train_images[slide]
spots = train_spots[slide]

ct_cols = [f"C{j}" for j in range(1, 36)]
n_cols = 7
n_rows = int(np.ceil(35 / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), constrained_layout=True)
axes = axes.flatten()

for idx, ct in enumerate(ct_cols):
    ax = axes[idx]
    ax.imshow(img)
    sc = ax.scatter(
        spots["x"], spots["y"],
        c=spots[ct], cmap="viridis", s=20, vmin=0, vmax=spots[ct].max()
    )
    ax.set_title(ct, fontsize=8)
    ax.axis("off")

for j in range(idx + 1, n_rows * n_cols):
    axes[j].axis('off')

cbar = fig.colorbar(sc, ax=axes.tolist(), orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label("Abundance", rotation=270, labelpad=15)

fig.suptitle(f"{slide} - Per Cell Type Overlays (No Shift)", y=1.02, fontsize=14)
plt.show()



def check_missing_values(spots_group, label):
    print(f"\n Checking for missing values in {label}...")
    for slide_name in spots_group.keys():
        slide_data = spots_group[slide_name][()]  
        df = pd.DataFrame(slide_data)

        missing = df.isnull().sum().sum()         
        nans = np.isnan(df.values).sum()           

        if missing > 0 or nans > 0:
            print(f" Slide {slide_name}: {missing} missing, {nans} NaN values found!")
        else:
            print(f" Slide {slide_name}: No missing or NaN values.")

with h5py.File(data_path, "r") as f:
    train_spots = f["spots/Train"]
    test_spots = f["spots/Test"]

    check_missing_values(train_spots, "Train Spots")
    check_missing_values(test_spots, "Test Spots")



def plot_slide_with_spots(slide_name, h5_file, zoom_region=None):
    with h5py.File(h5_file, "r") as f:
        image = np.array(f[f"images/Train/{slide_name}"])
        spots = pd.DataFrame(np.array(f[f"spots/Train/{slide_name}"]))
    
    fig, ax = plt.subplots(1, 1 + int(zoom_region is not None), 
                         figsize=(18, 6) if zoom_region else (10, 10))
    
    if zoom_region:
        x1, x2, y1, y2 = zoom_region
        zoom_img = image[y1:y2, x1:x2]
        zoom_spots = spots[(spots['x'].between(x1, x2)) & 
                          (spots['y'].between(y1, y2))]
        
        ax[0].imshow(image)
        ax[0].set_title(f"Full Slide {slide_name}")
        ax[1].imshow(zoom_img)
        ax[1].scatter(zoom_spots["x"]-x1, zoom_spots["y"]-y1, 
                     s=10, c="red", alpha=0.7)
        ax[1].set_title(f"Zoomed Region ({x1}-{x2}, {y1}-{y2})")
    else:
        ax.imshow(image)
        ax.scatter(spots["x"], spots["y"], s=5, c="red", alpha=0.5)
        ax.set_title(f"Slide {slide_name} | Spots: {len(spots)}")
    
    for a in fig.axes:
        a.axis('off')
    plt.tight_layout()
    plt.show()
plot_slide_with_spots(train_slides[0], data_path, 
                     zoom_region=(1000, 1500, 1000, 1500))



cell_data = []
for slide in train_slides:
    with h5py.File(data_path, "r") as f:
        spots = pd.DataFrame(np.array(f[f"spots/Train/{slide}"]))
        cell_data.append(spots.iloc[:, 2:37])  # Columns C1-C35

cell_types = [f"C{i}" for i in range(1, 36)]
all_cell_data = pd.concat(cell_data)

# Plot distribution of mean abundances
plt.figure(figsize=(15, 6))
mean_abundance = all_cell_data.mean().sort_values(ascending=False)
sns.barplot(x=mean_abundance.index, y=mean_abundance.values)
plt.xticks(rotation=90)
plt.title("Mean Cell-Type Abundance Across All Slides")
plt.ylabel("Mean Abundance")
plt.show()


top_cell_types = mean_abundance.index[:5]

with h5py.File(data_path, "r") as f:
    image = np.array(f["images/Train/S_1"])
    spots = pd.DataFrame(np.array(f["spots/Train/S_1"]))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, cell_type in enumerate(top_cell_types):
    sc = axes[i].scatter(
        spots["x"], 
        spots["y"], 
        c=spots[cell_type], 
        cmap="viridis", 
        s=10,
        alpha=0.7
    )
    axes[i].set_title(f"{cell_type} Distribution")
    plt.colorbar(sc, ax=axes[i])
    axes[i].axis('off')

# Remove empty subplot
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()



corr_matrix = all_cell_data.corr()

plt.figure(figsize=(15, 12))
sns.heatmap(
    corr_matrix, 
    cmap="coolwarm", 
    vmin=-0.5, 
    vmax=0.5,
    center=0,
    square=True,
    annot=False,
    cbar_kws={"shrink": 0.8}
)
plt.title("Cell-Type Correlation Matrix Across All Slides")
plt.show()


import os
import time
import random
from pathlib import Path
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import imgaug
import wandb  
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "seed": 42,
    "data_path": "/kaggle/input/el-hackathon-2025",
    "output_dir": "/kaggle/working/",
    "batch_size": 32,
    "num_workers": 6,
    "learning_rate": 0.003,
    "weight_decay": 1e-1,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.1,
    "num_classes": 35,
    "image_size": (162, 162),
    "patch_size": 54,
    "max_epochs": 5,
    "patience": 5,
    "min_delta": 0.001,
    "save_best_only": True,
    "checkpoint_epochs": [],
    "use_wandb": False,
    "model_type": "resnet18",
    "mixed_precision": False,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    imgaug.seed(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def spearman_rank_correlation(x, y):
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    return spearmanr(x, y)[0]

def spearman_corr(preds, targets):
    correlations = []
    for i in range(len(preds)):
        corr = spearman_rank_correlation(preds[i], targets[i])
        if not np.isnan(corr):
            correlations.append(corr)
    return np.mean(correlations)


class HackhathonDataset(Dataset):
    def __init__(self, data_path, transform=None, mode="train"):
        self.data_path = data_path
        self.materials = []
        self.transform = transform
        train_slides = ["S_1","S_2","S_3", "S_4", "S_5"]
        val_slide = ["S_6"]
        test_slide = ["S_7"]
        self.mode = mode
        slide_list = train_slides if mode == "train" else val_slide if mode == "val" else test_slide

        with h5py.File(f"{self.data_path}/elucidata_ai_challenge_data.h5", "r") as h5file:
            images_group = "images/Train" if mode != "test" else "images/Test"
            spots_group = "spots/Train" if mode != "test" else "spots/Test"
            train_images = h5file[images_group]
            train_spots = h5file[spots_group]

            for slide_name in tqdm(slide_list, desc=f"Loading {mode} data"):
                if slide_name in train_images.keys():
                    image = np.array(train_images[slide_name])
                    spots = np.array(train_spots[slide_name])
                    df = pd.DataFrame(spots)
                    self._split_into_patches(image, df, CONFIG["patch_size"])

    def __len__(self):
        return len(self.materials)

    def __getitem__(self, idx):
        image, stats = self.materials[idx]
        if self.transform:
            image = self.transform(image)
        stats = torch.tensor(stats[2:], dtype=torch.float32)
        return image, stats

    def _split_into_patches(self, arr, df, patch_size):
        h, w, c = arr.shape
        for idx in range(len(df)):
            row = df.iloc[idx]
            x, y = int(row["x"]), int(row["y"])
            half_size = patch_size // 2
            y_min = max(y - half_size, 0)
            y_max = min(y + half_size, h)
            x_min = max(x - half_size, 0)
            x_max = min(x + half_size, w)
            patch = arr[y_min:y_max, x_min:x_max, :]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                self.materials.append([patch, row])
            else:
                padded_patch = np.zeros((patch_size, patch_size, c), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                self.materials.append([padded_patch, row])

def get_transforms():
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize(CONFIG["image_size"]),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Resize(CONFIG["image_size"]),
    ])
    return train_transform, val_transform

def get_tta_transforms():
    tta_transforms = [
        T.Compose([T.ToTensor(), T.Resize(CONFIG["image_size"])]),
        T.Compose([T.ToTensor(), T.Resize(CONFIG["image_size"]), T.RandomHorizontalFlip(p=1.0)]),
        T.Compose([T.ToTensor(), T.Resize(CONFIG["image_size"]), T.RandomVerticalFlip(p=1.0)]),
        T.Compose([T.ToTensor(), T.Resize(CONFIG["image_size"]), T.RandomRotation(degrees=(90, 90))]),
        T.Compose([T.ToTensor(), T.Resize(CONFIG["image_size"]), T.RandomRotation(degrees=(180, 180))]),
        T.Compose([T.ToTensor(), T.Resize(CONFIG["image_size"]), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]),
    ]
    return tta_transforms


class DifferentiableSpearmanLoss(nn.Module):
    def __init__(self, regularization_strength=1.0):
        super().__init__()
        self.regularization_strength = regularization_strength

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()
        pred_rank = self._soft_rank(y_pred)
        true_rank = self._soft_rank(y_true)
        pred_rank = F.normalize(pred_rank, dim=1)
        true_rank = F.normalize(true_rank, dim=1)
        spearman = torch.sum(pred_rank * true_rank, dim=1)
        return 1 - spearman.mean()

    def _soft_rank(self, x, regularization_strength=None):
        if regularization_strength is None:
            regularization_strength = self.regularization_strength
        x = x.unsqueeze(-1)
        diff = x - x.transpose(-1, -2)
        P = torch.sigmoid(-regularization_strength * diff)
        ranks = P.sum(dim=-1)
        return ranks

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, regularization_strength=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.spearman = DifferentiableSpearmanLoss(regularization_strength)
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        l1_loss = self.l1(y_pred, y_true)
        spearman_loss = self.spearman(y_pred, y_true)
        return l1_loss + self.alpha * spearman_loss

def create_model(model_type, num_classes):
    if model_type == "resnet18":
        MODEL_PATH = '/kaggle/input/ckpt-file/tenpercent_resnet18.ckpt'
        RETURN_PREACTIVATION = False
        NUM_CLASSES = 35

        def load_model_weights(model, weights):
            model_dict = model.state_dict()
            weights = {k: v for k, v in weights.items() if k in model_dict}
            if weights == {}:
                print('No weight could be loaded..')
            model_dict.update(weights)
            model.load_state_dict(model_dict)
            return model

        model = models.resnet18(pretrained=False)
        state = torch.load(MODEL_PATH, map_location='cuda:0', weights_only=False)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        model = load_model_weights(model, state_dict)

        if RETURN_PREACTIVATION:
            model.fc = torch.nn.Sequential()
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_type == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "swin":
        model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_type == "efficientnet_b3":
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "convnext_large":
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, scaler=None):
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []
    progress_bar = tqdm(dataloader, desc="Training")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = epoch_loss / len(dataloader)
    spearman_score = spearman_corr(all_preds, all_labels)
    return avg_loss, spearman_score, all_preds, all_labels

def validate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(dataloader)
    spearman_score = spearman_corr(all_preds, all_labels)
    return avg_loss, spearman_score, all_preds, all_labels

def predict_test_set_with_tta(model, data_path, device):
    tta_transforms = get_tta_transforms()
    with h5py.File(f"{data_path}/elucidata_ai_challenge_data.h5", "r") as f:
        test_spots = f["spots/Test"]
        test_images = f["images/Test"]
        sample = 'S_7'
        image = np.array(test_images[sample])
        spots = np.array(test_spots[sample])
        x, y = spots["x"], spots["y"]
        outputs = []

        with torch.inference_mode():
            model.eval()
            patch_size = CONFIG["patch_size"]
            for x_, y_ in tqdm(zip(x, y), desc="Generating predictions with TTA", total=len(x)):
                half_size = patch_size // 2
                y_min = max(y_ - half_size, 0)
                y_max = min(y_ + half_size, image.shape[0])
                x_min = max(x_ - half_size, 0)
                x_max = min(x_ + half_size, image.shape[1])
                patch = image[y_min:y_max, x_min:x_max, :]

                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded_patch

                patch_predictions = []
                for transform in tta_transforms:
                    patch_tensor = transform(patch)
                    patch_tensor = patch_tensor.to(device)
                    with torch.no_grad():
                        output = model(patch_tensor.unsqueeze(0)).cpu().numpy()
                        patch_predictions.append(output[0])

                avg_prediction = np.mean(patch_predictions, axis=0)
                outputs.append(avg_prediction)

    return np.array(outputs), x, y


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)

def save_submission(predictions, data_path, epoch, model_name):
    example_df = pd.read_csv(f"/kaggle/input/sample-submission/submission (1).csv")
    ID = example_df["ID"]
    output_df = pd.DataFrame(predictions)
    submission_df = pd.concat([ID, output_df], axis=1)
    submission_df.columns = example_df.columns
    output_file = "submission.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")
    return output_file


'''
def main():
    set_seed(CONFIG["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms()
    train_dataset = HackhathonDataset(CONFIG["data_path"], transform=train_transform, mode="train")
    val_dataset = HackhathonDataset(CONFIG["data_path"], transform=val_transform, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    model = create_model(CONFIG["model_type"], CONFIG["num_classes"])
    model = model.to(device)
    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG["scheduler_gamma"],
        patience=3,
    )

    scaler = torch.cuda.amp.GradScaler() if CONFIG["mixed_precision"] and device.type == "cuda" else None

    if CONFIG["use_wandb"]:
        wandb.init(
            project="hackathon-gene-expression",
            config=CONFIG,
            name=f"{CONFIG['model_type']}_run"
        )
        wandb.watch(model)

    best_val_spearman = -1
    no_improvement_count = 0
    best_model_path = f"{CONFIG['output_dir']}/best_{CONFIG['model_type']}_model.pt"

    for epoch in range(CONFIG["max_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['max_epochs']}")

        train_loss, train_spearman, _, _ = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler
        )

        val_loss, val_spearman, _, _ = validate(
            model, val_loader, loss_fn, device
        )

        scheduler.step(val_loss)

        metrics = {
            "train_loss": train_loss,
            "train_spearman": train_spearman,
            "val_loss": val_loss,
            "val_spearman": val_spearman,
            "learning_rate": optimizer.param_groups[0]['lr']
        }

        print(f"Train Loss: {train_loss:.4f}, Train Spearman: {train_spearman:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Spearman: {val_spearman:.4f}")

        if CONFIG["use_wandb"]:
            wandb.log(metrics)

        improved = val_spearman > best_val_spearman + CONFIG["min_delta"]

        if improved:
            best_val_spearman = val_spearman
            no_improvement_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_spearman': val_spearman,
            }, best_model_path)

            print(f"Saved best model with validation Spearman: {val_spearman:.4f}")
            test_preds, x, y = predict_test_set_with_tta(model, CONFIG["data_path"], device)
            submission_file = save_submission(
                test_preds, CONFIG["data_path"], f"best_epoch_{epoch}", CONFIG["model_type"]
            )

            if CONFIG["use_wandb"]:
                wandb.save(submission_file)
        else:
            no_improvement_count += 1

        if epoch in CONFIG["checkpoint_epochs"]:
            test_preds, _, _ = predict_test_set_with_tta(model, CONFIG["data_path"], device)
            submission_file = save_submission(
                test_preds, CONFIG["data_path"], epoch, CONFIG["model_type"]
            )

            if CONFIG["use_wandb"]:
                wandb.save(submission_file)

        if no_improvement_count >= CONFIG["patience"]:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    checkpoint = torch.load(best_model_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation Spearman: {checkpoint['val_spearman']:.4f}")

    final_preds, _, _ = predict_test_set_with_tta(model, CONFIG["data_path"], device)
    final_submission = save_submission(
        final_preds, CONFIG["data_path"], "final", CONFIG["model_type"]
    )

    if CONFIG["use_wandb"]:
        wandb.save(final_submission)
        wandb.finish()

    print("Training completed!")

main()
'''


