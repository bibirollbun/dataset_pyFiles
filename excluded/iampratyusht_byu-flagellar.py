# Standard libraries
import os
import json
import random
import threading
import time
from pathlib import Path
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

# Numerical and data handling
import numpy as np
import pandas as pd
import yaml

# Image handling and visualization
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# PyTorch and torchvision
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

# Progress bar
from tqdm.notebook import tqdm


# !pip install ultralytics


!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


# YOLO model (Ultralytics)
from ultralytics import YOLO


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


root_dir = Path("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025")


# yaml_dir = "/kaggle/working/yaml_dir"
# os.makedirs(yaml_dir, exist_ok=True)


class Visualise:
    def __init__(self, path, transform=None, box_width=24):
        self.path = Path(path)
        self.transform = transform
        self.box_width = box_width

    def random_tomosplits(self, n):
        image_path = list(self.path.glob("*/*.jpg"))
        if n > len(image_path):
            raise ValueError(f"Requested {n} samples but only {len(image_path)} found.")
        image_list = random.sample(image_path, n)

        rows = (n + 4) // 5
        fig, axes = plt.subplots(rows, 5, figsize=(16, rows * 4))
        axes = axes.flatten()

        for i, img_path in enumerate(image_list):
            image = Image.open(img_path)
            img_size = image.size
            axes[i].imshow(image)
            axes[i].axis("off")

            tomo_id = img_path.parent.name.split('_')[1]
            slice_id = img_path.stem.split('_')[1]
            title = f"tomo_id : {tomo_id}\nslice : {slice_id}\nsize : {img_size}"
            axes[i].set_title(title, fontsize=12)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def display_transform(self, n):
        image_path = list(self.path.glob("*/*.jpg"))
        if n > len(image_path):
            raise ValueError(f"Requested {n} samples but only {len(image_path)} found.")
        image_list = random.sample(image_path, n)

        fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
        for i, img_path in enumerate(image_list):
            image = Image.open(img_path)
            axes[i][0].imshow(image)
            axes[i][0].axis("off")
            axes[i][0].set_title(f"Original\nsize: {image.size}", fontsize=12)

            if self.transform is not None:
                t_image = self.transform(image)
                if isinstance(t_image, torch.Tensor):
                    t_image = t_image.permute(1, 2, 0).numpy()
                axes[i][1].imshow(t_image)
                axes[i][1].axis("off")
                axes[i][1].set_title(f"Transformed\nsize: {t_image.size}", fontsize=12)
            else:
                axes[i][1].imshow(image)
                axes[i][1].axis("off")
                axes[i][1].set_title("No Transform", fontsize=12)

        plt.tight_layout()
        plt.show()

    def display_slices(self, n):
        tomo_dirs = [p for p in self.path.iterdir() if p.is_dir() and p.name.startswith("tomo_")]
        if not tomo_dirs:
            raise ValueError("No tomo_* directories found.")
        tomo_path = random.choice(tomo_dirs)
        image_list = sorted(tomo_path.glob("*.jpg"))[:n]
        if not image_list:
            raise ValueError(f"No .jpg files found in {tomo_path}")

        rows = (len(image_list) + 4) // 5
        fig, axes = plt.subplots(rows, 5, figsize=(16, rows * 4))
        axes = axes.flatten()

        for i, img_path in enumerate(image_list):
            image = Image.open(img_path)
            axes[i].imshow(image)
            axes[i].axis("off")
            axes[i].set_title(f"slice shape: {image.size}")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Random tomo: {tomo_path.name}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_with_bounding_boxes(self, n, label_path):
        df = pd.read_csv(label_path)
        
        tomos  = df[df["Motor axis 0"]!=-1]
        sampled_tomos = tomos["tomo_id"].drop_duplicates().sample(n=n-1, random_state=42)
        sampled_df = df[df["tomo_id"].isin(sampled_tomos)]
        
        rows = int(np.ceil(n / 2))
        cols = min(n, 2)
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (_, motor) in enumerate(sampled_df.iterrows()):
            z = int(motor["Motor axis 0"])
            if z == -1:
                continue
        
            img_path = Path(self.path) / motor["tomo_id"] / f"slice_{z:04d}.jpg"
            # if not img_path.exists():
            #     continue
        
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            img_rgb = image.convert('RGB')
        
            img_width, img_height = img_rgb.size
            overlay = Image.new('RGBA', img_rgb.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
        
            x_center = motor["Motor axis 2"]
            y_center = motor["Motor axis 1"]
            width = height = self.box_width  # Default box size if not defined
        
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(img_width, int(x_center + width / 2))
            y2 = min(img_height, int(y_center + height / 2))
        
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 64), outline=(255, 0, 0, 200))
            draw.text((x1, max(0, y1 - 10)), "Class 0", fill=(255, 0, 0, 255))
        
            img_rgb = Image.alpha_composite(img_rgb.convert('RGBA'), overlay).convert('RGB')
        
            axes[i].imshow(np.array(img_rgb))
            img_name = motor["tomo_id"]+"_"+os.path.basename(img_path)
            axes[i].set_title(f"Image: {img_name}")
            axes[i].axis('on')
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()


class NormalizeByPercentile:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        clipped = np.clip(img, p2, p98)
        normalized = 255 * (clipped - p2) / (p98 - p2 + 1e-5)
        return Image.fromarray(np.uint8(normalized))
data_transform = transforms.Compose([
    NormalizeByPercentile(),
])


# visualizer = Visualise(path=Path(root_dir/"train"), transform=data_transform)


# visualizer.plot_with_bounding_boxes(9,"/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv" )


# visualizer.random_tomosplits(n=10)


# visualizer.display_transform(n=5)


# visualizer.display_slices(n=5)


class BYUCustomDatasetPreparer:
    def __init__(self, root, yaml_dir, transform=None, target_transform=True, window_size=24, split_ratio=0.8, trust=2, mode="train", neg_include=False):
        self.root = Path(root)
        self.yaml_dir = Path(yaml_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.window_size = window_size
        self.mode = mode
        self.split_ratio = split_ratio
        self.trust = trust
        self.data = []
        self.neg_include = neg_include

        self.image_dir = self.root / mode
        self.paths = sorted(list(self.image_dir.glob("*/*.jpg")))

        if self.mode == "train":
            label_path = self.root / "train_labels.csv"
            self.labels = pd.read_csv(label_path)
            self.tomos = self.labels["tomo_id"].tolist()
            self.unique_tomos = list(set(self.tomos))
            print(f"No. of total motors present in dataset is {len(self.tomos)} out of {len(self.unique_tomos)} tomograms")
            self._init_folders()
        else:
            self._init_folders()

    def _init_folders(self):
        if self.mode == "train":
            self.yolo_images_train = self.yaml_dir / "images/train"
            self.yolo_images_val = self.yaml_dir / "images/val"
            self.yolo_labels_train = self.yaml_dir / "labels/train"
            self.yolo_labels_val = self.yaml_dir / "labels/val"
            paths = [self.yolo_images_train, self.yolo_images_val, self.yolo_labels_train, self.yolo_labels_val]
        else:
            self.yolo_images_test = self.yaml_dir / "images/test"
            self.yolo_labels_test = self.yaml_dir / "labels/test"
            paths = [self.yolo_images_test, self.yolo_labels_test]
        for p in paths:
            os.makedirs(p, exist_ok=True)

    def _split_train_val(self):
        train_tomos = random.sample(self.unique_tomos, int(self.split_ratio * len(self.unique_tomos)))
        val_tomos = [t for t in self.unique_tomos if t not in train_tomos]
        return train_tomos, val_tomos

    def _extract_unique_tomos(self):
        return list(set(self.tomos))

    def _process_split(self, tomo_list, images_dir, labels_dir):
        total_motor_count = 0
        total_image_count = 0
        total_unprocessed_tomos = 0
        desc = f"preparing training set" if self.mode == "train" else f"preparing validating set"
        for tomo in tqdm(tomo_list, desc=desc):
            motors = self.labels[self.labels["tomo_id"] == tomo]
            total_slices = int(motors["Array shape (axis 0)"].iloc[0])
            present_slices = motors["Motor axis 0"].dropna().astype(int).tolist()
            present_slices_set = set(present_slices)
    
            if self.neg_include:
                all_slices = list(range(total_slices))
                negative_slices = [i for i in all_slices if i not in present_slices_set]
                for i in negative_slices:
                    label = {"label_id": 1, "x": None, "y": None, "w": None, "h": None}  # dummy
                    self._save_image_and_label(tomo, i, label, images_dir, labels_dir)
                total_image_count += len(negative_slices)
    
            m_count, i_count, u_count = self._extract_labels(tomo, motors, total_slices, images_dir, labels_dir)
            total_motor_count += m_count
            total_image_count += i_count
            total_unprocessed_tomos += u_count if not self.neg_include else 0

    
        print(f"\nðŸ“Š [Summary] Under {desc} Motors: {total_motor_count}, Images: {total_image_count}, Unprocessed Motors: {total_unprocessed_tomos}\n")

    def _extract_labels(self, tomo, motors, total_slices, images_dir, labels_dir):
        motor_count = 0
        image_count = 0
        unprocessod_tomos = 0
        for _, motor in motors.iterrows():
            slice_idx = motor["Motor axis 0"]
            if int(motor["Motor axis 0"]) == -1:
                unprocessod_tomos += 1
                continue
            motor_count += 1
            slice_idx = int(slice_idx)
            label = {"label_id": 0}
            if self.target_transform:
                try:
                    label["y_orig"] = motor["Motor axis 1"]
                    label["x_orig"] = motor["Motor axis 2"]
                    label["y"] = motor["Motor axis 1"] / motor["Array shape (axis 1)"]
                    label["x"] = motor["Motor axis 2"] / motor["Array shape (axis 2)"]
                    label["h"] = self.window_size / motor["Array shape (axis 1)"]
                    label["w"] = self.window_size / motor["Array shape (axis 2)"]
                except:
                    continue
            else:
                label["y"] = motor["Motor axis 1"]
                label["x"] = motor["Motor axis 2"]
                label["h"] = label["w"] = self.window_size

            z_min = max(0, slice_idx - self.trust)
            z_max = min(total_slices - 1, slice_idx + self.trust)
            for z in range(z_min, z_max + 1):
                self._save_image_and_label(tomo, z, label, images_dir, labels_dir)
                image_count += 1
        return motor_count, image_count, unprocessod_tomos

    def _save_image_and_label(self, tomo, z, label, images_dir, labels_dir):
        src_path = self.root / self.mode / tomo / f"slice_{z:04d}.jpg"
        if not src_path.exists():
            return
        image = Image.open(src_path)
        if self.transform:
            image = self.transform(image)
        img_name = f"{tomo}_z{z:04d}_y{int(label['y_orig']):04d}_x{int(label['x_orig']):04d}.jpg"
        img_path = images_dir / img_name
        image.save(img_path)
        label_path = labels_dir / img_name.replace(".jpg", ".txt")
        with open(label_path, "w") as f:
            f.write(f"{int(label['label_id'])} {label['x']} {label['y']} {label['w']} {label['h']}")

    def prepare(self):
        if self.mode == "train":
            train_tomos, val_tomos = self._split_train_val()
            print(f"ðŸŸ¢ Train tomos: {len(train_tomos)}, ðŸ”µ Val tomos: {len(val_tomos)}")
            self._process_split(train_tomos, self.yolo_images_train, self.yolo_labels_train)
            self._process_split(val_tomos, self.yolo_images_val, self.yolo_labels_val)
        else:
            unique_tomos = self._extract_unique_tomos()
            print(f"ðŸŸ  Test tomos: {len(unique_tomos)}")
            self._process_split(unique_tomos, self.yolo_images_test, self.yolo_labels_test)

    def create_yaml(self):
        if self.neg_include:
            yaml_content = {
                'path': str(self.yaml_dir),
                'train': 'images/train',
                'val': 'images/val',
                'names': {0: 'motor', 1: 'no_motor'}
            }
        else:
            yaml_content = {
                'path': str(self.yaml_dir),
                'train': 'images/train',
                'val': 'images/val',
                'names': {0: 'motor'}
            }
    
        with open(self.yaml_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print("âœ… dataset.yaml created.")



# data = BYUCustomDatasetPreparer(root_dir, yaml_dir, transform = data_transform, trust = 4)


# data.prepare()
# data.create_yaml()


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S, self.B, self.C = S, B, C
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * (S // 4) * (S // 4), 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


class YOLOModelLoader:
    def __init__(self, model_version: str, num_classes: int, model_path: str = None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_version = model_version.lower()
        self.num_classes = num_classes
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        if self.model_version == 'yolov1':
            return YOLOv1(S=7, B=2, C=self.num_classes).to(self.device)

        model_source = self.model_path 
        if self.model_version.startswith('yolov') or self.model_version.startswith('yolo'):
            return YOLO(Path(model_source+f"/{self.model_version}.pt"))

        elif self.model_version.startswith("rtdetr"):
            return RTDETR(Path(model_source+f"/{self.model_version}.pt"))

        else:
            raise ValueError(f"Unsupported model version: {self.model_version}")

    def get_model(self):
        return self.model



# loader = YOLOModelLoader(model_version='yolov8n', num_classes=1, model_path='/kaggle/input/yolo/pytorch/default/1')
# model = loader.get_model()


class evaluate_data(Dataset):
    def __init__(self, root, evaluate_dir, transform=None, window_size=24, mode="evaluate"):
        self.root = Path(root)
        self.transform = transform
        self.mode = mode
        self.window_size = window_size
        self.evaluate_dir= evaluate_dir

        if self.mode == "evaluate":
            self.image_dir = Path(self.root / "train")
            label_path = self.root / "train_labels.csv"
            self.labels = pd.read_csv(label_path)
            self.paths = []
            for _, motor in self.labels.iterrows():
                if motor["Motor axis 0"] == -1:
                    continue
                self.paths.append(self.image_dir / motor["tomo_id"] / f'slice_{int(motor["Motor axis 0"]):04d}.jpg')
        else:
            self.image_dir = Path(self.root / "test")
            self.paths = sorted(list(self.image_dir.glob("*/*.jpg")))

        self.slice_info = [
            {
                "path": p,
                "tomo_id": p.parent.name,
                "slice_idx": int(p.stem.split("_")[1])
            }
            for p in self.paths
        ]

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        file = self.slice_info[idx]
        src_image = Image.open(file["path"])
        image = self.transform(src_image) if self.transform else src_image

        tomo = file["tomo_id"]
        slice_idx = file["slice_idx"]
        dest_path = os.path.join(self.evaluate_dir, tomo, os.path.basename(file["path"]))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        image.save(dest_path)

        if self.mode == "evaluate":
            slice_matches = self.labels[(self.labels["tomo_id"] == tomo) & (self.labels["Motor axis 0"] == slice_idx)]
            row = slice_matches.iloc[0]
            y = row["Motor axis 1"]
            x = row["Motor axis 2"]
            h = w =  self.window_size 
            return {
                "tomo": tomo,
                "slice": slice_idx,
                "image": image,
                "path": dest_path,
                "label": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                }
            }
        return {
                "tomo": tomo,
                "slice": slice_idx,
                "image": image,
                "path": dest_path}


class Evaluate:
    def __init__(self, dataset):
        self.data = dataset

    def plot_loss_curve(self, run_dir):
        results_csv = os.path.join(run_dir, 'results.csv')  
        if not os.path.exists(results_csv):
            print(f"Results file not found at {results_csv}")
            return
        
        results_df = pd.read_csv(results_csv)
        train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
        val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
        
        if not train_dfl_col or not val_dfl_col:
            print("DFL loss columns not found in results CSV")
            print(f"Available columns: {results_df.columns.tolist()}")
            return
        
        train_dfl_col = train_dfl_col[0]
        val_dfl_col = val_dfl_col[0]
        
        best_epoch = results_df[val_dfl_col].idxmin()
        best_val_loss = results_df.loc[best_epoch, val_dfl_col]
        
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
        plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
        plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                    label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
        
        plt.xlabel('Epoch')
        plt.ylabel('DFL Loss')
        plt.title('Training and Validation DFL Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        plt.close()
        return best_epoch, best_val_loss

    def predict_on_samples(self, model, num_samples=4):
        num_samples = min(num_samples, len(self.data))
        indices = random.sample(range(len(self.data)), num_samples)
        samples = [self.data[i] for i in indices]
    
        cols = 2
        rows = int(np.ceil(num_samples / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten() if rows > 1 else np.array(axes).reshape(-1)
    
        for i, file in enumerate(samples):
            img_path = file["path"]
            results = model.predict(img_path, conf=0.25)[0]
            img = Image.open(img_path)
            axes[i].imshow(np.array(img), cmap='gray')
    
            
            x_gt = file["label"]["x"]
            y_gt = file["label"]["y"]
            w_gt = file["label"]["w"]
            h_gt = file["label"]["h"]
            rect_gt = Rectangle((x_gt - w_gt / 2, y_gt - h_gt / 2),
                                w_gt, h_gt, linewidth=1, edgecolor='g', facecolor='none')
            axes[i].add_patch(rect_gt)
    
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    rect_pred = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          linewidth=1, edgecolor='r', facecolor='none')
                    axes[i].add_patch(rect_pred)
                    axes[i].text(x1, y1 - 5, f'{conf:.2f}', color='red')
    
            axes[i].set_title(f"Ground Truth (green) vs Prediction (red)")
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()



def train_yolo_model(yaml_path,yolo_weights_dir, model, epochs=30, batch_size=16, img_size=640):
    print(f"Loading model")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=yolo_weights_dir,
        name='motor_detector',
        exist_ok=True,
        patience=5,              # Early stopping if no improvement for 5 epochs
        save_period=5,           # Save checkpoints every 5 epochs
        val=True,                # Ensure validation is performed
        verbose=True             # Show detailed output during training
    )
    
    # Get the path to the run directory
    run_dir = os.path.join(yolo_weights_dir, 'motor_detector')    
    best_epoch_info = evaluate.plot_loss_curve(run_dir)
    
    if best_epoch_info:
        best_epoch, best_val_loss = best_epoch_info
        print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
    
    return model, results



# yolo_weights_dir = "/kaggle/working/yolo_weights"


# print("Starting YOLO training process...")
# yaml_path = Path("/kaggle/working/yaml_dir/dataset.yaml")
# print(f"Using YAML file: {yaml_path}")

# evaluate_dir = "/kaggle/working/evaluate"
# os.makedirs(evaluate_dir, exist_ok=True)
# data = evaluate_data(root_dir, evaluate_dir, transform=data_transform)
# evaluate = Evaluate(data)


# print("\nStarting YOLO training...")
# model, results = train_yolo_model(
#     yaml_path=yaml_path,
#     yolo_weights_dir=yolo_weights_dir,
#     model=model,
#     epochs=30  # Using 30 epochs instead of 100 for faster training
# )

# print("\nTraining complete!")




# data = evaluate_data(root_dir, evaluate_dir, transform=data_transform)
# evaluate = Evaluate(data)
# print("\nRunning predictions on sample images...")
# evaluate.predict_on_samples(model, num_samples=8)


np.random.seed(42)
torch.manual_seed(42)

data_path = root_dir
test_dir = os.path.join(data_path, "test")
submission_path = "/kaggle/working/submission.csv"

model_path = "/kaggle/input/v2/pytorch/default/1/best (2).pt"

CONFIDENCE_THRESHOLD = 0.45
MAX_DETECTIONS_PER_TOMO = 3
NMS_IOU_THRESHOLD = 0.2
CONCENTRATION = 1

class GPUProfiler:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        print(f"[PROFILE] {self.name}: {elapsed:.3f}s")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8

if device.startswith('cuda'):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")

    free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
    BATCH_SIZE = max(8, min(32, int(free_mem * 4)))
    print(f"Dynamic batch size set to {BATCH_SIZE} based on {free_mem:.2f}GB free memory")
else:
    print("GPU not available, using CPU")
    BATCH_SIZE = 4




class Test:
    def __init__(self, model, device, test_dir, batch_size, confidence_threshold,
             concentration, submission_path, nms_iou_threshold, GPUProfiler=None,
             MAX_DETECTIONS_PER_TOMO=1):
        self.MAX_DETECTIONS_PER_TOMO = MAX_DETECTIONS_PER_TOMO
        self.model = YOLO(model)
        self.device = device
        self.test_dir = test_dir 
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.concentration = concentration
        self.submission_path = submission_path
        self.nms_iou_threshold = nms_iou_threshold
        self.GPUProfiler = GPUProfiler

        self.model.to(self.device)
        if self.device.startswith('cuda'):
            self.model.fuse()
            if torch.cuda.get_device_capability(0)[0] >= 7:
                self.model.model.half()

    def preload_image_batch(self, file_paths):
        images = []
        for path in file_paths:
            img = cv2.imread(path)
            if img is None:
                img = np.array(Image.open(path))
            images.append(img)
        return images

    def process_tomogram(self, tomo_id, index=0, total=1):
        print(f"Processing tomogram {tomo_id} ({index}/{total})")
        tomo_dir = os.path.join(self.test_dir, tomo_id)
        slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
        selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * self.concentration))
        selected_indices = np.round(selected_indices).astype(int)
        slice_files = [slice_files[i] for i in selected_indices]

        print(f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices based on CONCENTRATION={self.concentration}")

        all_detections = []
        streams = [torch.cuda.Stream() for _ in range(min(4, self.batch_size))] if self.device.startswith('cuda') else [None]
        next_batch_thread = None
        next_batch_images = None

        for batch_start in range(0, len(slice_files), self.batch_size):
            if next_batch_thread is not None:
                next_batch_thread.join()
                next_batch_images = None

            batch_end = min(batch_start + self.batch_size, len(slice_files))
            batch_files = slice_files[batch_start:batch_end]

            next_batch_start = batch_end
            next_batch_end = min(next_batch_start + self.batch_size, len(slice_files))
            next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []

            if next_batch_files:
                next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
                next_batch_thread = threading.Thread(target=self.preload_image_batch, args=(next_batch_paths,))
                next_batch_thread.start()
            else:
                next_batch_thread = None

            sub_batches = np.array_split(batch_files, len(streams))
            sub_batch_results = []

            for i, sub_batch in enumerate(sub_batches):
                if len(sub_batch) == 0:
                    continue
                stream = streams[i % len(streams)]
                with torch.cuda.stream(stream) if stream and self.device.startswith('cuda') else nullcontext():
                    sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                    sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]

                    if self.GPUProfiler:
                        with self.GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                            sub_results = self.model(sub_batch_paths, verbose=False)
                    else:
                        sub_results = self.model(sub_batch_paths, verbose=False)

                    for j, result in enumerate(sub_results):
                        if len(result.boxes) > 0:
                            boxes = result.boxes
                            for box_idx, confidence in enumerate(boxes.conf):
                                if confidence >= self.confidence_threshold:
                                    x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu().numpy()
                                    x_center = (x1 + x2) / 2
                                    y_center = (y1 + y2) / 2
                                    all_detections.append({
                                        'z': round(sub_batch_slice_nums[j]),
                                        'y': round(y_center),
                                        'x': round(x_center),
                                        'confidence': float(confidence)
                                    })

            if self.device.startswith('cuda'):
                torch.cuda.synchronize()

        if next_batch_thread is not None:
            next_batch_thread.join()

        final_detections = self.perform_3d_nms(all_detections, self.nms_iou_threshold)
        final_detections.sort(key=lambda x: x['confidence'], reverse=True)

        if not final_detections:
            return [{
                'tomo_id': tomo_id,
                'Motor axis 0': -1,
                'Motor axis 1': -1,
                'Motor axis 2': -1
            }]

        top_detections = final_detections[:self.MAX_DETECTIONS_PER_TOMO]
        return [{
            'tomo_id': tomo_id,
            'Motor axis 0': round(det['z']),
            'Motor axis 1': round(det['y']),
            'Motor axis 2': round(det['x'])
        } for det in top_detections]


    def perform_3d_nms(self, detections, iou_threshold):
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        final_detections = []

        def distance_3d(d1, d2):
            return np.sqrt((d1['z'] - d2['z'])**2 + (d1['y'] - d2['y'])**2 + (d1['x'] - d2['x'])**2)

        box_size = 24
        distance_threshold = box_size * iou_threshold

        while detections:
            best_detection = detections.pop(0)
            final_detections.append(best_detection)
            detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]

        return final_detections

    def generate_submission(self):
        test_tomos = sorted([d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))])
        total_tomos = len(test_tomos)
        print(f"Found {total_tomos} tomograms in test directory")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = []
        motors_found = 0

        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_tomo = {}
            for i, tomo_id in enumerate(test_tomos, 1):
                future = executor.submit(self.process_tomogram, tomo_id, i, total_tomos)
                future_to_tomo[future] = tomo_id

            for future in future_to_tomo:
                tomo_id = future_to_tomo[future]
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    result_list = future.result()
                    results.extend(result_list)

                    motors_in_tomo = [r for r in result_list if r['Motor axis 0'] != -1]
                    if motors_in_tomo:
                        motors_found += len(motors_in_tomo)
                        for r in motors_in_tomo:
                            print(f"Motor found in {r['tomo_id']} at position: "
                                  f"z={r['Motor axis 0']}, y={r['Motor axis 1']}, x={r['Motor axis 2']}")
                    else:
                        print(f"No motor detected in {tomo_id}")
                    
                    print(f"Current detection count: {motors_found} motors in {len(results)} entries "
                          f"({motors_found / len(results) * 100:.1f}%)")

                except Exception as e:
                    print(f"Error processing {tomo_id}: {e}")
                    results.append({
                        'tomo_id': tomo_id,
                        'Motor axis 0': -1,
                        'Motor axis 1': -1,
                        'Motor axis 2': -1
                    })

        submission_df = pd.DataFrame(results)
        submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
        submission_df.to_csv(self.submission_path, index=False)

        print(f"\nSubmission complete!")
        print(f"Motors detected: {motors_found}/{total_tomos} ({motors_found/total_tomos*100:.1f}%)")
        print(f"Submission saved to: {self.submission_path}")
        print("\nSubmission preview:")
        print(submission_df.head())

        return submission_df



test = Test(
    model=model_path,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    test_dir=test_dir,
    batch_size=BATCH_SIZE,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    concentration=CONCENTRATION,
    submission_path='submission.csv',
    nms_iou_threshold=NMS_IOU_THRESHOLD,
    GPUProfiler=GPUProfiler,
    MAX_DETECTIONS_PER_TOMO=MAX_DETECTIONS_PER_TOMO
)



if __name__ == "__main__":
    # Time entire process
    start_time = time.time()
    
    # Generate submission
    submission = test.generate_submission()
    
    # Print total execution time
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")




