import os

dataset_path = "/kaggle/input/alaska2-image-steganalysis"

cover_path = os.path.join(dataset_path, "Cover")
jmp_path = os.path.join(dataset_path, "JMiPOD")
jnw_path = os.path.join(dataset_path, "JUNIWARD")
uerd_path = os.path.join(dataset_path, "UERD")
test_path = os.path.join(dataset_path, "Test")

num_cover = len(os.listdir(cover_path))
num_jmp = len(os.listdir(jmp_path))
num_jnw = len(os.listdir(jnw_path))
num_uerd = len(os.listdir(uerd_path))
num_test = len(os.listdir(test_path))

print(f"Cover images: {num_cover}")
print(f"JMiPOD images: {num_jmp}")
print(f"JUNIWARD images: {num_jnw}")
print(f"UERD images: {num_uerd}")
print(f"Test images: {num_test}")


import matplotlib.pyplot as plt
import cv2
import random

# Function to display images
def show_images(image_folder, num_samples=5):
    image_files = sorted(os.listdir(image_folder))[:num_samples]  # Get first 5 images

    plt.figure(figsize=(15, 5))
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)  # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(img_name)
        plt.axis("off")

    plt.show()

# Display cover images
print("Cover Images:")
show_images(cover_path)

# Display stego images
print("Stego Images:")
show_images(jmp_path)
show_images(jnw_path)
show_images(uerd_path)

# Display test images
print("Test Images:")
show_images(test_path)


import numpy as np
from PIL import Image

# Load a sample image
sample_image_path = os.path.join(cover_path, os.listdir(cover_path)[0])
image = Image.open(sample_image_path)

# Image properties
print(f"Image format: {image.format}")
print(f"Image size (width x height): {image.size}")
print(f"Color mode: {image.mode}")


def plot_histogram(image_path, title):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    plt.hist(img.ravel(), bins=256, color='blue', alpha=0.7, label='Pixel Values')
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Choose a sample image from each category
cover_sample = os.path.join(cover_path, os.listdir(cover_path)[0])
jmp_sample = os.path.join(jmp_path, os.listdir(jmp_path)[0])
jnw_sample = os.path.join(jnw_path, os.listdir(jnw_path)[0])
uerd_sample = os.path.join(uerd_path, os.listdir(uerd_path)[0])

plot_histogram(cover_sample, "Histogram of Cover Image")
plot_histogram(jmp_sample, "Histogram of JMiPOD Image")
plot_histogram(jnw_sample, "Histogram of JUNIWARD Image")
plot_histogram(uerd_sample, "Histogram of UERD Image")



from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.stats import entropy
import numpy as np

# Feature extraction function
def extract_features(image_path):
    img = imread(image_path)
    gray_img = rgb2gray(img)

    # Mean and Standard Deviation
    mean_intensity = np.mean(gray_img)
    std_intensity = np.std(gray_img)

    # Entropy
    hist, _ = np.histogram(gray_img.ravel(), bins=256, density=True)
    img_entropy = entropy(hist)

    return mean_intensity, std_intensity, img_entropy

# Extract features for cover and stego images
cover_features = extract_features(cover_sample)
jmp_features = extract_features(jmp_sample)
jnw_features = extract_features(jnw_sample)
uerd_features = extract_features(uerd_sample)

print("Cover Image Features (Mean, Std, Entropy):", cover_features)
print("JMiPOD Image Features (Mean, Std, Entropy):", jmp_features)
print("JUNIWARD Image Features (Mean, Std, Entropy):", jnw_features)
print("UERD Image Features (Mean, Std, Entropy):", uerd_features)


!pip install torch torchvision efficientnet-pytorch numpy pandas scikit-learn opencv-python tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn import metrics
import os
import glob
import random
import time
from datetime import datetime
import datetime
import warnings

warnings.filterwarnings("ignore")


class TrainGlobalConfig:
    # Removed csv_file reference since we'll generate folds
    fold_number = 0
    num_workers = 4
    batch_size = 16
    n_epochs = 10
    lr = 2e-4
    seed = 42

    verbose = True
    verbose_step = 1

    step_scheduler = True
    valid_scheduler = False

    SchedulerClass = OneCycleLR
    scheduler_params = dict(
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=None,
        pct_start=0.1,
        anneal_strategy="cos",
        cycle_momentum=True,
        div_factor=10.0,
    )
    def __init__(self):
        # Calculate steps_per_epoch after dataloader is created
        self.steps_per_epoch = None

    @property
    def scheduler_params(self):
        params = dict(
            max_lr=self.lr,
            epochs=self.n_epochs,
            steps_per_epoch=self.steps_per_epoch,  # Will be set later
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=True,
            div_factor=10.0,
        )
        return params


class BalanceClassSampler(Sampler):
    def __init__(self, labels, mode="downsampling"):
        """
        Args:
            labels: array of class labels
            mode: "downsampling" or "upsampling"
        """
        self.labels = np.array(labels)
        self.mode = mode

        # Get class counts
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique_labels, counts))

        # Determine sampling counts
        if mode == "downsampling":
            self.sample_count = min(counts)
        elif mode == "upsampling":
            self.sample_count = max(counts)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Generate indices
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        for label, count in self.class_counts.items():
            label_indices = np.where(self.labels == label)[0]
            if self.mode == "downsampling":
                selected = np.random.choice(label_indices, self.sample_count, replace=False)
            else:  # upsampling
                selected = np.random.choice(label_indices, self.sample_count, replace=True)
            indices.extend(selected)

        np.random.shuffle(indices)
        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



class AlaskaDataset(Dataset):
    def __init__(self, df, root_path, num_classes=4, transforms=None):
        self.df = df
        self.root_path = root_path
        self.num_classes = num_classes
        self.transforms = transforms

    def __getitem__(self, index):
        filename = self.df.iloc[index]["filename"]
        image = cv2.imread(f"{self.root_path}/{filename}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        label_idx = self.df.iloc[index]["label"]
        target = self._onehot(self.num_classes, label_idx)
        return torch.tensor(image).permute(2, 0, 1), target

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return list(self.df["label"].values)

    def _onehot(self, num_classes, target):
        vec = torch.zeros(num_classes, dtype=torch.float32)
        vec[target] = 1.0
        return vec





class AlaskaTestDataset(Dataset):
    def __init__(self, image_names, root_path, transforms=None):
        self.image_names = image_names
        self.root_path = root_path
        self.transforms = transforms

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = cv2.imread(f"{self.root_path}/Test/{image_name}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        return image_name, torch.tensor(image).permute(2, 0, 1)

    def __len__(self):
        return len(self.image_names)



def get_model():
    model = EfficientNet.from_pretrained("efficientnet-b2")
    model._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return model

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logits, targets):
        if self.training:
            logits = logits.float()
            targets = targets.float()

            log_probs = F.log_softmax(logits, dim=-1)
            nll_loss = (-log_probs * targets).sum(-1)
            smooth_loss = -log_probs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return F.cross_entropy(logits, targets.argmax(dim=1))

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        if sum(mask) != 0:
            x_padding = np.linspace(fpr[mask][-1], 1, 100)
            x = np.concatenate([fpr[mask], x_padding])
            y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
            y = y - y_min
            score = metrics.auc(x, y)
        else:
            score = 1.0

        submetric = score * weight
        competition_metric += submetric

    return competition_metric / normalization

class RocAucMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0.5, 0.5])
        self.score = 0

    def update(self, y_pred, y_true):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        y_pred = 1 - F.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score



class AlaskaLearner:
    def __init__(self, model, config, base_dir="./"):
        self.model = model.cuda()
        self.config = config
        self.base_dir = base_dir
        self.log_path = f"{self.base_dir}/log.txt"
        self.best_loss = 1e5
        self.start_time = time.time()  # Track overall training start time
        self.epoch_times = []  # To store epoch durations for estimation

        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
        self.optimizer,
        **config.scheduler_params  # Now gets updated params
    )
        self.criterion = LabelSmoothing().cuda()

        self.scaler = torch.cuda.amp.GradScaler()

        self.log("Learner initialized with mixed precision support.")

    def fit(self, train_loader, valid_loader):
        total_steps = len(train_loader) * self.config.n_epochs
        self.log(f"Total training steps: {total_steps}")
        total_start_time = time.time()

        for epoch in range(self.config.n_epochs):
            epoch_start_time = time.time()  # This is the correct variable name
            self.log(f"\nEpoch {epoch + 1}/{self.config.n_epochs} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Training
            train_loss, auc_scores = self._train_epoch(train_loader)
            train_time = time.time() - epoch_start_time
            self.epoch_times.append(train_time)

            # Calculate remaining time
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = self.config.n_epochs - (epoch + 1)
            remaining_time = remaining_epochs * avg_epoch_time

            self.log(
                f"[TRAIN] Loss: {train_loss.avg:.5f}, AUC: {auc_scores.avg:.5f}, "
                f"Time: {self._format_time(train_time)}, "
                f"Remaining: ~{self._format_time(remaining_time)}"
            )

            # Save checkpoint every epoch
            checkpoint_path = f"{self.base_dir}/fold{self.config.fold_number}-epoch{epoch + 1}-checkpoint.bin"
            self.save(checkpoint_path)
            self.log(f"Saved checkpoint to {checkpoint_path}")

            # Validation
            t = time.time()
            valid_loss, auc_scores = self._valid_epoch(valid_loader)
            valid_time = time.time() - t
            self.log(
                f"[VALID] Loss: {valid_loss.avg:.5f}, AUC: {auc_scores.avg:.5f}, Time: {valid_time:.1f}s"
            )

            if valid_loss.avg < self.best_loss:
                self.best_loss = valid_loss.avg
                best_path = f"{self.base_dir}/fold{self.config.fold_number}-best-checkpoint-{str(epoch + 1).zfill(3)}epoch.bin"
                self.save(best_path)
                self.log(f"New best model saved to {best_path}")

                # Keep only 3 best checkpoints
                for path in sorted(
                    glob.glob(f"{self.base_dir}/fold{self.config.fold_number}-best-checkpoint-*epoch.bin")
                )[:-3]:
                    os.remove(path)
                    self.log(f"Removed old checkpoint: {path}")

            if self.config.valid_scheduler:
                self.scheduler.step(metrics=valid_loss.avg)

            # Log epoch completion time - FIXED THIS LINE
            epoch_time = time.time() - epoch_start_time  # Changed from epoch_start to epoch_start_time
            self.log(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

        # Log total training time
        total_time = time.time() - total_start_time
        self.log(f"\nTraining completed in {self._format_time(total_time)}")

    def _train_epoch(self, train_loader):
        self.model.train()
        train_loss = AverageMeter()
        auc_scores = RocAucMeter()
        batch_times = AverageMeter()
        epoch_start_time = time.time()
        last_log_time = epoch_start_time

        for step, (images, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            images = images.cuda().float()
            targets = targets.cuda().float()

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config.step_scheduler:
                self.scheduler.step()

            auc_scores.update(outputs, targets)
            train_loss.update(loss.item(), images.size(0))

            batch_time = time.time() - batch_start_time
            batch_times.update(batch_time)

            # Live time estimation - update every 30 seconds or at end of epoch
            current_time = time.time()
            if current_time - last_log_time > 30 or step == len(train_loader) - 1:
                elapsed_time = current_time - epoch_start_time
                batches_remaining = len(train_loader) - step - 1
                estimated_remaining = batches_remaining * batch_times.avg

                self.log(
                    f"Step {step + 1}/{len(train_loader)} | "
                    f"Batch: {batch_time:.2f}s ({batch_times.avg:.2f}s avg) | "
                    f"Elapsed: {self._format_time(elapsed_time)} | "
                    f"ETA: {self._format_time(estimated_remaining)} | "
                    f"Loss: {train_loss.avg:.4f} | AUC: {auc_scores.avg:.4f}"
                )
                last_log_time = current_time

        return train_loss, auc_scores

    def _format_time(self, seconds):
        """Helper method to format time in human-readable way"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
        else:
            return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"

    def _valid_epoch(self, valid_loader):
        self.model.eval()
        valid_loss = AverageMeter()
        auc_scores = RocAucMeter()

        with torch.no_grad():
            for step, (images, targets) in enumerate(valid_loader):
                images = images.cuda().float()
                targets = targets.cuda().float()

                with torch.cuda.amp.autocast():  # <-- NEW
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                auc_scores.update(outputs, targets)
                valid_loss.update(loss.item(), images.size(0))

                if self.config.verbose and (step % self.config.verbose_step == 0):
                    print(
                        f"Validation Step {step}/{len(valid_loader)}, "
                        f"Loss: {valid_loss.avg:.4f}, AUC: {auc_scores.avg:.4f}",
                        end="\r"
                    )

        return valid_loss, auc_scores

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Special handling for scheduler
        if "scheduler_state_dict" in checkpoint:
            # Completely reinitialize the scheduler with current config
            self.scheduler = self.config.SchedulerClass(
                self.optimizer,
                **self.config.scheduler_params
            )

        self.best_loss = checkpoint["best_loss"]

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, "a+") as logger:
            logger.write(f"{message}\n")

    def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            best_loss = checkpoint.get("best_loss", float("inf"))
            config = checkpoint.get("config", None)

            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            return best_loss, config

        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {str(e)}")
            return float("inf"), None

# Utility Functions

def format_time(seconds):
    """Convert seconds to human-readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            # Only load scheduler state if requested
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_loss = checkpoint.get("best_loss", float("inf"))
        config = checkpoint.get("config", None)

        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return best_loss, config

    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {str(e)}")
        return float("inf"), None

def create_folds(data_root, n_splits=5):
    """Create cross-validation folds from the original dataset structure"""
    dataset = []
    classes = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]

    for label, class_name in enumerate(classes):
        image_paths = glob.glob(f"{data_root}/{class_name}/*.jpg")
        for path in image_paths:
            dataset.append({
                "filename": f"{class_name}/{os.path.basename(path)}",
                "label": label,
                "image_name": os.path.basename(path)
            })

    random.shuffle(dataset)
    df = pd.DataFrame(dataset)

    # Create stratified folds
    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = 0
    for fold_number, (train_index, val_index) in enumerate(
        gkf.split(X=df.index, y=df["label"], groups=df["image_name"])
    ):
        df.loc[df.iloc[val_index].index, "fold"] = fold_number

    return df

def train_pipeline(config, data_root="/kaggle/input/alaska2-image-steganalysis", resume_from=None):
    seed_everything(config.seed)

    # Generate folds from original dataset structure
    df = create_folds(data_root)

    # Create datasets
    train_dataset = AlaskaDataset(
        df=df[df["fold"] != config.fold_number],
        root_path=data_root
    )

    valid_dataset = AlaskaDataset(
        df=df[df["fold"] == config.fold_number],
        root_path=data_root
    )

    # Create data loaders with custom sampler
    train_loader = DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=True,
    )

    # Update scheduler steps
    config.steps_per_epoch = len(train_loader)

    # Initialize model and learner
    model = get_model()
    learner = AlaskaLearner(model=model, config=config)

    # Resume training if checkpoint provided
    if resume_from is not None:
        # Don't load scheduler state - we'll create a fresh one
        best_loss, loaded_config = load_checkpoint(
            model=model,
            checkpoint_path=resume_from,
            optimizer=learner.optimizer,
            scheduler=None  # Skip loading scheduler state
        )
        if loaded_config is not None:
            learner.best_loss = best_loss
            print(f"Resuming training from checkpoint: {resume_from}")

    # Train the model
    learner.fit(train_loader, valid_loader)

def create_submission(model_path, data_root="/kaggle/input/alaska2-image-steganalysis", output_file="submission.csv"):
    # Load model
    checkpoint = torch.load(model_path)
    config = checkpoint["config"]
    model = get_model().cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create test dataset
    test_image_names = [os.path.basename(x) for x in glob.glob(f"{data_root}/Test/*.jpg")]
    test_dataset = AlaskaTestDataset(
        image_names=test_image_names,
        root_path=data_root
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    # Run inference
    results = {"Id": [], "Label": []}
    with torch.no_grad():
        for image_names, images in test_loader:
            images = images.cuda().float()
            outputs = model(images)
            probs = 1 - F.softmax(outputs, dim=1).data.cpu().numpy()[:, 0]

            results["Id"].extend(image_names)
            results["Label"].extend(probs)

    # Save submission
    submission = pd.DataFrame(results)
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


if __name__ == "__main__":
    config = TrainGlobalConfig()

    # Check for existing checkpoints to resume from
    checkpoint_files = glob.glob("fold0-best-checkpoint-005epoch.bin") + glob.glob("fold*-epoch*-checkpoint.bin")

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        latest_checkpoint = "fold0-best-checkpoint-005epoch.bin"
        print(f"Found existing checkpoint: {latest_checkpoint}")
        resume = input("Do you want to resume training from this checkpoint? (y/n): ").lower()
        if resume == 'y':
            config = TrainGlobalConfig()  # Reinitialize config
            train_pipeline(config, resume_from=latest_checkpoint)
        else:
            train_pipeline(config)
    else:
        train_pipeline(config)


    # Create submission using the best model
    best_models = glob.glob(f"fold{config.fold_number}-best-checkpoint-*epoch.bin")
    if best_models:
        best_model = max(best_models, key=os.path.getctime)
        create_submission(best_model)
    else:
        print("No best model found for submission")


if __name__ == "__main__":
    config = TrainGlobalConfig()
    # Create submission using the best model
    best_models = glob.glob(f"fold{config.fold_number}-best-checkpoint-005epoch.bin")
    if best_models:
        create_submission(best_model)
    else:
        print("No best model found for submission")


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = "/kaggle/input/finalsubmission/result.png"

img = mpimg.imread(image_path)

plt.figure(figsize=(10, 10), dpi=300)

plt.imshow(img)
plt.axis('off')
plt.show()


