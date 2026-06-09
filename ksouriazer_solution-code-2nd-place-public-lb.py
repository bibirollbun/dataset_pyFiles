!nvidia-smi


# !pip install -r requirements.txt


import glob
import os
import random
import warnings
from datetime import datetime

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


classes = ["Naeimi", "Goat", "Roman", "Harri", "Sawakni", "Barbari", "Najdi"]

mapper = dict(zip(classes, range(len(classes))))
rev_mapper = {v: k for k, v in mapper.items()}
# read data
train_df = pd.read_csv("data/Sheep Classification Images/train_labels.csv")
train_df["path"] = train_df.filename.apply(
    lambda fn: os.path.join("data/Sheep Classification Images", "train", fn)
)
# create test set
test_img_paths = glob.glob("data/Sheep Classification Images/test/*")

test_df = pd.DataFrame(
    [os.path.basename(path) for path in test_img_paths], columns=["filename"]
)
test_df["path"] = test_img_paths
def make_stratified_folds(train_df, n_splits=5, label_col="label", seed=42):
    train_df = train_df.copy().reset_index(drop=True)
    train_df["folds"] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[label_col])):
        train_df.loc[val_idx, "folds"] = fold
    return train_df


# Example usage:
train_df = make_stratified_folds(train_df, n_splits=5, label_col="label", seed=42)


class SheepClassifDataset(Dataset):
    def __init__(self, dataframe, transform=None, to_train=True, mapper=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.to_train = to_train
        self.mapper = mapper  # Dict: string label → integer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["path"]
        # Read with OpenCV (BGR)
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Cannot read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            # For albumentations, expects 'image' kwarg and returns a dict
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor, (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)

        if self.to_train:
            label = row["label"]
            if self.mapper:
                label = self.mapper[label]
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        else:
            return image

class SheepClassifier(nn.Module):
    def __init__(
        self,
        base_name="convnext_large.fb_in22k_ft_in1k_384",
        num_classes=len(classes),
        pretrained=True,
        in_chans=3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            base_name, pretrained=pretrained, in_chans=in_chans, num_classes=0
        )
        # Infer feature dimension
        test_input = torch.zeros(1, in_chans, 384, 384)
        out = self.backbone(test_input)
        emb_dim = out.shape[-1] if len(out.shape) == 2 else out.shape[1]
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: (B, 3, H, W)
        emb = self.backbone(x)
        out = self.classifier(emb)
        return out

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    y_a, y_b = y, y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        if self.reduction == "mean":
            return loss.mean()
        return loss
class Trainer:
    def __init__(
        self,
        model_class,
        dataset_class,
        train_df,
        test_df,
        mapper,
        folds,
        fold_column="folds",
        num_classes=2,
        seed=42,
        img_size=224,
        batch_size=16,
        lr=1e-4,
        num_workers=2,
    ):
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.train_df = train_df
        self.test_df = test_df
        self.mapper = mapper
        self.folds = folds
        self.fold_column = fold_column
        self.num_classes = num_classes
        self.seed = seed
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            "approach-1", self.model_class().__class__.__name__, self.timestamp
        )

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def seed_worker(self, worker_id):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def get_transforms(self):
        self.set_seed()
        train_transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomResizedCrop(
                            self.img_size,
                            self.img_size,
                            scale=(0.7, 1.2),
                            ratio=(0.75, 1.33),
                            p=0.5,
                        ),
                        A.Resize(self.img_size, self.img_size, p=0.5),
                    ],
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    p=0.3, scale_limit=0.2, rotate_limit=30, border_mode=0
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(p=0.3),
                        A.GaussNoise(p=0.3),
                        A.ImageCompression(quality_lower=85, quality_upper=95, p=0.3),
                    ],
                    p=0.25,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=self.img_size // 8,
                    max_width=self.img_size // 8,
                    p=0.4,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        test_transforms = A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        return train_transforms, test_transforms

    def get_loaders(self, fold):
        train_transforms, test_transforms = self.get_transforms()
        train_data = self.train_df[self.train_df[self.fold_column] != fold].reset_index(
            drop=True
        )
        valid_data = self.train_df[self.train_df[self.fold_column] == fold].reset_index(
            drop=True
        )
        dataset_train = self.dataset_class(
            train_data, transform=train_transforms, to_train=True, mapper=self.mapper
        )
        dataset_valid = self.dataset_class(
            valid_data, transform=test_transforms, to_train=True, mapper=self.mapper
        )

        y_train = [self.mapper[row["label"]] for idx, row in train_data.iterrows()]
        class_sample_count = np.array(
            [(np.array(y_train) == t).sum() for t in np.unique(y_train)]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight).float()

        generator = torch.Generator()
        generator.manual_seed(self.seed + fold)
        sampler = WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True, generator=generator
        )
        train_loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=generator,
            drop_last=False,
        )
        valid_loader = DataLoader(
            dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            drop_last=False,
        )
        return train_loader, valid_loader, valid_data

    def train_fold(self, fold, num_epochs=18, model_save_dir=None, patience=7):
        self.set_seed(self.seed + fold)

        if model_save_dir is None:
            model_save_dir = os.path.join(self.experiment_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"best_model_fold{fold}.pth")

        train_loader, valid_loader, valid_data = self.get_loaders(fold)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_class().to(device)

        # AdamW + OneCycleLR
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * num_epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.15,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=10,
            final_div_factor=10,
        )

        y_train = []
        for idx, row in train_loader.dataset.dataframe.iterrows():
            y_train.append(self.mapper[row["label"]])
        y_train = np.array(y_train)
        class_weights = compute_class_weight(
            "balanced", classes=np.arange(self.num_classes), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        alpha = torch.tensor([1.0 for _ in range(self.num_classes)], device=device)
        criterion = FocalLoss(alpha=alpha, gamma=2)

        best_f1 = 0.0
        patience_counter = 0

        logger.info(f"Starting training for fold {fold}")
        for epoch in range(num_epochs):
            self.set_seed(self.seed + fold * 100 + epoch)
            model.train()
            epoch_loss = 0.0
            correct, total = 0, 0
            all_preds, all_targets = [], []

            for images, targets in tqdm(
                train_loader,
                desc=f"Fold {fold} | Epoch {epoch + 1}/{num_epochs} - Training",
            ):
                images, targets = images.to(device), targets.to(device)
                r = np.random.rand()
                if r < 0.45:
                    images, targets_a, targets_b, lam = mixup_data(
                        images, targets, alpha=0.5
                    )
                    outputs = model(images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                        outputs, targets_b
                    )
                elif r < 0.85:
                    images, targets_a, targets_b, lam = cutmix_data(
                        images, targets, alpha=0.5
                    )
                    outputs = model(images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                        outputs, targets_b
                    )
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                if r < 0.85:
                    correct += (
                        lam * (preds == targets_a).sum().item()
                        + (1 - lam) * (preds == targets_b).sum().item()
                    )
                    total += targets_a.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets_a.cpu().numpy())
                else:
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            train_loss = epoch_loss / len(train_loader)
            train_acc = correct / total
            train_f1 = f1_score(all_targets, all_preds, average="macro")

            val_loss, val_acc, val_f1 = self.validate(
                model, valid_loader, class_weights
            )
            logger.info(
                f"Fold {fold} | Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), model_save_path)
                logger.info(
                    f"Fold {fold} | >> Saved best model with val F1: {best_f1:.4f}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} (no improvement in {patience} epochs)"
                    )
                    break

        logger.info(f"Finished training for fold {fold}, best val F1: {best_f1:.4f}")
        return model_save_path

    def validate(self, model, valid_loader, class_weights):
        device = next(model.parameters()).device
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        val_all_preds, val_all_targets = [], []
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_targets.extend(targets.cpu().numpy())
        val_loss /= len(valid_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_all_targets, val_all_preds, average="macro")
        return val_loss, val_acc, val_f1

    @staticmethod
    def apply_tta(model, images):
        probas_list = []
        probas_list.append(F.softmax(model(images), dim=1))
        mean_probas = torch.stack(probas_list, dim=0).mean(dim=0)
        return mean_probas

    def predict_with_tta(self, model, loader, targets_provided=True):
        device = next(model.parameters()).device
        preds, true_vals, pred_probas = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="TTA Inference"):
                if targets_provided:
                    images, targets = batch
                    true_vals.append(targets.cpu().numpy())
                else:
                    images = batch
                images = images.to(device)
                tta_probas = self.apply_tta(model, images)
                probas = tta_probas.cpu().numpy()
                pred_classes = tta_probas.argmax(dim=1).cpu().numpy()
                preds.append(pred_classes)
                pred_probas.append(probas)
        preds = np.concatenate(preds, axis=0)
        pred_probas = np.concatenate(pred_probas, axis=0)
        if targets_provided:
            true_vals = np.concatenate(true_vals, axis=0)
            return preds, pred_probas, true_vals
        else:
            return preds, pred_probas

    def cross_val_training(self, num_epochs=18, model_save_dir=None, patience=7):
        if model_save_dir is None:
            model_save_dir = os.path.join(self.experiment_dir, "models")
        all_model_paths = []
        for fold in self.folds:
            model_path = self.train_fold(
                fold=fold,
                num_epochs=num_epochs,
                model_save_dir=model_save_dir,
                patience=patience,
            )
            all_model_paths.append(model_path)
        logger.info(f"All folds trained: {all_model_paths}")
        return all_model_paths

    def cross_val_inference(self, model_save_dir=None, oof_dir=None):
        if model_save_dir is None:
            model_save_dir = os.path.join(self.experiment_dir, "models")
        if oof_dir is None:
            oof_dir = os.path.join(self.experiment_dir, "train_oof_proba")
        os.makedirs(oof_dir, exist_ok=True)
        valid_data_all = []
        all_error_samples = []
        for fold in self.folds:
            model = self.model_class()
            model_path = os.path.join(model_save_dir, f"best_model_fold{fold}.pth")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model = model.cuda() if torch.cuda.is_available() else model

            _, valid_loader, valid_data = self.get_loaders(fold)
            preds, pred_probas, true_vals = self.predict_with_tta(model, valid_loader)

            valid_data_grouped = pd.DataFrame(
                {
                    "index": valid_data.index,
                    "path": valid_data.path,
                    "gt": true_vals,
                    "pred": preds,
                    "pred_proba": list(pred_probas),
                }
            )
            # -- error collection (uses 'path' or 'filename' column!) --
            rev_mapper = {v: k for k, v in self.mapper.items()}
            for idx, row in valid_data.iterrows():
                gt = row["label"] if "label" in row else row["Target"]
                pred_id = preds[idx - valid_data.index[0]]  # adjust index if needed!
                pred = rev_mapper[pred_id] if pred_id in rev_mapper else pred_id
                gt_id = self.mapper[gt] if gt in self.mapper else gt
                gt_str = rev_mapper[gt_id] if gt_id in rev_mapper else str(gt)
                img_path = row.get("path", row.get("filename", None))
                if img_path is not None and gt_str != pred:
                    all_error_samples.append(
                        {"img_path": img_path, "gt": gt_str, "pred": pred}
                    )

            valid_data_all.append(valid_data_grouped)
            fold_oof_path = os.path.join(oof_dir, f"fold{fold}.csv")
            save_df = valid_data_grouped.copy()
            save_df["pred_proba"] = save_df["pred_proba"].apply(
                lambda x: ",".join([f"{v:.8f}" for v in x])
            )
            save_df.to_csv(fold_oof_path, index=False)
            logger.info(f"Fold {fold} OOF probabilities saved to {fold_oof_path}")

            acc = accuracy_score(valid_data_grouped["gt"], valid_data_grouped["pred"])
            f1 = f1_score(
                valid_data_grouped["gt"], valid_data_grouped["pred"], average="macro"
            )
            print(f"Fold {fold} | Macro F1: {f1:.4f}")

        valid_data_all = pd.concat(valid_data_all).reset_index(drop=True)
        logger.info("Validation inference complete.")

        return valid_data_all

    def test_infer_cv_ensemble(
        self,
        model_save_dir=None,
        submission_path=None,
        id_col="filename",
        oof_dir=None,
        fold_indices=None,
    ):
        if model_save_dir is None:
            model_save_dir = os.path.join(self.experiment_dir, "models")
        if oof_dir is None:
            oof_dir = os.path.join(self.experiment_dir, "test_oof_proba")
        os.makedirs(oof_dir, exist_ok=True)
        test_df = self.test_df.copy()
        _, test_transforms = self.get_transforms()
        dataset_test = self.dataset_class(
            test_df, transform=test_transforms, to_train=False, mapper=self.mapper
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
        )
        all_fold_pred_probas = []

        for fold in self.folds:
            model = self.model_class()
            model_path = os.path.join(model_save_dir, f"best_model_fold{fold}.pth")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model = model.cuda() if torch.cuda.is_available() else model
            _, pred_probas = self.predict_with_tta(
                model, test_loader, targets_provided=False
            )
            all_fold_pred_probas.append(pred_probas)
            fold_oof_path = os.path.join(oof_dir, f"fold{fold}.csv")
            fold_df = pd.DataFrame(
                {
                    id_col: test_df[id_col],
                    "pred_proba": [
                        ",".join([f"{v:.8f}" for v in x]) for x in pred_probas
                    ],
                }
            )
            fold_df.to_csv(fold_oof_path, index=False)
            logger.info(f"Test probabilities for fold {fold} saved to {fold_oof_path}")

        logger.info("All folds' OOF saved, now blending...")

        all_fold_pred_probas = np.stack(all_fold_pred_probas)
        mean_pred_probas = all_fold_pred_probas.mean(axis=0)
        predicted_targets = mean_pred_probas.argmax(axis=1)

        submission_df = pd.DataFrame(
            {
                id_col: test_df[id_col],
                "label": [
                    list(self.mapper.keys())[list(self.mapper.values()).index(x)]
                    for x in predicted_targets
                ],
            }
        )

        if submission_path is None:
            submission_name = f"submission_{self.timestamp}.csv"
            submission_path = os.path.join(self.experiment_dir, submission_name)

        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Final blended submission saved to {submission_path}")
        return submission_path, submission_df

trainer = Trainer(
    model_class=SheepClassifier,
    dataset_class=SheepClassifDataset,
    train_df=train_df,
    test_df=test_df,
    mapper=mapper,
    folds=[0, 1, 2, 3, 4],
    fold_column="folds",
    num_classes=len(mapper),
    seed=42,
    img_size=448,
    batch_size=16,
    lr=1e-4,
    num_workers=os.cpu_count(),
)

# Train all folds (models saved under trainer.experiment_dir/models)
trainer.cross_val_training(num_epochs=100, patience=10)

# Inference and evaluation on validation sets across all folds
valid_data_all = trainer.cross_val_inference()

# compute F1 and accuracy from valid_data_all:
y_true = valid_data_all["gt"]
y_pred = np.array(
    [
        np.argmax([float(x) for x in row.split(",")])
        if isinstance(row, str)
        else np.argmax(row)
        for row in valid_data_all["pred_proba"]
    ]
)
print(f"Validation macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"Validation accuracy: {accuracy_score(y_true, y_pred):.4f}")

# Test ensemble prediction and submission
submission_path, submission_df = trainer.test_infer_cv_ensemble()

# Print submission path
print(f"Submission saved to: {submission_path}")
pd.read_csv(f"{submission_path}")
print(pd.read_csv(f"{submission_path}")["label"].value_counts())
print(30 * "--")
print(pd.read_csv(f"{submission_path}")["label"].value_counts(normalize=True))
train_df["label"].value_counts(normalize=True)
sub = pd.read_csv(f"{submission_path}")

pp_sub = pd.read_csv("postprocessing_rows.csv").sort_values("filename")

# Ensure both are sorted by filename for alignment
mask = sub["filename"].isin(pp_sub["filename"])
sub.loc[mask, "label"] = sub.loc[mask, "filename"].map(
    dict(zip(pp_sub["filename"], pp_sub["label"]))
)

sub.to_csv(submission_path, index=False)

