import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as v2
from torchvision import models
import timm
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import warnings
import random
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Config:
    TRAIN_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/'
    TEST_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test/'
    TRAIN_CSV = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
    
    MODELS = [
        
        {'name': 'coatnet_3_rw_224.sw_in12k', 'image_size': 224, 'weight': 0.35},
        {'name': 'eva02_base_patch14_224.mim_in22k', 'image_size': 224, 'weight': 0.3},
        {'name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k', 'image_size': 224, 'weight': 0.2},
        {'name': 'convnext_small.fb_in22k_ft_in1k_384', 'image_size': 384, 'weight': 0.15},
    ]

    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    NUM_WORKERS = 4
    BASE_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    NUM_FOLDS = 5
    
    CLASSES = ['Naeimi', 'Najdi', 'Harri', 'Goat', 'Sawakni', 'Roman', 'Barbari']
    NUM_CLASSES = len(CLASSES)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 0.8
    LABEL_SMOOTHING = 0.02

    ROTATION_DEGREES = 25
    COLOR_BRIGHTNESS = 0.25
    COLOR_CONTRAST = 0.25
    COLOR_SATURATION = 0.25
    COLOR_HUE = 0.25/2
    PERSPECTIVE_SCALE = 0.1
    CROP_SCALE = (0.85, 1.0)
    CROP_RATIO = (0.9, 1.1)
    ERASING_PROB = 0.1
    ERASING_SCALE = (0.02, 0.15)

    WARMUP_EPOCHS = 5
    FREEZE_EPOCHS = 10



class SheepDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, is_train=True, use_tta=False):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_train = is_train
        self.use_tta = use_tta
        
        if is_train:
            self.label_to_idx = {label: idx for idx, label in enumerate(Config.CLASSES)}
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            if min(image.size) < 224:
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.use_tta and not self.is_train:
            images = []
            for _ in range(8):
                if self.transforms:
                    images.append(self.transforms(image.copy()))
            return torch.stack(images), row['filename']
        if self.transforms:
            image = self.transforms(image)
        if self.is_train:
            label = self.label_to_idx[row['label']]
            return image, label
        else:
            return image, row['filename']


def get_transforms(image_size=224):    
    train_transforms = v2.Compose([
        v2.Resize((int(image_size * 1.1), int(image_size * 1.1))),
        v2.RandomResizedCrop(image_size, scale=Config.CROP_SCALE, ratio=Config.CROP_RATIO),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=Config.ROTATION_DEGREES),
        v2.ColorJitter(brightness=Config.COLOR_BRIGHTNESS, contrast=Config.COLOR_CONTRAST, 
                      saturation=Config.COLOR_SATURATION, hue=Config.COLOR_HUE),
        v2.RandomPerspective(distortion_scale=Config.PERSPECTIVE_SCALE, p=0.3),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.1),
        v2.RandomErasing(p=Config.ERASING_PROB, scale=Config.ERASING_SCALE),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = v2.Compose([
        v2.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        v2.RandomChoice([
            v2.Resize((image_size, image_size)),
            v2.Resize((int(image_size * 1.05), int(image_size * 1.05))),
            v2.Resize((int(image_size * 0.95), int(image_size * 0.95)))
        ]),
        v2.CenterCrop(image_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms, test_transforms


def get_model(model_config):
    model_name = model_config['name']
    try:
        model = timm.create_model(model_name, pretrained=True)
        
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(num_classes=0, global_pool="avg")

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            feat = model.forward_features(dummy_input)

        if feat.ndim == 2 or feat.ndim == 3:
            feature_dim = getattr(model, "num_features", None)
            if feature_dim is None:
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    feat = model.forward_features(dummy_input)
                    feature_dim = feat.shape[1]
    
            new_head = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(0.3),
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, Config.NUM_CLASSES),
            )
            
        elif feat.ndim == 4:
            feature_dim = feat.shape[1]
            
            new_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(0.3),
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, Config.NUM_CLASSES)
            )
        else:
            raise ValueError(
                f"Forward_features: {tuple(feat.shape)}"
            )
        
        if hasattr(model, "head") and isinstance(model.head, nn.Module):
            model.head = new_head
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
            model.classifier = new_head
        elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
            model.fc = new_head
        else:
            model.head = new_head
        
        print(f"Created {model_name}. shape={tuple(feat.shape)}")
        return model
    
    except Exception as e:
        print(f"Error creating {model_name}: {e}.")
        return timm.create_model("resnet50", pretrained=True, num_classes=Config.NUM_CLASSES)



def calculate_weight(df):
    class_counts = df['label'].value_counts()
    total = len(df)
    
    class_weights = {cls: total / (len(Config.CLASSES) * count) 
                    for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in df['label']]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        else:
            focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                          device, epoch, max_epochs, use_mixup=True):
    model.train()
    
    # Progressive unfreezing
    if epoch < Config.FREEZE_EPOCHS:
        for name, param in model.named_parameters():
            if not any(x in name.lower() for x in ['classifier', 'head', 'fc']):
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    mixup_prob = max(0.3, 0.5 - (epoch / max_epochs) * 0.3)
    cutmix = v2.CutMix(num_classes=Config.NUM_CLASSES, alpha=Config.CUTMIX_ALPHA)
    mixup = v2.MixUp(num_classes=Config.NUM_CLASSES, alpha=Config.MIXUP_ALPHA)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        original_target = target.clone()
        
        if use_mixup and np.random.random() < mixup_prob:
            data, target = cutmix_or_mixup(data, target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(output.data, 1)
        total += original_target.size(0)
        correct += (predicted == original_target).sum().item()
            
                
    if scheduler:
        scheduler.step()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc


def train_model_fold(train_df, val_df, model_config, fold_num):
    print(f"Training {model_config['name']} - Fold {fold_num + 1}")
    
    image_size = model_config['image_size']
    weighted_sampler = calculate_weight(train_df)
    
    model = get_model(model_config)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(Config.DEVICE)
    
    class_counts = train_df['label'].value_counts()
    alpha_weights = torch.tensor([1.0 / class_counts[cls] for cls in Config.CLASSES])
    alpha_weights = alpha_weights / alpha_weights.sum() * Config.NUM_CLASSES
    alpha_weights = alpha_weights.to(Config.DEVICE)
    
    focal_loss = FocalLoss(alpha=alpha_weights, gamma=2.0)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    def combined_loss(outputs, targets):
        if targets.dim() > 1 and targets.size(1) == Config.NUM_CLASSES:
            log_probs = F.log_softmax(outputs, dim=1)
            return -torch.sum(targets * log_probs, dim=1).mean()
        else:
            return 0.6 * ce_loss(outputs, targets) + 0.4 * focal_loss(outputs, targets)
    
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ['classifier', 'head', 'fc']):
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.BASE_LR * 0.1},
        {'params': head_params, 'lr': Config.BASE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=1e-7
    )
    
    best_f1 = 0.0
    best_model_state = None
    patience = 8
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        train_transforms, val_transforms, _ = get_transforms(image_size)
        
        train_dataset = SheepDataset(train_df, Config.TRAIN_DIR, 
                                           train_transforms, is_train=True)
        val_dataset = SheepDataset(val_df, Config.TRAIN_DIR, 
                                         val_transforms, is_train=True)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                                sampler=weighted_sampler, num_workers=Config.NUM_WORKERS, 
                                pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, combined_loss, optimizer, scheduler, 
            Config.DEVICE, epoch, Config.NUM_EPOCHS, use_mixup=(epoch >= Config.WARMUP_EPOCHS)
        )
        
        val_loss, val_f1, _, _ = validate_epoch(model, val_loader, ce_loss, Config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_f1 + 0.001:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
            
        print("-" * 50)
    
    print(f"Best F1 for {model_config['name']} - Fold {fold_num + 1}: {best_f1:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_f1


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return epoch_loss, f1, all_preds, all_targets


def predict_ensemble(models_dict, test_df):
    all_model_predictions = []
    
    # maximum mean score across all models
    max_mean_score = max([np.mean(model_data['scores']) for model_data in models_dict.values()])
    
    for model_name, model_data in models_dict.items():
        model_config = next(config for config in Config.MODELS if config['name'] == model_name)
        image_size = model_config['image_size']
        base_weight = model_config['weight']
        
        # Weight by average CV
        cv_scores = model_data['scores']
        performance_weight = np.mean(cv_scores) / max_mean_score
        final_weight = base_weight * performance_weight
        
        print(f"Model: {model_name}")
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {np.mean(cv_scores):.4f}")
        print(f"Performance weight: {performance_weight:.4f}")
        print(f"Final weight: {final_weight:.4f}\n")
        
        _, _, test_transforms = get_transforms(image_size)
        
        test_dataset = SheepDataset(test_df, Config.TEST_DIR, test_transforms, 
                                          is_train=False, use_tta=True)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE//4, 
                                shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        model_predictions = []
        
        for fold_idx, model in enumerate(model_data['models'][0]):
            print(f"Processing {model_name} fold {fold_idx + 1}...")
            
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(test_loader):
                    batch_size, num_tta = data.shape[:2]
                    data = data.view(-1, *data.shape[2:])
                    data = data.to(Config.DEVICE)
                    
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    
                    # Reshape & TTA
                    probs = probs.view(batch_size, num_tta, -1)
                    probs = probs.mean(dim=1)
                    
                    predictions.append(probs.cpu().numpy())
            
            model_predictions.append(np.concatenate(predictions))
        
        avg_model_predictions = np.mean(model_predictions, axis=0)
        weighted_predictions = avg_model_predictions * final_weight
        all_model_predictions.append(weighted_predictions)
        
        print(f"Completed predictions for {model_name}\n")
    
    ensemble_probs = np.sum(all_model_predictions, axis=0)
    
    temperature = 1.2
    ensemble_probs = ensemble_probs / temperature
    ensemble_probs = np.exp(ensemble_probs) / np.sum(np.exp(ensemble_probs), axis=1, keepdims=True)
    
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    pred_labels = [Config.CLASSES[pred] for pred in ensemble_preds]
    
    print("Ensemble prediction completed")
    return pred_labels, ensemble_probs


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
train_df = pd.read_csv(Config.TRAIN_CSV)
print(f"Total training samples: {len(train_df)}")
print("\nClass distribution:")
class_dist = train_df['label'].value_counts().sort_index()
print(class_dist)


skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=42)
all_models = {}
all_scores = {}
all_fold_preds = []
all_fold_targets = []

for model_config in Config.MODELS:
    model_name = model_config['name']
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    models = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"\nFold {fold + 1} - Train: {len(fold_train_df)}, Val: {len(fold_val_df)}")
        
        model, best_f1 = train_model_fold(fold_train_df, fold_val_df, model_config, fold)
        
        image_size = model_config['image_size']
        _, val_transforms, _ = get_transforms(image_size)
        val_dataset = SheepDataset(fold_val_df, Config.TRAIN_DIR, val_transforms, is_train=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        _, _, fold_preds, fold_targets = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        all_fold_preds.extend(fold_preds)
        all_fold_targets.extend(fold_targets)
        
        models.append(model)
        fold_scores.append(best_f1)
        
        torch.save(model.state_dict(), f'{model_name}_fold_{fold + 1}_enhanced.pth')
    
    all_models[model_name] = {
        'models': [models],
        'scores': fold_scores
    }
    all_scores[model_name] = fold_scores
    
    print(f"\n{model_name} CV Results:")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i + 1}: {score:.4f}")
    print(f"Mean CV Score: {np.mean(fold_scores):.4f} Â± {np.std(fold_scores):.4f}")



cm = confusion_matrix(all_fold_targets, all_fold_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Load images
# images = [
#     ("CoAtNet Confusion Matrix", "CoAtNet.png"),
#     ("EVA02 Confusion Matrix", "EVA02.png"),
#     ("ViT-Base & ConvNeXt Confusion Matrix", "ViT_ConvNext.png")
# ]

# # Display images
# for ax, (title, path) in zip(axes, images):
#     img = mpimg.imread(path)
#     ax.imshow(img)
#     ax.set_title(title, fontsize=10)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()


test_files = os.listdir(Config.TEST_DIR)
test_df = pd.DataFrame({'filename': test_files})
predictions, probabilities = predict_ensemble(all_models, test_df)

submission_df = pd.DataFrame({
    'filename': test_df['filename'],
    'label': predictions
})

submission_df.to_csv('submission.csv', index=False)





