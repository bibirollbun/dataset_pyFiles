import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import random
import warnings
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from PIL import ImageFilter

torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


data_path = kagglehub.competition_download('sheep-classification-challenge-2025') + "/Sheep Classification Images"

TRAIN_DIR = Path(data_path, "train")
TEST_DIR  = Path(data_path, "test")
TRAIN_CSV = Path(data_path, "train_labels.csv")
NUM_CLASSES = 7

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'Number of available GPUs: {torch.cuda.device_count()}')


class BASE_CFG:
    def __call__(self):
        raise NotImplementedError()
    def __repr__(self):
        return f"Model: {self.MODEL_NAME}, Batch Size: {self.BATCH_SIZE}, Epochs: {self.EPOCHS}, Num Folds: {self.NUM_FOLDS}, Dropout: {self.DROPRATE}, Learning Rate: {self.LR}"

class Dinov2SmallCfg(BASE_CFG):
    MODEL_NAME = "timm/vit_small_patch14_reg4_dinov2.lvd142m"
    IMAGE_SIZE = (518, 518)
    BATCH_SIZE = 32
    EPOCHS = 22
    NUM_FOLDS = 4
    DROPRATE = 0.05
    LR = 1e-4

class Dinov2BaseCfg(BASE_CFG):
    MODEL_NAME = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
    IMAGE_SIZE = (518, 518)
    BATCH_SIZE = 22
    EPOCHS = 22
    NUM_FOLDS = 2
    DROPRATE = 0.025
    LR = 1e-6

class Eva2BaseCfg(BASE_CFG):
    MODEL_NAME = "timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
    IMAGE_SIZE = (448, 448)
    BATCH_SIZE = 24
    EPOCHS = 26
    NUM_FOLDS = 4
    DROPRATE = 0.25
    LR = 2e-5

class ConvnextV2BaseCfg(BASE_CFG):
    MODEL_NAME = "timm/convnextv2_base.fcmae_ft_in22k_in1k_384"
    IMAGE_SIZE = (384, 384)
    BATCH_SIZE = 32
    EPOCHS = 36
    NUM_FOLDS = 4
    DROPRATE = 0.15
    LR = 4e-5

class DietV3LargeCfg(BASE_CFG):
    MODEL_NAME = "timm/deit3_large_patch16_224.fb_in22k_ft_in1k"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 36
    EPOCHS = 32
    NUM_FOLDS = 4
    DROPRATE = 0.15
    LR = 4e-5


MODEL_CFG = Dinov2BaseCfg


train_df = pd.read_csv(TRAIN_CSV)
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

test_files = sorted(os.listdir(TEST_DIR))


IMAGES: dict[str, Image.Image] = {}


from albumentations import CoarseDropout

class SheepDatasetV2(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False, is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.is_train = is_train
        # if not self.is_test:
        self.preprocess()

    def preprocess(self):
        """ remove background """

        print('Removing background...')
        filenames = list(self.df['filename'] if not self.is_test else self.df)
        remaining_files = [f for f in filenames if f not in IMAGES]
        train_files = os.listdir(TRAIN_DIR)
        if not remaining_files:
            return
            
        # Process in batches of 8
        batch_size = 6
        for i in tqdm(range(0, len(remaining_files), batch_size), desc="Processing batches"):
            batch_files = remaining_files[i:i + batch_size]
            batch_images = []
            
            for filename in batch_files:
                img_dir = self.img_dir if filename in train_files else TEST_DIR
                img_path = os.path.join(img_dir, filename)
                image = Image.open(img_path).convert("RGB")
                batch_images.append(transform_image(image))
            
            batch_tensor = torch.stack(batch_images)
            processed_masks = batch_process(batch_tensor)
            
            for filename, mask in zip(batch_files, processed_masks):
                img_dir = self.img_dir if filename in train_files else TEST_DIR
                img_path = os.path.join(img_dir, filename)
                original_image = Image.open(img_path).convert("RGB")
                image = original_image.copy()
                image.putalpha(mask.resize(original_image.size))
                image = create_sheep_herd_image(crop_image(image))
                IMAGES[filename] = (original_image, image)

        if not IMAGES:
            return


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.is_test:
            img_name = self.df[idx]
        else:
          img_name = self.df.iloc[idx]['filename']

        
        if not self.is_test:
            original_image, image = IMAGES[img_name]
            image = image.convert("RGB")
            # original_image = Image.fromarray(CoarseDropout(p=0.5)(image=np.array(original_image))['image'])
            # image = Image.fromarray(CoarseDropout(p=0.5)(image=np.array(image))['image'])
            return self.transform(image), self.transform(original_image), self.df.iloc[idx]['encoded_label']
        else:
            original_image, image = IMAGES[img_name]
            # original_image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
            # image = comp(image, original_image.copy(), 2).convert("RGBA")
            original_image = self.transform(original_image)
            image = self.transform(image.convert("RGB"))
            # return original_image, original_image, img_name
            return image, original_image, img_name


torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet = nn.DataParallel(birefnet).to("cuda").eval()

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def crop_image(image: Image.Image, padding=0) -> Image.Image:
    alpha = np.array(image)[:, :, 3]
    non_transparent = np.where(alpha > 0)

    top = max(0, non_transparent[0].min() - padding)
    bottom = min(image.height, non_transparent[0].max() + 1 + padding)
    left = max(0, non_transparent[1].min() - padding)
    right = min(image.width, non_transparent[1].max() + 1 + padding)

    # Crop the image
    cropped_img = image.crop((left, top, right, bottom))
    return cropped_img


def process(image: Image.Image) -> Image.Image:
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        with autocast(device_type="cuda"):
            preds = birefnet(input_images)
    pred = preds[-1].sigmoid().cpu()[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    image.putalpha(mask)
    # image = crop_image(image)
    return image

def batch_process(images: torch.Tensor) -> torch.Tensor:
    images = images.to("cuda")
    with torch.no_grad():
        with autocast(device_type="cuda"):
            preds = birefnet(images)
            
    preds = preds[-1].sigmoid().cpu().squeeze(1)
    # convert to PIL images
    processed_images = []
    for pred in preds:
        pred_pil = transforms.ToPILImage()(pred)
        processed_images.append(pred_pil)
    return processed_images


def rgba2rgb(image):
    background = Image.new("RGB", image.size, (0, 0, 0))  # White background
    image_rgb = Image.alpha_composite(Image.new("RGBA", image.size, (0, 0, 0, 0)), image).convert("RGB")
    return image_rgb



def unnormalize(image):
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  return (image * std + mean).clip(0,1)


def resize_padding(img, target_size, padding_color=(0, 0, 0)):
    img_width, img_height = img.size
    target_width, target_height = target_size
    aspect_ratio = min(target_width / img_width, target_height / img_height)
    
    new_width = int(img_width * aspect_ratio)
    new_height = int(img_height * aspect_ratio)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    new_img = Image.new("RGBA", target_size, padding_color)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_img.paste(resized_img, (paste_x, paste_y), resized_img)
    return new_img

def comp(img1, img2, gbr=6):
    size = img2.size
    img1 = img1.resize(size)
    img2 = img2.filter(ImageFilter.GaussianBlur(radius=gbr)).convert("RGBA")
    return Image.alpha_composite(img2, img1)
    

def create_sheep_herd_image(image: Image.Image):
    img_width, img_height = image.size

    placements = [
        {'rotation': 0, 'offset': (0, 0)},
        {'rotation': 0, 'offset': (40, 40)},
        {'rotation': 30, 'offset': (img_width, 0), "reverse": True},
        {'rotation': 30, 'offset': (0, img_height), "reverse": True},
        {'rotation': 0, 'offset': (img_width, img_height)}
    ]

    # Calculate canvas size
    canvas_width = int(img_width * 1.4)
    canvas_height = int(img_height * 1.4)

    # Create transparent canvas
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))


    # Composite each image in order (bottom to top)
    for placement in placements:
        rotation = placement['rotation']
        offset = placement['offset']

        if placement.get('reverse', False):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Rotate the image
        rotated_img = image.rotate(rotation, expand=True)

        temp_canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        temp_canvas.paste(rotated_img, offset)

        # Composite the temporary canvas onto the main canvas
        canvas = Image.alpha_composite(canvas, temp_canvas)

    return canvas



from timm.layers import AttentionPoolLatent

def resnet(num_classes):
    model = models.resnet34(pretrained=True)
    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def efficientnet(num_classes):
    model = models.efficientnet_v2_s(pretrained=True)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Linear(num_features, num_classes)
    return model.to(device)

def create_model():
    model = timm.create_model(
        MODEL_CFG.MODEL_NAME,
        pretrained=True,
        num_classes=7806,
        drop_rate=MODEL_CFG.DROPRATE
    )

    model.reset_classifier(NUM_CLASSES, "avg")
    model.global_pool == "map"
    model.attn_pool = AttentionPoolLatent(
        model.embed_dim,
        num_heads=model.blocks[0].attn.num_heads,
        mlp_ratio=4.0,
        norm_layer=lambda x: model.norm,
    )

    for i, layer in enumerate([model.patch_embed, model.norm]):
        for p in layer.parameters():
            p.requires_grad = False

    for i in range(0, len(model.blocks) - 4):
        for p in model.blocks[i].parameters():
            p.requires_grad = False
    
    return model


def get_optimizers_and_schedulers(model, total_steps):
    head_params = []
    bld_params = [] # attn_pool in our model
    blocks_params = []
    tokens_params = []
    other_params = [] # To catch any stray trainable parameters

    num_trainable_blocks = 4
    trainable_block_names = [f"blocks.{len(model.blocks) - i - 1}" for i in range(num_trainable_blocks)]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip frozen layers

        if name.startswith('head'):
            head_params.append(param)
        elif name.startswith('attn_pool'):
            bld_params.append(param)
        elif any(name.startswith(b_name) for b_name in trainable_block_names):
            blocks_params.append(param)
            
        # the cls_token and pos_embed are top-level parameters in timm ViT models
        elif name in ['cls_token', 'pos_embed']:
            tokens_params.append(param)
        else:
            other_params.append(param)


    param_groups = {
        'head': head_params,
        'bld': bld_params,
        'blocks': blocks_params,
        'tokens': tokens_params
    }

    # --- 2. Create Optimizers ---
    lr_mult = 0.8  # From the config
    base_lr = MODEL_CFG.LR # From the config
    weight_decay = 0.01

    optimizers = {
        'head': torch.optim.AdamW(param_groups['head'], lr=base_lr * lr_mult, weight_decay=weight_decay),
        'bld': torch.optim.AdamW(param_groups['bld'], lr=base_lr, weight_decay=weight_decay),
        'blocks': torch.optim.AdamW(param_groups['blocks'], lr=base_lr, weight_decay=weight_decay),
        'tokens': torch.optim.AdamW(param_groups['tokens'], lr=base_lr * lr_mult, weight_decay=weight_decay)
    }

    # --- 3. Create Schedulers ---
    schedulers = {
        'head': torch.optim.lr_scheduler.OneCycleLR(
            optimizers['head'], max_lr=1e-4, total_steps=total_steps,
            pct_start=0.2, anneal_strategy='cos', div_factor=1e1, final_div_factor=1e1
        ),
        'bld': torch.optim.lr_scheduler.OneCycleLR(
            optimizers['bld'], max_lr=9e-5, total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos', div_factor=1e7, final_div_factor=1e1
        ),
        'blocks': torch.optim.lr_scheduler.OneCycleLR(
            optimizers['blocks'], max_lr=1e-4, total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos', div_factor=1e7, final_div_factor=1e1
        ),
        'tokens': torch.optim.lr_scheduler.OneCycleLR(
            optimizers['tokens'], max_lr=1e-4, total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos', div_factor=1e7, final_div_factor=1e1
        )
    }

    # Return lists for easier iteration in the training loop
    return list(optimizers.values()), list(schedulers.values())


from torchvision.transforms import v2
from timm.data.mixup import Mixup

train_transform = lambda augs=False: transforms.Compose([i for i in [
    v2.Resize(MODEL_CFG.IMAGE_SIZE),
    # v2.RandomCrop(MODEL_CFG.IMAGE_SIZE, padding=4, padding_mode='reflect') if augs else None,
    v2.RandomChoice([v2.Grayscale(), v2.GaussianBlur(13)], p=[0.35, 0.4]) if augs else None,
    v2.RandomApply([v2.JPEG((14, 60))], p=0.3) if augs else None,
    v2.RandomHorizontalFlip(p=0.5) if augs else None,
    v2.TrivialAugmentWide() if augs else None,
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
] if i])

val_transform = lambda: transforms.Compose([
    v2.Resize(MODEL_CFG.IMAGE_SIZE),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mixup = v2.CutMix(num_classes=NUM_CLASSES, alpha=1.0)
# mixup = v2.MixUp(num_classes=NUM_CLASSES, alpha=0.85)
# cutmix_or_mixup = v2.RandomChoice([cutmix, mixup], p=[0.25, 0.75])

# mixup = Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0, mode='batch', num_classes=NUM_CLASSES)


def train_epoch(model, dataloader, criterion, optimizers, schedulers, scaler, epoch:int, is_mixup:bool=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for i, (real_images, original_images, real_labels) in enumerate(pbar):
        # if (rr:=random.random()) < 0.25:
        # images, labels = original_images, real_labels
        # else:
        if is_mixup:
            images, labels = mixup(real_images, real_labels)
        else:
            images, labels = original_images, real_labels

        images, labels = images.to(device), labels.to(device)
        
        with autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        for opt in optimizers:
            scaler.step(opt)
            opt.zero_grad()
        scaler.update() 

        for sch in schedulers:
            sch.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(real_labels.numpy())

        pbar.set_postfix(loss=loss.item())

    lrs = [sch.get_last_lr()[0] for sch in schedulers]
    print(
        f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}"
        f"\nLRs -> Head: {lrs[0]:.2e}, Bld/Pool: {lrs[1]:.2e}, Blocks: {lrs[2]:.2e}, Tokens: {lrs[3]:.2e}")

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1


def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i, (images, original_images, labels) in enumerate(tqdm(dataloader, desc='Validation')):
            images, labels = original_images.to(device), labels.to(device)
            
            with autocast(device_type="cuda"):
                outputs = model(original_images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1, np.array(all_probs), np.array(all_labels)


print(f"Fintuning {MODEL_CFG()}")
print(f"Length of train images: {len(train_df)}")


from sklearn.model_selection import train_test_split

def mixup_criterion(logits, soft_labels):
    kldivloss = nn.KLDivLoss(reduction='batchmean')
    
    return kldivloss(nn.functional.log_softmax(logits, dim=-1), soft_labels)

def stratified_train(BATCH_SIZE, EPOCHS, NUM_FOLDS, mixup: bool = True):
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    fold_models = []
    oof_predictions = np.zeros((len(train_df), NUM_CLASSES))
    oof_labels = np.zeros(len(train_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['encoded_label'])):
        print(f'\n{"="*50}')
        print(f'Fold {fold + 1}/{NUM_FOLDS}')
        print(f'{"="*50}')
        
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = SheepDatasetV2(train_data, TRAIN_DIR, train_transform())
        val_dataset = SheepDatasetV2(val_data, TRAIN_DIR, val_transform())
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        if mixup:
            criterion = mixup_criterion
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.073)
        model, best_model_state, best_val_f1 = train(train_loader, val_loader, criterion, EPOCHS, fold, mixup=mixup)

        # Get OOF predictions for best model
        model.load_state_dict(best_model_state)
        model.to(device)
        _, _, best_val_probs, best_val_labels = validate_epoch(model, val_loader, nn.CrossEntropyLoss(label_smoothing=0.073))
        model.to("cpu")
        oof_predictions[val_idx] = best_val_probs
        oof_labels[val_idx] = best_val_labels
        fold_models.append(model)
    
        print(f'\nFold {fold + 1} Best Val F1: {best_val_f1:.4f}')

    # Calculate OOF predictions
    oof_preds = np.argmax(oof_predictions, axis=1)
 
    # Overall OOF F1 score
    oof_f1 = f1_score(oof_labels, oof_preds, average='macro')
    print(f'\nOverall OOF F1 Score: {oof_f1:.4f}')
    return fold_models


def normal_train(BATCH_SIZE, EPOCHS, model=None, mixup: bool = True, augs: bool = True):
    train_data, val_data = train_test_split(train_df, test_size=0.2109, random_state=42, stratify=train_df['encoded_label'])
    print(len(train_data))
    train_dataset = SheepDatasetV2(train_data, TRAIN_DIR, train_transform(augs=augs))
    val_dataset = SheepDatasetV2(val_data, TRAIN_DIR, val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    if mixup:
        criterion = mixup_criterion
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.073)
    return train(train_loader, val_loader, criterion, EPOCHS, model=model, mixup=mixup)
    


def train(train_loader, val_loader, criterion, EPOCHS, fold=0, model=None, mixup: bool = False):
    if model is None:
        model = create_model()
    optimizers, schedulers = get_optimizers_and_schedulers(model, total_steps=len(train_loader) * EPOCHS)
        
    model = nn.DataParallel(model).to(device)

    scaler = GradScaler()
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizers, schedulers, scaler, epoch, mixup)
    
        # Validate
        val_loss, val_f1, val_probs, val_true_labels = validate_epoch(model, val_loader, nn.CrossEntropyLoss(label_smoothing=0.073))
    
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        print("GPU MEM USAGE:", torch.cuda.memory_allocated() / 1024**2 / 1024)
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
    
            # Save best model to disk
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }, f'best_model_fold_{fold}.pth')
    
    model.to("cpu")
    return model, best_model_state, best_val_f1


BATCH_SIZE = 32 # totall images should be multiple of batch_size; int(ttl_images * 0.7891) / 36 = 18 ✔️
EPOCHS = MODEL_CFG.EPOCHS
NUM_FOLDS = MODEL_CFG.NUM_FOLDS


import gc; gc.collect()
torch.cuda.empty_cache()


# model, best_model_state, best_f1_val = normal_train(BATCH_SIZE, 5, mixup=True, augs=True)
# print(f"Best F1 on validation set: {best_f1_val:.4f}")
# print("First round training completed. Now training without mixup...")
# model.load_state_dict(best_model_state)
model, best_model_state, best_f1_val = normal_train(BATCH_SIZE, 20, mixup=False, augs=True)
model.load_state_dict(best_model_state)


def create_submission(predicted_labels, avg_predictions, filename):
    submission_df = pd.DataFrame({
        'filename': test_files,
        'label': predicted_labels
    })
    
    submission_df.to_csv(f'submission-{filename}.csv' if 'submission' not in filename else filename+'.csv', index=False)
    print(f'\nSubmission saved! Shape: {submission_df.shape}')
    print(submission_df.head())
    
    test_pred_df = pd.DataFrame({
        'filename': test_files,
        'pred_label': predicted_labels
    })
    
    for i, class_name in enumerate(label_encoder.classes_):
        test_pred_df[f'prob_{class_name}'] = avg_predictions[:, i]

    test_pred_df.to_csv(f'{filename}-with-props.csv', index=False)
    print(f'\nTest predictions with probabilities saved to {filename}-with-props.csv')

def     test_inference(model, filename):
    test_dataset = SheepDatasetV2(test_files, TEST_DIR, transform=val_transform(), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print('\nGenerating test predictions...')

    model.to(device)

    model.eval()
    predictions = []

    with torch.no_grad():
        for images, orig, _ in tqdm(test_loader, desc=f'Test Inference'):
            images = images.to(device)
            original_images = orig.to(device)
            outputs = model(original_images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions.append(probabilities.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    final_predictions = np.argmax(predictions, axis=1)

    predicted_labels = label_encoder.inverse_transform(final_predictions)
    create_submission(predicted_labels, predictions, filename)

test_inference(model, "first-round-pesudo-label")


def get_top_two_probs(row):
    prob_cols = ['prob_Barbari', 'prob_Goat', 'prob_Harri', 'prob_Naeimi', 
                 'prob_Najdi', 'prob_Roman', 'prob_Sawakni']
    probs = {col.replace('prob_', ''): row[col] for col in prob_cols}
    
    top_two = sorted(probs.values(), reverse=True)[:2]
    return top_two[0], top_two[1] 

def remove_naeimi_images(df, fraction=0.25):
    naeimi_df = df[df['label'] == 'Naeimi']
    naeimi_to_remove = naeimi_df.sample(frac=fraction, random_state=42)
    
    final_df = pd.concat([df[df['label'] != 'Naeimi'], naeimi_df.drop(naeimi_to_remove.index)])
    
    return final_df

pesudo_labels = pd.read_csv("first-round-pesudo-label-with-props.csv")
pesudo_labels[['top1_prob', 'top2_prob']] = pesudo_labels.apply(
    lambda row: pd.Series(get_top_two_probs(row)), axis=1
)
pesudo_labels = pesudo_labels[pesudo_labels.apply(
    lambda row: row['top1_prob'] / row['top2_prob'] > 2.8, 
    axis=1
)].rename(columns={'pred_label': 'label'})
pesudo_labels['encoded_label'] = label_encoder.fit_transform(pesudo_labels['label'])
print(f"confident pesudo labels: {len(pesudo_labels)}")

train_df = remove_naeimi_images(pd.read_csv(TRAIN_CSV), 0.1)
# train_df = pd.read_csv(TRAIN_CSV)
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
train_df = pd.concat([train_df, pesudo_labels[['filename', 'label', 'encoded_label']]], axis=0, ignore_index=True)
print(len(train_df))


# offline second round training on pesudo labels and real labels
model, best_model_state, best_f1_val = normal_train(BATCH_SIZE, 16, mixup=True, augs=True)
print(f"Best F1 on validation set: {best_f1_val:.4f}")
print("First round training completed. Now training without mixup...")
model.load_state_dict(best_model_state)
model, best_model_state, best_f1_val = normal_train(BATCH_SIZE, 20, mixup=False, model=model.module, augs=True)
model.load_state_dict(best_model_state)
test_inference(model, "submission")




