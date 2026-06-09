import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
import timm
import joblib


# --- 1. Main Configuration ---
BASE_PATH = '/kaggle/input/2025-bamboo-summer-competiton-dl-pr/'
class CFG:
    # Paths
    TRAIN_CSV_PATH = BASE_PATH + 'train.csv'
    TEST_CSV_PATH = BASE_PATH + 'test.csv'
    SAMPLE_CSV_PATH = BASE_PATH + 'sample_submission'
    IMAGE_DIR_TRAIN = BASE_PATH + 'train'
    IMAGE_DIR_TEST = BASE_PATH + 'test'
    
    # Model & Training settings
    MODEL_NAME = 'tf_efficientnet_b5_ns' # A very powerful Noisy-Student model
    IMG_SIZE = 456 # Native image size for B5
    BATCH_SIZE = 16 # Reduce batch size for the larger model
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    LABEL_SMOOTHING = 0.1
    N_SPLITS = 5
    
    # Other settings
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2


# --- 2. Function for reproducibility ---
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.SEED)


# --- 3. Load data and encode labels ---
train_df = pd.read_csv(CFG.TRAIN_CSV_PATH)
test_df = pd.read_csv(CFG.TEST_CSV_PATH)

le = LabelEncoder()
train_df['label_enc'] = le.fit_transform(train_df['label'])
joblib.dump(le, 'label_encoder.pkl')
CFG.NUM_CLASSES = len(le.classes_)


# --- 4. Define the Dataset class ---
class ButterflyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.df.loc[index, 'filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image
        label = self.df.loc[index, 'label_enc']
        return image, torch.tensor(label, dtype=torch.long)


# --- 5. Define Image Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- 6. K-Fold Cross-Validation Training ---
skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)
model_paths = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label_enc'])):
    print(f"========== Fold {fold+1}/{CFG.N_SPLITS} ==========")
    
    # Create fold-specific data
    train_fold_df = train_df.iloc[train_idx]
    val_fold_df = train_df.iloc[val_idx]

    train_dataset = ButterflyDataset(train_fold_df, CFG.IMAGE_DIR_TRAIN, train_transform)
    val_dataset = ButterflyDataset(val_fold_df, CFG.IMAGE_DIR_TRAIN, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

    # Create a new model for each fold
    model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.NUM_CLASSES)
    model.to(CFG.DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    model_path = f'best_model_fold_{fold+1}.pth'

    for epoch in range(CFG.EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(CFG.DEVICE), labels.to(CFG.DEVICE)
            
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(CFG.DEVICE), labels.to(CFG.DEVICE)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        print(f"Fold {fold+1} | Epoch {epoch+1} | Val Acc: {val_acc:.5f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Best model for fold {fold+1} saved with accuracy: {best_acc:.5f}")
    
    model_paths.append(model_path)


# --- 7. Ensemble Prediction ---
print("\nStarting ensemble prediction on test data...")
test_dataset = ButterflyDataset(test_df, CFG.IMAGE_DIR_TEST, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

all_fold_preds = []
for path in model_paths:
    print(f"Loading model: {path}")
    model = timm.create_model(CFG.MODEL_NAME, pretrained=False, num_classes=CFG.NUM_CLASSES)
    model.load_state_dict(torch.load(path))
    model.to(CFG.DEVICE)
    model.eval()

    fold_probs = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc=f"Predicting with {path}"):
            imgs = imgs.to(CFG.DEVICE)
            
            with torch.cuda.amp.autocast():
                # TTA
                outputs1 = model(imgs)
                outputs2 = model(torch.flip(imgs, [3]))
            
            # Average probabilities from TTA
            avg_outputs = (torch.softmax(outputs1, 1) + torch.softmax(outputs2, 1)) / 2
            fold_probs.append(avg_outputs.cpu().numpy())
    
    all_fold_preds.append(np.vstack(fold_probs))

# Average the probabilities across all folds
final_avg_probs = np.mean(all_fold_preds, axis=0)
final_preds = np.argmax(final_avg_probs, axis=1)


# --- 8. Create Submission File ---
decoded_labels = le.inverse_transform(final_preds)
submission = pd.DataFrame({'filename': test_df['filename'], 'label': decoded_labels})
submission.to_csv('submission.csv', index=False)

print("\nsubmission.csv has been created successfully!")




