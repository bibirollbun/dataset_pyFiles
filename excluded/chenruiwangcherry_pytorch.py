
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ==========================================
# é…�ç½®ä¸�è·¯å¾„
# ==========================================
CONFIG = {
    "seed": 42,
    "img_size": 224,  # æˆ‘ä»¬åœ¨é¢„å¤„ç�†é˜¶æ®µå°± Resize åˆ°è¿™ä¸ªå°ºå¯¸
    "batch_size": 32,
    "epochs": 40,
    "lr": 1e-5,
    "num_classes": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # å�Ÿå§‹è·¯å¾„
    "csv_path": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "raw_img_dir": "/kaggle/input/aptos2019-blindness-detection/train_images",
    # ã€�æ–°ã€‘å¤„ç�†å��çš„å›¾ç‰‡ä¿�å­˜è·¯å¾„ (Kaggle è¾“å‡ºç›®å½•)
    "processed_img_dir": "/kaggle/working/processed_images_clahe_224"
}

# è®¾ç½®éš�æœºç§�å­�
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"Using device: {CONFIG['device']}")

from concurrent.futures import ProcessPoolExecutor
import functools

# ==========================================
# ç¬¬ä¸€éƒ¨åˆ†ï¼šç¦»çº¿é¢„å¤„ç�† (Offline Preprocessing) - å¤šè¿›ç¨‹åŠ é€Ÿç‰ˆ
# ==========================================

def process_one_image(img_name):
    """
    å�•ä¸ªå›¾ç‰‡çš„å¤„ç�†å‡½æ•°ï¼Œç”¨äº�å¤šè¿›ç¨‹è°ƒç”¨
    """
    load_path = os.path.join(CONFIG['raw_img_dir'], img_name + ".png")
    save_path = os.path.join(CONFIG['processed_img_dir'], img_name + ".png")
    
    # 1. è¯»å�–å�Ÿå§‹å›¾ç‰‡
    image = cv2.imread(load_path)
    if image is None:
        return # è¯»å�–å¤±è´¥ç›´æ�¥è·³è¿‡
        
    # 2. æ��å‰� Resize
    image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
    
    # 3. æ‰§è¡Œ CLAHE
    # æ³¨æ„�ï¼šåœ¨å‡½æ•°å†…éƒ¨åˆ›å»º CLAHE å¯¹è±¡ï¼Œç¡®ä¿�çº¿ç¨‹å®‰å…¨
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l) # å�ªå¢�å¼ºäº®åº¦é€šé�“
    merged = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    # 4. ä¿�å­˜ (è½¬å›� BGR)
    save_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_image_bgr)

def preprocess_and_save_images_parallel():
    # æ£€æŸ¥æ˜¯å�¦å·²å­˜åœ¨
    if os.path.exists(CONFIG['processed_img_dir']) and len(os.listdir(CONFIG['processed_img_dir'])) > 100:
        print(f"æ£€æµ‹åˆ° {CONFIG['processed_img_dir']} å·²æœ‰æ•°æ�®ï¼Œè·³è¿‡é¢„å¤„ç�†æ­¥éª¤...")
        return

    os.makedirs(CONFIG['processed_img_dir'], exist_ok=True)
    df = pd.read_csv(CONFIG['csv_path'])
    image_ids = df['id_code'].tolist()
    
    print(f"ğŸš€ å¼€å§‹å¤šè¿›ç¨‹åŠ é€Ÿé¢„å¤„ç�† (æ€»æ•°: {len(image_ids)})...")
    
    # ä½¿ç”¨ ProcessPoolExecutor è‡ªåŠ¨åˆ©ç”¨æ‰€æœ‰ CPU æ ¸å¿ƒ
    # max_workers=None ä¼šè‡ªåŠ¨è®¾ç½®ä¸º CPU æ ¸å¿ƒæ•° (Kaggle ä¸Šé€šå¸¸æ˜¯ 4)
    with ProcessPoolExecutor() as executor:
        # ä½¿ç”¨ tqdm æ˜¾ç¤ºè¿›åº¦æ�¡
        list(tqdm(executor.map(process_one_image, image_ids), total=len(image_ids), desc="Parallel Processing"))
        
    print("âœ… é¢„å¤„ç�†å®Œæˆ�ï¼�æ‰€æœ‰å¢�å¼ºå��çš„å›¾ç‰‡å·²ä¿�å­˜ã€‚")

# --- æ‰§è¡ŒåŠ é€Ÿç‰ˆé¢„å¤„ç�† ---
if __name__ == '__main__':
    preprocess_and_save_images_parallel()


# ==========================================
# ç¬¬äºŒéƒ¨åˆ†ï¼šæ•°æ�®åŠ è½½ä¸�è®­ç»ƒ (Training Pipeline)
# ==========================================

# 1. å®šä¹‰ Dataset (ç�°åœ¨å®ƒé��å¸¸è½»é‡�çº§ï¼Œå�ªè´Ÿè´£è¯»å›¾)
class ProcessedDRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        # æ³¨æ„�ï¼šè¿™é‡Œæˆ‘ä»¬è¯»å�–çš„æ˜¯ã€�æ–°ç›®å½•ã€‘ä¸‹çš„å›¾ç‰‡
        img_path = os.path.join(self.img_dir, img_name + ".png")
        label = self.df.iloc[idx]['diagnosis']

        # è¯»å�–å›¾ç‰‡ (å·²ç»�æ˜¯ 224x224 ä¸”å�šè¿‡ CLAHE çš„äº†)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # è½¬ RGB
        
        # PyTorch Transforms (ä¸»è¦�æ˜¯ Augmentation å’Œ Normalization)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# 2. æ•°æ�®å‡†å¤‡
df = pd.read_csv(CONFIG['csv_path'])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG['seed'], stratify=df['diagnosis'])

# è®¡ç®—ç±»åˆ«æ�ƒé‡� (ä¿�æŒ�è¿™ä¸€æ­¥ï¼Œè¿™å¯¹ä¸�å¹³è¡¡æ•°æ�®è‡³å…³é‡�è¦�)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_df['diagnosis']), 
    y=train_df['diagnosis']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(CONFIG['device'])

# å®šä¹‰ Transforms
# æ³¨æ„�ï¼šä¸�éœ€è¦�å†� Resize äº†ï¼Œå› ä¸ºé¢„å¤„ç�†æ—¶å·²ç»� Resize è¿‡äº†
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# å®�ä¾‹åŒ– DataLoader
train_dataset = ProcessedDRDataset(train_df, CONFIG['processed_img_dir'], transform=train_transforms)
val_dataset = ProcessedDRDataset(val_df, CONFIG['processed_img_dir'], transform=val_transforms)

# num_workers å�¯ä»¥è®¾ä¸º 2 æˆ– 4ï¼Œç�°åœ¨ CPU å�‹åŠ›å¾ˆå°�ï¼Œä¸»è¦�è´Ÿè´£ä»�ç¡¬ç›˜æ�¬è¿�æ•°æ�®
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

# 3. æ¨¡å�‹æ�„å»º (ResNet50)
def build_model(num_classes):
    print("æ­£åœ¨åŠ è½½ DenseNet121 (æ¯” ResNet50 æ›´é€‚å�ˆåŒ»å­¦å½±åƒ�)...")
    # åŠ è½½é¢„è®­ç»ƒçš„ DenseNet121
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    # è�·å�–åˆ†ç±»å±‚çš„è¾“å…¥ç‰¹å¾�æ•°
    num_ftrs = model.classifier.in_features
    
    # æ›¿æ�¢æœ€å��çš„å…¨è¿�æ�¥å±‚
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),  # é˜²æ­¢è¿‡æ‹Ÿå�ˆ
        nn.Linear(num_ftrs, num_classes)
    )
    return model

model = build_model(CONFIG['num_classes'])
model = model.to(CONFIG['device'])

# 4. ä¼˜åŒ–å™¨ä¸� Loss
criterion = nn.CrossEntropyLoss(weight=class_weights) # ä½¿ç”¨ç±»åˆ«æ�ƒé‡�
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# 5. è®­ç»ƒå¾ªç�¯
best_acc = 0.0
patience = 7
counter = 0

train_history = {'loss': [], 'acc': []}
val_history = {'loss': [], 'acc': []}

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    
    # === Training ===
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # è¿™é‡Œçš„ tqdm ä¼šè·‘å¾—å¿«å¾—å¤š
    train_bar = tqdm(train_loader, desc="Training")
    for images, labels in train_bar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        train_bar.set_postfix(loss=loss.item(), acc=correct/total)
    
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = correct / total
    train_history['loss'].append(epoch_train_loss)
    train_history['acc'].append(epoch_train_acc)
    
    # === Validation ===
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_acc = val_correct / val_total
    val_history['loss'].append(epoch_val_loss)
    val_history['acc'].append(epoch_val_acc)
    
    print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
    scheduler.step(epoch_val_loss)
    
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), "best_resnet50_offline.pth")
        print(f"ğŸ”¥ New Best Model Saved! Accuracy: {best_acc:.4f}")
        counter = 0
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter} out of {patience}")
        if counter >= patience:
            print("ğŸ›‘ Early stopping triggered.")
            break

# ==========================================
# ç»“æ�œå±•ç¤º
# ==========================================
model.load_state_dict(torch.load("best_resnet50_offline.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(CONFIG['device'])
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif']))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Offline Preprocessed)')
plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ==========================================
# é…�ç½®ä¸�è·¯å¾„
# ==========================================
CONFIG = {
    "seed": 42,
    "img_size": 224, 
    "batch_size": 32,
    "epochs": 40,
    "lr": 2e-4,  # å›�å½’ä»»åŠ¡é€šå¸¸å�¯ä»¥æ�¥å�—ç¨�å¾®å¤§ä¸€ç‚¹çš„åˆ�å§‹å­¦ä¹ ç�‡
    "num_classes": 1,  # ã€�ä¿®æ”¹ã€‘å›�å½’æ¨¡å¼�ï¼šè¾“å‡º1ä¸ªæ•°å€¼
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "csv_path": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "raw_img_dir": "/kaggle/input/aptos2019-blindness-detection/train_images",
    "processed_img_dir": "/kaggle/working/processed_images_clahe_224"
}

# è®¾ç½®éš�æœºç§�å­�
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"Using device: {CONFIG['device']}")

# ==========================================
# ç¬¬ä¸€éƒ¨åˆ†ï¼šç¦»çº¿é¢„å¤„ç�† (å¤šè¿›ç¨‹åŠ é€Ÿç‰ˆ)
# ==========================================

def process_one_image(img_name):
    load_path = os.path.join(CONFIG['raw_img_dir'], img_name + ".png")
    save_path = os.path.join(CONFIG['processed_img_dir'], img_name + ".png")
    
    image = cv2.imread(load_path)
    if image is None: return 
        
    image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
    
    # CLAHE å¢�å¼º
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    save_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_image_bgr)

def preprocess_and_save_images_parallel():
    if os.path.exists(CONFIG['processed_img_dir']) and len(os.listdir(CONFIG['processed_img_dir'])) > 100:
        print(f"æ£€æµ‹åˆ°å·²é¢„å¤„ç�†æ•°æ�®ï¼Œè·³è¿‡...")
        return

    os.makedirs(CONFIG['processed_img_dir'], exist_ok=True)
    df = pd.read_csv(CONFIG['csv_path'])
    image_ids = df['id_code'].tolist()
    
    print(f"ğŸš€ å¼€å§‹å¤šè¿›ç¨‹é¢„å¤„ç�†...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_one_image, image_ids), total=len(image_ids), desc="Parallel Preprocessing"))
    print("âœ… é¢„å¤„ç�†å®Œæˆ�")

if __name__ == '__main__':
    preprocess_and_save_images_parallel()


# ==========================================
# ç¬¬äºŒéƒ¨åˆ†ï¼šæ•°æ�®åŠ è½½ä¸�æ¨¡å�‹ (Regression Mode)
# ==========================================

class ProcessedDRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, img_name + ".png")
        label = self.df.iloc[idx]['diagnosis']

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        # ã€�ä¿®æ”¹ã€‘è¿™é‡Œè¿”å›� float ç±»å�‹ï¼Œç”¨äº�å›�å½’è®¡ç®—è·�ç¦»
        return image, torch.tensor(label, dtype=torch.float)

# æ•°æ�®å‡†å¤‡
df = pd.read_csv(CONFIG['csv_path'])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG['seed'], stratify=df['diagnosis'])

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), # å¢�åŠ å�‚ç›´ç¿»è½¬
    transforms.RandomRotation(20),   
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # è½»å¾®é¢œè‰²æŠ–åŠ¨
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ProcessedDRDataset(train_df, CONFIG['processed_img_dir'], transform=train_transforms)
val_dataset = ProcessedDRDataset(val_df, CONFIG['processed_img_dir'], transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

# ã€�ä¿®æ”¹ã€‘æ�„å»ºå›�å½’æ¨¡å�‹
def build_model():
    print("æ­£åœ¨åŠ è½½ DenseNet121 (Regression Mode)...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    
    # è¾“å‡ºå±‚æ”¹ä¸º 1 ä¸ªèŠ‚ç‚¹ï¼Œç›´æ�¥é¢„æµ‹ 0.0 ~ 4.0 çš„æ•°å€¼
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1) 
    )
    return model

model = build_model()
model = model.to(CONFIG['device'])

# ã€�ä¿®æ”¹ã€‘Loss æ”¹ä¸º MSELoss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ==========================================
# ç¬¬ä¸‰éƒ¨åˆ†ï¼šè®­ç»ƒå¾ªç�¯ (Regression Logic)
# ==========================================

best_acc = 0.0
patience = 8
counter = 0

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    
    # === Training ===
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_bar = tqdm(train_loader, desc="Training")
    for images, labels in train_bar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        # ã€�ä¿®æ”¹ã€‘Reshape æ ‡ç­¾ä»¥åŒ¹é…�è¾“å‡º (batch_size, 1)
        labels = labels.view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # ã€�ä¿®æ”¹ã€‘è®¡ç®—å‡†ç¡®ç�‡ï¼šæˆªæ–­ -> å››èˆ�äº”å…¥ -> è½¬æ•´æ•°
        with torch.no_grad():
            preds_clipped = torch.clamp(outputs, 0, 4)
            predicted = torch.round(preds_clipped)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_bar.set_postfix(mse_loss=loss.item(), acc=correct/total)
    
    epoch_train_loss = running_loss / len(train_loader)
    
    # === Validation ===
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            labels = labels.view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            
            # å�Œæ ·çš„è½¬æ�¢é€»è¾‘
            preds_clipped = torch.clamp(outputs, 0, 4)
            predicted = torch.round(preds_clipped)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_acc = val_correct / val_total
    
    print(f"Val MSE Loss: {epoch_val_loss:.4f} | Val Accuracy: {epoch_val_acc:.4f}")
    
    scheduler.step(epoch_val_loss)
    
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), "best_densenet_regression.pth")
        print(f"ğŸ”¥ New Best Model Saved! Accuracy: {best_acc:.4f}")
        counter = 0
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter} out of {patience}")
        if counter >= patience:
            print("ğŸ›‘ Early stopping triggered.")
            break

# ==========================================
# ç¬¬å››éƒ¨åˆ†ï¼šç»“æ�œå±•ç¤º (è½¬æ�¢å›�ç±»åˆ«)
# ==========================================
print("\nLoading Best Model for Evaluation...")
model.load_state_dict(torch.load("best_densenet_regression.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(CONFIG['device'])
        outputs = model(images)
        
        # å°†è¿�ç»­é¢„æµ‹å€¼è½¬å›�ç¦»æ•£ç±»åˆ«
        preds_clipped = torch.clamp(outputs, 0, 4)
        preds_int = torch.round(preds_clipped).long()
        
        all_preds.extend(preds_int.cpu().numpy().flatten()) # flatten å±•å¹³æ•°ç»„
        all_labels.extend(labels.long().numpy())

print("\n=== Classification Report (Regression Based) ===")
print(classification_report(all_labels, all_preds, target_names=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif']))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Regression Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label (Rounded)')
plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ==========================================
# æ ¸å¿ƒé…�ç½® (å…³é”®æ”¹åŠ¨åŒº)
# ==========================================
CONFIG = {
    "seed": 2025,
    # ã€�æ”¹åŠ¨1ã€‘åˆ†è¾¨ç�‡å¤§å¹…æ��å�‡ï¼Œè¿™æ˜¯è¯†åˆ«å¾®å°�ç—…ç�¶çš„å…³é”®
    "img_size": 300,  
    # å¦‚æ�œæ˜¾å­˜ä¸�å¤Ÿ (OOM)ï¼Œè¯·æŠŠ batch_size è°ƒå°�åˆ° 16 æˆ– 8
    "batch_size": 16, 
    "epochs": 35,
    "lr": 1e-4,
    "num_classes": 1,  # å›�å½’æ¨¡å¼�
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "csv_path": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "raw_img_dir": "/kaggle/input/aptos2019-blindness-detection/train_images",
    # ã€�æ”¹åŠ¨2ã€‘ä¿�å­˜è·¯å¾„æ�¢ä¸ªå��å­—ï¼Œé�¿å…�å’Œä¹‹å‰�çš„æ··æ·†
    "processed_img_dir": "/kaggle/working/processed_images_clahe_300"
}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"Using device: {CONFIG['device']} | Image Size: {CONFIG['img_size']}")

# ==========================================
# 1. é«˜æ¸…é¢„å¤„ç�† (ä¿�ç•™ CLAHEï¼Œä½†å°ºå¯¸å�˜å¤§)
# ==========================================
def process_one_image(img_name):
    load_path = os.path.join(CONFIG['raw_img_dir'], img_name + ".png")
    save_path = os.path.join(CONFIG['processed_img_dir'], img_name + ".png")
    
    image = cv2.imread(load_path)
    if image is None: return 
        
    # Resize åˆ° 300x300
    image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
    
    # CLAHE å¢�å¼º (å¯¹äº�è¯†åˆ«å¾®è¡€ç®¡ç˜¤è‡³å…³é‡�è¦�)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    # è½¬ BGR ä¿�å­˜
    save_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_image_bgr)

def preprocess_parallel():
    if os.path.exists(CONFIG['processed_img_dir']) and len(os.listdir(CONFIG['processed_img_dir'])) > 100:
        print(f"æ£€æµ‹åˆ°å·²å­˜åœ¨çš„ 300px æ•°æ�®ï¼Œè·³è¿‡é¢„å¤„ç�†...")
        return

    os.makedirs(CONFIG['processed_img_dir'], exist_ok=True)
    df = pd.read_csv(CONFIG['csv_path'])
    image_ids = df['id_code'].tolist()
    
    print(f"ğŸš€ å¼€å§‹ç”Ÿæˆ� {CONFIG['img_size']}x{CONFIG['img_size']} é«˜æ¸…å¢�å¼ºæ•°æ�®...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_one_image, image_ids), total=len(image_ids)))
    print("âœ… é¢„å¤„ç�†å®Œæˆ�")

if __name__ == '__main__':
    preprocess_parallel()

# ==========================================
# 2. æ•°æ�®åŠ è½½ä¸�å¢�å¼º (ä½¿ç”¨ Cutout/CoarseDropout æ€�æƒ³)
# ==========================================
class ProcessedDRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, img_name + ".png")
        label = self.df.iloc[idx]['diagnosis']

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float)

df = pd.read_csv(CONFIG['csv_path'])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG['seed'], stratify=df['diagnosis'])

# æ›´ä¸°å¯Œçš„æ•°æ�®å¢�å¼ºï¼Œé˜²æ­¢å¤§æ¨¡å�‹è¿‡æ‹Ÿå�ˆ
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), 
    transforms.RandomRotation(360), # çœ¼åº•å›¾æ˜¯åœ†çš„ï¼Œ360åº¦æ—‹è½¬éƒ½æ²¡é—®é¢˜
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # æ¨¡æ‹Ÿä¸�å�Œè®¾å¤‡çš„æ›�å…‰
    transforms.ToTensor(),
    # EfficientNet å®˜æ–¹æ�¨è��çš„å½’ä¸€åŒ–å�‚æ•°
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ProcessedDRDataset(train_df, CONFIG['processed_img_dir'], transform=train_transforms)
val_dataset = ProcessedDRDataset(val_df, CONFIG['processed_img_dir'], transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 3. æ ¸å¿ƒæ¨¡å�‹: EfficientNet-B3
# ==========================================
def build_model():
    print("æ­£åœ¨åŠ è½½ EfficientNet-B3 (æ€§èƒ½æ€ªå…½)...")
    # EfficientNet-B3 é€‚å�ˆ 300x300 å·¦å�³çš„åˆ†è¾¨ç�‡
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    # ä¿®æ”¹ classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) # å›�å½’è¾“å‡º
    
    return model

model = build_model()
model = model.to(CONFIG['device'])

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5) # AdamW é˜²æ­¢è¿‡æ‹Ÿå�ˆ
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ==========================================
# 4. è®­ç»ƒå¾ªç�¯
# ==========================================
best_kappa = -1.0 # ä½¿ç”¨ Kappa ä½œä¸ºæœ€ä½³æ¨¡å�‹ä¿�å­˜æ ‡å‡†ï¼Œè¿™æ¯” Accuracy æ›´é� è°±
patience = 10
counter = 0

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    
    # --- Training ---
    model.train()
    running_loss = 0.0
    
    train_bar = tqdm(train_loader, desc="Training")
    for images, labels in train_bar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        labels = labels.view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_bar.set_postfix(mse=loss.item())
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            labels = labels.view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # å›�å½’ -> æ•´æ•°ç±»åˆ«
            preds_clipped = torch.clamp(outputs, 0, 4)
            preds_int = torch.round(preds_clipped).long()
            
            all_preds.extend(preds_int.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
    avg_val_loss = val_loss / len(val_loader)
    
    # è®¡ç®— Cohen's Kappa (è¿™æ˜¯æœ€ä¸¥æ ¼çš„åŒ»å­¦æŒ‡æ ‡)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    # è®¡ç®— Accuracy
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    print(f"Val MSE: {avg_val_loss:.4f} | Accuracy: {acc:.4f} | Kappa Score: {kappa:.4f}")
    
    scheduler.step(avg_val_loss)
    
    # ä¿�å­˜é€»è¾‘ï¼šæˆ‘ä»¬ä¼˜å…ˆçœ‹ Accuracy æ˜¯å�¦çª�ç ´
    if acc > 0.88 or kappa > best_kappa: 
        if kappa > best_kappa: best_kappa = kappa
        torch.save(model.state_dict(), "best_efficientnet_b3.pth")
        print(f"ğŸ”¥ Model Saved! (Acc: {acc:.4f}, Kappa: {kappa:.4f})")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("ğŸ›‘ Early stopping")
            break

# ==========================================
# 5. æœ€ç»ˆéªŒè¯�ä¸�æ··æ·†çŸ©é˜µ
# ==========================================
model.load_state_dict(torch.load("best_efficientnet_b3.pth"))
model.eval()

# è¿™é‡Œå�¯ä»¥åŠ å…¥ TTA (Test Time Augmentation) é€»è¾‘è¿›ä¸€æ­¥åˆ·åˆ†
# æš‚æ—¶å…ˆç”¨æ ‡å‡†é¢„æµ‹
final_preds = []
final_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(CONFIG['device'])
        outputs = model(images)
        preds = torch.round(torch.clamp(outputs, 0, 4)).long()
        final_preds.extend(preds.cpu().numpy().flatten())
        final_labels.extend(labels.numpy())

print(classification_report(final_labels, final_preds, target_names=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif']))

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'EfficientNet-B3 Confusion Matrix\nAccuracy: {np.mean(np.array(final_preds)==np.array(final_labels)):.4f}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. å…¨å±€é…�ç½®
# ==========================================
CONFIG = {
    "seed": 2025,
    "img_size": 300,  # é«˜æ¸…å°ºå¯¸
    "batch_size": 16, # æ˜¾å­˜ä¸�å¤Ÿè¯·æ”¹å°�
    "epochs": 25,     # 25è½®è¶³å¤Ÿäº†
    "lr": 3e-4,       # ç¨�å¾®åŠ å¤§ä¸€ç‚¹åˆ�å§‹å­¦ä¹ ç�‡
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "csv_path": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "raw_img_dir": "/kaggle/input/aptos2019-blindness-detection/train_images",
    "processed_img_dir": "/kaggle/working/processed_images_clahe_300"
}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"Using device: {CONFIG['device']}")

# ==========================================
# 2. é¢„å¤„ç�† (å¦‚æ�œæ²¡æœ‰æ–‡ä»¶åˆ™è‡ªåŠ¨è¿�è¡Œ)
# ==========================================
def process_one_image(img_name):
    load_path = os.path.join(CONFIG['raw_img_dir'], img_name + ".png")
    save_path = os.path.join(CONFIG['processed_img_dir'], img_name + ".png")
    
    image = cv2.imread(load_path)
    if image is None: return 
    
    image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

def check_and_preprocess():
    if os.path.exists(CONFIG['processed_img_dir']) and len(os.listdir(CONFIG['processed_img_dir'])) > 100:
        print("âœ… æ£€æµ‹åˆ°å·²é¢„å¤„ç�†æ•°æ�®ï¼Œè·³è¿‡ç”Ÿæˆ�æ­¥éª¤ã€‚")
        return
    
    print(f"ğŸš€ æ­£åœ¨ç”Ÿæˆ� {CONFIG['img_size']}px é«˜æ¸…æ•°æ�® (å¤šè¿›ç¨‹)...")
    os.makedirs(CONFIG['processed_img_dir'], exist_ok=True)
    df = pd.read_csv(CONFIG['csv_path'])
    image_ids = df['id_code'].tolist()
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_one_image, image_ids), total=len(image_ids)))
    print("âœ… é¢„å¤„ç�†å®Œæˆ�")

check_and_preprocess()

# ==========================================
# 3. Dataset & Model
# ==========================================
class ProcessedDRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, img_name + ".png")
        label = self.df.iloc[idx]['diagnosis']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

# åŠ è½½æ•°æ�®
from sklearn.model_selection import train_test_split
df = pd.read_csv(CONFIG['csv_path'])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG['seed'], stratify=df['diagnosis'])

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(ProcessedDRDataset(train_df, CONFIG['processed_img_dir'], transform=train_transforms), 
                          batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(ProcessedDRDataset(val_df, CONFIG['processed_img_dir'], transform=val_transforms), 
                        batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

def build_model():
    print("æ­£åœ¨åŠ è½½ EfficientNet-B3...")
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

model = build_model().to(CONFIG['device'])
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
# ä½¿ç”¨ CosineAnnealingLRï¼Œè¿™å¯¹å¾®è°ƒé��å¸¸æœ‰æ•ˆ
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

# ==========================================
# 4. è®­ç»ƒå¾ªç�¯ (Training Loop)
# ==========================================
best_kappa = -1.0
print("\nğŸ”¥ å¼€å§‹è®­ç»ƒ (Training Started)...")

for epoch in range(CONFIG['epochs']):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device']).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device']).view(-1, 1)
            outputs = model(images)
            val_preds.extend(outputs.cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())
    
    # ç®€å�•èˆ�å…¥è®¡ç®—å½“å‰� Kappa
    cur_preds = np.round(np.clip(val_preds, 0, 4)).astype(int)
    cur_kappa = cohen_kappa_score(val_labels, cur_preds, weights='quadratic')
    
    print(f"Epoch {epoch+1} | Val Kappa: {cur_kappa:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    if cur_kappa > best_kappa:
        best_kappa = cur_kappa
        torch.save(model.state_dict(), "best_efficientnet_b3.pth")
        print(f"âœ… æ¨¡å�‹å·²ä¿�å­˜! Best Kappa: {best_kappa:.4f}")
    
    scheduler.step()

# ==========================================
# 5. é˜ˆå€¼ä¼˜åŒ– (Threshold Optimization)
# ==========================================
print("\nğŸ�† è®­ç»ƒç»“æ�Ÿï¼Œå¼€å§‹å¯»æ‰¾æœ€ä½³åˆ‡å‰²é˜ˆå€¼...")

# åŠ è½½æœ€ä½³æ�ƒé‡�
model.load_state_dict(torch.load("best_efficientnet_b3.pth"))
model.eval()

# è�·å�–æ‰€æœ‰éªŒè¯�é›†é¢„æµ‹ç»“æ�œ
valid_preds = []
valid_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(CONFIG['device'])
        outputs = model(images)
        valid_preds.extend(outputs.cpu().numpy().flatten())
        valid_labels.extend(labels.numpy())

# å®šä¹‰ä¼˜åŒ–å™¨
class OptimizedRounder(object):
    def __init__(self): self.coef_ = 0
    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]: X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]: X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]: X_p[i] = 3
            else: X_p[i] = 4
        return -cohen_kappa_score(y, X_p, weights='quadratic')
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = sp.optimize.minimize(loss_partial, [0.5, 1.5, 2.5, 3.5], method='nelder-mead')
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]: X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]: X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]: X_p[i] = 3
            else: X_p[i] = 4
        return X_p

# è¿�è¡Œä¼˜åŒ–
optR = OptimizedRounder()
optR.fit(valid_preds, valid_labels)
coefficients = optR.coef_['x']
print(f"ä¼˜åŒ–å��çš„é˜ˆå€¼: {coefficients}")

# æœ€ç»ˆé¢„æµ‹
final_preds = optR.predict(valid_preds, coefficients)
final_acc = np.mean(final_preds == valid_labels)
final_kappa = cohen_kappa_score(valid_labels, final_preds, weights='quadratic')

print(f"\nğŸ�‰ æœ€ç»ˆ Accuracy: {final_acc*100:.2f}%")
print(f"ğŸ�‰ æœ€ç»ˆ Kappa Score: {final_kappa:.4f}")

# ç»˜å›¾
cm = confusion_matrix(valid_labels, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'Final Confusion Matrix (Acc: {final_acc:.4f})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. å…¨å±€é…�ç½®
# ==========================================
CONFIG = {
    "seed": 2025,
    "img_size": 300,  # é«˜æ¸…å°ºå¯¸
    "batch_size": 16, # æ˜¾å­˜ä¸�å¤Ÿè¯·æ”¹å°�
    "epochs": 25,     # 25è½®è¶³å¤Ÿäº†
    "lr": 3e-4,       # ç¨�å¾®åŠ å¤§ä¸€ç‚¹åˆ�å§‹å­¦ä¹ ç�‡
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "csv_path": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "raw_img_dir": "/kaggle/input/aptos2019-blindness-detection/train_images",
    "processed_img_dir": "/kaggle/working/processed_images_clahe_300"
}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"Using device: {CONFIG['device']}")

# ==========================================
# 2. é¢„å¤„ç�† (å¦‚æ�œæ²¡æœ‰æ–‡ä»¶åˆ™è‡ªåŠ¨è¿�è¡Œ)
# ==========================================
def process_one_image(img_name):
    load_path = os.path.join(CONFIG['raw_img_dir'], img_name + ".png")
    save_path = os.path.join(CONFIG['processed_img_dir'], img_name + ".png")
    
    image = cv2.imread(load_path)
    if image is None: return 
    
    image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

def check_and_preprocess():
    if os.path.exists(CONFIG['processed_img_dir']) and len(os.listdir(CONFIG['processed_img_dir'])) > 100:
        print("âœ… æ£€æµ‹åˆ°å·²é¢„å¤„ç�†æ•°æ�®ï¼Œè·³è¿‡ç”Ÿæˆ�æ­¥éª¤ã€‚")
        return
    
    print(f"ğŸš€ æ­£åœ¨ç”Ÿæˆ� {CONFIG['img_size']}px é«˜æ¸…æ•°æ�® (å¤šè¿›ç¨‹)...")
    os.makedirs(CONFIG['processed_img_dir'], exist_ok=True)
    df = pd.read_csv(CONFIG['csv_path'])
    image_ids = df['id_code'].tolist()
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_one_image, image_ids), total=len(image_ids)))
    print("âœ… é¢„å¤„ç�†å®Œæˆ�")

check_and_preprocess()

# ==========================================
# 3. Dataset & Model (å�«å�‡è¡¡é‡‡æ ·ä¿®å¤�)
# ==========================================
class ProcessedDRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, img_name + ".png")
        label = self.df.iloc[idx]['diagnosis']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

# åŠ è½½æ•°æ�®
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler # <--- å¼•å…¥è¿™ä¸ªç¥�å™¨

df = pd.read_csv(CONFIG['csv_path'])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG['seed'], stratify=df['diagnosis'])

# === å…³é”®ä¿®æ”¹ï¼šè®¡ç®—é‡‡æ ·æ�ƒé‡� ===
# ç›®çš„ï¼šè®©æ¯�ä¸ªç±»åˆ«åœ¨è®­ç»ƒæ—¶å‡ºç�°çš„æ¦‚ç�‡ç›¸ç­‰
class_counts = train_df['diagnosis'].value_counts().sort_index().values
sample_weights = 1.0 / class_counts
samples_weights = np.array([sample_weights[t] for t in train_df['diagnosis']])
samples_weights = torch.from_numpy(samples_weights).double()

# åˆ›å»ºé‡‡æ ·å™¨
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
# ============================

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === å…³é”®ä¿®æ”¹ï¼šåœ¨ DataLoader ä¸­å�¯ç”¨ sampler ===
# æ³¨æ„�ï¼šä½¿ç”¨äº† sampler å��ï¼Œshuffle å¿…é¡»è®¾ä¸º False (å› ä¸º sampler å·²ç»�åœ¨éš�æœºæŠ½æ ·äº†)
train_loader = DataLoader(
    ProcessedDRDataset(train_df, CONFIG['processed_img_dir'], transform=train_transforms), 
    batch_size=CONFIG['batch_size'], 
    sampler=sampler,      # <--- æ³¨å…¥é‡‡æ ·å™¨
    shuffle=False,        # <--- å¿…é¡»ä¸º False
    num_workers=2, 
    pin_memory=True
)

val_loader = DataLoader(
    ProcessedDRDataset(val_df, CONFIG['processed_img_dir'], transform=val_transforms), 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True
)

def build_model():
    print("æ­£åœ¨åŠ è½½ EfficientNet-B3 (å�‡è¡¡é‡‡æ ·ç‰ˆ)...")
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

model = build_model().to(CONFIG['device'])
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

# ==========================================
# 4. è®­ç»ƒå¾ªç�¯ (Training Loop)
# ==========================================
best_kappa = -1.0
print("\nğŸ”¥ å¼€å§‹è®­ç»ƒ (Training Started)...")

for epoch in range(CONFIG['epochs']):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device']).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device']).view(-1, 1)
            outputs = model(images)
            val_preds.extend(outputs.cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())
    
    # ç®€å�•èˆ�å…¥è®¡ç®—å½“å‰� Kappa
    cur_preds = np.round(np.clip(val_preds, 0, 4)).astype(int)
    cur_kappa = cohen_kappa_score(val_labels, cur_preds, weights='quadratic')
    
    print(f"Epoch {epoch+1} | Val Kappa: {cur_kappa:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    if cur_kappa > best_kappa:
        best_kappa = cur_kappa
        torch.save(model.state_dict(), "best_efficientnet_b3.pth")
        print(f"âœ… æ¨¡å�‹å·²ä¿�å­˜! Best Kappa: {best_kappa:.4f}")
    
    scheduler.step()

# ==========================================
# 5. é˜ˆå€¼ä¼˜åŒ– (Threshold Optimization)
# ==========================================
print("\nğŸ�† è®­ç»ƒç»“æ�Ÿï¼Œå¼€å§‹å¯»æ‰¾æœ€ä½³åˆ‡å‰²é˜ˆå€¼...")

# åŠ è½½æœ€ä½³æ�ƒé‡�
model.load_state_dict(torch.load("best_efficientnet_b3.pth"))
model.eval()

# è�·å�–æ‰€æœ‰éªŒè¯�é›†é¢„æµ‹ç»“æ�œ
valid_preds = []
valid_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(CONFIG['device'])
        outputs = model(images)
        valid_preds.extend(outputs.cpu().numpy().flatten())
        valid_labels.extend(labels.numpy())

# å®šä¹‰ä¼˜åŒ–å™¨
class OptimizedRounder(object):
    def __init__(self): self.coef_ = 0
    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]: X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]: X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]: X_p[i] = 3
            else: X_p[i] = 4
        return -cohen_kappa_score(y, X_p, weights='quadratic')
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = sp.optimize.minimize(loss_partial, [0.5, 1.5, 2.5, 3.5], method='nelder-mead')
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]: X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]: X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]: X_p[i] = 3
            else: X_p[i] = 4
        return X_p

# è¿�è¡Œä¼˜åŒ–
optR = OptimizedRounder()
optR.fit(valid_preds, valid_labels)
coefficients = optR.coef_['x']
print(f"ä¼˜åŒ–å��çš„é˜ˆå€¼: {coefficients}")

# æœ€ç»ˆé¢„æµ‹
final_preds = optR.predict(valid_preds, coefficients)
final_acc = np.mean(final_preds == valid_labels)
final_kappa = cohen_kappa_score(valid_labels, final_preds, weights='quadratic')

print(f"\nğŸ�‰ æœ€ç»ˆ Accuracy: {final_acc*100:.2f}%")
print(f"ğŸ�‰ æœ€ç»ˆ Kappa Score: {final_kappa:.4f}")

# ç»˜å›¾
cm = confusion_matrix(valid_labels, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'Final Confusion Matrix (Acc: {final_acc:.4f})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F

# ==========================================
# 1. ç»ˆæ��é…�ç½® (Dual Backbone Mode)
# ==========================================
CONFIG = {
    "seed": 2025,
    "img_size": 256,  # 256 æ˜¯å�Œæ¨¡å�‹ä¸‹æ˜¾å­˜å’Œç²¾åº¦çš„æœ€ä½³å¹³è¡¡ç‚¹
    "batch_size": 12, # å�Œæ¨¡å�‹æ˜¾å­˜å� ç”¨å¤§ï¼Œè°ƒå°� batch_size é˜²æ­¢ç‚¸æ˜¾å­˜
    "epochs": 30,     
    "lr": 1e-4,       
    "num_classes": 5, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "csv_path": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "raw_img_dir": "/kaggle/input/aptos2019-blindness-detection/train_images",
    "processed_img_dir": "/kaggle/working/processed_images_clahe_256"
}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"Using device: {CONFIG['device']}")

# ==========================================
# 2. é¢„å¤„ç�† (CLAHE + Resize)
# ==========================================
def process_one_image(img_name):
    load_path = os.path.join(CONFIG['raw_img_dir'], img_name + ".png")
    save_path = os.path.join(CONFIG['processed_img_dir'], img_name + ".png")
    
    image = cv2.imread(load_path)
    if image is None: return 
    
    image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
    
    # CLAHE å¢�å¼º
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

def check_and_preprocess():
    if os.path.exists(CONFIG['processed_img_dir']) and len(os.listdir(CONFIG['processed_img_dir'])) > 100:
        print("âœ… æ£€æµ‹åˆ°å·²é¢„å¤„ç�†æ•°æ�®ï¼Œç›´æ�¥ä½¿ç”¨ã€‚")
        return
    
    print(f"ğŸš€ æ­£åœ¨ç”Ÿæˆ� {CONFIG['img_size']}px æ•°æ�® (å¤šè¿›ç¨‹)...")
    os.makedirs(CONFIG['processed_img_dir'], exist_ok=True)
    df = pd.read_csv(CONFIG['csv_path'])
    image_ids = df['id_code'].tolist()
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_one_image, image_ids), total=len(image_ids)))
    print("âœ… é¢„å¤„ç�†å®Œæˆ�")

check_and_preprocess()

# ==========================================
# 3. Dataset (åˆ†ç±»æ¨¡å¼�)
# ==========================================
class ProcessedDRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, img_name + ".png")
        label = self.df.iloc[idx]['diagnosis']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

df = pd.read_csv(CONFIG['csv_path'])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG['seed'], stratify=df['diagnosis'])

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360), # 360åº¦æ—‹è½¬
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(ProcessedDRDataset(train_df, CONFIG['processed_img_dir'], transform=train_transforms), 
                          batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(ProcessedDRDataset(val_df, CONFIG['processed_img_dir'], transform=val_transforms), 
                        batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 4. ğŸ�† æ ¸å¿ƒæ¨¡å�‹: Dual Backbone (DenseNet + ResNet)
# ==========================================
class DualBackboneModel(nn.Module):
    def __init__(self, num_classes=5):
        super(DualBackboneModel, self).__init__()
        
        print("ğŸ’¡ æ­£åœ¨æ�„å»ºå�Œæµ�æ¨¡å�‹: DenseNet201 + ResNet50 (Strong Baseline)...")
        
        # Branch 1: DenseNet201 (æ“…é•¿ç»†èŠ‚çº¹ç�†)
        self.densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        dens_out = self.densenet.classifier.in_features # 1920
        self.densenet.classifier = nn.Identity() # ç§»é™¤åˆ†ç±»å¤´ï¼Œå�ªå�–ç‰¹å¾�
        
        # Branch 2: ResNet50 (æ“…é•¿æ•´ä½“ç»“æ�„)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        res_out = self.resnet.fc.in_features # 2048
        self.resnet.fc = nn.Identity() # ç§»é™¤åˆ†ç±»å¤´
        
        # è��å�ˆå��çš„å…¨è¿�æ�¥å±‚
        self.fusion_fc = nn.Sequential(
            nn.Linear(dens_out + res_out, 256), # 1920 + 2048 -> 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # å·¦æ‰‹ç”»åœ†
        feat1 = self.densenet(x)
        # å�³æ‰‹ç”»æ–¹
        feat2 = self.resnet(x)
        
        # å�ˆä½“ï¼�
        concat_feat = torch.cat((feat1, feat2), dim=1)
        output = self.fusion_fc(concat_feat)
        return output

model = DualBackboneModel(num_classes=5).to(CONFIG['device'])

# ==========================================
# 5. è®­ç»ƒé…�ç½® (Class Weights + Label Smoothing)
# ==========================================
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['diagnosis']), y=train_df['diagnosis'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(CONFIG['device'])

# è¿™é‡Œçš„ label_smoothing=0.1 æ˜¯æ��åˆ†ç¥�å™¨ï¼Œé˜²æ­¢æ¨¡å�‹å¯¹ Class 2 è¿‡åº¦è‡ªä¿¡
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)

# ==========================================
# 6. è®­ç»ƒå¾ªç�¯
# ==========================================
best_acc = 0.0
print("\nğŸ”¥ å¼€å§‹å�Œæµ�æ¨¡å�‹è®­ç»ƒ (Training Started)...")

for epoch in range(CONFIG['epochs']):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    # Training
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    val_acc = val_correct / val_total
    kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
    
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Kappa: {kappa:.4f}")
    
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_dual_model.pth")
        print(f"âœ… æ–°çºªå½•! Best Accuracy: {best_acc*100:.2f}%")

# ==========================================
# 7. æœ€ç»ˆç»“æ�œå±•ç¤º
# ==========================================
print("\nğŸ�† åŠ è½½æœ€ä½³å�Œæµ�æ¨¡å�‹è¿›è¡Œè¯„ä¼°...")
model.load_state_dict(torch.load("best_dual_model.pth"))
model.eval()

final_preds = []
final_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        final_preds.extend(predicted.cpu().numpy())
        final_labels.extend(labels.cpu().numpy())

print("\n=== Classification Report ===")
print(classification_report(final_labels, final_preds, target_names=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif']))

final_acc = np.mean(np.array(final_preds) == np.array(final_labels))
print(f"ğŸ�‰ æœ€ç»ˆ Accuracy: {final_acc*100:.2f}%")

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'Dual Backbone Confusion Matrix (Acc: {final_acc:.4f})')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()




import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# 1. ç²¾å‡†è·¯å¾„é…�ç½® (åŸºäº�ä½ çš„æˆªå›¾)
# ==========================================
# æ ¹ç›®å½• (æ ¹æ�®ä½ çš„æˆªå›¾æ�¨æ–­)
BASE_DIR = "/kaggle/input/resized-2015-2019-blindness-detection-images"

CONFIG = {
    "seed": 2025,
    "img_size": 224,      
    "batch_size": 32,     
    "epochs": 15,        
    "lr": 1e-4,       
    "num_classes": 5, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==========================================
# 2. æ··å�ˆæ•°æ�®åŠ è½½å™¨ (2015 + 2019)
# ==========================================
def load_mixed_data_v2():
    print("ğŸš€ æ­£åœ¨åˆ†åˆ«è¯»å�– 2015 å’Œ 2019 çš„æ•°æ�®è¡¨...")
    
    # --- 1. å¤„ç�† 2015 æ•°æ�® ---
    # ä½ çš„æˆªå›¾æ˜¾ç¤º CSV åœ¨ labels æ–‡ä»¶å¤¹ä¸‹
    path_15 = os.path.join(BASE_DIR, "labels/trainLabels15.csv")
    df_15 = pd.read_csv(path_15)
    df_15.rename(columns={'image': 'id_code', 'level': 'diagnosis'}, inplace=True)
    
    # 2015 çš„å›¾ç‰‡é€šå¸¸åœ¨ 'resized train 15' æ–‡ä»¶å¤¹é‡Œ (æˆªå›¾é‡Œæœ‰ resized test 15ï¼Œæ�¨æµ‹ train ä¹Ÿåœ¨å�Œçº§)
    # æˆ‘ä»¬å…ˆè‡ªåŠ¨æ�œç´¢ä¸€ä¸‹ 'resized train 15' åœ¨å“ªé‡Œ
    dir_15 = None
    for root, dirs, files in os.walk(BASE_DIR):
        if "resized train 15" in dirs:
            dir_15 = os.path.join(root, "resized train 15")
            break
            
    if dir_15:
        # æˆªå›¾æ˜¾ç¤ºå��ç¼€æ˜¯ .jpg
        df_15['id_code'] = df_15['id_code'].apply(lambda x: os.path.join(dir_15, f"{x}.jpg"))
        print(f"âœ… 2015 æ•°æ�®åŠ è½½æˆ�åŠŸ: {len(df_15)} å¼  (è·¯å¾„: {dir_15})")
    else:
        print("âš ï¸� è­¦å‘Š: æ²¡æ‰¾åˆ° 'resized train 15' æ–‡ä»¶å¤¹ï¼Œå�¯èƒ½æ•°æ�®é›†è§£å�‹ç»“æ�„ä¸�å�Œï¼Œè·³è¿‡ 2015 æ•°æ�®ã€‚")
        df_15 = pd.DataFrame() # ç©ºè¡¨

    # --- 2. å¤„ç�† 2019 æ•°æ�® ---
    path_19 = os.path.join(BASE_DIR, "labels/trainLabels19.csv")
    df_19 = pd.read_csv(path_19)
    # 2019 çš„åˆ—å��é€šå¸¸æ˜¯ id_code, diagnosisï¼Œä½†ä¹Ÿå�¯èƒ½ä¸�ä¸€æ ·
    if 'image' in df_19.columns: df_19.rename(columns={'image': 'id_code'}, inplace=True)
    if 'level' in df_19.columns: df_19.rename(columns={'level': 'diagnosis'}, inplace=True)
    
    # å¯»æ‰¾ 2019 å›¾ç‰‡ç›®å½•
    dir_19 = None
    for root, dirs, files in os.walk(BASE_DIR):
        if "resized train 19" in dirs:
            dir_19 = os.path.join(root, "resized train 19")
            break
            
    if dir_19:
        df_19['id_code'] = df_19['id_code'].apply(lambda x: os.path.join(dir_19, f"{x}.jpg"))
        print(f"âœ… 2019 æ•°æ�®åŠ è½½æˆ�åŠŸ: {len(df_19)} å¼  (è·¯å¾„: {dir_19})")
    else:
        # å¦‚æ�œæ‰¾ä¸�åˆ° resized train 19ï¼Œå°±å°�è¯•ç”¨å®˜æ–¹å�Ÿå§‹æ•°æ�®é›†è·¯å¾„
        print("âš ï¸� æ²¡æ‰¾åˆ° resized 2019 ç›®å½•ï¼Œå°�è¯•ä½¿ç”¨å�Ÿå§‹è·¯å¾„...")
        raw_19_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
        if os.path.exists(raw_19_dir):
            df_19['id_code'] = df_19['id_code'].apply(lambda x: os.path.join(raw_19_dir, f"{x}.png"))
            print(f"âœ… å·²å›�é€€åˆ° APTOS 2019 å�Ÿå§‹è·¯å¾„: {len(df_19)} å¼ ")

    # --- 3. å�ˆå¹¶ä¸�æ¸…æ´— ---
    df_final = pd.concat([df_15, df_19], axis=0).reset_index(drop=True)
    
    # å†�æ¬¡æ£€æŸ¥æ–‡ä»¶æ˜¯å�¦å­˜åœ¨ (é��å¸¸é‡�è¦�ï¼Œé˜²æ­¢æŠ¥é”™)
    print("æ­£åœ¨æœ€ç»ˆæ ¡éªŒæ–‡ä»¶è·¯å¾„ (è¿™å�¯èƒ½éœ€è¦�å‡ ç§’é’Ÿ)...")
    # å�ªæ£€æŸ¥å‰� 10 ä¸ªå’Œå�� 10 ä¸ªä½œä¸ºå¿«é€ŸéªŒè¯�
    valid_count = 0
    # è¿™é‡Œçš„ lambda å‡½æ•°ä¼šæ£€æŸ¥æ–‡ä»¶æ˜¯å�¦çœŸçš„åœ¨ç¡¬ç›˜ä¸Š
    df_final['exists'] = df_final['id_code'].apply(os.path.exists)
    df_final = df_final[df_final['exists']].reset_index(drop=True)
    
    print(f"âœ… æœ€ç»ˆæœ‰æ•ˆå›¾ç‰‡æ•°é‡�: {len(df_final)}")
    
    # --- 4. æ™ºèƒ½é‡‡æ · (å�ªç•™ç²¾å��) ---
    df_0 = df_final[df_final['diagnosis'] == 0]
    df_others = df_final[df_final['diagnosis'] != 0]
    
    # å�¥åº·æ ·æœ¬å¤ªå¤šäº†ï¼Œå�– 5000 å¼ å¹³è¡¡ä¸€ä¸‹
    if len(df_0) > 5000:
        df_0 = df_0.sample(n=5000, random_state=2025)
        
    df_train = pd.concat([df_0, df_others], axis=0).sample(frac=1, random_state=2025).reset_index(drop=True)
    print(f"ğŸ“‰ è®­ç»ƒé›†ç²¾ç®€å��: {len(df_train)} å¼  (åŒ…å�«æ‰€æœ‰æ‚£ç—…æ ·æœ¬ + 5000å�¥åº·æ ·æœ¬)")
    
    return df_train

# æ‰§è¡ŒåŠ è½½
df = load_mixed_data_v2()

# åˆ’åˆ†
train_df, val_df = train_test_split(df, test_size=0.1, random_state=CONFIG['seed'], stratify=df['diagnosis'])
# 4. Dataset & DataLoader
# ==========================================
class RetinopathyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['id_code']
        label = self.df.iloc[idx]['diagnosis']
        
        image = cv2.imread(img_path)
        # å®¹é”™ï¼šä¸‡ä¸€è¯»ä¸�åˆ°å›¾
        if image is None:
            # print(f"Warning: Could not read {img_path}")
            image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size'])) # ç¡®ä¿�å°ºå¯¸ç»Ÿä¸€
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# å¢�å¼ºç­–ç•¥
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(RetinopathyDataset(train_df, transform=train_transforms), 
                          batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(RetinopathyDataset(val_df, transform=val_transforms), 
                        batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 5. å�Œæµ�æ¨¡å�‹ (Dual Backbone)
# ==========================================
class DualBackboneModel(nn.Module):
    def __init__(self, num_classes=5):
        super(DualBackboneModel, self).__init__()
        print("ğŸ’¡ æ�„å»ºå�Œæµ�æ¨¡å�‹ (DenseNet201 + ResNet50)...")
        
        # Branch 1: DenseNet
        self.densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        dens_out = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        
        # Branch 2: ResNet
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        res_out = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(dens_out + res_out, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat1 = self.densenet(x)
        feat2 = self.resnet(x)
        concat_feat = torch.cat((feat1, feat2), dim=1)
        output = self.fusion_fc(concat_feat)
        return output

model = DualBackboneModel(num_classes=5).to(CONFIG['device'])

# ==========================================
# 6. è®­ç»ƒå¾ªç�¯
# ==========================================
from sklearn.utils.class_weight import compute_class_weight
# è®¡ç®— Class Weights é˜²æ­¢ä¸�å¹³è¡¡
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['diagnosis']), y=train_df['diagnosis'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(CONFIG['device'])

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

best_acc = 0.0
print(f"\nğŸ”¥ å¼€å§‹è®­ç»ƒï¼�æ€»æ•°æ�®é‡�: {len(df)} (å·²è¿‡æ»¤ç²¾å��)")

for epoch in range(CONFIG['epochs']):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    for images, labels in pbar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    val_acc = val_correct / val_total
    kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
    
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Kappa: {kappa:.4f}")
    
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_mega_model.pth")
        print(f"âœ… æ–°çºªå½•! Best Accuracy: {best_acc*100:.2f}%")

# ==========================================
# 7. ç»“æ�œå±•ç¤º
# ==========================================
print("\nğŸ�† æœ€ç»ˆè¯„ä¼°...")
model.load_state_dict(torch.load("best_mega_model.pth"))
model.eval()

final_preds = []
final_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        final_preds.extend(predicted.cpu().numpy())
        final_labels.extend(labels.cpu().numpy())

print(classification_report(final_labels, final_preds, target_names=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif']))
acc = np.mean(np.array(final_preds) == np.array(final_labels))
print(f"ğŸ�‰ æœ€ç»ˆ Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'Mega Dataset Confusion Matrix (Acc: {acc:.4f})')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 1. æ ¸å¿ƒé…�ç½®
# ==========================================
CONFIG = {
    "seed": 2025,
    "img_size": 256,      # 256 å¹³è¡¡ç‚¹
    "batch_size": 16,     # ç¨�å¾®å¤§ä¸€ç‚¹
    "epochs": 15,         # 15è½®è¶³å¤Ÿ
    "lr": 1e-4,       
    "num_classes": 5, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # ä¸¤ä¸ªæ•°æ�®é›†çš„è·¯å¾„
    "path_2015_images": "/kaggle/input/resized-2015-2019-blindness-detection-images/resized_train_images",
    "path_2015_csv": "/kaggle/input/resized-2015-2019-blindness-detection-images/labels/trainLabels15.csv",
    "path_2019_base": "/kaggle/input/aptos2019-blindness-detection"
}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])

# ==========================================
# 2. Ben's Preprocessing (æ��åˆ†ç¥�å™¨ï¼šåœ†å½¢è£�å‰ª)
# ==========================================
def crop_image_from_gray(img, tol=7):
    """
    è‡ªåŠ¨åˆ‡æ�‰çœ¼åº•å›¾å‘¨å›´çš„é»‘è‰²åŒºåŸŸï¼Œå�ªä¿�ç•™çœ¼ç�ƒ
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def circle_crop(img, sigmaX=10):
    """
    è¿›ä¸€æ­¥å¤„ç�†ï¼šæŠŠå›¾åƒ�resizeå¹¶åº”ç”¨é«˜æ–¯æ¨¡ç³Šï¼Œçª�å‡ºè¡€ç®¡
    """
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (CONFIG['img_size'], CONFIG['img_size']))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img

# ==========================================
# 3. æ•°æ�®å‡†å¤‡ï¼šåˆ†å¼€å‡†å¤‡ Train å’Œ Val
# ==========================================
def prepare_data():
    print("ğŸš€ æ­£åœ¨æ�„å»ºæ•°æ�®é›†ï¼šè®­ç»ƒé›†ç”¨æ··å�ˆæ•°æ�®ï¼ŒéªŒè¯�é›†å�ªç”¨2019æ•°æ�®...")
    
    # --- A. å‡†å¤‡ 2019 æ•°æ�® (é«˜è´¨é‡�) ---
    df_19 = pd.read_csv(os.path.join(CONFIG['path_2019_base'], "train.csv"))
    df_19['id_code'] = df_19['id_code'].apply(lambda x: os.path.join(CONFIG['path_2019_base'], "train_images", x + ".png"))
    # ç»™ 2019 æ•°æ�®æ‰“æ ‡ï¼Œæ–¹ä¾¿è¯†åˆ«
    df_19['source'] = '2019'
    
    # --- B. å‡†å¤‡ 2015 æ•°æ�® (ä½œä¸ºè¡¥å……ç²®è�‰) ---
    # æ³¨æ„�ï¼šéœ€è¦�ä½ çš„ resized æ•°æ�®é›†è·¯å¾„æ­£ç¡®
    df_15 = pd.read_csv(CONFIG['path_2015_csv'])
    df_15.rename(columns={'image': 'id_code', 'level': 'diagnosis'}, inplace=True)
    
    # è‡ªåŠ¨å¯»æ‰¾ 2015 å›¾ç‰‡ç›®å½•
    real_2015_dir = None
    base_search = "/kaggle/input/resized-2015-2019-blindness-detection-images"
    for root, dirs, files in os.walk(base_search):
        if "resized train 15" in dirs:
            real_2015_dir = os.path.join(root, "resized train 15")
            break
    
    if real_2015_dir:
        df_15['id_code'] = df_15['id_code'].apply(lambda x: os.path.join(real_2015_dir, x + ".jpg"))
        df_15['source'] = '2015'
        print(f"âœ… 2015 æ•°æ�®å°±ç»ª: {len(df_15)} å¼ ")
    else:
        print("âš ï¸� æ²¡æ‰¾åˆ° 2015 å›¾ç‰‡ç›®å½•ï¼Œä»…ä½¿ç”¨ 2019 æ•°æ�®")
        df_15 = pd.DataFrame()

    # --- C. å…³é”®åˆ’åˆ† ---
    # éªŒè¯�é›† (Validation)ï¼šå¿…é¡»å…¨éƒ¨æ�¥è‡ª 2019ï¼�è¿™æ ·æµ‹å‡ºæ�¥çš„åˆ†æ‰�æ˜¯çœŸå®�çš„ï¼�
    # æˆ‘ä»¬ä»� 2019 é‡Œåˆ‡ 20% å‡ºæ�¥å�šéªŒè¯�
    train_19, val_19 = train_test_split(df_19, test_size=0.2, random_state=CONFIG['seed'], stratify=df_19['diagnosis'])
    
    # è®­ç»ƒé›† (Training)ï¼šå‰©ä¸‹çš„ 2019 + æ‰€æœ‰çš„ 2015
    # æˆ‘ä»¬å�¯ä»¥å¯¹ 2015 è¿›è¡Œé‡‡æ ·ï¼Œä¸�è¦�å…¨ç”¨ï¼Œå�¦åˆ™è·‘å¤ªæ…¢
    if len(df_15) > 10000:
        # å�ªå�– 1.5ä¸‡å¼  2015 çš„æ•°æ�®ï¼ŒåŠ ä¸Š 3000 å¼  2019 çš„æ•°æ�®
        df_15_sample = df_15.sample(n=15000, random_state=2025)
    else:
        df_15_sample = df_15
        
    df_train_final = pd.concat([train_19, df_15_sample], axis=0).sample(frac=1).reset_index(drop=True)
    df_val_final = val_19.reset_index(drop=True)
    
    print("-" * 30)
    print(f"ğŸ“š è®­ç»ƒé›† (æ··å�ˆ): {len(df_train_final)} å¼  (2015è¾…åŠ© + 2019æ ¸å¿ƒ)")
    print(f"ğŸ“� éªŒè¯�é›† (çº¯2019): {len(df_val_final)} å¼  (è¿™å°±æ˜¯ä½ çš„çœŸå®�æˆ�ç»©å�•)")
    print("-" * 30)
    
    return df_train_final, df_val_final

train_df, val_df = prepare_data()

# ==========================================
# 4. Dataset
# ==========================================
class RetinopathyDataset(Dataset):
    def __init__(self, df, transform=None, mode='train'):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['id_code']
        label = self.df.iloc[idx]['diagnosis']
        
        image = cv2.imread(img_path)
        if image is None: # å®¹é”™
            image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # åº”ç”¨ Ben's Preprocessing (åœ†å½¢è£�å‰ª+é«˜æ–¯æ¨¡ç³Š)
            # è¿™æ­¥å�¯èƒ½ä¼šç¨�å¾®æ…¢ä¸€ç‚¹ç‚¹ï¼Œä½†å¯¹æ��åˆ†é��å¸¸å…³é”®
            try:
                image = circle_crop(image)
            except:
                image = cv2.resize(image, (CONFIG['img_size'], CONFIG['img_size']))
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(RetinopathyDataset(train_df, transform=train_transforms), 
                          batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
# éªŒè¯�é›†ä¸�åŠ¨ï¼Œä¹Ÿä¸�Shuffle
val_loader = DataLoader(RetinopathyDataset(val_df, transform=val_transforms), 
                        batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 5. æ¨¡å�‹: EfficientNet-B5 (å�•ä½“æœ€å¼º)
# ==========================================
def build_model():
    print("ğŸ’¡ åŠ è½½ EfficientNet-B5 (Pretrained)...")
    model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, CONFIG['num_classes'])
    return model

model = build_model().to(CONFIG['device'])

# ==========================================
# 6. è®­ç»ƒå¾ªç�¯
# ==========================================
# ä½¿ç”¨ Class Weights è§£å†³ä¸�å¹³è¡¡
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['diagnosis']), y=train_df['diagnosis'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(CONFIG['device'])

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

best_acc = 0.0
print("\nğŸ”¥ å¼€å§‹æœ€ç»ˆå†²åˆº (Train on Mixed, Val on 2019)...")

for epoch in range(CONFIG['epochs']):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels in pbar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    # Validation (On 2019 Only)
    model.eval()
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    val_acc = val_correct / val_total
    kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
    
    print(f"Epoch {epoch+1} | Val Acc (2019): {val_acc:.4f} | Kappa: {kappa:.4f}")
    
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model_final.pth")
        print(f"âœ… æœ€ä½³æ¨¡å�‹ä¿�å­˜! Best Acc: {best_acc:.4f}")

# ==========================================
# 7. æœ€ç»ˆè¯„ä¼°
# ==========================================
print("\nğŸ�† åŠ è½½æœ€ä½³æ¨¡å�‹è¿›è¡Œæœ€ç»ˆ TTA è¯„ä¼°...")
model.load_state_dict(torch.load("best_model_final.pth"))
model.eval()

final_preds = []
final_labels = []

# ç®€å�•çš„ TTA: å�Ÿå›¾ + æ°´å¹³ç¿»è½¬
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="TTA Testing"):
        images = images.to(CONFIG['device'])
        
        p1 = model(images)
        p2 = model(torch.flip(images, dims=[3])) # Flip horizontal
        
        avg = (p1 + p2) / 2.0
        _, predicted = torch.max(avg.data, 1)
        
        final_preds.extend(predicted.cpu().numpy())
        final_labels.extend(labels.cpu().numpy())

print(classification_report(final_labels, final_preds, target_names=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif']))
acc = np.mean(np.array(final_preds) == np.array(final_labels))
print(f"ğŸ�‰ æœ€ç»ˆ Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'Final Validation (2019 Only)\nAcc: {acc:.4f}')
plt.show()


import numpy as np
import pandas as pd
import torch
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight 

# ======================================================
# 1. è·¯å¾„é…�ç½® (è¯·æ ¹æ�®ä½ çš„å®�é™…æƒ…å†µä¿®æ”¹è¿™é‡Œï¼�ï¼�ï¼�)
# ======================================================

# ã€�æƒ…å†µ Aã€‘å¦‚æ�œä½ ç”¨çš„æ˜¯ä¸Šä¼ å¥½çš„ Dataset (æ�¨è��):
# SAVE_DIR = "/kaggle/input/ä½ çš„æ•°æ�®é›†å��å­—/processed_images" 
SAVE_DIR = "/kaggle/input/aptos-2019-preprocessed-224/processed_images"


# æ£€æŸ¥ä¸€ä¸‹è·¯å¾„å¯¹ä¸�å¯¹ï¼Œä¸�å¯¹ç›´æ�¥æŠ¥é”™ï¼Œå…�å¾—å��é�¢è®­ç»ƒç™½è´¹åŠ²
if not os.path.exists(SAVE_DIR):
    raise FileNotFoundError(f"â�Œ æ‰¾ä¸�åˆ°æ–‡ä»¶å¤¹: {SAVE_DIR}ã€‚è¯·æ£€æŸ¥è·¯å¾„è®¾ç½®ï¼�")
else:
    print(f"âœ… å›¾ç‰‡æ–‡ä»¶å¤¹å®šä½�æˆ�åŠŸ: {SAVE_DIR}")
    print(f"æ–‡ä»¶å¤¹å†…å›¾ç‰‡æ•°é‡�: {len(os.listdir(SAVE_DIR))}")

CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
BATCH_SIZE = 32

# ======================================================
# 2. å®šä¹‰ Dataset ç±» (åŠ äº†ä¸€ä¸ª print è­¦å‘Š)
# ======================================================
class RetinopathyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.dataframe.iloc[idx]
        filename = row["filename"]
        label = row["diagnosis"]
        img_path = os.path.join(self.img_dir, filename)
        
        image = cv2.imread(img_path)
        
        # --- è¿™é‡Œçš„ä¿®æ”¹ï¼šå¦‚æ�œæ˜¯ Noneï¼ŒæŠ¥é”™è€Œä¸�æ˜¯ç»™é»‘å›¾ ---
        # å�¦åˆ™ä½ å�¯èƒ½è®­ç»ƒäº†å�Šå¤©å�‘ç�°å‡†ç¡®ç�‡ä¸�æ¶¨ï¼Œå…¶å®�å…¨æ˜¯é»‘å›¾
        if image is None:
            print(f"â�Œ è­¦å‘Š: æ— æ³•è¯»å�–å›¾ç‰‡ {img_path}")
            # å�ªæœ‰å½“ä½ ç¡®å®šä¸ªåˆ«å›¾ç‰‡æ�Ÿå��æ—¶æ‰�ç”¨å…¨é»‘å›¾ï¼Œå�¦åˆ™å»ºè®®ç›´æ�¥åœ¨è¿™é‡ŒæŠ¥é”™æ£€æŸ¥
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            
        return img_tensor, torch.tensor(label, dtype=torch.long)

# ======================================================
# 3. å®šä¹‰æ•°æ�®å¢�å¼º
# ======================================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ======================================================
# 4. è¯»å�–æ•°æ�®å¹¶åˆ’åˆ†
# ======================================================
df = pd.read_csv(CSV_PATH)
df["filename"] = df["id_code"].apply(lambda x: x + ".png")

# é‡�æ–°åˆ’åˆ†
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["diagnosis"], random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"è®­ç»ƒé›†å¤§å°�: {len(train_df)} | éªŒè¯�é›†å¤§å°�: {len(val_df)}")

# ======================================================
# 5. åˆ¶ä½œè¿‡é‡‡æ · Sampler (é€»è¾‘æ— è¯¯)
# ======================================================
print("æ­£åœ¨è®¡ç®—é‡‡æ ·æ�ƒé‡�...")
class_weights_np = compute_class_weight('balanced', classes=np.unique(train_df['diagnosis']), y=train_df['diagnosis'])

# æ˜ å°„æ�ƒé‡�åˆ°æ¯�ä¸ªæ ·æœ¬
train_targets = train_df['diagnosis'].to_numpy()
# åˆ—è¡¨æ�¨å¯¼å¼�å�¯èƒ½æ…¢ï¼Œnumpyæ˜ å°„æ›´å¿«ï¼Œä½†ä½ çš„å†™æ³•ä¹Ÿæ²¡é—®é¢˜
samples_weight = torch.DoubleTensor([class_weights_np[t] for t in train_targets])

sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# ======================================================
# 6. ç”Ÿæˆ� DataLoader
# ======================================================
# è¿™é‡Œçš„ transform è®°å¾—ä¼ è¿›å�»
train_ds = RetinopathyDataset(train_df, SAVE_DIR, transform=data_transforms['train'])
val_ds = RetinopathyDataset(val_df, SAVE_DIR, transform=data_transforms['val'])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print("\nâœ… DataLoader å‡†å¤‡å®Œæ¯•ï¼�å�¯ä»¥å¼€å§‹è®­ç»ƒäº†ã€‚")


from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

class RetinopathyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir  # è¿™é‡Œä¼ å…¥ä¿�å­˜é¢„å¤„ç�†å›¾ç‰‡çš„æ–‡ä»¶å¤¹è·¯å¾„
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row["filename"] # ä½¿ç”¨æˆ‘ä»¬å�¯ä»¥ç¡®å®šçš„æ–‡ä»¶å��
        label = row["diagnosis"]
        
        # 1. ä»�ç£�ç›˜æ�„å»ºè·¯å¾„
        img_path = os.path.join(self.img_dir, filename)
        
        # 2. è¯»å�–å›¾ç‰‡ (OpenCV è¯»å�–çš„æ˜¯ BGR)
        image = cv2.imread(img_path)
        
        if image is None:
            # å®¹é”™å¤„ç�†ï¼šå¦‚æ�œè¯»å�–å¤±è´¥ï¼Œåˆ›å»ºä¸€ä¸ªé»‘å›¾
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 3. BGR è½¬ RGB (é��å¸¸é‡�è¦�ï¼�å› ä¸º PyTorch/PIL æœŸæœ› RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 4. è½¬ä¸º PIL Image ä»¥ä¾¿åº”ç”¨ transforms
        img_pil = Image.fromarray(image)
        
        # 5. åº”ç”¨å¢�å¼º
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.long)

# --- æ•°æ�®å¢�å¼ºé…�ç½® (ä¿�æŒ�ä¸�å�˜) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


from torch.utils.data import WeightedRandomSampler

# 1. åˆ’åˆ†æ•°æ�®é›† (ä¿�æŒ�ä¸�å�˜)
train_df, test_val_df = train_test_split(df, test_size=0.3, stratify=df["diagnosis"], random_state=42)
val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df["diagnosis"], random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 2. è®¡ç®— Class Weights (ä¿�æŒ�ä¸�å�˜ï¼Œç”¨äº� Loss æˆ– é‡‡æ ·)
# æ³¨æ„�ï¼šè¿™é‡Œæˆ‘ä»¬ä¿�ç•™ numpy ç‰ˆæœ¬ç”¨äº�ç”Ÿæˆ�é‡‡æ ·å™¨æ�ƒé‡�
class_weights_np = compute_class_weight('balanced', classes=np.unique(train_df['diagnosis']), y=train_df['diagnosis'])
# è½¬ä¸º Tensor å¤‡ç”¨ (å¦‚æ�œä½ è¿˜æƒ³åœ¨ Loss é‡Œå�Œé‡�åŠ æ�ƒçš„è¯�ï¼Œè™½ç„¶é€šå¸¸ç”¨äº†é‡‡æ ·å°±ä¸�éœ€è¦� Loss åŠ æ�ƒäº†)
class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(torch.device("cuda"))
print(f"Class Weights: {class_weights}")

# === ã€�æ–°å¢�æ­¥éª¤ã€‘ä¸ºè¿‡é‡‡æ ·å‡†å¤‡æ ·æœ¬æ�ƒé‡� ===
# 2.1 è�·å�–è®­ç»ƒé›†æ‰€æœ‰ Label
train_targets = train_df['diagnosis'].to_numpy()

# 2.2 ç»™æ¯�ä¸ªæ ·æœ¬åˆ†é…�æ�ƒé‡� (æ ·æœ¬æ�ƒé‡� = å®ƒæ‰€å±�ç±»åˆ«çš„æ�ƒé‡�)
# åˆ—è¡¨æ�¨å¯¼å¼�ï¼šé��å�†æ¯�ä¸ªæ ·æœ¬çš„ labelï¼ŒæŸ¥è¡¨æ‰¾åˆ°å¯¹åº”çš„ weight
samples_weight = [class_weights_np[t] for t in train_targets]
samples_weight = torch.DoubleTensor(samples_weight) # é‡‡æ ·å™¨éœ€è¦� DoubleTensor

# 2.3 åˆ›å»ºé‡‡æ ·å™¨
# replacement=True è¡¨ç¤ºå…�è®¸é‡�å¤�æŠ½æ · (è¿™å°±æ˜¯è¿‡é‡‡æ ·çš„å�Ÿç�†)
sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# 3. å®�ä¾‹åŒ– Dataset (ä¿�æŒ�ä¸�å�˜)
train_ds = RetinopathyDataset(train_df, SAVE_DIR, transform=data_transforms['train'])
val_ds = RetinopathyDataset(val_df, SAVE_DIR, transform=data_transforms['val'])
test_ds = RetinopathyDataset(test_df, SAVE_DIR, transform=data_transforms['val'])

# 4. DataLoader (å…³é”®ä¿®æ”¹ï¼�)
BATCH_SIZE = 32

# === ä¿®æ”¹ç‚¹ï¼štrain_loader åŠ å…¥ samplerï¼Œå¹¶å…³é—­ shuffle ===
train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,   # <--- åŠ å…¥é‡‡æ ·å™¨
    shuffle=False,     # <--- å¿…é¡»æ”¹ä¸º Falseï¼�ï¼�Sampler å’Œ Shuffle äº’æ–¥
    num_workers=2, 
    pin_memory=True
)

# éªŒè¯�é›†å’Œæµ‹è¯•é›†ä¸�éœ€è¦�è¿‡é‡‡æ ·ï¼Œä¿�æŒ� shuffle=False å�³å�¯
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("DataLoaders with Oversampling are ready!")


import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiBranchModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiBranchModel, self).__init__()
        
        # --- Branch 1: DenseNet201 ---
        self.densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        self.densenet_features = self.densenet.features
        # DenseNet è¾“å‡ºé€šé�“: 1920
        
        # --- Branch 2: ResNet50 ---
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # å�»æ�‰æœ€å��ä¸¤å±‚ (AvgPool, FC)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-2]) 
        # ResNet è¾“å‡ºé€šé�“: 2048
        
        # å…¨å±€æ± åŒ–
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # åˆ†æ”¯ç‰¹å®šçš„ Dropout (é˜²æ­¢æŸ�ä¸€ä¸ªåˆ†æ”¯ä¸»å¯¼)
        self.branch_dropout = nn.Dropout(0.5)
        
        # å�ˆå¹¶å��çš„åˆ†ç±»å¤´
        # è¾“å…¥ç»´åº¦: 1920 + 2048 = 3968
        self.fc = nn.Sequential(
            nn.Linear(3968, 512),
            nn.BatchNorm1d(512), # åŠ  BN å±‚åŠ é€Ÿæ”¶æ•›
            nn.ReLU(),
            nn.Dropout(0.5),     # å¼ºåŠ› Dropout
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Branch 1
        x1 = self.densenet_features(x)
        x1 = nn.functional.relu(x1, inplace=True)
        x1 = self.global_pool(x1)
        x1 = torch.flatten(x1, 1)
        
        # Branch 2
        x2 = self.resnet_features(x)
        x2 = self.global_pool(x2)
        x2 = torch.flatten(x2, 1)
        
        # Concatenate
        x_cat = torch.cat((x1, x2), dim=1)
        x_cat = self.branch_dropout(x_cat) # åº”ç”¨ Dropout
        
        # Classification
        out = self.fc(x_cat)
        return out

model = MultiBranchModel(num_classes=5).to(device)
print("Model initialized.")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), correct / total


import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

# 1. ç¡®ä¿�å®šä¹‰äº†è®¾å¤‡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. å°†ä¹‹å‰�è®¡ç®—å‡ºçš„ list/numpy æ�ƒé‡�è½¬æ�¢ä¸º Tensorï¼Œå¹¶ç§»è‡³ device
# å�‡è®¾ class_weights = [0.40567866, 1.9790541, 0.73316646, 3.8038962, 2.4822035]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# 3. ä¼ å…¥è½¬æ�¢å��çš„ Tensor
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# ---------------------------------------------------------
# STAGE 1: Warmup (å†»ç»“éª¨å¹²ï¼Œå�ªè®­ç»ƒåˆ†ç±»å¤´)
# ---------------------------------------------------------
print("=== Stage 1: Training Head Only ===")

# 1. å†»ç»“æ‰€æœ‰å±‚
for param in model.parameters():
    param.requires_grad = False
# 2. è§£å†»åˆ†ç±»å¤´ (fc)
for param in model.fc.parameters():
    param.requires_grad = True

# Stage 1 ä¼˜åŒ–å™¨ (å­¦ä¹ ç�‡ç¨�å¤§)
optimizer_s1 = optim.Adam(model.fc.parameters(), lr=1e-3)
best_val_acc = 0.0

for epoch in range(5): # è®­ç»ƒ 5 ä¸ª epoch é¢„çƒ­
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_s1)
    val_loss, val_acc = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/5 - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model_stage1.pth")

print("Stage 1 Complete. Loading best weights...")
model.load_state_dict(torch.load("model_stage1.pth"))


# ---------------------------------------------------------
# STAGE 2: Fine-Tuning (è§£å†»éª¨å¹²ï¼Œå¾®è°ƒå…¨ç½‘)
# ---------------------------------------------------------
print("\n=== Stage 2: Global Fine-Tuning ===")

# 1. è§£å†»æ‰€æœ‰å±‚
for param in model.parameters():
    param.requires_grad = True

# Stage 2 ä¼˜åŒ–å™¨ (å­¦ä¹ ç�‡æ��å°�ï¼Œé…�å�ˆæ�ƒé‡�è¡°å‡�)
optimizer_s2 = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# å­¦ä¹ ç�‡è°ƒåº¦å™¨
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, mode='max', factor=0.1, patience=3, verbose=True)

best_val_acc = 0.0
patience_counter = 0
early_stopping_limit = 7 # å®¹å¿�æ¬¡æ•°

history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(20): # æœ€å¤š 20 epochs
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_s2)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # è®°å½•å�†å�²
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # æ›´æ–° LR
    scheduler.step(val_acc)
    
    print(f"Epoch {epoch+1}/20 - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    # ä¿�å­˜æœ€ä½³æ¨¡å�‹ & æ—©å�œ
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_multibranch_model.pth")
        patience_counter = 0 # é‡�ç½®è®¡æ•°å™¨
        print(">>> New Best Model Saved!")
    else:
        patience_counter += 1
        print(f">>> EarlyStopping counter: {patience_counter}/{early_stopping_limit}")
        
    if patience_counter >= early_stopping_limit:
        print("Early stopping triggered.")
        break


import numpy as np
import pandas as pd
import torch
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# === 1. é…�ç½®è·¯å¾„ (ç¡®ä¿�è¿™é‡Œå’Œè®­ç»ƒæ—¶ä¸€æ ·) ===
SAVE_DIR = "/kaggle/input/aptos-2019-preprocessed-224/processed_images" 
CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
BATCH_SIZE = 32

# === 2. é‡�æ–°å®šä¹‰ Dataset ç±» ===
class RetinopathyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row["filename"]
        label = row["diagnosis"]
        img_path = os.path.join(self.img_dir, filename)
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.long)

# === 3. é‡�æ–°å®šä¹‰æ•°æ�®å¢�å¼º (Valéƒ¨åˆ†) ===
data_transforms = {
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# === 4. é‡�æ–°åˆ’åˆ†æ•°æ�® (å¿…é¡»å’Œè®­ç»ƒæ—¶éš�æœºç§�å­�ä¸€è‡´ï¼�) ===
df = pd.read_csv(CSV_PATH)
df["filename"] = df["id_code"].apply(lambda x: x + ".png")

# è¿™é‡Œçš„ random_state=42 ä¿�è¯�äº†ä½ ç�°åœ¨çš„ val_df å’Œè®­ç»ƒæ—¶å®Œå…¨ä¸€æ ·
train_df, test_val_df = train_test_split(df, test_size=0.3, stratify=df["diagnosis"], random_state=42)
val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df["diagnosis"], random_state=42)

print(f"éªŒè¯�é›†æ�¢å¤�æˆ�åŠŸï¼Œæ•°é‡�: {len(val_df)}")

# === 5. å�ªç”Ÿæˆ� val_loader å�³å�¯ (ä¸�éœ€è¦� train_loader) ===
val_ds = RetinopathyDataset(val_df, SAVE_DIR, transform=data_transforms['val'])
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("âœ… val_loader å‡†å¤‡å°±ç»ªï¼�")


import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === æŠŠä½ å®šä¹‰çš„æ¨¡å�‹ç±»å�Ÿå°�ä¸�åŠ¨å¤�åˆ¶è¿‡æ�¥ ===
class MultiBranchModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiBranchModel, self).__init__()
        
        # Branch 1: DenseNet201
        self.densenet = models.densenet201(weights=None) # æ�¨ç�†æ—¶ä¸�éœ€è¦�ä¸‹è½½ ImageNet æ�ƒé‡�ï¼Œå��æ­£ä¼šè¢«è¦†ç›–
        self.densenet_features = self.densenet.features
        
        # Branch 2: ResNet50
        self.resnet = models.resnet50(weights=None)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-2]) 
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.branch_dropout = nn.Dropout(0.5)
        
        self.fc = nn.Sequential(
            nn.Linear(3968, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x1 = self.densenet_features(x)
        x1 = nn.functional.relu(x1, inplace=True)
        x1 = self.global_pool(x1)
        x1 = torch.flatten(x1, 1)
        
        x2 = self.resnet_features(x)
        x2 = self.global_pool(x2)
        x2 = torch.flatten(x2, 1)
        
        x_cat = torch.cat((x1, x2), dim=1)
        x_cat = self.branch_dropout(x_cat)
        
        out = self.fc(x_cat)
        return out

# === åˆ�å§‹åŒ–æ¨¡å�‹ ===
model = MultiBranchModel(num_classes=5).to(device)
print("âœ… æ¨¡å�‹ç»“æ�„å·²é‡�å»ºï¼ˆå½“å‰�æ˜¯éš�æœºåˆ�å§‹åŒ–çš„ï¼‰")


from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# === 1. åŠ è½½æ�ƒé‡� ===
weight_path = "best_multibranch_model.pth"

if os.path.exists(weight_path):
    print(f"æ­£åœ¨åŠ è½½æ�ƒé‡�: {weight_path} ...")
    # åŠ è½½æ�ƒé‡�åˆ°æ¨¡å�‹
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval() # æ��å…¶é‡�è¦�ï¼šåˆ‡æ�¢åˆ°è¯„ä¼°æ¨¡å¼� (å…³é—­ Dropout/BN)
    print("âœ… æ�ƒé‡�åŠ è½½æˆ�åŠŸï¼�ä¸�éœ€è¦�é‡�è®­ï¼�")
else:
    print(f"â�Œ æ‰¾ä¸�åˆ°æ–‡ä»¶: {weight_path}")
    print("å¦‚æ�œæ–‡ä»¶ä¸¢å¤±ï¼Œè¯´æ˜� Session é‡�ç½®å¯¼è‡´ä¸´æ—¶æ–‡ä»¶è¢«æ¸…ç©ºï¼Œè¿™ç§�æƒ…å†µä¸‹å�ªèƒ½é‡�è®­äº†...")
    # å¦‚æ�œè¿™é‡ŒæŠ¥é”™ï¼Œå�œæ­¢å¾€ä¸‹è¿�è¡Œ

# === 2. å�ªæœ‰æ�ƒé‡�åŠ è½½æˆ�åŠŸæ‰�è¿�è¡Œä¸‹é�¢çš„é¢„æµ‹ ===
if os.path.exists(weight_path):
    all_preds = []
    all_labels = []

    print("å¼€å§‹éªŒè¯�é›†æ�¨ç�†...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_val = np.array(all_labels)
    y_pred = np.array(all_preds)

    # === 3. æ‰“å�°æŠ¥å‘Š ===
    target_names = ['0: No DR', '1: Mild', '2: Mod', '3: Severe', '4: Prolif']
    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=target_names))

    # === 4. ç”»æ··æ·†çŸ©é˜µ ===
    cm = confusion_matrix(y_val, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Loaded Model)')
    plt.show()


# ç»˜åˆ¶ Loss å’Œ Accuracy æ›²çº¿
plt.figure(figsize=(12, 5))

# 1. Loss æ›²çº¿
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='.')
plt.plot(history['val_loss'], label='Val Loss', marker='.')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Accuracy æ›²çº¿
plt.subplot(1, 2, 2)
# æ³¨æ„�ï¼šä½ çš„historyé‡Œå¥½åƒ�å�ªè®°å½•äº† val_accï¼Œå¦‚æ�œ train_acc æ²¡å­˜è¿›å�»ä¼šæŠ¥é”™
# å�‡è®¾ä½ çš„ history å­—å…¸ç»“æ�„æ˜¯ {'train_loss':[], 'val_loss':[], 'val_acc':[]}
if 'train_acc' in history:
    plt.plot(history['train_acc'], label='Train Acc', marker='.')
plt.plot(history['val_acc'], label='Val Acc', marker='.')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# å‰�ç½®æ£€æŸ¥ï¼šç¡®ä¿�ä½ çš„ y_pred æ˜¯ç±»åˆ«æ ‡ç­¾ (0,1,2...) è€Œä¸�æ˜¯æ¦‚ç�‡
# å¦‚æ�œä½ çš„ y_pred æ˜¯æ¦‚ç�‡ (ä¾‹å¦‚ shapeæ˜¯ (n, 5))ï¼Œè¯·å�–æ¶ˆä¸‹é�¢è¿™è¡Œçš„æ³¨é‡Šï¼š
# y_pred = np.argmax(y_pred, axis=1)
# ==========================================

# 1. æ‰“å�°è¯¦ç»†çš„åˆ†ç±»æŠ¥å‘Š (åŒ…å�« Precision, Recall, F1-Score)
print("=== Classification Report ===")
# target_names è®¾ç½®ä¸ºä½ çš„5ä¸ªç±»åˆ«ï¼Œæ–¹ä¾¿æŸ¥çœ‹
report = classification_report(y_val, y_pred, target_names=['0: No DR', '1: Mild', '2: Mod', '3: Severe', '4: Prolif'])
print(report)

# 2. è®¡ç®—å¹¶ç»˜åˆ¶æ··æ·†çŸ©é˜µ
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))

# å½’ä¸€åŒ–æ··æ·†çŸ©é˜µ (æŒ‰çœŸå®�ç±»åˆ«å½’ä¸€åŒ– -> æŸ¥çœ‹å�¬å›�ç�‡ Recall)
# æ¯�ä¸€è¡Œçš„å’Œä¸º1ï¼Œè¡¨ç¤ºè¯¥çœŸå®�ç±»åˆ«è¢«é¢„æµ‹æˆ�å�„ç±»çš„æ¯”ä¾‹
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# ç»˜åˆ¶çƒ­åŠ›å›¾
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['0', '1', '2', '3', '4'],
            yticklabels=['0', '1', '2', '3', '4'])

plt.xlabel('Predicted Label (æ¨¡å�‹é¢„æµ‹)')
plt.ylabel('True Label (çœŸå®�æƒ…å†µ)')
plt.title('Normalized Confusion Matrix (Recall View)')
plt.show()

# 3. å�•ç‹¬æ‰“å�° Macro F1 Score
macro_f1 = f1_score(y_val, y_pred, average='macro')
print(f"=== Macro F1-Score: {macro_f1:.4f} ===")
print("(è¿™æ˜¯è¡¡é‡�ä¸�å¹³è¡¡æ•°æ�®é›†ä¸­æ¨¡å�‹ç»¼å�ˆæ€§èƒ½çš„å…³é”®æŒ‡æ ‡)")


from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# å‰�ç½®æ£€æŸ¥ï¼šç¡®ä¿�ä½ çš„ y_pred æ˜¯ç±»åˆ«æ ‡ç­¾ (0,1,2...) è€Œä¸�æ˜¯æ¦‚ç�‡
# å¦‚æ�œä½ çš„ y_pred æ˜¯æ¦‚ç�‡ (ä¾‹å¦‚ shapeæ˜¯ (n, 5))ï¼Œè¯·å�–æ¶ˆä¸‹é�¢è¿™è¡Œçš„æ³¨é‡Šï¼š
# y_pred = np.argmax(y_pred, axis=1)
# ==========================================

# 1. æ‰“å�°è¯¦ç»†çš„åˆ†ç±»æŠ¥å‘Š (åŒ…å�« Precision, Recall, F1-Score)
print("=== Classification Report ===")
# target_names è®¾ç½®ä¸ºä½ çš„5ä¸ªç±»åˆ«ï¼Œæ–¹ä¾¿æŸ¥çœ‹
report = classification_report(y_val, y_pred, target_names=['0: No DR', '1: Mild', '2: Mod', '3: Severe', '4: Prolif'])
print(report)

# 2. è®¡ç®—å¹¶ç»˜åˆ¶æ··æ·†çŸ©é˜µ
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))

# å½’ä¸€åŒ–æ··æ·†çŸ©é˜µ (æŒ‰çœŸå®�ç±»åˆ«å½’ä¸€åŒ– -> æŸ¥çœ‹å�¬å›�ç�‡ Recall)
# æ¯�ä¸€è¡Œçš„å’Œä¸º1ï¼Œè¡¨ç¤ºè¯¥çœŸå®�ç±»åˆ«è¢«é¢„æµ‹æˆ�å�„ç±»çš„æ¯”ä¾‹
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# ç»˜åˆ¶çƒ­åŠ›å›¾
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['0', '1', '2', '3', '4'],
            yticklabels=['0', '1', '2', '3', '4'])

plt.xlabel('Predicted Label (æ¨¡å�‹é¢„æµ‹)')
plt.ylabel('True Label (çœŸå®�æƒ…å†µ)')
plt.title('Normalized Confusion Matrix (Recall View)')
plt.show()

# 3. å�•ç‹¬æ‰“å�° Macro F1 Score
macro_f1 = f1_score(y_val, y_pred, average='macro')
print(f"=== Macro F1-Score: {macro_f1:.4f} ===")
print("(è¿™æ˜¯è¡¡é‡�ä¸�å¹³è¡¡æ•°æ�®é›†ä¸­æ¨¡å�‹ç»¼å�ˆæ€§èƒ½çš„å…³é”®æŒ‡æ ‡)")


# åŠ è½½æœ€ä½³æ¨¡å�‹
model.load_state_dict(torch.load("best_multibranch_model.pth"))
model.eval()

y_true = []
y_pred = []
y_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

# 1. å‡†ç¡®ç�‡
acc = accuracy_score(y_true, y_pred)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

# 2. æ··æ·†çŸ©é˜µ
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Multi-Branch)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 3. åˆ†ç±»æŠ¥å‘Š
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# 4. è®­ç»ƒæ›²çº¿
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Val Accuracy', color='green')
plt.title("Validation Accuracy")
plt.legend()
plt.show()


def get_saliency_map(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_() # å¼€å�¯æ¢¯åº¦è¿½è¸ª
    
    output = model(image_tensor.unsqueeze(0)) # Add batch dim
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    
    # å��å�‘ä¼ æ’­
    model.zero_grad()
    output_max.backward()
    
    # è�·å�–è¾“å…¥å›¾åƒ�çš„æ¢¯åº¦ (å�–ç»�å¯¹å€¼ max)
    saliency, _ = torch.max(image_tensor.grad.data.abs(), dim=0)
    return saliency.cpu().numpy()

# ä»�æµ‹è¯•é›†ä¸­å�–ä¸€å¼ å›¾ç‰‡
idx = 10 
img_tensor, label = test_ds[idx]

# è®¡ç®— Saliency
saliency_map = get_saliency_map(model, img_tensor)

# ç»˜å›¾
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# å��å½’ä¸€åŒ–ä»¥ä¾¿æ˜¾ç¤º
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)
img_display = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
img_display = np.clip(img_display, 0, 1) # ä¿®æ­£æ•°å€¼èŒƒå›´

plt.imshow(img_display)
plt.title(f"Original (Label: {label.item()})")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap='hot')
plt.title("Saliency Map")
plt.axis('off')

plt.show()


import torch
from collections import Counter

# å�‡è®¾ä½ çš„è®­ç»ƒé›†å�˜é‡�å��æ˜¯ train_dataset
# å¦‚æ�œä½ çš„å�˜é‡�å��ä¸�ä¸€æ ·ï¼ˆæ¯”å¦‚ train_dataï¼‰ï¼Œè¯·ä¿®æ”¹ä¸‹é�¢è¿™ä¸€è¡Œ
dataset_to_count = train_ds

print("æ­£åœ¨ç»Ÿè®¡ç±»åˆ«æ•°é‡�ï¼Œè¯·ç¨�å€™...")

try:
    # === æ–¹æ³• A: æ��é€Ÿç‰ˆ (é’ˆå¯¹ ImageFolder æˆ–å¸¸ç”¨æ•°æ�®é›†) ===
    # ImageFolder é€šå¸¸ä¼šæŠŠæ ‡ç­¾å­˜åœ¨ .targets é‡Œ
    if hasattr(dataset_to_count, 'targets'):
        labels = dataset_to_count.targets
    # æœ‰äº›è‡ªå®šä¹‰ Dataset å�¯èƒ½ä¼šå�« .labels
    elif hasattr(dataset_to_count, 'labels'):
        labels = dataset_to_count.labels
    else:
        raise AttributeError("æ²¡æ‰¾åˆ° targets å±�æ€§")
    
    # ç¡®ä¿� labels æ˜¯åˆ—è¡¨ï¼Œä¸�æ˜¯ Tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
        
    print("æˆ�åŠŸé€šè¿‡å±�æ€§ç›´æ�¥è¯»å�–æ ‡ç­¾ï¼�")

except AttributeError:
    # === æ–¹æ³• B: æš´åŠ›ç‰ˆ (ä¸‡èƒ½ï¼Œä½†å¦‚æ�œæ•°æ�®é›†å¾ˆå¤§å�¯èƒ½ä¼šæ…¢) ===
    print("æœªæ‰¾åˆ°ç›´æ�¥æ ‡ç­¾å±�æ€§ï¼Œæ­£åœ¨é��å�†æ•°æ�®é›†ç»Ÿè®¡ (å�¯èƒ½éœ€è¦�ä¸€ç‚¹æ—¶é—´)...")
    labels = []
    for i in range(len(dataset_to_count)):
        # å�ªå�– labelï¼Œä¸�åŠ è½½å›¾ç‰‡æ•°æ�®ä»¥åŠ å¿«é€Ÿåº¦
        # æ³¨æ„�ï¼šè¿™é‡Œå�‡è®¾ä½ çš„ dataset[i] è¿”å›�çš„æ˜¯ (image, label)
        _, label = dataset_to_count[i]
        
        # å¦‚æ�œ label æ˜¯ Tensorï¼Œè½¬æˆ�æ•°å­—
        if isinstance(label, torch.Tensor):
            label = label.item()
        labels.append(label)

# === ç»Ÿè®¡å¹¶æ‰“å�° ===
counts = Counter(labels)
sorted_counts = dict(sorted(counts.items())) # æŒ‰ç±»åˆ« ID æ�’åº�

print("\n=== è®­ç»ƒé›†ç±»åˆ«åˆ†å¸ƒ ===")
total_count = sum(counts.values())
for class_idx, count in sorted_counts.items():
    percent = (count / total_count) * 100
    print(f"Class {class_idx}: {count} å¼  ({percent:.2f}%)")

print(f"\næ€»æ ·æœ¬æ•°: {total_count}")

# ç®€å�•çš„åˆ¤æ–­
max_class = max(counts.values())
min_class = min(counts.values())
if max_class / min_class > 2:
    print(f"\nâš ï¸� è­¦å‘Š: ç±»åˆ«æ��å…¶ä¸�å¹³è¡¡ï¼�æœ€å¤šç±»æ˜¯æœ€å°�ç±»çš„ {max_class/min_class:.1f} å€�ã€‚")
    print("å¼ºçƒˆå»ºè®®ä½¿ç”¨ 'åŠ æ�ƒ Loss' æˆ– 'WeightedRandomSampler'ã€‚")
else:
    print("\nâœ… ç±»åˆ«ç›¸å¯¹å¹³è¡¡ï¼Œå�¯èƒ½ä¸�æ˜¯ä¸�å¹³è¡¡å¯¼è‡´çš„é—®é¢˜ã€‚")


import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch

# === 1. é‡�æ–°è¯»å�– CSV (æ‰¾å›� df) ===
CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
# å®šä¹‰ä¿�å­˜è·¯å¾„ (å��ç»­ Dataset éœ€è¦�ç”¨åˆ°)
SAVE_DIR = "/kaggle/working/processed_images" 

if not os.path.exists(CSV_PATH):
    print("é”™è¯¯ï¼šæ‰¾ä¸�åˆ° CSV æ–‡ä»¶ï¼Œè¯·æ£€æŸ¥è·¯å¾„ï¼�")
else:
    df = pd.read_csv(CSV_PATH)
    # åˆ«å¿˜äº†è¿™ä¸€æ­¥ï¼�åŠ ä¸Šå��ç¼€
    df["filename"] = df["id_code"].apply(lambda x: x + ".png")
    print(f"æˆ�åŠŸè¯»å�– Dataframe: {len(df)} è¡Œ")

# === 2. åˆ’åˆ†è®­ç»ƒé›†å’ŒéªŒè¯�é›† ===
# stratify=df['diagnosis'] ä¿�è¯�ç±»åˆ«æ¯”ä¾‹ä¸€è‡´
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['diagnosis']
)

print(f"è®­ç»ƒé›†æ•°é‡�: {len(train_df)}")
print(f"éªŒè¯�é›†æ•°é‡�: {len(val_df)}")

# === 3. ç»Ÿè®¡ç±»åˆ«å¹¶è®¡ç®—æ�ƒé‡� ===
class_counts = train_df['diagnosis'].value_counts().sort_index()
print("\n=== è®­ç»ƒé›†ç±»åˆ«åˆ†å¸ƒ ===")
print(class_counts)

# è®¡ç®—æ�ƒé‡�
total_samples = len(train_df)
num_classes = len(class_counts)
class_weights = [total_samples / (num_classes * count) for count in class_counts]

# è½¬æˆ� Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print(f"\nè®¡ç®—å‡ºçš„ç±»åˆ«æ�ƒé‡� (Class Weights): \n{class_weights_tensor.cpu().numpy()}")


import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim

# === 1. å®šä¹‰æ¨¡å�‹ (å¦‚æ�œä½ è¿˜æ²¡å®šä¹‰çš„è¯�) ===
print("æ­£åœ¨åŠ è½½ ResNet50 é¢„è®­ç»ƒæ¨¡å�‹...")
# ä½¿ç”¨ ResNet50ï¼Œå®ƒçš„æ€§èƒ½æ¯” ResNet18 å¼ºå¾ˆå¤šï¼Œé€‚å�ˆåŒ»å­¦å›¾åƒ�
# weights='DEFAULT' è¡¨ç¤ºåŠ è½½æœ€æ–°çš„ ImageNet é¢„è®­ç»ƒæ�ƒé‡�
model = models.resnet50(weights='DEFAULT')

# === 2. ä¿®æ”¹å…¨è¿�æ�¥å±‚ (é€‚é…� 5 åˆ†ç±») ===
# è�·å�– ResNet æœ€å��å…¨è¿�æ�¥å±‚çš„è¾“å…¥ç‰¹å¾�æ•°
num_ftrs = model.fc.in_features
# æ›¿æ�¢ä¸ºæ–°çš„å…¨è¿�æ�¥å±‚ï¼Œè¾“å‡ºç±»åˆ«æ•°ä¸º 5 (APTOS æ•°æ�®é›†)
model.fc = nn.Linear(num_ftrs, 5)

# === 3. æ�¬è¿�åˆ° GPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"æ¨¡å�‹å·²åŠ è½½å¹¶ç§»åŠ¨åˆ°: {device}")

# ==========================================
#       ä¸‹é�¢æ˜¯ Stage 1 è®­ç»ƒä»£ç �
# ==========================================

print("\n=== Stage 1: Training Head Only (with Class Weights) ===")

# 4. å†»ç»“æ‰€æœ‰å±‚ (å�ªè®­ç»ƒæœ€å��ä¸€å±‚)
for param in model.parameters():
    param.requires_grad = False

# 5. è§£å†»åˆ†ç±»å¤´ (fc å±‚)
for param in model.fc.parameters():
    param.requires_grad = True

# 6. å®šä¹‰å¸¦æ�ƒé‡�çš„ Loss
# ç¡®ä¿� class_weights_tensor å­˜åœ¨ (å¦‚æ�œæŠ¥é”™ï¼Œè¯´æ˜�ä½ éœ€è¦�é‡�æ–°è¿�è¡Œè®¡ç®—æ�ƒé‡�çš„ä»£ç �)
if 'class_weights_tensor' not in locals():
    print("é”™è¯¯: class_weights_tensor æœªå®šä¹‰ï¼�è¯·å…ˆè¿�è¡Œä¸Šé�¢è®¡ç®—æ�ƒé‡�çš„ä»£ç �ã€‚")
else:
    class_weights_tensor = class_weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 7. ä¼˜åŒ–å™¨ (å�ªä¼˜åŒ– fc)
    optimizer_s1 = optim.Adam(model.fc.parameters(), lr=1e-3)

    # 8. è®­ç»ƒå¾ªç�¯ (5 Epochs)
    best_val_loss = float('inf')

    for epoch in range(5): 
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer_s1.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) # åŠ æ�ƒ Loss
            loss.backward()
            optimizer_s1.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_ds)
        epoch_acc = running_corrects.double() / len(train_ds)
        
        # éªŒè¯�é€»è¾‘
        model.eval()
        val_running_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_running_loss / len(val_ds)
        val_acc = val_corrects.double() / len(val_ds)

        print(f"Epoch {epoch+1}/5 - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model_stage1_weighted.pth")

    print("Stage 1 å®Œæˆ�ï¼�å‡†å¤‡è¿›å…¥ Stage 2ã€‚")


import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader



# 1. é‡�æ–°ç”Ÿæˆ� Dataset å’Œ DataLoader (ç¡®ä¿�ä½¿ç”¨åˆšæ‰�åˆ‡åˆ†å¥½çš„å¸¦æ�ƒé‡�çš„ train_df)
# å�‡è®¾ SAVE_DIR æ˜¯ä½ åˆšæ‰�é¢„å¤„ç�†å›¾ç‰‡çš„è·¯å¾„ "/kaggle/working/processed_images"
train_ds = RetinopathyDataset(train_df, SAVE_DIR, transform=data_transforms['train'])
val_ds = RetinopathyDataset(val_df, SAVE_DIR, transform=data_transforms['val'])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# 2. å®šä¹‰å¸¦æ�ƒé‡�çš„ Loss (å…³é”®æ­¥éª¤ï¼�)
# ç¡®ä¿�æ�ƒé‡�åœ¨ GPU ä¸Š
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = class_weights_tensor.to(device) 
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 3. å‡†å¤‡ Stage 2 å¾®è°ƒ (è§£å†»æ‰€æœ‰å±‚ + å°�å­¦ä¹ ç�‡)
# å¦‚æ�œä½ éœ€è¦�é‡�æ–°åŠ è½½ä¹‹å‰�çš„æœ€ä½³æ�ƒé‡�ï¼Œè¯·å�–æ¶ˆä¸‹é�¢è¿™è¡Œçš„æ³¨é‡Š
# model.load_state_dict(torch.load("model_stage1.pth")) 

for param in model.parameters():
    param.requires_grad = True

# ä½¿ç”¨è¾ƒå°�çš„å­¦ä¹ ç�‡è¿›è¡Œå¾®è°ƒ
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# 4. å¼€å§‹è®­ç»ƒ
print(f"=== Starting Stage 2 Training with Weighted Loss (Weights: {class_weights_tensor.cpu().numpy().round(2)}) ===")
best_val_loss = float('inf')

for epoch in range(15): # å»ºè®®è·‘ 10-15 ä¸ª Epoch
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) # è¿™é‡Œä¼šè‡ªåŠ¨åº”ç”¨ class_weights
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_ds)
    epoch_acc = running_corrects.double() / len(train_ds)
    
    # éªŒè¯�é˜¶æ®µ
    model.eval()
    val_running_loss = 0.0
    val_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
    val_loss = val_running_loss / len(val_ds)
    val_acc = val_corrects.double() / len(val_ds)
    
    print(f"Epoch {epoch+1}/15 - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    # ä¿�å­˜æœ€ä½³æ¨¡å�‹ (æ ¹æ�® Val Loss è€Œä¸�æ˜¯ Accï¼Œå› ä¸ºæ•°æ�®ä¸�å¹³è¡¡æ—¶ Loss æ›´å�¯é� )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_weighted.pth")
        print(">>> New Best Model Saved (Improved Val Loss)!")
        
    scheduler.step(val_loss)

