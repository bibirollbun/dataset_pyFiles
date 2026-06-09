import os
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np

# --- ãƒ©ãƒ™ãƒ«ãƒ�ãƒƒãƒ”ãƒ³ã‚° ---
label_map = {
    'freshapples': 1,
    'freshoranges': 2,
    'freshpotato': 3,
    'freshtomato': 4,
    'rottenapples': 5,
    'rottenoranges': 6,
    'rottenpotato': 7,
    'rottentomato': 8
}

# --- å¼·åŒ–ã�•ã‚Œã�Ÿãƒ‡ãƒ¼ã‚¿å¤‰æ�› ---
# è¨“ç·´ç”¨ï¼šãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã‚’è¿½åŠ 
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# æ¤œè¨¼ãƒ»ãƒ†ã‚¹ãƒˆç”¨ï¼šæ­£è¦�åŒ–ã�®ã�¿
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®èª­ã�¿è¾¼ã�¿ ---
train_dataset = datasets.ImageFolder('/kaggle/input/responsi-komputasi-statistika-ii/Dataset-Responsi/Train', transform=train_transform)
val_dataset = datasets.ImageFolder('/kaggle/input/responsi-komputasi-statistika-ii/Dataset-Responsi/Validation', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# --- ã‚ˆã‚Šå¼·åŠ›ã�ªãƒ¢ãƒ‡ãƒ«å®šç¾©ï¼ˆResNet50 + Dropoutï¼‰ ---
class ImprovedResNet(nn.Module):
    def __init__(self, num_classes=8):
        super(ImprovedResNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        # ç‰¹å¾´é‡�æŠ½å‡ºéƒ¨åˆ†ã‚’éƒ¨åˆ†çš„ã�«å‡�çµ�
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
            
        # ã‚«ã‚¹ã‚¿ãƒ åˆ†é¡�ãƒ˜ãƒƒãƒ‰
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

model = ImprovedResNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --- å­¦ç¿’è¨­å®šã�®æ”¹å–„ ---
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ãƒ©ãƒ™ãƒ«ã‚¹ãƒ ãƒ¼ã‚¸ãƒ³ã‚°
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# å­¦ç¿’ç�‡ã‚¹ã‚±ã‚¸ãƒ¥ãƒ¼ãƒ©ãƒ¼
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

early_stopping = EarlyStopping(patience=10)

# --- æ¤œè¨¼é–¢æ•° ---
def validate_model(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), 100 * correct / total

# --- å­¦ç¿’ãƒ«ãƒ¼ãƒ—ã�®æ”¹å–„ ---
best_val_acc = 0
best_model_state = None

for epoch in range(50):  # ã‚¨ãƒ�ãƒƒã‚¯æ•°ã‚’å¢—åŠ 
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # å‹¾é…�ã‚¯ãƒªãƒƒãƒ”ãƒ³ã‚°
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    # æ¤œè¨¼
    val_loss, val_acc = validate_model(model, val_loader)
    scheduler.step()
    
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # ãƒ™ã‚¹ãƒˆãƒ¢ãƒ‡ãƒ«ã�®ä¿�å­˜
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"âœ… New best validation accuracy: {best_val_acc:.2f}%")
    
    # Early Stopping
    if early_stopping(val_loss):
        print("Early stopping triggered")
        break

# ãƒ™ã‚¹ãƒˆãƒ¢ãƒ‡ãƒ«ã‚’ãƒ­ãƒ¼ãƒ‰
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")

# --- ãƒ†ã‚¹ãƒˆæ™‚ã�®æ‹¡å¼µï¼ˆTTA: Test Time Augmentationï¼‰ ---
def tta_predict(model, image_path, n_tta=5):
    """Test Time Augmentation ã‚’ä½¿ç”¨ã�—ã�Ÿäºˆæ¸¬"""
    model.eval()
    
    # è¤‡æ•°ã�®å¤‰æ�›ã‚’é�©ç”¨
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]
    
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms[:n_tta]:
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            predictions.append(F.softmax(output, dim=1))
    
    # å¹³å�‡äºˆæ¸¬
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction.argmax(dim=1).item()

# --- ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã�‹ã‚‰ã‚¯ãƒ©ã‚¹å��ã�¸ã�®ãƒ�ãƒƒãƒ”ãƒ³ã‚° ---
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# --- sample_submission.csv ã�®èª­ã�¿è¾¼ã�¿ ---
sample_df = pd.read_csv('/kaggle/input/responsi-komputasi-statistika-ii/Dataset-Responsi/sample_submission.csv')

# --- ãƒ†ã‚¹ãƒˆç”»åƒ�ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒª ---
test_dir = '/kaggle/input/responsi-komputasi-statistika-ii/Dataset-Responsi/Test'

# --- ãƒ¢ãƒ‡ãƒ«æ�¨è«–ï¼ˆTTAä½¿ç”¨ï¼‰ ---
model.eval()
predicted_labels = []

print("Starting inference with Test Time Augmentation...")
for i, img_name in enumerate(sample_df['image']):
    img_path = os.path.join(test_dir, img_name)
    pred_class_idx = tta_predict(model, img_path, n_tta=5)
    
    # index â†’ class_name â†’ label_code ã�«å¤‰æ�›
    class_name = idx_to_class[pred_class_idx]
    label_code = label_map[class_name]
    predicted_labels.append(label_code)
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(sample_df)} images")

# --- submission.csv å‡ºåŠ› ---
sample_df['label'] = predicted_labels
sample_df.columns = ['image', 'label']
sample_df.to_csv('submission.csv', index=False)
print("âœ… æ­£ã�—ã�� submission.csv ã‚’ä¿�å­˜ã�—ã�¾ã�—ã�Ÿã€‚")
print(f"ğŸ“Š Best validation accuracy achieved: {best_val_acc:.2f}%")

