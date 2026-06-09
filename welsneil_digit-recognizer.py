# =============================================================================
# BƯỚC 0: CÀI ĐẶT THƯ VIỆN
# =============================================================================
!pip install timm -q
print("Thư viện 'timm' đã được cài đặt.")


# =============================================================================
# BƯỚC 1: IMPORT CÁC THƯ VIỆN
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import gc
import timm

print("Tất cả các thư viện đã được import thành công!")


# =============================================================================
# BƯỚC 2: CẤU HÌNH DỰ ÁN
# =============================================================================
class Config:
    TRAIN_FILE = '/kaggle/input/digit-recognizer-challenge/train.csv'
    TEST_FILE = '/kaggle/input/digit-recognizer-challenge/test.csv'
    SUBMISSION_FILE = '/kaggle/working/submission.csv'
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    RANDOM_STATE = 42
    N_SPLITS = 5
    N_TTA = 10
    LABEL_SMOOTHING = 0.1

print(f"Thiết bị đang sử dụng: {Config.DEVICE}")
print(f"Chiến lược FINAL BOSS: Ensemble lai (ResNet18 + Vision Transformer).")


# =============================================================================
# BƯỚC 3: XÂY DỰNG LỚP DATASET (Không thay đổi)
# =============================================================================
class MNISTDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28)
        label = self.labels[idx] if self.labels is not None else -1
        if self.transform: image = self.transform(image)
        return image, label



# =============================================================================
# BƯỚC 4: ĐỊNH NGHĨA CÁC PHÉP BIẾN ĐỔI (TRANSFORMS)
# =============================================================================
# --- Transform cho ResNet ---
resnet_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
])
resnet_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Thêm bước Resize để phóng to ảnh lên 224x224
vit_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), # <-- ĐÂY LÀ BƯỚC QUAN TRỌNG
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
])
vit_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), # <-- ĐÂY LÀ BƯỚC QUAN TRỌNG
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
print("Đã định nghĩa các bộ transform riêng cho ResNet và ViT.")



# =============================================================================
# BƯỚC 5: ĐỊNH NGHĨA CÁC KIẾN TRÚC MẠNG (Không thay đổi)
# =============================================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes); self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes); self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes: self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion*planes))
    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out)); out += self.shortcut(x); out = nn.ReLU(inplace=True)(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__(); self.in_planes = 64; self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64); self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1); self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2); self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1); layers = []
        for s in strides: layers.append(block(self.in_planes, planes, s)); self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x))); out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out); out = out.view(out.size(0), -1); out = self.linear(out)
        return out
def ResNet18(): return ResNet(BasicBlock, [2, 2, 2, 2])
def create_vit(): return timm.create_model("vit_tiny_patch16_224", pretrained=True, in_chans=1, num_classes=10)


# =============================================================================
# BƯỚC 6: ĐỊNH NGHĨA HÀM MẤT MÁT (Không thay đổi)
# =============================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__(); self.confidence = 1.0 - smoothing; self.smoothing = smoothing; self.cls = classes; self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, pred, target):
        pred = self.logsoftmax(pred)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred); true_dist.fill_(self.smoothing / (self.cls - 1)); true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))



# =============================================================================
# BƯỚC 7: HÀM HUẤN LUYỆN VÀ DỰ ĐOÁN CHUNG
# =============================================================================
def train_and_predict(model_name, X, y, X_test):
    test_predictions = []
    skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== FOLD {fold+1}/{Config.N_SPLITS} for {model_name} =====")
        X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
        
        # ### SỬA LỖI: Chọn bộ transform phù hợp với model ###
        if model_name == 'ResNet18':
            current_train_transform = resnet_train_transform
            current_test_transform = resnet_test_transform
        else: # ViT
            current_train_transform = vit_train_transform
            current_test_transform = vit_test_transform

        train_dataset = MNISTDataset(X_train, y_train, transform=current_train_transform)
        val_dataset = MNISTDataset(X_val, y_val, transform=current_test_transform)
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        if model_name == 'ResNet18': model = ResNet18().to(Config.DEVICE)
        else: model = create_vit().to(Config.DEVICE)
            
        criterion = LabelSmoothingCrossEntropy(classes=10, smoothing=Config.LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Config.LEARNING_RATE, total_steps=len(train_loader) * Config.EPOCHS)
        
        best_accuracy = 0.0
        model_path_fold = f"/kaggle/working/{model_name}_fold_{fold}.pth"

        for epoch in range(Config.EPOCHS):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(Config.DEVICE, non_blocking=True), labels.to(Config.DEVICE, non_blocking=True)
                optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels); loss.backward(); optimizer.step(); scheduler.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(Config.DEVICE, non_blocking=True), labels.to(Config.DEVICE, non_blocking=True)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            val_accuracy = 100 * correct / len(val_dataset)
            if val_accuracy > best_accuracy: best_accuracy = val_accuracy; torch.save(model.state_dict(), model_path_fold)
        
        print(f"Fold {fold+1} Best Val Acc: {best_accuracy:.2f}%")

        model.load_state_dict(torch.load(model_path_fold))
        model.eval()
        
        # ### SỬA LỖI: Dùng đúng transform cho TTA ###
        tta_dataset = MNISTDataset(X_test, transform=current_train_transform)
        tta_loader = DataLoader(tta_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        fold_preds = []
        with torch.no_grad():
            for _ in range(Config.N_TTA):
                pass_preds = []
                for images, _ in tta_loader: images = images.to(Config.DEVICE, non_blocking=True); outputs = model(images); pass_preds.append(torch.softmax(outputs, dim=1))
                fold_preds.append(torch.cat(pass_preds))
        test_predictions.append(torch.mean(torch.stack(fold_preds), dim=0))
        
        del model, train_loader, val_loader; gc.collect(); torch.cuda.empty_cache()

    return torch.mean(torch.stack(test_predictions), dim=0)



# =============================================================================
# BƯỚC 8: THỰC THI ENSEMBLE LAI VÀ TẠO SUBMISSION
# =============================================================================
train_df = pd.read_csv(Config.TRAIN_FILE)
test_df = pd.read_csv(Config.TEST_FILE)
X = train_df.drop(columns='label').values.astype(np.uint8)
y = train_df['label'].values
X_test = test_df.values.astype(np.uint8)

print("\n--- BƯỚC 8: Bắt đầu quá trình Huấn luyện và Dự đoán ---")
print(f"\n{'='*25} BẮT ĐẦU HUẤN LUYỆN ENSEMBLE RESNET-18 {'='*25}")
resnet_preds = train_and_predict('ResNet18', X, y, X_test)

print(f"\n{'='*25} BẮT ĐẦU HUẤN LUYỆN ENSEMBLE VISION TRANSFORMER {'='*25}")
vit_preds = train_and_predict('ViT', X, y, X_test)

print(f"\n{'='*25} KẾT HỢP KẾT QUẢ CUỐI CÙNG {'='*25}")
# Gán trọng số cao hơn cho ResNet vì nó đã được chứng minh là tốt hơn
final_preds_tensor = (resnet_preds * 0.7) + (vit_preds * 0.3) # Tăng trọng số cho ResNet
_, final_labels = torch.max(final_preds_tensor, 1)
final_labels = final_labels.cpu().numpy()

submission_df = pd.DataFrame({'ImageId': np.arange(1, len(final_labels) + 1), 'Label': final_labels})
submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
print(f"File '{Config.SUBMISSION_FILE}' đã được tạo thành công!")
print("Xem trước 5 dòng đầu tiên:")
print(submission_df.head())

