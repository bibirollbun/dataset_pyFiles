import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# 1. 경로 및 장치 설정
TRAIN_CSV = "/kaggle/input/butterfly-species-classification-dl-pr/butterfly_competition/train.csv"
TEST_CSV  = "/kaggle/input/butterfly-species-classification-dl-pr/butterfly_competition/test.csv"
TRAIN_IMG_DIR = "/kaggle/input/butterfly-species-classification-dl-pr/butterfly_competition/train"
TEST_IMG_DIR  = "/kaggle/input/butterfly-species-classification-dl-pr/butterfly_competition/test"

device = torch.device("cpu")  # CPU 전용





# 2. CSV 로드 및 레이블 인코딩
df = pd.read_csv(TRAIN_CSV)
label2id = {lbl:idx for idx,lbl in enumerate(sorted(df['label'].unique()))}
id2label = {v:k for k,v in label2id.items()}
df['label_idx'] = df['label'].map(label2id)




# 3. Train/Validation 분리 (stratify 해서 클래스 불균형 완화)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label_idx'], random_state=42
)




# 4. Dataset 클래스 정의
class ButterflyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.loc[idx, 'filename']
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, fname
        label = self.df.loc[idx, 'label_idx']
        return img, label




# 5. 데이터 전처리(Transforms)
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
test_transforms = val_transforms




# 6. DataLoader
batch_size = 32

train_ds = ButterflyDataset(train_df, TRAIN_IMG_DIR, transform=train_transforms)
val_ds   = ButterflyDataset(val_df,   TRAIN_IMG_DIR, transform=val_transforms)
test_df  = pd.read_csv(TEST_CSV)
test_ds  = ButterflyDataset(test_df,  TEST_IMG_DIR,  transform=test_transforms, is_test=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)




# 7. 모델 정의 (ResNet18)
model = models.resnet18(pretrained=True)
n_features = model.fc.in_features
model.fc = nn.Linear(n_features, len(label2id))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)




# 8. 학습·검증 함수
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(train_ds)
    acc = correct / len(train_ds)
    return avg_loss, acc

def validate():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(val_ds)
    acc = correct / len(val_ds)
    return avg_loss, acc




# 9. 메인 루프
n_epochs = 5
for epoch in range(1, n_epochs+1):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc     = validate()
    print(f"Epoch {epoch:02d} | "
          f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
          f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")




# (이전까지는 동일)

# 10. 테스트셋 예측 및 submission.csv 생성
model.eval()
preds = []
ids   = []
with torch.no_grad():
    for imgs, names in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        idxs = outputs.argmax(dim=1).cpu().numpy()
        preds.extend(idxs)
        ids.extend(names)

submission = pd.DataFrame({
    "id":    ids,                            # <-- 'filename'이 아니라 'id' 로 컬럼명 변경
    "label": [id2label[i] for i in preds]
})
submission.to_csv("baseline__submission.csv", index=False)
print("=> baseline__submission.csv 생성 완료")



































