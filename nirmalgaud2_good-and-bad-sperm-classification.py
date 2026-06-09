import numpy as np 
import pandas as pd 
import os


import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


bad_sperm_path = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Bad Sperm'
good_sperm_path = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm'


data = []

for img in os.listdir(bad_sperm_path):
    image_full_path = os.path.join(bad_sperm_path, img)
    data.append([image_full_path, 'bad'])

for img in os.listdir(good_sperm_path):
    image_full_path = os.path.join(good_sperm_path, img)
    data.append([image_full_path, 'good'])

df = pd.DataFrame(data, columns=['image_path', 'label'])


df.head()


df.tail()


df.shape


df.columns


df.duplicated().sum()


df.isnull().sum()


df.info()


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x="label", palette="viridis", ax=ax)

ax.set_title("Distribution of Sperm Types", fontsize=14, fontweight='bold')
ax.set_xlabel("Tumor Type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=11, color='black', 
                xytext=(0, 5), textcoords='offset points')

plt.show()

label_counts = df["label"].value_counts()

fig, ax = plt.subplots(figsize=(10, 8))
colors = sns.color_palette("viridis", len(label_counts))

ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
       startangle=140, colors=colors, textprops={'fontsize': 8, 'weight': 'bold'},
       wedgeprops={'edgecolor': 'black', 'linewidth': 1})

ax.set_title("Distribution of Sperm Types - Pie Chart", fontsize=14, fontweight='bold')

plt.show()


from PIL import Image

num_images = 5

unique_labels = df['label'].unique()

plt.figure(figsize=(15, len(unique_labels) * 3))

for row_idx, label in enumerate(unique_labels):
  
    label_images = df[df['label'] == label].head(num_images)['image_path'].tolist()
    
    for col_idx, img_path in enumerate(label_images):
        plt_idx = row_idx * num_images + col_idx + 1
        plt.subplot(len(unique_labels), num_images, plt_idx)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        if col_idx == 2:  
            plt.title(label, fontsize=10)

plt.tight_layout()
plt.show()


max_samples = df['label'].value_counts().max()

balanced_df = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=max_samples, replace=True, random_state=42)
).reset_index(drop=True)

balanced_df = balanced_df[['image_path', 'label']]


df = balanced_df


df


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models

warnings.filterwarnings("ignore")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print("\n--- Data Split Validation ---")
print("Test Set Label Distribution:")
print(test_df['label'].value_counts())
print("-----------------------------\n")

class SpermDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self): return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        true_label_string = row['label']
        if true_label_string == 'good':
            label = 1
        elif true_label_string == 'bad':
            label = 0
        else:
            label = 0 
            
        return img, label

batch_size = 32
train_loader = DataLoader(SpermDataset(train_df), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(SpermDataset(val_df),   batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(SpermDataset(test_df),  batch_size=batch_size, shuffle=False, num_workers=4)

class ResidualSurprisePredictor(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*2), nn.GELU(),
            nn.Linear(dim*2, dim*2), nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        self.register_buffer('m1', torch.zeros(dim))
        self.register_buffer('m2', torch.zeros(dim))
        self.alpha = 0.99

    def forward(self, h):

        self.m1 = self.m1.detach()
        self.m2 = self.m2.detach()
        
        batch_mean = h.mean(dim=0)
        
        self.m1 = self.alpha * self.m1 + (1 - self.alpha) * batch_mean
        self.m2 = self.alpha * self.m2 + (1 - self.alpha) * self.m1
        
        predicted_surprise_vector = self.net(self.m2).expand(h.size(0), -1)
        u_t = predicted_surprise_vector + self.m2.expand(h.size(0), -1) 
        
        return u_t

class SPERMNestedClassifier(nn.Module):
    def __init__(self, num_classes=2, dim=512):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.dim = dim
        
        self.W_fast = nn.Parameter(torch.randn(dim, dim) * 0.02)
        
        self.surprise_net = ResidualSurprisePredictor(dim) 
        
        self.slow_level1 = nn.Linear(dim, dim)
        self.slow_level2 = nn.Linear(dim, dim)
        self.register_buffer('step', torch.tensor(0))
        
        self.sparsity_temp = 0.1 
        self.sparsity_reg_lambda = 0.0001 

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        self.step += 1
        
        f = self.backbone(x)
        h = self.pool(f).flatten(1)
        h_norm = F.normalize(h, dim=-1)

        u_t = self.surprise_net(h)

        h_outer = torch.bmm(h_norm.unsqueeze(2), h_norm.unsqueeze(1))
        u_outer = torch.bmm(u_t.unsqueeze(2), h_norm.unsqueeze(1))
        
        delta_W_batched = -u_outer + torch.bmm(self.W_fast.expand(b, -1, -1), -h_outer)
        delta_W = delta_W_batched.mean(0)
        
        sparsity_mask = torch.sigmoid(delta_W / self.sparsity_temp)
        sparse_delta_W = delta_W * sparsity_mask
        
        with torch.no_grad():
            self.W_fast.data.add_(sparse_delta_W * 0.1) 
        
        h_fast = torch.matmul(h_norm, self.W_fast)

        h_slow = h_fast
        if self.training:
            if self.step % 4 == 0:
                h_slow = h_slow + self.slow_level1(h_slow)
            if self.step % 64 == 0:
                h_slow = h_slow + self.slow_level2(h_slow)
        else:
            h_slow = h_slow + self.slow_level1(h_slow)
            h_slow = h_slow + self.slow_level2(h_slow)

        logits = self.classifier(h_slow)
        
        sparsity_reg_loss = self.sparsity_reg_lambda * torch.norm(sparsity_mask, p=1)
        
        return logits, sparsity_reg_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SPERMNestedClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
patience = 5 
epochs = 10
min_delta = 0.001 
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc = 0.0

for epoch in range(epochs):
    if early_stop:
        print("\nStopping early due to non-improving validation loss.")
        break

    model.train()
    train_loss, correct, total = 0, 0, 0
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits, sparsity_reg_loss = model(imgs)
        loss = criterion(logits, labels) + sparsity_reg_loss 
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        if i % 30 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")
    
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            val_loss += criterion(logits, labels).item()
            val_correct += (logits.argmax(1) == labels).sum().item()
    
    current_val_loss = val_loss / len(val_loader)
    train_acc = correct / total
    val_acc = val_correct / len(val_loader.dataset)
    
    scheduler.step(current_val_loss)

    history["train_loss"].append(train_loss/len(train_loader))
    history["train_acc"].append(train_acc)
    history["val_loss"].append(current_val_loss)
    history["val_acc"].append(val_acc)
    
    print(f"\nEPOCH {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {current_val_loss:.4f} | Acc: {val_acc:.4f}")
    
    if current_val_loss < best_val_loss - min_delta:
        best_val_loss = current_val_loss
        epochs_no_improve = 0
        best_acc = val_acc
        torch.save(model.state_dict(), "best_sperm_final.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            early_stop = True

try:
    model.load_state_dict(torch.load("best_sperm_final.pth"))
except FileNotFoundError:
    print("Warning: Best model weights not found. Using last epoch's weights.")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        logits, _ = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)

print(f"\nFINAL TEST ACCURACY: {acc*100:.2f}%")
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=["Bad", "Good"], zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
plt.title('SPERM Nested Learning – Sperm Morphology Detection')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.show()

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.legend(); plt.title('Accuracy')
plt.show()

print(f"\nSPERM Nested Learning (SPERM) successfully trained!")

