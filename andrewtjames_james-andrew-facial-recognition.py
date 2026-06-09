import pandas as pd
import numpy as np
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt, seaborn as sns


# Constants
BATCH_SIZE = 64
EPOCHS = 30
BASE_LR = 1e-3
PATIENCE = 6
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

torch.manual_seed(SEED); np.random.seed(SEED)


# -----------------
# Load Data
# -----------------
def load_data(path):
    df = pd.read_csv(path)
    df = df[df['emotion'] < 7]
    return df
    
# Compute dataset mean/std for Normalize (sampled for speed)
def compute_mean_std(df, sample=25000):
    idx = np.random.choice(len(df), size=min(sample, len(df)), replace=False)
    arr = np.stack([np.fromstring(df.pixels[i], sep=' ', dtype=np.float32).reshape(48,48)/255. for i in idx])
    mean = float(arr.mean()); std = float(arr.std() + 1e-8)
    return mean, std



# -----------------
# Dataset
# -----------------
class FERDataset(Dataset):
    def __init__(self, df, tf):
        self.y = df.emotion.values.astype(np.int64)
        self.X = [np.fromstring(p, sep=' ', dtype=np.uint8).reshape(48,48) for p in df.pixels]
        self.tf = tf
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.tf(self.X[i]), self.y[i]




def build_transforms(mean, std):
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0) # cutout-like
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    return train_tf, val_tf


# ---------------------
# Model
# ---------------------
class CNN(nn.Module):
    def __init__(self, n=7, p=0.35):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()
            )
        self.b1 = block(1, 32);  self.p1 = nn.MaxPool2d(2)  # 24x24
        self.b2 = block(32,64);  self.p2 = nn.MaxPool2d(2)  # 12x12
        self.b3 = block(64,96);  self.p3 = nn.MaxPool2d(2)  # 6x6
        self.b4 = nn.Sequential(nn.Conv2d(96,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)
        self.fc  = nn.Linear(128, n)
    def forward(self, x):
        x = self.p1(self.b1(x))
        x = self.p2(self.b2(x))
        x = self.p3(self.b3(x))
        x = self.b4(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)




class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.reduction = reduction
    def forward(self, logits, target):
        ce = self.ce(logits, target)                 # [B]
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze() + 1e-8
        loss = (1 - pt)**self.gamma * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()


# ---------------------
# Utilities Imbalance
# ---------------------
def make_balanced_sampler(train_df):
    counts = train_df.emotion.value_counts().to_dict()
    class_w = {c: 1.0 / counts[c] for c in counts}
    weights = [class_w[y] for y in train_df.emotion.values]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------
# Training & Evaluation
# ---------------------
def epoch_eval(model, loader, criterion=None):
    model.eval()
    total_loss = 0.0; n_batches = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            if criterion is not None:
                total_loss += criterion(out, yb).item(); n_batches += 1
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS.keys()), zero_division=0)
    macro_f1 = f1.mean()
    val_loss = (total_loss/n_batches) if n_batches>0 else None
    return val_loss, (prec, rec, f1, sup, macro_f1), (y_true, y_pred)

def train(model, train_loader, val_loader, epochs, base_lr):
    criterion = FocalLoss(gamma=2.0)
    opt = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 10))
    best_f1 = -1.0; wait = 0
    t_losses, v_losses, f1_hist = [], [], []

    for ep in range(1, epochs+1):
        model.train(); run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward(); opt.step()
            run += loss.item()
        t_losses.append(run/len(train_loader))

        v_loss, metrics, _ = epoch_eval(model, val_loader, criterion)
        v_losses.append(v_loss if v_loss is not None else np.nan)
        macro_f1 = metrics[-1]; f1_hist.append(macro_f1)
        print(f"Epoch {ep}/{epochs} | Train {t_losses[-1]:.4f} | Val {v_losses[-1]:.4f} | Macro-F1 {macro_f1:.3f}")

        # early stop on macro-F1
        if macro_f1 > best_f1 + 1e-3:
            best_f1, wait = macro_f1, 0
            torch.save(model.state_dict(), "best_fix.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping (macro-F1 plateau)."); break
        sched.step()

    model.load_state_dict(torch.load("best_fix.pt", map_location=DEVICE))
    return t_losses, v_losses, f1_hist

def plot_losses(t, v):
    plt.figure(figsize=(6,4))
    plt.plot(t, label="Train Loss"); plt.plot(v, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.tight_layout(); plt.show()

def final_report(model, val_loader):
    _, (_, _, f1, _, macro_f1), (y, yhat) = epoch_eval(model, val_loader, None)
    print(f"Final Macro-F1: {macro_f1:.3f}\n")
    print("Classification Report:\n",
          classification_report(y, yhat, target_names=list(LABELS.values()), digits=3))
    # F1 bars
    plt.figure(figsize=(8,4))
    plt.bar(list(LABELS.values()), f1); plt.ylim(0,1)
    plt.title("Per-class F1"); plt.xticks(rotation=30); plt.tight_layout(); plt.show()
    # Confusion matrix
    cm = confusion_matrix(y, yhat, labels=list(LABELS.keys()))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(LABELS.values()), yticklabels=list(LABELS.values()))
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.show()




# -----------------
# Plots
# -----------------
def plot_losses(tloss, vloss):
    plt.figure(figsize=(6,4))
    plt.plot(tloss, label="Train Loss")
    plt.plot(vloss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(LABELS.values()),
                yticklabels=list(LABELS.values()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=LABELS.values()))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(LABELS.values()), yticklabels=list(LABELS.values()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# -----------------
# Execution
# -----------------
df = load_data("/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv")

mean, std = compute_mean_std(df)
print(f"Normalize with mean={mean:.4f}, std={std:.4f}")

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.emotion, random_state=SEED)
tf_train, tf_val = build_transforms(mean, std)

train_ds = FERDataset(train_df, tf_train)
val_ds   = FERDataset(val_df,  tf_val)

sampler = make_balanced_sampler(train_df)   # balanced batches
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = CNN().to(DEVICE)
t_loss, v_loss, f1_hist = train(model, train_loader, val_loader, EPOCHS, BASE_LR)
plot_losses(t_loss, v_loss)
final_report(model, val_loader)

