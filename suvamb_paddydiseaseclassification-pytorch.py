!pip install torch_lr_finder


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from pathlib import Path
from torch_lr_finder import LRFinder

path = Path("/kaggle/input/paddy-disease-classification")
trn_path = path / "train_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


train_tfms = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomResizedCrop(128, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

valid_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


full_ds = datasets.ImageFolder(trn_path, transform=train_tfms)
n_val = int(0.2 * len(full_ds))
n_train = len(full_ds) - n_val
train_ds, valid_ds = random_split(full_ds, [n_train, n_val])
valid_ds.dataset.transform = valid_tfms

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


model = timm.create_model("resnet26d", pretrained=True, num_classes=len(full_ds.classes))
model = model.to(device) # fp16 training


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, dl, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += preds.argmax(dim=1).eq(yb).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)

def validate(model, dl, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            correct += preds.argmax(dim=1).eq(yb).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)


def find_lr(model, train_dl, criterion, optimizer):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_dl, end_lr=1, num_iter=100)
    lr_finder.plot()  # This will plot loss vs lr
    lr_finder.reset()


print("Running learning rate finder...")
find_lr(model, train_dl, criterion, optimizer)





epochs = 3
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc = validate(model, valid_dl, criterion)
    print(f"Epoch {epoch+1}/{epochs}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


tst_path = path/"test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same transforms as validation (no augmentation!)
test_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



test_ds = datasets.ImageFolder(
    root=path, 
    transform=test_tfms,
    loader=datasets.folder.default_loader
)



from glob import glob
test_files = sorted(glob(str(tst_path/"*.jpg")))
test_ds.samples = [(f, 0) for f in test_files]
test_ds.targets = [0] * len(test_files)

test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)



model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(device)
        preds = model(xb)
        probs = torch.softmax(preds, dim=1)
        all_preds.append(probs.argmax(dim=1).cpu())

idxs = torch.cat(all_preds)   # predicted class indices



# Map indices back 
import pandas as pd
class_names = train_dl.dataset.dataset.classes  # from ImageFolder train dataset
mapping = dict(enumerate(class_names))

results = pd.Series(idxs.numpy(), name="label").map(mapping)


ss = pd.read_csv(path/"sample_submission.csv")
ss['label'] = results.values
ss.to_csv("submission.csv", index=False)

!head submission.csv



# Your solution here



""" 
train_tfms = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomResizedCrop(128, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),  # Added random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
"""


# Your solution here



""" 
# Add this import at the top
from torch.optim.lr_scheduler import StepLR

# In the main training loop, after initializing the optimizer:

scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # Initialize the scheduler

# Modify the training loop to step the scheduler
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc = validate(model, valid_dl, criterion)
    print(f"Epoch {epoch+1}/{epochs}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    scheduler.step()  # Step the scheduler after each epoch
""" 


# Your solution here



""" 
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

# At the end of the training loop, call the save_model function:
save_model(model, 'trained_model.pth')  # Save the model state
"""


# Your solution here



"""
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(full_ds.classes))
"""


# Your solution here



""" 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Modify the validate function to return predictions and true labels:

def validate(model, dl, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            correct += preds.argmax(dim=1).eq(yb).sum().item()
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    return total_loss / len(dl.dataset), correct / len(dl.dataset), all_labels, all_preds

# In the training loop, after validation:
val_loss, val_acc, val_labels, val_preds = validate(model, valid_dl, criterion)
plot_confusion_matrix(val_labels, val_preds, classes=full_ds.classes) 
"""





