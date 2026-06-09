#in Google Colab only


import os

data_path = '/kaggle/input/sheep-classification-challenge-2025/'
os.listdir(data_path)


import os

data_path = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/'
os.listdir(data_path)



import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score


# Load data
df = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts()}")


train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),  # Ø®Ù�Ù‘Ø¶ØªÙ‡Ø§ Ù…Ù† 10 Ø¥Ù„Ù‰ 5
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Ø®Ù�Ù�Ù†Ø§
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # ÙŠØ¬Ø¨ Ø£Ù† ÙŠØ¬ÙŠ Ù‚Ø¨Ù„ Normalize
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class SheepDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(dataframe['label'].unique()))}
        self.df['label'] = self.df['label'].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


label_counts = train_df['label'].value_counts()
class_weights = 1. / label_counts
train_df['weight'] = train_df['label'].map(class_weights)
sample_weights = train_df['weight'].values

train_dataset = SheepDataset(train_df, '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/', transform=train_transform)
val_dataset = SheepDataset(val_df, '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/', transform=val_transform)

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


num_classes = df['label'].nunique()
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)


def train_model(model, train_loader, val_loader, epochs=50, patience=10):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import accuracy_score, f1_score
    import copy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, 1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(f1)

        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f} Acc: {acc:.4f} F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, "best_model.pth")
            print("âœ… Best model saved with F1:", best_f1)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"â�³ No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"ğŸ›‘ Early stopping triggered after {epoch+1} epochs.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


model = train_model(model, train_loader, val_loader)


model.load_state_dict(torch.load("best_model.pth"))
model.eval()
torch.save(model.state_dict(), '/kaggle/working/best_model.pth')



import torch
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import pandas as pd

# Ø§Ù„ØªØ­Ù‚Ù‚ Ù…Ù† ÙˆØ¬ÙˆØ¯ Ø§Ù„Ù€ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ØªØ­Ø¯ÙŠØ¯ Ø§Ù„Ù…Ø³Ø§Ø± Ø§Ù„ØµØ­ÙŠØ­ Ù„Ù…Ù„Ù� Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø§Ù„Ù…Ø­Ù�ÙˆØ¸ Ù�ÙŠ /kaggle/working/
model_path = '/kaggle/working/best_model.pth'

# Ø§Ù„ØªØ­Ù‚Ù‚ Ù…Ù† ÙˆØ¬ÙˆØ¯ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ù�ÙŠ Ø§Ù„Ù…Ø³Ø§Ø± Ø§Ù„Ù…Ø­Ø¯Ø¯
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))  # ØªØ­Ù…ÙŠÙ„ Ø£Ù�Ø¶Ù„ Ù†Ù…ÙˆØ°Ø¬
    model = model.to(device)
    model.eval()
else:
    print("Ù„Ù… ÙŠØªÙ… Ø§Ù„Ø¹Ø«ÙˆØ± Ø¹Ù„Ù‰ Ø§Ù„Ù…Ù„Ù�ØŒ ØªØ£ÙƒØ¯ Ù…Ù† Ø§Ù„Ù…Ø³Ø§Ø± Ø§Ù„ØµØ­ÙŠØ­ Ù„Ù„Ù†Ù…ÙˆØ°Ø¬!")

# ØªØ¹Ø±ÙŠÙ� Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª Ø§Ù„ØªÙŠ Ø³ØªÙ�Ø·Ø¨Ù‚ Ø¹Ù„Ù‰ Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ØªØ­Ù…ÙŠÙ„ Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„ØªØ³Ù…ÙŠØ§Øª (labels) Ù…Ù† Ù…Ù„Ù� CSV
train_labels_path = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
df = pd.read_csv(train_labels_path)

# ØªØ¹ÙŠÙŠÙ† Ø§Ù„Ù…ØµÙ�ÙˆÙ�Ø© Ø¨ÙŠÙ† Ø§Ù„Ø£Ø³Ù…Ø§Ø¡ ÙˆØ§Ù„Ù�Ù‡Ø§Ø±Ø³
label_map = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
idx_to_label = {v: k for k, v in label_map.items()}

# ØªØ­Ø¯ÙŠØ¯ Ù…Ø³Ø§Ø± Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±
test_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test/'
test_images = os.listdir(test_dir)

# Ø§Ø®ØªÙŠØ§Ø± 7 ØµÙˆØ± Ø¹Ø´ÙˆØ§Ø¦ÙŠØ© Ù…Ù† Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±
random_images = random.sample(test_images, 7)

# Ø§Ù„ØªÙ†Ø¨Ø¤ Ø¨Ù€ 7 ØµÙˆØ± Ø¹Ø´ÙˆØ§Ø¦ÙŠØ© Ù…Ù† Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±
for img_name in random_images:
    img_path = os.path.join(test_dir, img_name)

    image = Image.open(img_path).convert("RGB")
    input_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        conf, pred_idx = torch.max(probabilities, 1)
        pred_label = idx_to_label[pred_idx.item()]
        confidence = conf.item() * 100

    plt.imshow(image)
    plt.title(f"Predicted: {pred_label} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
model = model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

label_map = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
idx_to_label = {v: k for k, v in label_map.items()}

test_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test/'
test_images = os.listdir(test_dir)
results = []

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()
        pred_label = idx_to_label[pred_idx]

    results.append({'filename': img_name, 'label': pred_label})


submission = pd.DataFrame(results)
submission.to_csv("submission(0.96).csv", index=False)
print("Done")

