import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

import torch.optim as optim


# Load FER2013 CSV
df = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')
print("Total samples in raw DataFrame:", len(df))
print("Emotion value counts:\n", df['emotion'].value_counts())


df.head(9)


# 1) Class distribution from our DataFrame `df`
emotion_counts = df['emotion'].value_counts().sort_index()
emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad",     5: "Surprise",
    6: "Neutral"
}

labels = [emotion_labels[i] for i in emotion_counts.index]


plt.figure(figsize=(8,5))
plt.bar(labels, emotion_counts.values)
plt.title("FER2013 Class Distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 2) A helper to fetch one sample per class from the PyTorch Dataset
def get_one_sample_per_class(dataset):
    seen, samples = set(), []
    for img, lbl in dataset:
        if lbl not in seen:
            seen.add(lbl)
            samples.append((img, lbl))
        if len(seen) == len(emotion_labels):
            break
    return samples



# Define basic transform
basic_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])



# Custom Dataset Class
class FER2013Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emotion = int(self.data.iloc[idx]['emotion'])
        img = np.fromstring(self.data.iloc[idx]['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, emotion




# Use the basic_transform Dataset (no augmentations)
full_dataset = FER2013Dataset(df, transform=basic_transform)
samples = get_one_sample_per_class(full_dataset)



# 3) Plot the seven samples
fig, axes = plt.subplots(1, 7, figsize=(14, 2))
for ax, (img, lbl) in zip(axes, samples):
    arr = img.squeeze().numpy()
    ax.imshow(arr, cmap='gray')
    ax.set_title(emotion_labels[lbl])
    ax.axis('off')
plt.suptitle("One SAMlpe  per Emotion Class")
plt.tight_layout()
plt.show()



# 1) Define your preprocessing transform
preprocess_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),      # ensure single channel
    transforms.ToTensor(),                            # scales pixels to [0,1]
    transforms.Normalize((0.5,), (0.5,))              # z-score normalization: (x−0.5)/0.5 ⇒ [−1,1]
])


# 2) Re-split the DataFrame (if you haven’t already) into train/val/test

# keep all 7 classes this time
df_full = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')

train_df, test_df = train_test_split(
    df_full, test_size=0.20, stratify=df_full['emotion'], random_state=42
)
train_df, val_df = train_test_split(
    train_df, test_size=0.10, stratify=train_df['emotion'], random_state=42
)



# 3) Create Dataset + DataLoaders
train_ds = FER2013Dataset(train_df, transform=preprocess_transform)
val_ds   = FER2013Dataset(val_df,   transform=preprocess_transform)
test_ds  = FER2013Dataset(test_df,  transform=preprocess_transform)


train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)


# 4) Sanity-check one batch
images, labels = next(iter(train_loader))
print(f"Batch image tensor shape: {images.shape}")   # expect [64, 1, 48, 48]
print(f"Batch label tensor shape: {labels.shape}")   # expect [64]
print(f"Pixel range: min={images.min():.3f}, max={images.max():.3f}")



from torchvision import transforms

# Define PyTorch-style augmentations
augmented_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(48, padding=4),   # Assuming image size is 48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



# Assuming FER2013Dataset is your custom dataset class
aug_train_ds = FER2013Dataset(train_df, transform=augmented_train_transform)
aug_train_loader = DataLoader(aug_train_ds, batch_size=64, shuffle=True, num_workers=2)



# Optional: visualize a few augmented samples
def visualize_augmented_samples(dataset, n=6):
    fig, axes = plt.subplots(1, n, figsize=(12, 2))
    for i in range(n):
        img, lbl = dataset[np.random.randint(len(dataset))]
        axes[i].imshow(img.squeeze().numpy(), cmap='gray')
        axes[i].set_title(f"{emotion_labels[lbl]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()



visualize_augmented_samples(aug_train_ds)


# Build CNN Model 

# 1) Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3, l2_reg=1e-4):
        super().__init__()
        # 1×1 conv for the shortcut
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        self.pool  = nn.MaxPool2d(2,2)
        self.drop  = nn.Dropout2d(dropout_prob)

        
        # Weight decay (L2) will be applied in optimizer via weight_decay

    def forward(self, x):
        # Shortcut path
        s = self.shortcut(x)
        
        # Main conv path
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        
        # Add & activate
        out = F.relu(s + y)
        out = self.pool(out)
        out = self.drop(out)
        return out






# 2) Assemble the full CNN
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Residual blocks with channels [32, 64, 128, 256]
        self.rb1 = ResidualBlock(1,  32)
        self.rb2 = ResidualBlock(32, 64)
        self.rb3 = ResidualBlock(64, 128)
        self.rb4 = ResidualBlock(128,256)
        
        # Global Average Pooling: adaptive to 1×1
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        # Final classifier
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.5)
        self.out = nn.Linear(128, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.global_pool(x)      # shape [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)    # shape [batch, 256]
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.out(x)
        return x

# 3) Instantiate and print summary
model = EmotionCNN(num_classes=7)
print(model)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")


# Test-set evaluation
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}%')



# 4) Save the model
model_path = "emotion_cnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation after each epoch
    model.eval()
    val_running_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')



import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Set model to evaluation mode
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


from sklearn.metrics import classification_report

report = classification_report(all_labels, all_preds, target_names=emotion_labels)
print("Classification Report:\n", report)



# Example assuming you tracked accuracy and loss
epochs = range(1, len(train_accuracies) + 1)

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()



# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


# Label map
emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad",     5: "Surprise",
    6: "Neutral"
}

# Ensure model is in eval mode
model.eval()

# Grab one batch
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Forward pass
with torch.no_grad():
    outputs = model(images)
    _, preds = outputs.max(1)

# Move to CPU for plotting
images = images.cpu()
labels = labels.cpu()
preds  = preds.cpu()

# Plot first 5 test images
num_display = 5
fig, axes = plt.subplots(1, num_display, figsize=(12, 3))
for i in range(num_display):
    img = images[i].squeeze().numpy()
    true_lbl = emotion_labels[int(labels[i])]
    pred_lbl = emotion_labels[int(preds[i])]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"T: {true_lbl}\nP: {pred_lbl}",
                      color=('green' if true_lbl == pred_lbl else 'red'))
    axes[i].axis('off')

plt.suptitle("Test Set: True vs. Predicted Labels")
plt.tight_layout()
plt.show()

