import os, random, shutil, math
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import timm


original_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train"
labels_csv = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load data
df = pd.read_csv(labels_csv)


df.head()


def show_images_with_labels(data_dir, labels_df, num_images=50, num_cols=10):
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
    num_images_to_show = min(num_images, len(image_files))
    num_rows = math.ceil(num_images_to_show / num_cols)

    plt.figure(figsize=(15, num_rows * 3))

    for i in range(num_images_to_show):
        img_filename = image_files[i]
        img_path = os.path.join(data_dir, img_filename)

        # Find the label for the current image filename in the labels_df
        label_row = labels_df[labels_df['filename'] == img_filename]
        label = label_row['label'].iloc[0] if not label_row.empty else "Label Not Found"

        img = Image.open(img_path)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


show_images_with_labels(original_dir, df)


countdf=df['label'].value_counts()
countdf


#Unbalance data ^^
#let's do augmentation


augmented_dir = "Sheep Classification Images/augmented"
target_per_class = 300 #maxmize data for learning


os.makedirs(augmented_dir, exist_ok=True)


#Copy original images to the augmented folder
new_rows = []
for _, row in df.iterrows():
    label = row['label']
    fname = row['filename']
    src = os.path.join(original_dir, fname)
    dst_dir = os.path.join(augmented_dir, label)
    dst = os.path.join(dst_dir, fname)
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        new_rows.append({'filename': fname, 'label': label})


augment = transforms.Compose([
    transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Resize((300, 300))  # for PIL saving
])


# Apply augmentation to underrepresented classes


label_counts = pd.DataFrame(new_rows)['label'].value_counts()
for label in label_counts.index:
    class_dir = os.path.join(augmented_dir, label)
    files = [f for f in os.listdir(class_dir) if f.endswith('.jpg') or f.endswith('.png')]
    current_count = len(files)
    if current_count >= target_per_class:
        continue
    needed = target_per_class - current_count
    for i in range(needed):
        fname = random.choice(files)
        img_path = os.path.join(class_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            aug_img = augment(img)
            new_name = f"{os.path.splitext(fname)[0]}_aug{i}.jpg"
            aug_img.save(os.path.join(class_dir, new_name))
            new_rows.append({'filename': new_name, 'label': label})
        except Exception as e:
            print(f"Error augmenting {fname}: {e}")


updated_df = pd.DataFrame(new_rows)
updated_df.to_csv("Sheep Classification Images/cleaned_labels.csv", index=False)# new csv has new labels of augmentd images


label_to_idx = {label: idx for idx, label in enumerate(sorted(updated_df['label'].unique()))}
print(sorted(updated_df['label'].unique()))


train_df, val_df = train_test_split(
    updated_df,
    test_size=0.2,
    stratify=updated_df['label'],
    random_state=42
)


train_df['label'].value_counts()


val_df['label'].value_counts()


label_to_idx = {label: idx for idx, label in enumerate(sorted(updated_df['label'].unique()))}
idx_to_label = {v: k for k, v in label_to_idx.items()}


class Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, label_to_idx=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.df['label_idx'] = self.df['label'].map(self.label_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['label'], row['filename'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row['label_idx']


train_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_dataset = Dataset(train_df, augmented_dir, transform=train_transform, label_to_idx=label_to_idx)
val_dataset = Dataset(val_df, augmented_dir, transform=val_transform, label_to_idx=label_to_idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


#convnext_base(CNN and Transformer-style training)


model = timm.create_model("convnext_tiny", pretrained=True, num_classes=7)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)


best_val_acc = 0
train_losses, val_accuracies = [], []
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))

    #Validation
    model.eval()
    val_correct, val_total = 0, 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_accuracies.append(val_acc)
    scheduler.step(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("model Saved")



#Final Classification Report
print(classification_report(val_labels, val_preds, target_names=list(label_to_idx.keys())))


#confusion_matrix
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

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_to_idx.keys()))
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


#Evaluation


model.load_state_dict(torch.load("/kaggle/working/best_model.pth", map_location=device))
model.to(device)
model.eval()


label_to_idx = {
    'Barbari': 0,
    'Goat': 1,
    'Harri': 2,
    'Naeimi': 3,
    'Najdi': 4,
    'Roman': 5,
    'Sawakni': 6
}
idx_to_label = {v: k for k, v in label_to_idx.items()}


transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


test_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test"
results = []


for filename in os.listdir(test_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(test_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
            label = idx_to_label[pred.item()]
            results.append({"filename": filename, "label": label})


#show predict
plt.figure(figsize=(16, 10))
num_images = min(len(results), 20)

for i in range(num_images):
    item = results[i]
    img_path = os.path.join(test_dir, item['filename'])
    img = Image.open(img_path).convert("RGB")

    plt.subplot(5, 10, i + 1)
    plt.imshow(img)
    plt.title(f"Predicted: {item['label']}", fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()

