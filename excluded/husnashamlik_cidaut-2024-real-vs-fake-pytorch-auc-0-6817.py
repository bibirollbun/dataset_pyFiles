


pip install torch torchvision pandas scikit-learn


import os
import torch
import torchvision
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Paths
df = pd.read_csv("/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv")
train_dir = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train"
test_dir = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test"
sample_submission_csv = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv"


df['label'].value_counts()


# Hyperparameters
batch_size = 32
epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


label_encoder=LabelEncoder()
df['label']=label_encoder.fit_transform(df['label'])
print(f"Label Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")



train_data,val_data=train_test_split(df,test_size=0.20,random_state=42)


# Save split datasets to temporary CSV files (optional)
train_data.to_csv("train_split.csv", index=False)
val_data.to_csv("val_split.csv", index=False)




# Step 2: Dataset and Transforms
class DrivingSceneDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label




train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Initialize DataLoaders
train_dataset = DrivingSceneDataset("train_split.csv", train_dir, train_transform)
val_dataset = DrivingSceneDataset("val_split.csv", train_dir, val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



# Model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model = model.to(device)


# Loss and Optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")



# AUC-ROC Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            preds = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return roc_auc_score(all_labels, all_preds)




# Evaluate on Validation Set
val_auc = evaluate_model(model, val_loader)
print(f"Validation AUC-ROC: {val_auc:.4f}")



class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.images = sorted(
            [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.images[idx]  # Return image and filename



test_dataset = TestDataset(test_dir, val_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Make Predictions on Test Images
predictions = []
filenames = []

model.eval()
with torch.no_grad():
    for images, image_names in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze(1)
        preds = torch.sigmoid(outputs).cpu().numpy()  # Convert to NumPy array
        predictions.extend(preds)
        filenames.extend(image_names)  # Collect filenames as strings



# Convert Predictions to Binary Labels (Threshold = 0.5)
binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

# Create Submission DataFrame
submission = pd.DataFrame({
    'image': filenames,  # Use the filenames directly
    'label': binary_predictions
})

# Save Submission File
submission.to_csv("submission.csv", index=False)



submission




