import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import timm  # For advanced models
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.optim as optim



# Define dataset path
DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"

print("Loading dataset...")
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df["diagnosis"] = df["diagnosis"].astype(int)

# Split dataset
print("Splitting dataset...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["diagnosis"])



class DRDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, "train_images", self.dataframe.iloc[idx, 0] + ".png")
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx, 1]
        return image, label



print("Applying transformations...")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Creating datasets and dataloaders...")
train_dataset = DRDataset(train_df, DATA_DIR, transform=transform)
val_dataset = DRDataset(val_df, DATA_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)



print("Initializing models...")
class DRModel(nn.Module):
    def __init__(self, model_name):
        super(DRModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=5)
        self.dropout = nn.Dropout(0.4)  # Added dropout to reduce overfitting
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x



model1 = DRModel("swin_large_patch4_window7_224").cuda()



model2 = DRModel("tf_efficientnet_b7").cuda()



model3 = DRModel("convnext_large").cuda()



print("Starting training...")

def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)  # Lower LR and weight decay
    best_val_loss = float('inf')
    patience = 3
    early_stop_count = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {100*correct/total:.2f}%")
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered.")
                break



train_model(model1, train_loader, val_loader)



def test_model(model, dataloader):
    model.eval()  # Set to evaluation mode
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            total_preds.extend(preds)
            total_labels.extend(labels.numpy())

    accuracy = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average='weighted')
    print(f"Model Test Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    return accuracy, f1



print("Testing Model 1 (Swin Large)...")
acc1, f1_1 = test_model(model1, val_loader)



train_model(model2, train_loader, val_loader)



print("Testing Model 2 (EfficientNet B7)...")
acc2, f1_2 = test_model(model2, val_loader)




train_model(model3, train_loader, val_loader)



print("Testing Model 3 (ConvNeXt Large)...")
acc3, f1_3 = test_model(model3, val_loader)



print("Starting weighted fusion prediction...")

def weighted_fusion(models, weights, dataloader):
    models = [m.eval() for m in models]
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            outputs = [m(images) for m in models]
            fused_output = sum(w * o for w, o in zip(weights, outputs)) / sum(weights)
            preds = torch.argmax(fused_output, dim=1).cpu().numpy()
            total_preds.extend(preds)
            total_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average='weighted')
    print(f"Fusion Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")



weights = [0.5, 0.0, 0.5]  # Adjusted based on validation performance
weighted_fusion([model1, model2, model3], weights, val_loader)



import torch

# Save individual model states and fusion weights
torch.save({
    'model1_state_dict': model1.state_dict(),
    'model2_state_dict': model2.state_dict(),
    'model3_state_dict': model3.state_dict(),
    'fusion_weights': weights
}, "/kaggle/working/weighted_fusion_model.pth")

print("Weighted fusion model and weights saved successfully at /kaggle/working/weighted_fusion_model.pth")



# Load the saved model and weights
checkpoint = torch.load("/kaggle/working/weighted_fusion_model.pth")

# Restore model states
model1.load_state_dict(checkpoint['model1_state_dict'])
model2.load_state_dict(checkpoint['model2_state_dict'])
model3.load_state_dict(checkpoint['model3_state_dict'])

# Restore fusion weights
weights = checkpoint['fusion_weights']

print("Weighted fusion model and weights loaded successfully!")



# Save all three models
model1_path = "/kaggle/working/model1.pth"
model2_path = "/kaggle/working/model2.pth"
model3_path = "/kaggle/working/model3.pth"

torch.save(model1.state_dict(), model1_path)
torch.save(model2.state_dict(), model2_path)
torch.save(model3.state_dict(), model3_path)

# Download models to local computer
from IPython.display import FileLink

display(FileLink(model1_path))
display(FileLink(model2_path))
display(FileLink(model3_path))





