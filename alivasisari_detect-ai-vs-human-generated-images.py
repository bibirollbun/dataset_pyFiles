import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split


class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Map numeric labels to themselves (if label is already numeric)
        self.label_mapping = {label: label for label in self.annotations['label'].unique()}
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Read the file name and label from the correct columns
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])  # `file_name`
        image = Image.open(img_name).convert("RGB")
        label = self.annotations.iloc[idx, 2]  # `label`
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.filenames[idx]



transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.0, hue=0.3),
    # transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     # transforms.RandomRotation(15),
#     # transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


train_dataset = TrainDataset(csv_file='/kaggle/input/ai-vs-human-generated-dataset/train.csv', root_dir='/kaggle/input/ai-vs-human-generated-dataset/', transform=transform)
test_dataset = TestDataset(root_dir='/kaggle/input/ai-vs-human-generated-dataset/test_data_v2', transform=transform)

train_size = int(0.5 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



image, label = train_dataset[0]
print("Label (Numeric):", label)


image, label = test_dataset[0]
print("Label (Numeric):", label)
print("Image :", image)


# from collections import Counter

# train_labels = [train_dataset[idx][1] for idx in train_data.indices]
# train_label_counts = Counter(train_labels)

# val_labels = [train_dataset[idx][1] for idx in val_data.indices]
# val_label_counts = Counter(val_labels)

# print("Training Labels Distribution:", train_label_counts)
# print("Validation Labels Distribution:", val_label_counts)



import matplotlib.pyplot as plt

data_iter = iter(train_loader)
images, labels = next(data_iter)

image = images[0].numpy().transpose((1, 2, 0))
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image = std * image + mean
image = image.clip(0, 1)

plt.imshow(image)
plt.title(f"Label: {labels[0].item()}")
plt.axis('off')
plt.show()


from transformers import (
    ViTForImageClassification,
    DeiTForImageClassification,
    SwinForImageClassification,
    ConvNextForImageClassification,
    ResNetForImageClassification,
    EfficientNetForImageClassification,
    BeitForImageClassification,
    AutoFeatureExtractor,
)
import torch
from torchvision import models

model_names = {
    "vit": "google/vit-base-patch16-224-in21k",
    "deit": "facebook/deit-base-distilled-patch16-224",
    "swin": "microsoft/swin-base-patch4-window7-224",
    "convnext": "facebook/convnext-base-224",
    "resnet": "microsoft/resnet-50",
    "efficientnet": "google/efficientnet-b0",
    "regnet": "facebook/regnet-y-8gf",
    "beit": "microsoft/beit-base-patch16-224",
    "convnext_large": "facebook/convnext-large-224",
    "swin_large": "microsoft/swin-large-patch4-window12-384",
    "vit_large": "google/vit-large-patch32-384",
    "efficientnet_v2_m": "efficientnet_v2_m",
    "alexnet": "alexnet",
    "vgg16": "vgg16",
    "densenet": "densenet121",
    "inception": "inception_v3",
    "mobilenet_v2": "mobilenet_v2",
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
}

selected_model_key = "swin"
selected_model_name = model_names[selected_model_key]
if selected_model_key == "vit":
    model = ViTForImageClassification.from_pretrained(
        selected_model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
elif selected_model_key == "deit":
    model = DeiTForImageClassification.from_pretrained(
        selected_model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
elif selected_model_key == "swin" or selected_model_key == "swin_large":
    model = SwinForImageClassification.from_pretrained(
        selected_model_name, 
        num_labels=2, 
        attention_probs_dropout_prob = 0.05,
        hidden_dropout_prob = 0.05,
        ignore_mismatched_sizes=True
    )
elif selected_model_key == "convnext":
    model = ConvNextForImageClassification.from_pretrained(
        selected_model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
elif selected_model_key == "resnet":
    model = ResNetForImageClassification.from_pretrained(
        selected_model_name, num_labels=2, ignore_mismatched_sizes=True
    )
elif selected_model_key == "efficientnet":
    model = EfficientNetForImageClassification.from_pretrained(
        selected_model_name, num_labels=2, ignore_mismatched_sizes=True
    )
elif selected_model_key == "beit":
    model = BeitForImageClassification.from_pretrained(
        selected_model_name, num_labels=2, ignore_mismatched_sizes=True
    )
elif selected_model_key == "efficientnet_v2_m":
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2) 
elif selected_model_key == "alexnet":
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
elif selected_model_key == "vgg16":
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
elif selected_model_key == "densenet":
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
elif selected_model_key == "inception":
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
elif selected_model_key == "mobilenet_v2":
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
elif selected_model_key == "mobilenet_v3_small":
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
elif selected_model_key == "mobilenet_v3_large":
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
else:
    raise ValueError("Invalid model key!")
# feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")


device


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []
train_auc_scores = []
val_auc_scores = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    true_labels = []
    predicted_labels = []
    predicted_probs = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        true_labels.extend(labels.cpu().numpy())
        _, predicted = torch.max(logits, 1)
        predicted_labels.extend(predicted.cpu().numpy())

        predicted_probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=1)
    auc = roc_auc_score(true_labels, predicted_probs)

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    print(f"Train Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(accuracy)
    train_f1_scores.append(f1)
    train_auc_scores.append(auc)

    model.eval()
    val_running_loss = 0.0
    val_true_labels = []
    val_predicted_labels = []
    val_predicted_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            loss = criterion(logits, labels)
            val_running_loss += loss.item()

            val_true_labels.extend(labels.cpu().numpy())
            _, predicted = torch.max(logits, 1)
            val_predicted_labels.extend(predicted.cpu().numpy())

            val_predicted_probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

    val_accuracy = accuracy_score(val_true_labels, val_predicted_labels)
    val_precision = precision_score(val_true_labels, val_predicted_labels, average='binary', zero_division=1)
    val_recall = recall_score(val_true_labels, val_predicted_labels, average='binary', zero_division=1)
    val_f1 = f1_score(val_true_labels, val_predicted_labels, average='binary', zero_division=1)
    val_auc = roc_auc_score(val_true_labels, val_predicted_probs)

    print(f"Validation Loss: {val_running_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

    val_losses.append(val_running_loss / len(val_loader))
    val_accuracies.append(val_accuracy)
    val_f1_scores.append(val_f1)
    val_auc_scores.append(val_auc)

print("Training complete.")



epochs = range(1, num_epochs + 1)


model.eval()
predictions = []

with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        _, predicted = torch.max(logits, 1)

        for filename, label in zip(filenames, predicted.cpu().numpy()):
            predictions.append({'id': filename, 'label': int(label)})


output_df = pd.DataFrame(predictions)
output_df['label'] = output_df['label'].replace({0: 1, 1: 0})
output_df['id'] = 'test_data_v2/' + output_df['id'].astype(str)
# output_df.rename(columns={'i': 'id'}, inplace=True)
output_df.to_csv('submission.csv', index=False)


cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 7))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


cm


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss per Epoch')

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy per Epoch')

plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(epochs, train_f1_scores, label='Train F1')
plt.plot(epochs, val_f1_scores, label='Validation F1')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 Score per Epoch')
plt.show()




