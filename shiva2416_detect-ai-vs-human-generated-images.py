import pandas as pd

# Load CSV files
train_csv_path = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
test_csv_path = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print("Train DataFrame:")
print(train_df.head())

print("Test DataFrame:")
print(test_df.head())



# Install required libraries
# !pip install torch torchvision transformers pandas pillow tqdm

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler  # Import from torch.amp
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  # For confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Check GPU (should show L4)
!nvidia-smi
!nproc
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. Load and Split train.csv

# Split: 70% train, 20% val, 10% test
train_df, temp_df = train_test_split(train_df, test_size=0.3, stratify=train_df['label'], random_state=42)
val_df, test1_df = train_test_split(temp_df, test_size=0.3333, stratify=temp_df['label'], random_state=42)

# Verify sizes and balance
print(f"Train: {len(train_df)} images")
print(f"Validation: {len(val_df)} images")
print(f"Test: {len(test1_df)} images")
print("Train label distribution:\n", train_df['label'].value_counts())


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.df = dataframe
        self.image_dir = image_dir
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['file_name'])  # Use 'file_name'
        label = self.df.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)


class TestImageDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.df = dataframe
        self.image_dir = image_dir
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id']
        img_path = os.path.join(self.image_dir, img_id)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found")
            image = Image.new('RGB', (224, 224))
        else:
            image = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)


image_dir = '/kaggle/input/ai-vs-human-generated-dataset/'
train_dataset = ImageDataset(train_df, image_dir)
val_dataset = ImageDataset(val_df, image_dir)
test_dataset = ImageDataset(test1_df, image_dir)
sumbit_dataset = TestImageDataset(test_df,image_dir)


# 3. DataLoaders with optimizations
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, prefetch_factor=2)
submit_loader = DataLoader(sumbit_dataset,batch_size=32,shuffle=False,num_workers=4,prefetch_factor=2)


print("Test CSV head:")
print(test_df.head())
print(f"Image dir contents: {os.listdir(image_dir)[:5]}")
print(f"Test CSV length: {len(test_df)}")


print(f"Test Dataset Size: {len(test_dataset)}")
print(f"Number of Batches: {len(test_loader)}")


# 4. Model Setup
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=2,  # Real vs AI
    ignore_mismatched_sizes=True
).to(device)


# Optimizer and scaler for mixed precision
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()


# 5. Training Loop
num_epochs = 4  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for pixel_values, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):  # Mixed precision
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
    
    val_accuracy = correct / len(val_dataset)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy}")



# 4. Test Evaluation with Confusion Matrix
test_correct = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for i, (pixel_values, labels) in enumerate(tqdm(test_loader, desc="Evaluating Test Set")):
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):  # cuda.amp, works in your setup
            outputs = model(pixel_values=pixel_values)
            preds = torch.argmax(outputs.logits, dim=1)
            test_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
       
test_accuracy = test_correct / len(test_dataset)
print(f"Test Accuracy: {test_accuracy}")
print(f"Total Predictions: {len(all_preds)}, Total Labels: {len(all_labels)}")

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['AI-generated', 'Human-generated'], 
            yticklabels=['AI-generated', 'Human-generated'])
plt.title('Confusion Matrix for Test Data')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# # Generate predictions
# predictions = []
# with torch.no_grad():
#     for batch in tqdm(sumbit_loader, desc="Generating predictions"):
#         pixel_values = batch.to(device)
#         outputs = model(pixel_values=pixel_values)
#         preds = torch.argmax(outputs.logits, dim=1)
#         predictions.extend(preds.cpu().numpy())

# # Create submission DataFrame
# submission_df = pd.DataFrame({
#     'id': test_df['id'],
#     'label': predictions  # Using 'label' as column name, adjust if needed
# })

# # Save to CSV
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created as 'submission.csv'")
# print(submission_df.head())


# # Generate probabilities
# all_probs = []
# with torch.no_grad():
#     for batch in tqdm(submit_loader, desc="Generating predictions"):
#         pixel_values = batch.to(device)
#         with autocast(device_type='cuda', dtype=torch.float16):
#             outputs = model(pixel_values=pixel_values)
#             scaled_logits = outputs.logits * 20  # Higher scaling
#             probs = torch.softmax(scaled_logits, dim=-1)[:, 1]
#         all_probs.extend(probs.cpu().numpy())
#         if len(all_probs) <= 5:
#             print(f"First 5 probs (AI): {probs[:5]}")

# # OHEM-inspired adjustment
# all_probs_adjusted = [prob * 1.5 if prob < 0.1 or prob > 0.9 else prob for prob in all_probs]

# # Exact 66% AI threshold
# target_ai_count = 3656
# sorted_probs = np.sort(all_probs_adjusted)
# threshold_66_exact = sorted_probs[len(test_df) - target_ai_count]
# balanced_preds = (all_probs_adjusted > threshold_66_exact).astype(int)

# # Create submission file
# submission_df = pd.DataFrame({
#     'id': test_df['id'],
#     'label': balanced_preds
# })
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file created as 'submission.csv'")
# print("Submission head:")
# print(submission_df.head())
# print(f"Prediction counts: 0 (Human): {(balanced_preds == 0).sum()}, 1 (AI): {(balanced_preds == 1).sum()}")



# Generate probabilities with higher scaling
all_probs = []
with torch.no_grad():
    for batch in tqdm(submit_loader, desc="Generating predictions"):
        pixel_values = batch.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(pixel_values=pixel_values)
            scaled_logits = outputs.logits * 10  # Tweak 1: Scale more
            probs = torch.softmax(scaled_logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        if len(all_probs) <= 5:
            print(f"First 5 probs (AI): {probs[:5]}")

# Save probabilities
probabilities_df = pd.DataFrame({
    'id': test_df['id'],
    'prob_class_1': all_probs
})
probabilities_df.to_csv('probabilities.csv', index=False)
print("Probabilities saved to 'probabilities.csv'")
print(probabilities_df.head())

# Tweak 2: Exact 66% AI threshold
target_ai_count = int(0.66 * len(test_df))  # 3656 AI
sorted_probs = np.sort(all_probs)
threshold_66_exact = sorted_probs[len(test_df) - target_ai_count]
print(f"Exact 66% AI threshold: {threshold_66_exact:.4f}")
balanced_preds = (all_probs > threshold_66_exact).astype(int)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': balanced_preds
})
submission_df.to_csv('submission.csv', index=False)
print("Submission file created as 'submission.csv'")
print("Submission head:")
print(submission_df.head())
print(f"Prediction counts: 0 (Human): {(balanced_preds == 0).sum()}, 1 (AI): {(balanced_preds == 1).sum()}")






# # Generate predictions with logits
# all_logits = []
# with torch.no_grad():
#     for batch in tqdm(submit_loader, desc="Generating predictions"):
#         pixel_values = batch
#         pixel_values = pixel_values.to(device)
#         with autocast():
#             outputs = model(pixel_values=pixel_values)
#             logits = outputs.logits
#         all_logits.extend(logits.cpu().numpy())

# # Balance predictions (~52% AI, 48% Human)
# all_logits = np.array(all_logits)  # Shape: [5540, 2]
# logit_diff = all_logits[:, 1] - all_logits[:, 0]  # Class 1 - Class 0
# threshold = np.percentile(logit_diff, 48)  # ~52% AI (1)
# balanced_preds = (logit_diff > threshold).astype(int)  # 0=Human, 1=AI

# # Create submission
# submission_balanced = pd.DataFrame({
#     'id': test_df['id'],
#     'label': balanced_preds
# })
# submission_balanced.to_csv('submission_balanced.csv', index=False)

# # Print results
# print("Balanced submission head:")
# print(submission_balanced.head())
# print(f"Balanced counts: 0 (Human): {(balanced_preds == 0).sum()}, 1 (AI): {(balanced_preds == 1).sum()}")
# print(f"Total predictions: {len(balanced_preds)}")


# 7. Save Model
torch.save(model.state_dict(), '/content/vit_classifier.pth')
print("Model saved to /content/vit_classifier.pth")




