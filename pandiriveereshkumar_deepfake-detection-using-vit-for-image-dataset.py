import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the device to GPU (For Kaggle, it will automatically use the available GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ViT Image Processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')


# Custom Dataset
# Define augmentations for training
train_augmentations = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, processor, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipping corrupted file: {img_path}")
            return self.__getitem__((idx + 1) % len(self.file_paths))
        
        # Apply augmentations only if specified
        if self.augment:
            image = train_augmentations(image)
        
        # Process using processor
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze()
        
        return {
            'pixel_values': pixel_values,
            'labels': self.labels[idx]
        }

# Get image paths
real_dir = "/kaggle/input/image-df-mini/Image-Df/real"
fake_dir = "/kaggle/input/image-df-mini/Image-Df/fake"

real_paths = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir)]
fake_paths = [os.path.join(fake_dir, fname) for fname in os.listdir(fake_dir)]

file_paths = real_paths + fake_paths
labels = [0]*len(real_paths) + [1]*len(fake_paths)  # 0: real, 1: fake

# Split data
train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.1, stratify=labels)

# Transforms (Use processor for pre-processing)
def transform_function(image):
    return processor(images=image, return_tensors="pt")["pixel_values"].squeeze()

# Datasets
train_dataset = DeepfakeDataset(train_paths, train_labels, processor, augment=True)
val_dataset = DeepfakeDataset(val_paths, val_labels, processor, augment=False)


# Load ViT model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=2
).to(device)

# Optimizer and Loss
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()



# Define the Trainer
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=160,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # Changed from eval_loss to accuracy
    greater_is_better=True,
    report_to="none",
    fp16=False  # Set to False if having issues with mixed precision
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    compute_metrics=lambda p: compute_metrics(p),
)



def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    eer_index = np.nanargmin(abs_diffs)
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    probs = p.predictions[:, 1]  # Probability of class 1 (fake)
    labels = p.label_ids

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    eer = compute_eer(labels, probs)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'eer': eer
    }



# Train the model
trainer.train()

# Save the model
trainer.save_model("final_model_v4")

# Evaluation
trainer.evaluate()


# To visualize confusion matrix post-evaluation
def plot_confusion_matrix(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

all_preds = []
all_labels = []

for idx in range(len(val_dataset)):
    batch = val_dataset[idx]
    img = batch['pixel_values'].unsqueeze(0).to(device)  # Add batch dimension
    label = torch.tensor(batch['labels']).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=img)
        pred = torch.argmax(outputs.logits, dim=1)

    all_preds.append(pred.item())
    all_labels.append(label.item())

    if idx %10000 == 0:
        print(f"Image:{idx}")

plot_confusion_matrix(all_preds, all_labels)


# === Load test dataset ===
test_dir = "/kaggle/input/image-df-mini/Image-Df/test"
test_label_path = os.path.join(test_dir, "test_labels.txt")

test_paths = []
test_labels = []

with open(test_label_path, 'r') as f:
    for line in f:
        label, fname = line.strip().split()
        test_paths.append(os.path.join(test_dir, fname))
        test_labels.append(0 if int(label) == 1 else 1) 

test_dataset = DeepfakeDataset(test_paths, test_labels, processor)

# === Predict on test data ===
all_preds_test = []
all_labels_test = []

for idx in range(len(test_dataset)):
    batch = test_dataset[idx]
    img = batch['pixel_values'].unsqueeze(0).to(device)
    label = torch.tensor(batch['labels']).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=img)
        pred = torch.argmax(outputs.logits, dim=1)

    all_preds_test.append(pred.item())
    all_labels_test.append(label.item())

    if idx % 1000 == 0:
        print(f"Test Image: {idx}")

# === Evaluate on test set ===
print("\nTest Set Evaluation:")
print(classification_report(all_labels_test, all_preds_test, target_names=["Real", "Fake"]))
plot_confusion_matrix(all_preds_test, all_labels_test)



metrics = trainer.state.log_history

train_loss = []
eval_loss = []
eval_accuracy = []

for entry in metrics:
    if 'loss' in entry and 'epoch' in entry and 'eval_loss' not in entry:
        train_loss.append(entry['loss'])
    if 'eval_loss' in entry:
        eval_loss.append(entry['eval_loss'])
    if 'eval_accuracy' in entry:
        eval_accuracy.append(entry['eval_accuracy'])

# Find correct epoch length (match with eval entries)
epochs = list(range(1, len(eval_loss) + 1))

# Safe plotting
plt.figure(figsize=(14, 5))

# Loss plot
plt.subplot(1, 2, 1)
if len(train_loss) >= len(epochs):  # Sometimes train_loss has more entries
    trimmed_train_loss = train_loss[:len(epochs)]
else:
    trimmed_train_loss = train_loss

if len(eval_loss) >= len(epochs):  # Sometimes train_loss has more entries
    trimmed_eval_loss = eval_loss[:len(epochs)]
else:
    trimmed_eval_loss = eval_loss

if len(eval_accuracy) >= len(epochs):  # Sometimes train_loss has more entries
    trimmed_eval_accuracy = eval_accuracy[:len(epochs)]
else:
    trimmed_eval_accuracy = eval_accuracy
    
plt.plot(epochs[:len(trimmed_train_loss)], trimmed_train_loss, label='Train Loss')
plt.plot(epochs[:len(trimmed_eval_loss)], trimmed_eval_loss, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, trimmed_eval_accuracy, marker='o', label='Eval Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()



!ls /kaggle/working/


import os
os.chdir(r'/kaggle/working')

!tar -czf output.tar.gz final_model_v4

from IPython.display import FileLink

FileLink(r'output.tar.gz')


!tar -czf output2.tar.gz final_model_v3

from IPython.display import FileLink

FileLink(r'output2.tar.gz')




