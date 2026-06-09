!pip install -U albumentations

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import timm
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import f1_score
from tqdm import tqdm
import copy


!pip install -U albumentations



class CustomDataset(Dataset):
    def __init__(self, df, data_dir, transforms=None, is_train=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing file names and labels.
            data_dir (str): Directory where the image files are stored.
            transforms (albu.Compose, optional): Albumentations transforms to apply to the images.
            is_train (bool, optional): Flag to indicate if the dataset is for training.
        """
        self.df = df
        self.data_dir = data_dir
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract the file name and label
        filename = self.df.iloc[idx]['file_name']
        label = self.df.iloc[idx]['label']

        # Load the image
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB mode
        image = np.array(image)  # Convert to NumPy array (Albumentations requires this format)

        # Apply transforms if specified
        if self.transforms:
            image = self.transforms(image=image)['image']

        # For training, return the image and label
        if self.is_train:
            return image, label
        # For inference, return only the image
        else:
            return image



BATCH_SIZE = 16
SEED = 42
IMG_SIZE = 384

# Load the train CSV
train_csv_path = "/kaggle/input/ai-vs-human-generated-dataset/train.csv"
train_df = pd.read_csv(train_csv_path)

# Define the base directory where images are stored
train_data_dir = "/kaggle/input/ai-vs-human-generated-dataset/"

train_transform = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.Resize(IMG_SIZE, IMG_SIZE),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = albu.Compose([
    albu.Resize(IMG_SIZE, IMG_SIZE),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=SEED, stratify=train_df['label'])

# Dataset and DataLoader
train_dataset = CustomDataset(train_df, train_data_dir, transforms=train_transform, is_train=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dataset = CustomDataset(val_df, train_data_dir, transforms=train_transform, is_train=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Iterate through the DataLoader
for images, labels in train_dataloader:
    print(images.shape, labels)
    break

for images, labels in val_dataloader:
    print(images.shape, labels)
    break


LR = 1e-5
NUM_EPOCHS = 2

# model = timm.create_model('vit_small_patch14_dinov2.lvd142m', 
#                           pretrained=True,
#                          num_classes=2)

model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', 
                          pretrained=True,
                         num_classes=2)

optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, num_epochs=10, save_path="best_model.pth"
):
    """
    Trains the model and evaluates it on the validation set. Saves the model with the best F1 score.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer, e.g., AdamW.
        scheduler: Learning rate scheduler, e.g., StepLR.
        loss_fn: Loss function, e.g., CrossEntropyLoss.
        device: Device to train on (e.g., 'cuda' or 'cpu').
        num_epochs: Number of epochs to train.
        save_path: Path to save the model with the best F1 score.

    Returns:
        best_model_wts: The weights of the model with the highest F1 score.
        history: A dictionary containing training and validation metrics.
    """
    model.to(device)  # Move model to the specified device

    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())  # To store the best weights
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    curr_step = 0
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        model.train()

        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Compute training metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            curr_step += 1
            if curr_step % 1000 == 0:
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_labels = []
                all_preds = []
        
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
        
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)
        
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
        
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
        
                        # Store labels and predictions for F1 score calculation
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())
        
                val_loss /= len(val_loader)
                val_f1 = f1_score(all_labels, all_preds, average="weighted")
        
                # Save the best model weights based on validation F1 score
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved with F1 score: {best_f1:.4f}")


                train_loss /= 1000
                
                # Log metrics
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)

                # Print results after 1k steps
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

                # reset
                train_loss = 0.0
                train_total = 0
                train_correct = 0
                
                # Step the learning rate scheduler
                scheduler.step()

    print("Training complete.")
    print(f"Best F1 Score: {best_f1:.4f}")
    return best_model_wts, history


best_model_wts, history = train_model(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    device=device,
    num_epochs=NUM_EPOCHS,
    save_path="best_model.pth",
)

# Load the best model weights after training
model.load_state_dict(best_model_wts)


import matplotlib.pyplot as plt

def visualize_history(history):
    """
    Visualizes the training and validation history.

    Args:
        history (dict): Dictionary containing training and validation metrics with keys:
                        - "train_loss": List of training loss values.
                        - "val_loss": List of validation loss values.
                        - "val_f1": List of validation F1 scores.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Plot validation F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_f1"], label="Val F1 Score", marker="o", color="green")
    plt.title("Validation F1 Score Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

visualize_history(history)



# Load the test CSV for predictions
test_csv_path = "/kaggle/input/ai-vs-human-generated-dataset/test.csv"
test_df = pd.read_csv(test_csv_path)

# Define the base directory where images are stored
test_data_dir = "/kaggle/input/ai-vs-human-generated-dataset"

# Update image paths in the test dataframe
test_df['img_path'] = test_df['id'].apply(lambda x: os.path.join(test_data_dir, x))


%%time

# Initialize a list to store results
predictions = []

# Perform inference on test images
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    # if idx==100:
    #     break
    
    img_path = row['img_path']  # Path to the image
    id_ = row['id']
    try:
        # Open and preprocess image
        image = np.array(Image.open(img_path).convert("RGB"))
        input_tensor = val_transform(image=image)['image'].unsqueeze(0).to(device)  # Add batch dimension

        # Model prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_label = torch.argmax(outputs, dim=-1).item()  # Get predicted class

        predictions.append((id_, predicted_label))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")



# Create a DataFrame for submission
submission_df = pd.DataFrame(predictions, columns=["id", "label"])

# Save to CSV for submission
submission_csv_path = "submission.csv"
submission_df.to_csv(submission_csv_path, index=False)
print(f"Submission file saved at {submission_csv_path}")


value_counts = submission_df['label'].value_counts()
value_counts


submission_df.head()

