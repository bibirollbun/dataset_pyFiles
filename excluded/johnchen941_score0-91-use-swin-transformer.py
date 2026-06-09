import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            labels (list, optional): List of labels corresponding to the images.
            transform (callable, optional): Transform to be applied to the images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image, os.path.basename(img_path)  # For test set


from transformers import SwinForImageClassification
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_vit_model(train_loader, val_loader, device, epochs=10, lr=5e-5):
    """Trains the Swin Transformer model."""

    model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",  # Swin transformer model
    num_labels=1,
    problem_type="binary_classification",
    ignore_mismatched_sizes=True
    
    )
    model.to(device)


    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()


    optimizer = optim.AdamW(model.parameters(), 
                            lr=lr,  
                            weight_decay=0.01)
    

    
    # å®šä¹‰å­¦ä¹ ç�‡è°ƒåº¦å™¨
    lr_scheduler = ReduceLROnPlateau(
        optimizer,             
        mode='min',            
        factor=0.1,            
        patience=3,            
        verbose=True,         
        threshold=1e-4,        
        threshold_mode='abs',  
        min_lr=1e-6            
    )

    best_auc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits.squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss / len(val_loader):.4f} | Val AUC: {val_auc:.4f}")

        lr_scheduler.step(val_loss)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_vit_model.pth")

    return model



def predict_with_vit(model, test_loader, device, submission_path):
    """Generates predictions using the trained ViT model."""
    model.eval()
    predictions = []
    image_names = []
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            outputs = model(images).logits.squeeze()
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            image_names.extend(names)

    # Save submission file
    submission_df = pd.DataFrame({
        "image": image_names,
        "label": predictions
    })
    submission_df.to_csv(submission_path, index=False)



if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train"
    TEST_DIR = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test"
    LABELS_PATH = "/kaggle/input/fvsvdvs/train.csv"  
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read labels
    df_labels = pd.read_csv(LABELS_PATH)
    image_paths = [os.path.join(TRAIN_DIR, img) for img in df_labels["image"]]
    labels = df_labels["label"].astype(float).values

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create datasets and loaders
    train_dataset = CustomDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train the model
    vit_model = train_vit_model(train_loader, val_loader, DEVICE, epochs=EPOCHS)

    # Prepare test dataset and loader
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")])
    test_paths = [os.path.join(TEST_DIR, f) for f in test_files]
    test_dataset = CustomDataset(test_paths, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Predict and save submission
    predict_with_vit(vit_model, test_loader, DEVICE, "submission.csv")


