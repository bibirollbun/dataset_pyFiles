# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import torchvision.transforms as T
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, ViTFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split


# Paths
data_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/"
train_images_dir = os.path.join(data_dir, "train")
test_images_dir = os.path.join(data_dir, "test")
train_csv_path = os.path.join(data_dir, "train_labels.csv")


# Load CSV labels
train_labels_df = pd.read_csv(train_csv_path)
label_mapping = {label: idx for idx, label in enumerate(train_labels_df['label'].unique())}
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

# Custom Dataset
class DateDataset(Dataset):
    def __init__(self, images_dir, labels_df=None, transform=None):
        self.images_dir = images_dir
        self.labels_df = labels_df
        self.transform = transform
        self.image_filenames = os.listdir(images_dir)
        if labels_df is not None:
            self.labels = {row["filename"]: label_mapping[row["label"]] for _, row in labels_df.iterrows()}
        else:
            self.labels = None

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[image_name] if self.labels else -1
        return image, image_name, label






# Load Model
def load_model():
    model_name = "google/vit-base-patch16-224-in21k"
    
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=len(label_mapping), ignore_mismatched_sizes=True)
    processor = ViTFeatureExtractor.from_pretrained(model_name)
    return model, processor


# Transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])



# Datasets and Loaders
train_dataset = DateDataset(train_images_dir, train_labels_df, transform)
test_dataset = DateDataset(test_images_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




# Train Model
def train_model(model, train_loader, epochs=10, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, _, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} completed, Average Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), "vit_model.pth")
    print("Model training complete and saved.")


# Training and Testing
torch.manual_seed(42)
model, processor = load_model()
train_model(model, train_loader)



def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, filenames,_ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            print(predicted)
            for filename, pred in zip(filenames, predicted.cpu().numpy()):
                results.append([filename, reverse_label_mapping[pred]])
    print(filename)
    results_df = pd.DataFrame(results, columns=["filename", "label"])
    results_df.to_csv("test_results.csv", index=False)
    print("Test results saved to test_results.csv")


model.load_state_dict(torch.load("vit_model.pth"))
test_model(model, test_loader)


import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load the CSV file that contains filenames and labels
csv_file = '/kaggle/working/test_results.csv'  # Replace with your CSV path
df = pd.read_csv(csv_file)

# Define the test folder where images are located
test_folder = '/kaggle/input/open-data-day-2025-dates-types-classification/test'  # Replace with your image folder path

# Function to display images with labels
def display_images_with_labels(df, test_folder):
    # Set up the plot grid
    fig, axes = plt.subplots(nrows=12, ncols=12, figsize=(15, 15))  # Adjust grid size based on number of images
    axes = axes.flatten()
    
    for i, (filename, label) in enumerate(zip(df['filename'], df['label'])):
        # Construct full image path
        img_path = os.path.join(test_folder, filename)
        
        # Open the image
        img = Image.open(img_path)
        
        # Display the image
        axes[i].imshow(img)
        axes[i].axis('off')  # Hide axes

        # Display the label on top of the image
        axes[i].set_title(label, fontsize=12, color='white', backgroundcolor='black', loc='center')

    plt.tight_layout()
    plt.show()

# Call the function to display the images
display_images_with_labels(df, test_folder)








