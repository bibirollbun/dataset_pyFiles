import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

# Set the seed for all libraries
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setup
np.random.seed(seed)

# Set deterministic behavior for cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



#Global Variables
batch_size = 64
imbalance_approach = True


import pandas as pd
from sklearn.model_selection import train_test_split

# # Map non-class 0 to a single binary class
# labels_df['binary_label'] = labels_df['diagnosis'].apply(lambda x: 0 if x == '0' else 1)

# # Split into Class 0 and Other Classes
# class0_data = labels_df[labels_df['binary_label'] == 0]
# other_data = labels_df[labels_df['binary_label'] == 1]

# print(len(class0_data),len(other_data))


import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import f1_score, cohen_kappa_score


import torch
import torch.nn as nn


class ResNetModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)  # Load ResNet18
        
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNetModel(pretrained=True).to(device)


import torch.optim as optim

# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Handle class imbalance
# class_weights = len(train_data) / (2.0 * train_data['binary_label'].value_counts().to_numpy())
# pos_weight = torch.tensor(np.float64(class_weights[1] / class_weights[0])).to(device)

# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score,classification_report



from PIL import Image, ImageChops

def crop_to_object(img):


    # Convert to grayscale
    gray_im = img.convert("L")

    # Set your threshold value
    threshold = 10  # Adjust this value as needed

    # Create a binary image: pixels > threshold become 255 (white), else 0 (black)
    binary_im = gray_im.point(lambda x: 255 if x > threshold else 0)

    # Get the bounding box of the white regions (non-background)
    bbox = binary_im.getbbox()

    if bbox:
        # Crop the image to the bounding box and save it
        cropped_img = img.crop(bbox)
        #cropped_img.save(output_path)
        #print(f"Cropped image saved as '{output_path}'")
        return cropped_img
        
# Demo usage:
# if __name__ == "__main__":
#     input_image_path = "retina_image.png"   # Path to your retina image
#     output_image_path = "cropped_retina.png"  # Desired path for the cropped image

#     # Crop the image and show the result if available
#     cropped = crop_to_object(input_image_path, output_image_path)
#     if cropped:
#         cropped.show()



# ============================
# Custom Dataset for Messidor1 (Handles .tif images)
# ============================
class MessidorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate over folders (0, 1, 2, 3) and collect image paths
        for label in range(4):  # Labels: 0, 1, 2, 3
            folder_path = os.path.join(root_dir, str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith(".tif"):
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if self.labels[idx] == 0:
            label = 0
        else:
            label = 1

        # Open the image and apply transformations
        image = Image.open(img_path).convert("RGB")
        image = crop_to_object(image)
        if self.transform:
            image = self.transform(image)

        return image, label


crop_to_object(Image.open("/kaggle/input/messifor2/messidor2/IMAGES/20051020_43808_0100_PP.png"))


# ============================
# Data Transformations & Dataloaders
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing for ResNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dir = "/kaggle/input/messidor1-data/P_Data/Test"

test_dataset = MessidorDataset(test_dir, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Testing samples: {len(test_dataset)}")


# ============================
# Evaluation Function (QWK & Accuracy)
# ============================
def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute Accuracy & Quadratic Weighted Kappa (QWK)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"✅ Test Accuracy: {accuracy:.4f}")
    
    print(classification_report(y_true,y_pred))
    return y_true ,y_pred



model.load_state_dict(torch.load("/kaggle/input/eyepacs-resnet/EyePacs_best_resnet_model_qwk.pth"))
model.eval()


# ============================
# Run Evaluation
# ============================
y_true ,y_pred = evaluate(model, test_loader)


# Define dataset paths
image_dir = "/kaggle/input/idrid-dataset/Imagenes/Imagenes"
csv_file = "/kaggle/input/idrid-dataset/idrid_labels.csv"


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define dataset class
class IDRiDDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)

        # Ensure column names are correctly read
        self.df.columns = self.df.columns.str.strip()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract image ID and label
        img_id = self.df.iloc[idx]["id_code"]

        if self.df.iloc[idx]["diagnosis"] == 0:
            label = 0
        else:
            label = 1
        #label = self.df.iloc[idx]["diagnosis"]

        # Load image
        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# Create dataset and dataloader
batch_size = 64
test_dataset = IDRiDDataset(image_dir=image_dir, csv_file=csv_file, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


import torch.nn.functional as F

# Evaluation function
def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print(classification_report(all_labels, all_preds))
    return all_labels, all_preds

# Run evaluation
all_labels, all_preds = evaluate_model(model, test_loader)

