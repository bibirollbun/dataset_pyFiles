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



import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import f1_score, cohen_kappa_score,confusion_matrix


import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)  # Load ResNet18
        
        # Remove the original FC layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last layer
        
        # Add custom classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),  # First Dense layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout to prevent overfitting
            nn.Linear(256, 128),  # Second Dense layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)   # Output layer for binary classification
        )

    def forward(self, x):
        x = self.resnet(x)  # Extract features from ResNet
        x = self.classifier(x)  # Pass through classifier
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNetModel(pretrained=False).to(device)


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



import os
from torch.utils.data import Dataset
from PIL import Image

class MessidorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate over folders (0, 1, 2, 3) and collect valid image paths
        for label in [0, 3]:  # Include only 0, 3
            folder_path = os.path.join(root_dir, str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith(".tif"):
                    self.image_paths.append(os.path.join(folder_path, filename))

                    # Assign binary labels: 0 (negative), 1 (positive)
                    binary_label = 0 if label == 0 else 1
                    self.labels.append(binary_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

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
            preds = (torch.sigmoid(outputs) > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute Accuracy & Quadratic Weighted Kappa (QWK)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"✅ Test Accuracy: {accuracy:.4f}")
    
    print(classification_report(y_true,y_pred))
    print(confusion_matrix(y_true,y_pred))
    return y_true ,y_pred



model.load_state_dict(torch.load("/kaggle/input/eyepacs-ddr-resnet-model/EyePacs_DDR_best_resnet_model.pth"))
model.eval()


# ============================
# Run Evaluation
# ============================
y_true ,y_pred = evaluate(model, test_loader)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



import numpy as np

# Convert predictions to numpy for histogram
y_pred = np.array(y_pred)

plt.figure(figsize=(6, 5))
plt.hist(y_pred, bins=[-0.5, 0.5, 1.5], edgecolor='black', alpha=0.7, color='blue')
plt.xticks([0, 1], labels=["No DR", "DR"])
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()



from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()



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
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    

    print(f"Accuracy: {accuracy:.4f}")
    
    print(classification_report(all_labels, all_preds))

    print(confusion_matrix(all_labels, all_preds))
    return all_labels, all_preds

# Run evaluation
all_labels, all_preds = evaluate_model(model, test_loader)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


import numpy as np

# Convert predictions to numpy for histogram
all_preds = np.array(all_preds)

plt.figure(figsize=(6, 5))
plt.hist(all_preds, bins=[-0.5, 0.5, 1.5], edgecolor='black', alpha=0.7, color='blue')
plt.xticks([0, 1], labels=["No DR", "DR"])
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(all_labels, all_preds)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()



from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()



# Define dataset paths
image_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define dataset class
class APTOSDataset(Dataset):
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
        img_path = os.path.join(self.image_dir, img_id + ".png")
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# Create dataset and dataloader
batch_size = 64
test_dataset = APTOSDataset(image_dir=image_dir, csv_file=csv_file, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# Run evaluation
all_labels, all_preds = evaluate_model(model, test_loader)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



import numpy as np

# Convert predictions to numpy for histogram
all_preds = np.array(all_preds)

plt.figure(figsize=(6, 5))
plt.hist(all_preds, bins=[-0.5, 0.5, 1.5], edgecolor='black', alpha=0.7, color='blue')
plt.xticks([0, 1], labels=["No DR", "DR"])
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(all_labels, all_preds)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()



from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()



import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Correct paths
csv_path = "/kaggle/input/messidor2-dr-grades/messidor_data.csv"
img_folder = "/kaggle/input/messifor2/messidor2/IMAGES"

# Load CSV file
Messidor2_data = pd.read_csv(csv_path)

# Ensure correct data types and drop missing labels
Messidor2_data = Messidor2_data.dropna(subset=["adjudicated_dr_grade"])  # Drop NaNs
Messidor2_data["adjudicated_dr_grade"] = Messidor2_data["adjudicated_dr_grade"].astype(int)  # Ensure integer labels

# Convert to binary classification (0: No DR, 1-4: Has DR)
Messidor2_data["binary_label"] = Messidor2_data["adjudicated_dr_grade"].apply(lambda x: 0 if x == 0 else 1)

class MessidorDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
        self.transform = transform

        # Ensure image_id column has ".png" extension
        self.data["image_id"] = self.data["image_id"].astype(str).apply(lambda x: x if x.endswith(".png") else x + ".png")

        # Get valid image files in the folder
        self.valid_images = set(os.listdir(root_dir))

        # Filter dataset to include only existing images
        self.data = self.data[self.data["image_id"].isin(self.valid_images)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image_id"]
        img_path = os.path.join(self.root_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Extract label as binary classification
        label = int(self.data.iloc[idx]["binary_label"])

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)  # BCEWithLogitsLoss expects float labels

# Create dataset and dataloader
test_dataset = MessidorDataset(Messidor2_data, img_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total images found: {len(test_dataset)}")


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/kaggle/input/eyepacs-resnet/EyePacs_best_resnet_model.pth", map_location=device))
model.to(device)
model.eval()


# Evaluate model
def evaluate_binary_model(model, dataloader):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)  # Convert to probability
            preds = (probs > 0.5).float()  # Convert to 0 or 1

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds

# Run evaluation
all_labels, all_preds = evaluate_binary_model(model, test_loader)

# Compute binary confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=["No DR", "Has DR"]))


import seaborn as sns

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative (0)", "Positive (1)"], yticklabels=["Negative (0)", "Positive (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Accuracy Bar Plot
def plot_accuracy(accuracy):
    plt.figure(figsize=(5, 5))
    plt.bar(["Accuracy"], [accuracy], color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Accuracy")
    plt.show()

# Compute Accuracy for Plot
accuracy = accuracy_score(all_labels, all_preds)

# Show Plots
plot_confusion_matrix(all_labels, all_preds)
plot_accuracy(accuracy)


import numpy as np

# Convert predictions to numpy for histogram
all_preds = np.array(all_preds)

plt.figure(figsize=(6, 5))
plt.hist(all_preds, bins=[-0.5, 0.5, 1.5], edgecolor='black', alpha=0.7, color='blue')
plt.xticks([0, 1], labels=["No DR", "DR"])
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(all_labels, all_preds)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()



from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()



import pandas as pd

# Set path to DDR dataset
DDR_DIR = "/kaggle/input/idriddiseasegrading/DDR-dataset/DR_grading"
TEST_CSV = f"{DDR_DIR}/test.csv"

# Load test.csv
df_test = pd.read_csv(TEST_CSV)

# Convert multi-class labels to binary labels
df_test['binary_label'] = df_test['Retinopathy grade'].apply(lambda x: 0 if x == 0 else 1)

# Display dataset preview
print(df_test.head())


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DDRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image file name and corresponding label
        img_id = str(self.df.iloc[idx]['Image name'])
        img_path = os.path.join(self.img_dir, img_id)
        label = int(self.df.iloc[idx]['binary_label'])

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformation (resize to match model input)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust based on your model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Set path to DDR test images
DDR_TEST_IMG_DIR = f"{DDR_DIR}/test"

# Create dataset
test_dataset = DDRDataset(df_test, DDR_TEST_IMG_DIR, transform=test_transforms)

# Create DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print("DDR Test DataLoader Ready!")


import torch
import torch.nn.functional as F

# Evaluate model
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Get predictions
        outputs = model(images)
        probs = torch.sigmoid(outputs)  # Convert to probability
        preds = (probs > 0.5).float()   # Convert to 0 or 1

        # Store results
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # Calculate accuracy
        correct += (preds == labels).sum().item()
        total += labels.size(0)

# Compute final accuracy
accuracy = correct / total * 100
print(f"Model Accuracy on DDR Test Data: {accuracy:.2f}%")


# Classification Report
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=["No DR", "Has DR"]))


import seaborn as sns

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative (0)", "Positive (1)"], yticklabels=["Negative (0)", "Positive (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Accuracy Bar Plot
def plot_accuracy(accuracy):
    plt.figure(figsize=(5, 5))
    plt.bar(["Accuracy"], [accuracy], color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Accuracy")
    plt.show()

# Compute Accuracy for Plot
accuracy = accuracy_score(all_labels, all_preds)

# Show Plots
plot_confusion_matrix(all_labels, all_preds)
plot_accuracy(accuracy)


import numpy as np

# Convert predictions to numpy for histogram
all_preds = np.array(all_preds)

plt.figure(figsize=(6, 5))
plt.hist(all_preds, bins=[-0.5, 0.5, 1.5], edgecolor='black', alpha=0.7, color='blue')
plt.xticks([0, 1], labels=["No DR", "DR"])
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(all_labels, all_preds)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()



from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

