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


class ResNetModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):  # Add `num_classes` parameter
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
            nn.Linear(128, num_classes)  # Output layer for multi-class classification
        )

    def forward(self, x):
        x = self.resnet(x)  # Extract features from ResNet
        x = self.classifier(x)  # Pass through classifier
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNetModel(pretrained=False).to(device)


import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score,classification_report


import os
from torch.utils.data import Dataset
from PIL import Image

class MessidorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate over all folders (0, 1, 2, 3) and collect valid image paths
        for label in [0, 1, 2, 3]:  # Include all classes
            folder_path = os.path.join(root_dir, str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith(".tif"):
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)  # Use the original class label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image and apply transformations
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


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


from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

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
            _, preds = torch.max(outputs, 1)  # Get predicted class indices

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute Accuracy & Quadratic Weighted Kappa (QWK)
    accuracy = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    print(f"✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(5)]))  # Adjust for 5 classes
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return y_true, y_pred


model.load_state_dict(torch.load("/kaggle/input/multi-class-eyepacs-ddr-resnet-model/EyePacs_DDR_best_resnet_model.pth"))
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
plt.figure(figsize=(8, 7))  # Adjust figure size for better readability
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=[f"Class {i}" for i in range(4)],  # Labels for predicted classes
    yticklabels=[f"Class {i}" for i in range(4)]   # Labels for true classes
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


import numpy as np

# Convert predictions to numpy for histogram
y_pred = np.array(y_pred)

# Define the number of classes
num_classes = 4  # Classes: 0, 1, 2, 3

plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.hist(
    y_pred, 
    bins=np.arange(-0.5, num_classes + 0.5, 1),  # Bin edges for class indices
    edgecolor='black', 
    alpha=0.7, 
    color='blue'
)
plt.xticks(range(num_classes), labels=[f"Class {i}" for i in range(num_classes)])  # Labels for classes
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class precision-recall curve
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])  # One-hot encode true labels
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3])  # One-hot encode predicted labels

# Compute precision-recall curve for each class
precision = dict()
recall = dict()
average_precision = dict()

n_classes = y_true_bin.shape[1]  # Number of classes

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
    average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])

# Compute micro-average precision-recall curve
precision["micro"], recall["micro"], _ = precision_recall_curve(
    y_true_bin.ravel(), y_pred_bin.ravel()
)
average_precision["micro"] = average_precision_score(y_true_bin, y_pred_bin, average="micro")

# Plot precision-recall curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        recall[i],
        precision[i],
        label=f"Class {i} (AP = {average_precision[i]:.2f})",
    )

# Plot micro-average precision-recall curve
plt.plot(
    recall["micro"],
    precision["micro"],
    label=f"Micro-Average (AP = {average_precision['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.legend(loc="lower left")
plt.grid()
plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class ROC curve
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])  # One-hot encode true labels
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3])  # One-hot encode predicted labels

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = y_true_bin.shape[1]  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(
    y_true_bin.ravel(), y_pred_bin.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
    )

# Plot micro-average ROC curve
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"Micro-Average (AUC = {roc_auc['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)")
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
        label = self.df.iloc[idx]["diagnosis"]  # Use the original class label (0 to 4)

        # Load image
        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

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


from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

# Evaluation function
def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices

            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(5)]))  # Adjust for 5 classes
    
    print("Confusion Matrix:")
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
plt.figure(figsize=(8, 7))  # Adjust figure size for better readability
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=[f"Class {i}" for i in range(5)],  # Labels for predicted classes
    yticklabels=[f"Class {i}" for i in range(5)]   # Labels for true classes
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


import numpy as np

# Convert predictions to numpy for histogram
all_preds = np.array(all_preds)

# Define the number of classes
num_classes = 5  # Classes: 0, 1, 2, 3, 4

plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.hist(
    all_preds, 
    bins=np.arange(-0.5, num_classes + 0.5, 1),  # Bin edges for class indices
    edgecolor='black', 
    alpha=0.7, 
    color='blue'
)
plt.xticks(range(num_classes), labels=[f"Class {i}" for i in range(num_classes)])  # Labels for classes
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class precision-recall curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])  # One-hot encode true labels
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4])    # One-hot encode predicted labels

# Compute precision-recall curve for each class
precision = dict()
recall = dict()
average_precision = dict()

n_classes = all_labels_bin.shape[1]  # Number of classes

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_preds_bin[:, i])
    average_precision[i] = average_precision_score(all_labels_bin[:, i], all_preds_bin[:, i])

# Compute micro-average precision-recall curve
precision["micro"], recall["micro"], _ = precision_recall_curve(
    all_labels_bin.ravel(), all_preds_bin.ravel()
)
average_precision["micro"] = average_precision_score(all_labels_bin, all_preds_bin, average="micro")

# Plot precision-recall curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        recall[i],
        precision[i],
        label=f"Class {i} (AP = {average_precision[i]:.2f})",
    )

# Plot micro-average precision-recall curve
plt.plot(
    recall["micro"],
    precision["micro"],
    label=f"Micro-Average (AP = {average_precision['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.legend(loc="lower left")
plt.grid()
plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class ROC curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])  # One-hot encode true labels
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4])    # One-hot encode predicted labels

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = all_labels_bin.shape[1]  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(
    all_labels_bin.ravel(), all_preds_bin.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
    )

# Plot micro-average ROC curve
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"Micro-Average (AUC = {roc_auc['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)")
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
        label = self.df.iloc[idx]["diagnosis"]  # Use the original class label (0 to 4)

        # Load image
        img_path = os.path.join(self.image_dir, img_id + ".png")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

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
plt.figure(figsize=(8, 7))  # Adjust figure size for better readability
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=[f"Class {i}" for i in range(5)],  # Labels for predicted classes
    yticklabels=[f"Class {i}" for i in range(5)]   # Labels for true classes
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


import numpy as np

# Convert predictions to numpy for histogram
all_preds = np.array(all_preds)

# Define the number of classes
num_classes = 5  # Classes: 0, 1, 2, 3, 4

plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.hist(
    all_preds, 
    bins=np.arange(-0.5, num_classes + 0.5, 1),  # Bin edges for class indices
    edgecolor='black', 
    alpha=0.7, 
    color='blue'
)
plt.xticks(range(num_classes), labels=[f"Class {i}" for i in range(num_classes)])  # Labels for classes
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class precision-recall curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])  # One-hot encode true labels
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4])    # One-hot encode predicted labels

# Compute precision-recall curve for each class
precision = dict()
recall = dict()
average_precision = dict()

n_classes = all_labels_bin.shape[1]  # Number of classes

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_preds_bin[:, i])
    average_precision[i] = average_precision_score(all_labels_bin[:, i], all_preds_bin[:, i])

# Compute micro-average precision-recall curve
precision["micro"], recall["micro"], _ = precision_recall_curve(
    all_labels_bin.ravel(), all_preds_bin.ravel()
)
average_precision["micro"] = average_precision_score(all_labels_bin, all_preds_bin, average="micro")

# Plot precision-recall curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        recall[i],
        precision[i],
        label=f"Class {i} (AP = {average_precision[i]:.2f})",
    )

# Plot micro-average precision-recall curve
plt.plot(
    recall["micro"],
    precision["micro"],
    label=f"Micro-Average (AP = {average_precision['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.legend(loc="lower left")
plt.grid()
plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class ROC curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])  # One-hot encode true labels
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4])    # One-hot encode predicted labels

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = all_labels_bin.shape[1]  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(
    all_labels_bin.ravel(), all_preds_bin.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
    )

# Plot micro-average ROC curve
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"Micro-Average (AUC = {roc_auc['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)")
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
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        # Extract label as multi-class classification
        label = int(self.data.iloc[idx]["adjudicated_dr_grade"])  # Use original class labels (0 to 4)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)  # CrossEntropyLoss expects long labels

# Create dataset and dataloader
test_dataset = MessidorDataset(Messidor2_data, img_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total images found: {len(test_dataset)}")


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


from sklearn.metrics import cohen_kappa_score

# Evaluate model
def evaluate_multiclass_model(model, dataloader):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds

# Run evaluation
all_labels, all_preds = evaluate_multiclass_model(model, test_loader)

# Compute multi-class confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(
    all_labels, 
    all_preds, 
    target_names=[f"Class {i}" for i in range(5)]  # Adjust for 5 classes
))

# Compute Quadratic Weighted Kappa (QWK)
qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 7))  # Adjust figure size for better readability
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=[f"Class {i}" for i in range(5)],  # Labels for predicted classes
    yticklabels=[f"Class {i}" for i in range(5)]   # Labels for true classes
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


import numpy as np

# Convert predictions to numpy for histogram
all_preds = np.array(all_preds)

# Define the number of classes
num_classes = 5  # Classes: 0, 1, 2, 3, 4

plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.hist(
    all_preds, 
    bins=np.arange(-0.5, num_classes + 0.5, 1),  # Bin edges for class indices
    edgecolor='black', 
    alpha=0.7, 
    color='blue'
)
plt.xticks(range(num_classes), labels=[f"Class {i}" for i in range(num_classes)])  # Labels for classes
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class precision-recall curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])  # One-hot encode true labels
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4])    # One-hot encode predicted labels

# Compute precision-recall curve for each class
precision = dict()
recall = dict()
average_precision = dict()

n_classes = all_labels_bin.shape[1]  # Number of classes

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_preds_bin[:, i])
    average_precision[i] = average_precision_score(all_labels_bin[:, i], all_preds_bin[:, i])

# Compute micro-average precision-recall curve
precision["micro"], recall["micro"], _ = precision_recall_curve(
    all_labels_bin.ravel(), all_preds_bin.ravel()
)
average_precision["micro"] = average_precision_score(all_labels_bin, all_preds_bin, average="micro")

# Plot precision-recall curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        recall[i],
        precision[i],
        label=f"Class {i} (AP = {average_precision[i]:.2f})",
    )

# Plot micro-average precision-recall curve
plt.plot(
    recall["micro"],
    precision["micro"],
    label=f"Micro-Average (AP = {average_precision['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.legend(loc="lower left")
plt.grid()
plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class ROC curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])  # One-hot encode true labels
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4])    # One-hot encode predicted labels

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = all_labels_bin.shape[1]  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(
    all_labels_bin.ravel(), all_preds_bin.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
    )

# Plot micro-average ROC curve
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"Micro-Average (AUC = {roc_auc['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid()
plt.show()


import pandas as pd

# Set path to DDR dataset
DDR_DIR = "/kaggle/input/idriddiseasegrading/DDR-dataset/DR_grading"
TEST_CSV = f"{DDR_DIR}/test.csv"

# Load test.csv
df_test = pd.read_csv(TEST_CSV)

# Ensure 'Retinopathy grade' column contains valid integer labels
df_test = df_test.dropna(subset=["Retinopathy grade"])  # Drop rows with missing labels
df_test["Retinopathy grade"] = df_test["Retinopathy grade"].astype(int)  # Ensure integer type

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

        # Filter out rows with label 5 (ignore class 5)
        self.df = self.df[self.df['Retinopathy grade'] != 5].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image file name and corresponding label
        img_id = str(self.df.iloc[idx]['Image name'])
        img_path = os.path.join(self.img_dir, img_id)
        label = int(self.df.iloc[idx]['Retinopathy grade'])  # Use original multi-class labels (0 to 4)

        # Load and preprocess image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformation (resize to match model input)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust based on your model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ImageNet
])

# Set path to DDR test images
DDR_TEST_IMG_DIR = f"{DDR_DIR}/test"

# Create dataset
test_dataset = DDRDataset(df_test, DDR_TEST_IMG_DIR, transform=test_transforms)

# Create DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print("DDR Test DataLoader Ready!")


from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

# Evaluate model
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # Get predicted class indices

        # Store results
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Compute metrics
accuracy = sum([1 for l, p in zip(all_labels, all_preds) if l == p]) / len(all_labels) * 100
qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# Classification Report
report = classification_report(
    all_labels,
    all_preds,
    target_names=[f"Class {i}" for i in range(5)],  # Adjust for 5 classes
    output_dict=False
)

# Print results
print(f"Model Accuracy on DDR Test Data: {accuracy:.2f}%")
print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 7))  # Adjust figure size for better readability
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=[f"Class {i}" for i in range(5)],  # Labels for predicted classes
    yticklabels=[f"Class {i}" for i in range(5)]   # Labels for true classes
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


import numpy as np

# Convert predictions to numpy for histogram
all_preds = np.array(all_preds)

# Define the number of classes
num_classes = 5  # Classes: 0, 1, 2, 3, 4

plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.hist(
    all_preds, 
    bins=np.arange(-0.5, num_classes + 0.5, 1),  # Bin edges for class indices
    edgecolor='black', 
    alpha=0.7, 
    color='blue'
)
plt.xticks(range(num_classes), labels=[f"Class {i}" for i in range(num_classes)])  # Labels for classes
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y")
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class precision-recall curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])  # One-hot encode true labels
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4])    # One-hot encode predicted labels

# Compute precision-recall curve for each class
precision = dict()
recall = dict()
average_precision = dict()

n_classes = all_labels_bin.shape[1]  # Number of classes

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_preds_bin[:, i])
    average_precision[i] = average_precision_score(all_labels_bin[:, i], all_preds_bin[:, i])

# Compute micro-average precision-recall curve
precision["micro"], recall["micro"], _ = precision_recall_curve(
    all_labels_bin.ravel(), all_preds_bin.ravel()
)
average_precision["micro"] = average_precision_score(all_labels_bin, all_preds_bin, average="micro")

# Plot precision-recall curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        recall[i],
        precision[i],
        label=f"Class {i} (AP = {average_precision[i]:.2f})",
    )

# Plot micro-average precision-recall curve
plt.plot(
    recall["micro"],
    precision["micro"],
    label=f"Micro-Average (AP = {average_precision['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.legend(loc="lower left")
plt.grid()
plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the labels for multi-class ROC curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])  # One-hot encode true labels
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4])    # One-hot encode predicted labels

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = all_labels_bin.shape[1]  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(
    all_labels_bin.ravel(), all_preds_bin.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
    )

# Plot micro-average ROC curve
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"Micro-Average (AUC = {roc_auc['micro']:.2f})",
    linestyle="--",
    linewidth=2,
)

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

