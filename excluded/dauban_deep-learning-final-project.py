# BLOCK 1: Import Core Libraries, Download Dataset, and Setup Modules

# Core libraries for numerical operations and data handling
import numpy as np              # Linear algebra operations
import pandas as pd             # Data processing and CSV file I/O
import os                       # Operating system interface
import kagglehub                # Module for downloading datasets from Kaggle

# Visualization libraries
import matplotlib.pyplot as plt # For plotting graphs

# PyTorch and related libraries for deep learning
import torch                    # PyTorch core library
import torch.nn as nn           # Neural network modules
import torch.nn.functional as F # Functional interface for neural networks
import torch.optim as optim     # Optimization algorithms

import torchvision              # Computer vision package in PyTorch
import torchvision.transforms as transforms  # Image transformations
from torch.utils.data import DataLoader, Dataset  # Data loading utilities

# Scikit-learn libraries for model evaluation and data splitting
from sklearn.model_selection import train_test_split  # Splitting data into training and validation sets
from sklearn.metrics import accuracy_score, f1_score    # Evaluation metrics

# PIL for image processing
from PIL import Image

# Download the dataset and store its local path
dataset_path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")

# Print the dataset path and list the downloaded files
print("Dataset downloaded to:", dataset_path)
print("Files in dataset directory:", os.listdir(dataset_path))

# Note:
# - Up to 20GB can be written to the current directory (/kaggle/working/), 
#   which is preserved when creating a version using "Save & Run All".
# - Files written to /kaggle/temp/ will not persist outside the current session.




# BLOCK 2: Setup Device
# Determine whether to use GPU (CUDA) or CPU for computation.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# BLOCK 3: Define Paths and Load CSV Files
# Set the base path for the dataset.
BASE_PATH = "/kaggle/input/ai-vs-human-generated-dataset"

# Define paths to the training and test image directories.
TRAIN_FOLDER = os.path.join(BASE_PATH, "train_data")
TEST_FOLDER  = os.path.join(BASE_PATH, "test_data_v2")

# Define paths to the CSV files.
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV  = os.path.join(BASE_PATH, "test.csv")

# Load the CSV files into pandas DataFrames.
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Display the first five rows of the training DataFrame.
print("Sample training data:")
print(train_df.head(5))



# BLOCK 4: Split Data into Training/Validation Sets and Define Image Transformations

from sklearn.model_selection import train_test_split

# Split the training DataFrame into training and validation sets (80% training, 20% validation)
train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

print("Training set size:", len(train_data))
print("Validation set size:", len(val_data))

# Define image transformations for the training set
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize images to 224x224 pixels
    transforms.RandomHorizontalFlip(),        # Apply random horizontal flipping for data augmentation
    transforms.ToTensor(),                    # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet's mean and standard deviation
                         std=[0.229, 0.224, 0.225])
])

# Define image transformations for the validation set
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize images to 224x224 pixels
    transforms.ToTensor(),                    # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet's mean and standard deviation
                         std=[0.229, 0.224, 0.225])
])




# BLOCK 5: Define Custom Dataset and Verify Sample Image Path

class CustomDataset(Dataset):
    """
    Custom Dataset class for loading images and their labels.
    
    Parameters:
      dataframe: pandas DataFrame containing image file names and labels.
      image_dir: Directory where the images are stored.
      transform: Optional image transformations.
      file_name_col: Column name in the DataFrame that contains image file names.
      label_col: Column name in the DataFrame that contains labels.
                 For unlabeled data, this can be set to None.
    """
    def __init__(self, dataframe, image_dir, transform=None, file_name_col='file_name', label_col='label'):
        self.dataframe = dataframe
        self.transform = transform
        self.image_dir = image_dir
        self.file_name_col = file_name_col
        self.label_col = label_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Retrieve the image file name from the DataFrame
        img_file = self.dataframe.iloc[idx][self.file_name_col]
        # Remove any directory components to obtain just the file name
        img_file = os.path.basename(img_file)
        
        # Construct the full path to the image
        img_path = os.path.join(self.image_dir, img_file)
        
        # Retrieve the label if available, otherwise return a dummy value
        if self.label_col is not None and self.label_col in self.dataframe.columns:
            label = self.dataframe.iloc[idx][self.label_col]
        else:
            label = -1  # Dummy label for unlabeled data
        
        # Check if the image file exists; print a warning if not
        if not os.path.exists(img_path):
            print("Warning: Image path does not exist -", img_path)
        
        # Open the image and apply transformations if specified
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Optional: Verify a sample image path from the training set
sample_img_path = os.path.join(TRAIN_FOLDER, train_df.iloc[0]['file_name'].replace("train_data/", ""))
print("Sample image path:", sample_img_path)
print("Image exists:", os.path.exists(sample_img_path))



# BLOCK 6: Create DataLoaders for Training, Validation, and Test Sets

batch_size = 32  # Define the mini-batch size

# Create Dataset objects for training and validation using the split data
train_dataset = CustomDataset(train_data, image_dir=TRAIN_FOLDER, transform=transform_train)
val_dataset = CustomDataset(val_data, image_dir=TRAIN_FOLDER, transform=transform_val)

# Create the test dataset using the 'id' column for file names
# Note: The test data is unlabeled, so label_col is set to None.
test_dataset = CustomDataset(test_df, image_dir=TEST_FOLDER, transform=transform_val, file_name_col='id', label_col=None)

# Create DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Test set size:", len(test_dataset), "samples")

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("Training set size (after split):", len(train_dataset), "samples")
print("Validation set size:", len(val_dataset), "samples")



# BLOCK 7: Define and Initialize the CNN Classifier

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened dimension after convolutions
        # For an input size of 224x224, the output after two pooling layers is 56x56 with 128 channels.
        self.flatten_dim = 128 * 56 * 56
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))  # Additional pooling for dimension reduction
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the CNN model and move it to the designated device
cnn_model = CNNClassifier(num_classes=2).to(device)

# Print the model architecture
print(cnn_model)



# BLOCK 8: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)  # Adam optimizer with learning rate 0.0001

print("Model running on:", device)

# BLOCK 9: Define Training Function for the CNN Model
def train_model_cnn(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """
    Train the CNN model on the training dataset.
    
    Parameters:
      model (nn.Module): The CNN model.
      train_loader (DataLoader): DataLoader for training data.
      criterion: Loss function.
      optimizer: Optimization algorithm.
      device: Computation device (CPU or GPU).
      num_epochs (int): Number of training epochs.
    """
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()       # Reset gradients
            outputs = model(images)     # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()             # Backward pass
            optimizer.step()            # Update weights
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")



# BLOCK 10: Define Evaluation Function for the CNN Model

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model_cnn(model, val_loader, device):
    """
    Evaluate the CNN model on the validation dataset.
    
    Parameters:
      model (nn.Module): The CNN model.
      val_loader (DataLoader): DataLoader for the validation data.
      device: Computation device (CPU or GPU).
    
    Returns:
      acc_percent (float): Accuracy on the validation set (percentage).
    """
    model.eval()  # Set the model to evaluation mode
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Human", "AI-Generated"]))
    
    # Plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Human", "AI-Generated"],
                yticklabels=["Human", "AI-Generated"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    return acc * 100.0  # Return accuracy as a percentage


# BLOCK 11: Train the CNN Model and Evaluate on the Validation Set

num_epochs = 1  # Number of training epochs
train_model_cnn(cnn_model, train_loader, criterion, optimizer, device, num_epochs)

# Evaluate the CNN model on the validation set and print the accuracy
val_accuracy = evaluate_model_cnn(cnn_model, val_loader, device)
print(f"Validation Accuracy: {val_accuracy:.2f}%")


# BLOCK 12: Hyperparameter Tuning Using the Updated Evaluation Function

# Define a list of learning rates to experiment with.
learning_rates = [0.1, 0.01, 0.001, 0.0001]
best_acc = 0
best_lr = None

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    # Reinitialize the CNN model and optimizer for this trial.
    cnn_model = CNNClassifier(num_classes=2).to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=lr)
    
    # Train the model for 1 epoch (quick experiment).
    train_model_cnn(cnn_model, train_loader, criterion, optimizer, device, num_epochs=1)
    
    # Evaluate the model on the validation set.
    acc = evaluate_model_cnn(cnn_model, val_loader, device)
    print(f"Validation accuracy with lr {lr}: {acc:.2f}%\n")
    
    # Update the best learning rate if this trial yields a higher accuracy.
    if acc > best_acc:
        best_acc = acc
        best_lr = lr

print(f"Best learning rate: {best_lr} with validation accuracy: {best_acc:.2f}%")




# BLOCK 13: Predict on Test Set and Visualize Prediction Distribution

def predict_test(model, test_loader, device):
    """
    Generate predictions on the test dataset using the provided model.
    
    Parameters:
      model (nn.Module): Trained CNN model.
      test_loader (DataLoader): DataLoader for test data.
      device (torch.device): Device on which computations will be performed.
      
    Returns:
      List of predicted class labels.
    """
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    with torch.no_grad():
        for images, _ in test_loader:  # Test labels are not available
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    return predictions

# Generate predictions on the test set using the CNN model
test_predictions = predict_test(cnn_model, test_loader, device)

# Display the first 20 predictions
print("First 20 predictions:", test_predictions[:20])

# Count the number of predictions for each class
unique, counts = np.unique(test_predictions, return_counts=True)
print("Prediction counts:")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count}")

# Visualize the distribution of predicted classes using a count plot
plt.figure(figsize=(8, 6))
sns.countplot(x=test_predictions)
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Distribution of Predicted Classes on Test Set")
plt.show()



# BLOCK 14: Compute and Plot Color Histograms and Statistics for an Example Image

import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the base path and paths to the training folder and CSV file.
BASE_PATH = "/kaggle/input/ai-vs-human-generated-dataset"
TRAIN_FOLDER = os.path.join(BASE_PATH, "train_data")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")

# Load the training CSV into a DataFrame.
train_df = pd.read_csv(TRAIN_CSV)
print("Train DataFrame columns:", train_df.columns)
print("First 5 rows of train_df:")
print(train_df.head())

def compute_color_histogram(image_path):
    """
    Compute the color histograms for the red, green, and blue channels of an image.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        hist_r, hist_g, hist_b: Histograms for the red, green, and blue channels.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
    
    return hist_r, hist_g, hist_b

def compute_color_statistics(image_path):
    """
    Compute the mean and standard deviation for each color channel of an image.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        means, stds: Flattened arrays containing the mean and standard deviation 
                     for the red, green, and blue channels.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    means, stds = cv2.meanStdDev(image)
    return means.flatten(), stds.flatten()

# Select an example image from the training set.
example_file = train_df.iloc[0]['file_name']
# Remove any unwanted prefix from the file name if present.
example_file = example_file.replace("train_data/", "")
image_path = os.path.join(TRAIN_FOLDER, example_file)

# Verify that the image exists and then compute statistics.
if not os.path.exists(image_path):
    print("Image not found:", image_path)
else:
    hist_r, hist_g, hist_b = compute_color_histogram(image_path)
    means, stds = compute_color_statistics(image_path)
    
    print("Channel Means (R, G, B):", means)
    print("Channel Standard Deviations (R, G, B):", stds)
    
    # Plot the color histograms in separate subplots.
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(hist_r, color='red')
    plt.title('Red Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.plot(hist_g, color='green')
    plt.title('Green Histogram')
    plt.xlabel('Pixel Value')
    
    plt.subplot(1, 3, 3)
    plt.plot(hist_b, color='blue')
    plt.title('Blue Histogram')
    plt.xlabel('Pixel Value')
    
    plt.tight_layout()
    plt.show()



# BLOCK 15: Compute and Visualize Additional Image Statistics

import cv2
import numpy as np
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Define functions to compute image statistics
# -----------------------------
def compute_saturation_statistics(image_path):
    """
    Compute the mean and standard deviation of the saturation channel of an image.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        tuple: (mean_saturation, std_saturation)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mean_sat = np.mean(saturation)
    std_sat = np.std(saturation)
    return mean_sat, std_sat

def compute_image_entropy(image_path):
    """
    Compute the Shannon entropy of an image.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        float: Shannon entropy of the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(gray)
    return entropy

def compute_unique_colors(image_path):
    """
    Compute the number of unique colors in an image.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        int: Number of unique colors.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, image.shape[2])
    unique_colors = np.unique(pixels, axis=0)
    return len(unique_colors)

def compute_signal_to_noise_ratio(image_path):
    """
    Compute an approximate signal-to-noise ratio (SNR) from a grayscale image.
    The SNR is approximated as the ratio of the mean to the standard deviation
    of pixel intensities.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        float: Approximate signal-to-noise ratio.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    snr = mean_intensity / std_intensity if std_intensity != 0 else float('inf')
    return snr

# -----------------------------
# Example usage: Compute statistics for an example image from the training set
# -----------------------------
# Assume that 'train_df' and 'TRAIN_FOLDER' are defined from previous blocks.
example_file = train_df.iloc[0]['file_name'].replace("train_data/", "")
image_path = os.path.join(TRAIN_FOLDER, example_file)

if not os.path.exists(image_path):
    print("Image not found:", image_path)
else:
    # Compute statistics
    mean_sat, std_sat = compute_saturation_statistics(image_path)
    entropy = compute_image_entropy(image_path)
    unique_colors = compute_unique_colors(image_path)
    snr = compute_signal_to_noise_ratio(image_path)
    
    print("Saturation Mean:", mean_sat)
    print("Saturation Standard Deviation:", std_sat)
    print("Image Entropy:", entropy)
    print("Unique Colors Count:", unique_colors)
    print("Approximate Signal-to-Noise Ratio:", snr)
    
    # -----------------------------
    # Plot 1: Saturation Histogram (Separate Figure)
    # -----------------------------
    # Load the image and convert to HSV to extract the saturation channel.
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_img[:, :, 1]
    
    plt.figure(figsize=(8, 6))
    plt.hist(saturation_channel.ravel(), bins=256, color='purple', alpha=0.7)
    plt.title("Saturation Histogram")
    plt.xlabel("Saturation Value (0-255)")
    plt.ylabel("Frequency")
    plt.text(30, plt.ylim()[1] * 0.9, 
             "Saturation indicates the intensity of color. Lower values correspond to dull colors; higher values indicate vivid colors.",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    
    # -----------------------------
    # Plot 2: Bar Chart for Additional Image Statistics (Separate Figure)
    # -----------------------------
    plt.figure(figsize=(8, 6))
    stats_names = ["Sat Mean", "Sat Std", "Entropy", "SNR"]
    stats_values = [mean_sat, std_sat, entropy, snr]
    
    bars = plt.bar(stats_names, stats_values, color=['red', 'green', 'blue', 'orange'])
    plt.title("Additional Image Statistics")
    plt.ylabel("Value")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom')
    plt.show()



# BLOCK A-C: Setup Vision Transformer Model, Loss Function, Optimizer, and Data Transformations

import timm
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# Load and configure a pre-trained Vision Transformer (ViT) model for binary classification.
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
vit_model = vit_model.to(device)
print("ViT Model Architecture:")
print(vit_model)

# Define the loss function (CrossEntropyLoss) for classification.
criterion_vit = nn.CrossEntropyLoss()

# Define the optimizer for the ViT model with fine-tuned hyperparameters.
optimizer_vit = optim.Adam(vit_model.parameters(), lr=1.8972443761018018e-05, weight_decay=7.813579510269746e-06)

# Define image transformations required by the ViT model.
# These transformations resize images to 224x224 and normalize them using ImageNet statistics.
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Note: Ensure that your DataLoader objects (e.g., train_loader, val_loader) use these transforms.



# BLOCK D: Training Function for the Vision Transformer (ViT) Model

def train_model_vit(model, train_loader, criterion, optimizer, device, num_epochs=1):
    """
    Train the Vision Transformer (ViT) model.

    Parameters:
      model (nn.Module): The ViT model.
      train_loader (DataLoader): DataLoader for training data.
      criterion: Loss function.
      optimizer: Optimization algorithm.
      device: Computation device (CPU or GPU).
      num_epochs (int): Number of epochs to train.
    """
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()          # Reset gradients
            outputs = model(images)        # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                # Backward pass
            optimizer.step()               # Update parameters
            
            running_loss += loss.item()
            
            # Calculate the number of correct predictions
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Compute loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"[ViT] Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")



# BLOCK E: Evaluation Function for the Vision Transformer (ViT) Model

def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on the validation dataset and display performance metrics.
    
    Parameters:
      model (nn.Module): The model to evaluate.
      val_loader (DataLoader): DataLoader for the validation data.
      device (torch.device): Computation device (CPU or GPU).
      
    Prints:
      - Classification report with precision, recall, and f1-score.
      - Confusion matrix plot.
    """
    model.eval()  # Set model to evaluation mode
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Human", "AI-Generated"]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Human", "AI-Generated"],
                yticklabels=["Human", "AI-Generated"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()



#BLOCK F: Train and Evaluate the ViT Model
# Set the number of training epochs for the ViT model
num_epochs_vit = 1

# Train the ViT model using the training DataLoader
train_model_vit(vit_model, train_loader, criterion_vit, optimizer_vit, device, num_epochs=num_epochs_vit)

# Evaluate the trained ViT model on the validation DataLoader
evaluate_model(vit_model, val_loader, device)


# BLOCK G: Hyperparameter Tuning with Optuna

import optuna

def objective(trial):
    """
    Objective function for hyperparameter tuning using Optuna.
    
    This function suggests values for the learning rate and weight decay,
    updates the optimizer accordingly, and runs a short training loop to 
    compute the average loss as a performance metric.
    
    Parameters:
      trial (optuna.trial.Trial): An Optuna trial object.
      
    Returns:
      float: Average loss over the trial epochs.
    """
    # Suggest values for the learning rate and weight decay parameters.
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    
    # Update the optimizer for the ViT model with the suggested parameters.
    optimizer = optim.Adam(vit_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Run a short training loop (e.g., 2 epochs) to obtain a metric.
    num_epochs_trial = 2
    vit_model.train()
    running_loss = 0.0
    for epoch in range(num_epochs_trial):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = vit_model(images)
            loss = criterion_vit(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    # Calculate the average loss over all batches and epochs.
    avg_loss = running_loss / (num_epochs_trial * len(train_loader))
    return avg_loss

# Create an Optuna study to minimize the average loss.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial
print("  Loss: {}".format(trial.value))
print("  Parameters:")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))



# BLOCK H: Create Test Dataset for ViT, Perform Inference, and Visualize Prediction Distribution

# Define a custom Dataset for the test set used by the Vision Transformer (ViT)
class CustomTestDatasetViT(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Initializes the custom test dataset.
        
        Parameters:
            dataframe (pandas.DataFrame): DataFrame containing test image file paths in the 'id' column.
            transform: Image transformations to apply.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.image_dir = TEST_FOLDER  # TEST_FOLDER is defined in BLOCK 4

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Retrieve the file path from the 'id' column and remove the "test_data_v2/" prefix.
        file_path = self.dataframe.iloc[idx]['id'].replace("test_data_v2/", "")
        img_path = os.path.join(self.image_dir, file_path)
        # Open the image and convert to RGB.
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image  # Test set does not contain labels

# Create the test dataset for ViT.
test_dataset_vit = CustomTestDatasetViT(test_df, transform=transform_val)

# Create a DataLoader for the test dataset.
test_loader_vit = DataLoader(test_dataset_vit, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
print("Test set size for ViT:", len(test_dataset_vit), "samples")

# Define a function to generate predictions on the test set using the ViT model.
def predict_test_set_vit(model, test_loader, device):
    """
    Perform inference on the test set using the ViT model.
    
    Parameters:
        model (nn.Module): The Vision Transformer model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Computation device (CPU or GPU).
    
    Returns:
        List of predicted class labels.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

# Run predictions on the test set using the ViT model.
test_predictions = predict_test_set_vit(vit_model, test_loader_vit, device)
print("Sample predictions from ViT on test set:", test_predictions[:20])

# Visualize the distribution of predictions.
import numpy as np
import matplotlib.pyplot as plt

pred_array = np.array(test_predictions)
unique, counts = np.unique(pred_array, return_counts=True)

# Define class labels (assuming 0 corresponds to 'Human' and 1 to 'AI-Generated')
class_labels = ["Human", "AI-Generated"]

plt.figure(figsize=(6, 4))
plt.bar(unique, counts, tick_label=class_labels, color=['skyblue', 'salmon'])
plt.xlabel("Class")
plt.ylabel("Number of Predictions")
plt.title("Distribution of Test Set Predictions")
plt.show()

# Create a DataFrame to display the distribution of predictions.
import pandas as pd
df_distribution = pd.DataFrame({
    "Class": class_labels,
    "Count": counts
})
print("Prediction Distribution:")
print(df_distribution)



# BLOCK I
#Model Comparison
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_return(model, data_loader, device):
    """
    Evaluates the given model on data_loader.
    Returns:
      y_true: List of true labels.
      y_pred: List of predicted labels.
      acc: Accuracy (in percentage).
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred) * 100.0
    return y_true, y_pred, acc

# --- Evaluate the CNN model ---
print("=== CNN Model Evaluation ===")
y_true_cnn, y_pred_cnn, cnn_accuracy = evaluate_model_return(cnn_model, val_loader, device)
print(f"CNN Validation Accuracy: {cnn_accuracy:.2f}%")
print("CNN Classification Report:")
print(classification_report(y_true_cnn, y_pred_cnn, target_names=["Human", "AI-Generated"]))

# --- Evaluate the ViT model ---
print("\n=== ViT Model Evaluation ===")
y_true_vit, y_pred_vit, vit_accuracy = evaluate_model_return(vit_model, val_loader, device)
print(f"ViT Validation Accuracy: {vit_accuracy:.2f}%")
print("ViT Classification Report:")
print(classification_report(y_true_vit, y_pred_vit, target_names=["Human", "AI-Generated"]))

# --- Plot Confusion Matrices Side by Side ---
cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
cm_vit = confusion_matrix(y_true_vit, y_pred_vit)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Human", "AI-Generated"], 
            yticklabels=["Human", "AI-Generated"], ax=axs[0])
axs[0].set_title("CNN Confusion Matrix")

sns.heatmap(cm_vit, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Human", "AI-Generated"], 
            yticklabels=["Human", "AI-Generated"], ax=axs[1])
axs[1].set_title("ViT Confusion Matrix")

plt.tight_layout()
plt.show()


