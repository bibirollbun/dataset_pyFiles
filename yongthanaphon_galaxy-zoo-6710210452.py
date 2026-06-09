import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import random
from PIL import Image
from cv2 import imread
import matplotlib.pyplot as plt


!unzip -q ../input/galaxy-zoo-the-galaxy-challenge/images_training_rev1.zip
!unzip -q ../input/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.zip


train_set = pd.read_csv('training_solutions_rev1.csv')
files = os.listdir('./images_training_rev1')

train_set.head()


plt.figure(1, figsize=(9, 9))
plt.axis('off')
n = 0
for i in range(16):
  n += 1
  random_img = './images_training_rev1/'+random.choice(files)
  imgs = imread(random_img)
  plt.subplot(4, 4, n)
  plt.axis('off')
  plt.imshow(imgs)

plt.show()


!pip install torch torchvision matplotlib scikit-learn


# Import libraries
import torch
import numpy as np
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset

# Create a simple model (for demonstration purposes)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(20, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split data into training and validation sets
train_size = int(0.8 * len(X))
train_X, val_X = X[:train_size], X[train_size:]
train_y, val_y = y[:train_size], y[train_size:]

# Create DataLoader for train and validation
train_data = TensorDataset(train_X, train_y)
val_data = TensorDataset(val_X, val_y)
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
valloader = DataLoader(val_data, batch_size=64)

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define Training Loop with Performance Metrics
def train_model(model, trainloader, valloader, criterion, optimizer, epoch):
    model.train()
    for ep in range(epoch):
        train_loss = 0.0
        correct, total = 0, 0
        y_true, y_pred = [], []
        
        for data, label in tqdm(trainloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (output > 0.5).float()  # Convert predictions to 0 or 1
            
            correct += (predicted == label).sum().item()
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        
        # Calculate metrics for this epoch
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        print(f"Epoch {ep+1}/{epoch} - Train Loss: {train_loss/len(trainloader):.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for data, label in valloader:
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
                
                predicted = (output > 0.5).float()
                y_true.extend(label.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        val_accuracy = accuracy_score(y_true, y_pred)
        val_precision = precision_score(y_true, y_pred)
        val_recall = recall_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred)
        val_auc = roc_auc_score(y_true, y_pred)
        
        print(f"Validation Loss: {val_loss/len(valloader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, "
              f"Validation Precision: {val_precision:.4f}, "
              f"Validation Recall: {val_recall:.4f}, "
              f"Validation F1 Score: {val_f1:.4f}, Validation AUC: {val_auc:.4f}")

# Run the training process
train_model(model, trainloader, valloader, criterion, optimizer, epoch=5)



# Import libraries
import torch
import numpy as np
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset

# Create a simple model (for demonstration purposes)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(20, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split data into training and validation sets
train_size = int(0.8 * len(X))
train_X, val_X = X[:train_size], X[train_size:]
train_y, val_y = y[:train_size], y[train_size:]

# Create DataLoader for train and validation
train_data = TensorDataset(train_X, train_y)
val_data = TensorDataset(val_X, val_y)
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
valloader = DataLoader(val_data, batch_size=64)

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define Training Loop with Performance Metrics and Graph Plotting
def train_model(model, trainloader, valloader, criterion, optimizer, epoch):
    # Lists to store metrics for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_aucs, val_aucs = [], []
    
    for ep in range(epoch):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0
        y_true, y_pred = [], []
        
        for data, label in tqdm(trainloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (output > 0.5).float()  # Convert predictions to 0 or 1
            
            correct += (predicted == label).sum().item()
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        
        # Calculate metrics for this epoch
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        train_losses.append(train_loss/len(trainloader))
        train_accuracies.append(accuracy)
        train_precisions.append(precision)
        train_recalls.append(recall)
        train_f1s.append(f1)
        train_aucs.append(auc)
        
        print(f"Epoch {ep+1}/{epoch} - Train Loss: {train_loss/len(trainloader):.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for data, label in valloader:
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
                
                predicted = (output > 0.5).float()
                y_true.extend(label.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        val_accuracy = accuracy_score(y_true, y_pred)
        val_precision = precision_score(y_true, y_pred)
        val_recall = recall_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred)
        val_auc = roc_auc_score(y_true, y_pred)
        
        val_losses.append(val_loss/len(valloader))
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        
        print(f"Validation Loss: {val_loss/len(valloader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, "
              f"Validation Precision: {val_precision:.4f}, "
              f"Validation Recall: {val_recall:.4f}, "
              f"Validation F1 Score: {val_f1:.4f}, Validation AUC: {val_auc:.4f}")
    
    # Plotting the metrics
    epochs = np.arange(1, epoch+1)
    
    # Loss plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision plot
    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_precisions, label='Train Precision')
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.title('Precision per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Recall plot
    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.title('Recall per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # F1 Score plot
    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_f1s, label='Train F1 Score')
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.title('F1 Score per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # AUC plot
    plt.subplot(2, 3, 6)
    plt.plot(epochs, train_aucs, label='Train AUC')
    plt.plot(epochs, val_aucs, label='Validation AUC')
    plt.title('AUC per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    # Display the plots
    plt.tight_layout()
    plt.show()

# Run the training process and plot graphs
train_model(model, trainloader, valloader, criterion, optimizer, epoch=5)


