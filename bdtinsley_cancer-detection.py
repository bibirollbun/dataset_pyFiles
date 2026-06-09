import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import os

# Use HLS palette for more fun diagram colorings
sns.set_palette("hls")

# Store the directory for easy access
in_dir = '/kaggle/input/histopathologic-cancer-detection'
out_dir = '/kaggle/working'

# Load the labels
labels_df = pd.read_csv(f'{in_dir}/train_labels.csv')

# Basic information about the dataset
print("\n=== Dataset Overview ===")
print(f"Total number of images: {len(labels_df)}")
print("\nLabel distribution:")
print(labels_df['label'].value_counts(normalize=True).round(3) * 100)


# Basic information about the dataset
print("\n=== Dataset Overview ===")
print(f"Total number of images: {len(labels_df)}")
print("\nLabel distribution:")
print(labels_df['label'].value_counts(normalize=True).round(3) * 100)


# Create a pie chart of the distribution
plt.figure(figsize=(8, 6))
labels_df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Cancer vs Non-Cancer Cases')
plt.ylabel('')
plt.savefig('label_distribution.png')
plt.plot()


# Basic statistics
print("\n=== Basic Statistics ===")
print(labels_df.describe())

# Create a bar plot of the distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=labels_df, x='label')
plt.title('Count of Cancer vs Non-Cancer Cases')
plt.xlabel('Label (0: No Cancer, 1: Cancer)')
plt.ylabel('Count')
plt.savefig('label_counts.png')
plt.plot()




# Checking for any missing values
print("\n=== Missing Values ===")
print(labels_df.isnull().sum())


def get_image_sizes(directory):
    sizes = []
    for img_path in Path(directory).glob('*.tif'):
        try:
            size = os.path.getsize(img_path)
            sizes.append(size)
        except:
            continue
    return sizes

image_sizes = get_image_sizes(f'{dir}/test')
print("\n=== Image Size Statistics ===")
print(pd.Series(image_sizes).describe())


# Display images as a grid of
n_samples = 25  # number of images to display
n_cols = 5     # number of columns in the grid
n_rows = (n_samples + n_cols - 1) // n_cols

# Create a figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
axes = axes.flatten()  # flatten the axes array for easier iteration

# Get random sample of indices
random_indices = np.random.choice(len(labels_df), n_samples, replace=False)

# Display images
for i, ax in enumerate(axes):
    if i < n_samples:
        # Get the image path and label
        img = random_indices[i]
        img_path = f'{in_dir}/train/{labels_df.iloc[img]["id"]}.tif' 
        label = labels_df.iloc[img]["label"]
        
        try:
            # Load and display the image
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f'Label: {label}\n(Cancer: {label == 1})')
            ax.axis('off')  # hide axes
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading image\n{str(e)}', 
                   ha='center', va='center')
            ax.axis('off')
    else:
        ax.axis('off')  # hide empty subplots if any

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
from tqdm import tqdm
from sklearn.metrics import silhouette_score

def extract_features(image_path, n_bins=16):
    """Extract basic features from an image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Basic statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Histogram features with variable number of bins
    hist = cv2.calcHist([gray], [0], None, [n_bins], [0, 256])
    hist = hist.flatten() / hist.sum()  # normalize histogram
    
    # Texture features (using Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.var(laplacian)
    
    # Combine all features
    features = np.concatenate([
        [mean_intensity, std_intensity, texture],
        hist
    ])
    
    return features

# Test different numbers of bins
n_bins_list = [8, 16, 32, 64]
results = []

# Determine the best number of bins to use for the model
for n_bins in n_bins_list:
    print(f"\nTesting with {n_bins} histogram bins...")
    
    # Extract features
    features_list = []
    valid_indices = []
    
    print(f"Extracting features for {len(labels_df)} images...")
    # Use tqdm to show progress for feature extraction
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        img_path = f'{in_dir}/train/{row["id"]}.tif'
        features = extract_features(img_path, n_bins)
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
    
    print(f"Converting to numpy array...")
    # Convert to numpy array
    X = np.array(features_list)
    y = labels_df.iloc[valid_indices]['label'].values
    
    print(f"Splitting and scaling data...")
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use a subset for silhouette score calculation
    sample_size = min(5000, len(X_train_scaled))
    sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    X_sample = X_train_scaled[sample_indices]
    y_sample = y_train[sample_indices]

    print("Calculating silhouette score on subset of data...")
    silhouette = silhouette_score(X_sample, y_sample)
    
    print(f"Training model...") 
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print(f"Getting predictions...")
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Evaluating model...")
    results.append({
        'n_bins': n_bins,
        'accuracy': accuracy,
        'silhouette': silhouette,
        'n_features': X.shape[1]
    })
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Number of features: {X.shape[1]}")

# Create DataFrame and find best bin size
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df.to_string(index=False))

# Find best bin size based on both metrics
best_accuracy_bins = results_df.loc[results_df['accuracy'].idxmax(), 'n_bins']
best_silhouette_bins = results_df.loc[results_df['silhouette'].idxmax(), 'n_bins']

print(f"\nBest bin size for accuracy: {best_accuracy_bins}")
print(f"Best bin size for silhouette score: {best_silhouette_bins}")

# Plot results
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(results_df['n_bins'], results_df['accuracy'], 'bo-')
plt.xlabel('Number of Histogram Bins')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Bins')
plt.grid(True, alpha=0.3)

# Plot silhouette score
plt.subplot(1, 2, 2)
plt.plot(results_df['n_bins'], results_df['silhouette'], 'ro-')
plt.xlabel('Number of Histogram Bins')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Bins')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Set the optimal number of bins
n_bins = 16
print(f"Building final model with {n_bins} histogram bins...")

# Extract features
features_list = []
valid_indices = []

for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    img_path = f'{in_dir}/train/{row["id"]}.tif'
    features = extract_features(img_path, n_bins)
    if features is not None:
        features_list.append(features)
        valid_indices.append(idx)

# Convert to numpy array
X = np.array(features_list)
y = labels_df.iloc[valid_indices]['label'].values

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train final model
final_model = LogisticRegression(random_state=42)
final_model.fit(X_train_scaled, y_train)

# Get predictions
y_pred = final_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Model Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Number of features: {X.shape[1]}")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce features to 2D using PCA
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

# Create a mesh grid for the decision boundary
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Fit logistic regression on 2D data
log_reg_2d = LogisticRegression(random_state=42)
log_reg_2d.fit(X_train_2d, y_train)

# Get predictions for the mesh grid
Z = log_reg_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Create the plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, alpha=0.8)
plt.colorbar(label='Probability of Cancer')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Logistic Regression Decision Boundary (2D PCA projection)')
plt.show()

# Print explained variance ratio
print("\nExplained variance ratio of first two components:")
print(pca.explained_variance_ratio_)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Define the CNN architecture
class CancerCNN(nn.Module):
    def __init__(self):
        super(CancerCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        x = self.pool(self.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool(self.relu(self.conv2(x)))
        
        # Third conv block
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        
        return x

# Custom Dataset class
class CancerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.data.iloc[idx]['id']}.tif"
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


# Create datasets
train_dataset = CancerDataset(in_dir + '/train_labels.csv', in_dir + '/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Split into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
# Use MPS (Apple Silicon) if available, otherwise use GPU if available, otherwise use CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnn_model = CancerCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            # Detach the loss before adding to running_loss
            running_loss += loss.detach().item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                
                val_running_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(cnn_model, train_loader, val_loader, criterion, optimizer)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



# Function to predict on new images
def predict_image(model, image_path, transform):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        output = model(image)
        prediction = (output.squeeze() > 0.5).float()
        probability = output.squeeze().item()
        return prediction.item(), probability

# Create a submission for the CNN model using  predict_image
def create_cnn_submission(model):
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Create a list to store predictions
    predictions = []
    
    # Get all test images
    test_images = [f for f in os.listdir(f'{in_dir}/test') if f.endswith('.tif')]
    print(f"Found {len(test_images)} test images")
    
    # Process each test image
    for img in tqdm(test_images):
        image_path = f'{in_dir}/test/{img}'
        try:
            # Get prediction
            prediction, probability = predict_image(model, image_path, transform)
            
            # Extract image ID (remove .tif extension)
            image_id = img.split('.')[0]
            
            # Add to predictions
            predictions.append({
                'id': image_id,
                'label': prediction
            })
            
        except Exception as e:
            print(f"Error processing {img}: {str(e)}")
    
    # Create DataFrame and save to csv
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(f'{out_dir}/submission.csv', index=False)
    print(f"\nCreated submission.csv with {len(submission_df)} predictions")
    
    # Display first few rows to verify format
    print("\nFirst few rows of submission file:")
    print(submission_df.head())
    
# Run and done
create_cnn_submission(cnn_model)

