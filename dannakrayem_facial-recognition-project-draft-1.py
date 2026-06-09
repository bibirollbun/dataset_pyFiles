import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



BATCH_SIZE = 64 # Defines the batch size for training
#provides number of samples processed before updating the model)
#Set the number of epochs or full passes over the entire training dataset
EPOCHS = 30  # Higher for better convergence
#Set the learning rate for the optimizer to control step size during optimization
LEARNING_RATE = 0.0003  # lower learning rate increases stability
# Select the device to run the model on i.e. GPU if available otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')
df = df[df['emotion'] < 7]  #Exclusion of unused class


# Print the shape of the dataset number of rows and columns
print("Dataset Shape:", df.shape)
# Print the count of each unique value in the 'emotion' column to show class distribution
print("\nEmotion Class Distribution:\n", df['emotion'].value_counts())
# Print header to indicate descriptive statistics will follow
print("\nDescriptive Statistics (pixel strings converted to float arrays):")
# Print summary statistics like mean, std, min, max for each numerical column in the dataframe
print(df.describe())


plt.figure(figsize=(8,5))
df['emotion'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Emotion Class Distribution")
plt.xlabel("Emotion Label")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.show()


# Select the first 9 rows from the 'pixels' column, convert each pixel string to a 48x48 numpy array of floats
pixels_sample = df['pixels'].iloc[:9].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48))
# Create a 3x3 grid of subplots with figure size 8x8 inches
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
# Add a main title to the entire figure
fig.suptitle("Sample Images from Dataset", fontsize=16)
# Loop through each subplot axis and the corresponding image data
for i, ax in enumerate(axes.flat):
    # Display the image in grayscale
    ax.imshow(pixels_sample.iloc[i], cmap='gray')
     # Set the title of the subplot to show the label of the image (emotion)
    ax.set_title(f"Label: {df['emotion'].iloc[i]}")
    # Remove the axis ticks and labels for cleaner visualization
    ax.axis('off')
# Adjust the padding between subplots to prevent overlap
plt.tight_layout()
plt.show()


# Convert each pixel string in the 'pixels' column into a numpy array of float32 values
pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
# Stack all the pixel arrays into a single numpy array X
# Shape: num_samples x num_pixels
X = np.stack(pixels.to_numpy())
# Extract the 'emotion' column as a numpy array to use as labels
y = df['emotion'].to_numpy()


# Normalize and reshape images to range [0, 1] by dividing by 255 max pixel intensity
X /= 255.0


# Define a sequence of image transformations to apply on each image
transform = transforms.Compose([
    # Convert numpy array or tensor image to PIL Image format
    transforms.ToPILImage(),
    # Convert the image to grayscale with 1 output channel
    transforms.Grayscale(num_output_channels=1),
    # Convert the PIL Image back to a tensor (C x H x W) with pixel values in [0,1]
    transforms.ToTensor(),
    # Normalize the tensor image by subtracting mean=0.5 and dividing by std=0.5 for each channel
    transforms.Normalize(mean=[0.5], std=[0.5])
])



class FERDataset(Dataset):
      # Initialize with images, labels, and optional transform function
    def __init__(self, images, labels, transform=None):
        self.images = images # Store image data (numpy arrays)
        self.labels = labels  # Store corresponding labels
        self.transform = transform # Store image transformations to apply

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.images)

    # Retrieve an image-label pair at the given index
    def __getitem__(self, idx):
        # Reshape the image to 48x48 and convert to unsigned 8 bit integer type pixel values
        img = self.images[idx].reshape(48, 48).astype(np.uint8)  # Ensures 48x48 input
         # Apply transformations
        if self.transform:
            img = self.transform(img)
        # Convert the label to a PyTorch tensor of type long required for classification
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # Return the processed image and its label
        return img, label


# Split the dataset into training and validation sets 80% train, 20% validation
#fixed random seed for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Create training dataset object applying the defined transforms
train_dataset = FERDataset(X_train, y_train, transform=transform)
# Create validation dataset object applying the same transforms
val_dataset = FERDataset(X_val, y_val, transform=transform)
# Create DataLoader for training dataset to load data in batches and shuffle each epoch
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Create DataLoader for validation dataset to load data in batches without shuffling
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# Define a convolutional neural network model for emotion recognition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            # 1st convolutional layer: input 1 channel, output 64 channels, 3x3 kernel, padding 1 to keep size
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Input is 1-channel
            nn.BatchNorm2d(64), # Batch normalization to stabilize learning
            nn.ReLU(), # Non-linear activation
            nn.MaxPool2d(2), # Downsample by factor of 2 (48x48 -> 24x24)

            # 2nd convolutional layer: input 64 channels, output 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # Downsample (24x24 -> 12x12)

            # 3rd convolutional layer: input 128 channels, output 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), # Downsample (12x12 -> 6x6)

            nn.Flatten(),  # Flatten feature maps to a vector
            # Fully connected layer: 256 channels * 6 * 6 spatial size to 512 units
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5), #dropout for regularization

            # Final output layer with 7 classes (for 7 emotions)
            nn.Linear(512, 7)
        )

    # Define the forward pass through the network
    def forward(self, x):
        return self.net(x)


# Instantiate the CNN model and move it to the specified device (GPU or CPU)
model = CNN().to(DEVICE)
# Define the loss function - CrossEntropyLoss is used for multi-class classification
criterion = nn.CrossEntropyLoss()
# Define the optimizer - Adam optimizer with the specified learning rate
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(EPOCHS):
    model.train() # Set the model to training mode to enables dropout, batchnorm, etc.
    running_loss = 0.0 # Accumulate loss over the epoch
    # For loop through the training data in batches
    for images, labels in train_loader:
        # Move data to the same device as the model
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        # Forward pass: compute model predictions
        outputs = model(images)
        # Compute loss between predicted outputs and true labels
        loss = criterion(outputs, labels)
        # Backward pass: compute gradients
        loss.backward()
        # Update model parameters based on gradients
        optimizer.step()
        # Accumulate the loss for reporting
        running_loss += loss.item()
    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")




# Set the model to evaluation mode (disables dropout, etc.)
model.eval()
# Initialize counters for correct predictions and total samples
correct = 0
total = 0
# Disable gradient calculation during evaluation for efficiency
with torch.no_grad():
    # Iterate over the validation data
    for images, labels in val_loader:
        # Move data to the same device as the model GPU or CPU
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Forward pass: compute predictions
        outputs = model(images)
        # Get the predicted class by taking the index of the max logit
        _, predicted = torch.max(outputs.data, 1)
        # Update total number of samples
        total += labels.size(0)
        # Count how many predictions were correct
        correct += (predicted == labels).sum().item()

# Print the final validation accuracy as a percentage
print(f"Validation Accuracy: {100 * correct / total:.2f}%")

