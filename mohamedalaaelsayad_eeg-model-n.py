# Import core libraries
import pandas as pd               # For handling CSV files and data manipulation
import numpy as np                # For numerical operations (used later)
import os                         # For general file operations
from pathlib import Path          # For handling file paths in a clean, OS-independent way
from tqdm.notebook import tqdm    # For displaying progress bars in notebooks

# Configuration
base_path = Path("/kaggle/input/hms-harmful-brain-activity-classification")  # Path to input dataset on Kaggle
output_path = Path("/kaggle/working/preprocessed_data")                      # Path to save any generated/preprocessed files
output_path.mkdir(exist_ok=True)  # Create output folder if it doesn't exist

# Load the metadata CSV file containing training labels and EEG segment info
df = pd.read_csv(base_path / "train.csv")

# Preview the first 5 rows to understand the structure
df.head(5)


# Calculate the starting row in the full spectrogram file
# Each 2 seconds = 1 row â‡’ divide seconds by 2
df["start_row"] = df["spectrogram_label_offset_seconds"] / 2

# Each sub-spectrogram spans 600 seconds â‡’ 600 / 2 = 300 rows
# So the end row is start_row + 300
df["end_row"] = df["start_row"] + 300

# Display the updated DataFrame to confirm changes
df.head()


# Select the first 25,082 rows from the training data
# This helps reduce processing time and memory usage during development
half_data = df.head(25082)

# Reset the index of the subset 
half_data = half_data.reset_index(drop=True)

# Display the first few rows of the subset
half_data.head()


# Required libraries
import numpy as np
import os
from tqdm import tqdm

# === Step 4: Processing EEG Spectrograms ===
processed_df_list = []  # To store paths and labels for processed files

# Iterate through each row of the dataset with a progress bar
for _, row in tqdm(half_data.iterrows(), total=half_data.shape[0], desc="Processing Spectrograms"):
    
    # Extract spectrogram and label IDs
    spec_id = row['spectrogram_id']
    label_id = row['label_id']
    
    # Build file paths
    input_spec_path = base_path / "train_spectrograms" / f"{spec_id}.parquet"
    output_npy_path = output_path / f"{label_id}.npy"

    # Skip if already processed
    if not os.path.exists(output_npy_path):
        try:
            # Load the .parquet spectrogram file
            spectrogram = pd.read_parquet(input_spec_path)

            # Extract the slice corresponding to the sub-segment (600 seconds â†’ 300 rows)
            start = int(row["start_row"])
            end = int(row["end_row"])
            spectrogram = spectrogram.iloc[start:end]

            # Fill any NaN values with column-wise means (prevents model failure)
            spectrogram = spectrogram.fillna(spectrogram.mean())

            # Drop the 'time' column and convert to NumPy array
            npy_data = spectrogram.drop('time', axis=1).to_numpy()

            # Save the array as .npy for faster loading during training
            np.save(output_npy_path, npy_data)

        except FileNotFoundError:
            print(f"Missing file for spectrogram_id: {spec_id}. Skipping.")
            continue

    # Append output path and label to list for later use
    processed_df_list.append({
        'npy_path': str(output_npy_path),
        'expert_consensus': row['expert_consensus']
    })

# Convert the list of dictionaries into a DataFrame for downstream use
processed_df = pd.DataFrame(processed_df_list)
print(f"\nCreated {len(processed_df)} .npy files.")
processed_df.head()


# Import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Create a bar plot to show the number of samples per class
sns.countplot(x="expert_consensus", data=processed_df)

# Set axis labels and title
plt.xlabel("Classes")  # X-axis: class labels like Seizure, LPD, etc.
plt.ylabel("Counts")   # Y-axis: number of samples per class
plt.title("Distribution of expert_consensus")

# Rotate x-axis labels for better readability if long
plt.xticks(rotation=45)

# Display the plot
plt.show()


# Import PyTorch core and neural network modules
import torch
import torch.nn as nn

# Check if CUDA (GPU) is available; else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the selected device (useful for debugging)
print(f"Using device: {device}")


from sklearn.preprocessing import LabelEncoder

# Create a label encoder instance
le = LabelEncoder()

# Fit the encoder on the expert_consensus labels and transform them into integers
processed_df["class"] = le.fit_transform(processed_df["expert_consensus"])

# Preview the DataFrame with the new numeric class column
processed_df.head()


# Drop the original label column (text) as it's no longer needed
processed_df.drop("expert_consensus", axis=1, inplace=True)

# Preview the updated DataFrame to confirm the change
processed_df.head()


import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df
        self.augment = augment

        # Define simple torchvision-style augmentations (applied only if augment=True)
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

    def __len__(self):
        # Return total number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # Load .npy spectrogram file for the given index
        path = self.df.iloc[idx]["npy_path"]
        data = np.load(path)

        # Normalize: zero mean, unit variance
        data = (data - data.mean()) / (data.std() + 1e-8)

        # Ensure fixed shape: width = 400 columns (time), height = 300 rows (frequencies)
        if data.shape[1] < 400:
            pad_cols = 400 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_cols)), mode='constant')
        elif data.shape[1] > 400:
            data = data[:, :400]

        if data.shape[0] < 300:
            pad_rows = 300 - data.shape[0]
            data = np.pad(data, ((0, pad_rows), (0, 0)), mode='constant')
        elif data.shape[0] > 300:
            data = data[:300, :]

        # Reshape into 3D format: [channels, height, width]
        # Here: [4, 300, 100] by slicing into 4 equal vertical segments (simulating multi-channel input)
        try:
            data = data.reshape(300, 4, 100).transpose(1, 0, 2)  # shape: [4, 300, 100]
        except Exception as e:
            raise ValueError(f"â�Œ Error reshaping file {path} with shape {data.shape}: {e}")

        # Convert to PyTorch tensor
        data = torch.from_numpy(data).float()

        # Apply augmentation if enabled
        if self.augment:
            data = self.transforms(data)

        # Get label and convert to tensor
        label = torch.tensor(self.df.iloc[idx]["class"], dtype=torch.long)

        return data, label


from sklearn.model_selection import train_test_split

# For reproducibility â€” ensures the same random split every run
torch.manual_seed(42)

# Split data into train and validation sets (80/20 split), keeping class distribution balanced
train_df, val_df = train_test_split(
    processed_df,
    test_size=0.2,
    random_state=42,
    stratify=processed_df['class']  # Ensures class distribution remains balanced in both sets
)

# Create dataset instances
# augment=False: no data augmentation applied here (you can enable later if needed)
train_dataset = SpectrogramDataset(train_df, augment=False)
val_dataset = SpectrogramDataset(val_df)

# Define batch size
BATCH_SIZE = 32

# Create DataLoaders for batch processing
# shuffle=True only for training (important for randomness)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Print number of batches in each loader
print(f"The Train Data contains {len(train_loader)} batches and Validation Data contains {len(val_loader)} batches. Each batch contains {BATCH_SIZE} examples.")


import torch
import torch.nn as nn

class EEG_Spectrogram_CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(EEG_Spectrogram_CNN, self).__init__()

        # === Convolutional Block 1 ===
        # Input: (Batch, 4, 300, 100)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(5, 5), stride=1, padding=2),  # â†’ (Batch, 32, 300, 100)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # â†’ (Batch, 32, 150, 50)
        )

        # === Convolutional Block 2 ===
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),  # â†’ (Batch, 64, 150, 50)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # â†’ (Batch, 64, 75, 25)
        )

        # === Convolutional Block 3 ===
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),  # â†’ (Batch, 128, 75, 25)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=3)  # â†’ (Batch, 128, 25, 8)
        )

        # Flatten layer to prepare for fully connected classification head
        self.flatten = nn.Flatten()  # Output size: 128 Ã— 25 Ã— 8 = 25,600 features

        # === Fully Connected Classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(128 * 25 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(1024, num_classes)  # Output: logits for 6 classes
        )

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)  # â†’ (Batch, 32, 150, 50)
        x = self.conv2(x)  # â†’ (Batch, 64, 75, 25)
        x = self.conv3(x)  # â†’ (Batch, 128, 25, 8)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


# Set seed to ensure reproducibility
torch.manual_seed(42)

# Initialize the model and move it to the GPU (or CPU if CUDA is not available)
model = EEG_Spectrogram_CNN().to(device)

# Create a dummy input with the same shape as actual data: [Batch, Channels, Height, Width]
dummy_input = torch.randn([1, 4, 300, 100]).to(device)

# Forward pass to ensure the model runs correctly
output = model(dummy_input)
print(output.shape)  # Should print: torch.Size([1, 6])


# Define the loss function
# CrossEntropyLoss expects raw logits and class labels (no softmax needed)
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
# Adam adjusts learning rates adaptively and usually converges faster
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


from tqdm import tqdm

epochs = 12

for epoch in tqdm(range(epochs)):
    # === Training Phase ===
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0

    for batch, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)

        # Backward pass (compute gradients)
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate loss and accuracy
        train_loss += loss.item()
        correct = (y_pred.argmax(dim=1) == y).sum().item()
        train_accuracy += correct / y.size(0)

        # Optional progress print every ~5000 samples
        if batch % 157 == 0:
            print(f"{batch * BATCH_SIZE} from {len(train_loader) * BATCH_SIZE}")

    # Average metrics over all batches
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # === Validation Phase ===
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():  # Disable gradient tracking for validation
        for batch, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            # Accumulate loss and accuracy
            val_loss += loss.item()
            correct = (y_pred.argmax(dim=1) == y).sum().item()
            val_accuracy += correct / y.size(0)

            # Optional progress print
            if batch % 25 == 0:
                print(f"{batch * BATCH_SIZE} from {len(val_loader) * BATCH_SIZE}")

    # Average validation metrics
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)

    print(f"         >> Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Step 1: Collect true and predicted labels ===
# Make sure your model is in evaluation mode and not tracking gradients
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)

        # Store predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# === Step 2: Plot Normalized Confusion Matrix (as % per true class) ===
ConfusionMatrixDisplay.from_predictions(
    y_true=all_labels,
    y_pred=all_preds,
    normalize='true',              # Normalize by true labels (row-wise)
    values_format='.0%',           # Display values as whole number percentages
    display_labels=le.classes_     # Use the original class names from LabelEncoder
)

# === Step 3: Plot Formatting ===
plt.title("Normalized Confusion Matrix (%)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Highlight errors only
sample_weight = (all_labels != all_preds).astype(int)

# Plot confusion matrix using errors only
ConfusionMatrixDisplay.from_predictions(
    y_true=all_labels,
    y_pred=all_preds,
    sample_weight=sample_weight,
    normalize='true',
    values_format='.0%',
    display_labels=le.classes_
)

plt.title("Confusion Matrix (Misclassifications Only, %)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# === Load and preprocess test sample ===
test_df = pd.read_parquet("/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/853520.parquet")

# Drop the 'time' column and convert to NumPy array
test_array = np.array(test_df.iloc[:, 1:])

# Normalize (zero mean, unit variance)
test_array = (test_array - test_array.mean()) / (test_array.std() + 1e-8)

# Reshape to match model input: (4, 300, 100)
test_array = test_array.reshape(300, 4, 100).transpose(1, 0, 2)

# Convert to torch tensor and move to device
test_tensor = torch.tensor(test_array, dtype=torch.float).to(device)

# Add batch dimension: shape becomes (1, 4, 300, 100)
test_tensor = test_tensor.unsqueeze(0)

# === Inference ===
model.eval()  # Ensure model is in evaluation mode
y_test_logit = model(test_tensor)
y_test_probs = torch.softmax(y_test_logit, dim=1)
predicted_class = torch.argmax(y_test_probs, dim=1)

# Display the predicted class index
predicted_class


# Save the full model (architecture + parameters)
torch.save(model, "EEGModel.pth")

# Save only the model weights (recommended for deployment)
torch.save(model.state_dict(), "EEGModelw.pth")

