!pip install torch torchvision transformers pillow matplotlib opencv-python pandas


!pip install tqdm


import os
import cv2
import torch
import json
from tqdm import tqdm  # For progress bar
from sklearn.model_selection import train_test_split  # For splitting dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Paths
train_videos_path = "/kaggle/input/deepfake-detection-challenge/train_sample_videos"  # Update this to the actual path
metadata_path = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json"  # Update this to the actual metadata.json path

# Load Metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Dataset Class
class DeepfakeDataset(Dataset):
    def __init__(self, video_dir, metadata, processor, frame_count=5, transform=None):
        self.video_dir = video_dir
        self.metadata = metadata
        self.processor = processor
        self.frame_count = frame_count
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.data = list(metadata.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_name = self.data[idx]
        label = 1 if self.metadata[video_name]["label"] == "FAKE" else 0
        video_path = os.path.join(self.video_dir, video_name)

        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(self.frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame)  # Convert NumPy array to PIL image
            frames.append(self.transform(frame))
        cap.release()

        # Pad frames if less than required (for videos with fewer frames)
        while len(frames) < self.frame_count:
            frames.append(torch.zeros_like(frames[0]))

        # Stack frames into a 5D tensor (batch_size, num_frames, channels, height, width)
        frames_tensor = torch.stack(frames)

        # Aggregate frames by averaging across the frame dimension (dim=1)
        aggregated_frame = frames_tensor.mean(dim=0)  # Shape: [channels, height, width]

        # Process the aggregated frame to ensure it fits the model input format
        inputs = self.processor(images=aggregated_frame, return_tensors="pt", do_rescale=False)

        # Ensure the shape of pixel_values is [batch_size, channels, height, width]
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove the extra batch dimension

        return pixel_values, torch.tensor(label)

# Initialize Dataset and DataLoader
processor = AutoImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
train_dataset = DeepfakeDataset(train_videos_path, metadata, processor)

# Split into training and validation sets (80% train, 20% validation)
train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

# Model Setup
model = AutoModelForImageClassification.from_pretrained("Wvolf/ViT_Deepfake_Detection")
model.config.num_labels = 2  # Set for binary classification (REAL vs FAKE)

# Training Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy

# Validation loop
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            # Forward pass through the model
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

            # Calculate accuracy
            accuracy = calculate_accuracy(outputs.logits, labels)

            # Accumulate loss and accuracy for validation
            val_loss += loss.item()
            val_accuracy += accuracy

    # Calculate the average validation loss and accuracy
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    return val_loss, val_accuracy

# Training Loop with Progress Bar and Epoch Display
for epoch in range(20):  # Number of epochs
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    # Progress bar using tqdm with epoch display
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
        for pixel_values, labels in tepoch:
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            # Forward pass through the model
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            
            # Calculate accuracy
            accuracy = calculate_accuracy(outputs.logits, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar description with loss and accuracy
            tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)
            
            # Accumulate loss and accuracy for the epoch
            epoch_loss += loss.item()
            epoch_accuracy += accuracy

    # Calculate average loss and accuracy for the epoch
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    print(f"Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Evaluate on validation data
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the Fine-tuned Model
model.save_pretrained("fine_tuned_deepfake_vit")
processor.save_pretrained("fine_tuned_deepfake_vit")






import os
import cv2
import torch
import json
from tqdm import tqdm  # For progress bar
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Paths
test_videos_path = "/kaggle/input/deepfake-detection-challenge/test_videos"  # Update this to the actual path

# Since there is no metadata, we will create a list of video names (assuming video files are present in the test directory)
test_video_files = os.listdir(test_videos_path)

# Dataset Class (same as before)
class DeepfakeDataset(Dataset):
    def __init__(self, video_dir, video_files, processor, frame_count=5, transform=None):
        self.video_dir = video_dir
        self.video_files = video_files
        self.processor = processor
        self.frame_count = frame_count
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_name)

        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(self.frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame)  # Convert NumPy array to PIL image
            frames.append(self.transform(frame))
        cap.release()

        # Pad frames if less than required (for videos with fewer frames)
        while len(frames) < self.frame_count:
            frames.append(torch.zeros_like(frames[0]))

        # Stack frames into a 5D tensor (batch_size, num_frames, channels, height, width)
        frames_tensor = torch.stack(frames)

        # Aggregate frames by averaging across the frame dimension (dim=1)
        aggregated_frame = frames_tensor.mean(dim=0)  # Shape: [channels, height, width]

        # Process the aggregated frame to ensure it fits the model input format
        inputs = self.processor(images=aggregated_frame, return_tensors="pt", do_rescale=False)

        # Ensure the shape of pixel_values is [batch_size, channels, height, width]
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove the extra batch dimension

        return pixel_values, video_name  # Return video name for saving predictions later

# Initialize Dataset and DataLoader for Test Data
processor = AutoImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
test_dataset = DeepfakeDataset(test_videos_path, test_video_files, processor)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model Setup (load the trained model)
model = AutoModelForImageClassification.from_pretrained("fine_tuned_deepfake_vit")  # Load the fine-tuned model
model.to(device)  # Ensure model is on the correct device

# Test loop with Progress Bar
def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for pixel_values, video_names in tepoch:
                pixel_values = pixel_values.to(device)

                # Forward pass through the model
                outputs = model(pixel_values=pixel_values)
                
                # Get predicted class (0 or 1)
                _, predicted = torch.max(outputs.logits, 1)

                # Store predictions along with video names
                for video_name, pred in zip(video_names, predicted):
                    predicted_class = "FAKE" if pred == 1 else "REAL"
                    predictions.append((video_name, predicted_class))
                
                # Update progress bar with batch completion
                tepoch.set_postfix()

    return predictions

# Run test on the test set
predictions = test(model, test_loader, device)

# Print predictions (video name and predicted label)
for video_name, label in predictions:
    print(f"{video_name}: {label}")


