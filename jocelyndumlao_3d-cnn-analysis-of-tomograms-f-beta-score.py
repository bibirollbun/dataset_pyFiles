import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2  # added for image processing

import warnings
warnings.filterwarnings("ignore")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


!pip download -d ./packages ultralytics
!tar cfvz archive.tar.gz ./packages


!tar xfvz archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages 


# --- 1. Data Loading and Preprocessing ---

# Define data paths
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
test_dir = os.path.join(data_path, "test")
train_dir = os.path.join(data_path, "train")  
train_labels_path = os.path.join(data_path, "train_labels.csv")  
submission_path = os.path.join(data_path, "submission.csv") 
output_path = 'submission.csv'  # Saves it in the /kaggle/working/ directory


# Load training labels
train_labels_df = pd.read_csv(train_labels_path).drop_duplicates(subset='tomo_id', keep='first').set_index('tomo_id')
print("Train Labels DataFrame:")
train_labels_df.head().style.background_gradient(cmap='plasma')



# Function to load a tomogram (stack of images)
def load_tomogram(tomo_dir, resize=(64, 64), fixed_depth=32):
    slices = []
    try:
        slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    except FileNotFoundError:
        print(f"Error: Tomo directory not found: {tomo_dir}")
        return None  # Handle the error and return None

    if len(slice_files) < fixed_depth:
        padding_needed = fixed_depth - len(slice_files)
        empty_img = Image.new('L', resize, color=0)
        slices.extend([np.array(empty_img)] * padding_needed)
    else:
        slice_files = slice_files[:fixed_depth]  # truncate if more than fixed_depth

    for slice_file in slice_files:
        img = Image.open(os.path.join(tomo_dir, slice_file)).convert('L')
        img = img.resize(resize, Image.Resampling.LANCZOS)
        slices.append(np.array(img))

    return np.stack(slices).astype(np.float32) / 255.0


# Custom Dataset class
class TomogramDataset(Dataset):
    def __init__(self, data_dir, labels_df, resize=(64, 64), fixed_depth=32, is_test=False):
        self.data_dir = data_dir
        self.labels_df = labels_df

        # Correctly handle the test dataset case where labels may not be available.
        if is_test:
            # Get a list of tomo_ids from the directory names
            self.tomogram_ids = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        else:
            self.tomogram_ids = list(labels_df.index)

        self.resize = resize
        self.fixed_depth = fixed_depth
        self.is_test = is_test

    def __len__(self):
        return len(self.tomogram_ids)

    def __getitem__(self, idx):
        tomo_id = self.tomogram_ids[idx]
        tomogram_path = os.path.join(self.data_dir, tomo_id)

        tomogram = load_tomogram(tomogram_path, resize=self.resize, fixed_depth=self.fixed_depth)

        if tomogram is None:
            print(f"Warning: Skipping tomo_id {tomo_id} due to loading error.")
            # Handle the error case by returning None or a zero-filled array.
            # Returning None will require changes to the DataLoader's collate_fn,
            # but returning a zero-filled array allows the training loop to continue.
            tomogram = np.zeros((1, self.fixed_depth, self.resize[0], self.resize[1]), dtype=np.float32)
            labels = np.array([-1.0, -1.0, -1.0], dtype=np.float32)  # Default labels
            return torch.tensor(tomogram).unsqueeze(0), torch.tensor(labels), tomo_id  # ***unsqueeze here

        tomogram = torch.tensor(tomogram).unsqueeze(0)  # *** Move unsqueeze here. Ensures consistent shape
        if not self.is_test:
            labels = self.labels_df.loc[tomo_id, ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values.astype(np.float32)
        else:
            labels = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

        return tomogram, torch.tensor(labels), tomo_id



# --- 2. Model Definition ---

class Simple3DCNN(nn.Module):
    def __init__(self, fixed_depth=32):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # Added depth dimension to pooling
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # Added depth dimension to pooling
        self.flatten = nn.Flatten()

        # Calculate the size of the flattened layer dynamically
        self.flattened_size = 32 * (fixed_depth // 4) * (64 // 4) * (64 // 4)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)



# --- 3. Training ---

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, display_images=False):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (tomogram, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")):
            # Skip batches where tomogram is None (due to loading errors)
            if tomogram is None:
                print("Skipping batch due to None tomogram.")
                continue

            tomogram, labels = tomogram.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tomogram)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Display sample tomogram images
            if display_images and batch_idx == 0:  # Show images from the first batch
                fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # Display 5 slices
                sample_tomo = tomogram[0].cpu().squeeze().numpy()  # Convert first tomogram in batch to NumPy, remove the channel dimension
                depth = sample_tomo.shape[0]  # Get depth of 3D volume
                slice_indices = np.linspace(0, depth - 1, 5, dtype=int)  # Select 5 slices

                for i, idx in enumerate(slice_indices):
                    axes[i].imshow(sample_tomo[idx], cmap='gray')  # Show slice
                    axes[i].set_title(f"Slice {idx}")
                    axes[i].axis('off')

                plt.show()

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {running_loss / len(train_loader):.4f}")



# --- 4. Evaluation Metric Implementation ---

def calculate_fbeta_score(predictions, ground_truths, threshold=1000, beta=2):
    """
    Calculates the F-beta score for the motor prediction task.

    Args:
        predictions (np.ndarray): Predicted motor locations (N x 3).
        ground_truths (np.ndarray): Ground truth motor locations (N x 3).
        threshold (float): Euclidean distance threshold for True Positive.
        beta (float): Beta parameter for F-beta score.

    Returns:
        float: F-beta score.
    """
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(ground_truths)):
        gt = ground_truths[i]
        pred = predictions[i]

        # Check if motor exists in ground truth
        motor_exists_gt = not all(gt == -1)

        # Check if motor is predicted
        motor_predicted = not all(pred == -1)

        if motor_exists_gt and motor_predicted:
            distance = np.linalg.norm(gt - pred)
            if distance <= threshold:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif motor_exists_gt and not motor_predicted:
            fn += 1
        elif not motor_exists_gt and motor_predicted:
            fp += 1

    # Calculate precision, recall, and F-beta score
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        fbeta = 0.0
    else:
        fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    return fbeta, precision, recall  # Return precision and recall



# --- 5. Prediction and Submission ---
# Modify predict function

def predict(model, test_loader, confidence_threshold=0.45, max_detections_per_tomo=1):  # Added params
    model.eval()
    predictions = {}
    with torch.no_grad():
        for tomogram, _, tomo_ids in tqdm(test_loader, desc="Predicting"):
            # Check if tomogram is None, skip if so.
            if tomogram is None:
                print(f"Skipping tomo_ids {tomo_ids} due to None tomogram.")
                continue

            outputs = model(tomogram.to(device)).cpu().numpy()  # Get predictions

            for i, tomo_id in enumerate(tomo_ids):
                # Get the prediction for the current tomogram
                prediction = outputs[i]

                # Filter predictions based on confidence threshold
                # No confidence scores available with this model, so returning the prediction directly
                predictions[tomo_id] = prediction

    return predictions


def create_submission_file(predictions, submission_path, output_path='submission.csv'):
    """
    Creates a submission file using an existing submission CSV as a template.

    Args:
        predictions (dict): A dictionary where keys are tomo_ids and values are predicted coordinates.
        submission_path (str): Path to the existing submission CSV file.
        output_path (str): Path to save the updated submission CSV.
    """
    try:
        submission = pd.read_csv(submission_path)
    except FileNotFoundError:
        print(f"Error: Submission file not found at {submission_path}")
        # *** IMPORTANT: If the submission file is NOT found, create a dummy one.
        # *** This is crucial for Kaggle to accept the submission.

        # Create a list of all tomo_ids in the test set
        test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

        # Create a DataFrame with tomo_ids and default predictions (-1)
        submission = pd.DataFrame({'tomo_id': test_tomos, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1})
        submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] = submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].astype(float) # important: cast to float

        print("Created a dummy submission file because the original was not found.")


    # Make sure the necessary columns exist.  If not, add them filled with -1.
    if 'Motor axis 0' not in submission.columns:
        submission['Motor axis 0'] = -1.0
    if 'Motor axis 1' not in submission.columns:
        submission['Motor axis 1'] = -1.0
    if 'Motor axis 2' not in submission.columns:
        submission['Motor axis 2'] = -1.0
    submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] = submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].astype(float)  # important: cast to float

    # Iterate through predictions and update the submission DataFrame
    for tomo_id, coords in predictions.items():
        try:
            submission.loc[submission['tomo_id'] == tomo_id, ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] = coords
        except KeyError:
            print(f"Warning: tomo_id {tomo_id} not found in submission file.")

    # Save the updated submission file
    submission.to_csv(output_path, index=False)
    print(f"Submission file created/updated at {output_path}")

    # Display the head of the submission file
    print("\nSubmission file head:")
    print(submission.head())  # added to show submission head

    # Force the file to be written to disk
    import sys
    sys.stdout.flush()
    # os.fsync(sys.stdout.fileno())




# --- Main Execution ---

if __name__ == '__main__':
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 5
    fixed_depth = 32
    resize_dims = (64, 64)
    threshold = 1000
    beta = 2

    # Define detection parameters (using values from the YOLO script)
    CONFIDENCE_THRESHOLD = 0.45
    MAX_DETECTIONS_PER_TOMO = 1  # Keep track of top N detections per tomogram

    # Create datasets
    train_dataset = TomogramDataset(train_dir, train_labels_df, resize=resize_dims, fixed_depth=fixed_depth)
    test_dataset = TomogramDataset(test_dir, train_labels_df, resize=resize_dims, fixed_depth=fixed_depth, is_test=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = Simple3DCNN(fixed_depth).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs, display_images=True)

    # Make predictions
    predictions = predict(model, test_loader, CONFIDENCE_THRESHOLD, MAX_DETECTIONS_PER_TOMO)  # pass threshold and max detections

    # Create submission file
    create_submission_file(predictions, submission_path)  # pass the path to existing submission CSV

    # --- Evaluation and Plotting ---

    # Load the ground truth labels for evaluation (using training labels for demonstration)
    ground_truths = train_labels_df[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values

    # Convert predictions dictionary to a numpy array, aligning with ground truth order
    predicted_values = []
    tomo_ids_in_order = train_labels_df.index.tolist()  # Get tomo_ids in the order they appear in ground_truths
    for tomo_id in tomo_ids_in_order:
        predicted_values.append(predictions.get(tomo_id, [-1, -1, -1]))  # Use get() to handle missing predictions
    predictions_array = np.array(predicted_values)

    # Calculate evaluation metrics
    fbeta, precision, recall = calculate_fbeta_score(predictions_array, ground_truths, threshold, beta)

    print(f"F-beta score: {fbeta:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Plotting Precision-Recall Curve
    precisions = [precision]  # For a single point
    recalls = [recall]
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', linestyle='-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True)
    plt.show()

    # Plotting F-beta score vs. Threshold

    thresholds = np.arange(500, 2000, 100)
    fbeta_scores = []
    for t in thresholds:
        fbeta_score, _, _ = calculate_fbeta_score(predictions_array, ground_truths, t, beta)
        fbeta_scores.append(fbeta_score)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fbeta_scores, marker='o', linestyle='-')
    plt.xlabel("Threshold")
    plt.ylabel("F-beta Score")
    plt.title("F-beta Score vs. Threshold")
    plt.grid(True)
    plt.show()




