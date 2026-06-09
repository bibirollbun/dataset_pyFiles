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
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data paths
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
test_dir = os.path.join(data_path, "test")
train_dir = os.path.join(data_path, "train")
train_labels_path = os.path.join(data_path, "train_labels.csv")
submission_path = os.path.join(data_path, "submission.csv")
output_path = 'submission.csv'

# Load training labels
try:
    train_labels_df = pd.read_csv(train_labels_path).drop_duplicates(subset='tomo_id', keep='first').set_index('tomo_id')
    print("Train Labels DataFrame:")
    print(train_labels_df.head())
except FileNotFoundError:
    print(f"Error: File not found at {train_labels_path}. Please ensure the dataset is correctly attached.")
    train_labels_df = None  # Or handle the error appropriately


def load_tomogram(tomogram_path, resize=(64, 64), fixed_depth=32):
    """
    Loads a tomogram from a directory containing image slices (for Kaggle).

    Args:
        tomogram_path (str): Path to the directory containing the tomogram slices.
        resize (tuple): Target size for resizing the slices.
        fixed_depth (int): Target depth of the tomogram.

    Returns:
        numpy.ndarray: The loaded and preprocessed tomogram, or None if an error occurs.
    """
    try:
        slices = sorted([f for f in os.listdir(tomogram_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        tomogram = []
        for slice_file in slices:
            slice_path = os.path.join(tomogram_path, slice_file)
            img = Image.open(slice_path).convert('L')  # Convert to grayscale
            img = img.resize(resize)
            tomogram.append(np.array(img))

        # Pad or truncate to the fixed depth
        if len(tomogram) < fixed_depth:
            padding = fixed_depth - len(tomogram)
            tomogram.extend([np.zeros_like(tomogram[0])] * padding)
        elif len(tomogram) > fixed_depth:
            tomogram = tomogram[:fixed_depth]

        tomogram = np.array(tomogram, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        return tomogram
    except Exception as e:
        print(f"Error loading tomogram from {tomogram_path}: {e}")
        return None

# Custom Dataset class
class TomogramDataset(Dataset):
    def __init__(self, data_dir, labels_df, resize=(64, 64), fixed_depth=32, is_test=False):
        self.data_dir = data_dir
        self.labels_df = labels_df
        if is_test:
            self.tomogram_ids = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        else:
            self.tomogram_ids = list(labels_df.index)
        self.resize = resize
        self.fixed_depth = fixed_depth
        self.is_test = is_test

    def __len__(self):
        return len(self.tomogram_ids)

    def __getitem__(self, idx):
        start_time = time.time() #add start time
        tomo_id = self.tomogram_ids[idx]
        tomogram_path = os.path.join(self.data_dir, tomo_id)
        tomogram = load_tomogram(tomogram_path, resize=self.resize, fixed_depth=self.fixed_depth)
        if tomogram is None:
            tomogram = np.zeros((1, self.fixed_depth, self.resize[0], self.resize[1]), dtype=np.float32)
            labels = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            return torch.tensor(tomogram).unsqueeze(0), torch.tensor(labels), tomo_id
        tomogram = torch.tensor(tomogram).unsqueeze(0)
        if not self.is_test:
            labels = self.labels_df.loc[tomo_id, ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values.astype(np.float32)
        else:
            labels = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        end_time = time.time() #add end time.
        print(f"Time to load {tomo_id}: {end_time - start_time:.4f} seconds") #print the time it took to load.
        return tomogram, torch.tensor(labels), tomo_id


# Model Definition
class Simple3DCNN(nn.Module):
    def __init__(self, fixed_depth=32, dropout_rate=0.2): # Added dropout rate as a parameter
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16) # Added batch normalization
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32) # Added batch normalization
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.flatten = nn.Flatten()
        self.flattened_size = 32 * (fixed_depth // 4) * (64 // 4) * (64 // 4)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.bn3 = nn.BatchNorm1d(128) # Added batch normalization
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) # Added dropout
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x)))) #batch normalization added.
        x = self.pool2(self.relu2(self.bn2(self.conv2(x)))) #batch normalization added.
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.bn3(self.fc1(x)))) #batch normalization and dropout added.
        return self.fc2(x)


# Training
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, display_images=False):
    model.train()
    scaler = torch.cuda.amp.GradScaler() # Add mixed precision scaler
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (tomogram, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")):
            print(f"Tomogram shape: {tomogram.shape}, Labels shape: {labels.shape}") #verify the shapes.
            if tomogram is None:
                continue
            tomogram, labels = tomogram.to(device), labels.to(device)
            model= model.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): #mixed precision
                outputs = model(tomogram)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward() #mixed precision.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            scaler.step(optimizer) #mixed precision
            scaler.update() #mixed precision.
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}: Loss = {loss.item()}")

        # --- Pair Plot ---
        if display_images: #moved outside of batch loop.
            df = pd.DataFrame(labels.cpu().numpy(), columns=['Motor axis 0', 'Motor axis 1', 'Motor axis 2'])
            sns.pairplot(df)
            plt.show()

            fig, axes = plt.subplots(1, 5, figsize=(15, 5))
            sample_tomo = tomogram[0].cpu().squeeze().numpy()
            depth = sample_tomo.shape[0]
            slice_indices = np.linspace(0, depth - 1, 5, dtype=int)
            for i, idx in enumerate(slice_indices):
                axes[i].imshow(sample_tomo[idx], cmap='gray')
                axes[i].set_title(f"Slice {idx}")
                axes[i].axis('off')
            plt.show()

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {running_loss / len(train_loader):.4f}")

    return model #Added return model, so that the trained model can be used.


# Evaluation Metric Implementation
import numpy as np
from scipy.spatial import distance

def calculate_fbeta_score(predictions, ground_truths, threshold=1000, beta=2):
    """
    Calculates the F-beta score, precision, and recall for motor detection.

    Args:
        predictions (list or np.ndarray): Predicted motor coordinates.
        ground_truths (list or np.ndarray): Ground truth motor coordinates.
        threshold (int): Distance threshold for considering a prediction as a true positive.
        beta (float): Beta value for F-beta score calculation.

    Returns:
        tuple: (fbeta, precision, recall)
    """

    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)

    # Identify where motors exist in ground truth and predictions
    motor_exists_gt = ~np.all(ground_truths == -1, axis=1)
    motor_predicted = ~np.all(predictions == -1, axis=1)

    # Calculate distances
    distances = np.array([distance.euclidean(gt, pred) if me_gt and mp else float('inf') for gt, pred, me_gt, mp in zip(ground_truths, predictions, motor_exists_gt, motor_predicted)])

    # Determine true positives, false positives, and false negatives
    tp_mask = motor_exists_gt & motor_predicted & (distances <= threshold)
    fp_mask = motor_predicted & ~tp_mask
    fn_mask = motor_exists_gt & ~motor_predicted

    tp = np.sum(tp_mask)
    fp = np.sum(fp_mask)
    fn = np.sum(fn_mask)

    # Calculate precision, recall, and F-beta score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0.0

    return fbeta, precision, recall


if __name__ == '__main__':
    batch_size = 4
    learning_rate = 0.00001  # Further reduced learning rate
    num_epochs = 10  # Increase epochs
    fixed_depth = 32
    resize_dims = (64, 64)
    threshold = 1000
    beta = 2
    CONFIDENCE_THRESHOLD = 0.45
    MAX_DETECTIONS_PER_TOMO = 1

    train_dataset = TomogramDataset(train_dir, train_labels_df, resize=resize_dims, fixed_depth=fixed_depth)
    test_dataset = TomogramDataset(test_dir, train_labels_df, resize=resize_dims, fixed_depth=fixed_depth, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) #added num_workers
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2) #added num_workers

    model = Simple3DCNN(fixed_depth).to(device)
    criterion = nn.HuberLoss()  # Use Huber Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') #Add learning rate scheduler

    def modified_train_model(model, train_loader, criterion, optimizer, num_epochs=10, display_images=False):
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (tomogram, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")):
                if tomogram is None:
                    continue
                tomogram, labels = tomogram.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(tomogram)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}: Loss = {loss.item()}")

            if display_images: #modified to be outside of batch_idx == 0
                for tomogram, labels, _ in train_loader: #modified to iterate over the entire train loader.
                    tomogram, labels = tomogram.to(device), labels.to(device)
                
                    df = pd.DataFrame(labels.cpu().numpy(), columns=['Motor axis 0', 'Motor axis 1', 'Motor axis 2'])
                    sns.pairplot(df)
                    plt.show()

                    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
                    sample_tomo = tomogram[0].cpu().squeeze().numpy()
                    depth = sample_tomo.shape[0]
                    slice_indices = np.linspace(0, depth - 1, 5, dtype=int)
                    for i, idx in enumerate(slice_indices):
                        axes[i].imshow(sample_tomo[idx], cmap='gray')
                        axes[i].set_title(f"Slice {idx}")
                        axes[i].axis('off')
                    plt.show()
                    break #added break, so that only one set of plots is produced per epoch.

            epoch_loss = running_loss / len(train_loader)
            scheduler.step(epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_loss:.4f}")

    modified_train_model(model, train_loader, criterion, optimizer, num_epochs, display_images=True)

    predictions = predict(model, test_loader, CONFIDENCE_THRESHOLD, MAX_DETECTIONS_PER_TOMO)

    create_submission_file(predictions, submission_path)

    ground_truths = train_labels_df[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
    predicted_values = []
    tomo_ids_in_order = train_labels_df.index.tolist()
    for tomo_id in tomo_ids_in_order:
        predicted_values.append(predictions.get(tomo_id, [-1, -1, -1]))
    predictions_array = np.array(predicted_values)

    fbeta, precision, recall = calculate_fbeta_score(predictions_array, ground_truths, threshold, beta)

    print(f"F-beta score: {fbeta:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    precisions = [precision]
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


# Prediction and Submission

def predict(model, test_loader, confidence_threshold=0.45, max_detections_per_tomo=1):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for tomogram, _, tomo_ids in tqdm(test_loader, desc="Predicting"):
            if tomogram is None:
                print(f"Skipping tomo_ids {tomo_ids} due to None tomogram.")
                continue
            outputs = model(tomogram.to(device)).cpu().numpy()
            for i, tomo_id in enumerate(tomo_ids):
                prediction = outputs[i]
                predictions[tomo_id] = prediction
    return predictions

def create_submission_file(predictions, submission_path, output_path='submission.csv'):
    try:
        submission = pd.read_csv(submission_path)
    except FileNotFoundError:
        print(f"Error: Submission file not found at {submission_path}")
        test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        submission = pd.DataFrame({'tomo_id': test_tomos, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1})
        submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] = submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].astype(float)
        print("Created a dummy submission file because the original was not found.")

    if 'Motor axis 0' not in submission.columns:
        submission['Motor axis 0'] = -1.0
    if 'Motor axis 1' not in submission.columns:
        submission['Motor axis 1'] = -1.0
    if 'Motor axis 2' not in submission.columns:
        submission['Motor axis 2'] = -1.0
    submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] = submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].astype(float)

    for tomo_id, coords in predictions.items():
        try:
            submission.loc[submission['tomo_id'] == tomo_id, ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] = coords
        except KeyError:
            print(f"Warning: tomo_id {tomo_id} not found in submission file.")
    submission.to_csv(output_path, index=False)
    print(f"Submission file created/updated at {output_path}")
    print("\nSubmission file head:")
    print(submission.head())
    import sys
    sys.stdout.flush()


# Assuming you have already run the previous cells and have access to:
# - train_labels_df
# - predictions (dictionary of predictions)
# - predictions_array (numpy array of predicted values)
# - ground_truths (numpy array of ground truth values)
# - fbeta, precision, recall (evaluation metrics)

# --- Graphical Dump ---

# 1. Pie Charts (Distribution of Number of Motors)
motor_counts = train_labels_df['Number of motors'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(motor_counts, labels=motor_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Number of Motors')
plt.show()

# 2. Pair Plot (Ground Truth Motor Coordinates)
sns.pairplot(train_labels_df[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']])
plt.suptitle('Pair Plot of Ground Truth Motor Coordinates', y=1.02)
plt.show()

# 3. KDE Plot (Distribution of Predicted vs. Ground Truth Motor Axis 0)
plt.figure(figsize=(10, 6))
sns.kdeplot(ground_truths[:, 0], label='Ground Truth Axis 0', fill=True)
sns.kdeplot(predictions_array[:, 0], label='Predicted Axis 0', fill=True)
plt.title('KDE Plot of Predicted vs. Ground Truth Motor Axis 0')
plt.legend()
plt.show()

# 4. Pair Grid (Comparison of Predicted vs. Ground Truth)
df_comparison = pd.DataFrame({
    'Ground Truth Axis 0': ground_truths[:, 0],
    'Predicted Axis 0': predictions_array[:, 0],
    'Ground Truth Axis 1': ground_truths[:, 1],
    'Predicted Axis 1': predictions_array[:, 1],
    'Ground Truth Axis 2': ground_truths[:, 2],
    'Predicted Axis 2': predictions_array[:, 2],
})
g = sns.PairGrid(df_comparison)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.fig.suptitle('Pair Grid: Predicted vs. Ground Truth Motor Coordinates', y=1.02)
plt.show()

# 5. Relplot (Scatter Plot of Predicted vs. Ground Truth Axis 1)
plt.figure(figsize=(10, 6))
sns.relplot(x=ground_truths[:, 1], y=predictions_array[:, 1], kind='scatter')
plt.title('Relplot: Predicted vs. Ground Truth Motor Axis 1')
plt.show()

# 6. Catplot (Distribution of Voxel Spacing)
plt.figure(figsize=(10, 6))
sns.catplot(x='Voxel spacing', kind='count', data=train_labels_df)
plt.title('Catplot: Distribution of Voxel Spacing')
plt.show()

# 7. Density Plot (2D Density of Predicted vs. Ground Truth Axis 2)
plt.figure(figsize=(10, 6))
sns.kdeplot(x=ground_truths[:, 2], y=predictions_array[:, 2], fill=True)
plt.title('2D Density Plot: Predicted vs. Ground Truth Motor Axis 2')
plt.show()

# 8. Area Chart (Distribution of Array Shape Axis 0)
array_shape_counts = train_labels_df['Array shape (axis 0)'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.fill_between(array_shape_counts.index, array_shape_counts.values)
plt.title('Area Chart: Distribution of Array Shape (Axis 0)')
plt.xlabel('Array Shape (Axis 0)')
plt.ylabel('Count')
plt.show()

# 9. fbeta vs threshold
thresholds = np.arange(500, 2000, 100)
fbeta_scores = []
beta = 2
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

# 10. Precision recall curve
precisions = [precision]
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

