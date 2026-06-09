import polars as pl
import pandas as pd
import numpy as np
import pickle # To save/load scaler and other objects
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from scipy.spatial.transform import Rotation as R

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score # For evaluating


import os
import json




"""
Hierarchical macro F1 metric for the CMI 2025 Challenge.

This script defines a single entry point `score(solution, submission, row_id_column_name)`
that the Kaggle metrics orchestrator will call.
It performs validation on submission IDs and computes a combined binary & multiclass F1 score.
"""
class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass


class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures

    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:

        # Validate gestures
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )

        # Compute binary F1 (Target vs Non-Target)
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )

        # Build multi-class labels for gestures
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')

        # Compute macro F1 over all gesture classes
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )
        print(f1_binary)
        print(f1_macro)
        return 0.5 * f1_binary + 0.5 * f1_macro


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str
) -> float:
    """
    Compute hierarchical macro F1 for the CMI 2025 challenge.

    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'sequence_id': unique identifier for each sequence
      - 'gesture': one of the eight target gestures or "Non-Target"

    This metric averages:
    1. Binary F1 on SequenceType (Target vs Non-Target)
    2. Macro F1 on gesture (mapping non-targets to "Non-Target")

    Raises ParticipantVisibleError for invalid submissions,
    including invalid SequenceType or gesture values.


    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': range(4), 'gesture': ['Eyebrow - pull hair']*4})
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Forehead - pull hairline']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.5
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Text on phone']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.0
    >>> score(solution, solution, row_id_column_name=row_id_column_name)
    1.0
    """
    # Validate required columns
    for col in (row_id_column_name, 'gesture'):
        if col not in solution.columns:
            raise ParticipantVisibleError(f"Solution file missing required column: '{col}'")
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission file missing required column: '{col}'")

    metric = CompetitionMetric()
    return metric.calculate_hierarchical_f1(solution, submission)




path1 = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
path2 = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"


# --- Configuration & Global Variables ---
# Define the features that will be used for training and prediction
TRAIN = False

FEATURE_COLUMNS = [
    'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w',
    'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag',
    'rot_angle', 'acc_mag_jerk', 'rot_angle_vel'
]
TARGET_COLUMN = 'gesture'
SEQUENCE_ID_COLUMN = 'sequence_id'

############################################################################
# needs attention while prediction
MAX_SEQUENCE_LENGTH = 700 # Based on prior EDA on the train.csv
                          # Also beware we don't know the lenght of a gesture in a
                          # test dataset
                          # This is crucial for padding/truncating sequences.
############################################################################

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 2
DROPOUT_RATE = 0.2
N_SPLITS = 5 # For StratifiedGroupKFold

MODEL_SAVE_PATH = 'trained_model.pth'
SCALER_SAVE_PATH = ''
MAPPING_SAVE_PATH = 'gesture_mapping.pkl'
MAX_SEQ_LEN_SAVE_PATH = 'max_sequence_length.pkl' # Save this as well!

LABEL_MAP_PATH = 'label_map.json'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


if TRAIN:
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)


if TRAIN:
    max_sequence_len = df2.groupby('sequence_id').size().max()
    print(max_sequence_len)


if TRAIN:
    # create a new dataframe df2_imu containing only the raw imu data
    df2_imu = df2[
                    [
                      "sequence_id",
                      "acc_x",
                      "acc_y",
                      "acc_z",
                      "rot_x",
                      "rot_y",
                      "rot_z",
                      "rot_w",
                      "gesture"
                    ]
                  ]
    df2_imu['sequence_id'] = df2_imu['sequence_id'].str.slice(start=4).astype(int)


if TRAIN:
    acc_values = df2[["acc_x","acc_y","acc_z"]]
    rot_values = df2[["rot_x","rot_y","rot_z","rot_w"]]


def remove_gravity_from_acc(acc_data:pd.DataFrame, rot_data:pd.DataFrame):

  # check if acc_data is a pandas dataframe
  # if yes acc_values convert it into a numpy array
  # else have it as asis

  if isinstance(acc_data, pd.DataFrame):
      acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
  else:
      acc_values = acc_data

  # check if rot_data is a pandas dataframe
  # if yes quat_values convert it into a numpy array
  # else have it as asis

  if isinstance(rot_data, pd.DataFrame):
      quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
  else:
      quat_values = rot_data

  num_samples = acc_values.shape[0] # no of rows in acc_values

  # create a numpy array linear_accel filled with 0s whose shape is like
  # acc_values
  linear_accel = np.zeros_like(acc_values)

  # create a numpy array with earths gravity
  gravity_world = np.array([0, 0, 9.81])


  # iterate as many times as there are rows in acc_values
  for i in range(num_samples):
      # IF all the values in i^th row of quat_values are "NaN (not a number)"
      #                                OR
      # IF all the values in i^th row of quat_values are close to 0
      # THEN copy ALL values in i^th row of acc_values to i^th row of
      #    linear_accel and goto the next iteration of i
      if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
          linear_accel[i, :] = acc_values[i, :]
          continue


      try:
          # create a rotation object using the i^th row of quat_values
          rotation = R.from_quat(quat_values[i])

          # rotate the gravity_world vector using the i^th row quarternion to
          # the sensor_frame (inverse=True means rotate the gravity vector to
          # the sensor frame)
          gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)

          # substract the rotated gravity vector (components) from the corresponding components of acc_values
          linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
      except ValueError:
            linear_accel[i, :] = acc_values[i, :]

  return linear_accel


if TRAIN:
    remove_gravity_from_acc(acc_values,rot_values)
    
    df2_imu['linear_acc_x'] = remove_gravity_from_acc(acc_values,rot_values)[:,0]
    df2_imu['linear_acc_y'] = remove_gravity_from_acc(acc_values,rot_values)[:,1]
    df2_imu['linear_acc_z'] = remove_gravity_from_acc(acc_values,rot_values)[:,2]


def linear_acc_mag(linear_acc : pd.DataFrame):
  return np.sqrt(
      linear_acc['linear_acc_x']**2 + \
      linear_acc['linear_acc_y']**2 + \
      linear_acc['linear_acc_z']**2
      )


if TRAIN:
    df2_imu['linear_acc_mag'] = linear_acc_mag(df2_imu[['linear_acc_x',   'linear_acc_y','linear_acc_z']])


if TRAIN:
    df2_imu['rot_angle'] = 2 * np.arccos(df2_imu['rot_w'].clip(-1, 1))


if TRAIN:
    df2_imu['acc_mag_jerk'] = df2_imu.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)


if TRAIN:
    df2_imu['rot_angle_vel'] = df2_imu.groupby('sequence_id')['rot_angle'].diff().fillna(0)


if TRAIN:
    df2_imu.head()


if TRAIN:
    df2_imu[FEATURE_COLUMNS] = df2_imu[FEATURE_COLUMNS].ffill().bfill().fillna(0)


if TRAIN:
    nan_present = df2_imu.isnull().any()
    
    print("Columns with at least one NaN value:")
    print(nan_present)


if TRAIN:
    # Create integer label mapping and save it
    label_map = {label: i for i, label in enumerate(df2_imu['gesture'].unique())}
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f)
    df2_imu['gesture_id'] = df2_imu['gesture'].map(label_map)
    print(f"Gesture to ID map: {label_map}")


if TRAIN:
    feature_cols = [col for col in df2_imu.columns if col not in ['sequence_id', 'gesture', 'gesture_id']]
    X = df2_imu[feature_cols]
    y = df2_imu['gesture_id']
    groups = df2_imu['sequence_id']



class GestureDataset(Dataset):
  def __init__(self, sequences):
    """
    Args:
        sequences (list): A list of dictionaries, where each dictionary
                          represents a single gesture sequence.
                          e.g., [{'data': np.array, 'label': int}, ...]
    """
    self.sequences = sequences

  def __len__(self):
    # The total number of sequences (gestures) in the dataset
    return len(self.sequences)

  def __getitem__(self, idx):
    # Fetch a single sequence by its index
    sequence_data = self.sequences[idx]

    # Convert the numpy array of features to a PyTorch tensor
    features = torch.tensor(sequence_data['data'], dtype=torch.float32)

    # Get the original length of the sequence before any padding
    original_length = features.shape[0]

    # Convert the label to a PyTorch tensor
    label = torch.tensor(sequence_data['label'], dtype=torch.long)

    sequence_id = sequence_data['seq_id']  # <-- This line is crucial

    return features, original_length, label, sequence_id


class PyTorchModel(nn.Module):
  def __init__(self, num_features, num_classes, hidden_size=64):
    super().__init__()

    self.hidden_size = hidden_size
    self.num_classes = num_classes

    # Bidirectional LSTM (input: num_features, output: 2*hidden_size)
    self.lstm = nn.LSTM(input_size=num_features,
                        hidden_size=self.hidden_size,
                        batch_first=True,
                        bidirectional=True)

    # Bidirectional GRU (input: 2*hidden_size (from Bi-LSTM), output: 2*hidden_size)
    self.gru = nn.GRU(input_size=2 * self.hidden_size, # Input from Bi-LSTM output
                      hidden_size=self.hidden_size,
                      batch_first=True,
                      bidirectional=True)

    # Attention mechanism components
    # Input to attention_dense is the output of Bi-GRU: (batch, seq_len, 2*hidden_size)
    self.attention_dense = nn.Linear(2 * self.hidden_size, 1) # Maps to (batch, seq_len, 1)

    # Output layer
    self.classifier = nn.Linear(2 * self.hidden_size, num_classes) # Input: context vector from attention


  def forward(self, x, lengths):
    # The input 'x' is a padded tensor from the DataLoader.
    # Its shape is: (batch_size, max_sequence_length, num_features)

    # --- 1. Pack the padded input sequence ---
    # This creates a 'PackedSequence' object which only contains the non-padded elements.
    # This is a crucial step for the RNNs to ignore padding.
    # The lengths tensor must be on the CPU for this operation.
    x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

    # --- 2. Pass the packed sequence through the LSTM and GRU layers ---
    # These layers are designed to accept and return PackedSequence objects.
    # The output of LSTM is a PackedSequence, and we just pass it to the GRU.
    lstm_out_packed, _ = self.lstm(x_packed)
    gru_out_packed, _ = self.gru(lstm_out_packed)

    # --- 3. Unpack the GRU output to get a padded tensor back ---
    # The attention mechanism and Linear layer expect a standard tensor, not a PackedSequence.
    # This converts the PackedSequence back to a padded tensor.
    gru_out, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
    # gru_out shape: (batch_size, max_sequence_length_in_batch, 2 * hidden_size)

    # The rest of the code is your attention mechanism, which is already correct for a padded tensor.
    # --- Simple Attention Mechanism ---

    # 1. Calculate raw attention scores for each time step
    attention_scores = torch.tanh(self.attention_dense(gru_out))
    attention_scores = attention_scores.squeeze(-1)

    # 2. Mask padding (sets scores for padded areas to -infinity before softmax)
    # Note: We need to create a mask for the attention mechanism
    max_sequence_length = x.size(1)
    mask = torch.arange(max_sequence_length, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

    # 3. Apply softmax to get attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)

    # 4. Prepare attention_weights for element-wise multiplication
    attention_weights = attention_weights.unsqueeze(-1)
    attention_weights = attention_weights.expand_as(gru_out)

    # 5. Element-wise multiplication: Weigh the GRU output features
    weighted_output = gru_out * attention_weights

    # 6. Sum to get the context vector
    context_vector = torch.sum(weighted_output, dim=1)

    # --- Output Layer ---
    logits = self.classifier(context_vector)

    return logits


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch_features, batch_lengths, batch_labels, _ in dataloader:
        batch_features = batch_features.to(DEVICE)
        batch_lengths = batch_lengths.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_features, batch_lengths)
        loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted_labels = torch.max(logits, 1)
        correct_predictions += (predicted_labels == batch_labels).sum().item()
        total_predictions += batch_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_features, batch_lengths, batch_labels, _ in dataloader:
            batch_features = batch_features.to(DEVICE)
            batch_lengths = batch_lengths.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            logits = model(batch_features, batch_lengths)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item()
            _, predicted_labels = torch.max(logits, 1)
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy



def custom_collate_fn(batch):
    """
    A custom collate function to handle padding and batching of variable-length sequences.
    This is passed to the DataLoader.

    Args:
        batch (list): A list of tuples, each containing (features, length, label)

    Returns:
        tuple: A tuple containing the padded features, sorted lengths, and sorted labels.
    """
    # 1. Unzip the batch: separates the tuples into three lists
    # e.g., batch_features = [features_1, features_2, ...]
    #       batch_lengths = [length_1, length_2, ...]
    #       batch_labels = [label_1, label_2, ...]
    batch_features, batch_lengths, batch_labels, batch_ids = zip(*batch)

    # 2. Pad the sequences to the length of the longest sequence in the batch
    # `pad_sequence` handles this efficiently. `batch_first=True` gives us (B, L, F) shape.
    padded_features = pad_sequence(batch_features, batch_first=True, padding_value=0.0)

    # 3. Convert lengths and labels to tensors
    lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long)
    labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

    ids_tensor = torch.tensor(batch_ids, dtype=torch.long)

    # 4. Sort the padded features, lengths, and labels by length in descending order
    # This is a requirement for torch.nn.utils.rnn.pack_padded_sequence
    lengths_sorted, sorted_indices = lengths_tensor.sort(descending=True)
    padded_features_sorted = padded_features[sorted_indices]
    labels_sorted = labels_tensor[sorted_indices]
    ids_sorted = ids_tensor[sorted_indices] # Sort the IDs too


    return padded_features_sorted, lengths_sorted, labels_sorted, ids_sorted


# Convert DataFrames to PyTorch-friendly lists of sequences ---
# These functions are simplified versions of what we discussed earlier
def df_to_sequences(dataframe):
    sequences = []
    for seq_id, group in dataframe.groupby('sequence_id'):
        sequences.append({
            'data': group[feature_cols].values,
            'label': group['gesture_id'].iloc[0],
            'seq_id': seq_id
        })
    return sequences


if TRAIN:
    # --- 3. Stratified Group K-Fold Cross-Validation Loop ---
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    all_fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    
        # Split the dataframe into train and validation sets using the indices
        train_df = df2_imu.iloc[train_idx]
        val_df = df2_imu.iloc[val_idx]
    
        # --- 4. SCALING: Fit on train data only, then transform both ---
        # Instantiate the scaler for this fold
        scaler = StandardScaler()
    
        # Fit the scaler on the training features only
        scaler.fit(train_df[feature_cols])
    
        # Save the scaler for this fold (e.g., if you want to use the best one later)
        # import joblib
        joblib.dump(scaler, os.path.join(SCALER_SAVE_PATH,'standard_scaler.pkl'))
    
        # Transform both train and validation sets
        train_df[feature_cols] = scaler.transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    
        train_sequences = df_to_sequences(train_df)
        val_sequences = df_to_sequences(val_df)
    
        # --- 6. Create PyTorch DataLoaders ---
        train_dataset = GestureDataset(train_sequences)
        val_dataset = GestureDataset(val_sequences)
    
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    
        print(f"Fold {fold+1} train gesture counts:\n{train_df['gesture'].value_counts()}")
        print(f"Fold {fold+1} val gesture counts:\n{val_df['gesture'].value_counts()}")
    
        # =========================================================================
        # --- MODEL TRAINING AND VALIDATION PART (Step 7) ---
        # =========================================================================
    
        # Instantiate the model, optimizer, and loss function for this fold
        # A fresh model is created for each fold to ensure independent training
        num_features = len(feature_cols)
        num_classes = len(label_map)
    
        model = PyTorchModel(num_features, num_classes, hidden_size=HIDDEN_SIZE).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
    
        best_val_accuracy = 0.0
        for epoch in range(NUM_EPOCHS):
            # Training phase
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    
            # Validation phase
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
    
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
            # Save the best model for this fold based on validation accuracy
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH.format(fold+1))
                print(f"  -> Model saved for fold {fold+1} with improved accuracy: {best_val_accuracy:.4f}")
    
        print(f"Finished training for Fold {fold+1}. Best Validation Accuracy: {best_val_accuracy:.4f}")
    
        ## Need to calculate the competition metrics here
        print("Calculating competition metric for the best model...")
    
        # 1. Load the best model's weights from the disk
        best_model = PyTorchModel(num_features, num_classes,
                                  hidden_size=HIDDEN_SIZE).to(DEVICE)
        best_model.load_state_dict(torch.load(MODEL_SAVE_PATH.format(fold+1)))
        best_model.eval() # Set the model to evaluation mode
    
        # 2. Run inference on the entire validation set and collect sequence_id
        val_sequence_ids = []
        val_true_labels = []
        val_pred_labels = []
        with torch.no_grad():
            # val_loader should now return sequence_id
            for batch_features, batch_lengths, batch_labels, batch_ids in val_loader:
                batch_features = batch_features.to(DEVICE)
                batch_lengths = batch_lengths.to(DEVICE)
    
                logits = best_model(batch_features, batch_lengths)
                _, predicted = torch.max(logits, 1)
    
                val_sequence_ids.extend(batch_ids.cpu().numpy())
                val_true_labels.extend(batch_labels.cpu().numpy())
                val_pred_labels.extend(predicted.cpu().numpy())
    
        # 3. Create a map from integer IDs back to gesture names
        # You need a variable that maps class IDs (integers) to gesture names (strings)
        id_to_gesture_map = {v: k for k, v in label_map.items()}
    
        # 4. Map integer labels to gesture names and create DataFrames
        true_gestures = [id_to_gesture_map[i] for i in val_true_labels]
        pred_gestures = [id_to_gesture_map[i] for i in val_pred_labels]
    
        solution_df = pd.DataFrame({
            'id': val_sequence_ids,
            'gesture': true_gestures
        })
    
        submission_df = pd.DataFrame({
            'id': val_sequence_ids,
            'gesture': pred_gestures
        })
    
        # 5. Instantiate the CompetitionMetric class and call the method
        metric_calculator = CompetitionMetric()
        competition_metric_value = metric_calculator.calculate_hierarchical_f1(
            sol=solution_df,
            sub=submission_df
        )
    
        print(f"Fold {fold+1} Competition Metric: {competition_metric_value:.4f}")
        all_fold_metrics.append(competition_metric_value)
    
    
        ## End of competition metric calculation
    
    print("\nCross-validation training complete.")
    print(f"Competition Metric per fold: {all_fold_metrics}")
    print(f"Final Average Competition Metric: {np.mean(all_fold_metrics):.4f}")



if TRAIN == False:
    MODEL_PATH = '/kaggle/input/artifact-dataaugmented/trained_model_DataAugmented.pth'
    SCALER_PATH = '/kaggle/input/artifact-dataaugmented/standard_scaler.pkl'
    LABEL_MAP_PATH = '/kaggle/input/artifact-dataaugmented/label_map.json'
    MAX_SEQUENCE_LENGTH = 700

    # Initialize global variables to None
    MODEL = None
    SCALER = None
    LABEL_MAP = None
    ID_TO_GESTURE_MAP = None
    NUM_FEATURES = len(FEATURE_COLUMNS)
    NUM_CLASSES = 0 # Will be set after loading label_map
    


if TRAIN == False:
    # Load artifacts
    try:
        # Load label map
        with open(LABEL_MAP_PATH, 'r') as f:
            LABEL_MAP = json.load(f)
        ID_TO_GESTURE_MAP = {v: k for k, v in LABEL_MAP.items()}
        NUM_CLASSES = len(LABEL_MAP)
    
        # Load scaler
        SCALER = joblib.load(SCALER_PATH)
    
        # Load model
        MODEL = PyTorchModel(NUM_FEATURES, NUM_CLASSES, hidden_size=HIDDEN_SIZE)
        # Use map_location='cpu' to ensure it loads correctly regardless of GPU availability
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        MODEL.to(torch.device("cpu")) # Ensure model is on CPU for inference
        MODEL.eval() # Set to evaluation mode for inference (disables dropout, batchnorm)

        print("Inference artifacts (model, scaler, label map) loaded successfully.")

    except Exception as e:
        print(f"Error loading inference artifacts: {e}")
        # In a real competition, you might want to raise an error here
        # or ensure a default behavior. For now, we'll let it be None.


# --- 3. The Predict Function for Kaggle Submission ---
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predicts the gesture for a given sequence of sensor data.

    Args:
        sequence (pl.DataFrame): Polars DataFrame containing sensor data for one gesture.
                                 Expected columns: sequence_id, acc_x, acc_y, etc.
        demographics (pl.DataFrame): Polars DataFrame containing demographic info (not used by this model).

    Returns:
        str: The predicted gesture string (e.g., 'Cheek - pinch skin').
    """
    # Defensive check: Ensure artifacts are loaded. This should ideally happen globally.
    if MODEL is None or SCALER is None or ID_TO_GESTURE_MAP is None:
        print("Error: Model or scaler not loaded. Returning default gesture.")
        return 'Text on phone' # Fallback gesture

    # 1. Convert Polars DataFrame to Pandas DataFrame for easier preprocessing
    sequence_pd = sequence.to_pandas()

    # 2. Preprocess sequence_id (if it's in 'SEQ_XXXXXX' format)
    # This must match how you processed sequence_id during training
    if 'sequence_id' in sequence_pd.columns and isinstance(sequence_pd['sequence_id'].iloc[0], str) and sequence_pd['sequence_id'].iloc[0].startswith('SEQ_'):
        sequence_pd['sequence_id'] = sequence_pd['sequence_id'].str.slice(start=4).astype(int)
    # Ensure sequence_id is int if it was already numeric but not int
    elif 'sequence_id' in sequence_pd.columns:
        sequence_pd['sequence_id'] = sequence_pd['sequence_id'].astype(int)

    # Do feature engineering here
    acc_values = sequence_pd[["acc_x","acc_y","acc_z"]]
    rot_values = sequence_pd[["rot_x","rot_y","rot_z","rot_w"]]
    remove_gravity_from_acc(acc_values,rot_values)

    sequence_pd['linear_acc_x'] = remove_gravity_from_acc(acc_values,rot_values)[:,0]
    sequence_pd['linear_acc_y'] = remove_gravity_from_acc(acc_values,rot_values)[:,1]
    sequence_pd['linear_acc_z'] = remove_gravity_from_acc(acc_values,rot_values)[:,2]

    sequence_pd['linear_acc_mag'] = linear_acc_mag(sequence_pd[['linear_acc_x',   'linear_acc_y','linear_acc_z']])
    
    sequence_pd['rot_angle'] = 2 * np.arccos(sequence_pd['rot_w'].clip(-1, 1))

    sequence_pd['acc_mag_jerk'] = sequence_pd.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    
    sequence_pd['rot_angle_vel'] = sequence_pd.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    
    # 3. Extract features and handle NaNs (ffill, bfill, fillna(0))
    # This must match your training preprocessing exactly
    # Apply imputation within the sequence if sequence_id is reliable for grouping
    if 'sequence_id' in sequence_pd.columns:
        sequence_features = sequence_pd.groupby('sequence_id')[FEATURE_COLUMNS].ffill().bfill().fillna(0)
    else: # If no sequence_id, treat as a single continuous sequence for imputation
        sequence_features = sequence_pd[FEATURE_COLUMNS].ffill().bfill().fillna(0)

    # 4. Apply the loaded StandardScaler
    # The scaler expects a 2D array (num_samples, num_features)
    scaled_features = SCALER.transform(sequence_features)

    # 5. Convert to PyTorch tensor and prepare for model
    # Model expects (batch_size, sequence_length, num_features)
    # For a single sequence, batch_size is 1
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0) # Add batch dimension
    
    # Original length for pack_padded_sequence. For a single sequence, it's its own length.
    original_length = torch.tensor([scaled_features.shape[0]], dtype=torch.long) 

    # 6. Handle MAX_SEQUENCE_LENGTH (truncation if needed)
    # This is crucial if inference sequences can be longer than MAX_SEQUENCE_LENGTH
    if input_tensor.shape[1] > MAX_SEQUENCE_LENGTH:
        input_tensor = input_tensor[:, :MAX_SEQUENCE_LENGTH, :]
        original_length = torch.tensor([MAX_SEQUENCE_LENGTH], dtype=torch.long)
    
    # Move tensor to CPU (Kaggle inference environments often don't have GPUs for submission)
    input_tensor = input_tensor.to(torch.device("cpu"))
    original_length = original_length.to(torch.device("cpu"))

    # 7. Run inference
    with torch.no_grad(): # Disable gradient calculation for inference
        logits = MODEL(input_tensor, original_length)
        _, predicted_class_id = torch.max(logits, 1) # Get the class with the highest logit
    
    # 8. Map predicted ID back to gesture string
    predicted_gesture_string = ID_TO_GESTURE_MAP[predicted_class_id.item()]

    return predicted_gesture_string


import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

