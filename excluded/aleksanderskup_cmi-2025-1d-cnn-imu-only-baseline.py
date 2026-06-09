import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import io
from PIL import Image
import polars as pl

import kaggle_evaluation.cmi_inference_server
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

device = "cuda"


training_data_path = Path("data/CMI/train.csv")
training_data_demo_path = Path("data/CMI/train_demographics.csv")

test_data_path = Path("data/CMI/test.csv")
test_data_demo_path = Path("data/CMI/test_demographics.csv")


train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_demo_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')


train_df.head(60)


class CMIIMUDataset(Dataset):
    def __init__(self, df, transform=None, device='cpu'):
        self.device = device
        self.df = df
        self.transform = transform
        self.features = []  # Initialize features list
        self.scaler = StandardScaler()

        excluded_cols = {
            'gesture', 'sequence_type', 'behavior', 'orientation',  # train-only
            'row_id', 'subject', 'phase',  # metadata
            'sequence_id', 'sequence_counter'  # identifiers
        }
        
        thermal_tof_cols = [col for col in df.columns if col.startswith('thm_') or col.startswith('tof_')]
        excluded_cols.update(thermal_tof_cols)

        # Get feature columns (all columns except excluded ones)
        feature_cols = [col for col in df.columns if col not in excluded_cols]

        # Build sequences
        for seq_id, group in df.groupby('sequence_id'):
            data = group[feature_cols].copy()

            # fill missing values forward and backward
            data = pd.DataFrame(data).ffill().bfill().fillna(0).values

            # Standardize features
            data = self.scaler.fit_transform(data)
            self.features.append(data)  # Convert to numpy array

        # Store both gesture labels and sequence types
        self.labels = df.groupby('sequence_id')['gesture'].first().values
        self.sequence_types = df.groupby('sequence_id')['sequence_type'].first().values

        # Pad sequences to the same length
        self.features = nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.float32, device=self.device) for f in self.features], batch_first=True)

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.labels = torch.tensor(self.labels, dtype=torch.long, device=self.device)

    def print(self):
        print(self.features.shape)
        print(self.labels.shape)
        print(self.features[:3])
        print(self.labels[:3])
        print("\nSequence types sample:", self.sequence_types[:3])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label


dataset = CMIIMUDataset(train_df, device=device)
dataset.print()

# Print unique gestures and their encoded values
print("\nUnique gestures and their encoded values:")
for gesture, encoded in zip(np.unique(train_df['gesture']), torch.unique(dataset.labels)):
    print(f"Gesture: {gesture} -> Encoded: {encoded}")

print("\nNumber of unique gestures:", len(np.unique(train_df['gesture'])))


print("\nUnique sequence types:")
print(train_df['sequence_type'].unique())

# Print example gestures for each sequence type
print("\nExample gestures for each sequence type:")
for seq_type in train_df['sequence_type'].unique():
    gestures = train_df[train_df['sequence_type'] == seq_type]['gesture'].unique()
    print(f"\n{seq_type}:")
    for g in gestures:
        print(f"- {g}")


print("\nChecking for NaN values in features:")
print("NaN values in features:", torch.isnan(dataset.features).any())
print("Number of NaN values:", torch.isnan(dataset.features).sum())

print("\nChecking feature statistics:")
print("Min value:", torch.min(dataset.features))
print("Max value:", torch.max(dataset.features))
print("Mean value:", torch.mean(dataset.features))
print("Std value:", torch.std(dataset.features))


print(dataset.features.shape)
print(dataset.labels.shape)


type(dataset.features), type(dataset.labels)


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size // 4, hidden_size // 4)
        self.relu4 = nn.ReLU()
        self.fcout = nn.Linear(hidden_size // 4, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fcout(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch normalization dla danych wejściowych
        self.batch_norm_input = nn.BatchNorm1d(input_size)
        
        # LSTM z dropout
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Batch normalization po LSTM
        self.batch_norm_hidden = nn.BatchNorm1d(hidden_size)
        
        # Dropout przed warstwą liniową
        self.dropout = nn.Dropout(dropout_rate)
        
        # Warstwa liniowa
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fcout = nn.Linear(hidden_size // 4, num_classes)

    def forward(self, x):

        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_length)
        # Normalizacja danych wejściowych
        x = self.batch_norm_input(x)

        x = x.permute(0, 2, 1)  # (batch_size, seq_length, input_size)
        
        # Forward LSTM
        out, _ = self.lstm(x, None)  # None oznacza, że stany ukryte są inicjalizowane zerami
        
        # Bierzemy ostatni output
        out = out[:, -1, :]
        
        # Normalizacja po LSTM
        out = self.batch_norm_hidden(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Warstwa liniowa
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fcout(out)
        
        return out


class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()

        #Block in
        self.convin = nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=7, padding=1)
        self.bnin = nn.BatchNorm1d(512)
        self.relubin = nn.ReLU()
        self.poolin = nn.MaxPool1d(kernel_size=2)
        self.dropoutin = nn.Dropout(0.2)
        
        #Block 1
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=768, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm1d(768)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        
        #Block 2
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        #Block 3
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=1536, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(1536)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.4)

        #Block 4
        self.conv4 = nn.Conv1d(in_channels=1536, out_channels=2048, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(2048)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AdaptiveAvgPool1d(1)
        self.dropout4 = nn.Dropout(0.4)

        #Dense layers
        self.fc1 = nn.Linear(2048, 2048)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(1024, 512)
        self.relu_fc3 = nn.ReLU()
        self.dropout_fc3 = nn.Dropout(0.3)

        self.fcout = nn.Linear(512, num_classes)


    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)  # Change to (batch_size, input_size, seq_length) for Conv1d

        x = self.convin(x)
        x = self.bnin(x)
        x = self.relubin(x)
        x = self.poolin(x)
        x = self.dropoutin(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)

        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout_fc3(x)

        x = self.fcout(x)

        return x


class CNNLSTMAttention(nn.Module):
    def __init__(self, input_channels, num_classes, lstm_hidden=256, lstm_layers=1, bidirectional=False, attn_hidden=128, dropout=0.3):
        super(CNNLSTMAttention, self).__init__()

        #CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=512, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=512, out_channels=768, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(768),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=768, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=1024, out_channels=1536, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1536),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        #LSTM
        self.lstm = nn.LSTM(
            input_size=1536,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if lstm_layers == 1 else 0.1,
            num_layers=lstm_layers,
            hidden_size=lstm_hidden
        )

        self.dir_mult = 2 if bidirectional else 1
        feat_dim = lstm_hidden * self.dir_mult

        #Additive Attention
        self.attn_fc = nn.Linear(feat_dim, attn_hidden)
        self.attn_v = nn.Linear(attn_hidden, 1, bias=False)

        #MLP
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (N, L, C_in) --> (N, C_in, L)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        # x: (N, C_feat, L') --> (N, L', C_feat)

        outputs, _ = self.lstm(x)   # (N, L', H*dirs)

        # Attention
        attn_scores = torch.tanh(self.attn_fc(outputs))     # (N, L', attn_hidden)
        attn_scores = self.attn_v(attn_scores).squeeze(-1)  # (N, L')
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)  # (N, H*dirs)

        #MLP
        logits = self.head(context) # (N, num_classes)

        return logits


generator1 = torch.Generator().manual_seed(42)
train_data, val_data = random_split(dataset, [0.8, 0.2], generator=generator1)
def get_dataloaders(train_data, val_data, batch_size=1024):
    train_dataloader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,  # Włącz shuffle dla danych treningowych
            num_workers=0  # Użyj wielu workerów do ładowania danych
            #pin_memory=True  # Przyspiesz transfer do GPU
        )
        
    val_dataloader = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False,  # Nie shuffle'uj danych walidacyjnych
            num_workers=0
            #pin_memory=True
        )
    return train_dataloader, val_dataloader


def calculate_training_metrics(predictions, labels):
    accuracy = (predictions == labels).mean()
    precision = precision_score(y_true=labels, y_pred=predictions, average='weighted', zero_division=0)
    recall = recall_score(y_true=labels, y_pred=predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_true=labels, y_pred=predictions, average='weighted', zero_division=0)

    # Add tp, fp, fn, tn calculations for each class
    cm = confusion_matrix(labels, predictions)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    return accuracy, precision, recall, f1, cm, tp, fp, fn, tn, labels, predictions


def train_one_epoch(train_loader, model, loss_fn, optimizer, epoch_idx):
    running_loss = 0.0
    last_loss = 0.0
    quarter = max(1, len(train_loader) // 4) # log every 1/4 of an epoch
    n_in_window = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        if outputs.dim() == 2:
            logits = outputs
        else:
            logits = outputs.permute(0, 2, 1).mean(dim=2)  # Adjust dimensions for CrossEntropyLoss
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        n_in_window += 1

        probabilities = torch.softmax(logits, dim=1)
        _, predicted = torch.max(logits.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.detach().cpu().numpy())

        if ((i + 1) % quarter == 0) or ((i + 1) == len(train_loader)):
            last_loss = running_loss / n_in_window  # średnia w oknie
            pct = int(round(100 * (i + 1) / len(train_loader)))
            print(f"  epoch {epoch_idx+1} {pct}% ({i+1}/{len(train_loader)}) - loss: {last_loss:.6f}")

            # reset okna
            running_loss = 0.
            n_in_window = 0

    # Calculate metrics at the end of epoch
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    metrics = calculate_training_metrics(all_predictions, all_labels)

    return last_loss, metrics



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
            raise ValueError(f"Invalid gesture values in submission: {invalid_types}")

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

        return 0.5 * f1_binary + 0.5 * f1_macro


class EarlyStopping_Accuracy:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_accuracy_max = 0.0
        self.delta = delta
        self.path = path

    def __call__(self, val_accuracy, model):

        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model):
        '''Saves model when validation accuracy increases.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_accuracy_max:.6f} --> {val_accuracy:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_accuracy_max = val_accuracy


def validate(val_loader: DataLoader, model: nn.Module, loss_fn: nn):
    running_loss = 0.0
    n_in_window = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if outputs.dim() == 2:
                logits = outputs
            else:
                logits = outputs.permute(0, 2, 1).mean(dim=2)  # Adjust dimensions for CrossEntropyLoss
            loss = loss_fn(logits, labels)

            running_loss += loss.item()
            n_in_window += 1

            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate standard metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    metrics = calculate_training_metrics(all_predictions, all_labels)
    
    avg_loss = running_loss / n_in_window
    print(f"  Validation - loss: {avg_loss:.6f}, accuracy: {metrics[0]*100:.2f}%, precision: {metrics[1]:.4f}, recall: {metrics[2]:.4f}, f1: {metrics[3]:.4f}")

    return avg_loss, metrics, all_probabilities


def calculate_metrics(predictions, labels):

    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    cm = confusion_matrix(labels, predictions)

    return precision, recall, f1, cm


def write_to_tensorboard(writer, epoch, train_loss, val_loss, val_metrics, train_metrics, learning_rate):
    # val_metrics: accuracy, precision, recall, f1, cm, tp, fp, fn, tn, labels, predictions
    #               0           1           2   3   4   5    6   7   8      9       10
    # train_metrics: accuracy, precision, recall, f1, cm, tp, fp, fn, tn, labels, predictions
    writer.add_scalar('Loss/train', train_loss, epoch + 1)
    writer.add_scalar('Loss/val', val_loss, epoch + 1)
    writer.add_scalars('Loss', {'Training': train_loss, 'Validation': val_loss}, epoch + 1)
    writer.add_scalar('Accuracy/val', val_metrics[0]*100, epoch + 1)
    writer.add_scalar('Precision/val', val_metrics[1], epoch + 1)
    writer.add_scalar('Recall/val', val_metrics[2], epoch + 1)
    writer.add_scalar('F1-score/val', val_metrics[3], epoch + 1)
    writer.add_scalar('Accuracy/train', train_metrics[0]*100, epoch + 1)
    writer.add_scalar('Precision/train', train_metrics[1], epoch + 1)
    writer.add_scalar('Recall/train', train_metrics[2], epoch + 1)
    writer.add_scalar('F1-score/train', train_metrics[3], epoch + 1)
    writer.add_scalar('Learning Rate', learning_rate, epoch + 1)

    writer.flush()


def train_loop(train_data: Subset, val_data: Subset, model: nn.Module, loss_fn: nn, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, name: str, patience=10, epochs=10, batch_size=32, warmup_epochs=0, warmup_lr=1e-5):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter('runs/{}_{}'.format(name, timestamp))

    early_stopping = EarlyStopping_Accuracy(patience=patience, verbose=True, path='runs/{}_{}/best_accuracy_model.pt'.format(name, timestamp))
    epoch_number = 0

    all_metrics = []
    all_vmetrics = []
    all_probabilities = []

    best_vloss = 1_000_000.
    best_vloss_epoch = -1

    train_dataloader, validation_loader = get_dataloaders(train_data, val_data, batch_size=batch_size)

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, metrics = train_one_epoch(train_dataloader, model, loss_fn, optimizer, epoch_number)

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        avg_vloss, vmetrics, probabilities = validate(validation_loader, model, loss_fn)

        # Step the scheduler
        if epoch_number < warmup_epochs:
            # Linear warmup
            warmup_lr = (epoch_number + 1) / warmup_epochs * optimizer.defaults['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"  Warmup LR: {warmup_lr:.6f}")
        else:
            scheduler.step(vmetrics[0])  # Step based on validation accuracy

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        write_to_tensorboard(writer, epoch_number, avg_loss, avg_vloss, vmetrics, metrics, 
                           optimizer.param_groups[0]['lr'])

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_vloss_epoch = epoch_number
            model_path = 'runs/{}_{}/best_loss_model.pt'.format(name, timestamp)
            torch.save(model.state_dict(), model_path)

        all_metrics.append(metrics)
        all_vmetrics.append(vmetrics)
        all_probabilities.append(probabilities)

        # Early stopping
        early_stopping(vmetrics[0], model) # Monitor validation accuracy
        if early_stopping.early_stop:
            print("Early stopping")
            break

        epoch_number += 1

    print("Training complete.")
    print(f"Best validation loss: {best_vloss:.6f} at epoch {best_vloss_epoch + 1}")

    data = {
        "model": model,
        "writer": writer,
        "all_metrics": all_metrics,
        "all_vmetrics": all_vmetrics,
        "all_probabilities": all_probabilities,
        "model_path": 'runs/{}_{}'.format(name, timestamp)
    }
    return data


def evaluate_model(model, val_data, batch_size=128):
    """
    Evaluate model on validation data, calculating both standard and competition metrics
    """
    model.eval()
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if outputs.dim() == 2:
                logits = outputs
            else:
                logits = outputs.permute(0, 2, 1).mean(dim=2)
                
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_true_labels = np.array(all_true_labels)
    
    # Get sequence types for validation samples
    val_indices = val_data.indices
    sequence_types = dataset.sequence_types[val_indices]
    
    # Convert predictions and true labels back to original gesture names
    original_predictions = dataset.label_encoder.inverse_transform(all_predictions)
    original_labels = dataset.label_encoder.inverse_transform(all_true_labels)
    
    # Calculate standard metrics
    accuracy = (all_predictions == all_true_labels).mean()
    precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    
    print("\nStandard Metrics:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate competition metrics using official implementation
    metric = CompetitionMetric()
    val_submission = pd.DataFrame({'gesture': original_predictions})
    val_solution = pd.DataFrame({'gesture': original_labels})
    competition_score = metric.calculate_hierarchical_f1(val_solution, val_submission)
    
    print("\nCompetition Metrics (Official):")
    print(f"Final Score: {competition_score:.4f}")
    
    # Print sequence type statistics
    target_mask = np.isin(original_labels, metric.target_gestures)
    print("\nSequence Statistics:")
    print(f"Target sequences: {target_mask.sum()}")
    print(f"Non-target sequences: {len(target_mask) - target_mask.sum()}")
    
    if target_mask.sum() > 0:
        print("\nTarget gestures present in validation set:")
        target_gestures = np.unique(original_labels[target_mask])
        for gesture in target_gestures:
            print(f"- {gesture}")
    
        # Print confusion matrix for target gestures
        target_indices = np.where(target_mask)[0]
        target_predictions = original_predictions[target_indices]
        target_true_labels = original_labels[target_indices]
        unique_targets = np.unique(target_true_labels)
        
        print("\nConfusion Matrix for Target Gestures:")
        cm = confusion_matrix(target_true_labels, target_predictions, labels=unique_targets)
        print("True labels (rows) vs Predicted labels (columns):")
        print("Labels:", unique_targets)
        print(cm)
    
    return {
        'standard_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'competition_score': competition_score,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'true_labels': all_true_labels,
        'sequence_types': sequence_types
    }


def plot_pr_curve(y_true, probabilities, writer, class_names=None):
    """
    Rysuje krzywe PR (Precision-Recall) dla każdej klasy.
    
    Args:
        y_true: prawdziwe etykiety
        probabilities: macierz prawdopodobieństw dla każdej klasy (wynik softmax)
        writer: obiekt SummaryWriter z TensorBoard
        class_names: opcjonalna lista nazw klas
    """
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    num_classes = probabilities.shape[1]
    
    for i in range(num_classes):
        class_name = class_names[i] if class_names is not None else f"class_{i}"
        
        # Konwertuj etykiety na format binarny (1 dla danej klasy, 0 dla pozostałych)
        y_true_binary = (y_true == i).astype(np.int32)
        
        # Pobierz prawdopodobieństwa dla danej klasy
        class_probs = probabilities[:, i]
        
        # Oblicz precision i recall dla różnych progów
        precision, recall, thresholds = precision_recall_curve(y_true_binary, class_probs)
        
        # Rysuj krzywą PR
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Class {class_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {class_name}')
        plt.grid(True)
        plt.legend()
        
        # Zapisz wykres do bufora i dodaj do TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Konwertuj bufor na tensor
        img = Image.open(buf)
        img_tensor = torch.from_numpy(np.array(img))
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Dodaj wykres do TensorBoard
        writer.add_image(f'PR_Curves/{class_name}', img_tensor, global_step=0)
        
        # Dodaj też surowe dane do interaktywnego wykresu
        writer.add_pr_curve(
            f'PR_Curves_Raw/{class_name}',
            y_true_binary,
            class_probs,
            global_step=0
        )
        
        plt.close()
    
    writer.flush()


model = LSTMModel(input_size=dataset.features.shape[2], hidden_size=512, num_layers=1, num_classes=len(torch.unique(dataset.labels)), dropout_rate=0.4)
model.to(device)
print(model)


model = ComplexNN(input_size=dataset.features.shape[2], hidden_size=256, num_classes=len(torch.unique(dataset.labels)))
model.to(device)
print(model)


model = CNN1D(input_size=dataset.features.shape[2], num_classes=len(torch.unique(dataset.labels)))
model.to(device)
print(model)


model = CNNLSTMAttention(input_channels=dataset.features.shape[2], num_classes=len(torch.unique(dataset.labels)), lstm_layers=2, bidirectional=True, dropout=0.3)
model.to(device)
print(model)


pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {pytorch_total_params:,}")


pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable model parameters: {pytorch_trainable_params:,}")


name = 'CMI_1D_CNN_v1'
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.1, 
    patience=8,
    min_lr=1e-7
)

# Inicjalizacja wag
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)



torch.cuda.empty_cache()


# Uruchomienie treningu
#results = train_loop(
#    model=model,
#    train_data=train_data,
#    val_data=val_data,
#    loss_fn=loss_fn,
#    optimizer=optimizer,
#    patience=15,
#    epochs=150,
#    batch_size=64,  # Zwiększony batch size dla lepszej stabilności
#    scheduler=scheduler,
#    warmup_epochs=15,
#    warmup_lr=1e-4,
#    name=name
#)

# Ewaluacja modelu na zbiorze walidacyjnym
#print("\nEwaluacja końcowa modelu:")
#evaluation = evaluate_model(model, val_data)


#plot_pr_curve(results["all_vmetrics"][-1][9], results["all_probabilities"][-1], results["writer"], class_names=dataset.label_encoder.classes_)


essa = torch.load('/kaggle/input/1d-cnn-lstmattention/pytorch/v1/1/best_accuracy_model.pt')
model.load_state_dict(essa)


print("\nEwaluacja na zbiorze walidacyjnym:")
val_evaluation = evaluate_model(model, val_data)


def predict_sequence(sequence_df: pd.DataFrame, model: nn.Module, device='cuda') -> str:
    """
    Predict gesture for a single sequence
    
    Args:
        sequence_df: DataFrame containing single sequence data
        model: Trained model
        device: Device to run inference on
    
    Returns:
        str: Predicted gesture name
    """
    # Get feature columns (same as in dataset)
    excluded_cols = {
        'gesture', 'sequence_type', 'behavior', 'orientation',  # train-only
        'row_id', 'subject', 'phase',  # metadata
        'sequence_id', 'sequence_counter'  # identifiers
    }
    
    thermal_tof_cols = [col for col in sequence_df.columns if col.startswith('thm_') or col.startswith('tof_')]
    excluded_cols.update(thermal_tof_cols)
    
    feature_cols = [col for col in sequence_df.columns if col not in excluded_cols]
    
    # Prepare features
    data = sequence_df[feature_cols].copy()
    data = data.ffill().bfill().fillna(0).values
    
    # Standardize features using the same scaler logic
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Convert to tensor and add batch dimension
    data = torch.tensor(data, dtype=torch.float32, device=device)
    data = data.unsqueeze(0)  # Add batch dimension
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        if outputs.dim() > 2:
            outputs = outputs.permute(0, 2, 1).mean(dim=2)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        
    # Convert back to gesture name
    predicted_gesture = dataset.label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    
    return predicted_gesture


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Competition prediction function.
    Called with one sequence at a time.
    
    Args:
        sequence: Polars DataFrame with sequence data
        demographics: Polars DataFrame with demographics data
        
    Returns:
        str: Predicted gesture name
    """
    # Convert Polars DataFrame to Pandas
    sequence_pd = sequence.to_pandas()
    
    # Make prediction using our model
    predicted_gesture = predict_sequence(sequence_pd, model)
    
    # Validate if prediction is valid
    metric = CompetitionMetric()
    if predicted_gesture not in metric.all_classes:
        # If prediction is invalid, return a safe default
        return 'Text on phone'  # One of the non-target gestures
        
    return predicted_gesture

# Test the prediction function on a single sequence from validation set
test_sequence_idx = val_data.indices[0]
test_sequence = train_df[train_df['sequence_id'] == train_df['sequence_id'].unique()[test_sequence_idx]]
test_demographics = train_demo_df[train_demo_df['subject'] == test_sequence['subject'].iloc[0]]

# Convert to Polars for testing
test_sequence_pl = pl.from_pandas(test_sequence)
test_demographics_pl = pl.from_pandas(test_demographics)

print("Testing prediction function:")
print(f"True gesture: {test_sequence['gesture'].iloc[0]}")
print(f"Predicted gesture: {predict(test_sequence_pl, test_demographics_pl)}")


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

