# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

! pip install mne
! pip install torch_xla

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import pprint
from matplotlib.lines import Line2D



markers = ['S', 'K', 'REM', 'Son', 'Soff', 'A', 'MS', ]
marker_colors = {
    'S': 'r',
    'K': 'g',
    'REM': 'b',
    'Son': 'm',
    'Soff': 'c',
    'A': 'y',
    'MS': 'k',
}

def load_eeg(data_dir:Path, subject: str) -> mne.io.Raw:
    fname = f"{subject}_eeg_raw.mat"
    fpath = data_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} does not exist")
    
    mat = loadmat(fpath)

    pprint.pprint(mat.keys(), depth=2)
    
    eeg_data = mat["EEG"][0, 0]['data']   
    
    info = mne.create_info(ch_names=['EEG1'], sfreq=250, ch_types=['eeg'])
    raw = mne.io.RawArray(eeg_data, info)
    print(raw.info)

    return raw

def load_labels(data_dir:Path, subject: str) -> pd.DataFrame:
    fname = f"{subject}_labels.csv"
    fpath = data_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} does not exist")
    
    df = pd.read_csv(fpath)

    # for each marker M, theres M0 and M1. Change marker names to just M for both
    df['Marker'] = df['Marker'].str.replace('0', '').str.replace('1', '')

    # # keep only rows in which Marker has the '1' suffix
    # df = df[df['Marker'].str.endswith('1')]

    # # remove the '1' suffix from the Marker column
    # df['Marker'] = df['Marker'].str.replace('1', '')

    return df


def get_event_epochs(raw: mne.io.Raw, events_labels: pd.DataFrame, event:str, tmin:int=0, tmax:int=30) -> mne.Epochs:
    df = events_labels
    event_times = df[df['Marker'] == event]["Timestamp_samples"].values
    event_epochs = df[df['Marker'] == event]["Epoch"].values


    # Create an events array for MNE
    events = np.column_stack([event_times, np.zeros(len(event_times), dtype=int), event_epochs])
    try:
        epochs = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    except ValueError as e:
        print(f"Error creating epochs for event {event}: {e}")
        return None

    return epochs



def sample2min(sample: int, sfreq: int) -> float:
    return sample / sfreq / 60



# Define stage names for legend labels
stage_names = {
    'S': 'Sleep stage',
    'K': 'K-complex',
    'REM': 'Rapid Eye Movement sleep',
    'Son': 'Sleep onset',
    'Soff': 'Sleep offset',
    'A': 'Arousal',
    'MS': 'Microstate'
}


def plot_eeg(raw: mne.io.Raw, subject: str, t0: float, t1: float, events_labels: pd.DataFrame=None):
    time = np.arange(raw._data.shape[1]) / raw.info['sfreq'] / 60

    f, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time, raw._data[0], alpha=0.5)

    if events_labels is not None:
        for marker in markers:
            events_samples = events_labels[events_labels['Marker'] == marker]["Timestamp_samples"].values
            events_time = sample2min(events_samples, raw.info['sfreq'])
            ax.vlines(events_time, -400, 400, alpha=0.5, color=marker_colors[marker])
        
        # Create legend handles with the full stage names
        unique_markers = events_labels["Marker"].unique()
        legend_handles = [
            Line2D([0], [0], color=marker_colors[marker], lw=4, label=stage_names[marker])
            for marker in markers if marker in unique_markers
        ]
        # Place the legend below the graph, centered and taking the full width.
        ax.legend(handles=legend_handles, title="Event Markers", 
                  loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=len(legend_handles))

    ax.set(
        xlabel='Time (min)',
        ylabel='EEG Amplitude',
        ylim=[-400, 400],
        xlim=[t0, t1],
        title=f'EEG data for subject {subject} between {t0:.2f} and {t1:.2f} minutes'
    )
    
    # Adjust layout to provide space for the legend at the bottom
    f.tight_layout(rect=[0, 0.1, 1, 1])
    return f, ax



INPUT_DATA_DIR_PATH = Path("/kaggle/input/bci-i-idun-eeg-analysis-challenge")
OUTPUT_DATA_DIR_PATH=Path("/kaggle/working/")
SUBJECT = "S001"
SUBJECTS_TRAIN =["S001","S002","S003"]


eeg = load_eeg(INPUT_DATA_DIR_PATH,SUBJECT)


labels = load_labels(INPUT_DATA_DIR_PATH, SUBJECT)

labels.head()


epoch_eeg_data = []
epoch_labels = []

for marker in markers:
    epochs = get_event_epochs(eeg, labels, marker)
    if epochs is None:
        print(f"!!! --- No epochs found for marker {marker}")
        continue

    _data = epochs.get_data()[:, 0, :]
    epoch_eeg_data.append(_data)
    epoch_labels.append(np.full(len(_data), marker))

assert len(epoch_eeg_data) == len(epoch_labels)


X = np.concatenate(epoch_eeg_data, axis=0)
y = np.concatenate(epoch_labels, axis=0)

print(X.shape, y.shape)

np.save(OUTPUT_DATA_DIR_PATH / f"{SUBJECT}_X.npy", X)
np.save(OUTPUT_DATA_DIR_PATH / f"{SUBJECT}_y.npy", y)


# Creating the npy files and storing
for subject in SUBJECTS_TRAIN:
    eeg = load_eeg(INPUT_DATA_DIR_PATH,subject)

    labels = load_labels(INPUT_DATA_DIR_PATH, subject)

    epoch_eeg_data = []
    epoch_labels = []
    
    for marker in markers:
        epochs = get_event_epochs(eeg, labels, marker)
        if epochs is None:
            print(f"!!! --- No epochs found for marker {marker}")
            continue
    
        _data = epochs.get_data()[:, 0, :]
        epoch_eeg_data.append(_data)
        epoch_labels.append(np.full(len(_data), marker))
    
    assert len(epoch_eeg_data) == len(epoch_labels)

    X = np.concatenate(epoch_eeg_data, axis=0)
    y = np.concatenate(epoch_labels, axis=0)
    
    print(X.shape, y.shape)
    
    np.save(OUTPUT_DATA_DIR_PATH / f"{subject}_X.npy", X)
    np.save(OUTPUT_DATA_DIR_PATH / f"{subject}_y.npy", y)

    
    
    
    


INPUT_DATA_DIR_PATH = Path("/kaggle/input/bci-i-idun-eeg-analysis-challenge")
OUTPUT_DATA_DIR_PATH=Path("/kaggle/working/")
SUBJECT = "S001"


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import scipy
import torch

# choose a subject [0-3 only, subject 4 is in the test set!]
subject = SUBJECT


# this function plots the EEG data. You can pass a start/end time in minutes to zoom in on a particular time window
plot = plot_eeg(eeg, SUBJECT, 40, 45)[0]


labels = load_labels(INPUT_DATA_DIR_PATH, SUBJECT)
labels = labels[labels.Marker.isin(markers)]
labels.head(10)


print("\nCounts of each marker:")
print(labels.Marker.value_counts())


f, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels.Marker.value_counts().index, labels.Marker.value_counts().values)
ax.set(title="Marker counts", xlabel="Marker", ylabel="Count")
plt.show()


plot = plot_eeg(eeg, subject, 0, 50, labels)[0]


epochs = get_event_epochs(eeg, labels, 'REM')
epochs.plot_image()





X = np.load(INPUT_DATA_DIR_PATH / f"{SUBJECT}_X.npy") # If we concat data for all subjects we will create a much better model
Y = np.load(INPUT_DATA_DIR_PATH / f"{SUBJECT}_y.npy")

print(X.shape, Y.shape)


Y[:5]


idxs = np.where(Y == 'REM')[0]
rem_X = X[idxs]

rem_average = rem_X.mean(axis=0)

f, ax = plt.subplots(figsize=(10, 5))
ax.plot(rem_average)
ax.set(title="Average REM signal", ylim=[-50, 50])
plt.show()


counts = {m: len(np.where(Y == m)[0]) for m in markers}
percents = {m: round(counts[m]/len(Y) * 100, 2) for m in markers}
summary = pd.DataFrame({'Count': counts, 'Percent': percents})
summary





from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import balanced_accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns


def print_class_proportions(Y, label:str):
    counts = {m: len(np.where(Y == m)[0]) for m in markers}
    percents = {m: round(counts[m]/len(Y) * 100, 2) for m in markers}
    summary = pd.DataFrame({'Count': counts, 'Percent': percents})
    # sort by count
    summary = summary.sort_values(by='Count', ascending=False)
    print(f"\n{label} dataset:")
    print(summary)


# counts for the whole dataset
print_class_proportions(Y, "For Whole Subject 1 ")


# splitting the data without stratification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print_class_proportions(Y_train, "S1 Training set")
print_class_proportions(Y_test, "S1 Test set")


# the classess are not balanced, so we need to stratify the split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
print_class_proportions(Y_train, "Training set")
print_class_proportions(Y_test, "Test set")


# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def plot_confusion_matrix(conf_mat, markers, title):
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, cmap='Blues', xticklabels=markers, yticklabels=markers, ax=ax)

    # highlight the diagonal elements
    for i in range(conf_mat.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=1))

    ax.set(
           title=title,
           xlabel='Predicted',
           ylabel='True',
           aspect='equal'
           )
    plt.show()


def fit_and_eval_sk_classifier(clf, X_train, Y_train, X_test, Y_test):
    clf.fit(X_train, Y_train)

    Y_pred_train = clf.predict(X_train)
    Y_pred_test = clf.predict(X_test)

    if set(Y_pred_train) != set(Y_train):
        print(f"Predicted classes {set(Y_pred_train)} do not match true classes {set(Y_train)}")
    if set(Y_pred_test) != set(Y_test):
        print(f"Predicted classes {set(Y_pred_test)} do not match true classes {set(Y_test)}")

    train_acc = balanced_accuracy_score(Y_train, Y_pred_train)
    test_acc = balanced_accuracy_score(Y_test, Y_pred_test)
    print(f"Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    train_f1 = f1_score(Y_train, Y_pred_train, average='weighted')
    test_f1 = f1_score(Y_test, Y_pred_test, average='weighted')
    print(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
    
    conf_mat = confusion_matrix(Y_test, Y_pred_test, normalize='true')
    title=f"{clf.__class__.__name__} - Train: {train_f1:.4f}, Test: {test_f1:.4f}"
    plot_confusion_matrix(conf_mat, markers, title)


# fit multi-class logistic regression
clf = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000, class_weight=None)
fit_and_eval_sk_classifier(clf, X_train, Y_train, X_test, Y_test)


print_class_proportions(Y, "Whole dataset")


# fit multi-class logistic regression with class weights
clf = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000, class_weight="balanced")    
fit_and_eval_sk_classifier(clf, X_train, Y_train, X_test, Y_test)


from sklearn.svm import SVC

svc_cls = SVC(kernel='rbf', random_state=42, class_weight="balanced")  # class_weight "balanced"
fit_and_eval_sk_classifier(svc_cls, X_train, Y_train, X_test, Y_test)


from sklearn.svm import SVC

svc_cls = SVC(kernel='rbf', random_state=42)  # class_weight None
fit_and_eval_sk_classifier(svc_cls, X_train, Y_train, X_test, Y_test)


weights = {
    'S' : 1,
    'A': 1,
    'K': 1,
    'MS': 1,
    'REM': 1,
    'Son': 6,
    'Soff': 6,
}

svc_cls = SVC(kernel='rbf', random_state=42, class_weight=weights)  # pass  your own weights dictionary
fit_and_eval_sk_classifier(svc_cls, X_train, Y_train, X_test, Y_test)


from sklearn.ensemble import RandomForestClassifier

rf_cls = RandomForestClassifier(n_estimators=5, random_state=42, class_weight="balanced")
fit_and_eval_sk_classifier(rf_cls, X_train, Y_train, X_test, Y_test)


#  define a time convolutional feed forward neural network with pytoimport torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_length, num_classes=7):
        super(TimeSeriesCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of flattened features
        self.flatten_size = self._get_flatten_size(input_length)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _get_flatten_size(self, input_length):
        # Helper function to calculate flattened size
        x = torch.randn(1, 1, input_length)
        x = self.conv_layers(x)
        return x.shape[1] * x.shape[2]
    
    def forward(self, x):
        # x shape: (batch_size, n_samples)
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, n_samples)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda', lr=0.001, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights) 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # weight_decay is L2 regularization
        
    model = model.to(device)
    best_val_loss = np.inf
    best_model = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
        
        val_acc = 100. * correct / total
        
        if epoch == 0 or (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
            print('--------------------')

    return best_model


from torch.utils.data import Dataset

class TimeSeriesAugmentation:
    def __init__(self, noise_level=0.05, shift_range=10):
        self.noise_level = noise_level
        self.shift_range = shift_range
    
    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise
    
    def time_shift(self, x):
        # Generate random shift value
        shift = torch.randint(-self.shift_range, self.shift_range, (1,)).item()
        # Roll the tensor along the time dimension
        return torch.roll(x, shifts=shift, dims=0)
    
    def scale(self, x):
        scale_factor = torch.rand(1) * 0.4 + 0.8  # Random scale between 0.8 and 1.2
        return x * scale_factor

class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = X
        self.y = y
        self.augment = augment
        self.aug = TimeSeriesAugmentation()
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()  # Create a copy to avoid modifying original data
        if self.augment:
            if torch.rand(1) < 0.5:
                x = self.aug.add_noise(x)
            if torch.rand(1) < 0.5:
                x = self.aug.time_shift(x)
            if torch.rand(1) < 0.5:
                x = self.aug.scale(x)
        return x, self.y[idx]
    
    def __len__(self):
        return len(self.X)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# convert Y from text labels to integer labels
_markers = list(set(Y))
_Y_train = [_markers.index(m) for m in Y_train]
_Y_test = [_markers.index(m) for m in Y_test]

# Convert to PyTorch tensors
_X_train = torch.FloatTensor(X_train)
_Y_train = torch.LongTensor(_Y_train)
_X_test = torch.FloatTensor(X_test)
_Y_test = torch.LongTensor(_Y_test)

# Create data loaders
train_dataset = AugmentedDataset(_X_train, _Y_train)
val_dataset = AugmentedDataset(_X_test, _Y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device


# Initialize model
input_length = X_train.shape[1]  # number of time samples
model = TimeSeriesCNN(input_length=input_length, num_classes=len(set(Y)))

model = model.to(device)

# Train model
best_model = train_model(model, train_loader, val_loader, num_epochs=30, device=device, lr=1e-4)


# load best model
model.load_state_dict(best_model)

# compute metric
Y_pred = model(_X_test.to(device)).argmax(1).cpu().numpy()
score = balanced_accuracy_score(_Y_test, Y_pred)
f1 = f1_score(_Y_test, Y_pred, average='weighted')

# plot confusion matrix
conf_mat = confusion_matrix(_Y_test, Y_pred, normalize='true')
plot_confusion_matrix(conf_mat, markers, f"tCNN - Accuracy: {score:.4f}, F1: {f1:.4f}")


# Calculate class weights based on frequency
def compute_class_weights(y_train):
    class_counts = torch.bincount(y_train)
    total = len(y_train)
    weights = total / (len(class_counts) * class_counts.float())
    return weights


weights = compute_class_weights(_Y_train).to(device)
balanced_model = train_model(model, train_loader, val_loader, num_epochs=30, device=device, lr=1e-4, class_weights=weights)

# load best model
model.load_state_dict(best_model)

# compute metric
Y_pred = model(_X_test.to(device)).argmax(1).cpu().numpy()
score = balanced_accuracy_score(_Y_test, Y_pred)
f1 = f1_score(_Y_test, Y_pred, average='weighted')

# plot confusion matrix
conf_mat = confusion_matrix(_Y_test, Y_pred, normalize='true')
plot_confusion_matrix(conf_mat, markers, f"tCNN - Accuracy: {score:.4f}, F1: {f1:.4f}")


#imports



# CONCAT 3 FILES
# List of subject IDs (adjust these IDs as needed)
subjects = ["S001", "S002", "S003"]

# Initialize empty lists to store data for each subject
X_list = []
Y_list = []

# Loop over each subject and load their data
for subject in subjects:
    # /kaggle/working/S001_X.npy
    print()
    X_subject = np.load(OUTPUT_DATA_DIR_PATH / f"{subject}_X.npy")
    Y_subject = np.load(OUTPUT_DATA_DIR_PATH / f"{subject}_y.npy")
    X_list.append(X_subject)
    Y_list.append(Y_subject)





# Concatenate the data from all subjects along the first axis
X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)

print(X.shape, Y.shape)


def print_class_proportions(Y, label:str):
    counts = {m: len(np.where(Y == m)[0]) for m in markers}
    percents = {m: round(counts[m]/len(Y) * 100, 2) for m in markers}
    summary = pd.DataFrame({'Count': counts, 'Percent': percents})
    # sort by count
    summary = summary.sort_values(by='Count', ascending=False)
    print(f"\n{label} dataset:")
    print(summary)


# counts for the whole dataset
print_class_proportions(Y, "For Whole ")


# splitting the data without stratification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print_class_proportions(Y_train, "Full Training set")
print_class_proportions(Y_test, "Full Test set")


# TPU
# import torch
# import torch_xla
# import torch_xla.core.xla_model as xm

# # Get the TPU device
# device = xm.xla_device()

# # Check if connected
# print(f"Using device: {device}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# convert Y from text labels to integer labels
_markers = list(set(Y))
_Y_train = [_markers.index(m) for m in Y_train]
_Y_test = [_markers.index(m) for m in Y_test]

# Convert to PyTorch tensors
_X_train = torch.FloatTensor(X_train)
_Y_train = torch.LongTensor(_Y_train)
_X_test = torch.FloatTensor(X_test)
_Y_test = torch.LongTensor(_Y_test)

# Create data loaders
train_dataset = AugmentedDataset(_X_train, _Y_train)
val_dataset = AugmentedDataset(_X_test, _Y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device


# Initialize model
input_length = X_train.shape[1]  # number of time samples
model = TimeSeriesCNN(input_length=input_length, num_classes=len(set(Y)))

model = model.to(device)

# Train model
best_model = train_model(model, train_loader, val_loader, num_epochs=30, device=device, lr=1e-4)


# Calculate class weights based on frequency
def compute_class_weights(y_train):
    class_counts = torch.bincount(y_train)
    total = len(y_train)
    weights = total / (len(class_counts) * class_counts.float())
    return weights


weights = compute_class_weights(_Y_train).to(device)
balanced_model = train_model(model, train_loader, val_loader, num_epochs=100, device=device, lr=1e-4, class_weights=weights)

# load best model
model.load_state_dict(best_model)

# compute metric
Y_pred = model(_X_test.to(device)).argmax(1).cpu().numpy()
score = balanced_accuracy_score(_Y_test, Y_pred)
f1 = f1_score(_Y_test, Y_pred, average='weighted')

# plot confusion matrix
conf_mat = confusion_matrix(_Y_test, Y_pred, normalize='true')
plot_confusion_matrix(conf_mat, markers, f"tCNN - Accuracy: {score:.4f}, F1: {f1:.4f}")


subject = "S004"
X_test = np.load(INPUT_DATA_DIR_PATH / f"{subject}_X.npy")

Y_pred = model(torch.FloatTensor(X_test).to(device)).argmax(1).cpu().numpy()
print(Y_pred[:5])


df.shape


df


_markers


df["Marker"].value_counts()


df["Marker"] = df["Marker"].apply(lambda x: _markers[x - 1])  # Adjust for 0-based indexing


df


IDs = np.arange(len(Y_pred))

df = pd.DataFrame({"ID":IDs, "Marker": Y_pred})
df.head()
df.to_csv("example_submission.csv", index=False)


df




