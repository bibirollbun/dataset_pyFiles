import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import random

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


TRAIN_PATH = "/kaggle/input/beyond-nti-r-1-c-2/train.csv"
TEST_PATH = "/kaggle/input/beyond-nti-r-1-c-2/test.csv"


df = pd.read_csv(TRAIN_PATH)


target = 'Cover_Type'
no_info_cols = [col for col in df.columns if df[col].nunique() == 1]
print(no_info_cols)
num_cols = [col for col in df.columns if df[col].nunique() > 7]
cat_cols = [col for col in df.columns if col not in num_cols]

print(df[target].value_counts().sort_values())


cat_cols.remove(target)


def process_data(data, train=True, scaler=None):
  data_mod = data.copy()
  ids = data_mod["Id"]
  data_mod.drop("Id", axis=1, inplace=True)
  data_mod.drop(no_info_cols, axis=1, inplace=True)

  if train:
    data_mod.drop(data_mod[data_mod[target] == 5].index, axis=0, inplace=True)
    X = data_mod.drop(target, axis=1)
    y = data_mod[target]

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y))

  else:
    X = data_mod
    del(data_mod)
    y_encoded = None
    label_encoder = None

  num_cols_new = [col for col in num_cols if col in X.columns]
  cat_cols_new = [col for col in cat_cols if col in X.columns]

  X_num = X[num_cols_new]
  X_cat = X[cat_cols_new]

  if scaler is None:
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols_new, index=X_num.index)
  else:
    X_num_scaled = pd.DataFrame(scaler.transform(X_num), columns=num_cols_new, index=X_num.index)

  X_mod = pd.concat([X_num_scaled, X_cat], axis=1)

  return X_mod, y_encoded, label_encoder, scaler, ids


X, y_encoded, label_encoder, scaler, _ = process_data(df)
del(df)


class ForestDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

print("Training features shape:", X_train.shape)
print("Validation features shape:", X_val.shape)
print("Training target shape:", y_train.shape)
print("Validation target shape:", y_val.shape)


num_features = X_train.shape[1]
num_classes = y_train.nunique()

print("Number of features:", num_features)
print("Number of classes:", num_classes)


train_dataset = ForestDataset(X_train, y_train)
val_dataset = ForestDataset(X_val, y_val)

batch_size = 1024

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Number of training batches:", len(train_dataloader))
print("Number of validation batches:", len(val_dataloader))


!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0


print(y_train.value_counts())


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline


n_class_1 = int(y_train.value_counts()[1] * 0.06)
n_class_0 = int(y_train.value_counts()[0] * 0.06)

sampling_strategy_under = {
    1: n_class_1,
    0: n_class_0
}


"""n_class_1 = int(y_train.value_counts()[1] * 0.02)
n_class_0 = int(y_train.value_counts()[0] * 0.02)
n_class_2 = int(y_train.value_counts()[2] * 0.3)

sampling_strategy_under = {
    1: n_class_1,
    0: n_class_0,
    2: n_class_2
}"""


# Initialize the undersampler and oversampler
undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42)
oversampler = ADASYN(random_state=42)

pipeline = Pipeline(steps=[
    ('under', undersampler),
    ('over', oversampler)
])

X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

print(y_resampled.value_counts())


mask = (y_resampled == 0) | (y_resampled == 1)

# Step 2: Apply the mask to both y_resampled and X_resampled
y_resampled_filtered = y_resampled[mask]
X_resampled_filtered = X_resampled[mask]

# Step 3: Concatenate the filtered resampled data with the original training data
X_new_train = pd.DataFrame(np.concatenate((X_train, X_resampled_filtered), axis=0), columns=X_train.columns)
y_new_train = pd.Series(np.concatenate((y_train, y_resampled_filtered), axis=0))





df_og = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
df_og.head()
df_og = df_og[df_og[target] != 5]

X_og, _, _, _, _ = process_data(df_og.drop(target, axis=1), train=False, scaler=scaler)
y_og = pd.Series(label_encoder.transform(df_og[target]))

X_new_train = pd.DataFrame(np.concatenate((X_new_train, X_og), axis=0), columns=X_new_train.columns)
y_new_train = pd.Series(np.concatenate((y_new_train, y_og), axis=0))


train_dataset = ForestDataset(X_new_train, y_new_train)
val_dataset = ForestDataset(X_val, y_val)

batch_size = 1024

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Number of training batches:", len(train_dataloader))
print("Number of validation batches:", len(val_dataloader))


class MultiClassClassifier(nn.Module):

  def __init__(self, num_features, num_classes):
    super().__init__()

    self.num_units = [512, 256, 128, 64]

    self.layer_stack = nn.Sequential(

        nn.Linear(num_features, self.num_units[0], bias=False), #bias is redundant when using BatchNorm
        nn.BatchNorm1d(self.num_units[0]),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(self.num_units[0], self.num_units[1], bias=False),
        nn.BatchNorm1d(self.num_units[1]),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(self.num_units[1], self.num_units[2], bias=False),
        nn.BatchNorm1d(self.num_units[2]),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(self.num_units[2], self.num_units[3], bias=False),
        nn.BatchNorm1d(self.num_units[3]),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(self.num_units[3], num_classes) #output raw logits
    )

  def forward(self, x):
    return self.layer_stack(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_1 = MultiClassClassifier(num_features, num_classes).to(device)
display(model_1)


# HyperParameters
learning_rate = 0.001
epochs = 30
patience = 3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",   # we want to minimize val_loss
    factor=0.5,   # new_lr = old_lr * 0.5
    patience=1,   
    verbose=True  
)

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float("inf")
epochs_no_improve = 0

# Training Loop
for epoch in range(epochs):
    # Training phase
    model_1.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False, mininterval=2.0)

    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model_1(inputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)
        _, pred_labels = predictions.max(1)
        correct_train += (pred_labels == labels).sum().item()
        total_train += labels.size(0)

        # Update progress bar with live info (last batch loss)
        train_bar.set_postfix(loss=loss.item())

    epoch_train_loss = running_train_loss / len(train_dataloader.dataset)
    epoch_train_accuracy = correct_train / total_train

    # Validation phase
    model_1.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model_1(inputs)
            loss = criterion(predictions, labels)

            running_val_loss += loss.item() * inputs.size(0)
            _, pred_labels = predictions.max(1)
            correct_val += (pred_labels == labels).sum().item()
            total_val += labels.size(0)

    epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
    epoch_val_accuracy = correct_val / total_val

    # Print and store metrics
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"  Training Loss: {epoch_train_loss:.4f} | Validation Loss: {epoch_val_loss:.4f}")
    print(f"  Training Accuracy: {epoch_train_accuracy:.4f} | Validation Accuracy: {epoch_val_accuracy:.4f}")

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_accuracy)
    val_accuracies.append(epoch_val_accuracy)

    scheduler.step(epoch_val_loss)
    
    # Early Stopping + Checkpoint
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model_1.state_dict(), "model_2.pth")  # save best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

print("\nTraining finished!")
print("Best model saved as 'model_2.pth'")


# Reload best model
model_1.load_state_dict(torch.load("model_2.pth"))
model_1.eval()


all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_dataloader:   # or test_dataloader if you have one
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model_1(inputs)
        _, pred_labels = predictions.max(1)

        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, normalize="true")  # normalized confusion matrix

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=range(len(cm)), yticklabels=range(len(cm)))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.show()

del(all_preds)
del(all_labels)


df_test = pd.read_csv(TEST_PATH)
df_test.head()


X_test, _, _, _, test_ids = process_data(df_test, train=False, scaler=scaler)
del(df_test)


model_1.eval()
with torch.no_grad():
    predictions = model_1(torch.tensor(X_test.values, dtype=torch.float32).to(device))
    _, pred_labels = predictions.max(1)


test_labels = pred_labels.cpu().numpy()
test_labels = label_encoder.inverse_transform(test_labels)
pd.DataFrame({"Id": test_ids, "Cover_Type": test_labels}).to_csv("/kaggle/working/submission_1.6.csv", index=False)




