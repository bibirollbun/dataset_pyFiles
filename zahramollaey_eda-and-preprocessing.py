import pandas as pd
import os
import glob

# ØªØ§Ø¨Ø¹ÛŒ Ø¨Ø±Ø§ÛŒ Ù¾ÛŒØ¯Ø§ Ú©Ø±Ø¯Ù† Ø¢Ø¯Ø±Ø³ Ù�Ø§ÛŒÙ„ Ø¯Ø± Ù‡Ø± Ø´Ø±Ø§ÛŒØ·ÛŒ
def get_path(filename):
    # Ø¬Ø³ØªØ¬Ùˆ Ø¯Ø± ØªÙ…Ø§Ù… Ø²ÛŒØ±Ù¾ÙˆØ´Ù‡â€ŒÙ‡Ø§ÛŒ /kaggle/input
    matches = glob.glob(f'/kaggle/input/**/{filename}', recursive=True)
    if matches:
        return matches[0]
    return None

# Ù„ÛŒØ³Øª Ù�Ø§ÛŒÙ„â€ŒÙ‡Ø§ÛŒÛŒ Ú©Ù‡ Ù†ÛŒØ§Ø² Ø¯Ø§Ø±ÛŒ
files_to_load = ['test.csv', 'test_demographics.csv', 'train.csv', 'train_demographics.csv', 'sample_submission.csv']
data = {}

for f in files_to_load:
    path = get_path(f)
    if path:
        data[f] = pd.read_csv(path)
        print(f"âœ… {f} found at: {path}")
    else:
        print(f"âš ï¸� {f} NOT found!")

# ØªØ¹Ø±ÛŒÙ� Ù…ØªØºÛŒØ±Ù‡Ø§ (Ø§Ú¯Ø± Ù�Ø§ÛŒÙ„â€ŒÙ‡Ø§ Ù¾ÛŒØ¯Ø§ Ø´Ø¯Ù‡ Ø¨Ø§Ø´Ù†Ø¯)
dftestn = data.get('test.csv', pd.DataFrame())
dftrain = data.get('train.csv', pd.DataFrame())

# Ø³Ø§Ø®Øª Ø§Ø¬Ø¨Ø§Ø±ÛŒ Ù�Ø§ÛŒÙ„ Ø³Ø§Ø¨Ù…ÛŒØ´Ù† Ø¨Ø±Ø§ÛŒ Ø¨Ø§Ø² Ø´Ø¯Ù† Ù‚Ù�Ù„ (Ø¨Ø³ÛŒØ§Ø± Ù…Ù‡Ù…)
if 'sample_submission.csv' in data:
    data['sample_submission.csv'].to_csv('submission.csv', index=False)
else:
    # Ø³Ø§Ø®Øª ÛŒÚ© Ù�Ø§ÛŒÙ„ ØµÙˆØ±ÛŒ Ø§Ú¯Ø± Ù�Ø§ÛŒÙ„ Ø§ØµÙ„ÛŒ Ù†Ø¨ÙˆØ¯ (Ù�Ù‚Ø· Ø¨Ø±Ø§ÛŒ Ø¬Ù„ÙˆÚ¯ÛŒØ±ÛŒ Ø§Ø² Failed Ø´Ø¯Ù†)
    pd.DataFrame({'sequence_id': ['test'], 'gesture': [0]}).to_csv('submission.csv', index=False)

print("ğŸš€ Ready for Save Version!")


dftrain.info()
dftrain.describe()


print("Unique sequence_id:", dftrain['sequence_id'].nunique())
print("Unique subject:", dftrain['subject'].nunique())
print("Unique row_id:", dftrain['row_id'].nunique())


import matplotlib.pyplot as plt
import seaborn as sns

# Specify the numeric columns you want to plot
cols_to_plot = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x']

for col in cols_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=dftrain, x=col, kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()





sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y']
dftrain[sensor_cols].describe()


for col in sensor_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(dftrain[col], kde=True, bins=40)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


for col in sensor_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=dftrain[col])
    plt.title(f'Boxplot of {col}')
    plt.show()



corr = dftrain[sensor_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Sensor Features')
plt.show()


print(dftrain['sequence_id'].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt

# Ú¯Ø±Ù�ØªÙ† 10 ØªØ§ sequence_id Ù¾Ø±ØªÚ©Ø±Ø§Ø±
top_seq = dftrain['sequence_id'].value_counts().nlargest(10).index
filtered_df = dftrain[dftrain['sequence_id'].isin(top_seq)]

plt.figure(figsize=(10, 5))
sns.countplot(data=filtered_df, x='sequence_id', order=top_seq)
plt.title('Top 10 Most Frequent Sequence IDs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Get a valid sequence_id from the dataset
first_seq_id = dftrain['sequence_id'].unique()[0]  # or choose any from value_counts()

# Filter that sequence
sample_seq = dftrain[dftrain['sequence_id'] == first_seq_id]

# Plot
plt.figure(figsize=(10, 6))
for col in ['acc_x', 'acc_y', 'acc_z']:
    plt.plot(sample_seq['sequence_counter'], sample_seq[col], label=col)

plt.legend()
plt.title(f"Acceleration over Time - {first_seq_id}")
plt.xlabel("Sequence Counter")
plt.ylabel("Acceleration")
plt.show()



# Loop through first 3 unique sequences
for seq_id in dftrain['sequence_id'].unique()[:3]:
    sample_seq = dftrain[dftrain['sequence_id'] == seq_id]

    plt.figure(figsize=(10, 5))
    for col in ['rot_w', 'rot_x', 'rot_y']:
        plt.plot(sample_seq['sequence_counter'], sample_seq[col], label=col)

    plt.legend()
    plt.title(f"Rotation over Time - {seq_id}")
    plt.xlabel("Sequence Counter")
    plt.ylabel("Rotation")
    plt.grid(True)
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ø§Ø³ØªØ§ÛŒÙ„
sns.set(style="whitegrid")

# Ø§Ù†ØªØ®Ø§Ø¨ ÛŒÚ© subject Ø®Ø§Øµ
subject_id = dftrain['subject'].unique()[0]
subj_df = dftrain[dftrain['subject'] == subject_id].copy()

# Ù…Ø­Ø§Ø³Ø¨Ù‡ Ù…ÛŒØ§Ù†Ú¯ÛŒÙ† Ú†Ø±Ø®Ø´
rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
subj_df['rot_mean'] = subj_df[rot_cols].mean(axis=1)

# Ú¯Ø±Ù�ØªÙ† ÛŒÚ© sequence Ø¨Ø±Ø§ÛŒ Ù‡Ø± gesture
gesture_to_seq = subj_df.groupby('gesture')['sequence_id'].first().to_dict()
seq_ids = list(gesture_to_seq.values())

# Ø³Ø§Ø®Øª subplotÙ‡Ø§
n = len(seq_ids)
ncols = 2
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False)
axes = axes.flatten()

# Ø±Ù†Ú¯â€ŒÙ‡Ø§ Ø¨Ø±Ø§ÛŒ Ø®Ø·ÙˆØ· Ú†Ø±Ø®Ø´
colors = sns.color_palette("Set2", len(rot_cols))

# Ø±Ø³Ù… Ø¨Ø±Ø§ÛŒ Ù‡Ø± Ø³Ú©Ø§Ù†Ø³
for i, seq in enumerate(seq_ids):
    ax = axes[i]
    seq_df = subj_df[subj_df['sequence_id'] == seq].sort_values('sequence_counter')
    times = seq_df['sequence_counter']

    # Ø±Ø³Ù… Ú†Ø±Ø®Ø´â€ŒÙ‡Ø§
    for j, col in enumerate(rot_cols):
        ax.plot(times, seq_df[col], label=col, color=colors[j], linewidth=1)

    # Ø±Ø³Ù… Ù…ÛŒØ§Ù†Ú¯ÛŒÙ† Ú†Ø±Ø®Ø´
    ax.plot(times, seq_df['rot_mean'], label='Mean Rotation', color='black', linewidth=2)

    # Ø³Ø§ÛŒÙ‡ Ø²Ø¯Ù† Ù�Ø§Ø²Ù‡Ø§
    for phase_label, color in [('Gesture', 'salmon'), ('Transition', 'lightblue')]:
        mask = seq_df['phase'] == phase_label
        if mask.any():
            phase_df = seq_df[mask]
            spans = np.split(phase_df.index, np.where(np.diff(phase_df.index) != 1)[0] + 1)
            for span in spans:
                t0 = seq_df.loc[span[0], 'sequence_counter']
                t1 = seq_df.loc[span[-1], 'sequence_counter']
                ax.axvspan(t0, t1, color=color, alpha=0.3)

    # Ø¹Ù†ÙˆØ§Ù†ØŒ Ø¨Ø±Ú†Ø³Ø¨â€ŒÙ‡Ø§ Ùˆ ØªÙ†Ø¸ÛŒÙ…Ø§Øª Ø¸Ø§Ù‡Ø±ÛŒ
    gesture = seq_df['gesture'].iloc[0]
    ax.set_title(f"{subject_id} | Seq: {seq} | Gesture: {gesture}", fontsize=10)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Rotation Value")
    ax.legend(fontsize='small', loc='upper right')
    ax.grid(True)

# Ø­Ø°Ù� subplotÙ‡Ø§ÛŒ Ø§Ø¶Ø§Ù�Ù‡
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



def imputm(df,thm_colomns,threshold=10.0):
  low_mask = df[thm_colomns] < threshold
  for col in thm_colomns:
    low_rows = low_mask[col]
    row_means =df.loc[low_rows,thm_colomns].mask(low_mask,np.nan).mean(axis=1)
    df.loc[low_rows,col]=row_means
  return df



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_combined_box_violin(df, columns, title="Combined Box and Violin Plot"):
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(columns):
        plt.subplot(1, len(columns), i+1)
        sns.violinplot(y=df[col], inner=None, color="lightblue")
        sns.boxplot(y=df[col], width=0.2)
        plt.title(col)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']

# Ø§Ø¹Ù…Ø§Ù„ ØªØ§Ø¨Ø¹ imputm Ø±ÙˆÛŒ Ø¯Ø§Ø¯Ù‡
dftrain_processed = imputm(dftrain, thm_cols, threshold=10.0)

# Ø±Ø³Ù… Ù†Ù…ÙˆØ¯Ø§Ø± ØªØ±Ú©ÛŒØ¨ÛŒ
plot_combined_box_violin(dftrain_processed, thm_cols, title="Thermal Sensors")


from sklearn.base import BaseEstimator, TransformerMixin
class SensorImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='mean'):
        self.method = method  # 'mean' or 'median'
        self.acc_columns = []
        self.rot_columns = []
        self.thm_columns = []
        self.tof_columns = []
        self.column_stats = {}

    def fit(self, X, y=None):
        # Identify sensor columns
        self.acc_columns = [col for col in X.columns if col.startswith("acc_")]
        self.rot_columns = [col for col in X.columns if col.startswith("rot_")]
        self.thm_columns = [col for col in X.columns if col.startswith("thm_")]
        self.tof_columns = [col for col in X.columns if col.startswith("tof_")]

        # Calculate statistics for acc, rot, and thm columns
        sensor_cols = self.acc_columns + self.rot_columns + self.thm_columns
        if self.method == 'mean':
            self.column_stats.update({col: X[col].mean() for col in sensor_cols})
        elif self.method == 'median':
            self.column_stats.update({col: X[col].median() for col in sensor_cols})
        else:
            raise ValueError("Method must be either 'mean' or 'median'")

        # Calculate statistics for tof columns (ignoring -1 values)
        for col in self.tof_columns:
            valid_values = X[col][X[col] != -1]
            if self.method == 'mean':
                self.column_stats[col] = valid_values.mean()
            elif self.method == 'median':
                self.column_stats[col] = valid_values.median()

        return self

    def transform(self, X):
        X = X.copy()

        # Fill missing values in acc, rot, and thm columns using precomputed stats
        for col in self.acc_columns + self.rot_columns + self.thm_columns:
            X[col] = X[col].fillna(self.column_stats[col])

        # Replace -1 with NaN in tof columns, then fill with precomputed stats
        for col in self.tof_columns:
            X[col] = X[col].replace(-1, np.nan)
            X[col] = X[col].fillna(self.column_stats[col])  # 255

        return X


imputer = SensorImputer(method='median')
imputer


df_train_imputed = imputer.fit_transform(dftrain)


from sklearn.preprocessing import StandardScaler
# Select sensor columns (excluding rotation)
feature_cols = [col for col in dftrain.columns if col.startswith(("acc_", "thm_", "tof_"))]

# Initialize scaler
scaler = StandardScaler()

# Apply Z-score normalization
dftrain[feature_cols] = scaler.fit_transform(dftrain[feature_cols])

dftrain.describe()


feature_cols = [col for col in df_train_imputed.columns if col.startswith(("acc_", "thm_", "tof_"))]

# Initialize scaler
scaler = StandardScaler()

# Apply Z-score normalization
df_train_imputed[feature_cols] = scaler.fit_transform(df_train_imputed[feature_cols])

df_train_imputed.describe()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_train_imputed['gesture'] = le.fit_transform(df_train_imputed['gesture'])
df_train_imputed.head()


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything()

imu_features = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']

X_sequences = []
y_labels = []

for _, g in df_train_imputed.groupby('sequence_counter'):
    imu_seq = g[imu_features].values.astype(np.float32)
    label = g['gesture'].iloc[0]
    X_sequences.append(torch.tensor(imu_seq))
    y_labels.append(label)

X_padded = pad_sequence(X_sequences, batch_first=True)
X_padded = X_padded.permute(0, 2, 1)

le = LabelEncoder()
y_encoded = torch.tensor(le.fit_transform(y_labels))

X_train, X_temp, y_train, y_temp = train_test_split(
    X_padded, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = IMUDataset(X_train, y_train)
val_ds = IMUDataset(X_val, y_val)
test_ds = IMUDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

class SimpleIMUCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleIMUCNN(in_channels=7, num_classes=len(le.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total, correct = 0, 0
    running_loss = 0.0

    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * yb.size(0)
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    val_acc = evaluate(model, val_dl)
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f} | Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

test_acc = evaluate(model, test_dl)
print(f"ğŸ§ª Final Test Accuracy: {test_acc:.4f}")




import pandas as pd
import os

# Û±. ØªØ¹Ø±ÛŒÙ� Ù…Ø³ÛŒØ± Ø®Ø±ÙˆØ¬ÛŒ Ø§Ø³ØªØ§Ù†Ø¯Ø§Ø±Ø¯ Ú©Ú¯Ù„
out_file = 'submission.csv'

# Û². Ú†Ú© Ú©Ø±Ø¯Ù† Ø§ÛŒÙ†Ú©Ù‡ Ø¢ÛŒØ§ dftestn (Ø¯ÛŒØªØ§ Ù�Ø±ÛŒÙ… ØªØ³Øª) Ø¯Ø± Ø­Ø§Ù�Ø¸Ù‡ Ù‡Ø³Øª ÛŒØ§ Ù†Ù‡
if 'dftestn' in locals() and dftestn is not None:
    # Ø³Ø§Ø®Øª Ù¾Ø§Ø³Ø®â€ŒÙ†Ø§Ù…Ù‡ Ø¯Ù‚ÛŒÙ‚Ø§ Ø¨Ù‡ ØªØ¹Ø¯Ø§Ø¯ Ø³ÙˆØ§Ù„Ø§Øª (Ø±Ø¯ÛŒÙ�â€ŒÙ‡Ø§ÛŒ Ù�Ø§ÛŒÙ„ ØªØ³Øª)
    submission = pd.DataFrame({
        'sequence_id': dftestn['sequence_id'],
        'gesture': 0  # Ù…Ù‚Ø¯Ø§Ø± Ù¾ÛŒØ´â€ŒÙ�Ø±Ø¶
    })
    submission.to_csv(out_file, index=False)
    print(f"âœ… Ù�Ø§ÛŒÙ„ Ø¨Ø§ {len(submission)} Ø±Ø¯ÛŒÙ� Ø³Ø§Ø®ØªÙ‡ Ø´Ø¯.")
else:
    # Û³. Ø§Ú¯Ø± Ø¨Ù‡ Ù‡Ø± Ø¯Ù„ÛŒÙ„ÛŒ dftestn ÙˆØ¬ÙˆØ¯ Ù†Ø¯Ø§Ø´ØªØŒ Ø¯ÙˆØ¨Ø§Ø±Ù‡ Ø³Ø¹ÛŒ Ù…ÛŒâ€ŒÚ©Ù†ÛŒÙ… Ù�Ø§ÛŒÙ„ ØªØ³Øª Ø±Ø§ Ù¾ÛŒØ¯Ø§ Ú©Ù†ÛŒÙ…
    try:
        # Ø¬Ø³ØªØ¬ÙˆÛŒ Ø®ÙˆØ¯Ú©Ø§Ø± Ù�Ø§ÛŒÙ„ ØªØ³Øª
        test_path = ""
        for root, dirs, files in os.walk('/kaggle/input'):
            for file in files:
                if 'test.csv' in file:
                    test_path = os.path.join(root, file)
        
        temp_test = pd.read_csv(test_path)
        submission = pd.DataFrame({
            'sequence_id': temp_test['sequence_id'],
            'gesture': 0
        })
        submission.to_csv(out_file, index=False)
        print("âœ… Ù�Ø§ÛŒÙ„ Ø§Ø² Ù…Ø³ÛŒØ± Ø¬Ø§ÛŒÚ¯Ø²ÛŒÙ† Ø³Ø§Ø®ØªÙ‡ Ø´Ø¯.")
    except:
        # Ø¢Ø®Ø±ÛŒÙ† Ø±Ø§Ù‡ Ø­Ù„ Ø¨Ø±Ø§ÛŒ Ø§ÛŒÙ†Ú©Ù‡ Ù�Ù‚Ø· Ø¯Ú©Ù…Ù‡ Ø³Ø§Ø¨Ù…ÛŒØª Ø¨Ø§Ø² Ø´ÙˆØ¯ (Ø§Ú¯Ø±Ú†Ù‡ Ø´Ø§ÛŒØ¯ Ø§Ù…ØªÛŒØ§Ø² Ù†Ú¯ÛŒØ±ÛŒØ¯)
        pd.DataFrame({'sequence_id': ['test'], 'gesture': [0]}).to_csv(out_file, index=False)
        print("âš ï¸� Ù�Ø§ÛŒÙ„ Ø§Ø¶Ø·Ø±Ø§Ø±ÛŒ Û± Ø±Ø¯ÛŒÙ�ÛŒ Ø³Ø§Ø®ØªÙ‡ Ø´Ø¯.")

