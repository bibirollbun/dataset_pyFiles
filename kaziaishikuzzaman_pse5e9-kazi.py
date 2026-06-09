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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


train.columns


def create_features(df):    
    # Interaction between RhythmScore and Energy
    df['Rhythm_Energy_Interaction'] = df['RhythmScore'] * df['Energy']
    
    # Normalize TrackDurationMs to minutes
    df['TrackDuration_Minutes'] = df['TrackDurationMs'] / 60000
    
    # Loudness per unit of Energy (handle divide by zero)
    df['Loudness_per_Energy'] = df['AudioLoudness'] / (df['Energy'] + 1e-6)
    
    # Vocal to Instrumental ratio (handle divide by zero)
    df['Vocal_to_Instrumental_Ratio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-6)
    
    # Mood and Energy product
    df['Mood_Energy_Product'] = df['MoodScore'] * df['Energy']
    
    # LivePerformanceLikelihood to Energy ratio (handle divide by zero)
    df['LivePerformance_to_Energy_Ratio'] = df['LivePerformanceLikelihood'] / (df['Energy'] + 1e-6)
    
    # Log transformation of TrackDurationMs
    df['Log_TrackDuration'] = np.log1p(df['TrackDurationMs'])
    
    # Sum of VocalContent and InstrumentalScore
    df['Vocal_Instrumental_Sum'] = df['VocalContent'] + df['InstrumentalScore']
    
    # Bin AudioLoudness into categories
    loud_bins = [-np.inf, -20, -10, 0, np.inf]
    loud_labels = ['quiet', 'soft', 'moderate', 'loud']
    df['AudioLoudness_Bin'] = pd.cut(df['AudioLoudness'], bins=loud_bins, labels=loud_labels)
    
    return df


from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_new_features(train_df, test_df, cols, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Choose 'minmax' or 'zscore'.")

    # Fit scaler on train, transform train and test
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    return train_df, test_df


from sklearn.preprocessing import LabelEncoder

def encode_audio_loudness_bin(train_df, test_df):
    encoder = LabelEncoder()
    # Fit on train and transform
    train_df['AudioLoudness_Bin'] = encoder.fit_transform(train_df['AudioLoudness_Bin'].astype(str))
    # Transform test using same encoder
    test_df['AudioLoudness_Bin'] = encoder.transform(test_df['AudioLoudness_Bin'].astype(str))
    return train_df, test_df


train = create_features(train)
test = create_features(test)


new_cols = [
    'Rhythm_Energy_Interaction', 'TrackDuration_Minutes', 'Loudness_per_Energy',
    'Vocal_to_Instrumental_Ratio', 'Mood_Energy_Product', 'LivePerformance_to_Energy_Ratio',
    'Log_TrackDuration', 'Vocal_Instrumental_Sum'
]


train, test = normalize_new_features(train, test, new_cols, method='zscore')
train, test = encode_audio_loudness_bin(train, test)


train.columns


test.columns


###Create PyTorch Dataset


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class MusicDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, is_test=False):
        self.df = df
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = None
        
        if not is_test:
            self.y = df[target_col].values.astype(np.float32)
            
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = torch.tensor(self.X[index])
        if self.y is not None:
            y = torch.tensor(self.y[index])
            return x, y
        return x


feature_cols = [col for col in test.columns if col != 'id']
target_col = "BeatsPerMinute"


feature_cols


train_dataset = MusicDataset(train, feature_cols, target_col)
test_dataset = MusicDataset(test, feature_cols, target_col, is_test=True)


data_iter = iter(train_dataset)

data = next(data_iter)
features, labels = data
print(features, labels)


data_iter = iter(test_dataset)

data = next(data_iter)
features = data
print(features)


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Quick check to verify shapes (run this first to confirm fix)
print("Train batch check:")
for batch_x, batch_y in train_loader:
    print(f"Train batch shapes: X {batch_x.shape}, Y {batch_y.shape}")
    break

print("Test batch check:")
for batch_x in test_loader:
    print(f"Test batch shape: X {batch_x.shape}")
    break


###Create a basic ANN using Pytorch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class ANN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(ANN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x


# Step 1: Instantiate model
input_size = len(feature_cols)
model = ANN(input_size).to(device)
print(f"Model input size: {input_size}")

# Step 2: Setup optimizer and loss (Adam for ANN, MSE for regression)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Mean Squared Error, common for continuous targets like scores

# Step 3: Train the model (simple loop; add validation later if needed)
num_epochs = 30  # Adjust based on dataset size/time
model.train()  # Set to training mode


for epoch in range(num_epochs):
    total_loss = 0
    num_batches = len(train_loader)  # For progress tracking
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)  # Squeeze for shape match
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log every 10 batches (adjust to 5 or 1 for more/less verbosity; skip for speed)
        # if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")  # Epoch summary

print("Training complete!")


model.eval()
predictions = []

with torch.no_grad():
    for batch_x in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds = outputs.squeeze().cpu().numpy()
        predictions.extend(preds)

submission = pd.DataFrame({
    'id': test['id'].values,
    target_col: predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission saved!")
print(submission.head())

