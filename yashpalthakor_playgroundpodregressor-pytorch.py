import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# EDA (very basic)
train_df.head()
train_df.info()
train_df.describe()


numeric_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
categorical_features = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


train_df[numeric_features] = train_df[numeric_features].fillna(train_df[numeric_features].median())
test_df[numeric_features] = test_df[numeric_features].fillna(test_df[numeric_features].median())


for col in categorical_features:
    train_df[col] = train_df[col].fillna('missing')
    test_df[col] = test_df[col].fillna('missing')


from sklearn.preprocessing import LabelEncoder

for col in categorical_features:
    le = LabelEncoder()
    full_data = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(full_data)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))


features = numeric_features + categorical_features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_df[features].values)
test_features = scaler.transform(test_df[features].values)


# TARGET = "Listening_Time_minutes"
# ID = "id"

# features = [col for col in train_df.columns if col not in [TARGET, ID]]

# # Handle missing values
# train_df.fillna(train_df.mean(), inplace=True)
# test_df.fillna(test_df.mean(), inplace=True)

# # Scaling
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_df[features])
# test_features = scaler.transform(test_df[features])

# train_targets = train_df[TARGET].values.astype(np.float32)


class PodcastDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


TARGET = "Listening_Time_minutes"
ID = "id"
features = [col for col in train_df.columns if col not in [TARGET, ID]]

# Assuming you handled missing data and encoded categoricals as discussed
train_targets = train_df[TARGET].values.astype(np.float32)
train_features = train_df[features].values.astype(np.float32)
test_features = test_df[features].values.astype(np.float32)



# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(train_features, train_targets, test_size=0.2, random_state=42)

train_dataset = PodcastDataset(X_train, y_train)
val_dataset = PodcastDataset(X_val, y_val)
test_dataset = PodcastDataset(test_features)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class PodcastRegressor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = PodcastRegressor(len(features)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50


train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0  # sum of losses over all training samples
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)  # sum loss weighted by batch size

    train_loss /= len(train_dataset)  # average train loss per sample
    train_losses.append(train_loss)   # append once per epoch

    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            
            val_preds = model(X_val_batch)
            vloss = criterion(val_preds, y_val_batch)
            
            val_loss += vloss.item() * X_val_batch.size(0)

    val_loss /= len(val_dataset)  # average val loss per sample
    val_losses.append(val_loss)   # append once per epoch

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()


model.eval()
test_preds = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        test_preds.extend(preds.cpu().numpy())

# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file created!")


