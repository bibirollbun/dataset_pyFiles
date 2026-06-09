import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv", index_col="id")


train.head()


train.isna().sum()


test.isna().sum()


def fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Rhythm_Energy'] = df['RhythmScore'] * df['Energy']
    df['Rhythm_Loudness'] = df['RhythmScore'] * df['AudioLoudness']
    df['Duration_Minutes'] = df['TrackDurationMs'] / 60000  
    df['Duration_Energy_Ratio'] = df['TrackDurationMs'] / (df['Energy'] * 10000 + 1)  
    df['RhythmScore_Squared'] = df['RhythmScore'] ** 2
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Log_Duration'] = np.log1p(df['TrackDurationMs']) 
    df['Acoustic_Instrumental_Ratio'] = df['AcousticQuality'] / (df['InstrumentalScore'] + 0.01) 
    df['Vocal_Energy'] = df['VocalContent'] * df['Energy']
    df['Live_Energy'] = df['LivePerformanceLikelihood'] * df['Energy']
    df['Mood_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Audio_Intensity'] = (df['Energy'] * np.abs(df['AudioLoudness'])) / 10  
    df['Performance_Character'] = (df['LivePerformanceLikelihood'] + df['MoodScore']) / 2
    df['Energy_Loudness_Ratio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 0.01)
    df['Rhythm_Duration_Density'] = df['RhythmScore'] / df['Duration_Minutes']

    return df

train_fe = fe(train)
test_fe = fe(test)


numerical_cols = train_fe.select_dtypes(include=[np.number]).columns
print(f"The numerical value columns are: {numerical_cols.values}")

numerical_cols = [col for col in numerical_cols if col != 'BeatsPerMinute']


scaler = MinMaxScaler()
train_fe[numerical_cols] = scaler.fit_transform(train_fe[numerical_cols])
test_fe[numerical_cols] = scaler.transform(test_fe[numerical_cols])


X = train_fe.drop('BeatsPerMinute', axis=1)
y = train_fe['BeatsPerMinute']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


X_train_tensor = torch.tensor(X_train.values.astype(np.float32)).to(device)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).unsqueeze(1).to(device)

X_val_tensor = torch.tensor(X_test.values.astype(np.float32)).to(device)
y_val_tensor = torch.tensor(y_test.values.astype(np.float32)).unsqueeze(1).to(device)

X_test_tensor = torch.tensor(test_fe.values.astype(np.float32)).to(device)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )


    def forward(self, x):
        return self.net(x)


model = RegressionModel(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)


best_rmse = float("inf")
patience = 20
counter = 0
num_epochs = 200

train_losses = []
val_losses = []
train_rmses = []
val_rmses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)
        total += batch_y.size(0)

    epoch_train_loss = running_loss / total
    epoch_train_rmse = (epoch_train_loss ** 0.5)
    train_losses.append(epoch_train_loss)
    train_rmses.append(epoch_train_rmse)

    model.eval()
    val_running_loss = 0.0
    val_total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).float().view(-1, 1)

            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            val_running_loss += loss.item() * batch_X.size(0)
            val_total += batch_y.size(0)

    epoch_val_loss = val_running_loss / val_total
    epoch_val_rmse = (epoch_val_loss ** 0.5)
    val_losses.append(epoch_val_loss)
    val_rmses.append(epoch_val_rmse)

    scheduler.step(epoch_val_rmse)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1}: "
            f"Train_Loss={epoch_train_loss:.4f}, Val_Loss={epoch_val_loss:.4f}, "
            f"Train_RMSE={epoch_train_rmse:.4f}, Val_RMSE={epoch_val_rmse:.4f}"
        )
        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")

    if epoch_val_rmse < best_rmse:
        best_rmse = epoch_val_rmse
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break


epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 4))
cmap = plt.get_cmap('Paired')

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss (MSE)', color=cmap(0))
plt.plot(epochs, val_losses, label='Val Loss (MSE)', color=cmap(1))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_rmses, label='Train RMSE', color=cmap(2))
plt.plot(epochs, val_rmses, label='Val RMSE', color=cmap(3))
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE per Epoch')
plt.legend()

plt.tight_layout()
plt.show()


model.load_state_dict(best_model_state)


model.eval()
with torch.no_grad():
    test_preds = model(X_val_tensor).squeeze().cpu().numpy()

rmse = np.sqrt(mean_squared_error(y_test, test_preds))
print(f"Test RMSE: {rmse:.4f}")


model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor).squeeze().cpu().numpy()

sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
submission = pd.DataFrame({
    "id": sub['id'],
    "BeatsPerMinute": test_preds
})

submission.to_csv("submission.csv", index=False)

