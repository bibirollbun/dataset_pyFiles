import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col="id")


train.head()


train.isna().sum()


test.isna().sum()


for df in [train, test]:
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_squared'] = df['duration'] ** 2
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_sqrt'] = np.sqrt(df['duration'])


categorical_cols = test.select_dtypes(include=['object']).columns
numerical_cols = test.select_dtypes(include=['int64', 'float64']).columns

print(f"The categorical value columns are: {categorical_cols.values}")
print(f"The numerical value columns are: {numerical_cols.values}")


y_counts = train['y'].value_counts()
plt.figure(figsize=(4, 4))

cmap = plt.get_cmap('flare')
colors = cmap(np.linspace(0, 1, len(y_counts)))

plt.pie(
    y_counts.values,
    labels=y_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    counterclock=False,
    textprops={'color': 'white'}
)
plt.title('Distribution of y')
plt.tight_layout()
plt.show()


n_plots = len(categorical_cols)
cols_per_row = math.ceil(n_plots / 3)

plt.figure(figsize=(5 * cols_per_row, 10))

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, cols_per_row, i)
    sns.countplot(x=col, hue='y', data=train, palette='mako')
    plt.title(f"{col} vs y count")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


n_plots = len(numerical_cols)
cols_per_row = math.ceil(n_plots / 3)

plt.figure(figsize=(5 * cols_per_row, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, cols_per_row, i)
    sns.histplot(x=col, hue='y', data=train, fill=True, palette='mako', bins=30)
    plt.title(f"{col} vs y count")

plt.tight_layout()
plt.show()


encoder = LabelEncoder()
for i in categorical_cols:
    train[i] = encoder.fit_transform(train[i])
    test[i] = encoder.transform(test[i])


scaler = MinMaxScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])


X = train.drop('y', axis=1)
y = train['y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


X_train_tensor = torch.tensor(X_train.values.astype(np.float32)).to(device)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).unsqueeze(1).to(device)

X_val_tensor = torch.tensor(X_test.values.astype(np.float32)).to(device)
y_val_tensor = torch.tensor(y_test.values.astype(np.float32)).unsqueeze(1).to(device)

X_test_tensor = torch.tensor(test.values.astype(np.float32)).to(device)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = BinaryClassifier(input_dim=X_train.shape[1]).to(device)
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
pos_weight_value = num_neg / num_pos
pos_weight = torch.tensor([pos_weight_value]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)


best_auc = 0.0
patience = 20
counter = 0
num_epochs = 200
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
roc_aucs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    epoch_train_loss = running_loss / total
    epoch_train_accuracy = correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            val_running_loss += loss.item() * batch_X.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.extend(probs.squeeze().cpu().numpy())
            all_labels.extend(batch_y.squeeze().cpu().numpy())

            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)

    epoch_val_loss = val_running_loss / val_total
    epoch_val_accuracy = val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)
    
    roc_auc = roc_auc_score(all_labels, all_preds)
    roc_aucs.append(roc_auc)
    scheduler.step(roc_auc)
    if ((epoch+1)%5 == 0):
        print(f"Epoch {epoch+1}: Train_Loss={epoch_train_loss:.4f}, Val_Loss={epoch_val_loss:.4f}, "
          f"Train_Acc={epoch_train_accuracy:.4f}, Val_Acc={epoch_val_accuracy:.4f}, ROC_AUC={roc_auc:.4f}")
        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")

    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break


epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(15, 4))
cmap = plt.get_cmap('Paired')

plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss', color=cmap(0))
plt.plot(epochs, val_losses, label='Val Loss', color=cmap(1))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, label='Train Acc', color=cmap(2))
plt.plot(epochs, val_accuracies, label='Val Acc', color=cmap(3))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, roc_aucs, label='Val ROC AUC', color=cmap(4))
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.title('ROC AUC per Epoch')
plt.legend()

plt.tight_layout()
plt.show()


model.load_state_dict(best_model_state)


model.eval()
with torch.no_grad():
    test_logits = model(X_val_tensor)
    test_probs = torch.sigmoid(test_logits).squeeze().cpu().numpy()
print(f"ROC AUC Score: {roc_auc_score(y_test, test_probs):.4f}")


model.eval()
with torch.no_grad():
    test_logits = model(X_test_tensor)
    test_probs = torch.sigmoid(test_logits).squeeze().cpu().numpy()


sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission = pd.DataFrame({
    "id": sub['id'],
    "y": test_probs 
})

submission.to_csv("submission.csv", index=False)

