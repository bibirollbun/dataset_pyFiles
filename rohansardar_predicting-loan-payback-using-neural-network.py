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


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv", index_col="id")


train.head()


train.isna().sum()


test.isna().sum()


categorical_cols = test.select_dtypes(include=['object']).columns
numerical_cols = test.select_dtypes(include=['int64', 'float64']).columns

print(f"The categorical value columns are: {categorical_cols.values}")
print(f"The numerical value columns are: {numerical_cols.values}")


plt.figure(figsize=(6,4))

sns.countplot(data=train, x='loan_paid_back', palette='cubehelix')

plt.title('Distribution of loan_paid_back')
plt.xlabel('loan_paid_back')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='loan_paid_back', data=train, palette='cubehelix')
    plt.title(f"{col} vs loan_paid_back count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


for col in numerical_cols:
    plt.figure(figsize=(4, 6))
    sns.boxplot(x='loan_paid_back', y=col, data=train, palette='mako')
    plt.title(f"Distribution of {col} by loan_paid_back")
    plt.tight_layout()
    plt.show()


for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=train, x=col, hue='loan_paid_back', fill=True, palette='mako')
    plt.title(f"Density of {col} by loan_paid_back")
    plt.tight_layout()
    plt.show()


encoder = LabelEncoder()
for i in categorical_cols:
    train[i] = encoder.fit_transform(train[i])
    test[i] = encoder.transform(test[i])


scaler = MinMaxScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])


X = train.drop('loan_paid_back', axis=1)
y = train['loan_paid_back']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


X_train_tensor = torch.tensor(X_train.values.astype(np.float32)).to(device)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).unsqueeze(1).to(device)

X_val_tensor = torch.tensor(X_test.values.astype(np.float32)).to(device)
y_val_tensor = torch.tensor(y_test.values.astype(np.float32)).unsqueeze(1).to(device)

X_test_tensor = torch.tensor(test.values.astype(np.float32)).to(device)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        
        return out + self.shortcut(x)  

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.layer1 = ResidualBlock(input_dim, 256, dropout)
        self.layer2 = ResidualBlock(256, 128, dropout)
        self.layer3 = ResidualBlock(128, 64, dropout / 2)
        self.fc_out = nn.Linear(64, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc_out(x)
        return x  


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


sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
submission = pd.DataFrame({
    "id": sub['id'],
    "loan_paid_back": test_probs 
})

submission.to_csv("submission.csv", index=False)




