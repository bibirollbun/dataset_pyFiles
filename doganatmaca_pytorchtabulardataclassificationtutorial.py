import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset


data_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


data_train.info()


data_train.head()


X = data_train.drop(['id','diagnosed_diabetes'],axis=1)
y = data_train['diagnosed_diabetes']


X["pulse_pressure"] = X["systolic_bp"] - X["diastolic_bp"]
X["bmi_age"] = X["bmi"] * X["age"]
X["waist_bmi"] = X["waist_to_hip_ratio"] * X["bmi"]
X["activity_per_age"] = X["physical_activity_minutes_per_week"] / (X["age"] + 1)
X["chol_ratio"] = X["ldl_cholesterol"] / (X["hdl_cholesterol"] + 1)



X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.15,stratify=y,random_state=42)
x_train = pd.get_dummies(X_train,drop_first=True)
x_test = pd.get_dummies(X_test,drop_first=True)



print(f"x_train shape = {x_train.shape}, x_test shape = {x_test.shape}")


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


x_train_tensor = torch.tensor(x_train_scaled, dtype = torch.float32)
y_train_tensor = torch.tensor(Y_train.values, dtype = torch.float32)

x_test_tensor = torch.tensor(x_test_scaled, dtype = torch.float32)
y_test_tensor = torch.tensor(Y_test.values, dtype = torch.float32)


print(type(x_train_tensor))
print(x_train_tensor.shape)


train_dataset = TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = TensorDataset(x_test_tensor,y_test_tensor)


train_loader = DataLoader(
    train_dataset,
    batch_size = 128,
    shuffle = True
)
test_loader = DataLoader(
    test_dataset,
    batch_size = 128,
    shuffle = False
)


class DiabetesModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


input_dim = x_train_tensor.shape[1]
model = DiabetesModel(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)


def train_one_epoch(model,loader):
    total_loss=0

    for x_batch, y_batch in tqdm(loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch).view(-1)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(loader)


def validate(model, loader):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu())
            all_targets.append(y_batch)

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    return roc_auc_score(all_targets, all_probs)



EPOCHS = 5
best_roc = 0

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    test_roc = validate(model, test_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test ROC-AUC: {test_roc:.4f}"
    )

    if test_roc > best_roc:
        best_roc = test_roc
        torch.save(model.state_dict(), "best_model.pt")


