# SYED HUSSAIN RAZA - AI_22036
# AGAH MIR HASAN - AI-22038
# TURKI AHMED - AI-22048


data_path = "/kaggle/input/ieee-fraud-detection/"

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch

# Load and merge data
train_transaction = pd.read_csv(f"{data_path}train_transaction.csv")
train_identity = pd.read_csv(f"{data_path}train_identity.csv")

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# Target + features
y = train['isFraud'].values
train.drop(['isFraud', 'TransactionID'], axis=1, inplace=True)

# Drop cols with >90% missing
missing_perc = train.isnull().mean()
train = train.loc[:, missing_perc < 0.9]

# Categorical / numeric
cat_cols = [col for col in train.columns if train[col].dtype == 'object']
num_cols = [col for col in train.columns if train[col].dtype != 'object']

# Impute missing
for col in num_cols:
    train[col].fillna(train[col].median(), inplace=True)   # FIX: median instead of -999
for col in cat_cols:
    train[col].fillna('missing', inplace=True)

# Label encode categoricals
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    label_encoders[col] = le

# Simple feature engineering
if 'TransactionDT' in train.columns:
    train['TransactionHour'] = (train['TransactionDT'] // 3600) % 24

for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
    if col in train.columns:
        freq = train[col].value_counts(normalize=True)
        train[col + '_freq'] = train[col].map(freq).fillna(0)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(train.values.astype('float32'))   # FIX: float32 to save RAM

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Fraud rate: {y_train.mean():.4f}")



from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # FIX: shape (N,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = FraudDataset(X_train, y_train)
val_ds = FraudDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)  # FIX: smaller batch size
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

class LightweightNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)   # FIX: smaller network
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)  # FIX: no sigmoid, logits only

input_dim = X_train.shape[1]
model = LightweightNN(input_dim).to(device)

# Class weight
pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)



def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

torch.save(model.state_dict(), "best_model.pth")




# Load best model


from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

val_preds, val_true, val_probs = [], [], []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()

        val_probs.extend(probs.cpu().numpy())
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(y_batch.cpu().numpy())

print(f"AUC: {roc_auc_score(val_true, val_probs):.4f}")
print(f"Precision: {precision_score(val_true, val_preds):.4f}")
print(f"Recall: {recall_score(val_true, val_preds):.4f}")
print(f"F1: {f1_score(val_true, val_preds):.4f}")



import shap
import lime
import lime.lime_tabular
import numpy as np

# KernelExplainer with a tiny background set
background_idx = np.random.choice(X_train.shape[0], 50, replace=False)
background = X_train[background_idx]

def shap_predict(x):
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        probs = torch.sigmoid(model(x_tensor)).cpu().numpy().flatten()
        return np.vstack([1 - probs, probs]).T

explainer = shap.KernelExplainer(shap_predict, background)
shap_values = explainer.shap_values(X_val[:20])  # keep small for speed

shap.summary_plot(shap_values, X_val[:20], feature_names=train.columns)




# -----------------------
# LIME
# -----------------------




def predict_fn(x):
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        probs = torch.sigmoid(model(x_tensor)).cpu().numpy().flatten()
        return np.vstack([1 - probs, probs]).T  # [P(not fraud), P(fraud)]

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=train.columns,
    class_names=["Not Fraud", "Fraud"],
    mode="classification"
)

# Pick one fraud case

fraud_idx = np.where(y_val == 1)[0][0]
exp = lime_explainer.explain_instance(
    X_val[fraud_idx],
    predict_fn,
    num_features=10,
    num_samples=1000   # default = 5000, reduce for speed
)

print("\n=== LIME Console Explanation ===")
for feat, weight in exp.as_list():
    print(f"  {feat}: {weight:.4f}")






# Load test data
test_transaction = pd.read_csv(f"{data_path}test_transaction.csv")
test_identity = pd.read_csv(f"{data_path}test_identity.csv")
test = pd.merge(test_transaction, test_identity, on="TransactionID", how="left")
test_ids = test["TransactionID"]

# Keep only training columns
test = test.reindex(columns=train.columns, fill_value=np.nan)

# Impute numeric
for col in num_cols:
    if col in test.columns:
        test[col].fillna(train[col].median(), inplace=True)

# Impute categorical + encode
for col, le in label_encoders.items():
    if col in test.columns:
        test[col] = test[col].fillna("missing")
        test[col] = test[col].map(lambda s: s if s in le.classes_ else "missing")
        if "missing" not in le.classes_:
            le.classes_ = np.append(le.classes_, "missing")
        test[col] = le.transform(test[col].astype(str))

# Feature engineering (same as train)
if "TransactionDT" in test.columns:
    test["TransactionHour"] = (test["TransactionDT"] // 3600) % 24

for col in ["card1", "card2", "addr1", "P_emaildomain"]:
    if col in test.columns:
        freq = train[col].value_counts(normalize=True)
        test[col + "_freq"] = test[col].map(freq).fillna(0)

# Scale
X_test = scaler.transform(test.values.astype("float32"))

# Predict with trained model
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    test_preds = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()

# Save submission
submission = pd.DataFrame({"TransactionID": test_ids, "isFraud": test_preds})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(submission.head())


