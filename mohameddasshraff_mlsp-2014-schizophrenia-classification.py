import os

DATA_PATH = "/kaggle/input"
os.listdir(DATA_PATH)


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)



import os

os.listdir("/kaggle/input/mlsp-2014-mri")



import zipfile
import os

BASE_PATH = "/kaggle/input/mlsp-2014-mri"
WORK_PATH = "/kaggle/working"

# Unzip Train
with zipfile.ZipFile(f"{BASE_PATH}/Train.zip", 'r') as zip_ref:
    zip_ref.extractall(WORK_PATH)

# Unzip Test
with zipfile.ZipFile(f"{BASE_PATH}/Test.zip", 'r') as zip_ref:
    zip_ref.extractall(WORK_PATH)

print("Unzipping done")
os.listdir(WORK_PATH)



DATA_PATH = "/kaggle/working"

train_labels = pd.read_csv(f"{DATA_PATH}/train_labels.csv")
train_fnc = pd.read_csv(f"{DATA_PATH}/train_FNC.csv")
train_sbm = pd.read_csv(f"{DATA_PATH}/train_SBM.csv")

test_fnc = pd.read_csv(f"{DATA_PATH}/test_FNC.csv")
test_sbm = pd.read_csv(f"{DATA_PATH}/test_SBM.csv")

y = train_labels["Class"].values

print(train_fnc.shape, train_sbm.shape, train_labels.shape)



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_fnc = train_fnc.drop(columns=["Id"], errors="ignore")
train_sbm = train_sbm.drop(columns=["Id"], errors="ignore")
test_fnc = test_fnc.drop(columns=["Id"], errors="ignore")
test_sbm = test_sbm.drop(columns=["Id"], errors="ignore")

scaler_fnc = StandardScaler()
scaler_sbm = StandardScaler()

X_fnc = scaler_fnc.fit_transform(train_fnc)
X_sbm = scaler_sbm.fit_transform(train_sbm)

X_fnc_test = scaler_fnc.transform(test_fnc)
X_sbm_test = scaler_sbm.transform(test_sbm)

pca = PCA(n_components=50, random_state=42)
X_fnc = pca.fit_transform(X_fnc)
X_fnc_test = pca.transform(X_fnc_test)

print(X_fnc.shape, X_sbm.shape)



import torch
from torch.utils.data import Dataset, DataLoader

class DatasetMM(Dataset):
    def __init__(self, fnc, sbm, y=None):
        self.fnc = torch.tensor(fnc, dtype=torch.float32)
        self.sbm = torch.tensor(sbm, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.fnc)

    def __getitem__(self, idx):
        if self.y is None:
            return self.fnc[idx], self.sbm[idx]
        return self.fnc[idx], self.sbm[idx], self.y[idx]



import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, fnc_dim, sbm_dim):
        super().__init__()
        self.fnc = nn.Sequential(
            nn.Linear(fnc_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.sbm = nn.Sequential(
            nn.Linear(sbm_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, f, s):
        f = self.fnc(f)
        s = self.sbm(s)
        x = torch.cat([f, s], dim=1)
        return self.fc(x).squeeze()


import torch.optim as optim
import numpy as np

# Dataset + DataLoader
ds = DatasetMM(X_fnc, X_sbm, y)
dl = DataLoader(ds, batch_size=16, shuffle=True)

# Model
model = FusionNet(X_fnc.shape[1], X_sbm.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

# Training
for epoch in range(30):
    model.train()
    losses = []
    for f, s, yb in dl:
        optimizer.zero_grad()
        preds = model(f, s)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}/30 - Loss: {np.mean(losses):.4f}")



# Test Dataset
test_ds = DatasetMM(X_fnc_test, X_sbm_test)
test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

model.eval()
preds = []
with torch.no_grad():
    for f, s in test_dl:
        pred = model(f, s)
        preds.extend(pred.numpy())

# Binary classification: 0 or 1
preds_binary = [1 if p >= 0.5 else 0 for p in preds]



import pandas as pd

submission = pd.DataFrame({
    "Id": range(len(preds_binary)),
    "Class": preds_binary
})

submission.to_csv("submission.csv", index=False)
print("Submission file created!")
submission.head()



import shutil

# copy to /kaggle/working (لو مش موجود هناك)
shutil.move("submission.csv", "/kaggle/working/submission.csv")



# Evaluation on training set
model.eval()
with torch.no_grad():
    f = torch.tensor(X_fnc, dtype=torch.float32)
    s = torch.tensor(X_sbm, dtype=torch.float32)
    preds_train = model(f, s).numpy()

preds_train_binary = [1 if p >= 0.5 else 0 for p in preds_train]

from sklearn.metrics import accuracy_score
acc = accuracy_score(y, preds_train_binary)
print(f"Training Accuracy: {acc*100:.2f}%")



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y, preds_train_binary)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()





