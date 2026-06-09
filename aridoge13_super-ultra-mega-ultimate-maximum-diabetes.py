import warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# LOAD DATA
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

TARGET = "diagnosed_diabetes"

y = train[TARGET].astype(int).values
train = train.drop(columns=[TARGET])


# BASIC FEATURE SPLIT
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()


# AUTOENCODER DEFINITION (SMALL + STABLE)
class AEDataset(Dataset):
    def __init__(self, X):
        self.x = torch.tensor(X, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.x[i]

class AutoEncoder(nn.Module):
    def __init__(self, d_in, d_latent=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_latent)
        )
        self.dec = nn.Sequential(
            nn.Linear(d_latent, 64), nn.ReLU(), nn.Linear(64, d_in)
        )
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


# CV LOOP
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train))

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
    print(f"\n--- Fold {fold} ---")

    X_tr, X_va = train.iloc[tr_idx].copy(), train.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    
    # CATEGORICAL ENCODING (FIT ON TRAIN ONLY)
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X_tr[c] = le.fit_transform(X_tr[c].astype(str))
        X_va[c] = X_va[c].astype(str).map(lambda x: x if x in le.classes_ else "<UNK>")
        if "<UNK>" not in le.classes_:
            le.classes_ = np.append(le.classes_, "<UNK>")
        X_va[c] = le.transform(X_va[c])
        encoders[c] = le

    
    # NUMERIC IMPUTATION + SCALING (TRAIN ONLY)
    
    scaler = StandardScaler()
    X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols].fillna(X_tr[num_cols].median()))
    X_va[num_cols] = scaler.transform(X_va[num_cols].fillna(X_tr[num_cols].median()))

   
    # UNSUPERVISED LEARNING (TRAIN FOLD ONLY)
    

    # AUTOENCODER
    ae_input_tr = X_tr[num_cols + cat_cols].values.astype(np.float32)
    ae_input_va = X_va[num_cols + cat_cols].values.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = AutoEncoder(ae_input_tr.shape[1]).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loader = DataLoader(AEDataset(ae_input_tr), batch_size=2048, shuffle=True)

    ae.train()
    for _ in range(5):
        for xb, _ in loader:
            xb = xb.to(device)
            opt.zero_grad()
            out, _ = ae(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()

    ae.eval()
    with torch.no_grad():
        _, z_tr = ae(torch.tensor(ae_input_tr).to(device))
        _, z_va = ae(torch.tensor(ae_input_va).to(device))

    z_tr = z_tr.cpu().numpy()
    z_va = z_va.cpu().numpy()

    # PCA
    pca = PCA(n_components=3, random_state=42)
    pca_tr = pca.fit_transform(z_tr)
    pca_va = pca.transform(z_va)

    # KMEANS 
    km = KMeans(n_clusters=8, random_state=42)
    cl_tr = km.fit_predict(pca_tr)
    cl_va = km.predict(pca_va)

    
    # FINAL SUPERVISED TABLE
    for i in range(z_tr.shape[1]):
        X_tr[f"ae_{i}"] = z_tr[:, i]
        X_va[f"ae_{i}"] = z_va[:, i]

    for i in range(pca_tr.shape[1]):
        X_tr[f"pca_{i}"] = pca_tr[:, i]
        X_va[f"pca_{i}"] = pca_va[:, i]

    X_tr["cluster"] = cl_tr
    X_va["cluster"] = cl_va

    
    # SUPERVISED MODEL
    # Single model only
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_tr, y_tr)
    oof[va_idx] = model.predict_proba(X_va)[:, 1]

    fold_auc = roc_auc_score(y_va, oof[va_idx])
    print(f"Fold AUC: {fold_auc:.5f}")

    del ae, model, X_tr, X_va
    gc.collect()


# FINAL CV SCORE
cv_auc = roc_auc_score(y, oof)
print(f"\nOOF AUC: {cv_auc:.5f}")


# FULL TRAIN PREPARATION (NO CV ANYMORE)

X_full = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
X_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

y_full = X_full["diagnosed_diabetes"].astype(int).values
X_full = X_full.drop(columns=["diagnosed_diabetes"])

num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_full.select_dtypes(include=["object"]).columns.tolist()



# CATEGORICAL ENCODING
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_full[c] = le.fit_transform(X_full[c].astype(str))
    X_test[c] = X_test[c].astype(str).map(lambda x: x if x in le.classes_ else "<UNK>")
    if "<UNK>" not in le.classes_:
        le.classes_ = np.append(le.classes_, "<UNK>")
    X_test[c] = le.transform(X_test[c])
    label_encoders[c] = le

# NUMERIC IMPUTATION + SCALING
scaler = StandardScaler()
X_full[num_cols] = scaler.fit_transform(
    X_full[num_cols].fillna(X_full[num_cols].median())
)
X_test[num_cols] = scaler.transform(
    X_test[num_cols].fillna(X_full[num_cols].median())
)



# AUTOENCODER TRAINING 
ae_input_full = X_full[num_cols + cat_cols].values.astype(np.float32)
ae_input_test = X_test[num_cols + cat_cols].values.astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ae = AutoEncoder(ae_input_full.shape[1]).to(device)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

loader = DataLoader(
    AEDataset(ae_input_full),
    batch_size=2048,
    shuffle=True
)

ae.train()
for _ in range(8):
    for xb, _ in loader:
        xb = xb.to(device)
        opt.zero_grad()
        out, _ = ae(xb)
        loss = loss_fn(out, xb)
        loss.backward()
        opt.step()

ae.eval()
with torch.no_grad():
    _, z_full = ae(torch.tensor(ae_input_full).to(device))
    _, z_test = ae(torch.tensor(ae_input_test).to(device))

z_full = z_full.cpu().numpy()
z_test = z_test.cpu().numpy()



# PCA
pca = PCA(n_components=3, random_state=42)
pca_full = pca.fit_transform(z_full)
pca_test = pca.transform(z_test)

# KMEANS
km = KMeans(n_clusters=8, random_state=42)
cl_full = km.fit_predict(pca_full)
cl_test = km.predict(pca_test)



# FINAL FEATURE TABLE
X_full_final = X_full.copy()
X_test_final = X_test.copy()

# AE features
for i in range(z_full.shape[1]):
    X_full_final[f"ae_{i}"] = z_full[:, i]
    X_test_final[f"ae_{i}"] = z_test[:, i]

# PCA features
for i in range(pca_full.shape[1]):
    X_full_final[f"pca_{i}"] = pca_full[:, i]
    X_test_final[f"pca_{i}"] = pca_test[:, i]

# Cluster label
X_full_final["cluster"] = cl_full
X_test_final["cluster"] = cl_test

print("Final train shape:", X_full_final.shape)
print("Final test shape :", X_test_final.shape)



from lightgbm import LGBMClassifier

final_model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42
)

final_model.fit(X_full_final, y_full)



test_preds = final_model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame({
    "id": pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")["id"],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()



print("Prediction summary:")
print(pd.Series(test_preds).describe())

print("Train target mean:", y_full.mean())
print("Test pred mean  :", test_preds.mean())


