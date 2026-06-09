import warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import rankdata

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# Global Configuration
TARGET = "diagnosed_diabetes"

SEEDS_LGB = [7, 11, 17, 23, 29, 31, 37, 41, 43, 47]
SEEDS_CAT = [13, 19, 27, 39, 53]

AE_LATENT = 8
N_PCA = 3
N_CLUSTERS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)

print("Config loaded.")


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

y = train[TARGET].astype(int).values
train = train.drop(columns=[TARGET])

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("Target mean:", y.mean())


X_train = train.copy()
X_test  = test.copy()

# CATEGORICAL ENCODING
encoders = {}

for c in cat_cols:
    le = LabelEncoder()

    # Fit on train
    X_train[c] = le.fit_transform(X_train[c].astype(str))

    # Handle unseen categories in test
    X_test[c] = X_test[c].astype(str).map(
        lambda x: x if x in le.classes_ else "<UNK>"
    )

    # Add UNK token if needed
    if "<UNK>" not in le.classes_:
        le.classes_ = np.append(le.classes_, "<UNK>")

    # Transform test
    X_test[c] = le.transform(X_test[c])

    encoders[c] = le


# NUMERICAL SCALING
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(
    X_train[num_cols].fillna(X_train[num_cols].median())
)

X_test[num_cols] = scaler.transform(
    X_test[num_cols].fillna(X_train[num_cols].median())
)


# SANITY CHECKS (SAFE)
assert not X_train.isna().any().any(), "NaNs in X_train"
assert not X_test.isna().any().any(),  "NaNs in X_test"

assert np.isfinite(X_train[num_cols].values).all(), "Non-finite in X_train numeric"
assert np.isfinite(X_test[num_cols].values).all(),  "Non-finite in X_test numeric"

print("Preprocessing complete.")



# AUTOENCODER DATASET

class AEDataset(Dataset):
    def __init__(self, X):
        self.x = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.x[i]



# AUTOENCODER MODEL
class AutoEncoder(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Linear(64, d_latent)
        )

        self.dec = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.ReLU(),
            nn.Linear(64, d_in)
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


# AE TRAINING

ae_input_train = X_train[num_cols + cat_cols].values.astype(np.float32)
ae_input_test  = X_test[num_cols + cat_cols].values.astype(np.float32)

ae = AutoEncoder(ae_input_train.shape[1], AE_LATENT).to(DEVICE)
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

loader = DataLoader(
    AEDataset(ae_input_train),
    batch_size=2048,
    shuffle=True
)

ae.train()
for _ in range(8):
    for xb, _ in loader:
        xb = xb.to(DEVICE)

        optimizer.zero_grad()
        out, _ = ae(xb)
        loss = loss_fn(out, xb)
        loss.backward()
        optimizer.step()

ae.eval()
with torch.no_grad():
    _, z_train = ae(torch.tensor(ae_input_train).to(DEVICE))
    _, z_test  = ae(torch.tensor(ae_input_test).to(DEVICE))

z_train = z_train.cpu().numpy()
z_test  = z_test.cpu().numpy()

print("AE embeddings:", z_train.shape)



pca = PCA(n_components=N_PCA, random_state=42)
pca_train = pca.fit_transform(z_train)
pca_test = pca.transform(z_test)

km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
cl_train = km.fit_predict(pca_train)
cl_test = km.predict(pca_test)

print("PCA + clustering done.")


X_train_final = X_train.copy()
X_test_final = X_test.copy()

for i in range(AE_LATENT):
    X_train_final[f"ae_{i}"] = z_train[:, i]
    X_test_final[f"ae_{i}"] = z_test[:, i]

for i in range(N_PCA):
    X_train_final[f"pca_{i}"] = pca_train[:, i]
    X_test_final[f"pca_{i}"] = pca_test[:, i]

X_train_final["cluster"] = cl_train
X_test_final["cluster"] = cl_test

assert X_train_final.shape[1] == X_test_final.shape[1]

print("Final feature count:", X_train_final.shape[1])


lgb_preds = []

for seed in SEEDS_LGB:
    print(f"LGB seed {seed}")

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=seed
    )

    model.fit(X_train_final, y)

    preds = model.predict_proba(X_test_final)[:, 1]
    lgb_preds.append(preds)

lgb_preds = np.array(lgb_preds)

print("LGB preds shape:", lgb_preds.shape)



cat_preds = []

for seed in SEEDS_CAT:
    print(f"CatBoost seed {seed}")
    
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False
    )
    model.fit(X_train_final, y)
    
    preds = model.predict_proba(X_test_final)[:, 1]
    cat_preds.append(preds)

cat_preds = np.array(cat_preds)

print("CatBoost preds shape:", cat_preds.shape)


# Rank-average within each family

lgb_rank = np.mean([rankdata(p) for p in lgb_preds], axis=0)
cat_rank = np.mean([rankdata(p) for p in cat_preds], axis=0)

# Normalize ranks to [0, 1]

lgb_rank /= lgb_rank.max()
cat_rank /= cat_rank.max()

# Final stable blend
final_pred = 0.65 * lgb_rank + 0.35 * cat_rank

print("Prediction summary:")
print(pd.Series(final_pred).describe())
print("Train target mean:", y.mean())
print("Test pred mean :", final_pred.mean())


submission = pd.DataFrame({
    "id": test["id"],
    TARGET: final_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()

