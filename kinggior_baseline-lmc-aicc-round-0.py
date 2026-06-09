"""
Configuration & imports
"""
import os
from pathlib import Path
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.metrics import accuracy_score, confusion_matrix

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INPUT_DIM = 100
EMBED_DIM = 10
LOGIT_DIM = 5
N = 5000
P_A = 0.5


OUT_DIR = Path("/kaggle/input/latent-model-classification-aicc-round-0/")
TRAIN_CSV = OUT_DIR / "dataset.csv"
PEN_A_PTH = OUT_DIR / "modelA_penultimate.pth"
PEN_B_PTH = OUT_DIR / "modelB_penultimate.pth"
OUT_PRED_CSV = "/kaggle/working/predictions.csv"


df = pd.read_csv(TRAIN_CSV)
x_cols = [c for c in df.columns if c.startswith("x")]
X = df[x_cols].values.astype(np.float32)
class PenultimateNet(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, hidden=64, out_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

device = "cpu" # or "cuda"
penA = PenultimateNet().to(device)
penB = PenultimateNet().to(device)
penA.load_state_dict(torch.load(str(PEN_A_PTH), map_location=device))
penB.load_state_dict(torch.load(str(PEN_B_PTH), map_location=device))
penA.eval(); penB.eval()


from sklearn.cluster import KMeans

with torch.no_grad():
    Xt = torch.from_numpy(X).to(device)
    ZA = penA(Xt).cpu().numpy()
    ZB = penB(Xt).cpu().numpy()

X_cat = np.hstack([ZA, ZB])
kmeans = KMeans(n_clusters=2, random_state=SEED)
clusters = kmeans.fit_predict(X_cat)
labels = np.where(clusters == 0, "A", "B")

ids = np.arange(1, len(labels) + 1)
out_df = pd.DataFrame({"ID": ids, "source": labels})
out_df.to_csv(OUT_PRED_CSV, index=False)
print(f"Saved predictions to: {OUT_PRED_CSV}  (IDs 0..{len(labels)-1})")


"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score

pred = pd.read_csv(OUT_PRED_CSV).set_index("ID")
sol  = pd.read_csv(OUT_DIR / "solution.csv").set_index("ID")

# case-insensitive lookup for 'source' column
pcol = next(c for c in pred.columns if c.lower() == "source")
scol = next(c for c in sol.columns  if c.lower() == "source")

idx = pred.index.intersection(sol.index)
y_pred = pred.loc[idx, pcol].astype(str).str.strip().str.lower()
y_true =  sol.loc[idx, scol].astype(str).str.strip().str.lower()

print(f"Evaluated {len(idx)} samples. Accuracy: {accuracy_score(y_true, y_pred):.6f}")
"""

