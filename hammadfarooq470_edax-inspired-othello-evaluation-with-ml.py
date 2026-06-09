# Full pipeline for "Evaluate Othello Boards" (competitive baseline)
# Requirements: numpy, pandas, sklearn, lightgbm, torch, torchvision
# Run in Kaggle notebook (GPU recommended for CNN)
import os, gc, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

# To ignore specific categories of warnings, for example, DeprecationWarning
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Your Python code that might generate warnings goes here


# Paths (Kaggle default)
INPUT_DIR = "/kaggle/input/evaluate-othello-boards"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH  = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

print("train shape:", train.shape, "test shape:", test.shape)
# Quick look
train.head()


# -----------------------------
# Utility functions: board conversions & transforms
# -----------------------------
def rowcol_to_idx(r,c): return r*8 + c

def flat_to_board(flat):
    # flat: length-64 array of ints (1,-1,0)
    return np.array(flat).reshape(8,8)

def board_to_flat(board):
    return board.reshape(-1)

# symmetry transforms (8): 4 rotations x reflect/no-reflect
def all_symmetries(board):
    # board: (8,8)
    mats = []
    for k in range(4):
        r = np.rot90(board, k)
        mats.append(r.copy())
        mats.append(np.fliplr(r).copy())
    return mats

# convert 1 / -1 / 0 -> two channels: black (1) and white (1)
def board_to_channels(board):
    # returns 2x8x8: channel0 for black (1), channel1 for white (1)
    black = (board==1).astype(np.float32)
    white = (board==-1).astype(np.float32)
    return np.stack([black, white])

# some engineered features
def engineered_features_from_flat(flat):
    b = flat.reshape(8,8)
    # counts
    black_count = (b==1).sum()
    white_count = (b==-1).sum()
    empties = (b==0).sum()
    diff = black_count - white_count
    # corner occupancy
    corners = [(0,0),(0,7),(7,0),(7,7)]
    corner_vals = [b[r,c] for r,c in corners]
    corner_count_turn = sum(1 for v in corner_vals if v==1) # but careful: 'turn player' unknown - we'll add both black/white counts
    # edges (perimeter excluding corners)
    edges = []
    edges += [(0,c) for c in range(1,7)]
    edges += [(7,c) for c in range(1,7)]
    edges += [(r,0) for r in range(1,7)]
    edges += [(r,7) for r in range(1,7)]
    edge_black = sum(1 for (r,c) in edges if b[r,c]==1)
    edge_white = sum(1 for (r,c) in edges if b[r,c]==-1)
    # mobility rough (number of empty squares adjacent to opponent pieces) - heuristic
    mob = 0
    for r in range(8):
        for c in range(8):
            if b[r,c]==0:
                # check neighbors if exists opponent (1 or -1)
                neigh = b[max(0,r-1):r+2, max(0,c-1):c+2]
                if np.any(neigh!=0):
                    mob += 1
    return [black_count, white_count, diff, empties, edge_black, edge_white, mob, sum(np.array(corner_vals)==1), sum(np.array(corner_vals)==-1)]



# apply engineered features to df
def add_features(df):
    feats = []
    for i,row in df.iterrows():
        flat = row[[f"cell_{i}" for i in range(64)]].values.astype(int)
        feats.append(engineered_features_from_flat(flat))
    feats = np.array(feats)
    colnames = ["bcount","wcount","disc_diff","empties","edge_b","edge_w","mobility","corner_b","corner_w"]
    fdf = pd.DataFrame(feats, columns=colnames, index=df.index)
    return pd.concat([df.reset_index(drop=True), fdf.reset_index(drop=True)], axis=1)



# -----------------------------
# Prepare training features for LightGBM
# -----------------------------
train_feat = add_features(train)
test_feat  = add_features(test)

# raw board cells already in dataframe as cell_0..cell_63, keep them
lgb_features = [f"cell_{i}" for i in range(64)] + ["bcount","wcount","disc_diff","empties","edge_b","edge_w","mobility","corner_b","corner_w"]
len(lgb_features), lgb_features[:5]


# -----------------------------
# LightGBM cross-validated training
# -----------------------------
TARGET = "turn_player_advantage"
X = train_feat[lgb_features].values
y = train_feat[TARGET].values
X_test = test_feat[lgb_features].values

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
preds_test = np.zeros(len(test))
models = []

lgb_params = {
    "objective":"regression",
    "metric":"rmse",
    "verbosity":-1,
    "boosting_type":"gbdt",
    "learning_rate":0.02,
    "num_leaves":64,
    "feature_fraction":0.8,
    "bagging_fraction":0.8,
    "bagging_freq":5,
    "seed":SEED
}

for fold,(tr_idx,va_idx) in enumerate(kf.split(X,y)):
    print("LGB Fold", fold)
    tr_x, tr_y = X[tr_idx], y[tr_idx]
    va_x, va_y = X[va_idx], y[va_idx]

    dtrain = lgb.Dataset(tr_x, label=tr_y)
    dval   = lgb.Dataset(va_x, label=va_y)

    bst = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dval],
        valid_names=["train","valid"],
        callbacks=[
            lgb.early_stopping(100),       # replaces early_stopping_rounds
            lgb.log_evaluation(200)        # replaces verbose_eval
        ]
    )

    pred_va = bst.predict(va_x, num_iteration=bst.best_iteration)
    oof[va_idx] = pred_va
    preds_test += bst.predict(X_test, num_iteration=bst.best_iteration) / kf.n_splits
    models.append(bst)

    print("Fold RMSE:", math.sqrt(mean_squared_error(va_y, pred_va)))

print("LGB OOF RMSE:", math.sqrt(mean_squared_error(y, oof)))



# -----------------------------
# CNN model (PyTorch): process channels and symmetry augmentation
# -----------------------------
# We will create a small 2-channel CNN that takes (2,8,8) inputs and outputs scalar
class OthelloDataset(Dataset):
    def __init__(self, df, target_col=None, augment=False):
        self.df = df
        self.target_col = target_col
        self.augment = augment
        self.indices = df.index.to_list()
        self.cells = [ [f"cell_{i}" for i in range(64)] ]
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        flat = row[[f"cell_{i}" for i in range(64)]].values.astype(np.int8)
        board = flat.reshape(8,8)
        # choose random symmetry during training if augment True
        if self.augment:
            syms = all_symmetries(board)
            board = random.choice(syms)
        channels = board_to_channels(board) # 2x8x8 float32
        x = torch.tensor(channels, dtype=torch.float32)
        if self.target_col is None:
            return x
        y = torch.tensor(row[self.target_col], dtype=torch.float32)
        return x, y

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1), nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x).squeeze(1)


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)   # regression output

    def forward(self, x):
        # conv layers + relu + pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# -----------------------------
# Dataset
# -----------------------------
class OthelloDataset(Dataset):
    def __init__(self, df, target_col=None, augment=False):
        self.df = df
        self.target_col = target_col
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        board = row[[f"cell_{i}" for i in range(64)]].values.astype(np.int8)
        board = board.reshape(8, 8)

        # two input planes: black and white
        black_plane = (board == 1).astype(np.float32)
        white_plane = (board == -1).astype(np.float32)
        x = np.stack([black_plane, white_plane], axis=0)  # shape (2, 8, 8)

        if self.target_col is not None:
            y = np.float32(row[self.target_col])
            return torch.tensor(x), torch.tensor(y)
        else:
            return torch.tensor(x)

# -----------------------------
# CNN Model
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Training Helper
# -----------------------------
def train_cnn_fold(train_df, valid_df, epochs=12, batch_size=256, lr=1e-3, device="cpu"):
    best_rmse = 1e9
    best_state = None

    train_ds = OthelloDataset(train_df, target_col=TARGET, augment=True)
    valid_ds = OthelloDataset(valid_df, target_col=TARGET, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb).squeeze().detach().cpu().numpy()
                preds.append(p)
                truths.append(yb.detach().cpu().numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        rmse = math.sqrt(mean_squared_error(truths, preds))

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, best_rmse

# -----------------------------
# 5-Fold CV
# -----------------------------
kf_cnn = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_cnn = np.zeros(len(train))
preds_test_cnn = np.zeros(len(test))
device = "cuda" if torch.cuda.is_available() else "cpu"

for fold, (tr_idx, va_idx) in enumerate(kf_cnn.split(train)):
    print("CNN Fold", fold)
    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)

    model, best_rmse = train_cnn_fold(
        tr_df, va_df, epochs=12, batch_size=256, lr=1e-3, device=device
    )

    # predict val
    va_ds = OthelloDataset(va_df, target_col=TARGET, augment=False)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    preds_val = []
    with torch.no_grad():
        model.eval()
        for xb, yb in va_loader:
            xb = xb.to(device)
            p = model(xb).detach().cpu().numpy()
            preds_val.append(p)
    preds_val = np.concatenate(preds_val)
    oof_cnn[va_idx] = preds_val.ravel()



    # predict test
    test_ds = OthelloDataset(test, target_col=None, augment=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    preds_t = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            p = model(xb).detach().cpu().numpy()
            preds_t.append(p)
    preds_t = np.concatenate(preds_t)
    preds_test_cnn += preds_t.ravel() / kf_cnn.n_splits


    # cleanup
    del model
    gc.collect()

print("CNN OOF RMSE:", math.sqrt(mean_squared_error(train[TARGET].values, oof_cnn)))



# -----------------------------
# Simple stacking: stack LGB + CNN predictions with a Ridge (out-of-fold)
# -----------------------------
stack_train = np.vstack([oof, oof_cnn]).T
stack_test  = np.vstack([preds_test, preds_test_cnn]).T
stack_oof = np.zeros(len(train))
stack_test_pred = np.zeros(len(test))

kf2 = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx,va_idx in kf2.split(stack_train):
    r = Ridge(alpha=1.0)
    r.fit(stack_train[tr_idx], y[tr_idx])
    stack_oof[va_idx] = r.predict(stack_train[va_idx])
    stack_test_pred += r.predict(stack_test) / kf2.n_splits
print("Stack OOF RMSE:", math.sqrt(mean_squared_error(y, stack_oof)))


# -----------------------------
# Final submission
# -----------------------------
sub = sample.copy()
sub['turn_player_advantage'] = stack_test_pred
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")

