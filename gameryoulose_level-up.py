!pip install tabpfn xgboost scikit-learn pandas numpy


import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CONFIGURATION
# ==========================================
N_FOLDS = 5
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 35

def load_data():
    train_df = None; test_df = None
    for root, dirs, files in os.walk("../input"):
        for file in files:
            p = os.path.join(root, file)
            if "train" in file and "csv" in file: train_df = pd.read_csv(p, compression='zip' if 'zip' in file else None)
            if "test" in file and "csv" in file: test_df = pd.read_csv(p, compression='zip' if 'zip' in file else None)
    
    if train_df is None: train_df = pd.read_csv("train.csv")
    if test_df is None: test_df = pd.read_csv("test.csv")
    return train_df, test_df

# ==========================================
# 2. FEATURE ENGINEERING (The ID Strategy)
# ==========================================
def process_data(train, test):
    print("ğŸ› ï¸�  Processing Data (Restoring ID Strategy)...")
    
    # 1. Conservative Outlier Removal
    train = train[train['y'] < 170].copy()
    
    y_train = train['y'].values
    train_ids = train['ID'].values
    test_ids = test['ID'].values
    
    # 2. SAVE ID for Feature Engineering
    ntrain = train.shape[0]
    train.drop(['y'], axis=1, inplace=True)
    
    combined = pd.concat([train, test], axis=0)
    
    # --- 3. THE ID LEAK FEATURES ---
    combined['ID_Scaled'] = MinMaxScaler().fit_transform(combined[['ID']])
    combined['ID_Group'] = combined['ID'].apply(lambda x: int(x / 20))
    
    # --- 4. CATEGORICAL GROUPING ---
    cat_cols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
    for c in cat_cols:
        counts = combined[c].value_counts()
        keep = counts.index[counts >= 5]
        combined[c] = combined[c].apply(lambda x: x if x in keep else 'Other')
        le = LabelEncoder()
        combined[c] = le.fit_transform(combined[c].astype(str))

    # --- 5. DECOMPOSITION ---
    n_comp = 12
    pca = PCA(n_components=n_comp, random_state=SEED)
    pca_res = pca.fit_transform(combined.drop(['ID'], axis=1).select_dtypes(include=np.number))
    
    ica = FastICA(n_components=n_comp, random_state=SEED, whiten='unit-variance')
    ica_res = ica.fit_transform(combined.drop(['ID'], axis=1).select_dtypes(include=np.number))
    
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=SEED)
    grp_res = grp.fit_transform(combined.drop(['ID'], axis=1).select_dtypes(include=np.number))
    
    for i in range(n_comp):
        combined[f'pca_{i}'] = pca_res[:, i]
        combined[f'ica_{i}'] = ica_res[:, i]
        combined[f'grp_{i}'] = grp_res[:, i]

    combined.drop(['ID'], axis=1, inplace=True)

    # Split
    train_x = combined.iloc[:ntrain].copy()
    test_x = combined.iloc[ntrain:].copy()
    
    # Scaling for NN
    scaler = StandardScaler()
    train_x_sc = scaler.fit_transform(train_x)
    test_x_sc = scaler.transform(test_x)
    
    return train_x, test_x, train_x_sc, test_x_sc, y_train, test_ids

# ==========================================
# 3. NEURAL NETWORK
# ==========================================
class MercDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

class MercNet(nn.Module):
    def __init__(self, input_dim):
        super(MercNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.layers(x)

# ==========================================
# 4. TRAINING THE STACK
# ==========================================
def train_stack(X_df, X_sc, y, X_test_df, X_test_sc):
    ntrain = X_df.shape[0]; ntest = X_test_df.shape[0]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_train = pd.DataFrame()
    oof_test = pd.DataFrame()
    
    print(f"\nğŸš€ Training Stack (XGB + LGB + Cat + NN)...")
    
    xgb_oof, xgb_pred = np.zeros(ntrain), np.zeros(ntest)
    lgb_oof, lgb_pred = np.zeros(ntrain), np.zeros(ntest)
    cat_oof, cat_pred = np.zeros(ntrain), np.zeros(ntest)
    nn_oof, nn_pred = np.zeros(ntrain), np.zeros(ntest)
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_df, y)):
        X_tr, X_val = X_df.iloc[tr_idx], X_df.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # 1. XGBoost
        dtr = xgb.DMatrix(X_tr, y_tr); dval = xgb.DMatrix(X_val, y_val); dtest = xgb.DMatrix(X_test_df)
        model = xgb.train({'n_trees': 500, 'eta': 0.005, 'max_depth': 4, 'subsample': 0.95, 'objective': 'reg:squarederror', 'verbosity': 0}, 
                          dtr, 1200, [(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        xgb_oof[val_idx] = model.predict(dval); xgb_pred += model.predict(dtest) / N_FOLDS
        
        # 2. LightGBM
        lgb_ds = lgb.Dataset(X_tr, y_tr); lgb_val = lgb.Dataset(X_val, y_val)
        model = lgb.train({'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.005, 'num_leaves': 10, 'max_depth': 4, 'verbosity': -1}, 
                          lgb_ds, 1200, valid_sets=[lgb_val], callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_oof[val_idx] = model.predict(X_val); lgb_pred += model.predict(X_test_df) / N_FOLDS

        # 3. CatBoost
        model = CatBoostRegressor(iterations=1200, learning_rate=0.005, depth=4, verbose=False, allow_writing_files=False)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        cat_oof[val_idx] = model.predict(X_val); cat_pred += model.predict(X_test_df) / N_FOLDS
        
        # 4. Neural Net
        tr_ds = MercDataset(X_sc[tr_idx], y_tr); val_ds = MercDataset(X_sc[val_idx], y_val)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True); val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
        nn_model = MercNet(X_sc.shape[1]).to(DEVICE)
        opt = optim.Adam(nn_model.parameters(), lr=0.004); crit = nn.MSELoss()
        
        for ep in range(EPOCHS):
            nn_model.train()
            for x, t in tr_dl: opt.zero_grad(); loss = crit(nn_model(x.to(DEVICE)), t.to(DEVICE)); loss.backward(); opt.step()
        
        nn_model.eval()
        with torch.no_grad():
            p = []; [p.extend(nn_model(x.to(DEVICE)).cpu().numpy().flatten()) for x, _ in val_dl]
            nn_oof[val_idx] = p
            p_test = []; [p_test.extend(nn_model(x.to(DEVICE)).cpu().numpy().flatten()) for x in DataLoader(MercDataset(X_test_sc), batch_size=BATCH_SIZE)]
            nn_pred += np.array(p_test) / N_FOLDS
            
    oof_train['xgb'] = xgb_oof; oof_test['xgb'] = xgb_pred
    oof_train['lgb'] = lgb_oof; oof_test['lgb'] = lgb_pred
    oof_train['cat'] = cat_oof; oof_test['cat'] = cat_pred
    oof_train['nn'] = nn_oof;   oof_test['nn'] = nn_pred
    
    print("\n   [Correlation Matrix]")
    print(oof_train.corr())
    
    return oof_train, oof_test

# ==========================================
# 5. MAIN (RIDGE STACKING)
# ==========================================
if __name__ == "__main__":
    tr, te = load_data()
    X, X_test, X_sc, X_test_sc, y, ids = process_data(tr, te)
    
    L1_tr, L1_te = train_stack(X, X_sc, y, X_test, X_test_sc)
    
    print("\nğŸ§  Meta-Learning (Ridge Stacking)...")
    # This was the magic step: Ridge learns how to fix the NN scaling automatically
    meta = Ridge(alpha=40) 
    meta.fit(L1_tr, y)
    print(f"   Weights: {dict(zip(L1_tr.columns, meta.coef_.round(3)))}")
    
    final_preds = meta.predict(L1_te)
    
    pd.DataFrame({'ID': ids, 'y': final_preds}).to_csv('submission_id_leak_stack.csv', index=False)
    print("âœ… Done! Saved 'submission_id_leak_stack.csv'")

