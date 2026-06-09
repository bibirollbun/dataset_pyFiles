!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# === Paths ===
BASE_PATH      = '/kaggle/input/neurips-open-polymer-prediction-2025/'
EXTRA_BASE     = '/kaggle/input/'
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV_PATH  = os.path.join(BASE_PATH, 'test.csv')
EXTRA_PATHS = {
    'tc':   os.path.join(EXTRA_BASE, 'tc-smiles/Tc_SMILES.csv'),
    'tg2':  os.path.join(EXTRA_BASE, 'smiles-extra-data/JCIM_sup_bigsmiles.csv'),
    'tg3':  os.path.join(EXTRA_BASE, 'smiles-extra-data/data_tg3.xlsx'),
    'dnst': os.path.join(EXTRA_BASE, 'smiles-extra-data/data_dnst1.xlsx'),
}

TARGET_COLUMNS = ['Tg','FFV','Tc','Density','Rg']

# === Utility functions ===
def make_smile_canonical(smiles: str) -> str:
    """Convert SMILES to canonical form for consistent grouping"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return smiles  # fallback to original if conversion fails
    except:
        return smiles
        
def add_extra_data(df_train: pd.DataFrame, df_extra: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    1) Canonicalize SMILES in df_extra.
    2) Drop rows missing SMILES or the target.
    3) Group by SMILES, take mean(target).
    4) Impute into df_train where SMILES overlap and df_train[target] is null.
    5) Append rows for SMILES unique to df_extra.
    """
    before = df_train[target].notnull().sum()

    # 1+2+3
    df_extra['SMILES'] = df_extra['SMILES'].apply(make_smile_canonical)
    df_extra = df_extra.dropna(subset=['SMILES', target])
    df_extra = df_extra.groupby('SMILES', as_index=False)[[target]].mean()

    set_train = set(df_train['SMILES'])
    set_extra = set(df_extra['SMILES'])
    overlap   = set_train & set_extra
    unique    = set_extra - set_train

    # 4) Impute missing in df_train
    for smi in overlap:
        mask = df_train['SMILES'] == smi
        if df_train.loc[mask, target].isna().any():
            val = df_extra.loc[df_extra['SMILES']==smi, target].iloc[0]
            df_train.loc[mask, target] = val

    # 5) Append unique SMILES
    new_rows = df_extra[df_extra['SMILES'].isin(unique)].copy()
    for col in TARGET_COLUMNS:
        if col not in new_rows:
            new_rows[col] = pd.NA
    new_rows = new_rows[['SMILES'] + TARGET_COLUMNS]
    df_train = pd.concat([df_train, new_rows], ignore_index=True)

    after = df_train[target].notnull().sum()
    print(f'For "{target}", added {after - before} new samples.')
    return df_train

# === Load main data ===
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df  = pd.read_csv(TEST_CSV_PATH)

# === Merge extra sources ===
print("Adding Tc data...")
tc_extra = (pd.read_csv(EXTRA_PATHS['tc'], usecols=['SMILES','TC_mean'])
              .rename(columns={'TC_mean':'Tc'}))
train_df = add_extra_data(train_df, tc_extra, 'Tc')

print("Adding Tg data (source 2)...")
tg2 = (pd.read_csv(EXTRA_PATHS['tg2'], usecols=['SMILES','Tg (C)'])
         .rename(columns={'Tg (C)':'Tg'}))
train_df = add_extra_data(train_df, tg2, 'Tg')

print("Adding Tg data (source 3)...")
tg3 = (pd.read_excel(EXTRA_PATHS['tg3'], usecols=['SMILES','Tg [K]'])
         .rename(columns={'Tg [K]':'Tg'}))
tg3['Tg'] = tg3['Tg'] - 273.15  # K â†’ Â°C
train_df = add_extra_data(train_df, tg3, 'Tg')

print("Adding Density data...")
dnst = (pd.read_excel(EXTRA_PATHS['dnst'], usecols=['SMILES','density(g/cm3)'])
          .rename(columns={'density(g/cm3)':'Density'}))
dnst['SMILES'] = dnst['SMILES'].apply(make_smile_canonical)
dnst = dnst.dropna(subset=['SMILES','Density'])
dnst = dnst[dnst['Density'] != 'nylon']
dnst['Density'] = dnst['Density'].astype(float) - 0.118
train_df = add_extra_data(train_df, dnst, 'Density')

print("\nFinal counts:")
for t in TARGET_COLUMNS:
    print(f"  {t}: {train_df[t].notnull().sum()}")
print("-"*40)


print(f"Total polymers in train set: {len(train_df)}\n")

# Count non-null samples per property
for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    n = train_df[prop].notnull().sum()
    print(f"{prop}: {n} samples")


           id          Tg       FFV        Tc   Density         Rg
0  1109053969  167.816630  0.375729  0.215749  1.132733  23.194366
1  1422188626  176.811480  0.378414  0.251642  1.056754  21.202372
2  2032016830  141.433996  0.351483  0.265111  1.117257  18.583658
==================================================


# Quick EDA: target summary
print("Target Summary:")
print(train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].describe(), "\n")

# Histograms for each target
for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    plt.figure()
    train[col].hist(bins=30)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# Feature engineering from SMILES
def featurize(smiles):
    features = {
        'len': len(smiles),
        'count_digits': sum(c.isdigit() for c in smiles),
        'count_paren_open': smiles.count('('),
        'count_paren_close': smiles.count(')'),
        'count_stars': smiles.count('*'),
    }
    for atom in ['C', 'O', 'N', 'F', 'S', 'P', 'H']:
        features[f'count_{atom}'] = smiles.count(atom)
    return features

X = pd.DataFrame([featurize(s) for s in train['SMILES']])
y = train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Correlation matrix of features vs targets
corr = pd.concat([X, y], axis=1).corr()
corr_feat_targets = corr.loc[X.columns, y.columns]
print("Feature-Target Correlations:")
print(corr_feat_targets, "\n")

# Baseline model per-property Random Forest
print("Baseline MAE per property:")
for prop in y.columns:
    mask = y[prop].notnull()
    X_prop = X[mask]
    y_prop = y.loc[mask, prop]
    X_tr, X_val, y_tr, y_val = train_test_split(X_prop, y_prop, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"{prop}: {mae:.3f}")



import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# === Featurization ===
def featurize(smiles: str) -> dict:
    d = {
        'len': len(smiles),
        'count_digits': sum(c.isdigit() for c in smiles),
        'count_paren_open': smiles.count('('),
        'count_paren_close': smiles.count(')'),
        'count_stars': smiles.count('*'),
    }
    for atom in ['C','O','N','F','S','P','H']:
        d[f'count_{atom}'] = smiles.count(atom)
    return d

X_train = pd.DataFrame([featurize(s) for s in train_df['SMILES']])
X_test  = pd.DataFrame([featurize(s) for s in test_df['SMILES']])

# === Hyperparameter tuning & training ===
param_dist = {
    'n_estimators':     [100, 200],
    'max_depth':        [None, 10, 20],
    'max_features':     ['auto','sqrt'],
    'min_samples_split':[2,5],
    'min_samples_leaf': [1,2]
}
submission = pd.DataFrame({'id': test_df['id']})

for prop in TARGET_COLUMNS:
    mask   = train_df[prop].notna()
    X_prop = X_train.loc[mask].reset_index(drop=True)
    y_prop = train_df.loc[mask, prop].reset_index(drop=True)

    rs = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=5, cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, random_state=42
    )
    rs.fit(X_prop, y_prop)

    best   = rs.best_estimator_
    cv_mae = -rs.best_score_
    print(f"{prop}: CV MAE={cv_mae:.3f}, params={rs.best_params_}")

    best.fit(X_prop, y_prop)
    submission[prop] = best.predict(X_test)

# === Save ===
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission preview:")
print(submission.head())


# import pandas as pd
# import numpy as np
# import re
# from rdkit import Chem, DataStructs, RDLogger
# from rdkit.Chem import AllChem
# from sklearn.ensemble import RandomForestRegressor

# # â€”â€”â€” Disable RDKit warnings â€”â€”â€”
# RDLogger.DisableLog('rdApp.*')

# # â€”â€”â€” Fingerprint featurizer (strip '*', empty branches, stereo slashes, leading '=') â€”â€”â€”
# def mol_fp(smiles, radius=2, n_bits=1024):
#     # remove asterisks, empty branches, stereo indicators, and stray leading '='
#     clean = smiles.replace('*', '').replace('()', '')
#     clean = re.sub(r'[\\/]', '', clean)      # remove / and \
#     clean = re.sub(r'^=+', '', clean)        # strip leading '='
    
#     try:
#         mol = Chem.MolFromSmiles(clean, sanitize=False)
#         if mol is None:
#             raise ValueError("Could not parse")
#         Chem.SanitizeMol(mol)  # explicit sanitization
#         fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
#         arr = np.zeros((n_bits,), dtype=int)
#         DataStructs.ConvertToNumpyArray(fp, arr)
#         return arr
#     except Exception:
#         # parsing or sanitization failed
#         return np.zeros((n_bits,), dtype=int)


# # â€”â€”â€” Build feature matrices â€”â€”â€”
# X_train_full = np.vstack(train['SMILES'].apply(mol_fp).values)
# X_test       = np.vstack(test ['SMILES'].apply(mol_fp).values)

# # â€”â€”â€” Train & predict per property â€”â€”â€”
# props = ['Tg','FFV','Tc','Density','Rg']
# submission = pd.DataFrame({'id': test['id']})

# for p in props:
#     mask = train[p].notnull()
#     Xp   = X_train_full[mask.values]
#     yp   = train.loc[mask, p].values
    
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(Xp, yp)
#     submission[p] = model.predict(X_test)

# # â€”â€”â€” Save submission â€”â€”â€”
# #submission.to_csv('/kaggle/working/submission.csv', index=False)
# #print(submission.head())


# from rdkit.Chem import AllChem, Descriptors
# from sklearn.linear_model import Ridge
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_absolute_error

# # Suppress RDKit warnings
# RDLogger.DisableLog('rdApp.*')

# # Featurizers with reduced fingerprint size
# def safe_mol(smiles):
#     s = smiles.lstrip('/\\').replace('()','').replace('*','')
#     try:
#         m = Chem.MolFromSmiles(s, sanitize=False)
#         if m is None:
#             return None
#         Chem.SanitizeMol(m)
#         return m
#     except:
#         return None

# def basic_stats(smiles):
#     stats = {
#         'len': len(smiles),
#         'count_digits': sum(c.isdigit() for c in smiles),
#         'count_paren_open': smiles.count('('),
#         'count_paren_close': smiles.count(')'),
#         'count_stars': smiles.count('*'),
#     }
#     stats.update({f'count_{atom}': smiles.count(atom) for atom in ['C','O','N','F','S','P','H']})
#     return stats

# descriptor_list = [
#     ('MolWt', Descriptors.MolWt), ('MolLogP', Descriptors.MolLogP),
#     ('TPSA', Descriptors.TPSA), ('NumRotatableBonds', Descriptors.NumRotatableBonds),
#     ('NumHDonors', Descriptors.NumHDonors), ('NumHAcceptors', Descriptors.NumHAcceptors),
#     ('NumAliphaticRings', Descriptors.NumAliphaticRings), ('NumAromaticRings', Descriptors.NumAromaticRings),
# ]

# def rdkit_desc(smiles):
#     mol = safe_mol(smiles)
#     if mol is None:
#         return [np.nan] * len(descriptor_list)
#     return [func(mol) for _, func in descriptor_list]

# def mol_fp(smiles, radius=2, n_bits=256):
#     mol = safe_mol(smiles)
#     if mol is None:
#         return np.zeros((n_bits,), dtype=int)
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
#     arr = np.zeros((n_bits,), dtype=int)
#     DataStructs.ConvertToNumpyArray(fp, arr)
#     return arr

# properties = ['Tg','FFV','Tc','Density','Rg']

# # Compute features
# basic_train = pd.DataFrame([basic_stats(s) for s in train['SMILES']])
# desc_train  = pd.DataFrame([rdkit_desc(s) for s in train['SMILES']], columns=[n for n,_ in descriptor_list])
# desc_train  = desc_train.fillna(desc_train.median())
# fp_train    = np.vstack([mol_fp(s) for s in train['SMILES']])
# X_all       = np.hstack([basic_train.values, desc_train.values, fp_train])

# basic_test = pd.DataFrame([basic_stats(s) for s in test['SMILES']])
# desc_test  = pd.DataFrame([rdkit_desc(s) for s in test['SMILES']], columns=[n for n,_ in descriptor_list])
# desc_test  = desc_test.fillna(desc_train.median())
# fp_test    = np.vstack([mol_fp(s) for s in test['SMILES']])
# X_test_all = np.hstack([basic_test.values, desc_test.values, fp_test])

# # Ensemble models: Random Forest, Ridge, MLP
# models = {
#     'RF':    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
#     'Ridge': Ridge(alpha=1.0),
#     'MLP':   MLPRegressor(hidden_layer_sizes=(100,), max_iter=100, random_state=42)
# }

# # Storage
# val_scores = {prop: {} for prop in properties}
# test_preds = {prop: [] for prop in properties}

# # Train and evaluate per property
# for prop in properties:
#     mask = train[prop].notnull().values
#     X_prop = X_all[mask]
#     y_prop = train.loc[mask, prop].values
#     X_tr, X_val, y_tr, y_val = train_test_split(X_prop, y_prop, test_size=0.2, random_state=42)
    
#     val_preds = []
#     for name, model in models.items():
#         m = model.__class__(**model.get_params())
#         m.fit(X_tr, y_tr)
#         pred_val = m.predict(X_val)
#         val_scores[prop][name] = mean_absolute_error(y_val, pred_val)
#         val_preds.append(pred_val)
        
#         m.fit(X_prop, y_prop)
#         test_preds[prop].append(m.predict(X_test_all))
    
#     val_scores[prop]['Ensemble'] = mean_absolute_error(y_val, np.mean(val_preds, axis=0))

# # Display validation MAEs
# val_df = pd.DataFrame(val_scores).T
# print("Validation MAE per model & ensemble:")
# print(val_df)

# # Assemble ensemble submission
# submission = pd.DataFrame({'id': test['id']})
# for prop in properties:
#     submission[prop] = np.mean(test_preds[prop], axis=0)

# submission.to_csv('/kaggle/working/submission.csv', index=False)
# print("\nEnsemble submission preview:")
# print(submission.head())




# === Prepare data ===
mask_full = train_df[TARGETS].notnull().all(axis=1)
full_df   = train_df[mask_full].reset_index(drop=True)
all_smiles = pd.concat([full_df['SMILES'], test_df['SMILES']])
chars = sorted({c for s in all_smiles for c in s})
char2idx = {c:i+1 for i,c in enumerate(chars)}
max_len = all_smiles.str.len().max()

def encode(s):
    seq = [char2idx.get(c,0) for c in s][:max_len]
    return seq + [0]*(max_len-len(seq))

X = np.vstack(full_df['SMILES'].apply(encode).values)
y = full_df[TARGETS].values.astype(np.float32)
X_test = np.vstack(test_df['SMILES'].apply(encode).values)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

class SMIDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

train_loader = DataLoader(SMIDataset(X_tr, y_tr), batch_size=64, shuffle=True)
val_loader   = DataLoader(SMIDataset(X_val, y_val), batch_size=64)
test_loader  = DataLoader(SMIDataset(X_test), batch_size=64)

class Net(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, out_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv  = nn.Conv1d(embed_dim, 128, 5, padding=2)
        self.pool  = nn.MaxPool1d(2)
        self.lstm  = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1   = nn.Linear(2*hidden_dim, 128)
        self.drop  = nn.Dropout(0.2)
        self.fc2   = nn.Linear(128, out_dim)
    def forward(self, x):
        x = self.embed(x).permute(0,2,1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).permute(0,2,1)
        x, _ = self.lstm(x)
        x = x.mean(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

device = torch.device('cpu')
model  = Net(len(char2idx)+1, 128, 64, len(TARGETS)).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
crit   = nn.MSELoss()

best_loss = float('inf')
for epoch in range(1,51):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += crit(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch} Val MSE {val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best.pth')

model.load_state_dict(torch.load('best.pth'))
model.eval()
preds = []
with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
preds = np.vstack(preds)

submission = pd.DataFrame(preds, columns=TARGETS)
submission.insert(0, 'id', test_df['id'])
submission.to_csv('submission_pytorch.csv', index=False)




