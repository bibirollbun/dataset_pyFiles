# Single-cell MLP & Tree-based Ensemble Pipeline with Feature Selection

import time
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# 1) Load & merge, drop NaNs
train_df1 = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train_df2 = pd.read_csv("/kaggle/input/train-fert-2/train2.csv")
train_df  = pd.concat([train_df1, train_df2], ignore_index=True).dropna()
test_df   = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv").dropna()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    device_name = torch.cuda.get_device_name(0)
else:
    device_name = 'CPU'
print(f"Using device: {device} ({device_name})")

# 2) Target encoding
le_target = LabelEncoder()
train_df['target'] = le_target.fit_transform(train_df['Fertilizer Name'])

# 3) Feature engineering (LEGENDARY)
soil_CEC = {
    "Sandy": 4.0,
    "Red":   4.0,
    "Loamy": 15.0,
    "Clayey": 35.0,
    "Black": 35.0
}
fc_lookup = {
    "Sandy": 0.17,
    "Red": 0.17,
    "Loamy": 0.25,
    "Clayey": 0.32,
    "Black": 0.32
}
T_base = {
    "Wheat": 0,
    "Barley": 0,
    "Maize": 10,
    "Paddy": 10,
    "Millets": 10,
    "Sugarcane": 18,
    "Cotton": 15,
    "Tobacco": 10,
    "Oil seeds": 5,
    "Pulses": 8
}

# For some Alpha feature engineering
optimal_npk = {
    "Barley":     (120,  60,  40),
    "Cotton":     (120,  40,  40),
    "Ground Nuts":( 20,  40,  20),   # legumes fix N, so lower N
    "Maize":      (120,  60,  40),
    "Millets":     ( 80,  40,  30),
    "Oil seeds":  ( 60,  30,  30),
    "Paddy":      (100,  50,  50),
    "Pulses":     ( 20,  40,  20),   # similar to Ground Nuts
    "Sugarcane":  (150,  60,  40),
    "Tobacco":    (100,  30,  40),
    "Wheat":      (120,  60,  40)
}

optimal_moisture = {
    "Sandy": 0.17,   # 17% (vol/vol)
    "Red":   0.17,   # (similar to ‘Sandy’ in most regions)
    "Loamy": 0.25,   # 25%
    "Clayey":0.32,   # 32%
    "Black": 0.32    # (similar to heavy‐textured/clayey)
}


unique_ferts = train_df["Fertilizer Name"].unique()


train_df["Fertilizer_Name_Encoded"] = LabelEncoder().fit_transform(train_df["Fertilizer Name"].astype(str))

def soil_crop_compatibility_score(soil_type, crop_type):
    """
    Retrieves the compatibility score between a soil type and a crop type.
    
    Parameters:
    - soil_type: Type of the soil
    - crop_type: Type of the crop
    - compatibility_matrix: Dictionary with compatibility scores
    
    Returns:
    - SCCS: Compatibility score
    """
    # Compatibility scores (0.0 to 1.0)
    compatibility_matrix = {
        "Black": {
            "Cotton": 1.0,
            "Pulses": 0.9,
            "Oil seeds": 0.85,
            "Sugarcane": 0.8,
            "Wheat": 0.75,
            "Barley": 0.7
        },
        "Clayey": {
            "Paddy": 1.0,
            "Wheat": 0.85,
            "Tobacco": 0.8,
            "Pulses": 0.75
        },
        "Loamy": {
            "Wheat": 1.0,
            "Barley": 0.95,
            "Maize": 0.9,
            "Sugarcane": 0.85,
            "Oil seeds": 0.8,
            "Pulses": 0.75
        },
        "Red": {
            "Millets": 0.9,
            "Ground Nuts": 0.85,
            "Pulses": 0.8,
            "Cotton": 0.75,
            "Tobacco": 0.7
        },
        "Sandy": {
            "Ground Nuts": 1.0,
            "Millets": 0.95,
            "Cotton": 0.9,
            "Pulses": 0.85,
            "Oil seeds": 0.8,
            "Tobacco": 0.75
        }
    }
    return compatibility_matrix.get(soil_type, {}).get(crop_type, 0)

def compute_N_effective(N, crop_type):
    # If crop is a legume, ignore measured N; else use measured N.
    if crop_type in ['Pulses', 'Ground Nuts']:
        return 0.0
    else:
        return N

# Pre-fit encoders for soil and crop
le_soil = LabelEncoder().fit(
    pd.concat([train_df['Soil Type'], test_df['Soil Type']], ignore_index=True).astype(str)
)
le_crop = LabelEncoder().fit(
    pd.concat([train_df['Crop Type'], test_df['Crop Type']], ignore_index=True).astype(str)
)

# Example feature engineering function
def legendary_feature_engineering(df_input):
    df_feat = df_input.copy()
    # Only extract or encode fertilizer name if it exists
    if 'Fertilizer Name' in df_feat.columns:
        # your existing LabelEncoder / NPK‐regex code here
        df_feat['Fertilizer_Name_Encoded'] = LabelEncoder().fit_transform(
            df_feat['Fertilizer Name'].astype(str)
        )
    else:
        # for test rows, either fill with zeros or a default
        df_feat['Fertilizer_Name_Encoded'] = 0

    df_feat["Soil_Type_Encoded"] = le_soil.transform(df_feat["Soil Type"].astype(str))
    df_feat["Crop_Type_Encoded"] = le_crop.transform(df_feat["Crop Type"].astype(str))
     # Omega -> Beta Feature Engineering

    df_feat["Total_Nutrients"] = df_feat["Nitrogen"] + df_feat["Potassium"] + df_feat["Phosphorous"]
    df_feat["Env_Index"] = (df_feat["Temparature"] + df_feat["Humidity"]) / 2
    df_feat["Wet_Index"] = (7*df_feat["Humidity"] + 3*df_feat["Moisture"]) / 10

    df_feat["Humidity_norm"] = (df_feat["Humidity"] - df_feat["Humidity"].mean()) / df_feat["Humidity"].std()
    df_feat["Moisture_norm"] = (df_feat["Moisture"] - df_feat["Moisture"].mean()) / df_feat["Moisture"].std()
    df_feat["Wet_Index_norm"] = df_feat["Humidity_norm"] + df_feat["Moisture_norm"]
    
    df_feat["Drought_Stress"] = df_feat["Temparature"] / (df_feat["Humidity"] + 1e-5)
    df_feat["N_P_Ratio"]      = df_feat["Nitrogen"]    / (df_feat["Phosphorous"] + 1e-5)
    df_feat["N_K_Ratio"]      = df_feat["Nitrogen"]    / (df_feat["Potassium"] + 1e-5)
    df_feat["K_P_Ratio"]      = df_feat["Potassium"]    / (df_feat["Phosphorous"] + 1e-5)
    df_feat["K_N_Ratio"]      = df_feat["Potassium"]   / (df_feat["Nitrogen"] + 1e-5)
    df_feat["P_K_Ratio"]      = df_feat["Phosphorous"]    / (df_feat["Potassium"] + 1e-5)
    df_feat["P_N_Ratio"]      = df_feat["Phosphorous"]    / (df_feat["Nitrogen"] + 1e-5)
    
    df_feat["Temp_Humidity_Interaction"]     = df_feat["Temparature"] * df_feat["Humidity"]
    df_feat["Moisture_Humidity_Interaction"] = df_feat["Moisture"]    * df_feat["Humidity"]
    df_feat["Temp_Moisture_Interaction"]     = df_feat["Moisture"]    * df_feat["Temparature"]
    
    df_feat["Moisture_Nitrogen_Interaction"] = df_feat["Moisture"]    * df_feat["Nitrogen"]
    df_feat["Moisture_Potassium_Interaction"] = df_feat["Moisture"]    * df_feat["Potassium"]
    df_feat["Moisture_Phosphorous_Interaction"] = df_feat["Moisture"]    * df_feat["Phosphorous"]
    
    df_feat["Nutrient_Mean"] = df_feat[["Nitrogen", "Potassium", "Phosphorous"]].mean(axis=1)
    
    # Additional Features
    df_feat["Moisture_Adjusted_N"] = df_feat["Nitrogen"] / (df_feat["Moisture"] + 1e-5)
    df_feat["Moisture_Adjusted_K"] = df_feat["Potassium"] / (df_feat["Moisture"] + 1e-5)
    df_feat["Moisture_Adjusted_P"] = df_feat["Phosphorous"] / (df_feat["Moisture"] + 1e-5
                                                    )
    df_feat["Humidity_Adjusted_N"] = df_feat["Nitrogen"] / (df_feat["Humidity"] + 1e-5)
    df_feat["Humidity_Adjusted_K"] = df_feat["Potassium"] / (df_feat["Humidity"] + 1e-5)
    df_feat["Humidity_Adjusted_P"] = df_feat["Phosphorous"] / (df_feat["Humidity"] + 1e-5)
    
    df_feat["Env_Nutrient_Sum"]    = df_feat["Temparature"] + df_feat["Humidity"] + df_feat["Moisture"] + df_feat["Total_Nutrients"]
    df_feat["Env_Nutrient_Mean"]   = df_feat[["Temparature", "Humidity", "Moisture", "Nitrogen", "Phosphorous", "Potassium"]].mean(axis=1)
    
    # ---------
    # Alpha Feature Engineering

    # Α.1: THI (Temperature Humidity Index)
    df_feat["Temparture_Humidity_Index"] = 0.8 * df_feat["Temparature"] + (df_feat["Humidity"]/100) * (df_feat["Temparature"] - 14.4) + 46.4
    # Α.2: Nutrient Efficiency Ratio
    df_feat["Nutrient_Efficiency_Ratio"] = 2 * (df_feat["Nitrogen"] + df_feat["Potassium"] + df_feat["Phosphorous"]) / (df_feat["Temparature"] + df_feat["Humidity"])
    
    # Convert optimal_npk to DataFrame with proper index
    opt_npk_df = pd.DataFrame.from_dict(optimal_npk, orient='index', columns=['Opt_N', 'Opt_P', 'Opt_K'])
    opt_npk_df.index.name = 'Crop Type'
    
    # Reset index to merge safely
    opt_npk_df = opt_npk_df.reset_index()
    
    # Merge optimal NPK values based on crop type
    df_feat = df_feat.merge(opt_npk_df, on="Crop Type", how="left")
    
    # Compute NBI
    df_feat["NBI"] = np.sqrt(
        (df_feat["Nitrogen"] - df_feat["Opt_N"]) ** 2 +
        (df_feat["Phosphorous"] - df_feat["Opt_P"]) ** 2 +
        (df_feat["Potassium"] - df_feat["Opt_K"]) ** 2
    )
    
    # Map optimal moisture and compute SMD
    df_feat["Opt_Moisture"] = df_feat["Soil Type"].map(optimal_moisture)
    df_feat["SMD"] = df_feat["Opt_Moisture"] - df_feat["Moisture"]
    
    # Optionally drop intermediate columns
    df_feat.drop(columns=["Opt_N", "Opt_P", "Opt_K", "Opt_Moisture"], inplace=True)



    # ---------
    # Sigma Feature Engineering

    # Σ.1 Legume Nitrogen Fixation Factor
    df_feat["N_effective"] = df_feat.apply(
            lambda row: compute_N_effective(row["Nitrogen"], row["Crop Type"]),
            axis=1
        )
    # Σ.2 Compute retention_factor = CEC / 50 (normalize by an approximate max of 50)
    df_feat["retention_factor"] = df_feat["Soil Type"].map(lambda s: soil_CEC.get(s, 0.0)) / 50.0

    # Σ.3 Now create “retained” nutrient columns
    df_feat["N_retained"] = df_feat["Nitrogen"] * df_feat["retention_factor"]
    df_feat["P_retained"] = df_feat["Phosphorous"] * df_feat["retention_factor"]
    df_feat["K_retained"] = df_feat["Potassium"] * df_feat["retention_factor"]

    df_feat["Total_Nutrients_retained"] = (df_feat["N_retained"] + df_feat["P_retained"] + df_feat["K_retained"])

    # Σ.4 Growing Degree Days (GDD)
    df_feat["T_base"] = df_feat["Crop Type"].map(lambda c: T_base.get(c, 0))
    df_feat["GDD"] = (df_feat["Temparature"] - df_feat["T_base"]).clip(lower=0.0)

    # Σ.5 Nutrient Proportions (Stoichiometric Fractions)
    df_feat["sumNPK"] = df_feat["Nitrogen"] + df_feat["Phosphorous"] + df_feat["Potassium"] + 1e-5
    df_feat["N_frac"] = df_feat["Nitrogen"] / df_feat["sumNPK"]
    df_feat["P_frac"] = df_feat["Phosphorous"] / df_feat["sumNPK"]
    df_feat["K_frac"] = df_feat["Potassium"] / df_feat["sumNPK"]

    # Σ.6 One‐Hot Soil & Crop Indicators
    soil_dummies = pd.get_dummies(df_feat["Soil Type"], prefix="soil")
    crop_dummies = pd.get_dummies(df_feat["Crop Type"], prefix="crop")
    df_feat[soil_dummies.columns] = soil_dummies
    df_feat[crop_dummies.columns] = crop_dummies
    
    # Σ.7 Total Retained Nutrient Availability
    df_feat["Env_Nutrient_Availability"] = (
        df_feat["Temparature"] 
        + df_feat["Humidity"] 
        + df_feat["Moisture"] 
        + df_feat["Total_Nutrients_retained"]
    )

    # Σ.8 Relative Soil Moisture
    df_feat["field_capacity"] = df_feat["Soil Type"].map(lambda s: fc_lookup.get(s, 0.25))
    df_feat["Moisture_Index"] = df_feat["Moisture"] / (df_feat["field_capacity"] + 1e-5)

    df_feat.drop(columns=["Soil Type", "Crop Type"], inplace=True)

    return df_feat

# Apply feature engineering
def_train = legendary_feature_engineering(train_df)
def_test  = legendary_feature_engineering(test_df)

# 4) Drop raw IDs & types
drop_cols = ['Fertilizer Name', 'id', 'Soil Type', 'Crop Type']
def_train.drop(columns=drop_cols, errors='ignore', inplace=True)
def_test.drop(columns=drop_cols, errors='ignore', inplace=True)

# 5) Prepare features and target
X_full = def_train.drop(columns='target')
y_full = def_train['target'].values
X_test = def_test.copy()

# 6) Scale features
common_cols = [c for c in def_train.columns if c != 'target']
X_full = def_train[common_cols]
X_test = def_test[common_cols]

scaler = StandardScaler()
X_scaled      = scaler.fit_transform(X_full)
X_test_scaled = scaler.transform(X_test)   # now columns match exactly

# 7) Feature selection via RandomForest
fs_rf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
y_full = y_full.astype(np.int8) # ensure correct dtype
# Sample 100,000 rows randomly
sample_indices = np.random.choice(len(X_scaled), 300_000, replace=False) #avoid memory overload
X_sample = X_scaled[sample_indices]
y_sample = y_full[sample_indices]

# Fit RF for feature importance
fs_rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=1)
fs_rf.fit(X_sample, y_sample)

importances   = fs_rf.feature_importances_
feature_names = X_full.columns
idxs = np.argsort(importances)[::-1][:30]  # top 30
selected_feats = feature_names[idxs]
print("Selected features:", list(selected_feats))

# 8) Reduce dataset
X_scaled = X_scaled.astype(np.float32)
X_sel      = X_scaled[:, idxs]
X_test_sel = X_test_scaled[:, idxs]

# 9) Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_sel, y_full, test_size=0.1, stratify=y_full, random_state=42
)

# 10) Train RF and record val_acc
rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=100)
rf_model.fit(X_train, y_train)
rf_val_acc = rf_model.score(X_val, y_val)
print(f"RF val_acc: {rf_val_acc:.4f}")

# 11) DataLoaders for NN
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=64, shuffle=True, num_workers=0)
val_loader   = DataLoader(TabularDataset(X_val, y_val), batch_size=128, shuffle=False, num_workers=0)
test_loader  = DataLoader(TabularDataset(X_test_sel), batch_size=128, shuffle=False, num_workers=0)

# 12) MLP definition
class FertMLP(nn.Module):
    def __init__(self, n_feats, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)
# 13) Train MLP
mlp = FertMLP(len(selected_feats), len(le_target.classes_)).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

best_nn = 0
for ep in range(1,31):
    mlp.train(); tl=0
    for xb,yb in tqdm(train_loader,desc='Train'):
        xb,yb=xb.to(device),yb.to(device)
        opt.zero_grad(); l=loss_fn(mlp(xb),yb); l.backward(); opt.step(); tl+=l.item()
    mlp.eval(); cor=0; tot=0
    with torch.inference_mode():
        for xb,yb in val_loader:
            xb,yb=xb.to(device),yb.to(device)
            pr=mlp(xb).argmax(1); cor+=(pr==yb).sum().item(); tot+=xb.size(0)
    acc=cor/tot
    best_nn=max(best_nn,acc)
    print(f"Ep{ep:02d}|tr_loss={tl/len(train_loader):.3f}|val_acc={acc:.3f}")
print("Best NN acc:",best_nn)

# 14) Inference choose best
use_rf = rf_val_acc>=best_nn
print("Using RF" if use_rf else "Using NN")
if use_rf:
    probs=rf_model.predict_proba(X_test_sel)
else:
    mlp.load_state_dict(torch.load('best_mlp.pt',map_location=device))
    mlp.eval(); ps=[]
    with torch.inference_mode():
        for xb in test_loader: ps.append(mlp(xb.to(device)).softmax(1).cpu().numpy())
    probs=np.vstack(ps)
ids=test_df.index.values+750000
preds=np.argsort(probs,1)[:,-3:][:,::-1]
names=[" ".join(le_target.inverse_transform(p)) for p in preds]
pd.DataFrame({'id':ids,'Fertilizer Name':names}).to_csv('submission.csv',index=False)
print('saved.')

