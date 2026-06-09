# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
dataset1 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")
dataset2 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv")
dataset3 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
dataset4 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")


def dataset_info(name, df, show_head=True):
    print(f"--- {name} ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nEksik deÄŸer sayÄ±sÄ±:")
    print(df.isnull().sum())
    if show_head:
        print("\nHead:")
        print(df.head())
    print("\n" + "="*50 + "\n")

# Ä°nceleme
dataset_info("Train", train)
dataset_info("Test", test)
dataset_info("Dataset1", dataset1)
dataset_info("Dataset2", dataset2, show_head=False)  # dataset2 bÃ¼yÃ¼k ve sadece SMILES, istersen head gÃ¶stermeyebiliriz
dataset_info("Dataset3", dataset3)
dataset_info("Dataset4", dataset4)


train['FFV'].value_counts(dropna=False)


def missing_data_report(df, name):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    print(f"\n--- {name} ---")
    print("Eksik veri sayÄ±larÄ±:")
    print(missing_count)
    print("\nEksik veri oranlarÄ± (%):")
    print(missing_percent)

# Veri setlerini listele
datasets = [
    (train, "Train"),
    (test, "Test"),
    (dataset1, "Dataset1"),
    (dataset2, "Dataset2"),
    (dataset3, "Dataset3"),
    (dataset4, "Dataset4")
]

# Her veri seti iÃ§in eksik veriyi yazdÄ±r
for df, name in datasets:
    missing_data_report(df, name)



import kagglehub
download_path = kagglehub.dataset_download("senkin13/rdkit-2025-3-3-cp311")
print("Path to dataset files", download_path)


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl



from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem,Lipinski, rdMolDescriptors
from rdkit.Chem import Draw
from IPython.display import display




def check_smiles_validity(smiles_list):
    invalid_smiles = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles.append((idx, smi))
    return invalid_smiles

# Her dataset iÃ§in geÃ§ersiz SMILES kontrolÃ¼
invalid_train = check_smiles_validity(train["SMILES"])
invalid_test = check_smiles_validity(test["SMILES"])
invalid_dataset1 = check_smiles_validity(dataset1["SMILES"])
invalid_dataset2 = check_smiles_validity(dataset2["SMILES"])
invalid_dataset3 = check_smiles_validity(dataset3["SMILES"])
invalid_dataset4 = check_smiles_validity(dataset4["SMILES"])

# SonuÃ§larÄ± yazdÄ±r
print(f"\nGeÃ§ersiz SMILES sayÄ±sÄ± (train): {len(invalid_train)}")
print(f"GeÃ§ersiz SMILES sayÄ±sÄ± (test): {len(invalid_test)}")
print(f"GeÃ§ersiz SMILES sayÄ±sÄ± (dataset1): {len(invalid_dataset1)}")
print(f"GeÃ§ersiz SMILES sayÄ±sÄ± (dataset2): {len(invalid_dataset2)}")
print(f"GeÃ§ersiz SMILES sayÄ±sÄ± (dataset3): {len(invalid_dataset3)}")
print(f"GeÃ§ersiz SMILES sayÄ±sÄ± (dataset4): {len(invalid_dataset4)}")

# Ä°stersen Ã¶rnek invalid SMILES'leri de gÃ¶sterelim
if invalid_train:
    print("\nÃ–rnek geÃ§ersiz SMILES (train):", invalid_train[:5])
if invalid_test:
    print("\nÃ–rnek geÃ§ersiz SMILES (test):", invalid_test[:5])
if invalid_dataset1:
    print("\nÃ–rnek geÃ§ersiz SMILES (dataset1):", invalid_dataset1[:5])
if invalid_dataset2:
    print("\nÃ–rnek geÃ§ersiz SMILES (dataset2):", invalid_dataset2[:5])
if invalid_dataset3:
    print("\nÃ–rnek geÃ§ersiz SMILES (dataset3):", invalid_dataset3[:5])
if invalid_dataset4:
    print("\nÃ–rnek geÃ§ersiz SMILES (dataset4):", invalid_dataset4[:5])



import matplotlib.pyplot as plt
import seaborn as sns

targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

for col in targets:
    if col in train.columns:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.histplot(train[col], kde=True)
        plt.title(f"{col} Histogram")
        
        plt.subplot(1,2,2)
        sns.boxplot(x=train[col])
        plt.title(f"{col} Boxplot")
        
        plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(train[targets].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Hedef DeÄŸiÅŸken Korelasyonu")
plt.show()



from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys, RDKFingerprint

import pandas as pd
import numpy as np

def mol_from_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except:
        return None


def compute_full_features(df, smi_col='SMILES', morgan_bits=2048):
    print(f"Computing features for {len(df)} molecules...")
    mols = df[smi_col].apply(lambda s: mol_from_smiles(s))
    
    # 1) RDKit descriptors - daha gÃ¼venli hesaplama
    desc_names = [d[0] for d in Descriptors._descList]
    desc_data = []
    
    for i, m in enumerate(mols):
        if i % 1000 == 0:
            print(f"Processing molecule {i}/{len(mols)}")
            
        if m is None:
            desc_data.append([np.nan]*len(desc_names))
        else:
            vals = []
            for name, func in Descriptors._descList:
                try:
                    val = func(m)
                    # Sonsuz deÄŸerleri kontrol et
                    if np.isinf(val) or np.isnan(val):
                        vals.append(0.0)
                    else:
                        vals.append(val)
                except:
                    vals.append(0.0)
            desc_data.append(vals)
    
    desc_df = pd.DataFrame(desc_data, columns=desc_names, index=df.index)
    
    # 2) Morgan fingerprint radius=2
    morgan_fp2 = []
    for m in mols:
        if m is None:
            morgan_fp2.append(np.zeros(morgan_bits, dtype=np.int8))
        else:
            try:
                bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, nBits=morgan_bits)
                arr = np.zeros((morgan_bits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                morgan_fp2.append(arr)
            except:
                morgan_fp2.append(np.zeros(morgan_bits, dtype=np.int8))
    
    morgan_fp2_df = pd.DataFrame(morgan_fp2, columns=[f"MFp2_{i}" for i in range(morgan_bits)], index=df.index)
    
    # 3) Morgan fingerprint radius=3
    morgan_fp3 = []
    for m in mols:
        if m is None:
            morgan_fp3.append(np.zeros(1024, dtype=np.int8))
        else:
            try:
                bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=3, nBits=1024)
                arr = np.zeros((1024,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                morgan_fp3.append(arr)
            except:
                morgan_fp3.append(np.zeros(1024, dtype=np.int8))
    
    morgan_fp3_df = pd.DataFrame(morgan_fp3, columns=[f"MFp3_{i}" for i in range(1024)], index=df.index)
    
    # 4) MACCS Keys
    maccs = []
    for m in mols:
        if m is None:
            maccs.append(np.zeros(167, dtype=np.int8))
        else:
            try:
                bv = MACCSkeys.GenMACCSKeys(m)
                arr = np.zeros((167,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                maccs.append(arr)
            except:
                maccs.append(np.zeros(167, dtype=np.int8))
    
    maccs_df = pd.DataFrame(maccs, columns=[f"MACCS_{i}" for i in range(167)], index=df.index)
    
    # 5) RDKit fingerprint
    rdkit_fp = []
    for m in mols:
        if m is None:
            rdkit_fp.append(np.zeros(2048, dtype=np.int8))
        else:
            try:
                bv = RDKFingerprint(m, fpSize=2048)
                arr = np.zeros((2048,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                rdkit_fp.append(arr)
            except:
                rdkit_fp.append(np.zeros(2048, dtype=np.int8))
    
    rdkit_fp_df = pd.DataFrame(rdkit_fp, columns=[f"RDKFP_{i}" for i in range(2048)], index=df.index)
    
    # 6) BirleÅŸtir
    features = pd.concat([desc_df, morgan_fp2_df, morgan_fp3_df, maccs_df, rdkit_fp_df], axis=1)
    
    # NaN ve inf kontrolÃ¼
    print(f"NaN values before cleaning: {features.isnull().sum().sum()}")
    print(f"Inf values before cleaning: {np.isinf(features.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Temizleme
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)  # NaN'larÄ± 0 ile doldur
    
    print(f"Final feature shape: {features.shape}")
    return features


       



import warnings
from rdkit import RDLogger

# RDKit uyarÄ±larÄ±nÄ± kapat
RDLogger.DisableLog('rdApp.*')

# Python uyarÄ±larÄ±nÄ± kapat
warnings.filterwarnings("ignore", category=DeprecationWarning)


train_features = compute_full_features(train)
test_features = compute_full_features(test)
dataset1_features = compute_full_features(dataset1)
dataset3_features = compute_full_features(dataset3)
dataset4_features = compute_full_features(dataset4)




# Duplicate column kontrolÃ¼ ve temizleme
print("Duplicate column kontrolÃ¼...")
print(f"Train duplicate columns: {train.columns.duplicated().sum()}")
print(f"Train features duplicate columns: {train_features.columns.duplicated().sum()}")



# Concat iÅŸlemi - duplicate column'larÄ± kaldÄ±r
train = pd.concat([train, train_features], axis=1)
train = train.loc[:, ~train.columns.duplicated()]

test = pd.concat([test, test_features], axis=1)
test = test.loc[:, ~test.columns.duplicated()]

dataset1 = pd.concat([dataset1, dataset1_features], axis=1)
dataset1 = dataset1.loc[:, ~dataset1.columns.duplicated()]

dataset3 = pd.concat([dataset3, dataset3_features], axis=1)
dataset3 = dataset3.loc[:, ~dataset3.columns.duplicated()]

dataset4 = pd.concat([dataset4, dataset4_features], axis=1)
dataset4 = dataset4.loc[:, ~dataset4.columns.duplicated()]

print(f"Final train shape: {train.shape}")


train.isnull().any().sum()


import lightgbm as lgb

def supervised_impute_target(target_col, train_df, num_boost_round=1000):
    """
    Eksik target deÄŸerlerini supervised LightGBM ile tahmin edip doldurur.
    """
    missing_idx = train_df[train_df[target_col].isnull()].index
    if len(missing_idx) == 0:
        print(f"{target_col}: Eksik deÄŸer yok.")
        return 0
    
    print(f"{target_col}: {len(missing_idx)} eksik deÄŸer bulundu.")
    
    # Feature listesi
    feature_cols = [c for c in train_df.columns if c not in [target_col, 'SMILES']]
    
    # Eksik olmayan veriler
    train_known = train_df[train_df[target_col].notna()].copy()
    X_train = train_known[feature_cols]
    y_train = train_known[target_col].astype(float)
    
    # Veri temizleme
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())
    
    # LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'verbose': -1,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Eksik deÄŸerlerin tahmini
    X_pred = train_df.loc[missing_idx, feature_cols]
    X_pred = X_pred.replace([np.inf, -np.inf], np.nan)
    X_pred = X_pred.fillna(X_pred.median())
    
    preds = model.predict(X_pred, num_iteration=model.best_iteration)
    train_df.loc[missing_idx, target_col] = preds
    
    print(f"{target_col}: {len(missing_idx)} deÄŸer dolduruldu.")
    return len(missing_idx)





targets = ['Tc', 'Tg', 'FFV', 'Density', 'Rg']
for target in targets:
    supervised_impute_target(target, train)



train.isnull().any().sum()


train.shape


test.shape


# DÃ¼zeltilmiÅŸ Feature Selection - Train ve Test iÃ§in tutarlÄ±
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Veri hazÄ±rlama
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
feature_cols = [c for c in train.columns if c not in targets + ["SMILES"]]

print(f"BaÅŸlangÄ±Ã§ feature sayÄ±sÄ±: {len(feature_cols)}")

# X ve y
X = train[feature_cols].copy()
y = train[targets].astype(float).copy()

# Test verisini aynÄ± feature'larla hazÄ±rla
test_clean = test[feature_cols].copy()

# Veri temizleme - HER Ä°KÄ°SÄ°NE DE AYNI Ä°ÅžLEMÄ° YAP
print("Veri temizleme baÅŸlÄ±yor...")

# 1. Infinity kontrolÃ¼ - BOTH
X = X.replace([np.inf, -np.inf], np.nan)
test_clean = test_clean.replace([np.inf, -np.inf], np.nan)
print(f"X NaN after inf replacement: {X.isnull().sum().sum()}")
print(f"Test NaN after inf replacement: {test_clean.isnull().sum().sum()}")

# 2. NaN doldurma - BOTH (Train'den median hesapla, her ikisine uygula)
train_medians = X.median()
X = X.fillna(train_medians)
test_clean = test_clean.fillna(train_medians)  # Train medianÄ± ile doldur!
print(f"X NaN after fillna: {X.isnull().sum().sum()}")
print(f"Test NaN after fillna: {test_clean.isnull().sum().sum()}")

# 3. Extreme value clipping - BOTH
X = X.clip(-1e6, 1e6)
test_clean = test_clean.clip(-1e6, 1e6)

# 4. Constant feature'larÄ± kaldÄ±r - TRAIN'DEN BULLA, HER Ä°KÄ°SÄ°NDEN KALDIR
constant_features = X.columns[X.std() == 0].tolist()
if len(constant_features) > 0:
    print(f"Removing {len(constant_features)} constant features")
    print(f"Constant features: {constant_features[:10]}...")  # Ä°lk 10'unu gÃ¶ster
    
    # HER Ä°KÄ°SÄ°NDEN DE KALDIR
    X = X.drop(columns=constant_features)
    test_clean = test_clean.drop(columns=constant_features)
    
    # Feature listesini gÃ¼ncelle
    feature_cols = [c for c in feature_cols if c not in constant_features]

print(f"Feature count after constant removal: {X.shape[1]}")

# 5. High correlation feature'larÄ± kaldÄ±r - TRAIN'DEN BULLA, HER Ä°KÄ°SÄ°NDEN KALDIR
print("High correlation analysis...")
correlation_matrix = X.corr().abs()
upper_tri = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

high_corr_features = []
for column in upper_tri.columns:
    if any(upper_tri[column] > 0.95):
        high_corr_features.append(column)

if len(high_corr_features) > 0:
    print(f"Removing {len(high_corr_features)} highly correlated features")
    print(f"High corr features: {high_corr_features[:10]}...")  # Ä°lk 10'unu gÃ¶ster
    
    # HER Ä°KÄ°SÄ°NDEN DE KALDIR
    X = X.drop(columns=high_corr_features)
    test_clean = test_clean.drop(columns=high_corr_features)
    
    # Feature listesini gÃ¼ncelle  
    feature_cols = [c for c in feature_cols if c not in high_corr_features]

print(f"Final feature count: {X.shape[1]}")

# Ã–NEMLÄ°: Åžimdi X ve test_clean aynÄ± sÃ¼tunlara sahip!
print(f"X shape: {X.shape}")
print(f"Test shape: {test_clean.shape}")
print(f"X columns == Test columns: {list(X.columns) == list(test_clean.columns)}")

# KaldÄ±rÄ±lan toplam feature sayÄ±sÄ±
removed_features = set(constant_features) | set(high_corr_features)
print(f"Toplam kaldÄ±rÄ±lan unique feature sayÄ±sÄ±: {len(removed_features)}")

# Scaling - AYNI SCALER'I HER Ä°KÄ°SÄ°NE UYGULA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Train'den fit, train'e transform
test_scaled = scaler.transform(test_clean)  # Train fit'ini test'e uygula

# Target scaling
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y)

print("âœ“ TutarlÄ± feature selection ve scaling tamamlandÄ±!")
print(f"âœ“ X_scaled shape: {X_scaled.shape}")
print(f"âœ“ test_scaled shape: {test_scaled.shape}")
print(f"âœ“ y_scaled shape: {y_scaled.shape}")

# Feature selection sonuÃ§larÄ±nÄ± kaydet (debug iÃ§in)
feature_selection_info = {
    'original_feature_count': len([c for c in train.columns if c not in targets + ["SMILES"]]),
    'constant_features_removed': len(constant_features),
    'high_corr_features_removed': len(high_corr_features),
    'final_feature_count': X.shape[1],
    'constant_features_list': constant_features,
    'high_corr_features_list': high_corr_features
}

print(f"\nðŸ“Š FEATURE SELECTION SUMMARY:")
for key, value in feature_selection_info.items():
    if isinstance(value, list) and len(value) > 10:
        print(f"{key}: {len(value)} items (showing first 5: {value[:5]})")
    else:
        print(f"{key}: {value}")


import numpy as np

def weighted_mae(y_true, y_pred):
    """
    Weighted MAE hesapla (5 target iÃ§in)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n_samples, n_properties = y_true.shape
    K = n_properties
    
    weights = []
    for i in range(n_properties):
        valid_mask = ~np.isnan(y_true[:, i])
        ni = np.sum(valid_mask)
        valid_values = y_true[valid_mask, i]
        ri = np.max(valid_values) - np.min(valid_values)
        
        sqrt_inv_n_values = []
        for j in range(n_properties):
            nj = np.sum(~np.isnan(y_true[:, j]))
            sqrt_inv_n_values.append(np.sqrt(1/nj))
        sum_sqrt_inv_n = np.sum(sqrt_inv_n_values)
        
        wi = (1/ri) * (K * np.sqrt(1/ni) / sum_sqrt_inv_n)
        weights.append(wi)
    weights = np.array(weights)
    
    # Weighted MAE
    total_error = np.sum(np.abs(y_true - y_pred) * weights)
    return total_error / n_samples



X_scaled.shape


y.shape


test_scaled.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_wmae = []
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n===== Fold {fold+1} =====")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Model (her foldda yeniden kurulur)
    model = Sequential([
        Dense(512, input_shape=(X_scaled.shape[1],)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),

        Dense(512),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),

       

        Dense(256),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.1),

        Dense(128),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.1),

        Dense(y.shape[1], activation='linear')

        
       
        
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
    )

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)

    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=400,
    batch_size=64,
    verbose=1,            # train ve val lossâ€™u her epoch gÃ¶rebilirsin
    callbacks=[early_stop,reduce_lr] # Early stopping aktif
)


    # Val tahminleri
    y_val_pred = model.predict(X_val, verbose=0)
    wmae_score = weighted_mae(y_val, y_val_pred)
    print(f"Fold {fold+1} wMAE: {wmae_score:.6f}")
    fold_wmae.append(wmae_score)

    # Test tahmini
    fold_pred = model.predict(test_scaled, verbose=0)
    test_preds.append(fold_pred)

print("\nCV Weighted MAE sonuÃ§larÄ±:", fold_wmae)
print("Ortalama wMAE:", np.mean(fold_wmae))

# Test tahminlerini fold'lar Ã¼zerinden ortalama
final_test_pred = np.mean(np.array(test_preds), axis=0)
    



# Basit Submission OluÅŸturma

import pandas as pd
import numpy as np

# Orijinal test verisini oku
test_original = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

# Test ID'lerini al
test_ids = test_original['id'].values  # kÃ¼Ã§Ã¼k 'id' 

# Target kolonlarÄ±
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Submission DataFrame oluÅŸtur
submission = pd.DataFrame()
submission['id'] = test_ids  # kÃ¼Ã§Ã¼k 'id'

# Tahminleri ekle
for i, target in enumerate(target_columns):
    submission[target] = final_test_pred[:, i]

# NaN kontrolÃ¼ ve temizleme
if submission[target_columns].isnull().sum().sum() > 0:
    print("NaN deÄŸerler medyan ile dolduruluyor...")
    for target in target_columns:
        if submission[target].isnull().sum() > 0:
            submission[target].fillna(submission[target].median(), inplace=True)

# CSV olarak kaydet
submission.to_csv('submission.csv', index=False)

print(f"âœ… Submission hazÄ±r!")
print(f"Shape: {submission.shape}")
print(f"Columns: {list(submission.columns)}")
print("\n")
submission.head()

