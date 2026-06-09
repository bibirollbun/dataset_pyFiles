!ls ../input/my-rdkit-wheel



 import glob

wheel_files = glob.glob("../input/my-rdkit-wheel/rdkit_pypi-2022.9.5-*.whl")
if not wheel_files:
    raise FileNotFoundError("â�Œ RDKit wheel not found. Attach dataset correctly.")

wheel_path = wheel_files[0]
print(f"ğŸ“¦ Installing RDKit from: {wheel_path}")
!pip install "{wheel_path}" --quiet

# Verify installation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles("CCO")
    print("âœ… RDKit Installed Successfully! Test MolWt:", Descriptors.MolWt(mol))
except Exception as e:
    print("â�Œ RDKit installation failed:", e)


 import pandas as pd

train = pd.read_csv("../input/neurips-open-polymer-prediction-2025/train.csv")
test  = pd.read_csv("../input/neurips-open-polymer-prediction-2025/test.csv")
sample_submission = pd.read_csv("../input/neurips-open-polymer-prediction-2025/sample_submission.csv")

print("âœ… Train shape:", train.shape)
print("âœ… Test shape:", test.shape)
print("âœ… Sample submission shape:", sample_submission.shape)


 import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import DataStructs

FP_RADIUS = 2
FP_NBITS = 1024

def rdkit_featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, np.zeros(FP_NBITS, dtype=np.uint8)
    
    desc = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRings": Chem.GetSSSR(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
    }
    
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    arr = np.zeros(FP_NBITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    return desc, arr

def build_features(df):
    descs, fps = [], []
    for s in df['SMILES']:
        d, a = rdkit_featurize(s)
        descs.append(d)
        fps.append(a)
    
    X_desc = pd.DataFrame(descs).fillna(0)
    X_fp = np.array(fps, dtype=np.uint8)
    X_all = pd.concat([X_desc, pd.DataFrame(X_fp)], axis=1)
    return X_all

X_train = build_features(train)
X_test  = build_features(test)

print("âœ… Feature engineering complete!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce')
X_test_numeric  = X_test.apply(pd.to_numeric, errors='coerce')

X_train_numeric.fillna(0, inplace=True)
X_test_numeric.fillna(0, inplace=True)

X_train_numeric.columns = X_train_numeric.columns.astype(str)
X_test_numeric.columns  = X_test_numeric.columns.astype(str)

print("âœ… Numeric preprocessing done!")
  


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

TARGETS = ["Tg","FFV","Tc","Density","Rg"]
preds = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for t in TARGETS:
    print(f"ğŸ�¯ Training target: {t}")
    y_full = train[t].values
    not_nan_idx = ~np.isnan(y_full)
    y = y_full[not_nan_idx]
    X = X_train_numeric.iloc[not_nan_idx].copy()
    
    fold_preds = np.zeros(len(X_test_numeric))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_tr, y_tr)
        fold_preds += model.predict(X_test_numeric) / kf.n_splits
    
    fold_preds = np.clip(fold_preds, 0, None)
    preds[t] = fold_preds
    print(f"âœ… Target {t} done.")



submission = sample_submission.copy()

for t in TARGETS:
    submission[t] = preds[t].astype(float)

submission.fillna(0, inplace=True)
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file ready! Shape:", submission.shape)


