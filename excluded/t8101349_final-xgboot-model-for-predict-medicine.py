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



!pip install pandas
!pip install tqdm
!pip install pickle
!pip install rdkit
!pip install numpy
     


# åˆ†å¸ƒçµ±è¨ˆ
import pandas as pd
from tqdm import tqdm
import pickle

# å�ƒæ•¸è¨­å®š
filename = "/kaggle/input/leash-BELKA/train.csv"
chunksize = 1_000_000
bb_columns = ["buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles"]

# è¨ˆç®—åˆ†å±¤åˆ†å¸ƒ
def compute_strata_counts(bb_col):
    print(f"ğŸ”� çµ±è¨ˆ {bb_col} åˆ†å±¤åˆ†å¸ƒ...")
    strata_counts = {"pos": {}, "neg": {}}
    for chunk in tqdm(pd.read_csv(filename, chunksize=chunksize, usecols=[bb_col, "binds"]), desc=f"Counting {bb_col}"):
        for bb, group in chunk.groupby(bb_col):
            pos_count = len(group[group["binds"] == 1])
            neg_count = len(group[group["binds"] == 0])
            strata_counts["pos"][bb] = strata_counts["pos"].get(bb, 0) + pos_count
            strata_counts["neg"][bb] = strata_counts["neg"].get(bb, 0) + neg_count
    total_pos = sum(strata_counts["pos"].values())
    total_neg = sum(strata_counts["neg"].values())
    return {"strata_counts": strata_counts, "total_pos": total_pos, "total_neg": total_neg}

# åŸ·è¡Œä¸¦å„²å­˜
strata_data = {}
for bb_col in bb_columns:
    strata_data[bb_col] = compute_strata_counts(bb_col)
    print(f"{bb_col} - ç¸½æ­£é¡�: {strata_data[bb_col]['total_pos']}, ç¸½è² é¡�: {strata_data[bb_col]['total_neg']}")

# å„²å­˜é �è™•ç�†è³‡æ–™
with open("strata_data.pkl", "wb") as f:
    pickle.dump(strata_data, f)
print("âœ… åˆ†å¸ƒçµ±è¨ˆå·²å„²å­˜è‡³ 'strata_data.pkl'")
     

# æŠ½æ¨£
import pandas as pd
from tqdm import tqdm
import pickle

# è¼‰å…¥é �è™•ç�†è³‡æ–™
with open("strata_data.pkl", "rb") as f:
    strata_data = pickle.load(f)




# å�ƒæ•¸è¨­å®š å�¯èª¿æ•´ç‰¹å¾µæŠ½æ¨£æ¯”ä¾‹èˆ‡æ­£è² é¡�æ¯”ä¾‹
filename = "/kaggle/input/leash-BELKA/train.csv"
chunksize = 1_000_000
targets = [
    {"bb": "buildingblock1_smiles", "pos": 80000, "neg": 240000}, # ç‰¹å¾µä¸€
    {"bb": "buildingblock2_smiles", "pos": 10000, "neg": 30000}, # ç‰¹å¾µäºŒ
    {"bb": "buildingblock3_smiles", "pos": 10000, "neg": 30000}, # ç‰¹å¾µä¸‰
]

# æŠ½æ¨£å‡½æ•¸
def stratified_sample(bb_col, pos_target, neg_target, strata_data):
    strata_counts = strata_data[bb_col]["strata_counts"]
    total_pos = strata_data[bb_col]["total_pos"]
    total_neg = strata_data[bb_col]["total_neg"]

    print(f"ğŸ�² é€²è¡Œ {bb_col} åˆ†å±¤æŠ½æ¨£...")
    pos_samples = []
    neg_samples = []
    required_cols = ["molecule_smiles", "buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles", "protein_name", "binds"]

    for chunk in tqdm(pd.read_csv(filename, chunksize=chunksize, usecols=required_cols), desc=f"Sampling {bb_col}"):
        for bb, group in chunk.groupby(bb_col):
            pos_chunk = group[group["binds"] == 1]
            neg_chunk = group[group["binds"] == 0]

            pos_size = min(len(pos_chunk), int(pos_target * (strata_counts["pos"].get(bb, 0) / total_pos)))
            neg_size = min(len(neg_chunk), int(neg_target * (strata_counts["neg"].get(bb, 0) / total_neg)))

            if pos_size > 0 and len(pos_samples) < pos_target:
                pos_sample = pos_chunk.sample(n=min(pos_size, pos_target - len(pos_samples)), random_state=42)
                pos_samples.append(pos_sample)

            if neg_size > 0 and len(neg_samples) < neg_target:
                neg_sample = neg_chunk.sample(n=min(neg_size, neg_target - len(neg_samples)), random_state=42)
                neg_samples.append(neg_sample)

        if len(pos_samples) >= pos_target and len(neg_samples) >= neg_target:
            break

    df = pd.concat(pos_samples + neg_samples, ignore_index=True)
    df_pos = df[df["binds"] == 1].sample(n=min(pos_target, len(df[df["binds"] == 1])), random_state=42)
    df_neg = df[df["binds"] == 0].sample(n=min(neg_target, len(df[df["binds"] == 0])), random_state=42)
    return pd.concat([df_pos, df_neg], ignore_index=True)

# åŸ·è¡Œåˆ†å±¤æŠ½æ¨£
train_dfs = []
for target in targets:
    df = stratified_sample(target["bb"], target["pos"], target["neg"], strata_data)
    train_dfs.append(df)

# å�ˆä½µæ¨£æœ¬
train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42)

# æª¢æŸ¥çµ�æ�œ
print("ğŸ”� æª¢æŸ¥å»ºæ§‹å¡Šåˆ†å¸ƒ...")
bb1_unique = train_df["buildingblock1_smiles"].nunique()
bb2_unique = train_df["buildingblock2_smiles"].nunique()
bb3_unique = train_df["buildingblock3_smiles"].nunique()

print(f"ç¸½æ¨£æœ¬æ•¸: {len(train_df)}")
print(f"æ­£é¡�è¨˜éŒ„æ•¸: {len(train_df[train_df['binds'] == 1])}")
print(f"è² é¡�è¨˜éŒ„æ•¸: {len(train_df[train_df['binds'] == 0])}")
print(f"ç�¨ç‰¹åˆ†å­�æ•¸: {train_df['molecule_smiles'].nunique()}")
print(f"buildingblock1_smilesç›¸ç•°è¨ˆæ•¸: {bb1_unique}ï¼ˆå�Ÿå§‹271ï¼‰")
print(f"buildingblock2_smilesç›¸ç•°è¨ˆæ•¸: {bb2_unique}ï¼ˆå�Ÿå§‹693ï¼‰")
print(f"buildingblock3_smilesç›¸ç•°è¨ˆæ•¸: {bb3_unique}ï¼ˆå�Ÿå§‹872ï¼‰")

# å„²å­˜çµ�æ�œ
train_df.to_csv("1030_40_data.csv", index=False)
     


import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from joblib import Parallel, delayed

# å®šç¾© SMILES è½‰æ�›å‡½æ•¸
def smiles_to_morgan_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    else:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(generator.GetFingerprint(mol), dtype=int)

# ä¸¦è¡Œè™•ç�† SMILES è½‰æ�›
def parallel_smiles_conversion(smiles_series, n_jobs=4):
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(smiles_to_morgan_fingerprint)(smiles) for smiles in smiles_series
    )
    return results

# è¼‰å…¥æª”æ¡ˆ
input_file = '/kaggle/working/1030_40_data.csv'
df_test = pd.read_csv(input_file)

# åˆ†æ‰¹è™•ç�†å�ƒæ•¸
batch_size = 100_000  # æ¯�æ‰¹è™•ç�† 10 è�¬ç­†ï¼Œå�¯æ ¹æ“šè¨˜æ†¶é«”æƒ…æ³�èª¿æ•´
n_batches = (len(df_test) + batch_size - 1) // batch_size

# å„²å­˜ä¸­é–“çµ�æ�œ
output_X_dir = 'temp_X_batches'
os.makedirs(output_X_dir, exist_ok=True)

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(df_test))
    batch_df = df_test.iloc[start_idx:end_idx].copy()

    print(f"Processing batch {i+1}/{n_batches} ({start_idx} to {end_idx})")

    # å°�ç•¶å‰�æ‰¹æ¬¡çš„ "molecule_smiles" é€²è¡Œè½‰æ�›
    batch_df["molecule_smiles"] = parallel_smiles_conversion(batch_df["molecule_smiles"], n_jobs=4)

    # è½‰æ�›ç‚ºæŒ‡ç´‹æ•¸æ“šæ¡†
    fingerprints_df = pd.DataFrame(batch_df['molecule_smiles'].to_list())
    protein_onehot = pd.get_dummies(batch_df["protein_name"], prefix="protein").astype(int).reset_index(drop=True)
    X_batch = pd.concat([fingerprints_df, protein_onehot], axis=1)
    X_batch.columns = X_batch.columns.astype(str)

    # X è½‰æˆ� int8
    int_cols = X_batch.select_dtypes(include=['int64']).columns
    for col in int_cols:
        X_batch[col] = X_batch[col].astype(np.int8)

    # å„²å­˜ç•¶å‰�æ‰¹æ¬¡åˆ°è‡¨æ™‚æª”æ¡ˆ
    batch_file = os.path.join(output_X_dir, f'X_batch_{i}.parquet')
    X_batch.to_parquet(batch_file)

    # æ¸…ç�†è¨˜æ†¶é«”
    del batch_df, fingerprints_df, protein_onehot, X_batch

# å�ˆä½µæ‰€æœ‰ X æ‰¹æ¬¡
X_test = pd.concat([pd.read_parquet(os.path.join(output_X_dir, f))
                   for f in os.listdir(output_X_dir) if f.endswith('.parquet')],
                   axis=0)

# è™•ç�† y
df_test['binds'] = df_test['binds'].astype(np.int8)
y_test = df_test['binds'].reset_index(drop=True)  # é‡�ç½®ç´¢å¼•ç‚ºé€£çºŒçš„ RangeIndex

from sklearn.utils import shuffle

# æ‰“äº‚æ•¸æ“šï¼Œä½†ä¿�æŒ� X å’Œ y çš„å°�æ‡‰é—œä¿‚
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# å†�æ¬¡å„²å­˜ç‚º Parquet
X_test.to_parquet('mg1030_X.parquet', index=False)
y_test.to_frame().to_parquet('mg1030_y.parquet', index=False)

'''
# å„²å­˜ X_test ç‚º Parquet æª”æ¡ˆ
X_test.to_parquet('mg1030_X.parquet', index=False)

# å°‡ y_test è½‰æ�›ç‚º DataFrame ä¸¦å„²å­˜ç‚º Parquet æª”æ¡ˆ
y_test.to_frame().to_parquet('mg1030_y.parquet', index=False)
'''

# å�¯é�¸ï¼šæ¸…ç�†è‡¨æ™‚ç›®éŒ„
import shutil
shutil.rmtree('temp_X_batches')

# restart kernal é‡‹æ”¾è¨˜æ†¶é«”


X_train = pd.read_parquet('/kaggle/working/mg1030_X.parquet')
y_train = pd.read_parquet('/kaggle/working/mg1030_y.parquet')


X_train.shape


# -------------------- XGBoost --------------------
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint
# æœ€ä½³ param
xgb_model = XGBClassifier(colsample_bytree=0.7, gamma=0.3, learning_rate=0.5,
                          max_depth=11, n_estimators=258, reg_alpha=0, reg_lambda=10,
                          subsample=1.0)
xgb_model.fit(X_train, y_train)



#å„²å­˜æ¨¡å�‹
import pickle
with open("/kaggle/working/15_05_new_xgb_model.bin", "wb") as f:
    pickle.dump(xgb_model, f)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

import duckdb
import pandas as pd
from tqdm import tqdm
import numpy as np # linear algebra



import pickle
xg =  open("/kaggle/input/new-xgb/pytorch/default/1/15_05_new_xgb_model.bin", "rb")
xgb_15_05_model =  pickle.load(xg) #è¼‰å…¥model
xgb_15_05_model


def smiles_to_morgan_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    else:
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(generator.GetFingerprint(mol), dtype=int)



import os

# Process the test.parquet file chunk by chunk
test_file = '/kaggle/input/leash-BELKA/test.csv' #è¼‰å…¥æª”æ¡ˆå��ç¨±

df_test = pd.read_csv(test_file)


df_test.shape


from rdkit.Chem import AllChem
from rdkit import Chem

output_file = 'submission15_05_.csv'  # è¼¸å‡ºæª”æ¡ˆå��ç¨±

# Read the test.parquet file into a pandas DataFrame
for df_test in pd.read_csv(test_file, chunksize=104681):
    
    
    # å°� "molecule_smiles" æ¬„ä½�é€²è¡Œè½‰æ�›ä¸¦é¡¯ç¤ºé€²åº¦æ¢�
    tqdm.pandas(desc="Transforming molecule_smiles")
    df_test["molecule_smiles"] = df_test["molecule_smiles"].progress_apply(lambda x: smiles_to_morgan_fingerprint(x))
    df_test.columns = df_test.columns.astype(str)
    
    
    # è½‰æˆ�int8
    int_cols = df_test.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df_test[col] = df_test[col].astype(np.int8)
    
    
    
    fingerprints_df = pd.DataFrame(df_test['molecule_smiles'].to_list())
    print(f"fingerprints_df shape: {fingerprints_df.shape}")  # æ‡‰è©²æ˜¯ (104681, 2048)
    
    protein_onehot = pd.get_dummies(df_test["protein_name"], prefix="protein").astype(int).reset_index(drop=True)
    print(f"protein_onehot shape: {protein_onehot.shape}")  # æ‡‰è©²æ˜¯ (104681, X)
    
    X_test = pd.concat([fingerprints_df, protein_onehot], axis=1)
    print(f"X_test shape: {X_test.shape}")  # æ‡‰è©²æ˜¯ (104681, 2048 + X)

    
    print(X_test)
    
    # Predict the probabilities
    probabilities = xgb_15_05_model.predict_proba(X_test)[:, 1]
    
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)
    
    # ç”¢ç”Ÿæ–°çš„ idï¼Œç¯„åœ�å¾� 295246830 åˆ° 296921725
    df_test['id'] = range(295246830, 295246830 + len(df_test))
    
    # å»ºç«‹è¼¸å‡º DataFrame
    output_df = pd.DataFrame({'id': df_test['id'], 'binds': predictions})
    
    
    # Save the output DataFrame to a CSV file
    output_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))


import pandas as pd

input_file = '/kaggle/working/submission15_05_.csv'
output_file = '/kaggle/working/submission15_05_new.csv'
# è®€å�–å·²å„²å­˜çš„ CSV
output_df = pd.read_csv(input_file)

print(len(output_df))

# ä¿®æ”¹ id æ¬„ä½�
output_df['id'] = range(295246830, 295246830 + len(output_df))

print(output_df.shape)

# å°‡ä¿®æ”¹å¾Œçš„ DataFrame å„²å­˜å›� CSV
output_df.to_csv(output_file, index=False)

