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


import pandas as pd

# Read training data
train = pd.read_csv("/kaggle/input/ai-drug-discovery/training-set.csv")

# Show first 5 rows
train.head()



train.info()


!pip install rdkit-pypi

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
tqdm.pandas()



import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from rdkit import Chem
from rdkit.Chem import AllChem


# ===============================
# 3) Keep important columns only
# ===============================
df = train[["SMILES", "Label"]].dropna()
df.head()


# ===============================
# 4) Convert SMILES â†’ Morgan Fingerprint
# ===============================
from tqdm import tqdm
tqdm.pandas()

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return None

df["FP"] = df["SMILES"].progress_apply(smiles_to_fp)
df = df[df["FP"].notnull()]  # remove faile




# ===============================
# 5) Convert fingerprints to numpy array
# ===============================
X = np.array([list(fp) for fp in df["FP"]])
y = df["Label"].values

print("X shape =", X.shape)
print("y shape =", y.shape)

# ===============================
# 6) Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# ===============================





X_train



# ===============================
# 7) Train Model (Random Forest)
# ===============================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    class_weight="balanced",   # Ù…Ù‡Ù… Ù„Ø£Ù† Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª ØºØ§Ù„Ø¨Ù‹Ø§ Ù…Ø´ Ù…ØªÙˆØ§Ø²Ù†Ø©
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 8) Evaluate Model
# ===============================
y_pred = model.predict(X_test)

print("\n=== Classification Report ===\n")
print(classification_report(y_test, y_pred))




