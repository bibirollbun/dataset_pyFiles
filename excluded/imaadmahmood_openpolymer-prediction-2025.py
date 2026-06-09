from IPython.display import Image, display

img_path = "/kaggle/input/polymer-version-3/polymer_3.png"

display(Image(filename=img_path))


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


# âœ… OFFLINE ChemBERTa Pipeline for NeurIPS Polymer Challenge 2025
# Uses only offline files & saves outputs correctly

import os
import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data import DataLoader
from tqdm import tqdm

# ========= File Paths =========
TRAIN_CSV = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
TEST_CSV = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"
SAMPLE_SUB = "/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv"
CHEMBERTA_PATH = "/kaggle/input/chemberta-zinc-base-offline"
WORK_DIR = "/kaggle/working"

# ========= Load Data =========
df = pd.read_csv(TRAIN_CSV)
X_smiles = df['SMILES'].tolist()
y = df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].values

# ========= Load Offline ChemBERTa =========
tokenizer = RobertaTokenizer.from_pretrained(CHEMBERTA_PATH)
transformer = RobertaModel.from_pretrained(CHEMBERTA_PATH)
transformer.eval()

# ========= Embedding Function =========
def get_chemberta_embeddings(smiles_list, save_path):
    embeddings = []
    loader = DataLoader(smiles_list, batch_size=32)
    for batch in tqdm(loader, desc="ChemBERTa Encoding"):
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = transformer(**tokens).pooler_output
        embeddings.append(output.cpu().numpy())
    arr = np.vstack(embeddings)
    np.save(save_path, arr)
    return arr

# ========= Train Embeddings =========
train_emb_file = os.path.join(WORK_DIR, "train_bert_embs.npy")
X_features = np.load(train_emb_file) if os.path.exists(train_emb_file) else get_chemberta_embeddings(X_smiles, train_emb_file)

# ========= Cross-Validation =========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros_like(y)
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_features)):
    X_train, X_val = X_features[train_idx], X_features[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    val_preds = np.zeros_like(y_val)
    fold_models = []
    for t in range(5):
        mask = ~np.isnan(y_train[:, t])
        model = GradientBoostingRegressor(n_estimators=200)
        model.fit(X_train[mask], y_train[mask, t])
        val_preds[:, t] = model.predict(X_val)
        fold_models.append(model)
    oof_preds[val_idx] = val_preds
    models.append(fold_models)

# ========= Evaluation =========
print("\nMSE per task:")
for i, name in enumerate(['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    mask = ~np.isnan(y[:, i])
    mse = mean_squared_error(y[mask, i], oof_preds[mask, i])
    print(f"{name}: {mse:.4f}")

# ========= Test Predictions =========
test_df = pd.read_csv(TEST_CSV)
test_smiles = test_df['SMILES'].tolist()
test_emb_file = os.path.join(WORK_DIR, "test_bert_embs.npy")
X_test = np.load(test_emb_file) if os.path.exists(test_emb_file) else get_chemberta_embeddings(test_smiles, test_emb_file)

test_preds = np.zeros((len(X_test), 5))
for fold_models in models:
    for i in range(5):
        test_preds[:, i] += fold_models[i].predict(X_test) / len(models)

# ========= Submission =========
sub = pd.read_csv(SAMPLE_SUB)
sub[['Tg', 'FFV', 'Tc', 'Density', 'Rg']] = test_preds
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… submission.csv saved.")


