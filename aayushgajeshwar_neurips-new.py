# Upgrade pip and install packages from offline folder
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("/kaggle/input/polybert/polyBERT")


import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rdkit import Chem # Import Chem from rdkit to potentially validate SMILES before using psmiles
import warnings


from rdkit.Chem import MolToSmiles

def safe_canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


ps_slile = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
ps_slile = ps_slile[['SMILES', 'Tg', 'Density', 'FFV', 'Tc', 'Rg']] # Assuming this slicing was intended
ps_slile['SMILES'] = ps_slile['SMILES'].apply(safe_canonicalize)


import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from tqdm import tqdm

# Load PolyBERT
checkpoint = "/kaggle/input/polybert/polyBERT"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
bert_model = AutoModel.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# Extract PolyBERT features (mean pooling of last hidden state)
def extract_bert_features(smiles_list):
    features = []
    with torch.no_grad():
        for smi in tqdm(smiles_list, desc="Extracting BERT features"):
            encoded = tokenizer(smi, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            features.append(pooled)
    return np.array(features)


# Train and evaluate using XGBoost
def train_xgboost_on_bert_features(X, Y, property_name):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n{property_name} Results:")
    print(f"MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}")

    return Y_test, y_pred


def test_xgboost_on_bert_features(X_train, Y_train, X_test, property_name):
   
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return y_pred


df1= pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df2= pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")
df3= pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv")
df4= pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
df5= pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")

for  df in [df2, df4, df5]:
  count=0
  for i in df["SMILES"].to_list():
    if i in df1["SMILES"].to_list():
      count+=1
  print(len(df["SMILES"].to_list()), count, sep= ",")


# Load the two datasets
df1 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df2 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")

# Drop rows with missing SMILES or properties
df1 = df1.dropna(subset=['SMILES', 'Tc'])
df2 = df2.dropna(subset=['SMILES', 'TC_mean'])
df1['SMILES'] = df1['SMILES'].apply(safe_canonicalize)
df2['SMILES'] = df2['SMILES'].apply(safe_canonicalize)
# Convert df1 SMILES to a set for fast lookup
smiles_set1 = set(df1['SMILES'])

# Filter df2 to keep only SMILES not in df1
df2_unique = df2[~df2['SMILES'].isin(smiles_set1)]

# Select desired columns from both
df1_selected = df1[['SMILES', 'Tc']]
df2_selected = df2_unique[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})

# Combine the dataframes
df_Tc = pd.concat([df1_selected, df2_selected], ignore_index=True)

# Preview
print(f"Combined dataframe shape: {df_Tc.shape}")
print(df_Tc.head())


df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df = df[['SMILES', 'Tg', 'Density', 'FFV', 'Tc', 'Rg']] # Assuming this slicing was intended
df['SMILES'] = df['SMILES'].apply(safe_canonicalize)


# Load the two datasets
df1 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df2 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")

# Drop rows with missing SMILES or properties
df1 = df1.dropna(subset=['SMILES', 'Tc'])
df2 = df2.dropna(subset=['SMILES', 'TC_mean'])
df1['SMILES'] = df1['SMILES'].apply(safe_canonicalize)
df2['SMILES'] = df2['SMILES'].apply(safe_canonicalize)
# Convert df1 SMILES to a set for fast lookup
smiles_set1 = set(df1['SMILES'])

# Filter df2 to keep only SMILES not in df1
df2_unique = df2[~df2['SMILES'].isin(smiles_set1)]

# Select desired columns from both
df1_selected = df1[['SMILES', 'Tc']]
df2_selected = df2_unique[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})

# Combine the dataframes
df_Tc = pd.concat([df1_selected, df2_selected], ignore_index=True)

# Preview
print(f"Combined dataframe shape: {df_Tc.shape}")
print(df_Tc.head())


# Load the two datasets
df1 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df2 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")

# Drop rows with missing SMILES or properties
df1 = df1.dropna(subset=['SMILES', 'Tg'])
df2 = df2.dropna(subset=['SMILES', 'Tg'])
df1['SMILES'] = df1['SMILES'].apply(safe_canonicalize)
df2['SMILES'] = df2['SMILES'].apply(safe_canonicalize)
# Convert df1 SMILES to a set for fast lookup
smiles_set1 = set(df1['SMILES'])

# Filter df2 to keep only SMILES not in df1
df2_unique = df2[~df2['SMILES'].isin(smiles_set1)]

# Select desired columns from both
df1_selected = df1[['SMILES', 'Tg']]
df2_selected = df2_unique[['SMILES', 'Tg']]

# Combine the dataframes
df_Tg = pd.concat([df1_selected, df2_selected], ignore_index=True)

# Preview
print(f"Combined dataframe shape: {df_Tg.shape}")
print(df_Tg.head())


# Load the two datasets
df1 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df2 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")

# Drop rows with missing SMILES or properties
df1 = df1.dropna(subset=['SMILES', 'FFV'])
df2 = df2.dropna(subset=['SMILES', 'FFV'])
df1['SMILES'] = df1['SMILES'].apply(safe_canonicalize)
df2['SMILES'] = df2['SMILES'].apply(safe_canonicalize)
# Convert df1 SMILES to a set for fast lookup
smiles_set1 = set(df1['SMILES'])

# Filter df2 to keep only SMILES not in df1
df2_unique = df2[~df2['SMILES'].isin(smiles_set1)]

# Select desired columns from both
df1_selected = df1[['SMILES', 'FFV']]
df2_selected = df2_unique[['SMILES', 'FFV']]

# Combine the dataframes
df_FFV = pd.concat([df1_selected, df2_selected], ignore_index=True)

# Preview
print(f"Combined dataframe shape: {df_FFV.shape}")
print(df_FFV.head())


y_true_dict, y_pred_dict = {}, {}


# Load data
df= ps_slile.copy()
target_columns = ['Density', 'Rg']


# Main loop
for prop in target_columns:
    prop_df = df[['SMILES', prop]].dropna(subset=['SMILES', prop]).reset_index(drop=True)
    X_smiles = [str(s) for s in prop_df['SMILES'] if s is not None]
    Y = prop_df[prop].values

    if len(X_smiles) < 2:
        print(f"Not enough valid data for target '{prop}'. Skipping.")
        continue

    X_features = extract_bert_features(X_smiles)
    y_true, y_pred = train_xgboost_on_bert_features(X_features, Y, prop)
    y_true_dict[prop] = y_true
    y_pred_dict[prop] = y_pred


# Load data
df= df_FFV.copy()
target_columns = ['FFV']

# Main loop
for prop in target_columns:
    prop_df = df[['SMILES', prop]].dropna(subset=['SMILES', prop]).reset_index(drop=True)
    X_smiles = [str(s) for s in prop_df['SMILES'] if s is not None]
    Y = prop_df[prop].values

    if len(X_smiles) < 2:
        print(f"Not enough valid data for target '{prop}'. Skipping.")
        continue

    X_features = extract_bert_features(X_smiles)
    y_true, y_pred = train_xgboost_on_bert_features(X_features, Y, prop)
    y_true_dict[prop] = y_true
    y_pred_dict[prop] = y_pred


# Load data
df= df_Tc.copy()
target_columns = ['Tc']

# Main loop
for prop in target_columns:
    prop_df = df[['SMILES', prop]].dropna(subset=['SMILES', prop]).reset_index(drop=True)
    X_smiles = [str(s) for s in prop_df['SMILES'] if s is not None]
    Y = prop_df[prop].values

    if len(X_smiles) < 2:
        print(f"Not enough valid data for target '{prop}'. Skipping.")
        continue

    X_features = extract_bert_features(X_smiles)
    y_true, y_pred = train_xgboost_on_bert_features(X_features, Y, prop)
    y_true_dict[prop] = y_true
    y_pred_dict[prop] = y_pred


# Load data
df= df_Tg.copy()
target_columns = ['Tg']

# Main loop
for prop in target_columns:
    prop_df = df[['SMILES', prop]].dropna(subset=['SMILES', prop]).reset_index(drop=True)
    X_smiles = [str(s) for s in prop_df['SMILES'] if s is not None]
    Y = prop_df[prop].values

    if len(X_smiles) < 2:
        print(f"Not enough valid data for target '{prop}'. Skipping.")
        continue

    X_features = extract_bert_features(X_smiles)
    y_true, y_pred = train_xgboost_on_bert_features(X_features, Y, prop)
    y_true_dict[prop] = y_true
    y_pred_dict[prop] = y_pred


K = len(y_true_dict)
ranges = {k: np.nanmax(y_true_dict[k]) - np.nanmin(y_true_dict[k]) for k in y_true_dict}
n_vals = {k: np.count_nonzero(~np.isnan(y_true_dict[k])) for k in y_true_dict}
denom = sum([np.sqrt(1 / n_vals[k]) for k in n_vals])
weights = {
    k: (1 / ranges[k]) * ((K * np.sqrt(1 / n_vals[k])) / denom)
    for k in n_vals
}

wmae_total = 0
count_total = 0
for k in y_true_dict:
    y_t = y_true_dict[k]
    y_p = y_pred_dict[k]
    mask = ~np.isnan(y_t)
    err = np.abs(y_t[mask] - y_p[mask])
    wmae_total += weights[k] * np.sum(err)
    count_total += np.sum(mask)

wmae_final = wmae_total / count_total
print(f"\nFinal wMAE (across all models): {wmae_final:.4f}")


df_sub = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
df_sub['SMILES'] = df_sub['SMILES'].apply(safe_canonicalize)
sub_columns = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']

submission = dict.fromkeys(sub_columns)
submission['id'] = df_sub['id']


# Load data
df= ps_slile.copy()
target_columns = ['Density', 'Rg']


# Main loop
for prop in target_columns:
    prop_df = df[['SMILES', prop]].dropna(subset=['SMILES', prop]).reset_index(drop=True)
    X_smiles = [str(s) for s in prop_df['SMILES'] if s is not None]
    X_smiles_test = [str(s) for s in df_sub['SMILES'] if s is not None]
    Y = prop_df[prop].values

    if len(X_smiles) < 2:
        print(f"Not enough valid data for target '{prop}'. Skipping.")
        continue

    X_features = extract_bert_features(X_smiles)
    X_test = extract_bert_features(X_smiles_test)
    y_pred = test_xgboost_on_bert_features(X_features, Y, X_test, prop)
    submission[prop] = y_pred


# Load data
df= df_FFV.copy()
target_columns = ['FFV']


# Main loop
for prop in target_columns:
    prop_df = df[['SMILES', prop]].dropna(subset=['SMILES', prop]).reset_index(drop=True)
    X_smiles = [str(s) for s in prop_df['SMILES'] if s is not None]
    X_smiles_test = [str(s) for s in df_sub['SMILES'] if s is not None]
    Y = prop_df[prop].values

    if len(X_smiles) < 2:
        print(f"Not enough valid data for target '{prop}'. Skipping.")
        continue

    X_features = extract_bert_features(X_smiles)
    X_test = extract_bert_features(X_smiles_test)
    y_pred = test_xgboost_on_bert_features(X_features, Y, X_test, prop)
    submission[prop] = y_pred


# Load data
df= df_Tg.copy()
target_columns = ['Tg']


# Main loop
for prop in target_columns:
    prop_df = df[['SMILES', prop]].dropna(subset=['SMILES', prop]).reset_index(drop=True)
    X_smiles = [str(s) for s in prop_df['SMILES'] if s is not None]
    X_smiles_test = [str(s) for s in df_sub['SMILES'] if s is not None]
    Y = prop_df[prop].values

    if len(X_smiles) < 2:
        print(f"Not enough valid data for target '{prop}'. Skipping.")
        continue

    X_features = extract_bert_features(X_smiles)
    X_test = extract_bert_features(X_smiles_test)
    y_pred = test_xgboost_on_bert_features(X_features, Y, X_test, prop)
    submission[prop] = y_pred


# Load data
df= df_Tc.copy()
target_columns = ['Tc']


# Main loop
for prop in target_columns:
    prop_df = df[['SMILES', prop]].dropna(subset=['SMILES', prop]).reset_index(drop=True)
    X_smiles = [str(s) for s in prop_df['SMILES'] if s is not None]
    X_smiles_test = [str(s) for s in df_sub['SMILES'] if s is not None]
    Y = prop_df[prop].values

    if len(X_smiles) < 2:
        print(f"Not enough valid data for target '{prop}'. Skipping.")
        continue

    X_features = extract_bert_features(X_smiles)
    X_test = extract_bert_features(X_smiles_test)
    y_pred = test_xgboost_on_bert_features(X_features, Y, X_test, prop)
    submission[prop] = y_pred


submission = pd.DataFrame(submission)
submission


submission.to_csv('submission.csv',index=False)




