/kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


# Cell 1: Imports
# [Unchanged]
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import shap
import catboost
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from tqdm import tqdm

# Cell 2: Load Data
# [Unchanged]
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

# Cell 3: Load and Clean Extra Data
# ==============================================================================
extra_tg_file_path = "/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv"
extra_tc_file_path = "/kaggle/input/tc-smiles/Tc_SMILES.csv"
extra_tg_df = pd.read_csv(extra_tg_file_path)
extra_tc_df = pd.read_csv(extra_tc_file_path)

# Clean and prepare extra_tg_df
extra_tg_clean = extra_tg_df[['SMILES', 'PID', 'Tg', 'Polymer Class']].rename(columns={'PID': 'id'})
extra_tg_clean[['FFV', 'Tc', 'Density', 'Rg']] = float('nan')

# Clean and prepare extra_tc_clean
extra_tc_clean = extra_tc_df[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})
extra_tc_clean['id'] = range(len(train) + len(extra_tg_df), len(train) + len(extra_tg_df) + len(extra_tc_df))
extra_tc_clean[['Tg', 'FFV', 'Density', 'Rg', 'Polymer Class']] = 'Unknown'  # Set Polymer Class to 'Unknown'

# Reorder columns
extra_tg_clean = extra_tg_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'Polymer Class']]
extra_tc_clean = extra_tc_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'Polymer Class']]

# Add Polymer Class to train (default to 'Unknown')
train['Polymer Class'] = 'Unknown'

# Tanimoto similarity for deduplication
def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=256)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=256)
    return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)

train_all = pd.concat([train, extra_tg_clean, extra_tc_clean], ignore_index=True)
train_all = train_all.drop_duplicates(subset=['SMILES'], keep='first')
train_all['Polymer Class'] = train_all['Polymer Class'].fillna('Unknown')  # Ensure no NaN in Polymer Class
train_all = train_all.reset_index(drop=True)
train = train_all
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
print("Target counts:")
print(train[targets].count())
print("\nPolymer Class distribution:")
print(train['Polymer Class'].value_counts())

# Cell 4: Install RDKit
# [Unchanged]
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl

# Cell 5: Script 1 - CatBoost Stacking Model
# [Unchanged]
# Outputs: submission_catboost.csv, oof_catboost.csv

# Cell 6: Empty Cell
# [Unchanged]
# Placeholder for future use

# Cell 7: Script 2 - Single-Task SMILES-BERT TTA Model
# ==============================================================================
# SCRIPT 2: SINGLE-TASK SMILES-BERT TTA MODEL (ADVANCED)
# ==============================================================================
import pandas as pd
import numpy as np
import warnings
import random
import joblib
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer
from transformers.activations import ACT2FN
from rdkit import Chem
import gc
warnings.filterwarnings('ignore')
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

# Configuration
train_df = train
test_df = test
sample_df = sample
TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]
N_AUGMENTATIONS = 300
RANDOM_STATE = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = {'Tg': 16, 'FFV': 32, 'Tc': 16, 'Density': 32, 'Rg': 32}
SEEDS = [42, 123, 456]
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Model Definition
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.2)  # For MCD
        self.activation = ACT2FN[config.hidden_act]
    def forward(self, hidden_states):
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        self.pooler = ContextPooler(config)
        self.output = nn.Linear(config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.pooler(outputs.last_hidden_state)
        return self.output(pooled_output)

# Helper Functions
def load_model_and_scaler(model_path, scaler_path, target):
    config = AutoConfig.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
    model = CustomModel(config).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    scalers = joblib.load(scaler_path)
    scaler_index = TARGET_VARIABLES.index(target)
    scaler = scalers[scaler_index]
    return model, scaler

def augment_smiles(smiles: str, n_augs: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    augmented = {smiles}
    for _ in range(n_augs * 2):
        if len(augmented) >= n_augs:
            break
        aug_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
        augmented.add(aug_smiles)
    return list(augmented)

def prepare_input(smiles, polymer_class):
    return f"SMILES: {smiles} | Class: {polymer_class}"

# Inference Pipeline
print("\n--- RUNNING SCRIPT 2: SMILES-BERT MODEL (OOF + TTA) ---")
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
scaler_path = '/kaggle/input/smiles-bert-models/target_scalers.pkl'
bert_oof_df = pd.DataFrame(index=train_df.index)
bert_predictions_df = pd.DataFrame({'id': test_df['id']})
uncertainty_df = pd.DataFrame(index=train_df.index)  # For OOF
test_uncertainty_df = pd.DataFrame(index=test_df.index)  # For test

for target in TARGET_VARIABLES:
    print(f"\n  Processing for {target}...")
    y = train_df[target].dropna()
    X_subset_smiles = train_df.loc[y.index]
    model_path = f'/kaggle/input/smiles-bert-models/trained_smiles_model_{target}_target.pth'
    model, scaler = load_model_and_scaler(model_path, scaler_path, target)
    batch_size = BATCH_SIZE[target]
    
    # Debug Polymer Class
    print(f"    Polymer Class distribution for {target}:")
    print(X_subset_smiles['Polymer Class'].value_counts(dropna=False))
    
    # OOF Predictions
    print(f"    Generating OOF predictions for {target}...")
    oof_preds = pd.Series(0.0, index=X_subset_smiles.index, dtype=np.float32)
    oof_uncertainty = pd.Series(0.0, index=X_subset_smiles.index, dtype=np.float32)
    batches = [X_subset_smiles.index[i:i + batch_size] for i in range(0, len(X_subset_smiles), batch_size)]
    
    for seed in SEEDS:
        torch.manual_seed(seed)
        with torch.no_grad():
            for batch_idx in tqdm(batches, desc="    OOF Batches"):
                batch_smiles = X_subset_smiles.loc[batch_idx]['SMILES'].tolist()
                batch_classes = X_subset_smiles.loc[batch_idx]['Polymer Class'].fillna('Unknown').tolist()
                batch_inputs = [prepare_input(s, c) for s, c in zip(batch_smiles, batch_classes)]
                inputs = tokenizer(batch_inputs, return_tensors='pt', truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                preds = model(**inputs).squeeze(-1)
                scaled_preds = preds.cpu().numpy()
                unscaled_preds = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
                oof_preds.loc[batch_idx] += unscaled_preds / len(SEEDS)
    
    bert_oof_df[target] = oof_preds
    
    # Compute OOF uncertainty
    print(f"    Computing OOF uncertainty for {target}...")
    for idx in tqdm(X_subset_smiles.index, desc="    OOF Uncertainty"):
        smiles = X_subset_smiles.loc[idx, 'SMILES']
        polymer_class = X_subset_smiles.loc[idx, 'Polymer Class']
        if pd.isna(polymer_class):
            polymer_class = 'Unknown'
        augmented_smiles_list = augment_smiles(smiles, N_AUGMENTATIONS)
        aug_inputs = [prepare_input(s, polymer_class) for s in augmented_smiles_list]
        pred_samples = []
        for chunk in [aug_inputs[i:i + batch_size] for i in range(0, len(aug_inputs), batch_size)]:
            inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                for _ in range(5):  # MCD with 5 samples
                    preds = model(**inputs).squeeze(-1)
                    scaled_preds = preds.cpu().numpy()
                    unscaled_preds = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
                    pred_samples.extend(unscaled_preds)
        oof_uncertainty.loc[idx] = np.std(pred_samples)
    uncertainty_df[target] = oof_uncertainty
    
    # Test Predictions with Uncertainty-Aware TTA
    print(f"    Generating Test predictions with TTA for {target}...")
    target_preds = np.zeros(len(test_df))
    target_uncertainty = np.zeros(len(test_df))
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        augmented_smiles_list = augment_smiles(row['SMILES'], N_AUGMENTATIONS)
        polymer_class = row.get('Polymer Class', 'Unknown')
        aug_inputs = [prepare_input(s, polymer_class) for s in augmented_smiles_list]
        pred_samples = []
        for chunk in [aug_inputs[i:i + batch_size] for i in range(0, len(aug_inputs), batch_size)]:
            inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                for _ in range(5):  # MCD with 5 samples
                    preds = model(**inputs).squeeze(-1)
                    scaled_preds = preds.cpu().numpy()
                    unscaled_preds = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
                    pred_samples.extend(unscaled_preds)
        target_preds[i] = np.mean(pred_samples)
        target_uncertainty[i] = np.std(pred_samples)
    
    bert_predictions_df[target] = target_preds
    test_uncertainty_df[target] = target_uncertainty
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

bert_predictions_df.to_csv('submission_bert.csv', index=False)
bert_oof_df.to_csv('oof_bert.csv')
uncertainty_df.to_csv('uncertainty_bert_oof.csv')
test_uncertainty_df.to_csv('uncertainty_bert_test.csv')
print("\n--- SCRIPT 2 COMPLETE: BERT predictions and OOF file saved. ---")
gc.collect()
torch.cuda.empty_cache()

# Cell 8: Define Metrics
# [Unchanged]
MINMAX_DICT = {
    'Tg': [-148.0297376, 472.25],
    'FFV': [0.2269924, 0.77709707],
    'Tc': [0.0465, 0.524],
    'Density': [0.748691234, 1.840998909],
    'Rg': [9.7283551, 34.672905605],
}
NULL_FOR_SUBMISSION = -9999
target_weights = {'Tg': 0.5, 'FFV': 2.0, 'Tc': 1.0, 'Density': 1.0, 'Rg': 0.8}

def scaling_error(labels, preds, property):
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    return np.mean(error / label_range)

def wmae(labels, preds, weights):
    return np.mean(np.abs(labels - preds) * weights)

# Cell 9: Script 3 - Optimal Blending
# ==============================================================================
# SCRIPT 3: FINAL OPTIMAL BLENDING (ADVANCED)
# ==============================================================================
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

# Configuration
train_df = train
TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]

print("\n--- RUNNING SCRIPT 3: OPTIMAL WEIGHT FINDING & BLENDING ---")

# Load predictions and OOF files
bert_test_preds = pd.read_csv('submission_bert.csv')
catboost_test_preds = pd.read_csv('submission_catboost.csv')
bert_oof_df = pd.read_csv('oof_bert.csv', index_col=0)
catboost_oof_df = pd.read_csv('oof_catboost.csv', index_col=0)
uncertainty_df = pd.read_csv('uncertainty_bert_oof.csv', index_col=0)
test_uncertainty_df = pd.read_csv('uncertainty_bert_test.csv', index_col=0)

final_submission = pd.DataFrame({'id': bert_test_preds['id']})

print("Finding optimal blend weights for each target...")
for target in TARGET_VARIABLES:
    print(f"  Processing blending for {target}...")
    
    # Filter non-NaN true labels
    valid_indices = train_df[target].dropna().index
    if len(valid_indices) == 0:
        print(f"  Warning: No non-NaN labels for {target}. Using CatBoost predictions.")
        final_submission[target] = catboost_test_preds[target]
        print(f"  wMAE for {target}: Not calculated (no valid OOF samples).")
        continue
    
    # Debug index alignment
    print(f"  Number of valid indices for {target}: {len(valid_indices)}")
    print(f"  Available indices in uncertainty_df: {len(uncertainty_df.index)}")
    
    # Align OOF predictions with valid indices
    try:
        oof_df = pd.concat([
            train_df.loc[valid_indices, target],
            bert_oof_df.loc[valid_indices, target],
            catboost_oof_df.loc[valid_indices, target],
            uncertainty_df.loc[valid_indices, target]
        ], axis=1)
        oof_df.columns = ['true', 'bert', 'catboost', 'uncertainty']
        oof_df.dropna(inplace=True)
    except KeyError as e:
        print(f"  Error: Index alignment failed for {target}: {e}")
        print(f"  Using CatBoost predictions due to alignment failure.")
        final_submission[target] = catboost_test_preds[target]
        print(f"  wMAE for {target}: Not calculated (alignment error).")
        continue
    
    if len(oof_df) == 0:
        print(f"  Warning: No valid OOF samples for {target} after alignment. Using CatBoost predictions.")
        final_submission[target] = catboost_test_preds[target]
        print(f"  wMAE for {target}: Not calculated (no valid OOF samples).")
        continue
    
    print(f"  Number of valid OOF samples for {target}: {len(oof_df)}")
    
    # Stack with uncertainty
    X_stack = oof_df[['bert', 'catboost', 'uncertainty']]
    y_stack = oof_df['true']
    model = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=6, random_seed=42, verbose=False)
    train_pool = Pool(X_stack, y_stack)
    model.fit(train_pool)
    
    # Predict on test
    X_test_stack = pd.DataFrame({
        'bert': bert_test_preds[target],
        'catboost': catboost_test_preds[target],
        'uncertainty': test_uncertainty_df[target]
    })
    final_submission[target] = model.predict(X_test_stack)
    
    # Evaluate wMAE
    blend_preds = model.predict(X_stack)
    wmae_score = wmae(oof_df['true'], blend_preds, target_weights[target])
    print(f"  wMAE for {target}: {wmae_score:.5f}")

# Ensure correct column order
sample_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
final_submission = final_submission[sample_df.columns]
final_submission.to_csv('submission.csv', index=False)

print("\nFinal submission file created using advanced stacking!")
print("Preview:")
print(final_submission.head())

# Cell 10: Empty Cell
# [Unchanged]
# Placeholder for future use


d=pd.read_csv("/kaggle/working/oof_catboost.csv")
d

