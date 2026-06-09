import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

sample = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


# ==============================================================================
# SCRIPT 1: GBDT STACKING MODEL (CORRECTED)
# ==============================================================================
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')

# --- Link to environment variables ---
# This notebook environment uses 'train', 'test', 'sample'
# We assign them to the names used in the script.
train_df = train
test_df = test
sample_df = sample

# --- Configuration ---
TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]
N_SPLITS = 5
RANDOM_STATE = 42

XGB_PARAMS = {'n_estimators': 2000, 'learning_rate': 0.02, 'max_depth': 6, 'subsample': 0.7, 'colsample_bytree': 0.6, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist'}
LGBM_PARAMS = {'n_estimators': 2000, 'learning_rate': 0.02, 'max_depth': 7, 'subsample': 0.7, 'colsample_bytree': 0.6, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1}
META_MODEL = Ridge(alpha=1.0, random_state=RANDOM_STATE)

# --- Feature Engineering ---
def generate_rdkit_features(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    desc_list = [d[0] for d in Descriptors._descList]
    morgan_fp_size = 1024
    if mol is None: return np.full(len(desc_list) + morgan_fp_size, np.nan)
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    descriptors = np.array(calculator.CalcDescriptors(mol))
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=morgan_fp_size)
    mfp_array = np.array(list(mfp.ToBitString())).astype(int)
    return np.concatenate([descriptors, mfp_array])

# --- Data Preparation & Training ---
print("--- RUNNING SCRIPT 1: GBDT STACKING MODEL ---")
desc_list_names = [d[0] for d in Descriptors._descList]
fp_morgan_cols = [f'mfp_{i}' for i in range(1024)]
feature_columns = desc_list_names + fp_morgan_cols
X = pd.DataFrame(np.vstack([generate_rdkit_features(s) for s in train_df['SMILES']]), columns=feature_columns)
X_test = pd.DataFrame(np.vstack([generate_rdkit_features(s) for s in test_df['SMILES']]), columns=feature_columns)
f32_max = np.finfo(np.float32).max
for df in [X, X_test]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[df > f32_max] = np.nan
    df[df < -f32_max] = np.nan
impute_values = X.mean()
X.fillna(impute_values, inplace=True)
X_test = X_test.reindex(columns=X.columns).fillna(impute_values)

gbdt_oof_df = pd.DataFrame(index=train_df.index)
gbdt_predictions_df = pd.DataFrame({'id': test_df['id']})

for target in TARGET_VARIABLES:
    print(f"  Training for {target}...")
    y = train_df[target].dropna()
    X_subset = X.loc[y.index]
    
    oof_preds_xgb = pd.Series(np.zeros(len(X_subset)), index=X_subset.index)
    oof_preds_lgb = pd.Series(np.zeros(len(X_subset)), index=X_subset.index)
    test_preds_xgb_folds = np.zeros((len(X_test), N_SPLITS))
    test_preds_lgb_folds = np.zeros((len(X_test), N_SPLITS))
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_subset, y)):
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = X_subset.iloc[train_idx], y.iloc[train_idx], X_subset.iloc[val_idx], y.iloc[val_idx]
        
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[xgb.callback.EarlyStopping(50, save_best=True)], verbose=0)
        oof_preds_xgb.iloc[val_idx] = xgb_model.predict(X_val_fold)
        test_preds_xgb_folds[:, fold] = xgb_model.predict(X_test)

        lgb_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds_lgb.iloc[val_idx] = lgb_model.predict(X_val_fold)
        test_preds_lgb_folds[:, fold] = lgb_model.predict(X_test)
        
    X_meta_train = pd.concat([oof_preds_xgb, oof_preds_lgb], axis=1)
    X_meta_test = pd.DataFrame({'xgb': np.mean(test_preds_xgb_folds, axis=1), 'lgb': np.mean(test_preds_lgb_folds, axis=1)})
    
    meta_model = META_MODEL
    meta_model.fit(X_meta_train, y)
    gbdt_predictions_df[target] = meta_model.predict(X_meta_test)
    gbdt_oof_df.loc[X_meta_train.index, target] = meta_model.predict(X_meta_train)

gbdt_predictions_df.to_csv('submission_gbdt.csv', index=False)
gbdt_oof_df.to_csv('oof_gbdt.csv')
print("\n--- SCRIPT 1 COMPLETE: GBDT predictions and OOF file saved. ---")


# ==============================================================================
# SCRIPT 2: POLYMER-BERT TTA MODEL (Corrected for Memory Error)
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

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

# --- Link to environment variables ---
train_df = train
test_df = test
sample_df = sample

# --- Configuration ---
TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]
N_AUGMENTATIONS = 101
RANDOM_STATE = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32 # Process in batches of 32 to save memory
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# --- Model Definition ---
class ContextPooler(nn.Module):
    def __init__(self, config): super().__init__(); self.dense = nn.Linear(config.hidden_size, config.hidden_size); self.dropout = nn.Dropout(config.hidden_dropout_prob); self.activation = ACT2FN[config.hidden_act]
    def forward(self, hidden_states): context_token = hidden_states[:, 0]; context_token = self.dropout(context_token); pooled_output = self.dense(context_token); pooled_output = self.activation(pooled_output); return pooled_output
class CustomModel(PreTrainedModel):
    def __init__(self, config): super().__init__(config); self.backbone = AutoModel.from_config(config); self.pooler = ContextPooler(config); self.output = nn.Linear(config.hidden_size, 1)
    def forward(self, input_ids, attention_mask): outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask); pooled_output = self.pooler(outputs.last_hidden_state); return self.output(pooled_output)

# --- Helper Functions ---
def load_model_and_scaler(model_path, scaler_path, target_name):
    config = AutoConfig.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer'); model = CustomModel(config).to(DEVICE); model.load_state_dict(torch.load(model_path, map_location=DEVICE)); model.eval()
    scalers = joblib.load(scaler_path); scaler_index = TARGET_VARIABLES.index(target_name); scaler = scalers[scaler_index]; return model, scaler
def augment_smiles(smiles: str, n_augs: int):
    mol = Chem.MolFromSmiles(smiles);
    if mol is None: return [smiles]
    augmented = {smiles};
    for _ in range(n_augs * 2):
        if len(augmented) >= n_augs: break
        aug_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True); augmented.add(aug_smiles)
    return list(augmented)

# --- Inference Pipeline ---
print("\n--- RUNNING SCRIPT 2: BERT MODEL (OOF + TTA) ---")
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
scaler_path = '/kaggle/input/smiles-bert-models/target_scalers.pkl'
bert_oof_df = pd.DataFrame(index=train_df.index)
bert_predictions_df = pd.DataFrame({'id': test_df['id']})

for target in TARGET_VARIABLES:
    print(f"\n  Processing for {target}...")
    y = train_df[target].dropna()
    X_subset_smiles = train_df.loc[y.index]
    
    # --- Generate OOF predictions (with Mini-Batching) ---
    print(f"    Generating OOF predictions for {target}...")
    model_path = f'/kaggle/input/smiles-bert-models/trained_smiles_model_{target}_target.pth'
    model, scaler = load_model_and_scaler(model_path, scaler_path, target)
    
    oof_preds = pd.Series(index=X_subset_smiles.index, dtype=np.float32)
    batches = [X_subset_smiles.index[i:i + BATCH_SIZE] for i in range(0, len(X_subset_smiles), BATCH_SIZE)]

    with torch.no_grad():
        for batch_idx in tqdm(batches, desc="    OOF Batches"):
            batch_smiles = X_subset_smiles.loc[batch_idx]['SMILES'].tolist()
            inputs = tokenizer(batch_smiles, return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            preds = model(**inputs)
            scaled_preds = preds.cpu().numpy()
            unscaled_preds = scaler.inverse_transform(scaled_preds).flatten()
            oof_preds.loc[batch_idx] = unscaled_preds
    
    bert_oof_df[target] = oof_preds
    
    # --- Generate Test predictions (with TTA) ---
    print(f"    Generating Test predictions with TTA for {target}...")
    target_preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        augmented_smiles_list = augment_smiles(row['SMILES'], N_AUGMENTATIONS)
        inputs = tokenizer(augmented_smiles_list, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad(): preds = model(**inputs)
        scaled_preds = preds.cpu().numpy(); unscaled_preds = scaler.inverse_transform(scaled_preds).flatten(); final_pred = np.median(unscaled_preds)
        target_preds.append(final_pred)
    
    bert_predictions_df[target] = target_preds

    # Clean up memory after each target
    del model, scaler
    gc.collect()
    torch.cuda.empty_cache()

bert_predictions_df.to_csv('submission_bert.csv', index=False)
bert_oof_df.to_csv('oof_bert.csv')
print("\n--- SCRIPT 2 COMPLETE: BERT predictions and OOF file saved. ---")


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

print("\n--- RUNNING SCRIPT 3: OPTIMAL WEIGHT FINDING & BLENDING ---")

# Carregar predições e OOFs
bert_test_preds = pd.read_csv('submission_bert.csv')
gbdt_test_preds = pd.read_csv('submission_gbdt.csv')
bert_oof_df = pd.read_csv('oof_bert.csv', index_col=0)
gbdt_oof_df = pd.read_csv('oof_gbdt.csv', index_col=0)

# Garantir alinhamento dos índices
train_df = train  # assumindo que 'train' foi carregado anteriormente
TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]

final_submission = pd.DataFrame({'id': bert_test_preds['id']})
best_weights = {}

print("Finding optimal blend weights for each target...")

for target in TARGET_VARIABLES:
    try:
        # Alinhar os índices entre true labels e predições
        bert_oof_aligned = bert_oof_df[target].align(train_df[[target]], axis=0)[0]
        gbdt_oof_aligned = gbdt_oof_df[target].align(train_df[[target]], axis=0)[0]

        oof_df = pd.concat([
            train_df[target],
            bert_oof_aligned.rename('bert'),
            gbdt_oof_aligned.rename('gbdt')
        ], axis=1).dropna()

        oof_df.columns = ['true', 'bert', 'gbdt']

        best_rmse = float('inf')
        best_w = 0.18

        for w in np.arange(0.0, 1.01, 0.01):
            blend_preds = w * oof_df['bert'] + (1 - w) * oof_df['gbdt']
            rmse = np.sqrt(mean_squared_error(oof_df['true'], blend_preds))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = w

        best_weights[target] = best_w
        print(f"  Best weight for {target} is {best_w:.2f} (Local RMSE: {best_rmse:.5f})")

    except Exception as e:
        print(f"Error processing {target}: {e}")
        best_weights[target] = 0.18  # fallback

print("\nBlending test set predictions with optimal weights...")
for col in bert_test_preds.columns:
    if col != 'id':
        w = best_weights.get(col, 0.18)
        final_submission[col] = w * bert_test_preds[col] + (1 - w) * gbdt_test_preds[col]

# Garantir que a ordem das colunas esteja correta
sample_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
final_submission = final_submission[sample_df.columns]

# Salvar submissão final
final_submission.to_csv('submission.csv', index=False)

print("\nFinal submission file created using data-driven optimal weights!")
print("Preview:")
print(final_submission.head())


# ==============================================================================
# SCRIPT 3: FINAL OPTIMAL BLENDING (CORRECTED)
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# --- Link to environment variables ---
train_df = train

print("\n--- RUNNING SCRIPT 3: OPTIMAL WEIGHT FINDING & BLENDING ---")

# Load all predictions and OOF files
bert_test_preds = pd.read_csv('submission_bert.csv')
gbdt_test_preds = pd.read_csv('submission_gbdt.csv')
bert_oof_df = pd.read_csv('oof_bert.csv', index_col=0)
gbdt_oof_df = pd.read_csv('oof_gbdt.csv', index_col=0)

final_submission = pd.DataFrame({'id': bert_test_preds['id']})
best_weights = {}

print("Finding optimal blend weights for each target...")
# Note: TARGET_VARIABLES should be defined here if this script is run completely separately
TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]
for target in TARGET_VARIABLES:
    
    # Prepare OOF data for optimization
    oof_df = pd.concat([train_df[target], bert_oof_df[target], gbdt_oof_df[target]], axis=1)
    oof_df.columns = ['true', 'bert', 'gbdt']
    oof_df.dropna(inplace=True)

    best_rmse = float('inf')
    best_w = 0.18
    
    # Search for the best weight
    for w in np.arange(0.0, 1.01, 0.01):
        blend_preds = w * oof_df['bert'] + (1 - w) * oof_df['gbdt']
        rmse = np.sqrt(mean_squared_error(oof_df['true'], blend_preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
            
    best_weights[target] = best_w
    print(f"  Best weight for {target} is {best_w:.2f} (Local RMSE: {best_rmse:.5f})")

print("\nBlending test set predictions with optimal weights...")
for col in bert_test_preds.columns:
    if col != 'id':
        w = best_weights.get(col, 0.18)
        final_submission[col] = (w * bert_test_preds[col]) + ((1 - w) * gbdt_test_preds[col])

# Make sure final submission has the right columns in the right order
sample_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
final_submission = final_submission[sample_df.columns]
final_submission.to_csv('submission.csv', index=False)

print("\nFinal submission file created using data-driven optimal weights!")
print("Preview:")
print(final_submission.head())

