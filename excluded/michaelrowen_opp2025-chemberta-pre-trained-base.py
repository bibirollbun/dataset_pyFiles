import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train.head()


train.info()


print("Dataset shape:", train.shape)
print("\nMissing values:")
train.isnull().sum()


print("\nSMILES length statistics:")
train['smiles_length'] = train['SMILES'].str.len()
train['smiles_length'].describe()


def prepare_data_for_modeling(df, target_properties=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    """Prepare data for different modeling approaches"""
    
    mask = df[target_properties].notna().any(axis=1)
    df_clean = df[mask].copy()
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Original dataset shape: {df.shape}")

    smiles = df_clean['SMILES'].tolist()
    targets = df_clean[target_properties].values
    
    return smiles, targets, df_clean

smiles_list, target_values, clean_df = prepare_data_for_modeling(train)

print(f"Number of SMILES: {len(smiles_list)}")
print(f"Target shape: {target_values.shape}")

print("\nMissing values in targets:")
for i, prop in enumerate(['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    missing = np.isnan(target_values[:, i]).sum()
    print(f"{prop}: {missing}/{len(target_values)} ({missing/len(target_values)*100:.1f}%)")


import matplotlib.pyplot as plt
import seaborn as sns

properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, prop in enumerate(properties):
    if i < len(axes):
        train[prop].hist(ax=axes[i], bins=30, alpha=0.7)
        axes[i].set_title(f'{prop} Distribution')
        axes[i].set_xlabel(prop)
        axes[i].set_ylabel('Frequency')

# Remove the last empty subplot
if len(properties) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()


from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline

MODEL = "/kaggle/input/c/transformers/default/1/ChemBERTa-77M-MLM"
# pipe = pipeline("fill-mask", model=MODEL)
class ChemBERTaEmbedder:
    def __init__(self, model_name=MODEL):
        """
        - seyonec/ChemBERTa-zinc-base-v1
        - seyonec/ChemBERTa-zinc-250k-v1
        - seyonec/PubChem10M_SMILES_BPE_450k
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

print("Loading ChemBERTa model...")
try:
    chemberta = ChemBERTaEmbedder()
    print("ChemBERTa loaded successfully!")
except Exception as e:
    print(f"Error loading ChemBERTa: {e}")


def extract_simple_molecular_features(smiles_list):
    features = []
    for smiles in smiles_list:
        feature_vector = [
            len(smiles),  # SMILES length
            smiles.count('C'),  # Carbon count
            smiles.count('N'),  # Nitrogen count
            smiles.count('O'),  # Oxygen count
            smiles.count('S'),  # Sulfur count
            smiles.count('P'),  # Phosphorus count
            smiles.count('F'),  # Fluorine count
            smiles.count('Cl'),  # Chlorine count
            smiles.count('Br'),  # Bromine count
            smiles.count('I'),  # Iodine count
            smiles.count('='),  # Double bonds
            smiles.count('#'),  # Triple bonds
            smiles.count('-'),  # Single bonds
            smiles.count('(') + smiles.count(')'),  # Branching
            smiles.count('[') + smiles.count(']'),  # Bracket atoms
            smiles.count('@'),  # Chirality centers
            smiles.count('c'),  # Aromatic carbon
            smiles.count('n'),  # Aromatic nitrogen
            smiles.count('o'),  # Aromatic oxygen
            smiles.count('s'),  # Aromatic sulfur
        ]
        features.append(feature_vector)
    
    return np.array(features)
    
def extract_chemberta_features(smiles_list, batch_size=32, use_gpu=True):
    try:
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        if hasattr(chemberta, 'model'):
            chemberta.model = chemberta.model.to(device)
        
        all_embeddings = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]

            batch_embeddings = []
            for smiles in batch:
                inputs = chemberta.tokenizer(
                    smiles, 
                    return_tensors='pt', 
                    max_length=512, 
                    truncation=True, 
                    padding='max_length'
                )
                
                # Move inputs to GPU
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = chemberta.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                    # Move back to CPU and convert to numpy
                    embedding = embedding.cpu().numpy()
                    batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
            
            if i % 100 == 0:
                print(f"Processed {min(i+batch_size, len(smiles_list))}/{len(smiles_list)} SMILES")
        
        return np.array(all_embeddings)
        
    except Exception as e:
        print(f"ChemBERTa not available: {e}")
        print("Using alternative approach")
        return None
simple_features = extract_simple_molecular_features(smiles_list)
chemberta_features = extract_chemberta_features(smiles_list, batch_size=32, use_gpu=True)

simple_features.shape, chemberta_features.shape


total_features = np.concatenate((simple_features, chemberta_features), axis=1)
total_features.shape


test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
submission_template = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


test_simple_features = extract_simple_molecular_features(test_df['SMILES'].tolist())


test_chemberta_features = extract_chemberta_features(
    test_df['SMILES'].tolist(), 
    32, 
    True
)

test_features = np.concatenate((test_simple_features, test_chemberta_features), axis=1)


import lightgbm as lgb
CONFIG = {
    'K_FOLDS': 10, 
    'RANDOM_STATE': 42,
    'MAX_LENGTH': 512,
    'VERBOSE': False,
    
    'LGB_PARAMS': {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.07,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_data_in_leaf': 10,
        'max_depth': -1,
        'n_estimators': 1000,
        'force_col_wise': True,
        'max_bin': 255,
        'verbose': -1
    }
}


def calculate_wmae(y_true, y_pred, property_names=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    """
    Calculate weighted Mean Absolute Error (wMAE)
    wMAE = sum(weight_i * MAE_i) / sum(weight_i)
    where weight_i = number of valid samples for property i
    """
    total_weighted_mae = 0
    total_weights = 0
    individual_maes = {}
    
    for i, prop in enumerate(property_names):
        mask = ~(np.isnan(y_true[:, i]) | np.isnan(y_pred[:, i]))
        
        if mask.sum() > 0:
            true_vals = y_true[mask, i]
            pred_vals = y_pred[mask, i]
            
            mae = mean_absolute_error(true_vals, pred_vals)
            weight = len(true_vals)  
            
            total_weighted_mae += weight * mae
            total_weights += weight
            individual_maes[prop] = {'MAE': mae, 'weight': weight, 'n_samples': len(true_vals)}
        else:
            individual_maes[prop] = {'MAE': np.nan, 'weight': 0, 'n_samples': 0}
    
    wmae = total_weighted_mae / total_weights if total_weights > 0 else np.nan
    
    return wmae, individual_maes


def train_kfold_with_test_predictions(X_train, y_train, X_test, k_folds=5, random_state=42):

    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    oof_predictions = np.full_like(y_train, np.nan)  # Out-of-fold predictions
    test_predictions_folds = np.zeros((k_folds, len(X_test), len(property_names)))  # Test predictions per fold
    fold_scores = {prop: [] for prop in property_names}
    
    print(f"Starting {k_folds}-fold cross-validation...")
    print("="*60)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"\nFold {fold + 1}/{k_folds}")
        
        X_train_fold = X_train[train_idx]
        X_val_fold = X_train[val_idx] 
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        fold_test_preds = np.zeros((len(X_test), len(property_names)))
        
        for i, prop in enumerate(property_names):
            print(f"Training {prop}")
            mask_train = ~np.isnan(y_train_fold[:, i])
            mask_val = ~np.isnan(y_val_fold[:, i])
            lgb_params = CONFIG['LGB_PARAMS'].copy()
            lgb_params.update({
                'device_type': 'gpu',
                'random_state': random_state + fold,
                'verbose': -1,
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
            

            train_indices = np.where(mask_train)[0]
            X_tr = X_train_fold[mask_train]
            y_tr = y_train_fold[mask_train, i]
            
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr, y_tr)
            

            if mask_val.sum() > 0:
                val_pred = model.predict(X_val_fold[mask_val])
                oof_predictions[val_idx[mask_val], i] = val_pred
                
                val_score = mean_absolute_error(y_val_fold[mask_val, i], val_pred)
                fold_scores[prop].append(val_score)
            else:
                fold_scores[prop].append(np.nan)
                print(f"    No validation samples available")
            
            test_pred = model.predict(X_test)
            fold_test_preds[:, i] = test_pred
            test_predictions_folds[fold, :, i] = test_pred
        
        print(f"  Fold {fold + 1} completed")
        print("="*60)
    print("Cross-Validation Results:")
    print("="*60)
    
    cv_scores = {}
    for i, prop in enumerate(property_names):
        valid_scores = [score for score in fold_scores[prop] if not np.isnan(score)]
        if valid_scores:
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            cv_scores[prop] = {
                'mean': mean_score,
                'std': std_score,
                'n_folds': len(valid_scores)
            }
            print(f"{prop:8s}: {mean_score:.4f} ± {std_score:.4f} ({len(valid_scores)}/{k_folds} folds)")
        else:
            cv_scores[prop] = {'mean': np.nan, 'std': np.nan, 'n_folds': 0}
            print(f"{prop:8s}: No valid predictions")
    
    oof_wmae, _ = calculate_wmae(y_train, oof_predictions, property_names)
    print(f"\nOverall Out-of-Fold wMAE: {oof_wmae:.4f}")

    blended_test_predictions = np.mean(test_predictions_folds, axis=0)
    
    return {
        'oof_predictions': oof_predictions,
        'test_predictions': blended_test_predictions,
        'test_predictions_folds': test_predictions_folds,
        'cv_scores': cv_scores,
        'oof_wmae': oof_wmae
    }
kfold_results = train_kfold_with_test_predictions(
    X_train=total_features,
    y_train=target_values, 
    X_test=test_features,
    k_folds=CONFIG['K_FOLDS'],
    random_state=CONFIG['RANDOM_STATE']
)


properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for i, prop in enumerate(properties):
        submission_template[prop] = kfold_results['test_predictions'][:, i]
submission_template.to_csv('submission.csv', index=False)

submission_template.head()

