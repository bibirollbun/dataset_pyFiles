# ğŸ“¦ Install RDKit wheel from Kaggle input
print("ğŸš€ Installing RDKit from local wheel...")

!pip install /kaggle/input/neurips-opp2025-download-libraries/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl

print("âœ… RDKit installation complete!")


# ğŸ“¦ Install Mordred (molecular descriptor calculator) from local Kaggle input path
print("ğŸ”§ Installing Mordred (v1.2.0) ...")
!pip install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  /kaggle/input/neurips-opp2025-download-libraries/mordred-1.2.0.tar.gz

print("âœ… Mordred installation completed!")


# Standard library imports
import os          # Operating system interfaces
import json        # For working with JSON data
import types       # Dynamic type creation and manipulation
import joblib      # For saving and loading models/data
import numbers     # Numeric abstract base classes
import warnings    # To handle and filter warnings
from pathlib import Path  # Object-oriented filesystem paths


# Data processing and numerical libraries
import numpy as np     # Numerical computing
import polars as pl    # Fast DataFrame library
import pandas as pd    # Data manipulation and analysis

# RDKit for chemistry/molecular data
from rdkit import Chem   # Core RDKit chemistry functions
from mordred import Calculator  # Molecular descriptor calculator
from mordred import descriptors # Predefined molecular descriptors


# Machine learning models
import lightgbm as lgb                    # Gradient boosting framework
from metric import score                  # Custom scoring function
from xgboost import XGBRegressor          # XGBoost regression model
from catboost import CatBoostRegressor    # CatBoost regression model


# PyTorch deep learning
import torch
import torch.nn as nn                     # Neural network layers and functions
from rdkit import Chem                    # (Duplicate import, already imported above)
from metric import score                  # (Duplicate import, already imported above)
from transformers import AutoModel        # Pretrained transformer models
from transformers import AutoConfig       # Transformer configuration class
from torch.utils.data import Dataset      # Custom dataset handling
from transformers import AutoTokenizer    # Tokenizer for NLP models
from torch.utils.data import DataLoader   # Data batching and loading
from sklearn.model_selection import KFold # Cross-validation splitting
from torch.utils.data import TensorDataset # Tensor dataset wrapper
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence 


# Utilities for variable-length sequences in RNNs

# Pandas display option
pd.options.display.max_columns = None  # Show all columns when printing DataFrames


# Disable RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from transformers.utils.logging import disable_progress_bar
disable_progress_bar()


# Configuration class to store paths, hyperparameters, and model settings
class CFG:
    
    # External datasets (chemical/molecular data)
    ext1_data_path = Path('/kaggle/input/tc-smiles/Tc_SMILES.csv')          # External dataset 1
    ext2_data_path = Path('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv') # External dataset 2
    ext3_data_path = Path('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')       # External dataset 3
    ext4_data_path = Path('/kaggle/input/smiles-extra-data/data_tg3.xlsx')        # External dataset 4

    # Competition dataset paths
    train_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')  # Training data
    test_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')    # Test data

    # Pretrained model paths
    chemberta_root_path = Path('/kaggle/input/neurips-opp2025-a-chemberta-t')  # ChemBERTa pretrained model
    fastsmiles_root_path = Path('/kaggle/input/neurips-opp2025-a-fastsmiles-t')# FastSMILES pretrained model
    
    # Training / inference configuration
    threshold = 0.80                 # Threshold for classification / probability cutoff
    checkpoint = 'DeepChem/ChemBERTa-77M-MTR'  # Pretrained checkpoint model to use
    n_splits = 5                     # Number of folds for cross-validation
    batch_size = 1024                # Training batch size
    hidden_size = 384                # Hidden layer size for models
    context = 512                     # Context window length (sequence length)
    seed = 42                         # Random seed for reproducibility
    
    # Ensemble weights for classical ML models
    ctb_weight = 0.3   # CatBoost weight
    lgb_weight = 0.4   # LightGBM weight
    xgb_weight = 0.3   # XGBoost weight
    
    # Ensemble weights for transformer-based models
    chemberta_weight = 0.4   # ChemBERTa contribution
    fastsmiles_weight = 0.6  # FastSMILES contribution



# Feature Engineering (FE) class for handling external data, merging, descriptors, and preprocessing
class FE:
    
    def __init__(
        self, 
        ext1_data_path, 
        ext2_data_path,
        ext3_data_path,
        ext4_data_path,
        threshold
    ):
        # Store paths to external datasets
        self._ext1_data_path = ext1_data_path
        self._ext2_data_path = ext2_data_path
        self._ext3_data_path = ext3_data_path
        self._ext4_data_path = ext4_data_path
            
        # General parameters
        self._threshold = threshold        # Threshold for dropping bad features
        self._batch_size = 16384           # Polars CSV batch size (fast loading)
        
        # Feature calculation configs
        self._ignore_3D = True             # Ignore 3D descriptors
        self._bad_features = None          # Features to remove later
        
        # Target properties to track
        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
    def _load_data(self, path):
        """Load CSV with Polars (fast) and convert to pandas"""
        print(f"ğŸ“‚ Loading dataset from {path} ...")
        return pl.read_csv(path, batch_size=self._batch_size).to_pandas()
    
    def _to_canonical(self, smiles):
        """Convert SMILES string to canonical form"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return np.nan
    
    def _load(self, path, columns_dict, excel=False):
        """Load external dataset (CSV or Excel), rename columns, and canonicalize SMILES"""
        print(f"ğŸ”„ Loading external data from {path} ...")
        if not excel:
            ext_data = pd.read_csv(path)
        else:
            ext_data = pd.read_excel(path)
        
        # Rename target columns (unify names across datasets)
        ext_data = ext_data.rename(columns=columns_dict)
        ext_data['SMILES'] = ext_data['SMILES'].apply(self._to_canonical)
        
        return ext_data

    def _merge(self, data, ext_data, target):
        """Merge external data into main dataset"""
        print(f"ğŸ”— Merging external data for target: {target} ...")
        shared_smiles = set(data['SMILES']) & set(ext_data['SMILES'])
        unique_smiles = set(ext_data['SMILES']) - set(data['SMILES'])
        
        # Only update missing targets
        populated = set(data.loc[data[target].notna(), 'SMILES'])
        shared_smiles -= populated
        
        for s in shared_smiles:
            value = ext_data.loc[ext_data['SMILES'] == s, target].iat[0]
            data.loc[data['SMILES'] == s, target] = value
        
        # Add new rows from external data
        new_rows = ext_data[ext_data['SMILES'].isin(unique_smiles)]
        data = pd.concat([data, new_rows], axis=0).reset_index(drop=True)
        
        return data
    
    def _merge_with_ext1_data(self, df):
        ext_data = self._load(self._ext1_data_path, {'TC_mean': 'Tc'})        
        ext_data = ext_data[['SMILES', 'Tc']].groupby('SMILES', as_index=False)['Tc'].mean()
        return self._merge(df, ext_data, 'Tc')

    def _merge_with_ext2_data(self, df):
        ext_data = self._load(self._ext2_data_path, {'Tg (C)': 'Tg'})
        ext_data = ext_data[['SMILES', 'Tg']].groupby('SMILES', as_index=False)['Tg'].mean()
        return self._merge(df, ext_data, 'Tg')

    def _merge_with_ext3_data(self, df):
        ext_data = self._load(self._ext3_data_path, {'density(g/cm3)': 'Density'}, excel=True)
        ext_data = ext_data[['SMILES', 'Density']]
        
        # Filter invalid rows
        mask = ext_data['SMILES'].notnull() & ext_data['Density'].notnull() & (ext_data['Density'] != 'nylon')
        ext_data = ext_data[mask].copy()        
        
        # Normalize density
        ext_data['Density'] -= 0.118
        ext_data = ext_data.groupby('SMILES', as_index=False)['Density'].mean()
        
        return self._merge(df, ext_data, 'Density')

    def _merge_with_ext4_data(self, df):
        ext_data = self._load(self._ext4_data_path, {'Tg [K]': 'Tg'}, excel=True)
        ext_data = ext_data[['SMILES', 'Tg']]      
        
        # Convert Kelvin â†’ Celsius
        ext_data['Tg'] -= 273.15
        ext_data = ext_data.groupby('SMILES', as_index=False)['Tg'].mean()
        
        return self._merge(df, ext_data,'Tg')
        
    def _merge_data(self, df):  
        """Merge all external datasets into training/test data"""
        print("ğŸ”— Starting merge with all external datasets ...")
        df = self._merge_with_ext1_data(df)
        df = self._merge_with_ext2_data(df)
        df = self._merge_with_ext3_data(df)
        df = self._merge_with_ext4_data(df)      
        return df
    
    def _aggregate_smiles(self, df):
        """Compute molecular descriptors using Mordred"""
        print("ğŸ§® Calculating molecular descriptors ...")
        mols = df['SMILES'].map(Chem.MolFromSmiles)
        calc = Calculator(descriptors, ignore_3D=self._ignore_3D)
        
        features = calc.pandas(mols).apply(pd.to_numeric, errors='coerce')
        
        # Drop bad features if too many NaNs or low variance
        if self._bad_features is None:
            self._bad_features = [
                c for c in features.columns 
                if features[c].nunique(dropna=False) <= 1 
                or features[c].isna().mean() > self._threshold
            ]
        
        return pd.concat([df, features], axis=1)
    
    def _cast_datatypes(self, df):
        """Convert column datatypes to optimize memory"""
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            
            val = non_null.iloc[0]
            
            if isinstance(val, numbers.Integral):
                target = 'int32'
            elif isinstance(val, numbers.Real):
                target = 'float32'
            else:
                target = 'string'
                
            df[col] = df[col].astype(target)
        
        return df

    def _stats(self, df):
        """Print dataset stats"""
        unique_smiles = df['SMILES'].unique()
        print(f"ğŸ“Š Dataset contains {len(unique_smiles)} unique SMILES.\n") 
        
        for p in self._properties:                
            count = df[p].notna().sum()
            print(f"âœ… Target {p} has {count} non-missing values.")        
    
    def info(self, df):
        """Print basic info about dataframe"""
        print(f"\nğŸ”� Shape of dataframe: {df.shape}")
        mem = df.memory_usage().sum() / 1024**2
        print(f"ğŸ’¾ Memory usage: {mem:.2f} MB\n")
        
        if 'Density' in df.columns: 
            self._stats(df)
    
    def apply_fe(self, path):
        """Main function: load, merge, process, and engineer features"""
        print(f"\nğŸš€ Applying Feature Engineering on {path} ...")
        
        # Step 1: Load dataset
        df = self._load_data(path)
        
        # Step 2: Merge external datasets if train (more than 2 cols)
        if df.shape[1] > 2:
            df = self._merge_data(df)
        
        # Keep raw copy before feature engineering
        raw_data = df.copy()
        raw_data = self._cast_datatypes(raw_data)     
            
        # Step 3: Generate molecular descriptors
        df = self._aggregate_smiles(df)
        
        # Step 4: Drop bad features
        df = df.drop(columns=self._bad_features, errors='ignore')        
        
        # Step 5: Optimize datatypes
        df = self._cast_datatypes(df)        
        
        # Step 6: Print summary
        self.info(df)
        
        # Step 7: Handle categorical features
        cat_cols = df.select_dtypes(include=['string']).columns.tolist()
        if cat_cols:
            df[cat_cols] = df[cat_cols].fillna('Missing').astype('category')
            print(f"ğŸ”  Converted {len(cat_cols)} string columns to categorical.")
        
        print("âœ… Feature Engineering Completed!\n")
        return df, raw_data, cat_cols



# Configure Feature Engineering object with paths & threshold
fe = FE(
    CFG.ext1_data_path, 
    CFG.ext2_data_path,
    CFG.ext3_data_path,
    CFG.ext4_data_path,
    CFG.threshold
)


# Apply FE on training data
print("ğŸ�‹ï¸� Processing Training Data ...")
train_data, raw_data, cat_cols = fe.apply_fe(CFG.train_path)



# Apply FE on test data
print("ğŸ§ª Processing Test Data ...")
test_data, _, _ = fe.apply_fe(CFG.test_path)



# Custom PyTorch dataset for handling SMILES strings
class SmilesDataset(Dataset):
    
    def __init__(self, smiles, targets, mode, n_aug=10):
        # Convert SMILES strings into RDKit Mol objects
        self.mols = [Chem.MolFromSmiles(s) for s in smiles]
        self.targets = targets
        self.mode = mode
        self.n_aug = n_aug
        print(f"[INIT] Dataset created with {len(self.mols)} molecules, mode={mode}, n_aug={n_aug}")

    def __len__(self):
        # Data augmentation: multiply dataset size by n_aug during training
        if self.mode == 'train':
            return len(self.mols) * self.n_aug
        return len(self.mols)

    def __getitem__(self, index):
        # Get real index (since augmented dataset is bigger)
        real_index = index % len(self.mols)        
        mol = self.mols[real_index]

        # Generate random SMILES for training, canonical SMILES otherwise
        if self.mode == 'train':
            s = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True, canonical=False)
        else:
            s = Chem.MolToSmiles(mol, canonical=True)

        print(f"[GETITEM] Index={index}, RealIdx={real_index}, SMILES={s}, Target={self.targets[real_index]}")
        return s, self.targets[real_index]


# Mean pooling over token embeddings
class MeanPooling(nn.Module):
    
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, last_hidden_state, attention_mask):
        # Expand mask to match hidden state shape
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        summed = (last_hidden_state * mask).sum(dim=1)   # Sum embeddings
        counts = mask.sum(dim=1).clamp(min=1e-9)         # Avoid division by zero
        result = summed / counts
        print(f"[MeanPooling] Output shape: {result.shape}")
        return result


# Pooling based on CLS token with dense layer + dropout
class ContextPooling(nn.Module):
    
    def __init__(self, hidden_size, dropout=0.4):
        super(ContextPooling, self).__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, last_hidden_state, attention_mask):
        # Take CLS token (first token representation)
        cls_token = last_hidden_state[:, 0]
        
        x = self.dense(cls_token)
        x = self.activation(x)
        x = self.dropout(x)

        print(f"[ContextPooling] Output shape: {x.shape}")
        return x


# ChemBERTa model wrapper
class ChemBERTa(nn.Module):

    def __init__(
        self,
        hidden_size,
        checkpoint,
        config_path=None,
        pretrained=True
    ):        
        super(ChemBERTa, self).__init__()

        # Load model config
        if config_path is None:
            self._config = AutoConfig.from_pretrained(checkpoint)
        else:
            with open(config_path, 'r') as f:
                self._config = AutoConfig.from_dict(json.load(f))

        # Load pretrained transformer or initialize from scratch
        if pretrained:
            self.transformer = AutoModel.from_pretrained(checkpoint, config=self._config)
            print(f"[ChemBERTa] Loaded pretrained model: {checkpoint}")
        else:
            self.transformer = AutoModel(self._config)
            print(f"[ChemBERTa] Initialized new model with config")

        # Pooling layers
        self.mean_pool = MeanPooling()
        self.context_pool = ContextPooling(hidden_size, dropout=self._config.hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def feature(self, input_ids, attention_mask):
        # Get hidden states from transformer
        hidden = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        print(f"[ChemBERTa.feature] Hidden shape: {hidden.shape}")

        # Combine mean and context pooling
        mean_features = self.mean_pool(hidden, attention_mask)
        context_features = self.context_pool(hidden, attention_mask)
        
        combined = mean_features + context_features
        print(f"[ChemBERTa.feature] Combined feature shape: {combined.shape}")
        return combined

    def forward(self, inputs):
        # Extract features then apply linear layer for regression
        feats = self.feature(inputs['input_ids'], inputs['attention_mask'])
        out = self.fc(feats)
        print(f"[ChemBERTa.forward] Output shape: {out.shape}")
        return out


class ChemBERTa_Inference:
    
    def __init__(
        self,
        data,
        cat_cols,
        root_path,
        n_splits,
        hidden_size,
        checkpoint,
        batch_size,
        context
    ):
        # Initialize inference class with dataset, configs, and model parameters
        self.data = data
        self.cat_cols = cat_cols
        self._root_path = root_path
        self._n_splits = n_splits
        self._hidden_size = hidden_size
        self._checkpoint = checkpoint
        self._batch_size = batch_size
        self._context = context
        
        # Tokenizer and config placeholders (to be loaded later)
        self._tokenizer = None
        self._config = None
        # Use GPU if available, otherwise CPU
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Properties we want to predict
        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        # Number of augmentations per sample during test-time augmentation (TTA)
        self._aug_factor = 300

    def _target_folder(self, target):
        # Locate the folder corresponding to a given property target
        for directory in os.listdir(self._root_path):
            full = os.path.join(self._root_path, directory)
            if os.path.isdir(full) and target.lower() in directory.lower():
                return full
                
        raise FileNotFoundError(f'No sub-folder for target {target} under {self._root_path}')

    def _load_models_for_target(self, target, title):
        # Load pretrained models, configs, and tokenizer for a specific property target
        base_path = self._target_folder(target)
        config_dir = os.path.join(base_path, 'Config')
        models_dir = os.path.join(base_path, 'Models')
        tokenizer_dir = os.path.join(base_path, 'Tokenizer')
        
        # Load tokenizer and config locally
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self._config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
        
        models = []
        
        for fold in range(1, self._n_splits + 1):
            model_path = os.path.join(models_dir, f'{title.lower()}-{target.lower()}-fold-{fold}.pth')
            print(f"Loading model from: {model_path}")  # Debugging print
            
            try:
                model = torch.load(model_path, map_location=self._device, weights_only=False)
            except TypeError:
                model = torch.load(model_path, map_location=self._device)
                
            model.to(self._device).eval()
            models.append(model)

        # Load OOF predictions file for debugging/validation
        title = title.replace('-', '_')
        csv_name = f'{title.lower()}_{target.lower()}_oof_preds.csv'
        oof_preds = pd.read_csv(os.path.join(base_path, csv_name))['oof_preds'].values
        
        print(f"Loaded {len(models)} models and OOF predictions for target: {target}")  # Debugging print
        return models, oof_preds

    def _predict_with_models(self, df, models):
        # Run inference with multiple models and apply test-time augmentation
        smiles_list = df['SMILES'].tolist()
        dummy_targets = [0] * len(smiles_list)  # Placeholder targets for dataset
        dataset = SmilesDataset(smiles_list, dummy_targets, mode='train', n_aug=self._aug_factor)

        # Collate function to tokenize SMILES strings into model inputs
        def tta_collate(batch):
            smiles_batch, _ = zip(*batch)
            enc = self._tokenizer(
                list(smiles_batch),
                padding=True,
                truncation=True,
                max_length=self._context,
                return_tensors='pt'
            )
            return enc['input_ids'], enc['attention_mask']

        # DataLoader for batching
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, collate_fn=tta_collate)

        fold_preds = []
        
        for model in models:
            preds = []
            with torch.no_grad():
                for ids, mask in loader:
                    batch = {'input_ids': ids.to(self._device), 'attention_mask': mask.to(self._device)}
                    out = model(batch).squeeze().cpu().numpy()
                    preds.extend(np.atleast_1d(out).tolist())
            
            # Average across augmentation factor
            preds = np.array(preds).reshape(self._aug_factor, len(smiles_list)).mean(axis=0)
            fold_preds.append(preds)
            print(f"Completed predictions for one model with {len(smiles_list)} samples")  # Debugging print
            
        # Average across all folds
        return np.mean(fold_preds, axis=0)

    def create_submission(self, test_data, title):
        # Create final ensemble predictions and submission file
        ids = test_data['id'].values
        sorted_idx = np.argsort(ids)  # Ensure correct ordering by ID
        inverse_idx = np.argsort(sorted_idx)
        
        sorted_test = test_data.iloc[sorted_idx].reset_index(drop=True)
        ensemble_preds = np.zeros((len(sorted_test), len(self._properties)), dtype='float32')
        
        for col_index, target in enumerate(self._properties):
            print(f"Predicting property: {target}")  # Debugging print
            models, _ = self._load_models_for_target(target, title)
            ensemble_preds[:, col_index] = self._predict_with_models(sorted_test, models)
            
        # Restore original test order
        ensemble_preds = ensemble_preds[inverse_idx]
        
        # Build submission DataFrame
        subm_data = pd.DataFrame(ensemble_preds, columns=self._properties)
        subm_data['id'] = ids
        subm_data = subm_data[['id'] + self._properties]
        
        # Save submission
        subm_data.to_csv('/kaggle/working/chemberta_submission.csv', index=False)
        print("Submission saved to chemberta_submission.csv")  # Debugging print
        
        display(subm_data.head())


class FastSMILES_Inference:
    
    def __init__(
        self,
        train_data,
        test_data, 
        cat_cols,
        root_path,
        ctb_weight,
        lgb_weight,
        xgb_weight,
        n_splits
    ):
        # Target properties to predict
        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Store inputs
        self.train_data = train_data
        self.test_data = test_data
        self.cat_cols = cat_cols
        self.root_path = root_path
        
        # Model weights for ensemble
        self._ctb_weight = ctb_weight
        self._lgb_weight = lgb_weight
        self._xgb_weight = xgb_weight
        
        # Number of folds
        self._n_splits = n_splits

        # To store OOF (out-of-fold) predictions and test predictions
        self._ctb_oof_preds = []
        self._lgb_oof_preds = []
        self._xgb_oof_preds = []

        self._ctb_preds = []
        self._lgb_preds = []
        self._xgb_preds = []

    def find_folder(self, target):
        """Find the folder for a given target property inside root_path"""
        print(f"Searching for folder of target: {target}")
        
        for d in os.listdir(self.root_path):
            full = os.path.join(self.root_path, d)
            if os.path.isdir(full) and d.lower() == target.lower():
                print(f"Found folder: {full}")
                return full
                
        raise FileNotFoundError(f'No folder for target {target} under {self.root_path}')
    
    def load_model(self, target, title):
        """Load models and OOF predictions for a specific target and model type"""
        base = os.path.join(self.find_folder(target), title)
        models_dir = os.path.join(base, 'Models')
        models = []
        
        print(f"Loading {title} models for target: {target}")
        
        for fold in range(1, self._n_splits + 1):
            path = os.path.join(models_dir, f'{title.lower()}-fold-{fold}.pkl')
            print(f" -> Loading model from: {path}")
            models.append(joblib.load(path))
        
        # Load OOF predictions
        path = os.path.join(base, f'{title.lower().replace("-", "_")}_oof_preds.csv')
        print(f" -> Loading OOF predictions from: {path}")
        oof_preds = pd.read_csv(path)['oof_preds'].values
        
        return models, oof_preds

    def infer_model(self, models):
        """Run inference using all folds and return averaged predictions"""
        print(f"Running inference on test set with {len(models)} models")
        data = self.test_data.drop(['id'], axis=1)
        return np.mean([model.predict(data) for model in models], axis=0)
    
    def load_and_infer_model(self, title):
        """Load models, get OOF/test predictions, and calculate performance"""
        print(f"\n=== Processing {title} models ===")
        
        # Convert categorical columns
        for col in self.cat_cols:
            self.train_data[col] = self.train_data[col].astype('category')
            self.test_data[col] = self.test_data[col].astype('category')
            print(f"Converted {col} to category type")
        
        for target in self._properties:
            print(f"\nTarget: {target}")
            
            # Only train where target is available
            valid_mask = self.train_data[target].notna().values
            valid_indices = np.where(valid_mask)[0]
            print(f" -> Using {len(valid_indices)} valid rows for training {target}")
            
            # Load models and run inference
            models, oof_preds = self.load_model(target, title)
            preds = self.infer_model(models)
            
            # Save predictions
            if title.startswith('CTB'):
                self._ctb_oof_preds.append(oof_preds)
                self._ctb_preds.append(preds)          
            elif title.startswith('LGB'):
                self._lgb_oof_preds.append(oof_preds)
                self._lgb_preds.append(preds)           
            elif title.startswith('XGB'):
                self._xgb_oof_preds.append(oof_preds)
                self._xgb_preds.append(preds)
        
        # Evaluate performance
        solution_data = self.train_data[['id'] + self._properties]
        
        if title.startswith('CTB'):
            oof_list = self._ctb_oof_preds
        elif title.startswith('LGB'):
            oof_list = self._lgb_oof_preds
        elif title.startswith('XGB'):
            oof_list = self._xgb_oof_preds
        
        oof_matrix = np.vstack(oof_list).T
        oof_data = pd.DataFrame(oof_matrix, columns=self._properties)
        oof_data['id'] = self.train_data['id'].values
        oof_data = oof_data[['id'] + self._properties]
        
        model_score = score(solution_data, oof_data, 'id')
        print(f" >> wMAE for {title}: {model_score:.3f}")           
    
    def inference(self):
        """Perform ensemble inference across CTB, LGB, and XGB models"""
        print("\n=== Performing Ensemble Inference ===")
        
        # Weighted OOF and test predictions
        ensemble_oof_preds = np.vstack([
            (self._ctb_weight * ctb +
             self._lgb_weight * lgb +
             self._xgb_weight * xgb)
            for ctb, lgb, xgb in zip(self._ctb_oof_preds, self._lgb_oof_preds, self._xgb_oof_preds)
        ]).T
        
        ensemble_preds = np.vstack([
            (self._ctb_weight * ctb +
             self._lgb_weight * lgb +
             self._xgb_weight * xgb)
            for ctb, lgb, xgb in zip(self._ctb_preds, self._lgb_preds, self._xgb_preds)
        ]).T
        
        # Build evaluation dataframe
        solution_data = self.train_data[['id'] + self._properties]
        oof_data = pd.DataFrame(ensemble_oof_preds, columns=self._properties)
        oof_data['id'] = self.train_data['id'].values
        oof_data = oof_data[['id'] + self._properties]
        
        # Compute ensemble score
        ensemble_score = score(solution_data, oof_data, 'id')
        print(f" >> wMAE for Ensemble: {ensemble_score:.3f}")
        
        # Save submission
        subm_data = pd.DataFrame(ensemble_preds, columns=self._properties)
        subm_data['id'] = self.test_data['id'].values
        subm_data = subm_data[['id'] + self._properties]
        subm_data.to_csv('/kaggle/working/fastsmiles_submission.csv', index=False)
        print("Saved submission file: fastsmiles_submission.csv")
        
        display(subm_data.head())


# Initialize ChemBERTa inference class
chemberta_inference = ChemBERTa_Inference(
    raw_data,                 # Raw dataset containing train/test splits
    cat_cols,                 # List of categorical columns
    CFG.chemberta_root_path,  # Path where pretrained ChemBERTa models are stored
    CFG.n_splits,             # Number of CV folds
    CFG.hidden_size,          # Transformer hidden size
    CFG.checkpoint,           # HuggingFace checkpoint path
    CFG.batch_size,           # Batch size for inference
    CFG.context               # Boolean: use ContextPooling or not
)

# # Debug prints
# print("âœ… ChemBERTa_Inference initialized")
# print(f"Data shape: {raw_data.shape}")
# print(f"Categorical columns: {cat_cols}")
# print(f"Model checkpoint: {CFG.checkpoint}")
# print(f"Hidden size: {CFG.hidden_size}, Batch size: {CFG.batch_size}, Context: {CFG.context}")


# Initialize FastSMILES inference class
fastsmiles_inference = FastSMILES_Inference(
    train_data,               # Training dataset (with target properties)
    test_data,                # Test dataset (for submission predictions)
    cat_cols,                 # List of categorical columns to cast properly
    CFG.fastsmiles_root_path, # Root path where trained FastSMILES models are stored
    CFG.ctb_weight,           # Weight for CatBoost predictions in ensemble
    CFG.lgb_weight,           # Weight for LightGBM predictions in ensemble
    CFG.xgb_weight,           # Weight for XGBoost predictions in ensemble
    CFG.n_splits              # Number of CV folds used in training
)

# # Debug prints (optional for sanity checks)
# print("âœ… FastSMILES_Inference initialized")
# print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
# print(f"Categorical columns: {cat_cols}")
# print(f"Weights -> CatBoost: {CFG.ctb_weight}, LightGBM: {CFG.lgb_weight}, XGBoost: {CFG.xgb_weight}")
# print(f"Number of folds: {CFG.n_splits}")


# 1) Run ChemBERTa inference (single model submission)
chemberta_inference.create_submission(test_data, 'ChemBERTa-A')
# -> Generates predictions from ChemBERTa for test_data
# -> Saves submission CSV (likely chemberta_submission.csv)


# 2) Run FastSMILES CatBoost inference
fastsmiles_inference.load_and_infer_model('CTB-A')
# -> Loads all CatBoost models for Tg, FFV, Tc, Density, Rg
# -> Collects OOF preds + test preds
# -> Prints wMAE score for CatBoost


# 3) Run FastSMILES LightGBM inference
fastsmiles_inference.load_and_infer_model('LGB-A')
# -> Same as above, but for LightGBM


# 4) Run FastSMILES XGBoost inference
fastsmiles_inference.load_and_infer_model('XGB-A')
# -> Same as above, but for XGBoost


# 5) Ensemble inference
fastsmiles_inference.inference()
# -> Combines CTB/LGB/XGB predictions using weights
# -> Computes ensemble OOF score (wMAE)
# -> Saves final ensemble submission as fastsmiles_submission.csv
# -> Displays preview of submission dataframe


# ğŸ“‚ Load both individual submissions
chemberta_submission = pd.read_csv('/kaggle/working/chemberta_submission.csv')
fastsmiles_submission = pd.read_csv('/kaggle/working/fastsmiles_submission.csv')

# ğŸ�¯ Get target property columns (exclude 'id')
target_cols = [col for col in chemberta_submission.columns if col != 'id']

# ğŸ§ª Start ensemble as a copy of ChemBERTa
ensemble = chemberta_submission.copy()

# â�• Weighted blending: ChemBERTa ğŸ§  + FastSMILES âš¡
for col in target_cols:
    ensemble[col] = (
        CFG.chemberta_weight * chemberta_submission[col] + 
        CFG.fastsmiles_weight * fastsmiles_submission[col]
    )

# ğŸ’¾ Save final blended submission
ensemble.to_csv('/kaggle/working/submission.csv', index=False)

# ğŸ‘€ Peek at results
display(ensemble.head())

