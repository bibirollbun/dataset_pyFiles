!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import torch
import pandas as pd
import joblib
from transformers import PreTrainedModel, AutoConfig, BertModel, BertTokenizerFast, BertConfig, AutoModel, AutoTokenizer
from sklearn.metrics import mean_absolute_error
from torch import nn
from transformers.activations import ACT2FN
from tqdm import tqdm
import numpy as np

class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
        self.dense = nn.Linear(pooler_size, pooler_size)
        
        dropout_prob = getattr(config, 'pooler_dropout', config.hidden_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.activation = getattr(config, 'pooler_hidden_act', config.hidden_act)
        self.config = config

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0] # CLS token
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.activation](pooled_output)
        return pooled_output

class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        
        self.pooler = ContextPooler(config)

        pooler_output_dim = getattr(config, 'pooler_hidden_size', config.hidden_size)
        self.output = torch.nn.Linear(pooler_output_dim, 1) # Still predicting one label at a time. Kinda stupid

    def forward(
        self,
        input_ids,
        scaler,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        pooled_output = self.pooler(outputs.last_hidden_state)
        
        # Final regression output
        regression_output = self.output(pooled_output)

        loss = None
        true_loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()

            unscaled_labels = scaler.inverse_transform(labels.cpu().numpy())
            unscaled_outputs = scaler.inverse_transform(regression_output.cpu().detach().numpy())
            
            loss = loss_fn(regression_output, labels)
            true_loss = mean_absolute_error(unscaled_outputs, unscaled_labels)

        return {
            "loss": loss,
            "logits": regression_output,
            "true_loss": true_loss
        }


import torch
import pandas as pd
import joblib
from transformers import PreTrainedModel, AutoConfig, BertModel, BertTokenizerFast, BertConfig, AutoModel, AutoTokenizer
from sklearn.metrics import mean_absolute_error
from torch import nn
from transformers.activations import ACT2FN
from tqdm import tqdm
import numpy as np

class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
        self.dense = nn.Linear(pooler_size, pooler_size)
        
        dropout_prob = getattr(config, 'pooler_dropout', config.hidden_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.activation = getattr(config, 'pooler_hidden_act', config.hidden_act)
        self.config = config

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0] # CLS token
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.activation](pooled_output)
        return pooled_output

class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        
        self.pooler = ContextPooler(config)

        pooler_output_dim = getattr(config, 'pooler_hidden_size', config.hidden_size)
        self.output = torch.nn.Linear(pooler_output_dim, 1) # Still predicting one label at a time. Kinda stupid

    def forward(
        self,
        input_ids,
        scaler,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        pooled_output = self.pooler(outputs.last_hidden_state)
        
        # Final regression output
        regression_output = self.output(pooled_output)

        loss = None
        true_loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()

            unscaled_labels = scaler.inverse_transform(labels.cpu().numpy())
            unscaled_outputs = scaler.inverse_transform(regression_output.cpu().detach().numpy())
            
            loss = loss_fn(regression_output, labels)
            true_loss = mean_absolute_error(unscaled_outputs, unscaled_labels)

        return {
            "loss": loss,
            "logits": regression_output,
            "true_loss": true_loss
        }
BATCH_SIZE = 16

def tokenize_smiles(seq):
    seq = [tokenizer.cls_token + smiles for smiles in seq] # If we pass a string, tokenizer will smartly think we want to create a sequence for each symbol
    tokenized = tokenizer(seq, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    return tokenized

def load_model(path):
    config = AutoConfig.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
    model = CustomModel(config).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    return model


def make_predictions(model, scaler, smiles_seq):
    aggregated_preds = []
    for smiles in smiles_seq:
        smiles = [smiles]
        smiles_tokenized = tokenize_smiles(smiles)

        input_ids = smiles_tokenized['input_ids'].cuda()
        attention_mask = smiles_tokenized['attention_mask'].cuda()
        with torch.no_grad():
            preds = model(input_ids=input_ids, scaler=scaler, attention_mask=attention_mask)['logits'].cpu().numpy()
        
        true_preds = scaler.inverse_transform(preds).flatten()
        aggregated_preds.append(true_preds.tolist())
    return np.array(aggregated_preds)


test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
test_copy = test.copy()

smiles_test = test['SMILES'].to_list()

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

scalers = joblib.load('/kaggle/input/smiles-bert-models/target_scalers.pkl')
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
import pandas as pd
import numpy as np
from rdkit import Chem
import random
from typing import Optional, List, Union

def augment_smiles_dataset(df: pd.DataFrame,
                               smiles_column: str = 'SMILES',
                               augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
                               n_augmentations: int = 100,
                               preserve_original: bool = True,
                               random_seed: Optional[int] = None) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def apply_augmentation_strategy(smiles: str, strategy: str) -> List[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = []
            
            if strategy == 'enumeration':
                # Standard SMILES enumeration
                for _ in range(n_augmentations):
                    enum_smiles = Chem.MolToSmiles(mol, 
                                                 canonical=False, 
                                                 doRandom=True,
                                                 isomericSmiles=True)
                    augmented.append(enum_smiles)
            
            elif strategy == 'kekulize':
                # Kekulization variants
                try:
                    Chem.Kekulize(mol)
                    kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    augmented.append(kek_smiles)
                except:
                    pass
            
            elif strategy == 'stereo_enum':
                # Stereochemistry enumeration
                for _ in range(n_augmentations // 2):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    no_stereo = Chem.MolToSmiles(mol)
                    augmented.append(no_stereo)
            
            return list(set(augmented))  # Remove duplicates
            
        except Exception as e:
            print(f"Error in {strategy} for {smiles}: {e}")
            return [smiles]
    
    augmented_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_smiles = row[smiles_column]
        
        if preserve_original:
            original_row = row.to_dict()
            original_row['augmentation_strategy'] = 'original'
            original_row['is_augmented'] = False
            augmented_rows.append(original_row)
        
        for strategy in augmentation_strategies:
            strategy_smiles = apply_augmentation_strategy(original_smiles, strategy)
            
            for aug_smiles in strategy_smiles:
                if aug_smiles != original_smiles:
                    new_row = row.to_dict().copy()
                    new_row[smiles_column] = aug_smiles
                    new_row['augmentation_strategy'] = strategy
                    new_row['is_augmented'] = True
                    augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")
    
    return augmented_df

test = augment_smiles_dataset(test)
preds_mapping = {}

for i in tqdm(range(len(targets))):
    target = targets[i]
    scaler = scalers[i]

    model_path = f'/kaggle/input/private-smile-bert-models/warm_smiles_model_{target}_target.pth' # Not actually warm
    
    model = load_model(model_path)
    true_preds = []

    for i, data in test.groupby('id'):
        test_smiles = data['SMILES'].to_list()
        augmented_preds = make_predictions(model, scaler, test_smiles)
    
        average_pred = np.median(augmented_preds)
    
        true_preds.append(float(average_pred.flatten()[0]))

    preds_mapping[target] = true_preds
submission = pd.DataFrame(preds_mapping)
submission['id'] = test_copy['id']
submission.to_csv('submission.csv', index=False)


import pandas as pd
import numpy as np
from rdkit import Chem
import random
from typing import Optional, List, Union

def augment_smiles_dataset(df: pd.DataFrame,
                               smiles_column: str = 'SMILES',
                               augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
                               n_augmentations: int = 100,
                               preserve_original: bool = True,
                               random_seed: Optional[int] = None) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def apply_augmentation_strategy(smiles: str, strategy: str) -> List[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = []
            
            if strategy == 'enumeration':
                # Standard SMILES enumeration
                for _ in range(n_augmentations):
                    enum_smiles = Chem.MolToSmiles(mol, 
                                                 canonical=False, 
                                                 doRandom=True,
                                                 isomericSmiles=True)
                    augmented.append(enum_smiles)
            
            elif strategy == 'kekulize':
                # Kekulization variants
                try:
                    Chem.Kekulize(mol)
                    kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    augmented.append(kek_smiles)
                except:
                    pass
            
            elif strategy == 'stereo_enum':
                # Stereochemistry enumeration
                for _ in range(n_augmentations // 2):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    no_stereo = Chem.MolToSmiles(mol)
                    augmented.append(no_stereo)
            
            return list(set(augmented))  # Remove duplicates
            
        except Exception as e:
            print(f"Error in {strategy} for {smiles}: {e}")
            return [smiles]
    
    augmented_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_smiles = row[smiles_column]
        
        if preserve_original:
            original_row = row.to_dict()
            original_row['augmentation_strategy'] = 'original'
            original_row['is_augmented'] = False
            augmented_rows.append(original_row)
        
        for strategy in augmentation_strategies:
            strategy_smiles = apply_augmentation_strategy(original_smiles, strategy)
            
            for aug_smiles in strategy_smiles:
                if aug_smiles != original_smiles:
                    new_row = row.to_dict().copy()
                    new_row[smiles_column] = aug_smiles
                    new_row['augmentation_strategy'] = strategy
                    new_row['is_augmented'] = True
                    augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")
    
    return augmented_df

test = augment_smiles_dataset(test)


preds_mapping = {}

for i in tqdm(range(len(targets))):
    target = targets[i]
    scaler = scalers[i]

    model_path = f'/kaggle/input/private-smile-bert-models/warm_smiles_model_{target}_target.pth' # Not actually warm
    
    model = load_model(model_path)
    true_preds = []

    for i, data in test.groupby('id'):
        test_smiles = data['SMILES'].to_list()
        augmented_preds = make_predictions(model, scaler, test_smiles)
    
        average_pred = np.median(augmented_preds)
    
        true_preds.append(float(average_pred.flatten()[0]))

    preds_mapping[target] = true_preds


submission = pd.DataFrame(preds_mapping)
submission['id'] = test_copy['id']
submission.to_csv('submission.csv', index=False)


submission

