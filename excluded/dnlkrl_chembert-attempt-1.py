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

!pip install transformers scikit-learn pandas numpy --no-index --find-links /kaggle/input/downloadedlibs/
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'

from transformers import AutoTokenizer, AutoModel

import torch
from torch import nn

from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from torch import nn
from torch.optim import AdamW
import matplotlib.pyplot as plt

import os
import joblib 

df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
local_model_path = "/kaggle/input/openpoly-chembert-dataset-1/openpoly_chembert_dataset/model_files"
absolute_path = os.path.abspath(local_model_path)



class ChemBERTRegressor(nn.Module):
    def __init__(self, model_name, dropout_p=0.1):
        super(ChemBERTRegressor, self).__init__()
        self.chembert = AutoModel.from_pretrained(absolute_path)
        # Get the size of the ChemBERT output embeddings
        embedding_dim = self.chembert.config.hidden_size 
        # The regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_p),      # <--- Add more layers? 
            nn.Linear(embedding_dim, 1) # Single target, set to 5 after completing imputation expmnts. 
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.chembert(input_ids=input_ids, attention_mask=attention_mask)  # Pass tokenised input to ChemBERT
        cls_embedding = outputs.last_hidden_state[:, 0] # Using first token embedding
        prediction = self.regressor(cls_embedding)
        return prediction

class PolymerDataset(Dataset):
    def __init__(self, dataframe, tokenizer, target_name):
        self.tokenizer = tokenizer
        self.smiles = dataframe['SMILES'].tolist()
        self.targets = dataframe[target_name].tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_str = self.smiles[idx]
        target_val = self.targets[idx]
        
        inputs = self.tokenizer(
            smiles_str,
            padding='max_length',
            truncation=True,
            max_length=512, # A reasonable max length for polymers
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze() # squueze req for DataLoader to be happy
        attention_mask = inputs['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': torch.tensor(target_val, dtype=torch.float)
        }


HF_HUB_OFFLINE=1 # Poss redundant

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

target_name_list = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
df_test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
df_submission = pd.DataFrame(data = df_test.loc[:, 'id'])

INPUT_PATH = "/kaggle/input/openpoly-weights-2/openpoly_weights_2/"  # Path for models and scalers from training notebook

# LOOPING OVER ALL TARGETS 

for target_col in target_name_list:
    print(f"--- Predicting for target: {target_col} ---")
   
    # 2. Prepare the test data
    df_test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

    tokenizer = AutoTokenizer.from_pretrained(absolute_path)
    
    test_dataset = PolymerDataset(df_test, tokenizer, target_name='id') # target_name as id is a bodge
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 0) # num_workers = 0 is an attempted debug, may not be ideal

    model = ChemBERTRegressor(absolute_path)
    model_path = os.path.join(INPUT_PATH, f'best_model_{target_col}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler_path = os.path.join(INPUT_PATH, f'scaler_{target_col}.pkl')
    scaler = joblib.load(scaler_path)
    
    
    print("NOW PREDICTING FOR "+target_col)
    # 3. Generate predictions
    predictions_gpu = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Move predictions to CPU and convert to a NumPy array or list
            predictions_gpu.append(outputs)
    
    predictions_scaled = torch.cat(predictions_gpu, dim=0) 
    final_predictions = scaler.inverse_transform(predictions_scaled.cpu().numpy()) 

    df_submission[target_col] = final_predictions



tg_mean = 96.452313684
ffv_mean = 0.367211995
density_mean = 0.9854843785473083
tc_mean = 0.25633409252439104
rg_mean = 16.41978670954397

df_submission = df_submission.replace(-np.inf,np.nan)
df_submission = df_submission.replace(np.inf,np.nan)
df_submission['Tg'].fillna(tg_mean, inplace=True)
df_submission['FFV'].fillna(ffv_mean, inplace=True)
df_submission['Tc'].fillna(tc_mean, inplace=True)
df_submission['Density'].fillna(density_mean, inplace=True)
df_submission['Rg'].fillna(rg_mean, inplace=True)

print(df_submission)

df_submission['id'] = df_submission['id'].astype(str)

df_submission.to_csv('submission.csv', index = False)

