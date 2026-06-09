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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

test_ids = test['id']


interaction_features = [
    ('humidity', 'cloud', 'humidity_cloud_interaction'),
    ('humidity', 'sunshine', 'humidity_sunshine_interaction')
]

for col1, col2, new_col in interaction_features:
    train[new_col] = train[col1] * train[col2]
    test[new_col] = test[col1] * test[col2]

train['cloud_sunshine_ratio'] = train['cloud'] / (train['sunshine'] + 1e-5)
test['cloud_sunshine_ratio'] = test['cloud'] / (test['sunshine'] + 1e-5)

train['relative_dryness'] = 100 - train['humidity']
test['relative_dryness'] = 100 - test['humidity']

train['sunshine_percentage'] = train['sunshine'] / (train['sunshine'] + train['cloud'] + 1e-5)
test['sunshine_percentage'] = test['sunshine'] / (test['sunshine'] + test['cloud'] + 1e-5)

train['weather_index'] = (0.4 * train['humidity']) + (0.3 * train['cloud']) - (0.3 * train['sunshine'])
test['weather_index'] = (0.4 * test['humidity']) + (0.3 * test['cloud']) - (0.3 * test['sunshine'])

train.drop(columns=['day'], axis=1, inplace=True, errors='ignore')
test.drop(columns=['day'], axis=1, inplace=True, errors='ignore')



def preprocess_data(train, test):
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    test['winddirection'].fillna(train['winddirection'].median(), inplace=True)
    
    X = train.drop('rainfall', axis=1)
    y = train['rainfall']
    
    return X, y, test

X, y, test = preprocess_data(train, test)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled), 
    torch.FloatTensor(y_train.values)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_scaled), 
    torch.FloatTensor(y_val.values)
)
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))



BATCH_SIZE = 64

class RainfallDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def train_dataloader(self):
        return DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=2
        )
    
    def val_dataloader(self):
        return DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            num_workers=2
        )

class RainfallModel(pl.LightningModule):
    def __init__(self, input_size, pos_weight):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.pos_weight = pos_weight
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        
        acc = accuracy_score(y.cpu(), (preds > 0.5).cpu())
        auc = roc_auc_score(y.cpu(), preds.cpu())
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_auc', auc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

pos_weight = torch.tensor([len(y_train) / sum(y_train)])



model = RainfallModel(
    input_size=X_train.shape[1],
    pos_weight=pos_weight
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    verbose=True
)

checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./',
    filename='best_model',
    save_top_k=1
)

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[early_stop, checkpoint],
    accelerator='auto',
    devices=1
)


summary(model)


data_module = RainfallDataModule()
trainer.fit(model, datamodule=data_module)


best_model = RainfallModel.load_from_checkpoint(
    'best_model.ckpt',
    input_size=X_train.shape[1],
    pos_weight=pos_weight
)


best_model = best_model.to('cuda')  

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
preds = []
best_model.eval()
with torch.no_grad():
    for x in test_loader:
        inputs = x[0].to(best_model.device)
        
        logits = best_model(inputs).squeeze()
        batch_preds = torch.sigmoid(logits)
        
        preds.extend(batch_preds.cpu().numpy())


submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': np.array(preds) 
})

submission.to_csv('submission.csv', index=False, float_format='%.6f')


submission.head()

