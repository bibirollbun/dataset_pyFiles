import polars as pl
import numpy as np


train = pl.read_csv('/kaggle/input/pump-fun-graduation-february-2025/train.csv')
train = train.filter(pl.col('is_valid'))

test = pl.read_csv('/kaggle/input/pump-fun-graduation-february-2025/test_unlabeled.csv')


chunk_data = pl.read_csv('/kaggle/input/pump-fun-graduation-february-2025/chunk_*.csv')


slot_mins = pl.concat([test[['mint', 'slot_min']], train[['mint', 'slot_min']]])


all_mints = pl.concat([train[['mint']], test[['mint']]])


all_mints_slots = all_mints.join(pl.DataFrame({'relative_slot': range(100)}), how='cross')


aggregated = chunk_data \
          .join(slot_mins, left_on='base_coin', right_on='mint', how='inner') \
          .with_columns(((pl.col('slot') - pl.col('slot_min'))).alias('relative_slot')) \
          .with_columns((pl.col('direction') == 'sell').alias('is_sell'),
                        (pl.col('direction') != 'sell').alias('is_buy'),) \
          .with_columns((pl.col('quote_coin_amount') * pl.col('is_sell')).alias('sell_amount'),
                        (pl.col('quote_coin_amount') * pl.col('is_buy')).alias('buy_amount')) \
          .group_by('base_coin', 'relative_slot') \
          .agg(pl.sum('buy_amount'), 
               pl.sum('sell_amount'),
               pl.sum('is_sell').alias('sell_transactions'),
               pl.sum('is_buy').alias('buy_transactions'),
               pl.sum('fee'),
               pl.sum('consumed_gas'),) \
          .join(all_mints_slots, left_on=['base_coin', 'relative_slot'], right_on=['mint', 'relative_slot'], how='right') \
          .fill_null(0) \
          .sort(["mint", "relative_slot"])


columns = [
    'buy_amount',
    'sell_amount',
    'sell_transactions',
    'buy_transactions',
    'fee',
    'consumed_gas',
]

numpied_all = aggregated.select(columns).to_numpy().reshape(-1, 100, len(columns))


sorted_mints = pl.concat([train[['mint']].with_columns(is_train=True),
                          test[['mint']].with_columns(is_train=False)]).sort('mint')


sorted_train = train.sort('mint')
sorted_test = test.sort('mint')


assert aggregated[['mint']][::100].equals(sorted_mints[['mint']])


del aggregated, chunk_data, all_mints_slots


numpied_train = numpied_all[sorted_mints['is_train']]
numpied_test = numpied_all[~sorted_mints['is_train']]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl

X = torch.tensor(numpied_train, dtype=torch.float32)
y = torch.tensor(sorted_train["has_graduated"].to_numpy(), dtype=torch.float32)

full_dataset = TensorDataset(X, y)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

X_test = torch.tensor(numpied_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


del numpied_all, numpied_train, numpied_test


class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.bn_input = nn.BatchNorm1d(input_size) 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.reset_parameters()
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        if type(x) == list:
            x = x[0]
        batch_size, seq_len, input_size = x.size()
        
        x = x.transpose(1, 2)
        x = self.bn_input(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(self.dropout(last_hidden))
        return torch.sigmoid(out).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        
        self.log('train_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches
            ),
            "interval": "step",
            "frequency": 1
        }
    
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


model = LSTMClassifier(input_size=len(columns))


trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices="auto",
    log_every_n_steps=10
)

trainer.fit(model, train_loader, val_loader)


test_predictions = np.concatenate(trainer.predict(model, dataloaders=test_loader))


submission = sorted_test[['mint']]
submission = submission.with_columns(has_graduated=test_predictions)
submission.write_csv('submission.csv')




