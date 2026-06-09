!pip install -q lightning


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, StandardScaler


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col = "id")
# X_train = pd.read_pickle("/kaggle/input/optimal-fertilizers-datasets/FFS_train_v5.pkl")
# X_test = pd.read_pickle("/kaggle/input/optimal-fertilizers-datasets/FFS_test_v5.pkl")
X_train = pd.read_pickle("/kaggle/input/optimal-fertilizers-datasets/train_v5.pkl")
X_test = pd.read_pickle("/kaggle/input/optimal-fertilizers-datasets/test_v5.pkl")


le_target = LabelEncoder()
train['Fertilizer Name'] = le_target.fit_transform(train['Fertilizer Name'])
class_names = le_target.classes_


categorical_features = ["Soil_Type", "Crop_Type","Cluster"]
numerical_fetures = [col for col in X_train.columns if col not in categorical_features]


from torch.utils.data import random_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader,TensorDataset
import lightning as L
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os


x_train_numerical = torch.from_numpy(X_train[numerical_fetures].values).to(torch.float32)
x_train_categorical = torch.from_numpy(X_train[categorical_features].values).to(torch.int64)

x_test_numerical = torch.from_numpy(X_test[numerical_fetures].values).to(torch.float32)
x_test_categorical = torch.from_numpy(X_test[categorical_features].values).to(torch.int64)


X_torch = torch.cat([x_train_numerical,x_train_categorical],1)
y_torch = torch.from_numpy(train["Fertilizer Name"].values).to(torch.int64)
categorical_embedding_sizes = [(X_train[col].nunique(), 16) for col in categorical_features]

X_test_torch = torch.cat([x_test_numerical,x_test_categorical],1)


class_sample_count = np.unique(y_torch, return_counts=True)[1]
weight = torch.tensor(1. / class_sample_count).to(torch.float32)


from sklearn.model_selection import train_test_split

X_torch_train, X_torch_val, y_torch_train, y_torch_val = train_test_split(X_torch, y_torch, test_size=0.2, 
                                                                          random_state=42, stratify=y_torch)


train_dataset = TensorDataset(X_torch_train, y_torch_train)
val_dataset = TensorDataset(X_torch_val, y_torch_val)


class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.3):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        for i in layers:
            all_layers.append(nn.LazyLinear(i))
            all_layers.append(nn.SiLU())
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
    

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    
    def forward(self, X):
        x_categorical = X[:,-len(categorical_features):].to(torch.int64)
        x_numerical = X[:,:-len(categorical_features)]
        
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        
        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x


def mapk(y_true, y_pred, k=3):
    pred_top3 = torch.topk(nn.functional.softmax(y_pred),3).indices
    matches = pred_top3 == y_true.unsqueeze(-1)
    max_indices = torch.argmax(matches.to(torch.long) ,axis = 1) + 1
    all_false = matches.sum(axis = 1) == 0
    scores = 1/max_indices
    scores[all_false] = 0
    return torch.mean(scores)


class LighningModel(L.LightningModule):
    def __init__(self, model, num_epochs, learning_rate, batch_size):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(weight = weight)
        self.apply(self._init_weights)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
    def forward(self,X):
        out = self.model(X)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_pred = self(x)
        
        loss = self.loss_fn(y_pred, y.squeeze())
        self.log("train_loss", loss, prog_bar = True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y.squeeze())        
        self.log("val_loss", loss, prog_bar = True, on_epoch=True)

        score = mapk(y,y_pred)
        self.log("MAP@3", score, prog_bar = True, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, eps=1e-08, amsgrad = True)

        return {
        "optimizer": opt,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(opt,factor=0.5, min_lr = 3e-8),
            "monitor": "val_loss",
            "frequency": 1,
        },
    }
    
    def on_validation_epoch_end(self):
        # Log the learning rate.
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True)
                 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers = os.cpu_count(),shuffle=True)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers = os.cpu_count())


from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

L.seed_everything(42)
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
# checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="MAP@3", mode = "max")

batch_size = 2048
num_epochs = 5000
lr = 3e-4
layers = [1024,512, 256]
dropout = 0.3
pytorch_model = Model(categorical_embedding_sizes, len(numerical_fetures), train["Fertilizer Name"].nunique(), layers = layers, p = dropout)
model = LighningModel(pytorch_model, num_epochs, learning_rate = lr, batch_size = batch_size)
logger = CSVLogger(save_dir="logs/")


#Initialize Parameter for Lazy Layer
model(torch.zeros(next(iter(model.train_dataloader()))[0].shape))

trainer = L.Trainer(
    max_epochs=num_epochs,
    accelerator="auto",
    devices="auto",
    logger=logger,
	callbacks=[
        EarlyStopping('val_loss', patience=num_epochs//20), 
        checkpoint_callback,
        StochasticWeightAveraging(swa_lrs=1e-2),
    ],
    gradient_clip_val= 0.5,
    precision="16-mixed",
    # precision="bf16-mixed",
    accumulate_grad_batches=5,
    # detect_anomaly=True,
    # overfit_batches = 0.1
)


trainer.fit(model)


import matplotlib.pyplot as plt


metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
metrics = metrics.groupby('epoch').mean() 
plt.plot(metrics["train_loss_epoch"])
plt.plot(metrics["val_loss"])


device = "cuda" if torch.cuda.is_available() else "cpu"
pytorch_model = Model(categorical_embedding_sizes, len(numerical_fetures), train["Fertilizer Name"].nunique(), layers = layers)

model = LighningModel.load_from_checkpoint(checkpoint_callback.best_model_path,
                                           model = pytorch_model, num_epochs = num_epochs, 
                                           learning_rate = lr, batch_size = batch_size)
model.to(device)




model.eval()

with torch.inference_mode():
    y_hat = model(X_test_torch.to(device))
    y_hat_top3 = torch.topk(nn.functional.softmax(y_hat),3).indices
    y_hat_top3 = y_hat_top3.cpu()
    pred_names = [le_target.inverse_transform(pred) for pred in y_hat_top3]
    pred_labels = [' '.join(name) for name in pred_names]
    
results = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv",index_col = "id")
results["Fertilizer Name"] = pred_labels
results.to_csv("submission.csv")

