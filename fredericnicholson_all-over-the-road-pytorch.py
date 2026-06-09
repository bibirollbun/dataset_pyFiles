# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import polars  as pl # data processing, CSV file I/O (e.g. pd.read_csv)
from polars import selectors as cs

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data_df = pl.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
print (data_df.schema)




print (f" accident risk 1 in training data : {data_df.filter ( pl.col('accident_risk') >= 0.98).shape [0]}")

print (data_df.group_by ("num_lanes").agg (pl.col('id').len(), pl.col('accident_risk').mean()))
print (data_df.group_by ("num_reported_accidents").agg (pl.col('id').len(), pl.col('accident_risk').mean()))
print (data_df.group_by ("speed_limit").agg (pl.col('id').len(), pl.col('accident_risk').mean()))





'''
input polars Dataframe containing either the train or the test data 
num lanes, reported accidents and speed limit will be converted using one hot encoding 
categorical features are encoded  using one hot encoding
all values converted to Float 32 

drop "id", "accident_risk", "speed_limit", so only categorical features are used 
return converted polars Dataframe
'''
def transform (df :pl.DataFrame) -> pl.DataFrame :
    
     result = df.with_columns (more_than_50 = pl.col('speed_limit') > 50) 
     result = result.with_columns (speed_limit_fraction =  pl.col('speed_limit') / 70, 
                                    base_risk = pl.col("curvature") * 0.3 +
                                               (pl.col("lighting")  == "night") * 0.2 +
                                               (pl.col("weather") != "clear") * 0.2 +  
                                               (pl.col("speed_limit") > 55) * 0.2 +
                                               (pl.col("num_reported_accidents") > 2) * 0.1,
                                     first_curvator_digit = (pl.col("curvature") * 10).cast(pl.UInt8),
                                     second_curvator_digit = (pl.col("curvature") * 100).cast(pl.UInt8) % 10) 
 
     result =  result.to_dummies (    cs.string() | cs.by_name ("speed_limit", "num_reported_accidents", "num_lanes"  ))
     # drop_list = [c for c in ["id", "accident_risk", "speed_limit"] if c in df.columns]
          
     result = result.drop (cs.by_name ("id", "accident_risk", "speed_limit", require_all = False))
     result = result.select (cs.all().cast (pl.Float32))
     return result 


transform (data_df).head (3)
    


# data_df = data_df.filter (pl.col("accident_risk") < 1)

# data_df = data_df.sample (fraction = 0.2)

features = transform (data_df).to_numpy() 
print (f" {features.shape=}")
print (f" {features [0] = }")


labels = data_df.select ("accident_risk")
labels = labels.to_numpy().squeeze(-1)

print (f" {labels.shape=}")
print (f" {labels = }")



%%time
!pip install lightning 


from torch.utils.data import DataLoader, Dataset

import torch
from torch import nn, optim

class RoadDataset(Dataset):
    def __init__(self, data, labels=None, is_train=True):
        self.data = data
        self.labels = labels
        self.is_train = is_train
        print ("Building Road dataset:")
        print (f"data type = { type (data) =} data shape  {data.shape = }")
        print (f"data row 0 : { data [0]}")
        print (f"labels shape  {labels.shape = }")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = torch.tensor (self.data[idx])
        if self.is_train:
            label = torch.tensor (self.labels[idx])
            return sequence, label
        else:
            return sequence

road_dataset = RoadDataset (features, labels, True)

print (road_dataset.__getitem__(11))


%%time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

sample_length = len (labels)
split = 0.2
val_len = int (sample_length * split) 
train_len = sample_length - val_len 

train_set, val_set = torch.utils.data.random_split(road_dataset, [train_len, val_len])
 
dataloader_train = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)

dataloader_validate = DataLoader(
    val_set, batch_size=BATCH_SIZE)




class RoadDNN (nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1)
        )
        
    def forward(self, x):
        h = self.shared(x)
        return h.squeeze(-1)



import lightning as L
import torchmetrics 


from torchmetrics.regression import MeanSquaredError
# define the LightningModule
class Road_training(L.LightningModule):
    def __init__(self, x_size, lr = 1e-5):
        super().__init__()
        self.model = RoadDNN (x_size)
        self.loss_fn = MeanSquaredError(False)   
        self.lr = lr 
    def forward (self, x, labels = None) :
        output = self.model (x)
        loss = 0 
        if labels is not None :
            loss = self.loss_fn(output, labels)
        return loss, output
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        if torch.all(y == 0) :
           print (f" batch {batch_idx =} received label tensor y with all zeros") 
           loss = 1 
        else : 
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            self.log("loss", loss)
        return {"loss" : loss
                }
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop.
        # it is independent of forward
        x, y = batch
        
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)
    
        return {"val_loss" : loss 
                }    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# taken from the rate finder 
road_training = Road_training(features.shape [1])


from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_callback = ModelCheckpoint(
    dirpath = "checkpoints", 
    filename = "best_checkpoint", 
    save_top_k = 1, 
    verbose = True, 
    monitor = "val_loss",
    mode = "min" 
)


earlyStopping = EarlyStopping("val_loss", min_delta=0.0, patience=10)


trainer = L.Trainer(limit_train_batches=BATCH_SIZE, max_epochs=1500, 
                     log_every_n_steps =32, 
                     # auto_lr_find=True,
                     
                      callbacks = [checkpoint_callback, 
                                   earlyStopping])


from lightning.pytorch.tuner import Tuner
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(road_training, train_dataloaders=dataloader_train , min_lr = 0.5e-5, max_lr = 1e-3, num_training = 50)
best_lr = lr_finder.suggestion()
print(f"Suggested learning rate: {best_lr}")

road_training.hparams.lr = best_lr



%%time
if (__name__ == "__main__") : 
  trainer.fit(model=road_training, train_dataloaders=dataloader_train, val_dataloaders=dataloader_validate)


def extract_number (string) -> int :

    number = ""
    for char in string:
        if char.isdigit():
            number += char
        elif number:  # Stop once the first number is fully captured
            break

    if number:
        first_number = int(number)
        
    else :
        first_number = -1 
    return first_number


import os
checkpoints = []
for dirname, _, filenames in os.walk('/kaggle/working/checkpoints'):
    for filename in filenames:
        checkpoints.append(filename)
        print (filename)
if len (checkpoints) > 1 :
    high_version  = max ([ extract_number (filename) for filename in checkpoints] )
    last_checkpoint = f'/kaggle/working/checkpoints/best_checkpoint-v{high_version}.ckpt' 
else :  
    high_version = -1
    last_checkpoint = f'/kaggle/working/checkpoints/best_checkpoint.ckpt' 
print (f"the last checkpoint is { high_version = } ")


print (f"using {last_checkpoint = } ")
trained_model = Road_training.load_from_checkpoint (
        last_checkpoint,
        x_size = features.shape [1])

trained_model.freeze()



comp_df = pl.DataFrame ({"true" : [0], "predict" : [0] })

for x_val, y_val in (dataloader_validate) :
# Move the tensor to the GPU
    if torch.cuda.is_available():
        gpu_tensor = x_val.to('cuda')  
        
    y_pred = trained_model (gpu_tensor) 
    #print (y_pred)
    #print (type(y_pred))
    #print (y_pred [1].shape)
    #print ("now detach")
    y_pred  = y_pred [1].cpu().detach().numpy()
    #print (y_pred)
    #print (type(y_pred))
    #print (y_pred.shape)

    comp_df_iter = pl.DataFrame ({"true" : y_val.numpy(), "predict" : y_pred })
    comp_df = pl.concat ([comp_df, comp_df_iter], how = "vertical_relaxed")

display (comp_df)




comp_df = comp_df.with_columns ( error = ((pl.col ('true') - pl.col ('predict'))/0.025).abs().round (0) )
import seaborn as sns

sns.scatterplot (data = comp_df.to_pandas(), x = "true", y = "predict", hue = "error")


comp_df.describe()


test_df = pl.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_df = pl.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


test_features = transform (test_df) 

test_features = test_features.to_torch().to(device) 

predict = trained_model (test_features)

print (predict)


p = pl.Series (predict [1].cpu().detach().numpy())

p


submission = sample_df.with_columns (p.alias ("accident_risk"))

submission.write_csv ("submission.csv")




display (submission.select ("accident_risk").describe())

display (data_df.select ("accident_risk").describe())


