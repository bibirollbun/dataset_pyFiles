!pip install rdkit



# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score
from torch.utils.data import Dataset,DataLoader
from transformers import TFAutoModel, AutoTokenizer, RobertaTokenizer,RobertaModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



#importing datasets
df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
df.head()


#Now we will be exploring df_train
df.describe()


df.isnull().mean() * 100


cols = ['Tg','FFV','Tc','Density','Rg']
sns.pairplot(df[cols])
plt.show()


# from the above graphs we can see that
# Even thought there are many missing values in the data
# whatever available data is very consistent
# outliers are few in number, so removing them might lead
# to overfitting, so I'll keep them as it is


train, val = train_test_split(df, test_size = 0.1, random_state = 69 )



from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5)
train[cols] = imputer.fit_transform(train[cols])
train


#preparing data
smiles_train,smiles_val = train['SMILES'],val['SMILES']
target_train, target_val = train[['FFV', 'Tg', 'Tc','Density','Rg']].values, val[['FFV', 'Tg', 'Tc','Density','Rg']].values
train_mask = ~torch.isnan(torch.tensor(target_train, dtype=torch.float32))
val_mask = ~torch.isnan(torch.tensor(target_val, dtype=torch.float32))


columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
N_t = [df[col].notna().sum() for col in columns]
R_t = [df[col].max() - df[col].min() for col in columns]
N_t = torch.tensor(N_t, dtype=torch.float32)
R_t = torch.tensor(R_t, dtype=torch.float32)
num_train, target_size = target_train.shape
num_val, _  = target_val.shape
print("Number of Examples  int train set is" ,num_train)
print("Number of Examples  int train set is" ,num_val)
print("Number of Target_size is" ,target_size)
print("N_t is" ,N_t)
print("R_t is" ,R_t)



# tokenizer function, we will be using the default RobertaTokenizer
def tokenized_smiles(smiles):
  tokenizer = RobertaTokenizer.from_pretrained("/kaggle/input/chemberta-zinc-base-v1/transformers/seyonec-chemberta-zinc-base-v1/1/models/ChemBERTa-zinc-base-v1")
  tokens = tokenizer(smiles,max_length = 128, padding = 'max_length', truncation = True, return_tensors = 'pt')
  return tokens


#tokenizing the inputs
train_tokens = tokenized_smiles(smiles_train.tolist())
val_tokens = tokenized_smiles(smiles_val.tolist())


# Our model class, We used the pretrained ChemBERTa model, and added a hidden and output layer which we will fine tune
class myModel(nn.Module):
  def __init__(self,output_dim = 5,hidden_dim = 256,dropout = 0.2):
    super(myModel,self).__init__()
    self.chemberta = RobertaModel.from_pretrained("/kaggle/input/chemberta-zinc-base-v1/transformers/seyonec-chemberta-zinc-base-v1/1/models/ChemBERTa-zinc-base-v1")
    for param in self.chemberta.parameters():
      param.requires_grad = False
    self.fc1 = nn.Linear(self.chemberta.config.hidden_size,hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
    self.dropout  = nn.Dropout(dropout)
    self.out = nn.Linear(hidden_dim // 2,output_dim)
  def forward(self,input_ids,attention_mask):
    outputs = self.chemberta(input_ids = input_ids,attention_mask = attention_mask)
    cls_output = outputs.last_hidden_state[:,0]
    x = self.fc1(cls_output)
    x = F.relu(x)
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.out(x)
    x = F.relu(x)
    return x


#class for making our final inputs organized
class SMILESdataset(Dataset):
  def __init__(self,tokens,targets,mask):
    self.input_ids = tokens['input_ids']
    self.attention_mask = tokens['attention_mask']
    self.targets = targets
    self.mask = mask

  def __len__(self):
    return len(self.targets)

  def __getitem__(self,idx):
    return{
        'input_ids' : self.input_ids[idx],
        'attention_mask':self.attention_mask[idx],
        'targets' : self.targets[idx],
        'mask' : self.mask[idx]
    }


train_dataset = SMILESdataset(train_tokens, torch.tensor(target_train, dtype = torch.float32),train_mask)
val_dataset  = SMILESdataset(val_tokens,  torch.tensor(target_val, dtype=torch.float32),val_mask)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32,shuffle = False)



class WeightedMAELoss(nn.Module):
    def __init__(self, N_t, R_t):
        super().__init__()
        self.register_buffer("weights", self.compute_weights(N_t, R_t))

    def compute_weights(self, N_t, R_t):
        raw_weights = 1.0 / (torch.sqrt(N_t) * R_t)
        normalized_weights = raw_weights / raw_weights.sum()
        return normalized_weights  # shape: [num_tasks]

    def forward(self, preds, targets, mask):
        preds = preds.float()
        targets = targets.float()
        preds = preds[mask]
        targets = targets[mask]
        mask = mask.float()
        abs_error = torch.abs(preds - targets) # [batch, num_tasks]
        count = mask.sum(dim=0).clamp(min=1.0)          # [num_tasks]
        mae_per_task = abs_error.sum(dim=0) / count     # [num_tasks]

        # Weighted sum of per-task MAE
        return (mae_per_task * self.weights).sum()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = myModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
N_t = N_t.to(device)
R_t = R_t.to(device)
best_val_loss = float('inf')
patience = 3
wait = 0
num_epochs = 100
torch.autograd.set_detect_anomaly(True)
fn_loss = WeightedMAELoss(N_t, R_t)
for epoch in range(num_epochs):
    model.train()
    train_losses = []

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = fn_loss(outputs, targets, mask)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['mask'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = fn_loss(outputs, targets, mask)
            val_losses.append(loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
  #early stopping and saving the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), "OpenPolymerModel1Best.pt")
        print("Best Model Saved")
    else:
        wait += 1
        if wait >= patience:
            print("EARLY STOPPING TRIGGERED")
            break
        print(f"No improvement. Early stop patience:{wait}/{patience}")




# torch.save(model.state_dict(), "FinalWeightsModel1.pt")
# print("Final model weights saved to FinalWeightsModel1.pt")



# Final Loss was
# Training Loss = 75.9
# Val loss = 108.0





df_test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')




sub_tokens = tokenized_smiles(df_test['SMILES'].tolist())

#making these just for dataloader
sub_masks = torch.ones((len(df_test), 5), dtype=torch.bool)
sub_targets = torch.zeros((len(df_test), 5))


test_dataset = SMILESdataset(sub_tokens, sub_targets, sub_masks)
test_loader = DataLoader(test_dataset, batch_size=32)

model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_preds.append(outputs.cpu())

preds = torch.cat(all_preds, dim=0).numpy()



submission_df = pd.DataFrame()
submission_df["id"] = df_test["id"]
submission_df["Tg"] = preds[:, 1]
submission_df["FFV"] = preds[:, 0]
submission_df["Tc"] = preds[:, 2]
submission_df["Density"] = preds[:, 3]
submission_df["Rg"] = preds[:, 4]


submission_df.to_csv("/kaggle/working/submission.csv", index=False)





