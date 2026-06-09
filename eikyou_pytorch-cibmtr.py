!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")


import warnings
warnings.filterwarnings("ignore")


RMV = ["ID","efs","efs_time","y"]
RACES = train['race_group']
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
        CATS.append(c)
    elif not "age" in c:
        train[c] = train[c].astype("str")
        test[c] = test[c].astype("str")
        CATS.append(c)
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


CAT_SIZE = []
CAT_EMB = []
NUMS = []

combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

print("We LABEL ENCODE the CATEGORICAL FEATURES: ")

for c in FEATURES:
    if c in CATS:
        # LABEL ENCODE
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        #combined[c] = combined[c].astype("category")

        n = combined[c].nunique()
        mn = combined[c].min()
        mx = combined[c].max()
        print(f'{c} has ({n}) unique values')

        CAT_SIZE.append(mx+1) 
        CAT_EMB.append( int(np.ceil( np.sqrt(mx+1))) ) 
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
            
        m = combined[c].mean()
        s = combined[c].std()
        combined[c] = (combined[c]-m)/s
        combined[c] = combined[c].fillna(0)
        
        NUMS.append(c)
        
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += 2*len(train)
train.y = train.y / train.y.max()
train.y = np.log( train.y )
train.y -= train.y.mean()
train.y *= -1.0

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlim((-5,5))
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


import torch
from torch import nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, KFold


import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ResidualBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, p=0.1):
        super(ResidualBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p), 
        )

        self.activation = nn.LeakyReLU()

        

    def forward(self, X):
        residual = X
        
        out = self.fc(X)
        # Add residual connection
        out += residual
        
        return self.activation(out)  # Apply activation after residual connection



class KaplanMeierNet(nn.Module):

    def __init__(self, cat_sizes, cat_emb_dims, num_features, hidden_dim, num_blocks, p=0.3):
        super().__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_size, emb_dim) 
            for cat_size, emb_dim in zip(cat_sizes, cat_emb_dims)
        ])

        input_dim = sum(cat_emb_dims) + len(num_features)

        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p),
            nn.LeakyReLU()
        )
        

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, p) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)
        
        
    def forward(self, X_cats, X_nums):

        X_cats = X_cats.long()
        
        embs = [emb(X_cats[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = [emb.flatten(start_dim=1) for emb in embs]

        X = torch.cat(embs + [X_nums], dim=-1)

        X = self.input_fc(X)

        for block in self.residual_blocks:
            X = block(X)
        
        return self.output_layer(X)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X_cats, X_nums, y=None):
        self.X_cats = torch.tensor(X_cats).float()
        self.X_nums = torch.tensor(X_nums).float()
        if y is not None:
            self.y = torch.tensor(y).float()
        else:
            self.y = None

    def __len__(self):
        return len(self.X_cats)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cats[idx], self.X_nums[idx], self.y[idx].unsqueeze(0)
        return self.X_cats[idx], self.X_nums[idx]


dataset = MyDataset(train[CATS].values, train[NUMS].values, train['y'].values)

test_dataset = MyDataset(test[CATS].values, test[NUMS].values)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def lr_lambda(epoch):
    return 0.9 ** epoch


%%time
REPEATS = 3
FOLDS = 5
EPOCHS = 10
LR = 1e-2
batch_size = 32
criterion = nn.MSELoss()

oof = np.zeros(len(train))
pred = np.zeros(len(test))

for r in range(REPEATS):
    print("#"*25)
    print(f"### REPEAT {r+1} ###")
    print("#"*25)

    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42 + r)

    for i, (train_index, val_index) in enumerate(kfold.split(train, RACES)):

        print(f"   Fold {i + 1}/{FOLDS}")
        
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        seed_everything(42 + r)
        model = KaplanMeierNet(CAT_SIZE, CAT_EMB, NUMS, 256, 10, 0.15).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0
            for X_cats_batch, X_nums_batch, y_batch in train_loader:
                optimizer.zero_grad()
                out = model(X_cats_batch.to(device), X_nums_batch.to(device))
                loss = criterion(out, y_batch.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            #Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_cats_batch, X_nums_batch, y_batch in val_loader:
                    out = model(X_cats_batch.to(device), X_nums_batch.to(device))
                    loss = criterion(out, y_batch.to(device))
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            print(f'      Train loss: {train_loss}  Val loss: {val_loss} Learning Rate: {scheduler.get_lr()[0]:.6f}')
            scheduler.step()
        
        model.eval()
        #OOF prediction
        val_pred = []
        with torch.no_grad():
            for X_cats_batch, X_nums_batch, y_batch in val_loader:
                out = model(X_cats_batch.to(device), X_nums_batch.to(device))
                val_pred.append(out.cpu())
                
        oof[val_index] = np.concatenate(val_pred).squeeze()

        #TEST prediction
        test_pred = []
        with torch.no_grad():
            for X_cats_batch, X_nums_batch in test_loader:
                out = model(X_cats_batch.to(device), X_nums_batch.to(device))
                test_pred.append(out.cpu())

        pred += np.concatenate(test_pred).squeeze()


oof /= REPEATS
pred /= (FOLDS*REPEATS)


from metric import score
from scipy.stats import rankdata

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for NN =",m)


ids = test['ID']

output = pd.DataFrame(data={'ID': ids, 'prediction': pred})
output.to_csv('submission.csv', index=False)

