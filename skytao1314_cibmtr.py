import pandas as pd, numpy as np
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import os
import imageio
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import KFold



train_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


train_df.head()


print(f'train_df shape: {train_df.shape}')
print(f'test_df shape: {test_df.shape}')



plt.hist(train_df.loc[train_df.efs==1,'efs_time'],bins = 100, label  = "efs = 1")
plt.hist(train_df.loc[train_df.efs==0,'efs_time'],bins = 100, label  = "efs = 0")
plt.xlabel('efs_time')
plt.ylabel('count')
plt.title('efs_time distribution')
plt.legend()
plt.show()



train_df['y'] = train_df.efs_time.values
mx = train_df.loc[train_df.efs==1,'efs_time'].max()
mn = train_df.loc[train_df.efs==0,'efs_time'].min()
train_df.loc[train_df.efs == 0, 'y'] = train_df.loc[train_df.efs == 0, 'y'] + mx - mn
train_df.y = train_df.y.rank()
train_df.loc[train_df.efs == 0, 'y'] += 2*len(train_df)
train_df.y = train_df.y / train_df.y.max()
train_df.y = np.log( train_df.y )
train_df.y -= train_df.y.mean()
train_df.y *= -1.0

plt.hist(train_df.loc[train_df.efs==1,'y'],bins = 100, label  = "efs = 1")
plt.hist(train_df.loc[train_df.efs==0,'y'],bins = 100, label  = "efs = 0")
plt.xlabel('y')
plt.ylabel('count')
plt.title('y distribution')
plt.legend()
plt.show()



RMV = ['ID','efs','efs_time', 'y']

FEATURES = [col for col in train_df.columns if col not in RMV]

print(f'特征总计: {len(FEATURES)}，分别是: {FEATURES}')


CATS = []
for c in FEATURES:
    if train_df[c].dtype=="object":
        train_df[c] = train_df[c].fillna("NAN")
        test_df[c] = test_df[c].fillna("NAN")
        CATS.append(c)
    elif not "age" in c:
        train_df[c] = train_df[c].astype("str")
        test_df[c] = test_df[c].astype("str")
        CATS.append(c)
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")



CAT_SIZE = []
CAT_EMB = []
NUMS = []

combined = pd.concat([train_df,test_df],axis=0,ignore_index=True)
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
        
train_df = combined.iloc[:len(train_df)].copy()
test_df = combined.iloc[len(train_df):].reset_index(drop=True).copy()


%%time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

class SurvivalModel(nn.Module):
    def __init__(self, cat_sizes, cat_embs, num_features):
        super(SurvivalModel, self).__init__()
        self.cat_emb_layers = nn.ModuleList([
            nn.Embedding(cat_size, cat_emb) for cat_size, cat_emb in zip(cat_sizes, cat_embs)
        ])
        self.num_features = num_features
        self.fc1 = nn.Linear(sum(cat_embs) + num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x_cats, x_nums):
        # Process categorical features with embeddings
        embs = [emb_layer(x_cats[:, i]) for i, emb_layer in enumerate(self.cat_emb_layers)]
        
        # Combine embeddings and numerical features
        x = torch.cat(embs + [x_nums], dim=1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x




def train_model(model, train_loader, valid_loader, optimizer, criterion, device, epochs):
    for epoch in range(epochs):
        model.train()
        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x_cat, x_num)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_cat, x_num, y in valid_loader:
                x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
                output = model(x_cat, x_num)
                valid_loss += criterion(output, y).item()
        print(f"Epoch {epoch+1}, Validation Loss: {valid_loss/len(valid_loader)}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 15
REPEATS = 3
FOLDS = 5
kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)

oof_nn = np.zeros(len(train_df))
pred_nn = np.zeros(len(test_df))


    
for i, (train_index, test_index) in enumerate(kf.split(train_df)):
    print(f"### Fold {i+1} ###")
        
    X_train_cats = torch.tensor(train_df.loc[train_index, CATS].values, dtype=torch.long)
    X_train_nums = torch.tensor(train_df.loc[train_index, NUMS].values, dtype=torch.float32)
    y_train = torch.tensor(train_df.loc[train_index, "y"].values, dtype=torch.float32).unsqueeze(1)
        
    X_valid_cats = torch.tensor(train_df.loc[test_index, CATS].values, dtype=torch.long)
    X_valid_nums = torch.tensor(train_df.loc[test_index, NUMS].values, dtype=torch.float32)
    y_valid = torch.tensor(train_df.loc[test_index, "y"].values, dtype=torch.float32).unsqueeze(1)
        
    train_dataset = TensorDataset(X_train_cats, X_train_nums, y_train)
    valid_dataset = TensorDataset(X_valid_cats, X_valid_nums, y_valid)
        
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512)
        
    model = SurvivalModel(CAT_SIZE, CAT_EMB, len(NUMS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
        
    train_model(model, train_loader, valid_loader, optimizer, criterion, device, EPOCHS)
        
    model.eval()
    with torch.no_grad():
        oof_nn[test_index] += model(X_valid_cats.to(device), X_valid_nums.to(device)).cpu().numpy().flatten()

    X_test_cats = torch.tensor(test_df[CATS].values, dtype=torch.long)
    X_test_nums = torch.tensor(test_df[NUMS].values, dtype=torch.float32)
    with torch.no_grad():
        pred_nn += model(X_test_cats.to(device), X_test_nums.to(device)).cpu().numpy().flatten()

oof_nn /= REPEATS
pred_nn /= (FOLDS * REPEATS)


from torchmetrics import MeanSquaredError
import torch

def score(y_true, y_pred):
    """
    使用 torchmetrics 计算 MSE。
    """
    mse = MeanSquaredError()
    y_true_tensor = torch.tensor(y_true["efs"].values, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred["prediction"].values, dtype=torch.float32)
    return mse(y_pred_tensor, y_true_tensor).item()

# 调用方法
y_true = train_df[["ID", "efs", "efs_time", "race_group"]].copy()
y_pred = train_df[["ID"]].copy()
y_pred["prediction"] = oof_nn
m = score(y_true, y_pred)
print(f"\nOverall CV for NN = {m:.4f}")



sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = pred_nn
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

