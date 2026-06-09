!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl



!pip download lifelines



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)

train.head()


plt.hist(train.loc[train.efs==1,"efs_time"], bins = 100, label = "efs = 1, Yes event")
plt.hist(train.loc[train.efs==0,"efs_time"], bins = 100, label = "efs = 0, Maybe event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"efs_time"] = train.loc[train.efs==0,"efs_time"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += 2*len(train)
train.y = train.y/train.y.max()
train.y = np.log(train.y)
train.y -= train.y.mean()
train.y *= -1.0

plt.hist(train.loc[train.efs==1,"y"], bins = 100, label = "efs = 1, Yes event")
plt.hist(train.loc[train.efs==0,"y"], bins = 100, label = "efs = 0, Maybe event")
plt.xlim((-5,5))
plt.xlabel("Transformed target y")
plt.ylabel("Dennsity")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
Features = []
for c in train.columns:
    if c not in RMV:
        Features.append(c)

print(f"There are {len(Features)} FEATURES: {Features}")


Cats = []
for c in Features:
    if train[c].dtype == "object":
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
        Cats.append(c)
    elif not "age" in c:
        train[c] = train[c].astype("str")
        test[c] = test[c].astype("str")
        Cats.append(c)

print(f"In these features, there are {len(Cats)} CATEGORICAL FEATURES: {Cats}")


Cat_Size = []
Cat_Emb = []
Nums = []

combined = pd.concat([train,test],axis=0,ignore_index = True)

#print("Combined data shape:", combined.shape )

print("We LABEL ENCODE the CATEGORICAL FEATURES: ")


for c in Features:
    if c in Cats:
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")

        n = combined[c].nunique()
        mx = combined[c].max()
        mn = combined[c].min()
        print(f'{c} has ({n}) unique values')

        Cat_Size.append(mx+1)
        Cat_Emb.append(int(np.ceil(np.sqrt(mx+1))))
    else:
        if combined[c].dtype == "float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype == "int64":
            combined[c] == combined[c].astype("int32")

        mean = combined[c].mean()
        std_dev = combined[c].std()

        combined[c] = (combined[c] - mean)/std_dev
        combined[c] = combined[c].fillna(0)

        Nums.append(c)


train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from sklearn.model_selection import KFold
import numpy as np

def build_model(Cat_Size, Cat_Emb, Nums):
    class TabularModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Embedding layers
            self.embeddings = nn.ModuleList([
                nn.Embedding(size, emb_size) 
                for size, emb_size in zip(Cat_Size, Cat_Emb)
            ])
            
            # Calculate input size
            total_emb_dim = sum(Cat_Emb)
            
            # Dense layers
            self.layers = nn.Sequential(
                nn.Linear(total_emb_dim + len(Nums), 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            
        def forward(self, x_cat, x_num):
            embs = []
            for i, emb in enumerate(self.embeddings):
                x = emb(x_cat[:, i])
                x = x.view(x.size(0), -1)  # Explicit flatten
                embs.append(x)
            x = torch.cat(embs + [x_num], dim=1)
            return self.layers(x)
            
    return TabularModel()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPEATS = 3
FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 512

# Initialize predictions
oof_nn = np.zeros(len(train))
pred_nn = np.zeros(len(test))

# Convert test data to tensors once
X_test_cats = torch.tensor(test[Cats].values, dtype=torch.long).to(DEVICE)
X_test_nums = torch.tensor(test[Nums].values, dtype=torch.float32).to(DEVICE)

for r in range(REPEATS):
    print(f"{'#'*25}\n### REPEAT {r+1} ###\n{'#'*25}")
    
    kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f" {'#'*25}\n ### Fold {fold+1} ###\n {'#'*25}")
        
        # Prepare data
        X_train_cats = torch.tensor(train.loc[train_idx, Cats].values, dtype=torch.long).to(DEVICE)
        X_train_nums = torch.tensor(train.loc[train_idx, Nums].values, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(train.loc[train_idx, "y"].values, dtype=torch.float32).to(DEVICE)
        
        X_valid_cats = torch.tensor(train.loc[valid_idx, Cats].values, dtype=torch.long).to(DEVICE)
        X_valid_nums = torch.tensor(train.loc[valid_idx, Nums].values, dtype=torch.float32).to(DEVICE)
        y_valid = torch.tensor(train.loc[valid_idx, "y"].values, dtype=torch.float32).to(DEVICE)
        
        # Initialize model
        model = build_model(Cat_Size, Cat_Emb, Nums).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.01)         # CHANGE: Increased initial learning rate from 0.001 to 0.01
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        criterion = nn.MSELoss()
        
        # Training loop
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Training
            model.train()
            # CHANGE: Added separate training phase with loss calculation
            # Training phase
            train_loss_sum = 0.0
            num_batches = 0

            for i in range(0, len(X_train_cats), BATCH_SIZE):
                batch_cats = X_train_cats[i:i+BATCH_SIZE]
                batch_nums = X_train_nums[i:i+BATCH_SIZE]
                batch_y = y_train[i:i+BATCH_SIZE]
                
                optimizer.zero_grad()
                outputs = model(batch_cats, batch_nums)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() # CHANGE: Accumulate training loss
                num_batches += 1

            train_loss_avg = train_loss_sum / num_batches             # CHANGE: Calculate average training loss

            
            # Validation
            model.eval()
            valid_loss_sum = 0.0
            with torch.no_grad():
                valid_preds = model(X_valid_cats, X_valid_nums).squeeze()
                valid_loss = criterion(valid_preds, y_valid).item()

            valid_loss_avg = valid_loss_sum
                
            scheduler.step(valid_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {valid_loss:.4f}, Learning Rate: {current_lr:.6f}")
            # Early stopping
            print(f"Epoch {epoch+1}/{EPOCHS} - "
                  f"loss: {train_loss_avg:.4f} - val_loss: {valid_loss_avg:.4f} - "
                  f"learning_rate: {current_lr:.4e}")
            
            # Early stopping logic (optional)
            if valid_loss_avg < best_loss:
                best_loss = valid_loss_avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:  # Stop if no improvement for 10 epochs
                    print("Early stopping triggered.")
                    break
        
        # Predict OOF
        model.eval()
        with torch.no_grad():
            oof_nn[valid_idx] += model(X_valid_cats, X_valid_nums).squeeze().cpu().numpy()
            pred_nn += model(X_test_cats, X_test_nums).squeeze().cpu().numpy()

# Average predictions
oof_nn /= REPEATS
pred_nn /= (FOLDS * REPEATS)

print("Training completed!")
print(f"OOF MSE: {np.mean((oof_nn - train['y'].values) ** 2)}")


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_nn
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for NN =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = pred_nn
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

