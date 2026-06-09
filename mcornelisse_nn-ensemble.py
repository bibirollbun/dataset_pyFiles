!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


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


RMV = ["ID","efs","efs_time","y"]
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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold


# Define the number of epochs and the learning rate schedule
EPOCHS = 4
LRS = [0.01]*2 + [0.001]*1 + [0.0001]*1  # 4 epochs: 0.01, 0.01, 0.001, 0.0001

def lrfn(epoch):
    if epoch < len(LRS):
        return LRS[epoch]
    else:
        return LRS[-1]

# Visualize the Learning Rate Schedule
rng = list(range(EPOCHS))
lr_y = [lrfn(x) for x in rng]

plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.show()

print(f"Learning rate schedule: {lr_y[0]:.3g} → {max(lr_y):.3g} → {lr_y[-1]:.3g}")


class Model(nn.Module):
    def __init__(self, cat_sizes, cat_embs, num_numerical):
        super().__init__()
        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_size, emb_dim) 
            for cat_size, emb_dim in zip(cat_sizes, cat_embs)
        ])
        # Calculate total embedding dimension
        total_emb_dim = sum(cat_embs)
        # Define dense layers
        self.fc1 = nn.Linear(total_emb_dim + num_numerical, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x_cat, x_num):
        # Process each categorical feature through its embedding
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            # Extract the i-th categorical column and apply embedding
            emb = emb_layer(x_cat[:, i])
            embeddings.append(emb)
        # Concatenate embeddings and numerical features
        concatenated = torch.cat(embeddings + [x_num], dim=1)
        # Pass through dense layers
        x = torch.relu(self.fc1(concatenated))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_model(model, train_loader, valid_loader, optimizer, criterion, scheduler, device, epochs):
    # Lists to keep track of training and validation losses each epoch
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        #  Training Loop
        model.train()
        running_train_loss = 0.0
        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x_cat, x_num)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

        # Compute average training loss for this epoch
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, y in valid_loader:
                x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
                output = model(x_cat, x_num)
                val_loss = criterion(output, y).item()
                running_val_loss += val_loss
        
        epoch_val_loss = running_val_loss / len(valid_loader)
        val_losses.append(epoch_val_loss)
        
        # Print validation loss each epoch
        #print(f"Epoch {epoch+1}/{epochs}, "
        #      f"Train Loss: {epoch_train_loss:.4f}, "
        #      f"Validation Loss: {epoch_val_loss:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        # Step the scheduler to update the learning rate for the next epoch
        scheduler.step()
    
    return train_losses, val_losses



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import RepeatedKFold

EPOCHS = 4
REPEATS = 3
FOLDS = 5

# Define ensemble seeds to produce different model initializations and randomness
#ensemble_seeds = [42, 123, 999]
ensemble_seeds = [42, 123, 999, 2024, 17, 88, 314, 7777, 555, 8675309]

# Initialize arrays to hold ensemble predictions
ensemble_oof = np.zeros(len(train))
ensemble_pred = np.zeros(len(test))
ensemble_count = 0

all_ensemble_fold_train_losses = []
all_ensemble_fold_val_losses = []

for seed in ensemble_seeds:
    print(f"--- Training ensemble member with seed {seed} ---")
    # Set seeds for reproducibility for this ensemble member
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a RepeatedKFold object for this seed
    rkf = RepeatedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=seed)
    
    # Initialize predictions for this ensemble member
    oof_nn = np.zeros(len(train))
    pred_nn = np.zeros(len(test))
    
    for fold, (train_index, test_index) in enumerate(rkf.split(train)):
        print(f"### Seed {seed} - Fold {fold+1} ###")
        
        # Prepare training data for this fold
        X_train_cats = torch.tensor(train.loc[train_index, CATS].values, dtype=torch.long)
        X_train_nums = torch.tensor(train.loc[train_index, NUMS].values, dtype=torch.float32)
        y_train = torch.tensor(train.loc[train_index, "y"].values, dtype=torch.float32).unsqueeze(1)
        
        # Prepare validation data for this fold
        X_valid_cats = torch.tensor(train.loc[test_index, CATS].values, dtype=torch.long)
        X_valid_nums = torch.tensor(train.loc[test_index, NUMS].values, dtype=torch.float32)
        y_valid = torch.tensor(train.loc[test_index, "y"].values, dtype=torch.float32).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_cats, X_train_nums, y_train)
        valid_dataset = TensorDataset(X_valid_cats, X_valid_nums, y_valid)
        
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=512)
        
        # Initialize a new model for this fold
        model = Model(CAT_SIZE, CAT_EMB, len(NUMS)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lrfn(epoch) / 0.01)
        criterion = nn.MSELoss()
        
        # Train the model on the current fold
        train_losses, val_losses = train_model(model, train_loader, valid_loader,
                                               optimizer, criterion, scheduler, device, EPOCHS)
        all_ensemble_fold_train_losses.append(train_losses)
        all_ensemble_fold_val_losses.append(val_losses)
        
        # Get out-of-fold predictions
        model.eval()
        with torch.no_grad():
            oof_nn[test_index] += model(X_valid_cats.to(device), X_valid_nums.to(device))\
                                        .cpu().numpy().flatten()
        
        # Predict on the test data
        X_test_cats = torch.tensor(test[CATS].values, dtype=torch.long)
        X_test_nums = torch.tensor(test[NUMS].values, dtype=torch.float32)
        with torch.no_grad():
            pred_nn += model(X_test_cats.to(device), X_test_nums.to(device))\
                                         .cpu().numpy().flatten()
    
    # Average predictions across folds (and repeats) for this ensemble member
    oof_nn /= REPEATS
    pred_nn /= (FOLDS * REPEATS)
    
    # Accumulate ensemble predictions
    ensemble_oof += oof_nn
    ensemble_pred += pred_nn
    ensemble_count += 1

# Final ensemble predictions: average over all ensemble members
ensemble_oof /= ensemble_count
ensemble_pred /= ensemble_count


avg_train_losses = np.mean(all_ensemble_fold_train_losses, axis=0)  # shape: (EPOCHS,)
avg_valid_losses = np.mean(all_ensemble_fold_val_losses, axis=0)  # shape: (EPOCHS,)


plt.figure(figsize=(8, 6))
plt.plot(avg_train_losses, label="Average Training Loss")
plt.plot(avg_valid_losses, label="Average Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Average Loss Across Folds")
plt.legend()
plt.show()



# Compute overall CV score (using your scoring routine)
from metric import score
y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = ensemble_oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for NN = {m}")


# Create submission file with ensemble test predictions
sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = ensemble_pred
sub.to_csv("submission.csv", index=False)
print("Sub shape:", sub.shape)
sub.head()




