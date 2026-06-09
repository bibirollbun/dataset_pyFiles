import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')


# Read the data 
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train.head()


y= train['y']
train= train.drop('y',axis=1)



# We want to check out any problems with our data - everything looks good on that front 
print(f"train summary: \n {train.isna().sum()}")
print(f"test summary: \n {test.isna().sum()}")



categorical_cols = test.select_dtypes(include=['object']).columns 
numerical_cols = test.select_dtypes(include=['int64','float64']).columns 

print(f"Our categorical columns : {categorical_cols.values}")
print(f"Our numerical columns : {numerical_cols.values}")



# Dealing with the day problem
numerical_cols = numerical_cols.drop('day')
categorical_cols = categorical_cols.append(pd.Index(['day']))


# Dealing with the pdays problem, create a marker for being contacted + replacing
for df in [train,test]:
    df['was contacted'] = (df['pdays']!=-1).astype(int)
    df['pdays'] = df['pdays'].replace(-1,100000)


print(f"our new categorical cols: {categorical_cols} " )
train.head()


encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)


train_encoded = encoder.fit_transform(train[categorical_cols])
test_encoded = encoder.transform(test[categorical_cols])


train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=train.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=test.index)


train = pd.concat([train.drop(columns=categorical_cols), train_encoded_df], axis=1)
test = pd.concat([test.drop(columns=categorical_cols), test_encoded_df], axis=1)



scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class BinaryClassifier(nn.Module):
    def __init__(self,input_dim):
        super(BinaryClassifier,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128), # Applies a standard learnable linear weighted sum + bias to the input, alters dimension 
            nn.BatchNorm1d(128), # Performing a Batch normalisation to stabilise the training
            nn.LeakyReLU(),# LeakyReLU is like ReLU but allows for a small slope at negative values, this avoids the "dying ReLU problem"
            nn.Dropout(0.1),# Performing a dropout to prevent overfitting
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(), # Ensures non-linearity 
            nn.Dropout(0.1),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x): 
        return self.net(x)
    


class EarlyStopping:
    def __init__(self,patience=5,mode='max'):
        self.patience = patience # How many epochs without improvement before stopping
        self.mode = mode # "max" if higher is better (e.g. AUC), "min" if lower is better (e.g. loss)
        self.best_score = None # Stores the best metric value we've seen
        self.counter = 0 # Consecutive epochs without seeing improvment
        self.early_stop = False  # Flags to True when we stop training

    def __call__(self,score,model):
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(model)
            # We didn't improve
        elif (self.mode == "max" and score <= self.best_score) or (self.mode=="min" and score >=self.best_score):
            self.counter += 1 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # We did improve
            self.best_score = score 
            self.save_checkpoint(model)
            self.counter = 0 
    # Saving our best model
    def save_checkpoint(self,model):
        self.best_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Loading the best model
    def load_checkpoint(self,model):
        model.load_state_dict(self.best_model_wts)
            
            


X = train.values 
X_test = test.values


# This is just for faster testing
X_small = X[:5000]
y_small = y[:5000]


X_data = X
y_data = y

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "gpu_hist", # Ensure we are working on the GPU
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0)
    }

    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in skf.split(X_data, y_data):
        X_train, X_valid = X_data[train_idx], X_data[valid_idx]
        y_train, y_valid = y_data[train_idx], y_data[valid_idx]

        model = xgb.XGBClassifier(**params, use_label_encoder=False)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, y_pred))

    return np.mean(auc_scores)

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30) 

print("Best trial:")
print(study.best_trial.params)

best_params = study.best_trial.params




skf = StratifiedKFold(n_splits=5,shuffle=True)

# OOF predictions
oof_nn = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_ensemble = np.zeros(len(X))

# Test predictions 
nn_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))
ensemble_test_preds = np.zeros(len(X_test))


# For tracking our neural network metrics
history = {
    "fold":[],
    "epoch":[],
    "train_loss":[],
    "val_loss": [],
    "val_auc": []
}
# For tracking our XGBoost and Ensemble metrics 
other_metrics = {
    "fold": [],
    "nn_auc": [],
    "xgb_auc": [],
    "ensemble_auc":[]    
}


for fold, (train_idx,val_idx) in enumerate(skf.split(X,y)):
    print(f"\n Fold {fold+1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train,y_val = y[train_idx], y[val_idx]

   
    
    # Convert everything to tensors to use the NN model, making sure it's on the GPU
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_tensor   = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor   = torch.tensor(y_val.values, dtype=torch.float32, device=device).unsqueeze(1)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(TensorDataset(X_train_tensor,y_train_tensor),
                             batch_size=64,shuffle = True)
   

    # Need a model per fold 
    model = BinaryClassifier(X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay = 0.00001)
    early_stopping = EarlyStopping(patience=5,mode="max")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience =3 , verbose = True
    )
    
    # Actually training 
    for epoch in range(50):
        model.train()
        running_loss = 0
        for xb,yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds,yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation of each epoch
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits,y_val_tensor).item()
            val_preds = torch.sigmoid(val_logits).squeeze().cpu()
        val_auc = roc_auc_score(y_val,val_preds)

        # We must step our scheduler 
        scheduler.step(val_auc)
        early_stopping(val_auc,model)
        
        # Save our metrics for later plotting
        history["fold"].append(fold+1)
        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")

        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    early_stopping.load_checkpoint(model)
    model.eval()
    with torch.no_grad():
        nn_val_preds = torch.sigmoid(model(X_val_tensor)).squeeze().cpu().numpy()
        nn_test_fold_preds = torch.sigmoid(model(X_test_tensor)).squeeze().cpu().numpy()
    

    # XGBoost setup + predictions
    xgb_model = xgb.XGBClassifier(
        **best_params, eval_metric = "auc", use_label_encoder = False
    )

    xgb_model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False,early_stopping_rounds = 50)

    xgb_calibrated = CalibratedClassifierCV(xgb_model,method="sigmoid",cv="prefit")
    xgb_calibrated.fit(X_val,y_val)
    xgb_val_preds = xgb_calibrated.predict_proba(X_val)[:,1]
    xgb_test_fold_preds = xgb_calibrated.predict_proba(X_test)[:,1]

    # Finding our best weights
    best_auc = 0 
    best_w = 0.5 
    for w in np.linspace(0,1,101):
        blended_val = w* nn_val_preds +(1-w)*xgb_val_preds
        auc = roc_auc_score(y_val,blended_val)
        if auc> best_auc:
            best_auc=auc
            best_w=w
    print(f"Fold {fold+1} best weight: NN={best_w:.3f}, XGB={1-best_w:.3f}, AUC={best_auc:.4f}")
    #Ensemble predictions 
    ensemble_preds_val = best_w*nn_val_preds + (1-best_w)*xgb_val_preds
    
    # Get our predictions
    oof_nn[val_idx] = nn_val_preds
    oof_xgb[val_idx] = xgb_val_preds
    oof_ensemble[val_idx] = ensemble_preds_val
    
    nn_test_preds += nn_test_fold_preds / skf.n_splits
    xgb_test_preds += xgb_test_fold_preds /skf.n_splits
    ensemble_test_preds += (best_w*nn_test_fold_preds + (1-best_w)*xgb_test_fold_preds)/skf.n_splits
    
    
    # Update our metrics for later plotting
    nn_auc = roc_auc_score(y_val, nn_val_preds)
    xgb_auc = roc_auc_score(y_val,xgb_val_preds)
    ensemble_auc = roc_auc_score(y_val, ensemble_preds_val)
    
    other_metrics["fold"].append(fold+1)
    other_metrics["nn_auc"].append(nn_auc)
    other_metrics["xgb_auc"].append(xgb_auc)
    other_metrics["ensemble_auc"].append(ensemble_auc)
    print(f"Fold {fold+1} Results:")
    print(f"  NN Best Val AUC: {nn_auc:.4f}")
    print(f"  XGBoost Val AUC: {xgb_auc:.4f}")
    print("-"*40)



nn_auc_overall = roc_auc_score(y, oof_nn)
xgb_auc_overall = roc_auc_score(y, oof_xgb)
ensemble_auc_overall = roc_auc_score(y, oof_ensemble)

print("\n--- Final OOF AUC Scores ---")
print(f"Neural Network OOF AUC: {nn_auc_overall:.4f}")
print(f"XGBoost OOF AUC: {xgb_auc_overall:.4f}")
print(f"Ensemble OOF AUC: {ensemble_auc_overall:.4f}")



history_df = pd.DataFrame(history)


grouped = history_df.groupby("epoch").agg({
    "train_loss": ["mean", "std"],
    "val_loss": ["mean", "std"],
    "val_auc": ["mean", "std"]
})

epochs = grouped.index
train_loss_mean = grouped["train_loss"]["mean"]
train_loss_std = grouped["train_loss"]["std"]
val_loss_mean = grouped["val_loss"]["mean"]
val_loss_std = grouped["val_loss"]["std"]
val_auc_mean = grouped["val_auc"]["mean"]
val_auc_std = grouped["val_auc"]["std"]

plt.figure(figsize=(15, 5))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_mean, label="Train Loss", color="blue")
plt.fill_between(epochs, train_loss_mean-train_loss_std, train_loss_mean+train_loss_std, alpha=0.2, color="blue")
plt.plot(epochs, val_loss_mean, label="Val Loss", color="orange")
plt.fill_between(epochs, val_loss_mean-val_loss_std, val_loss_mean+val_loss_std, alpha=0.2, color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("NN Loss per Epoch")
plt.legend()

# ROC AUC curve
plt.subplot(1, 2, 2)
plt.plot(epochs, val_auc_mean, label="Val ROC AUC", color="green")
plt.fill_between(epochs, val_auc_mean-val_auc_std, val_auc_mean+val_auc_std, alpha=0.2, color="green")
plt.xlabel("Epoch")
plt.ylabel("ROC AUC")
plt.title("NN ROC AUC per Epoch")
plt.legend()

plt.tight_layout()
plt.show()



folds = np.arange(1, len(other_metrics["fold"]) + 1)
nn_auc = other_metrics["nn_auc"]
xgb_auc = other_metrics["xgb_auc"]
ensemble_auc = other_metrics["ensemble_auc"]


fig, axes = plt.subplots(2,1,figsize=(15,20))

axes[0].plot(folds, nn_auc, marker="o", label="NN AUC", color="steelblue")
axes[0].plot(folds, xgb_auc, marker="o", label="XGB AUC", color="orange")
axes[0].plot(folds, ensemble_auc, marker="o", label="Ensemble AUC", color="green")
axes[0].set_xlabel("Fold",fontsize=20)
axes[0].set_ylabel("Validation AUC",fontsize=20)
axes[0].set_title("NN vs XGB vs Ensemble AUC per Fold",fontsize=25)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: Overall OOF AUC
axes[1].bar(["NN", "XGB", "Ensemble"], [nn_auc_overall, xgb_auc_overall, ensemble_auc_overall], 
            color=["steelblue", "orange", "green"])
axes[1].set_ylabel("ROC AUC",fontsize=20)
axes[1].set_title("Overall OOF ROC AUC",fontsize=25)
axes[1].set_ylim(0.0, 1.0)
for i, v in enumerate([nn_auc_overall, xgb_auc_overall, ensemble_auc_overall]):
    axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()


sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission = pd.DataFrame({
    "id": sub['id'],
    "y": ensemble_test_preds
})

submission.to_csv("submission.csv", index=False)





