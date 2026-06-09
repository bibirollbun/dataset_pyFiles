import pandas as pd
import matplotlib.pyplot as plt


## Local 
# train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
# og = pd.read_csv('../data/abalone.csv')

## Kaggle
train = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')
og = pd.read_csv('/kaggle/input/abalone-dataset/abalone.csv')


train.columns


og.columns


plt.figure(figsize=(5,5))
train['Whole weight.2'].plot(kind='kde')
og['Viscera weight'].plot(kind='kde')
plt.legend()


plt.figure(figsize=(5,5))
train['Whole weight.1'].plot(kind='kde')
og['Shucked weight'].plot(kind='kde')
plt.legend()


rename_columns = {
    'Whole weight.2': 'Viscera weight',
    'Whole weight.1': 'Shucked weight'
}


train = train.rename(columns=rename_columns)
test = test.rename(columns=rename_columns)


def ohe(df,column):
    return pd.concat([
        df.drop(column,axis=1),
        pd.get_dummies(df[column],drop_first=True, prefix=column).astype(int)
    ],axis=1)


train = ohe(train, 'Sex')
test = ohe(test, 'Sex')
og = ohe(og, 'Sex')


display(train.head(2))
display(test.head(2))
display(og.head(2))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def abalation_study(train, test, actual_target='Rings') : 
    
    df1 = train.copy()
    df2 = test.copy()
    
    df1['abalation_target'] = 0
    df2['abalation_target'] = 1
    df = pd.concat([
        df1.drop(actual_target,axis=1,errors='ignore'),
        df2.drop(actual_target,axis=1,errors='ignore')
    ], axis=0) 
    
    min_samples_leaf = int(0.03 * df.shape[0])
    class_weight = 1/df.abalation_target.mean()
    
    previous_model = None
    for n_est in range(10,250,25): 
        
        current_model = RandomForestClassifier(
            n_estimators = n_est,
            min_samples_leaf = min_samples_leaf,
            oob_score = True, 
            class_weight={0:1,1:class_weight}
        )
        
        current_model = current_model.fit(df.drop('abalation_target',axis=1),df['abalation_target'])
        
        if previous_model :
            if current_model.oob_score_ > previous_model.oob_score_ : 
                previous_model = current_model 
            else : 
                break 
        else : 
            previous_model = current_model 
            
        print("n_estimators = ",n_est,"\toob_score = ", current_model.oob_score_)
        
    final_model = previous_model 
    prediction = final_model.predict_proba(df.drop('abalation_target', axis=1))
    print("ROC-AUC:", roc_auc_score(df['abalation_target'], prediction[:, 1]), end="\t")
    


import warnings
warnings.filterwarnings("ignore")


abalation_study(
    train.drop('id',axis=1), 
    test.drop('id',axis=1)
)


abalation_study(
    train.drop('id',axis=1), 
    og
)


abalation_study(
    test.drop('id',axis=1), 
    og
)


FOLDS = 5
PATIENCE = 3
N_ITER = 100


import torch
import torch.nn as nn 
import torch.functional as f 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import numpy as np


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


class RegressionModel(nn.Module):
    
    def __init__(self, input_dim, latent_dim, dropout=0.2):
        
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*2, 1),
        )
        
    def forward(self, x):
        
        return self.model(x)


from sklearn.preprocessing import StandardScaler
import torch

def get_scaled_train_val_test(train, val, test=None):
    
    sc = StandardScaler()

    # Scale features
    train_scaled = sc.fit_transform(train.drop(['id','Rings'], axis=1))
    val_scaled   = sc.transform(val.drop(['id','Rings'], axis=1))
    
    # Convert to float32 and move to device
    X_train = torch.tensor(train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(train['Rings'].values, dtype=torch.float32).to(device)
    
    X_val = torch.tensor(val_scaled, dtype=torch.float32).to(device)
    y_val = torch.tensor(val['Rings'].values, dtype=torch.float32).to(device)

    if test is not None:
        test_scaled  = sc.transform(test.drop(['id'], axis=1))
        X_test = torch.tensor(test_scaled, dtype=torch.float32).to(device)
        return (X_train, y_train), (X_val, y_val), X_test
    else:
        return (X_train, y_train), (X_val, y_val)



from torch.utils.data import DataLoader, TensorDataset

def train_model_and_get_val_pred(X_train, y_train, X_val, y_val,
                                 N_iterations=100,
                                 latent_dim=32,
                                 batch_size=32, 
                                 lr=1e-4, 
                                 weight_decay=1e-5,
                                 patience=PATIENCE):  
    """
    Train RegressionModel with early stopping.
    """
    input_dim = X_train.size(1)
    model = RegressionModel(input_dim=input_dim, latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    loss_fn_val = torch.nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_DL = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    wait = 0  # counter for early stopping

    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    wait = 0  # counter for early stopping
    patience = PATIENCE 
    N_iterations = N_ITER
    for epoch in range(N_iterations):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_DL:
            y_pred = model(X_batch)
            batch_loss = loss_fn(y_pred, y_batch.unsqueeze(1))  # ensure same shape
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * len(y_batch)  # batch_sq_error = batch_mean_sq_error * batch_n
            total_samples += len(y_batch)

        avg_loss = epoch_loss / total_samples # total_sq_error / total_n = toal_mean_sq_error
        train_loss_list.append(avg_loss)
            
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn_val(y_val_pred, y_val.unsqueeze(1)).item()  # ensure same shape
            val_loss_list.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_state = model.state_dict()  # save best model
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping triggered at iteration {epoch}")
            break

        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss = {avg_loss:.6f} | val_loss = {val_loss:.6f} ")
        
    # Load the best model before returning predictions
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)

    return y_val_pred



def objective(trial):
    """
    Optuna objective function for tuning RegressionModel hyperparameters.
    """

    # -------------------------------
    # 1️⃣ Hyperparameters to tune
    # -------------------------------
    N_iterations = 100
    latent_dim   = trial.suggest_categorical("latent_dim", [128,256,512])
    batch_size   = trial.suggest_categorical("batch_size", [128, 256])
    lr           = trial.suggest_float("lr", 5e-5, 7e-4, log=True)  
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    
    print(f"N_iterations: {N_iterations}, latent_dim: {latent_dim}, batch_size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}")

    # -------------------------------
    # 2️⃣ Prepare data
    # -------------------------------
    X = train.drop("Rings", axis=1)
    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    pred_oof = np.zeros((train.shape[0], 1), dtype=float)

    # -------------------------------
    # 3️⃣ Cross-validation loop
    # -------------------------------
    for train_index, val_index in skf.split(X):
        
        train_temp = train.iloc[train_index]
        val_temp = train.iloc[val_index]
        
        (X_train,y_train), (X_val,y_val) = get_scaled_train_val_test(train_temp, val_temp)

        pred = train_model_and_get_val_pred(
            X_train,y_train, X_val,y_val,
            N_iterations = N_iterations,
            latent_dim = latent_dim,
            batch_size = batch_size,
            lr = lr,
            weight_decay = weight_decay
        )
        
        pred_oof[val_index] = pred.detach().cpu().numpy().reshape(-1, 1)


    # -------------------------------
    # 4️⃣ Compute RMSE
    # -------------------------------
    y_true = train['Rings'].values.reshape(-1, 1)  # target as NumPy
    rmse = np.sqrt(mean_squared_error(y_true, pred_oof))

    print("CV RMSE:", rmse)

    return rmse  # return as float for Optuna



# import optuna
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=20, show_progress_bar=True)


# # Convert all trials to a DataFrame
# df_results = study.trials_dataframe()

# # Sort by best (lowest) value
# df_results = df_results.sort_values(by="value", ascending=True).reset_index(drop=True)


# df_results


# study.best_trial.params


# best_param_dict = study.best_trial.params

best_param_dict = {
    'latent_dim': 256,
    'batch_size': 128,
    'lr': 0.0003882362869046006,
    'weight_decay': 1.5884884614136126e-05
}


from torch.utils.data import DataLoader, TensorDataset

def train_model_and_get_val_test_pred(X_train, y_train, X_val, y_val, X_test,
                                N_iterations=100,
                                latent_dim=64,
                                batch_size=128, 
                                lr=0.00075, 
                                weight_decay=1.5e-5,
                                patience=3):  
    """
    Train RegressionModel with early stopping.
    """
    input_dim = X_train.size(1)
    model = RegressionModel(input_dim=input_dim, latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_DL = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    wait = 0  # counter for early stopping
    N_iterations = N_ITER
    for n in range(N_iterations):
        model.train()
        for X_batch, y_batch in train_DL:
            y_pred = model(X_batch)
            batch_loss = loss_fn(y_pred, y_batch.unsqueeze(1))  # ensure same shape
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            

        # Compute validation loss at the end of each epoch
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred, y_val.unsqueeze(1)).item()

        if n%5==0:
            print(f"\tIteration {n}: val_loss = {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_state = model.state_dict()  # save best model
        else:
            wait += 1

        if wait >= patience:
            print(f"\tEarly stopping triggered at iteration {n}")
            break

    # Load the best model before returning predictions
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        y_test_pred = model(X_test)

    return y_val_pred, y_test_pred



# -------------------------------
# 1️⃣ Hyperparameters 
# -------------------------------
N_iterations = 100
latent_dim   = best_param_dict['latent_dim']
batch_size   = best_param_dict['batch_size']
lr           = best_param_dict['lr']
weight_decay = best_param_dict['weight_decay']

# -------------------------------
# 2️⃣ Prepare data
# -------------------------------
X = train.drop("Rings", axis=1)

skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

pred_oof = np.zeros((train.shape[0], 1), dtype=float)
pred_test = np.zeros((test.shape[0], 1), dtype=float)

# -------------------------------
# 3️⃣ Cross-validation loop
# -------------------------------
for fold, (train_index, val_index) in enumerate(skf.split(X)):
    
    print(f"FOLD : {fold+1}")
    
    train_temp = train.iloc[train_index]
    val_temp = train.iloc[val_index]
    
    (X_train,y_train), (X_val,y_val), X_test = get_scaled_train_val_test(train_temp, val_temp, test)

    val_pred, test_pred  = train_model_and_get_val_test_pred(
        X_train,y_train, X_val,y_val, X_test,
        N_iterations = N_iterations,
        latent_dim = latent_dim,
        batch_size = batch_size,
        lr = lr,
        weight_decay = weight_decay,
        patience = 5
    )
    
    pred_oof[val_index] = val_pred.detach().cpu().numpy().reshape(-1, 1)
    pred_test = pred_test + test_pred.detach().cpu().numpy().reshape(-1, 1)/FOLDS

# -------------------------------
# 4️⃣ Compute RMSLE
# -------------------------------
from sklearn.metrics import mean_squared_log_error

y_true = train['Rings'].values.reshape(-1, 1)
y_pred = np.clip(pred_oof, a_min=0, a_max=None)  # avoid negative predictions

rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

print("-"*50)
print("CV RMSLE:", rmsle)


# test.drop('Ring',axis=1,inplace=True)
# train.drop('Rings_pred',axis=1,inplace=True)


train['Rings_pred'] = pred_oof


train['Rings'].describe()


train['Rings_pred'].describe()


test['Ring'] = pred_test


test['Ring'].describe()


test[['id','Ring']].to_csv("v2_with_OOF_strategy.csv", index=False)

