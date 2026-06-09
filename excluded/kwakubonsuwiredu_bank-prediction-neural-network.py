# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
from torchmetrics.classification import BinaryAccuracy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
from torch import optim
from torch.nn import BCELoss
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import optuna


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')

target = 'y'
features = [ col for col in data.columns if col not in ['id', 'y']]

cat_col = [col for col in features if data[col].dtype == 'object']


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
data[cat_col] = encoder.fit_transform(data[cat_col])


data.nunique()


data.head()


X = data[features]
y = data[[target]]


X_tensor = torch.tensor(np.array(X.astype(float)))
y_tensor = torch.tensor(np.array(y.astype(float)))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size =0.2 )

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size =0.1 )



dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
dataloader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=2, pin_memory =True)

dataset1 = TensorDataset(X_test, y_test)
validationloader = DataLoader(dataset1, batch_size=200, shuffle = True,  num_workers=2, pin_memory =True)


import math, torch, torch.nn as nn

def init_uniform_all(m, a=-0.1, b=0.1):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=a, b=b)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=a, b=b)



# Hyper Parameter Tuning with optuna
def objective(trial):
    # ---- hyperparameters to tune ----
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.6)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.6)
    dropout3 = trial.suggest_float("dropout3", 0.1, 0.6)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # ---- build model (inline version of your Sequential) ----
    model = nn.Sequential(
        nn.Linear(len(features), 256),
        nn.Dropout(dropout1),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.Dropout(dropout2),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.Dropout(dropout3),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(device)

    model.apply(init_uniform_all)

    # ---- optimizer with momentum only if SGD ----
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # SGD
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    # ---- loss ----
    criterion = nn.BCEWithLogitsLoss()  # binary classification

    # ---- training loop -------
    for epoch in range(20):  
        model.train()
        for xb, yb in dataloader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # ---- validation ----
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in validationloader:
            xb, yb = xb.to(device).float(), yb.to(device).float()
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    val_loss = total_loss / n

    return val_loss



'''
# run the study
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=100, timeout=None)

'''


# best_params = study.best_trial.params

best_params = {'dropout1': 0.11191084822432693, 'dropout2': 0.10045500269761898, 'dropout3': 0.5981602821713188, 'lr': 0.00023808995670469342, 'weight_decay': 8.656316838062717e-06, 'optimizer': 'Adam'}
print("Best params:", best_params)


model = nn.Sequential(
        nn.Linear(len(features), 256),
        nn.Dropout(best_params['dropout1']),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.Dropout(best_params['dropout2']),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.Dropout(best_params['dropout3']),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(device)

model.apply(init_uniform_all)


if best_params['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
else:
    optimizer = optim.SGD(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        momentum=best_params['momentum'] 
    )

optimizer


# Train the model
import copy
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

criterion = nn.BCEWithLogitsLoss()

# --- Early stopping helper ---
class EarlyStopper:
    def __init__(self, patience=10, mode="min", min_delta=0.0, restore_best=True):
        self.patience = patience
        self.mode = mode  # "min" for loss, "max" for a score like AUC
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def step(self, value, model):
        if self.best_score is None:
            self.best_score = value
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
            return False

        improved = (value < self.best_score - self.min_delta) if self.mode == "min" \
                   else (value > self.best_score + self.min_delta)

        if improved:
            self.best_score = value
            self.counter = 0
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# --- Training with early stopping (monitoring validation loss) ---
val_loss = []
num_epochs = 200
earlystop = EarlyStopper(patience=5, mode="min", min_delta=0.0, restore_best=True)

for i in range(num_epochs):
    print(f'EPOCH {i}:', end=' ')

    # -------- train --------
    model.train()
    running_train_loss, n_train = 0.0, 0
    for feature, labels in dataloader:
        feature = feature.to(device).float()
        labels  = labels.to(device).float().view(-1, 1)  # (B,1) to match logits

        optimizer.zero_grad(set_to_none=True)
        outputs = model(feature)                          # logits (B,1)
        loss = criterion(outputs, labels)                 # BCEWithLogitsLoss
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * feature.size(0)
        n_train += feature.size(0)

    # -------- validate --------
    model.eval()
    eval_loader = validationloader if 'validationloader' in globals() else dataloader

    running_val_loss, n_val = 0.0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for feature, labels in eval_loader:
            feature = feature.to(device).float()
            labels  = labels.to(device).float().view(-1, 1)

            outputs = model(feature)                      # logits
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * feature.size(0)
            n_val += feature.size(0)

            probs = torch.sigmoid(outputs).detach().cpu().view(-1)
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().view(-1))

    epoch_val_loss = running_val_loss / max(n_val, 1)
    val_loss.append(epoch_val_loss)

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    preds = (all_probs > 0.5).astype("int32")
    acc = (preds == all_labels).mean()
    try:
        roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc = float("nan")

    print(f"loss: {epoch_val_loss:.4f} accuracy: {acc:.4f} roc_auc_score: {roc:.4f}")

    # ---- early stopping on validation loss ----
    if earlystop.step(epoch_val_loss, model):
        print(f"Early stopping triggered (no improvement for {earlystop.patience} epochs).")
        break

# restore best weights (if enabled)
if earlystop.restore_best and earlystop.best_state is not None:
    model.load_state_dict(earlystop.best_state)

print(np.mean(val_loss))



test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


test[cat_col] = encoder.fit_transform(test[cat_col])


predictions = torch.sigmoid(model(torch.tensor(np.array(test[features])).to(device).float()))


a = torch.sigmoid(predictions)


submission = pd.DataFrame()
submission['id'] = test['id']
submission['y'] = a.cpu().detach().numpy()

submission.to_csv('submission23.csv', index=False)

