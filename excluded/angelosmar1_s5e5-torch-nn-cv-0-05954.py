import os
import time
import joblib
import gc
import math
import itertools

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import StepLR

print(f"numpy version {np.__version__}")
print(f"pd version {pd.__version__}")
print(f"sklearn version {sklearn.__version__}")
print(f"torch version {torch.__version__}")


def convert_dtypes(X, *from_to_tuples):
    for from_type, to_type in from_to_tuples:
        cols = X.select_dtypes(from_type).columns.tolist()
        X[cols] = X[cols].astype(to_type)
    return X


def train_single_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = F.mse_loss(output, targets, reduction='sum')
            total_loss += loss.item()
            num_samples += targets.size(0)
    return total_loss / num_samples


def train(model, train_dataloader, num_epochs, optimizer, criterion,
          val_dataloader=None, lr_scheduler=None, early_stop_epochs=None,
          save_path=None, num_print_decimals=5):

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    min_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_single_epoch(model, train_dataloader, optimizer, criterion)
        train_loss = evaluate(model, train_dataloader)
        train_loss_per_epoch.append(np.sqrt(train_loss))
        print_message = f"Epoch: {epoch}, Train Loss: {round(np.sqrt(train_loss), num_print_decimals)}"

        if val_dataloader is not None:
            val_loss = evaluate(model, val_dataloader)
            val_loss_per_epoch.append(np.sqrt(val_loss))
            print_message += f", Val Loss: {round(np.sqrt(val_loss), num_print_decimals)}"
            if val_loss < min_val_loss:
                epochs_since_improvement = 0
                min_val_loss = val_loss
                if save_path is not None:
                    save_model(save_path, model, optimizer, train_loss_per_epoch,
                                val_loss_per_epoch, epoch)
                    print_message += f"\nSaved checkpoint '{save_path}'"
            else:
                epochs_since_improvement += 1

        print(print_message)

        if early_stop_epochs is not None and epochs_since_improvement == early_stop_epochs:
            print(f"Early Stopping.")
            break

        if lr_scheduler is not None:
            lr_scheduler.step()
            print(f"Learning Rate: {lr_scheduler.get_last_lr()}")

    return train_loss_per_epoch, val_loss_per_epoch


def plot_loss_curves(train_loss_per_epoch, val_loss_per_epoch,
                     best_epoch=None, title=None, figsize=(6, 4)):
    plt.figure(figsize=figsize)
    x_axis_values = list(range(1, len(train_loss_per_epoch) + 1))
    plt.plot(x_axis_values, train_loss_per_epoch, color="blue", label="train loss")
    plt.plot(x_axis_values, val_loss_per_epoch, color="red", label="validation loss")
    if best_epoch is not None:
        plt.axvline(x=best_epoch, linestyle="--", color="green", label="best model")
    plt.legend()
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.tight_layout()
    plt.show()


def save_model(save_path, model, optimizer, train_losses, val_losses, epoch):
    if not save_path.endswith(".pth"):
        save_path += ".pth"
    model_info = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
             }
    torch.save(model_info, save_path)


def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            output = model(inputs)
            predictions.append(output)
    return torch.cat(predictions).squeeze(-1).cpu().numpy()


class CustomDataset(Dataset):

    def __init__(self, X, y, X_dtype=torch.float32, y_dtype=torch.float32):
        self.features_num = torch.tensor(X.to_numpy(), dtype=X_dtype)
        self.labels = torch.tensor(y.to_numpy(), dtype=y_dtype).unsqueeze(dim=-1)

    def __len__(self):
        return self.features_num.shape[0]

    def __getitem__(self, idx):
        return self.features_num[idx], self.labels[idx]


class ResidualBlock(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.shortcut = (
            nn.Sequential(
                nn.Linear(in_features, out_features),
            ) if in_features != out_features else nn.Identity()
        )

    def forward(self, x):
        out = self.fc1(x)
        out = F.silu(out)
        out = self.fc2(out)
        shortcut = self.shortcut(x)
        out = F.silu(out + shortcut)
        return out


class CustomModel(nn.Module):
    
    def __init__(self, units, input_dim):
        super().__init__()
        layers = []
        in_features = input_dim

        for i in range(0, len(units), 2):
            u1 = units[i]
            u2 = units[i+1] if i+1 < len(units) else units[i]
            layers.append(ResidualBlock(in_features, u1, u2))
            in_features = u2

        self.res_blocks = nn.Sequential(*layers)
        self.out = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.res_blocks(x)
        return self.out(x)


data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col="id")
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col="id")


y = np.log1p(data["Calories"])
X = data.drop(columns="Calories")
X_test = test.copy()


X["Sex"] = X["Sex"].map({"female": 0, "male": 1})
X_test["Sex"] = X_test["Sex"].map({"female": 0, "male": 1})


combined = pd.concat([X, X_test])

combined_drop_sex = combined.drop(columns="Sex")

    
interactions = {}
for col1, col2 in itertools.combinations(combined_drop_sex.columns, 2):
    interactions[col1 + "_mul_" + col2] = combined_drop_sex[col1] * combined_drop_sex[col2]
    interactions[col1 + "_div_" + col2] = combined_drop_sex[col1] / combined_drop_sex[col2]
    interactions[col2 + "_div_" + col1] = combined_drop_sex[col2] / combined_drop_sex[col1]
    interactions[col1 + "_div_sq" + col2] = combined_drop_sex[col1] / combined_drop_sex[col2] ** 2
    interactions[col2 + "_div_sq" + col1] = combined_drop_sex[col2] / combined_drop_sex[col1] ** 2
interactions = pd.DataFrame(interactions)


combined = pd.concat([combined, 
                     (combined_drop_sex ** 2).add_prefix("sq_"),
                      np.log(combined_drop_sex).add_prefix("log_"),
                      (1 / combined_drop_sex).add_prefix("inv_"),
                      np.sqrt(combined_drop_sex).add_prefix("sqrt_"),
                      (1 / combined_drop_sex ** 2).add_prefix("inv_sq"),
                      interactions], axis=1)


to_scale = [col for col in combined.columns if col != "Sex"]


mean = combined[to_scale].mean()
std = combined[to_scale].std()
combined[to_scale] = (combined[to_scale] - mean) / std


X = combined.loc[X.index]
X_test = combined.loc[X_test.index]

del combined


train_dataset = CustomDataset(X, y)
test_dataset = CustomDataset(X_test, y)

criterion =  nn.MSELoss()
num_epochs = 50
batch_size = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=5)

time0 = time.perf_counter()

oof_preds = np.zeros(shape=(X.shape[0],))
test_preds = np.zeros(shape=(X_test.shape[0], num_folds))

scores = []
model_iters = []
training_history = []


test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


for fold_number, (train_indices, val_indices) in enumerate(kfold.split(data)):

    print(f"fold {fold_number + 1}")
    
    model = CustomModel(units=(86, 86, 64, 32), input_dim=X.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    train_dataloader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=False)

    train_loss_per_epoch, val_loss_per_epoch = train(model, train_dataloader, num_epochs,
                                                     optimizer, criterion, val_dataloader, 
                                                     lr_scheduler, save_path=None, early_stop_epochs=None)

    training_history.append((train_loss_per_epoch, val_loss_per_epoch))

    val_preds = predict(model, val_dataloader)
    oof_preds[val_indices] = val_preds

    cur_test_preds = predict(model, test_dataloader)
    test_preds[:, fold_number] = cur_test_preds

    score = np.sqrt(mean_squared_error(y.iloc[val_indices], val_preds))
    print(score)
    model_iters.append(np.argmin(val_loss_per_epoch) + 1)
    scores.append(score)

time1 = time.perf_counter()

print("----------------------------------")
print(f"time {round(time1 - time0, 2)}s")
print(f"num iterations {model_iters} mean {int(np.mean(model_iters))}")
print("scores ", scores)
print("cv mean", np.mean(scores))
print("cv std", np.std(scores))
print("oof cv", np.sqrt(mean_squared_error(y, oof_preds)))


for fold_number, (train_losses, val_losses) in enumerate(training_history):
    plot_loss_curves(train_losses, val_losses, title=f"fold {fold_number + 1}", figsize=(6, 4),)


oof_preds = pd.DataFrame(np.expm1(oof_preds).clip(1, 314), index=X.index)
test_preds = pd.DataFrame(np.expm1(test_preds.mean(axis=1)).clip(1, 314), index=X_test.index)


plt.hist(oof_preds, bins=100)
plt.title("oof preds")
plt.show()


plt.hist(test_preds, bins=100)
plt.title("test preds")
plt.show()


oof_preds.to_csv("oof.csv")
test_preds.to_csv("submission.csv")

