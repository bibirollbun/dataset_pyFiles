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


train = pd.read_csv("/kaggle/input/playground-series-s4e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e3/test.csv")


target_list = [
    'Pastry', 
    'Z_Scratch', 
    'K_Scatch', 
    'Stains',
    'Dirtiness', 
    'Bumps', 
    'Other_Faults'
]


train.head(2)


test.head(2)


train.shape , test.shape


train[target_list].mean()


from sklearn.preprocessing import StandardScaler

dataset = pd.concat([train.drop(target_list,axis=1),test],axis=0)
dataset = dataset.drop(['id'], axis=1)

sc = StandardScaler()
dataset_scaled = sc.fit_transform(dataset)


LATENT_DIM = 64


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ----- DataLoader -----
X = torch.tensor(dataset_scaled, dtype=torch.float32)

input_dim = X.shape[1] 
latent_dim = LATENT_DIM
batch_size = 32
epochs = 200

train_loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)


# ----- Model with Dropout -----
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, latent_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*4, latent_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*2, latent_dim)
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*2, latent_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*4, output_dim),
            nn.Sigmoid()  # remove if inputs not normalized
        )

    def forward(self, x):
        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, input_dim, dropout)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
        

# ----- Initialize model -----
model = AutoEncoder(input_dim, latent_dim, dropout=0.2)

# ----- Training setup -----
criterion = nn.MSELoss()
# Add weight decay for L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)


loss_history = []

# ----- Training loop -----
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    for (batch,) in train_loader:
        optimizer.zero_grad()
        x_recon, _ = model(batch)
        loss = criterion(x_recon, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{epochs}] ➤ Loss: {avg_loss:.6f}")

# ----- Plot training loss -----
plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), loss_history, linewidth=2)
plt.title("Training Loss per Epoch (with Dropout + L2 Reg)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()




# Split train and test back
train_scaled = dataset_scaled[:len(train)]
test_scaled = dataset_scaled[len(train):]

X_train = torch.tensor(train_scaled, dtype=torch.float32)
y_train = torch.tensor(train[target_list].values, dtype=torch.float32)  # multi-label

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

X_test = torch.tensor(test_scaled, dtype=torch.float32)


# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt

# def train_evaluate_classifier(
#     encoder,
#     X,
#     y,
#     latent_dim=64,
#     n_labels=7,
#     dropout=0.1,
#     lr=5e-4,
#     weight_decay=5e-6,
#     batch_size=32,
#     epochs=150,
#     val_frac=0.2,
#     plot_loss=True,
#     print_loss=True,
#     device='cpu'
# ):
#     """
#     Train and evaluate a multi-label classifier on top of a frozen encoder.
#     Returns the micro-average ROC-AUC on the validation set.
#     """
#     # Move data to device
#     X = torch.tensor(X, dtype=torch.float32).to(device)
#     y = torch.tensor(y, dtype=torch.float32).to(device)

#     # Freeze encoder
#     for param in encoder.parameters():
#         param.requires_grad = False

#     # Define classifier
#     class MultiLabelClassifier(nn.Module):
#         def __init__(self, encoder, latent_dim, n_labels, dropout=0.1):
#             super().__init__()
#             self.encoder = encoder
#             self.classifier = nn.Sequential(
#                 nn.Linear(latent_dim, 32),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(32, n_labels)
#             )

#         def forward(self, x):
#             z = self.encoder(x)
#             logits = self.classifier(z)
#             return logits

#     clf_model = MultiLabelClassifier(encoder, latent_dim, n_labels, dropout).to(device)

#     # Weighted BCE for imbalanced labels
#     label_means = y.mean(dim=0)
#     pos_weight = 1.0 / (label_means + 1e-6)
#     pos_weight = pos_weight / pos_weight.max()
#     pos_weight = pos_weight.to(device)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     # Optimizer with L2 regularization
#     optimizer = torch.optim.Adam(clf_model.parameters(), lr=lr, weight_decay=weight_decay)

#     # Create train/validation split
#     dataset = TensorDataset(X, y)
#     val_size = int(len(dataset) * val_frac)
#     train_size = len(dataset) - val_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # Training loop
#     loss_history = []
#     for epoch in range(1, epochs + 1):
#         clf_model.train()
#         epoch_loss = 0
#         for batch_x, batch_y in train_loader:
#             optimizer.zero_grad()
#             logits = clf_model(batch_x)
#             loss = criterion(logits, batch_y)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         avg_loss = epoch_loss / len(train_loader)
#         loss_history.append(avg_loss)

#         if (epoch % 10 == 0) and print_loss:
#             print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

#     # Evaluate on validation set
#     clf_model.eval()
#     all_probs, all_true = [], []
#     with torch.no_grad():
#         for val_x, val_y in val_loader:
#             logits = clf_model(val_x)
#             probs = torch.sigmoid(logits)
#             all_probs.append(probs.cpu())
#             all_true.append(val_y.cpu())

#     all_probs = torch.cat(all_probs).numpy()
#     all_true = torch.cat(all_true).numpy()

#     # # Per-label ROC-AUC
#     # for i, label in enumerate(target_list):
#     #     auc = roc_auc_score(y_true[:, i], train_probs[:, i])
#     #     print(f"ROC-AUC for {label}: {auc:.4f}")
    
#     # print("-"*50)
    
#     # Micro-average ROC-AUC
#     micro_auc = roc_auc_score(all_true, all_probs, average='micro')
#     print(f"Validation Micro-average ROC-AUC: {micro_auc:.4f}")

#     return clf_model, micro_auc, loss_history


# !pip install optuna --quiet  # install optuna if not already

# import optuna


# # Define objective function for Optuna
# def objective(trial):
#     # Hyperparameter search space
#     lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
#     weight_decay = trial.suggest_loguniform('weight_decay', 1e-7, 1e-3)
#     dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
#     batch_size = 64
#     epochs = trial.suggest_int('epochs', 50, 200)

#     # Train & evaluate using your function
#     _, micro_auc, _ = train_evaluate_classifier(
#         encoder=model.encoder,       # pretrained encoder
#         X=train_scaled,
#         y=y_train.numpy(),
#         latent_dim=LATENT_DIM,       # ignored in classifier
#         n_labels=7,
#         dropout=dropout,
#         lr=lr,
#         weight_decay=weight_decay,
#         batch_size=batch_size,
#         epochs=epochs,
#         val_frac=0.07,
#         print_loss=False,             # disable plots during optimization
#         device='cpu'                 # change to 'cuda' if GPU available
#     )
#     # Return micro-ROC-AUC to maximize
#     return micro_auc

# # Create study
# study = optuna.create_study(direction='maximize')

# # Run optimization
# study.optimize(objective, n_trials=40)  # you can increase n_trials

# # Print best hyperparameters
# print("Best Hyperparameters:")
# for key, value in study.best_params.items():
#     print(f"{key}: {value}")

# print(f"Best Micro-average ROC-AUC: {study.best_value:.4f}")


# import optuna
# from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

# # ----- Optimization history -----
# plot_optimization_history(study)


# # ----- Convert study results to DataFrame -----
# trials_df = study.trials_dataframe()

# # Sort by value (best first)
# trials_df = trials_df.sort_values(by="value", ascending=False)
# trials_df


# # ----- Parameter importance -----
# fig2 = plot_param_importances(study)
# fig2.show()


# study.best_params


#Choosing the most balanced Hyper-param combination
best_pramas_dict = {
 'lr': 0.000257,
 'weight_decay': 1.604507e-04,
 'dropout': 0.115806,
 'epochs': 68    			
}

# best_pramas_dict = study.best_params


def train_best_classifier(
    encoder,
    X,
    y,
    latent_dim=64,
    n_labels=7,
    dropout=0.1,
    lr=5e-4,
    weight_decay=5e-6,
    batch_size=32,
    epochs=150,
    plot_loss=True,
    print_loss=True,
    device='cpu'
):
   
    # Move data to device
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # Define classifier
    class MultiLabelClassifier(nn.Module):
        def __init__(self, encoder, latent_dim, n_labels, dropout=0.1):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_labels)
            )

        def forward(self, x):
            z = self.encoder(x)
            logits = self.classifier(z)
            return logits

    clf_model = MultiLabelClassifier(encoder, latent_dim, n_labels, dropout).to(device)

    # Weighted BCE for imbalanced labels
    label_means = y.mean(dim=0)
    pos_weight = 1.0 / (label_means + 1e-6)
    pos_weight = pos_weight / pos_weight.max()
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer with L2 regularization
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create train/validation split
    dataset = TensorDataset(X, y)
    train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    loss_history = []
    for epoch in range(1, epochs + 1):
        clf_model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = clf_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch % 10 == 0) and print_loss:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    return clf_model, loss_history


clf_model_best, loss_history = train_best_classifier(
    encoder=model.encoder,       # pretrained encoder
    X=train_scaled,              # training features
    y=y_train.numpy(),           # multi-label targets
    latent_dim=64,
    n_labels=7,
    dropout=best_pramas_dict['dropout'],
    lr=best_pramas_dict['lr'],
    weight_decay=best_pramas_dict['weight_decay'],
    batch_size=32,
    epochs=best_pramas_dict['epochs'],
    print_loss=True,
    device='cpu'                 # or 'cuda' if GPU available
)



# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, best_pramas_dict['epochs'] + 1), loss_history, marker='o', linestyle='-', linewidth=2)
plt.title("Multi-Label Classifier Training Loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


clf_model_best.eval()
with torch.no_grad():
    logits_test = clf_model_best(X_test)
    probs_test = torch.sigmoid(logits_test)  # probabilities for each label
    preds_test = (probs_test > 0.5).int()   # binary predictions


test[target_list] = probs_test


test[['id']+target_list]


test[['id']+target_list].to_csv('submission.csv',index=False)

