import torch
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


df = pd.read_csv(f'/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv(f'/kaggle/input/playground-series-s5e7/test.csv')


df.info()


null_percentage = (df_test.isna().sum()/ len(df_test)) * 100
null_percentage


Null_bi_col = ['Stage_fear', 'Drained_after_socializing']
for i in Null_bi_col:
    df[i] = df[i].map({
        'Yes': 1,
        'No' : 0
    })
    df_test[i] = df_test[i].map({
        'Yes': 1,
        'No' : 0
    })
df['Personality'] = df['Personality'].map({
    'Introvert' : 0,
    'Extrovert' : 1
})



# Use KNN imputer 
X = df.drop(columns=['Personality', 'id'])
X_test = df_test.drop(columns=['id'])
imputer = KNNImputer(n_neighbors=5, weights='uniform')
df_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
df_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


df_set = pd.concat([df_imputed, df['Personality']], axis = 1)


df_set.info()


X = df_set.drop(columns = ['Personality']).to_numpy().astype(np.float32) 
y = df['Personality'].to_numpy().astype(np.int64)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCHSIZE = 32
criterion = nn.CrossEntropyLoss()
CLASSES = 2


class PersonalityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = PersonalityDataset(X_train, y_train)
val_ds   = PersonalityDataset(X_val, y_val)

train_loader = DataLoader(train_ds, shuffle=True, batch_size = BATCHSIZE)
val_loader   = DataLoader(val_ds, batch_size = BATCHSIZE)


def optuna_classifier(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    in_features = 7 

    for i in range(n_layers):
        out_features = trial.suggest_int(f'out_features_l{i}', 4, 256)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        dropout = trial.suggest_float(f'dropout_l{i}', 0.01, 0.5)
        layers.append(nn.Dropout(dropout))
        in_features = out_features  

    # Output layer
    layers.append(nn.Linear(in_features, CLASSES))

    return nn.Sequential(*layers)



DEVICE


def objective(trial):
    model = optuna_classifier(trial).to(DEVICE)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr=trial.suggest_float("lr", 1e-4, 1.0, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == yb).sum().item()

        val_acc = correct / len(val_loader.dataset)
        
    return val_acc



# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)
# print('Best score', study.best_trial.params)
# print('Best score', study.best_value)


class PersonalityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 203),
            nn.ReLU(),
            nn.Dropout(0.10754101553989988),
            nn.Linear(203, 2) 
        )
    def forward(self, x):
        return self.model(x)

model1 = PersonalityClassifier()


model1.to(DEVICE)


torch.manual_seed(42)
optimizer = torch.optim.Adam(model1.parameters(), lr=0.00907584992383887)

for epoch in range(EPOCHS):
    # --- Train ---
    model1.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model1(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Validation ---
    model1.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model1(xb)
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == yb).sum().item()

    val_acc = correct / len(val_loader.dataset)
    print(f"Epoch {epoch:2d} | Val Acc: {val_acc:.6f}")



model1.eval()
with torch.no_grad():
    test_tensor = torch.from_numpy(df_test_imputed.values.astype(np.float32)).to(DEVICE)
    test_preds = model1(test_tensor)
    test_labels = test_preds.argmax(dim=1).cpu().numpy()

result = pd.Series(test_labels).map({
    0: 'Introvert',
    1: 'Extrovert'
})


submission_df = pd.DataFrame({
    'id': df_test['id'],
    'Personality': result
})


submission_df.to_csv(f'submission.csv', index = False)


submission_df.head()

