import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
pd.set_option('mode.use_inf_as_na', False)

import matplotlib.pyplot as plt
import seaborn  as sns

import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

import random


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def preprocess(data):
    _col = data.columns

    for c in _col:
        if data[c].dtype == 'int64':
            data[c] = data[c].astype('int32')

        elif data[c].dtype == 'float64':
            data[c] = data[c].astype('float32')
    
    data['col_1'] = data['maxtemp'] - data['mintemp']
    data['col_2'] = data['temparature'] - data['mintemp']
    data['col_3'] = data['maxtemp'] - data['temparature']

    data['col_4'] = data['humidity'] - data['dewpoint']

    data['col_5'] = data['humidity'] / 100

    data['col_rad'] = np.deg2rad(data['winddirection'])
    data['col_6'] = np.sin(data['col_rad'])
    data['col_7'] = np.cos(data['col_rad'])
    data.drop(columns=['col_rad'], inplace=True)

    data['WIND_FACT'] = (
    35.74 + 
    (0.6215 * data['temparature']) - 
    (35.75 * (data['winddirection']**0.16)) + 
    (0.4275 * data['temparature'] * (data['winddirection']**0.16))
    )

    return data


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)  
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

mean_wind_tr = (df_train['winddirection'].mean()).round()
df_test['winddirection'].fillna(mean_wind_tr, inplace=True)


df_train = df_train.drop(columns=['id', 'day'])
df_test_id = df_test['id']
df_test = df_test.drop(columns=['id', 'day'])


df_train_y = df_train['rainfall']
df_train = df_train.drop(columns=['rainfall'])


df_train = preprocess(df_train)
df_test = preprocess(df_test)


pd.options.display.float_format = '{:.2f}'.format


scaler = StandardScaler()
df_train = scaler.fit_transform(df_train)
df_test = scaler.transform(df_test)

df_train_y = pd.Series(df_train_y)


X_train, X_val, y_train, y_val = train_test_split(df_train, df_train_y, test_size=0.2, random_state=SEED)


def objective(trial, X_train, X_val, y_train, y_val):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden_size = trial.suggest_int('hidden_size', 20, 100)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_epochs = trial.suggest_int('num_epochs', 10, 200)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, dropout_rate=0.5):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)  
            self.fc2 = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.dropout(out)  
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out
    
    input_size = X_train.shape[1]
    model = MLP(input_size, hidden_size, dropout_rate)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1)) 
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()

            epoch_loss += loss.item()  

  
            predicted = (outputs > 0.5).float()  
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

        epoch_loss /= len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}'
              .format(epoch + 1, num_epochs, epoch_loss, epoch_accuracy))
    
    with torch.no_grad():
        outputs = model(X_val)
        predicted_probs = outputs.cpu().numpy() 
        auc = roc_auc_score(y_val.cpu().numpy(), predicted_probs) 
        print('Test AUC: {:.4f}'.format(auc))
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return auc




def run_optuna(X_train, X_val, y_train, y_val, n_trials=10):
    def lambda_objective(trial):
        return objective(trial, X_train, X_val, y_train, y_val)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda_objective, n_trials=n_trials)

    print("Best Trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study.best_params

best_params = run_optuna(X_train, X_val, y_train, y_val, n_trials=100)


input_size = df_train.shape[1]
hidden_size = 61

model = MLP(input_size, hidden_size, dropout_rate= 0.23631507605369712)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)


train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=7.650759418753573e-05)

num_epochs = 117
batch_size = 64

train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1)) 

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        epoch_loss += loss.item()  

        # Doğruluk hesaplama
        predicted = (outputs > 0.5).float() 
        correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
        total_samples += labels.size(0)

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

    epoch_loss /= len(train_loader)
    epoch_accuracy = correct_predictions / total_samples

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}'
          .format(epoch + 1, num_epochs, epoch_loss, epoch_accuracy))

with torch.no_grad():
    outputs = model(X_val)
    predicted_probs = outputs.cpu().numpy() 
    auc = roc_auc_score(y_val.cpu().numpy(), predicted_probs) 
    print('Test AUC: {:.4f}'.format(auc))

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)
plt.show()


PATH = "/kaggle/working/mlp_model_state.pth"
torch.save(model.state_dict(), PATH)
print('The models state dictionary has been saved.: {}'.format(PATH))


model = MLP(input_size, hidden_size)
model.load_state_dict(torch.load(PATH))
model.eval()  
print("The model's state dictionary has been loaded.: {}".format(PATH))


test_tensor = torch.tensor(df_test, dtype=torch.float32)

with torch.no_grad():
    outputs = model(test_tensor)
    predicted_probs = outputs.cpu().numpy()  
    print("Olasılıklar:", predicted_probs)


output = pd.DataFrame({
    'id': df_test_id, 
    'rainfall': predicted_probs[:,0]
})

output


output.to_csv('submission.csv', index=False)

