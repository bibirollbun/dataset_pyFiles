# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt #gr谩ficos
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import random
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings('ignore')


if torch.cuda.is_available():
    print("Utilizamos la primera GPU disponible")
    DEVICE=device = torch.device('cuda:0')
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    print("No hay GPU, toca correr todo en CPU")
    DEVICE=device = torch.device('cpu')

DEVICE


os.environ["CUDA_LAUNCH_BLOCKING"] = "1" if torch.cuda.is_available() else "0"


def reset_seed():
    SEED = 42
    torch.backends.cudnn.enabled = True
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


reset_seed()


df = pd.read_parquet('/kaggle/input/aa-v-2025-i-pronosticos-nn-rnn-cnn/df_train.parquet')
print(df.shape)
df.head(5)


# Establecemos una variable que se llame fecha, donde solo te tienen en cuenta el a帽o y el mes
df['fecha'] = df['anio'].astype(str) + df['semana'].astype(str).str.zfill(2)
df = df.sort_values(by=['fecha','id_bar'])



# Group by year and week, summing 'Casos_Dengue'
dengue_by_week = df.groupby(['fecha'])['dengue'].sum().reset_index()

# Rename columns for clarity
dengue_by_week.columns = ['fecha', 'dengue']

plt.figure(figsize=(14,7))
ax = plt.gca() # get current axis
plt.plot(dengue_by_week.fecha, dengue_by_week.dengue)
plt.xlabel('Semana')
plt.ylabel('Casos Dengue')
plt.title('Evoluci贸n de los casos de dengue en funci贸n de la semana')
plt.xticks(np.arange(0,368,9), rotation=90)
plt.show()


# Extract week number from the 'fecha' column
dengue_by_week['week'] = dengue_by_week['fecha'].str[4:].astype(int)

# Group by week number and create boxplots
plt.figure(figsize=(14, 7))
dengue_by_week.boxplot(column='dengue', by='week', figsize=(12, 8))
plt.title('Distribuci贸n de casos de dengue por semana del a帽o')
plt.xlabel('N煤mero de semana')
plt.ylabel('Casos de dengue')
plt.suptitle('') # Remove the default boxplot title
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



plt.figure(figsize=(14,7))
ax = plt.gca()  # Obtener el eje actual

# Graficar cada barrio en el mismo gr谩fico
for barrio_id in df['id_bar'].unique():
    plt.plot(df[df['id_bar'] == barrio_id]['fecha'], df[df['id_bar'] == barrio_id]['dengue'], label=f'Barrio {barrio_id}')

plt.xlabel('Semana')
plt.ylabel('Casos Dengue')
plt.title('Evoluci贸n de los casos de dengue en funci贸n de la semana')
plt.xticks(np.arange(0,368,9), rotation=90)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # Ajustar la leyenda para que no se sobreponga
plt.show()


plt.figure(figsize=(14,7))
ax = plt.gca()  # Obtener el eje actual

# Graficar cada barrio en el mismo gr谩fico
plt.plot(df[df['id_bar'] == barrio_id]['fecha'], df[df['id_bar'] == barrio_id]['dengue'], label=f'Barrio {barrio_id}')

plt.xlabel('Semana')
plt.ylabel('Casos Dengue')
plt.title('Evoluci贸n de los casos de dengue en funci贸n de la semana')
plt.xticks(np.arange(0,368,9), rotation=90)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # Ajustar la leyenda para que no se sobreponga
plt.show()


SEQUENCE_LENGTH = 5
TARGET_COLUMN = 'dengue'
GROUP_COLUMN = 'id_bar'


# Ordenar datos
df = df.sort_values(by=[GROUP_COLUMN, 'anio', 'semana'])
features_cols = df.columns.difference(['anio', 'semana', TARGET_COLUMN, GROUP_COLUMN, 'id', 'fecha'])


df.anio.value_counts(1).sort_index().cumsum()


train_df = df[df.anio <= 2020].copy()
val_df = df[df.anio >= 2021].copy()


scaler = StandardScaler().fit(train_df[features_cols])
train_df[features_cols] = scaler.transform(train_df[features_cols])
val_df[features_cols] = scaler.transform(val_df[features_cols])


class DengueDataset(Dataset):
    def __init__(self, df, sequence_length, is_train=True):
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.data = []
        grouped = df.groupby(GROUP_COLUMN)
        for _, group in grouped:
            group = group.reset_index(drop=True)
            if len(group) > sequence_length:
                for i in range(len(group) - sequence_length):
                    seq_x = group.loc[i:i+sequence_length-1, features_cols].values.flatten()
                    if self.is_train:
                        seq_y = group.loc[i+sequence_length, TARGET_COLUMN]
                        self.data.append((seq_x, seq_y))
                    else:
                        seq_id = group.loc[i+sequence_length, 'id']
                        self.data.append((seq_x, seq_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            x, y = self.data[idx]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            x, seq_id = self.data[idx]
            return torch.tensor(x, dtype=torch.float32), seq_id


# Crear loaders
train_loader = DataLoader(DengueDataset(train_df, SEQUENCE_LENGTH), batch_size=32, shuffle=True)
val_loader = DataLoader(DengueDataset(val_df, SEQUENCE_LENGTH), batch_size=32, shuffle=False)


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze()


input_size = len(features_cols) * SEQUENCE_LENGTH
model = SimpleMLP(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(20):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = criterion(model(x_batch), y_batch)
            val_loss += loss.item()

    print(f'Epoch {epoch+1}/20 | Train MSE: {train_loss/len(train_loader):.4f} | Val MSE: {val_loss/len(val_loader):.4f}')


# Guardar modelo entrenado
torch.save(model.state_dict(), 'dengue_model.pth')


# 1. Unificar train + val
full_train_df = pd.concat([train_df, val_df]).sort_values(by=[GROUP_COLUMN, 'anio', 'semana'])

# 2. (Importante) No volvemos a hacer .fit() al scaler, usamos el original
full_train_df[features_cols] = scaler.transform(full_train_df[features_cols])

# 3. Crear dataset y dataloader
full_dataset = DengueDataset(full_train_df, SEQUENCE_LENGTH, is_train=True)
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)


# 4. Reinicializar modelo (si quieres entrenar desde cero)
model = SimpleMLP(input_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Entrenar con todo el dataset
print("\n���� Reentrenando con todo el dataset (train + val)...\n")
for epoch in range(20):
    model.train()
    train_loss = 0
    for x_batch, y_batch in full_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"[FINAL] Epoch {epoch+1}/20 | Train MSE: {train_loss / len(full_loader):.4f}")


# Cargar datos nuevos y preparar submission
df_test = pd.read_parquet('/kaggle/input/aa-v-2025-i-pronosticos-nn-rnn-cnn/df_test.parquet')
df_test = df_test.sort_values(by=[GROUP_COLUMN, 'anio', 'semana'])
df_test[features_cols] = scaler.transform(df_test[features_cols])




# Agregar 煤ltimas SEQUENCE_LENGTH filas del entrenamiento a cada grupo de test
historical_sequences = df.groupby(GROUP_COLUMN).tail(SEQUENCE_LENGTH)
df_test_extended = pd.concat([historical_sequences, df_test], ignore_index=True)
df_test_extended = df_test_extended.sort_values(by=[GROUP_COLUMN, 'anio', 'semana']).reset_index(drop=True)


# Crear dataset para predicci贸n con todos los registros
class DengueTestDataset(Dataset):
    def __init__(self, df, sequence_length):
        self.sequence_length = sequence_length
        self.data = []
        grouped = df.groupby(GROUP_COLUMN)
        for _, group in grouped:
            group = group.reset_index(drop=True)
            for i in range(sequence_length, len(group)):
                seq_x = group.loc[i-sequence_length:i-1, features_cols].values.flatten()
                seq_id = group.loc[i, 'id']
                self.data.append((seq_x, seq_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, seq_id = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), seq_id


# Crear loader para predicci贸n
test_loader = DataLoader(DengueTestDataset(df_test_extended, SEQUENCE_LENGTH), batch_size=32, shuffle=False)


# Generar predicciones
model.eval()
predictions = []
ids = []

with torch.no_grad():
    for x_batch, id_batch in test_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch).cpu().numpy()
        predictions.extend(preds)
        ids.extend(id_batch)



# Preparar submission con todos los registros
df_submission = pd.DataFrame({'id': ids, 'dengue': predictions})

# Exportar a CSV
df_submission.to_csv('submission.csv', index=False)
print(f'Submission guardado en submission.csv, con {len(df_submission)} predicciones.')


df_submission


data = {
    'id_bar': [1, 1, 1, 1, 1, 1, 1],  # Barrio 1
    'anio': [2020, 2020, 2020, 2020, 2020, 2020, 2020],  # A帽o 2020
    'semana': [1, 2, 3, 4, 5, 6, 7],  # Semanas consecutivas
    'dengue': [2, 3, 5, 7, 11, 13, 17],  # Casos de dengue (target)
    'temp_max': [30.1, 30.5, 31.0, 30.8, 29.9, 30.2, 30.6],  # Temperatura m谩xima
    'precipitacion': [12.0, 14.2, 10.5, 11.7, 13.3, 12.8, 14.0],  # Precipitaci贸n
}

train_df = pd.DataFrame(data)
features_cols = ['temp_max', 'precipitacion']
SEQUENCE_LENGTH = 3
TARGET_COLUMN = 'dengue'
GROUP_COLUMN = 'id_bar'


data


# Crear dataset de entrenamiento
dataset_train = DengueDataset(train_df, SEQUENCE_LENGTH, is_train=True)

# Mostrar el tama帽o del dataset
print(f"N煤mero de secuencias en el dataset: {len(dataset_train)}")

# Ver algunas muestras del dataset
for i in range(len(dataset_train)):
    print(f"Ejemplo {i+1}:")
    print(f"Entrada (X): {dataset_train[i][0]}")
    print(f"Salida (Y): {dataset_train[i][1]}")
    print("-" * 50)




