import numpy as np # linear algebra
import pandas as pd # data processing,datos tabulares

pd.set_option('display.max_columns', None) # para mostrar todas las columnas del dataframe al imprimir

import matplotlib.pyplot as plt  # para grÃ¡ficos
import seaborn as sns

# Redes neuronales tradicionales y mÃ©tricas
from sklearn.neural_network import MLPRegressor  # perceptrÃ³n multicapa tradicional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # mÃ©tricas de evaluaciÃ³n
from math import sqrt  # raÃ­z cuadrada (para RMSE)

# LibrerÃ­as para deep learning
import torch  # PyTorch
from torch.utils.data import Dataset, DataLoader  # para definir datasets personalizados y cargarlos en lotes
import torch.nn as nn  # para definir arquitecturas de redes neuronales
import torch.optim as optim  # para definir algoritmos de optimizaciÃ³n (como Adam)
from sklearn.preprocessing import StandardScaler #NormalizaciÃ³n
import random
import os

# CÃ³digo para mostrar archivos 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# Desactiva los warnings para evitar ruido en la salida
import warnings
warnings.filterwarnings('ignore')


# ConfiguraciÃ³n de uso de GPU si estÃ¡ disponible
if torch.cuda.is_available():
    print("Utilizamos la primera GPU disponible")
    DEVICE = device = torch.device('cuda:0')  # selecciona la GPU
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # ayuda en la depuraciÃ³n de errores en CUDA
else:
    print("No hay GPU, toca correr todo en CPU")
    DEVICE = device = torch.device('cpu')  # usa CPU


# Reafirma esa configuraciÃ³n por compatibilidad
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" if torch.cuda.is_available() else "0"


# FunciÃ³n para fijar semillas aleatorias y garantizar resultados reproducibles
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


reset_seed() # se llama a la funciÃ³n para fijar la semilla


# Utilizaremos parquet ya que es un formato eficiente usado en entornos como Kaggle
df = pd.read_parquet('/kaggle/input/fa-ii-2025-i-pronosticos-nn-rnn-cnn/df_train.parquet')
print(df.shape)
df.head(5)


# EXPLORACIÃ“N INICIAL DE LOS DATOS

print("\nResumen estadÃ­stico de las variables numÃ©ricas:")
df.describe()

print("\nValores faltantes por columna:")
print(df.isnull().sum())

print("\nTipos de variables:")
print(df.dtypes)


# Establecemos una variable que se llame fecha, donde solo te tienen en cuenta el aÃ±o y el mes

# Se crea una columna 'fecha' con formato "AÃ‘OSEMANA" (ej: 201501)
df['fecha'] = df['anio'].astype(str) + df['semana'].astype(str).str.zfill(2)

# Se ordena por fecha y luego por barrio
df = df.sort_values(by=['fecha', 'id_bar'])

print(df.shape)
df.head(5)



# Convertir cÃ³digo de fecha (ej. 201501) a fecha real usando aÃ±o y semana
# 1. Convertimos a string para separar aÃ±o y semana
df['fecha'] = df['fecha'].astype(str)

# 2. Creamos columnas separadas
df['anio'] = df['fecha'].str[:4].astype(int)
df['semana'] = df['fecha'].str[4:].astype(int)

# 3. Creamos una fecha vÃ¡lida usando la semana epidemiolÃ³gica (lunes de esa semana)
df['fecha'] = pd.to_datetime(df['anio'].astype(str) + df['semana'].astype(str) + '1', format='%G%V%u', errors='coerce')

# Verificamos que no haya fechas nulas
print(df['fecha'].isnull().sum())



# Agrupamos por semana y graficamos la evoluciÃ³n para detectar patrones estacionales

df['fecha'] = pd.to_datetime(df['fecha'])
df_weekly = df.groupby('fecha')['dengue'].sum().reset_index()

plt.figure(figsize=(12,5))
plt.plot(df_weekly['fecha'], df_weekly['dengue'], marker='o', linewidth=1)
plt.title("EvoluciÃ³n semanal de casos de dengue")
plt.xlabel("Fecha")
plt.ylabel("NÃºmero de casos")
plt.grid(True)
plt.tight_layout()
plt.show()

# Comentario:
# Este grÃ¡fico nos permite observar picos de contagio y estacionalidad,
# clave para cualquier modelo temporal.


dengue_yearly = df.groupby('anio')['dengue'].sum().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=dengue_yearly, x='anio', y='dengue', palette='Blues_d')
plt.title('ğŸ“Š Total anual de casos de dengue')
plt.xlabel('AÃ±o')
plt.ylabel('Casos de dengue')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# CORRELACIÃ“N ENTRE VARIABLES NUMÃ‰RICAS
# Eliminamos columnas irrelevantes antes de la correlaciÃ³n
variables_numericas = ['dengue', 'lluvia_mean', 'lluvia_max', 'lluvia_var',
                       'temperatura_mean', 'temperatura_max', 'temperatura_min',
                       'sumideros', 'equipesado', 'maquina', 'concentraciones']

correlation_matrix = df[variables_numericas].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlaciÃ³n entre variables")
plt.show()

# Algunas variables climÃ¡ticas y operativas tienen correlaciones leves con el dengue.
# Temperatura mÃ­nima parece estar dÃ©bilmente asociada. Esto orientarÃ¡ la selecciÃ³n de variables para los modelos.


plt.figure(figsize=(14,7))
ax = plt.gca()  # Obtener el eje actual

# Graficar cada barrio en el mismo grÃ¡fico
plt.plot(df[df['id_bar'] == barrio_id]['fecha'], df[df['id_bar'] == barrio_id]['dengue'], label=f'Barrio {barrio_id}')

plt.xlabel('Semana')
plt.ylabel('Casos Dengue')
plt.title('EvoluciÃ³n de los casos de dengue en funciÃ³n de la semana')
plt.xticks(np.arange(0,368,9), rotation=90)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # Ajustar la leyenda para que no se sobreponga
plt.show()


# Agrupar casos totales de dengue por barrio
top_barrios = df.groupby('id_bar')['dengue'].sum().reset_index()

# Ordenar de mayor a menor
top_barrios = top_barrios.sort_values(by='dengue', ascending=False).head(10)

# Graficar
plt.figure(figsize=(10, 6))
sns.barplot(data=top_barrios, x='id_bar', y='dengue', palette='Reds_r')
plt.title('ğŸ�˜ï¸� Top 10 barrios con mÃ¡s casos de dengue (2015â€“2021)')
plt.xlabel('ID del Barrio')
plt.ylabel('Casos Totales de Dengue')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Agrupar por estrato y sumar casos de dengue
dengue_por_estrato = df.groupby('ESTRATO')['dengue'].sum().reset_index()

# Graficar distribuciÃ³n
plt.figure(figsize=(8, 5))
sns.barplot(data=dengue_por_estrato, x='ESTRATO', y='dengue', palette='Purples')
plt.title('ğŸ“Š Casos de dengue por estrato socioeconÃ³mico')
plt.xlabel('Estrato')
plt.ylabel('Casos Totales de Dengue')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# GrÃ¡ficos de dispersiÃ³n entre dengue y variables climÃ¡ticas

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# DispersiÃ³n: dengue vs lluvia promedio
sns.scatterplot(data=df, x='lluvia_mean', y='dengue', ax=axs[0], alpha=0.5)
axs[0].set_title('ğŸŒ§ï¸� Dengue vs Lluvia Promedio')
axs[0].set_xlabel('Lluvia Promedio (lluvia_mean)')
axs[0].set_ylabel('Casos de Dengue')

# DispersiÃ³n: dengue vs temperatura promedio
sns.scatterplot(data=df, x='temperatura_mean', y='dengue', ax=axs[1], alpha=0.5, color='orange')
axs[1].set_title('ğŸŒ¡ï¸� Dengue vs Temperatura Promedio')
axs[1].set_xlabel('Temperatura Promedio (temperatura_mean)')
axs[1].set_ylabel('Casos de Dengue')

plt.tight_layout()
plt.show()



# DETECCIÃ“N DE OUTLIERS (IQR)
# Utilizamos el mÃ©todo del rango intercuartÃ­lico para detectar valores atÃ­picos en variables crÃ­ticas

outlier_summary = []

for var in variables_numericas:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[var] < lower) | (df[var] > upper)]
    outlier_summary.append({
        'Variable': var,
        'Total Outliers': len(outliers),
        'Porcentaje del Total (%)': round(100 * len(outliers) / len(df), 2),
        'Min': df[var].min(),
        'Q1': Q1,
        'Q3': Q3,
        'Max': df[var].max()
    })

outlier_df = pd.DataFrame(outlier_summary)
print("\nResumen de outliers detectados:")
print(outlier_df)

# Las decisiones sobre si eliminar, reemplazar por media o mediana, o transformar logarÃ­tmicamente
# se harÃ¡n en funciÃ³n del % de outliers por variable y su relevancia en el modelo.


# TRATAMIENTO DE OUTLIERS Y TRANSFORMACIONES

# Reemplazo de outliers por media (porcentaje < 5%)
for var in ['lluvia_mean', 'lluvia_max', 'temperatura_mean', 'temperatura_max', 'sumideros']:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    media = df[var].mean()
    df[var] = df[var].apply(lambda x: media if x < lower or x > upper else x)

# Reemplazo de outliers por mediana para variables discretas (sin sesgo evidente)
for var in ['lluvia_var', 'temperatura_min']:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mediana = df[var].median()
    df[var] = df[var].apply(lambda x: mediana if x < lower or x > upper else x)

# TransformaciÃ³n logarÃ­tmica para variables con muchos ceros y sesgo
for var in ['equipesado', 'maquina', 'concentraciones']:
    df[var] = np.log1p(df[var])  # log1p(x) = log(1 + x) evita log(0)

# Comentario:
# Este tratamiento balancea los datos sin perder observaciones importantes,
# preservando la estructura temporal y el contexto epidemiolÃ³gico y operativo.



# Variables ANTES y DESPUÃ‰S del tratamiento de outliers

# Seleccionamos las variables que sufrieron transformaciÃ³n o reemplazo
vars_transformadas = ['lluvia_max', 'sumideros', 'concentraciones', 'maquina']

# Creamos una copia del dataframe original para comparar
df_original = pd.read_parquet("/kaggle/input/fa-ii-2025-i-pronosticos-nn-rnn-cnn/df_train.parquet")

# GrÃ¡ficos de distribuciÃ³n antes y despuÃ©s
fig, axs = plt.subplots(len(vars_transformadas), 2, figsize=(12, 3 * len(vars_transformadas)))
fig.suptitle("ComparaciÃ³n antes y despuÃ©s del tratamiento de outliers", fontsize=16)

for i, var in enumerate(vars_transformadas):
    # Antes
    sns.boxplot(x=df_original[var], ax=axs[i, 0], color='salmon')
    axs[i, 0].set_title(f"{var} - Antes")
    
    # DespuÃ©s
    sns.boxplot(x=df[var], ax=axs[i, 1], color='lightgreen')
    axs[i, 1].set_title(f"{var} - DespuÃ©s")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# Antes y despuÃ©s del tratamiento de outliers

# Variables tratadas
vars_transformadas = ['lluvia_max', 'sumideros', 'concentraciones', 'maquina']

# Crear diccionarios para guardar los resÃºmenes
resumen_antes = df_original[vars_transformadas].describe().loc[['mean', 'std', 'min', 'max']].T
resumen_despues = df[vars_transformadas].describe().loc[['mean', 'std', 'min', 'max']].T

# Renombrar columnas para distinguirlas
resumen_antes.columns = ['Media_antes', 'STD_antes', 'Min_antes', 'Max_antes']
resumen_despues.columns = ['Media_despues', 'STD_despues', 'Min_despues', 'Max_despues']

# Combinar ambos resÃºmenes en un solo DataFrame
resumen_comparativo = pd.concat([resumen_antes, resumen_despues], axis=1)
resumen_comparativo = resumen_comparativo.round(3)

# Mostrar tabla en notebook
display(resumen_comparativo)



# Definimos variable objetivo y columnas clave
TARGET_COLUMN = 'dengue'
GROUP_COLUMN = 'id_bar'  # Para separar los datos por barrio
SEQUENCE_LENGTH = 5      # NÃºmero de semanas que se usarÃ¡n para predecir la siguiente


features_cols_ini


df.anio.value_counts(1).sort_index().cumsum()


train_df = df[df.anio <= 2020].copy()
val_df = df[df.anio >= 2021].copy()


features_cols = features_cols_ini.copy()
scaler = StandardScaler().fit(train_df[features_cols])
train_df[features_cols] = scaler.transform(train_df[features_cols])
val_df[features_cols] = scaler.transform(val_df[features_cols])


train_df


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


#data = train_df[train_df.id_bar==0].head(10).copy()


#features_cols = ['temperatura_mean', 'lluvia_mean']
#SEQUENCE_LENGTH_p = 3


#data[['semana','id_bar','dengue']+features_cols]


'''# Crear dataset de entrenamiento
dataset_train = DengueDataset(data, SEQUENCE_LENGTH_p, is_train=True)

# Mostrar el tamaÃ±o del dataset
print(f"NÃºmero de secuencias en el dataset: {len(dataset_train)}")

# Ver algunas muestras del dataset
for i in range(len(dataset_train)):
    print(f"Ejemplo {i+1}:")
    print(f"Entrada (X): {dataset_train[i][0]}")
    print(f"Salida (Y): {dataset_train[i][1]}")
    print("-" * 50)
'''


# Crear loaders PARA ENTRENAMIENTO Y VALIDACIÃ“N
train_loader = DataLoader(DengueDataset(train_df, SEQUENCE_LENGTH), batch_size=32, shuffle=True)
val_loader = DataLoader(DengueDataset(val_df, SEQUENCE_LENGTH), batch_size=32, shuffle=False)


#DefiniciÃ³n del modelo
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Capa oculta
            nn.ReLU(),
            nn.Linear(hidden_size, 1)            # Capa de salida (regresiÃ³n)
            #nn.ReLU()  #Se elimina para permitir predicciones cercanas a cero
        )

    def forward(self, x):
        return self.fc(x).squeeze()


#InicializaciÃ³n del modelo y entorno
features_cols = features_cols_ini.copy()  # Lista de variables usadas como entrada
input_size = len(features_cols) * SEQUENCE_LENGTH

model = SimpleMLP(input_size).to(device)  # Mover a GPU o CPU

criterion = nn.MSELoss()  # PÃ©rdida cuadrÃ¡tica
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Recalcular input_size si cambiaste features_cols
input_size = len(features_cols) * SEQUENCE_LENGTH

# Definir y enviar el modelo al dispositivo
mlp_model = SimpleMLP(input_size).to(device)

# Funciones de pÃ©rdida y optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

# Entrenamiento con early stopping
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(50):
    mlp_model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(mlp_model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    mlp_model.eval()
    val_loss = 0
    val_mae = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = mlp_model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
            val_mae += torch.mean(torch.abs(preds - y_batch)).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)
    print(f"Epoch {epoch+1} | Train MSE: {train_loss/len(train_loader):.4f} | "
          f"Val MSE: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(mlp_model.state_dict(), 'dengue_model.pth')  # âœ… Se guarda el modelo nuevo
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping en la Ã©poca {epoch+1}")
            break



class CNNForecast(nn.Module):
    def __init__(self, sequence_length, num_features):
        super(CNNForecast, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # reduce la dimensiÃ³n temporal
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = x.view(x.size(0), SEQUENCE_LENGTH, -1)   # (batch, seq_len, features)
        x = x.permute(0, 2, 1)                       # (batch, features, seq_len) para Conv1d
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze()


# Entrenamiento CNN
num_features = len(features_cols)
model = CNNForecast(sequence_length=SEQUENCE_LENGTH, num_features=num_features).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# EarlyStopping
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(50):
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
    val_mae = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
            val_mae += torch.mean(torch.abs(preds - y_batch)).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)
    print(f"Epoch {epoch+1} | Train MSE: {train_loss/len(train_loader):.4f} | "
          f"Val MSE: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'cnn_dengue_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping en la Ã©poca {epoch+1}")
            break


import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), SEQUENCE_LENGTH, -1)  # (batch, time, features)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Salida del Ãºltimo paso temporal
        return self.fc(out).squeeze()



# Inicializar modelo RNN
input_size = len(features_cols)
model = SimpleRNN(input_size=input_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento con EarlyStopping
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(50):
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
    val_mae = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
            val_mae += torch.mean(torch.abs(preds - y_batch)).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)
    print(f"Epoch {epoch+1} | Train MSE: {train_loss/len(train_loader):.4f} | "
          f"Val MSE: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'rnn_dengue_model.pth')  # âœ… GUARDAR MEJOR MODELO
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping en la Ã©poca {epoch+1}")
            break



import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), SEQUENCE_LENGTH, -1)  # (batch, time, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # tomar el Ãºltimo paso temporal
        return self.fc(out).squeeze()



import optuna
from optuna.trial import TrialState

def objective(trial):
    # Sugerencia de hiperparÃ¡metros
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # Modelo
    input_size = len(features_cols)
    model = LSTMForecast(input_size, hidden_size, num_layers, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Entrenamiento simplificado con EarlyStopping
    best_val_loss = float('inf')
    counter = 0
    patience = 3

    for epoch in range(15):  # menos Ã©pocas para pruebas rÃ¡pidas
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        # ValidaciÃ³n
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                val_loss += criterion(preds, y_batch).item()

        avg_val_loss = val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)

        # EarlyStopping interno de Optuna
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_val_loss



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # Puedes aumentar n_trials si tienes tiempo

print("Mejores hiperparÃ¡metros encontrados:")
print(study.best_params)



best_params = study.best_params
model = LSTMForecast(
    input_size=len(features_cols),
    hidden_size=best_params['hidden_size'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout']
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

# Entrenamiento final
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(50):
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
            preds = model(x_batch)
            val_loss += criterion(preds, y_batch).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'lstm_dengue_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break



import torch
import torch.nn as nn

class GRUForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUForecast, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), SEQUENCE_LENGTH, -1)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Ãºltimo paso temporal
        return self.fc(out).squeeze()



import optuna

def objective_gru(trial):
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = GRUForecast(len(features_cols), hidden_size, num_layers, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    counter = 0
    patience = 3

    for epoch in range(15):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                val_loss += criterion(preds, y_batch).item()

        avg_val_loss = val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_val_loss



study_gru = optuna.create_study(direction='minimize')
study_gru.optimize(objective_gru, n_trials=50)

print("Mejores hiperparÃ¡metros GRU:")
print(study_gru.best_params)



best_params_gru = study_gru.best_params

model = GRUForecast(
    input_size=len(features_cols),
    hidden_size=best_params_gru['hidden_size'],
    num_layers=best_params_gru['num_layers'],
    dropout=best_params_gru['dropout']
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params_gru['lr'])

best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(50):
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
            preds = model(x_batch)
            val_loss += criterion(preds, y_batch).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'gru_dengue_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break



# ğŸ–¼ï¸� GrÃ¡fico de dispersiÃ³n
plt.figure(figsize=(8, 5))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel("Casos reales de dengue")
plt.ylabel("Casos predichos por GRU")
plt.title("PredicciÃ³n de casos con GRU")
plt.grid(True)
plt.show()


import optuna
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

def objective(trial):
    # HiperparÃ¡metros sugeridos por Optuna
    num_layers = trial.suggest_int('num_layers', 2, 4)
    channels = [trial.suggest_categorical(f'n_ch_{i}', [32, 64, 128]) for i in range(num_layers)]
    kernel_size = trial.suggest_categorical('kernel_size', [2, 3, 5])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # Modelo
    model = TCN(input_size=input_size,
                num_channels=channels,
                kernel_size=kernel_size,
                dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Entrenamiento corto (solo 10 epochs con early stopping)
    best_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(10):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                preds = model(x_batch).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_batch.numpy())

        rmse = mean_squared_error(val_targets, val_preds, squared=False)
        if rmse < best_loss:
            best_loss = rmse
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_loss



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)



print("ğŸ�† Mejor configuraciÃ³n encontrada:")
print(study.best_params)



# Recuperar mejores hiperparÃ¡metros encontrados por Optuna
best_params = study.best_params

num_layers = best_params['num_layers']
channels = [best_params[f'n_ch_{i}'] for i in range(num_layers)]
kernel_size = best_params['kernel_size']
dropout = best_params['dropout']
lr = best_params['lr']

# Inicializar modelo final
model_tcn = TCN(
    input_size=input_size,
    num_channels=channels,
    kernel_size=kernel_size,
    dropout=dropout
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model_tcn.parameters(), lr=lr)

# Entrenamiento completo con early stopping
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(50):
    model_tcn.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model_tcn(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model_tcn.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = criterion(model_tcn(x_batch), y_batch)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model_tcn.state_dict(), 'tcn_dengue_model_optuna.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"â�¹ï¸� Early stopping en la Ã©poca {epoch+1}")
            break



from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Cargar modelo entrenado
model_tcn.load_state_dict(torch.load("tcn_dengue_model_optuna.pth"))
model_tcn.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        preds = model_tcn(x_batch).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

# Convertir a arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calcular mÃ©tricas
mae_tcn = mean_absolute_error(y_true, y_pred)
rmse_tcn = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"ğŸ“Š TCN Optuna â†’ MAE: {mae_tcn:.2f} | RMSE: {rmse_tcn:.2f}")

# GrÃ¡fico
plt.figure(figsize=(8, 5))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel("Casos reales de dengue")
plt.ylabel("Casos predichos")
plt.title("Predicciones del TCN (Optuna)")
plt.grid(True)
plt.show()



model_gru.eval()
model_tcn.eval()

y_true_ens, y_pred_gru, y_pred_tcn = [], [], []

with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)

        preds_gru = model_gru(x_batch).cpu().numpy()
        preds_tcn = model_tcn(x_batch).cpu().numpy()

        y_true_ens.extend(y_batch.numpy())
        y_pred_gru.extend(preds_gru)
        y_pred_tcn.extend(preds_tcn)

# ConversiÃ³n a arrays
y_true_ens = np.array(y_true_ens)
y_pred_gru = np.array(y_pred_gru)
y_pred_tcn = np.array(y_pred_tcn)

# Ensemble: promedio ponderado
alpha = 0.7  # peso GRU (ajustable)
y_pred_ens = alpha * y_pred_gru + (1 - alpha) * y_pred_tcn

# MÃ©tricas
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_ens = mean_absolute_error(y_true_ens, y_pred_ens)
rmse_ens = np.sqrt(mean_squared_error(y_true_ens, y_pred_ens))

print(f"ğŸ¤� Ensemble (GRU + TCN) â†’ MAE: {mae_ens:.2f} | RMSE: {rmse_ens:.2f}")


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluar_modelo(modelo, loader, nombre_modelo):
    modelo.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds = modelo(x_batch).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{nombre_modelo} â†’ MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    return y_true, y_pred



# CNN
model_cnn = CNNForecast(SEQUENCE_LENGTH, len(features_cols)).to(device)
model_cnn.load_state_dict(torch.load("cnn_dengue_model.pth"))
y_true_cnn, y_pred_cnn = evaluar_modelo(model_cnn, val_loader, "CNN")



# RNN
model_rnn = SimpleRNN(len(features_cols)).to(device)
model_rnn.load_state_dict(torch.load("rnn_dengue_model.pth"))
y_true_rnn, y_pred_rnn = evaluar_modelo(model_rnn, val_loader, "RNN")



# LSTM
model_lstm = LSTMForecast(
    input_size=len(features_cols),
    hidden_size=study.best_params['hidden_size'],
    num_layers=study.best_params['num_layers'],
    dropout=study.best_params['dropout']
).to(device)
model_lstm.load_state_dict(torch.load("lstm_dengue_model.pth"))
y_true_lstm, y_pred_lstm = evaluar_modelo(model_lstm, val_loader, "LSTM")



# GRU
model_gru = GRUForecast(
    input_size=len(features_cols),
    hidden_size=study_gru.best_params['hidden_size'],
    num_layers=study_gru.best_params['num_layers'],
    dropout=study_gru.best_params['dropout']
).to(device)
model_gru.load_state_dict(torch.load("gru_dengue_model.pth"))
y_true_gru, y_pred_gru = evaluar_modelo(model_gru, val_loader, "GRU")



# Cargar datos nuevos y preparar submission
features_cols = features_cols_ini.copy()
df_test = pd.read_parquet('/kaggle/input/fa-ii-2025-i-pronosticos-nn-rnn-cnn/df_test.parquet')
df_test = df_test.sort_values(by=[GROUP_COLUMN, 'anio', 'semana'])

# NormalizaciÃ³n con el mismo scaler del entrenamiento
df_test[features_cols] = scaler.transform(df_test[features_cols])




# Unir Ãºltimas secuencias del entrenamiento
historical_sequences = df.groupby(GROUP_COLUMN).tail(SEQUENCE_LENGTH)
df_test_extended = pd.concat([historical_sequences, df_test], ignore_index=True)
df_test_extended = df_test_extended.sort_values(by=[GROUP_COLUMN, 'anio', 'semana']).reset_index(drop=True)


df_test_extended


# Crear loader para predicciÃ³n
test_loader = DataLoader(DengueDataset(df_test_extended, SEQUENCE_LENGTH, is_train=False), batch_size=32, shuffle=False)


# Cargar modelo GRU entrenado 
model_gru = GRUForecast(
    input_size=len(features_cols),
    hidden_size=study_gru.best_params['hidden_size'],
    num_layers=study_gru.best_params['num_layers'],
    dropout=study_gru.best_params['dropout']
).to(device)

model_gru.load_state_dict(torch.load("gru_dengue_model.pth"))
model_gru.eval()

# Generar predicciones 
predictions = []
ids = []

with torch.no_grad():
    for x_batch, id_batch in test_loader:
        x_batch = x_batch.to(device)
        preds = model_gru(x_batch).cpu().numpy()
        predictions.extend(preds)
        ids.extend(id_batch)



len(ids)


# Preparar submission con todos los registros
df_submission = pd.DataFrame({'id': ids, 'dengue': predictions})
df_submission = df_submission.sort_values(by='id')

# Exportar a CSV
df_submission.to_csv('submission.csv', index=False)
print(f'Submission guardado en submission.csv, con {len(df_submission)} predicciones.')

df_submission.head()


df_submission




