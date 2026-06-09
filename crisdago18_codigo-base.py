# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt #grÃ¡ficos
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


df = pd.read_parquet('/kaggle/input/fa-ii-2025-ii-pronosticos-nn-rnn-lstm-tcn/df_train.parquet')
print(df.shape)
df.head(5)


df.tail(5)


df.describe()


# Crear columna con el aÃ±o
df["year"] = df["date"].dt.year

# Boxplot por aÃ±o de la variable seleccionada
variable = "avg_temp"  # <-- Cambia por la variable que quieras analizar, importante para tu EDA

plt.figure(figsize=(18,6))
df.boxplot(column=variable, by="year", grid=False, showfliers=True)  # showfliers=False quita outliers extremos, True los deja
plt.title(f"DistribuciÃ³n anual de {variable}")
plt.suptitle("")  
plt.xlabel("AÃ±o")
plt.ylabel(variable)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date") # Tengan en cuenta asegurar el ordenamiento de esta serie de tiempo

# Visualizar la serie de tiempo de precipitaciÃ³n
plt.figure(figsize=(14,5))
plt.plot(df["date"], df["min_temp"], color="blue", linewidth=1)
plt.title("Serie de tiempo - min_temp")
plt.xlabel("Fecha")
plt.ylabel("min_temp")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Graficar solo el Ãºltimo aÃ±o (365 dÃ­as)
plt.figure(figsize=(14,5))
plt.plot(df.tail(365)["date"], df.tail(365)["precipitation"], color="blue", linewidth=1)
plt.title("Serie de tiempo - PrecipitaciÃ³n diaria (Ãºltimo aÃ±o)")
plt.xlabel("Fecha")
plt.ylabel("PrecipitaciÃ³n (mm)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Calcular media mÃ³vil de 21 dÃ­as
df["precipitation_ma21"] = df["precipitation"].rolling(window=21).mean()

# Graficar serie original vs media mÃ³vil
plt.figure(figsize=(14,5))
plt.plot(df["date"], df["precipitation"], color="lightblue", alpha=0.6, label="PrecipitaciÃ³n diaria")
plt.plot(df["date"], df["precipitation_ma21"], color="red", linewidth=2, label="Media mÃ³vil (21 dÃ­as)")
plt.title("Serie de tiempo - PrecipitaciÃ³n diaria con media mÃ³vil de 21 dÃ­as")
plt.xlabel("Fecha")
plt.ylabel("PrecipitaciÃ³n (mm)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Calcular medias mÃ³viles
df["precipitation_ma7"] = df["precipitation"].rolling(window=7).mean()
df["precipitation_ma14"] = df["precipitation"].rolling(window=14).mean()
df["precipitation_ma30"] = df["precipitation"].rolling(window=30).mean()

# Filtrar Ãºltimo aÃ±o (365 dÃ­as)
df_last_year = df.tail(365)

# Graficar serie original y suavizadas
plt.figure(figsize=(14,6))
plt.plot(df_last_year["date"], df_last_year["precipitation"], color="lightblue", alpha=0.5, label="PrecipitaciÃ³n diaria")
plt.plot(df_last_year["date"], df_last_year["precipitation_ma7"], color="red", linewidth=2, label="Media mÃ³vil 7 dÃ­as")
plt.plot(df_last_year["date"], df_last_year["precipitation_ma14"], color="green", linewidth=2, label="Media mÃ³vil 14 dÃ­as")
plt.plot(df_last_year["date"], df_last_year["precipitation_ma30"], color="purple", linewidth=2, label="Media mÃ³vil 30 dÃ­as")

plt.title("Serie de tiempo - PrecipitaciÃ³n (Ãºltimo aÃ±o) con medias mÃ³viles")
plt.xlabel("Fecha")
plt.ylabel("PrecipitaciÃ³n (mm)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Filtrar Ãºltimos dos aÃ±o
df_last_year = df.tail(730)

# Crear figura y eje
fig, ax1 = plt.subplots(figsize=(14,6))

# Primer eje Y: PrecipitaciÃ³n
color = "blue"
ax1.set_xlabel("Fecha")
ax1.set_ylabel("PrecipitaciÃ³n (mm)")
ax1.plot(df_last_year["date"], df_last_year["precipitation"], color=color, linewidth=1.5, label="PrecipitaciÃ³n")
ax1.tick_params(axis="y")

# Segundo eje Y: Temperatura promedio
ax2 = ax1.twinx()
color = "red"
ax2.set_ylabel("Temperatura promedio (Â°C)")
ax2.plot(df_last_year["date"], df_last_year["avg_temp"], color=color, linewidth=1.5, label="Temperatura promedio")
ax2.tick_params(axis="y")

# TÃ­tulo y ajustes
plt.title("PrecipitaciÃ³n y Temperatura promedio - Ãšltimos dos aÃ±o")
fig.tight_layout()
plt.show()


import seaborn as sns
# Asegurar que la columna date no entre en el cÃ¡lculo
df_corr = df.drop(columns=["date", "year"])

# Calcular la matriz de correlaciones
corr_matrix = df_corr.corr()

# Visualizar con un mapa de calor
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de correlaciones")
plt.tight_layout()
plt.show()


df


sequence_length = 5
TARGET_COLUMN = 'precipitation'


# Contar registros por aÃ±o
counts_per_year = df.groupby(df["date"].dt.year).size()

# Calcular acumulado por aÃ±o (cumsum)
cumsum_per_year = counts_per_year.cumsum()

# Mostrar resultado
print(cumsum_per_year)


# Convertir a porcentaje
cumsum_perc = (cumsum_per_year / cumsum_per_year.iloc[-1]) * 100

# Mostrar resultado
print(cumsum_perc)


df


train_df = df[df.year <= 2023].copy()
val_df = df[df.year >= 2024].copy()


train_df


val_df


features_cols = [
    'avg_rel_humidity', 
    'avg_temp', 
    'evapotranspiration', 
    'max_rel_humidity', 
    'max_temp', 
    'min_rel_humidity', 
    'min_temp', 
    'solar_radiation'
]


features_cols


train_df


val_df


# Columnas a eliminar
cols_to_drop = ["precipitation_ma21", "precipitation_ma7", "precipitation_ma14", "precipitation_ma30"]

# Eliminar de train y val
train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
val_df   = val_df.drop(columns=cols_to_drop, errors="ignore")


train_df


val_df


scaler = StandardScaler().fit(train_df[features_cols])
train_df[features_cols] = scaler.transform(train_df[features_cols])
val_df[features_cols] = scaler.transform(val_df[features_cols])


train_df


val_df


'''
from sklearn.preprocessing import MinMaxScaler
# Inicializar el MinMaxScaler
scaler = MinMaxScaler().fit(train_df[features_cols])

# Transformar train, val con el mismo fit
train_df[features_cols] = scaler.transform(train_df[features_cols])
val_df[features_cols]   = scaler.transform(val_df[features_cols])
'''


'''
from sklearn.preprocessing import Normalizer
normalizer = Normalizer().fit(train_df[features_cols])

# Aplicar la transformaciÃ³n usando el mismo fit
train_df[features_cols] = normalizer.transform(train_df[features_cols])
val_df[features_cols]   = normalizer.transform(val_df[features_cols])
'''


df


# Contar nÃºmero de NaN por columna
nan_count = df.isna().sum()

print(nan_count)


# ==== 1) Limpieza de datos para evitar NaN/Inf en features y target ====
cols_needed = features_cols + [TARGET_COLUMN]

def clean_split(df):
    df2 = df.copy()
    # Reemplaza inf por NaN y elimina filas con NaN en lo necesario
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna(subset=cols_needed)
    return df2

train_df_clean = clean_split(train_df)
val_df_clean   = clean_split(val_df)


val_df_clean


# ==== 2) Dataset que entrega ventanas aplanadas (MLP) con (X, y) ====
class PrecipitationDataset(Dataset):
    def __init__(self, df, sequence_length, features_cols, TARGET_COLUMN, is_train=True, horizon=1): # validar return_mode="sequence"
        self.seq_len = sequence_length
        self.features_cols = list(features_cols)
        self.TARGET_COLUMN = TARGET_COLUMN
        self.h = horizon
        self.data = []

        g = df.reset_index(drop=True)
        n = len(g)
        max_i = n - self.seq_len - (self.h - 1)
        if max_i <= 0:
            return

        for i in range(max_i):
            X = g.iloc[i : i + self.seq_len][self.features_cols].values  # (W, F)
            X = X.reshape(-1)  # aplanado para MLP
            y = g.iloc[i + self.seq_len + (self.h - 1)][self.TARGET_COLUMN]
            self.data.append((X, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y


# ==== 3) Crear datasets
sequence_length = sequence_length  # ya lo tienes definido
train_dataset = PrecipitationDataset(train_df_clean, sequence_length, features_cols, TARGET_COLUMN, horizon=1)
val_dataset   = PrecipitationDataset(val_df_clean,   sequence_length, features_cols, TARGET_COLUMN, horizon=1)


# ==== 4) DataLoaders (ahora sÃ­) ====
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"NÃºmero de batches en train_loader: {len(train_loader)}")
print(f"NÃºmero de batches en val_loader:   {len(val_loader)}")


# ==== 5) Modelo MLP 
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.fc(x).squeeze(-1)  # <- evita problemas con batch=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = len(features_cols) * sequence_length
model = SimpleMLP(input_size).to(device)


# ==== 6) Entrenamiento con loaders ====
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Este hiperparametro pueden ensayarlo con otros valores

for epoch in range(20):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # sanity check: sin NaN
        if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
            raise ValueError("Hay NaN en x_batch o y_batch (train)")

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        if torch.isnan(loss):
            raise ValueError("Loss NaN en train; revisa datos/targets.")
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- Val ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                raise ValueError("Hay NaN en x_batch o y_batch (val)")

            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            if torch.isnan(loss):
                raise ValueError("Loss NaN en val; revisa datos/targets.")
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/20 | Train MSE: {train_loss/len(train_loader):.4f} | Val MSE: {val_loss/len(val_loader):.4f}")


torch.save(model.state_dict(), 'preciÃ­tation_model.pth')


# --- 1) Unificar train + val y ordenar temporalmente ---
# Si existe GROUP_COLUMN en los dataframes, se ordena por grupo y fecha; si no, solo por fecha.
def build_full_train(train_df, val_df):
    full = pd.concat([train_df, val_df], ignore_index=True)
    # columnas de ordenamiento
    sort_cols = []
    if 'GROUP_COLUMN' in globals() and GROUP_COLUMN in full.columns:
        sort_cols.append(GROUP_COLUMN)
    # usa 'anio' y 'semana' si existen; en tu caso suele existir 'date'
    if 'anio' in full.columns:   sort_cols.append('anio')
    if 'semana' in full.columns: sort_cols.append('semana')
    if 'date' in full.columns and 'anio' not in full.columns:
        sort_cols.append('date')

    if sort_cols:
        full = full.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        full = full.sort_values(by=full.columns.tolist()).reset_index(drop=True)
    return full

full_train_df = build_full_train(train_df, val_df)

# --- 2) (Importante) No volver a hacer .fit() ---
# Usa el mismo transformador que ya tengas (scaler o normalizer). No re-ajustes para evitar leakage.
# Intentamos aplicar 'scaler' si existe; si no, 'normalizer'. Omite si no quieres transformar.
if 'scaler' in globals() and scaler is not None:
    full_train_df[features_cols] = scaler.transform(full_train_df[features_cols])
elif 'normalizer' in globals() and normalizer is not None:
    full_train_df[features_cols] = normalizer.transform(full_train_df[features_cols])
# Si no quieres ninguna transformaciÃ³n, comenta el bloque anterior.

# (Opcional pero recomendado) Limpiar inf/NaN por si quedaron al unir
full_train_df = full_train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_cols + [TARGET_COLUMN])

# --- 3) Crear dataset y dataloader (usa tu clase PrecipitationDataset) ---
sequence_length = sequence_length  # ya definido por ti
BATCH_SIZE = 32

# Para MLP (entrada aplanada):
full_dataset = PrecipitationDataset(
    full_train_df,
    sequence_length=sequence_length,
    features_cols=features_cols,
    TARGET_COLUMN=TARGET_COLUMN,
    is_train=True,
    #return_mode="flat",   # "sequence" si vas a usar RNN/LSTM/GRU/TCN
    horizon=1
)

full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 4. Reinicializar modelo (si quieres entrenar desde cero)
model = SimpleMLP(input_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Entrenar con todo el dataset
print("\nğŸ”� Reentrenando con todo el dataset (train + val)...\n")
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


# --- 1) Cargar test y ordenar ---
df_test = pd.read_parquet('/kaggle/input/fa-ii-2025-ii-pronosticos-nn-rnn-lstm-tcn/df_test.parquet')
df_test['date'] = pd.to_datetime(df_test['date'])
df_test = df_test.sort_values('date').reset_index(drop=True) #Se los dejo por si en otro caso lo consideran necesario


# (opcional) limpieza defensiva
df_test = df_test.replace([np.inf, -np.inf], np.nan)
# Si algÃºn feature tiene NaN, dependerÃ¡ de tu caso: puedes rellenar o descartar.
# AquÃ­ supongo que no hay NaN en test. Si los hay, considera un fillna sensato.
# df_test[features_cols] = df_test[features_cols].fillna(method='ffill').fillna(method='bfill')


# --- 2) Aplicar misma transformaciÃ³n de entrenamiento
if 'scaler' in globals() and scaler is not None:
    df_test[features_cols] = scaler.transform(df_test[features_cols])
elif 'normalizer' in globals() and normalizer is not None:
    df_test[features_cols] = normalizer.transform(df_test[features_cols])


# --- 3) AÃ±adir el "histÃ³rico" del train para poder construir la 1Âª ventana de test ---
# AsegÃºrate de que 'train_df' tiene las mismas columnas y estÃ¡ ordenado por fecha
train_hist = train_df.sort_values('date').reset_index(drop=True)
# (Si aplicaste scaler/normalizer a train/val, NO vuelvas a fittear. Si necesitas,
# aplica transform al hist usando el mismo objeto)
if 'scaler' in globals() and scaler is not None:
    train_hist_feat = train_hist[features_cols].copy()
    train_hist[features_cols] = scaler.transform(train_hist_feat)
elif 'normalizer' in globals() and normalizer is not None:
    train_hist_feat = train_hist[features_cols].copy()
    train_hist[features_cols] = normalizer.transform(train_hist_feat)

# Tomar las Ãºltimas sequence_length filas del train (solo para construir la 1Âª ventana)
hist_tail = train_hist.tail(sequence_length)[features_cols + ['date']]

# Concatenar: histÃ³rico + test
df_test_extended = pd.concat([hist_tail, df_test[features_cols + ['date']]], ignore_index=True)
df_test_extended = df_test_extended.sort_values('date').reset_index(drop=True)



# --- 4) Dataset de test (una sola serie temporal, sin 'id'; usamos 'date' como identificador) ---

class PrecipitationTestDataset(Dataset):
    def __init__(self, df, sequence_length, features_cols):
        self.seq_len = sequence_length
        self.features_cols = list(features_cols)
        self.data = []
        g = df.reset_index(drop=True)
        n = len(g)
        for i in range(self.seq_len, n):
            X = g.loc[i-self.seq_len:i-1, self.features_cols].values.flatten()
            date_i = str(g.loc[i, 'date'])  # ğŸ‘ˆ convertir a string
            self.data.append((X, date_i))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        x, date_i = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), date_i



# Crear loader de test
test_dataset = PrecipitationTestDataset(df_test_extended, sequence_length, features_cols)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- 5) Generar predicciones ---
model.eval()
predictions = []
dates = []

with torch.no_grad():
    for x_batch, date_batch in test_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch).detach().cpu().numpy().ravel()
        predictions.extend(preds.tolist())
        # 'date_batch' es un tensor de objetos (pytorch lo deja como list nativa al iterar)
        dates.extend([pd.Timestamp(d) for d in date_batch])

# --- 6) Armar submission ---
df_submission = pd.DataFrame({
    'date': dates,
    'Precipitation': predictions
}).sort_values('date').reset_index(drop=True)

# VerificaciÃ³n: deben haber tantas predicciones como filas en df_test
assert len(df_submission) == len(df_test), f"Predicciones {len(df_submission)} != filas test {len(df_test)}"

# (Opcional) Si la competencia requiere no negativos:
# df_submission['Precipitation'] = np.clip(df_submission['Precipitation'], 0, None)

# Guardar CSV
df_submission.to_csv('submission.csv', index=False)
print(f"Submission guardado en submission.csv, con {len(df_submission)} predicciones.")


df_submission


data_sim = {
    'dia': [1, 2, 3, 4, 5, 6, 7],
    'rad_solar': [2, 3, 5, 7, 11, 13, 17],
    'temp_max': [30.1, 30.5, 31.0, 30.8, 29.9, 30.2, 30.6],
    'precipitation': [12.0, 14.2, 10.5, 11.7, 13.3, 12.8, 14.0]
}

train_df = pd.DataFrame(data_sim)
SEQUENCE_LENGTH = 3
features_cols = ['temp_max', 'precipitation']   # para replicar tu X
target_col = 'rad_solar'                        # tu Y (7, 11, 13, 17)


data_sim


train_df


class PrecDataset(Dataset):
    """
    Dataset para MLP con ventanas deslizantes aplanadas.
    - df: DataFrame con columnas de features y target, ordenado por tiempo.
    - features_cols: lista de columnas usadas como predictores (en el orden deseado).
    - target_col: nombre de la columna objetivo.
    - sequence_length: tamaÃ±o de la ventana (W).
    - horizon: pasos hacia adelante a predecir (default=1 -> siguiente instante).
    
    La forma de X es [W * len(features_cols)], concatenando por tiempo:
    [feat1_t0, feat2_t0, ..., featK_t0, feat1_t1, feat2_t1, ..., featK_t(W-1)]
    """
    def __init__(self, df, features_cols, target_col, sequence_length=3, horizon=1):
        self.df = df.reset_index(drop=True)
        self.features_cols = list(features_cols)
        self.target_col = target_col
        self.W = int(sequence_length)
        self.h = int(horizon)
        self.data = []

        n = len(self.df)
        # nÃºmero de muestras posibles
        max_i = n - self.W - (self.h - 1)
        if max_i <= 0:
            return

        for i in range(max_i):
            # ventana [i, i+W)
            X_win = self.df.loc[i:i+self.W-1, self.features_cols].values  # (W, F)
            X = X_win.reshape(-1)  # aplanado para MLP
            y = self.df.loc[i + self.W + (self.h - 1), self.target_col]
            self.data.append((X, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y



ds = PrecDataset(train_df, features_cols, target_col, sequence_length=SEQUENCE_LENGTH, horizon=1)

print("NÃºmero de secuencias en el dataset:", len(ds))
for i in range(len(ds)):
    x, y = ds[i]
    print(f"\nEjemplo {i+1}:")
    print("Entrada (X):", x)
    print("Salida  (Y):", y.item())

