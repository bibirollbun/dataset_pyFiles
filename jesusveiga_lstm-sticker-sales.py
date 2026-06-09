import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")
data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/sample_submission.csv")


test_data.head()


# Convertir las fechas
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Extraer características temporales relevantes
for df in [train_data, test_data]:
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Variables cíclicas (útiles para capturar estacionalidades)
    df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 365.0)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)

# Conversión de las variables categóricas a tipo string
for df in [train_data, test_data]:
    df['Month'] = df['Month'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['week_of_year'] = df['week_of_year'].astype(str)

# Solo conservamos las columnas necesarias para modelado en train_data
features = ['id', 'date', 'country', 'product', 'store', 'Year', 'Month', 'Day', 'day_of_week', 'week_of_year', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'num_sold']

# Filtramos solo las columnas necesarias en train_data
train_data = train_data[features]

# Para test_data, no incluimos 'num_sold'
features_test = ['id', 'date', 'country', 'product', 'store', 'Year', 'Month', 'Day', 'day_of_week', 'week_of_year', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

# Filtramos solo las columnas necesarias en test_data
test_data = test_data[features_test]


# Categorical features
cat_features = ['country', 'product', 'store']

# Eliminar filas con 100% NaN en 'num_sold' para combinaciones de categorías
train_data = train_data[train_data.groupby(cat_features)['num_sold'].transform(lambda x: x.isna().mean() < 1)]

# Mostrar el resultado
print(f"Shape after dropping NaN categories: {len(train_data)}")


def interpolate_missing_by_date(group):
    group['date'] = pd.to_datetime(group['date'])
    
    group = group.sort_values(by='date')
    
    # Interpola los valores faltantes en 'num_sold' por fecha (interpolación lineal)
    group['num_sold'] = group['num_sold'].interpolate(method='linear', limit_direction='both')
    
    return group

# Aplica la interpolación para todos los grupos
train_data = train_data.groupby(['country', 'store', 'product'], group_keys=False).apply(interpolate_missing_by_date)

# Verifica el resultado
train_data.head()


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Preprocesamiento
train_data['date'] = pd.to_datetime(train_data['date'])
train_data['Year'] = train_data['date'].dt.year
train_data['Month'] = train_data['date'].dt.month
train_data['Day'] = train_data['date'].dt.day
train_data = train_data.drop(['id', 'date'], axis=1)
train_data['num_sold'] = np.log1p(train_data['num_sold'])  # Aplicamos log a 'num_sold'

# Codificación de columnas categóricas
categorical_cols = ['country', 'store', 'product']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Separar las características y el objetivo
X = train_data.drop('num_sold', axis=1)
y = train_data['num_sold']

# División entrenamiento-validación
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Asegurarse de que X e y tienen el mismo número de muestras
train_data = train_data.dropna(subset=['num_sold'])  # Eliminar filas donde 'num_sold' sea NaN

# Volver a dividir después de eliminar filas con NaN en 'num_sold'
X = train_data.drop('num_sold', axis=1)
y = train_data['num_sold']

# División entrenamiento-validación
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características de entrada
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Escalar la variable objetivo
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1))

# Verificar las formas de los datos
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"y_train_scaled shape: {y_train_scaled.shape}")

# Reformatear las entradas para LSTM (debe tener la forma [muestras, pasos de tiempo, características])
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_valid_reshaped = X_valid_scaled.reshape(X_valid_scaled.shape[0], 1, X_valid_scaled.shape[1])

print(f"X_train_reshaped shape: {X_train_reshaped.shape}")
print(f"X_valid_reshaped shape: {X_valid_reshaped.shape}")


from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Crear el modelo con Dropout y Batch Normalization
model = Sequential([
    # Primera capa LSTM
    LSTM(64, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    BatchNormalization(),  # Normalización después de la primera capa LSTM
    Dropout(0.2),          # Dropout del 20%

    # Segunda capa LSTM
    LSTM(32, return_sequences=False),
    BatchNormalization(),  # Normalización después de la segunda capa LSTM
    Dropout(0.2),          # Dropout del 20%

    # Capa densa con activación ReLU
    Dense(16, activation='relu'),
    Dropout(0.2),          # Dropout del 20% después de la capa densa

    # Capa de salida
    Dense(1)               # Salida para predicción
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Resumen del modelo
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)

# Entrenar el modelo
history = model.fit(
    X_train_reshaped, y_train_scaled,  # Usa los datos escalados y reshaped
    epochs=50,
    batch_size=32,
    validation_data=(X_valid_reshaped, y_valid_scaled),  # Usa los datos de validación escalados
    shuffle=False,
    callbacks=[early_stopping, checkpoint]  # Añadir los callbacks
)

# Evaluar el modelo
loss, mae = model.evaluate(X_valid_reshaped, y_valid_scaled)
print(f"Validation Loss: {loss}")
print(f"Validation MAE: {mae}")



from tensorflow.keras.models import load_model

# Cargar el mejor modelo guardado
best_model = load_model('best_model.keras')


best_model = model.save('final_model.keras')


# Preprocesamiento de test_data (como se hizo con train_data)
test_data['Year'] = test_data['date'].dt.year
test_data['Month'] = test_data['date'].dt.month
test_data['Day'] = test_data['date'].dt.day
test_data = test_data.drop(['id', 'date'], axis=1)

# Codificar las columnas categóricas usando los mismos LabelEncoders que para train_data
for col in categorical_cols:
    test_data[col] = label_encoders[col].transform(test_data[col])

# Escalar las características del test_data
X_test_scaled = scaler.transform(test_data)

# Reformatear las entradas de test_data para LSTM
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Verificar la forma final de X_test_reshaped
print(f"X_test_reshaped shape: {X_test_reshaped.shape}")

# Realizar predicciones con el modelo
y_test_scaled_pred = model.predict(X_test_reshaped)

# Desescalar las predicciones de la variable objetivo
y_test_pred = target_scaler.inverse_transform(y_test_scaled_pred)

# Asegurarse de que y_test_pred tenga el mismo orden que test_data
test_data['num_sold'] = np.expm1(y_test_pred)  # Aplicar la inversa de log1p



# Crear un dataframe de salida con las columnas 'id' y 'num_sold'
submission = pd.DataFrame({
    'id': pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")['id'],
    'num_sold': test_data['num_sold']
})

# Guardar el archivo CSV
submission.to_csv('submission.csv', index=False)

# Verificar las primeras filas
submission.head()


