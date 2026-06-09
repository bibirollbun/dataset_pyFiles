# === LibrerÃ­as del sistema ===
import os
import warnings
from pathlib import Path

# === Manejo y procesamiento de datos ===
import numpy as np
import pandas as pd

# === VisualizaciÃ³n ===
import matplotlib.pyplot as plt
import seaborn as sns


# === IngenierÃ­a de caracterÃ­sticas y reducciÃ³n de dimensionalidad ===
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

# === ValidaciÃ³n cruzada y particiÃ³n de datos ===
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    cross_val_predict,
    train_test_split
)

# === MÃ©tricas de evaluaciÃ³n para regresiÃ³n ===
from sklearn.metrics import (
    mean_squared_log_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

# === Preprocesamiento y pipelines ===
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
    FunctionTransformer
)

# === ConfiguraciÃ³n de estilo de grÃ¡ficos ===
plt.style.use("seaborn-darkgrid")
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/playground-series-s3e11/train.csv", index_col = "id")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e11/test.csv", index_col = "id")


df_train


Y = df_train.pop("cost")
Y


df_train.isna().sum()


Y.plot(kind = "hist")


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# # ðŸš« NO usamos mixed_precision en CPU â€” se elimina completamente

# # Suponiendo que ya tenÃ©s tus datos en df_train_ y Y
# X_train, X_val, y_train, y_val = train_test_split(df_train, Y, test_size=0.22, random_state=42)

# # # Escalado estÃ¡ndar
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_val = scaler.transform(X_val)

# # Datasets eficientes con tf.data
# batch_size = 512
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = train_dataset.shuffle(100000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# # Modelo secuencial profundo
# model = Sequential([
#     Dense(512, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),

#     Dense(512, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),

#     Dense(384, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.25),

#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.2),

#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.2),

#     Dense(1)  # Salida regresiÃ³n (float32 por defecto en CPU)
# ])

# # CompilaciÃ³n
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss='mse',
#     metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
# )

# # Callbacks
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_rmse', patience=10, restore_best_weights=True),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.5, patience=5, verbose=1),
#     # tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_rmse')
# ]

# # Entrenamiento
# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=50,
#     callbacks=callbacks,
#     verbose=1
# )

# # EvaluaciÃ³n final
# loss, final_rmse = model.evaluate(val_dataset)
# print(f"\nâœ… RMSE final en validaciÃ³n: {final_rmse:.4f}")

# # ðŸ“Š Graficar entrenamiento por RMSE
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['rmse'], label='Train RMSE')
# plt.plot(history.history['val_rmse'], label='Val RMSE')
# plt.xlabel('Epochs')
# plt.ylabel('RMSE')
# plt.title('RMSE durante el entrenamiento')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# model.save("modelo_final.keras")

from tensorflow.keras.models import load_model

model = load_model("/kaggle/input/cost-predict-tensorflow-regression/tensorflow2/default/1/modelo_final.keras")



preds = model.predict(df_test)


preds


sub = pd.DataFrame({
    "id": df_test.index,
    "cost": preds.ravel()  # o preds.flatten()
})
sub.to_csv("submission.csv", index=False)


