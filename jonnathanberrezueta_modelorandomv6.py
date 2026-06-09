"""
This project focuses on predicting Attention Deficit Hyperactivity Disorder (ADHD) in women 
using a supervised machine learning approach. We implemented the Random Forest algorithm due to 
its robustness and effectiveness in handling complex datasets with minimal preprocessing. The 
goal is to accurately identify individuals at risk based on relevant features in the dataset.

The entire workflow was developed on Google Colaboratory, leveraging its cloud-based environment 
for easy collaboration and seamless integration with Google Drive. Datasets were imported directly 
into the Colab notebook using Google Drive URLs and file IDs, ensuring a reproducible and 
accessible data pipeline.

Our submission package includes a `.ipynb` notebook that details the data preprocessing, model 
training, evaluation, and final prediction steps. Additionally, the corresponding `.csv` prediction 
file generated from the trained model is included in the zip file for review.

You can view the notebook and results directly through the following link:
https://drive.google.com/drive/folders/1qLu0fPCM5T0oq3hlRBavDvNwdbLdXOT7?usp=sharing

///

Este proyecto se centra en la predicción del Trastorno por Déficit de Atención e Hiperactividad (ADHD) 
en mujeres, utilizando un enfoque de aprendizaje automático supervisado. Implementamos el algoritmo 
Random Forest debido a su robustez y eficacia para manejar conjuntos de datos complejos con un 
mínimo de preprocesamiento. El objetivo es identificar con precisión a las personas en riesgo a 
partir de características relevantes del conjunto de datos.

Todo el flujo de trabajo fue desarrollado en Google Colaboratory, aprovechando su entorno en la nube 
para facilitar la colaboración y la integración fluida con Google Drive. Los conjuntos de datos fueron 
importados directamente al notebook de Colab utilizando URLs e IDs de archivos de Google Drive, lo que 
garantiza una canalización de datos reproducible y accesible.

El paquete enviado incluye un archivo `.ipynb` que detalla el preprocesamiento de los datos, el 
entrenamiento del modelo, su evaluación y los pasos de predicción final. Además, se incluye el archivo 
`.csv` con las predicciones generadas por el modelo entrenado.

Puedes ver el notebook y los resultados directamente en el siguiente enlace:
https://drive.google.com/drive/folders/1qLu0fPCM5T0oq3hlRBavDvNwdbLdXOT7?usp=sharing
"""


from IPython import get_ipython
from IPython.display import display
from IPython.display import Markdown

team_info = """
# Team / Equipo

**Team Name / Nombre del equipo:** FK² Team

**Members / Miembros:**
- Keyla Bueno
- Kely Juca
- Francisco Lopez
- Jonnathan Berrezueta"""

display(Markdown(team_info))


# 1. Data Loading and Preparation / 1. Carga y preparación de datos
#    - Load of used bookstores / Carga de librerias usadas.
#    - Load the datasets. / - Cargar los datasets.
#    - Merge the datasets into one. / - Unir los datasets en uno solo.
#    - Review of null values / Revisar si hay valores nulos
#    - Data cleaning: impute missing values and drop 'participant_id'. / - Limpieza de datos: imputar valores faltantes y eliminar 'participant_id'.
#    - Separate features (X) and targets (y_adhd, y_sex). / - Separar features (X) y targets (y_adhd, y_sex).
#    - Handle class imbalance for Sex_F using SMOTE. / - Manejar el desbalance de clases para Sex_F usando SMOTE.
#    - Scale the features. / - Escalar las features.


# Load of used bookstores / Carga de librerias usadas
!pip install imblearn
!pip install gdown --upgrade  # Install or update gdown / Instala o actualiza gdown
import requests
import pandas as pd
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, r2_score
from IPython.display import display
from imblearn.over_sampling import SMOTE


# Load the datasets from Google Drive / Cargar los datasets desde Google Drive
# IDs of the files in Google Drive  / IDs de los archivos en Google Drive
file_ids = {
    'data1': '1z1MDH90Roc9cmR8W5uGv6cAJDavfzcZD',  # ID of TRAIN_CATEGORICAL_METADATA_new.xlsx / ID de TRAIN_CATEGORICAL_METADATA_new.xlsx
    'data2': '1bIPufkvU_SmsJRMETjEb1L_lmmnsJHK0',  # ID of TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv / ID de TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv
    'data3': '1EgnKdeBND_q12SHSI1YzserBITqiyJqN',  # ID of TRAIN_QUANTITATIVE_METADATA_new.xlsx / ID de TRAIN_QUANTITATIVE_METADATA_new.xlsx
    'data4': '1zrbifo2QvaRFcS_hz0VcL9Zi7VPqrJBi'  # ID of TRAINING_SOLUTIONS.xlsx / ID de TRAINING_SOLUTIONS.xlsx
}

# Destination paths to save the files / Rutas de destino para guardar los archivos
destinations = {
    'data1': '/content/TRAIN_CATEGORICAL_METADATA_new.xlsx',
    'data2': '/content/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv',
    'data3': '/content/TRAIN_QUANTITATIVE_METADATA_new.xlsx',
    'data4': '/content/TRAINING_SOLUTIONS.xlsx'
}

# Download each file / Descargar cada archivo
for data_name, file_id in file_ids.items():
    destination = destinations[data_name]
    print(f"Downloading {data_name} from Google Drive... / Descargando {data_name} desde Google Drive...")
    gdown.download(id=file_id, output=destination, quiet=False)  # Using gdown / Usando gdown
    print(f"{data_name} successfully downloaded to {destination}\n / {data_name} descargado correctamente en {destination}\n")

# Read the files into pandas DataFrames / Leer los archivos en DataFrames de pandas
data1 = pd.read_excel(destinations['data1'])
data2 = pd.read_csv(destinations['data2'])
data3 = pd.read_excel(destinations['data3'])
data4 = pd.read_excel(destinations['data4'])


# Merge the datasets / Unir los datasets
df = data1.merge(data3, on='participant_id', how='inner')
df = df.merge(data2, on='participant_id', how='inner')
df = df.merge(data4, on='participant_id', how='inner')

print(f'Dimensions after merge / Dimensiones después del merge: {df.shape}')


# Review of null values / Revisar si hay valores nulos
print("   DataSet number 1")
print(data1.isnull().sum())
print("")
print("   DataSet number 2")
print(data2.isnull().sum())
print("")
print("   DataSet number 3")
print(data3.isnull().sum())
print("")
print("   DataSet number 4")
print(data4.isnull().sum())


# Data cleaning / Limpieza de datos
# Impute missing values / Imputar valores faltantes
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].fillna(df[col].mean())

# Drop the 'participant_id' column / Eliminar la columna 'participant_id'
df = df.drop('participant_id', axis=1)


# Separate features (X) and targets (y_adhd, y_sex) / Separar features (X) y targets (y_adhd, y_sex)
X = df.drop(['ADHD_Outcome', 'Sex_F'], axis=1)
y_adhd = df['ADHD_Outcome']
y_sex = df['Sex_F']


# Handle class imbalance for Sex_F using SMOTE / Manejar el desbalance de clases para Sex_F usando SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_sex_resampled = smote.fit_resample(X, y_sex)

# Create y_adhd_resampled with the original values of y_adhd / Crear y_adhd_resampled con los valores originales de y_adhd
y_adhd_resampled = pd.Series(index=X_resampled.index, dtype=y_adhd.dtype)
for i, original_index in enumerate(X_resampled.index):
    if original_index in y_adhd.index:
        y_adhd_resampled.iloc[i] = y_adhd.loc[original_index]
    else:
        # Assign a default value  / Asignar un valor por defecto
        y_adhd_resampled.iloc[i] = y_adhd.mode()[0]


# Scale the features / Escalar las features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=X_resampled.columns)

# Split the data / Dividir los datos
X_train, X_test, y_adhd_train, y_adhd_test, y_sex_train, y_sex_test = train_test_split(
    X_scaled, y_adhd_resampled, y_sex_resampled, test_size=0.2, random_state=42)


# 2. Model Training / 2. Entrenamiento de modelos
#    - Train a model for ADHD (classification). / - Entrenar un modelo para ADHD (clasificación).
#    - Train a model for sex (classification). / - Entrenar un modelo para sexo (clasificación).


# Model for ADHD / Modelo para ADHD
model_adhd = RandomForestClassifier(n_estimators=100, random_state=42)
model_adhd.fit(X_train, y_adhd_train)


# Model for sex / Modelo para sexo
model_sex = RandomForestClassifier(n_estimators=100, random_state=42)
model_sex.fit(X_train, y_sex_train)


# 3. Predictions and Evaluation / 3. Predicciones y evaluación
#    - Make predictions on the test set. / - Realizar predicciones en el conjunto de prueba.
#    - Calculate accuracy, recall, and R2 for both models. / - Calcular accuracy, recall y R2 para ambos modelos.
#    - Visualize the table. / Visualizar la tabla.


# Predictions / Predicciones
# Predict on the original test set / Predecir sobre el conjunto de prueba original
X_original_test = scaler.transform(X)
X_original_test = pd.DataFrame(X_original_test, columns=X_scaled.columns)

y_adhd_pred = model_adhd.predict(X_original_test)
y_sex_pred = model_sex.predict(X_original_test)


# Calculate accuracy, recall, and R2 / Calcular accuracy, recall y R2
accuracy_adhd = accuracy_score(y_adhd, y_adhd_pred)
recall_adhd = recall_score(y_adhd, y_adhd_pred)
r2_adhd = r2_score(y_adhd, y_adhd_pred)

accuracy_sex = accuracy_score(y_sex, y_sex_pred)
recall_sex = recall_score(y_sex, y_sex_pred)
r2_sex = r2_score(y_sex, y_sex_pred)

print(f"Accuracy ADHD / Exactitud ADHD: {accuracy_adhd:.4f}")
print(f"Recall ADHD / Sensibilidad ADHD: {recall_adhd:.4f}")
print(f"R2 ADHD / R2 ADHD: {r2_adhd:.4f}")
print(f"Accuracy Sex / Exactitud Sexo: {accuracy_sex:.4f}")
print(f"Recall Sex / Sensibilidad Sexo: {recall_sex:.4f}")
print(f"R2 Sex / R2 Sexo: {r2_sex:.4f}")


# Create results table / Crear tabla de resultados
results = pd.DataFrame({'participant_id': data1['participant_id'],
                        'ADHD_outcome_pred': y_adhd_pred,
                        'sexf_pred': y_sex_pred})

# Visualize the table / Visualizar la tabla
display(results)


# Download to CSV / Descargar a CSV
from google.colab import files
results.to_csv('predictions.csv', index=False)
files.download('predictions.csv')
print("File 'predictions.csv' downloaded! / ¡Archivo 'predictions.csv' descargado!")

