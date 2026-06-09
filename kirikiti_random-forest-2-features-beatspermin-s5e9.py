import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

import itertools

test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col='id')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
print("Shape of dataset:", df.shape)

# Vista Rapida
df.head()



# Suppose df is your original DataFrame
# Remove rows with missing data in numeric fields

# Supongamos que df es tu DataFrame original
# Eliminar filas con datos faltantes en campos numéricos

df_clean = df.dropna()


# Remove outliers using the DBSCAN
# Eliminar outliers usando DBSCAN

#Data Normalization
#Normalizacion de datos
scaler=StandardScaler()
df_norm=scaler.fit_transform(df_clean)

#Apply DBSCAN
#Aplicar DBSCAN
dbscan=DBSCAN(eps=0.5, min_samples=5)
labels=dbscan.fit_predict(df_norm)

#Add labels to original df
#Agregar etiquetas al df original
df_clean['outlier']=labels

#Filter dt to clean outlier
#Filtrar el df para eliminar outlier
df_clean=df_clean[df_clean['outlier']!=-1].drop(columns=['outlier'])

df_clean.head()



num_features = df.select_dtypes(include=[np.number]).columns.tolist()


# We define the transformations to try
# Definimos las transformaciones a probar
def log_transform(x):
    return np.log1p(x)

def poly_transform(x, degree=2):
    return x ** degree

def exp_transform(x):
    return np.exp(x)

# Create a dictionary to store the transformations
# Crear un diccionario para almacenar las transformaciones
transformations = {
    'original': lambda x: x,
    'log': log_transform,
    'poly2': lambda x: poly_transform(x, degree=2),
    'poly3': lambda x: poly_transform(x, degree=3),
    'exp': exp_transform
}

# Create a list to store the results
# Crear una lista para almacenar los resultados
correlation_results = []

# Test all combinations of transformations
# Probar todas las combinaciones de transformaciones
for feature1, feature2 in itertools.combinations(num_features, 2):
    for trans1_name, trans1_func in transformations.items():
        for trans2_name, trans2_func in transformations.items():
            # Apply transformations
            # Aplicar transformaciones
            transformed1 = trans1_func(df_clean[feature1])
            transformed2 = trans2_func(df_clean[feature2])

            # Calculate the correlation
            # Calcular la correlación
            correlation = transformed1.corr(transformed2)

            # Store the results
            # Almacenar los resultados
            correlation_results.append({
                'Var1': feature1,
                'Var2': feature2,
                'Transformation1': trans1_name,
                'Transformation2': trans2_name,
                'Correlation': correlation
            })

# Convert the list of results to a DataFrame
# Convertir la lista de resultados a un DataFrame
correlation_results_df = pd.DataFrame(correlation_results)


# Filter the DataFrame to only include rows where 'BeatsPerMinute' is involved
# Filtrar el DataFrame para que solo incluya filas donde 'BeatsPerMinute' esté involucrada
filtered_df = correlation_results_df[
    (correlation_results_df['Var1'] == 'BeatsPerMinute') | 
    (correlation_results_df['Var2'] == 'BeatsPerMinute')
]

# Calculate the absolute value of the correlations
# Calcular el valor absoluto de las correlaciones
filtered_df['AbsCorrelation'] = filtered_df['Correlation'].abs()

# Sort the DataFrame by the absolute correlation column from highest to lowest
# Ordenar el DataFrame por la columna de correlación absoluta de mayor a menor
sorted_filtered_df = filtered_df.sort_values(by='AbsCorrelation', ascending=False)

# Select the top 20 highest
# Seleccionar los 20 más altos
sorted_filtered_df.head(20)


# Elijo el 420 y 215 por su más alta correlación 
#y por que para la variable objetivo "BeatsPerMinute" en ambos casos NO se usa tranformación
#lo que facilita el modelo


# Add columns with the requested transformations
# Agregar columnas con las transformaciones solicitadas
df_clean['ExpAudioLoudness'] = np.expm1(df_clean['AudioLoudness']) # Exponencial de AudioLoudness
df_clean['Poly3RhythmScore'] = poly_transform(df_clean['RhythmScore'], degree=3)  # Polinómica de grado 3 de RhythmScore


# Graficar histogramas para todas las características numéricas
# Plot histograms for all numeric features
mod_features=['ExpAudioLoudness','Poly3RhythmScore','BeatsPerMinute']
df_clean[mod_features].hist(bins=30, figsize=(15,12), layout=(4,3))

plt.suptitle("Feature Distributions")
plt.show()


sns.pairplot(df_clean[mod_features])
plt.show()


correlation_matrix = df_clean[mod_features].corr()
print(correlation_matrix)


target_variable = 'BeatsPerMinute'

# Split the DataFrame into independent variables (X) and dependent variable (y)
# Dividir el DataFrame en variables independientes (X) y dependientes (y)
X = df_clean[mod_features].drop(columns=[target_variable])  # Eliminar la variable objetivo
y = df_clean[target_variable]  # Variable objetivo

# Split into training and test sets
# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
# Crear y entrenar el modelo de Bosque Aleatorio
model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calculate the RMSE
# Calcular el RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Show el RMSE
# Mostrar el RMSE
print(f'RMSE: {rmse}')


# Calcular los errores
errors = y_test_deslog - y_pred_deslog

#  histograma
# Crear el histograma
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True)
plt.title('Histograma de Errores de Predicción')
plt.xlabel('Errores')
plt.ylabel('Frecuencia')
plt.axvline(0, color='red', linestyle='--')  # Línea vertical en x=0
plt.grid(True)
plt.show()


# The distribution of the errors is asymmetric, 
# so if we transformed the variables differently, 
# we would be able to improve the model

# La distribución de los errores es asimetrica, 
# por lo que si transformasemos las variables de otra forma, 
# lograriamos mejorar el modelo


# Add columns with the requested transformations to the test DataFrame
# Agregar columnas con las transformaciones solicitadas al data frame de test

test['ExpAudioLoudness'] = np.expm1(test['AudioLoudness']) # Exponencial de AudioLoudness
test['Poly3RhythmScore'] = poly_transform(test['RhythmScore'], degree=3)  # Polinómica de grado 3 de RhythmScore

X_submit = test[['ExpAudioLoudness','Poly3RhythmScore']]  # Eliminar la variable objetivo

# Make predictions on the test set for submission
# Realizar predicciones en el conjunto de test para el submit
y_pred_submit = model.predict(X_submit)

y_pred_submit


# Preparation and submission of the Output
#Preparacion y envio del Output

df_sub['BeatsPerMinute'] = y_pred_submit
df_sub.to_csv('test_predictions_dbscan.csv', index=False)
df_sub.head()


